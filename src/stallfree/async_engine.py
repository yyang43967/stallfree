"""Run the synchronous engine behind an asyncio server.

One background task (the *loop*) owns the ``LLMEngine``; nothing else touches it. Each
``engine.step()`` runs in a single worker thread (``asyncio.to_thread``), so while the GPU works
the event loop keeps accepting connections and streaming tokens already produced.

Requests and cancellations reach the loop through an inbox and are applied between steps, never
while a step is running. Each request gets its own ``asyncio.Queue`` of ``StepOutput`` items,
which the HTTP handler reads. If the handler stops reading (client disconnected, task cancelled),
the request is aborted and its KV blocks are freed at the next step boundary.

Backpressure: at most ``max_queued_requests`` requests may be in the system at once; beyond that
``generate`` raises ``QueueFullError`` immediately, and the API turns it into a 503.

Metrics: when a request ends, one structured log line records its queue time, time to first
token, time per output token, and end-to-end latency, measured on the server's monotonic clock.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
import logging
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field

from stallfree.engine import LLMEngine, StepOutput
from stallfree.sequence import FinishReason, SamplingParams

logger = logging.getLogger("stallfree.metrics")


class QueueFullError(RuntimeError):
    """Too many requests in the system; the caller should retry later (HTTP 503)."""


class EngineDeadError(RuntimeError):
    """The engine loop stopped because of an error; no further requests can run."""


@dataclass
class _Request:
    """One request's bookkeeping on the event-loop side."""

    rid: int
    prompt: list[int]
    params: SamplingParams
    queue: asyncio.Queue[StepOutput | BaseException] = field(default_factory=asyncio.Queue)
    seq_id: int | None = None
    arrival: float = field(default_factory=time.monotonic)
    first_step_start: float | None = None  # start of the step that produced the first token
    first_token: float | None = None
    last_token: float | None = None
    num_tokens: int = 0
    done: bool = False


class AsyncEngine:
    """asyncio front end for one ``LLMEngine``.

    Args:
        engine: the synchronous engine; owned exclusively by this object's loop from ``start``.
        max_queued_requests: most requests (waiting or running) accepted at once.
    """

    def __init__(self, engine: LLMEngine, max_queued_requests: int = 1024) -> None:
        if max_queued_requests < 1:
            raise ValueError("max_queued_requests must be >= 1")
        self.engine = engine
        self.max_queued_requests = max_queued_requests
        self._inbox: list[tuple[str, _Request]] = []
        self._wake = asyncio.Event()
        self._by_seq: dict[int, _Request] = {}
        self._active = 0  # accepted and not yet finished, aborted, or failed
        self._ids = itertools.count()
        self._task: asyncio.Task[None] | None = None
        self._error: BaseException | None = None
        self._stopping = False

    # --- lifecycle -----------------------------------------------------------------------------

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="engine-loop")

    async def stop(self) -> None:
        self._stopping = True
        self._wake.set()
        if self._task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    @property
    def num_active(self) -> int:
        return self._active

    # --- requests ------------------------------------------------------------------------------

    def validate(self, num_prompt_tokens: int, max_tokens: int) -> None:
        """Raise ``ValueError`` if the request could never run (safe to call at any time)."""
        self.engine.scheduler.validate(num_prompt_tokens, max_tokens)

    async def generate(
        self, prompt_token_ids: Sequence[int], params: SamplingParams
    ) -> AsyncIterator[StepOutput]:
        """Yield the request's outputs as they are produced; the last one has ``finished``.

        Raises:
            QueueFullError: if ``max_queued_requests`` requests are already in the system.
            EngineDeadError: if the engine loop has failed.
            ValueError: if the request can never run (too long for the model or the cache).
        """
        if self._error is not None:
            raise EngineDeadError("engine loop failed") from self._error
        if self._active >= self.max_queued_requests:
            raise QueueFullError(f"{self._active} requests in flight")
        self.validate(len(prompt_token_ids), params.max_tokens)
        req = _Request(next(self._ids), list(prompt_token_ids), params)
        self._active += 1
        self._post("add", req)
        try:
            while True:
                item = await req.queue.get()
                if isinstance(item, BaseException):
                    raise item
                yield item
                if item.finished:
                    return
        finally:
            # Reached on normal completion, on an error, and when the consumer stops early
            # (GeneratorExit / CancelledError). Only the last case needs an abort.
            if not req.done:
                self._post("abort", req)

    def _post(self, kind: str, req: _Request) -> None:
        self._inbox.append((kind, req))
        self._wake.set()

    # --- the loop ------------------------------------------------------------------------------

    async def _run(self) -> None:
        try:
            while not self._stopping:
                if not self._inbox and not self.engine.has_unfinished():
                    self._wake.clear()
                    await self._wake.wait()  # idle: sleep until a request or abort arrives
                    continue
                self._apply_inbox()
                if not self.engine.has_unfinished():
                    continue
                step_start = time.monotonic()
                outputs = await asyncio.to_thread(self.engine.step)
                self._deliver(outputs, step_start)
        except BaseException as exc:
            self._error = exc
            self._fail_all(exc)
            if not isinstance(exc, asyncio.CancelledError):
                logger.exception("engine loop failed")
            raise

    def _apply_inbox(self) -> None:
        inbox, self._inbox = self._inbox, []
        for kind, req in inbox:
            if kind == "add":
                if req.done:  # aborted before it was ever added
                    continue
                try:
                    req.seq_id = self.engine.add_request(req.prompt, req.params, req.arrival)
                except ValueError as exc:  # validated earlier; kept as a safety net
                    self._finish(req, "rejected")
                    req.queue.put_nowait(exc)
                    continue
                self._by_seq[req.seq_id] = req
            elif kind == "abort" and not req.done:
                if req.seq_id is not None:
                    self.engine.abort(req.seq_id)
                    self._by_seq.pop(req.seq_id, None)
                self._finish(req, FinishReason.ABORT.value)

    def _deliver(self, outputs: list[StepOutput], step_start: float) -> None:
        now = time.monotonic()
        for out in outputs:
            req = self._by_seq.get(out.seq_id)
            if req is None or req.done:
                continue  # aborted while this step ran
            if req.first_token is None:
                req.first_step_start, req.first_token = step_start, now
            req.last_token = now
            req.num_tokens += 1
            req.queue.put_nowait(out)
            if out.finished:
                self._by_seq.pop(out.seq_id, None)
                reason = out.finish_reason.value if out.finish_reason else "unknown"
                self._finish(req, reason)

    def _finish(self, req: _Request, reason: str) -> None:
        req.done = True
        self._active -= 1
        self._log_metrics(req, reason)

    def _fail_all(self, exc: BaseException) -> None:
        err = EngineDeadError("engine loop failed")
        err.__cause__ = exc
        for _, req in self._inbox:
            if not req.done:
                req.queue.put_nowait(err)
        for req in list(self._by_seq.values()):
            if not req.done:
                req.done = True
                req.queue.put_nowait(err)
        self._inbox.clear()
        self._by_seq.clear()

    @staticmethod
    def _log_metrics(req: _Request, reason: str) -> None:
        end = time.monotonic()
        record: dict[str, object] = {
            "event": "request_done",
            "request": req.rid,
            "finish_reason": reason,
            "prompt_tokens": len(req.prompt),
            "completion_tokens": req.num_tokens,
            "e2e_s": round(end - req.arrival, 6),
        }
        if req.first_token is not None and req.first_step_start is not None:
            record["queue_s"] = round(req.first_step_start - req.arrival, 6)
            record["ttft_s"] = round(req.first_token - req.arrival, 6)
            if req.num_tokens > 1 and req.last_token is not None:
                record["tpot_s"] = round(
                    (req.last_token - req.first_token) / (req.num_tokens - 1), 6
                )
        logger.info(json.dumps(record))
