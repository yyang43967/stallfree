"""Load generator: replays a trace against an OpenAI-compatible server and records SSE timestamps.

    python -m benchmarks.loadgen --base-url http://127.0.0.1:8000/v1 --trace trace.jsonl.gz \
        --mode open --rate 12.5 --out run.jsonl
    python -m benchmarks.loadgen ... --mode closed --concurrency 256 --out sat.jsonl

Single-threaded asyncio with one httpx client. It runs in its own process, separate from the
server and from the CPU sampler, so its CPU use can be measured (a run where it exceeds 80% of
one core is flagged by the analysis).

Every request streams with ``ignore_eos`` and ``max_tokens`` from the trace, greedy sampling, and
``stream_options.include_usage`` so the server reports the prompt and completion token counts it
saw. Times are seconds on the monotonic clock, relative to the start of the run.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx

from benchmarks.workload import (
    DEFAULT_WINDOW,
    MIN_MEASURE_S,
    MIN_MEASURED,
    WARMUP_S,
    TraceRequest,
    Window,
    arrivals,
    loads,
    plan,
)

MODEL = "Qwen/Qwen3-0.6B"
CLOSED_MEASURE_S = 120.0


def body(req: TraceRequest, model: str) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": req.prompt}],
        "max_tokens": req.max_tokens,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


async def send(
    client: httpx.AsyncClient, url: str, req: TraceRequest, model: str, t0: float
) -> dict[str, Any]:
    """One streaming request. ``chunks`` holds the arrival time of every SSE event that carried
    generated text; the first is the time to first token."""
    rec: dict[str, Any] = {
        "index": req.index,
        "target_prompt_tokens": req.prompt_tokens,
        "max_tokens": req.max_tokens,
        "long": req.long,
        "sent": time.monotonic() - t0,
        "chunks": [],
        "prompt_tokens": None,
        "completion_tokens": None,
        "finish_reason": None,
        "status": None,
        "error": None,
    }
    chunks: list[float] = rec["chunks"]
    try:
        async with client.stream("POST", url, json=body(req, model)) as resp:
            rec["status"] = resp.status_code
            if resp.status_code != 200:
                rec["error"] = (await resp.aread()).decode(errors="replace")[:500]
                return rec
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                now = time.monotonic() - t0
                event = json.loads(data)
                if "error" in event and "choices" not in event:
                    rec["error"] = json.dumps(event["error"])[:500]
                    break
                usage = event.get("usage")
                if usage:
                    rec["prompt_tokens"] = usage.get("prompt_tokens")
                    rec["completion_tokens"] = usage.get("completion_tokens")
                for choice in event.get("choices") or ():
                    if (choice.get("delta") or {}).get("content"):
                        chunks.append(now)
                    if choice.get("finish_reason"):
                        rec["finish_reason"] = choice["finish_reason"]
    except (httpx.HTTPError, asyncio.CancelledError) as exc:
        rec["error"] = f"{type(exc).__name__}: {exc}"[:500]
        if isinstance(exc, asyncio.CancelledError):
            rec["cancelled"] = True
    rec["end"] = time.monotonic() - t0
    return rec


def _client() -> httpx.AsyncClient:
    limits = httpx.Limits(max_connections=None, max_keepalive_connections=None)
    return httpx.AsyncClient(limits=limits, timeout=httpx.Timeout(None))


async def open_loop(
    base_url: str,
    trace: Sequence[TraceRequest],
    rate: float,
    model: str = MODEL,
    window: Window = DEFAULT_WINDOW,
) -> dict[str, Any]:
    """Poisson arrivals at ``rate``: warm-up requests, then the measured ones; waits for all."""
    warmup, measured = plan(trace, rate, window)
    reqs = list(trace[: warmup + measured])
    times = arrivals(reqs, rate)
    url = f"{base_url}/chat/completions"
    async with _client() as client:
        t0, t0_wall = time.monotonic(), time.time()
        tasks = []
        lag = 0.0
        for req, at in zip(reqs, times, strict=True):
            delay = at - (time.monotonic() - t0)
            if delay > 0:
                await asyncio.sleep(delay)
            else:
                lag = max(lag, -delay)
            tasks.append(asyncio.create_task(send(client, url, req, model, t0)))
        records = await asyncio.gather(*tasks)
    for rec, at in zip(records, times, strict=True):
        rec["scheduled"] = at
        rec["phase"] = "warmup" if rec["index"] < warmup else "measured"
    return {
        "mode": "open",
        "t0_wall": t0_wall,
        "rate": rate,
        "warmup_requests": warmup,
        "measured_requests": measured,
        "max_send_lag_s": lag,
        "records": records,
    }


async def closed_loop(
    base_url: str,
    trace: Sequence[TraceRequest],
    concurrency: int,
    warmup_s: float = WARMUP_S,
    measure_s: float = CLOSED_MEASURE_S,
    model: str = MODEL,
) -> dict[str, Any]:
    """``concurrency`` clients, each sending its next request as soon as the last one ends,
    cycling through the trace. Requests still running at the end are cancelled. Throughput is
    the number of requests that finished inside the measurement window divided by its length."""
    url = f"{base_url}/chat/completions"
    order = itertools.cycle(trace)
    records: list[dict[str, Any]] = []
    stop_at = warmup_s + measure_s

    async with _client() as client:
        t0, t0_wall = time.monotonic(), time.time()

        async def worker() -> None:
            while time.monotonic() - t0 < stop_at:
                records.append(await send(client, url, next(order), model, t0))

        workers = [asyncio.create_task(worker()) for _ in range(concurrency)]
        await asyncio.sleep(stop_at)
        for w in workers:
            w.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
    done = [
        r
        for r in records
        if r["error"] is None and r.get("end") is not None and warmup_s <= r["end"] < stop_at
    ]
    tokens = sum(r["completion_tokens"] or 0 for r in done)
    return {
        "mode": "closed",
        "t0_wall": t0_wall,
        "concurrency": concurrency,
        "warmup_s": warmup_s,
        "measure_s": measure_s,
        "completed_in_window": len(done),
        "request_throughput": len(done) / measure_s,
        "output_token_throughput_completed": tokens / measure_s,
        "records": records,
    }


def write_jsonl(path: Path, result: dict[str, Any]) -> None:
    """Per-request records as JSON lines; the run summary goes in a ``.summary.json`` file."""
    records = result.pop("records")
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    path.with_suffix(".summary.json").write_text(json.dumps(result, indent=1) + "\n")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--base-url", required=True)
    p.add_argument("--trace", required=True, type=Path, help="gzipped JSON lines")
    p.add_argument("--mode", choices=["open", "closed"], required=True)
    p.add_argument("--rate", type=float)
    p.add_argument("--concurrency", type=int)
    p.add_argument("--warmup-s", type=float, default=WARMUP_S)
    p.add_argument("--measure-s", type=float, default=CLOSED_MEASURE_S, help="closed loop")
    p.add_argument("--min-measured", type=int, default=MIN_MEASURED, help="open loop")
    p.add_argument("--min-measure-s", type=float, default=MIN_MEASURE_S, help="open loop")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args(argv)
    trace = loads(a.trace.read_bytes())
    if a.mode == "open":
        window = Window(a.warmup_s, a.min_measured, a.min_measure_s)
        result = asyncio.run(open_loop(a.base_url, trace, a.rate, a.model, window))
    else:
        result = asyncio.run(
            closed_loop(a.base_url, trace, a.concurrency, a.warmup_s, a.measure_s, a.model)
        )
    write_jsonl(a.out, result)


if __name__ == "__main__":
    main()
