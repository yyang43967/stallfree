"""OpenAI-compatible HTTP server: chat completions (JSON or SSE), model list, health.

    python -m stallfree.server.api --model <dir> [--device cuda --attention flash] ...

Requests are validated and tokenized here; the async engine sees token ids only. A streaming
request whose client disconnects is cancelled, which aborts it in the engine and frees its blocks.
"""

from __future__ import annotations

import argparse
import contextlib
import hmac
import json
import logging
import os
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from stallfree.async_engine import AsyncEngine, EngineDeadError, QueueFullError
from stallfree.config import AttentionBackendName, EngineConfig, SchedulerConfig
from stallfree.sequence import FinishReason, SamplingParams
from stallfree.server.protocol import (
    AssistantMessage,
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionRequest,
    Choice,
    ChunkChoice,
    Delta,
    ModelCard,
    ModelList,
    Usage,
    error_body,
    new_completion_id,
    now,
)
from stallfree.tokenizer import ChatTokenizer, IncrementalDetokenizer

logger = logging.getLogger("stallfree.server")

# OpenAI's defaults when a field is omitted.
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 1.0


def _error(status: int, message: str, type_: str, param: str | None = None) -> JSONResponse:
    return JSONResponse(error_body(message, type_, param), status_code=status)


def _finish_reason(reason: FinishReason | None) -> str | None:
    if reason is FinishReason.STOP:
        return "stop"
    if reason is FinishReason.LENGTH:
        return "length"
    return None


def create_app(
    engine: AsyncEngine,
    tokenizer: ChatTokenizer,
    *,
    served_model_name: str,
    max_model_len: int,
    api_key: str | None = None,
) -> FastAPI:
    """Build the app. The engine loop starts and stops with the app's lifespan."""

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        await engine.start()
        try:
            yield
        finally:
            await engine.stop()

    app = FastAPI(title="stallfree", lifespan=lifespan)
    created = now()

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        loc = ".".join(str(p) for p in first.get("loc", ())[1:]) or None
        message = f"invalid request: {first.get('msg', exc)}"
        return _error(400, message, "invalid_request_error", loc)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        return _error(exc.status_code, str(exc.detail), "invalid_request_error")

    @app.middleware("http")
    async def require_key(request: Request, call_next: Any) -> Any:
        if api_key and request.url.path.startswith("/v1"):
            given = request.headers.get("authorization", "")
            if not hmac.compare_digest(given.encode(), f"Bearer {api_key}".encode()):
                return _error(401, "invalid or missing API key", "authentication_error")
        return await call_next(request)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models() -> ModelList:
        return ModelList(data=[ModelCard(id=served_model_name, created=created)])

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(req: ChatCompletionRequest) -> Any:
        if req.model != served_model_name:
            return _error(
                404, f"model {req.model!r} is not served here", "invalid_request_error", "model"
            )
        unsupported = req.unsupported()
        if unsupported:
            return _error(
                400,
                f"unsupported parameter(s): {', '.join(unsupported)}",
                "invalid_request_error",
                unsupported[0],
            )
        prompt = tokenizer.encode_chat(req.chat(), req.chat_template_kwargs or {})
        max_tokens = req.requested_max_tokens() or max(1, max_model_len - len(prompt))
        try:
            params = SamplingParams(
                temperature=DEFAULT_TEMPERATURE if req.temperature is None else req.temperature,
                top_p=DEFAULT_TOP_P if req.top_p is None else req.top_p,
                max_tokens=max_tokens,
                ignore_eos=req.ignore_eos,
            )
            engine.validate(len(prompt), max_tokens)
        except ValueError as exc:
            return _error(400, str(exc), "invalid_request_error", "max_tokens")
        if engine.num_active >= engine.max_queued_requests:
            return _error(503, "server is at capacity; retry later", "server_error")
        stream = engine.generate(prompt, params)
        rid = new_completion_id()
        if req.stream:
            include_usage = bool(req.stream_options and req.stream_options.include_usage)
            return StreamingResponse(
                _sse(stream, tokenizer, rid, served_model_name, len(prompt), include_usage),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache"},
            )
        return await _complete(stream, tokenizer, rid, served_model_name, len(prompt))

    return app


async def _complete(
    stream: AsyncGenerator[Any, None],
    tokenizer: ChatTokenizer,
    rid: str,
    model: str,
    prompt_len: int,
) -> Any:
    ids: list[int] = []
    reason: str | None = None
    try:
        async for out in stream:
            ids.append(out.token_id)
            reason = _finish_reason(out.finish_reason)
    except QueueFullError as exc:
        return _error(503, str(exc), "server_error")
    except EngineDeadError as exc:
        return _error(500, str(exc), "server_error")
    return ChatCompletion(
        id=rid,
        created=now(),
        model=model,
        choices=[
            Choice(message=AssistantMessage(content=tokenizer.decode(ids)), finish_reason=reason)
        ],
        usage=Usage(
            prompt_tokens=prompt_len, completion_tokens=len(ids), total_tokens=prompt_len + len(ids)
        ),
    )


def _chunk(rid: str, model: str, created: int, delta: Delta, finish: str | None = None) -> str:
    c = ChatCompletionChunk(
        id=rid,
        created=created,
        model=model,
        choices=[ChunkChoice(delta=delta, finish_reason=finish)],
    )
    return f"data: {c.model_dump_json(exclude_none=True)}\n\n"


async def _sse(
    stream: AsyncGenerator[Any, None],
    tokenizer: ChatTokenizer,
    rid: str,
    model: str,
    prompt_len: int,
    include_usage: bool,
) -> AsyncIterator[str]:
    """One SSE event per text delta. If the client disconnects, Starlette cancels this generator,
    which closes ``stream`` and aborts the request in the engine."""
    created = now()
    detok = IncrementalDetokenizer(tokenizer)
    completion = 0
    yield _chunk(rid, model, created, Delta(role="assistant", content=""))
    try:
        async for out in stream:
            completion += 1
            text = detok.add(out.token_id)
            if out.finished:
                text += detok.flush()
                if text:
                    yield _chunk(rid, model, created, Delta(content=text))
                yield _chunk(rid, model, created, Delta(), _finish_reason(out.finish_reason))
            elif text:
                yield _chunk(rid, model, created, Delta(content=text))
    except (QueueFullError, EngineDeadError) as exc:
        yield f"data: {json.dumps(error_body(str(exc), 'server_error'))}\n\n"
        return
    finally:
        await stream.aclose()  # closing the generator aborts the request if it is unfinished
    if include_usage:
        usage = ChatCompletionChunk(
            id=rid,
            created=created,
            model=model,
            choices=[],
            usage=Usage(
                prompt_tokens=prompt_len,
                completion_tokens=completion,
                total_tokens=prompt_len + completion,
            ),
        )
        yield f"data: {usage.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Serve a model over an OpenAI-compatible API.")
    p.add_argument("--model", required=True, help="checkpoint directory")
    p.add_argument("--served-model-name", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--attention", default="torch_ref", choices=["torch_ref", "flash"])
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--num-blocks", type=int, default=None)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--max-num-seqs", type=int, default=256)
    p.add_argument("--max-num-batched-tokens", type=int, default=4096)
    p.add_argument(
        "--enable-mixed-batching",
        action="store_true",
        help="stall-free mixed batching: prefill chunks and decodes share each step under the "
        "--max-num-batched-tokens budget",
    )
    p.add_argument("--max-queued-requests", type=int, default=1024)
    p.add_argument(
        "--api-key-env",
        default="STALLFREE_API_KEY",
        help="environment variable holding the bearer key (unset: no auth)",
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    return p.parse_args(argv)


def engine_config(a: argparse.Namespace, eos_token_id: int | None) -> EngineConfig:
    return EngineConfig(
        model=a.model,
        num_blocks=a.num_blocks,
        block_size=a.block_size,
        dtype=a.dtype,
        device=a.device,
        attention=AttentionBackendName(a.attention),
        scheduler=SchedulerConfig(
            max_num_seqs=a.max_num_seqs,
            max_num_batched_tokens=a.max_num_batched_tokens,
            max_model_len=a.max_model_len,
            enable_mixed_batching=a.enable_mixed_batching,
            eos_token_id=eos_token_id,
        ),
    )


def main(argv: list[str] | None = None) -> None:
    import uvicorn

    from stallfree.engine import LLMEngine
    from stallfree.tokenizer import HFChatTokenizer

    a = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    tokenizer = HFChatTokenizer(a.model)
    cfg = engine_config(a, tokenizer.eos_token_id)
    engine = AsyncEngine(LLMEngine(cfg), max_queued_requests=a.max_queued_requests)
    app = create_app(
        engine,
        tokenizer,
        served_model_name=a.served_model_name or a.model,
        max_model_len=a.max_model_len,
        api_key=os.environ.get(a.api_key_env) or None,
    )
    uvicorn.run(app, host=a.host, port=a.port, log_level="info")


if __name__ == "__main__":
    main()
