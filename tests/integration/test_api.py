"""The HTTP server end to end on CPU: a real uvicorn server in a thread, the tiny model, a
byte-level tokenizer, and the official ``openai`` client."""

from __future__ import annotations

import asyncio
import contextlib
import json
import random
import socket
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import openai
import pytest
import uvicorn
from helpers.byte_tokenizer import ByteTokenizer
from helpers.tiny_qwen3 import hf_greedy
from transformers import Qwen3ForCausalLM as HFQwen3ForCausalLM

from stallfree.async_engine import AsyncEngine
from stallfree.config import EngineConfig, SchedulerConfig
from stallfree.engine import LLMEngine
from stallfree.server.api import create_app

KEY = "test-key"
MODEL = "tiny"
MAX_LEN = 256


@dataclass
class Server:
    url: str
    engine: AsyncEngine
    tokenizer: ByteTokenizer

    def client(self, key: str = KEY) -> openai.OpenAI:
        return openai.OpenAI(base_url=f"{self.url}/v1", api_key=key, max_retries=0)

    def async_client(self) -> openai.AsyncOpenAI:
        return openai.AsyncOpenAI(base_url=f"{self.url}/v1", api_key=KEY, max_retries=0)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@contextlib.contextmanager
def run_server(
    model_dir: Path, *, max_queued_requests: int = 64, eos_token_id: int = 0
) -> Iterator[Server]:
    cfg = EngineConfig(
        model=str(model_dir),
        num_blocks=128,
        block_size=4,
        dtype="float32",
        scheduler=SchedulerConfig(
            max_num_seqs=16,
            max_num_batched_tokens=512,
            max_model_len=MAX_LEN,
            watermark=0.0,
            eos_token_id=eos_token_id,
        ),
    )
    engine = AsyncEngine(LLMEngine(cfg), max_queued_requests=max_queued_requests)
    tok = ByteTokenizer()
    app = create_app(engine, tok, served_model_name=MODEL, max_model_len=MAX_LEN, api_key=KEY)
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 30
    while not server.started:
        assert time.monotonic() < deadline, "server did not start"
        time.sleep(0.02)
    try:
        yield Server(f"http://127.0.0.1:{port}", engine, tok)
    finally:
        server.should_exit = True
        thread.join(timeout=30)


@pytest.fixture(scope="module")
def server(tiny_model_dir: Path) -> Iterator[Server]:
    with run_server(tiny_model_dir) as s:
        yield s


def user(text: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": text}]


def expected(hf: HFQwen3ForCausalLM, srv: Server, text: str, n: int, **kw: Any) -> str:
    """What greedy decoding must produce: HF's continuation of the same prompt ids, as text."""
    ids = srv.tokenizer.encode_chat(user(text), kw)
    return srv.tokenizer.decode(hf_greedy(hf, ids, n))


GREEDY: dict[str, Any] = {"temperature": 0.0, "extra_body": {"ignore_eos": True}}


def wait_until(cond: Callable[[], bool], timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < deadline, "condition not reached"
        time.sleep(0.01)


# --- basics ----------------------------------------------------------------------------------


def test_health_and_models(server: Server) -> None:
    assert httpx.get(f"{server.url}/health").json() == {"status": "ok"}
    models = server.client().models.list()
    assert [m.id for m in models.data] == [MODEL]


def test_api_key_is_required_on_v1_only(server: Server) -> None:
    with pytest.raises(openai.AuthenticationError):
        server.client(key="wrong").models.list()
    r = httpx.get(f"{server.url}/v1/models")
    assert r.status_code == 401 and r.json()["error"]["type"] == "authentication_error"


def test_non_streaming_matches_hf(server: Server, tiny_hf_model: HFQwen3ForCausalLM) -> None:
    r = server.client().chat.completions.create(
        model=MODEL,
        messages=user("hello"),
        max_tokens=12,
        **GREEDY,  # type: ignore[arg-type]
    )
    assert r.choices[0].message.content == expected(tiny_hf_model, server, "hello", 12)
    assert r.choices[0].finish_reason == "length"
    prompt_len = len(server.tokenizer.encode_chat(user("hello"), {}))
    assert r.usage is not None
    assert (r.usage.prompt_tokens, r.usage.completion_tokens) == (prompt_len, 12)
    assert r.usage.total_tokens == prompt_len + 12


def test_streaming_equals_non_streaming(server: Server) -> None:
    client = server.client()
    args: dict[str, Any] = {
        "model": MODEL,
        "messages": user("stream me"),
        "max_tokens": 20,
        **GREEDY,
    }
    full = client.chat.completions.create(**args).choices[0].message.content
    chunks = list(
        client.chat.completions.create(**args, stream=True, stream_options={"include_usage": True})
    )
    assert chunks[0].choices[0].delta.role == "assistant"
    text = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
    assert text == full
    finishes = [
        c.choices[0].finish_reason for c in chunks if c.choices and c.choices[0].finish_reason
    ]
    assert finishes == ["length"]
    assert chunks[-1].choices == [] and chunks[-1].usage is not None
    assert chunks[-1].usage.completion_tokens == 20
    assert len({c.id for c in chunks}) == 1


def test_eight_concurrent_streams_are_each_correct(
    server: Server, tiny_hf_model: HFQwen3ForCausalLM
) -> None:
    texts = [f"request {i}: " + "xyz" * i for i in range(8)]
    lengths = [6 + 2 * i for i in range(8)]

    async def one(client: openai.AsyncOpenAI, text: str, n: int) -> tuple[str, set[str]]:
        stream = await client.chat.completions.create(
            model=MODEL,
            messages=user(text),
            max_tokens=n,
            stream=True,
            **GREEDY,  # type: ignore[arg-type]
        )
        parts, ids = [], set()
        async for chunk in stream:
            ids.add(chunk.id)
            if chunk.choices:
                parts.append(chunk.choices[0].delta.content or "")
        return "".join(parts), ids

    async def main() -> list[tuple[str, set[str]]]:
        client = server.async_client()
        pairs = zip(texts, lengths, strict=True)
        return await asyncio.gather(*(one(client, t, n) for t, n in pairs))

    results = asyncio.run(asyncio.wait_for(main(), 60))
    for (text, ids), prompt_text, n in zip(results, texts, lengths, strict=True):
        assert len(ids) == 1, "chunks from different responses mixed into one stream"
        assert text == expected(tiny_hf_model, server, prompt_text, n)
    assert len({next(iter(ids)) for _, ids in results}) == 8


def test_chat_template_kwargs_reach_the_template(server: Server) -> None:
    client = server.client()
    base = client.chat.completions.create(model=MODEL, messages=user("q"), max_tokens=1, **GREEDY)  # type: ignore[arg-type]
    off = client.chat.completions.create(
        model=MODEL,
        messages=user("q"),
        max_tokens=1,
        temperature=0.0,
        extra_body={"ignore_eos": True, "chat_template_kwargs": {"enable_thinking": False}},
    )
    assert base.usage is not None and off.usage is not None
    # The stub template appends "(no thinking) " (14 bytes) when thinking is turned off.
    assert off.usage.prompt_tokens - base.usage.prompt_tokens == len("(no thinking) ")


# --- errors ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "param"),
    [
        ({"stop": ["x"]}, "stop"),
        ({"seed": 1}, "seed"),
        ({"n": 2}, "n"),
        ({"tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}]}, "tools"),
    ],
)
def test_unsupported_parameters_get_a_400_naming_them(
    server: Server, kwargs: dict[str, Any], param: str
) -> None:
    with pytest.raises(openai.BadRequestError) as err:
        server.client().chat.completions.create(model=MODEL, messages=user("x"), **kwargs)
    body = err.value.body
    assert isinstance(body, dict) and param in body["message"] and body["param"] == param


def test_other_errors_are_openai_shaped(server: Server) -> None:
    client = server.client()
    with pytest.raises(openai.NotFoundError):
        client.chat.completions.create(model="other", messages=user("x"))
    with pytest.raises(openai.BadRequestError):  # longer than max_model_len
        client.chat.completions.create(model=MODEL, messages=user("x"), max_tokens=MAX_LEN)
    r = httpx.post(
        f"{server.url}/v1/chat/completions",
        headers={"authorization": f"Bearer {KEY}"},
        json={"model": MODEL, "messages": [], "max_tokens": 0},
    )
    assert r.status_code == 400
    assert set(r.json()["error"]) == {"message", "type", "param", "code"}


# --- disconnect and capacity ------------------------------------------------------------------


@contextlib.contextmanager
def slow_steps(srv: Server, seconds: float) -> Iterator[None]:
    engine = srv.engine.engine
    real = engine.step

    def slow() -> list[Any]:
        time.sleep(seconds)
        return real()

    engine.step = slow  # type: ignore[method-assign]
    try:
        yield
    finally:
        engine.step = real  # type: ignore[method-assign]


def test_client_disconnect_aborts_and_frees_blocks(server: Server) -> None:
    eng = server.engine.engine
    with slow_steps(server, 0.02):  # 200 tokens would take ~4 s
        with httpx.stream(
            "POST",
            f"{server.url}/v1/chat/completions",
            headers={"authorization": f"Bearer {KEY}"},
            json={
                "model": MODEL,
                "messages": user("long"),
                "max_tokens": 200,
                "stream": True,
                "temperature": 0.0,
                "ignore_eos": True,
            },
            timeout=30,
        ) as r:
            lines = 0
            for line in r.iter_lines():
                if line.startswith("data: "):
                    lines += 1
                if lines >= 4:
                    break  # leaving the block closes the connection
        steps_at_close = eng.num_steps
        wait_until(lambda: not eng.has_unfinished() and server.engine.num_active == 0)
    assert eng.block_manager.num_free_blocks == eng.block_manager.num_blocks
    assert eng.num_steps - steps_at_close < 50, "generation kept going after the disconnect"


def test_full_server_answers_503(tiny_model_dir: Path) -> None:
    with run_server(tiny_model_dir, max_queued_requests=1) as srv, slow_steps(srv, 0.05):
        holder = threading.Thread(
            target=lambda: srv.client().chat.completions.create(
                model=MODEL, messages=user("hold"), max_tokens=20, **GREEDY
            )
        )  # type: ignore[arg-type]
        holder.start()
        wait_until(lambda: srv.engine.num_active == 1)
        with pytest.raises(openai.InternalServerError) as err:
            srv.client().chat.completions.create(model=MODEL, messages=user("x"), max_tokens=2)
        assert err.value.status_code == 503
        holder.join(timeout=30)
        assert (
            srv.client()
            .chat.completions.create(
                model=MODEL,
                messages=user("x"),
                max_tokens=2,
                **GREEDY,  # type: ignore[arg-type]
            )
            .usage
            is not None
        )


def test_metrics_lines_are_json(server: Server, caplog: pytest.LogCaptureFixture) -> None:
    import logging

    with caplog.at_level(logging.INFO, logger="stallfree.metrics"):
        server.client().chat.completions.create(
            model=MODEL,
            messages=user("m"),
            max_tokens=3,
            **GREEDY,  # type: ignore[arg-type]
        )
        wait_until(lambda: any(r.name == "stallfree.metrics" for r in caplog.records))
    line = next(r for r in caplog.records if r.name == "stallfree.metrics")
    record = json.loads(line.getMessage())
    assert record["completion_tokens"] == 3 and "ttft_s" in record


def test_random_prompts_round_trip(server: Server, tiny_hf_model: HFQwen3ForCausalLM) -> None:
    rng = random.Random(0)
    for _ in range(3):
        text = "".join(rng.choice("abc 你好 😀") for _ in range(rng.randint(1, 20)))
        r = server.client().chat.completions.create(
            model=MODEL,
            messages=user(text),
            max_tokens=8,
            **GREEDY,  # type: ignore[arg-type]
        )
        assert r.choices[0].message.content == expected(tiny_hf_model, server, text, 8)


def test_ignore_eos_is_honoured(tiny_model_dir: Path, tiny_hf_model: HFQwen3ForCausalLM) -> None:
    """Make the model's actual first greedy token the end-of-sequence token: a request stops
    there unless it sets ignore_eos."""
    tok = ByteTokenizer()
    first = hf_greedy(tiny_hf_model, tok.encode_chat(user("hello"), {}), 1)[0]
    with run_server(tiny_model_dir, eos_token_id=first) as srv:
        client = srv.client()
        stopped = client.chat.completions.create(
            model=MODEL, messages=user("hello"), max_tokens=12, temperature=0.0
        )
        ignored = client.chat.completions.create(
            model=MODEL,
            messages=user("hello"),
            max_tokens=12,
            temperature=0.0,
            extra_body={"ignore_eos": True},
        )
    assert stopped.choices[0].finish_reason == "stop"
    assert stopped.usage is not None and stopped.usage.completion_tokens == 1
    assert ignored.choices[0].finish_reason == "length"
    assert ignored.usage is not None and ignored.usage.completion_tokens == 12


def test_full_server_answers_503_before_streaming(tiny_model_dir: Path) -> None:
    """At capacity, a streaming request gets a 503 status, not a 200 stream carrying an error."""
    with run_server(tiny_model_dir, max_queued_requests=1) as srv, slow_steps(srv, 0.05):
        holder = threading.Thread(
            target=lambda: srv.client().chat.completions.create(
                model=MODEL, messages=user("hold"), max_tokens=20, **GREEDY
            )
        )  # type: ignore[arg-type]
        holder.start()
        wait_until(lambda: srv.engine.num_active == 1)
        r = httpx.post(
            f"{srv.url}/v1/chat/completions",
            headers={"authorization": f"Bearer {KEY}"},
            json={"model": MODEL, "messages": user("x"), "max_tokens": 2, "stream": True},
            timeout=30,
        )
        assert r.status_code == 503
        assert r.json()["error"]["type"] == "server_error"
        holder.join(timeout=30)
