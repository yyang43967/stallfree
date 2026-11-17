"""Eight concurrent streaming clients against the real server on a GPU (flash, bf16, Qwen3-0.6B).

Each stream is checked structurally (one response id, a finish reason, usage that matches) and
for content: the streamed text, re-tokenized, is compared with Hugging Face's bf16 greedy output
for the same chat under the margin rule (identical, or first difference where the reference's
top-1/top-2 gap is at most tau).
"""

from __future__ import annotations

import asyncio
import socket
import threading
import time
from typing import Any

import torch
import uvicorn

from stallfree.async_engine import AsyncEngine
from stallfree.config import AttentionBackendName, EngineConfig, SchedulerConfig
from stallfree.engine import LLMEngine
from stallfree.server.api import create_app
from stallfree.tokenizer import HFChatTokenizer
from tests.gpu.calibration import MAX_NEW, chat_ids, first_diff, greedy, load, reference_gaps
from tests.gpu.margin import BF16_MARGIN_TAU, EOS
from tests.gpu.prompts import PROMPTS

KEY = "concurrency-test"
NAME = "Qwen/Qwen3-0.6B"
CLIENTS = 8


def _port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def run_concurrency_check(model_dir: str) -> dict[str, Any]:
    import openai

    tok = HFChatTokenizer(model_dir)
    hf = load(model_dir, torch.bfloat16, "eager")
    prompts = [chat_ids(tok._tok, p) for p in PROMPTS[:CLIENTS]]
    refs = greedy(hf, tok._tok, prompts, 1, EOS)
    gaps = [reference_gaps(hf, p, r) for p, r in zip(prompts, refs, strict=True)]

    cfg = EngineConfig(
        model=model_dir,
        num_blocks=None,
        dtype="bfloat16",
        device="cuda",
        attention=AttentionBackendName.FLASH,
        scheduler=SchedulerConfig(max_model_len=4096, eos_token_id=tok.eos_token_id),
    )
    engine = AsyncEngine(LLMEngine(cfg))
    app = create_app(engine, tok, served_model_name=NAME, max_model_len=4096, api_key=KEY)
    port = _port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.05)

    async def one(client: Any, text: str) -> dict[str, Any]:
        start = time.monotonic()
        stream = await client.chat.completions.create(
            model=NAME,
            messages=[{"role": "user", "content": text}],
            max_tokens=MAX_NEW,
            temperature=0.0,
            stream=True,
            stream_options={"include_usage": True},
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        parts: list[str] = []
        ids: set[str] = set()
        finish = None
        usage = None
        first = None
        async for chunk in stream:
            ids.add(chunk.id)
            if chunk.usage is not None:
                usage = chunk.usage.model_dump()
            for c in chunk.choices:
                if c.delta.content:
                    first = first or time.monotonic() - start
                    parts.append(c.delta.content)
                finish = c.finish_reason or finish
        return {
            "text": "".join(parts),
            "ids": sorted(ids),
            "finish": finish,
            "usage": usage,
            "ttft_client_s": first,
            "e2e_client_s": time.monotonic() - start,
        }

    async def all_clients() -> list[dict[str, Any]]:
        client = openai.AsyncOpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key=KEY)
        return await asyncio.gather(*(one(client, p) for p in PROMPTS[:CLIENTS]))

    results = asyncio.run(asyncio.wait_for(all_clients(), 300))
    server.should_exit = True
    thread.join(timeout=60)

    rows = []
    for i, (res, ref) in enumerate(zip(results, refs, strict=True)):
        ours = tok._tok.encode(res["text"], add_special_tokens=False)
        ref_text_ids = [t for t in ref if t not in EOS]
        ref_stopped = bool(ref) and ref[-1] in EOS
        exact = ours == ref_text_ids and (res["finish"] == "stop") == ref_stopped
        t = None if exact else first_diff(ref_text_ids, ours)
        gap = gaps[i][t] if t is not None and t < len(gaps[i]) else None
        usage_ok = bool(res["usage"]) and res["usage"]["prompt_tokens"] == len(prompts[i])
        rows.append(
            {
                "prompt": i,
                "exact": exact,
                "position": t,
                "gap": gap,
                "passes": exact or (gap is not None and gap <= BF16_MARGIN_TAU),
                "one_response_id": len(res["ids"]) == 1,
                "finish": res["finish"],
                "usage_ok": usage_ok,
                "ref_len": len(ref),
                "our_tokens": len(ours),
                "ttft_client_s": res["ttft_client_s"],
                "e2e_client_s": res["e2e_client_s"],
            }
        )
    return {
        "gpu": torch.cuda.get_device_name(),
        "clients": CLIENTS,
        "tau": BF16_MARGIN_TAU,
        "distinct_response_ids": len({r["ids"][0] for r in results if r["ids"]}),
        "results": rows,
    }
