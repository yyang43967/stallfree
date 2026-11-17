"""Serve Qwen3-0.6B on a Modal GPU behind the OpenAI-compatible API.

    STALLFREE_API_KEY=<key> uv run modal serve modal_apps/serve.py    # temporary URL, stops on exit
    STALLFREE_API_KEY=<key> uv run modal deploy modal_apps/serve.py   # stays until `modal app stop`

The bearer key is read from the local environment when the app is created and passed to the
container as an ephemeral secret; no named secret is stored in the workspace. One container at a
time, scaled to zero after 60 idle seconds.
"""

from __future__ import annotations

import os
from typing import Any

import modal

from modal_apps.common import GPU, MODEL_DIR, MODEL_REPO, WEIGHTS_DIR, gpu_image, weights_volume

app = modal.App("stallfree-serve")
API_KEY_SECRET = modal.Secret.from_dict(
    {"STALLFREE_API_KEY": os.environ.get("STALLFREE_API_KEY", "")}
)


@app.function(
    image=gpu_image,
    gpu=GPU,
    volumes={WEIGHTS_DIR: weights_volume},
    secrets=[API_KEY_SECRET],
    cpu=4.0,
    memory=16384,
    timeout=3600,
    scaledown_window=60,
    max_containers=1,
)
@modal.concurrent(max_inputs=256)
@modal.asgi_app()
def api() -> Any:
    import logging

    from stallfree.async_engine import AsyncEngine
    from stallfree.config import AttentionBackendName, EngineConfig, SchedulerConfig
    from stallfree.engine import LLMEngine
    from stallfree.server.api import create_app
    from stallfree.tokenizer import HFChatTokenizer

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    key = os.environ.get("STALLFREE_API_KEY") or None
    tokenizer = HFChatTokenizer(MODEL_DIR)
    cfg = EngineConfig(
        model=MODEL_DIR,
        num_blocks=None,
        dtype="bfloat16",
        device="cuda",
        attention=AttentionBackendName.FLASH,
        scheduler=SchedulerConfig(max_model_len=4096, eos_token_id=tokenizer.eos_token_id),
    )
    engine = AsyncEngine(LLMEngine(cfg))
    return create_app(
        engine, tokenizer, served_model_name=MODEL_REPO, max_model_len=4096, api_key=key
    )
