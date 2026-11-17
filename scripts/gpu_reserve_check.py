"""Chat with a stallfree server through GPU Reserve's own model-provider code.

Run with GPU Reserve's Python, from anywhere:

    STALLFREE_URL=https://.../v1 STALLFREE_API_KEY=... \\
        <gpu-reserve>/backend/.venv/bin/python scripts/gpu_reserve_check.py <gpu-reserve>/backend

It builds the chat model exactly as GPU Reserve does for ASSISTANT_MODEL=vllm:<model>, with only
the base URL and key pointing at this server, and makes a plain chat call (no tools), once with
thinking off and once with thinking on, invoking and streaming. It runs from an empty temporary
directory so GPU Reserve's .env is never read, and prints no URL or key.
"""

from __future__ import annotations

import json
import os
import secrets
import sys
import tempfile

QUESTION = "What is the capital of France? Answer in one short sentence."


def main() -> None:
    backend = sys.argv[1]
    url, key = os.environ["STALLFREE_URL"], os.environ["STALLFREE_API_KEY"]
    sys.path.insert(0, backend)
    os.chdir(tempfile.mkdtemp())  # no .env here
    os.environ.update(
        ASSISTANT_MODEL="vllm:Qwen/Qwen3-0.6B",
        VLLM_BASE_URL=url,
        VLLM_API_KEY=key,
        JWT_SECRET=secrets.token_urlsafe(48),  # required by Settings; unused here
        ASSISTANT_REQUEST_TIMEOUT_SECONDS="300",
    )
    from app.core.config import Settings
    from app.llm.registry import build_chat_model

    results = []
    for thinking in ("false", "true"):
        os.environ["VLLM_ENABLE_THINKING"] = thinking
        model = build_chat_model(Settings())
        invoked = model.invoke([("user", QUESTION)])
        streamed = "".join(chunk.text for chunk in model.stream([("user", QUESTION)]))
        results.append(
            {
                "enable_thinking": thinking,
                "provider_class": type(model).__name__,
                "invoke_text": invoked.text,
                "invoke_usage": getattr(invoked, "usage_metadata", None),
                "stream_text": streamed,
                "reasoning_in_reply": "<think>" in invoked.text or "</think>" in invoked.text,
            }
        )
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
