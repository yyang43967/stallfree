"""Development runs on a Modal GPU (L4, or A10 when no L4 is free).

    uv run modal run -m modal_apps.dev::download        # CPU only: weights into the volume
    uv run modal run -m modal_apps.dev::smoke           # versions, cache sizing, one generation
    uv run modal run -m modal_apps.dev::gpu_tests       # pytest -m gpu tests/gpu
    uv run modal run -m modal_apps.dev::calibrate       # bf16 noise data -> docs/data/

Every GPU function reports which GPU ran and how long the function body took.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

import modal

from modal_apps.common import (
    GPU,
    MODEL_DIR,
    MODEL_REPO,
    MODEL_REVISION,
    WEIGHTS_DIR,
    download_image,
    gpu_image,
    weights_volume,
)

app = modal.App("stallfree-dev")
GPU_FN = {
    "image": gpu_image,
    "gpu": GPU,
    "volumes": {WEIGHTS_DIR: weights_volume},
    "cpu": 4.0,
    "memory": 16384,
}


@app.function(
    image=download_image, volumes={WEIGHTS_DIR: weights_volume}, cpu=1.0, memory=2048, timeout=1800
)
def download_weights() -> str:
    from huggingface_hub import snapshot_download

    start = time.time()
    path = snapshot_download(
        MODEL_REPO,
        revision=MODEL_REVISION,
        local_dir=MODEL_DIR,
        allow_patterns=["*.json", "*.safetensors", "*.txt"],
    )
    weights_volume.commit()
    files = sorted(p.name for p in Path(path).iterdir())
    return f"{path}: {files} ({time.time() - start:.0f} s)"


def _gpu_name() -> str:
    import torch

    return torch.cuda.get_device_name()


@app.function(**GPU_FN, timeout=900)  # type: ignore[arg-type]
def smoke_run() -> dict[str, Any]:
    start = time.time()
    import flash_attn
    import torch
    from transformers import AutoTokenizer

    from stallfree.config import AttentionBackendName, EngineConfig, SchedulerConfig
    from stallfree.engine import LLMEngine
    from stallfree.sequence import SamplingParams

    driver = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    cfg = EngineConfig(
        model=MODEL_DIR,
        num_blocks=None,
        dtype="bfloat16",
        device="cuda",
        attention=AttentionBackendName.FLASH,
        scheduler=SchedulerConfig(eos_token_id=tok.convert_tokens_to_ids("<|im_end|>")),
    )
    engine = LLMEngine(cfg)
    prompts = ["What is the capital of France?", "Write a haiku about a kitchen."]
    ids = [
        tok.encode(
            tok.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            ),
            add_special_tokens=False,
        )
        for p in prompts
    ]
    outs = engine.generate(ids, SamplingParams(max_tokens=40))
    return {
        "gpu": _gpu_name(),
        "driver": driver,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "flash_attn": flash_attn.__version__,
        "memory": engine.memory_report.summary() if engine.memory_report else None,
        "outputs": [tok.decode(o) for o in outs],
        "seconds": round(time.time() - start, 1),
    }


@app.function(**GPU_FN, timeout=1800)  # type: ignore[arg-type]
def run_gpu_tests(select: str = "") -> dict[str, Any]:
    start = time.time()
    cmd = ["python", "-m", "pytest", "-q", "-s", "-m", "gpu", "-p", "no:cacheprovider", "tests/gpu"]
    if select:
        cmd += ["-k", select]
    proc = subprocess.run(cmd, cwd="/root", capture_output=True, text=True)
    return {
        "gpu": _gpu_name(),
        "returncode": proc.returncode,
        "output": proc.stdout[-12000:],
        "stderr": proc.stderr[-3000:],
        "seconds": round(time.time() - start, 1),
    }


@app.function(**GPU_FN, timeout=1800)  # type: ignore[arg-type]
def run_calibration() -> dict[str, Any]:
    import sys

    start = time.time()
    sys.path.insert(0, "/root")
    from tests.gpu.calibration import run_calibration as calibrate_fn

    result = calibrate_fn(MODEL_DIR)
    result["seconds"] = round(time.time() - start, 1)
    return result


@app.local_entrypoint()
def download() -> None:
    print(download_weights.remote())


@app.local_entrypoint()
def smoke() -> None:
    print(json.dumps(smoke_run.remote(), indent=2))


@app.local_entrypoint()
def gpu_tests(select: str = "") -> None:
    r = run_gpu_tests.remote(select)
    print(r["output"])
    print(r["stderr"])
    print(f"gpu={r['gpu']} returncode={r['returncode']} seconds={r['seconds']}")


@app.local_entrypoint()
def calibrate() -> None:
    r = run_calibration.remote()
    out = Path("docs/data/tau_calibration.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r, indent=2) + "\n")
    print(f"wrote {out}; gpu={r['gpu']} seconds={r['seconds']}")
