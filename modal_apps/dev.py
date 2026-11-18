"""Development runs on a Modal GPU (L4, or A10 when no L4 is free).

    uv run modal run -m modal_apps.dev::download        # CPU only: weights into the volume
    uv run modal run -m modal_apps.dev::smoke           # versions, cache sizing, one generation
    uv run modal run -m modal_apps.dev::gpu_tests       # pytest -m gpu tests/gpu
    uv run modal run -m modal_apps.dev::calibrate       # bf16 noise data -> docs/data/
    uv run modal run -m modal_apps.dev::phase3_gpu      # chunked margin suite + micro-benchmark

Every GPU function reports which GPU ran and how long the function body took.
"""

from __future__ import annotations

import json
import os
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
    report = Path("/tmp/margin_report.json")
    cmd = ["python", "-m", "pytest", "-q", "-s", "-m", "gpu", "-p", "no:cacheprovider", "tests/gpu"]
    if select:
        cmd += ["-k", select]
    env = {**os.environ, "STALLFREE_REPORT": str(report)}
    proc = subprocess.run(cmd, cwd="/root", capture_output=True, text=True, env=env)
    return {
        "gpu": _gpu_name(),
        "returncode": proc.returncode,
        "output": proc.stdout[-12000:],
        "stderr": proc.stderr[-3000:],
        "margin_report": json.loads(report.read_text()) if report.exists() else None,
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


@app.function(**GPU_FN, timeout=2400)  # type: ignore[arg-type]
def run_calibration_and_margin(mixed_budget: int = 0) -> dict[str, Any]:
    """Calibration and the margin suite in one container, so both run on the same GPU.
    ``mixed_budget`` > 0 runs the suite with mixed batching on and that token budget."""
    return _calibration_and_margin(mixed_budget)


def _calibration_and_margin(mixed_budget: int) -> dict[str, Any]:
    import sys

    start = time.time()
    sys.path.insert(0, "/root")
    from tests.gpu.calibration import run_calibration as calibrate_fn

    calibration = calibrate_fn(MODEL_DIR)
    cal_path = Path("/tmp/tau_calibration.json")
    cal_path.write_text(json.dumps(calibration))
    report = Path("/tmp/margin_report.json")
    env = {**os.environ, "STALLFREE_REPORT": str(report), "STALLFREE_CALIBRATION": str(cal_path)}
    if mixed_budget:
        env["STALLFREE_MIXED_BUDGET"] = str(mixed_budget)
    cmd = [
        "python",
        "-m",
        "pytest",
        "-q",
        "-s",
        "-m",
        "gpu",
        "-p",
        "no:cacheprovider",
        "tests/gpu",
        "-k",
        "margin",
    ]
    proc = subprocess.run(cmd, cwd="/root", capture_output=True, text=True, env=env)
    return {
        "gpu": _gpu_name(),
        "calibration": calibration,
        "margin_report": json.loads(report.read_text()) if report.exists() else None,
        "returncode": proc.returncode,
        "output": proc.stdout[-8000:],
        "seconds": round(time.time() - start, 1),
    }


def _microbench(budgets: list[int | None], repeats: int, commit: str) -> dict[str, Any]:
    """The stall micro-benchmark in this container (see benchmarks/microbench.py)."""
    import sys
    from dataclasses import asdict
    from importlib.metadata import version

    sys.path.insert(0, "/root")
    import torch

    from benchmarks.microbench import Scenario, run_suite
    from stallfree.config import AttentionBackendName, EngineConfig, SchedulerConfig
    from stallfree.engine import LLMEngine

    start = time.time()
    engine = LLMEngine(
        EngineConfig(
            model=MODEL_DIR,
            num_blocks=None,
            dtype="bfloat16",
            device="cuda",
            attention=AttentionBackendName.FLASH,
            scheduler=SchedulerConfig(max_model_len=4096),
        )
    )
    scenario = Scenario()
    runs = run_suite(engine, budgets, repeats, scenario, torch.cuda.synchronize)
    return {
        "meta": {
            "gpu": _gpu_name(),
            "model": MODEL_REPO,
            "model_revision": MODEL_REVISION,
            "dtype": "bfloat16",
            "attention": "flash",
            "block_size": engine.config.block_size,
            "num_blocks": engine.block_manager.num_blocks,
            "off_scheduler": asdict(engine.config.scheduler),
            "torch": torch.__version__,
            "flash_attn": version("flash-attn"),
            "commit": commit,
            "seconds": round(time.time() - start, 1),
        },
        "scenario": asdict(scenario),
        "budgets": budgets,
        "repeats": repeats,
        "runs": runs,
    }


@app.function(**GPU_FN, timeout=2400)  # type: ignore[arg-type]
def run_phase3_gpu(
    margin_budget: int, budgets: list[int | None], repeats: int, commit: str
) -> dict[str, Any]:
    """The chunked margin suite (with calibration) and the stall micro-benchmark, in one container
    so both report the same GPU."""
    margin = _calibration_and_margin(margin_budget)
    return {"margin": margin, "microbench": _microbench(budgets, repeats, commit)}


@app.function(**GPU_FN, timeout=1800)  # type: ignore[arg-type]
def run_concurrency() -> dict[str, Any]:
    import sys

    start = time.time()
    sys.path.insert(0, "/root")
    from tests.gpu.concurrency import run_concurrency_check

    result = run_concurrency_check(MODEL_DIR)
    result["seconds"] = round(time.time() - start, 1)
    return result


@app.local_entrypoint()
def download() -> None:
    print(download_weights.remote())


@app.local_entrypoint()
def smoke() -> None:
    print(json.dumps(smoke_run.remote(), indent=2))


@app.local_entrypoint()
def gpu_tests(select: str = "", gpu: str = "") -> None:
    """``--gpu A10`` pins one GPU type (calibration and the margin suite must share one)."""
    fn = run_gpu_tests.with_options(gpu=gpu) if gpu else run_gpu_tests
    r = fn.remote(select)
    if r.get("margin_report"):
        out = Path("docs/data/margin_suite.json")
        out.write_text(json.dumps(r["margin_report"], indent=2) + "\n")
        print(f"wrote {out}")
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


@app.local_entrypoint()
def calibrate_and_margin(gpu: str = "A10", mixed_budget: int = 0) -> None:
    fn = run_calibration_and_margin.with_options(gpu=gpu)
    r = fn.remote(mixed_budget)
    suffix = f"_mixed{mixed_budget}" if mixed_budget else ""
    Path(f"docs/data/tau_calibration{suffix}.json").write_text(
        json.dumps(r["calibration"], indent=2) + "\n"
    )
    if r["margin_report"]:
        Path(f"docs/data/margin_suite{suffix}.json").write_text(
            json.dumps(r["margin_report"], indent=2) + "\n"
        )
    print(r["output"])
    print(f"gpu={r['gpu']} returncode={r['returncode']} seconds={r['seconds']}")


@app.local_entrypoint()
def concurrency(gpu: str = "") -> None:
    fn = run_concurrency.with_options(gpu=gpu) if gpu else run_concurrency
    r = fn.remote()
    Path("docs/data/concurrency_gpu.json").write_text(json.dumps(r, indent=2) + "\n")
    ok = all(x["passes"] and x["one_response_id"] and x["usage_ok"] for x in r["results"])
    print(
        f"gpu={r['gpu']} clients={r['clients']} distinct ids={r['distinct_response_ids']} "
        f"all checks pass={ok} seconds={r['seconds']}"
    )
    for x in r["results"]:
        print(x)


def _git_commit() -> str:
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True
    )
    return head.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")


@app.local_entrypoint()
def phase3_gpu(gpu: str = "A10", margin_budget: int = 16, repeats: int = 3) -> None:
    """Margin suite with every prompt chunked, then the micro-benchmark (off, on-256/512/1024)."""
    budgets: list[int | None] = [None, 256, 512, 1024]
    r = run_phase3_gpu.with_options(gpu=gpu).remote(margin_budget, budgets, repeats, _git_commit())
    m = r["margin"]
    suffix = f"_mixed{margin_budget}"
    Path(f"docs/data/tau_calibration{suffix}.json").write_text(
        json.dumps(m["calibration"], indent=2) + "\n"
    )
    if m["margin_report"]:
        Path(f"docs/data/margin_suite{suffix}.json").write_text(
            json.dumps(m["margin_report"], indent=2) + "\n"
        )
    print(m["output"])
    print(f"margin: gpu={m['gpu']} returncode={m['returncode']} seconds={m['seconds']}")
    mb = r["microbench"]
    gpu_tag = mb["meta"]["gpu"].replace("NVIDIA ", "").replace(" ", "_").lower()
    out = Path("benchmarks/data/microbench")
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"stall_{gpu_tag}_{mb['meta']['commit']}.json"
    path.write_text(json.dumps(mb, indent=1) + "\n")
    print(f"microbench: gpu={mb['meta']['gpu']} seconds={mb['meta']['seconds']} -> {path}")
