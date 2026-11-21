"""The serving benchmark on one Modal L40S per container: stallfree (mixed batching off and on)
and vLLM (default and --enforce-eager), each alone in its own container, with the load generator
in the same container against localhost.

    uv run modal run -m modal_apps.bench::saturation               # closed loop, stallfree-off
    uv run modal run -m modal_apps.bench::dryrun --rate 0.8        # all 4 systems, 1 repeat
    uv run modal run -m modal_apps.bench --system vllm --rates 0.2,0.5,0.8,1.0,1.2 --repeats 3

Rates are multiples of stallfree-off's saturation throughput, read from
``benchmarks/data/saturation/saturation.json``. Traces are generated locally from the seed (with
the pinned tokenizer) and sent to the containers, so every system replays identical requests;
each run records the trace's and the source text's SHA-256. Results are written to
``benchmarks/data/[<group>/]<system>/<label>/seed<k>.*`` (see benchmarks/driver.py).
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import gzip
import json
import os
import platform
import time
from importlib import metadata
from pathlib import Path
from typing import Any

import modal

from modal_apps.common import (
    MODEL_DIR,
    MODEL_REPO,
    MODEL_REVISION,
    WEIGHTS_DIR,
    git_commit,
    gpu_image,
    weights_volume,
)

app = modal.App("stallfree-bench")

BENCH_GPU = "L40S"
CPU = 8.0
MEMORY_MIB = 32768
DATA = Path("benchmarks/data")
SATURATION_FILE = DATA / "saturation" / "saturation.json"
SATURATION_CONCURRENCY = 256  # = max_num_seqs: every client's request can run at once
SATURATION_TRACE_REQUESTS = 3000  # the closed loop cycles through these
SEEDS = (1, 2, 3)  # repeat k uses SEEDS[k]; the same seeds for every system

VLLM_VERSION = "0.33.0"
# vLLM's official server image, so vLLM runs with everything its defaults expect (its FlashInfer
# sampler JIT-compiles kernels with nvcc, which a slim pip-only image lacks). Python 3.12 inside.
vllm_image = (
    modal.Image.from_registry(f"vllm/vllm-openai:v{VLLM_VERSION}")
    .entrypoint([])
    .run_commands(
        "ln -sf $(command -v python3) /usr/local/bin/python",
        "python -m pip install httpx==0.28.1 psutil==7.2.2"
        " || uv pip install --system httpx==0.28.1 psutil==7.2.2",
    )
    .env({"HF_HUB_OFFLINE": "1", "VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1"})
    .workdir("/root")  # the image's /vllm-workspace has its own `benchmarks` package
    .add_local_python_source("benchmarks")
)

RESOURCES: dict[str, Any] = {
    "gpu": BENCH_GPU,
    "cpu": CPU,
    "memory": MEMORY_MIB,
    "volumes": {WEIGHTS_DIR: weights_volume},
    "timeout": 3 * 3600,
}


def _versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for pkg in ("torch", "flash-attn", "vllm", "transformers", "tokenizers", "httpx", "psutil"):
        with contextlib.suppress(metadata.PackageNotFoundError):
            out[pkg] = metadata.version(pkg)
    return out


def _vllm_flag_check(cmd: list[str]) -> dict[str, bool]:
    """Every ``--flag`` in the server command must appear in ``vllm serve --help=all``, so a
    renamed or removed flag stops the run instead of being silently ignored."""
    import subprocess

    proc = subprocess.run(["vllm", "serve", "--help=all"], capture_output=True, text=True)
    text = proc.stdout + proc.stderr
    flags = {a: a in text for a in cmd if a.startswith("--")}
    missing = [f for f, ok in flags.items() if not ok]
    if proc.returncode != 0 or missing:
        keys = [f.removeprefix("--no-").removeprefix("--") for f in missing]
        lines = [ln.strip() for ln in text.splitlines() if any(k in ln for k in keys)]
        raise RuntimeError(
            f"vllm serve --help=all: rc={proc.returncode}, missing {missing}; "
            f"related help lines: {lines[:20]}"
        )
    return flags


def _run_system(
    system: str, points: list[dict[str, Any]], traces: dict[int, bytes], meta: dict[str, Any]
) -> dict[str, bytes]:
    """Start ``system``'s server once, run every point against it, return the files written
    (path relative to the data directory -> bytes)."""
    from benchmarks import driver
    from benchmarks.workload import loads, trace_sha256

    work = Path("/tmp/bench")
    out_root = work / "out"
    out_root.mkdir(parents=True, exist_ok=True)
    trace_paths, trace_hashes = {}, {}
    for seed, blob in traces.items():
        path = work / f"trace_seed{seed}.jsonl.gz"
        path.write_bytes(blob)
        trace_paths[seed] = path
        trace_hashes[seed] = trace_sha256(loads(blob))
    sys_enum = driver.System(system)
    cmd = driver.server_command(sys_enum, MODEL_DIR)
    flag_check = _vllm_flag_check(cmd) if system.startswith("vllm") else None
    base_url = f"http://127.0.0.1:{driver.PORT}/v1"
    log_path = work / "server.log"
    tag = meta.get("container_tag", "")
    start = time.time()
    common = {
        **meta,
        "system": system,
        "server_command": cmd,
        "model": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "versions": _versions(),
        "cpu_request": CPU,
        "memory_mib": MEMORY_MIB,
        "gpu_requested": BENCH_GPU,
        "host_cpu_count": os.cpu_count(),
        "vllm_flag_check": flag_check,
    }
    try:
        with driver.server(cmd, log_path, base_url, env={**os.environ}) as proc:
            common["server_startup_s"] = round(time.time() - start, 1)
            for p in points:
                seed = p["seed"]
                out = out_root / system / p["label"] / f"seed{seed}.jsonl"
                if p["mode"] == "open":
                    args = ["--mode", "open", "--rate", repr(p["rate"])]
                else:
                    args = ["--mode", "closed", "--concurrency", str(p["concurrency"])]
                run_meta = {
                    **common,
                    **p,
                    "trace_seed": seed,
                    "trace_sha256": trace_hashes[seed],
                }
                driver.run_point(
                    base_url=base_url,
                    server_pid=proc.pid,
                    trace_path=trace_paths[seed],
                    out=out,
                    loadgen_args=args,
                    run_meta=run_meta,
                )
    except Exception as exc:
        log = log_path.read_text(errors="replace") if log_path.exists() else ""
        errors = [ln for ln in log.splitlines() if "ERROR" in ln or "Error" in ln][-30:]
        raise RuntimeError(
            f"{type(exc).__name__}: {exc}\n--- server log error lines ---\n"
            + "\n".join(errors)
            + f"\n--- server log tail ---\n{log[-1500:]}"
        ) from exc
    finally:
        if log_path.exists():
            log_out = out_root / system / f"server{tag}.log.gz"
            log_out.parent.mkdir(parents=True, exist_ok=True)
            log_out.write_bytes(gzip.compress(log_path.read_bytes()))
    files = [f for f in out_root.rglob("*") if f.is_file()]
    return {str(f.relative_to(out_root)): f.read_bytes() for f in files}


@app.function(image=gpu_image, **RESOURCES)
def run_stallfree(
    system: str, points: list[dict[str, Any]], traces: dict[int, bytes], meta: dict[str, Any]
) -> dict[str, bytes]:
    return _run_system(system, points, traces, meta)


@app.function(image=vllm_image, **RESOURCES)
def run_vllm(
    system: str, points: list[dict[str, Any]], traces: dict[int, bytes], meta: dict[str, Any]
) -> dict[str, bytes]:
    return _run_system(system, points, traces, meta)


def _runner(system: str) -> modal.Function:
    return run_vllm if system.startswith("vllm") else run_stallfree


# --- local side -------------------------------------------------------------------------------


def _traces(seeds: list[int], num_requests: int) -> dict[int, bytes]:
    from benchmarks.workload import TEXT_PATH, HFWorkloadTokenizer, dumps, make_trace

    tok = HFWorkloadTokenizer(MODEL_REPO, revision=MODEL_REVISION)
    text = TEXT_PATH.read_text()
    return {s: dumps(make_trace(tok, text, s, num_requests)) for s in seeds}


def _meta(group: str) -> dict[str, Any]:
    from benchmarks.workload import TEXT_SHA256, Mixture, Window

    return {
        "group": group,
        "commit": git_commit(),
        "text_sha256": TEXT_SHA256,
        "mixture": Mixture().__dict__,
        "window": Window().__dict__,
        "local_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def _write(files: dict[str, bytes], group: str) -> list[Path]:
    root = DATA / group if group else DATA
    written = []
    for rel, blob in sorted(files.items()):
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
        written.append(path)
    return written


def _saturation_rate() -> float:
    return float(json.loads(SATURATION_FILE.read_text())["request_throughput"])


def _label(mult: float) -> str:
    return f"r{mult:g}x"


@app.local_entrypoint()
def saturation(concurrency: int = SATURATION_CONCURRENCY, seed: int = SEEDS[0]) -> None:
    """Closed-loop saturation throughput of stallfree-off; writes saturation.json."""
    traces = _traces([seed], SATURATION_TRACE_REQUESTS)
    point = {"mode": "closed", "label": f"c{concurrency}", "concurrency": concurrency, "seed": seed}
    files = run_stallfree.remote("stallfree-off", [point], traces, _meta("saturation"))
    _write(files, "saturation")
    summary_rel = f"stallfree-off/c{concurrency}/seed{seed}.summary.json"
    summary = json.loads(files[summary_rel])
    result = {
        "system": "stallfree-off",
        "concurrency": concurrency,
        "seed": seed,
        "request_throughput": summary["request_throughput"],
        "output_token_throughput_completed": summary["output_token_throughput_completed"],
        "completed_in_window": summary["completed_in_window"],
        "measure_s": summary["measure_s"],
        "source": summary_rel,
    }
    SATURATION_FILE.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))


def _sweep(
    systems: list[str], mults: list[float], seeds: list[int], group: str
) -> dict[str, list[Path]]:
    """One container per (system, seed); each runs every rate in order. Containers run in
    parallel; each server runs alone in its own container."""
    from benchmarks.workload import requests_needed

    sat = _saturation_rate()
    rates = {m: m * sat for m in mults}
    traces = _traces(seeds, requests_needed(max(rates.values())))
    meta = {**_meta(group), "saturation_rate": sat}
    calls = []
    for system in systems:
        for seed in seeds:
            points = [
                {
                    "mode": "open",
                    "label": _label(m),
                    "rate": rates[m],
                    "rate_multiplier": m,
                    "seed": seed,
                }
                for m in mults
            ]
            m2 = {**meta, "container_tag": f"_seed{seed}"}
            call = _runner(system).spawn(system, points, {seed: traces[seed]}, m2)
            calls.append((system, seed, call))
    written: dict[str, list[Path]] = {}
    with concurrent.futures.ThreadPoolExecutor(len(calls)) as pool:
        futures = {pool.submit(c.get): (s, k) for s, k, c in calls}
        for fut in concurrent.futures.as_completed(futures):
            system, seed = futures[fut]
            try:
                written[f"{system}/seed{seed}"] = _write(fut.result(), group)
                print(f"done: {system} seed {seed}")
            except Exception as exc:
                print(f"FAILED: {system} seed {seed}: {type(exc).__name__}: {exc}")
    return written


@app.local_entrypoint()
def dryrun(
    rate: float = 0.8,
    seed: int = SEEDS[0],
    systems: str = "stallfree-off,stallfree-on,vllm,vllm-eager",
) -> None:
    """All four systems (or a comma-separated subset) at one rate, a multiple of the saturation
    rate, one repeat."""
    written = _sweep(systems.split(","), [rate], [seed], "dryrun")
    for key, paths in sorted(written.items()):
        print(key, len(paths), "files")


@app.local_entrypoint()
def main(system: str, rates: str = "0.2,0.5,0.8,1.0,1.2", repeats: int = 3) -> None:
    """The full sweep for one system: every rate, ``repeats`` seeds (one container each)."""
    mults = [float(x) for x in rates.split(",")]
    written = _sweep([system], mults, list(SEEDS[:repeats]), "")
    for key, paths in sorted(written.items()):
        print(key, len(paths), "files")


@app.function(image=vllm_image, cpu=2.0, memory=8192, timeout=1800)
def vllm_image_versions() -> dict[str, str]:
    """CPU only: package versions in the vLLM image. (vLLM's CLI needs a GPU even to print its
    help, so the server flags are checked in the benchmark container, before startup.)"""
    return _versions()


@app.local_entrypoint()
def check_vllm_image() -> None:
    """Build the vLLM image and print its package versions (no GPU)."""
    print(json.dumps(vllm_image_versions.remote(), indent=1))
