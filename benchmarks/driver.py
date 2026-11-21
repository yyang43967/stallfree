"""Runs one benchmark container's work: start a server, replay traces against it, sample CPU.

The server, the load generator, and this driver are three processes. The driver samples CPU
utilization of the server's process tree and of the load generator every second with psutil
(and GPU utilization and SM clock with nvidia-smi when a GPU is present), and writes everything
for one run next to its per-request records:

    <stem>.jsonl.gz       per-request records from the load generator (gzipped JSON lines)
    <stem>.summary.json   load generator summary (rate, warm-up and measured counts, ...)
    <stem>.cpu.jsonl      one CPU/GPU sample per second
    <stem>.run.json       system, flags, versions, GPU, CPU request, trace seed and hashes, commit
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

SAMPLE_INTERVAL_S = 1.0
HEALTH_TIMEOUT_S = 1200.0  # vLLM compiles and captures CUDA graphs at startup
STOP_TIMEOUT_S = 30.0
PORT = 8000
SERVED_MODEL = "Qwen/Qwen3-0.6B"
MAX_MODEL_LEN = 4096
MAX_NUM_SEQS = 256
MIXED_BUDGET = 512  # stallfree mixed batching and vLLM's chunked-prefill budget
OFF_PREFILL_CAP = 4096  # stallfree with mixed batching off: the cap on one prefill step
GPU_MEMORY_UTILIZATION = 0.9  # both engines' default
# stallfree rejects requests with 503 above this many active requests; set high enough that it
# never triggers, since vLLM has no such limit.
MAX_QUEUED = 100_000

GPU_FIELDS = (
    "name,driver_version,clocks.sm,clocks.mem,clocks.max.sm,clocks.max.mem,power.draw,"
    "power.limit,temperature.gpu,utilization.gpu,memory.used,memory.total"
)


class System(StrEnum):
    STALLFREE_OFF = "stallfree-off"
    STALLFREE_ON = "stallfree-on"
    VLLM = "vllm"
    VLLM_EAGER = "vllm-eager"


def server_command(system: System, model_dir: str, port: int = PORT) -> list[str]:
    """The server command line. Settings that exist in both engines are matched: model
    length, sequence limit, token budget (for chunked prefill), memory fraction, bf16, and no
    prefix caching (stallfree has none)."""
    common = ["--served-model-name", SERVED_MODEL, "--max-model-len", str(MAX_MODEL_LEN)]
    common += ["--max-num-seqs", str(MAX_NUM_SEQS), "--host", "127.0.0.1", "--port", str(port)]
    if system in (System.STALLFREE_OFF, System.STALLFREE_ON):
        cmd = [sys.executable, "-m", "stallfree.server.api", "--model", model_dir, *common]
        cmd += ["--device", "cuda", "--attention", "flash", "--dtype", "bfloat16"]
        cmd += ["--max-queued-requests", str(MAX_QUEUED)]
        if system is System.STALLFREE_ON:
            return [*cmd, "--enable-mixed-batching", "--max-num-batched-tokens", str(MIXED_BUDGET)]
        return [*cmd, "--max-num-batched-tokens", str(OFF_PREFILL_CAP)]
    cmd = ["vllm", "serve", model_dir, *common, "--dtype", "bfloat16"]
    cmd += ["--max-num-batched-tokens", str(MIXED_BUDGET), "--no-enable-prefix-caching"]
    cmd += ["--gpu-memory-utilization", str(GPU_MEMORY_UTILIZATION)]
    if system is System.VLLM_EAGER:
        cmd.append("--enforce-eager")
    return cmd


def gpu_info() -> dict[str, str] | None:
    if shutil.which("nvidia-smi") is None:
        return None
    out = subprocess.run(
        ["nvidia-smi", f"--query-gpu={GPU_FIELDS}", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not out:
        return None
    values = [v.strip() for v in out.splitlines()[0].split(",")]
    return dict(zip(GPU_FIELDS.split(","), values, strict=True))


@dataclass
class CpuSampler:
    """Samples CPU% per process (100 = one full core) for a server's process tree and the load
    generator, plus GPU utilization and SM clock, every ``interval`` seconds in a thread."""

    server_pid: int
    loadgen_pid: int
    interval: float = SAMPLE_INTERVAL_S
    gpu: bool = True
    samples: list[dict[str, Any]] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> list[dict[str, Any]]:
        self._stop.set()
        if self._thread:
            self._thread.join()
        return self.samples

    def _run(self) -> None:
        import psutil

        procs: dict[int, Any] = {}

        def track(pid: int) -> Any | None:
            if pid not in procs:
                try:
                    p = psutil.Process(pid)
                    p.cpu_percent(None)  # the first call only starts the measurement
                    procs[pid] = p
                except psutil.Error:
                    return None
            return procs[pid]

        def tree() -> list[int]:
            try:
                root = psutil.Process(self.server_pid)
                return [self.server_pid, *(c.pid for c in root.children(recursive=True))]
            except psutil.Error:
                return []

        for pid in [*tree(), self.loadgen_pid]:
            track(pid)
        while not self._stop.wait(self.interval):
            sample: dict[str, Any] = {"wall": time.time(), "procs": []}
            server_total = 0.0
            for pid in tree():
                p = track(pid)
                if p is None:
                    continue
                try:
                    pct = p.cpu_percent(None)
                    name = p.name()
                except psutil.Error:
                    continue
                server_total += pct
                sample["procs"].append([pid, name, round(pct, 1)])
            lg = track(self.loadgen_pid)
            try:
                sample["loadgen_pct"] = round(lg.cpu_percent(None), 1) if lg else None
            except psutil.Error:
                sample["loadgen_pct"] = None
            sample["server_pct"] = round(server_total, 1)
            if self.gpu:
                info = gpu_info()
                if info:
                    sample["gpu_util"] = info["utilization.gpu"]
                    sample["sm_clock"] = info["clocks.sm"]
            self.samples.append(sample)


@contextlib.contextmanager
def server(
    cmd: Sequence[str], log_path: Path, base_url: str, env: dict[str, str] | None = None
) -> Iterator[subprocess.Popen[bytes]]:
    """Start the server in its own process group, wait for /health, stop it afterwards."""
    log = log_path.open("wb")
    proc = subprocess.Popen(
        list(cmd), stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True
    )
    try:
        wait_healthy(base_url, proc)
        yield proc
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(STOP_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        log.close()


def wait_healthy(base_url: str, proc: subprocess.Popen[bytes]) -> float:
    start = time.monotonic()
    root = base_url.removesuffix("/v1")
    while time.monotonic() - start < HEALTH_TIMEOUT_S:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with {proc.returncode} before becoming healthy")
        try:
            with urllib.request.urlopen(f"{root}/health", timeout=5) as r:
                if r.status == 200:
                    return time.monotonic() - start
        except OSError:
            pass
        time.sleep(1.0)
    raise TimeoutError("server did not become healthy")


def metrics_text(base_url: str) -> str | None:
    """The Prometheus /metrics page (vLLM has one; stallfree does not)."""
    try:
        with urllib.request.urlopen(f"{base_url.removesuffix('/v1')}/metrics", timeout=10) as r:
            return str(r.read().decode())
    except OSError:
        return None


def run_point(
    *,
    base_url: str,
    server_pid: int,
    trace_path: Path,
    out: Path,
    loadgen_args: Sequence[str],
    run_meta: dict[str, Any],
    sample_interval: float = SAMPLE_INTERVAL_S,
) -> dict[str, Any]:
    """One measured run: the load generator in a subprocess, sampled; writes the run's files."""
    out.parent.mkdir(parents=True, exist_ok=True)
    gpu_before = gpu_info()
    metrics_before = metrics_text(base_url)
    cmd = [sys.executable, "-m", "benchmarks.loadgen", "--base-url", base_url]
    cmd += ["--trace", str(trace_path), "--out", str(out), *loadgen_args]
    start = time.time()
    lg = subprocess.Popen(cmd)
    sampler = CpuSampler(server_pid, lg.pid, sample_interval, gpu=gpu_before is not None)
    sampler.start()
    rc = lg.wait()
    samples = sampler.stop()
    if rc != 0:
        raise RuntimeError(f"load generator exited with {rc}")
    with out.with_suffix(".cpu.jsonl").open("w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    meta = {
        **run_meta,
        "loadgen_command": cmd[1:],
        "started_wall": start,
        "seconds": round(time.time() - start, 1),
        "gpu_before": gpu_before,
        "gpu_after": gpu_info(),
    }
    if metrics_before is not None:
        meta["metrics_before"] = metrics_before
        meta["metrics_after"] = metrics_text(base_url)
    out.with_suffix(".run.json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta
