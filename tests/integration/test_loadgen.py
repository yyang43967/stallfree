"""The load generator, CPU sampler, driver, and analysis against a real stallfree server on CPU
(tiny model, byte tokenizer). Checks the request shape (ignore_eos, usage, streaming), the
open-loop warm-up split, closed-loop throughput accounting, the files a run writes, and the
sanity checks the analysis reports. Timings on CPU are not meaningful."""

from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest
import uvicorn
from helpers.byte_tokenizer import ByteTokenizer

from benchmarks.analyze import load_run, run_metrics
from benchmarks.driver import CpuSampler, System, run_point, server_command
from benchmarks.loadgen import closed_loop, open_loop
from benchmarks.workload import Mixture, TraceRequest, Window, dumps, make_trace
from stallfree.async_engine import AsyncEngine
from stallfree.config import EngineConfig, SchedulerConfig
from stallfree.engine import LLMEngine
from stallfree.server.api import create_app

MODEL = "tiny"
MAX_LEN = 256
MIXTURE = Mixture(short_prompt=(20, 40), long_prompt=(80, 120), long_fraction=0.2, output=(3, 8))
WINDOW = Window(warmup_s=0.5, min_measured=12, min_measure_s=1.0)
TEXT = (
    "It is a truth universally acknowledged, that a single man in possession of a good fortune "
    "must be in want of a wife. However little known the feelings or views of such a man may be "
    "on his first entering a neighbourhood, this truth is so well fixed in the minds of the "
    "surrounding families, that he is considered as the rightful property of some one or other "
    "of their daughters. "
) * 4


class ByteWorkloadTokenizer:
    """The workload's view of the server's byte tokenizer."""

    def __init__(self) -> None:
        self._chat = ByteTokenizer()

    def encode_text(self, text: str) -> list[int]:
        return list(text.encode())

    def decode(self, ids: Sequence[int]) -> str:
        return self._chat.decode(ids)

    def chat_len(self, content: str) -> int:
        return len(self._chat.encode_chat([{"role": "user", "content": content}], {}))


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture(scope="module")
def base_url(tiny_model_dir: Path) -> Iterator[str]:
    cfg = EngineConfig(
        model=str(tiny_model_dir),
        num_blocks=256,
        block_size=4,
        dtype="float32",
        scheduler=SchedulerConfig(
            max_num_seqs=16, max_num_batched_tokens=512, max_model_len=MAX_LEN, eos_token_id=0
        ),
    )
    engine = AsyncEngine(LLMEngine(cfg))
    app = create_app(engine, ByteTokenizer(), served_model_name=MODEL, max_model_len=MAX_LEN)
    port = _free_port()
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 30
    while not srv.started:
        assert time.monotonic() < deadline
        time.sleep(0.02)
    try:
        yield f"http://127.0.0.1:{port}/v1"
    finally:
        srv.should_exit = True
        thread.join(timeout=30)


@pytest.fixture(scope="module")
def trace() -> list[TraceRequest]:
    return make_trace(ByteWorkloadTokenizer(), TEXT, seed=11, num_requests=60, mixture=MIXTURE)


def test_open_loop_requests_are_counted_by_the_server_and_run_to_max_tokens(
    base_url: str, trace: list[TraceRequest]
) -> None:
    result = asyncio.run(open_loop(base_url, trace, rate=20.0, model=MODEL, window=WINDOW))
    recs = result["records"]
    warmup, measured = result["warmup_requests"], result["measured_requests"]
    assert len(recs) == warmup + measured and warmup > 0 and measured >= 12
    assert [r["phase"] for r in recs] == ["warmup"] * warmup + ["measured"] * measured
    for r in recs:
        assert r["status"] == 200 and r["error"] is None
        assert r["prompt_tokens"] == r["target_prompt_tokens"]  # server-side count
        assert r["completion_tokens"] == r["max_tokens"]  # ignore_eos honoured
        assert r["finish_reason"] == "length"
        # One chunk per token, except where the detokenizer holds back an incomplete UTF-8
        # sequence (the random tiny model emits non-ASCII bytes) and sends it with the next.
        assert 0 < len(r["chunks"]) <= r["max_tokens"]
        assert r["sent"] <= r["chunks"][0] <= r["chunks"][-1] <= r["end"]
        assert r["sent"] >= r["scheduled"] - 0.05


def test_closed_loop_counts_only_requests_finished_in_the_window(
    base_url: str, trace: list[TraceRequest]
) -> None:
    result = asyncio.run(
        closed_loop(base_url, trace, concurrency=4, warmup_s=0.5, measure_s=1.5, model=MODEL)
    )
    done = [r for r in result["records"] if r["error"] is None and 0.5 <= r.get("end", -1) < 2.0]
    assert result["completed_in_window"] == len(done) > 0
    assert result["request_throughput"] == pytest.approx(len(done) / 1.5)


def test_cpu_sampler_sees_a_busy_load_generator() -> None:
    busy = subprocess.Popen([sys.executable, "-c", "while True: pass"])
    idle = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        sampler = CpuSampler(idle.pid, busy.pid, interval=0.3, gpu=False)
        sampler.start()
        time.sleep(1.6)
        samples = sampler.stop()
    finally:
        busy.kill()
        idle.kill()
    assert len(samples) >= 3
    assert max(s["loadgen_pct"] for s in samples) > 50
    assert all(s["server_pct"] < 50 for s in samples)
    assert samples[-1]["procs"][0][0] == idle.pid


def test_run_point_writes_the_run_files_and_the_analysis_passes_its_checks(
    base_url: str, trace: list[TraceRequest], tmp_path: Path
) -> None:
    trace_path = tmp_path / "trace.jsonl.gz"
    trace_path.write_bytes(dumps(trace))
    out = tmp_path / "stallfree-off" / "r20" / "seed11.jsonl"
    args = ["--mode", "open", "--rate", "20", "--model", MODEL, "--warmup-s", "0.5"]
    args += ["--min-measured", "12", "--min-measure-s", "1.0"]
    meta = run_point(
        base_url=base_url,
        server_pid=os.getpid(),
        trace_path=trace_path,
        out=out,
        loadgen_args=args,
        run_meta={"system": "stallfree-off", "seed": 11},
        sample_interval=0.1,  # the measured window is under a second on CPU
    )
    assert meta["system"] == "stallfree-off" and meta["gpu_before"] is None
    for suffix in (".jsonl", ".summary.json", ".cpu.jsonl", ".run.json"):
        assert out.with_suffix(suffix).exists(), suffix
    m = run_metrics(load_run(out))
    assert m["errors"] == 0 and m["ok"] == m["measured"] >= 12
    assert m["prompt_token_mismatches"] == 0
    assert m["ignore_eos_violations"] == 0
    assert m["missing_usage"] == 0
    assert m["tokens_per_chunk"] >= 1.0
    assert m["ttft"]["p50"] > 0 and m["itl"]["max"] >= m["itl"]["p99"] >= m["itl"]["p50"]
    assert "cpu" in m and m["cpu"]["loadgen_mean"] is not None


def test_server_commands_match_settings_across_systems() -> None:
    cmds = {s: server_command(s, "/m") for s in System}

    def flag(cmd: list[str], name: str) -> str | None:
        return cmd[cmd.index(name) + 1] if name in cmd else None

    for cmd in cmds.values():
        assert flag(cmd, "--max-model-len") == "4096"
        assert flag(cmd, "--max-num-seqs") == "256"
        assert flag(cmd, "--served-model-name") == "Qwen/Qwen3-0.6B"
    assert flag(cmds[System.STALLFREE_ON], "--max-num-batched-tokens") == "512"
    assert flag(cmds[System.VLLM], "--max-num-batched-tokens") == "512"
    assert "--enable-mixed-batching" in cmds[System.STALLFREE_ON]
    assert "--enable-mixed-batching" not in cmds[System.STALLFREE_OFF]
    assert "--no-enable-prefix-caching" in cmds[System.VLLM]
    assert "--enforce-eager" in cmds[System.VLLM_EAGER]
    assert "--enforce-eager" not in cmds[System.VLLM]
