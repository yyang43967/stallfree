"""Assigning py-spy samples to server components, and the report's arithmetic."""

from __future__ import annotations

import subprocess
import sys

import pytest

from benchmarks.driver import System, resolve, server_command
from benchmarks.profile_summary import component, parse, report, summarize

ROOT = "/root/stallfree"
LIB = "/usr/local/lib/python3.12/site-packages"
LINES = [
    # model forward from the worker thread, inside a CUDA launch
    f"thread (0x1): asyncio_0;run (threading.py:1012);step ({ROOT}/engine.py:80);"
    f"execute ({ROOT}/model_runner.py:120);forward ({ROOT}/model/qwen3.py:200);"
    f"linear ({LIB}/torch/nn/functional.py:1);cudaLaunchKernel (libcudart.so.13) 30",
    # building the batch in the runner
    f"thread (0x1): asyncio_0;step ({ROOT}/engine.py:80);"
    f"prepare ({ROOT}/model_runner.py:60);tensor ({LIB}/torch/_tensor.py:5) 10",
    # sampling waits for the GPU
    f"thread (0x1): asyncio_0;step ({ROOT}/engine.py:80);sample ({ROOT}/sampler.py:30);"
    f"tolist (x);cudaStreamSynchronize (libcudart.so.13) 5",
    # scheduler
    f"thread (0x1): asyncio_0;step ({ROOT}/engine.py:80);schedule ({ROOT}/scheduler.py:200) 4",
    # detokenization on the event loop thread, called from the SSE generator
    f"thread (0x2): MainThread;_sse ({ROOT}/server/api.py:210);"
    f"add ({ROOT}/tokenizer.py:90);decode ({LIB}/transformers/x.py:1) 6",
    # HTTP with no stallfree frame
    f"thread (0x2): MainThread;run ({LIB}/uvicorn/server.py:1);send ({LIB}/h11/_connection.py:1) 3",
    # event loop
    f"thread (0x2): MainThread;run_forever ({LIB}/asyncio/base_events.py:1) 2",
]


def test_components_follow_the_innermost_stallfree_frame() -> None:
    samples = parse(LINES)
    assert [component(s.frames) for s in samples] == [
        "model forward",
        "batch metadata (runner inputs)",
        "sampling",
        "scheduler and block manager",
        "detokenization",
        "HTTP and SSE",
        "event loop",
    ]
    assert samples[0].count == 30 and samples[0].thread == "thread (0x1): asyncio_0"


def test_summary_counts_launches_waits_threads_and_cores() -> None:
    s = summarize(parse(LINES))
    assert s.total == 60 and s.native
    assert s.launch["model forward"] == 30
    assert s.gpu_wait["sampling"] == 5
    assert s.by_thread == {"asyncio_0": 49, "MainThread": 11}
    text = report(s, "p.txt", rate=10, duration_s=3.0)
    assert "60 samples = 2.00 cores" in text
    assert "| model forward | 30 | 50.0% | 1.00 | 100.0% | 0.0% |" in text
    assert "`forward (stallfree/model/qwen3.py:200)`" in text


def test_budget_variants_and_the_profiling_launcher() -> None:
    assert resolve("stallfree-on-b1024") == (System.STALLFREE_ON, 1024)
    assert resolve("vllm-eager") == (System.VLLM_EAGER, None)
    cmd = server_command(System.STALLFREE_ON, "/m", budget=256, ptraceable=True)
    assert cmd[1:4] == ["-m", "benchmarks.ptraceable", "stallfree.server.api"]
    assert cmd[cmd.index("--max-num-batched-tokens") + 1] == "256"
    plain = server_command(System.STALLFREE_ON, "/m")
    assert plain[1:3] == ["-m", "stallfree.server.api"]
    assert plain[plain.index("--max-num-batched-tokens") + 1] == "512"


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="prctl is Linux-only")
def test_ptraceable_runs_the_module_as_main() -> None:
    out = subprocess.run(
        [sys.executable, "-m", "benchmarks.ptraceable", "json.tool", "--compact"],
        input='{"a": 1}',
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == '{"a":1}'
    assert "prctl" not in out.stderr
