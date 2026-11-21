"""Assigning py-spy samples to server components, and the report's arithmetic."""

from __future__ import annotations

import subprocess
import sys

import pytest

from benchmarks.driver import System, resolve, server_command
from benchmarks.profile_summary import (
    IO_WAIT,
    LOCK_WAIT,
    RUNNING,
    component,
    parse,
    report,
    state,
    summarize,
)

ROOT = "/root/stallfree"
LIB = "/usr/local/lib/python3.12/site-packages"
LINES = [
    # model forward from the worker thread, inside a CUDA launch
    f"thread (0x1): asyncio_0;run (threading.py:1012);step ({ROOT}/engine.py:80);"
    f"execute ({ROOT}/model_runner.py:120);forward ({ROOT}/model/qwen3.py:200);"
    f"linear ({LIB}/torch/nn/functional.py:1);cuLaunchKernel (libcuda.so.610) 30",
    # building the batch: slot mapping is metadata even though it lives in kv_cache.py
    f"thread (0x1): asyncio_0;step ({ROOT}/engine.py:80);prepare ({ROOT}/model_runner.py:60);"
    f"slots_for_positions ({ROOT}/kv_cache.py:31);x (/lib/libtorch_cpu.so) 10",
    # the step thread waiting for the GIL inside sampling
    f"thread (0x1): asyncio_0;step ({ROOT}/engine.py:80);sample ({ROOT}/sampler.py:30);"
    f"0x2ae1 (python3.12);pthread_cond_timedwait (/lib/libc.so.6);0x2ae2 (/lib/libc.so.6) 5",
    # scheduler
    f"thread (0x1): asyncio_0;step ({ROOT}/engine.py:80);schedule ({ROOT}/scheduler.py:200) 4",
    # prompt tokenization on the event loop thread, under the server entry point
    f"thread (0x2): MainThread;main ({ROOT}/server/api.py:305);"
    f"chat_completions ({ROOT}/server/api.py:120);"
    f"HFChatTokenizer.encode_chat ({ROOT}/tokenizer.py:50);encode ({LIB}/transformers/x.py:1) 6",
    # detokenization from the SSE generator
    f"thread (0x2): MainThread;main ({ROOT}/server/api.py:305);_sse ({ROOT}/server/api.py:210);"
    f"IncrementalDetokenizer.add ({ROOT}/tokenizer.py:90) 2",
    # HTTP with only the entry point as stallfree frame
    f"thread (0x2): MainThread;main ({ROOT}/server/api.py:305);run ({LIB}/uvicorn/server.py:1);"
    f"send ({LIB}/h11/_connection.py:1) 3",
    # event loop idle in epoll
    f"thread (0x2): MainThread;main ({ROOT}/server/api.py:305);"
    f"run_forever ({LIB}/asyncio/base_events.py:1);epoll_wait (/lib/libc.so.6) 7",
]


def test_components_and_states() -> None:
    samples = parse(LINES)
    assert [component(s.frames) for s in samples] == [
        "model forward",
        "batch metadata",
        "sampling",
        "scheduler and block manager",
        "tokenization (chat template)",
        "detokenization",
        "HTTP and SSE",
        "event loop",
    ]
    assert [state(s.frames) for s in samples] == [
        RUNNING,
        RUNNING,
        LOCK_WAIT,
        RUNNING,
        RUNNING,
        RUNNING,
        RUNNING,
        IO_WAIT,
    ]


def test_summary_counts_running_cpu_launches_and_libraries() -> None:
    s = summarize(parse(LINES))
    assert s.total == 67 and s.native
    assert s.by_state == {RUNNING: 55, LOCK_WAIT: 5, IO_WAIT: 7}
    assert s.launch["model forward"] == 30
    assert s.forward_libs == {"CUDA driver/runtime (kernel launch)": 30}
    text = report(s, "p.txt", rate=10, duration_s=5.5)
    assert "55 running (1.00 cores on average)" in text
    assert "| model forward | 30 | 54.5% | 0.55 | 100.0% | 0 |" in text
    assert "| sampling | 0 | 0.0% | 0.00 | - | 5 |" in text
    assert "`HFChatTokenizer.encode_chat (stallfree/tokenizer.py:50)`" in text


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
