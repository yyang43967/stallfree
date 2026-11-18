"""The stall micro-benchmark harness on CPU: step structure with mixed batching off and on, and
the analysis. Timings on CPU are not meaningful; only step counts and token placement are checked.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

from benchmarks.microbench import (
    Scenario,
    analyze,
    make_prompts,
    percentile,
    plot,
    report,
    run_suite,
    summarize,
)
from stallfree.config import AttentionBackendName, EngineConfig, SchedulerConfig
from stallfree.engine import LLMEngine

SCENARIO = Scenario(
    num_decodes=3,
    decode_prompt_len=6,
    long_prompt_len=40,
    warmup_tokens=2,
    measured_tokens=6,
    token_low=1,
    token_high=256,
)
BUDGET = 8


@pytest.fixture(scope="module")
def runs(tiny_model_dir: Path) -> list[dict[str, Any]]:
    engine = LLMEngine(
        EngineConfig(
            model=str(tiny_model_dir),
            num_blocks=64,
            block_size=4,
            dtype="float32",
            device="cpu",
            attention=AttentionBackendName.TORCH_REF,
            scheduler=SchedulerConfig(max_model_len=128),
        )
    )
    out = run_suite(engine, [None, BUDGET], repeats=2, scenario=SCENARIO, sync=lambda: None)
    assert engine.block_manager.num_free_blocks == 64
    assert not engine.config.scheduler.enable_mixed_batching
    return out


def after_arrival(run: dict[str, Any]) -> list[dict[str, Any]]:
    return [s for s in run["steps"] if s["start"] > 0]


def test_runs_are_interleaved_after_warmup(runs: list[dict[str, Any]]) -> None:
    assert [(r["config"], r["repeat"]) for r in runs] == [
        ("off", 0),
        ("on-8", 0),
        ("off", 1),
        ("on-8", 1),
    ]


def test_off_runs_the_long_prompt_alone_and_stalls_the_decodes(
    runs: list[dict[str, Any]],
) -> None:
    run = runs[0]
    first = after_arrival(run)[0]
    assert first["long_prompt_tokens"] == SCENARIO.long_prompt_len
    assert first["num_seqs"] == 1
    m = analyze(run, SCENARIO)
    assert m["long_prefill_steps"] == 1
    # One prefill step, then 6 decode steps for the decodes (the long one ends a step earlier).
    assert m["steps"] == 1 + SCENARIO.measured_tokens
    # The long request's first token comes from that prefill step.
    assert run["long_token_times"][0] == pytest.approx(first["end"])


def test_on_chunks_the_long_prompt_and_keeps_every_decode_running(
    runs: list[dict[str, Any]],
) -> None:
    run = runs[1]
    steps = after_arrival(run)
    chunk_steps = [s for s in steps if s["long_prompt_tokens"] > 0]
    per_chunk = BUDGET - SCENARIO.num_decodes
    assert len(chunk_steps) == math.ceil(SCENARIO.long_prompt_len / per_chunk)
    assert sum(s["long_prompt_tokens"] for s in chunk_steps) == SCENARIO.long_prompt_len
    assert all(s["num_tokens"] <= BUDGET for s in run["steps"])
    # Every step after arrival gives every decode a token until it has the measured count.
    for ts in run["decode_token_times"]:
        post = [t for t in ts if t > 0]
        ends = [s["end"] for s in steps[: len(post)]]
        assert post == pytest.approx(ends)
    m = analyze(run, SCENARIO)
    assert m["long_prefill_steps"] == len(chunk_steps)
    assert m["steps"] == len(chunk_steps) + SCENARIO.measured_tokens - 1


def test_summary_report_and_plots(runs: list[dict[str, Any]], tmp_path: Path) -> None:
    data = {
        "scenario": SCENARIO.__dict__,
        "meta": {
            "gpu": "cpu",
            "model": "tiny",
            "dtype": "float32",
            "attention": "torch_ref",
            "block_size": 4,
            "num_blocks": 64,
            "torch": "x",
            "flash_attn": "none",
            "commit": "abc",
        },
        "runs": runs,
    }
    s = summarize(data)
    assert list(s) == ["off", "on-8"]
    assert all(len(v) == 2 for v in s["off"].values())
    text = report(data, "x.json")
    assert "| off |" in text and "| on-8 |" in text
    assert "in-process" in text
    paths = plot(data, tmp_path, "run")
    assert [p.name for p in paths] == ["run_itl_timeline.png", "run_summary.png"]
    assert all(p.stat().st_size > 0 for p in paths)


def test_percentile_is_nearest_rank() -> None:
    values = list(range(1, 101))
    assert percentile(values, 99) == 99
    assert percentile(values, 100) == 100
    assert percentile([5.0], 99) == 5.0
    assert percentile([1, 2, 3, 4], 50) == 2


def test_prompts_are_deterministic() -> None:
    assert make_prompts(SCENARIO) == make_prompts(SCENARIO)
    decodes, long = make_prompts(SCENARIO)
    assert len(decodes) == 3 and all(len(p) == 6 for p in decodes) and len(long) == 40
