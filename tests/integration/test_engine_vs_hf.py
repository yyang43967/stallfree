"""End to end: the engine with the real block manager and scheduler against Hugging Face.

Tiny random Qwen3, fp32, PyTorch paged attention, block size 4 so short prompts cross many block
boundaries. Every request's greedy output must equal HF ``generate()`` token for token. The
reference's smallest top-1/top-2 logit gap over the generated positions is recorded for the run
summary.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from helpers.tiny_qwen3 import TINY_CONFIG, hf_greedy, hf_logits, record_gap
from transformers import Qwen3ForCausalLM as HFQwen3ForCausalLM

from stallfree.config import EngineConfig, SchedulerConfig
from stallfree.engine import LLMEngine
from stallfree.sequence import SamplingParams

BS = 4


# Every scenario in this file runs with mixed batching off and on; assertions are the same.
MODE: dict[str, bool] = {"mixed": False}


@pytest.fixture(autouse=True, params=[False, True], ids=["off", "on"])
def batching_mode(request: pytest.FixtureRequest) -> Iterator[None]:
    MODE["mixed"] = request.param
    yield
    MODE["mixed"] = False


def make_engine(
    model_dir: Path,
    *,
    num_blocks: int = 128,
    max_num_seqs: int = 16,
    max_num_batched_tokens: int = 256,
    watermark: float = 0.0,
) -> LLMEngine:
    scheduler = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=512,
        watermark=watermark,
        eos_token_id=None,
        enable_mixed_batching=MODE["mixed"],
    )
    cfg = EngineConfig(
        model=str(model_dir),
        num_blocks=num_blocks,
        block_size=BS,
        dtype="float32",
        scheduler=scheduler,
    )
    return LLMEngine(cfg)


def prompt(n: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    vocab: int = TINY_CONFIG["vocab_size"]  # type: ignore[assignment]
    return [rng.randrange(vocab) for _ in range(n)]


def params(max_tokens: int) -> SamplingParams:
    return SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)


def check_against_hf(
    hf: HFQwen3ForCausalLM,
    prompts: list[list[int]],
    outputs: list[list[int]],
    max_tokens: list[int],
    label: str,
) -> None:
    for p, out, n in zip(prompts, outputs, max_tokens, strict=True):
        ref = hf_greedy(hf, p, n)
        record_gap(hf_logits(hf, p + ref[:-1])[len(p) - 1 :], label)
        assert out == ref, f"prompt of {len(p)} tokens: first difference at {_first_diff(out, ref)}"


def _first_diff(a: list[int], b: list[int]) -> int:
    diffs = (i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y)
    return next(diffs, min(len(a), len(b)))


def run_to_completion(engine: LLMEngine, ids: list[int]) -> list[list[int]]:
    outputs: dict[int, list[int]] = {i: [] for i in ids}
    while engine.has_unfinished():
        for out in engine.step():
            outputs[out.seq_id].append(out.token_id)
    return [outputs[i] for i in ids]


def test_single_request(tiny_model_dir: Path, tiny_hf_model: HFQwen3ForCausalLM) -> None:
    p = prompt(10, seed=1)
    out = make_engine(tiny_model_dir).generate([p], params(20))
    check_against_hf(tiny_hf_model, [p], out, [20], "e2e single")


def test_batched_requests_of_different_lengths(
    tiny_model_dir: Path, tiny_hf_model: HFQwen3ForCausalLM
) -> None:
    lengths = [1, 3, 4, 5, 8, 9, 13, 20]
    prompts = [prompt(n, seed=100 + n) for n in lengths]
    max_tokens = [12, 7, 15, 9, 4, 11, 6, 10]
    engine = make_engine(tiny_model_dir)
    outs = engine.generate(prompts, [params(n) for n in max_tokens])
    check_against_hf(tiny_hf_model, prompts, outs, max_tokens, "e2e batched")


def test_requests_arriving_mid_run(tiny_model_dir: Path, tiny_hf_model: HFQwen3ForCausalLM) -> None:
    engine = make_engine(tiny_model_dir)
    prompts: list[list[int]] = []
    max_tokens: list[int] = []
    ids: list[int] = []
    outputs: dict[int, list[int]] = {}
    arrivals = {0: [(6, 14), (11, 9)], 3: [(4, 12), (9, 8), (2, 10)], 7: [(15, 6)], 12: [(5, 5)]}
    step = 0
    while step <= max(arrivals) or engine.has_unfinished():
        for n, m in arrivals.get(step, []):
            p = prompt(n, seed=200 + len(prompts))
            sid = engine.add_request(p, params(m))
            prompts.append(p)
            max_tokens.append(m)
            ids.append(sid)
            outputs[sid] = []
        for out in engine.step():
            outputs[out.seq_id].append(out.token_id)
        step += 1
    check_against_hf(tiny_hf_model, prompts, [outputs[i] for i in ids], max_tokens, "e2e arrivals")


@pytest.mark.parametrize("watermark", [0.0, 0.2])
def test_forced_preemption(
    tiny_model_dir: Path, tiny_hf_model: HFQwen3ForCausalLM, watermark: float
) -> None:
    # 10 blocks of 4 tokens. Each request may reach 5 + 12 = 17 tokens (5 blocks), so four of
    # them cannot all keep growing: some must be preempted and recomputed.
    engine = make_engine(tiny_model_dir, num_blocks=10, watermark=watermark)
    prompts = [prompt(5, seed=300 + i) for i in range(4)]
    ids = [engine.add_request(p, params(12)) for p in prompts]
    seqs = [engine.scheduler.get(i) for i in ids]
    outs = run_to_completion(engine, ids)
    assert engine.num_preemptions >= 1, "this setup must preempt"
    assert sum(s.num_preemptions for s in seqs if s is not None) == engine.num_preemptions
    check_against_hf(tiny_hf_model, prompts, outs, [12] * 4, "e2e preemption")
    assert engine.block_manager.num_free_blocks == engine.block_manager.num_blocks


@pytest.mark.parametrize("prompt_len", [3, 4, 5, 7, 8, 9])
def test_lengths_at_block_boundaries(
    tiny_model_dir: Path, tiny_hf_model: HFQwen3ForCausalLM, prompt_len: int
) -> None:
    """Prompts of block_size - 1, block_size, block_size + 1 (and the same around 2 blocks);
    generation then crosses further boundaries."""
    p = prompt(prompt_len, seed=400 + prompt_len)
    out = make_engine(tiny_model_dir).generate([p], params(6))
    check_against_hf(tiny_hf_model, [p], out, [6], "e2e block boundaries")


def test_block_boundary_lengths_together(
    tiny_model_dir: Path, tiny_hf_model: HFQwen3ForCausalLM
) -> None:
    prompts = [prompt(n, seed=500 + n) for n in (3, 4, 5, 7, 8, 9)]
    outs = make_engine(tiny_model_dir).generate(prompts, params(6))
    check_against_hf(tiny_hf_model, prompts, outs, [6] * 6, "e2e block boundaries")


def test_token_budget_and_sequence_limit(
    tiny_model_dir: Path, tiny_hf_model: HFQwen3ForCausalLM
) -> None:
    # Budget 8 tokens per step, at most 2 running: the 20-token prompt must run alone.
    engine = make_engine(tiny_model_dir, max_num_seqs=2, max_num_batched_tokens=8)
    prompts = [prompt(n, seed=600 + n) for n in (3, 20, 5, 2, 6)]
    outs = engine.generate(prompts, params(5))
    check_against_hf(tiny_hf_model, prompts, outs, [5] * 5, "e2e limits")


@pytest.mark.parametrize("seed", range(5))
def test_random_workload_with_small_cache(
    tiny_model_dir: Path, tiny_hf_model: HFQwen3ForCausalLM, seed: int
) -> None:
    rng = random.Random(seed)
    engine = make_engine(
        tiny_model_dir,
        num_blocks=rng.randint(8, 14),
        max_num_seqs=rng.randint(2, 6),
        max_num_batched_tokens=rng.choice([16, 32, 64]),
        watermark=rng.choice([0.0, 0.1]),
    )
    prompts, max_tokens, ids = [], [], []
    outputs: dict[int, list[int]] = {}
    for step in range(10_000):
        if len(prompts) < 10 and rng.random() < 0.3:
            p, m = prompt(rng.randint(1, 14), seed=1000 * seed + len(prompts)), rng.randint(1, 12)
            sid = engine.add_request(p, params(m))
            prompts.append(p)
            max_tokens.append(m)
            ids.append(sid)
            outputs[sid] = []
        if len(prompts) == 10 and not engine.has_unfinished():
            break
        for out in engine.step():
            outputs[out.seq_id].append(out.token_id)
        assert step < 9_999, "no progress"
    check_against_hf(
        tiny_hf_model, prompts, [outputs[i] for i in ids], max_tokens, "e2e random workload"
    )
    assert engine.block_manager.num_free_blocks == engine.block_manager.num_blocks


# --- scenarios specific to mixed batching (still run in both modes; the mid-prefill preemption
# assertion only applies when prefills are chunked) --------------------------------------------


def test_budget_smaller_than_every_prompt(
    tiny_model_dir: Path, tiny_hf_model: HFQwen3ForCausalLM
) -> None:
    engine = make_engine(tiny_model_dir, max_num_seqs=4, max_num_batched_tokens=4)
    prompts = [prompt(n, seed=700 + n) for n in (5, 9, 13, 20)]
    outs = engine.generate(prompts, params(6))
    check_against_hf(tiny_hf_model, prompts, outs, [6] * 4, "e2e chunked")


def test_preemption_during_chunked_prefill(
    tiny_model_dir: Path, tiny_hf_model: HFQwen3ForCausalLM
) -> None:
    mid_prefill: list[tuple[int, int, int]] = []
    total = 0
    for seed in range(6):
        rng = random.Random(900 + seed)
        engine = make_engine(
            tiny_model_dir,
            num_blocks=rng.randint(7, 10),
            max_num_seqs=4,
            max_num_batched_tokens=rng.choice([4, 6, 8]),
        )
        real_preempt = engine.scheduler._preempt

        def spy(seq: Any, real: Any = real_preempt) -> None:
            if 0 < seq.num_computed_tokens < seq.num_prompt_tokens:
                mid_prefill.append((seq.seq_id, seq.num_computed_tokens, seq.num_prompt_tokens))
            real(seq)

        engine.scheduler._preempt = spy  # type: ignore[method-assign]
        prompts = [prompt(rng.randint(8, 18), seed=1000 * seed + i) for i in range(4)]
        max_tokens = [rng.randint(4, 10) for _ in range(4)]
        outs = engine.generate(prompts, [params(n) for n in max_tokens])
        total += engine.num_preemptions
        check_against_hf(tiny_hf_model, prompts, outs, max_tokens, "e2e chunked preemption")
        assert engine.block_manager.num_free_blocks == engine.block_manager.num_blocks
    assert total > 0
    if MODE["mixed"]:
        assert mid_prefill, "no sequence was preempted in the middle of a chunked prefill"
