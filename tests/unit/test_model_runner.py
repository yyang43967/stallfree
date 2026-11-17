"""Model runner driven by a scripted plan (no scheduler), against Hugging Face in fp32.

Every plan below is written by hand: start positions, chunk sizes, block tables, and which
entries sample. Tokens are the prompt followed by HF's own greedy continuation, so at every
sampled row the runner must predict exactly HF's next token, and the row's logits must match
HF's logits at that position.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest
import torch
from helpers.scripted import Planned, scripted_steps
from helpers.tiny_qwen3 import (
    TINY_CONFIG,
    assert_matches_reference,
    hf_greedy,
    hf_logits,
    make_hf_model,
    save_hf_model,
)
from transformers import Qwen3ForCausalLM as HFQwen3ForCausalLM

from stallfree.kv_cache import KVCache
from stallfree.model.loader import load_model
from stallfree.model.qwen3 import Qwen3ForCausalLM
from stallfree.model_runner import ModelRunner
from stallfree.scheduler import SchedulerOutput

BS = 4


@pytest.fixture(scope="module")
def hf_model() -> HFQwen3ForCausalLM:
    return make_hf_model(seed=0)


@pytest.fixture(scope="module")
def model(
    hf_model: HFQwen3ForCausalLM, tmp_path_factory: pytest.TempPathFactory
) -> Qwen3ForCausalLM:
    path: Path = save_hf_model(hf_model, tmp_path_factory.mktemp("tiny_qwen3"))
    return load_model(path, dtype=torch.float32)


def make_runner(model: Qwen3ForCausalLM, num_blocks: int = 32) -> ModelRunner:
    cfg = model.config
    cache = KVCache(
        num_layers=cfg.num_hidden_layers,
        num_blocks=num_blocks,
        block_size=BS,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        dtype=torch.float32,
    )
    return ModelRunner(model, cache)


def reference(hf_model: HFQwen3ForCausalLM, prompt_len: int, new: int, seed: int) -> list[int]:
    """Prompt followed by HF's greedy continuation of ``new`` tokens."""
    rng = random.Random(seed)
    vocab: int = TINY_CONFIG["vocab_size"]  # type: ignore[assignment]
    prompt = [rng.randrange(vocab) for _ in range(prompt_len)]
    return prompt + hf_greedy(hf_model, prompt, new)


def run_and_check(
    runner: ModelRunner,
    hf_model: HFQwen3ForCausalLM,
    plan: list[list[Planned]],
    tokens: dict[int, list[int]],
    label: str,
) -> int:
    """Execute the plan; check every sampled row against HF. Returns the number of rows checked."""
    ref = {sid: hf_logits(hf_model, toks[:-1]) for sid, toks in tokens.items()}
    checked = 0
    for out, step in zip(scripted_steps(plan, tokens), plan, strict=True):
        result = runner.execute(out)
        sampled = [p for p in step if p.sample]
        assert result.seq_ids == tuple(p.seq_id for p in sampled)
        for row, p in zip(result.logits, sampled, strict=True):
            pos = p.start + p.num_new - 1  # position whose next token this row predicts
            assert int(row.argmax()) == tokens[p.seq_id][pos + 1], f"seq {p.seq_id} pos {pos}"
            assert_matches_reference(row[None], ref[p.seq_id][pos][None], label)
            checked += 1
    return checked


def decodes(seq_id: int, start: int, end: int, table: tuple[int, ...]) -> list[list[Planned]]:
    """One single-token step per position in [start, end)."""
    return [[Planned(seq_id, pos, 1, table)] for pos in range(start, end)]


# --- prepare ---------------------------------------------------------------------------------


def test_prepare_builds_the_flat_batch(model: Qwen3ForCausalLM) -> None:
    runner = make_runner(model)
    tokens = {0: list(range(100, 112)), 1: list(range(200, 210)), 2: [7, 8]}
    plan = [
        [
            Planned(0, 5, 3, (7, 2)),  # chunk continuation, sampled
            Planned(1, 9, 1, (0, 1, 4)),  # decode
            Planned(2, 0, 2, (3,), sample=False),  # first chunk of a prefill, not sampled
        ]
    ]
    (out,) = scripted_steps(plan, tokens)
    step = runner.prepare(out)
    assert step.input_ids.tolist() == [105, 106, 107, 209, 7, 8]
    assert step.positions.tolist() == [5, 6, 7, 9, 0, 1]
    assert step.metadata.slot_mapping.tolist() == [9, 10, 11, 17, 12, 13]
    assert step.metadata.cu_seqlens_q.tolist() == [0, 3, 4, 6]
    assert step.metadata.seq_lens.tolist() == [8, 10, 2]
    assert step.sample_rows.tolist() == [2, 3]
    assert step.sample_seq_ids == (0, 1)


def test_prepare_rejects_an_empty_step(model: Qwen3ForCausalLM) -> None:
    with pytest.raises(ValueError):
        make_runner(model).prepare(SchedulerOutput())


def test_unsampled_entries_produce_no_logits(model: Qwen3ForCausalLM) -> None:
    runner = make_runner(model)
    plan = [[Planned(0, 0, 3, (0,), sample=False)]]
    (out,) = scripted_steps(plan, {0: [1, 2, 3]})
    result = runner.execute(out)
    assert result.seq_ids == ()
    assert result.logits.shape == (0, model.config.vocab_size)


# --- scripted plans against HF ---------------------------------------------------------------


def test_prefill_then_decode(model: Qwen3ForCausalLM, hf_model: HFQwen3ForCausalLM) -> None:
    toks = reference(hf_model, prompt_len=6, new=10, seed=1)
    table = (9, 3, 6, 0)  # 16 tokens -> 4 blocks, scattered
    plan = [[Planned(0, 0, 6, table)], *decodes(0, 6, 15, table)]
    assert run_and_check(make_runner(model), hf_model, plan, {0: toks}, "runner, scripted") == 10


def test_chunked_prefill_with_unsampled_chunks(
    model: Qwen3ForCausalLM, hf_model: HFQwen3ForCausalLM
) -> None:
    toks = reference(hf_model, prompt_len=23, new=8, seed=2)
    table = (4, 11, 0, 7, 2, 9, 5, 1)
    plan = [
        [Planned(0, 0, 5, table, sample=False)],
        [Planned(0, 5, 8, table, sample=False)],
        [Planned(0, 13, 10, table)],  # last chunk samples the first new token
        *decodes(0, 23, 30, table),
    ]
    assert run_and_check(make_runner(model), hf_model, plan, {0: toks}, "runner, scripted") == 8


def test_mixed_step_chunk_decode_and_fresh_prefill(
    model: Qwen3ForCausalLM, hf_model: HFQwen3ForCausalLM
) -> None:
    a = reference(hf_model, prompt_len=11, new=4, seed=3)
    b = reference(hf_model, prompt_len=9, new=4, seed=4)
    c = reference(hf_model, prompt_len=6, new=3, seed=5)
    ta, tb, tc = (0, 5, 10, 15), (1, 6, 11), (2, 7)
    plan = [
        [Planned(0, 0, 5, ta, sample=False), Planned(1, 0, 9, tb)],
        # One step: A continues its prompt mid-block, B decodes, C starts.
        [Planned(0, 5, 6, ta), Planned(1, 9, 1, tb), Planned(2, 0, 6, tc)],
        [Planned(0, 11, 1, ta), Planned(1, 10, 1, tb), Planned(2, 6, 1, tc)],
        [Planned(0, 12, 1, ta), Planned(1, 11, 1, tb), Planned(2, 7, 1, tc)],
    ]
    n = run_and_check(make_runner(model), hf_model, plan, {0: a, 1: b, 2: c}, "runner, scripted")
    assert n == 1 + 3 + 3 + 3


def test_recompute_into_new_blocks_after_dropping_the_cache(
    model: Qwen3ForCausalLM, hf_model: HFQwen3ForCausalLM
) -> None:
    """What a preemption looks like to the runner: the same sequence is prefilled again from
    position 0 (prompt + generated tokens) into different blocks, then keeps decoding."""
    toks = reference(hf_model, prompt_len=7, new=12, seed=6)
    first, second = (0, 1, 2), (8, 5, 3, 6, 4)
    plan = [
        [Planned(0, 0, 7, first)],
        *decodes(0, 7, 10, first),  # generated tokens at positions 7, 8, 9, 10
        [Planned(0, 0, 11, second)],  # recompute all 11 tokens into new blocks
        *decodes(0, 11, 18, second),
    ]
    runner = make_runner(model, num_blocks=10)
    tokens = {0: toks}
    ref = hf_logits(hf_model, toks[:-1])
    for i, (out, step) in enumerate(zip(scripted_steps(plan, tokens), plan, strict=True)):
        if i == 4:  # just before the recompute: the old blocks now hold someone else's data
            with torch.no_grad():
                runner.cache.data[:, :, list(first)] = 1e4
        result = runner.execute(out)
        (p,) = step
        pos = p.start + p.num_new - 1
        assert int(result.logits[0].argmax()) == toks[pos + 1]
        assert_matches_reference(result.logits, ref[pos][None], "runner, scripted")
