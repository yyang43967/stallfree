"""Paged KV cache, slot mapping, and the PyTorch reference attention.

Two levels:
1. Attention alone on random Q/K/V, against ``DenseCausalAttention`` on the full sequence.
2. The tiny Qwen3 with paged attention, driven by hand-written steps (fixed block tables), against
   Hugging Face transformers in fp32.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import torch
from helpers.tiny_qwen3 import (
    TINY_CONFIG,
    DenseCausalAttention,
    assert_matches_reference,
    hf_greedy,
    hf_logits,
    make_hf_model,
    save_hf_model,
)
from transformers import Qwen3ForCausalLM as HFQwen3ForCausalLM

from stallfree.attention.metadata import (
    BLOCK_TABLE_PAD,
    SegmentSpec,
    build_attention_metadata,
)
from stallfree.attention.torch_ref import TorchRefAttention
from stallfree.kv_cache import KVCache, slots_for_positions
from stallfree.model.loader import load_model
from stallfree.model.qwen3 import Qwen3ForCausalLM

H, HKV, D = 4, 2, 8  # attention-only tests: 2 query heads per KV head, as in Qwen3


# --- slot mapping and metadata -------------------------------------------------------------


def test_slot_mapping_example() -> None:
    # block_size 4, table (2, 0, 5): positions 3..8 live in block 2 offset 3, block 0 offsets 0-3,
    # then block 5 offset 0.
    slots = slots_for_positions((2, 0, 5), torch.arange(3, 9), block_size=4)
    assert slots.tolist() == [11, 0, 1, 2, 3, 20]


def test_metadata_for_a_mixed_step() -> None:
    md = build_attention_metadata(
        [
            SegmentSpec(start_pos=5, num_new=3, block_table=(7, 2)),  # chunk continuation
            SegmentSpec(start_pos=9, num_new=1, block_table=(0, 1, 4)),  # decode
            SegmentSpec(start_pos=0, num_new=2, block_table=(3,)),  # fresh prefill
        ],
        block_size=4,
    )
    assert md.slot_mapping.tolist() == [9, 10, 11, 17, 12, 13]
    assert md.cu_seqlens_q.tolist() == [0, 3, 4, 6]
    assert md.seq_lens.tolist() == [8, 10, 2]
    assert md.block_tables.tolist() == [
        [7, 2, BLOCK_TABLE_PAD],
        [0, 1, 4],
        [3, BLOCK_TABLE_PAD, BLOCK_TABLE_PAD],
    ]
    assert (md.max_seqlen_q, md.max_seqlen_k) == (3, 10)
    assert md.slot_mapping.dtype == torch.long
    assert md.cu_seqlens_q.dtype == md.seq_lens.dtype == md.block_tables.dtype == torch.int32


@pytest.mark.parametrize(
    "segments",
    [
        [],
        [SegmentSpec(start_pos=0, num_new=0, block_table=(0,))],
        [SegmentSpec(start_pos=-1, num_new=1, block_table=(0,))],
        [SegmentSpec(start_pos=3, num_new=2, block_table=(0,))],  # needs 5 slots, has 4
    ],
)
def test_metadata_rejects_invalid_segments(segments: list[SegmentSpec]) -> None:
    with pytest.raises(ValueError):
        build_attention_metadata(segments, block_size=4)


# --- KV cache storage ----------------------------------------------------------------------


def make_cache(num_blocks: int = 16, block_size: int = 4, layers: int = 1) -> KVCache:
    return KVCache(
        num_layers=layers,
        num_blocks=num_blocks,
        block_size=block_size,
        num_kv_heads=HKV,
        head_dim=D,
        dtype=torch.float32,
    )


def test_write_lands_in_block_and_offset() -> None:
    cache = make_cache(num_blocks=5, layers=2)
    k = torch.randn(6, HKV, D)
    v = torch.randn(6, HKV, D)
    md = build_attention_metadata([SegmentSpec(0, 6, (3, 1))], block_size=4)
    cache.write(1, k, v, md.slot_mapping)
    for p in range(6):
        block, offset = (3, p) if p < 4 else (1, p - 4)
        assert torch.equal(cache.data[1, 0, block, offset], k[p])
        assert torch.equal(cache.data[1, 1, block, offset], v[p])
    assert not cache.data[0].any(), "other layers untouched"
    for b in (0, 2, 4):
        assert not cache.data[1, :, b].any(), "blocks outside the table untouched"
    assert not cache.data[1, :, 1, 2:].any(), "unused tail of the last block untouched"


def test_bytes_per_token_for_qwen3_0_6b() -> None:
    # 28 layers x (K and V) x 8 KV heads x 128 dims x 2 bytes (bf16) = 112 KiB.
    assert KVCache.bytes_per_token(28, 8, 128, torch.bfloat16) == 114_688 == 112 * 1024


# --- attention alone vs dense --------------------------------------------------------------


@dataclass
class Seq:
    """Random per-position Q/K/V for one sequence; the dense result is the reference."""

    length: int
    seed: int
    table: tuple[int, ...]
    q: torch.Tensor = field(init=False)
    k: torch.Tensor = field(init=False)
    v: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        g = torch.Generator().manual_seed(self.seed)
        self.q = torch.randn(self.length, H, D, generator=g)
        self.k = torch.randn(self.length, HKV, D, generator=g)
        self.v = torch.randn(self.length, HKV, D, generator=g)

    def dense(self) -> torch.Tensor:
        return DenseCausalAttention([self.length])(0, self.q, self.k, self.v)


def paged_step(
    cache: KVCache, parts: list[tuple[Seq, int, int]], block_size: int
) -> list[torch.Tensor]:
    """One attention call for several (sequence, start, num_new) segments; per-segment outputs."""
    md = build_attention_metadata(
        [SegmentSpec(start, n, s.table) for s, start, n in parts], block_size
    )
    q = torch.cat([s.q[start : start + n] for s, start, n in parts])
    k = torch.cat([s.k[start : start + n] for s, start, n in parts])
    v = torch.cat([s.v[start : start + n] for s, start, n in parts])
    out = TorchRefAttention(cache, md)(0, q, k, v)
    return list(out.split([n for _, _, n in parts]))


def table_for(length: int, block_size: int, first: int = 0, stride: int = 1) -> tuple[int, ...]:
    return tuple(first + stride * i for i in range(math.ceil(length / block_size)))


@pytest.mark.parametrize(
    ("block_size", "length"),
    [(4, 1), (4, 3), (4, 4), (4, 5), (4, 7), (4, 8), (4, 9), (256, 255), (256, 256), (256, 257)],
)
def test_prefill_matches_dense_at_block_boundaries(block_size: int, length: int) -> None:
    s = Seq(length, seed=length, table=table_for(length, block_size, first=1, stride=2)[::-1])
    cache = make_cache(num_blocks=2 * len(s.table) + 2, block_size=block_size)
    (out,) = paged_step(cache, [(s, 0, length)], block_size)
    torch.testing.assert_close(out, s.dense())


def test_chunked_prefill_matches_dense() -> None:
    s = Seq(13, seed=1, table=(6, 0, 3, 5))
    cache = make_cache(num_blocks=8)
    outs, start = [], 0
    for n in (3, 4, 5, 1):  # chunks starting mid-block and on block boundaries
        (o,) = paged_step(cache, [(s, start, n)], 4)
        outs.append(o)
        start += n
    torch.testing.assert_close(torch.cat(outs), s.dense())


def test_token_by_token_decode_matches_dense() -> None:
    s = Seq(13, seed=2, table=(1, 7, 4, 2))
    cache = make_cache(num_blocks=8)
    outs = paged_step(cache, [(s, 0, 5)], 4)
    for pos in range(5, 13):
        outs += paged_step(cache, [(s, pos, 1)], 4)
    torch.testing.assert_close(torch.cat(outs), s.dense())


def test_mixed_segments_in_one_call() -> None:
    a = Seq(11, seed=3, table=(4, 0, 6))
    b = Seq(10, seed=4, table=(1, 5, 7))
    c = Seq(6, seed=5, table=(2, 3))
    cache = make_cache(num_blocks=8)
    first = paged_step(cache, [(a, 0, 5), (b, 0, 9)], 4)  # A first chunk, B prefill
    # One call: A continues mid-block (start 5), B decodes, C starts fresh.
    a2, b2, c1 = paged_step(cache, [(a, 5, 6), (b, 9, 1), (c, 0, 6)], 4)
    torch.testing.assert_close(torch.cat([first[0], a2]), a.dense())
    torch.testing.assert_close(torch.cat([first[1], b2]), b.dense())
    torch.testing.assert_close(c1, c.dense())


def test_interleaved_noncontiguous_blocks_stay_isolated() -> None:
    a = Seq(12, seed=6, table=(5, 1, 3))
    b = Seq(12, seed=7, table=(0, 4, 2))
    cache = make_cache(num_blocks=8)
    a_out, b_out = paged_step(cache, [(a, 0, 6), (b, 0, 7)], 4)
    a_outs, b_outs = [a_out], [b_out]
    for pos in range(6, 11):
        a_o, b_o = paged_step(cache, [(a, pos, 1), (b, pos + 1, 1)], 4)
        a_outs.append(a_o)
        b_outs.append(b_o)
    # Poison everything A does not own; A's last decode must not change.
    with torch.no_grad():
        for blk in range(8):
            if blk not in a.table:
                cache.data[:, :, blk] = 1e4
    (a_last,) = paged_step(cache, [(a, 11, 1)], 4)
    a_outs.append(a_last)
    torch.testing.assert_close(torch.cat(a_outs), a.dense())
    torch.testing.assert_close(torch.cat(b_outs), b.dense()[:12])


# --- the model with paged attention vs Hugging Face ----------------------------------------

BS = 4  # small block size so short prompts cross block boundaries


@pytest.fixture(scope="module")
def hf_model() -> HFQwen3ForCausalLM:
    return make_hf_model(seed=0)


@pytest.fixture(scope="module")
def model(
    hf_model: HFQwen3ForCausalLM, tmp_path_factory: pytest.TempPathFactory
) -> Qwen3ForCausalLM:
    path: Path = save_hf_model(hf_model, tmp_path_factory.mktemp("tiny_qwen3"))
    return load_model(path, dtype=torch.float32)


def model_cache(model: Qwen3ForCausalLM, num_blocks: int) -> KVCache:
    cfg = model.config
    return KVCache(
        num_layers=cfg.num_hidden_layers,
        num_blocks=num_blocks,
        block_size=BS,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        dtype=torch.float32,
    )


def model_step(
    model: Qwen3ForCausalLM,
    cache: KVCache,
    parts: list[tuple[list[int], int, int, tuple[int, ...]]],
) -> list[torch.Tensor]:
    """Run one step of (tokens, start, num_new, block_table) segments; logits for every new row."""
    md = build_attention_metadata([SegmentSpec(st, n, tb) for _, st, n, tb in parts], BS)
    ids = torch.tensor([t for toks, st, n, _ in parts for t in toks[st : st + n]])
    positions = torch.cat([torch.arange(st, st + n) for _, st, n, _ in parts])
    with torch.no_grad():
        logits = model.compute_logits(model(ids, positions, TorchRefAttention(cache, md)))
    return list(logits.split([n for _, _, n, _ in parts]))


def random_prompt(n: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    vocab: int = TINY_CONFIG["vocab_size"]  # type: ignore[assignment]
    return [rng.randrange(vocab) for _ in range(n)]


@pytest.mark.parametrize("prompt_len", [3, 4, 5, 9])
def test_prefill_then_decode_matches_hf(
    model: Qwen3ForCausalLM, hf_model: HFQwen3ForCausalLM, prompt_len: int
) -> None:
    new = 15
    tokens = random_prompt(prompt_len, seed=prompt_len)
    blocks = math.ceil((prompt_len + new) / BS)
    table = tuple(range(2 * blocks - 1, -1, -2))  # descending, every other block
    cache = model_cache(model, num_blocks=2 * blocks)
    rows = model_step(model, cache, [(tokens, 0, prompt_len, table)])
    tokens.append(int(rows[0][-1].argmax()))
    while len(tokens) < prompt_len + new:
        (row,) = model_step(model, cache, [(tokens, len(tokens) - 1, 1, table)])
        rows.append(row)
        tokens.append(int(row[-1].argmax()))
    assert tokens[prompt_len:] == hf_greedy(hf_model, tokens[:prompt_len], new)
    ref = hf_logits(hf_model, tokens[:-1])  # every position whose next token we predicted
    assert_matches_reference(torch.cat(rows), ref, "paged attention, decode")


def test_chunked_prefill_matches_hf(model: Qwen3ForCausalLM, hf_model: HFQwen3ForCausalLM) -> None:
    tokens = random_prompt(23, seed=11)
    table = (9, 2, 7, 0, 5, 3, 8, 1)  # 8 blocks for 23 + 8 tokens, scattered
    cache = model_cache(model, num_blocks=10)
    rows, start = [], 0
    for n in (5, 8, 10):  # chunks crossing block boundaries mid-block
        rows += model_step(model, cache, [(tokens, start, n, table)])
        start += n
    tokens.append(int(rows[-1][-1].argmax()))
    for _ in range(7):
        (row,) = model_step(model, cache, [(tokens, len(tokens) - 1, 1, table)])
        rows.append(row)
        tokens.append(int(row[-1].argmax()))
    assert tokens[23:] == hf_greedy(hf_model, tokens[:23], 8)
    assert_matches_reference(torch.cat(rows), hf_logits(hf_model, tokens[:-1]), "paged, chunked")


def test_two_sequences_batched_with_interleaved_blocks_match_hf(
    model: Qwen3ForCausalLM, hf_model: HFQwen3ForCausalLM
) -> None:
    a, b = random_prompt(7, seed=21), random_prompt(10, seed=22)
    ta, tb = (0, 2, 4, 6, 8, 10), (1, 3, 5, 7, 9, 11)  # interleaved
    cache = model_cache(model, num_blocks=12)
    ra, rb = model_step(model, cache, [(a, 0, 7, ta), (b, 0, 10, tb)])
    rows_a, rows_b = [ra], [rb]
    a.append(int(ra[-1].argmax()))
    b.append(int(rb[-1].argmax()))
    for _ in range(12):
        ra, rb = model_step(model, cache, [(a, len(a) - 1, 1, ta), (b, len(b) - 1, 1, tb)])
        rows_a.append(ra)
        rows_b.append(rb)
        a.append(int(ra[-1].argmax()))
        b.append(int(rb[-1].argmax()))
    assert a[7:] == hf_greedy(hf_model, a[:7], 13)
    assert b[10:] == hf_greedy(hf_model, b[:10], 13)
    assert_matches_reference(torch.cat(rows_a), hf_logits(hf_model, a[:-1]), "paged, batched")
    assert_matches_reference(torch.cat(rows_b), hf_logits(hf_model, b[:-1]), "paged, batched")
