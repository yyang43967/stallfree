"""CPU-side checks for the GPU path: cache sizing arithmetic, varlen key boundaries, and the
flash backend's guards. The backend itself runs in the GPU tests."""

from __future__ import annotations

import pytest
import torch

from stallfree.attention.flash import FlashAttention
from stallfree.attention.metadata import SegmentSpec, build_attention_metadata
from stallfree.kv_cache import KVCache
from stallfree.memory import num_kv_blocks

GIB = 2**30
MIB = 2**20


def test_num_kv_blocks_example() -> None:
    # 24 GiB card at 0.9 -> 21.6 GiB; minus 2 GiB in use and 1 GiB activations -> 18.6 GiB;
    # Qwen3-0.6B bf16 block of 256 tokens = 28 MiB -> 680 whole blocks.
    n = num_kv_blocks(
        total_bytes=24 * GIB,
        utilization=0.9,
        used_bytes=2 * GIB,
        activation_peak_bytes=1 * GIB,
        block_bytes=28 * MIB,
    )
    assert n == int(24 * GIB * 0.9 - 3 * GIB) // (28 * MIB) == 680


def test_num_kv_blocks_rounds_down() -> None:
    assert (
        num_kv_blocks(
            total_bytes=1000,
            utilization=1.0,
            used_bytes=0,
            activation_peak_bytes=0,
            block_bytes=300,
        )
        == 3
    )


def test_num_kv_blocks_fails_when_nothing_fits() -> None:
    with pytest.raises(ValueError, match="no room"):
        num_kv_blocks(
            total_bytes=10 * GIB,
            utilization=0.5,
            used_bytes=4 * GIB,
            activation_peak_bytes=1 * GIB,
            block_bytes=28 * MIB,
        )


def test_qwen3_0_6b_block_is_28_mib() -> None:
    assert KVCache.bytes_per_token(28, 8, 128, torch.bfloat16) * 256 == 28 * MIB


def test_cu_seqlens_k_accumulates_context_lengths() -> None:
    md = build_attention_metadata(
        [SegmentSpec(5, 3, (7, 2)), SegmentSpec(9, 1, (0, 1, 4)), SegmentSpec(0, 2, (3,))],
        block_size=4,
    )
    assert md.seq_lens.tolist() == [8, 10, 2]
    assert md.cu_seqlens_k.tolist() == [0, 8, 18, 20]
    assert md.cu_seqlens_k.dtype == torch.int32


def _cache(block_size: int) -> KVCache:
    return KVCache(
        num_layers=1,
        num_blocks=2,
        block_size=block_size,
        num_kv_heads=1,
        head_dim=8,
        dtype=torch.float32,
    )


def test_flash_backend_refuses_a_cpu_cache() -> None:
    md = build_attention_metadata([SegmentSpec(0, 2, (0,))], block_size=256)
    with pytest.raises(ValueError, match="CUDA"):
        FlashAttention(_cache(256), md)
