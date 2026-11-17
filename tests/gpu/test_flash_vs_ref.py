"""FlashAttention backend against the PyTorch reference on the same paged cache (CUDA).

Accuracy criterion, fixed before the first GPU run: against an fp32 "truth" (the reference
backend in fp32 on the same inputs), the flash backend's largest absolute error must be at most
twice the reference backend's own error in the same low-precision dtype, plus one unit of
rounding of that dtype at 1.0. Flash must be no less accurate than the plain PyTorch path.
"""

from __future__ import annotations

import math

import pytest
import torch

from stallfree.attention.flash import FlashAttention
from stallfree.attention.metadata import SegmentSpec, build_attention_metadata
from stallfree.attention.torch_ref import TorchRefAttention
from stallfree.kv_cache import KVCache

BS = 256
H, HKV, D = 16, 8, 128  # Qwen3-0.6B attention shape
UNIT = {torch.bfloat16: 2.0**-8, torch.float16: 2.0**-11}
DEV = "cuda"


def make_cache(num_blocks: int, dtype: torch.dtype) -> KVCache:
    return KVCache(
        num_layers=1,
        num_blocks=num_blocks,
        block_size=BS,
        num_kv_heads=HKV,
        head_dim=D,
        dtype=dtype,
        device=DEV,
    )


def run(
    backend: type,
    dtype: torch.dtype,
    num_blocks: int,
    steps: list[list[tuple[int, int, int, tuple[int, ...]]]],
    qkv: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> list[torch.Tensor]:
    """steps: per step, (seq, start, num_new, table) segments; returns per-step outputs (fp32)."""
    cache = make_cache(num_blocks, dtype)
    outs = []
    for step in steps:
        md = build_attention_metadata([SegmentSpec(s, n, t) for _, s, n, t in step], BS, DEV)
        q = torch.cat([qkv[i][0][s : s + n] for i, s, n, _ in step]).to(dtype)
        k = torch.cat([qkv[i][1][s : s + n] for i, s, n, _ in step]).to(dtype)
        v = torch.cat([qkv[i][2][s : s + n] for i, s, n, _ in step]).to(dtype)
        outs.append(backend(cache, md)(0, q, k, v).float())
    return outs


def random_qkv(length: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator(device=DEV).manual_seed(seed)
    return (
        torch.randn(length, H, D, device=DEV, generator=g),
        torch.randn(length, HKV, D, device=DEV, generator=g),
        torch.randn(length, HKV, D, device=DEV, generator=g),
    )


def check(dtype: torch.dtype, num_blocks: int, steps: list, qkv: dict) -> None:
    truth = run(TorchRefAttention, torch.float32, num_blocks, steps, qkv)
    ref = run(TorchRefAttention, dtype, num_blocks, steps, qkv)
    flash = run(FlashAttention, dtype, num_blocks, steps, qkv)
    err_ref = max(float((r - t).abs().max()) for r, t in zip(ref, truth, strict=True))
    err_flash = max(float((f - t).abs().max()) for f, t in zip(flash, truth, strict=True))
    print(f"{dtype}: flash err {err_flash:.2e}, reference err {err_ref:.2e}")
    assert err_flash <= 2 * err_ref + UNIT[dtype]


def table(n_tokens: int, first: int = 0, stride: int = 1) -> tuple[int, ...]:
    return tuple(first + stride * i for i in range(math.ceil(n_tokens / BS)))


DTYPES = [torch.bfloat16, torch.float16]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("length", [1, 255, 256, 257, 513])
def test_prefill_at_block_boundaries(dtype: torch.dtype, length: int) -> None:
    t = table(length, first=1, stride=2)[::-1]
    check(dtype, 2 * len(t) + 2, [[(0, 0, length, t)]], {0: random_qkv(length, length)})


@pytest.mark.parametrize("dtype", DTYPES)
def test_chunked_prefill_then_decode(dtype: torch.dtype) -> None:
    t = (5, 0, 3)
    steps = [[(0, 0, 200, t)], [(0, 200, 100, t)], [(0, 300, 213, t)]]  # chunks cross 256 and 512
    steps += [[(0, p, 1, t)] for p in range(513, 520)]
    check(dtype, 6, steps, {0: random_qkv(520, 1)})


@pytest.mark.parametrize("dtype", DTYPES)
def test_mixed_segments_and_interleaved_blocks(dtype: torch.dtype) -> None:
    ta, tb, tc = (0, 2, 4), (1, 3, 5), (6, 7)
    qkv = {0: random_qkv(600, 2), 1: random_qkv(520, 3), 2: random_qkv(301, 4)}
    steps = [
        [(0, 0, 300, ta), (1, 0, 511, tb)],
        [(0, 300, 250, ta), (1, 511, 1, tb), (2, 0, 300, tc)],  # chunk, decode, fresh prefill
        [(0, 550, 1, ta), (1, 512, 1, tb), (2, 300, 1, tc)],  # three decodes
    ]
    check(dtype, 8, steps, qkv)
