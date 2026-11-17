"""FlashAttention-2 paged attention (CUDA only).

Writes the step's new K/V into the cache through ``slot_mapping`` (the same ``index_copy_`` as the
reference backend), then makes one ``flash_attn_varlen_func`` call per layer for the whole mixed
batch: every segment, prefill chunk or decode, is one variable-length sequence whose keys are read
through its block table. Causal masking in FlashAttention >= 2.1 is aligned to the end of each
segment's context, which is what a chunk continuation or a decode needs.

``flash_attn`` is imported when the backend is built, so the package imports without it.
"""

from __future__ import annotations

from typing import Any

import torch

from stallfree.attention.metadata import AttentionMetadata
from stallfree.kv_cache import KVCache


class FlashAttention:
    """``AttentionBackend`` over a paged ``KVCache`` using FlashAttention-2.

    Raises:
        ValueError: if the cache is not on a CUDA device or its block size is not a multiple of
            256 (FlashAttention-2's paged-KV requirement).
        ImportError: if ``flash_attn`` is not installed.
    """

    def __init__(self, cache: KVCache, metadata: AttentionMetadata) -> None:
        if cache.data.device.type != "cuda":
            raise ValueError("the flash backend needs a CUDA device")
        if cache.block_size % 256:
            raise ValueError("FlashAttention-2 paged KV needs block_size divisible by 256")
        from flash_attn import flash_attn_varlen_func  # lazy: optional, CUDA-only dependency

        self._varlen: Any = flash_attn_varlen_func
        self.cache = cache
        self.md = metadata

    def __call__(
        self, layer_idx: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        # q: [T, H, D]; k, v: [T, Hkv, D] for the step's new tokens.
        md = self.md
        self.cache.write(layer_idx, k, v, md.slot_mapping)
        out: torch.Tensor = self._varlen(
            q,
            self.cache.data[layer_idx, 0],  # [num_blocks, block_size, Hkv, D]
            self.cache.data[layer_idx, 1],
            cu_seqlens_q=md.cu_seqlens_q,
            cu_seqlens_k=md.cu_seqlens_k,
            max_seqlen_q=md.max_seqlen_q,
            max_seqlen_k=md.max_seqlen_k,
            softmax_scale=q.shape[-1] ** -0.5,
            causal=True,
            block_table=md.block_tables,
        )
        return out  # [T, H, D]
