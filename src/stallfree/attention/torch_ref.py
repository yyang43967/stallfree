"""Reference paged attention in plain PyTorch (any device, any dtype).

For each layer: write the step's new K/V into the cache through ``slot_mapping``, then, segment by
segment, gather the segment's whole context from the cache through its block table and run causal
attention. Keys and values are read only from the cache, never from the ``k``/``v`` arguments, so
this path exercises exactly what the GPU backend will read.

Slow (a Python loop over segments) but simple and exact; used for CPU correctness tests and as a
cross-check of the FlashAttention backend.
"""

from __future__ import annotations

import torch

from stallfree.attention.metadata import AttentionMetadata
from stallfree.kv_cache import KVCache, slots_for_positions


class TorchRefAttention:
    """``AttentionBackend`` over a paged ``KVCache`` for one step's ``AttentionMetadata``."""

    def __init__(self, cache: KVCache, metadata: AttentionMetadata) -> None:
        if metadata.block_size != cache.block_size:
            raise ValueError("metadata and cache disagree on block_size")
        self.cache = cache
        self.md = metadata

    def __call__(
        self, layer_idx: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        # q: [T, H, D]; k, v: [T, Hkv, D] for the step's new tokens only.
        md = self.md
        self.cache.write(layer_idx, k, v, md.slot_mapping)
        k_slots = self.cache.k_slots(layer_idx)  # [num_blocks * block_size, Hkv, D]
        v_slots = self.cache.v_slots(layer_idx)
        groups = q.shape[1] // k.shape[1]  # query heads per KV head (GQA)
        scale = q.shape[-1] ** -0.5
        cu = md.cu_seqlens_q.tolist()
        seq_lens = md.seq_lens.tolist()
        out = torch.empty_like(q)
        for i in range(md.num_segments):
            q_start, q_end = cu[i], cu[i + 1]
            n = q_end - q_start  # new tokens (queries) in this segment
            ctx = seq_lens[i]  # cached + new positions this segment may attend to
            start_pos = ctx - n  # absolute position of the segment's first query
            positions = torch.arange(ctx, device=q.device)
            slots = slots_for_positions(md.block_tables[i], positions, md.block_size)
            # Gather the context from the cache and expand KV heads to query heads.
            ks = k_slots[slots].repeat_interleave(groups, dim=1).transpose(0, 1)  # [H, ctx, D]
            vs = v_slots[slots].repeat_interleave(groups, dim=1).transpose(0, 1)  # [H, ctx, D]
            qs = q[q_start:q_end].transpose(0, 1)  # [H, n, D]
            scores = torch.matmul(qs, ks.transpose(1, 2)) * scale  # [H, n, ctx]
            # Causal mask aligned to the end: query j sits at position start_pos + j and may see
            # keys at positions <= start_pos + j.
            mask = torch.full((n, ctx), torch.finfo(q.dtype).min, dtype=q.dtype, device=q.device)
            mask = mask.triu(start_pos + 1)
            weights = torch.softmax(scores + mask, dim=-1, dtype=torch.float32).to(q.dtype)
            out[q_start:q_end] = torch.matmul(weights, vs).transpose(0, 1)  # [n, H, D]
        return out
