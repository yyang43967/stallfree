"""Size the KV cache from GPU memory.

After the weights are loaded, one forward pass of the largest step the scheduler can produce is
run on a throwaway cache, and the allocator's peak is recorded. The KV cache then gets

    utilization * total  -  memory already in use  -  peak activation memory of that forward

divided into whole blocks. "In use" is measured from the device (``total - free``), so it includes
the weights, the CUDA context, and anything else on the GPU.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch

from stallfree.attention.base import AttentionBackend
from stallfree.attention.metadata import AttentionMetadata, SegmentSpec, build_attention_metadata
from stallfree.kv_cache import KVCache
from stallfree.model.qwen3 import Qwen3ForCausalLM


def num_kv_blocks(
    *,
    total_bytes: int,
    utilization: float,
    used_bytes: int,
    activation_peak_bytes: int,
    block_bytes: int,
) -> int:
    """Whole KV blocks that fit in the memory left for the cache.

    Raises:
        ValueError: if not even one block fits.
    """
    budget = int(total_bytes * utilization) - used_bytes - activation_peak_bytes
    if budget < block_bytes:
        raise ValueError(
            f"no room for the KV cache: budget {budget / 2**30:.2f} GiB "
            f"< one block {block_bytes / 2**20:.1f} MiB (raise gpu_memory_utilization or "
            "lower max_num_batched_tokens / max_model_len)"
        )
    return budget // block_bytes


@dataclass(frozen=True, slots=True)
class MemoryReport:
    """What the profiling pass measured, in bytes, and the resulting block count."""

    total: int
    used_after_load: int
    activation_peak: int
    block_bytes: int
    num_blocks: int
    profile_tokens: int

    def summary(self) -> str:
        gib = 2**30
        return (
            f"total {self.total / gib:.2f} GiB, in use {self.used_after_load / gib:.2f} GiB, "
            f"activation peak {self.activation_peak / gib:.2f} GiB ({self.profile_tokens} tokens), "
            f"{self.num_blocks} blocks of {self.block_bytes / 2**20:.1f} MiB "
            f"= {self.num_blocks * self.block_bytes / gib:.2f} GiB KV cache"
        )


@torch.inference_mode()
def profile_num_blocks(
    model: Qwen3ForCausalLM,
    *,
    attention: Callable[[KVCache, AttentionMetadata], AttentionBackend],
    block_size: int,
    dtype: torch.dtype,
    device: torch.device,
    profile_tokens: int,
    sampled_rows: int,
    utilization: float,
) -> MemoryReport:
    """Run the largest forward on a throwaway cache and size the real cache from the peak."""
    cfg = model.config
    blocks = math.ceil(profile_tokens / block_size)
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    before = torch.cuda.memory_allocated(device)
    scratch = KVCache(
        num_layers=cfg.num_hidden_layers,
        num_blocks=blocks,
        block_size=block_size,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        dtype=dtype,
        device=device,
    )
    scratch_bytes = scratch.data.numel() * scratch.data.element_size()
    md = build_attention_metadata(
        [SegmentSpec(0, profile_tokens, tuple(range(blocks)))], block_size, device
    )
    ids = torch.zeros(profile_tokens, dtype=torch.long, device=device)
    positions = torch.arange(profile_tokens, device=device)
    hidden = model(ids, positions, attention(scratch, md))
    model.compute_logits(hidden[-min(sampled_rows, profile_tokens) :])
    torch.cuda.synchronize(device)
    activation_peak = torch.cuda.max_memory_allocated(device) - before - scratch_bytes
    del scratch, md, ids, positions, hidden
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info(device)
    block_bytes = (
        KVCache.bytes_per_token(cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim, dtype)
        * block_size
    )
    n = num_kv_blocks(
        total_bytes=total,
        utilization=utilization,
        used_bytes=total - free,
        activation_peak_bytes=activation_peak,
        block_bytes=block_bytes,
    )
    return MemoryReport(total, total - free, activation_peak, block_bytes, n, profile_tokens)
