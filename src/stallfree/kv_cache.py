"""Paged KV cache storage.

One tensor holds every layer's keys and values:
``[num_layers, 2, num_blocks, block_size, num_kv_heads, head_dim]`` (index 0 = K, 1 = V).
Block ``b`` of a layer holds positions ``offset = 0 .. block_size-1`` of whichever sequence owns
that block. A *slot* is a flat index ``b * block_size + offset``; writes and reads address the
cache by slot through a ``[num_blocks * block_size, num_kv_heads, head_dim]`` view.

Which sequence owns which block is the block manager's business; this module only stores numbers.
"""

from __future__ import annotations

import torch


def slots_for_positions(
    block_table: torch.Tensor | tuple[int, ...], positions: torch.Tensor, block_size: int
) -> torch.Tensor:
    """Slot index of each position: ``block_table[p // block_size] * block_size + p % block_size``.

    Args:
        block_table: the sequence's block ids, in position order.
        positions: ``[N]`` int64 positions within the sequence.
        block_size: tokens per block.

    Returns:
        ``[N]`` int64 slot indices.
    """
    table = torch.as_tensor(block_table, dtype=torch.long, device=positions.device)
    return table[positions // block_size] * block_size + positions % block_size


class KVCache:
    """Preallocated K/V storage for all layers.

    Args:
        num_layers, num_blocks, block_size, num_kv_heads, head_dim: cache geometry.
        dtype, device: storage format and location.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: str | torch.device = "cpu",
    ) -> None:
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        # Zero-filled so that a read of a never-written slot is deterministic in tests.
        shape = (num_layers, 2, num_blocks, block_size, num_kv_heads, head_dim)
        self.data = torch.zeros(shape, dtype=dtype, device=device)

    @staticmethod
    def bytes_per_token(
        num_layers: int, num_kv_heads: int, head_dim: int, dtype: torch.dtype
    ) -> int:
        """Cache bytes one token needs across all layers (K and V)."""
        itemsize = torch.empty((), dtype=dtype).element_size()
        return num_layers * 2 * num_kv_heads * head_dim * itemsize

    def k_slots(self, layer_idx: int) -> torch.Tensor:
        """Keys of one layer addressed by slot: ``[num_blocks * block_size, Hkv, D]`` (a view)."""
        return self.data[layer_idx, 0].view(-1, self.num_kv_heads, self.head_dim)

    def v_slots(self, layer_idx: int) -> torch.Tensor:
        """Values of one layer addressed by slot: ``[num_blocks * block_size, Hkv, D]`` (a view)."""
        return self.data[layer_idx, 1].view(-1, self.num_kv_heads, self.head_dim)

    def write(
        self, layer_idx: int, k: torch.Tensor, v: torch.Tensor, slot_mapping: torch.Tensor
    ) -> None:
        """Store the step's new keys/values. ``k, v: [T, Hkv, D]``; ``slot_mapping: [T]`` int64."""
        self.k_slots(layer_idx).index_copy_(0, slot_mapping, k)
        self.v_slots(layer_idx).index_copy_(0, slot_mapping, v)
