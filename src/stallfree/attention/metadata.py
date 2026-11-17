"""Per-step attention metadata, built once per step and shared by every layer.

A step's batch is a list of *segments*, one per scheduled sequence, concatenated in order. Segment
``i`` has ``num_new`` query tokens at positions ``start_pos .. start_pos + num_new - 1`` and attends
to its whole context, positions ``0 .. start_pos + num_new - 1``: the cached ones plus the new ones
(which are written to the cache first). A decode is a segment with ``num_new == 1``; a prefill
chunk that continues an earlier one has ``start_pos > 0``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from stallfree.kv_cache import slots_for_positions

BLOCK_TABLE_PAD = -1  # padding for rows shorter than the longest block table; never read


@dataclass(frozen=True, slots=True)
class SegmentSpec:
    """One sequence's part of a step, as the runner sees it."""

    start_pos: int
    num_new: int
    block_table: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class AttentionMetadata:
    """Tensors describing a step's segments.

    Attributes:
        slot_mapping: ``[T]`` int64, cache slot for each new token's K/V.
        cu_seqlens_q: ``[B + 1]`` int32, cumulative query lengths; segment ``i`` owns query rows
            ``cu_seqlens_q[i] .. cu_seqlens_q[i+1]-1``.
        seq_lens: ``[B]`` int32, context length of each segment (``start_pos + num_new``).
        block_tables: ``[B, max_blocks]`` int32, padded with ``BLOCK_TABLE_PAD``.
        max_seqlen_q, max_seqlen_k: longest query and context length in the step.
        block_size: tokens per cache block.
    """

    slot_mapping: torch.Tensor
    cu_seqlens_q: torch.Tensor
    seq_lens: torch.Tensor
    block_tables: torch.Tensor
    max_seqlen_q: int
    max_seqlen_k: int
    block_size: int

    @property
    def num_segments(self) -> int:
        return int(self.seq_lens.numel())


def build_attention_metadata(
    segments: Sequence[SegmentSpec], block_size: int, device: str | torch.device = "cpu"
) -> AttentionMetadata:
    """Build the step's tensors from its segments, in order.

    Raises:
        ValueError: if a segment is empty, starts at a negative position, or its block table does
            not cover ``start_pos + num_new`` positions.
    """
    if not segments:
        raise ValueError("a step needs at least one segment")
    slot_parts: list[torch.Tensor] = []
    cu = [0]
    seq_lens: list[int] = []
    max_blocks = max(len(s.block_table) for s in segments)
    tables = torch.full((len(segments), max_blocks), BLOCK_TABLE_PAD, dtype=torch.int32)
    for i, s in enumerate(segments):
        if s.num_new < 1 or s.start_pos < 0:
            raise ValueError(f"segment {i}: start_pos={s.start_pos}, num_new={s.num_new}")
        end = s.start_pos + s.num_new
        if len(s.block_table) * block_size < end:
            raise ValueError(
                f"segment {i}: {len(s.block_table)} blocks of {block_size} cannot hold {end} tokens"
            )
        positions = torch.arange(s.start_pos, end, dtype=torch.long)
        slot_parts.append(slots_for_positions(s.block_table, positions, block_size))
        cu.append(cu[-1] + s.num_new)
        seq_lens.append(end)
        tables[i, : len(s.block_table)] = torch.tensor(s.block_table, dtype=torch.int32)
    return AttentionMetadata(
        slot_mapping=torch.cat(slot_parts).to(device),
        cu_seqlens_q=torch.tensor(cu, dtype=torch.int32, device=device),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=device),
        block_tables=tables.to(device),
        max_seqlen_q=max(s.num_new for s in segments),
        max_seqlen_k=max(seq_lens),
        block_size=block_size,
    )
