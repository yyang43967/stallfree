"""Paged KV cache block manager.

The KV cache is a fixed pool of ``num_blocks`` blocks. Each block holds the keys and values of
``block_size`` consecutive token positions of one sequence, in every layer. A sequence's
*block table* lists, in order, the block ids that hold its positions: position ``p`` lives in
block ``table[p // block_size]`` at offset ``p % block_size``. The model runner turns block
tables into slot indices; this module never touches tensors.

This module owns the block tables. Sequences are identified by an integer ``seq_id``; the block
manager keeps one table per sequence that currently holds blocks.

Scope of this version: allocation, reference counts, and free. Every allocated block has exactly
one owner, so its reference count is 1 while allocated and 0 while free. Reference counts exist
now so that prefix caching can later share a block between sequences without changing this
interface.

Invariants that must hold after every public call:

- ``num_free_blocks + num_used_blocks == num_blocks``.
- A block is free if and only if its reference count is 0.
- Reference counts are never negative.
- Every block id in any block table is in ``range(num_blocks)`` and has reference count >= 1.
- No block id appears in two block tables, or twice in one table (no sharing in this version).
- A sequence that holds blocks for ``n`` tokens has exactly ``ceil(n / block_size)`` blocks.
- Failed calls change nothing (allocation is all-or-nothing).
- Same sequence of calls on two fresh managers gives the same block ids (deterministic).

Policies such as the admission watermark and preemption belong to the scheduler, which reads
``num_free_blocks`` and ``blocks_needed`` and decides; the block manager only does what it is
asked or refuses.

No torch import here.
"""

from __future__ import annotations

from collections import deque


class OutOfBlocksError(RuntimeError):
    """Raised by ``allocate`` when there are not enough free blocks.

    The manager's state is unchanged when this is raised.
    """


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


class BlockManager:
    """Allocates fixed-size KV cache blocks to sequences and frees them.

    Args:
        num_blocks: total number of blocks in the pool. Must be >= 1.
        block_size: tokens per block. Must be >= 1. (The engine uses 256 because FlashAttention-2
            requires it; this class works with any positive size, and tests use small sizes.)

    Raises:
        ValueError: if ``num_blocks`` or ``block_size`` is not a positive integer.

    A fresh manager has all blocks free and no sequences.

    State:
        ``_ref_counts[b]``: reference count of block ``b``.
        ``_free``: ids of the blocks with reference count 0, in the order they will be handed out
            (taken from the left, returned on the right). A deque makes both ends O(1).
        ``_tables[seq_id]``: block ids of that sequence in position order.
    """

    def __init__(self, num_blocks: int, block_size: int) -> None:
        if num_blocks < 1:
            raise ValueError(f"num_blocks must be >= 1, got {num_blocks}")
        if block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {block_size}")
        self._num_blocks = num_blocks
        self._block_size = block_size
        self._ref_counts: list[int] = [0] * num_blocks
        self._free: deque[int] = deque(range(num_blocks))
        self._tables: dict[int, list[int]] = {}

    @property
    def num_blocks(self) -> int:
        """Total number of blocks in the pool (constant)."""
        return self._num_blocks

    @property
    def block_size(self) -> int:
        """Tokens per block (constant)."""
        return self._block_size

    @property
    def num_free_blocks(self) -> int:
        """Number of blocks with reference count 0."""
        return len(self._free)

    @property
    def num_used_blocks(self) -> int:
        """Number of blocks with reference count >= 1."""
        return self._num_blocks - len(self._free)

    def has_seq(self, seq_id: int) -> bool:
        """Whether ``seq_id`` currently holds blocks (has a block table)."""
        return seq_id in self._tables

    def blocks_needed(self, seq_id: int, num_tokens: int) -> int:
        """Additional blocks ``seq_id`` needs so that its table covers ``num_tokens`` tokens.

        ``num_tokens`` is the total the sequence must hold afterwards (already computed tokens
        plus the tokens about to be computed), not an increment.

        - Unknown ``seq_id``: ``ceil(num_tokens / block_size)``.
        - Known ``seq_id`` holding ``k`` blocks: ``max(0, ceil(num_tokens / block_size) - k)``.

        Examples with ``block_size = 4``: a new sequence needs 1 block for 1..4 tokens and 2 for
        5..8. A sequence holding 1 block needs 0 more for 4 tokens and 1 more for 5.

        Pure query: never changes state.

        Raises:
            ValueError: if ``num_tokens < 1``.
        """
        if num_tokens < 1:
            raise ValueError(f"num_tokens must be >= 1, got {num_tokens}")
        held = len(self._tables.get(seq_id, ()))
        return max(0, _ceil_div(num_tokens, self._block_size) - held)

    def can_allocate(self, seq_id: int, num_tokens: int) -> bool:
        """Whether ``allocate(seq_id, num_tokens)`` would succeed right now.

        Equivalent to ``blocks_needed(seq_id, num_tokens) <= num_free_blocks``. Pure query.

        Raises:
            ValueError: if ``num_tokens < 1``.
        """
        return self.blocks_needed(seq_id, num_tokens) <= len(self._free)

    def allocate(self, seq_id: int, num_tokens: int) -> None:
        """Grow ``seq_id``'s block table so it covers ``num_tokens`` tokens.

        Used for every case where a sequence needs room: a new prompt (unknown ``seq_id``, creates
        the table), the next prefill chunk, or one decode token (often needs no new block, and a
        new block exactly when the previous one is full).

        - Takes ``blocks_needed(seq_id, num_tokens)`` blocks from the free pool, sets each one's
          reference count to 1, and appends them to the end of the table in order.
        - If the table already covers ``num_tokens``, does nothing (tables never shrink here).
        - All-or-nothing: if there are not enough free blocks, raises and changes nothing. In
          particular an unknown ``seq_id`` does not get an empty table.
        - Block choice must be deterministic for a given call history. Which free block is taken
          is otherwise up to the implementation.

        Raises:
            ValueError: if ``num_tokens < 1``.
            OutOfBlocksError: if fewer than ``blocks_needed`` blocks are free.
        """
        needed = self.blocks_needed(seq_id, num_tokens)  # validates num_tokens
        if needed > len(self._free):
            raise OutOfBlocksError(
                f"seq {seq_id} needs {needed} blocks for {num_tokens} tokens, "
                f"{len(self._free)} free"
            )
        if needed == 0:
            return  # already covered; only possible for a known sequence (num_tokens >= 1)
        # The check above guarantees the loop cannot fail halfway, so the call is atomic.
        table = self._tables.setdefault(seq_id, [])
        for _ in range(needed):
            block_id = self._free.popleft()
            assert self._ref_counts[block_id] == 0, f"free block {block_id} has references"
            self._ref_counts[block_id] = 1
            table.append(block_id)

    def free(self, seq_id: int) -> None:
        """Release every block held by ``seq_id`` and forget its table.

        Decrements the reference count of each block in the table; blocks that reach 0 return
        to the free pool. Called when a sequence finishes, is aborted, or is preempted
        (recompute preemption frees everything and re-prefills later).

        Blocks are returned last block first. Today this only fixes the reuse order; with prefix
        caching, the tail of a sequence (least likely to be shared) is then evicted before its
        head.

        Raises:
            KeyError: if ``seq_id`` holds no blocks (unknown, or already freed). A double free is
                a scheduler bug and must not pass silently.
        """
        if seq_id not in self._tables:
            raise KeyError(f"seq {seq_id} holds no blocks (unknown or already freed)")
        table = self._tables.pop(seq_id)
        for block_id in reversed(table):
            assert self._ref_counts[block_id] > 0, f"block {block_id} freed twice"
            self._ref_counts[block_id] -= 1
            if self._ref_counts[block_id] == 0:
                self._free.append(block_id)

    def block_table(self, seq_id: int) -> tuple[int, ...]:
        """The block ids of ``seq_id`` in position order, as an immutable snapshot.

        Raises:
            KeyError: if ``seq_id`` holds no blocks.
        """
        if seq_id not in self._tables:
            raise KeyError(f"seq {seq_id} holds no blocks")
        return tuple(self._tables[seq_id])

    def ref_count(self, block_id: int) -> int:
        """Reference count of ``block_id`` (0 if free).

        Raises:
            IndexError: if ``block_id`` is not in ``range(num_blocks)``.
        """
        # Explicit check: a negative index would otherwise wrap around silently.
        if not 0 <= block_id < self._num_blocks:
            raise IndexError(f"block_id {block_id} out of range [0, {self._num_blocks})")
        return self._ref_counts[block_id]
