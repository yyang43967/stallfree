"""Contract tests for the paged KV block manager (allocation, reference counts, free)."""

from __future__ import annotations

import math
import random

import pytest

from stallfree.block_manager import BlockManager, OutOfBlocksError

BS = 4  # small block size keeps the arithmetic readable


def check_invariants(bm: BlockManager, seq_ids: list[int]) -> None:
    """Assert every invariant from the module docstring that is visible through the public API.

    ``seq_ids`` are the sequences the caller believes currently hold blocks.
    """
    assert bm.num_free_blocks + bm.num_used_blocks == bm.num_blocks
    refs = [bm.ref_count(b) for b in range(bm.num_blocks)]
    assert all(r >= 0 for r in refs)
    assert sum(1 for r in refs if r == 0) == bm.num_free_blocks

    seen: list[int] = []
    for sid in seq_ids:
        assert bm.has_seq(sid)
        table = bm.block_table(sid)
        assert all(0 <= b < bm.num_blocks for b in table)
        assert all(bm.ref_count(b) == 1 for b in table)
        seen.extend(table)
    assert len(seen) == len(set(seen)), "a block appears in two tables or twice in one"
    assert len(seen) == bm.num_used_blocks, "used blocks must all belong to some table"


# --- construction ---------------------------------------------------------------------------


def test_fresh_manager_has_all_blocks_free() -> None:
    bm = BlockManager(num_blocks=8, block_size=BS)
    assert bm.num_blocks == 8
    assert bm.block_size == BS
    assert bm.num_free_blocks == 8
    assert bm.num_used_blocks == 0
    assert not bm.has_seq(0)
    check_invariants(bm, [])


@pytest.mark.parametrize(("num_blocks", "block_size"), [(0, 4), (-1, 4), (8, 0), (8, -4)])
def test_invalid_construction_raises(num_blocks: int, block_size: int) -> None:
    with pytest.raises(ValueError):
        BlockManager(num_blocks=num_blocks, block_size=block_size)


# --- blocks_needed and can_allocate ----------------------------------------------------------


@pytest.mark.parametrize(
    ("num_tokens", "expected"),
    [(1, 1), (BS - 1, 1), (BS, 1), (BS + 1, 2), (2 * BS, 2), (2 * BS + 1, 3)],
)
def test_blocks_needed_for_new_sequence(num_tokens: int, expected: int) -> None:
    bm = BlockManager(num_blocks=8, block_size=BS)
    assert bm.blocks_needed(seq_id=0, num_tokens=num_tokens) == expected


def test_blocks_needed_for_existing_sequence_counts_only_new_blocks() -> None:
    bm = BlockManager(num_blocks=8, block_size=BS)
    bm.allocate(seq_id=0, num_tokens=BS)  # exactly one full block
    assert bm.blocks_needed(0, BS) == 0
    assert bm.blocks_needed(0, 1) == 0  # never negative, tables never shrink
    assert bm.blocks_needed(0, BS + 1) == 1
    assert bm.blocks_needed(0, 3 * BS) == 2


def test_queries_do_not_change_state() -> None:
    bm = BlockManager(num_blocks=8, block_size=BS)
    bm.allocate(0, 3)
    before = (bm.num_free_blocks, bm.block_table(0))
    bm.blocks_needed(0, 100)
    bm.blocks_needed(1, 5)
    bm.can_allocate(0, 100)
    bm.can_allocate(1, 5)
    assert (bm.num_free_blocks, bm.block_table(0)) == before
    assert not bm.has_seq(1)


@pytest.mark.parametrize("bad", [0, -1])
def test_non_positive_num_tokens_raises(bad: int) -> None:
    bm = BlockManager(num_blocks=8, block_size=BS)
    with pytest.raises(ValueError):
        bm.blocks_needed(0, bad)
    with pytest.raises(ValueError):
        bm.can_allocate(0, bad)
    with pytest.raises(ValueError):
        bm.allocate(0, bad)
    assert bm.num_free_blocks == 8
    assert not bm.has_seq(0)


def test_can_allocate_matches_free_blocks() -> None:
    bm = BlockManager(num_blocks=3, block_size=BS)
    assert bm.can_allocate(0, 3 * BS)
    assert not bm.can_allocate(0, 3 * BS + 1)
    bm.allocate(0, 2 * BS)
    assert bm.can_allocate(1, BS)
    assert not bm.can_allocate(1, BS + 1)
    assert bm.can_allocate(0, 3 * BS)  # existing sequence needs only one more
    assert not bm.can_allocate(0, 3 * BS + 1)


# --- allocate --------------------------------------------------------------------------------


def test_allocate_new_sequence_creates_table() -> None:
    bm = BlockManager(num_blocks=8, block_size=BS)
    bm.allocate(seq_id=7, num_tokens=2 * BS + 1)
    table = bm.block_table(7)
    assert len(table) == 3
    assert bm.num_free_blocks == 5
    assert bm.num_used_blocks == 3
    check_invariants(bm, [7])


def test_decode_growth_adds_a_block_exactly_when_crossing_a_boundary() -> None:
    bm = BlockManager(num_blocks=8, block_size=BS)
    prev: tuple[int, ...] = ()
    for n in range(1, 3 * BS + 2):
        bm.allocate(0, n)
        table = bm.block_table(0)
        assert len(table) == math.ceil(n / BS)
        assert table[: len(prev)] == prev, "existing blocks must keep their positions"
        prev = table
        check_invariants(bm, [0])


def test_chunked_growth_appends_in_order() -> None:
    bm = BlockManager(num_blocks=8, block_size=BS)
    bm.allocate(0, BS + 2)  # first chunk: 2 blocks
    first = bm.block_table(0)
    bm.allocate(0, 4 * BS)  # second chunk: total 4 blocks
    second = bm.block_table(0)
    assert second[:2] == first
    assert len(second) == 4
    check_invariants(bm, [0])


def test_allocate_covering_fewer_tokens_is_a_no_op() -> None:
    bm = BlockManager(num_blocks=8, block_size=BS)
    bm.allocate(0, 3 * BS)
    table = bm.block_table(0)
    bm.allocate(0, 1)
    bm.allocate(0, 3 * BS)
    assert bm.block_table(0) == table
    assert bm.num_free_blocks == 5


def test_allocate_exactly_all_blocks() -> None:
    bm = BlockManager(num_blocks=3, block_size=BS)
    bm.allocate(0, 3 * BS)
    assert bm.num_free_blocks == 0
    assert not bm.can_allocate(1, 1)
    assert not bm.can_allocate(0, 3 * BS + 1)
    bm.allocate(0, 3 * BS)  # still fits, no new block needed
    check_invariants(bm, [0])


def test_failed_allocate_of_new_sequence_changes_nothing() -> None:
    bm = BlockManager(num_blocks=3, block_size=BS)
    bm.allocate(0, BS)
    with pytest.raises(OutOfBlocksError):
        bm.allocate(1, 3 * BS)  # needs 3, only 2 free
    assert not bm.has_seq(1), "a failed allocation must not leave an empty table"
    assert bm.num_free_blocks == 2
    check_invariants(bm, [0])


def test_failed_allocate_of_existing_sequence_changes_nothing() -> None:
    bm = BlockManager(num_blocks=3, block_size=BS)
    bm.allocate(0, BS)
    table = bm.block_table(0)
    with pytest.raises(OutOfBlocksError):
        bm.allocate(0, 4 * BS)  # needs 3 more, only 2 free
    assert bm.block_table(0) == table, "no partial growth"
    assert bm.num_free_blocks == 2
    check_invariants(bm, [0])


def test_sequences_never_share_blocks() -> None:
    bm = BlockManager(num_blocks=10, block_size=BS)
    bm.allocate(0, 2 * BS)
    bm.allocate(1, 3 * BS)
    bm.allocate(0, 3 * BS)  # interleaved growth
    bm.allocate(2, 1)
    assert set(bm.block_table(0)).isdisjoint(bm.block_table(1))
    assert set(bm.block_table(0)).isdisjoint(bm.block_table(2))
    assert set(bm.block_table(1)).isdisjoint(bm.block_table(2))
    check_invariants(bm, [0, 1, 2])


def test_block_table_is_an_immutable_snapshot() -> None:
    bm = BlockManager(num_blocks=8, block_size=BS)
    bm.allocate(0, BS)
    snap = bm.block_table(0)
    assert isinstance(snap, tuple)
    bm.allocate(0, 2 * BS)
    assert len(snap) == 1, "an earlier snapshot must not change when the table grows"
    assert len(bm.block_table(0)) == 2


def test_block_table_of_unknown_sequence_raises() -> None:
    bm = BlockManager(num_blocks=8, block_size=BS)
    with pytest.raises(KeyError):
        bm.block_table(0)


# --- free ------------------------------------------------------------------------------------


def test_free_returns_all_blocks_and_forgets_table() -> None:
    bm = BlockManager(num_blocks=8, block_size=BS)
    bm.allocate(0, 2 * BS + 1)
    bm.allocate(1, BS)
    blocks = bm.block_table(0)
    bm.free(0)
    assert not bm.has_seq(0)
    assert bm.num_free_blocks == 7
    assert all(bm.ref_count(b) == 0 for b in blocks)
    with pytest.raises(KeyError):
        bm.block_table(0)
    check_invariants(bm, [1])


def test_free_unknown_sequence_raises() -> None:
    bm = BlockManager(num_blocks=8, block_size=BS)
    with pytest.raises(KeyError):
        bm.free(0)


def test_double_free_raises_and_changes_nothing() -> None:
    bm = BlockManager(num_blocks=8, block_size=BS)
    bm.allocate(0, BS)
    bm.allocate(1, BS)
    bm.free(0)
    with pytest.raises(KeyError):
        bm.free(0)
    assert bm.num_free_blocks == 7
    check_invariants(bm, [1])


def test_freed_blocks_are_reused() -> None:
    bm = BlockManager(num_blocks=4, block_size=BS)
    bm.allocate(0, 2 * BS)
    bm.allocate(1, 2 * BS)
    freed = set(bm.block_table(0))
    bm.free(0)
    bm.allocate(2, 2 * BS)  # pool was full, so it must get exactly the freed blocks
    assert set(bm.block_table(2)) == freed
    check_invariants(bm, [1, 2])


def test_same_seq_id_can_be_allocated_again_after_free() -> None:
    """Recompute preemption: free everything, later allocate the same sequence from scratch."""
    bm = BlockManager(num_blocks=4, block_size=BS)
    bm.allocate(0, 3 * BS)
    bm.free(0)
    bm.allocate(0, 3 * BS + 1)
    assert len(bm.block_table(0)) == 4
    check_invariants(bm, [0])


# --- ref_count -------------------------------------------------------------------------------


def test_ref_counts_are_one_when_allocated_and_zero_when_free() -> None:
    bm = BlockManager(num_blocks=4, block_size=BS)
    assert all(bm.ref_count(b) == 0 for b in range(4))
    bm.allocate(0, 2 * BS)
    for b in bm.block_table(0):
        assert bm.ref_count(b) == 1


@pytest.mark.parametrize("bad", [-1, 4, 100])
def test_ref_count_out_of_range_raises(bad: int) -> None:
    bm = BlockManager(num_blocks=4, block_size=BS)
    with pytest.raises(IndexError):
        bm.ref_count(bad)


# --- determinism and randomized invariants ---------------------------------------------------


def _replay(ops: list[tuple[str, int, int]]) -> dict[int, tuple[int, ...]]:
    bm = BlockManager(num_blocks=16, block_size=BS)
    for op, sid, n in ops:
        if op == "alloc":
            bm.allocate(sid, n)
        else:
            bm.free(sid)
    return {sid: bm.block_table(sid) for sid in range(10) if bm.has_seq(sid)}


def test_allocation_is_deterministic() -> None:
    ops = [
        ("alloc", 0, 5),
        ("alloc", 1, 9),
        ("free", 0, 0),
        ("alloc", 2, 13),
        ("alloc", 1, 12),
        ("alloc", 3, 1),
    ]
    assert _replay(ops) == _replay(ops)


@pytest.mark.parametrize("seed", range(20))
def test_random_operations_keep_invariants(seed: int) -> None:
    """Random allocate/grow/free traffic checked against a simple model of expected sizes."""
    rng = random.Random(seed)
    num_blocks = rng.randint(4, 24)
    bm = BlockManager(num_blocks=num_blocks, block_size=BS)
    tokens: dict[int, int] = {}  # model: seq_id -> tokens covered
    next_id = 0

    for _ in range(300):
        r = rng.random()
        if tokens and r < 0.25:
            sid = rng.choice(list(tokens))
            bm.free(sid)
            del tokens[sid]
        elif tokens and r < 0.75:
            sid = rng.choice(list(tokens))
            target = tokens[sid] + rng.choice([1, 1, 1, rng.randint(1, 3 * BS)])
            expected_new = math.ceil(target / BS) - math.ceil(tokens[sid] / BS)
            assert bm.blocks_needed(sid, target) == expected_new
            fits = expected_new <= bm.num_free_blocks
            assert bm.can_allocate(sid, target) == fits
            if fits:
                bm.allocate(sid, target)
                tokens[sid] = target
            else:
                with pytest.raises(OutOfBlocksError):
                    bm.allocate(sid, target)
        else:
            sid, next_id = next_id, next_id + 1
            target = rng.randint(1, 4 * BS)
            fits = math.ceil(target / BS) <= bm.num_free_blocks
            assert bm.can_allocate(sid, target) == fits
            if fits:
                bm.allocate(sid, target)
                tokens[sid] = target
            else:
                with pytest.raises(OutOfBlocksError):
                    bm.allocate(sid, target)
                assert not bm.has_seq(sid)

        for sid, n in tokens.items():
            assert len(bm.block_table(sid)) == math.ceil(n / BS)
        assert bm.num_used_blocks == sum(math.ceil(n / BS) for n in tokens.values())
        check_invariants(bm, list(tokens))

    for sid in list(tokens):
        bm.free(sid)
    assert bm.num_free_blocks == num_blocks
    check_invariants(bm, [])


def test_module_does_not_import_torch() -> None:
    import subprocess
    import sys

    code = "import sys, stallfree.block_manager; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)
