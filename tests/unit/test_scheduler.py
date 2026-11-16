"""Contract tests for the scheduler with mixed batching off (prefill-priority).

``Sim`` stands in for the model runner. It keeps its own per-sequence "KV cache" (the list of
tokens whose positions were computed) and checks that every entry continues exactly where the
cache ends. The next token is a deterministic function of the whole token history, so a wrong
``token_ids`` or ``start_pos`` (for example after a recompute preemption) changes the output and
fails the comparison with a reference generated without any scheduler.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable

import pytest

from stallfree.block_manager import BlockManager
from stallfree.config import SchedulerConfig
from stallfree.scheduler import Scheduler, SchedulerOutput
from stallfree.sequence import FinishReason, SamplingParams, Sequence, SequenceStatus

BS = 4
EOS = 0


def fake_next(tokens: list[int]) -> int:
    """Deterministic next token from the full history; never EOS."""
    return 1 + sum((i + 1) * t for i, t in enumerate(tokens)) % 997


def reference_output(prompt: tuple[int, ...], max_tokens: int) -> list[int]:
    tokens = list(prompt)
    out: list[int] = []
    for _ in range(max_tokens):
        t = fake_next(tokens)
        tokens.append(t)
        out.append(t)
    return out


def make_seq(
    seq_id: int, prompt_len: int, max_tokens: int = 4, *, ignore_eos: bool = True, base: int = 1
) -> Sequence:
    prompt = [base + 100 * seq_id + i for i in range(prompt_len)]
    return Sequence(seq_id, prompt, SamplingParams(max_tokens=max_tokens, ignore_eos=ignore_eos))


def make(
    num_blocks: int = 64,
    *,
    max_num_seqs: int = 8,
    max_num_batched_tokens: int = 64,
    max_model_len: int = 256,
    watermark: float = 0.0,
    eos_token_id: int | None = EOS,
) -> tuple[Scheduler, BlockManager]:
    bm = BlockManager(num_blocks=num_blocks, block_size=BS)
    cfg = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=max_model_len,
        watermark=watermark,
        eos_token_id=eos_token_id,
    )
    return Scheduler(cfg, bm), bm


class Sim:
    """Drives a scheduler like the engine would and checks every step."""

    def __init__(
        self,
        sched: Scheduler,
        bm: BlockManager,
        cfg_budget: int,
        cfg_max_seqs: int,
        next_token: Callable[[list[int]], int] = fake_next,
    ) -> None:
        self.sched = sched
        self.bm = bm
        self.budget = cfg_budget
        self.max_seqs = cfg_max_seqs
        self.next_token = next_token
        self.cache: dict[int, list[int]] = {}
        self.seqs: dict[int, Sequence] = {}
        self.finished: dict[int, Sequence] = {}
        self.preemptions = 0

    def add(self, seq: Sequence) -> None:
        self.sched.add(seq)
        self.seqs[seq.seq_id] = seq

    def step(self) -> tuple[SchedulerOutput, list[Sequence]]:
        out = self.sched.schedule()
        self.check_plan(out)
        for sid in out.preempted_seq_ids:
            self.cache.pop(sid, None)
            self.preemptions += 1
        sampled: dict[int, int] = {}
        for e in out.entries:
            c = self.cache.setdefault(e.seq_id, [])
            assert e.start_pos == len(c), "entry must continue exactly where the cache ends"
            c.extend(e.token_ids)
            if e.sample:
                sampled[e.seq_id] = self.next_token(c)
        finished = self.sched.update(out, sampled)
        for f in finished:
            self.cache.pop(f.seq_id, None)
            self.finished[f.seq_id] = f
        self.check_state()
        return out, finished

    def run(self, max_steps: int = 5000) -> int:
        steps = 0
        while self.sched.has_unfinished():
            self.step()
            steps += 1
            assert steps < max_steps, "no progress (livelock or deadlock)"
        return steps

    def check_plan(self, out: SchedulerOutput) -> None:
        ids = [e.seq_id for e in out.entries]
        assert len(ids) == len(set(ids)), "a sequence appears twice in one step"
        assert len(ids) <= self.max_seqs
        assert set(ids).isdisjoint(out.preempted_seq_ids)
        assert out.num_batched_tokens <= self.budget or len(ids) == 1
        # Prefill-priority: a step is all prefills (start 0) or all decodes (start > 0).
        kinds = {e.start_pos == 0 for e in out.entries}
        assert len(kinds) <= 1, "prefill and decode must not share a step with mixed batching off"
        for e in out.entries:
            seq = self.sched.get(e.seq_id)
            assert seq is not None
            assert seq.status is SequenceStatus.RUNNING
            assert e.start_pos == seq.num_computed_tokens
            end = e.start_pos + e.num_new_tokens
            assert e.token_ids == seq.token_ids[e.start_pos : end]
            assert e.sample == (end == seq.num_tokens)
            assert e.block_table == self.bm.block_table(e.seq_id)
            assert len(e.block_table) >= math.ceil(end / BS)
            assert e.sampling_params == seq.sampling_params
        for sid in out.preempted_seq_ids:
            seq = self.sched.get(sid)
            assert seq is not None
            assert seq.status is SequenceStatus.WAITING
            assert seq.num_computed_tokens == 0
            assert not self.bm.has_seq(sid)

    def check_state(self) -> None:
        running = self.sched.running_seq_ids
        waiting = self.sched.waiting_seq_ids
        assert self.sched.num_running == len(running)
        assert self.sched.num_waiting == len(waiting)
        assert set(running).isdisjoint(waiting)
        used = 0
        for sid in running:
            seq = self.sched.get(sid)
            assert seq is not None and seq.status is SequenceStatus.RUNNING
            assert self.bm.has_seq(sid)
            used += len(self.bm.block_table(sid))
        for sid in waiting:
            seq = self.sched.get(sid)
            assert seq is not None and seq.status is SequenceStatus.WAITING
            assert not self.bm.has_seq(sid), "waiting sequences hold no blocks"
        for sid in self.finished:
            assert self.sched.get(sid) is None
            assert not self.bm.has_seq(sid)
        assert used == self.bm.num_used_blocks, "blocks held by nobody (leak)"


def sim_for(
    num_blocks: int = 64,
    *,
    max_num_seqs: int = 8,
    max_num_batched_tokens: int = 64,
    watermark: float = 0.0,
    next_token: Callable[[list[int]], int] = fake_next,
) -> Sim:
    sched, bm = make(
        num_blocks,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        watermark=watermark,
    )
    return Sim(sched, bm, max_num_batched_tokens, max_num_seqs, next_token)


# --- construction and add ------------------------------------------------------------------


def test_fresh_scheduler_is_empty() -> None:
    sched, _ = make()
    assert not sched.has_unfinished()
    assert sched.num_waiting == sched.num_running == 0
    assert sched.waiting_seq_ids == sched.running_seq_ids == ()
    out = sched.schedule()
    assert out.is_empty
    assert out.preempted_seq_ids == ()


def test_mixed_batching_not_implemented_yet() -> None:
    bm = BlockManager(num_blocks=8, block_size=BS)
    with pytest.raises(NotImplementedError):
        Scheduler(SchedulerConfig(enable_mixed_batching=True), bm)


def test_watermark_blocks_rounds_down() -> None:
    bm = BlockManager(num_blocks=50, block_size=BS)
    sched = Scheduler(SchedulerConfig(watermark=0.05), bm)
    assert sched.watermark_blocks == 2


def test_add_queues_in_fifo_order() -> None:
    sched, _ = make()
    for i in (3, 1, 2):
        sched.add(make_seq(i, 2))
    assert sched.waiting_seq_ids == (3, 1, 2)
    assert sched.has_unfinished()
    seq = sched.get(1)
    assert seq is not None and seq.seq_id == 1
    assert sched.get(99) is None


def test_add_rejects_duplicate_id() -> None:
    sched, _ = make()
    sched.add(make_seq(0, 2))
    with pytest.raises(ValueError):
        sched.add(make_seq(0, 3))


def test_add_rejects_non_fresh_sequence() -> None:
    sched, _ = make()
    seq = make_seq(0, 2)
    seq.num_computed_tokens = 1
    with pytest.raises(ValueError):
        sched.add(seq)
    seq2 = make_seq(1, 2)
    seq2.append_token(5)
    with pytest.raises(ValueError):
        sched.add(seq2)
    seq3 = make_seq(2, 2)
    seq3.status = SequenceStatus.RUNNING
    with pytest.raises(ValueError):
        sched.add(seq3)
    assert sched.num_waiting == 0


def test_add_rejects_request_longer_than_max_model_len() -> None:
    sched, _ = make(max_model_len=10)
    sched.add(make_seq(0, 6, max_tokens=4))  # exactly 10: accepted
    with pytest.raises(ValueError):
        sched.add(make_seq(1, 6, max_tokens=5))


def test_add_rejects_request_that_could_never_run_alone() -> None:
    # 10 blocks, watermark 0.2 -> 2 reserved; 8 blocks = 32 tokens is the most one request may use.
    sched, _ = make(num_blocks=10, watermark=0.2)
    sched.add(make_seq(0, 28, max_tokens=4))  # 32 tokens: 8 blocks
    with pytest.raises(ValueError):
        sched.add(make_seq(1, 29, max_tokens=4))  # 33 tokens: 9 blocks


# --- one sequence, step by step ------------------------------------------------------------


def test_entry_fields_through_prefill_and_decode() -> None:
    sched, bm = make()
    seq = Sequence(0, [5, 6, 7], SamplingParams(max_tokens=3, ignore_eos=True))
    sched.add(seq)

    out = sched.schedule()
    (e,) = out.entries
    assert (e.seq_id, e.start_pos, e.token_ids, e.sample) == (0, 0, (5, 6, 7), True)
    assert len(e.block_table) == 1
    assert sched.running_seq_ids == (0,)
    assert seq.status is SequenceStatus.RUNNING
    assert seq.num_computed_tokens == 0, "schedule() must not advance num_computed_tokens"
    assert sched.update(out, {0: 11}) == []
    assert seq.output_token_ids == [11]
    assert seq.num_computed_tokens == 3

    out = sched.schedule()
    (e,) = out.entries
    assert (e.start_pos, e.token_ids, e.sample) == (3, (11,), True)
    assert len(e.block_table) == 1  # position 3 still fits block 0
    sched.update(out, {0: 12})

    out = sched.schedule()
    (e,) = out.entries
    assert (e.start_pos, e.token_ids) == (4, (12,))
    assert len(e.block_table) == 2  # position 4 starts block 1
    finished = sched.update(out, {0: 13})

    assert finished == [seq]
    assert seq.output_token_ids == [11, 12, 13]
    assert seq.status is SequenceStatus.FINISHED
    assert seq.finish_reason is FinishReason.LENGTH
    assert not bm.has_seq(0)
    assert bm.num_free_blocks == bm.num_blocks
    assert not sched.has_unfinished()
    assert sched.get(0) is None


def test_max_tokens_one_finishes_after_prefill() -> None:
    sim = sim_for()
    sim.add(make_seq(0, 5, max_tokens=1))
    _, finished = sim.step()
    assert [f.seq_id for f in finished] == [0]
    assert finished[0].num_output_tokens == 1
    assert finished[0].finish_reason is FinishReason.LENGTH


def test_eos_stops_unless_ignored() -> None:
    sim = sim_for(next_token=lambda _tokens: EOS)
    sim.add(make_seq(0, 3, max_tokens=5, ignore_eos=False))
    sim.add(make_seq(1, 3, max_tokens=5, ignore_eos=True))
    sim.run()
    a, b = sim.finished[0], sim.finished[1]
    assert a.output_token_ids == [EOS]
    assert a.finish_reason is FinishReason.STOP
    assert b.output_token_ids == [EOS] * 5
    assert b.finish_reason is FinishReason.LENGTH


def test_eos_disabled_when_eos_token_id_is_none() -> None:
    sched, bm = make(eos_token_id=None)
    sim = Sim(sched, bm, 64, 8, next_token=lambda _tokens: EOS)
    sim.add(make_seq(0, 3, max_tokens=3, ignore_eos=False))
    sim.run()
    assert sim.finished[0].finish_reason is FinishReason.LENGTH


def test_update_without_sampled_token_raises() -> None:
    sched, _ = make()
    sched.add(make_seq(0, 3))
    out = sched.schedule()
    with pytest.raises(KeyError):
        sched.update(out, {})


# --- prefill-priority policy ---------------------------------------------------------------


def test_new_prefill_stalls_running_decodes() -> None:
    sim = sim_for()
    sim.add(make_seq(0, 3, max_tokens=8))
    sim.step()  # prefill A
    out, _ = sim.step()  # decode A
    assert [e.start_pos > 0 for e in out.entries] == [True]
    sim.add(make_seq(1, 2, max_tokens=8))
    out, _ = sim.step()
    assert [(e.seq_id, e.start_pos) for e in out.entries] == [(1, 0)], "decode of A must wait"
    out, _ = sim.step()
    assert [e.seq_id for e in out.entries] == [0, 1], "decodes in admission order"
    assert all(e.num_new_tokens == 1 for e in out.entries)


def test_prefills_batched_in_fifo_order_within_budget() -> None:
    sim = sim_for(max_num_batched_tokens=8)
    sim.add(make_seq(0, 3))
    sim.add(make_seq(1, 4))
    sim.add(make_seq(2, 3))  # 3 + 4 + 3 = 10 > 8
    out, _ = sim.step()
    assert [e.seq_id for e in out.entries] == [0, 1]
    assert sim.sched.waiting_seq_ids == (2,)
    out, _ = sim.step()
    assert [(e.seq_id, e.start_pos) for e in out.entries] == [(2, 0)]


def test_prompt_longer_than_budget_runs_alone() -> None:
    sim = sim_for(max_num_batched_tokens=8)
    sim.add(make_seq(0, 2))
    sim.add(make_seq(1, 10))
    sim.add(make_seq(2, 2))
    out, _ = sim.step()
    assert [e.seq_id for e in out.entries] == [0], "strict FIFO: no skipping ahead to 2"
    out, _ = sim.step()
    assert [(e.seq_id, e.num_new_tokens) for e in out.entries] == [(1, 10)]
    out, _ = sim.step()
    assert [e.seq_id for e in out.entries] == [2]


def test_admission_limited_by_max_num_seqs() -> None:
    sim = sim_for(max_num_seqs=2, max_num_batched_tokens=64)
    for i in range(3):
        sim.add(make_seq(i, 2, max_tokens=4))
    out, _ = sim.step()
    assert [e.seq_id for e in out.entries] == [0, 1]
    out, _ = sim.step()
    assert [e.seq_id for e in out.entries] == [0, 1], "running is full, so decode"
    assert all(e.start_pos > 0 for e in out.entries)
    assert sim.sched.waiting_seq_ids == (2,)


def test_admission_limited_by_free_blocks() -> None:
    sim = sim_for(num_blocks=4)
    sim.add(make_seq(0, 8, max_tokens=4))  # 2 blocks now
    sim.add(make_seq(1, 9, max_tokens=1))  # 3 blocks now; only 2 left
    out, _ = sim.step()
    assert [e.seq_id for e in out.entries] == [0]
    out, _ = sim.step()
    assert [(e.seq_id, e.start_pos) for e in out.entries] == [(0, 8)], "blocked head -> decode"


def test_watermark_applies_to_admission_not_to_decode() -> None:
    sim = sim_for(num_blocks=10, watermark=0.2)  # 2 blocks reserved
    assert sim.sched.watermark_blocks == 2
    sim.add(make_seq(0, 28, max_tokens=4))  # 7 blocks -> 3 free
    sim.add(make_seq(1, 4, max_tokens=1))  # 1 block -> 2 free (== watermark, allowed)
    sim.add(make_seq(2, 4, max_tokens=3))  # would leave 1 < 2: must wait
    out, _ = sim.step()
    assert [e.seq_id for e in out.entries] == [0, 1]
    assert sim.bm.num_free_blocks == 3
    # Seq 1 finished (max_tokens 1) -> 3 free. Seq 2 now fits: 3 - 1 = 2.
    out, _ = sim.step()
    assert [e.seq_id for e in out.entries] == [2]
    assert sim.bm.num_free_blocks == 2
    # Decodes: seq 0 at position 28 needs an 8th block, seq 2 at position 4 needs a 2nd one.
    # Decode growth may use the reserve.
    out, _ = sim.step()
    assert [(e.seq_id, len(e.block_table)) for e in out.entries] == [(0, 8), (2, 2)]
    assert sim.bm.num_free_blocks == 0


# --- preemption ----------------------------------------------------------------------------


def test_preemption_picks_newest_and_recomputes_correctly() -> None:
    sim = sim_for(num_blocks=4)
    sim.add(make_seq(0, 4, max_tokens=8))  # 12 tokens max -> 3 blocks
    sim.add(make_seq(1, 4, max_tokens=8))
    preempt_step: SchedulerOutput | None = None
    while sim.sched.has_unfinished():
        out, _ = sim.step()
        if out.preempted_seq_ids and preempt_step is None:
            preempt_step = out
            b = sim.sched.get(1)
            assert b is not None
            assert b.num_preemptions == 1
            assert b.num_output_tokens > 0, "generated tokens are kept"
            assert sim.sched.waiting_seq_ids[0] == 1, "preempted goes to the front"
    assert preempt_step is not None, "this setup must preempt"
    assert preempt_step.preempted_seq_ids == (1,), "victim is the most recently admitted"
    assert [e.seq_id for e in preempt_step.entries] == [0]
    for sid in (0, 1):
        seq = sim.finished[sid]
        assert seq.output_token_ids == reference_output(seq.prompt_token_ids, 8)


def test_recompute_prefill_covers_prompt_and_generated_tokens() -> None:
    sim = sim_for(num_blocks=4)
    sim.add(make_seq(0, 4, max_tokens=8))
    sim.add(make_seq(1, 4, max_tokens=8))
    seen_recompute = False
    while sim.sched.has_unfinished():
        before = {sid: sim.sched.get(sid) for sid in (0, 1)}
        prev = {sid: (s.num_preemptions, s.token_ids) for sid, s in before.items() if s}
        out, _ = sim.step()
        for e in out.entries:
            if e.seq_id in prev and prev[e.seq_id][0] > 0 and e.start_pos == 0:
                assert e.token_ids == prev[e.seq_id][1]
                assert len(e.token_ids) > 4
                seen_recompute = True
    assert seen_recompute


def test_sequence_preempts_itself_when_no_newer_victim() -> None:
    sim = sim_for(num_blocks=3)
    sim.add(make_seq(0, 4, max_tokens=4))  # 8 tokens max -> 2 blocks
    sim.add(make_seq(1, 4, max_tokens=4))
    out, _ = sim.step()  # prefill both: 1 block each, 1 free
    assert [e.seq_id for e in out.entries] == [0, 1]
    out, _ = sim.step()  # A takes the last block for position 4; B has no newer victim
    assert [e.seq_id for e in out.entries] == [0]
    assert out.preempted_seq_ids == (1,)
    sim.run()
    for sid in (0, 1):
        seq = sim.finished[sid]
        assert seq.output_token_ids == reference_output(seq.prompt_token_ids, 4)


def test_multiple_preemptions_keep_admission_order_in_waiting_queue() -> None:
    sim = sim_for(num_blocks=5)
    for i in range(4):
        sim.add(make_seq(i, 4, max_tokens=4))  # each needs 1 block now, 2 at most
    out, _ = sim.step()
    assert [e.seq_id for e in out.entries] == [0, 1, 2, 3]  # 1 block free
    out, _ = sim.step()
    # 0 takes the free block; 1 preempts 3 (newest); 2 has no newer victim and preempts itself.
    assert [e.seq_id for e in out.entries] == [0, 1]
    assert out.preempted_seq_ids == (3, 2)
    assert sim.sched.waiting_seq_ids == (2, 3)
    sim.run()
    for sid in range(4):
        seq = sim.finished[sid]
        assert seq.output_token_ids == reference_output(seq.prompt_token_ids, 4)


# --- abort ---------------------------------------------------------------------------------


def test_abort_waiting_sequence() -> None:
    sched, bm = make()
    sched.add(make_seq(0, 3))
    sched.add(make_seq(1, 3))
    seq = sched.abort(0)
    assert seq is not None
    assert seq.status is SequenceStatus.FINISHED
    assert seq.finish_reason is FinishReason.ABORT
    assert sched.waiting_seq_ids == (1,)
    assert sched.get(0) is None
    assert bm.num_free_blocks == bm.num_blocks


def test_abort_running_sequence_frees_blocks() -> None:
    sched, bm = make()
    sched.add(make_seq(0, 9))
    sched.update(sched.schedule(), {0: 3})
    assert bm.num_used_blocks == 3
    seq = sched.abort(0)
    assert seq is not None and seq.finish_reason is FinishReason.ABORT
    assert sched.running_seq_ids == ()
    assert bm.num_free_blocks == bm.num_blocks
    assert not sched.has_unfinished()


def test_abort_unknown_or_finished_returns_none() -> None:
    sim = sim_for()
    sim.add(make_seq(0, 2, max_tokens=1))
    sim.run()
    assert sim.sched.abort(0) is None
    assert sim.sched.abort(42) is None


def test_aborted_sequence_is_not_scheduled_again() -> None:
    sim = sim_for()
    sim.add(make_seq(0, 3, max_tokens=8))
    sim.add(make_seq(1, 3, max_tokens=8))
    sim.step()
    sim.sched.abort(0)
    out, _ = sim.step()
    assert [e.seq_id for e in out.entries] == [1]


# --- randomized end-to-end and determinism -------------------------------------------------


def _random_run(seed: int) -> tuple[Sim, list[tuple[tuple[int, int, int], ...]]]:
    rng = random.Random(seed)
    num_blocks = rng.randint(4, 12)
    max_num_seqs = rng.randint(1, 4)
    budget = rng.randint(max_num_seqs, 16)
    watermark = rng.choice([0.0, 0.1, 0.25])
    sim = sim_for(
        num_blocks,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=budget,
        watermark=watermark,
    )
    limit_tokens = (num_blocks - sim.sched.watermark_blocks) * BS
    trace: list[tuple[tuple[int, int, int], ...]] = []
    next_id = 0
    for step in range(5000):
        if next_id < 25 and rng.random() < 0.4:
            prompt_len = rng.randint(1, 12)
            max_tokens = rng.randint(1, 10)
            if prompt_len + max_tokens <= limit_tokens:
                sim.add(make_seq(next_id, prompt_len, max_tokens=max_tokens))
            next_id += 1
        if next_id >= 25 and not sim.sched.has_unfinished():
            break
        out, _ = sim.step()
        trace.append(tuple((e.seq_id, e.start_pos, e.num_new_tokens) for e in out.entries))
        assert step < 4999, "no progress"
    return sim, trace


@pytest.mark.parametrize("seed", range(15))
def test_random_workload_matches_reference(seed: int) -> None:
    sim, _ = _random_run(seed)
    assert not sim.sched.has_unfinished()
    for sid, seq in sim.seqs.items():
        assert sid in sim.finished
        assert seq.finish_reason is FinishReason.LENGTH
        assert seq.output_token_ids == reference_output(
            seq.prompt_token_ids, seq.sampling_params.max_tokens
        )
    assert sim.bm.num_free_blocks == sim.bm.num_blocks


def test_random_workloads_exercise_preemption() -> None:
    assert sum(_random_run(seed)[0].preemptions for seed in range(15)) > 0


def test_scheduling_is_deterministic() -> None:
    assert _random_run(3)[1] == _random_run(3)[1]


def test_module_does_not_import_torch() -> None:
    import subprocess
    import sys

    code = "import sys, stallfree.scheduler; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)
