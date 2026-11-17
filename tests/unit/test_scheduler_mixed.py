"""Contract tests for the scheduler with mixed batching on (stall-free).

``MixedSim`` stands in for the model runner like ``Sim`` in test_scheduler.py: it keeps its own
record of computed positions per sequence, checks that every entry continues exactly where that
record ends, and produces the next token from the whole history, so a wrong chunk boundary or a
wrong recompute changes the output. Every step it also records each sequence's state *before* the
step, which tells decodes, chunk continuations, and admissions apart, and checks the policy:

- decodes first, then chunk continuations, then admissions;
- the token budget is never exceeded, not even by a long prompt;
- every running sequence that is ready to decode is scheduled, unless it was preempted;
- a sequence in the middle of its prefill is skipped only when the budget is used up.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass

import pytest

from stallfree.block_manager import BlockManager
from stallfree.config import SchedulerConfig
from stallfree.scheduler import Scheduler, SchedulerOutput
from stallfree.sequence import FinishReason, SamplingParams, Sequence, SequenceStatus

BS = 4
EOS = 0


def fake_next(tokens: list[int]) -> int:
    return 1 + sum((i + 1) * t for i, t in enumerate(tokens)) % 997


def reference_output(prompt: tuple[int, ...], max_tokens: int) -> list[int]:
    tokens, out = list(prompt), []
    for _ in range(max_tokens):
        t = fake_next(tokens)
        tokens.append(t)
        out.append(t)
    return out


def make_seq(seq_id: int, prompt_len: int, max_tokens: int = 4) -> Sequence:
    prompt = [1 + 100 * seq_id + i for i in range(prompt_len)]
    return Sequence(seq_id, prompt, SamplingParams(max_tokens=max_tokens, ignore_eos=True))


@dataclass
class Before:
    status: SequenceStatus
    uncomputed: int


class MixedSim:
    def __init__(
        self,
        num_blocks: int = 64,
        *,
        max_num_seqs: int = 8,
        budget: int = 16,
        watermark: float = 0.0,
        next_token: Callable[[list[int]], int] = fake_next,
    ) -> None:
        self.bm = BlockManager(num_blocks=num_blocks, block_size=BS)
        self.cfg = SchedulerConfig(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=budget,
            max_model_len=512,
            watermark=watermark,
            enable_mixed_batching=True,
            eos_token_id=EOS,
        )
        self.sched = Scheduler(self.cfg, self.bm)
        self.next_token = next_token
        self.cache: dict[int, list[int]] = {}
        self.seqs: dict[int, Sequence] = {}
        self.finished: dict[int, Sequence] = {}
        self.preemptions = 0
        self.partial_prefill_preemptions = 0
        self.steps = 0

    def add(self, seq: Sequence) -> None:
        self.sched.add(seq)
        self.seqs[seq.seq_id] = seq

    def before(self) -> dict[int, Before]:
        return {
            sid: Before(s.status, s.num_uncomputed_tokens)
            for sid, s in self.seqs.items()
            if not s.is_finished
        }

    def step(self) -> SchedulerOutput:
        pre = self.before()
        computed_before = {sid: s.num_computed_tokens for sid, s in self.seqs.items()}
        out = self.sched.schedule()
        self.check_plan(out, pre)
        for sid in out.preempted_seq_ids:
            self.preemptions += 1
            seq = self.seqs[sid]
            if 0 < computed_before[sid] < seq.num_tokens - 1:
                self.partial_prefill_preemptions += 1  # preempted in the middle of a prefill
            self.cache.pop(sid, None)
        sampled: dict[int, int] = {}
        for e in out.entries:
            c = self.cache.setdefault(e.seq_id, [])
            assert e.start_pos == len(c), "entry must continue exactly where the cache ends"
            c.extend(e.token_ids)
            if e.sample:
                sampled[e.seq_id] = self.next_token(c)
        for f in self.sched.update(out, sampled):
            self.cache.pop(f.seq_id, None)
            self.finished[f.seq_id] = f
        self.check_state()
        self.steps += 1
        return out

    def run(self, max_steps: int = 5000) -> None:
        while self.sched.has_unfinished():
            self.step()
            assert self.steps < max_steps, "no progress"

    def check_plan(self, out: SchedulerOutput, pre: dict[int, Before]) -> None:
        ids = [e.seq_id for e in out.entries]
        assert len(ids) == len(set(ids))
        assert set(ids).isdisjoint(out.preempted_seq_ids)
        assert out.num_batched_tokens <= self.cfg.max_num_batched_tokens, "budget is strict"
        assert self.sched.num_running <= self.cfg.max_num_seqs

        def kind(sid: int) -> int:
            b = pre[sid]
            if b.status is SequenceStatus.WAITING:
                return 2  # admission
            return 0 if b.uncomputed == 1 else 1  # decode / chunk continuation

        kinds = [kind(sid) for sid in ids]
        assert kinds == sorted(kinds), "order must be decodes, continuations, admissions"
        for sid, b in pre.items():
            if b.status is SequenceStatus.RUNNING and b.uncomputed == 1:
                assert sid in ids or sid in out.preempted_seq_ids, f"decode of {sid} skipped"
            if b.status is SequenceStatus.RUNNING and b.uncomputed > 1 and sid not in ids:
                assert (
                    sid in out.preempted_seq_ids
                    or out.num_batched_tokens == self.cfg.max_num_batched_tokens
                ), f"prefill of {sid} skipped with budget left"
        for e in out.entries:
            seq = self.sched.get(e.seq_id)
            assert seq is not None and seq.status is SequenceStatus.RUNNING
            assert e.start_pos == seq.num_computed_tokens
            end = e.start_pos + e.num_new_tokens
            assert e.token_ids == seq.token_ids[e.start_pos : end]
            assert e.sample == (end == seq.num_tokens)
            assert e.block_table == self.bm.block_table(e.seq_id)
            assert len(e.block_table) >= math.ceil(end / BS)

    def check_state(self) -> None:
        used = 0
        for sid in self.sched.running_seq_ids:
            used += len(self.bm.block_table(sid))
        for sid in self.sched.waiting_seq_ids:
            assert not self.bm.has_seq(sid)
        for sid in self.finished:
            assert not self.bm.has_seq(sid)
        assert used == self.bm.num_used_blocks, "block leak"


def entries(out: SchedulerOutput) -> list[tuple[int, int, int]]:
    return [(e.seq_id, e.start_pos, e.num_new_tokens) for e in out.entries]


# --- the policy, step by step ----------------------------------------------------------------


def test_long_prompt_is_chunked_while_every_decode_keeps_running() -> None:
    """The SPEC test: no running sequence skips a step while a long prefill is in progress."""
    sim = MixedSim(budget=8)
    for i in range(3):
        sim.add(make_seq(i, 2, max_tokens=30))
    sim.step()  # three 2-token prefills
    sim.step()  # three decodes
    sim.add(make_seq(9, 40, max_tokens=2))
    chunks = []
    while 9 not in sim.finished or not sim.seqs[9].output_token_ids:
        out = sim.step()
        assert [sid for sid, _, n in entries(out)[:3]] == [0, 1, 2], "decodes first, every step"
        chunks += [n for sid, _, n in entries(out) if sid == 9]
        if sim.seqs[9].output_token_ids:
            break
    assert chunks == [5] * 8, "budget 8 minus 3 decodes = 5 prompt tokens per step"
    assert sim.seqs[9].output_token_ids, "the first token comes with the last chunk"


def test_decodes_then_continuation_then_admission() -> None:
    sim = MixedSim(budget=8)
    sim.add(make_seq(0, 3, max_tokens=20))
    sim.add(make_seq(1, 20, max_tokens=2))
    assert entries(sim.step()) == [(0, 0, 3), (1, 0, 5)]  # two admissions share the budget
    sim.add(make_seq(2, 4, max_tokens=2))
    # A decodes (1), B continues (7): budget used up, C waits.
    assert entries(sim.step()) == [(0, 3, 1), (1, 5, 7)]
    assert sim.sched.waiting_seq_ids == (2,)
    # A decodes, B continues with min(8 left, 7) = 7, C still waits.
    assert entries(sim.step()) == [(0, 4, 1), (1, 12, 7)]
    # A decodes; B's last token is a one-token piece and joins the decodes; C is admitted.
    assert entries(sim.step()) == [(0, 5, 1), (1, 19, 1), (2, 0, 4)]


def test_fifo_admission_fills_the_leftover_budget_with_a_partial_chunk() -> None:
    sim = MixedSim(budget=10)
    sim.add(make_seq(0, 4))
    sim.add(make_seq(1, 9))
    sim.add(make_seq(2, 3))
    assert entries(sim.step()) == [(0, 0, 4), (1, 0, 6)]  # 1 gets a partial chunk; 2 waits
    assert sim.sched.waiting_seq_ids == (2,)


def test_budget_smaller_than_every_prompt() -> None:
    sim = MixedSim(budget=4, max_num_seqs=4)
    for i, n in enumerate([6, 7, 9, 10]):
        sim.add(make_seq(i, n, max_tokens=5))
    sim.run()
    for sid, seq in sim.seqs.items():
        assert seq.output_token_ids == reference_output(seq.prompt_token_ids, 5), sid


def test_max_num_seqs_counts_partially_prefilled_sequences() -> None:
    sim = MixedSim(budget=8, max_num_seqs=2)
    for i in range(3):
        sim.add(make_seq(i, 12, max_tokens=2))
    out = sim.step()
    assert entries(out) == [(0, 0, 8)]  # the first prompt's chunk uses the whole budget
    while sim.sched.has_unfinished():
        sim.step()
        assert sim.sched.num_running <= 2
    assert set(sim.finished) == {0, 1, 2}


def test_watermark_shrinks_the_admitted_chunk() -> None:
    sim = MixedSim(num_blocks=10, budget=64, watermark=0.2)  # 2 blocks reserved
    sim.add(make_seq(0, 28, max_tokens=4))  # 7 blocks -> 3 free
    sim.add(make_seq(1, 12, max_tokens=1))  # needs 3 blocks; only 1 may be used
    out = sim.step()
    assert entries(out) == [(0, 0, 28), (1, 0, 4)]
    assert sim.bm.num_free_blocks == 2


def test_eos_and_length_still_finish_sequences() -> None:
    sim = MixedSim(budget=8, next_token=lambda _t: EOS)
    seq = Sequence(0, [5, 6, 7, 8, 9, 10, 11, 12, 13, 14], SamplingParams(max_tokens=5))
    sim.add(seq)
    sim.run()
    assert seq.output_token_ids == [EOS] and seq.finish_reason is FinishReason.STOP


# --- preemption -----------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(15))
def test_random_workloads_match_reference(seed: int) -> None:
    sim, _ = random_run(seed)
    for sid, seq in sim.seqs.items():
        assert sid in sim.finished
        assert seq.output_token_ids == reference_output(
            seq.prompt_token_ids, seq.sampling_params.max_tokens
        )
    assert sim.bm.num_free_blocks == sim.bm.num_blocks


def test_random_workloads_preempt_in_the_middle_of_a_prefill() -> None:
    sims = [random_run(seed)[0] for seed in range(15)]
    assert sum(s.preemptions for s in sims) > 0
    assert sum(s.partial_prefill_preemptions for s in sims) > 0, "a chunked prefill was preempted"


def test_mixed_scheduling_is_deterministic() -> None:
    assert random_run(4)[1] == random_run(4)[1]


def random_run(seed: int) -> tuple[MixedSim, list[list[tuple[int, int, int]]]]:
    rng = random.Random(seed)
    num_blocks = rng.randint(5, 12)
    max_num_seqs = rng.randint(1, 4)
    sim = MixedSim(
        num_blocks,
        max_num_seqs=max_num_seqs,
        budget=rng.randint(max_num_seqs, 12),
        watermark=rng.choice([0.0, 0.1, 0.25]),
    )
    limit = (num_blocks - sim.sched.watermark_blocks) * BS
    trace: list[list[tuple[int, int, int]]] = []
    next_id = 0
    for step in range(5000):
        if next_id < 25 and rng.random() < 0.4:
            prompt_len, max_tokens = rng.randint(1, 16), rng.randint(1, 10)
            if prompt_len + max_tokens <= limit:
                sim.add(make_seq(next_id, prompt_len, max_tokens=max_tokens))
            next_id += 1
        if next_id >= 25 and not sim.sched.has_unfinished():
            break
        trace.append(entries(sim.step()))
        assert step < 4999, "no progress"
    return sim, trace


def test_module_has_no_torch_dependency() -> None:
    import subprocess
    import sys

    code = "import sys, stallfree.scheduler; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)
