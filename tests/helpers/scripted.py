"""A scripted stand-in for the scheduler, used only to test the model runner.

It replays a hand-written list of steps. Each step lists, per sequence, where computation starts,
how many tokens, the block table, and whether to sample, exactly as written in the test. It makes
no decisions: no free lists, no queues, no budgets, no preemption logic. Token ids come from a
fixed per-sequence token list (prompt followed by the reference continuation), so every decode
feeds the reference token (teacher forcing) and the runner's prediction can be checked position
by position.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from stallfree.scheduler import ScheduledSeq, SchedulerOutput
from stallfree.sequence import SamplingParams

GREEDY = SamplingParams(temperature=0.0, max_tokens=1)


@dataclass(frozen=True, slots=True)
class Planned:
    """One sequence's part of a scripted step."""

    seq_id: int
    start: int
    num_new: int
    table: tuple[int, ...]
    sample: bool = True


def scripted_steps(
    plan: Sequence[Sequence[Planned]], tokens: dict[int, list[int]]
) -> Iterator[SchedulerOutput]:
    """Yield one ``SchedulerOutput`` per planned step, in order."""
    for step in plan:
        yield SchedulerOutput(
            entries=tuple(
                ScheduledSeq(
                    seq_id=p.seq_id,
                    start_pos=p.start,
                    token_ids=tuple(tokens[p.seq_id][p.start : p.start + p.num_new]),
                    block_table=p.table,
                    sample=p.sample,
                    sampling_params=GREEDY,
                )
                for p in step
            )
        )
