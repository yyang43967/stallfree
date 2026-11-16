"""Continuous-batching scheduler with recompute preemption.

Each engine step calls ``schedule()`` to decide which sequences run and how many new tokens each
one computes, the model runner executes that plan, and ``update()`` records the sampled tokens.

Per sequence, the scheduler works only with ``num_tokens`` (prompt + generated) and
``num_computed_tokens`` (tokens whose KV is cached); see ``stallfree.sequence``. An entry in a
step computes positions ``start_pos .. start_pos + num_new_tokens - 1`` where
``start_pos == num_computed_tokens``. If the entry reaches ``num_tokens``, the runner samples one
new token from the logits at its last position.

Policy with mixed batching off (prefill-priority; the baseline):

1. Prefill step. If the head of the waiting queue can be admitted, the step contains only
   prefills: admit waiting sequences in FIFO order, each with all of its uncomputed tokens, while
   - running + admitted < ``max_num_seqs``,
   - the step's token total stays within ``max_num_batched_tokens`` (a sequence longer than the
     budget is admitted only as the first and only entry of the step),
   - after allocating its blocks, at least ``watermark_blocks`` blocks remain free.
   Admission stops at the first sequence that does not fit (strict FIFO, no skipping ahead).
2. Decode step. Otherwise every running sequence computes its one uncomputed token, in admission
   order (oldest first). If a sequence needs a new block and none is free, the most recently
   admitted running sequence that is not yet in this step is preempted (blocks freed, cache
   dropped, moved to the front of the waiting queue, generated tokens kept). If no such victim is
   left, the sequence preempts itself and is not scheduled this step.

Prefill and decode never share a step, so while prompts are admitted every running sequence
stalls. Mixed batching (``enable_mixed_batching``) is the planned fix and is not implemented yet.

Guarantees:

- A request is accepted by ``add`` only if it can finish when running alone: prompt + max_tokens
  fits ``max_model_len`` and its blocks fit in the pool minus the watermark. Together with
  newest-first preemption this guarantees progress: the oldest running sequence is never chosen
  as a victim while others remain, and it can always complete alone.
- Waiting sequences hold no blocks; running sequences hold blocks covering their computed tokens.
- Deterministic: the same calls produce the same outputs.

No torch import here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from stallfree.block_manager import BlockManager
from stallfree.config import SchedulerConfig
from stallfree.sequence import SamplingParams, Sequence


@dataclass(frozen=True, slots=True)
class ScheduledSeq:
    """One sequence's work in a step: compute ``num_new_tokens`` positions from ``start_pos``.

    Attributes:
        seq_id: the sequence.
        start_pos: first position to compute (== the sequence's ``num_computed_tokens``).
        token_ids: the token at each of those positions; ``len(token_ids) == num_new_tokens``.
        block_table: the sequence's block table after allocation for this step; covers at least
            ``start_pos + num_new_tokens`` positions.
        sample: whether the runner samples a token from this entry's last position (true when
            the entry reaches the end of the sequence's tokens; false for a non-final prefill
            chunk).
        sampling_params: how to sample.
    """

    seq_id: int
    start_pos: int
    token_ids: tuple[int, ...]
    block_table: tuple[int, ...]
    sample: bool
    sampling_params: SamplingParams

    @property
    def num_new_tokens(self) -> int:
        return len(self.token_ids)


@dataclass(frozen=True, slots=True)
class SchedulerOutput:
    """The plan for one step.

    Attributes:
        entries: sequences to run, in a fixed order the runner must preserve.
        preempted_seq_ids: sequences preempted while building this step, in the order they were
            preempted (their cache is gone).
    """

    entries: tuple[ScheduledSeq, ...] = ()
    preempted_seq_ids: tuple[int, ...] = ()

    @property
    def num_batched_tokens(self) -> int:
        return sum(e.num_new_tokens for e in self.entries)

    @property
    def is_empty(self) -> bool:
        return not self.entries


class Scheduler:
    """Owns the waiting and running queues and decides each step.

    Args:
        config: limits and policy.
        block_manager: the KV block pool. The scheduler is its only caller during serving.

    Raises:
        NotImplementedError: if ``config.enable_mixed_batching`` is true (not implemented yet).

    ``watermark_blocks = int(config.watermark * block_manager.num_blocks)``.
    """

    def __init__(self, config: SchedulerConfig, block_manager: BlockManager) -> None:
        raise NotImplementedError

    @property
    def num_waiting(self) -> int:
        """Sequences in the waiting queue (new or preempted)."""
        raise NotImplementedError

    @property
    def num_running(self) -> int:
        """Sequences holding KV cache blocks."""
        raise NotImplementedError

    @property
    def waiting_seq_ids(self) -> tuple[int, ...]:
        """Ids in the waiting queue, front first."""
        raise NotImplementedError

    @property
    def running_seq_ids(self) -> tuple[int, ...]:
        """Ids in the running queue, in admission order (oldest first)."""
        raise NotImplementedError

    @property
    def watermark_blocks(self) -> int:
        """Blocks that must stay free after admitting a prefill."""
        raise NotImplementedError

    def has_unfinished(self) -> bool:
        """Whether any sequence is waiting or running."""
        raise NotImplementedError

    def get(self, seq_id: int) -> Sequence | None:
        """The waiting or running sequence with this id, or None."""
        raise NotImplementedError

    def add(self, seq: Sequence) -> None:
        """Append a new request to the end of the waiting queue.

        Raises:
            ValueError: if ``seq`` is not a fresh request (status WAITING, no computed tokens,
                no output tokens); if its id is already waiting or running; if
                ``num_prompt_tokens + max_tokens > max_model_len``; or if that many tokens need
                more blocks than ``num_blocks - watermark_blocks`` (it could never run alone).
        """
        raise NotImplementedError

    def abort(self, seq_id: int) -> Sequence | None:
        """Cancel a waiting or running sequence.

        Frees its blocks if it holds any, removes it from its queue, and marks it FINISHED with
        ``FinishReason.ABORT``. Returns the sequence, or None if the id is not waiting or running
        (already finished or unknown); that is not an error, because a client may disconnect
        after its request finished.
        """
        raise NotImplementedError

    def schedule(self) -> SchedulerOutput:
        """Plan the next step (see the module docstring for the policy).

        Side effects: allocates blocks for every entry, moves admitted sequences from waiting to
        running (status RUNNING), and applies preemptions (status WAITING,
        ``num_computed_tokens = 0``, ``num_preemptions += 1``, blocks freed, front of the waiting
        queue in their original admission order). Does not change ``num_computed_tokens`` of
        scheduled sequences; ``update`` does.

        Returns an empty output when nothing can run (nothing waiting or running).
        """
        raise NotImplementedError

    def update(self, output: SchedulerOutput, sampled: Mapping[int, int]) -> list[Sequence]:
        """Record the result of executing ``output``.

        For each entry in order: advance the sequence's ``num_computed_tokens`` by the entry's
        ``num_new_tokens``. If ``entry.sample``, append ``sampled[seq_id]`` and check stop
        conditions in this order:

        1. the token is ``eos_token_id`` and the request does not ignore EOS -> ``STOP``;
        2. the request has ``max_tokens`` output tokens -> ``LENGTH``.

        A finished sequence is marked FINISHED with its reason, its blocks are freed, and it
        leaves the running queue.

        Returns the sequences that finished in this step, in entry order.

        Raises:
            KeyError: if an entry with ``sample`` has no token in ``sampled``.
        """
        raise NotImplementedError
