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

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass

from stallfree.block_manager import BlockManager
from stallfree.config import SchedulerConfig
from stallfree.sequence import FinishReason, SamplingParams, Sequence, SequenceStatus


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

    State:
        ``_waiting``: FIFO of sequences holding no blocks (new, or preempted at the front).
        ``_running``: sequences holding blocks, in admission order (oldest first). The end of the
            list is the newest, which is where preemption victims come from.
        ``_seqs``: every waiting or running sequence by id.
    """

    def __init__(self, config: SchedulerConfig, block_manager: BlockManager) -> None:
        if config.enable_mixed_batching:
            raise NotImplementedError("mixed batching is not implemented yet")
        self._config = config
        self._bm = block_manager
        self._watermark_blocks = int(config.watermark * block_manager.num_blocks)
        self._waiting: deque[Sequence] = deque()
        self._running: list[Sequence] = []
        self._seqs: dict[int, Sequence] = {}

    @property
    def num_waiting(self) -> int:
        """Sequences in the waiting queue (new or preempted)."""
        return len(self._waiting)

    @property
    def num_running(self) -> int:
        """Sequences holding KV cache blocks."""
        return len(self._running)

    @property
    def waiting_seq_ids(self) -> tuple[int, ...]:
        """Ids in the waiting queue, front first."""
        return tuple(seq.seq_id for seq in self._waiting)

    @property
    def running_seq_ids(self) -> tuple[int, ...]:
        """Ids in the running queue, in admission order (oldest first)."""
        return tuple(seq.seq_id for seq in self._running)

    @property
    def watermark_blocks(self) -> int:
        """Blocks that must stay free after admitting a prefill."""
        return self._watermark_blocks

    def has_unfinished(self) -> bool:
        """Whether any sequence is waiting or running."""
        return bool(self._seqs)

    def get(self, seq_id: int) -> Sequence | None:
        """The waiting or running sequence with this id, or None."""
        return self._seqs.get(seq_id)

    def validate(self, num_prompt_tokens: int, max_tokens: int) -> None:
        """Raise ``ValueError`` if a request of this size could never finish running alone.

        Pure: reads only constants (limits, pool size, watermark), so it is safe to call from
        another thread while a step runs, e.g. to reject a request before streaming begins.
        """
        max_len = num_prompt_tokens + max_tokens
        if max_len > self._config.max_model_len:
            raise ValueError(
                f"prompt ({num_prompt_tokens}) + max_tokens ({max_tokens}) exceeds "
                f"max_model_len ({self._config.max_model_len})"
            )
        # Conservative by one token: the last sampled token never needs a KV slot.
        usable = self._bm.num_blocks - self._watermark_blocks
        block_size = self._bm.block_size
        if -(-max_len // block_size) > usable:
            raise ValueError(
                f"request needs up to {max_len} tokens of KV cache, more than the "
                f"{usable * block_size} available to one request"
            )

    def add(self, seq: Sequence) -> None:
        """Append a new request to the end of the waiting queue.

        Raises:
            ValueError: if ``seq`` is not a fresh request (status WAITING, no computed tokens,
                no output tokens); if its id is already waiting or running; if
                ``num_prompt_tokens + max_tokens > max_model_len``; or if that many tokens need
                more blocks than ``num_blocks - watermark_blocks`` (it could never run alone).
        """
        if (
            seq.status is not SequenceStatus.WAITING
            or seq.num_computed_tokens != 0
            or seq.num_output_tokens != 0
        ):
            raise ValueError(f"{seq!r} is not a fresh request")
        if seq.seq_id in self._seqs:
            raise ValueError(f"seq_id {seq.seq_id} is already waiting or running")
        self.validate(seq.num_prompt_tokens, seq.sampling_params.max_tokens)
        self._waiting.append(seq)
        self._seqs[seq.seq_id] = seq

    def abort(self, seq_id: int) -> Sequence | None:
        """Cancel a waiting or running sequence.

        Frees its blocks if it holds any, removes it from its queue, and marks it FINISHED with
        ``FinishReason.ABORT``. Returns the sequence, or None if the id is not waiting or running
        (already finished or unknown); that is not an error, because a client may disconnect
        after its request finished.
        """
        seq = self._seqs.pop(seq_id, None)
        if seq is None:
            return None
        if seq.status is SequenceStatus.RUNNING:
            self._running.remove(seq)
            self._bm.free(seq_id)
        else:
            self._waiting.remove(seq)
        seq.status = SequenceStatus.FINISHED
        seq.finish_reason = FinishReason.ABORT
        return seq

    def schedule(self) -> SchedulerOutput:
        """Plan the next step (see the module docstring for the policy).

        Side effects: allocates blocks for every entry, moves admitted sequences from waiting to
        running (status RUNNING), and applies preemptions (status WAITING,
        ``num_computed_tokens = 0``, ``num_preemptions += 1``, blocks freed, front of the waiting
        queue in their original admission order). Does not change ``num_computed_tokens`` of
        scheduled sequences; ``update`` does.

        Returns an empty output when nothing can run (nothing waiting or running).
        """
        prefills = self._schedule_prefills()
        if prefills:
            return SchedulerOutput(entries=tuple(prefills))
        return self._schedule_decodes()

    def _schedule_prefills(self) -> list[ScheduledSeq]:
        """Admit waiting sequences in strict FIFO order; empty if the head does not fit."""
        cfg = self._config
        entries: list[ScheduledSeq] = []
        num_batched = 0
        while self._waiting:
            seq = self._waiting[0]
            if len(self._running) >= cfg.max_num_seqs:
                break
            num_new = seq.num_uncomputed_tokens  # whole prompt (+ generated, after preemption)
            # Over budget stops admission, except as the first entry: a long prompt runs alone.
            if entries and num_batched + num_new > cfg.max_num_batched_tokens:
                break
            needed = self._bm.blocks_needed(seq.seq_id, seq.num_tokens)
            if self._bm.num_free_blocks - needed < self._watermark_blocks:
                break
            self._bm.allocate(seq.seq_id, seq.num_tokens)
            self._waiting.popleft()
            seq.status = SequenceStatus.RUNNING
            self._running.append(seq)
            entries.append(self._make_entry(seq, num_new))
            num_batched += num_new
        return entries

    def _schedule_decodes(self) -> SchedulerOutput:
        """One token for every running sequence, oldest first, preempting newest on shortage."""
        entries: list[ScheduledSeq] = []
        preempted: list[int] = []
        i = 0
        while i < len(self._running):
            seq = self._running[i]
            assert seq.num_uncomputed_tokens == 1, f"{seq!r} is not ready to decode"
            # The uncomputed token sits at position num_tokens - 1, so the table must cover
            # num_tokens positions; that needs a new block exactly at a block boundary.
            while not self._bm.can_allocate(seq.seq_id, seq.num_tokens):
                # Newest first. Entries 0..i-1 are already in this step and are never touched.
                # When seq itself is the last one left (no newer victim), it preempts itself.
                victim = self._running.pop()
                self._preempt(victim)
                preempted.append(victim.seq_id)
                if victim is seq:
                    break
            else:
                self._bm.allocate(seq.seq_id, seq.num_tokens)
                entries.append(self._make_entry(seq, 1))
                i += 1
        return SchedulerOutput(entries=tuple(entries), preempted_seq_ids=tuple(preempted))

    def _preempt(self, seq: Sequence) -> None:
        """Recompute preemption: drop the cache, keep the tokens, wait at the front.

        Victims are taken newest first, so pushing each one to the front leaves the waiting
        queue in admission order (oldest preempted first).
        """
        self._bm.free(seq.seq_id)
        seq.status = SequenceStatus.WAITING
        seq.num_computed_tokens = 0
        seq.num_preemptions += 1
        self._waiting.appendleft(seq)

    def _make_entry(self, seq: Sequence, num_new: int) -> ScheduledSeq:
        start = seq.num_computed_tokens
        end = start + num_new
        return ScheduledSeq(
            seq_id=seq.seq_id,
            start_pos=start,
            token_ids=seq.token_ids[start:end],
            block_table=self._bm.block_table(seq.seq_id),
            sample=end == seq.num_tokens,
            sampling_params=seq.sampling_params,
        )

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
        # Validate before changing anything, so a bad call leaves the state intact.
        missing = [e.seq_id for e in output.entries if e.sample and e.seq_id not in sampled]
        if missing:
            raise KeyError(f"no sampled token for sequences {missing}")
        eos = self._config.eos_token_id
        finished: list[Sequence] = []
        for entry in output.entries:
            seq = self._seqs[entry.seq_id]
            seq.num_computed_tokens += entry.num_new_tokens
            if not entry.sample:
                continue
            token = sampled[entry.seq_id]
            seq.append_token(token)
            if eos is not None and token == eos and not seq.sampling_params.ignore_eos:
                reason = FinishReason.STOP
            elif seq.num_output_tokens >= seq.sampling_params.max_tokens:
                reason = FinishReason.LENGTH
            else:
                continue
            self._finish(seq, reason)
            finished.append(seq)
        return finished

    def _finish(self, seq: Sequence, reason: FinishReason) -> None:
        seq.status = SequenceStatus.FINISHED
        seq.finish_reason = reason
        self._running.remove(seq)
        self._bm.free(seq.seq_id)
        del self._seqs[seq.seq_id]
