"""Request state shared by the scheduler, the engine, and the API layer.

A sequence's tokens are its prompt followed by the tokens generated so far. The KV cache holds
entries for the first ``num_computed_tokens`` of them. Everything the scheduler does can be stated
in these two numbers:

- prefill: compute positions ``num_computed_tokens .. num_tokens - 1`` (possibly in chunks);
- decode: exactly one uncomputed token remains (the last sampled one); compute it and sample the
  next;
- recompute preemption: drop the cache (``num_computed_tokens = 0``) and keep the tokens, so the
  next prefill covers prompt + generated tokens.

No torch import here.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True, slots=True)
class SamplingParams:
    """Per-request sampling settings.

    ``temperature == 0`` means greedy decoding. ``ignore_eos`` makes the request generate exactly
    ``max_tokens`` tokens (used by the benchmark).
    """

    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 16
    ignore_eos: bool = False

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        if not 0 < self.top_p <= 1:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {self.max_tokens}")


class SequenceStatus(Enum):
    WAITING = "waiting"  # no KV cache held; in the waiting queue
    RUNNING = "running"  # holds KV cache blocks; scheduled every step until it finishes
    FINISHED = "finished"


class FinishReason(Enum):
    STOP = "stop"  # sampled the EOS token
    LENGTH = "length"  # generated max_tokens tokens
    ABORT = "abort"  # cancelled by the caller


class Sequence:
    """One request's tokens and progress.

    Args:
        seq_id: unique id, assigned by the caller.
        prompt_token_ids: the prompt; must not be empty.
        sampling_params: sampling settings.
        arrival_time: when the request arrived (seconds, caller's clock); used for metrics only.
    """

    __slots__ = (
        "arrival_time",
        "finish_reason",
        "num_computed_tokens",
        "num_preemptions",
        "output_token_ids",
        "prompt_token_ids",
        "sampling_params",
        "seq_id",
        "status",
    )

    def __init__(
        self,
        seq_id: int,
        prompt_token_ids: Iterable[int],
        sampling_params: SamplingParams | None = None,
        arrival_time: float = 0.0,
    ) -> None:
        self.seq_id = seq_id
        self.prompt_token_ids: tuple[int, ...] = tuple(prompt_token_ids)
        if not self.prompt_token_ids:
            raise ValueError("prompt must contain at least one token")
        self.sampling_params = sampling_params if sampling_params is not None else SamplingParams()
        self.arrival_time = arrival_time
        self.output_token_ids: list[int] = []
        self.status = SequenceStatus.WAITING
        self.finish_reason: FinishReason | None = None
        self.num_computed_tokens = 0
        self.num_preemptions = 0

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self.output_token_ids)

    @property
    def num_tokens(self) -> int:
        """Prompt plus generated tokens."""
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    @property
    def token_ids(self) -> tuple[int, ...]:
        """Prompt followed by generated tokens."""
        return self.prompt_token_ids + tuple(self.output_token_ids)

    @property
    def num_uncomputed_tokens(self) -> int:
        """Tokens whose KV is not yet in the cache (1 during decode)."""
        return self.num_tokens - self.num_computed_tokens

    @property
    def is_finished(self) -> bool:
        return self.status is SequenceStatus.FINISHED

    def append_token(self, token_id: int) -> None:
        self.output_token_ids.append(token_id)

    def __repr__(self) -> str:
        return (
            f"Sequence(id={self.seq_id}, status={self.status.value}, "
            f"prompt={self.num_prompt_tokens}, out={self.num_output_tokens}, "
            f"computed={self.num_computed_tokens})"
        )
