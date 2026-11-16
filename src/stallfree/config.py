"""Configuration dataclasses. Validated at construction; no torch import."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    """Scheduler limits and policy.

    Attributes:
        max_num_seqs: most sequences that may be running (holding KV cache) at once.
        max_num_batched_tokens: token budget of one step (sum of new tokens over all entries).
            A single prefill longer than the budget is still admitted, alone, when mixed batching
            is off.
        max_model_len: longest sequence (prompt + generated) the model accepts.
        watermark: fraction of all KV blocks that must stay free after admitting a new prefill.
            Running sequences may use the reserve for decode, which lowers the chance that a just
            admitted request immediately forces a preemption.
        enable_mixed_batching: stall-free scheduling (prefill chunks mixed with decodes). Not
            implemented yet; must be False.
        eos_token_id: token that ends a request unless ``ignore_eos``; None disables EOS.
    """

    max_num_seqs: int = 256
    max_num_batched_tokens: int = 4096
    max_model_len: int = 4096
    watermark: float = 0.01
    enable_mixed_batching: bool = False
    eos_token_id: int | None = None

    def __post_init__(self) -> None:
        if self.max_num_seqs < 1:
            raise ValueError(f"max_num_seqs must be >= 1, got {self.max_num_seqs}")
        if self.max_num_batched_tokens < self.max_num_seqs:
            # Otherwise a decode step for every running sequence would not fit in one step.
            raise ValueError(
                f"max_num_batched_tokens ({self.max_num_batched_tokens}) must be >= "
                f"max_num_seqs ({self.max_num_seqs})"
            )
        if self.max_model_len < 1:
            raise ValueError(f"max_model_len must be >= 1, got {self.max_model_len}")
        if not 0 <= self.watermark < 1:
            raise ValueError(f"watermark must be in [0, 1), got {self.watermark}")
