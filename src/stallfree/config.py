"""Configuration dataclasses. Validated at construction; no torch import."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


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


# FlashAttention-2's paged KV path requires the block size to be a multiple of this.
FLASH_BLOCK_SIZE_MULTIPLE = 256
DTYPES = ("float32", "bfloat16", "float16")


class AttentionBackendName(Enum):
    TORCH_REF = "torch_ref"  # plain PyTorch, any device; correctness reference
    FLASH = "flash"  # FlashAttention-2, CUDA only


@dataclass(frozen=True, slots=True)
class EngineConfig:
    """Everything needed to build an engine.

    Attributes:
        model: directory with ``config.json`` and ``*.safetensors``.
        num_blocks: KV cache blocks to allocate. ``None`` sizes the cache from GPU memory after
            a profiling forward pass (CUDA only); tests and CPU runs set it explicitly.
        gpu_memory_utilization: fraction of GPU memory the engine may use in total (weights,
            activations, and KV cache) when ``num_blocks`` is ``None``.
        block_size: tokens per KV block (256 for FlashAttention-2; any size for the reference
            backend, which tests use with small blocks).
        dtype: ``"float32"``, ``"bfloat16"``, or ``"float16"`` for weights, activations, and cache.
        device: ``"cpu"``, ``"cuda"``, ...
        attention: which attention backend runs the steps.
        scheduler: scheduler limits and policy.
        seed: seed for the sampler's random generator.
    """

    model: str
    num_blocks: int | None = None
    block_size: int = FLASH_BLOCK_SIZE_MULTIPLE
    dtype: str = "bfloat16"
    device: str = "cpu"
    attention: AttentionBackendName = AttentionBackendName.TORCH_REF
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    seed: int = 0
    gpu_memory_utilization: float = 0.9

    def __post_init__(self) -> None:
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError(
                f"gpu_memory_utilization must be in (0, 1], got {self.gpu_memory_utilization}"
            )
        if self.num_blocks is not None and self.num_blocks < 1:
            raise ValueError(f"num_blocks must be >= 1, got {self.num_blocks}")
        if self.block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {self.block_size}")
        if (
            self.attention is AttentionBackendName.FLASH
            and self.block_size % FLASH_BLOCK_SIZE_MULTIPLE
        ):
            raise ValueError(
                f"the flash backend needs block_size divisible by {FLASH_BLOCK_SIZE_MULTIPLE}"
            )
        if self.dtype not in DTYPES:
            raise ValueError(f"dtype must be one of {DTYPES}, got {self.dtype!r}")
