"""Turn one scheduler step into tensors, run the model, return logits for the rows to sample.

The runner sees only the ``SchedulerOutput``: for each entry, the tokens to compute, where they
start, the sequence's block table, and whether to sample. It knows nothing about queues, blocks
being free, or why an entry is a prefill, a chunk, a decode, or a recompute.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from stallfree.attention.base import AttentionBackend
from stallfree.attention.metadata import AttentionMetadata, SegmentSpec, build_attention_metadata
from stallfree.attention.torch_ref import TorchRefAttention
from stallfree.kv_cache import KVCache
from stallfree.model.qwen3 import Qwen3ForCausalLM
from stallfree.scheduler import SchedulerOutput

AttentionFactory = Callable[[KVCache, AttentionMetadata], AttentionBackend]


@dataclass(frozen=True, slots=True)
class PreparedStep:
    """The flat batch for one step.

    Attributes:
        input_ids: ``[T]`` int64, every entry's new tokens concatenated in entry order.
        positions: ``[T]`` int64, each token's position within its own sequence.
        metadata: attention metadata (slot mapping, segment boundaries, block tables).
        sample_rows: ``[N]`` int64, row in the batch whose logits are sampled, one per entry
            with ``sample`` (the entry's last token), in entry order.
        sample_seq_ids: the sequence of each sampled row.
    """

    input_ids: torch.Tensor
    positions: torch.Tensor
    metadata: AttentionMetadata
    sample_rows: torch.Tensor
    sample_seq_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class RunnerOutput:
    """Logits ``[N, vocab]`` for the sampled rows, and which sequence each row belongs to."""

    seq_ids: tuple[int, ...]
    logits: torch.Tensor


class ModelRunner:
    """Executes scheduler steps on a model and its paged KV cache.

    Args:
        model: the model, already on ``device``.
        cache: the KV cache the scheduler's block tables index into.
        attention: builds the attention backend for a step (default: the PyTorch reference).
        device: where the batch tensors are created.
    """

    def __init__(
        self,
        model: Qwen3ForCausalLM,
        cache: KVCache,
        *,
        attention: AttentionFactory = TorchRefAttention,
        device: str | torch.device = "cpu",
    ) -> None:
        self.model = model
        self.cache = cache
        self.attention = attention
        self.device = torch.device(device)

    def prepare(self, output: SchedulerOutput) -> PreparedStep:
        """Build the flat batch for ``output``.

        Raises:
            ValueError: if the step has no entries.
        """
        if output.is_empty:
            raise ValueError("nothing to run: the step has no entries")
        ids: list[int] = []
        positions: list[int] = []
        segments: list[SegmentSpec] = []
        sample_rows: list[int] = []
        sample_seq_ids: list[int] = []
        for entry in output.entries:
            n = entry.num_new_tokens
            ids.extend(entry.token_ids)
            positions.extend(range(entry.start_pos, entry.start_pos + n))
            segments.append(SegmentSpec(entry.start_pos, n, entry.block_table))
            if entry.sample:
                sample_rows.append(len(ids) - 1)  # the entry's last token predicts the next one
                sample_seq_ids.append(entry.seq_id)
        dev = self.device
        return PreparedStep(
            input_ids=torch.tensor(ids, dtype=torch.long, device=dev),
            positions=torch.tensor(positions, dtype=torch.long, device=dev),
            metadata=build_attention_metadata(segments, self.cache.block_size, dev),
            sample_rows=torch.tensor(sample_rows, dtype=torch.long, device=dev),
            sample_seq_ids=tuple(sample_seq_ids),
        )

    @torch.inference_mode()
    def execute(self, output: SchedulerOutput) -> RunnerOutput:
        """Run one step: write the new tokens' K/V to the cache and return the sampled rows' logits.

        Every entry's KV is written even when it is not sampled (a non-final prefill chunk), so
        the next chunk can attend to it.
        """
        step = self.prepare(output)
        attn = self.attention(self.cache, step.metadata)
        hidden = self.model(step.input_ids, step.positions, attn)  # [T, hidden]
        logits = self.model.compute_logits(hidden[step.sample_rows])  # [N, vocab]
        return RunnerOutput(seq_ids=step.sample_seq_ids, logits=logits)
