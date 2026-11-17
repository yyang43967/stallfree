"""Synchronous engine: one ``step()`` schedules, runs the model, samples, and records results.

The engine owns every component and is not thread-safe; the async server (later) calls it from a
single loop. Requests are token ids in and token ids out; text and chat templates belong to the
API layer.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import count

import torch

from stallfree.block_manager import BlockManager
from stallfree.config import AttentionBackendName, EngineConfig
from stallfree.kv_cache import KVCache
from stallfree.model.loader import load_model
from stallfree.model_runner import ModelRunner
from stallfree.sampler import Sampler
from stallfree.scheduler import Scheduler
from stallfree.sequence import FinishReason, SamplingParams
from stallfree.sequence import Sequence as Request


@dataclass(frozen=True, slots=True)
class StepOutput:
    """One sequence's result from one step: the token it produced and whether it finished."""

    seq_id: int
    token_id: int
    finished: bool
    finish_reason: FinishReason | None


class LLMEngine:
    """Loads the model and runs continuous batching over a paged KV cache.

    Raises:
        NotImplementedError: for the flash backend (not implemented yet).
        ValueError: if ``max_model_len`` exceeds what the model supports.
    """

    def __init__(self, config: EngineConfig) -> None:
        if config.attention is not AttentionBackendName.TORCH_REF:
            raise NotImplementedError(f"attention backend {config.attention.value!r}")
        self.config = config
        dtype: torch.dtype = getattr(torch, config.dtype)
        self.model = load_model(config.model, dtype=dtype, device=config.device)
        mcfg = self.model.config
        if config.scheduler.max_model_len > mcfg.max_position_embeddings:
            raise ValueError(
                f"max_model_len {config.scheduler.max_model_len} exceeds the model's "
                f"max_position_embeddings {mcfg.max_position_embeddings}"
            )
        self.cache = KVCache(
            num_layers=mcfg.num_hidden_layers,
            num_blocks=config.num_blocks,
            block_size=config.block_size,
            num_kv_heads=mcfg.num_key_value_heads,
            head_dim=mcfg.head_dim,
            dtype=dtype,
            device=config.device,
        )
        self.block_manager = BlockManager(config.num_blocks, config.block_size)
        self.scheduler = Scheduler(config.scheduler, self.block_manager)
        self.runner = ModelRunner(self.model, self.cache, device=config.device)
        self.sampler = Sampler(seed=config.seed, device=config.device)
        self._ids = count()
        self.num_steps = 0
        self.num_preemptions = 0

    def add_request(
        self,
        prompt_token_ids: Sequence[int],
        sampling_params: SamplingParams | None = None,
        arrival_time: float = 0.0,
    ) -> int:
        """Queue a request; returns its id. Raises ``ValueError`` if it can never fit."""
        seq = Request(next(self._ids), prompt_token_ids, sampling_params, arrival_time)
        self.scheduler.add(seq)
        return seq.seq_id

    def abort(self, seq_id: int) -> bool:
        """Cancel a waiting or running request; False if it is unknown or already finished."""
        return self.scheduler.abort(seq_id) is not None

    def has_unfinished(self) -> bool:
        return self.scheduler.has_unfinished()

    def step(self) -> list[StepOutput]:
        """Run one step. Returns one output per sequence that produced a token, in batch order."""
        plan = self.scheduler.schedule()
        self.num_preemptions += len(plan.preempted_seq_ids)
        if plan.is_empty:
            return []
        result = self.runner.execute(plan)
        params = {e.seq_id: e.sampling_params for e in plan.entries}
        tokens = self.sampler.sample(result.logits, [params[s] for s in result.seq_ids])
        sampled = dict(zip(result.seq_ids, tokens, strict=True))
        finished = {seq.seq_id: seq.finish_reason for seq in self.scheduler.update(plan, sampled)}
        self.num_steps += 1
        return [
            StepOutput(sid, tok, sid in finished, finished.get(sid))
            for sid, tok in zip(result.seq_ids, tokens, strict=True)
        ]

    def generate(
        self,
        prompts: Sequence[Sequence[int]],
        sampling_params: SamplingParams | Sequence[SamplingParams] | None = None,
    ) -> list[list[int]]:
        """Offline convenience: run the given prompts to completion; outputs in prompt order."""
        if sampling_params is None or isinstance(sampling_params, SamplingParams):
            sampling_params = [sampling_params or SamplingParams()] * len(prompts)
        ids = [self.add_request(p, sp) for p, sp in zip(prompts, sampling_params, strict=True)]
        outputs: dict[int, list[int]] = {i: [] for i in ids}
        while self.has_unfinished():
            for out in self.step():
                outputs[out.seq_id].append(out.token_id)
        return [outputs[i] for i in ids]
