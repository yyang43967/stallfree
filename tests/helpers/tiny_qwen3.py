"""A tiny, randomly initialised Qwen3 built in the test process, and reference helpers.

No downloads: the Hugging Face model is constructed from a config, its weights seeded, and saved
to a temporary directory in the same safetensors format as a real checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import Qwen3Config as HFQwen3Config
from transformers import Qwen3ForCausalLM as HFQwen3ForCausalLM

# Shapes mirror Qwen3-0.6B's structure at toy scale, including num_heads * head_dim != hidden_size
# (0.6B: 16 * 128 = 2048 vs hidden 1024) and 2 query heads per KV head.
TINY_CONFIG = {
    "vocab_size": 256,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 3,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 32,
    "max_position_embeddings": 1024,
    "rms_norm_eps": 1e-6,
    "rope_parameters": {"rope_theta": 1_000_000.0, "rope_type": "default"},
    "tie_word_embeddings": True,
    "initializer_range": 0.1,
}


def make_hf_model(seed: int = 0, **overrides: object) -> HFQwen3ForCausalLM:
    """HF reference model in fp32 with eager attention and seeded random weights.

    Norm weights are randomised too (they start at 1.0), so a norm weight that fails to load
    cannot go unnoticed.
    """
    torch.manual_seed(seed)
    cfg = HFQwen3Config(**{**TINY_CONFIG, **overrides})  # type: ignore[arg-type]
    model = HFQwen3ForCausalLM(cfg).to(torch.float32)
    model.config._attn_implementation = "eager"
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("norm.weight"):
                p.uniform_(0.5, 1.5)
    return model.eval()


def save_hf_model(model: HFQwen3ForCausalLM, path: Path) -> Path:
    model.save_pretrained(path)
    return path


def hf_logits(model: HFQwen3ForCausalLM, token_ids: list[int]) -> torch.Tensor:
    """``[len(token_ids), vocab]`` logits from one full forward pass, no cache."""
    with torch.no_grad():
        out = model(input_ids=torch.tensor([token_ids]), use_cache=False)
    return out.logits[0]


def hf_greedy(model: HFQwen3ForCausalLM, prompt: list[int], max_new_tokens: int) -> list[int]:
    """Greedy continuation with HF's own generate() (eager attention, its KV cache)."""
    with torch.no_grad():
        out = model.generate(
            input_ids=torch.tensor([prompt]),
            attention_mask=torch.ones(1, len(prompt), dtype=torch.long),
            max_new_tokens=max_new_tokens,
            min_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=0,
        )
    return [int(t) for t in out[0, len(prompt) :]]


class DenseCausalAttention:
    """Attention without any cache: each segment attends causally to itself only.

    Used to check the model's arithmetic before the paged backend exists. Mirrors the HF eager
    path: scores = q k^T * scale, additive mask with the dtype's minimum, softmax in fp32, then
    weights @ v.
    """

    def __init__(self, seq_lens: list[int]) -> None:
        self.seq_lens = seq_lens

    def __call__(
        self, layer_idx: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        del layer_idx
        groups = q.shape[1] // k.shape[1]
        scale = q.shape[-1] ** -0.5
        out = []
        start = 0
        for n in self.seq_lens:
            qs = q[start : start + n].transpose(0, 1)  # [H, n, D]
            ks = k[start : start + n].repeat_interleave(groups, dim=1).transpose(0, 1)
            vs = v[start : start + n].repeat_interleave(groups, dim=1).transpose(0, 1)
            scores = torch.matmul(qs, ks.transpose(1, 2)) * scale  # [H, n, n]
            mask = torch.full((n, n), torch.finfo(q.dtype).min, dtype=q.dtype).triu(1)
            weights = torch.softmax(scores + mask, dim=-1, dtype=torch.float32).to(q.dtype)
            out.append(torch.matmul(weights, vs).transpose(0, 1))  # [n, H, D]
            start += n
        return torch.cat(out, dim=0)


# --- comparison with the reference, and near-tie tracking ---------------------------------

_MIN_GAP: dict[str, float] = {}


def record_gap(ref_logits: torch.Tensor, label: str) -> float:
    """Remember the smallest top-1 minus top-2 gap of the reference over these rows.

    Not an assertion. If a later scenario ever disagrees on a token, compare the gap at that
    position with this number: a gap near fp32 rounding (~1e-5) means a near-tie on the random
    model; a large gap means a real bug.
    """
    top2 = ref_logits.float().topk(2, dim=-1).values
    gap = float((top2[..., 0] - top2[..., 1]).min())
    _MIN_GAP[label] = min(gap, _MIN_GAP.get(label, float("inf")))
    return gap


def min_gaps() -> dict[str, float]:
    return dict(_MIN_GAP)


def assert_matches_reference(got: torch.Tensor, ref: torch.Tensor, label: str) -> None:
    """Identical predicted tokens and logits within the default fp32 tolerances."""
    record_gap(ref, label)
    assert torch.equal(got.argmax(-1), ref.argmax(-1)), "predicted tokens differ"
    torch.testing.assert_close(got, ref)  # default tolerances, never loosened
