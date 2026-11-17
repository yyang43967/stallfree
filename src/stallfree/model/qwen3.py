"""Qwen3 dense decoder over a flat batch of tokens.

All tokens of a step (from any number of sequences, prefill chunks and decodes alike) are
concatenated into one dimension ``T``. Positions are given per token, so RoPE needs no notion of
sequences. Only attention needs to know where sequences start and end; that is the backend's job
(see ``stallfree.attention.base``).

The arithmetic mirrors Hugging Face transformers' Qwen3 implementation operation by operation
(RMSNorm in fp32, RoPE from ``positions * inv_freq`` computed on the fly, SwiGLU MLP), so results
can be compared exactly in fp32.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from stallfree.attention.base import AttentionBackend


@dataclass(frozen=True, slots=True)
class Qwen3Config:
    """The subset of a Qwen3 ``config.json`` this engine uses."""

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    tie_word_embeddings: bool

    def __post_init__(self) -> None:
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be a multiple of num_key_value_heads")
        if self.head_dim % 2:
            raise ValueError("head_dim must be even for RoPE")

    @property
    def num_kv_groups(self) -> int:
        """Query heads that share one KV head (grouped-query attention)."""
        return self.num_attention_heads // self.num_key_value_heads

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Qwen3Config:
        """Parse a Hugging Face ``config.json`` dict, rejecting features this engine lacks."""
        if d.get("hidden_act", "silu") != "silu":
            raise ValueError(f"unsupported hidden_act {d.get('hidden_act')!r}")
        if d.get("attention_bias", False):
            raise ValueError("attention_bias is not supported")
        if d.get("use_sliding_window", False):
            raise ValueError("sliding-window attention is not supported")
        rope = d.get("rope_parameters") or d.get("rope_scaling") or {}
        if rope.get("rope_type", "default") != "default":
            raise ValueError(f"unsupported RoPE type {rope.get('rope_type')!r}")
        # Older configs keep rope_theta at the top level; newer ones in rope_parameters.
        rope_theta = rope.get("rope_theta", d.get("rope_theta"))
        if rope_theta is None:
            raise ValueError("config has no rope_theta")
        num_heads = d["num_attention_heads"]
        return cls(
            vocab_size=d["vocab_size"],
            hidden_size=d["hidden_size"],
            intermediate_size=d["intermediate_size"],
            num_hidden_layers=d["num_hidden_layers"],
            num_attention_heads=num_heads,
            num_key_value_heads=d.get("num_key_value_heads") or num_heads,
            head_dim=d.get("head_dim") or d["hidden_size"] // num_heads,
            rms_norm_eps=d["rms_norm_eps"],
            rope_theta=float(rope_theta),
            max_position_embeddings=d["max_position_embeddings"],
            tie_word_embeddings=d.get("tie_word_embeddings", False),
        )

    @classmethod
    def from_pretrained(cls, model_dir: str | Path) -> Qwen3Config:
        with open(Path(model_dir) / "config.json") as f:
            return cls.from_dict(json.load(f))


class RMSNorm(nn.Module):
    """Scale a vector to unit root-mean-square, then multiply by a learned per-channel weight."""

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x32 = x.to(torch.float32)  # statistics in fp32, as in the reference
        variance = x32.pow(2).mean(-1, keepdim=True)
        x32 = x32 * torch.rsqrt(variance + self.eps)
        return self.weight * x32.to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """``[x1, x2] -> [-x2, x1]`` over the last dimension (the rotation in RoPE)."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class RotaryEmbedding(nn.Module):
    """Rotary position embedding: rotate pairs of q/k channels by position-dependent angles."""

    def __init__(self, head_dim: int, theta: float) -> None:
        super().__init__()
        # Kept as a plain fp32 tensor (not a buffer), so casting the model to bf16 leaves it fp32.
        self.inv_freq = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim)
        )

    def forward(
        self, positions: torch.Tensor, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # positions: [T]; q: [T, H, D]; k: [T, Hkv, D]
        inv_freq = self.inv_freq.to(positions.device)
        freqs = positions[:, None].float() * inv_freq  # [T, D/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [T, D]
        cos = emb.cos().to(q.dtype)[:, None, :]  # [T, 1, D], broadcast over heads
        sin = emb.sin().to(q.dtype)[:, None, :]
        q = (q * cos) + (rotate_half(q) * sin)
        k = (k * cos) + (rotate_half(k) * sin)
        return q, k


class Qwen3Attention(nn.Module):
    def __init__(self, cfg: Qwen3Config, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.q_proj = nn.Linear(cfg.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, cfg.hidden_size, bias=False)
        # Qwen3 normalizes each head's query and key vector before RoPE.
        self.q_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        rope: RotaryEmbedding,
        attn: AttentionBackend,
    ) -> torch.Tensor:
        t = x.shape[0]  # x: [T, hidden]
        q = self.q_norm(self.q_proj(x).view(t, self.num_heads, self.head_dim))  # [T, H, D]
        k = self.k_norm(self.k_proj(x).view(t, self.num_kv_heads, self.head_dim))  # [T, Hkv, D]
        v = self.v_proj(x).view(t, self.num_kv_heads, self.head_dim)  # [T, Hkv, D]
        q, k = rope(positions, q, k)
        o = attn(self.layer_idx, q, k, v)  # [T, H, D]
        out: torch.Tensor = self.o_proj(o.reshape(t, self.num_heads * self.head_dim))
        return out  # [T, hidden]


class Qwen3MLP(nn.Module):
    """SwiGLU: ``down(silu(gate(x)) * up(x))``."""

    def __init__(self, cfg: Qwen3Config) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return out


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, cfg: Qwen3Config, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(cfg, layer_idx)
        self.mlp = Qwen3MLP(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        rope: RotaryEmbedding,
        attn: AttentionBackend,
    ) -> torch.Tensor:
        # Pre-norm residual blocks: x + f(norm(x)).
        x = x + self.self_attn(self.input_layernorm(x), positions, rope, attn)
        mlp_out: torch.Tensor = self.mlp(self.post_attention_layernorm(x))
        return x + mlp_out


class Qwen3Model(nn.Module):
    def __init__(self, cfg: Qwen3Config) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(Qwen3DecoderLayer(cfg, i) for i in range(cfg.num_hidden_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(cfg.head_dim, cfg.rope_theta)

    def forward(
        self, input_ids: torch.Tensor, positions: torch.Tensor, attn: AttentionBackend
    ) -> torch.Tensor:
        x = self.embed_tokens(input_ids)  # [T] -> [T, hidden]
        for layer in self.layers:
            x = layer(x, positions, self.rotary_emb, attn)
        out: torch.Tensor = self.norm(x)
        return out  # [T, hidden]


class Qwen3ForCausalLM(nn.Module):
    """Parameter names match the Hugging Face checkpoint, so weights load by name."""

    def __init__(self, cfg: Qwen3Config) -> None:
        super().__init__()
        self.config = cfg
        self.model = Qwen3Model(cfg)
        # With tied embeddings the output projection reuses the input embedding matrix.
        self.lm_head: nn.Linear | None = None
        if not cfg.tie_word_embeddings:
            self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def forward(
        self, input_ids: torch.Tensor, positions: torch.Tensor, attn: AttentionBackend
    ) -> torch.Tensor:
        """Final hidden states ``[T, hidden]`` for a flat batch of ``T`` tokens."""
        hidden: torch.Tensor = self.model(input_ids, positions, attn)
        return hidden

    def compute_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """``[N, hidden] -> [N, vocab]``. Called only for positions that are sampled."""
        weight = self.model.embed_tokens.weight if self.lm_head is None else self.lm_head.weight
        return F.linear(hidden, weight)
