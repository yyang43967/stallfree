"""The interface between the model and an attention backend.

The model computes Q, K, and V for every token in the step's flat batch and hands them to the
backend together with the layer index. The backend knows, from per-step metadata prepared by the
runner, which tokens belong to which sequence, where each token's K/V goes in the cache, and which
cached positions each query may attend to. It returns one output vector per query head per token.
"""

from __future__ import annotations

from typing import Protocol

import torch


class AttentionBackend(Protocol):
    def __call__(
        self, layer_idx: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Causal attention for one layer.

        Args:
            layer_idx: which layer (selects that layer's KV cache).
            q: ``[T, num_heads, head_dim]`` queries, RoPE already applied.
            k: ``[T, num_kv_heads, head_dim]`` keys for the step's new tokens, RoPE applied.
            v: ``[T, num_kv_heads, head_dim]`` values for the step's new tokens.

        Returns:
            ``[T, num_heads, head_dim]`` attention output.
        """
        ...
