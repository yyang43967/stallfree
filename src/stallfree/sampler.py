"""Pick the next token for each sampled row.

- ``temperature == 0``: greedy, the highest-scoring token (ties go to the lowest token id, as
  ``torch.argmax`` and Hugging Face do).
- otherwise: divide logits by the temperature, turn them into probabilities, keep the smallest
  set of most likely tokens whose total probability reaches ``top_p`` (nucleus sampling), and
  draw one token from that set with the sampler's own random generator.

The generator is seeded once per sampler, so a run with the same seed and the same sequence of
steps draws the same tokens.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from stallfree.sequence import SamplingParams


def top_p_mask(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    """Boolean mask over ``probs [V]`` of the tokens nucleus sampling keeps.

    A token is kept if the probability of all strictly more likely tokens is below ``top_p``, so
    the most likely token is always kept and the kept set is the smallest one reaching ``top_p``.
    """
    sorted_probs, order = probs.sort(descending=True)
    before = sorted_probs.cumsum(0) - sorted_probs  # mass of the tokens ranked above each one
    keep_sorted = before < top_p
    keep = torch.zeros_like(keep_sorted)
    keep[order] = keep_sorted
    return keep


class Sampler:
    """Samples one token per row of a logits matrix.

    Args:
        seed: seed for the random generator used by non-greedy rows.
        device: where the generator lives (must match the logits).
    """

    def __init__(self, seed: int = 0, device: str | torch.device = "cpu") -> None:
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(seed)

    def sample(self, logits: torch.Tensor, params: Sequence[SamplingParams]) -> list[int]:
        """``logits [N, V]`` and one ``SamplingParams`` per row -> N token ids.

        Raises:
            ValueError: if the number of rows and params differ.
        """
        if logits.shape[0] != len(params):
            raise ValueError(f"{logits.shape[0]} logit rows but {len(params)} sampling params")
        tokens: list[int] = []
        for row, p in zip(logits, params, strict=True):
            if p.temperature == 0:
                tokens.append(int(row.argmax()))
                continue
            probs = torch.softmax(row.float() / p.temperature, dim=-1)  # [V], in fp32
            if p.top_p < 1.0:
                probs = probs * top_p_mask(probs, p.top_p)  # multinomial renormalises the rest
            tokens.append(int(torch.multinomial(probs, 1, generator=self.generator)))
        return tokens
