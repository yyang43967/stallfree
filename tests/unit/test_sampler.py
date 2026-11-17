"""Greedy, temperature, and top-p sampling."""

from __future__ import annotations

import math

import pytest
import torch

from stallfree.sampler import Sampler, top_p_mask
from stallfree.sequence import SamplingParams

GREEDY = SamplingParams(temperature=0.0)
N = 20_000  # draws for frequency checks; 3-sigma of a proportion near 0.5 is about 0.011


def logits_for(probs: list[float]) -> torch.Tensor:
    return torch.tensor([math.log(p) for p in probs])


def frequencies(sampler: Sampler, row: torch.Tensor, params: SamplingParams) -> list[float]:
    tokens = sampler.sample(row.expand(N, -1), [params] * N)
    counts = torch.bincount(torch.tensor(tokens), minlength=row.numel())
    return (counts.float() / N).tolist()


def test_greedy_picks_the_highest_score() -> None:
    logits = torch.tensor([[0.1, 2.0, -1.0, 1.9], [5.0, 0.0, 0.0, 0.0]])
    assert Sampler().sample(logits, [GREEDY, GREEDY]) == [1, 0]


def test_greedy_ties_go_to_the_lowest_token_id() -> None:
    assert Sampler().sample(torch.tensor([[1.0, 3.0, 3.0, 0.0]]), [GREEDY]) == [1]


def test_greedy_ignores_top_p_and_uses_no_randomness() -> None:
    s = Sampler(seed=0)
    state = s.generator.get_state()
    logits = torch.tensor([[0.0, 1.0, 0.5]])
    assert s.sample(logits, [SamplingParams(temperature=0.0, top_p=0.1)]) == [1]
    assert torch.equal(s.generator.get_state(), state), "greedy must not consume random numbers"


def test_temperature_one_follows_the_softmax() -> None:
    probs = [0.5, 0.3, 0.2]
    freq = frequencies(Sampler(seed=1), logits_for(probs), SamplingParams(temperature=1.0))
    for f, p in zip(freq, probs, strict=True):
        assert abs(f - p) < 0.015


def test_low_temperature_sharpens_the_distribution() -> None:
    # Temperature 0.5 squares the probabilities before renormalising: 0.25, 0.09, 0.04 -> /0.38.
    probs = [0.5, 0.3, 0.2]
    freq = frequencies(Sampler(seed=2), logits_for(probs), SamplingParams(temperature=0.5))
    expected = [p * p / 0.38 for p in probs]
    for f, e in zip(freq, expected, strict=True):
        assert abs(f - e) < 0.015


@pytest.mark.parametrize(
    ("top_p", "kept"),
    [
        (0.4, [0]),
        (0.5, [0]),
        (0.7, [0, 1]),
        (0.9, [0, 1, 2]),
        (0.99, [0, 1, 2, 3]),
        (1.0, [0, 1, 2, 3]),
    ],
)
def test_top_p_keeps_the_smallest_set_reaching_p(top_p: float, kept: list[int]) -> None:
    probs = torch.tensor([0.5, 0.3, 0.15, 0.05])
    assert top_p_mask(probs, top_p).nonzero().flatten().tolist() == kept


def test_top_p_mask_follows_probability_not_token_order() -> None:
    probs = torch.tensor([0.05, 0.15, 0.5, 0.3])
    assert top_p_mask(probs, 0.7).nonzero().flatten().tolist() == [2, 3]


def test_top_p_sampling_never_draws_outside_the_nucleus() -> None:
    probs = [0.5, 0.3, 0.15, 0.05]
    params = SamplingParams(temperature=1.0, top_p=0.7)
    freq = frequencies(Sampler(seed=3), logits_for(probs), params)
    assert freq[2] == freq[3] == 0.0
    assert abs(freq[0] - 0.5 / 0.8) < 0.015  # renormalised over {0, 1}
    assert abs(freq[1] - 0.3 / 0.8) < 0.015


def test_same_seed_same_draws() -> None:
    logits = torch.randn(64, 50, generator=torch.Generator().manual_seed(0))
    params = [SamplingParams(temperature=0.8, top_p=0.9)] * 64
    assert Sampler(seed=7).sample(logits, params) == Sampler(seed=7).sample(logits, params)
    assert Sampler(seed=7).sample(logits, params) != Sampler(seed=8).sample(logits, params)


def test_mixed_rows_greedy_and_random() -> None:
    logits = torch.tensor([[0.0, 0.0, 9.0], [0.0, 0.0, 9.0]])
    s = Sampler(seed=4)
    params = [GREEDY, SamplingParams(temperature=50.0)]  # very flat second row
    draws = [s.sample(logits, params) for _ in range(200)]
    assert all(d[0] == 2 for d in draws)
    assert {d[1] for d in draws} == {0, 1, 2}


def test_bf16_logits_are_sampled_in_fp32() -> None:
    logits = logits_for([0.5, 0.3, 0.2]).to(torch.bfloat16)
    freq = frequencies(Sampler(seed=5), logits, SamplingParams(temperature=1.0))
    assert abs(freq[0] - 0.5) < 0.02


def test_row_and_param_count_must_match() -> None:
    with pytest.raises(ValueError):
        Sampler().sample(torch.zeros(2, 3), [GREEDY])
