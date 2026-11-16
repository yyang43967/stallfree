"""Tests for request state and scheduler configuration."""

from __future__ import annotations

import pytest

from stallfree.config import SchedulerConfig
from stallfree.sequence import SamplingParams, Sequence, SequenceStatus


def test_sequence_counts_and_tokens() -> None:
    seq = Sequence(3, [10, 11, 12], SamplingParams(max_tokens=5))
    assert seq.status is SequenceStatus.WAITING
    assert seq.finish_reason is None
    assert (seq.num_prompt_tokens, seq.num_output_tokens, seq.num_tokens) == (3, 0, 3)
    assert seq.num_uncomputed_tokens == 3
    seq.num_computed_tokens = 3
    seq.append_token(20)
    assert seq.token_ids == (10, 11, 12, 20)
    assert seq.num_uncomputed_tokens == 1
    assert seq.prompt_token_ids == (10, 11, 12)
    assert seq.output_token_ids == [20]
    assert not seq.is_finished


def test_sequence_copies_prompt() -> None:
    prompt = [1, 2]
    seq = Sequence(0, prompt)
    prompt.append(3)
    assert seq.prompt_token_ids == (1, 2)


def test_empty_prompt_rejected() -> None:
    with pytest.raises(ValueError):
        Sequence(0, [])


@pytest.mark.parametrize(
    "kwargs",
    [{"temperature": -0.1}, {"top_p": 0.0}, {"top_p": 1.1}, {"max_tokens": 0}],
)
def test_invalid_sampling_params(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        SamplingParams(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_num_seqs": 0},
        {"max_num_seqs": 8, "max_num_batched_tokens": 4},
        {"max_model_len": 0},
        {"watermark": -0.1},
        {"watermark": 1.0},
    ],
)
def test_invalid_scheduler_config(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        SchedulerConfig(**kwargs)  # type: ignore[arg-type]
