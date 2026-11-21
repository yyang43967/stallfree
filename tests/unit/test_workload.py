"""The benchmark trace: exact templated prompt lengths, the length mixture, determinism, the
warm-up/measurement plan, and the pinned source text."""

from __future__ import annotations

import itertools
import statistics
from collections.abc import Sequence

import pytest

from benchmarks.workload import (
    MIN_MEASURE_S,
    MIN_MEASURED,
    TEXT_PATH,
    TEXT_SHA256,
    WARMUP_S,
    HFWorkloadTokenizer,
    Mixture,
    Window,
    arrivals,
    dumps,
    file_sha256,
    loads,
    make_trace,
    plan,
    requests_needed,
    trace_sha256,
)

QWEN = "Qwen/Qwen3-0.6B"
QWEN_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


class CharTokenizer:
    """One token per character; the 'template' wraps the content in fixed text."""

    def encode_text(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, ids: Sequence[int]) -> str:
        return "".join(chr(i) for i in ids)

    def chat_len(self, content: str) -> int:
        return len(f"<user>{content}</user><assistant>")


SMALL = Mixture(short_prompt=(30, 60), long_prompt=(200, 300), long_fraction=0.2, output=(4, 9))


@pytest.fixture(scope="module")
def text() -> str:
    return TEXT_PATH.read_text()


@pytest.fixture(scope="module")
def qwen() -> HFWorkloadTokenizer:
    return HFWorkloadTokenizer(QWEN, revision=QWEN_REVISION)


def test_source_text_is_pinned() -> None:
    assert file_sha256(TEXT_PATH) == TEXT_SHA256
    readme = (TEXT_PATH.parent / "README.md").read_text()
    assert TEXT_SHA256 in readme


def test_qwen_prompts_have_exact_templated_lengths(qwen: HFWorkloadTokenizer, text: str) -> None:
    trace = make_trace(qwen, text, seed=7, num_requests=60)
    for r in trace:
        assert qwen.chat_len(r.prompt) == r.prompt_tokens
        lo, hi = Mixture().long_prompt if r.long else Mixture().short_prompt
        assert lo <= r.prompt_tokens <= hi
        assert 64 <= r.max_tokens <= 512


def test_mixture_proportions_and_ranges(text: str) -> None:
    trace = make_trace(CharTokenizer(), text, seed=1, num_requests=2000, mixture=SMALL)
    long = [r for r in trace if r.long]
    assert abs(len(long) / len(trace) - 0.2) < 0.03
    assert all(200 <= r.prompt_tokens <= 300 for r in long)
    assert all(30 <= r.prompt_tokens <= 60 for r in trace if not r.long)
    assert {r.max_tokens for r in trace} == set(range(4, 10))
    assert abs(statistics.fmean(r.gap for r in trace) - 1.0) < 0.06


def test_same_seed_same_trace_and_longer_traces_extend_shorter_ones(text: str) -> None:
    tok = CharTokenizer()
    a = make_trace(tok, text, seed=3, num_requests=40, mixture=SMALL)
    b = make_trace(tok, text, seed=3, num_requests=80, mixture=SMALL)
    c = make_trace(tok, text, seed=4, num_requests=40, mixture=SMALL)
    assert b[:40] == a
    assert trace_sha256(a) == trace_sha256(b[:40])
    assert trace_sha256(a) != trace_sha256(c)


def test_dumps_loads_round_trip(text: str) -> None:
    trace = make_trace(CharTokenizer(), text, seed=2, num_requests=30, mixture=SMALL)
    assert loads(dumps(trace)) == trace


def test_arrivals_scale_with_rate(text: str) -> None:
    trace = make_trace(CharTokenizer(), text, seed=2, num_requests=30, mixture=SMALL)
    slow, fast = arrivals(trace, 1.0), arrivals(trace, 4.0)
    assert fast == pytest.approx([t / 4 for t in slow])
    assert all(b > a for a, b in itertools.pairwise(slow))


@pytest.mark.parametrize("rate", [0.5, 2.0, 10.0])
def test_plan_uses_warmup_then_the_longer_of_count_and_duration(text: str, rate: float) -> None:
    trace = make_trace(CharTokenizer(), text, seed=5, num_requests=4000, mixture=SMALL)
    warmup, measured = plan(trace, rate)
    times = arrivals(trace, rate)
    assert all(t < WARMUP_S for t in times[:warmup])
    assert times[warmup] >= WARMUP_S
    assert measured >= MIN_MEASURED
    end_of_warmup = times[warmup - 1]
    if measured > MIN_MEASURED:  # duration decides: every arrival in the 120 s is measured
        assert times[warmup + measured - 1] < end_of_warmup + MIN_MEASURE_S
        assert times[warmup + measured] >= end_of_warmup + MIN_MEASURE_S
    else:  # count decides: 300 requests take at least 120 s to arrive
        assert times[warmup + measured - 1] >= end_of_warmup + MIN_MEASURE_S
    assert requests_needed(rate) >= warmup + measured


def test_plan_rejects_a_short_trace(text: str) -> None:
    trace = make_trace(CharTokenizer(), text, seed=5, num_requests=50, mixture=SMALL)
    with pytest.raises(ValueError, match="needs"):
        plan(trace, 1.0)
    w, m = plan(trace, 10.0, Window(warmup_s=1.0, min_measured=20, min_measure_s=1.0))
    assert w == sum(1 for t in arrivals(trace, 10.0) if t < 1.0)
    assert m == 20  # 1 s at 10 req/s is about 10 requests, so the count decides
