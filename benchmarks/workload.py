"""Seeded benchmark trace: prompt texts, prompt and output lengths, and Poisson arrival gaps.

A trace is generated once per seed and replayed unchanged against every system. Each request
gets a prompt whose length, after the server applies the chat template, is an exact number of
tokens: a slice of token ids from a fixed public-domain text is decoded and, if re-encoding with
the template changes the count, the slice is lengthened or shortened until it matches. Servers
report the prompt token count they saw, and the analysis checks it against the trace.

Arrival gaps are stored as unit-mean exponentials; replaying at rate lambda divides them by
lambda, so every rate uses the same requests in the same order.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

TEXT_PATH = Path(__file__).parent / "data" / "text" / "pride_and_prejudice.txt"
# Pins the source text; also in data/text/README.md and in every run.json.
TEXT_SHA256 = "518e6f9e8e2efccefce84aeeff981df5c2cb016a75526c99f5609a56b5bd3209"

# Requests arriving in the first WARMUP_S seconds are sent but excluded from the metrics.
WARMUP_S = 30.0
# After warm-up, at least MIN_MEASURED requests and at least MIN_MEASURE_S seconds of arrivals.
MIN_MEASURED = 300
MIN_MEASURE_S = 120.0
# How many times the slice length is corrected before giving up on a prompt.
MAX_LENGTH_FIXES = 8


@dataclass(frozen=True)
class Mixture:
    """Prompt and output length distribution (see docs/decisions.md, benchmark workload)."""

    short_prompt: tuple[int, int] = (128, 512)
    long_prompt: tuple[int, int] = (1536, 3072)
    long_fraction: float = 0.2
    output: tuple[int, int] = (64, 512)


@dataclass(frozen=True)
class TraceRequest:
    index: int
    prompt: str  # user message content; the server applies the chat template
    prompt_tokens: int  # length of the templated prompt in tokens
    max_tokens: int  # generated exactly, with ignore_eos
    gap: float  # unit-mean exponential inter-arrival gap
    long: bool


class WorkloadTokenizer(Protocol):
    def encode_text(self, text: str) -> list[int]:
        """Token ids of plain text, no special tokens."""
        ...

    def decode(self, ids: Sequence[int]) -> str: ...

    def chat_len(self, content: str) -> int:
        """Length of a one-message user chat after the server's chat template."""
        ...


class HFWorkloadTokenizer:
    """The Qwen3 tokenizer, templated exactly as the stallfree server does it."""

    def __init__(self, path: str, revision: str | None = None) -> None:
        from transformers import AutoTokenizer

        from stallfree.tokenizer import HFChatTokenizer

        self._tok: Any = AutoTokenizer.from_pretrained(path, revision=revision)
        self._chat = HFChatTokenizer(path, revision=revision)

    def encode_text(self, text: str) -> list[int]:
        ids: list[int] = self._tok.encode(text, add_special_tokens=False)
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        return self._chat.decode(ids)

    def chat_len(self, content: str) -> int:
        return len(self._chat.encode_chat([{"role": "user", "content": content}], {}))


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prompt(
    tok: WorkloadTokenizer, book: Sequence[int], offset: int, target: int, overhead: int
) -> str:
    k = target - overhead
    for _ in range(MAX_LENGTH_FIXES):
        text = tok.decode(book[offset : offset + k])
        n = tok.chat_len(text)
        if n == target:
            return text
        k += target - n
    raise ValueError(f"no slice at offset {offset} gives a {target}-token prompt")


def make_trace(
    tok: WorkloadTokenizer,
    text: str,
    seed: int,
    num_requests: int,
    mixture: Mixture = Mixture(),
) -> list[TraceRequest]:
    """``num_requests`` requests for ``seed``. A longer trace with the same seed starts with the
    same requests, because every request draws the same random values in the same order."""
    book = tok.encode_text(text)
    overhead = tok.chat_len("")
    longest = mixture.long_prompt[1]
    if len(book) < 2 * longest:
        raise ValueError("source text is too short for the longest prompt")
    rng = random.Random(seed)
    out = []
    for i in range(num_requests):
        long = rng.random() < mixture.long_fraction
        lo, hi = mixture.long_prompt if long else mixture.short_prompt
        target = rng.randint(lo, hi)
        max_tokens = rng.randint(*mixture.output)
        gap = rng.expovariate(1.0)
        offset = rng.randrange(len(book) - 2 * longest)
        prompt = _prompt(tok, book, offset, target, overhead)
        out.append(TraceRequest(i, prompt, target, max_tokens, gap, long))
    return out


def trace_sha256(trace: Sequence[TraceRequest]) -> str:
    h = hashlib.sha256()
    for r in trace:
        h.update(json.dumps(asdict(r), sort_keys=True, ensure_ascii=False).encode())
        h.update(b"\n")
    return h.hexdigest()


def arrivals(trace: Sequence[TraceRequest], rate: float) -> list[float]:
    """Send times in seconds from the start of the run, at ``rate`` requests per second."""
    t, out = 0.0, []
    for r in trace:
        t += r.gap / rate
        out.append(t)
    return out


@dataclass(frozen=True)
class Window:
    """Warm-up and measurement lengths of one open-loop run (defaults: the benchmark's)."""

    warmup_s: float = WARMUP_S
    min_measured: int = MIN_MEASURED
    min_measure_s: float = MIN_MEASURE_S


DEFAULT_WINDOW = Window()


def plan(
    trace: Sequence[TraceRequest], rate: float, window: Window = DEFAULT_WINDOW
) -> tuple[int, int]:
    """(warm-up requests, measured requests) for one run at ``rate``: everything arriving in the
    first ``warmup_s`` seconds is warm-up; then the longer of ``min_measured`` requests and
    ``min_measure_s`` seconds of arrivals is measured."""
    times = arrivals(trace, rate)
    warmup = sum(1 for t in times if t < window.warmup_s)
    end = times[warmup - 1] if warmup else 0.0
    by_time = sum(1 for t in times[warmup:] if t < end + window.min_measure_s)
    measured = max(window.min_measured, by_time)
    if warmup + measured > len(trace):
        raise ValueError(f"trace has {len(trace)} requests; rate {rate} needs {warmup + measured}")
    return warmup, measured


def requests_needed(rate: float) -> int:
    """A trace length that covers ``plan`` at ``rate`` with margin for Poisson variation."""
    expected = WARMUP_S * rate + max(MIN_MEASURED, MIN_MEASURE_S * rate)
    return math.ceil(expected * 1.15) + 50


def dumps(trace: Sequence[TraceRequest]) -> bytes:
    """Gzipped JSON lines, for passing a trace to a remote container."""
    lines = (json.dumps(asdict(r), ensure_ascii=False) for r in trace)
    return gzip.compress("\n".join(lines).encode())


def loads(blob: bytes) -> list[TraceRequest]:
    return [TraceRequest(**json.loads(line)) for line in gzip.decompress(blob).decode().split("\n")]
