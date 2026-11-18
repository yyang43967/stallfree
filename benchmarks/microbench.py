"""Stall micro-benchmark: steady decodes, then one long prompt arrives; mixed batching off vs on.

The engine is driven in-process (no HTTP server, no async loop), so the numbers isolate the
scheduling policy; the end-to-end comparison over HTTP is a separate benchmark.

Scenario: ``num_decodes`` requests with short prompts are admitted and decode until each has
produced ``warmup_tokens`` tokens. Then one request with a ``long_prompt_len``-token prompt arrives.
The run continues until the long request has produced ``measured_tokens`` tokens and every decode
has produced ``measured_tokens`` more; the remaining decodes are then aborted.

Metrics per run (times relative to the long request's arrival):
- inter-token latency (ITL) of the decodes: the gaps between consecutive tokens of each decode,
  starting with the gap that spans the arrival, ``measured_tokens`` gaps per decode; max, p99
  (nearest rank), and median over all decodes;
- TTFT of the long request: arrival to its first token;
- steps and output tokens from arrival to the end of the window, and output tokens per second.

GPU run:  uv run modal run -m modal_apps.dev::phase3_gpu
Report:   uv run python -m benchmarks.microbench benchmarks/data/microbench --out benchmarks/results
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

OFF = "off"


@dataclass(frozen=True)
class Scenario:
    num_decodes: int = 16
    decode_prompt_len: int = 128
    long_prompt_len: int = 3000
    warmup_tokens: int = 8
    measured_tokens: int = 64
    seed: int = 0
    # Prompt tokens are drawn uniformly from this range (ordinary vocabulary ids for Qwen3);
    # attention and matmul cost do not depend on which ids they are.
    token_low: int = 1000
    token_high: int = 100_000


def config_name(budget: int | None) -> str:
    return OFF if budget is None else f"on-{budget}"


def make_prompts(s: Scenario) -> tuple[list[list[int]], list[int]]:
    rng = random.Random(s.seed)

    def draw(n: int) -> list[int]:
        return [rng.randrange(s.token_low, s.token_high) for _ in range(n)]

    return [draw(s.decode_prompt_len) for _ in range(s.num_decodes)], draw(s.long_prompt_len)


# --------------------------------------------------------------------------------------------
# Running (needs torch; imported lazily so the report runs anywhere)


def run_once(engine: Any, scenario: Scenario, sync: Callable[[], None]) -> dict[str, Any]:
    """One run of the scenario on an idle engine with whatever scheduler it currently has."""
    from stallfree.sequence import SamplingParams

    bm = engine.block_manager
    assert bm.num_free_blocks == bm.num_blocks and not engine.has_unfinished()
    decode_prompts, long_prompt = make_prompts(scenario)
    # Decodes are aborted once measured; max_tokens only has to be large enough not to end them.
    max_len = engine.config.scheduler.max_model_len
    unbounded = SamplingParams(max_tokens=max_len - scenario.decode_prompt_len, ignore_eos=True)
    decode_ids = [engine.add_request(p, unbounded) for p in decode_prompts]
    long_ref = {"id": -1}
    times: dict[int, list[float]] = {i: [] for i in decode_ids}
    steps: list[dict[str, Any]] = []

    schedule = engine.scheduler.schedule
    last_plan: dict[str, Any] = {}

    def recording_schedule() -> Any:
        plan = schedule()
        last_plan["num_tokens"] = sum(e.num_new_tokens for e in plan.entries)
        last_plan["num_seqs"] = len(plan.entries)
        last_plan["long_prompt_tokens"] = sum(
            min(e.num_new_tokens, scenario.long_prompt_len - e.start_pos)
            for e in plan.entries
            if e.seq_id == long_ref["id"] and e.start_pos < scenario.long_prompt_len
        )
        return plan

    engine.scheduler.schedule = recording_schedule
    origin = time.perf_counter()

    def step() -> None:
        t0 = time.perf_counter()
        outs = engine.step()
        sync()
        t1 = time.perf_counter()
        for o in outs:
            times[o.seq_id].append(t1 - origin)
        steps.append({"start": t0 - origin, "end": t1 - origin, **last_plan})

    try:
        while min(len(times[i]) for i in decode_ids) < scenario.warmup_tokens:
            step()
        arrival = time.perf_counter() - origin
        long_id = engine.add_request(
            long_prompt, SamplingParams(max_tokens=scenario.measured_tokens, ignore_eos=True)
        )
        long_ref["id"] = long_id
        times[long_id] = []

        def after(i: int) -> int:
            return sum(t > arrival for t in times[i])

        while (
            after(long_id) < scenario.measured_tokens
            or min(after(i) for i in decode_ids) < scenario.measured_tokens
        ):
            step()
    finally:
        del engine.scheduler.schedule
    for i in decode_ids:
        engine.abort(i)
    while engine.has_unfinished():
        engine.step()

    def rel(ts: list[float]) -> list[float]:
        return [t - arrival for t in ts]

    return {
        "decode_token_times": [rel(times[i]) for i in decode_ids],
        "long_token_times": rel(times[long_id]),
        "steps": [{**s, "start": s["start"] - arrival, "end": s["end"] - arrival} for s in steps],
    }


def scheduler_for(base: Any, budget: int | None) -> Any:
    """The off configuration is the engine's own; mixed batching swaps in a budget."""
    if budget is None:
        return base
    return replace(
        base,
        enable_mixed_batching=True,
        max_num_batched_tokens=budget,
        max_num_seqs=min(base.max_num_seqs, budget),
    )


def run_suite(
    engine: Any,
    budgets: Sequence[int | None],
    repeats: int,
    scenario: Scenario,
    sync: Callable[[], None],
) -> list[dict[str, Any]]:
    """One discarded warm-up run per configuration, then ``repeats`` rounds over all
    configurations in turn (interleaved, so slow drift affects every configuration alike).
    One engine (model, cache, block manager) serves every run; only the scheduler changes."""
    from stallfree.scheduler import Scheduler

    base = engine.config.scheduler
    assert not base.enable_mixed_batching

    def run(budget: int | None) -> dict[str, Any]:
        engine.scheduler = Scheduler(scheduler_for(base, budget), engine.block_manager)
        return run_once(engine, scenario, sync)

    for b in budgets:
        run(b)
    runs = []
    for r in range(repeats):
        for b in budgets:
            runs.append({"config": config_name(b), "budget": b, "repeat": r, **run(b)})
    engine.scheduler = Scheduler(base, engine.block_manager)
    return runs


# --------------------------------------------------------------------------------------------
# Analysis (pure Python)


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile: the smallest value with at least ``q`` percent at or below it."""
    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100 * len(ordered)))
    return ordered[rank - 1]


def analyze(run: dict[str, Any], scenario: Scenario) -> dict[str, Any]:
    n = scenario.measured_tokens
    itls: list[float] = []
    window_end = run["long_token_times"][n - 1]
    for ts in run["decode_token_times"]:
        before = [t for t in ts if t <= 0]
        post = [t for t in ts if t > 0][:n]
        assert before and len(post) == n, "decode not measured over the full window"
        seq = [before[-1], *post]
        itls += [b - a for a, b in itertools.pairwise(seq)]
        window_end = max(window_end, post[-1])
    window_steps = [s for s in run["steps"] if s["start"] > 0 and s["end"] <= window_end + 1e-9]
    tokens = sum(
        sum(0 < t <= window_end + 1e-9 for t in ts)
        for ts in [*run["decode_token_times"], run["long_token_times"]]
    )
    return {
        "itl_max_ms": 1e3 * max(itls),
        "itl_p99_ms": 1e3 * percentile(itls, 99),
        "itl_median_ms": 1e3 * statistics.median(itls),
        "long_ttft_ms": 1e3 * run["long_token_times"][0],
        "long_prefill_steps": sum(s["long_prompt_tokens"] > 0 for s in window_steps),
        "steps": len(window_steps),
        "window_ms": 1e3 * window_end,
        "output_tokens": tokens,
        "output_tokens_per_s": tokens / window_end,
    }


METRICS = [
    ("itl_max_ms", "max ITL (ms)", 1),
    ("itl_p99_ms", "p99 ITL (ms)", 1),
    ("itl_median_ms", "median ITL (ms)", 1),
    ("long_ttft_ms", "long TTFT (ms)", 1),
    ("long_prefill_steps", "long prefill steps", 0),
    ("steps", "steps", 0),
    ("output_tokens_per_s", "output tok/s", 0),
]


def summarize(data: dict[str, Any]) -> dict[str, dict[str, list[float]]]:
    """Per configuration (in run order), each metric's values across repeats."""
    scenario = Scenario(**data["scenario"])
    out: dict[str, dict[str, list[float]]] = {}
    for run in data["runs"]:
        m = analyze(run, scenario)
        cfg = out.setdefault(run["config"], {k: [] for k, _, _ in METRICS})
        for k, _, _ in METRICS:
            cfg[k].append(m[k])
    return out


def fmt(values: list[float], digits: int) -> str:
    med = statistics.median(values)
    if min(values) == max(values):
        return f"{med:.{digits}f}"
    return f"{med:.{digits}f} ({min(values):.{digits}f}-{max(values):.{digits}f})"


def table(data: dict[str, Any]) -> str:
    summary = summarize(data)
    head = "| config | " + " | ".join(label for _, label, _ in METRICS) + " |"
    lines = [head, "|" + "---|" * (len(METRICS) + 1)]
    for cfg, metrics in summary.items():
        cells = [fmt(metrics[k], d) for k, _, d in METRICS]
        lines.append(f"| {cfg} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def report(data: dict[str, Any], source: str) -> str:
    s = Scenario(**data["scenario"])
    meta = data["meta"]
    repeats = 1 + max(r["repeat"] for r in data["runs"])
    return "\n".join(
        [
            "# Stall micro-benchmark",
            "",
            f"Generated by `python -m benchmarks.microbench` from `{source}`; do not edit by hand.",
            "",
            "The engine is driven in-process (no HTTP server, no async loop), so this isolates the "
            "scheduling policy; the end-to-end comparison over HTTP is a separate benchmark.",
            "",
            f"- Scenario: {s.num_decodes} decodes ({s.decode_prompt_len}-token prompts) each "
            f"{s.warmup_tokens} tokens in, then one {s.long_prompt_len}-token prompt arrives; "
            f"window = until the long request has {s.measured_tokens} tokens and every decode "
            f"{s.measured_tokens} more. Greedy, `ignore_eos`, random prompt ids (seed {s.seed}).",
            f"- GPU: {meta['gpu']}; model {meta['model']} in {meta['dtype']}, "
            f"{meta['attention']} attention, block size {meta['block_size']}, "
            f"{meta['num_blocks']} KV blocks.",
            f"- Versions: torch {meta['torch']}, flash-attn {meta['flash_attn']}; "
            f"commit `{meta['commit']}`.",
            f"- {repeats} repeats per configuration after one discarded warm-up each, interleaved; "
            "cells are median (min-max) over repeats.",
            "- off = prefill priority (the long prompt runs alone in one step); on-N = mixed "
            "batching with an N-token budget per step.",
            "- ITL is over the decodes only, from the gap that spans the arrival onward; p99 is "
            "nearest rank over all decodes' gaps.",
            "",
            table(data),
            "",
        ]
    )


# --------------------------------------------------------------------------------------------
# Plots

INK, MUTED, GRID, SERIES = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"


def plot(data: dict[str, Any], out: Path, stem: str) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
        }
    )
    first = [r for r in data["runs"] if r["repeat"] == 0]
    gpu = data["meta"]["gpu"]

    # 1. Every decode's inter-token gaps over time, one panel per configuration (repeat 0).
    fig, axes = plt.subplots(1, len(first), figsize=(3.0 * len(first), 3.0), sharey=True)
    for ax, run in zip(axes, first, strict=True):
        xs, ys = [], []
        for ts in run["decode_token_times"]:
            for a, b in itertools.pairwise(ts):
                if b > 0:
                    xs.append(1e3 * b)
                    ys.append(1e3 * (b - a))
        ax.scatter(xs, ys, s=8, color=SERIES, linewidths=0)
        ttft = 1e3 * run["long_token_times"][0]
        ax.axvline(ttft, color=MUTED, linestyle="--", linewidth=1)
        ax.annotate(
            "long prompt's\nfirst token",
            (ttft, 0.97),
            xycoords=("data", "axes fraction"),
            xytext=(4, 0),
            textcoords="offset points",
            va="top",
            color=MUTED,
            fontsize=8,
        )
        ax.set_title(run["config"], color=INK)
        ax.set_xlabel("time since the long prompt arrived (ms)")
    axes[0].set_ylabel("gap since the decode's previous token (ms)")
    axes[0].set_ylim(bottom=0)
    fig.suptitle(f"Inter-token gaps of the {len(first[0]['decode_token_times'])} decodes ({gpu})")
    fig.tight_layout()
    timeline = out / f"{stem}_itl_timeline.png"
    fig.savefig(timeline, dpi=150)
    plt.close(fig)

    # 2. Max ITL and long-request TTFT per configuration: bar = median, dots = repeats.
    summary = summarize(data)
    names = list(summary)
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.0))
    panels = [
        ("itl_max_ms", "max ITL of the decodes (ms)"),
        ("long_ttft_ms", "long request's TTFT (ms)"),
    ]
    for ax, (key, label) in zip(axes, panels, strict=True):
        meds = [statistics.median(summary[n][key]) for n in names]
        ax.bar(names, meds, width=0.6, color=SERIES)
        for i, n in enumerate(names):
            ax.scatter([i] * len(summary[n][key]), summary[n][key], s=10, color=INK, zorder=3)
            ax.annotate(
                f"{meds[i]:.0f}",
                (i, meds[i]),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                color=INK,
            )
        ax.set_title(label, color=INK)
        ax.grid(axis="x", visible=False)
    fig.suptitle(f"Median of repeats (bars) and each repeat (dots), {gpu}")
    fig.tight_layout()
    bars = out / f"{stem}_summary.png"
    fig.savefig(bars, dpi=150)
    plt.close(fig)
    return [timeline, bars]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--out", type=Path, default=Path("benchmarks/results"))
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    files = sorted(args.data_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"no *.json in {args.data_dir}")
    parts = []
    for path in files:
        data = json.loads(path.read_text())
        parts.append(report(data, str(path)))
        for p in plot(data, args.out, path.stem):
            print(f"wrote {p}")
    md = args.out / "microbench.md"
    md.write_text("\n".join(parts))
    print(f"wrote {md}")


if __name__ == "__main__":
    main()
