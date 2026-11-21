"""Tables from the raw benchmark data: one row per (system, rate), median across repeats with
min-max, plus every run's sanity checks and CPU use.

    python -m benchmarks.summarize benchmarks/data --group dryrun --out benchmarks/results
    python -m benchmarks.summarize benchmarks/data --out benchmarks/results    # the full sweep

Reads ``<data>/[<group>/]<system>/<label>/seed<k>.jsonl.gz`` and the files next to each.
"""

from __future__ import annotations

import argparse
import statistics
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from benchmarks.analyze import LOADGEN_CPU_FLAG_PCT, load_run, run_metrics

SYSTEM_ORDER = ["stallfree-off", "stallfree-on", "vllm", "vllm-eager"]


def find_runs(root: Path) -> list[Path]:
    """Run stems (``.../seed<k>.jsonl``) under ``root``, skipping other groups' directories."""
    stems = []
    for gz in sorted(root.glob("*/*/seed*.jsonl.gz")):
        stems.append(gz.with_name(gz.name.removesuffix(".gz")))
    return stems


def collect(root: Path) -> dict[tuple[str, str], list[dict[str, Any]]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for stem in find_runs(root):
        run = load_run(stem)
        m = run_metrics(run)
        meta = run["run"]
        m["seed"] = meta.get("trace_seed")
        m["rate_multiplier"] = meta.get("rate_multiplier")
        m["driver"] = (meta.get("gpu_before") or {}).get("driver_version")
        m["gpu"] = (meta.get("gpu_before") or {}).get("name")
        m["trace_sha256"] = meta.get("trace_sha256")
        m["text_sha256"] = meta.get("text_sha256")
        m["commit"] = meta.get("commit")
        system, label = stem.parent.parent.name, stem.parent.name
        groups.setdefault((system, label), []).append(m)
    return groups


def _ms(x: float | None) -> float | None:
    return None if x is None else 1000 * x


def _agg(runs: Sequence[dict[str, Any]], get: Callable[[dict[str, Any]], float | None]) -> str:
    vals = [v for v in (get(r) for r in runs) if v is not None]
    if not vals:
        return "-"
    med = statistics.median(vals)
    if len(vals) == 1:
        return f"{med:.1f}"
    return f"{med:.1f} ({min(vals):.1f}-{max(vals):.1f})"


def _order(key: tuple[str, str]) -> tuple[int, float, str]:
    system, label = key
    rank = SYSTEM_ORDER.index(system) if system in SYSTEM_ORDER else len(SYSTEM_ORDER)
    try:
        mult = float(label.removeprefix("r").removesuffix("x"))
    except ValueError:
        mult = 0.0
    return rank, mult, label


LATENCY_COLUMNS: list[tuple[str, Callable[[dict[str, Any]], float | None]]] = [
    ("TTFT p50 (ms)", lambda m: _ms(m["ttft"]["p50"])),
    ("TTFT p99 (ms)", lambda m: _ms(m["ttft"]["p99"])),
    ("TPOT p50 (ms)", lambda m: _ms(m["tpot"]["p50"])),
    ("TPOT p99 (ms)", lambda m: _ms(m["tpot"]["p99"])),
    ("ITL p99 (ms)", lambda m: _ms(m["itl"]["p99"])),
    ("ITL max (ms)", lambda m: _ms(m["itl"]["max"])),
    ("E2E p50 (s)", lambda m: m["e2e"]["p50"]),
    ("output tok/s", lambda m: m["output_token_throughput"]),
]


def latency_table(groups: dict[tuple[str, str], list[dict[str, Any]]]) -> str:
    head = ["system", "rate (req/s)", "repeats", *(c for c, _ in LATENCY_COLUMNS), "saturated"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for key in sorted(groups, key=_order):
        runs = groups[key]
        rate = statistics.median(r["offered_rate"] for r in runs)
        sat = sum(1 for r in runs if r.get("saturated"))
        cells = [key[0], f"{rate:.2f} ({key[1].removeprefix('r')})", str(len(runs))]
        cells += [_agg(runs, get) for _, get in LATENCY_COLUMNS]
        cells.append(f"{sat}/{len(runs)}" if sat else "no")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def checks_table(groups: dict[tuple[str, str], list[dict[str, Any]]]) -> str:
    head = [
        "system",
        "rate",
        "seed",
        "measured",
        "errors",
        "prompt-token mismatches",
        "ignore_eos violations",
        "tokens/chunk",
        "load-gen CPU mean/max (%)",
        "server CPU mean/max (%)",
        "GPU util (%)",
        "driver",
    ]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for key in sorted(groups, key=_order):
        for m in sorted(groups[key], key=lambda r: r["seed"] or 0):
            c = m.get("cpu") or {}
            flag = " **flagged**" if c.get("loadgen_flag") else ""
            lines.append(
                "| "
                + " | ".join(
                    [
                        key[0],
                        key[1].removeprefix("r"),
                        str(m["seed"]),
                        str(m["measured"]),
                        str(m["errors"]),
                        str(m["prompt_token_mismatches"]),
                        str(m["ignore_eos_violations"]),
                        f"{m['tokens_per_chunk']:.4f}" if m["tokens_per_chunk"] else "-",
                        f"{c.get('loadgen_mean', 0):.0f}/{c.get('loadgen_max', 0):.0f}{flag}",
                        f"{c.get('server_mean', 0):.0f}/{c.get('server_max', 0):.0f}",
                        f"{c.get('gpu_util_mean', 0):.0f}" if c.get("gpu_util_mean") else "-",
                        str(m["driver"]),
                    ]
                )
                + " |"
            )
    return "\n".join(lines)


def report(groups: dict[tuple[str, str], list[dict[str, Any]]], title: str, source: str) -> str:
    runs = [m for rs in groups.values() for m in rs]
    hashes = sorted({(m["seed"], m["trace_sha256"]) for m in runs}, key=lambda x: x[0] or 0)
    texts = sorted({m["text_sha256"] for m in runs if m["text_sha256"]})
    gpus = sorted({m["gpu"] for m in runs if m["gpu"]})
    return "\n".join(
        [
            f"# {title}",
            "",
            f"Generated by `python -m benchmarks.summarize` from `{source}`; do not edit by hand.",
            "",
            f"GPU: {', '.join(gpus)}. Source text SHA-256: {', '.join(texts)}.",
            "Trace SHA-256 per seed (identical for every system): "
            + ", ".join(f"seed {s}: `{(h or '-')[:16]}`" for s, h in hashes)
            + ".",
            "",
            "Client-side metrics over the measured requests (warm-up excluded). Cells are the "
            "median across repeats with (min-max) when there is more than one. A point is "
            "labelled saturated when TTFT keeps growing through the run (see "
            "`benchmarks/analyze.py`).",
            "",
            latency_table(groups),
            "",
            "## Sanity checks and CPU per run",
            "",
            "CPU in percent of one core, inside the measured window. A run is flagged when the "
            f"load generator averages more than {LOADGEN_CPU_FLAG_PCT:.0f}%.",
            "",
            checks_table(groups),
            "",
        ]
    )


BUDGET_COLUMNS: list[tuple[str, Callable[[dict[str, Any]], float | None]]] = [
    ("TTFT p50 (ms)", lambda m: _ms(m["ttft"]["p50"])),
    ("TTFT p99 (ms)", lambda m: _ms(m["ttft"]["p99"])),
    ("TPOT p50 (ms)", lambda m: _ms(m["tpot"]["p50"])),
    ("ITL p99 (ms)", lambda m: _ms(m["itl"]["p99"])),
    ("ITL max (ms)", lambda m: _ms(m["itl"]["max"])),
    ("E2E p50 (s)", lambda m: m["e2e"]["p50"]),
]


def budget_table(data: Path, label: str = "r0.8x") -> str | None:
    """stallfree-on at one rate with each token budget: the budget runs plus the 512-token
    runs of the main sweep at the same rate and seeds."""
    rows: dict[int, list[dict[str, Any]]] = {}
    budget_root = data / "budget"
    if not budget_root.exists():
        return None
    for (system, lab), runs in collect(budget_root).items():
        if lab == label:
            rows[int(system.rsplit("-b", 1)[1])] = runs
    main = collect(data).get(("stallfree-on", label))
    if main:
        rows[512] = main
    head = ["budget (tokens)", "repeats", *(c for c, _ in BUDGET_COLUMNS)]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for budget in sorted(rows):
        runs = rows[budget]
        cells = [str(budget), str(len(runs))] + [_agg(runs, g) for _, g in BUDGET_COLUMNS]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def saturation_table(data: Path) -> str | None:
    """Closed-loop saturation throughput of each system that has a saturation file."""
    import json

    files = sorted((data / "saturation").glob("saturation*.json"))
    if not files:
        return None
    lines = [
        "| system | concurrency | requests/s | output tokens/s | requests finished in window |",
        "|---|---|---|---|---|",
    ]
    for f in files:
        d = json.loads(f.read_text())
        lines.append(
            f"| {d['system']} | {d['concurrency']} | {d['request_throughput']:.2f} | "
            f"{d['output_token_throughput_completed']:.0f} | {d['completed_in_window']} |"
        )
    return "\n".join(lines)


def host_table(groups: dict[tuple[str, str], list[dict[str, Any]]]) -> str:
    """TPOT p50 and TTFT p50 per NVIDIA driver version (a marker of the host type Modal chose),
    median over the repeats that ran on that host type, with the count in brackets."""
    drivers = sorted({str(m["driver"]) for runs in groups.values() for m in runs})
    head = ["system", "rate"]
    for d in drivers:
        head += [f"TPOT p50 (ms), driver {d}", f"TTFT p50 (ms), driver {d}"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for key in sorted(groups, key=_order):
        cells = [key[0], key[1].removeprefix("r")]
        for d in drivers:
            runs = [m for m in groups[key] if str(m["driver"]) == d]
            for get in (lambda m: _ms(m["tpot"]["p50"]), lambda m: _ms(m["ttft"]["p50"])):
                cells.append(f"{_agg(runs, get)} [{len(runs)}]" if runs else "-")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("data", type=Path)
    p.add_argument("--group", default="", help="subdirectory such as dryrun; empty: the sweep")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    root = a.data / a.group if a.group else a.data
    groups = collect(root)
    if not groups:
        raise SystemExit(f"no runs under {root}")
    name = a.group or "sweep"
    title = "Benchmark dry run" if a.group == "dryrun" else "Benchmark results"
    a.out.mkdir(parents=True, exist_ok=True)
    out = a.out / f"{name}.md"
    text = report(groups, title, str(root))
    if not a.group:
        text += (
            "\n## By host type\n\nModal placed containers on two host types, told apart by "
            "their NVIDIA driver version. Repeats are split by host type here; [n] is the number "
            "of repeats on that host type.\n\n" + host_table(groups) + "\n"
        )
        sat = saturation_table(a.data)
        if sat:
            text += (
                "\n## Closed-loop saturation throughput\n\n256 concurrent clients, 30 s warm-up, "
                "requests finished in the next 120 s. stallfree-off's figure sets the sweep "
                "rates; the others are reported for reference and are not sweep points.\n\n"
                + sat
                + "\n"
            )
        budget = budget_table(a.data)
        if budget:
            text += (
                "\n## stallfree-on token budget at 0.8x saturation\n\nThe 512-token row is the "
                "main sweep's stallfree-on point; the others come from `budget/`. Same rate and "
                "seeds.\n\n" + budget + "\n"
            )
    out.write_text(text)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
