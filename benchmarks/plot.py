"""Plots of the sweep, generated from the raw data (never edited by hand).

    python -m benchmarks.plot benchmarks/data --out benchmarks/plots

One PNG per metric against the offered rate: TTFT, TPOT, and ITL (p50 solid and p99 dashed, or
p99 and max for ITL), and output-token throughput. Each point is the median across repeats with
a min-max bar; saturated points are drawn hollow. Latency axes are logarithmic because the
systems differ by more than an order of magnitude.
"""

from __future__ import annotations

import argparse
import statistics
from collections.abc import Callable
from pathlib import Path
from typing import Any

from benchmarks.summarize import SYSTEM_ORDER, collect

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
# Categorical slots 1-4 (blue, orange, aqua, yellow), checked for colour-vision separation;
# direct labels and distinct markers carry identity as well.
COLORS = {
    "stallfree-off": "#2a78d6",
    "stallfree-on": "#eb6834",
    "vllm": "#1baf7a",
    "vllm-eager": "#eda100",
}
MARKERS = {"stallfree-off": "o", "stallfree-on": "s", "vllm": "^", "vllm-eager": "D"}

Getter = Callable[[dict[str, Any]], float | None]
PLOTS: list[tuple[str, str, list[tuple[str, str, Getter]], bool]] = [
    (
        "ttft",
        "Time to first token (ms)",
        [
            ("p50", "-", lambda m: 1000 * m["ttft"]["p50"]),
            ("p99", "--", lambda m: 1000 * m["ttft"]["p99"]),
        ],
        True,
    ),
    (
        "tpot",
        "Time per output token (ms)",
        [
            ("p50", "-", lambda m: 1000 * m["tpot"]["p50"]),
            ("p99", "--", lambda m: 1000 * m["tpot"]["p99"]),
        ],
        True,
    ),
    (
        "itl",
        "Inter-token latency (ms)",
        [
            ("p99", "-", lambda m: 1000 * m["itl"]["p99"]),
            ("max", "--", lambda m: 1000 * m["itl"]["max"]),
        ],
        True,
    ),
    (
        "throughput",
        "Output tokens per second",
        [("", "-", lambda m: m["output_token_throughput"])],
        False,
    ),
]


def series(
    groups: dict[tuple[str, str], list[dict[str, Any]]], system: str, get: Getter
) -> list[tuple[float, float, float, float, bool]]:
    """(rate, median, min, max, any repeat saturated) per rate, sorted by rate."""
    out = []
    for (sysname, _), runs in groups.items():
        if sysname != system:
            continue
        vals = [v for v in (get(r) for r in runs) if v is not None]
        if not vals:
            continue
        rate = statistics.median(r["offered_rate"] for r in runs)
        sat = any(r.get("saturated") for r in runs)
        out.append((rate, statistics.median(vals), min(vals), max(vals), sat))
    return sorted(out)


LABEL_GAP_PT = 11.0  # minimum vertical distance between direct labels, in points


def _label_ends(ax: Any, ends: list[tuple[str, float, float]], color: str) -> None:
    """Direct labels at the line ends, pushed apart vertically so none overlap."""
    if not ends:
        return
    fig = ax.figure
    fig.canvas.draw()  # fixes the axis limits and scales before converting to display units
    to_px = ax.transData.transform
    gap = LABEL_GAP_PT * fig.dpi / 72
    placed = sorted(((to_px((x, y))[1], name, x, y) for name, x, y in ends))
    last = -float("inf")
    for py, name, x, y in placed:
        target = max(py, last + gap)
        last = target
        ax.annotate(
            name,
            (x, y),
            xytext=(6, (target - py) * 72 / fig.dpi),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=color,
        )


def plot(groups: dict[tuple[str, str], list[dict[str, Any]]], out: Path) -> list[Path]:
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
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
        }
    )
    systems = [s for s in SYSTEM_ORDER if any(k[0] == s for k in groups)]
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, ylabel, lines, log in PLOTS:
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        ends: list[tuple[str, float, float]] = []
        for system in systems:
            color = COLORS.get(system, MUTED)
            for stat, style, get in lines:
                pts = series(groups, system, get)
                if not pts:
                    continue
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                lo = [p[1] - p[2] for p in pts]
                hi = [p[3] - p[1] for p in pts]
                label = f"{system} {stat}".strip()
                ax.errorbar(
                    xs,
                    ys,
                    yerr=[lo, hi],
                    color=color,
                    linestyle=style,
                    linewidth=2,
                    elinewidth=1,
                    capsize=2,
                    label=label,
                )
                for x, y, *_, sat in pts:
                    ax.plot(
                        x,
                        y,
                        marker=MARKERS.get(system, "o"),
                        markersize=8,
                        color=color,
                        markerfacecolor=SURFACE if sat else color,
                        markeredgewidth=1.5,
                        linestyle="none",
                    )
                if style == "-":  # direct label at the right end of the solid line
                    ends.append((system, xs[-1], ys[-1]))
        if log:
            ax.set_yscale("log")
        _label_ends(ax, ends, INK)
        ax.set_xlabel("Offered rate (requests/s)")
        ax.set_ylabel(ylabel)
        ax.margins(x=0.15)
        ax.legend(fontsize=7, frameon=False, ncol=2, loc="best")
        ax.set_title(f"{ylabel} vs rate (hollow = saturated)", color=INK, fontsize=10)
        fig.tight_layout()
        path = out / f"{name}_vs_rate.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("data", type=Path)
    p.add_argument("--group", default="")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    root = a.data / a.group if a.group else a.data
    for path in plot(collect(root), a.out):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
