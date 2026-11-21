"""Where the stallfree server's CPU time goes, from a py-spy profile in raw (collapsed) format.

    python -m benchmarks.profile_summary \
        benchmarks/data/profile/stallfree-on/r0.8x/seed1.pyspy.txt \
        --out benchmarks/results/profile.md

py-spy samples every thread that is not idle, ``rate`` times a second, so the number of samples
divided by ``rate x duration`` is the CPU time in units of one core. Each sample is assigned to
one component: the innermost stallfree frame decides (for example, a PyTorch op called from
``stallfree/model/`` counts as model forward); samples with no stallfree frame are assigned by
library (uvicorn/starlette/h11 to HTTP, asyncio to the event loop). Native frames, when present,
show how much of each component was spent inside CUDA's kernel-launch call or waiting for the GPU.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

# Innermost stallfree file -> component. Checked in order; the first prefix that matches wins.
STALLFREE_COMPONENTS: list[tuple[str, str]] = [
    ("stallfree/model/", "model forward"),
    ("stallfree/attention/", "model forward"),
    ("stallfree/kv_cache.py", "model forward"),
    ("stallfree/model_runner.py", "batch metadata (runner inputs)"),
    ("stallfree/sampler.py", "sampling"),
    ("stallfree/scheduler.py", "scheduler and block manager"),
    ("stallfree/block_manager.py", "scheduler and block manager"),
    ("stallfree/sequence.py", "scheduler and block manager"),
    ("stallfree/tokenizer.py", "detokenization"),
    ("stallfree/server/", "HTTP and SSE"),
    ("stallfree/async_engine.py", "engine loop (async side)"),
    ("stallfree/engine.py", "engine step (other)"),
]
# Samples with no stallfree frame: by library.
LIBRARY_COMPONENTS: list[tuple[str, str]] = [
    ("/uvicorn/", "HTTP and SSE"),
    ("/starlette/", "HTTP and SSE"),
    ("/fastapi/", "HTTP and SSE"),
    ("/h11/", "HTTP and SSE"),
    ("/httptools/", "HTTP and SSE"),
    ("/pydantic", "HTTP and SSE"),
    ("/anyio/", "HTTP and SSE"),
    ("/asyncio/", "event loop"),
    ("/selectors.py", "event loop"),
    ("/concurrent/futures/", "thread handoff"),
    ("/threading.py", "thread handoff"),
]
COMPONENT_ORDER = [
    "model forward",
    "batch metadata (runner inputs)",
    "sampling",
    "scheduler and block manager",
    "engine step (other)",
    "detokenization",
    "HTTP and SSE",
    "engine loop (async side)",
    "event loop",
    "thread handoff",
    "other",
]
LAUNCH = re.compile(r"cudaLaunchKernel|cuLaunchKernel|cudaLaunchKernelExC|cuLaunchKernelEx")
GPU_WAIT = re.compile(
    r"cudaStreamSynchronize|cuStreamSynchronize|cudaDeviceSynchronize|cuCtxSynchronize|"
    r"cudaMemcpy|cuMemcpy|cudaEventSynchronize|cuEventSynchronize"
)
FRAME = re.compile(r"^(?P<func>.*?) \((?P<loc>[^()]*)\)$")


@dataclass
class Sample:
    thread: str
    frames: list[str]  # root first
    count: int


def parse(lines: Iterable[str]) -> list[Sample]:
    """Lines of ``thread (tid): name;frame;...;frame count``."""
    out = []
    for line in lines:
        line = line.rstrip("\n")
        if not line:
            continue
        stack, _, count = line.rpartition(" ")
        parts = stack.split(";")
        thread = parts[0]
        out.append(Sample(thread, parts[1:], int(count)))
    return out


def thread_name(label: str) -> str:
    """``thread (0x7f..): MainThread`` -> ``MainThread``; unnamed threads keep their label."""
    return label.split(": ", 1)[1] if ": " in label else label


def component(frames: list[str]) -> str:
    for frame in reversed(frames):  # innermost first
        m = FRAME.match(frame)
        loc = m.group("loc") if m else frame
        for prefix, name in STALLFREE_COMPONENTS:
            if prefix in loc:
                return name
    for frame in reversed(frames):
        for marker, name in LIBRARY_COMPONENTS:
            if marker in frame:
                return name
    return "other"


def innermost_stallfree(frames: list[str]) -> str | None:
    for frame in reversed(frames):
        if "stallfree/" in frame:
            func, _, loc = frame.partition(" (")
            path = loc.rstrip(")")
            return f"{func} ({path[path.index('stallfree/') :]})"
    return None


@dataclass
class Summary:
    total: int = 0
    native: bool = False
    by_component: Counter[str] = field(default_factory=Counter)
    launch: Counter[str] = field(default_factory=Counter)
    gpu_wait: Counter[str] = field(default_factory=Counter)
    by_thread: Counter[str] = field(default_factory=Counter)
    thread_component: Counter[tuple[str, str]] = field(default_factory=Counter)
    functions: Counter[str] = field(default_factory=Counter)


def summarize(samples: list[Sample]) -> Summary:
    s = Summary()
    for smp in samples:
        comp = component(smp.frames)
        stack = ";".join(smp.frames)
        name = thread_name(smp.thread)
        s.total += smp.count
        s.by_component[comp] += smp.count
        s.by_thread[name] += smp.count
        s.thread_component[(name, comp)] += smp.count
        if "(lib" in stack or ".so" in stack:
            s.native = True
        if LAUNCH.search(stack):
            s.launch[comp] += smp.count
        if GPU_WAIT.search(stack):
            s.gpu_wait[comp] += smp.count
        fn = innermost_stallfree(smp.frames)
        if fn:
            s.functions[fn] += smp.count
    return s


def _pct(n: int, d: int) -> str:
    return f"{100 * n / d:.1f}%" if d else "-"


def report(s: Summary, source: str, rate: int, duration_s: float) -> str:
    ticks = rate * duration_s
    lines = [
        "# stallfree server CPU profile",
        "",
        f"Generated by `python -m benchmarks.profile_summary` from `{source}`; "
        "do not edit by hand.",
        "",
        f"py-spy, {rate} Hz for {duration_s:.0f} s, all non-idle threads"
        + (", with native frames" if s.native else ", Python frames only")
        + f". {s.total} samples = {s.total / ticks:.2f} cores busy on average "
        "(samples / (rate x duration)).",
        "",
        "## By component",
        "",
        "| component | samples | share of samples | cores | of which in CUDA kernel launch "
        "| of which waiting for the GPU |",
        "|---|---|---|---|---|---|",
    ]
    for comp in [c for c in COMPONENT_ORDER if s.by_component[c]]:
        n = s.by_component[comp]
        lines.append(
            f"| {comp} | {n} | {_pct(n, s.total)} | {n / ticks:.2f} | "
            f"{_pct(s.launch[comp], n) if s.native else '-'} | "
            f"{_pct(s.gpu_wait[comp], n) if s.native else '-'} |"
        )
    lines += ["", "## By thread", "", "| thread | samples | cores | main components |"]
    lines.append("|---|---|---|---|")
    for name, n in s.by_thread.most_common():
        comps = Counter({c: k for (t, c), k in s.thread_component.items() if t == name})
        top = ", ".join(f"{c} {_pct(k, n)}" for c, k in comps.most_common(3))
        lines.append(f"| {name} | {n} | {n / ticks:.2f} | {top} |")
    lines += ["", "## Innermost stallfree functions (top 20)", ""]
    lines += ["| function | samples | share |", "|---|---|---|"]
    for fn, n in s.functions.most_common(20):
        lines.append(f"| `{fn}` | {n} | {_pct(n, s.total)} |")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("profile", type=Path, help="py-spy raw output (<stem>.pyspy.txt)")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    run = json.loads(
        a.profile.with_name(a.profile.name.replace(".pyspy.txt", ".run.json")).read_text()
    )
    spec = run["profile"]
    s = summarize(parse(a.profile.read_text().splitlines()))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(report(s, str(a.profile), int(spec["rate"]), float(spec["duration_s"])))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
