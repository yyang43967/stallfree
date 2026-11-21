"""Where the stallfree server's CPU time goes, from a py-spy profile in raw (collapsed) format.

    python -m benchmarks.profile_summary \\
        benchmarks/data/profile/stallfree-on/r0.8x/seed1.pyspy.txt.gz \\
        --out benchmarks/results/profile.md

Each sample is one thread at one instant (``rate`` samples a second per thread). Three things are
decided per sample:

- **State.** In the benchmark sandbox py-spy also reports threads that are blocked, so the
  innermost native frames decide: ``epoll_wait`` is the event loop waiting for I/O;
  ``pthread_cond_*wait``, a Python lock acquire, or pybind11's GIL acquire is a thread waiting
  for the GIL or a lock; anything else is running. Running samples / (rate x duration) is CPU time
  in units of one core.
- **Component.** The innermost stallfree frame decides (a PyTorch op called from
  ``stallfree/model/`` counts as model forward), ignoring the server's entry point; samples with
  no stallfree frame are assigned by library (uvicorn/starlette/h11 to HTTP, asyncio to the event
  loop).
- **Native library** (model forward only): the innermost frame's library, to separate Python and
  PyTorch dispatch from the CUDA driver call that launches kernels.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

RUNNING, LOCK_WAIT, IO_WAIT = "running", "waiting for GIL or lock", "waiting for I/O"
TAIL = 8  # innermost frames inspected for the state

IO_WAIT_RE = re.compile(r"^(epoll_wait|epoll_pwait|poll|select|ppoll)\b")
LOCK_WAIT_RE = re.compile(
    r"^(pthread_cond_timedwait|pthread_cond_wait|sem_wait|sem_timedwait|"
    r"PyThread_acquire_lock_timed|pybind11::gil_scoped_acquire|PyEval_RestoreThread|take_gil)"
)

# (path fragment, function-name prefix or None, component); first match wins.
STALLFREE_RULES: list[tuple[str, str | None, str]] = [
    ("stallfree/kv_cache.py", "slots_for_positions", "batch metadata"),
    ("stallfree/attention/metadata.py", None, "batch metadata"),
    ("stallfree/model/", None, "model forward"),
    ("stallfree/attention/", None, "model forward"),
    ("stallfree/kv_cache.py", None, "model forward"),
    ("stallfree/model_runner.py", None, "batch metadata"),
    ("stallfree/sampler.py", None, "sampling"),
    ("stallfree/scheduler.py", None, "scheduler and block manager"),
    ("stallfree/block_manager.py", None, "scheduler and block manager"),
    ("stallfree/sequence.py", None, "scheduler and block manager"),
    ("stallfree/tokenizer.py", "HFChatTokenizer.encode_chat", "tokenization (chat template)"),
    ("stallfree/tokenizer.py", "HFChatTokenizer.render", "tokenization (chat template)"),
    ("stallfree/tokenizer.py", None, "detokenization"),
    ("stallfree/server/", None, "HTTP and SSE"),
    ("stallfree/async_engine.py", None, "engine loop (async side)"),
    ("stallfree/engine.py", None, "engine step (other)"),
]
# The server's entry point is at the root of every main-thread stack; it says nothing.
ENTRY_FRAMES = ("main (", "<module> (")
ENTRY_FILES = ("stallfree/server/api.py", "benchmarks/ptraceable.py")
LIBRARY_RULES: list[tuple[str, str]] = [
    ("/uvicorn/", "HTTP and SSE"),
    ("/starlette/", "HTTP and SSE"),
    ("/fastapi/", "HTTP and SSE"),
    ("/h11/", "HTTP and SSE"),
    ("/httptools/", "HTTP and SSE"),
    ("/pydantic", "HTTP and SSE"),
    ("/anyio/", "HTTP and SSE"),
    ("/asyncio/", "event loop"),
    ("_asyncio.cpython", "event loop"),
    ("/selectors.py", "event loop"),
    ("/concurrent/futures/", "thread handoff"),
    ("/threading.py", "thread handoff"),
    ("/queue.py", "thread handoff"),
]
COMPONENT_ORDER = [
    "model forward",
    "batch metadata",
    "sampling",
    "scheduler and block manager",
    "engine step (other)",
    "tokenization (chat template)",
    "detokenization",
    "HTTP and SSE",
    "engine loop (async side)",
    "event loop",
    "thread handoff",
    "other",
]
# Innermost frame of a model-forward sample -> where the time is spent.
LIBRARY_GROUPS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"libcuda\.so|libcudart\.so"), "CUDA driver/runtime (kernel launch)"),
    (re.compile(r"libcublas"), "cuBLAS host code"),
    (re.compile(r"libtorch|libc10|_C\.cpython|flash_attn"), "PyTorch / extension C++ dispatch"),
    (re.compile(r"libc\.so|libstdc\+\+|ld-linux|libm\.so"), "C library (malloc, memcpy, locks)"),
    (re.compile(r"\.py:\d+\)$|<frozen"), "Python code (model and torch.nn)"),
]
LAUNCH = re.compile(r"cudaLaunchKernel|cuLaunchKernel")


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
        out.append(Sample(parts[0], parts[1:], int(count)))
    return out


def read_profile(path: Path) -> list[Sample]:
    raw = path.read_bytes()
    text = gzip.decompress(raw).decode() if path.suffix == ".gz" else raw.decode()
    return parse(text.splitlines())


def thread_name(label: str) -> str:
    """``thread (0x7f..): MainThread`` -> ``MainThread``; unnamed threads keep their label."""
    return label.split(": ", 1)[1] if ": " in label else label


def _name(frame: str) -> str:
    return frame.split(" (", 1)[0]


def state(frames: list[str]) -> str:
    for frame in reversed(frames[-TAIL:]):
        name = _name(frame)
        if IO_WAIT_RE.match(name):
            return IO_WAIT
        if LOCK_WAIT_RE.match(name):
            return LOCK_WAIT
    return RUNNING


def _is_entry(frame: str) -> bool:
    return frame.startswith(ENTRY_FRAMES) and any(f in frame for f in ENTRY_FILES)


def component(frames: list[str]) -> str:
    for frame in reversed(frames):  # innermost first
        if "stallfree/" not in frame or _is_entry(frame):
            continue
        for path, func, name in STALLFREE_RULES:
            if path in frame and (func is None or frame.startswith(func)):
                return name
    for frame in reversed(frames):
        for marker, name in LIBRARY_RULES:
            if marker in frame:
                return name
    return "other"


def library_group(frames: list[str]) -> str:
    leaf = frames[-1] if frames else ""
    for pattern, name in LIBRARY_GROUPS:
        if pattern.search(leaf):
            return name
    return "Python interpreter (unsymbolized)" if leaf.startswith("0x") else "other"


def innermost_stallfree(frames: list[str]) -> str | None:
    for frame in reversed(frames):
        if "stallfree/" in frame and not _is_entry(frame):
            func, _, loc = frame.partition(" (")
            path = loc.rstrip(")")
            return f"{func} ({path[path.index('stallfree/') :]})"
    return None


@dataclass
class Summary:
    total: int = 0
    native: bool = False
    by_state: Counter[str] = field(default_factory=Counter)
    comp_state: Counter[tuple[str, str]] = field(default_factory=Counter)
    launch: Counter[str] = field(default_factory=Counter)
    thread_state: Counter[tuple[str, str]] = field(default_factory=Counter)
    thread_comp: Counter[tuple[str, str]] = field(default_factory=Counter)
    forward_libs: Counter[str] = field(default_factory=Counter)
    functions: Counter[str] = field(default_factory=Counter)


def summarize(samples: list[Sample]) -> Summary:
    s = Summary()
    for smp in samples:
        st = state(smp.frames)
        comp = component(smp.frames)
        name = thread_name(smp.thread)
        n = smp.count
        s.total += n
        s.by_state[st] += n
        s.comp_state[(comp, st)] += n
        s.thread_state[(name, st)] += n
        if any(".so" in f for f in smp.frames):
            s.native = True
        if st != RUNNING:
            continue
        s.thread_comp[(name, comp)] += n
        if LAUNCH.search(";".join(smp.frames[-TAIL:])):
            s.launch[comp] += n
        if comp == "model forward":
            s.forward_libs[library_group(smp.frames)] += n
        fn = innermost_stallfree(smp.frames)
        if fn:
            s.functions[fn] += n
    return s


def _pct(n: int, d: int) -> str:
    return f"{100 * n / d:.1f}%" if d else "-"


def report(s: Summary, source: str, rate: int, duration_s: float) -> str:
    ticks = rate * duration_s
    running = s.by_state[RUNNING]
    lines = [
        "# stallfree server CPU profile",
        "",
        f"Generated by `python -m benchmarks.profile_summary` from `{source}`; "
        "do not edit by hand.",
        "",
        f"py-spy at {rate} Hz for {duration_s:.0f} s on the stallfree server process, every "
        "thread" + (", with native frames" if s.native else ", Python frames only") + ". "
        f"{s.total} samples: {running} running ({running / ticks:.2f} cores on average), "
        f"{s.by_state[LOCK_WAIT]} waiting for the GIL or a lock, "
        f"{s.by_state[IO_WAIT]} waiting for I/O.",
        "",
        "## CPU time by component (running samples)",
        "",
        "| component | running samples | share of CPU | cores | of which in CUDA launch call "
        "| samples waiting for GIL or lock |",
        "|---|---|---|---|---|---|",
    ]
    for comp in COMPONENT_ORDER:
        n = s.comp_state[(comp, RUNNING)]
        waits = s.comp_state[(comp, LOCK_WAIT)]
        if not n and not waits:
            continue
        launch = _pct(s.launch[comp], n) if s.native else "-"
        lines.append(
            f"| {comp} | {n} | {_pct(n, running)} | {n / ticks:.2f} | {launch} | {waits} |"
        )
    lines += [
        "",
        "## Model forward: where its CPU time is spent (innermost frame)",
        "",
        "| innermost frame | samples | share of model forward |",
        "|---|---|---|",
    ]
    fwd = sum(s.forward_libs.values())
    for lib, n in s.forward_libs.most_common():
        lines.append(f"| {lib} | {n} | {_pct(n, fwd)} |")
    lines += [
        "",
        "## By thread",
        "",
        "| thread | running | waiting for GIL or lock | waiting for I/O "
        "| main running components |",
        "|---|---|---|---|---|",
    ]
    running_by_thread = Counter({t: k for (t, st), k in s.thread_state.items() if st == RUNNING})
    for t in sorted({t for t, _ in s.thread_state}, key=lambda t: -running_by_thread[t]):
        comps = Counter({c: k for (th, c), k in s.thread_comp.items() if th == t})
        top = ", ".join(f"{c} {_pct(k, running_by_thread[t])}" for c, k in comps.most_common(3))
        lines.append(
            f"| {t} | {s.thread_state[(t, RUNNING)]} | {s.thread_state[(t, LOCK_WAIT)]} | "
            f"{s.thread_state[(t, IO_WAIT)]} | {top} |"
        )
    lines += ["", "## Innermost stallfree functions, running samples (top 20)", ""]
    lines += ["| function | samples | share of CPU |", "|---|---|---|"]
    for fn, n in s.functions.most_common(20):
        lines.append(f"| `{fn}` | {n} | {_pct(n, running)} |")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("profile", type=Path, help="py-spy raw output (<stem>.pyspy.txt[.gz])")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    stem = a.profile.name.split(".pyspy.txt")[0]
    spec = json.loads((a.profile.parent / f"{stem}.run.json").read_text())["profile"]
    s = summarize(read_profile(a.profile))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(report(s, str(a.profile), int(spec["rate"]), float(spec["duration_s"])))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
