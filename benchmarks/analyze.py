"""Metrics for one benchmark run, computed from the raw files the driver writes.

Client-side, from SSE timestamps, over the measured requests only (warm-up excluded):
- TTFT: first chunk with generated text minus the time the request was sent.
- TPOT: (last chunk - first chunk) / (completion tokens - 1), per request.
- ITL: gaps between consecutive text chunks, pooled over all measured requests; p99 and max.
- E2E: last event minus send time.
- Throughput: measured requests and their completion tokens divided by the measured window
  (first measured send to last measured end).

Sanity checks per run: the server's prompt token count equals the trace's; every request
generated exactly ``max_tokens`` tokens (``ignore_eos`` honoured); no errors; how many tokens
each text chunk carried; and the load generator's CPU use (flagged above 80% of one core).
"""

from __future__ import annotations

import gzip
import json
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from benchmarks.microbench import percentile

LOADGEN_CPU_FLAG_PCT = 80.0
# An open-loop point is labelled saturated when the queue keeps growing during the run: the
# median TTFT of the last fifth of measured requests (in send order) is more than this many
# times that of the first fifth. Below saturation TTFT is stationary and the ratio stays near 1.
TTFT_GROWTH_SATURATED = 2.0


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """JSON lines from ``path``, or from ``path.gz`` when only the gzipped file exists."""
    if path.exists():
        text = path.read_text()
    else:
        text = gzip.decompress(path.with_name(path.name + ".gz").read_bytes()).decode()
    return [json.loads(line) for line in text.splitlines() if line]


def load_run(jsonl: Path) -> dict[str, Any]:
    """All files of the run whose records are ``<stem>.jsonl`` (or ``<stem>.jsonl.gz``)."""
    run_json = jsonl.with_suffix(".run.json")
    cpu = jsonl.with_suffix(".cpu.jsonl")
    return {
        "records": read_jsonl(jsonl),
        "summary": json.loads(jsonl.with_suffix(".summary.json").read_text()),
        "cpu": read_jsonl(cpu) if cpu.exists() else [],
        "run": json.loads(run_json.read_text()) if run_json.exists() else {},
    }


def _ok(r: dict[str, Any]) -> bool:
    return r["error"] is None and r["status"] == 200 and len(r["chunks"]) > 0


def _stats(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"p50": None, "p99": None, "mean": None, "max": None}
    return {
        "p50": percentile(values, 50),
        "p99": percentile(values, 99),
        "mean": statistics.fmean(values),
        "max": max(values),
    }


def cpu_stats(
    samples: Sequence[dict[str, Any]], start_wall: float, end_wall: float
) -> dict[str, Any]:
    """Mean and max CPU% (100 = one core) of the server tree and the load generator inside the
    measured window, and the per-process breakdown of the server tree (mean)."""
    inside = [s for s in samples if start_wall <= s["wall"] <= end_wall]
    lg = [s["loadgen_pct"] for s in inside if s.get("loadgen_pct") is not None]
    srv = [s["server_pct"] for s in inside]
    per_proc: dict[str, list[float]] = {}
    for s in inside:
        seen: dict[str, float] = {}
        for _pid, name, pct in s["procs"]:
            seen[name] = seen.get(name, 0.0) + pct
        for name, pct in seen.items():
            per_proc.setdefault(name, []).append(pct)
    gpu = [float(s["gpu_util"]) for s in inside if s.get("gpu_util") not in (None, "")]
    return {
        "samples": len(inside),
        "loadgen_mean": statistics.fmean(lg) if lg else None,
        "loadgen_max": max(lg) if lg else None,
        "server_mean": statistics.fmean(srv) if srv else None,
        "server_max": max(srv) if srv else None,
        "server_procs_mean": {k: statistics.fmean(v) for k, v in sorted(per_proc.items())},
        "gpu_util_mean": statistics.fmean(gpu) if gpu else None,
        "loadgen_flag": bool(lg) and statistics.fmean(lg) > LOADGEN_CPU_FLAG_PCT,
    }


def ttft_growth(ok: Sequence[dict[str, Any]]) -> float | None:
    """Median TTFT of the last fifth of requests (by send time) over that of the first fifth."""
    ordered = sorted(ok, key=lambda r: r["sent"])
    k = len(ordered) // 5
    if k == 0:
        return None
    first = statistics.median(r["chunks"][0] - r["sent"] for r in ordered[:k])
    last = statistics.median(r["chunks"][0] - r["sent"] for r in ordered[-k:])
    return last / first if first > 0 else None


def run_metrics(run: dict[str, Any]) -> dict[str, Any]:
    summary = run["summary"]
    measured = [r for r in run["records"] if r.get("phase", "measured") == "measured"]
    ok = [r for r in measured if _ok(r)]
    ttft = [r["chunks"][0] - r["sent"] for r in ok]
    e2e = [r["end"] - r["sent"] for r in ok]
    tpot = [
        (r["chunks"][-1] - r["chunks"][0]) / (r["completion_tokens"] - 1)
        for r in ok
        if (r["completion_tokens"] or 0) > 1
    ]
    itl = [b - a for r in ok for a, b in zip(r["chunks"], r["chunks"][1:], strict=False)]
    start = min((r["sent"] for r in measured), default=0.0)
    end = max((r["end"] for r in ok), default=start)
    window = max(end - start, 1e-9)
    tokens = sum(r["completion_tokens"] or 0 for r in ok)
    chunks = sum(len(r["chunks"]) for r in ok)
    out: dict[str, Any] = {
        "measured": len(measured),
        "ok": len(ok),
        "errors": len(measured) - len(ok),
        "window_s": window,
        "ttft": _stats(ttft),
        "tpot": _stats(tpot),
        "itl": _stats(itl),
        "e2e": _stats(e2e),
        "request_throughput": len(ok) / window,
        "output_token_throughput": tokens / window,
        "prompt_token_mismatches": sum(
            1 for r in ok if r["prompt_tokens"] != r["target_prompt_tokens"]
        ),
        "ignore_eos_violations": sum(1 for r in ok if r["completion_tokens"] != r["max_tokens"]),
        "missing_usage": sum(1 for r in ok if r["prompt_tokens"] is None),
        "tokens_per_chunk": tokens / chunks if chunks else None,
        "max_send_lag_s": summary.get("max_send_lag_s"),
    }
    rate = summary.get("rate")
    if rate:
        out["offered_rate"] = rate
        growth = ttft_growth(ok)
        out["ttft_growth"] = growth
        out["saturated"] = growth is not None and growth > TTFT_GROWTH_SATURATED
    t0 = summary.get("t0_wall")
    if t0 is not None and run["cpu"]:
        out["cpu"] = cpu_stats(run["cpu"], t0 + start, t0 + end)
    return out
