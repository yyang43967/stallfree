"""Real Qwen3-0.6B in bf16 on the flash backend: margin rule, teacher-forced check, preemption.

Writes the full report to ``$STALLFREE_REPORT`` (default /tmp/margin_report.json)."""

from __future__ import annotations

import json
import os

from tests.gpu.margin import BF16_MARGIN_TAU, run_margin_suite


def test_margin_rule_teacher_forcing_and_preemption(model_dir: str) -> None:
    report = run_margin_suite(model_dir)
    path = os.environ.get("STALLFREE_REPORT", "/tmp/margin_report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    for mode in ("single", "batched"):
        assert report["summary"][mode]["failed_rule"] == 0, mode
    budget = report["mixed_batching_budget"]
    if budget is not None:
        # A budget below every prompt length must split every prompt run alone.
        single = report["results"]["single"]
        if all(r["prompt_len"] > budget for r in single):
            assert all(r["prefill_chunks"] > 1 for r in single)
    tf = report["teacher_forced"]
    assert tf["flash_mismatches"] == [], f"top-1 differs where the gap > {BF16_MARGIN_TAU}"
    assert tf["torch_ref_mismatches"] == []
    pre = report["preemption"]
    assert pre["num_preemptions"] >= 1, "the preemption scenario must preempt"
    assert all(r["passes"] for r in pre["results"])
