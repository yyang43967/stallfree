"""Eight concurrent streaming clients against the real server on the GPU (see concurrency.py)."""

from __future__ import annotations

from tests.gpu.concurrency import CLIENTS, run_concurrency_check


def test_eight_concurrent_streams_on_gpu(model_dir: str) -> None:
    report = run_concurrency_check(model_dir)
    assert report["distinct_response_ids"] == CLIENTS
    for row in report["results"]:
        assert row["one_response_id"] and row["usage_ok"] and row["finish"] in ("stop", "length")
        assert row["passes"], row
