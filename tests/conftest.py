"""Shared pytest hooks."""

from __future__ import annotations

import pytest
from helpers.tiny_qwen3 import min_gaps


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter) -> None:
    gaps = min_gaps()
    if not gaps:
        return
    terminalreporter.section("smallest reference top-1 minus top-2 logit gap")
    for label, gap in sorted(gaps.items()):
        terminalreporter.write_line(f"{label}: {gap:.3e}")
    terminalreporter.write_line(f"overall: {min(gaps.values()):.3e}")


@pytest.fixture(scope="session")
def tiny_hf_model():  # type: ignore[no-untyped-def]
    """The Hugging Face reference: tiny random Qwen3, fp32, eager attention, seed 0."""
    from helpers.tiny_qwen3 import make_hf_model

    return make_hf_model(seed=0)


@pytest.fixture(scope="session")
def tiny_model_dir(tiny_hf_model, tmp_path_factory):  # type: ignore[no-untyped-def]
    """The same model saved as a checkpoint directory (config.json + safetensors)."""
    from helpers.tiny_qwen3 import save_hf_model

    return save_hf_model(tiny_hf_model, tmp_path_factory.mktemp("tiny_qwen3_ckpt"))
