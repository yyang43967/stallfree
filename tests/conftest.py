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
