"""The README's micro-benchmark table must be the generated one, row for row."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def table_rows(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.startswith("| ")]


def test_readme_benchmark_table_matches_generated_results() -> None:
    generated = table_rows((ROOT / "benchmarks/results/microbench.md").read_text())
    readme = (ROOT / "README.md").read_text()
    assert generated, "no table in benchmarks/results/microbench.md"
    for row in generated:
        assert row in readme, f"README is missing or has edited the generated row: {row}"
