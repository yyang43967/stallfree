"""benchmarks/results.md quotes generated tables; every quoted row must match the generated one."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_quoted_table_rows_match_the_generated_tables() -> None:
    doc = (ROOT / "benchmarks" / "results.md").read_text()
    generated = (ROOT / "benchmarks" / "results" / "sweep.md").read_text().splitlines()
    quoted = [
        line
        for line in doc.splitlines()
        if line.startswith("| ")
        and line.split(" | ")[0].lstrip("| ")
        in {"stallfree-off", "stallfree-on", "vllm", "vllm-eager", "256", "512", "1024", "2048"}
    ]
    # Rows of the hand-written summary tables carry a % or text cells, not generated cells.
    quoted = [q for q in quoted if q.count("|") >= 6 and "%" not in q]
    assert len(quoted) >= 14
    for line in quoted:
        assert line in generated, line


def test_readme_benchmark_rows_match_the_generated_table() -> None:
    readme = (ROOT / "README.md").read_text().splitlines()
    generated = (ROOT / "benchmarks" / "results" / "sweep.md").read_text().splitlines()
    rows = [line for line in readme if " | 6.41 (0.8x) | " in line]
    assert len(rows) == 4
    for line in rows:
        assert line in generated, line
