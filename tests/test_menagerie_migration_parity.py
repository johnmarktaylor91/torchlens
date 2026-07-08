"""Tests for the Phase-2c migration parity checker."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pytest

from menagerie.catalog import SOURCE_COLUMNS
from menagerie.tools import migration_parity
from menagerie.tools.tsv_to_jsonl import migrate_tsv


def _write_tsv(path: Path, rows: list[list[str]]) -> None:
    """Write a small TSV fixture.

    Parameters
    ----------
    path:
        Destination path.
    rows:
        Source rows.
    """

    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(SOURCE_COLUMNS)
        writer.writerows(rows)


def test_run_parity_reports_matching_sample(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Parity compares old and typed build paths for candidate records."""

    source = tmp_path / "source.tsv"
    candidate = tmp_path / "candidate.jsonl"
    stats = tmp_path / "stats.json"
    report_path = tmp_path / "report.json"
    _write_tsv(
        source,
        [
            [
                "ExprToy",
                "unit",
                "torch.nn.Identity()",
                "(1, 3)",
                "float32",
                "toy",
                "vision",
                "2026",
                "verified",
            ],
            [
                "StatementToy",
                "unit",
                "import torch; model=torch.nn.Identity()",
                "(1, 4)",
                "float32",
                "toy",
                "vision",
                "2026",
                "verified",
            ],
        ],
    )
    migrate_tsv(source, candidate, stats)

    def fake_fingerprint(model: Any, example_input: Any) -> str:
        """Return a deterministic signature for a built model/input pair."""

        return f"{type(model).__name__}:{tuple(example_input.shape)}"

    monkeypatch.setattr(migration_parity, "structural_fingerprint", fake_fingerprint)

    report = migration_parity.run_parity(candidate, source, report_path, target=2)

    assert report["tested"] == 2
    assert report["matched"] == 2
    assert report["mismatch_count"] == 0
    assert report_path.exists()
