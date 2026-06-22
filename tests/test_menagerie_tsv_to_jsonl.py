"""Tests for the menagerie TSV-to-JSONL migration tool."""

from __future__ import annotations

import csv
from pathlib import Path

from menagerie.catalog import SOURCE_COLUMNS
from menagerie.schema import load_jsonl
from menagerie.tools.tsv_to_jsonl import migrate_tsv


def _write_tsv(path: Path, rows: list[list[str]]) -> None:
    """Write a small source TSV fixture.

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


def test_migrate_tsv_skips_classics_and_preserves_non_classics(tmp_path: Path) -> None:
    """The migration writes one candidate record per non-classics row."""

    source = tmp_path / "source.tsv"
    output = tmp_path / "candidate.jsonl"
    deferred_output = tmp_path / "deferred.jsonl"
    stats_path = tmp_path / "stats.json"
    _write_tsv(
        source,
        [
            [
                "ClassicToy",
                "classics-pytorch",
                "menagerie.classics.toy.build()",
                "(1, 3)",
                "float32",
                "classic",
                "history",
                "2026",
                "verified",
            ],
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
                "WrapperToy",
                "unit",
                "import builtins, torch; _W=builtins.type('W',(torch.nn.Module,),{'forward':(lambda self,x: x)}); model=_W()",
                "(1,)",
                "float32",
                "toy",
                "vision",
                "2026",
                "verified",
            ],
            [
                "DeferredToy",
                "unit",
                "torch.nn.Identity()",
                "text tokens",
                "float32",
                "toy",
                "text",
                "2026",
                "source-only",
            ],
        ],
    )

    stats = migrate_tsv(source, output, stats_path, deferred_output_path=deferred_output)
    records = load_jsonl(output)
    deferred_records = load_jsonl(deferred_output)

    assert stats["classics_rows_skipped"] == 1
    assert stats["non_classics_records_produced"] == 3
    assert stats["forward_required_records_written"] == 2
    assert stats["deferred_count"] == 1
    assert [record.name for record in records] == ["ExprToy", "WrapperToy"]
    assert [record.name for record in deferred_records] == ["DeferredToy"]
    assert records[0].recipe.type == "import-callable"
    assert records[1].recipe.type == "statement"
    assert records[1].input.kind == "none"
    assert records[1].input_is_real is False
    assert deferred_records[0].verification_expectation == "deferred"
    assert deferred_records[0].deferral is not None
