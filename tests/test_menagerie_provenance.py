"""Tests for menagerie sweep provenance."""

from __future__ import annotations

import json
from pathlib import Path

from menagerie.provenance import read_sweeps, record_sweep
from menagerie.status import build_provenance_status, format_provenance_status
from menagerie.tools.import_provenance import import_historical_provenance


def test_record_and_read_sweep_provenance(tmp_path: Path) -> None:
    """Sweep provenance persists structured fields in the verification database."""

    db_path = tmp_path / "verification.db"

    record_sweep(
        sweep_id="2026-06-22-unit",
        date="2026-06-22",
        agent="unit",
        sources=["source-a", "source-b"],
        selection_rule="add only genuinely new architecture families",
        families_considered=["A", "B"],
        families_added=["A"],
        families_rejected=["B"],
        git_commit="abc123",
        notes="raw notes",
        db_path=db_path,
    )

    rows = read_sweeps(db_path)

    assert len(rows) == 1
    assert rows[0].sweep_id == "2026-06-22-unit"
    assert rows[0].sources == ["source-a", "source-b"]
    assert rows[0].families_added == ["A"]


def test_import_historical_provenance_and_status_view(tmp_path: Path) -> None:
    """Historical crawl logs import into queryable rows used by status provenance."""

    crawl_history = tmp_path / "crawl_history.json"
    crawl_log = tmp_path / "CRAWL_LOG.md"
    harvest_sources = tmp_path / "HARVEST_SOURCES.md"
    source_jsonl = tmp_path / "master_catalog.jsonl"
    deferred_jsonl = tmp_path / "deferred.jsonl"
    db_path = tmp_path / "verification.db"

    crawl_history.write_text(
        json.dumps(
            {
                "last_exhaustive_crawl": "2026-06-22",
                "crawls": [
                    {
                        "date": "2026-06-22",
                        "type": "unit sweep",
                        "axes": ["axis-a"],
                        "method": "grep, render, validate",
                        "result": {"families_normalized": 1},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    crawl_log.write_text("## 2026-06-22\nRaw markdown details\n", encoding="utf-8")
    harvest_sources.write_text("## Hubs\n- **UnitHub** - example\n", encoding="utf-8")
    source_jsonl.write_text(
        (
            '{"name":"UnitNet","zoo":"unit","family":"unit","domain":"vision","era":"2026",'
            '"recipe":{"type":"import-callable","expr":"torch.nn.Identity()",'
            '"imports":["torch"],"input":{"kind":"tensor","spec":{"shape":[1]}}},'
            '"input":{"kind":"tensor","spec":{"shape":[1]}},"added_wave":"2026-06-22-unit-sweep"}\n'
        ),
        encoding="utf-8",
    )
    deferred_jsonl.write_text("", encoding="utf-8")

    rows = import_historical_provenance(
        crawl_history=crawl_history,
        crawl_log=crawl_log,
        harvest_sources=harvest_sources,
        db_path=db_path,
    )
    status = build_provenance_status(
        ledger_db=db_path,
        source_jsonl=source_jsonl,
        deferred_jsonl=deferred_jsonl,
        crawl_history=crawl_history,
    )

    assert len(rows) == 1
    assert rows[0].sweep_id == "2026-06-22-unit-sweep"
    assert "Raw markdown details" in rows[0].notes
    assert status.last_exhaustive_crawl == "2026-06-22"
    assert status.models_with_known_sweep == 1
    assert "structured sweeps: 1" in format_provenance_status(status)
