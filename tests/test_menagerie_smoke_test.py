"""Tests for menagerie smoke-test orchestration helpers."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from menagerie import smoke_test
from menagerie.catalog import CatalogRow, write_catalog


def _case(
    stable_id: str, include: str = "default", synthetic: bool = False
) -> smoke_test.SmokeCase:
    """Build one smoke case fixture.

    Parameters
    ----------
    stable_id:
        Stable ID.
    include:
        Inclusion group.
    synthetic:
        Whether the row is synthetic.

    Returns
    -------
    smoke_test.SmokeCase
        Smoke case.
    """

    return smoke_test.SmokeCase(
        {
            "stable_id": stable_id,
            "include": include,
            "synthetic": synthetic,
            "expected_env": "base",
            "expected_status": "validated",
        }
    )


def _row() -> CatalogRow:
    """Build a compact catalog row fixture."""

    return CatalogRow(
        model_id=1,
        display_index=1,
        stable_id="m1",
        name="UnitNet",
        variant="",
        family="unit",
        family_normalized="unit",
        domain="unit",
        zoo="unit",
        constructor_call="torch.nn.Identity()",
        input_shape="(1,)",
        input_dtype="float32",
        era="2026",
        verified=True,
        notes="",
        source="test",
        recipe_revision_sha256="recipe-a",
    )


def test_select_cases_respects_optional_groups() -> None:
    """Case selection honors all-islands, heavy, and no-cluster flags."""

    cases = [
        _case("m1"),
        _case("m2", "all_islands"),
        _case("m3", "heavy"),
        _case("m4", "cluster"),
    ]

    default = smoke_test.select_cases(
        cases, all_islands=False, with_heavy_giant=False, no_cluster=False
    )
    extended = smoke_test.select_cases(
        cases, all_islands=True, with_heavy_giant=True, no_cluster=True
    )

    assert [case.stable_id for case in default] == ["m1", "m4"]
    assert [case.stable_id for case in extended] == ["m1", "m2", "m3"]


def test_insert_synthetic_rows_adds_smoke_catalog_rows(tmp_path: Path) -> None:
    """Synthetic smoke rows are injected only into the copied catalog DB."""

    catalog_db = tmp_path / "catalog.db"
    write_catalog([_row()], canonical_tsv=tmp_path / "catalog.tsv", db_path=catalog_db)
    smoke_test._insert_synthetic_rows(  # noqa: SLF001
        catalog_db,
        [_case("smoke_exc_1", synthetic=True)],
    )

    with sqlite3.connect(catalog_db) as connection:
        row = connection.execute(
            "SELECT stable_id, name, source FROM models WHERE stable_id = 'smoke_exc_1'"
        ).fetchone()

    assert tuple(row) == ("smoke_exc_1", "smoke_exc_1", "smoke")


def test_smoke_manifest_jsonl_is_valid() -> None:
    """The committed smoke manifest parses as JSONL."""

    cases = smoke_test.load_cases(smoke_test.DEFAULT_SMOKE_MANIFEST)

    assert cases
    assert len({case.stable_id for case in cases}) == len(cases)
    assert all(json.dumps(case.payload) for case in cases)
