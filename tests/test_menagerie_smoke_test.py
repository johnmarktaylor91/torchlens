"""Tests for menagerie smoke-test orchestration helpers."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from menagerie import smoke_gate, smoke_test
from menagerie.catalog import CatalogRow, write_catalog
from menagerie.cluster_runner import GIANT_REGISTRY


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


def test_select_cases_drops_cluster_runner_under_no_cluster() -> None:
    """--no-cluster drops any case that expects a remote cluster runner.

    A genuine force_cluster giant is tagged include="heavy" (opt-in via
    --with-heavy-giant) and expected_runner="cluster". Under --no-cluster the
    run uses --runner local, so such a case must be dropped rather than routed
    local against a remote expectation.
    """

    forced_id = next(sid for sid, entry in GIANT_REGISTRY.items() if entry.force_cluster)
    forced = smoke_test.SmokeCase(
        {
            "stable_id": forced_id,
            "include": "heavy",
            "synthetic": False,
            "expected_env": "base",
            "expected_runner": "cluster",
            "expected_status": "validated",
        }
    )
    cases = [_case("m1"), forced]

    with_giant = smoke_test.select_cases(
        cases, all_islands=False, with_heavy_giant=True, no_cluster=False
    )
    no_cluster = smoke_test.select_cases(
        cases, all_islands=False, with_heavy_giant=True, no_cluster=True
    )

    assert [case.stable_id for case in with_giant] == ["m1", forced_id]
    assert [case.stable_id for case in no_cluster] == ["m1"]


def test_smoke_manifest_cluster_cases_are_force_cluster_giants() -> None:
    """Every committed cluster-runner expectation follows the true routing policy.

    A model is expected remote IFF it genuinely force-routes to the shared
    cluster (force_cluster=True). A stale cluster tag on a model that fits
    locally would demand a forbidden remote dispatch.
    """

    cases = smoke_test.load_cases(smoke_test.DEFAULT_SMOKE_MANIFEST)
    cluster_ids = [
        case.stable_id for case in cases if case.payload.get("expected_runner") == "cluster"
    ]
    assert cluster_ids, "smoke manifest must retain at least one forced-cluster case"
    for stable_id in cluster_ids:
        entry = GIANT_REGISTRY.get(stable_id)
        assert entry is not None, f"{stable_id} missing from GIANT_REGISTRY"
        assert entry.force_cluster, f"{stable_id} expected remote but force_cluster=False"


def test_smoke_gate_rejects_stale_cluster_expectation(tmp_path: Path) -> None:
    """The gate fails loudly on a cluster expectation for a local-routing model."""

    fitting_giant = next(sid for sid, entry in GIANT_REGISTRY.items() if not entry.force_cluster)
    cases = [{"stable_id": fitting_giant, "expected_runner": "cluster"}]
    with pytest.raises(RuntimeError, match="stale cluster expectation"):
        smoke_gate._assert_cluster(cases, tmp_path)  # noqa: SLF001


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
