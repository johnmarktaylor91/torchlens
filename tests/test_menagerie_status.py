"""Tests for honest menagerie status reporting."""

from __future__ import annotations

from pathlib import Path

from menagerie.catalog import CatalogRow, write_catalog
from menagerie.ledger import VerificationRun, append_verification_run, connect
from menagerie.status import build_status, main
from menagerie.tools.distinct_report import build_distinct_report


def _row(**overrides: object) -> CatalogRow:
    """Build a compact catalog row fixture.

    Parameters
    ----------
    overrides:
        Field overrides for the default row.

    Returns
    -------
    CatalogRow
        Catalog row.
    """

    data = {
        "model_id": 1,
        "display_index": 1,
        "stable_id": "m1",
        "name": "UnitNet",
        "variant": "",
        "family": "unit",
        "family_normalized": "unit",
        "domain": "unit",
        "zoo": "unit-zoo",
        "constructor_call": "torch.nn.Identity()",
        "input_shape": "(1,)",
        "input_dtype": "float32",
        "era": "2026",
        "verified": True,
        "notes": "",
        "source": "catalog",
        "recipe_revision_sha256": "recipe-a",
        "input_is_real": True,
        "verification_expectation": "forward_required",
        "quarantine": False,
    }
    data.update(overrides)
    return CatalogRow(**data)


def _run(**overrides: object) -> VerificationRun:
    """Build a compact verification run fixture.

    Parameters
    ----------
    overrides:
        Field overrides for the default run.

    Returns
    -------
    VerificationRun
        Verification run.
    """

    data = {
        "stable_id": "m1",
        "recipe_revision_sha256": "recipe-a",
        "name": "UnitNet",
        "zoo": "unit-zoo",
        "variant": "",
        "scope": "forward",
        "status": "passed",
        "forward_pass": 1,
        "backward_pass": None,
        "backward_na_reason": None,
        "metadata_ok": 1,
        "n_ops": 3,
        "graph_shape_hash": "hash-a",
        "svg_sha256": None,
        "torchlens_version": "tl-test",
        "torch_version": "torch-test",
        "python_version": "py-test",
        "device_requested": "cpu",
        "device_actual": "cpu",
        "env_hash": None,
        "runner_host": "unit-host",
        "started_at": "2026-06-22T00:00:00+00:00",
        "finished_at": "2026-06-22T00:00:01+00:00",
        "duration_sec": 1.0,
        "error_class": None,
        "error_message": None,
        "run_id": "run-a",
    }
    data.update(overrides)
    return VerificationRun(**data)  # type: ignore[arg-type]


def _write_catalog(tmp_path: Path, rows: list[CatalogRow]) -> Path:
    """Write a synthetic catalog database.

    Parameters
    ----------
    tmp_path:
        Temporary test directory.
    rows:
        Catalog rows.

    Returns
    -------
    Path
        Catalog database path.
    """

    catalog_db = tmp_path / "catalog.db"
    write_catalog(rows, canonical_tsv=tmp_path / "catalog.tsv", db_path=catalog_db)
    return catalog_db


def test_status_funnel_tiers_are_consistent(tmp_path: Path) -> None:
    """Status reports honest internally consistent funnel tiers."""

    catalog_db = _write_catalog(
        tmp_path,
        [
            _row(model_id=1, display_index=1, stable_id="m1", name="RealNet"),
            _row(
                model_id=2,
                display_index=2,
                stable_id="m2",
                name="WrapperNet",
                input_is_real=False,
            ),
            _row(
                model_id=3,
                display_index=3,
                stable_id="m3",
                name="DeferredNet",
                verification_expectation="deferred",
            ),
            _row(
                model_id=4,
                display_index=4,
                stable_id="m4",
                name="QuarantineNet",
                quarantine=True,
            ),
            _row(
                model_id=5,
                display_index=5,
                stable_id="m5",
                name="CodeExecNet",
                constructor_call="import torch\nmodel = torch.nn.Identity()",
            ),
        ],
    )
    ledger_db = tmp_path / "verification.db"
    conn = connect(ledger_db)
    append_verification_run(conn, _run(run_id="run-real", stable_id="m1", name="RealNet"))
    append_verification_run(
        conn,
        _run(
            run_id="run-wrapper",
            stable_id="m2",
            name="WrapperNet",
            graph_shape_hash="hash-b",
            finished_at="2026-06-22T00:00:02+00:00",
        ),
    )

    status = build_status(
        catalog_db=catalog_db,
        ledger_db=ledger_db,
        torchlens_version="tl-test",
        render_manifest=tmp_path / "missing.tsv",
    )

    assert status.total_catalog_models == 5
    assert status.expected_models == 4
    assert status.verified_models == 2
    assert status.headline_verified_real_input == 1
    assert status.verified_wrapper_input == 1
    assert status.deferred_models == 1
    assert status.quarantined_models == 1
    assert status.code_execution_models == 1
    assert (
        status.headline_verified_real_input + status.verified_wrapper_input
        <= status.verified_models
        <= status.expected_models
        <= status.total_catalog_models
    )


def test_distinct_count_collapses_duplicate_graph_shape_hashes(tmp_path: Path) -> None:
    """Distinct reporting counts unique shape-blind graph-shape hashes."""

    catalog_db = _write_catalog(
        tmp_path,
        [
            _row(model_id=1, display_index=1, stable_id="m1", name="NetA"),
            _row(model_id=2, display_index=2, stable_id="m2", name="NetB"),
            _row(model_id=3, display_index=3, stable_id="m3", name="NetC"),
        ],
    )
    ledger_db = tmp_path / "verification.db"
    conn = connect(ledger_db)
    append_verification_run(conn, _run(run_id="run-a", stable_id="m1", graph_shape_hash="same"))
    append_verification_run(
        conn,
        _run(
            run_id="run-b",
            stable_id="m2",
            name="NetB",
            graph_shape_hash="same",
            finished_at="2026-06-22T00:00:02+00:00",
        ),
    )
    append_verification_run(
        conn,
        _run(
            run_id="run-c",
            stable_id="m3",
            name="NetC",
            graph_shape_hash="different",
            finished_at="2026-06-22T00:00:03+00:00",
        ),
    )

    report = build_distinct_report(
        catalog_db=catalog_db,
        ledger_db=ledger_db,
        torchlens_version="tl-test",
    )

    assert report.hashed_model_count == 3
    assert report.total_distinct_architectures == 2
    assert report.verified_model_count == 3
    assert report.verified_distinct_architectures == 2
    assert report.by_name_vs_architecture_gap == 1


def test_status_reflects_current_recipe_verified_count(tmp_path: Path) -> None:
    """Status does not count stale-recipe ledger passes."""

    catalog_db = _write_catalog(
        tmp_path,
        [_row(model_id=1, display_index=1, stable_id="m1", recipe_revision_sha256="recipe-b")],
    )
    ledger_db = tmp_path / "verification.db"
    conn = connect(ledger_db)
    append_verification_run(conn, _run(run_id="run-old", recipe_revision_sha256="recipe-a"))

    stale_status = build_status(
        catalog_db=catalog_db,
        ledger_db=ledger_db,
        torchlens_version="tl-test",
        render_manifest=tmp_path / "missing.tsv",
    )
    assert stale_status.verified_models == 0

    append_verification_run(
        conn,
        _run(
            run_id="run-current",
            recipe_revision_sha256="recipe-b",
            graph_shape_hash="hash-b",
            finished_at="2026-06-22T00:00:02+00:00",
        ),
    )

    current_status = build_status(
        catalog_db=catalog_db,
        ledger_db=ledger_db,
        torchlens_version="tl-test",
        render_manifest=tmp_path / "missing.tsv",
    )
    assert current_status.verified_models == 1


def test_status_fraction_thresholds_exit_nonzero(tmp_path: Path) -> None:
    """Deferred and quarantine fraction ceilings are CI-able."""

    catalog_db = _write_catalog(
        tmp_path,
        [
            _row(model_id=1, display_index=1, stable_id="m1", name="ExpectedNet"),
            _row(
                model_id=2,
                display_index=2,
                stable_id="m2",
                name="DeferredNet",
                verification_expectation="deferred",
                quarantine=True,
            ),
        ],
    )
    ledger_db = tmp_path / "verification.db"
    connect(ledger_db).close()

    assert (
        main(
            [
                "--catalog-db",
                str(catalog_db),
                "--ledger-db",
                str(ledger_db),
                "--torchlens-version",
                "tl-test",
                "--render-manifest",
                str(tmp_path / "missing.tsv"),
                "--max-deferred-frac",
                "0.40",
            ]
        )
        == 1
    )
    assert (
        main(
            [
                "--catalog-db",
                str(catalog_db),
                "--ledger-db",
                str(ledger_db),
                "--torchlens-version",
                "tl-test",
                "--render-manifest",
                str(tmp_path / "missing.tsv"),
                "--max-quarantine-frac",
                "0.40",
            ]
        )
        == 1
    )
