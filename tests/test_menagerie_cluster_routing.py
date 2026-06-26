"""Tests for automatic menagerie cluster routing integration."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from menagerie import validate_menagerie
from menagerie.catalog import CatalogRow, write_catalog
from menagerie.cluster_runner import DispatchResult, MergeReport
from menagerie.ledger import VerificationRun, append_verification_run, connect
from menagerie.runtime import DependencyPlan


def _row(**overrides: object) -> CatalogRow:
    """Build a compact catalog row fixture.

    Parameters
    ----------
    overrides:
        Catalog field overrides.

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
    }
    data.update(overrides)
    return CatalogRow(**data)


def _run(**overrides: object) -> VerificationRun:
    """Build a compact verification run fixture.

    Parameters
    ----------
    overrides:
        Verification field overrides.

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
        "graph_shape_hash": "shape-a",
        "svg_sha256": None,
        "torchlens_version": "tl-test",
        "torch_version": "torch-test",
        "python_version": "py-test",
        "device_requested": "cpu",
        "device_actual": "cpu",
        "env_hash": "env-a",
        "lock_hash": "lock-a",
        "torchlens_source_hash": "source-a",
        "input_scale": 1.0,
        "runner_host": "unit-host",
        "started_at": "2026-06-25T00:00:00+00:00",
        "finished_at": "2026-06-25T00:00:01+00:00",
        "duration_sec": 1.0,
        "peak_rss_mb": 64,
        "error_class": None,
        "error_message": None,
        "run_id": "run-a",
    }
    data.update(overrides)
    return VerificationRun(**data)  # type: ignore[arg-type]


def _plan() -> DependencyPlan:
    """Build a base dependency plan fixture.

    Returns
    -------
    DependencyPlan
        Dependency plan.
    """

    return DependencyPlan(
        cluster_key="base",
        packages=(),
        top_modules=(),
        environment="base",
    )


def _args(tmp_path: Path, *extra: str) -> Any:
    """Parse validator arguments for a temporary test run.

    Parameters
    ----------
    tmp_path:
        Test temporary directory.
    extra:
        Additional CLI arguments.

    Returns
    -------
    Any
        Parsed argument namespace.
    """

    return validate_menagerie.build_parser().parse_args(
        [
            "--out-dir",
            str(tmp_path / "out"),
            "--manifest",
            str(tmp_path / "manifest.tsv"),
            "--db",
            str(tmp_path / "catalog.db"),
            "--base-env-only",
            "--jobs",
            "1",
            *extra,
        ]
    )


def test_auto_runner_dispatches_static_giant_and_keeps_non_giant_local(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Auto routing sends static giants to cluster and leaves normal rows local."""

    giant = _row(stable_id="m3635", name="beit_large_patch16_512")
    local = _row(model_id=2, display_index=2, stable_id="m_local", name="SmallNet")
    dispatch_calls: list[tuple[str, ...]] = []
    local_calls: list[str] = []
    merge_calls: list[tuple[Path, Path]] = []

    def fake_dispatch(stable_ids: list[str], **_: object) -> DispatchResult:
        """Capture cluster dispatches."""

        dispatch_calls.append(tuple(stable_ids))
        return DispatchResult(
            campaign_id="campaign-a",
            attempt_id="attempt-a",
            assignments=(),
            local_artifact_dir=tmp_path / "cluster",
            remote_artifact_dir="~/cluster",
            sbatch_job_ids=("123",),
            commands=(),
        )

    def fake_collect(_: DispatchResult) -> tuple[Path, Path]:
        """Return placeholder collected result paths."""

        rows_path = tmp_path / "cluster_results.jsonl"
        manifest_path = tmp_path / "cluster_results.manifest.json"
        rows_path.write_text("", encoding="utf-8")
        manifest_path.write_text("{}", encoding="utf-8")
        return rows_path, manifest_path

    def fake_merge(rows_path: Path, manifest_path: Path, **_: object) -> MergeReport:
        """Capture cluster merge calls."""

        merge_calls.append((rows_path, manifest_path))
        return MergeReport("campaign-a", "attempt-a", inserted=1, duplicates=0, assignments=1)

    def fake_append_cluster(rows: list[CatalogRow], manifest_path: Path) -> None:
        """Append a manifest row for mocked cluster results."""

        for row in rows:
            validate_menagerie.append_manifest(
                manifest_path,
                validate_menagerie.ValidationResult(
                    row.name,
                    row.model_id,
                    "validated",
                    3,
                    True,
                    "forward",
                    0.1,
                    "cluster",
                    "",
                    stable_id=row.stable_id,
                    recipe_revision_sha256=row.recipe_revision_sha256,
                ),
            )

    def fake_validate_with_timeout(
        row: CatalogRow,
        *_: object,
        **__: object,
    ) -> validate_menagerie.ValidationResult:
        """Capture local validation calls."""

        local_calls.append(row.stable_id)
        return validate_menagerie.ValidationResult(
            row.name,
            row.model_id,
            "validated",
            2,
            True,
            "forward",
            0.1,
            "base",
            "",
            stable_id=row.stable_id,
            recipe_revision_sha256=row.recipe_revision_sha256,
        )

    monkeypatch.setattr(validate_menagerie, "select_rows", lambda _: [giant, local])
    monkeypatch.setattr(validate_menagerie.cluster_runner, "dispatch_giants", fake_dispatch)
    monkeypatch.setattr(validate_menagerie.cluster_runner, "collect_cluster_results", fake_collect)
    monkeypatch.setattr(validate_menagerie.cluster_runner, "merge_cluster_results", fake_merge)
    monkeypatch.setattr(
        validate_menagerie,
        "_append_cluster_rows_from_ledger",
        fake_append_cluster,
    )
    monkeypatch.setattr(
        validate_menagerie,
        "group_by_dependency",
        lambda rows: [(_plan(), list(rows))],
    )
    monkeypatch.setattr(validate_menagerie, "install_dependency_plan", lambda *_: None)
    monkeypatch.setattr(validate_menagerie, "validate_with_timeout", fake_validate_with_timeout)
    monkeypatch.setattr(validate_menagerie, "append_validation_ledger", lambda *_: None)

    assert validate_menagerie.run(_args(tmp_path)) == 0

    assert dispatch_calls == [("m3635",)]
    assert local_calls == ["m_local"]
    assert len(merge_calls) == 1
    records = validate_menagerie.manifest_records(tmp_path / "manifest.tsv")
    assert set(records) == {"m3635", "m_local"}


def test_auto_runner_uses_cold_start_and_peak_rss_giant_routes(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Auto routing covers cold-start heuristics and measured peak-RSS giants."""

    ledger_db = tmp_path / "verification.db"
    with connect(ledger_db) as conn:
        append_verification_run(
            conn,
            _run(
                run_id="peak-run",
                stable_id="peak-model",
                peak_rss_mb=130 * validate_menagerie.MB_PER_GB,
            ),
        )
    monkeypatch.setattr(validate_menagerie.cluster_runner, "VERIFICATION_DB", ledger_db)
    cold_start = _row(stable_id="cold-start", name="Research 2B MoE")
    peak = _row(stable_id="peak-model", name="MeasuredGiant")
    small = _row(stable_id="small", name="SmallNet")

    routed = validate_menagerie._cluster_route_rows([cold_start, peak, small], "auto")

    assert [row.stable_id for row in routed] == ["cold-start", "peak-model"]


def test_auto_runner_routes_oom_but_not_native_crash(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Ledger OOM evidence is cluster-eligible, but native crashes are local-only."""

    ledger_db = tmp_path / "verification.db"
    with connect(ledger_db) as conn:
        append_verification_run(
            conn,
            _run(
                run_id="oom-run",
                stable_id="oom-model",
                status="oom",
                forward_pass=0,
                metadata_ok=0,
                n_ops=None,
                peak_rss_mb=None,
            ),
        )
        append_verification_run(
            conn,
            _run(
                run_id="crash-run",
                stable_id="crash-model",
                status="native_crash",
                forward_pass=0,
                metadata_ok=0,
                n_ops=None,
                peak_rss_mb=None,
            ),
        )
    monkeypatch.setattr(validate_menagerie.cluster_runner, "VERIFICATION_DB", ledger_db)
    oom = _row(stable_id="oom-model", name="OOMNet")
    crash = _row(stable_id="crash-model", name="CrashNet")

    routed = validate_menagerie._cluster_route_rows([oom, crash], "auto")

    assert [row.stable_id for row in routed] == ["oom-model"]


def test_no_build_catalog_selection_uses_read_only_loader(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """The worker no-build path selects rows without the build-capable loader."""

    row = _row(stable_id="m1", name="UnitNet")
    catalog_db = tmp_path / "catalog.db"
    write_catalog([row], canonical_tsv=tmp_path / "catalog.tsv", db_path=catalog_db)

    def fail_load_rows(*_: object, **__: object) -> list[CatalogRow]:
        """Fail if no-build mode calls the normal loader."""

        raise AssertionError("load_rows should not run in no-build mode")

    monkeypatch.setattr(validate_menagerie, "load_rows", fail_load_rows)

    selected = validate_menagerie._select_rows_for_validation(
        _args(tmp_path, "--db", str(catalog_db), "--no-build-catalog", "--stable-ids", "m1")
    )

    assert [item.stable_id for item in selected] == ["m1"]


def test_cluster_routing_resume_uses_ledger_not_manifest(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """A ledger-complete giant is not dispatched or run locally without manifest state."""

    ledger_db = tmp_path / "verification.db"
    giant = _row(stable_id="m3635", name="beit_large_patch16_512")
    local_calls: list[str] = []
    monkeypatch.setenv("TORCHLENS_MENAGERIE_ENV_HASH", "env-a")
    monkeypatch.setenv("TORCHLENS_MENAGERIE_LOCK_HASH", "lock-a")
    monkeypatch.setenv("TORCHLENS_SOURCE_HASH", "source-a")
    with connect(ledger_db) as conn:
        append_verification_run(
            conn,
            _run(stable_id="m3635", name="beit_large_patch16_512", runner_host="axon-test"),
        )

    def fail_dispatch(*_: object, **__: object) -> DispatchResult:
        """Fail if ledger-complete rows are dispatched."""

        raise AssertionError("ledger-complete giant was dispatched")

    def fake_validate_with_timeout(
        row: CatalogRow,
        *_: object,
        **__: object,
    ) -> validate_menagerie.ValidationResult:
        """Capture unexpected local validation calls."""

        local_calls.append(row.stable_id)
        return validate_menagerie.ValidationResult(
            row.name,
            row.model_id,
            "validated",
            2,
            True,
            "forward",
            0.1,
            "base",
            "",
            stable_id=row.stable_id,
            recipe_revision_sha256=row.recipe_revision_sha256,
        )

    monkeypatch.setattr(validate_menagerie.cluster_runner, "VERIFICATION_DB", ledger_db)
    monkeypatch.setattr(validate_menagerie, "select_rows", lambda _: [giant])
    monkeypatch.setattr(validate_menagerie.cluster_runner, "dispatch_giants", fail_dispatch)
    monkeypatch.setattr(validate_menagerie, "validate_with_timeout", fake_validate_with_timeout)
    monkeypatch.setattr(validate_menagerie, "write_reports", lambda *_args, **_kwargs: None)

    assert validate_menagerie.run(_args(tmp_path)) == 0

    assert local_calls == []
    records = validate_menagerie.manifest_records(tmp_path / "manifest.tsv")
    assert records["m3635"]["status"] == "validated"


def test_cluster_unreachable_writes_terminal_rows_and_continues(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Transport failure records env_unavailable rows instead of aborting validation."""

    ledger_db = tmp_path / "verification.db"
    giant = _row(stable_id="m3635", name="beit_large_patch16_512")
    catalog_db = tmp_path / "catalog.db"
    write_catalog([giant], canonical_tsv=tmp_path / "catalog.tsv", db_path=catalog_db)

    def fail_dispatch(*_: object, **__: object) -> DispatchResult:
        """Simulate an unreachable cluster transport command."""

        raise subprocess.CalledProcessError(
            255,
            ("ssh", "axon", "sbatch"),
            output="",
            stderr="ssh: connect failed",
        )

    monkeypatch.setattr(validate_menagerie.cluster_runner, "VERIFICATION_DB", ledger_db)
    monkeypatch.setattr(validate_menagerie, "select_rows", lambda _: [giant])
    monkeypatch.setattr(validate_menagerie.cluster_runner, "dispatch_giants", fail_dispatch)

    assert validate_menagerie.run(_args(tmp_path, "--db", str(catalog_db))) == 0

    with connect(ledger_db) as conn:
        row = conn.execute(
            "SELECT status, error_message FROM current_verification WHERE stable_id = 'm3635'"
        ).fetchone()
    records = validate_menagerie.manifest_records(tmp_path / "manifest.tsv")
    assert row["status"] == "env_unavailable"
    assert "reason=cluster_unreachable" in row["error_message"]
    assert records["m3635"]["status"] == "skipped:cluster_unavailable"


def test_cluster_timeout_writes_terminal_rows_and_continues(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Blocking sbatch timeouts record terminal rows instead of aborting."""

    ledger_db = tmp_path / "verification.db"
    giant = _row(stable_id="m3635", name="beit_large_patch16_512")
    catalog_db = tmp_path / "catalog.db"
    write_catalog([giant], canonical_tsv=tmp_path / "catalog.tsv", db_path=catalog_db)

    def timeout_dispatch(*_: object, **__: object) -> DispatchResult:
        """Simulate a blocking sbatch wait timeout."""

        raise subprocess.TimeoutExpired(("ssh", "axon", "sbatch --wait"), timeout=1.0)

    monkeypatch.setattr(validate_menagerie.cluster_runner, "VERIFICATION_DB", ledger_db)
    monkeypatch.setattr(validate_menagerie, "select_rows", lambda _: [giant])
    monkeypatch.setattr(validate_menagerie.cluster_runner, "dispatch_giants", timeout_dispatch)

    assert validate_menagerie.run(_args(tmp_path, "--db", str(catalog_db))) == 0

    with connect(ledger_db) as conn:
        row = conn.execute(
            "SELECT status, error_message FROM current_verification WHERE stable_id = 'm3635'"
        ).fetchone()
    assert row["status"] == "env_unavailable"
    assert "reason=cluster_timeout" in row["error_message"]
