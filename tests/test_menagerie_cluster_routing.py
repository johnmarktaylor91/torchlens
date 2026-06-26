"""Tests for automatic menagerie cluster routing integration."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from menagerie import validate_menagerie
from menagerie.catalog import CatalogRow, write_catalog
from menagerie.cluster_runner import (
    ClusterAssignment,
    CollectedClusterResults,
    DispatchResult,
    MergeReport,
)
from menagerie.ledger import VerificationRun, append_verification_run, connect
from menagerie.runtime import DependencyPlan


def _assignment(stable_id: str, array_index: int, **overrides: object) -> ClusterAssignment:
    """Build a compact cluster assignment fixture.

    Parameters
    ----------
    stable_id:
        Durable model identity.
    array_index:
        SLURM array task index.
    overrides:
        Assignment field overrides.

    Returns
    -------
    ClusterAssignment
        Cluster assignment.
    """

    data = {
        "campaign_id": "campaign-a",
        "attempt_id": "attempt-a",
        "assignment_id": f"campaign-a:attempt-a:{array_index}:{stable_id}",
        "stable_id": stable_id,
        "array_index": array_index,
        "node_mem_gb": 180,
        "worker_memory_cap_gb": 170,
        "partition": "nklab",
        "reason": "unit",
    }
    data.update(overrides)
    return ClusterAssignment(**data)  # type: ignore[arg-type]


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
            assignments=tuple(
                _assignment(stable_id, index) for index, stable_id in enumerate(stable_ids)
            ),
            local_artifact_dir=tmp_path / "cluster",
            remote_artifact_dir="~/cluster",
            sbatch_job_ids=("123",),
            commands=(),
        )

    def fake_collect(dispatch: DispatchResult, **_: object) -> CollectedClusterResults:
        """Return per-model collection with every task present (validated)."""

        rows_path = tmp_path / "cluster_results.jsonl"
        manifest_path = tmp_path / "cluster_results.manifest.json"
        rows_path.write_text("", encoding="utf-8")
        manifest_path.write_text("{}", encoding="utf-8")
        return CollectedClusterResults(
            present_assignments=dispatch.assignments,
            missing_assignments=(),
            result_rows_path=rows_path,
            result_manifest_path=manifest_path,
            result_dir=tmp_path / "cluster" / "results",
            log_dir=tmp_path / "cluster" / "logs",
        )

    def fake_merge(rows_path: Path, manifest_path: Path, **_: object) -> MergeReport:
        """Capture cluster merge calls."""

        merge_calls.append((rows_path, manifest_path))
        return MergeReport("campaign-a", "attempt-a", inserted=1, duplicates=0, assignments=1)

    def fake_append_cluster(rows: list[CatalogRow], manifest_path: Path, **_: object) -> None:
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
    monkeypatch.setattr(
        validate_menagerie.cluster_runner, "collect_cluster_results_partial", fake_collect
    )
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

    assert (
        validate_menagerie.run(
            _args(tmp_path, "--verification-db", str(tmp_path / "verification.db"))
        )
        == 0
    )

    assert dispatch_calls == [("m3635",)]
    assert local_calls == ["m_local"]
    assert len(merge_calls) == 1
    records = validate_menagerie.manifest_records(tmp_path / "manifest.tsv")
    assert set(records) == {"m3635", "m_local"}


def test_cluster_array_partial_failure_attributes_per_model_not_cascade(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """A partial array failure attributes per-model, NEVER a blanket cascade.

    Regression for the cluster result-handling cascade: array job 6014456 had
    82 tasks validate + 30 fail, but the batch-level ``sbatch --wait`` non-zero
    return code was stamped onto ALL ~290 dispatched models. The fix attributes
    each model by its OWN task outcome:

    * the task that validated -> ``validated``,
    * the task that honestly failed validation -> its real ``failed:*`` (the
      tripwire stays armed; the failure is surfaced, not masked),
    * the task that crashed with no result -> an honest
      ``failed:cluster_task_failed`` carrying its own ``.err`` tail,

    and NONE are recorded as a benign ``skipped:cluster_unavailable``.
    """

    from menagerie.cluster_runner import ClusterJobFailed, CollectedClusterResults
    from menagerie.cluster_runner import write_result_manifest, write_result_rows_jsonl
    from menagerie.cluster_runner import ClusterResultRow

    ledger_db = tmp_path / "verification.db"
    connect(ledger_db).close()
    monkeypatch.setenv("TORCHLENS_MENAGERIE_ENV_HASH", "env-a")
    monkeypatch.setenv("TORCHLENS_MENAGERIE_LOCK_HASH", "lock-a")
    monkeypatch.setenv("TORCHLENS_SOURCE_HASH", "source-a")

    good = _row(stable_id="m-good", name="GoodGiant")
    bad_validation = _row(model_id=2, display_index=2, stable_id="m-bad", name="BadGiant")
    crashed = _row(model_id=3, display_index=3, stable_id="m-crash", name="CrashGiant")
    rows = [good, bad_validation, crashed]

    good_assign = _assignment("m-good", 0)
    bad_assign = _assignment("m-bad", 1)
    crash_assign = _assignment("m-crash", 2)
    dispatch_result = DispatchResult(
        campaign_id="campaign-a",
        attempt_id="attempt-a",
        assignments=(good_assign, bad_assign, crash_assign),
        local_artifact_dir=tmp_path / "cluster",
        remote_artifact_dir="~/out/campaign-a/attempt-a",
        sbatch_job_ids=("6014456",),
        commands=(),
    )

    def fail_dispatch(*_: object, **__: object) -> DispatchResult:
        """Simulate a submitted array where one task failed (non-zero --wait)."""

        raise ClusterJobFailed(
            ("6014456",),
            "sbatch --wait returncode=1; Submitted batch job 6014456",
            dispatch=dispatch_result,
        )

    def fake_collect(_: DispatchResult, **__: object) -> CollectedClusterResults:
        """Return per-model results: two tasks present, one crashed/missing."""

        result_dir = dispatch_result.local_artifact_dir / "results"
        log_dir = dispatch_result.local_artifact_dir / "logs"
        result_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        good_row = ClusterResultRow(
            campaign_id="campaign-a",
            attempt_id="attempt-a",
            assignment_id=good_assign.assignment_id,
            run=_run(
                run_id="run-good",
                stable_id="m-good",
                name="GoodGiant",
                status="passed",
                runner_host="axon-test",
            ),
        )
        bad_row = ClusterResultRow(
            campaign_id="campaign-a",
            attempt_id="attempt-a",
            assignment_id=bad_assign.assignment_id,
            run=_run(
                run_id="run-bad",
                stable_id="m-bad",
                name="BadGiant",
                status="failed",
                forward_pass=0,
                metadata_ok=0,
                n_ops=None,
                peak_rss_mb=None,
                runner_host="axon-test",
                error_class="failed:replay",
                error_message="replay mismatch at op 7",
            ),
        )
        rows_path = dispatch_result.local_artifact_dir / "cluster_results.jsonl"
        manifest_path = dispatch_result.local_artifact_dir / "cluster_results.manifest.json"
        write_result_rows_jsonl((good_row, bad_row), rows_path)
        write_result_manifest((good_row, bad_row), manifest_path)
        (log_dir / "giant_6014456_2.err").write_text(
            "slurmstepd: error: Detected 1 oom-kill event\nMemoryError: tensor alloc\n",
            encoding="utf-8",
        )
        return CollectedClusterResults(
            present_assignments=(good_assign, bad_assign),
            missing_assignments=(crash_assign,),
            result_rows_path=rows_path,
            result_manifest_path=manifest_path,
            result_dir=result_dir,
            log_dir=log_dir,
        )

    monkeypatch.setattr(validate_menagerie.cluster_runner, "dispatch_giants", fail_dispatch)
    monkeypatch.setattr(
        validate_menagerie.cluster_runner, "collect_cluster_results_partial", fake_collect
    )

    args = _args(tmp_path, "--verification-db", str(ledger_db))
    validate_menagerie._run_cluster_validation(
        rows, args, tmp_path / "out", tmp_path / "manifest.tsv"
    )

    with connect(ledger_db) as conn:
        ledger = {
            row["stable_id"]: (row["status"], row["error_class"], row["error_message"])
            for row in conn.execute(
                "SELECT stable_id, status, error_class, error_message FROM current_verification"
            )
        }
    # Per-model attribution: validated, honest validation-failure, honest crash.
    assert ledger["m-good"][0] == "passed"
    assert ledger["m-bad"][0] == "failed"
    assert ledger["m-bad"][1] == "failed:replay"  # honest failure NOT masked
    assert ledger["m-crash"][0] == "failed"
    assert ledger["m-crash"][1] == "failed:cluster_task_failed"
    # The crashed model's message is its OWN .err tail, not the batch message.
    assert "MemoryError" in ledger["m-crash"][2]
    assert "Submitted batch job 6014456" not in ledger["m-crash"][2]
    # Nothing was cascaded to a benign cluster-unavailable skip.
    statuses = {value[0] for value in ledger.values()}
    assert "env_unavailable" not in statuses

    records = validate_menagerie.manifest_records(tmp_path / "manifest.tsv")
    assert records["m-good"]["status"] == "validated"
    assert records["m-bad"]["status"] == "failed:replay"
    assert records["m-crash"]["status"] == "failed:exception"


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
    cold_start = _row(stable_id="cold-start", name="Research 2B MoE")
    peak = _row(stable_id="peak-model", name="MeasuredGiant")
    small = _row(stable_id="small", name="SmallNet")

    routed = validate_menagerie._cluster_route_rows(
        [cold_start, peak, small], "auto", ledger_db=ledger_db
    )

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
    oom = _row(stable_id="oom-model", name="OOMNet")
    crash = _row(stable_id="crash-model", name="CrashNet")

    routed = validate_menagerie._cluster_route_rows([oom, crash], "auto", ledger_db=ledger_db)

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

    monkeypatch.setattr(validate_menagerie, "select_rows", lambda _: [giant])
    monkeypatch.setattr(validate_menagerie.cluster_runner, "dispatch_giants", fail_dispatch)
    monkeypatch.setattr(validate_menagerie, "validate_with_timeout", fake_validate_with_timeout)
    monkeypatch.setattr(validate_menagerie, "write_reports", lambda *_args, **_kwargs: None)

    assert validate_menagerie.run(_args(tmp_path, "--verification-db", str(ledger_db))) == 0

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

    monkeypatch.setattr(validate_menagerie, "select_rows", lambda _: [giant])
    monkeypatch.setattr(validate_menagerie.cluster_runner, "dispatch_giants", fail_dispatch)

    assert (
        validate_menagerie.run(
            _args(tmp_path, "--db", str(catalog_db), "--verification-db", str(ledger_db))
        )
        == 0
    )

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

    monkeypatch.setattr(validate_menagerie, "select_rows", lambda _: [giant])
    monkeypatch.setattr(validate_menagerie.cluster_runner, "dispatch_giants", timeout_dispatch)

    assert (
        validate_menagerie.run(
            _args(tmp_path, "--db", str(catalog_db), "--verification-db", str(ledger_db))
        )
        == 0
    )

    with connect(ledger_db) as conn:
        row = conn.execute(
            "SELECT status, error_message FROM current_verification WHERE stable_id = 'm3635'"
        ).fetchone()
    assert row["status"] == "env_unavailable"
    assert "reason=cluster_timeout" in row["error_message"]


def test_cluster_candidates_exclude_pixi_island_giants() -> None:
    """A pixi-island giant (deps absent on the cluster) is never cluster-routed.

    Bug: m7069 (mmseg:ann) is RAM-classified as a giant but assigned to the
    mmlab_core island. Routing it to the cluster sends it to a node lacking its
    island dependencies. It must validate locally in its island env instead,
    while base-env giants (m3635, m5651) still route to the cluster.
    """

    from menagerie import envs
    from menagerie.catalog import CATALOG_DB
    from menagerie.cluster_runner import is_giant, load_catalog_rows_ro

    rows = load_catalog_rows_ro(CATALOG_DB, stable_ids=["m3635", "m5651", "m7069"])
    assignments = envs.assign(rows)
    by_id = {row.stable_id: row for row in rows}

    # Precondition: m7069 is both a RAM giant AND a non-base island assignment.
    assert is_giant(by_id["m7069"])
    assert assignments["m7069"] != "base"
    assert assignments["m3635"] == "base"
    assert assignments["m5651"] == "base"

    candidates = validate_menagerie._cluster_candidate_rows(rows, "auto", ledger_db=None)
    routed = {row.stable_id for row in candidates}

    assert "m7069" not in routed
    assert routed == {"m3635", "m5651"}

    # Explicit --runner cluster must also exclude the island giant.
    forced = validate_menagerie._cluster_candidate_rows(rows, "cluster", ledger_db=None)
    assert "m7069" not in {row.stable_id for row in forced}
