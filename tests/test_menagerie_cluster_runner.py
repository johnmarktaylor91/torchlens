"""Tests for the menagerie SLURM cluster runner."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from menagerie.catalog import CatalogRow, write_catalog
from menagerie.cluster_runner import (
    GIANT_REGISTRY,
    ClusterAssignment,
    ClusterConfig,
    DispatchResult,
    ClusterMergeConflict,
    ClusterResultIntegrityError,
    ClusterResultRow,
    dispatch_giants,
    collect_cluster_results,
    is_giant,
    ledger_completed_stable_ids,
    merge_cluster_results,
    node_tier_for_row,
    pending_assignments_for_resume,
    run_worker_assignment,
    write_assignment_manifest,
    write_result_manifest,
    write_result_rows_jsonl,
)
from menagerie.ledger import (
    LEGACY_UNKNOWN,
    VerificationRun,
    VerificationTarget,
    append_verification_run,
    connect,
)


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
        "runner_host": "axon-test",
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


def _write_catalog(tmp_path: Path, rows: list[CatalogRow]) -> Path:
    """Write a synthetic catalog database.

    Parameters
    ----------
    tmp_path:
        Temporary directory.
    rows:
        Rows to persist.

    Returns
    -------
    pathlib.Path
        Catalog database path.
    """

    catalog_db = tmp_path / "catalog.db"
    write_catalog(rows, canonical_tsv=tmp_path / "catalog.tsv", db_path=catalog_db)
    return catalog_db


def _target(row: CatalogRow) -> VerificationTarget:
    """Build a matching verification target for a row.

    Parameters
    ----------
    row:
        Catalog row.

    Returns
    -------
    VerificationTarget
        Verification target.
    """

    return VerificationTarget(
        recipe_revision_sha256=row.recipe_revision_sha256,
        torchlens_source_hash="source-a",
        env_hash="env-a",
        lock_hash="lock-a",
        device_requested="cpu",
        scope="forward",
    )


def _result_row(run: VerificationRun | None = None) -> ClusterResultRow:
    """Build a cluster result row fixture.

    Parameters
    ----------
    run:
        Optional verification run.

    Returns
    -------
    ClusterResultRow
        Cluster result row.
    """

    return ClusterResultRow(
        campaign_id="campaign-a",
        attempt_id="attempt-a",
        assignment_id="assign-a",
        run=run or _run(),
    )


def _write_results(tmp_path: Path, rows: list[ClusterResultRow]) -> tuple[Path, Path]:
    """Write result JSONL and checksum manifest.

    Parameters
    ----------
    tmp_path:
        Temporary directory.
    rows:
        Result rows.

    Returns
    -------
    tuple[pathlib.Path, pathlib.Path]
        Result rows path and manifest path.
    """

    rows_path = tmp_path / "results.jsonl"
    manifest_path = tmp_path / "results.manifest.json"
    write_result_rows_jsonl(rows, rows_path)
    write_result_manifest(rows, manifest_path)
    return rows_path, manifest_path


def test_static_registry_contains_all_axon_giant_seeds() -> None:
    """The static registry seeds all 17 axon campaign giants."""

    expected = {
        "m920",
        "m2064",
        "m3635",
        "m4165",
        "m4246",
        "m4494",
        "m4495",
        "m4523",
        "m4524",
        "m4525",
        "m4526",
        "m4527",
        "m4797",
        "m4808",
        "m5187",
        "m5651",
        "m11112",
    }

    assert set(GIANT_REGISTRY) == expected
    assert GIANT_REGISTRY["m4246"].node_mem_gb == 250
    assert GIANT_REGISTRY["m5651"].node_mem_gb == 500


def test_is_giant_routes_static_heuristic_oom_and_blocks_native_crash(tmp_path: Path) -> None:
    """Routing honors static seeds, first-contact heuristics, OOMs, and native crashes."""

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
                stable_id="m4246",
                status="native_crash",
                forward_pass=0,
                metadata_ok=0,
                n_ops=None,
                peak_rss_mb=None,
            ),
        )

    assert is_giant(_row(stable_id="m3635", name="beit_large_patch16_512"), ledger=ledger_db)
    assert is_giant(_row(stable_id="oom-model"), ledger=ledger_db)
    assert is_giant(_row(stable_id="new-moe", name="Research 2B MoE"), ledger=ledger_db)
    assert is_giant(_row(stable_id="large-input", input_shape="(1, 3, 1024, 1024)"))
    assert not is_giant(_row(stable_id="m4246", name="depth_pro"), ledger=ledger_db)
    assert not is_giant(_row(stable_id="small", name="SmallNet"), ledger={"small": 4 * 1024})


def test_node_tier_right_sizes_incrementally_with_terabyte_escape() -> None:
    """Node tiers step through registered tiers and allow 1TB for huge peaks."""

    assert node_tier_for_row(_row(stable_id="m3635")).mem_gb == 180
    assert node_tier_for_row(_row(stable_id="m4246")).mem_gb == 250
    assert node_tier_for_row(_row(stable_id="m5651")).mem_gb == 500
    assert node_tier_for_row(_row(stable_id="huge"), ledger={"huge": 700 * 1024}).mem_gb == 1000


def test_repeated_unregistered_moe_oom_escalates_to_terabyte(tmp_path: Path) -> None:
    """Repeated OOMs for unregistered MoE monsters escalate to the largest tier."""

    ledger_db = tmp_path / "verification.db"
    with connect(ledger_db) as conn:
        append_verification_run(
            conn,
            _run(
                run_id="oom-1",
                stable_id="new-moe",
                name="Research 4B MoE",
                status="oom",
                forward_pass=0,
                metadata_ok=0,
                n_ops=None,
                graph_shape_hash=None,
                finished_at="2026-06-25T00:00:01+00:00",
            ),
        )
        append_verification_run(
            conn,
            _run(
                run_id="oom-2",
                stable_id="new-moe",
                name="Research 4B MoE",
                status="oom",
                forward_pass=0,
                metadata_ok=0,
                n_ops=None,
                graph_shape_hash=None,
                finished_at="2026-06-25T00:00:02+00:00",
            ),
        )

    tier = node_tier_for_row(_row(stable_id="new-moe", name="Research 4B MoE"), ledger=ledger_db)

    assert tier.mem_gb == 1000


def test_dispatch_uses_mocked_commands_and_one_sbatch_per_tier(tmp_path: Path) -> None:
    """Dispatch prepares rsync/ssh/sbatch commands without live cluster access."""

    catalog_db = _write_catalog(
        tmp_path,
        [
            _row(model_id=1, stable_id="m3635", name="beit_large_patch16_512"),
            _row(model_id=2, display_index=2, stable_id="m5651", name="longcat_flash"),
        ],
    )
    ledger_db = tmp_path / "verification.db"
    connect(ledger_db).close()
    commands: list[tuple[str, ...]] = []

    def fake_runner(command: Any) -> subprocess.CompletedProcess[str]:
        """Capture a cluster command and return a fake sbatch result."""

        command_tuple = tuple(str(item) for item in command)
        commands.append(command_tuple)
        stdout = "Submitted batch job 12345\n" if "sbatch" in command_tuple[-1] else ""
        return subprocess.CompletedProcess(command_tuple, 0, stdout=stdout, stderr="")

    result = dispatch_giants(
        ["m3635", "m5651"],
        catalog_db=catalog_db,
        ledger_db=ledger_db,
        repo_root=tmp_path,
        local_artifact_root=tmp_path / "cluster",
        config=ClusterConfig(host="axon-test", remote_repo="~/repo", remote_artifact_root="~/out"),
        command_runner=fake_runner,
        campaign_id="campaign-a",
        attempt_id="attempt-a",
    )

    assert result.sbatch_job_ids == ("12345", "12345")
    assert sum("sbatch" in command[-1] for command in commands) == 2
    assert (result.local_artifact_dir / "catalog.db").exists()
    assert (result.local_artifact_dir / "assignments.json").exists()


def test_merge_is_idempotent_and_conflicts_fail_loud(tmp_path: Path) -> None:
    """Merging duplicate rows is idempotent, but conflicting keys raise."""

    ledger_db = tmp_path / "verification.db"
    rows_path, manifest_path = _write_results(tmp_path, [_result_row()])

    report = merge_cluster_results(rows_path, manifest_path, local_ledger_db=ledger_db)
    duplicate = merge_cluster_results(rows_path, manifest_path, local_ledger_db=ledger_db)

    assert report.inserted == 1
    assert report.duplicates == 0
    assert duplicate.inserted == 0
    assert duplicate.duplicates == 1
    with connect(ledger_db) as conn:
        row = conn.execute("SELECT status, runner_host FROM current_verification").fetchone()
    assert tuple(row) == ("passed", "axon-test")

    conflicting = _result_row(_run(error_message="different payload"))
    conflict_rows, conflict_manifest = _write_results(tmp_path / "conflict", [conflicting])
    with pytest.raises(ClusterMergeConflict):
        merge_cluster_results(conflict_rows, conflict_manifest, local_ledger_db=ledger_db)


def test_merge_run_id_collision_is_not_counted_as_duplicate(tmp_path: Path) -> None:
    """A pre-existing run ID without an import marker is a collision, not a duplicate."""

    ledger_db = tmp_path / "verification.db"
    with connect(ledger_db) as conn:
        append_verification_run(conn, _run())
    rows_path, manifest_path = _write_results(tmp_path, [_result_row()])

    with pytest.raises(ClusterMergeConflict, match="without a matching cluster import"):
        merge_cluster_results(rows_path, manifest_path, local_ledger_db=ledger_db)


def test_merge_checksum_mismatch_fails_loud(tmp_path: Path) -> None:
    """A mismatched per-assignment checksum fails before inserting rows."""

    ledger_db = tmp_path / "verification.db"
    rows_path, manifest_path = _write_results(tmp_path, [_result_row()])
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["assignments"][0]["result_checksum"] = "bad-checksum"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ClusterResultIntegrityError):
        merge_cluster_results(rows_path, manifest_path, local_ledger_db=ledger_db)


def test_collect_cluster_results_verifies_and_aggregates_local_artifacts(tmp_path: Path) -> None:
    """Collection verifies per-task artifacts before writing aggregate merge inputs."""

    result_dir = tmp_path / "remote-results"
    _write_results(result_dir, [_result_row()])
    dispatch_result = DispatchResult(
        campaign_id="campaign-a",
        attempt_id="attempt-a",
        assignments=(),
        local_artifact_dir=tmp_path / "dispatch",
        remote_artifact_dir="~/out/campaign-a/attempt-a",
        sbatch_job_ids=(),
        commands=(),
        dry_run=True,
    )

    rows_path, manifest_path = collect_cluster_results(
        dispatch_result,
        local_result_dir=result_dir,
        dry_run=True,
    )

    assert rows_path.exists()
    assert manifest_path.exists()
    report = merge_cluster_results(rows_path, manifest_path, local_ledger_db=tmp_path / "ledger.db")
    assert report.inserted == 1


def test_ledger_keyed_resume_uses_identity_tuple(tmp_path: Path) -> None:
    """Resume derives completion from current ledger identity targets."""

    row_a = _row(stable_id="m1", name="DoneNet")
    row_b = _row(stable_id="m2", name="PendingNet", recipe_revision_sha256="recipe-b")
    ledger_db = tmp_path / "verification.db"
    with connect(ledger_db) as conn:
        append_verification_run(conn, _run(stable_id="m1", name="DoneNet"))
        append_verification_run(
            conn,
            _run(
                run_id="stale-run",
                stable_id="m2",
                name="PendingNet",
                recipe_revision_sha256="stale-recipe",
            ),
        )
    assignments = (
        ClusterAssignment("campaign-a", "attempt-a", "assign-a", "m1", 0, 180, 170, "nklab", ""),
        ClusterAssignment("campaign-a", "attempt-a", "assign-b", "m2", 1, 180, 170, "nklab", ""),
    )
    targets = {"m1": _target(row_a), "m2": _target(row_b)}

    assert ledger_completed_stable_ids(targets, ledger_db=ledger_db) == {"m1"}
    pending = pending_assignments_for_resume(assignments, targets, ledger_db=ledger_db)
    assert [assignment.stable_id for assignment in pending] == ["m2"]


def test_ledger_keyed_resume_excludes_half_migrated_rows(tmp_path: Path) -> None:
    """Half-migrated legacy rows are not counted complete for cluster resume."""

    row = _row(stable_id="m1")
    ledger_db = tmp_path / "verification.db"
    with connect(ledger_db) as conn:
        append_verification_run(
            conn,
            _run(
                stable_id="m1",
                torchlens_source_hash=LEGACY_UNKNOWN,
                lock_hash="lock-a",
            ),
        )
    targets = {
        "m1": VerificationTarget(
            recipe_revision_sha256=row.recipe_revision_sha256,
            torchlens_source_hash=LEGACY_UNKNOWN,
            env_hash="env-a",
            lock_hash="lock-a",
            device_requested="cpu",
            scope="forward",
        )
    }

    assert ledger_completed_stable_ids(targets, ledger_db=ledger_db) == set()


def test_worker_forwards_no_build_catalog_to_validator(tmp_path: Path) -> None:
    """Cluster workers invoke validation with the read-only no-build catalog flag."""

    assignment = ClusterAssignment(
        "campaign-a",
        "attempt-a",
        "assign-a",
        "m1",
        0,
        180,
        170,
        "nklab",
        "unit",
    )
    manifest_path = tmp_path / "assignments.json"
    write_assignment_manifest((assignment,), manifest_path)
    ledger_db = tmp_path / "menagerie" / "data" / "verification.db"
    with connect(ledger_db) as conn:
        append_verification_run(conn, _run())
    commands: list[tuple[str, ...]] = []

    def fake_runner(command: Any) -> subprocess.CompletedProcess[str]:
        """Capture the worker validator command."""

        command_tuple = tuple(str(item) for item in command)
        commands.append(command_tuple)
        return subprocess.CompletedProcess(command_tuple, 0, stdout="", stderr="")

    run_worker_assignment(
        manifest_path,
        0,
        repo_root=tmp_path,
        result_dir=tmp_path / "results",
        command_runner=fake_runner,
    )

    assert commands
    assert "--no-build-catalog" in commands[0]
