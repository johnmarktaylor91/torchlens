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
    REPO_ROOT,
    ClusterAssignment,
    ClusterConfig,
    ClusterJobFailed,
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
    render_sbatch_script,
    run_worker_assignment,
    write_assignment_manifest,
    write_result_manifest,
    write_result_rows_jsonl,
)
from menagerie.ledger import (
    ENV_VERIFICATION_DB,
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
        if command_tuple[:2] == ("ssh", "axon-test") and command_tuple[-1] == "echo $HOME":
            return subprocess.CompletedProcess(command_tuple, 0, stdout="/home/jt3295\n", stderr="")
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
    sbatch_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in result.local_artifact_dir.glob("cluster_runner_*.sbatch")
    )
    assert f"export {ENV_VERIFICATION_DB}=" in sbatch_text
    # SLURM #SBATCH log paths must be absolute (no literal '~'); home is resolved.
    assert "#SBATCH --output=/home/jt3295/out/" in sbatch_text
    assert "#SBATCH --error=/home/jt3295/out/" in sbatch_text
    assert "#SBATCH --output=~/" not in sbatch_text
    # The ledger path here lives outside the repo (like a /tmp smoke ledger), so
    # the worker writes to a node-local ledger under the remote artifact dir.
    assert f'--verification-db "{ledger_db}"' not in sbatch_text
    assert "/worker_ledger/" in sbatch_text


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


def _assignment(stable_id: str, array_index: int, **overrides: object) -> ClusterAssignment:
    """Build a compact cluster assignment fixture.

    Parameters
    ----------
    stable_id:
        Durable model identity.
    array_index:
        SLURM array task index.
    overrides:
        Field overrides for the default assignment.

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


def _write_task_result(
    result_dir: Path, assignment: ClusterAssignment, run: VerificationRun
) -> None:
    """Write one task's per-assignment result + manifest artifacts.

    Parameters
    ----------
    result_dir:
        Directory the worker writes per-task ``*.jsonl`` / ``*.manifest.json`` to.
    assignment:
        The assignment whose task produced the result.
    run:
        The verification run the worker recorded.
    """

    result_dir.mkdir(parents=True, exist_ok=True)
    row = ClusterResultRow(
        campaign_id=assignment.campaign_id,
        attempt_id=assignment.attempt_id,
        assignment_id=assignment.assignment_id,
        run=run,
    )
    write_result_rows_jsonl((row,), result_dir / f"{assignment.assignment_id}.jsonl")
    write_result_manifest((row,), result_dir / f"{assignment.assignment_id}.manifest.json")


def test_collect_partial_attributes_mixed_outcomes_per_task(tmp_path: Path) -> None:
    """A mixed array (some tasks present, some missing) is split per-model.

    This is the regression for the cluster cascade: a non-zero array
    ``sbatch --wait`` must NOT blanket-fail every model. Each present task keeps
    its own validated/honest-failed result; each missing task is reported as a
    missing assignment so the caller can surface its honest per-model failure.
    """

    from menagerie.cluster_runner import collect_cluster_results_partial, read_task_error_log

    good = _assignment("m-good", 0)
    bad_validation = _assignment("m-bad-validation", 1)
    crashed = _assignment("m-crashed", 2)
    dispatch_result = DispatchResult(
        campaign_id="campaign-a",
        attempt_id="attempt-a",
        assignments=(good, bad_validation, crashed),
        local_artifact_dir=tmp_path / "dispatch",
        remote_artifact_dir="~/out/campaign-a/attempt-a",
        sbatch_job_ids=("6014456",),
        commands=(),
        dry_run=True,
    )
    result_dir = dispatch_result.local_artifact_dir / "results"
    log_dir = dispatch_result.local_artifact_dir / "logs"
    # Task 0 validated; task 1 produced an HONEST failed result (a real
    # validation failure must still surface, not be masked); task 2 crashed and
    # left only a SLURM .err log with no result artifact.
    _write_task_result(
        result_dir,
        good,
        _run(run_id="run-good", stable_id="m-good", status="passed"),
    )
    _write_task_result(
        result_dir,
        bad_validation,
        _run(
            run_id="run-bad",
            stable_id="m-bad-validation",
            status="failed",
            forward_pass=0,
            metadata_ok=0,
            n_ops=None,
            error_class="failed:replay",
            error_message="replay mismatch at op 3",
        ),
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "giant_6014456_2.err").write_text(
        "Traceback (most recent call last):\nMemoryError: out of memory allocating tensor\n",
        encoding="utf-8",
    )

    collected = collect_cluster_results_partial(dispatch_result, dry_run=True)

    present_ids = {assignment.stable_id for assignment in collected.present_assignments}
    missing_ids = {assignment.stable_id for assignment in collected.missing_assignments}
    # The validated AND the honestly-failed task both come back as present rows
    # (the honest failure is NOT masked); only the crashed task is missing.
    assert present_ids == {"m-good", "m-bad-validation"}
    assert missing_ids == {"m-crashed"}
    assert collected.result_rows_path is not None
    assert collected.result_manifest_path is not None
    # The crashed task's real error is recoverable from its own .err log, so its
    # honest per-model message is its actual failure, not the batch message.
    err_tail = read_task_error_log(collected.log_dir, ("6014456",), 2)
    assert err_tail is not None
    assert "MemoryError" in err_tail


def test_collect_partial_rejects_corrupt_artifact_as_missing(tmp_path: Path) -> None:
    """A present-but-corrupt result artifact is treated as missing, not trusted."""

    from menagerie.cluster_runner import collect_cluster_results_partial

    good = _assignment("m-good", 0)
    corrupt = _assignment("m-corrupt", 1)
    dispatch_result = DispatchResult(
        campaign_id="campaign-a",
        attempt_id="attempt-a",
        assignments=(good, corrupt),
        local_artifact_dir=tmp_path / "dispatch",
        remote_artifact_dir="~/out/campaign-a/attempt-a",
        sbatch_job_ids=("6014456",),
        commands=(),
        dry_run=True,
    )
    result_dir = dispatch_result.local_artifact_dir / "results"
    _write_task_result(
        result_dir, good, _run(run_id="run-good", stable_id="m-good", status="passed")
    )
    # Corrupt task: a manifest whose checksum no longer matches the rows.
    _write_task_result(
        result_dir, corrupt, _run(run_id="run-corrupt", stable_id="m-corrupt", status="passed")
    )
    corrupt_manifest = result_dir / f"{corrupt.assignment_id}.manifest.json"
    payload = json.loads(corrupt_manifest.read_text(encoding="utf-8"))
    payload["assignments"][0]["result_checksum"] = "tampered"
    corrupt_manifest.write_text(json.dumps(payload), encoding="utf-8")

    collected = collect_cluster_results_partial(dispatch_result, dry_run=True)

    assert {a.stable_id for a in collected.present_assignments} == {"m-good"}
    assert {a.stable_id for a in collected.missing_assignments} == {"m-corrupt"}


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
        verification_db=ledger_db,
        command_runner=fake_runner,
    )

    assert commands
    assert "--no-build-catalog" in commands[0]
    assert "--verification-db" in commands[0]
    assert str(ledger_db) in commands[0]
    # The worker validates in-place; it must force local + base-env-only so the
    # child validator does not re-route the giant back to the cluster (an
    # infinite nested dispatch that records env_unavailable instead of running).
    worker_cmd = commands[0]
    runner_idx = worker_cmd.index("--runner")
    assert worker_cmd[runner_idx + 1] == "local"
    assert "--base-env-only" in worker_cmd


def _giant_assignment() -> ClusterAssignment:
    """Return a base-env giant assignment fixture."""

    return ClusterAssignment(
        "campaign-a", "attempt-a", "assign-a", "m3635", 1, 180, 170, "nklab", "seed"
    )


def test_render_sbatch_expands_tilde_in_slurm_log_directives() -> None:
    """SLURM #SBATCH log paths are absolute; SLURM does not expand '~' itself."""

    config = ClusterConfig(remote_home="/home/jt3295")
    script = render_sbatch_script(
        [_giant_assignment()],
        config=config,
        remote_artifact_dir="~/projects/torchlens/.cluster_runner/c/att",
        verification_db=Path("/tmp/smoke/verification_smoke.db"),
    )

    assert "#SBATCH --output=~/" not in script
    assert "#SBATCH --error=~/" not in script
    assert (
        "#SBATCH --output=/home/jt3295/projects/torchlens/.cluster_runner/c/att/logs/"
        "giant_%A_%a.log" in script
    )
    assert (
        "#SBATCH --error=/home/jt3295/projects/torchlens/.cluster_runner/c/att/logs/"
        "giant_%A_%a.err" in script
    )


def test_render_sbatch_redirects_offrepo_worker_ledger_to_node_local_path() -> None:
    """A smoke /tmp ledger (absent on the node) is redirected to a node-local path."""

    config = ClusterConfig(remote_home="/home/jt3295")
    script = render_sbatch_script(
        [_giant_assignment()],
        config=config,
        remote_artifact_dir="~/projects/torchlens/.cluster_runner/c/att",
        verification_db=Path("/tmp/smoke/verification_smoke.db"),
    )

    expected = (
        "/home/jt3295/projects/torchlens/.cluster_runner/c/att/worker_ledger/verification_smoke.db"
    )
    assert f'--verification-db "{expected}"' in script
    assert f'export TORCHLENS_MENAGERIE_VERIFICATION_DB="{expected}"' in script
    # No bash-unsafe quoted tilde survives into the worker ledger usages.
    assert '"/tmp/smoke/verification_smoke.db"' not in script


def test_render_sbatch_reroots_repo_ledger_to_remote_repo() -> None:
    """A repo-relative ledger maps to the REMOTE repo root, not the local path."""

    config = ClusterConfig(remote_home="/home/jt3295", remote_repo="~/projects/torchlens")
    repo_ledger = REPO_ROOT / "menagerie" / "data" / "verification.db"
    script = render_sbatch_script(
        [_giant_assignment()],
        config=config,
        remote_artifact_dir="~/projects/torchlens/.cluster_runner/c/att",
        verification_db=repo_ledger,
    )

    expected = "/home/jt3295/projects/torchlens/menagerie/data/verification.db"
    assert f'--verification-db "{expected}"' in script
    assert str(repo_ledger) not in script


def test_render_sbatch_bakes_lock_hash_without_remote_python() -> None:
    """The lock hash is baked in; no bare remote `python` runs before the pixi env.

    The cluster nodes ship only Python 2.7 as ``python`` (no ``python3``), so a
    remote ``python -m menagerie.cluster_runner lock-hash`` invocation raised a
    SyntaxError and failed the job before any validation ran. The hash is now
    computed locally from the committed lock files and emitted literally.
    """

    from menagerie.cluster_runner import compute_lock_hash_for_env

    config = ClusterConfig(remote_home="/home/jt3295", pixi_env="misc")
    script = render_sbatch_script(
        [_giant_assignment()],
        config=config,
        remote_artifact_dir="~/projects/torchlens/.cluster_runner/c/att",
        verification_db=Path("/tmp/smoke/verification_smoke.db"),
    )

    expected_hash = compute_lock_hash_for_env("misc")
    assert f'LOCK_HASH="{expected_hash}"' in script
    # No command substitution runs a bare remote python before the pixi env exists.
    assert "$(python -m menagerie.cluster_runner lock-hash" not in script
    assert "LOCK_HASH=$(" not in script


def test_render_sbatch_invokes_pixi_by_absolute_path() -> None:
    """Pixi is invoked by absolute staged path; the cluster has no pixi on PATH."""

    config = ClusterConfig(
        remote_home="/home/jt3295", remote_pixi_bin="~/.cache/torchlens/bin/pixi"
    )
    script = render_sbatch_script(
        [_giant_assignment()],
        config=config,
        remote_artifact_dir="~/projects/torchlens/.cluster_runner/c/att",
        verification_db=Path("/tmp/smoke/verification_smoke.db"),
    )

    assert 'PIXI_BIN="/home/jt3295/.cache/torchlens/bin/pixi"' in script
    assert '"$PIXI_BIN" install --manifest-path' in script
    assert '"$PIXI_BIN" run --manifest-path' in script
    # No bare `pixi install` / `pixi run` that would hit an absent PATH entry.
    assert "\npixi install" not in script
    assert "\npixi run" not in script


def test_dispatch_stages_pixi_binary_to_cluster(tmp_path: Path, monkeypatch: Any) -> None:
    """Dispatch rsyncs the local pixi binary to the cluster before submitting."""

    from menagerie import cluster_runner

    fake_pixi = tmp_path / "pixi"
    fake_pixi.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setattr(cluster_runner, "_local_pixi_bin", lambda: fake_pixi)

    catalog_db = _write_catalog(
        tmp_path, [_row(model_id=1, stable_id="m3635", name="beit_large_patch16_512")]
    )
    ledger_db = tmp_path / "verification.db"
    connect(ledger_db).close()
    commands: list[tuple[str, ...]] = []

    def fake_runner(command: Any) -> subprocess.CompletedProcess[str]:
        """Capture commands; resolve $HOME and accept sbatch."""

        command_tuple = tuple(str(item) for item in command)
        commands.append(command_tuple)
        if command_tuple[-1] == "echo $HOME":
            return subprocess.CompletedProcess(command_tuple, 0, stdout="/home/jt3295\n", stderr="")
        stdout = "Submitted batch job 999\n" if "sbatch" in command_tuple[-1] else ""
        return subprocess.CompletedProcess(command_tuple, 0, stdout=stdout, stderr="")

    dispatch_giants(
        ["m3635"],
        catalog_db=catalog_db,
        ledger_db=ledger_db,
        repo_root=tmp_path,
        local_artifact_root=tmp_path / "cluster",
        config=ClusterConfig(host="axon-test", remote_home="/home/jt3295"),
        command_runner=fake_runner,
        campaign_id="campaign-a",
        attempt_id="attempt-a",
    )

    staged = [
        command
        for command in commands
        if command[0] == "rsync" and command[-1].endswith(":~/.cache/torchlens/bin/pixi")
    ]
    assert staged == [("rsync", "-az", str(fake_pixi), "axon-test:~/.cache/torchlens/bin/pixi")]


def test_render_sbatch_requires_remote_home() -> None:
    """Rendering without a resolved remote home fails loud (no silent literal '~')."""

    with pytest.raises(ValueError, match="remote_home is required"):
        render_sbatch_script(
            [_giant_assignment()],
            config=ClusterConfig(),
            remote_artifact_dir="~/out",
            verification_db=Path("/tmp/x.db"),
        )


def test_dispatch_submitted_then_failed_raises_cluster_job_failed(tmp_path: Path) -> None:
    """A submitted job that runs and exits non-zero raises ClusterJobFailed, not a skip."""

    catalog_db = _write_catalog(
        tmp_path, [_row(model_id=1, stable_id="m3635", name="beit_large_patch16_512")]
    )
    ledger_db = tmp_path / "verification.db"
    connect(ledger_db).close()

    def fake_runner(command: Any) -> subprocess.CompletedProcess[str]:
        """Resolve $HOME, then fail the sbatch --wait after the job was accepted."""

        command_tuple = tuple(str(item) for item in command)
        if command_tuple[-1] == "echo $HOME":
            return subprocess.CompletedProcess(command_tuple, 0, stdout="/home/jt3295\n", stderr="")
        if "sbatch" in command_tuple[-1]:
            # sbatch --wait printed a job ID (accepted) but the array task failed.
            raise subprocess.CalledProcessError(
                1,
                command_tuple,
                output="Submitted batch job 6013771\n",
                stderr="slurmstepd: task 0 exited with non-zero status",
            )
        return subprocess.CompletedProcess(command_tuple, 0, stdout="", stderr="")

    with pytest.raises(ClusterJobFailed) as excinfo:
        dispatch_giants(
            ["m3635"],
            catalog_db=catalog_db,
            ledger_db=ledger_db,
            repo_root=tmp_path,
            local_artifact_root=tmp_path / "cluster",
            config=ClusterConfig(host="axon-test", remote_home="/home/jt3295"),
            command_runner=fake_runner,
            campaign_id="campaign-a",
            attempt_id="attempt-a",
        )

    assert excinfo.value.job_ids == ("6013771",)
    assert "returncode=1" in excinfo.value.detail
    assert excinfo.value.dispatch is not None


def test_dispatch_submit_rejected_propagates_transport_error(tmp_path: Path) -> None:
    """A genuine submit rejection (no job ID) stays a transport error, not a job failure."""

    catalog_db = _write_catalog(
        tmp_path, [_row(model_id=1, stable_id="m3635", name="beit_large_patch16_512")]
    )
    ledger_db = tmp_path / "verification.db"
    connect(ledger_db).close()

    def fake_runner(command: Any) -> subprocess.CompletedProcess[str]:
        """Resolve $HOME, then reject the submission outright."""

        command_tuple = tuple(str(item) for item in command)
        if command_tuple[-1] == "echo $HOME":
            return subprocess.CompletedProcess(command_tuple, 0, stdout="/home/jt3295\n", stderr="")
        if "sbatch" in command_tuple[-1]:
            raise subprocess.CalledProcessError(
                1, command_tuple, output="", stderr="sbatch: error: invalid partition"
            )
        return subprocess.CompletedProcess(command_tuple, 0, stdout="", stderr="")

    with pytest.raises(subprocess.CalledProcessError):
        dispatch_giants(
            ["m3635"],
            catalog_db=catalog_db,
            ledger_db=ledger_db,
            repo_root=tmp_path,
            local_artifact_root=tmp_path / "cluster",
            config=ClusterConfig(host="axon-test", remote_home="/home/jt3295"),
            command_runner=fake_runner,
            campaign_id="campaign-a",
            attempt_id="attempt-a",
        )
