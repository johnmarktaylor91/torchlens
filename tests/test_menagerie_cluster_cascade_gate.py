"""Golden-chaos regression gate for the cluster batch-cascade failure class.

This is the STAY-FIXED gate for the cluster cascade (FINAL_PLAN bucket E / B0).
Background: a SLURM array ``sbatch --wait`` returns non-zero if ANY single task
fails, so a PARTIAL array failure (some tasks validated, some failed) once
cascaded the batch-level return code onto EVERY dispatched model -- array job
6014456 stamped ~290 models ``failed:cluster_job_failed`` though ~73% had
actually validated. The de-cascade fix (commit ``8e9dd386``) replaced
batch-granularity attribution with per-model attribution driven by each task's
OWN result artifact, via the tolerant ``collect_cluster_results_partial``.

These tests drive a chaotic partial SLURM array end-to-end through the real
``collect_cluster_results_partial`` (NOT a mock) into a real ledger and assert:

* every model is attributed by its OWN task outcome (pass / honest fail / crash),
* a present-but-corrupt artifact is treated as missing (tripwire stays armed),
* the merge-conflict mid-collect residual edge (PREP_FINDINGS 1c) still attributes
  per-model and never blanket-cascades,
* and -- the load-bearing assertion -- the frozen pre-fix
  ``failed:cluster_job_failed`` ``error_class`` is NEVER written by current code.

If a future change reintroduces blanket-cascade attribution, these assertions
fail, so the class cannot silently come back.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import pytest

from menagerie import validate_menagerie
from menagerie.catalog import CatalogRow
from menagerie.cluster_runner import (
    ClusterAssignment,
    ClusterJobFailed,
    ClusterResultRow,
    DispatchResult,
    collect_cluster_results_partial,
    write_result_manifest,
    write_result_rows_jsonl,
)
from menagerie.validate_menagerie import ClusterDispatchHandle
from menagerie.ledger import (
    CASCADE_ARTIFACT_ERROR_CLASS,
    VerificationRun,
    append_verification_run,
    connect,
)

CASCADE_ARTIFACT = Path(__file__).resolve().parents[1] / "menagerie" / "data" / "catalog.db"
pytestmark = pytest.mark.skipif(
    not CASCADE_ARTIFACT.exists(),
    reason="requires local menagerie data artifacts; not part of the portable suite",
)


def _assignment(stable_id: str, array_index: int) -> ClusterAssignment:
    """Build a compact cluster assignment fixture.

    Parameters
    ----------
    stable_id:
        Durable model identity.
    array_index:
        SLURM array task index.

    Returns
    -------
    ClusterAssignment
        Cluster assignment.
    """

    return ClusterAssignment(
        campaign_id="campaign-a",
        attempt_id="attempt-a",
        assignment_id=f"campaign-a:attempt-a:{array_index}:{stable_id}",
        stable_id=stable_id,
        array_index=array_index,
        node_mem_gb=180,
        worker_memory_cap_gb=170,
        partition="nklab",
        reason="unit",
    )


def _row(stable_id: str, index: int, name: str) -> CatalogRow:
    """Build a compact catalog row fixture.

    Parameters
    ----------
    stable_id:
        Durable model identity.
    index:
        Catalog model/display index.
    name:
        Catalog display name.

    Returns
    -------
    CatalogRow
        Catalog row.
    """

    return CatalogRow(
        model_id=index,
        display_index=index,
        stable_id=stable_id,
        name=name,
        variant="",
        family="unit",
        family_normalized="unit",
        domain="unit",
        zoo="unit-zoo",
        constructor_call="torch.nn.Identity()",
        input_shape="(1,)",
        input_dtype="float32",
        era="2026",
        verified=True,
        notes="",
        source="catalog",
        recipe_revision_sha256="recipe-a",
    )


def _run(stable_id: str, name: str, **overrides: object) -> VerificationRun:
    """Build a compact verification run fixture.

    Parameters
    ----------
    stable_id:
        Durable model identity.
    name:
        Catalog display name.
    overrides:
        Verification field overrides.

    Returns
    -------
    VerificationRun
        Verification run.
    """

    data: dict[str, object] = {
        "stable_id": stable_id,
        "recipe_revision_sha256": "recipe-a",
        "name": name,
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
        "run_id": f"run-{stable_id}",
    }
    data.update(overrides)
    return VerificationRun(**data)  # type: ignore[arg-type]


def _write_task_artifact(
    result_dir: Path, assignment: ClusterAssignment, run: VerificationRun
) -> None:
    """Write one task's per-assignment result + manifest artifacts to disk.

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


def _args(tmp_path: Path, ledger_db: Path) -> Any:
    """Parse validator arguments wired to a temporary ledger.

    Parameters
    ----------
    tmp_path:
        Test temporary directory.
    ledger_db:
        Verification ledger path.

    Returns
    -------
    Any
        Parsed argument namespace.
    """

    args = validate_menagerie.build_parser().parse_args(
        [
            "--out-dir",
            str(tmp_path / "out"),
            "--manifest",
            str(tmp_path / "manifest.tsv"),
            "--db",
            str(tmp_path / "catalog.db"),
            "--jobs",
            "1",
            "--verification-db",
            str(ledger_db),
        ]
    )
    # Mirror the post-parse timeout resolution that ``run()`` performs before
    # dispatch (this gate calls ``_run_cluster_validation`` directly).
    args.resolved_timeout_base_sec = (
        args.timeout_base_sec if args.timeout_base_sec is not None else args.timeout_sec
    )
    return args


def _ledger_snapshot(ledger_db: Path) -> dict[str, tuple[str, str | None, str | None]]:
    """Return the current per-model ledger status snapshot.

    Parameters
    ----------
    ledger_db:
        Verification ledger path.

    Returns
    -------
    dict[str, tuple[str, str | None, str | None]]
        Mapping of stable ID to ``(status, error_class, error_message)``.
    """

    with connect(ledger_db) as conn:
        return {
            str(row["stable_id"]): (
                str(row["status"]),
                row["error_class"],
                row["error_message"],
            )
            for row in conn.execute(
                "SELECT stable_id, status, error_class, error_message FROM current_verification"
            )
        }


def _assert_no_cascade_rows_ever(ledger_db: Path) -> None:
    """Assert the frozen pre-fix cascade error_class is never in the ledger.

    Parameters
    ----------
    ledger_db:
        Verification ledger path.
    """

    with connect(ledger_db) as conn:
        cascade_rows = conn.execute(
            "SELECT COUNT(*) FROM verification_runs WHERE error_class = ?",
            (CASCADE_ARTIFACT_ERROR_CLASS,),
        ).fetchone()[0]
    assert cascade_rows == 0, "current code must never write the blanket cascade error_class"


def test_golden_chaos_partial_array_attributes_per_model_not_cascade(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """A chaotic partial SLURM array attributes per-model, never a blanket cascade.

    Drives the REAL ``collect_cluster_results_partial`` against on-disk artifacts
    for a mixed array: one task validated, one honestly failed validation, one
    crashed (no artifact), and one wrote a CORRUPT artifact (treated as missing).
    Every model must land its own honest per-model status and no model may carry
    the frozen ``failed:cluster_job_failed`` cascade marker.
    """

    ledger_db = tmp_path / "verification.db"
    connect(ledger_db).close()
    monkeypatch.setenv("TORCHLENS_MENAGERIE_ENV_HASH", "env-a")
    monkeypatch.setenv("TORCHLENS_MENAGERIE_LOCK_HASH", "lock-a")
    monkeypatch.setenv("TORCHLENS_SOURCE_HASH", "source-a")

    good = _row("m-good", 1, "GoodGiant")
    bad = _row("m-bad", 2, "BadGiant")
    crashed = _row("m-crash", 3, "CrashGiant")
    corrupt = _row("m-corrupt", 4, "CorruptGiant")
    rows = [good, bad, crashed, corrupt]

    good_assign = _assignment("m-good", 0)
    bad_assign = _assignment("m-bad", 1)
    crash_assign = _assignment("m-crash", 2)
    corrupt_assign = _assignment("m-corrupt", 3)
    dispatch_result = DispatchResult(
        campaign_id="campaign-a",
        attempt_id="attempt-a",
        assignments=(good_assign, bad_assign, crash_assign, corrupt_assign),
        local_artifact_dir=tmp_path / "cluster",
        remote_artifact_dir="~/out/campaign-a/attempt-a",
        sbatch_job_ids=("6014456",),
        commands=(),
        dry_run=True,
    )

    def fail_dispatch(*_: object, **__: object) -> DispatchResult:
        """Write per-task artifacts then raise a partial-array job failure."""

        result_dir = dispatch_result.local_artifact_dir / "results"
        log_dir = dispatch_result.local_artifact_dir / "logs"
        result_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        # Task 0 validated; task 1 honestly failed validation (must surface).
        _write_task_artifact(result_dir, good_assign, _run("m-good", "GoodGiant", status="passed"))
        _write_task_artifact(
            result_dir,
            bad_assign,
            _run(
                "m-bad",
                "BadGiant",
                status="failed",
                forward_pass=0,
                metadata_ok=0,
                n_ops=None,
                graph_shape_hash=None,
                peak_rss_mb=None,
                error_class="failed:replay",
                error_message="replay mismatch at op 7",
            ),
        )
        # Task 2 crashed and left ONLY a SLURM .err log (no result artifact).
        (log_dir / "giant_6014456_2.err").write_text(
            "slurmstepd: error: Detected 1 oom-kill event\nMemoryError: tensor alloc\n",
            encoding="utf-8",
        )
        # Task 3 wrote a CORRUPT result (manifest checksum will not verify); the
        # tolerant collector must treat it as missing, NOT silently trust it.
        _write_task_artifact(
            result_dir, corrupt_assign, _run("m-corrupt", "CorruptGiant", status="passed")
        )
        (result_dir / f"{corrupt_assign.assignment_id}.jsonl").write_text(
            "{ this is not valid result jsonl }\n", encoding="utf-8"
        )
        raise ClusterJobFailed(
            ("6014456",),
            "sbatch --wait returncode=1; Submitted batch job 6014456",
            dispatch=dispatch_result,
        )

    # Use the REAL partial collector, only forcing dry_run so no network rsync runs;
    # the present/missing split + per-artifact manifest verification is real.
    real_collect = functools.partial(collect_cluster_results_partial, dry_run=True)
    monkeypatch.setattr(validate_menagerie.cluster_runner, "dispatch_giants", fail_dispatch)
    monkeypatch.setattr(
        validate_menagerie.cluster_runner, "collect_cluster_results_partial", real_collect
    )

    args = _args(tmp_path, ledger_db)
    validate_menagerie._run_cluster_validation(
        rows, args, tmp_path / "out", tmp_path / "manifest.tsv"
    )

    ledger = _ledger_snapshot(ledger_db)
    # Per-model attribution by each task's OWN outcome.
    assert ledger["m-good"][0] == "passed"
    assert ledger["m-bad"] == ("failed", "failed:replay", "replay mismatch at op 7")
    assert ledger["m-crash"][0] == "failed"
    assert ledger["m-crash"][1] == "failed:cluster_task_failed"
    assert "MemoryError" in ledger["m-crash"][2]
    assert "Submitted batch job 6014456" not in ledger["m-crash"][2]
    # A corrupt artifact is treated as missing -> honest per-model failure, never
    # silently trusted as a pass.
    assert ledger["m-corrupt"][0] == "failed"
    assert ledger["m-corrupt"][1] == "failed:cluster_task_failed"

    # The cascade-class invariant: nobody is blanket-failed, and the frozen
    # cascade marker is never written by current code.
    statuses = {value[0] for value in ledger.values()}
    assert statuses == {"passed", "failed"}
    assert "env_unavailable" not in statuses
    _assert_no_cascade_rows_ever(ledger_db)

    records = validate_menagerie.manifest_records(tmp_path / "manifest.tsv")
    assert records["m-good"]["status"] == "validated"
    assert records["m-bad"]["status"] == "failed:replay"
    assert records["m-crash"]["status"] == "failed:exception"
    assert records["m-corrupt"]["status"] == "failed:exception"


def test_golden_chaos_merge_conflict_mid_collect_does_not_cascade(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """A merge-conflict mid-collect still attributes per-model, never cascades.

    Residual edge from PREP_FINDINGS 1c: if ``merge_cluster_results`` raises
    ``ClusterMergeConflict`` mid-merge, ``_surface_cluster_job_failure`` falls back
    to ``_append_cluster_task_failed_rows`` on ALL rows -- but that re-queries the
    already-merged stable IDs, so any pre-merged row is protected and genuinely
    unmerged rows get the honest ``failed:cluster_task_failed``, NEVER the blanket
    ``failed:cluster_job_failed``.

    Concretely: a present cluster result whose ``run_id`` already exists in the
    ledger (without a matching cluster import) collides on merge. The conflict
    must not poison every model; the model with a prior real outcome keeps it and
    the conflicting/unmerged model surfaces an honest per-model cluster failure.
    """

    ledger_db = tmp_path / "verification.db"
    monkeypatch.setenv("TORCHLENS_MENAGERIE_ENV_HASH", "env-a")
    monkeypatch.setenv("TORCHLENS_MENAGERIE_LOCK_HASH", "lock-a")
    monkeypatch.setenv("TORCHLENS_SOURCE_HASH", "source-a")

    # Seed a prior real PASS for m-keep, and a colliding run_id for m-conflict.
    with connect(ledger_db) as conn:
        append_verification_run(
            conn,
            _run("m-keep", "KeepGiant", run_id="run-keep", status="passed"),
        )
        # Pre-existing row whose run_id the incoming cluster result will reuse,
        # WITHOUT a matching cluster import -> forces ClusterMergeConflict.
        append_verification_run(
            conn,
            _run(
                "m-conflict",
                "ConflictGiant",
                run_id="run-collision",
                status="failed",
                forward_pass=0,
                metadata_ok=0,
                n_ops=None,
                graph_shape_hash=None,
                peak_rss_mb=None,
                error_class="failed:exception",
                error_message="prior local failure",
            ),
        )

    keep = _row("m-keep", 1, "KeepGiant")
    conflict = _row("m-conflict", 2, "ConflictGiant")
    rows = [keep, conflict]

    keep_assign = _assignment("m-keep", 0)
    conflict_assign = _assignment("m-conflict", 1)
    dispatch_result = DispatchResult(
        campaign_id="campaign-a",
        attempt_id="attempt-a",
        assignments=(keep_assign, conflict_assign),
        local_artifact_dir=tmp_path / "cluster",
        remote_artifact_dir="~/out/campaign-a/attempt-a",
        sbatch_job_ids=("6014456",),
        commands=(),
        dry_run=True,
    )

    def fail_dispatch(*_: object, **__: object) -> DispatchResult:
        """Write a present result whose run_id collides, then fail the array."""

        result_dir = dispatch_result.local_artifact_dir / "results"
        result_dir.mkdir(parents=True, exist_ok=True)
        # m-conflict's incoming result reuses an existing ledger run_id with a
        # different payload -> merge raises ClusterMergeConflict mid-collect.
        _write_task_artifact(
            result_dir,
            conflict_assign,
            _run(
                "m-conflict",
                "ConflictGiant",
                run_id="run-collision",
                status="passed",
            ),
        )
        raise ClusterJobFailed(
            ("6014456",),
            "sbatch --wait returncode=1; Submitted batch job 6014456",
            dispatch=dispatch_result,
        )

    real_collect = functools.partial(collect_cluster_results_partial, dry_run=True)
    monkeypatch.setattr(validate_menagerie.cluster_runner, "dispatch_giants", fail_dispatch)
    monkeypatch.setattr(
        validate_menagerie.cluster_runner, "collect_cluster_results_partial", real_collect
    )

    args = _args(tmp_path, ledger_db)
    validate_menagerie._run_cluster_validation(
        rows, args, tmp_path / "out", tmp_path / "manifest.tsv"
    )

    ledger = _ledger_snapshot(ledger_db)
    # The merge conflict did not blanket-cascade: m-keep retains its real prior
    # pass; m-conflict surfaces an honest per-model cluster failure (its prior
    # real outcome or an honest cluster_task_failed), NEVER the cascade marker.
    assert ledger["m-keep"][0] == "passed"
    assert ledger["m-conflict"][0] == "failed"
    assert ledger["m-conflict"][1] in {"failed:exception", "failed:cluster_task_failed"}
    _assert_no_cascade_rows_ever(ledger_db)


def test_running_array_at_deadline_does_not_stamp_in_progress_rows_failed(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """C1-1 gate: a still-RUNNING array does not get its in-progress rows failed.

    REVIEW_scheduling.md C1-1 (HIGH): ``collect_cluster_async`` polled
    ``poll_cluster_terminal`` but attributed UNCONDITIONALLY -- so when the local
    lane finished and the poll timed out with the array still RUNNING, every row
    without an artifact yet was stamped ``failed:cluster_task_failed``. A 4-hour
    giant still executing on axon was thus masked as a hard failure.

    This drives the REAL collector against a partial on-disk array where one task
    has FINISHED (artifact present) and one is still RUNNING (no artifact), with
    ``poll_cluster_terminal`` returning False (deadline reached, array not
    terminal). The finished row keeps its real status; the in-progress row is
    LEFT PENDING -- absent from the ledger and never stamped failed -- resumable
    by a later collect.
    """

    ledger_db = tmp_path / "verification.db"
    connect(ledger_db).close()
    monkeypatch.setenv("TORCHLENS_MENAGERIE_ENV_HASH", "env-a")
    monkeypatch.setenv("TORCHLENS_MENAGERIE_LOCK_HASH", "lock-a")
    monkeypatch.setenv("TORCHLENS_SOURCE_HASH", "source-a")

    finished = _row("m-finished", 1, "FinishedGiant")
    running = _row("m-running", 2, "RunningGiant")
    rows = (finished, running)

    finished_assign = _assignment("m-finished", 0)
    running_assign = _assignment("m-running", 1)
    dispatch_result = DispatchResult(
        campaign_id="campaign-a",
        attempt_id="attempt-a",
        assignments=(finished_assign, running_assign),
        local_artifact_dir=tmp_path / "cluster",
        remote_artifact_dir="~/out/campaign-a/attempt-a",
        sbatch_job_ids=("6014999",),
        commands=(),
        dry_run=True,
    )
    # Only the finished task has written its artifact; the running task has none.
    result_dir = dispatch_result.local_artifact_dir / "results"
    (dispatch_result.local_artifact_dir / "logs").mkdir(parents=True, exist_ok=True)
    _write_task_artifact(
        result_dir, finished_assign, _run("m-finished", "FinishedGiant", status="passed")
    )

    # The array is still RUNNING at the collect deadline: poll returns False.
    monkeypatch.setattr(
        validate_menagerie.cluster_runner,
        "poll_cluster_terminal",
        lambda *_, **__: False,
    )
    real_collect = functools.partial(collect_cluster_results_partial, dry_run=True)
    monkeypatch.setattr(
        validate_menagerie.cluster_runner, "collect_cluster_results_partial", real_collect
    )

    args = _args(tmp_path, ledger_db)
    manifest_path = tmp_path / "manifest.tsv"
    handle = ClusterDispatchHandle(rows, dispatch_result, tmp_path / "handle.json")

    validate_menagerie.collect_cluster_async(handle, args, manifest_path)

    ledger = _ledger_snapshot(ledger_db)
    # The finished task keeps its real per-model status.
    assert ledger["m-finished"][0] == "passed"
    # The still-running task is LEFT PENDING: never written, never stamped failed.
    assert "m-running" not in ledger
    _assert_no_cascade_rows_ever(ledger_db)

    # And the manifest never carries a false failure for the in-progress giant.
    records = validate_menagerie.manifest_records(manifest_path)
    assert records["m-finished"]["status"] == "validated"
    assert "m-running" not in records


def test_gate_d_pending_tasks_surface_in_summary(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """GATE (d): cluster tasks left pending surface in the run summary (N1).

    REVIEW_scheduling_fix.md N1: a SLURM array still running at the collect
    deadline leaves its tasks pending (correctly NOT failed); they are absent
    from the manifest, so they must be surfaced separately so the operator knows
    to resume-collect. This drives the real still-running collect, then asserts:

    * a ``cluster_tasks_pending.json`` sidecar is written beside the manifest,
    * ``write_reports`` lifts it into ``validation_summary.json`` with the pending
      stable IDs + a resume hint,
    * run_all's reader exposes the same block.
    """

    import json

    from menagerie.run_all import _cluster_tasks_pending

    ledger_db = tmp_path / "verification.db"
    connect(ledger_db).close()
    monkeypatch.setenv("TORCHLENS_MENAGERIE_ENV_HASH", "env-a")
    monkeypatch.setenv("TORCHLENS_MENAGERIE_LOCK_HASH", "lock-a")
    monkeypatch.setenv("TORCHLENS_SOURCE_HASH", "source-a")

    finished = _row("m-finished", 1, "FinishedGiant")
    running = _row("m-running", 2, "RunningGiant")
    rows = (finished, running)

    finished_assign = _assignment("m-finished", 0)
    running_assign = _assignment("m-running", 1)
    dispatch_result = DispatchResult(
        campaign_id="campaign-a",
        attempt_id="attempt-a",
        assignments=(finished_assign, running_assign),
        local_artifact_dir=tmp_path / "cluster",
        remote_artifact_dir="~/out/campaign-a/attempt-a",
        sbatch_job_ids=("6015050",),
        commands=(),
        dry_run=True,
    )
    result_dir = dispatch_result.local_artifact_dir / "results"
    (dispatch_result.local_artifact_dir / "logs").mkdir(parents=True, exist_ok=True)
    _write_task_artifact(
        result_dir, finished_assign, _run("m-finished", "FinishedGiant", status="passed")
    )

    monkeypatch.setattr(
        validate_menagerie.cluster_runner,
        "poll_cluster_terminal",
        lambda *_, **__: False,
    )
    real_collect = functools.partial(collect_cluster_results_partial, dry_run=True)
    monkeypatch.setattr(
        validate_menagerie.cluster_runner, "collect_cluster_results_partial", real_collect
    )

    # The manifest lives inside out_dir here so write_reports' out_dir read also hits.
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "validation_manifest.tsv"
    args = validate_menagerie.build_parser().parse_args(
        [
            "--out-dir",
            str(out_dir),
            "--manifest",
            str(manifest_path),
            "--db",
            str(tmp_path / "catalog.db"),
            "--jobs",
            "1",
            "--verification-db",
            str(ledger_db),
        ]
    )
    args.resolved_timeout_base_sec = args.timeout_sec
    handle = ClusterDispatchHandle(rows, dispatch_result, tmp_path / "handle.json")

    validate_menagerie.collect_cluster_async(handle, args, manifest_path)

    # 1) The sidecar records the pending task beside the manifest.
    sidecar = out_dir / validate_menagerie.CLUSTER_PENDING_JSON
    assert sidecar.exists()
    sidecar_payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert "m-running" in sidecar_payload["stable_ids"]
    assert "m-finished" not in sidecar_payload["stable_ids"]

    # 2) write_reports lifts it into the validation summary.
    validate_menagerie.write_reports(out_dir, manifest_path, list(rows), no_build_catalog=True)
    summary = json.loads((out_dir / validate_menagerie.SUMMARY_JSON).read_text(encoding="utf-8"))
    pending = summary.get("cluster_tasks_pending")
    assert pending is not None
    assert pending["count"] == 1
    assert pending["stable_ids"] == ["m-running"]
    assert "resume" in pending["resume_hint"].lower()

    # 3) run_all's reader exposes the same block.
    run_all_pending = _cluster_tasks_pending(out_dir)
    assert run_all_pending is not None
    assert run_all_pending["stable_ids"] == ["m-running"]


def test_terminal_array_still_attributes_missing_task_failed(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """The C1-1 guard must NOT mask a genuine cluster failure on a TERMINAL array.

    The narrow carve-out is "do not attribute a still-RUNNING task as failed."
    Once the array is TERMINAL (poll returns True), a task that produced no valid
    result IS a genuine failure and must still surface ``failed:cluster_task_
    failed`` -- the tripwire stays armed.
    """

    ledger_db = tmp_path / "verification.db"
    connect(ledger_db).close()
    monkeypatch.setenv("TORCHLENS_MENAGERIE_ENV_HASH", "env-a")
    monkeypatch.setenv("TORCHLENS_MENAGERIE_LOCK_HASH", "lock-a")
    monkeypatch.setenv("TORCHLENS_SOURCE_HASH", "source-a")

    finished = _row("m-finished", 1, "FinishedGiant")
    failed = _row("m-failed", 2, "FailedGiant")
    rows = (finished, failed)

    finished_assign = _assignment("m-finished", 0)
    failed_assign = _assignment("m-failed", 1)
    dispatch_result = DispatchResult(
        campaign_id="campaign-a",
        attempt_id="attempt-a",
        assignments=(finished_assign, failed_assign),
        local_artifact_dir=tmp_path / "cluster",
        remote_artifact_dir="~/out/campaign-a/attempt-a",
        sbatch_job_ids=("6015000",),
        commands=(),
        dry_run=True,
    )
    result_dir = dispatch_result.local_artifact_dir / "results"
    (dispatch_result.local_artifact_dir / "logs").mkdir(parents=True, exist_ok=True)
    _write_task_artifact(
        result_dir, finished_assign, _run("m-finished", "FinishedGiant", status="passed")
    )
    # m-failed crashed terminally and left no artifact.

    monkeypatch.setattr(
        validate_menagerie.cluster_runner,
        "poll_cluster_terminal",
        lambda *_, **__: True,
    )
    real_collect = functools.partial(collect_cluster_results_partial, dry_run=True)
    monkeypatch.setattr(
        validate_menagerie.cluster_runner, "collect_cluster_results_partial", real_collect
    )

    args = _args(tmp_path, ledger_db)
    manifest_path = tmp_path / "manifest.tsv"
    handle = ClusterDispatchHandle(rows, dispatch_result, tmp_path / "handle.json")

    validate_menagerie.collect_cluster_async(handle, args, manifest_path)

    ledger = _ledger_snapshot(ledger_db)
    assert ledger["m-finished"][0] == "passed"
    # Terminal + no result -> genuine honest failure (tripwire armed).
    assert ledger["m-failed"][0] == "failed"
    assert ledger["m-failed"][1] == "failed:cluster_task_failed"
    _assert_no_cascade_rows_ever(ledger_db)
