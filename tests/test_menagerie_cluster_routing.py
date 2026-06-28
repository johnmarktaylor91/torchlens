"""Tests for automatic menagerie cluster routing integration."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from menagerie import validate_menagerie
from menagerie.catalog import SOURCE_JSONL, CatalogRow, write_catalog
from menagerie.cluster_runner import (
    ClusterAssignment,
    ClusterConfig,
    CollectedClusterResults,
    DispatchResult,
    MergeReport,
    NodeTier,
    probe_local_gpu_vram_bytes,
    render_sbatch_script,
    requires_cuda,
    route_resources,
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


def test_resource_route_plain_small_row_is_local_cpu() -> None:
    """Rows without CUDA or RAM-giant evidence stay on local CPU."""

    route = route_resources(_row(stable_id="m-small"), ledger={})

    assert route.lane == "local-cpu"
    assert route.device == "cpu"
    assert route.cluster is False


def test_resource_route_cuda_required_fits_local_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """CUDA-required rows use local GPU only when the estimate fits local VRAM."""

    from menagerie import cluster_runner

    monkeypatch.setattr(cluster_runner, "REQUIRES_CUDA", {"m-cuda": "unit cuda"})
    route = route_resources(
        _row(stable_id="m-cuda", name="CudaNet 100M"),
        ledger={},
        local_gpu_vram_bytes=11 * 1024**3,
    )

    assert route.lane == "local-gpu"
    assert route.device == "cuda"
    assert route.cluster is False


def test_catalog_cuda_required_ids_route_to_local_rtx_2080_ti() -> None:
    """Catalog-confirmed CUDA-only recipes route to the local 11 GiB GPU."""

    cuda_ids = {
        "m4921",
        "m4922",
        "m4928",
        "m4932",
        "m5624",
        "m5625",
        "m5626",
        "m11955",
        "m11956",
    }

    for stable_id in cuda_ids:
        row = _row(stable_id=stable_id, name="CudaOnly 100M")
        route = route_resources(row, ledger={}, local_gpu_vram_bytes=11 * 1024**3)

        assert requires_cuda(row)
        assert route.lane == "local-gpu"
        assert route.device == "cuda"
        assert route.cluster is False


def test_cuda_required_catalog_recipes_request_cuda() -> None:
    """The hard-CUDA recipe records carry explicit CUDA device metadata."""

    expected = {
        ("fla_gated_deltanet", ""),
        ("fla_gated_deltanet2", ""),
        ("fla_gated_deltaproduct", ""),
        ("fla_gla", ""),
        ("lightweight_gan_discriminator", ""),
        ("lightweight_gan_generator", ""),
        ("lightweight_gan_simple_decoder", ""),
        ("lightweight_gan_discriminator", "variant-2"),
        ("lightweight_gan_generator", "variant-2"),
    }
    records = {
        (record["name"], record.get("variant", "")): record
        for record in (json.loads(line) for line in SOURCE_JSONL.read_text().splitlines())
        if (record["name"], record.get("variant", "")) in expected
    }

    assert set(records) == expected
    assert all(record["recipe"].get("device_requested") == "cuda" for record in records.values())


def test_resource_route_cuda_required_no_fit_uses_cluster_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CUDA-required rows that do not fit locally route to the GPU cluster lane."""

    from menagerie import cluster_runner

    monkeypatch.setattr(cluster_runner, "REQUIRES_CUDA", {"m-cuda-big": "unit cuda"})
    route = route_resources(
        _row(stable_id="m-cuda-big", name="CudaNet 70B"),
        ledger={},
        local_gpu_vram_bytes=11 * 1024**3,
    )
    no_vram_route = route_resources(
        _row(stable_id="m-cuda-big", name="CudaNet 70B"),
        ledger={},
        local_gpu_vram_bytes=None,
    )

    assert route.lane == "cluster-gpu"
    assert route.device == "cuda"
    assert route.cluster is True
    assert no_vram_route.lane == "cluster-gpu"


def test_resource_route_cuda_giant_prefers_cluster_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CUDA-required RAM giants route cluster-gpu before the RAM cluster rule."""

    from menagerie import cluster_runner

    monkeypatch.setattr(cluster_runner, "REQUIRES_CUDA", {"m4527": "unit cuda"})
    route = route_resources(
        _row(stable_id="m4527", name="effdet_tf_efficientdet_d7"),
        ledger={},
        local_gpu_vram_bytes=None,
    )

    assert route.lane == "cluster-gpu"
    assert route.device == "cuda"
    assert route.cluster is True


def test_resource_route_giant_only_uses_cluster_ram() -> None:
    """RAM giants without CUDA requirement route to the CPU RAM cluster lane."""

    route = route_resources(_row(stable_id="m4527", name="effdet_tf_efficientdet_d7"), ledger={})

    assert route.lane == "cluster-ram"
    assert route.device == "cpu"
    assert route.cluster is True


def test_resource_route_local_first_ignores_large_vram_estimate() -> None:
    """Large-looking CPU-eligible rows are never escalated by the VRAM estimate."""

    route = route_resources(
        _row(stable_id="m-huge-local", name="HugeLocalNet 70B"),
        ledger={},
        local_gpu_vram_bytes=11 * 1024**3,
    )

    assert route.lane == "local-cpu"
    assert route.device == "cpu"
    assert route.cluster is False


def test_gpu_tier_sbatch_requests_gres_but_ram_tier_does_not() -> None:
    """Only GPU-tier sbatch scripts include a GPU GRES directive."""

    config = ClusterConfig(
        remote_home="/home/unit",
        gpu_node_tier=NodeTier(180, 170, "gpu-test", 130, gpu=True),
    )
    ram_assignment = _assignment("m-ram", 0)
    gpu_assignment = _assignment("m-gpu", 0, partition="gpu-test", gpu=True)

    ram_script = render_sbatch_script(
        [ram_assignment],
        config=config,
        remote_artifact_dir="~/cluster/campaign/attempt",
        verification_db=Path("/tmp/verification.db"),
    )
    gpu_script = render_sbatch_script(
        [gpu_assignment],
        config=config,
        remote_artifact_dir="~/cluster/campaign/attempt",
        verification_db=Path("/tmp/verification.db"),
    )

    assert "#SBATCH --gres=gpu:1" not in ram_script
    assert "#SBATCH --gres=gpu:1" in gpu_script


def test_probe_local_gpu_vram_bytes_parses_nvidia_smi_and_handles_missing() -> None:
    """VRAM probe returns bytes on a MiB line and None when nvidia-smi is absent."""

    def ok_runner(command: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
        """Return fake nvidia-smi output."""

        return subprocess.CompletedProcess(command, 0, stdout="11264\n", stderr="")

    def missing_runner(command: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
        """Simulate missing nvidia-smi."""

        raise FileNotFoundError(command[0])

    assert probe_local_gpu_vram_bytes(ok_runner) == 11264 * 1024**2
    assert probe_local_gpu_vram_bytes(missing_runner) is None


def test_resource_routing_does_not_initialize_torch_cuda() -> None:
    """Route calculation must not initialize CUDA in the orchestrator process."""

    torch = pytest.importorskip("torch")
    cuda_initialized = torch.cuda.is_initialized()

    route_resources(_row(stable_id="m-small"), ledger={}, local_gpu_vram_bytes=None)

    assert torch.cuda.is_initialized() is cuda_initialized


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

    giant = _row(stable_id="m4527", name="effdet_tf_efficientdet_d7")
    local = _row(model_id=2, display_index=2, stable_id="m_local", name="SmallNet")
    dispatch_calls: list[tuple[str, ...]] = []
    local_calls: list[tuple[str, str]] = []
    merge_calls: list[tuple[Path, Path]] = []
    call_log: list[str] = []

    def fake_dispatch(stable_ids: list[str], **_: object) -> DispatchResult:
        """Capture cluster dispatches."""

        call_log.append("dispatch")
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

        call_log.append("collect")
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
        *args: object,
        **__: object,
    ) -> validate_menagerie.ValidationResult:
        """Capture local validation calls."""

        call_log.append("local")
        local_calls.append((row.stable_id, str(args[2])))
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
        validate_menagerie.cluster_runner,
        "poll_cluster_terminal",
        lambda *_args, **_kwargs: True,
    )
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

    assert dispatch_calls == [("m4527",)]
    assert local_calls == [("m_local", "cpu")]
    assert call_log == ["dispatch", "local", "collect"]
    assert len(merge_calls) == 1
    records = validate_menagerie.manifest_records(tmp_path / "manifest.tsv")
    assert set(records) == {"m4527", "m_local"}


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


def test_local_first_routes_measured_giant_keeps_unmeasured_and_small_local(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """LOCAL-FIRST: only a measured >=115 GiB peak routes; unmeasured/small stay local.

    The shared axon cluster is opt-IN, not opt-OUT. A model routes there ONLY with
    hard measured evidence it cannot fit locally:

    * a MEASURED peak RSS at/above the usable-local-RAM threshold -> cluster,
    * a model with a measured peak BELOW the threshold -> local,
    * an UNMEASURED model -> local (never preemptively shipped to the cluster),
      even when its name/params/shape "look" giant.
    """

    ledger_db = tmp_path / "verification.db"
    with connect(ledger_db) as conn:
        append_verification_run(
            conn,
            _run(
                run_id="giant-run",
                stable_id="measured-giant",
                peak_rss_mb=130 * validate_menagerie.MB_PER_GB,
            ),
        )
        append_verification_run(
            conn,
            _run(
                run_id="fits-run",
                stable_id="fits-model",
                peak_rss_mb=90 * validate_menagerie.MB_PER_GB,
            ),
        )
    measured_giant = _row(stable_id="measured-giant", name="MeasuredGiant")
    fits_local = _row(stable_id="fits-model", name="FitsLocally")
    # Unmeasured AND giant-LOOKING by name/params/shape -- must still stay local.
    unmeasured = _row(
        stable_id="unmeasured",
        name="Research 2B MoE longcat",
        input_shape="(1, 3, 1024, 1024)",
    )
    small = _row(stable_id="small", name="SmallNet")

    routed = validate_menagerie._cluster_route_rows(
        [measured_giant, fits_local, unmeasured, small], "auto", ledger_db=ledger_db
    )

    assert [row.stable_id for row in routed] == ["measured-giant"]


def test_local_first_routes_cluster_measured_peak_as_nonfit_evidence(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """A >=115 GiB peak measured on the cluster is genuine local-nonfit evidence.

    The latest terminal row may be a smaller cluster-validated peak; the MAX over
    history is what proves the model is too large for the workstation, so it must
    still route to the cluster on a resume.
    """

    ledger_db = tmp_path / "verification.db"
    with connect(ledger_db) as conn:
        # Earlier cluster run measured a 200 GiB peak (proof of nonfit) ...
        append_verification_run(
            conn,
            _run(
                run_id="cluster-peak",
                stable_id="cluster-giant",
                runner_host="ax17.rc.zi.columbia.edu",
                peak_rss_mb=200 * validate_menagerie.MB_PER_GB,
                started_at="2026-06-25T00:00:00+00:00",
            ),
        )
        # ... and the LATEST terminal row carries a smaller cluster peak.
        append_verification_run(
            conn,
            _run(
                run_id="cluster-peak-2",
                stable_id="cluster-giant",
                runner_host="ax17.rc.zi.columbia.edu",
                peak_rss_mb=80 * validate_menagerie.MB_PER_GB,
                started_at="2026-06-25T01:00:00+00:00",
            ),
        )
    cluster_giant = _row(stable_id="cluster-giant", name="ClusterMeasuredGiant")

    routed = validate_menagerie._cluster_route_rows([cluster_giant], "auto", ledger_db=ledger_db)

    assert [row.stable_id for row in routed] == ["cluster-giant"]


def test_local_first_escalates_local_ram_failures_only_near_full_ram(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """A prior LOCAL RAM failure escalates; a small-cap kill does NOT.

    * a LOCAL OOM -> cluster,
    * a LOCAL ``failed:memory_cap`` at a cap NEAR full local RAM -> cluster,
    * a LOCAL ``failed:memory_cap`` at a SMALL protective cap -> local (an
      early-sweep low cap proves nothing about 115 GiB feasibility), and
    * a native (non-RAM) crash is local-only.
    """

    monkeypatch.setattr(cluster_runner_socket(), "gethostname", lambda: "test-workstation")
    ledger_db = tmp_path / "verification.db"
    with connect(ledger_db) as conn:
        append_verification_run(
            conn,
            _run(
                run_id="oom-run",
                stable_id="oom-model",
                runner_host="test-workstation",
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
                run_id="bigcap-run",
                stable_id="bigcap-model",
                runner_host="test-workstation",
                status="failed",
                forward_pass=0,
                metadata_ok=0,
                n_ops=None,
                peak_rss_mb=110 * validate_menagerie.MB_PER_GB,
                error_class="failed:memory_cap",
                error_message="worker RSS exceeded --worker-memory-cap-gb=110.000; killed",
            ),
        )
        append_verification_run(
            conn,
            _run(
                run_id="smallcap-run",
                stable_id="smallcap-model",
                runner_host="test-workstation",
                status="failed",
                forward_pass=0,
                metadata_ok=0,
                n_ops=None,
                peak_rss_mb=30 * validate_menagerie.MB_PER_GB,
                error_class="failed:memory_cap",
                error_message="worker RSS exceeded --worker-memory-cap-gb=30.000; killed",
            ),
        )
        append_verification_run(
            conn,
            _run(
                run_id="crash-run",
                stable_id="crash-model",
                runner_host="test-workstation",
                status="native_crash",
                forward_pass=0,
                metadata_ok=0,
                n_ops=None,
                peak_rss_mb=None,
            ),
        )
    oom = _row(stable_id="oom-model", name="OOMNet")
    bigcap = _row(stable_id="bigcap-model", name="BigCapNet")
    smallcap = _row(stable_id="smallcap-model", name="SmallCapNet")
    crash = _row(stable_id="crash-model", name="CrashNet")

    routed = validate_menagerie._cluster_route_rows(
        [oom, bigcap, smallcap, crash], "auto", ledger_db=ledger_db
    )

    assert sorted(row.stable_id for row in routed) == ["bigcap-model", "oom-model"]


def test_local_first_remote_ram_failure_does_not_escalate(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """A memory-cap kill on a CLUSTER host is not local nonfit evidence.

    Only a failure on THIS workstation proves the model cannot run locally; a
    cluster-host memory-cap row (small peak, no >=115 GiB measurement) must keep
    the model local-first.
    """

    monkeypatch.setattr(cluster_runner_socket(), "gethostname", lambda: "test-workstation")
    ledger_db = tmp_path / "verification.db"
    with connect(ledger_db) as conn:
        append_verification_run(
            conn,
            _run(
                run_id="remote-cap",
                stable_id="remote-model",
                runner_host="ax14.rc.zi.columbia.edu",
                status="failed",
                forward_pass=0,
                metadata_ok=0,
                n_ops=None,
                peak_rss_mb=40 * validate_menagerie.MB_PER_GB,
                error_class="failed:memory_cap",
                error_message="worker RSS exceeded --worker-memory-cap-gb=120.000; killed",
            ),
        )
    remote = _row(stable_id="remote-model", name="RemoteCapNet")

    routed = validate_menagerie._cluster_route_rows([remote], "auto", ledger_db=ledger_db)

    assert routed == ()


def test_giant_registry_force_cluster_only_for_genuine_giants() -> None:
    """Only the four measured-nonfit giants keep ``force_cluster=True``.

    Every registry entry whose measured peak fits locally (<115 GiB) must be
    local-first (``force_cluster=False``); preserving a force flag for a fittable
    model is exactly the opt-OUT over-routing the local-first policy removes.
    """

    from menagerie.cluster_runner import (
        GIANT_REGISTRY,
        LOCAL_FIRST_CLUSTER_THRESHOLD_GB,
        MB_PER_GB,
    )

    forced = {sid for sid, entry in GIANT_REGISTRY.items() if entry.force_cluster}
    assert forced == {"m4246", "m4525", "m4526", "m4527"}

    threshold_mb = int(LOCAL_FIRST_CLUSTER_THRESHOLD_GB * MB_PER_GB)
    for sid, entry in GIANT_REGISTRY.items():
        if entry.force_cluster:
            assert entry.measured_peak_rss_mb is not None
            assert entry.measured_peak_rss_mb >= threshold_mb, sid
        elif entry.measured_peak_rss_mb is not None:
            assert entry.measured_peak_rss_mb < threshold_mb, sid


def cluster_runner_socket() -> Any:
    """Return the ``socket`` module object used by the cluster runner.

    The local-RAM-failure check resolves the workstation hostname via
    ``socket.gethostname()`` inside :mod:`menagerie.cluster_runner`; tests patch
    that module's ``socket`` so a fixture run reads as "local".

    Returns
    -------
    Any
        The :mod:`socket` module imported by the cluster runner.
    """

    from menagerie import cluster_runner

    return cluster_runner.socket


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
    giant = _row(stable_id="m4527", name="effdet_tf_efficientdet_d7")
    local_calls: list[str] = []
    monkeypatch.setenv("TORCHLENS_MENAGERIE_ENV_HASH", "env-a")
    monkeypatch.setenv("TORCHLENS_MENAGERIE_LOCK_HASH", "lock-a")
    monkeypatch.setenv("TORCHLENS_SOURCE_HASH", "source-a")
    with connect(ledger_db) as conn:
        append_verification_run(
            conn,
            _run(stable_id="m4527", name="effdet_tf_efficientdet_d7", runner_host="axon-test"),
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
    assert records["m4527"]["status"] == "validated"


def test_cluster_unreachable_writes_terminal_rows_and_continues(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Transport failure records env_unavailable rows instead of aborting validation."""

    ledger_db = tmp_path / "verification.db"
    giant = _row(stable_id="m4527", name="effdet_tf_efficientdet_d7")
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
            "SELECT status, error_message FROM current_verification WHERE stable_id = 'm4527'"
        ).fetchone()
    records = validate_menagerie.manifest_records(tmp_path / "manifest.tsv")
    assert row["status"] == "env_unavailable"
    assert "reason=cluster_unreachable" in row["error_message"]
    assert records["m4527"]["status"] == "skipped:cluster_unavailable"


def test_cluster_timeout_writes_terminal_rows_and_continues(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Blocking sbatch timeouts record terminal rows instead of aborting."""

    ledger_db = tmp_path / "verification.db"
    giant = _row(stable_id="m4527", name="effdet_tf_efficientdet_d7")
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
            "SELECT status, error_message FROM current_verification WHERE stable_id = 'm4527'"
        ).fetchone()
    assert row["status"] == "env_unavailable"
    assert "reason=cluster_timeout" in row["error_message"]


def test_cluster_candidates_exclude_pixi_island_giants() -> None:
    """A pixi-island giant (deps absent on the cluster) is never cluster-routed.

    Bug: m7069 (mmseg:ann) is a non-base island assignment. Even when it is a RAM
    giant by MEASURED evidence (a >=115 GiB peak), routing it to the cluster sends
    it to a node lacking its island dependencies, so it must validate locally in
    its island env instead -- while a base-env forced giant (m4527) still routes
    to the cluster. The island exclusion runs BEFORE the local-first giant check,
    so it holds regardless of how m7069 became a giant.
    """

    from menagerie import envs
    from menagerie.catalog import CATALOG_DB
    from menagerie.cluster_runner import MB_PER_GB, is_giant, load_catalog_rows_ro

    rows = load_catalog_rows_ro(CATALOG_DB, stable_ids=["m4527", "m7069"])
    assignments = envs.assign(rows)
    by_id = {row.stable_id: row for row in rows}

    # Precondition: m7069 is a non-base island assignment, and under local-first it
    # is a RAM giant only by MEASURED evidence (a >=115 GiB peak in the ledger).
    giant_ledger = {"m7069": 130 * MB_PER_GB}
    assert is_giant(by_id["m7069"], ledger=giant_ledger)
    assert assignments["m7069"] != "base"
    assert assignments["m4527"] == "base"

    candidates = validate_menagerie._cluster_candidate_rows(rows, "auto", ledger_db=None)
    routed = {row.stable_id for row in candidates}

    assert "m7069" not in routed
    # m4527 is a base-env forced giant; it still routes to the cluster.
    assert routed == {"m4527"}

    # Explicit --runner cluster must also exclude the island giant.
    forced = validate_menagerie._cluster_candidate_rows(rows, "cluster", ledger_db=None)
    assert "m7069" not in {row.stable_id for row in forced}
