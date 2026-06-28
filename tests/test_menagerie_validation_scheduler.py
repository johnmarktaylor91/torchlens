"""Tests for menagerie validation scheduler admission policy."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from menagerie.catalog import CatalogRow
from menagerie.ledger import VerificationRun, append_verification_run, connect
from menagerie.cluster_runner import ResourceRoute
from menagerie.runtime import DependencyPlan
from menagerie.validate_menagerie import (
    BIG_MODEL_THRESHOLD_MB,
    LEDGER_MEMORY_HEADROOM,
    MAX_CONCURRENT_BIG_MODELS,
    MB_PER_GB,
    MEMORY_BUDGET_HEADROOM_FRACTION,
    TIMEOUT_CEILING_FLOOR_SEC,
    TIMEOUT_CEILING_MIN_SEC,
    TIMEOUT_DEFAULT_SCALE,
    TIMEOUT_GENEROUS_MIN_SEC,
    SmokeCaseSettings,
    ValidationResult,
    ValidationWorkItem,
    _admit_memory_budgeted_items,
    _actual_available_memory_mb,
    _case_timeout,
    _lpt_sort_key,
    _memory_estimate_for_row,
    _resolve_memory_budget_gb,
    _resolve_row_device,
    default_worker_torch_threads,
    default_validation_jobs,
    latest_scheduler_memory_estimates,
    resolve_worker_torch_threads,
    resolve_timeout_ceiling_sec,
    validate_one,
    validate_with_timeout,
)


def _row(stable_id: str = "m1", name: str = "UnitNet") -> CatalogRow:
    """Build a compact catalog row fixture.

    Parameters
    ----------
    stable_id:
        Stable model identity.
    name:
        Model name.

    Returns
    -------
    CatalogRow
        Catalog row.
    """

    return CatalogRow(
        model_id=1,
        display_index=1,
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


def _plan() -> DependencyPlan:
    """Build a compact dependency plan fixture.

    Returns
    -------
    DependencyPlan
        Dependency plan.
    """

    return DependencyPlan(
        cluster_key="unit",
        packages=(),
        top_modules=(),
        environment="unit",
    )


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
        "runner_host": "workstation",
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


def _item(estimated_gb: int, stable_id: str = "m1", name: str = "UnitNet") -> ValidationWorkItem:
    """Build a validation work-item fixture.

    Parameters
    ----------
    estimated_gb:
        Estimated peak RSS in GB.
    stable_id:
        Stable model identity.
    name:
        Model name.

    Returns
    -------
    ValidationWorkItem
        Validation work item.
    """

    return ValidationWorkItem(
        plan=_plan(),
        row=_row(stable_id=stable_id, name=name),
        estimated_memory_mb=estimated_gb * MB_PER_GB,
        estimate_source="default",
    )


def test_actual_free_memory_gate_throttles_when_psutil_available_is_low(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mocked low psutil availability blocks admission despite estimate headroom."""

    pending = [_item(estimated_gb=4)]
    monkeypatch.setitem(
        sys.modules,
        "psutil",
        SimpleNamespace(
            virtual_memory=lambda: SimpleNamespace(available=8 * 1024**3),
        ),
    )

    decision = _admit_memory_budgeted_items(
        pending=pending,
        in_flight_memory_mb=4 * MB_PER_GB,
        in_flight_count=1,
        budget_mb=32 * MB_PER_GB,
        memory_floor_mb=12 * MB_PER_GB,
        actual_available_memory_mb=_actual_available_memory_mb(),
        available_slots=1,
    )

    assert decision.admitted == ()
    assert decision.throttled is True
    assert decision.throttle_reason == "actual_free"
    assert len(pending) == 1


def test_actual_free_memory_gate_allows_first_job_to_avoid_deadlock() -> None:
    """Low actual free memory still admits one job when nothing is in flight."""

    pending = [_item(estimated_gb=4)]

    decision = _admit_memory_budgeted_items(
        pending=pending,
        in_flight_memory_mb=0,
        in_flight_count=0,
        budget_mb=32 * MB_PER_GB,
        memory_floor_mb=12 * MB_PER_GB,
        actual_available_memory_mb=8 * MB_PER_GB,
        available_slots=4,
    )

    assert [item.row.name for item in decision.admitted] == ["UnitNet"]
    assert decision.throttled is False
    assert decision.throttle_reason is None
    assert pending == []


def test_case_timeout_scales_real_pass_priors() -> None:
    """A real-PASS prior anchors a scaled timeout within floor and ceiling.

    ``duration_estimates`` here is the PASS-only map: only a genuine finish
    anchors the scale. Scale >= 3.0 so a model that passed at ``D`` is not
    killed for running slower this time.
    """

    row = _row(stable_id="m-slow")
    # 3x * 1500 = 4500 -- the scale dominates the generous floor (above the finish
    # time + margin, above 3600, below the ceiling) so the scaled value is used.
    assert (
        _case_timeout(
            row,
            {},
            240.0,
            {"m-slow": 1500.0},
            timeout_scale=3.0,
            timeout_ceiling_sec=14400.0,
            generous_sec=3600.0,
        )
        == 4500
    )
    # A prior far beyond the ceiling clamps to the ceiling (hang-catcher cap).
    assert (
        _case_timeout(
            row,
            {},
            240.0,
            {"m-slow": 6000.0},
            timeout_scale=3.0,
            timeout_ceiling_sec=14400.0,
            generous_sec=3600.0,
        )
        == 14400.0
    )
    # A tiny PASS prior is floored at the GENEROUS minimum, NOT a thin base wall:
    # a model that passed fast in a prior run can run much slower this time
    # (single-thread pin, cold cache, GC, load), so the floor must be generous so
    # we never kill a finisher. 3x * 10 = 30 < 3600 -> floored at 3600.
    assert (
        _case_timeout(
            row,
            {},
            240.0,
            {"m-slow": 10.0},
            timeout_scale=3.0,
            timeout_ceiling_sec=14400.0,
            generous_sec=3600.0,
        )
        == 3600.0
    )


def test_case_timeout_generous_lane_for_timed_out_or_unmeasured() -> None:
    """A timed-out (truncated) or unmeasured prior gets the GENEROUS lane.

    This is the crux of the JMT directive: a row that timed out once -- its
    recorded duration is the *truncated cap*, which the PASS-only map excludes --
    must NOT be scaled (no 1.5x-of-360 trap); it gets a big fixed wall so it can
    actually finish on the rerun.
    """

    row = _row(stable_id="m-timed-out")
    # No PASS prior in the map (timed-out rows are excluded) -> generous lane.
    assert (
        _case_timeout(
            row,
            {},
            240.0,
            {},  # PASS-only map: timed-out prior is absent here by construction
            timeout_scale=3.0,
            timeout_ceiling_sec=14400.0,
            generous_sec=3600.0,
        )
        == 3600.0
    )
    # The generous lane is at least 3600s and never the flat ~240/360 default.
    assert (
        _case_timeout(
            row,
            {},
            240.0,
            None,
            timeout_scale=3.0,
            timeout_ceiling_sec=14400.0,
            generous_sec=3600.0,
        )
        == 3600.0
    )


def test_case_timeout_smoke_override_wins() -> None:
    """Explicit smoke timeout overrides duration-based sizing."""

    row = _row(stable_id="m-smoke")
    settings = {"m-smoke": SmokeCaseSettings(timeout_sec=77.0, input_scale=1.0)}

    timeout = _case_timeout(
        row,
        settings,
        240.0,
        {"m-smoke": 900.0},
        timeout_scale=3.0,
        timeout_ceiling_sec=14400.0,
        generous_sec=3600.0,
    )

    assert timeout == 77.0


def test_resolve_timeout_ceiling_is_generous_and_data_calibrated() -> None:
    """Ceiling >= max(14400, 3x corpus-max-passed) and never below the 3hr floor."""

    # No corpus evidence -> at least the 4hr minimum, above the 3hr floor.
    assert resolve_timeout_ceiling_sec(None, 0.0) >= TIMEOUT_CEILING_MIN_SEC
    assert resolve_timeout_ceiling_sec(None, 0.0) >= TIMEOUT_CEILING_FLOOR_SEC
    # Corpus max passed = 2868s (effdet) -> 3x = 8604, still below the 4hr floor,
    # so the 14400s minimum wins.
    assert resolve_timeout_ceiling_sec(None, 2868.0) == TIMEOUT_CEILING_MIN_SEC
    # A very slow finisher pushes the ceiling above 14400 (3x dominates).
    assert resolve_timeout_ceiling_sec(None, 6000.0) == 18000.0
    # An explicit higher ceiling is honored; an explicit LOWER one is clamped up
    # (a future "optimization" can never tighten it back into a limiter).
    assert resolve_timeout_ceiling_sec(36000.0, 2868.0) == 36000.0
    assert resolve_timeout_ceiling_sec(600.0, 2868.0) == TIMEOUT_CEILING_MIN_SEC
    assert resolve_timeout_ceiling_sec(600.0, 0.0) >= TIMEOUT_CEILING_FLOOR_SEC


def test_no_plausibly_finishing_model_is_killed() -> None:
    """STAY-FIXED gate for the JMT timeout directive (TIMEOUT_POLICY.md).

    A timeout must never kill a model that plausibly would finish:

    * The slow-fittable class (beit_large/eva_giant/depth_pro/deepseek_vl/OuteTTS)
      and the slowest KNOWN finishers (effdet 2868s, d2 937s) each get a timeout
      STRICTLY ABOVE their finish time + margin once a PASS prior exists.
    * A previously-TIMED-OUT stable_id (absent from the PASS-only map) resolves to
      the GENEROUS lane (>= 3600s), NOT ~360s (no 1.5x-of-a-truncated-cap trap).
    """

    ceiling = resolve_timeout_ceiling_sec(None, 2868.0)
    generous = TIMEOUT_GENEROUS_MIN_SEC

    # 1) Slowest known finishers get a wall well above finish-time + margin.
    known_finishers = {"m4527-effdet": 2868.0, "m-d2": 937.0}
    for stable_id, finish_sec in known_finishers.items():
        row = _row(stable_id=stable_id)
        timeout = _case_timeout(
            row,
            {},
            240.0,
            {stable_id: finish_sec},  # real PASS prior
            timeout_scale=TIMEOUT_DEFAULT_SCALE,
            timeout_ceiling_sec=ceiling,
            generous_sec=generous,
        )
        assert timeout > finish_sec + 60.0, (stable_id, timeout, finish_sec)
        assert timeout <= ceiling

    # 2) The slow-fittable class: even with only a truncated/timed-out history
    # (excluded from the PASS-only map), each gets the generous lane, never ~360s.
    slow_fittable = (
        "beit_large",
        "eva_giant",
        "depth_pro",
        "deepseek_vl",
        "OuteTTS",
    )
    for name in slow_fittable:
        row = _row(stable_id=f"m-{name}", name=name)
        timeout = _case_timeout(
            row,
            {},
            240.0,
            {},  # PASS-only map: their only history was a timeout -> absent here
            timeout_scale=TIMEOUT_DEFAULT_SCALE,
            timeout_ceiling_sec=ceiling,
            generous_sec=generous,
        )
        assert timeout >= 3600.0, (name, timeout)
        # Explicitly NOT the old 1.5x-of-truncated-240 (~360s) trap.
        assert timeout > 360.0

    # 3) A previously-timed-out stable_id resolves to the generous lane, not a
    # scaled truncated cap. The truncated 240s value lives in the ALL-durations
    # map but NOT in the PASS-only map fed to _case_timeout.
    timed_out = _row(stable_id="m-timed-out-prior")
    all_durations_with_truncated_cap = {"m-timed-out-prior": 240.0}
    pass_only = {}  # excludes the timed-out prior by construction
    timeout = _case_timeout(
        timed_out,
        {},
        240.0,
        pass_only,
        timeout_scale=TIMEOUT_DEFAULT_SCALE,
        timeout_ceiling_sec=ceiling,
        generous_sec=generous,
    )
    assert timeout >= 3600.0
    # Sanity: scaling the truncated cap would have produced ~360s -- which we
    # explicitly avoid.
    assert timeout != 1.5 * all_durations_with_truncated_cap["m-timed-out-prior"]

    # 4) The SMALL-real-pass-prior case (the F1 regression): a model that PASSED
    # fast (~30s) in a prior run -- recorded BEFORE this sprint's single-thread
    # validation pin -- must NOT get a thin scaled wall (3x * 30 = 90s). The
    # single-thread pin, cold cache, GC, and tracemalloc overhead can make the
    # same forward several-x slower this time, so the scaled real-pass timeout is
    # floored at the GENEROUS minimum (>= 3600s), never killed at <=4 min.
    fast_prior = _row(stable_id="m-fast-prior")
    timeout = _case_timeout(
        fast_prior,
        {},
        240.0,
        {"m-fast-prior": 30.0},  # genuine fast PASS prior
        timeout_scale=TIMEOUT_DEFAULT_SCALE,
        timeout_ceiling_sec=ceiling,
        generous_sec=generous,
    )
    assert timeout >= 3600.0, timeout
    # Explicitly NOT the thin 3x-of-30 (~90s) or the old 240s base wall.
    assert timeout > 240.0


def test_validate_with_timeout_records_peak_rss_on_timeout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Timeout kill path returns the sampled worker peak RSS."""

    class FakePipe:
        """Minimal readable text pipe that is already at EOF."""

        def readline(self) -> str:
            """Return EOF immediately."""

            return ""

        def close(self) -> None:
            """No-op close."""

    class FakeProcess:
        """Minimal ``Popen`` stand-in that never exits before timeout."""

        pid = 12345
        returncode = None

        def __init__(self) -> None:
            """Provide drainable stdout/stderr pipes for the reader threads."""

            self.stdout = FakePipe()
            self.stderr = FakePipe()

        def poll(self) -> int | None:
            """Return no exit status before the timeout branch kills the process."""

            return self.returncode

        def kill(self) -> None:
            """Record process termination."""

            self.returncode = -9

        def wait(self, timeout: float | None = None) -> int:
            """Return the (post-kill) exit status."""

            return self.returncode or 0

    class FakePsutilProcess:
        """Minimal psutil process with deterministic RSS."""

        def __init__(self, pid: int) -> None:
            """Store the process ID."""

            self.pid = pid

        def memory_info(self) -> SimpleNamespace:
            """Return deterministic resident memory."""

            return SimpleNamespace(rss=12 * 1024**2)

        def children(self, recursive: bool = False) -> list[object]:
            """Return no child processes."""

            return []

    monkeypatch.setattr(
        "menagerie.validate_menagerie.subprocess.Popen", lambda *_, **__: FakeProcess()
    )
    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(Process=FakePsutilProcess))

    result = validate_with_timeout(
        _row(stable_id="m-timeout"),
        dry_run=True,
        scope="forward",
        device="cpu",
        timeout_sec=0.0,
        tmp_dir=tmp_path,
    )

    assert result.status == "failed:timeout"
    assert result.peak_rss_mb == 12


def test_validate_with_timeout_drains_chatty_stderr_without_false_timeout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A worker that floods stderr (>64KB) past the pipe buffer must still PASS.

    REGRESSION for W3a-1 (REVIEW_scheduling.md): the Popen+poll loop never drained
    stdout/stderr until exit, so a chatty worker (torch/torchlens warnings) that
    wrote past the ~64KB OS pipe buffer BLOCKED on write() -> never finished ->
    false ``failed:timeout``. This drives the REAL chatty path: a genuine
    subprocess writes ~256KB to stderr then emits a passing ``worker_result`` line
    on stdout and exits 0. With concurrent reader-thread drains it PASSES; without
    them the worker deadlocks on the full pipe and this times out.
    """

    payload = {
        "name": "ChattyNet",
        "model_id": 1,
        "status": "validated",
        "validate_metadata_ok": True,
        "scope": "forward",
        "elapsed": 0.01,
        "dependency_cluster": "unit",
        "error": "",
        "stable_id": "m-chatty",
        "recipe_revision_sha256": "recipe-a",
        "n_ops": 3,
        "graph_shape_hash": "shape-chatty",
        "input_scale": 1.0,
    }
    event_line = json.dumps({"event": "worker_result", "result": payload})
    # A real worker that floods stderr WELL past the 64KB pipe buffer, then writes
    # its small result line to stdout and exits cleanly.
    worker_src = (
        "import sys\n"
        "sys.stderr.write('W' * (256 * 1024))\n"
        "sys.stderr.flush()\n"
        f"sys.stdout.write({event_line!r} + chr(10))\n"
        "sys.stdout.flush()\n"
    )
    real_popen = subprocess.Popen

    def chatty_popen(_command: object, **kwargs: object) -> subprocess.Popen:
        """Replace the worker command with the real chatty subprocess."""

        return real_popen([sys.executable, "-c", worker_src], **kwargs)

    monkeypatch.setattr("menagerie.validate_menagerie.subprocess.Popen", chatty_popen)

    result = validate_with_timeout(
        _row(stable_id="m-chatty", name="ChattyNet"),
        dry_run=True,
        scope="forward",
        device="cpu",
        timeout_sec=30.0,
        tmp_dir=tmp_path,
    )

    assert result.status == "validated", result.status
    assert result.n_ops == 3


def test_validate_with_timeout_retries_replay_failure_in_single_thread_worker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A replay failure in a multi-thread worker gets one fresh thread-1 process retry."""

    row = _row(stable_id="m-retry-process", name="RetryNet")
    failed = ValidationResult(
        row.name,
        row.model_id,
        "failed:replay",
        0,
        False,
        "forward",
        0.1,
        "unit-zoo",
        "first replay failed",
        stable_id=row.stable_id,
        recipe_revision_sha256=row.recipe_revision_sha256,
        input_scale=1.0,
    )
    passed = ValidationResult(
        row.name,
        row.model_id,
        "validated",
        9,
        True,
        "forward",
        0.2,
        "unit-zoo",
        "forward=True",
        "shape-retry",
        stable_id=row.stable_id,
        recipe_revision_sha256=row.recipe_revision_sha256,
        input_scale=1.0,
    )
    events = [
        json.dumps({"event": "worker_result", "result": failed.__dict__}) + "\n",
        json.dumps({"event": "worker_result", "result": passed.__dict__}) + "\n",
    ]
    commands: list[list[str]] = []

    class FakePipe:
        """Pipe returning one scripted text payload, then EOF."""

        def __init__(self, text: str) -> None:
            """Store text as individual readable lines."""

            self._lines = iter(text.splitlines(keepends=True))

        def readline(self) -> str:
            """Return the next line, or EOF."""

            return next(self._lines, "")

        def close(self) -> None:
            """No-op close."""

    class FakeProcess:
        """Already-exited worker process carrying scripted stdout."""

        pid = 123
        returncode = 0

        def __init__(self, stdout: str) -> None:
            """Create stdout/stderr pipes."""

            self.stdout = FakePipe(stdout)
            self.stderr = FakePipe("")

        def poll(self) -> int:
            """Return the exit status."""

            return self.returncode

        def wait(self, timeout: float | None = None) -> int:
            """Return the exit status."""

            del timeout
            return self.returncode

    def fake_popen(command: list[str], **kwargs: object) -> FakeProcess:
        """Return a failed worker first, then a passing single-thread worker."""

        del kwargs
        commands.append(command)
        return FakeProcess(events[len(commands) - 1])

    monkeypatch.setattr("menagerie.validate_menagerie.subprocess.Popen", fake_popen)

    result = validate_with_timeout(
        row,
        dry_run=False,
        scope="forward",
        device="cpu",
        timeout_sec=5.0,
        tmp_dir=tmp_path,
        worker_torch_threads=3,
    )

    assert result.status == "validated"
    assert result.n_ops == 9
    assert "worker_replay_retry=single_thread_process" in result.error
    assert len(commands) == 2
    assert commands[0][commands[0].index("--worker-torch-threads") + 1] == "3"
    assert commands[1][commands[1].index("--worker-torch-threads") + 1] == "1"


def test_resolve_row_device_is_route_owned_and_honors_or_errors_explicit_device() -> None:
    """R1-1: the route owns device; an explicit --device is honored or rejected.

    The route is the single source of the device used by the worker AND the
    ledger identity, so provenance never splits. The local-first default
    (--device cpu) defers to the route. An explicit --device that CONFLICTS with
    the resolved route is rejected loudly instead of being silently discarded
    (running CPU-eligible rows on CPU regardless, the quiet footgun the review
    flagged).
    """

    cpu_route = ResourceRoute("local-cpu", "cpu", False, "local_first_default")
    gpu_route = ResourceRoute("local-gpu", "cuda", False, "requires_cuda")

    # Default --device cpu defers to the route (cpu for bulk, cuda for GPU rows).
    assert _resolve_row_device(cpu_route, SimpleNamespace(device="cpu")) == "cpu"
    assert _resolve_row_device(gpu_route, SimpleNamespace(device="cpu")) == "cuda"
    # --device auto also defers to the route.
    assert _resolve_row_device(cpu_route, SimpleNamespace(device="auto")) == "cpu"
    # An explicit --device that MATCHES the route is honored.
    assert _resolve_row_device(gpu_route, SimpleNamespace(device="cuda")) == "cuda"
    # An explicit --device that CONFLICTS with the route errors loudly (never
    # silently overridden to the route's device).
    with pytest.raises(ValueError, match="conflicts with the resolved route device"):
        _resolve_row_device(cpu_route, SimpleNamespace(device="cuda"))


def test_lpt_sort_front_loads_giants_by_memory_then_duration() -> None:
    """LPT ordering places large and slow items before small rows."""

    pending = [
        _item(estimated_gb=2, stable_id="small-a", name="SmallA"),
        _item(estimated_gb=12, stable_id="giant-fast", name="GiantFast"),
        _item(estimated_gb=2, stable_id="small-b", name="SmallB"),
        _item(estimated_gb=12, stable_id="giant-slow", name="GiantSlow"),
        _item(estimated_gb=1, stable_id="small-c", name="SmallC"),
    ]
    duration_estimates = {"giant-fast": 30.0, "giant-slow": 90.0, "small-a": 1000.0}

    pending.sort(key=lambda item: _lpt_sort_key(item, duration_estimates), reverse=True)

    assert [item.row.stable_id for item in pending[:2]] == ["giant-slow", "giant-fast"]


def test_lpt_sorted_admission_admits_giant_first_when_budget_allows() -> None:
    """Greedy admission receives LPT-sorted pending and admits a giant first."""

    pending = [
        _item(estimated_gb=2, stable_id="small-a", name="SmallA"),
        _item(estimated_gb=12, stable_id="giant-fast", name="GiantFast"),
        _item(estimated_gb=2, stable_id="small-b", name="SmallB"),
        _item(estimated_gb=12, stable_id="giant-slow", name="GiantSlow"),
        _item(estimated_gb=1, stable_id="small-c", name="SmallC"),
    ]
    duration_estimates = {"giant-fast": 30.0, "giant-slow": 90.0}
    pending.sort(key=lambda item: _lpt_sort_key(item, duration_estimates), reverse=True)

    decision = _admit_memory_budgeted_items(
        pending=pending,
        in_flight_memory_mb=0,
        in_flight_count=0,
        budget_mb=16 * MB_PER_GB,
        memory_floor_mb=1 * MB_PER_GB,
        actual_available_memory_mb=64 * MB_PER_GB,
        available_slots=2,
    )

    assert decision.admitted[0].row.stable_id == "giant-slow"


def _big_item(
    estimated_mb: int,
    stable_id: str,
    name: str = "BigNet",
    source: str = "ledger",
) -> ValidationWorkItem:
    """Build a work item with an explicit MB estimate and source.

    Parameters
    ----------
    estimated_mb:
        Estimated peak RSS in MB.
    stable_id:
        Stable model identity.
    name:
        Model name.
    source:
        Estimate source tag.

    Returns
    -------
    ValidationWorkItem
        Validation work item.
    """

    return ValidationWorkItem(
        plan=_plan(),
        row=_row(stable_id=stable_id, name=name),
        estimated_memory_mb=estimated_mb,
        estimate_source=source,
    )


def test_default_validation_jobs_scales_to_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """Gate (a): default jobs and worker threads keep jobs*threads near nproc."""

    monkeypatch.setattr("menagerie.validate_menagerie.os.cpu_count", lambda: 20)
    assert default_validation_jobs() == 5
    assert default_worker_torch_threads(default_validation_jobs()) == 4
    assert default_validation_jobs() * default_worker_torch_threads(default_validation_jobs()) == 20

    monkeypatch.setattr("menagerie.validate_menagerie.os.cpu_count", lambda: 64)
    assert default_validation_jobs() == 16
    assert default_worker_torch_threads(default_validation_jobs()) == 4
    assert default_validation_jobs() * default_worker_torch_threads(default_validation_jobs()) == 64

    # Small/unknown core counts still yield at least one worker.
    monkeypatch.setattr("menagerie.validate_menagerie.os.cpu_count", lambda: 1)
    assert default_validation_jobs() == 1
    assert default_worker_torch_threads(default_validation_jobs()) == 1
    monkeypatch.setattr("menagerie.validate_menagerie.os.cpu_count", lambda: None)
    assert default_validation_jobs() == 1
    assert default_worker_torch_threads(default_validation_jobs()) == 1


def test_explicit_worker_torch_threads_overrides_auto() -> None:
    """An explicit per-worker torch thread count is honored."""

    assert resolve_worker_torch_threads(5, jobs=10) == 5


def test_forward_replay_failure_retries_single_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A falsy multi-thread forward result retries once with ``num_threads=1``."""

    class ProbeModel:
        """Tiny model fixture with an ``eval`` method."""

        def eval(self) -> "ProbeModel":
            """Return self, matching ``nn.Module.eval``."""

            return self

    attempts: list[int | None] = []

    def fake_validate_forward_pass_torch(
        model: object,
        input_args: object,
        input_kwargs: dict[Any, Any] | None = None,
        random_seed: int | None = None,
        verbose: bool = False,
        validate_metadata: bool = True,
        *,
        num_threads: int | None = None,
        _trace_observer: object | None = None,
    ) -> bool:
        """Fail the default-thread attempt and pass the single-thread retry."""

        del model, input_args, input_kwargs, random_seed, verbose, validate_metadata
        attempts.append(num_threads)
        if callable(_trace_observer):
            suffix = "retry" if num_threads == 1 else "first"
            _trace_observer(SimpleNamespace(num_ops=7, graph_shape_hash=f"shape-{suffix}"))
        return num_threads == 1

    monkeypatch.setattr("menagerie.validate_menagerie._build_input", lambda *_: object())
    monkeypatch.setattr("menagerie.validate_menagerie.instantiate_model", lambda _row: ProbeModel())
    monkeypatch.setattr(
        "torchlens.user_funcs._validate_forward_pass_torch",
        fake_validate_forward_pass_torch,
    )

    result = validate_one(_row(stable_id="retry-model"), False, "forward", "cpu")

    assert result.status == "validated"
    assert attempts == [None, 1]
    assert result.n_ops == 7
    assert result.graph_shape_hash == "shape-retry"
    assert "forward_retry=single_thread" in result.error


def test_ledger_peak_drives_a_high_memory_estimate_with_headroom() -> None:
    """Gate (b): a large measured ledger peak yields a HIGH estimate, not 4GB."""

    row = _row(stable_id="tf_efficientnet_l2", name="tf_efficientnet_l2")
    # 59 GB measured prior, like the real ledger entry for this model.
    ledger_mb = 59 * MB_PER_GB
    estimate = _memory_estimate_for_row(row, {"tf_efficientnet_l2": ledger_mb})

    assert estimate.source == "ledger"
    # Not the 4GB default -- the measured peak times the headroom factor.
    assert estimate.estimated_mb == int(round(ledger_mb * LEDGER_MEMORY_HEADROOM))
    assert estimate.estimated_mb > 70 * MB_PER_GB


def test_unmeasured_model_keeps_low_default_estimate() -> None:
    """Gate (d): an unmeasured, non-heavy model still gets the small default."""

    row = _row(stable_id="unmeasured", name="TinyNet")
    estimate = _memory_estimate_for_row(row, {})

    assert estimate.source == "default"
    assert estimate.estimated_mb == 4 * MB_PER_GB


def test_local_oom_scheduler_estimate_uses_high_floor_not_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cluster-escalated local OOM receives high scheduler memory, not 4GB."""

    local_host = "workstation"
    monkeypatch.setattr("menagerie.cluster_runner.socket.gethostname", lambda: local_host)
    ledger_db = tmp_path / "verification.db"
    with connect(ledger_db) as conn:
        append_verification_run(
            conn,
            _run(
                run_id="local-oom",
                stable_id="m9025",
                name="samvit_large",
                runner_host=local_host,
                status="oom",
                forward_pass=0,
                metadata_ok=0,
                n_ops=None,
                graph_shape_hash=None,
                peak_rss_mb=None,
                error_class="oom",
            ),
        )

    estimates = latest_scheduler_memory_estimates(ledger_db)
    estimate = _memory_estimate_for_row(_row(stable_id="m9025", name="samvit_large"), estimates)

    assert estimates["m9025"] == 960 * MB_PER_GB
    assert estimate.source == "ledger"
    assert estimate.estimated_mb > 4 * MB_PER_GB


def test_sigkill_scheduler_estimate_uses_high_floor_not_4gb(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A local SIGKILL with no peak gets the high floor, not the 4GB default."""

    local_host = "workstation"
    monkeypatch.setattr("menagerie.cluster_runner.socket.gethostname", lambda: local_host)
    ledger_db = tmp_path / "verification.db"
    with connect(ledger_db) as conn:
        append_verification_run(
            conn,
            _run(
                run_id="local-killed",
                stable_id="m-sigkill",
                name="SigkillGiant",
                runner_host=local_host,
                status="killed",
                forward_pass=0,
                metadata_ok=0,
                n_ops=None,
                graph_shape_hash=None,
                peak_rss_mb=None,
                error_class="failed:killed",
            ),
        )

    estimates = latest_scheduler_memory_estimates(ledger_db)
    estimate = _memory_estimate_for_row(_row(stable_id="m-sigkill", name="SigkillGiant"), estimates)

    assert estimates["m-sigkill"] == 960 * MB_PER_GB
    assert estimate.source == "ledger"
    assert estimate.estimated_mb > 4 * MB_PER_GB


def test_big_model_concurrency_cap_blocks_models_beyond_cap() -> None:
    """Gate (c): the >40GB big-model cap admits two big models, then throttles."""

    # Three big models (50GB each) plus a small one, plenty of aggregate budget.
    pending = [
        _big_item(50 * MB_PER_GB, stable_id="big-a", name="BigA"),
        _big_item(50 * MB_PER_GB, stable_id="big-b", name="BigB"),
        _big_item(50 * MB_PER_GB, stable_id="big-c", name="BigC"),
        _big_item(2 * MB_PER_GB, stable_id="small-a", name="SmallA"),
    ]

    decision = _admit_memory_budgeted_items(
        pending=pending,
        in_flight_memory_mb=0,
        in_flight_count=0,
        budget_mb=200 * MB_PER_GB,
        memory_floor_mb=1 * MB_PER_GB,
        actual_available_memory_mb=200 * MB_PER_GB,
        available_slots=8,
        in_flight_big_count=0,
        max_concurrent_big_models=MAX_CONCURRENT_BIG_MODELS,
    )

    admitted_ids = [item.row.stable_id for item in decision.admitted]
    big_admitted = [
        item for item in decision.admitted if item.estimated_memory_mb > BIG_MODEL_THRESHOLD_MB
    ]
    # At most two big models are admitted even though budget+slots allow all.
    assert len(big_admitted) == MAX_CONCURRENT_BIG_MODELS
    # The small model still gets in alongside the big models.
    assert "small-a" in admitted_ids
    # The third big model is held back.
    assert "big-c" in [item.row.stable_id for item in pending]


def test_big_model_cap_holds_against_in_flight_big_model() -> None:
    """Gate (c): with cap-saturating big models in flight, no further big admits."""

    pending = [
        _big_item(50 * MB_PER_GB, stable_id="big-b", name="BigB"),
        _big_item(2 * MB_PER_GB, stable_id="small-a", name="SmallA"),
    ]

    decision = _admit_memory_budgeted_items(
        pending=pending,
        in_flight_memory_mb=100 * MB_PER_GB,
        in_flight_count=MAX_CONCURRENT_BIG_MODELS,
        budget_mb=200 * MB_PER_GB,
        memory_floor_mb=1 * MB_PER_GB,
        actual_available_memory_mb=200 * MB_PER_GB,
        available_slots=7,
        in_flight_big_count=MAX_CONCURRENT_BIG_MODELS,
        max_concurrent_big_models=MAX_CONCURRENT_BIG_MODELS,
    )

    admitted_ids = [item.row.stable_id for item in decision.admitted]
    # The small model is admitted; the big one waits for the in-flight big to finish.
    assert admitted_ids == ["small-a"]
    assert "big-b" in [item.row.stable_id for item in pending]


def test_big_model_runs_alone_when_nothing_in_flight() -> None:
    """Gate (c): a lone big model is never wedged -- the first job always admits."""

    pending = [_big_item(50 * MB_PER_GB, stable_id="big-a", name="BigA")]

    decision = _admit_memory_budgeted_items(
        pending=pending,
        in_flight_memory_mb=0,
        in_flight_count=0,
        budget_mb=200 * MB_PER_GB,
        memory_floor_mb=1 * MB_PER_GB,
        actual_available_memory_mb=200 * MB_PER_GB,
        available_slots=8,
        in_flight_big_count=0,
        max_concurrent_big_models=MAX_CONCURRENT_BIG_MODELS,
    )

    assert [item.row.stable_id for item in decision.admitted] == ["big-a"]


def test_small_models_keep_high_concurrency_under_big_cap() -> None:
    """Gate (d): the big-model cap never throttles a batch of small models."""

    pending = [_item(estimated_gb=2, stable_id=f"small-{i}", name=f"Small{i}") for i in range(8)]

    decision = _admit_memory_budgeted_items(
        pending=pending,
        in_flight_memory_mb=0,
        in_flight_count=0,
        budget_mb=200 * MB_PER_GB,
        memory_floor_mb=1 * MB_PER_GB,
        actual_available_memory_mb=200 * MB_PER_GB,
        available_slots=8,
        in_flight_big_count=0,
        max_concurrent_big_models=MAX_CONCURRENT_BIG_MODELS,
    )

    assert len(decision.admitted) == 8
    assert not decision.throttled


def test_auto_memory_budget_leaves_headroom_below_total_ram(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gate (e): the auto budget is capped below total RAM (OOM-killer headroom)."""

    total_gb = 125.0
    # Nearly-idle box: available approaches total, so auto reaches the total-RAM
    # headroom cap and remains below the machine's physical RAM.
    monkeypatch.setitem(
        sys.modules,
        "psutil",
        SimpleNamespace(
            virtual_memory=lambda: SimpleNamespace(
                available=int(122 * 1024**3),
                total=int(total_gb * 1024**3),
            ),
        ),
    )

    budget_gb = _resolve_memory_budget_gb(None)

    assert budget_gb <= total_gb * MEMORY_BUDGET_HEADROOM_FRACTION + 1e-6
    assert budget_gb > 105.0
    assert budget_gb < total_gb  # never targets the full machine


def test_explicit_memory_budget_overrides_auto() -> None:
    """An explicit --memory-budget-gb is honored verbatim (override path)."""

    assert _resolve_memory_budget_gb(48.0) == 48.0
