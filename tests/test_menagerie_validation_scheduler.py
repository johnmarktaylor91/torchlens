"""Tests for menagerie validation scheduler admission policy."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from menagerie.catalog import CatalogRow
from menagerie.runtime import DependencyPlan
from menagerie.validate_menagerie import (
    MB_PER_GB,
    TIMEOUT_CEILING_FLOOR_SEC,
    TIMEOUT_CEILING_MIN_SEC,
    TIMEOUT_DEFAULT_SCALE,
    TIMEOUT_GENEROUS_MIN_SEC,
    SmokeCaseSettings,
    ValidationWorkItem,
    _admit_memory_budgeted_items,
    _actual_available_memory_mb,
    _case_timeout,
    _lpt_sort_key,
    resolve_timeout_ceiling_sec,
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
    # 3x * 900 = 2700 -- comfortably above the finish time, below the ceiling.
    assert (
        _case_timeout(
            row,
            {},
            240.0,
            {"m-slow": 900.0},
            timeout_scale=3.0,
            timeout_ceiling_sec=14400.0,
            generous_sec=3600.0,
        )
        == 2700
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
    # A tiny PASS prior never drops below the base wall.
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
        == 240.0
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
