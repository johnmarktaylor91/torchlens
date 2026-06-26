"""Tests for menagerie validation scheduler admission policy."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from menagerie.catalog import CatalogRow
from menagerie.runtime import DependencyPlan
from menagerie.validate_menagerie import (
    MB_PER_GB,
    SmokeCaseSettings,
    ValidationWorkItem,
    _admit_memory_budgeted_items,
    _actual_available_memory_mb,
    _case_timeout,
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


def test_case_timeout_uses_size_tiered_duration_clamp() -> None:
    """Prior row durations scale timeout within base and ceiling bounds."""

    row = _row(stable_id="m-slow")
    assert (
        _case_timeout(
            row,
            {},
            240.0,
            {"m-slow": 900.0},
            timeout_scale=1.5,
            timeout_ceiling_sec=1800.0,
        )
        == 1350
    )
    assert (
        _case_timeout(
            row,
            {},
            240.0,
            {"m-slow": 2000.0},
            timeout_scale=1.5,
            timeout_ceiling_sec=1800.0,
        )
        == 1800.0
    )
    assert _case_timeout(row, {}, 240.0, {}, timeout_scale=1.5) == 240.0
    assert (
        _case_timeout(
            row,
            {},
            240.0,
            {"m-slow": 10.0},
            timeout_scale=1.5,
            timeout_ceiling_sec=1800.0,
        )
        == 240.0
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
        timeout_scale=1.5,
        timeout_ceiling_sec=1800.0,
    )

    assert timeout == 77.0


def test_validate_with_timeout_records_peak_rss_on_timeout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Timeout kill path returns the sampled worker peak RSS."""

    class FakeProcess:
        """Minimal ``Popen`` stand-in that never exits before timeout."""

        pid = 12345
        returncode = None

        def poll(self) -> int | None:
            """Return no exit status before the timeout branch kills the process."""

            return self.returncode

        def kill(self) -> None:
            """Record process termination."""

            self.returncode = -9

        def communicate(self) -> tuple[str, str]:
            """Return empty captured worker output."""

            return "", ""

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
