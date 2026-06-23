"""Tests for menagerie validation scheduler admission policy."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from menagerie.catalog import CatalogRow
from menagerie.runtime import DependencyPlan
from menagerie.validate_menagerie import (
    MB_PER_GB,
    ValidationWorkItem,
    _admit_memory_budgeted_items,
    _actual_available_memory_mb,
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
