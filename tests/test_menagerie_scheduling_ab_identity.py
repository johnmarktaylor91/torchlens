"""A/B identity proof for menagerie scheduling and routing changes."""

from __future__ import annotations

from collections.abc import Iterable

from menagerie.catalog import CatalogRow
from menagerie.cluster_runner import route_resources
from menagerie.runtime import DependencyPlan
from menagerie.validate_menagerie import (
    MB_PER_GB,
    ValidationWorkItem,
    _admit_memory_budgeted_items,
    _lpt_sort_key,
)


def _row(stable_id: str, model_id: int, name: str) -> CatalogRow:
    """Build a compact catalog row fixture.

    Parameters
    ----------
    stable_id:
        Stable model identity.
    model_id:
        Catalog model ID.
    name:
        Model name.

    Returns
    -------
    CatalogRow
        Catalog row.
    """

    return CatalogRow(
        model_id=model_id,
        display_index=model_id,
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

    return DependencyPlan(cluster_key="base", packages=(), top_modules=(), environment="base")


def _item(stable_id: str, model_id: int, memory_gb: int) -> ValidationWorkItem:
    """Build one validation work item.

    Parameters
    ----------
    stable_id:
        Stable model identity.
    model_id:
        Catalog model ID.
    memory_gb:
        Estimated memory in GiB.

    Returns
    -------
    ValidationWorkItem
        Scheduler item.
    """

    return ValidationWorkItem(
        plan=_plan(),
        row=_row(stable_id, model_id, f"UnitNet{model_id}"),
        estimated_memory_mb=memory_gb * MB_PER_GB,
        estimate_source="unit",
    )


def _drain_order(items: Iterable[ValidationWorkItem]) -> list[str]:
    """Drain scheduler admission and return every admitted stable ID.

    Parameters
    ----------
    items:
        Pending items in the desired scheduler order.

    Returns
    -------
    list[str]
        Stable IDs admitted over the full drain.
    """

    pending = list(items)
    admitted: list[str] = []
    while pending:
        decision = _admit_memory_budgeted_items(
            pending=pending,
            in_flight_memory_mb=0,
            in_flight_count=0,
            budget_mb=16 * MB_PER_GB,
            memory_floor_mb=1 * MB_PER_GB,
            actual_available_memory_mb=64 * MB_PER_GB,
            available_slots=2,
        )
        admitted.extend(item.row.stable_id for item in decision.admitted)
    return admitted


def test_new_scheduling_routing_preserves_validation_membership_and_cpu_bulk() -> None:
    """new scheduling/routing -> IDENTICAL set of per-model validations.

    The validation body that produces status+n_ops+graph_shape_hash is untouched;
    this proof covers the operational levers only: admission order and explicit
    route device. The bulk sample has no CUDA-required or RAM-giant rows, so
    every row remains local CPU and the same rows are validated.
    """

    original_items = [
        _item("m-small-a", 1, 2),
        _item("m-giant-a", 2, 12),
        _item("m-small-b", 3, 1),
        _item("m-giant-b", 4, 10),
        _item("m-small-c", 5, 3),
    ]
    duration_estimates = {"m-giant-a": 60.0, "m-giant-b": 120.0, "m-small-a": 5.0}
    old_order = _drain_order(original_items)
    new_items = list(original_items)
    new_items.sort(key=lambda item: _lpt_sort_key(item, duration_estimates), reverse=True)
    new_order = _drain_order(new_items)

    assert set(new_order) == set(old_order)
    assert set(new_order) == {item.row.stable_id for item in original_items}
    for item in original_items:
        route = route_resources(item.row, ledger={}, local_gpu_vram_bytes=None)
        assert route.lane == "local-cpu"
        assert route.device == "cpu"
        assert route.cluster is False
