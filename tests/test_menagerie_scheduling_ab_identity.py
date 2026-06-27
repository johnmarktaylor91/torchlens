"""A/B identity proof for menagerie scheduling and routing changes."""

from __future__ import annotations

from collections.abc import Iterable

import pytest

from menagerie.catalog import CatalogRow
from menagerie.cluster_runner import route_resources
from menagerie.runtime import DependencyPlan
from menagerie.validate_menagerie import (
    MB_PER_GB,
    ValidationWorkItem,
    _admit_memory_budgeted_items,
    _lpt_sort_key,
    _resolve_row_device,
    validate_one,
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


def _traceable_row(stable_id: str = "m-real", model_id: int = 1) -> CatalogRow:
    """Build a catalog row for a tiny REAL model that traces end-to-end.

    Parameters
    ----------
    stable_id:
        Stable model identity.
    model_id:
        Catalog model ID.

    Returns
    -------
    CatalogRow
        Catalog row whose recipe instantiates a real traceable module.
    """

    return CatalogRow(
        model_id=model_id,
        display_index=model_id,
        stable_id=stable_id,
        name="LinearReLU",
        variant="",
        family="unit",
        family_normalized="unit",
        domain="unit",
        zoo="unit-zoo",
        constructor_call="torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU())",
        input_shape="(2, 4)",
        input_dtype="float32",
        era="2026",
        verified=True,
        notes="",
        source="catalog",
        recipe_revision_sha256="recipe-real",
    )


def test_old_vs_new_scheduling_run_yields_byte_identical_validation_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TEST-1 (REVIEW_scheduling.md): a REAL validation is byte-identical A/B.

    The earlier proof only checked admission membership + the route table. This
    strengthens it: run the SAME real model through the OLD scheduling (the blunt
    global ``args.device``) and the NEW scheduling (LPT order + the route-resolved
    per-row device), then assert the validation OUTPUT -- ``status``, ``n_ops``,
    and ``graph_shape_hash`` -- is IDENTICAL. The scheduling/device levers are
    operational only; the validation body that produces those fields must not move
    a single bit. If a scheduling change ever perturbed the trace, this fails.
    """

    torch = pytest.importorskip("torch")
    pytest.importorskip("torchlens")

    row = _traceable_row()

    # The recipe input path expects a typed JSONL record; supply a deterministic
    # real tensor so validate_one runs the REAL trace+replay+metadata body on the
    # fabricated row (the model itself instantiates straight from constructor_call).
    def _fixed_input(_row: CatalogRow) -> object:
        """Return a deterministic example input for the fabricated row."""

        torch.manual_seed(0)
        return torch.randn(2, 4)

    monkeypatch.setattr("menagerie.validate_menagerie.build_input_for_row", _fixed_input)

    # OLD scheduling: device taken from the blunt global args.device.
    old_args_device = "cpu"
    old_result = validate_one(row, dry_run=False, scope="forward", device=old_args_device)

    # NEW scheduling: device resolved per-row from the route (LPT order does not
    # touch the validation body, so running once with the route device proves the
    # identity for the levers that COULD touch device placement).
    route = route_resources(row, ledger={}, local_gpu_vram_bytes=None)
    new_device = _resolve_row_device(
        route,
        type("Args", (), {"device": old_args_device})(),
    )
    new_result = validate_one(row, dry_run=False, scope="forward", device=new_device)

    # The route keeps a CPU-eligible row on CPU: same device, same everything.
    assert new_device == old_args_device == "cpu"
    assert new_result.status == old_result.status == "validated", (
        old_result.status,
        new_result.status,
        old_result.error,
    )
    # The load-bearing tripwire fields are byte-identical across the A/B.
    assert new_result.n_ops == old_result.n_ops
    assert new_result.n_ops is not None and new_result.n_ops > 0
    assert new_result.graph_shape_hash == old_result.graph_shape_hash
    assert new_result.graph_shape_hash != ""
