"""``Bundle.measure_members``: recompute a metric over the members that still exist.

Released effect-table candidates are reported ``unmeasured``, never silently
re-scored; ``order="chain"`` also orders the measured members by the bundle's
validated operation ledger. The name stays on the surface pending the glossary
round-2 ruling on ``Bundle.measure_members`` (``Bundle.map`` is the flat form).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from ..errors.episode import BundleExperimentError
from ._lineage import validate_operation_chain

if TYPE_CHECKING:
    from . import Bundle

__all__ = ["measure_members"]


def measure_members(
    bundle: Bundle,
    *,
    metric: Callable[[Any], Any],
    order: str = "member",
) -> dict[str, Any]:
    """Return ``values`` (live members), ``unmeasured`` (released candidates) and ``order``.

    ``order="chain"`` adds ``chain`` (rows ``seq``, ``operation_id``, ``kind``,
    ``member``, ``value`` in ledger order) and ``unchained`` (members no
    operation recorded, never placed by a guessed position).
    """

    if order not in ("member", "chain"):
        raise BundleExperimentError(
            f"measure_members(order={order!r}) is not available: use "
            "order='member' (per member) or order='chain' (by the bundle's "
            "operation ledger).",
            code="measure_members_order_unavailable",
            order=order,
        )
    if not callable(metric):
        raise BundleExperimentError(
            "measure_members(metric=...) must be a callable(member) -> value",
            code="site_sweep_metric_invalid",
        )
    values = {name: metric(member) for name, member in bundle.members.items()}
    report: dict[str, Any] = {"values": values, "unmeasured": bundle._released_candidates()}
    if order == "chain":
        report.update(_chain_keyed_values(bundle, values))
    report["order"] = order
    return report


def _chain_keyed_values(bundle: Bundle, values: Mapping[str, Any]) -> dict[str, Any]:
    """Order measured members by the validated operation chain."""

    operations = bundle.history
    validate_operation_chain(operations)
    by_id = {operation.operation_id: operation for operation in operations}
    construction = bundle._member_construction
    chained: list[tuple[int, int, str]] = []
    unchained: list[str] = []
    for position, name in enumerate(values):
        operation_id = construction.get(name, {}).get("operation_id")
        operation = by_id.get(operation_id) if isinstance(operation_id, str) else None
        if operation is None:
            unchained.append(name)
        else:
            chained.append((operation.seq, position, name))
    chain = [
        {
            "seq": by_id[construction[name]["operation_id"]].seq,
            "operation_id": by_id[construction[name]["operation_id"]].operation_id,
            "kind": by_id[construction[name]["operation_id"]].kind,
            "member": name,
            "value": values[name],
        }
        for _seq, _position, name in sorted(chained)
    ]
    return {"chain": chain, "unchained": unchained}
