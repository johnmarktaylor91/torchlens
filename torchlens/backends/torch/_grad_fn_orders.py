"""Resolve GradFn ``order`` values from creators, roots and known topology.

The order of a GradFn record is the forward op order of the tensor whose
backward it computes. These helpers attribute it from the creator op, infer it
to a fixed point through child records when the creator is not logged, and
measure the coverage of the attribution for one backward pass.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from ..._data_classes.grad_fn import GradFn
    from ..._data_classes.grad_fn_call import GradFnCall


def _resolved_order_for_creator(
    creator_object_id: int | None,
    grad_fn_logs: dict[int, GradFn],
) -> int | None:
    """Return the derived order for a creator-attributed grad-fn node."""

    if creator_object_id is None:
        return 1
    creator = grad_fn_logs.get(creator_object_id)
    creator_order = None if creator is None else creator.order
    if creator_order is None:
        return None
    return creator_order + 1


def _pass_order_from_roots(
    root_grad_fn_ids: Sequence[int],
    grad_fn_logs: dict[int, GradFn],
    pass_orders: dict[int, int],
) -> int | None:
    """Derive a backward pass order from its root grad-fn nodes."""

    root_orders = [
        _root_based_order(grad_fn_logs[root_id], pass_orders)
        for root_id in root_grad_fn_ids
        if root_id in grad_fn_logs
    ]
    if not root_orders:
        return None
    if any(root_order is None for root_order in root_orders):
        return None
    if len(root_orders) != len(root_grad_fn_ids):
        return None
    unique_orders = set(root_orders)
    if len(unique_orders) != 1:
        return None
    return cast(int, root_orders[0])


def _root_based_order(root_grad_fn: GradFn, pass_orders: dict[int, int]) -> int | None:
    """Return the ROOT-based pass order for one root grad-fn.

    Parameters
    ----------
    root_grad_fn:
        Root autograd node for the backward pass.
    pass_orders:
        Already materialized pass orders keyed by pass index.

    Returns
    -------
    int | None
        Root-based order, or ``None`` when the root cannot be attributed.
    """

    origin_backward_pass = root_grad_fn.origin_backward_pass
    if origin_backward_pass is None:
        return root_grad_fn.order
    origin_order = pass_orders.get(origin_backward_pass)
    if origin_order is None:
        return root_grad_fn.order
    return origin_order + 1


def _order_attribution_coverage(
    calls: Sequence[GradFnCall],
    grad_fn_logs: dict[int, GradFn],
) -> float | None:
    """Return the fraction of calls with resolved grad-fn order metadata."""

    if not calls:
        return None
    labels_with_order = {
        grad_fn.label for grad_fn in grad_fn_logs.values() if grad_fn.order is not None
    }
    covered = sum(1 for call in calls if call.grad_fn_label in labels_with_order)
    return covered / len(calls)


def _max_call_order(
    calls: Sequence[GradFnCall],
    grad_fn_logs: dict[int, GradFn],
) -> int | None:
    """Return the maximum resolved GradFn order among calls in one pass.

    Parameters
    ----------
    calls:
        Materialized GradFn calls for a backward pass.
    grad_fn_logs:
        GradFn records keyed by object id.

    Returns
    -------
    int | None
        Highest order observed among the pass calls, or ``None`` when no call has
        resolved order metadata.
    """

    label_to_order = {
        grad_fn.label: grad_fn.order
        for grad_fn in grad_fn_logs.values()
        if grad_fn.order is not None
    }
    call_orders = [
        label_to_order[call.grad_fn_label] for call in calls if call.grad_fn_label in label_to_order
    ]
    return max(call_orders) if call_orders else None


def _assign_grad_fn_orders(grad_fn_logs: dict[int, GradFn]) -> bool:
    """Populate resolved GradFn order values from creators and known topology.

    A creator-attributed node sits one order above its creator. A creator-less
    node is inferred into the highest order among the nodes it feeds, whether
    or not a forward op is paired with it: only a node created after a
    ``create_graph`` gradient can feed a higher-order node (an op applied to
    that gradient, captured when the gradient is taken inside forward), and
    ordinary forward nodes, which feed only order-1 nodes, stay at 1.
    The two rules run to a fixed point: inferring a node upward re-derives the
    orders of the nodes its own differentiation created, so every
    creator-attributed order stays ``creator.order + 1`` even when the
    creator itself was only inferred. Returns whether the sweeps reached the
    fixed point (``False`` only when the cycle guard stopped them).
    """

    for grad_fn_record in grad_fn_logs.values():
        grad_fn_record.order = _resolved_order_for_creator(
            grad_fn_record.creator_object_id,
            grad_fn_logs,
        )

    # Orders only rise, by at least one per productive sweep, and a record's
    # order is bounded by the number of records; the cap only guards a cyclic
    # topology (ids reused across records), which the backward-graph
    # invariants then report.
    for _sweep in range(len(grad_fn_logs) + 1):
        if not _raise_grad_fn_orders_once(grad_fn_logs):
            return True
    return False


def _assign_new_grad_fn_orders(
    grad_fn_logs: Mapping[int, GradFn], new_records: Mapping[int, GradFn]
) -> bool:
    """Resolve orders for creator-less records appended after converged ``grad_fn_logs``.

    No existing record feeds or is created by an appended one, so the existing
    orders stay at their fixed point and the appended records rise to theirs
    alone, as in a scratch ``_assign_grad_fn_orders``. Returns ``False`` when
    the appended records do not converge (a cycle), so the caller rebuilds.
    """

    for grad_fn_record in new_records.values():
        grad_fn_record.order = 1
    for _sweep in range(len(new_records) + 1):
        changed = False
        for grad_fn_record in new_records.values():
            child_orders = [
                child.order
                for child_id in grad_fn_record.next_grad_fn_ids
                if (child := new_records.get(child_id) or grad_fn_logs.get(child_id)) is not None
                and child.order is not None
            ]
            if child_orders and max(child_orders) > 1 and grad_fn_record.order != max(child_orders):
                grad_fn_record.order = max(child_orders)
                changed = True
        if not changed:
            return True
    return False


def _raise_grad_fn_orders_once(grad_fn_logs: dict[int, GradFn]) -> bool:
    """Run one order sweep over ``grad_fn_logs``; return whether any order moved."""

    changed = False
    for grad_fn_record in grad_fn_logs.values():
        if grad_fn_record.creator_object_id is not None:
            resolved_order = _resolved_order_for_creator(
                grad_fn_record.creator_object_id, grad_fn_logs
            )
            if resolved_order is not None and grad_fn_record.order != resolved_order:
                grad_fn_record.order = resolved_order
                changed = True
            continue
        child_orders: list[int] = []
        for child_id in grad_fn_record.next_grad_fn_ids:
            child = grad_fn_logs.get(child_id)
            if child is not None and child.order is not None:
                child_orders.append(child.order)
        if not child_orders:
            continue
        inferred_order = max(child_orders)
        if inferred_order > 1 and grad_fn_record.order != inferred_order:
            grad_fn_record.order = inferred_order
            changed = True
    return changed
