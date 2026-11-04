"""Module membership of ``GradFn`` records: paired addresses and inferred containment.

A record paired with a forward op or a parameter takes that op's or that
parameter's module. An intervening node (one no forward op records) inherits
the module of its best neighbor, swept to a fixed point. The incremental
projection fold replays the same sweeps for newly discovered nodes only, so a
fold and a scratch rebuild agree; ``_infer_grad_fn_module_membership`` records
the sweep at which each record was set so the replay can see the timeline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from ..._data_classes.grad_fn import _paired_pass_op

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from ..._data_classes.grad_fn import GradFn

MembershipSweeps = dict[int, "int | None"]
"""Object id to the inference sweep that set its module (0 = set when built).

``None`` marks a record the sweeps may still set (eligible, never set); an
absent id is one the sweeps skip (carved out or creator-attributed)."""


def _param_module_address(trace: Any, param_ref: object | None) -> str | None:
    """Return the owning module address for a parameter reference.

    Parameters
    ----------
    trace:
        Trace containing parameter logs.
    param_ref:
        Parameter address recorded on an ``AccumulateGrad`` discovery event.

    Returns
    -------
    str | None
        Owning module address, or ``None`` when the parameter is unknown.
    """

    if not isinstance(param_ref, str):
        return None
    param_logs = trace._param_logs
    if param_ref not in param_logs:
        return None
    param_log = param_logs[param_ref]
    return None if param_log is None else getattr(param_log, "module_address", None)


def _op_module_address(
    trace: Any,
    op_label: str | None,
    grad_fn_object_id: int | None = None,
) -> str | None:
    """Return the containing module address for a paired forward op label.

    Parameters
    ----------
    trace:
        Trace containing op logs.
    op_label:
        Forward layer label paired to a grad-fn record.
    grad_fn_object_id:
        Autograd identity of the grad-fn node, selecting its own pass on a
        multi-pass layer.

    Returns
    -------
    str | None
        Module address for the paired op, if known.
    """

    op = _paired_pass_op(trace, op_label, grad_fn_object_id)
    if op is None:
        return None
    module_address = getattr(op, "module_address", None)
    if module_address is not None:
        return cast(str, module_address)
    atomic_module_address = getattr(op, "atomic_module_address", None)
    if atomic_module_address is not None:
        return cast(str, atomic_module_address)
    output_module_calls = list(getattr(op, "output_of_module_calls", []) or [])
    if output_module_calls:
        return str(output_module_calls[-1]).rsplit(":", 1)[0]
    module_calls = list(getattr(op, "modules", []) or [])
    if not module_calls:
        return None
    return str(module_calls[-1]).rsplit(":", 1)[0]


def _op_raw_index(trace: Any, grad_fn_record: GradFn) -> int | None:
    """Return the forward raw index backing a grad-fn's module attribution.

    Parameters
    ----------
    trace:
        Trace containing op logs.
    grad_fn_record:
        GradFn whose attributed op position should be inspected.

    Returns
    -------
    int | None
        Forward raw index, or ``None`` for parameter-only and unresolved records.
    """

    op = _paired_pass_op(trace, grad_fn_record.op_label, grad_fn_record.grad_fn_object_id)
    if op is None:
        return None
    return int(getattr(op, "raw_index"))


def _post_forward_grad_fn_ids(grad_fn_logs: dict[int, GradFn]) -> set[int]:
    """Return loss-construction grad-fn ids before the first op-anchored node.

    Parameters
    ----------
    grad_fn_logs:
        Projected GradFn records in backward discovery order.

    Returns
    -------
    set[int]
        Unpaired, non-creator-attributed ids that should not receive inferred
        module containment.
    """

    op_steps = [
        grad_fn_record.step_index
        for grad_fn_record in grad_fn_logs.values()
        if grad_fn_record.has_op
    ]
    if not op_steps:
        return set()
    first_op_step = min(op_steps)
    return {
        object_id
        for object_id, grad_fn_record in grad_fn_logs.items()
        if (
            grad_fn_record.step_index < first_op_step
            and not grad_fn_record.has_op
            and grad_fn_record.creator_object_id is None
        )
    }


def _candidate_module_sort_key(trace: Any, candidate: GradFn) -> tuple[int, int, str]:
    """Return the deterministic P5 tie-break key for a containment candidate.

    Parameters
    ----------
    trace:
        Trace containing forward op logs.
    candidate:
        Neighbor GradFn with resolved module containment.

    Returns
    -------
    tuple[int, int, str]
        Sort key where smaller values are preferred.
    """

    raw_index = _op_raw_index(trace, candidate)
    consumer_rank = -(raw_index if raw_index is not None else -1)
    return (consumer_rank, candidate.step_index, candidate.label)


def _inherit_module_from_best_neighbor(
    trace: Any, grad_fn_record: GradFn, candidates: Sequence[GradFn]
) -> bool:
    """Give ``grad_fn_record`` the module of its best set neighbor; return whether it took one."""

    if not candidates:
        return False
    winner = min(candidates, key=lambda candidate: _candidate_module_sort_key(trace, candidate))
    if winner.module_address is None:
        return False
    grad_fn_record.module_address = winner.module_address
    grad_fn_record.modules = list(winner.modules) or [winner.module_address]
    grad_fn_record.module_membership_source = "inferred"
    return True


def _infer_grad_fn_module_membership(trace: Any) -> MembershipSweeps:
    """Populate inferred module containment for intervening backward nodes.

    Parameters
    ----------
    trace:
        Trace whose ``grad_fn_logs`` already have graph relations synchronized.

    Returns
    -------
    MembershipSweeps
        The sweep that set each record's module (see ``MembershipSweeps``).
        GradFn ``modules``, ``module_address``, and ``module_membership_source``
        fields are updated in place.
    """

    grad_fn_logs = trace._grad_fn_logs
    label_to_id = {grad_fn.label: object_id for object_id, grad_fn in grad_fn_logs.items()}
    carve_out_ids = _post_forward_grad_fn_ids(grad_fn_logs)
    sweeps: MembershipSweeps = {}
    for object_id, grad_fn_record in grad_fn_logs.items():
        if grad_fn_record.module_membership_source is not None:
            sweeps[object_id] = 0
        elif object_id not in carve_out_ids and grad_fn_record.creator_object_id is None:
            sweeps[object_id] = None
    sweep = 0
    changed = True
    while changed:
        changed = False
        sweep += 1
        for object_id, grad_fn_record in grad_fn_logs.items():
            if sweeps.get(object_id, 0) is not None:
                continue
            candidates = [
                grad_fn_logs[label_to_id[label]]
                for label in [*grad_fn_record.parents, *grad_fn_record.children]
                if label in label_to_id
                and grad_fn_logs[label_to_id[label]].module_membership_source is not None
            ]
            if _inherit_module_from_best_neighbor(trace, grad_fn_record, candidates):
                sweeps[object_id] = sweep
                changed = True
    return sweeps


def _replay_membership_for_new_grad_fns(
    trace: Any,
    grad_fn_logs: Mapping[int, GradFn],
    new_records: Mapping[int, GradFn],
    neighbors: Mapping[int, Sequence[int]],
    sweeps: MembershipSweeps,
) -> MembershipSweeps | None:
    """Replay the inference sweeps for records appended after ``grad_fn_logs``.

    The appended records are creator-less intervening nodes discovered after
    every existing record; none is a parent of an existing record. Each is set
    by the sweep that would set it in a scratch rebuild: an existing neighbor
    is visible from the sweep recorded for it in ``sweeps``. An existing
    record's own sweep could move only when an appended parent becomes a
    candidate before the record was set; that case returns ``None`` so the
    caller rebuilds from scratch.

    Parameters
    ----------
    trace:
        Trace owning the records.
    grad_fn_logs:
        The existing records, unchanged.
    new_records:
        The appended records in discovery order; their module fields are set here.
    neighbors:
        Object ids of each appended record's parents then children, as the
        scratch sweep lists them.
    sweeps:
        The existing records' sweeps.

    Returns
    -------
    MembershipSweeps | None
        The appended records' sweeps, or ``None`` when only a rebuild is exact.
    """

    new_sweeps: MembershipSweeps = dict.fromkeys(new_records)
    last_existing_sweep = max((value for value in sweeps.values() if value), default=0)
    sweep = 0
    changed = True
    while changed or sweep < last_existing_sweep:
        changed = False
        sweep += 1
        for object_id, grad_fn_record in new_records.items():
            if new_sweeps[object_id] is not None:
                continue
            candidates = [
                new_records[neighbor_id]
                if neighbor_id in new_records
                else grad_fn_logs[neighbor_id]
                for neighbor_id in neighbors[object_id]
                if _visible_at(neighbor_id, sweep, new_sweeps, sweeps)
            ]
            if _inherit_module_from_best_neighbor(trace, grad_fn_record, candidates):
                new_sweeps[object_id] = sweep
                changed = True
    for object_id, set_sweep in new_sweeps.items():
        if set_sweep is None:
            continue
        for neighbor_id in neighbors[object_id]:
            if neighbor_id in new_records or neighbor_id not in sweeps:
                continue
            existing_sweep = sweeps[neighbor_id]
            if existing_sweep != 0 and (existing_sweep is None or set_sweep + 1 < existing_sweep):
                return None
    return new_sweeps


def _visible_at(
    object_id: int, sweep: int, new_sweeps: MembershipSweeps, sweeps: MembershipSweeps
) -> bool:
    """Return whether a record's module is set when an appended record reads it in ``sweep``."""

    if object_id in new_sweeps:
        return new_sweeps[object_id] is not None
    # An existing record absent from ``sweeps`` was skipped by every sweep and
    # never set; a preset record is recorded at sweep 0.
    set_sweep = sweeps.get(object_id)
    return set_sweep is not None and set_sweep <= sweep


def _prefer_direct_pairing_source_for_paired_ops(trace: Any) -> None:
    """Mark direct op pairings as paired after containment inference.

    Parameters
    ----------
    trace:
        Trace whose ``grad_fn_logs`` should be normalized.

    Returns
    -------
    None
        Directly paired GradFn records keep any inferred module address but expose
        ``module_membership_source="paired"``.
    """

    for grad_fn_record in trace._grad_fn_logs.values():
        op = grad_fn_record.op
        if (
            op is not None
            and (getattr(op, "is_compute_op", False) or getattr(op, "is_compute_layer", False))
            and grad_fn_record.module_membership_source == "inferred"
        ):
            grad_fn_record.module_membership_source = "paired"
