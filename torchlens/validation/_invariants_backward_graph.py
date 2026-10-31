"""Backward graph and grad-function invariants."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..data_classes.trace import Trace
    from .invariants import (
        MetadataInvariantError,
        _check_backward_event_flow_invariants,
        _check_backward_pass_domain_invariants,
        _check_journal_seq_invariants,
        _layer_feeds_recorded_backward_roots,
        _layer_postdates_all_backward_triggers,
    )

__all__ = (
    "_check_backward_graph_invariants",
    "_check_backward_grad_fn_registry",
    "_check_backward_grad_fn_handle_records",
    "_check_backward_layer_backpointers",
    "_check_backward_saved_grad_records",
    "_check_backward_pass_record_consistency",
    "_check_backward_pass_index_density",
    "_check_grad_fn_topology_invariants",
    "_check_grad_fn_relation_list",
)


def _check_backward_graph_invariants(trace: Trace) -> None:
    """Check T: backward grad-fn metadata consistency.

    A forward layer with a recorded ``grad_fn_object_id`` must retain a
    layer-to-GradFn backpointer unless it is structurally outside the backward
    pass contract: its final ``step_index`` must be greater than every recorded
    backward trigger's structural forward boundary. The boundary starts from the
    trigger-time forward op count and is tightened by paired root GradFns and
    observed op-gradient events when those identify the walked forward prefix.
    This admits legitimate mid-forward ``autograd.grad`` cases where later
    forward layers did not exist when backward graph walking ran, while still
    failing pre-trigger layers whose backpointer was accidentally severed.
    A pre-trigger layer is also exempt when it provably sat OUTSIDE the
    backward walk (b7-opus R24-X): its recorded handle id was never projected
    into ``grad_fn_logs`` AND no autograd-visited op is reachable from it
    through grad-carrying edges — the dead-branch case (``_ = h.mean()``) and
    grad-severed subgraphs, where autograd never constructs a backpointer to
    retain.

    Parameters
    ----------
    trace:
        Postprocessed model log to validate.

    Raises
    ------
    MetadataInvariantError
        If backward metadata is internally inconsistent.
    """

    name = "backward_graph_invariants"
    sync_projection = getattr(trace, "_sync_backward_projection_if_needed", None)
    if callable(sync_projection):
        sync_projection()
    _check_journal_seq_invariants(trace, name)
    _check_backward_event_flow_invariants(trace, name)
    if not trace.grad_fn_logs:
        _refuse_backward_evidence_without_registry(trace, name)
        return

    valid_pass_indices = _check_backward_grad_fn_registry(trace, name)
    _check_backward_grad_fn_handle_records(trace, name, valid_pass_indices)
    _check_backward_layer_backpointers(trace, name)
    _check_backward_saved_grad_records(trace, name)
    _check_backward_pass_index_density(trace, name)
    _check_grad_fn_topology_invariants(trace, name)
    _check_backward_pass_domain_invariants(trace, name, valid_pass_indices)
    _check_backward_pass_record_consistency(trace, name, valid_pass_indices)


def _refuse_backward_evidence_without_registry(trace: Trace, name: str) -> None:
    """Refuse an empty grad-fn registry that sits beside backward evidence.

    The rest of the backward invariant family is gated on ``grad_fn_logs``
    being non-empty, and the capture-event journal is a run-scoped live fact
    that never survives pickle restore, fork, or bundle load -- so on a
    detached trace a WIPED registry used to silently skip every registry,
    backpointer, saved-grad, density, topology, and domain check. Backward
    projection can never legitimately leave ``backward_pass_logs``,
    ``num_backward_passes``, ``grad_fn_order``, or
    ``backward_root_grad_fn_object_ids`` populated while the registry is
    empty, so that state is corruption and the gate REFUSES instead of
    skipping. Forward-time ``grad_fn_object_id`` stamps are deliberately NOT
    treated as evidence: they exist on every grad-enabled forward-only
    capture. A pass record whose ``status`` is not ``"ok"`` is a DISCLOSED
    failed walk (the walk died before discovering any grad fn), so it is
    legitimately registry-free and does not count as evidence either.

    Parameters
    ----------
    trace:
        Trace whose ``grad_fn_logs`` is empty.
    name:
        Invariant check name for raised errors.

    Raises
    ------
    MetadataInvariantError
        If backward-projection evidence exists without a grad-fn registry.
    """

    evidence: list[str] = []
    backward_pass_logs = getattr(trace, "backward_pass_logs", None) or {}
    ok_pass_count = sum(
        1 for record in backward_pass_logs.values() if getattr(record, "status", "ok") == "ok"
    )
    if ok_pass_count:
        evidence.append(f"backward_pass_logs holds {ok_pass_count} ok-status pass record(s)")
    num_backward_passes = getattr(trace, "num_backward_passes", 0) or 0
    if isinstance(num_backward_passes, int) and num_backward_passes > len(backward_pass_logs):
        evidence.append(
            f"num_backward_passes={num_backward_passes} exceeds the "
            f"{len(backward_pass_logs)} recorded pass record(s)"
        )
    grad_fn_order = getattr(trace, "grad_fn_order", None) or ()
    if grad_fn_order:
        evidence.append(f"grad_fn_order holds {len(grad_fn_order)} id(s)")
    root_ids = getattr(trace, "backward_root_grad_fn_object_ids", None) or ()
    if root_ids:
        evidence.append(f"backward_root_grad_fn_object_ids holds {len(root_ids)} id(s)")
    if evidence:
        raise MetadataInvariantError(
            name,
            "grad_fn_logs is empty while backward evidence is present "
            f"({'; '.join(evidence)}): the grad-fn registry was wiped or never "
            "recorded, so the backward invariant family cannot verify this "
            "trace and refuses rather than silently skipping.",
        )


def _check_backward_grad_fn_registry(trace: Trace, name: str) -> set[int]:
    """Check backward GradFn registry and root references.

    Parameters
    ----------
    trace:
        Trace with populated backward GradFn metadata.
    name:
        Invariant check name for raised errors.

    Returns
    -------
    set[int]
        Valid backward pass indices.

    Raises
    ------
    MetadataInvariantError
        If registry or root references are inconsistent.
    """

    grad_fn_ids = set(trace.grad_fn_logs)
    order_ids = set(trace.grad_fn_order)
    if not order_ids <= grad_fn_ids:
        missing = sorted(order_ids - grad_fn_ids)
        raise MetadataInvariantError(name, f"grad_fn_order contains unknown ids {missing!r}")

    root_ids = trace.backward_root_grad_fn_object_ids
    if not isinstance(root_ids, list):
        raise MetadataInvariantError(
            name,
            f"backward_root_grad_fn_object_ids must be a list, got {type(root_ids).__name__}",
        )
    missing_root_ids = [root_id for root_id in root_ids if root_id not in trace.grad_fn_logs]
    if missing_root_ids:
        raise MetadataInvariantError(
            name,
            f"backward_root_grad_fn_object_ids {missing_root_ids!r} are not present in grad_fn_logs",
        )

    return set(getattr(trace, "backward_pass_logs", {}).keys())


def _check_backward_grad_fn_handle_records(
    trace: Trace,
    name: str,
    valid_pass_indices: set[int],
) -> None:
    """Check per-GradFn handle metadata consistency.

    Parameters
    ----------
    trace:
        Trace with populated backward GradFn metadata.
    name:
        Invariant check name for raised errors.
    valid_pass_indices:
        Backward pass indices known on the trace.

    Raises
    ------
    MetadataInvariantError
        If a GradFn handle has inconsistent fields or call records.
    """

    layer_labels = set(trace.layer_labels)
    for grad_fn_object_id, grad_fn_handle in trace.grad_fn_logs.items():
        if not re.fullmatch(r"[a-z0-9_]+_back_[1-9]\d*_[1-9]\d*", grad_fn_handle.label):
            raise MetadataInvariantError(
                name,
                f"{grad_fn_handle.label!r} does not match backward-native label grammar",
            )
        if grad_fn_handle.has_op and grad_fn_handle.op_label not in layer_labels:
            raise MetadataInvariantError(
                name,
                f"{grad_fn_handle.label} points to missing layer {grad_fn_handle.op_label!r}",
            )
        membership_source = getattr(grad_fn_handle, "module_membership_source", None)
        if membership_source not in {None, "paired", "inferred"}:
            raise MetadataInvariantError(
                name,
                f"{grad_fn_handle.label} has invalid module_membership_source "
                f"{membership_source!r}",
            )
        if membership_source is None:
            if grad_fn_handle.module_address is not None or grad_fn_handle.modules:
                raise MetadataInvariantError(
                    name,
                    f"{grad_fn_handle.label} has module containment without a source",
                )
        elif grad_fn_handle.module_address is None or not grad_fn_handle.modules:
            raise MetadataInvariantError(
                name,
                f"{grad_fn_handle.label} has incomplete {membership_source!r} module containment",
            )
        op = grad_fn_handle.op
        if grad_fn_handle.has_op != (op is not None):
            raise MetadataInvariantError(
                name,
                f"{grad_fn_handle.label} has inconsistent has_op/op fields",
            )
        if grad_fn_handle.grad_fn_object_id != grad_fn_object_id:
            raise MetadataInvariantError(
                name,
                f"{grad_fn_handle.label} stored id {grad_fn_handle.grad_fn_object_id!r} under {grad_fn_object_id!r}",
            )
        # Domain check for EVERY handle, not only higher-order ones. This used to
        # live under ``creator_object_id is not None``, which is populated only by
        # ``create_graph=True`` autograd, so an ordinary ``log_backward`` never
        # reached it. ``origin_backward_pass`` is declared ``int | None`` and is
        # legitimately None on handles built outside a backward pass, so None is
        # explicitly allowed here; the stricter creator-branch form below is
        # retained unchanged so the higher-order path loses nothing.
        origin_backward_pass = getattr(grad_fn_handle, "origin_backward_pass", None)
        if origin_backward_pass is not None and origin_backward_pass not in valid_pass_indices:
            raise MetadataInvariantError(
                name,
                f"{grad_fn_handle.label} has invalid origin backward pass {origin_backward_pass!r}",
            )
        creator_object_id = getattr(grad_fn_handle, "creator_object_id", None)
        if creator_object_id is not None:
            creator = trace.grad_fn_logs.get(creator_object_id)
            if creator is None:
                raise MetadataInvariantError(
                    name,
                    f"{grad_fn_handle.label} points to missing creator id {creator_object_id!r}",
                )
            if grad_fn_handle.origin_backward_pass not in valid_pass_indices:
                raise MetadataInvariantError(
                    name,
                    f"{grad_fn_handle.label} has invalid origin backward pass "
                    f"{grad_fn_handle.origin_backward_pass!r}",
                )
            if creator.order is not None and grad_fn_handle.order is not None:
                expected_order = creator.order + 1
                if grad_fn_handle.order != expected_order:
                    raise MetadataInvariantError(
                        name,
                        f"{grad_fn_handle.label} order {grad_fn_handle.order!r} does not "
                        f"match creator order + 1 ({expected_order!r})",
                    )
        call_ordinals = sorted(grad_fn_handle.calls.keys())
        if call_ordinals != list(range(1, len(call_ordinals) + 1)):
            raise MetadataInvariantError(
                name,
                f"{grad_fn_handle.label} has non-dense local call ordinals {call_ordinals!r}",
            )
        for ordinal, call in grad_fn_handle.calls.items():
            if call.ordinal != ordinal or call.call_index != ordinal:
                raise MetadataInvariantError(
                    name,
                    f"{grad_fn_handle.label}:{ordinal} has inconsistent local ordinal fields",
                )
            if call.backward_pass_index is None:
                raise MetadataInvariantError(
                    name,
                    f"{grad_fn_handle.label}:{ordinal} is missing backward_pass_index",
                )


def _check_backward_layer_backpointers(trace: Trace, name: str) -> None:
    """Check forward-layer backpointers into backward GradFn handles.

    Parameters
    ----------
    trace:
        Trace with populated backward GradFn metadata.
    name:
        Invariant check name for raised errors.

    Raises
    ------
    MetadataInvariantError
        If a layer points to a missing or severed GradFn handle.
    """

    for layer in trace.layer_list:
        grad_fn_object_id = layer.grad_fn_object_id
        if grad_fn_object_id is None:
            continue
        if layer.grad_fn is None:
            if _layer_postdates_all_backward_triggers(trace, layer):
                continue
            # Dead-branch carve-out (b7-opus R24-X): autograd visits a grad_fn
            # only when its output feeds a backward root along grad-carrying
            # edges, so a provably non-contributing layer (unconsumed output,
            # or a path severed by a grad-less op) never receives a
            # backpointer BY CONTRACT. The exemption requires BOTH independent
            # signals: the recorded handle id was never projected into
            # ``grad_fn_logs`` (a projected handle is proof of visitation, so
            # a dropped backpointer on it still raises) AND the grad-carrying
            # reachability walk — which fails closed on every resolution
            # gap — cannot reach any visited op downstream.
            if grad_fn_object_id not in getattr(
                trace, "grad_fn_logs", {}
            ) and not _layer_feeds_recorded_backward_roots(trace, layer):
                continue
            raise MetadataInvariantError(
                name,
                f"Layer {layer.layer_label} with grad_fn_handle id {grad_fn_object_id!r} "
                "is missing its GradFn backpointer",
            )
        if grad_fn_object_id not in trace.grad_fn_logs:
            raise MetadataInvariantError(
                name,
                f"Layer {layer.layer_label} points to missing grad_fn_handle id "
                f"{grad_fn_object_id!r}",
            )


def _check_backward_saved_grad_records(trace: Trace, name: str) -> None:
    """Check saved gradient-op records match layers that have gradients.

    Parameters
    ----------
    trace:
        Trace with populated backward GradFn metadata.
    name:
        Invariant check name for raised errors.

    Raises
    ------
    MetadataInvariantError
        If saved-grad op labels and layer ``has_grad`` flags disagree.
    """

    expected_saved_grad_labels = {layer.label for layer in trace.layer_list if layer.has_grad}
    saved_grad_labels = {op.label for op in trace.saved_grad_ops}
    if saved_grad_labels != expected_saved_grad_labels:
        raise MetadataInvariantError(
            name,
            "saved_grad_ops does not match layers with saved grad tensors",
        )


def _check_backward_pass_record_consistency(
    trace: Trace,
    name: str,
    valid_pass_indices: set[int],
) -> None:
    """Check backward pass logs and call references are internally consistent.

    Parameters
    ----------
    trace:
        Trace with populated backward GradFn metadata.
    name:
        Invariant check name for raised errors.
    valid_pass_indices:
        Backward pass indices known on the trace.

    Raises
    ------
    MetadataInvariantError
        If backward pass logs are not dense or calls reference missing passes.
    """

    for pass_index, backward_pass in getattr(trace, "backward_pass_logs", {}).items():
        if backward_pass.pass_index != pass_index:
            raise MetadataInvariantError(
                name,
                f"BackwardPass stored index {backward_pass.pass_index!r} under {pass_index!r}",
            )
        for call in backward_pass.grad_fn_calls:
            if call.backward_pass_index != pass_index:
                raise MetadataInvariantError(
                    name,
                    f"{call.call_label} is attached to pass {pass_index} but records "
                    f"pass {call.backward_pass_index}",
                )
    for grad_fn_handle in trace.grad_fn_logs.values():
        for call in grad_fn_handle.calls.values():
            if call.backward_pass_index not in valid_pass_indices:
                raise MetadataInvariantError(
                    name,
                    f"{call.label} references missing backward pass {call.backward_pass_index}",
                )


def _check_backward_pass_index_density(trace: Trace, name: str) -> None:
    """Check backward pass log keys are dense.

    Parameters
    ----------
    trace:
        Trace with populated backward pass metadata.
    name:
        Invariant check name for raised errors.

    Raises
    ------
    MetadataInvariantError
        If backward pass log keys are not exactly ``1..num_backward_passes``.
    """

    expected_pass_indices = list(range(1, trace.num_backward_passes + 1))
    actual_pass_indices = sorted(getattr(trace, "backward_pass_logs", {}).keys())
    if actual_pass_indices != expected_pass_indices:
        raise MetadataInvariantError(
            name,
            f"backward_pass_logs keys {actual_pass_indices!r} are not dense "
            f"1..{trace.num_backward_passes}",
        )


def _check_grad_fn_topology_invariants(trace: Trace, name: str) -> None:
    """Check backward GradFn relation lists for reciprocal, resolvable links.

    The precondition contract is backward-capture only: callers invoke this
    helper after proving ``trace.grad_fn_logs`` is populated. Forward-only
    traces legitimately have no GradFn topology and are skipped by
    ``_check_backward_graph_invariants`` before this helper is reached.

    Parameters
    ----------
    trace:
        Trace with materialized backward GradFn projections.
    name:
        Invariant check name to use in raised errors.

    Raises
    ------
    MetadataInvariantError
        If a GradFn relation references a missing node or lacks its reciprocal
        back-reference.
    """

    grad_fns_by_label = {
        grad_fn_handle.label: grad_fn_handle for grad_fn_handle in trace.grad_fn_logs.values()
    }
    grad_fn_ids = set(trace.grad_fn_logs)
    for grad_fn_handle in trace.grad_fn_logs.values():
        missing_next_ids = [
            next_id for next_id in grad_fn_handle.next_grad_fn_ids if next_id not in grad_fn_ids
        ]
        if missing_next_ids:
            raise MetadataInvariantError(
                name,
                f"{grad_fn_handle.label} has next_grad_fn_ids missing from grad_fn_logs: "
                f"{missing_next_ids!r}",
            )

        _check_grad_fn_relation_list(
            grad_fn_handle,
            "parents",
            "children",
            grad_fns_by_label,
            name,
        )
        _check_grad_fn_relation_list(
            grad_fn_handle,
            "children",
            "parents",
            grad_fns_by_label,
            name,
        )
        _check_grad_fn_relation_list(
            grad_fn_handle,
            "siblings",
            "siblings",
            grad_fns_by_label,
            name,
        )
        _check_grad_fn_relation_list(
            grad_fn_handle,
            "co_parents",
            "co_parents",
            grad_fns_by_label,
            name,
        )

        for flag_name, relation_name in (
            ("has_parents", "parents"),
            ("has_children", "children"),
            ("has_siblings", "siblings"),
            ("has_co_parents", "co_parents"),
        ):
            if getattr(grad_fn_handle, flag_name) != bool(getattr(grad_fn_handle, relation_name)):
                raise MetadataInvariantError(
                    name,
                    f"{grad_fn_handle.label} has inconsistent {flag_name}/{relation_name}",
                )


def _check_grad_fn_relation_list(
    grad_fn_handle: object,
    relation_name: str,
    reciprocal_name: str,
    grad_fns_by_label: Mapping[str, object],
    name: str,
) -> None:
    """Check that one GradFn relation list resolves and reciprocates.

    Parameters
    ----------
    grad_fn_handle:
        GradFn object whose relation list is being validated.
    relation_name:
        Name of the outbound relation list.
    reciprocal_name:
        Name of the relation list expected on each target.
    grad_fns_by_label:
        Mapping from GradFn label to GradFn object.
    name:
        Invariant check name to use in raised errors.

    Raises
    ------
    MetadataInvariantError
        If a relation label is missing or lacks the reciprocal label.
    """

    source_label = str(getattr(grad_fn_handle, "label"))
    related_labels = getattr(grad_fn_handle, relation_name)
    if not isinstance(related_labels, list):
        raise MetadataInvariantError(
            name,
            f"{source_label} has non-list {relation_name}: {related_labels!r}",
        )
    for related_label in related_labels:
        related = grad_fns_by_label.get(related_label)
        if related is None:
            raise MetadataInvariantError(
                name,
                f"{source_label} {relation_name} references missing GradFn {related_label!r}",
            )
        reciprocal_labels = getattr(related, reciprocal_name)
        if source_label not in reciprocal_labels:
            raise MetadataInvariantError(
                name,
                f"{source_label} {relation_name} references {related_label!r}, but "
                f"{related_label!r} does not list {source_label!r} in {reciprocal_name}",
            )
