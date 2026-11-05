"""Ancestor storage, producer policy, and exhaustive record freezing."""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import torch

from ..._data_classes.op import (
    Op,
)
from ..._ir.events import (
    ParentEdge,
)
from ..._utils._mutation_names import _is_inplace_augmented_assignment_dunder
from ..._utils._tensor_label import (
    active_label_session_token,
    get_tensor_label,
    set_tensor_label,
)
from ..._utils._torch_compat import (
    tensor_version_or_none,
)
from . import module_stack as _mstack
from ._completeness_origins import internal_scalar_read
from ._same_object_handoff import set_same_object_live_value, set_same_object_mutation

if TYPE_CHECKING:
    from ..._data_classes.trace import Trace
    from .ops import (  # annotations only; runtime binding at the bottom
        CaptureProducerMode,
        CaptureProducerPolicy,
    )


__all__ = (
    "_get_ancestor_field",
    "_set_ancestor_field",
    "_delete_ancestor_field",
    "_compact_ancestor_sets",
    "_is_inplace_augmented_assignment_dunder",
    "_record_label_version_snapshot",
    "_bind_call_alias_labels",
    "_restore_call_alias_labels",
    "_label_version_baseline",
    "_stamp_same_object",
    "get_capture_producer_policy",
    "set_capture_producer_policy",
    "_should_keep_alias_mutation_contract",
    "_snapshot_exhaustive_module_stack",
    "_parent_edges_from_fields",
)


def _get_ancestor_field(op: Op, field_name: str) -> set[str] | frozenset[str]:
    """Return one public ancestor closure view.

    During postprocess the cell holds the mutable staging set and is returned
    as-is. Once ``_compact_ancestor_sets`` has interned the closure, reads
    return the bitset's cached ``frozenset`` view (one per distinct closure,
    shared by every member op — safe because it is immutable) WITHOUT writing
    it back, so the compact encoding stays the storage authority.

    Parameters
    ----------
    op
        Operation whose field is being read.
    field_name
        Ancestor field name backed by an original ``Op`` slot descriptor.

    Returns
    -------
    set[str] | frozenset[str]
        Staging set during postprocess; immutable view once finished.
    """

    descriptor = _ANCESTOR_SLOT_DESCRIPTORS[field_name]
    value = descriptor.__get__(op, type(op))
    if isinstance(value, _AncestorBitset):
        return value.frozen_view()
    return cast(set[str], value)


def _set_ancestor_field(op: Op, field_name: str, value: set[str]) -> None:
    """Store one ancestor field through its original ``Op`` slot.

    Parameters
    ----------
    op
        Operation whose field is being assigned.
    field_name
        Ancestor field name backed by an original ``Op`` slot descriptor.
    value
        Mutable public set to retain.
    """

    _ANCESTOR_SLOT_DESCRIPTORS[field_name].__set__(op, value)


def _delete_ancestor_field(op: Op, field_name: str) -> None:
    """Delete one ancestor field through its original ``Op`` slot.

    Parameters
    ----------
    op
        Operation whose field is being deleted during cleanup.
    field_name
        Ancestor field name backed by an original ``Op`` slot descriptor.
    """

    _ANCESTOR_SLOT_DESCRIPTORS[field_name].__delete__(op)


def _get_root_ancestors(op: Op) -> set[str] | frozenset[str]:
    """Return ``op.root_ancestors``: staging set or frozen closure view."""

    return _get_ancestor_field(op, "root_ancestors")


def _set_root_ancestors(op: Op, value: set[str]) -> None:
    """Assign ``op.root_ancestors`` through its preserved slot descriptor."""

    _set_ancestor_field(op, "root_ancestors", value)


def _delete_root_ancestors(op: Op) -> None:
    """Delete ``op.root_ancestors`` through its preserved slot descriptor."""

    _delete_ancestor_field(op, "root_ancestors")


def _get_internal_source_ancestors(op: Op) -> set[str] | frozenset[str]:
    """Return ``op.internal_source_ancestors``: staging set or frozen closure view."""

    return _get_ancestor_field(op, "internal_source_ancestors")


def _set_internal_source_ancestors(op: Op, value: set[str]) -> None:
    """Assign ``op.internal_source_ancestors`` through its preserved slot descriptor."""

    _set_ancestor_field(op, "internal_source_ancestors", value)


def _delete_internal_source_ancestors(op: Op) -> None:
    """Delete ``op.internal_source_ancestors`` through its preserved slot descriptor."""

    _delete_ancestor_field(op, "internal_source_ancestors")


def _compact_ancestor_sets(trace: Trace) -> None:
    """Replace finished Ops' two large ancestor sets with interned integer bitmaps.

    The original slot descriptors remain the storage authority. Public attribute
    reads lazily restore and cache a real mutable ``set``, preserving the declared
    API, ordinary mutation behavior, pickling, and serialization. Ops whose fields
    remain unread share one trace-local label table and one immutable bitmap object
    per distinct closure.

    Parameters
    ----------
    trace
        Finished trace whose retained Op metadata should be compacted.
    """

    ops = trace._layer_list
    if not ops:
        return

    raw_fields: list[tuple[Op, str, set[str] | _AncestorBitset]] = []
    labels_by_first_use: dict[str, None] = {}
    for op in ops:
        for field_name in _ANCESTOR_FIELD_NAMES:
            descriptor = _ANCESTOR_SLOT_DESCRIPTORS[field_name]
            value = descriptor.__get__(op, type(op))
            raw_fields.append((op, field_name, value))
            if isinstance(value, _AncestorBitset):
                labels_by_first_use.update(dict.fromkeys(value.materialize()))
            else:
                labels_by_first_use.update(dict.fromkeys(value))

    labels = tuple(labels_by_first_use)
    label_indices = {label: index for index, label in enumerate(labels)}
    bitset_pool: dict[int, _AncestorBitset] = {}
    for op, field_name, value in raw_fields:
        if isinstance(value, _AncestorBitset) and value.labels == labels:
            bits = value.bits
        else:
            values = value.materialize() if isinstance(value, _AncestorBitset) else value
            bits = 0
            for label in values:
                bits |= 1 << label_indices[label]
        bitset = bitset_pool.get(bits)
        if bitset is None:
            bitset = _AncestorBitset(labels, bits)
            bitset_pool[bits] = bitset
        _ANCESTOR_SLOT_DESCRIPTORS[field_name].__set__(op, bitset)


def _record_label_version_snapshot(t: Any, *, mutated_via: tuple[Any, Any] | None = None) -> None:
    """Record ``t``'s current version, keyed to the active label session.

    Parameters
    ----------
    t
        Tensor that was just labeled (as an op output, or by live in-place /
        alias label propagation).
    mutated_via
        ``(view, prior_label)`` when ``t`` is a storage alias whose label was just
        advanced because an in-place op wrote through ``view``; remembered for
        :func:`_bind_call_alias_labels`, chained to the alias's previous advance
        when that advance produced ``prior_label`` in this session.
    """

    if not isinstance(t, torch.Tensor):
        return
    if mutated_via is not None:
        view, prior_label = mutated_via
        token = active_label_session_token()
        previous = _VIEW_MUTATION_RELABELS.get(t)
        if previous is not None and (previous[0] != token or previous[2] != prior_label):
            previous = None
        _VIEW_MUTATION_RELABELS[t] = (token, prior_label, get_tensor_label(t), id(view), previous)
    # ``_version`` is a WITNESSED input-metadata property (r33): the scoped patch
    # records a genuine USER ``x._version`` read on a model-input leaf. This snapshot is
    # TorchLens's OWN bookkeeping read, so it must run under the internal marker or it
    # spuriously records a ``_version`` fact for every model, falsely diverging any
    # runtime input whose version counter differs from capture (an over-trigger).
    with internal_scalar_read():
        version = tensor_version_or_none(t)
    if version is not None:
        _LABEL_VERSION_SNAPSHOT[t] = (active_label_session_token(), version)


def _call_alias_tensors(args: tuple[Any, ...], kwargs: dict[str, Any]) -> list[torch.Tensor]:
    """Return the distinct rebindable tensors a call reads, top level and one list deep."""

    found: dict[int, torch.Tensor] = {}
    for value in (*args, *kwargs.values()):
        for item in value if isinstance(value, (list, tuple)) else (value,):
            if (
                isinstance(item, torch.Tensor)
                and not isinstance(item, torch.nn.Parameter)
                and item.device.type != "meta"
            ):
                found.setdefault(id(item), item)
    return list(found.values())


def _resolve_call_alias_labels(tensors: list[torch.Tensor]) -> dict[int, str]:
    """Return the label each colliding argument of one call binds to, by tensor id.

    An argument whose label another argument also carries, and which holds it only
    because an in-place op wrote through a view of it (a live relabel entry of this
    session naming that label), steps back to its label before that write; the step
    repeats along the alias's chained relabels while a collision remains. A tensor
    that produced its label itself never moves, so the written view keeps the
    mutation's label. A collision no relabel explains is left as is (fail closed).

    Parameters
    ----------
    tensors
        The call's distinct argument tensors.

    Returns
    -------
    dict[int, str]
        ``id(tensor)`` to its label for this call, only for tensors that move.
    """

    token = active_label_session_token()
    bound: dict[int, str] = {}
    cursor: dict[int, Any] = {}
    for tensor in tensors:
        label = get_tensor_label(tensor)
        if label is None:
            continue
        entry = _VIEW_MUTATION_RELABELS.get(tensor)
        live = entry is not None and entry[0] == token and entry[2] == label
        bound[id(tensor)] = label
        cursor[id(tensor)] = entry if live else None
    moved: dict[int, str] = {}
    changed = True
    while changed:
        changed = False
        holders: dict[str, list[int]] = {}
        for tensor_id, label in bound.items():
            holders.setdefault(label, []).append(tensor_id)
        for tensor_ids in holders.values():
            for tensor_id in tensor_ids if len(tensor_ids) > 1 else ():
                entry = cursor[tensor_id]
                if entry is None or entry[1] is None:
                    continue
                bound[tensor_id] = moved[tensor_id] = entry[1]
                cursor[tensor_id] = entry[4]
                changed = True
    return moved


def _bind_call_alias_labels(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> list[tuple[torch.Tensor, str | None, str]]:
    """Bind each argument of one call to the label of the value it is, for this call.

    An in-place op on a view advances its base's label to that op, so the base's
    later consumers see the write (``wrappers._propagate_mutation_label_to_storage_aliases``).
    A call that reads the base AND the written view (``y + col`` after
    ``col.add_(1)``; the write-back ``x[idx] = t`` of ``x[idx] += v``) then held one
    label for two different values, which a per-(parent, child) version snapshot cannot
    carry, so argument logging failed. Here the base binds to its pre-write label (its
    actual content reaches replay through the per-child version snapshot) and the view
    keeps the mutation's label; see :func:`_resolve_call_alias_labels`. The caller hands
    the result to :func:`_restore_call_alias_labels` once the call's argument reads are
    recorded, so the base's later consumers still bind to the mutation. A capture with
    no live view-write relabel pays one emptiness check per call.

    Parameters
    ----------
    args
        The call's live positional arguments.
    kwargs
        The call's live keyword arguments.

    Returns
    -------
    list[tuple[torch.Tensor, str | None, str]]
        ``(tensor, live_label, bound_label)`` for every argument that moved.
    """

    if not _VIEW_MUTATION_RELABELS:
        return []
    tensors = _call_alias_tensors(args, kwargs)
    moved = _resolve_call_alias_labels(tensors) if len(tensors) > 1 else {}
    bound = []
    for tensor in tensors if moved else ():
        label = moved.get(id(tensor))
        if label is not None:
            bound.append((tensor, get_tensor_label(tensor), label))
            set_tensor_label(tensor, label)
    return bound


def _restore_call_alias_labels(bound: list[tuple[torch.Tensor, str | None, str]]) -> None:
    """Give every argument :func:`_bind_call_alias_labels` moved its live label back.

    Parameters
    ----------
    bound
        What :func:`_bind_call_alias_labels` returned for the call.
    """

    for tensor, live_label, bound_label in bound:
        if live_label is not None and get_tensor_label(tensor) == bound_label:
            set_tensor_label(tensor, live_label)


def _label_version_baseline(t: Any) -> int | None:
    """Return ``t``'s labeled-version baseline, only if the ACTIVE session recorded it.

    Parameters
    ----------
    t
        Tensor being logged as an op output.

    Returns
    -------
    int | None
        The version recorded when this session last labeled ``t``, or ``None``
        when no baseline exists or the entry belongs to another (stale) session.
    """

    entry = _LABEL_VERSION_SNAPSHOT.get(t)
    if entry is None:
        return None
    session_token, version = entry
    if session_token is None or session_token != active_label_session_token():
        return None
    return version


def _stamp_same_object(
    logged: Any, receiver: Any, mutation_signature: bool, distinct_return: bool = False
) -> None:
    """Stamp a same-object call's mutation verdict on the copy it is logged against.

    The op-output classifier only sees the fresh logging copy, never the live
    receiver, so the wrapper decides here: the call's mutation signature (an
    in-place name, a setter, or an ``inplace=True`` request), refined by the
    receiver's version counter. An unchanged counter since the receiver was last
    labeled proves no write (eval-mode ``dropout_``); an unknown counter keeps
    the signature.

    Parameters
    ----------
    logged
        The copy the call's output is logged against.
    receiver
        The live first argument the call returned.
    mutation_signature
        Whether the call's signature requests mutation.
    distinct_return
        Whether the caller receives the copy itself (the module-boundary
        ``identity`` mint), so the copy is the live object.

    A non-mutating return also stamps the live receiver: the caller and every
    downstream op consume it, not the copy, so a by-reference payload must be
    it. A Parameter receiver is logged through its own mutation path and keeps
    the copy.
    """

    if not isinstance(logged, torch.Tensor):
        return
    verdict = mutation_signature
    baseline = _label_version_baseline(receiver) if verdict else None
    if baseline is not None:
        # TorchLens bookkeeping read: the receiver may be registered model state.
        with internal_scalar_read():
            current = tensor_version_or_none(receiver)
        verdict = current is None or current != baseline
    with contextlib.suppress(AttributeError):
        set_same_object_mutation(logged, verdict)
        if not (verdict or distinct_return or isinstance(receiver, torch.nn.Parameter)):
            set_same_object_live_value(logged, receiver)


def get_capture_producer_policy(mode: CaptureProducerMode) -> CaptureProducerPolicy:
    """Return the precomputed producer policy for ``mode``.

    Parameters
    ----------
    mode
        Capture mode to route.

    Returns
    -------
    CaptureProducerPolicy
        Cached policy object used on the decorated-operation hot path.
    """

    policy = _CAPTURE_PRODUCER_POLICIES.get(mode)
    if policy is None:
        emit = _EMIT_BY_MODE[mode]
        policy = CaptureProducerPolicy(mode, emit)
        _CAPTURE_PRODUCER_POLICIES[mode] = policy
    return policy


def set_capture_producer_policy(trace: Trace, mode: CaptureProducerMode) -> None:
    """Attach a precomputed producer policy to ``trace``.

    Compiled once at session setup; the hot path only ever touches the
    precompiled policy object.

    Parameters
    ----------
    trace
        Trace receiving the hot-path producer policy.
    mode
        Capture mode to compile into the policy.

    Returns
    -------
    None
        Mutates ``trace`` in place.
    """

    trace._capture_producer_policy = get_capture_producer_policy(mode)
    if mode == "predicate":
        install_lazy_record_contexts(trace)


def _should_keep_alias_mutation_contract(trace: Trace) -> bool:
    """Return whether mutation-position alias contracts can be consumed.

    Parameters
    ----------
    trace
        Active trace object.

    Returns
    -------
    bool
        True when replay, intervention, backward, or validation paths may use
        mutation-position metadata.
    """

    save_grads = trace._options_save_grads
    return bool(
        trace._options_save_args
        or getattr(trace, "is_replayable", False)
        or trace._options_backward_ready
        or save_grads not in (None, False)
        or trace.__dict__.get("_validation_active", False)
    )


def _snapshot_exhaustive_module_stack(self: Trace) -> list[tuple[str, int]]:
    """Return the raw hook-stack module context.

    Parameters
    ----------
    self:
        Active trace.

    Returns
    -------
    list[tuple[str, int]]
        Raw ``(module_address, pass_index)`` stack snapshot.
    """

    return [
        (frame.address, frame.pass_number)
        for frame in _mstack.snapshot(self._module_capture_ws.exhaustive_module_stack)
    ]


def _parent_edges_from_fields(fields_dict: dict[str, Any]) -> tuple[ParentEdge, ...]:
    """Convert raw parent labels to IR parent edges.

    Parameters
    ----------
    fields_dict
        Raw field mapping used to construct the corresponding ``Op``.

    Returns
    -------
    tuple[ParentEdge, ...]
        Parent edges for the operation event.
    """

    positions_by_label: dict[str, tuple[Any, str]] = {}
    for location, label in fields_dict["parent_arg_positions"]["args"].items():
        positions_by_label.setdefault(label, (location, "arg"))
    for location, label in fields_dict["parent_arg_positions"]["kwargs"].items():
        positions_by_label.setdefault(label, (location, "kwarg"))

    return tuple(
        ParentEdge(
            parent_label_raw=label,
            arg_position=positions_by_label.get(label, (None, "arg"))[0],
            _edge_use=positions_by_label.get(label, (None, "arg"))[1],
        )
        for label in fields_dict["parents"]
    )


# Late imports: the split modules of this package import one another (cycles), so each
# imports its siblings once every definition above exists.
from ._ops_exhaustive import _emit_exhaustive_operation_events  # noqa: E402
from ._ops_predicate_events import (  # noqa: E402
    _emit_predicate_operation_events,
    install_lazy_record_contexts,
)
from .ops import (  # noqa: E402
    _ANCESTOR_FIELD_NAMES,
    _ANCESTOR_SLOT_DESCRIPTORS,
    _CAPTURE_PRODUCER_POLICIES,
    _LABEL_VERSION_SNAPSHOT,
    _VIEW_MUTATION_RELABELS,
    CaptureProducerPolicy,
    _AncestorBitset,
)

# Producer routing by capture mode; the emitters live in their home modules above.
_EMIT_BY_MODE: dict[CaptureProducerMode, Callable[..., Any]] = {
    "exhaustive": _emit_exhaustive_operation_events,
    "predicate": _emit_predicate_operation_events,
}
