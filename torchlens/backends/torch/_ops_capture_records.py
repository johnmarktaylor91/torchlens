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
    "_bind_writeback_receiver_label",
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
        :func:`_bind_writeback_receiver_label`.
    """

    if not isinstance(t, torch.Tensor):
        return
    if mutated_via is not None:
        view, prior_label = mutated_via
        _VIEW_MUTATION_RELABELS[t] = (
            active_label_session_token(),
            prior_label,
            get_tensor_label(t),
            id(view),
        )
    # ``_version`` is a WITNESSED input-metadata property (r33): the scoped patch
    # records a genuine USER ``x._version`` read on a model-input leaf. This snapshot is
    # TorchLens's OWN bookkeeping read, so it must run under the internal marker or it
    # spuriously records a ``_version`` fact for every model, falsely diverging any
    # runtime input whose version counter differs from capture (an over-trigger).
    with internal_scalar_read():
        version = tensor_version_or_none(t)
    if version is not None:
        _LABEL_VERSION_SNAPSHOT[t] = (active_label_session_token(), version)


def _bind_writeback_receiver_label(func_name: str, args: tuple[Any, ...]) -> None:
    """Bind an augmented-assignment write-back's receiver to its pre-mutation producer.

    ``x[idx] += v`` runs as ``t = x[idx]; t += v; x[idx] = t``. The in-place op wrote
    through the view ``t``, so storage-alias propagation advanced ``x``'s label to that
    op, and the write-back ``__setitem__`` then recorded ONE parent at slot 0 (the whole
    tensor) and slot 2 (the slice): one label for two different values, which a
    per-(parent, child) version snapshot cannot carry, so argument logging failed on
    every such call. When slot 2 IS the view the in-place op wrote through, slot 0
    binds to the pre-mutation label for this call, so each slot has its own label:
    the value slot the mutation op, the receiver slot the base's own lineage. For a
    same-region write-back the binding is exact on its own (the call overwrites
    precisely the region the mutation wrote); when the view is written to ANOTHER
    region of its base (``v = m[:, 0:2]; v.mul_(3); m[:, 2:4] = v``) the receiver
    slot consumes a later version of that label and its actual content reaches
    replay through the per-child version snapshot. The setitem's own label then
    advances ``x`` like any in-place op. Advanced indexing or a later relabel of
    the receiver is left as is.

    Parameters
    ----------
    func_name
        Name of the call being logged.
    args
        The call's live positional arguments.
    """

    if func_name != "__setitem__" or len(args) < 3:
        return
    receiver, value = args[0], args[2]
    if (
        not isinstance(receiver, torch.Tensor)
        or not isinstance(value, torch.Tensor)
        or isinstance(receiver, torch.nn.Parameter)
        or receiver.device.type == "meta"
    ):
        return
    entry = _VIEW_MUTATION_RELABELS.get(receiver)
    if entry is None:
        return
    session_token, prior_label, mutation_label, view_id = entry
    if (
        prior_label is None
        or session_token is None
        or session_token != active_label_session_token()
        or view_id != id(value)
        or get_tensor_label(receiver) != mutation_label
        or get_tensor_label(value) != mutation_label
    ):
        return
    set_tensor_label(receiver, prior_label)


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
