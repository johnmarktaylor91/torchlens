"""Ancestor storage, producer policy, and exhaustive record freezing."""

import contextlib
import dataclasses
from typing import TYPE_CHECKING, Any, cast

import torch

from ..._io import BlobRef
from ...data_classes.op import (
    Op,
)
from ...ir.events import (
    FunctionCallRef,
    ModuleFrame,
    OutputRef,
    ParentEdge,
)
from ...ir.refs import ParamRef, TensorRef
from ...ir.semantics import BackendSemantics, CapturePolicy
from ...utils._torch_compat import (
    tensor_version_or_none,
)
from . import module_stack as _mstack
from ._tl import (
    active_label_session_token,
    get_param_meta,
)
from .completeness_witness import internal_scalar_read

if TYPE_CHECKING:
    from ...data_classes.trace import Trace

from .ops import (
    CaptureProducerMode,
    CaptureProducerPolicy,
)

if TYPE_CHECKING:
    from .ops import (
        _ANCESTOR_FIELD_NAMES,
        _ANCESTOR_SLOT_DESCRIPTORS,
        _CAPTURE_PRODUCER_POLICIES,
        _EMIT_BY_MODE,
        _FUNCTION_REF_PER_OUTPUT_FIELDS,
        _INPLACE_AUGMENTED_ASSIGNMENT_DUNDER_EXCLUSIONS,
        _LABEL_VERSION_SNAPSHOT,
        _AncestorBitset,
    )

__all__ = (
    "_get_ancestor_field",
    "_set_ancestor_field",
    "_delete_ancestor_field",
    "_compact_ancestor_sets",
    "_is_inplace_augmented_assignment_dunder",
    "_record_label_version_snapshot",
    "_label_version_baseline",
    "_stamp_same_object_mutation",
    "get_capture_producer_policy",
    "set_capture_producer_policy",
    "_should_keep_alias_mutation_contract",
    "_snapshot_exhaustive_module_stack",
    "_tensor_ref_from_fields",
    "_module_frames_from_fields",
    "_param_refs_from_fields",
    "_parent_edges_from_fields",
    "_function_call_ref_from_fields",
    "_resolve_call_function_ref",
    "_exhaustive_freeze_refs",
    "_exhaustive_output_ref",
    "_exhaustive_capture_policy",
)


def _get_ancestor_field(op: Op, field_name: str) -> "set[str] | frozenset[str]":
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


def _compact_ancestor_sets(trace: "Trace") -> None:
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

    ops = getattr(trace, "layer_list", None)
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


def _is_inplace_augmented_assignment_dunder(name: str) -> bool:
    """Return whether ``name`` denotes an in-place augmented-assignment dunder.

    Parameters
    ----------
    name:
        Function name recorded by the torch eager wrapper.

    Returns
    -------
    bool
        True for ``__i*`` mutation dunders except known non-mutating dunders.
    """

    return name.startswith("__i") and name not in _INPLACE_AUGMENTED_ASSIGNMENT_DUNDER_EXCLUSIONS


def _record_label_version_snapshot(t: Any) -> None:
    """Record ``t``'s current version, keyed to the active label session.

    Parameters
    ----------
    t
        Tensor that was just labeled (as an op output, or by live in-place /
        alias label propagation).
    """

    if not isinstance(t, torch.Tensor):
        return
    # ``_version`` is a WITNESSED input-metadata property (r33): the scoped patch
    # records a genuine USER ``x._version`` read on a model-input leaf. This snapshot is
    # TorchLens's OWN bookkeeping read, so it must run under the internal marker or it
    # spuriously records a ``_version`` fact for every model, falsely diverging any
    # runtime input whose version counter differs from capture (an over-trigger).
    with internal_scalar_read():
        version = tensor_version_or_none(t)
    if version is not None:
        _LABEL_VERSION_SNAPSHOT[t] = (active_label_session_token(), version)


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


def _stamp_same_object_mutation(logged: Any, receiver: Any, mutation_signature: bool) -> None:
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
        logged.tl_same_object_mutation = verdict


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
        emit = globals()[_EMIT_BY_MODE[mode]]
        policy = CaptureProducerPolicy(mode, emit)
        _CAPTURE_PRODUCER_POLICIES[mode] = policy
    return policy


def set_capture_producer_policy(trace: "Trace", mode: CaptureProducerMode) -> None:
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


def _should_keep_alias_mutation_contract(trace: "Trace") -> bool:
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

    save_grads = getattr(trace, "save_grads", None)
    return bool(
        getattr(trace, "save_arg_values", False)
        or getattr(trace, "intervention_ready", False)
        or getattr(trace, "backward_ready", False)
        or save_grads not in (None, False)
        or getattr(trace, "_validation_active", False)
    )


def _snapshot_exhaustive_module_stack(self: "Trace") -> list[tuple[str, int]]:
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
        (frame.address, frame.pass_index)
        for frame in _mstack.snapshot(self._module_capture_ws.exhaustive_module_stack)
    ]


def _tensor_ref_from_fields(tensor: torch.Tensor, fields_dict: dict[str, Any]) -> TensorRef:
    """Build an IR tensor reference from one raw fields dictionary.

    Parameters
    ----------
    tensor
        Tensor represented by the operation event.
    fields_dict
        Raw field mapping used to construct the corresponding ``Op``.

    Returns
    -------
    TensorRef
        Backend-neutral tensor metadata and optional payload reference.
    """

    # r65: TorchLens's OWN IR-bookkeeping ``requires_grad`` read runs under the explicit
    # internal-read marker so the r65 state-metadata property observer never mistakes it
    # for a user autograd read on a registered buffer/param receiver.
    with internal_scalar_read():
        _requires_grad = bool(tensor.requires_grad)
    return TensorRef(
        label_raw=fields_dict["_label_raw"],
        shape=fields_dict["shape"],
        dtype=str(fields_dict["dtype"]),
        device=str(tensor.device),
        requires_grad=_requires_grad,
        memory=fields_dict["activation_memory"],
        payload=fields_dict["out"],
        blob_ref=(
            BlobRef(blob_id=fields_dict["_pending_blob_id"], kind="out")  # type: ignore[arg-type]
            if fields_dict.get("_pending_blob_id") is not None
            else None
        ),
        backend_handle_id=str(id(tensor)),
    )


def _module_frames_from_fields(fields_dict: dict[str, Any]) -> tuple[ModuleFrame, ...]:
    """Convert raw module stack tuples to IR module frames.

    Parameters
    ----------
    fields_dict
        Raw field mapping used to construct the corresponding ``Op``.

    Returns
    -------
    tuple[ModuleFrame, ...]
        Module stack snapshot for the operation event.
    """

    return tuple(
        ModuleFrame(
            address=address,
            address_normalized=None,
            module_type="",
            call_index=pass_index,
            fx_qualpath=None,
            entry_argnames=(),
        )
        for address, pass_index in fields_dict["modules"]
    )


def _param_refs_from_fields(fields_dict: dict[str, Any]) -> tuple[ParamRef, ...]:
    """Convert raw parameter metadata to IR parameter references.

    Parameters
    ----------
    fields_dict
        Raw field mapping used to construct the corresponding ``Op``.

    Returns
    -------
    tuple[ParamRef, ...]
        Parameter references for the operation event.
    """

    refs: list[ParamRef] = []
    # One ref per RAW parent-param occurrence, each carrying ITS OWN barcode
    # from the param's meta. ``_param_barcodes`` is the DEDUPED key list of
    # ``parent_param_ops`` (a dict), so zipping it against the raw
    # ``parent_params`` misaligned every pairing after a repeated param
    # (weight-tied einsum/hypernetwork class): ``[A, B, A, C]`` persisted
    # ``ParamRef(barcode=C, geometry-of-A)`` and dropped a ref while
    # ``num_params`` said four (R23-3).
    for param in fields_dict["parent_params"]:
        param_meta = get_param_meta(param)
        if param_meta is None or param_meta.param_barcode is None:
            raise RuntimeError(
                "TorchLens internal error: a parent parameter reached ParamRef "
                "construction without an assigned barcode; param metadata was "
                "not processed for this op event."
            )
        barcode = param_meta.param_barcode
        param_address = (
            ""
            if param_meta is None or param_meta.param_address is None
            else param_meta.param_address
        )
        # r65: TorchLens's OWN per-op parent-param ``requires_grad`` read runs under the
        # explicit internal-read marker so the r65 state-metadata property observer never
        # mistakes it for a user autograd read on the registered parameter (it fires for
        # EVERY op with parent params -- unmarked it would spuriously fact-record every
        # consumed slot of every model).
        with internal_scalar_read():
            trainable = bool(param.requires_grad)
        refs.append(
            ParamRef(
                barcode=barcode,
                address=param_address,
                shape=tuple(param.shape),
                dtype=str(param.dtype),
                trainable=trainable,
                module_address=None,
            )
        )
    return tuple(refs)


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
            edge_use=positions_by_label.get(label, (None, "arg"))[1],
        )
        for label in fields_dict["parents"]
    )


def _function_call_ref_from_fields(fields_dict: dict[str, Any]) -> FunctionCallRef:
    """Build the frozen function-call summary for one operation event."""

    return FunctionCallRef(
        func=fields_dict["func"],
        func_name=fields_dict["func_name"],
        func_qualname=fields_dict["func_qualname"],
        func_call_id=fields_dict["func_call_id"],
        code_context=tuple(fields_dict["code_context"]),
        func_duration=fields_dict["func_duration"],
        flops_forward=fields_dict["flops_forward"],
        flops_backward=fields_dict["flops_backward"],
        func_rng_states=fields_dict["func_rng_states"],
        func_autocast_state=fields_dict["func_autocast_state"],
        arg_names=tuple(fields_dict["arg_names"]),
        num_args_total=fields_dict["num_args_total"],
        num_pos_args=fields_dict["num_pos_args"],
        num_kwargs=fields_dict["num_kwargs"],
        non_tensor_pos_args=tuple(fields_dict["non_tensor_pos_args"]),
        non_tensor_kwargs=tuple(fields_dict["non_tensor_kwargs"].items()),
        func_non_tensor_args=tuple(fields_dict["func_non_tensor_args"]),
        is_inplace=fields_dict["is_inplace"],
        func_config=tuple(fields_dict["func_config"].items()),
        func_id=fields_dict.get("func_id"),
    )


def _resolve_call_function_ref(
    fields_dict: dict[str, Any],
    call_ref_box: list[FunctionCallRef] | None,
) -> FunctionCallRef:
    """Return the (shared) function-call ref for one output event.

    The first output of a wrapped call builds the ref and parks it in
    ``call_ref_box``; sibling outputs reuse it verbatim when their per-output
    facts match, otherwise derive via ``dataclasses.replace`` (container
    fields stay shared by reference either way).
    """

    if call_ref_box:
        shared = call_ref_box[0]
        overrides = {
            name: fields_dict[name]
            for name in _FUNCTION_REF_PER_OUTPUT_FIELDS
            if getattr(shared, name) != fields_dict[name]
        }
        return dataclasses.replace(shared, **overrides) if overrides else shared
    ref = _function_call_ref_from_fields(fields_dict)
    if call_ref_box is not None:
        call_ref_box.append(ref)
    return ref


def _exhaustive_freeze_refs(
    fields_dict: dict[str, Any],
    tensor: torch.Tensor,
    module_stack: tuple[ModuleFrame, ...] | None,
) -> tuple[Any, tuple[ModuleFrame, ...], Any, BackendSemantics]:
    """Compute the exhaustive freeze's shared nested refs ONCE per record.

    Both freeze shapes (legacy ``OpEvent`` and decomposed ``OpRecord``) build
    from these exact values, so the shapes cannot drift on the ref layer.
    """

    tensor_ref = _tensor_ref_from_fields(tensor, fields_dict)
    if module_stack is None:
        module_stack = _module_frames_from_fields(fields_dict)
    transformed_ref = (
        None
        if fields_dict["transformed_out"] is None
        else TensorRef(
            label_raw=fields_dict["_label_raw"],
            shape=fields_dict["transformed_out_shape"],
            dtype=str(fields_dict["transformed_out_dtype"]),
            device=fields_dict["output_device"],
            requires_grad=None,
            memory=fields_dict["transformed_activation_memory"],
            payload=fields_dict["transformed_out"],
            blob_ref=(
                BlobRef(
                    blob_id=fields_dict["_pending_transformed_out_blob_id"],
                    kind="transformed_out",
                )  # type: ignore[arg-type]
                if fields_dict.get("_pending_transformed_out_blob_id") is not None
                else None
            ),
            backend_handle_id=None,
        )
    )
    backend_semantics = fields_dict.get("backend_semantics")
    if backend_semantics is None:
        backend_semantics = BackendSemantics(
            backend_grad_handle=fields_dict["grad_fn_handle"],
            grad_fn_class_name=fields_dict["grad_fn_class_name"],
            autograd_memory=fields_dict["autograd_memory"],
            num_autograd_tensors=fields_dict["num_autograd_tensors"],
            mutated_input_positions=(),
            aliased_output_inputs=(),
            unknown_aliasing=False,
            bytes_delta_at_call=fields_dict["bytes_delta_at_call"],
            bytes_peak_at_call=fields_dict["bytes_peak_at_call"],
        )
    return tensor_ref, module_stack, transformed_ref, backend_semantics


def _exhaustive_output_ref(
    fields_dict: dict[str, Any], tensor_ref: Any, transformed_ref: Any
) -> OutputRef:
    """Return the exhaustive freeze's output ref (shared by both shapes)."""

    return OutputRef(
        tensor=tensor_ref,
        transformed_tensor=transformed_ref,
        has_saved_activation=fields_dict["has_saved_activation"],
        output_device=fields_dict["output_device"],
        activation_transform=fields_dict["activation_transform"],
        detach_saved_activations=fields_dict["detach_saved_activations"],
        visualizer_path=fields_dict["visualizer_path"],
        multi_output_index=fields_dict["multi_output_index"],
        in_multi_output=fields_dict["in_multi_output"],
        container_path=tuple(fields_dict["container_path"]),
        container_spec=fields_dict["container_spec"],
        child_versions=tuple(fields_dict["out_versions_by_child"].items()),
    )


def _exhaustive_capture_policy(trace: "Trace", fields_dict: dict[str, Any]) -> CapturePolicy:
    """Return the exhaustive freeze's capture policy (shared by both shapes)."""

    return CapturePolicy(
        save_payload=fields_dict["has_saved_activation"],
        save_grad=fields_dict["save_grads"],
        save_mode=getattr(trace, "save_mode", "copy"),
    )
