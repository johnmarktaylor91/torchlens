"""Torch output decoding and marking for the FINALIZE stage.

Moved verbatim from ``torchlens.backends.torch.backend`` by the capture seam:
the module-level output helpers and the bodies of ``TorchBackend.extract_and_mark_outputs``
and ``TorchBackend.finalize_forward_session``, which delegate here.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, NoReturn, cast

import torch

from torchlens.backends.torch import _tl

from ..._errors import OutputAttributionError
from ..._io import BlobRef as PortableBlobRef
from ..._ir._output_occurrences import container_path_to_address, occurrences_by_raw_label
from ..._ir.container import ContainerSpec, OutputPathComponent
from ..._ir.container_registry import ContainerLeafOccurrence, ModelSite, Phase, Role
from ..._ir.events import OpEvent, OutputRef
from ..._ir.op_record import amend_output_parent_promotion
from ..._ir.refs import TensorRef
from ..._ir.semantics import CapturePolicy
from ..._ir.workspaces import RawGraphWorkspace
from ..._utils.introspection import (
    get_vars_of_type_from_obj,
    nested_assign,
)
from ...recording.types import CaptureSpec, StorageIntent
from ._ops_containers import _walk_output_tensors_with_paths, runnable_output_losslessness
from .buffer_writes import reconcile_buffer_writes

if TYPE_CHECKING:
    from torchlens.backends.torch.backend import TorchBackend

    from ..._data_classes.trace import Trace


def _tensor_memory_bytes(tensor: torch.Tensor) -> int:
    """Return the byte size of a tensor payload.

    Parameters
    ----------
    tensor:
        Tensor whose payload memory should be measured.

    Returns
    -------
    int
        Number of bytes occupied by the tensor storage view.
    """

    return int(tensor.nelement() * tensor.element_size())


def _write_output_parent_blob(
    trace: Trace,
    label_raw: str,
    payload: torch.Tensor | None,
    kind: str,
) -> PortableBlobRef | None:
    """Write a promoted output-parent payload to the active bundle writer.

    Parameters
    ----------
    trace:
        Trace that may own an output bundle writer.
    label_raw:
        Raw label for the output-parent operation.
    payload:
        Tensor payload to persist, if any.
    kind:
        Bundle payload kind, such as ``"out"`` or ``"transformed_out"``.

    Returns
    -------
    PortableBlobRef | None
        Blob reference for the persisted payload, or ``None`` when no payload
        was written.
    """

    writer = getattr(trace, "_out_writer", None)
    if writer is None or payload is None:
        return None
    blob_id = writer.next_blob_id()
    writer.submit_blob(blob_id, payload, kind=kind, label=label_raw)
    return PortableBlobRef(blob_id=blob_id, kind=kind)


def _output_parent_storage_intent(
    trace: Trace,
) -> tuple[StorageIntent, str | None, CaptureSpec]:
    """Build the storage intent, output device and capture spec for one promotion."""

    options = trace._predicate_save_options
    streaming = options.streaming
    intent = StorageIntent(
        in_ram=streaming is None or streaming.path is None or streaming.retain_in_memory,
        on_disk=streaming is not None and streaming.path is not None,
    )
    output_device = trace._options_output_device
    if output_device == "same":
        output_device = None
    spec = CaptureSpec(
        save_out=True,
        save_metadata=True,
        save_grad=False,
        device=output_device,
        save_mode=cast(Any, trace._options_save_mode),
    )
    return intent, output_device, spec


def _retain_output_parent_payloads(
    trace: Trace,
    event: OpEvent,
    tensor: torch.Tensor,
    storage: tuple[StorageIntent, str | None, CaptureSpec],
    ctx: Any,
) -> tuple[Any, Any, Any, PortableBlobRef | None, PortableBlobRef | None]:
    """Admit, resolve and persist the promoted output-parent payloads.

    ``storage`` is the ``(intent, output_device, spec)`` triple from
    ``_output_parent_storage_intent``.
    """

    intent, output_device, spec = storage

    from ...recording._storage_resolver import _resolve_storage

    # Output-parent promotion retains real payloads and must be visible to
    # the save-budget accountant like every other RAM retention: this path
    # kept ram/transformed payloads with no admit/commit, silently
    # undercounting on every selective capture whose outputs were not
    # selected (grind-r6 b5 R34-N2). Admit BEFORE the copy allocates;
    # disk-only routes stay exempt like the other predicate disk saves.
    budget = getattr(trace, "_save_budget_accountant", None)
    reservation = None
    if budget is not None and intent.in_ram:
        target_device = torch.device(output_device) if output_device is not None else tensor.device
        reservation = budget.admit(
            str(event.label_raw),
            target_device,
            int(tensor.nelement() * tensor.element_size()),
            site="primary",
        )
    ram_payload, disk_payload, transformed_ram_payload, transformed_disk_payload = _resolve_storage(
        tensor,
        spec,
        intent,
        activation_transform=trace._options_activation_transform,
        save_raw_activations=trace._options_save_raw_activations,
        ctx=ctx,
        kind="activation",
        detach_saved_activations=bool(trace._options_detach_saved_activations),
    )
    if budget is not None and reservation is not None:
        budget.commit(reservation, (ram_payload, transformed_ram_payload))
    raw_blob_ref = _write_output_parent_blob(trace, event.label_raw, disk_payload, "out")
    transformed_blob_ref = _write_output_parent_blob(
        trace,
        event.label_raw,
        transformed_disk_payload,
        "transformed_out",
    )
    return (
        ram_payload,
        transformed_ram_payload,
        transformed_disk_payload,
        raw_blob_ref,
        transformed_blob_ref,
    )


def _output_parent_output_ref(
    event: OpEvent,
    tensor: torch.Tensor,
    retained: tuple[Any, Any, Any, PortableBlobRef | None, PortableBlobRef | None],
) -> OutputRef:
    """Rebind the event output to the retained output-parent payload refs.

    ``retained`` is the ``(ram_payload, transformed_ram_payload,
    transformed_disk_payload, raw_blob_ref, transformed_blob_ref)`` tuple
    from ``_retain_output_parent_payloads``.
    """

    (
        ram_payload,
        transformed_ram_payload,
        transformed_disk_payload,
        raw_blob_ref,
        transformed_blob_ref,
    ) = retained

    tensor_ref = dataclasses.replace(
        event.output.tensor,
        shape=tuple(tensor.shape),
        dtype=str(tensor.dtype),
        device=str(tensor.device),
        requires_grad=tensor.requires_grad,
        memory=_tensor_memory_bytes(tensor),
        payload=ram_payload,
        blob_ref=cast(Any, raw_blob_ref),
        backend_handle_id=str(id(tensor)),
    )
    transformed_ref = event.output.transformed_tensor
    if transformed_ram_payload is not None:
        transformed_ref = TensorRef(
            label_raw=event.label_raw,
            shape=tuple(transformed_ram_payload.shape),
            dtype=str(transformed_ram_payload.dtype),
            device=str(transformed_ram_payload.device),
            requires_grad=transformed_ram_payload.requires_grad,
            memory=_tensor_memory_bytes(transformed_ram_payload),
            payload=transformed_ram_payload,
            blob_ref=cast(Any, transformed_blob_ref),
            backend_handle_id=str(id(transformed_ram_payload)),
        )
    elif transformed_disk_payload is not None and transformed_ref is not None:
        transformed_ref = dataclasses.replace(
            transformed_ref,
            blob_ref=cast(Any, transformed_blob_ref),
        )
    output_ref = dataclasses.replace(
        event.output,
        tensor=tensor_ref,
        transformed_tensor=transformed_ref,
        has_saved_activation=True,
    )
    return output_ref


def _promote_layers_to_save_output_parent(
    trace: Trace,
    event: OpEvent,
    tensor: torch.Tensor,
) -> tuple[OutputRef, CapturePolicy, bool, object]:
    """Resolve the ``output_parent_promotion`` amendment values for one output.

    Parameters
    ----------
    trace:
        Predicate-mode trace whose selective ``layers_to_save`` request must
        preserve the output-parent retention rule.
    event:
        Existing operation event for the tensor returned by the model.
    tensor:
        Live model-output tensor corresponding to ``event``.

    Returns
    -------
    tuple[OutputRef, CapturePolicy, bool, object]
        ``(output, policy, predicate_matched, capture_spec)`` for the caller's
        ``amend_output_parent_promotion`` — the event's current values when no
        payload retention is required, otherwise the saved-payload rebinds.
        The computed record context is deliberately NOT returned: the family
        schema preserves the documented legacy quirk of dropping it.
    """

    save_request = getattr(trace, "_rerun_save_request", None) or {}
    if (
        not save_request.get("retain_output_parents_for_layers_to_save", False)
        or getattr(trace, "_predicate_save_options", None) is None
        or event.output.has_saved_activation
    ):
        return event.output, event.policy, event.predicate_matched, event.capture_spec

    from ..._capture.projections import _record_context_from_event

    ctx = dataclasses.replace(_record_context_from_event(event), is_output_parent=True)
    storage = _output_parent_storage_intent(trace)
    retained = _retain_output_parent_payloads(trace, event, tensor, storage, ctx)
    output_ref = _output_parent_output_ref(event, tensor, retained)
    spec = storage[2]
    policy = dataclasses.replace(event.policy, save_payload=True)
    return output_ref, policy, True, spec


def _dedupe_output_addresses(
    entries: list[tuple[torch.Tensor, str, Any]],
) -> tuple[list[torch.Tensor], list[str]]:
    """Drop entries repeating an already-seen structural output address.

    Parameters
    ----------
    entries:
        ``(tensor, address, container_spec)`` output entries in walk order.

    Returns
    -------
    tuple[list[torch.Tensor], list[str]]
        First-occurrence output tensors and their display addresses.
    """

    addresses_seen: set[str] = set()
    deduped = []
    for entry in entries:
        if entry[1] in addresses_seen:
            continue
        deduped.append(entry)
        addresses_seen.add(entry[1])
    return [t for t, _, _ in deduped], [addr for _, addr, _ in deduped]


def _register_model_output_container_snapshot(
    trace: Trace,
    output: object,
    output_entries: list[
        tuple[torch.Tensor, tuple[OutputPathComponent, ...], ContainerSpec | None]
    ],
) -> None:
    """Register the final model-output container snapshot when present.

    Parameters
    ----------
    trace:
        Active trace.
    output:
        Raw model output object.
    output_entries:
        Path-aware tensor entries from the existing output walker.
    """

    spec = next((container_spec for _, _, container_spec in output_entries if container_spec), None)
    if spec is None:
        return
    # An opaque model-output container (custom Mapping, unsafe defaultdict, unknown
    # dict subclass, or an unrepresentable non-tensor leaf) is recorded but marked
    # NON-reconstructable so producer preflight refuses to advertise it runnable and
    # a live run reports UNVERIFIABLE -- never a silent bare-tensor/plain-dict.
    reconstructable = spec.kind != "opaque"
    occurrences: list[ContainerLeafOccurrence] = []
    for occ_index, (tensor, path, _container_spec) in enumerate(output_entries):
        producer_label = _tl.get_tensor_label(tensor)
        occurrences.append(
            ContainerLeafOccurrence(
                path=path,
                producer_op_label=producer_label,
                tensor_identity=producer_label,
                occ_index=occ_index,
            )
        )
    registry = trace._wrapper_runtime_ws.container_registry
    registry.register_snapshot(
        output,
        site=ModelSite(model_ref="self:1", position="return"),
        role=Role.MODEL_OUTPUT,
        phase=Phase.POST_CALL,
        observed_at_event_index=trace._raw_graph_ws.layer_counter,
        spec=spec,
        leaf_occurrences=tuple(occurrences),
        reconstructable=reconstructable,
    )
    registry.register_snapshot(
        output,
        site=ModelSite(model_ref="self:1", position="return"),
        role=Role.CALL_OUTPUT,
        phase=Phase.POST_CALL,
        observed_at_event_index=trace._raw_graph_ws.layer_counter,
        spec=spec,
        leaf_occurrences=tuple(occurrences),
        reconstructable=reconstructable,
    )


_OUTPUT_ATTRIBUTION_REMEDY = (
    "use ordinary torch module attributes during forward, or "
    "bind/import torch functions after TorchLens has wrapped torch"
)


def output_attribution_remedy(trace: Any) -> str:
    """Remedy for ``output_attribution_failed``, aimed at export products.

    An fx ``GraphModule`` root (``torch.export.export(...).module()`` or
    ``torch.fx.symbolic_trace``) runs generated code whose call targets were
    bound when the graph was built, so the torch-function-binding advice does
    not apply; the fix is to trace the eager source module.

    Parameters
    ----------
    trace:
        The capture whose output could not be attributed.

    Returns
    -------
    str
        The remedy sentence.
    """

    model_ref = trace._source_model_ref  # noqa: SLF001 -- declared Trace field (live capture)
    model = model_ref() if model_ref is not None else None
    if isinstance(model, torch.fx.GraphModule):
        return (
            "trace the source nn.Module before export: this root is a torch.fx "
            "GraphModule (torch.export ... .module() or torch.fx.symbolic_trace) whose "
            "generated forward calls targets bound at graph build time; call "
            "tl.trace(source_model, x) on the eager module instead"
        )
    return _OUTPUT_ATTRIBUTION_REMEDY


def _is_direct_registered_buffer_output(trace: Trace, tensor: torch.Tensor) -> bool:
    """Return whether an unlabeled output is a registered source-model buffer.

    Parameters
    ----------
    trace:
        Active trace that may hold a weak reference to the source model.
    tensor:
        Unlabeled output tensor returned by ``forward``.

    Returns
    -------
    bool
        True when ``tensor`` is exactly one of the source model's registered
        buffers. Such tensors are late-logged during postprocess; other
        unlabeled outputs fail loud at the output boundary.
    """

    model_ref = getattr(trace, "_source_model_ref", None)
    model = model_ref() if model_ref is not None else None
    if model is None:
        return False
    return any(tensor is buffer for _address, buffer in model.named_buffers())


def _model_input_output_label(trace: Trace, tensor: torch.Tensor) -> str | None:
    """Return the input-source label when an unlabeled output is a model input.

    Parameters
    ----------
    trace:
        Active trace carrying the tensors that were explicitly marked as model
        inputs for this capture.
    tensor:
        Unlabeled model output tensor.

    Returns
    -------
    str | None
        Raw input label for ``tensor`` when structural tensor identity/storage
        proves it is one of the marked model inputs; otherwise ``None``.
    """

    input_tensors = getattr(trace, "_output_attribution_input_tensors", ())
    input_labels = tuple(getattr(trace, "input_layers", ()))
    for index, input_tensor in enumerate(input_tensors):
        if not isinstance(input_tensor, torch.Tensor):
            continue
        if not _same_tensor_storage_identity(tensor, input_tensor):
            continue
        live_label = _tl.get_tensor_label(input_tensor)
        if live_label is not None:
            return live_label
        if index < len(input_labels):
            return str(input_labels[index])
    return None


def _same_tensor_storage_identity(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Return whether two tensors are the same object or identical storage view.

    Parameters
    ----------
    left:
        Candidate output tensor.
    right:
        Marked input tensor.

    Returns
    -------
    bool
        True when the tensors are the same Python object, or when their storage
        pointer, offset, shape, stride, dtype, and device all match.
    """

    if left is right:
        return True
    if left.dtype != right.dtype or left.device != right.device:
        return False
    try:
        if tuple(left.shape) != tuple(right.shape) or tuple(left.stride()) != tuple(right.stride()):
            return False
        if left.storage_offset() != right.storage_offset():
            return False
        return left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()
    except RuntimeError:
        # Nested tensors raise from shape/stride/storage reads; a tensor whose
        # layout metadata is unreadable cannot be structurally proven to be a
        # marked input, so identity attribution conservatively says no (R65).
        return False


def _assign_nested_input_value(
    obj: Any,
    addr: list[tuple[Any, Any]],
    value: Any,
) -> Any:
    """Assign ``value`` inside nested input containers, rebuilding tuples.

    Parameters
    ----------
    obj:
        Root object to update.
    addr:
        Address path from ``get_vars_of_type_from_obj``.
    value:
        Replacement tensor.

    Returns
    -------
    Any
        Updated root object.
    """

    if not addr:
        return value
    entry_type, entry_val = addr[0]
    rest = addr[1:]
    if entry_type == "ind":
        if isinstance(obj, tuple):
            items = list(obj)
            items[entry_val] = _assign_nested_input_value(items[entry_val], rest, value)
            obj_type = type(obj)
            if hasattr(obj_type, "_fields"):
                return obj_type(*items)
            return obj_type(items)
        if isinstance(obj, list):
            obj[entry_val] = _assign_nested_input_value(obj[entry_val], rest, value)
            return obj
        if isinstance(obj, dict):
            obj[entry_val] = _assign_nested_input_value(obj[entry_val], rest, value)
            return obj
    if entry_type == "attr":
        child = getattr(obj, entry_val)
        new_child = _assign_nested_input_value(child, rest, value)
        try:
            setattr(obj, entry_val, new_child)
        except AttributeError:
            # Immutable attribute — e.g. a real (non-property) NamedTuple
            # field on a NamedTuple subclass such as a GNN batch container.
            # ``_replace`` returns a new instance with that field swapped in.
            obj_fields = getattr(type(obj), "_fields", None)
            if hasattr(obj, "_replace") and obj_fields is not None and entry_val in obj_fields:
                return obj._replace(**{entry_val: new_child})
            # Otherwise this is a read-only *derived* property (e.g. a
            # convenience accessor that returns a value already stored in a
            # mutable nested container). The tensor's real backing storage is
            # reached and moved to device through its own container address;
            # there is nothing else to update at this alias.
        return obj
    nested_assign(obj, addr, value)
    return obj


def _collect_output_addresses(
    self_trace: Trace,
    outputs: object,
    output_entries: list[Any],
) -> list[Any]:
    """Record output container metadata and return every output tensor with its address."""

    # The per-op container_spec is only user-facing metadata when explicitly
    # opted into via capture_container_structure (or implied by
    # replayable); with the default OFF it must stay None on output
    # layers. The container *path*, however, is always preserved so
    # forward-replay validation can slice multi-output containers back to the
    # right leaf, and the FINAL-output ContainerSpec is always registered as
    # the model-output snapshot (below): the live ``run()`` provider rebuilds
    # the exact tuple/dict/ModelOutput the model returned from that one
    # snapshot, so a default capture settles VERIFIED instead of the
    # remedy-only ``container_contract_unrecorded`` (W051-HONESTY H2).
    persist_container_spec = getattr(self_trace, "is_replayable", False) or getattr(
        self_trace, "_capture_container_structure", False
    )
    if output_entries:
        output_tensors_w_addresses_all = [
            (tensor, container_path_to_address(path), None)
            for tensor, path, _container_spec in output_entries
        ]
        # One output op per occurrence; see ``_ir/_output_occurrences.py``.
        output_specs_by_raw_label = occurrences_by_raw_label(
            (_tl.get_tensor_label(tensor), path, spec if persist_container_spec else None)
            for tensor, path, spec in output_entries
        )
        setattr(self_trace, "_output_container_specs_by_raw_label", output_specs_by_raw_label)
    else:
        output_tensors_w_addresses_all = []
    if output_entries:
        # Every capture (one spec per capture): the model-output snapshot is
        # the contract the live provider reconstructs from.
        _register_model_output_container_snapshot(self_trace, outputs, output_entries)
    # (container_path is stored above for validation replay even when the spec is None)
    if not output_entries:
        output_tensors_w_addresses_all = get_vars_of_type_from_obj(
            outputs,
            torch.Tensor,
            search_depth=5,
            return_addresses=True,
            allow_repeats=True,
        )
    return output_tensors_w_addresses_all


def _raise_output_attribution_error(
    self_trace: Trace,
    t: torch.Tensor,
    output_address: str,
) -> NoReturn:
    """Raise the typed refusal for a model output tensor no traced op produced."""

    try:
        shape_text = str(tuple(t.shape))
    except RuntimeError:
        # Nested tensors raise from ``.shape``; the refusal must
        # stay typed instead of crashing on its own message (R65).
        shape_text = "<unavailable: nested>" if t.is_nested else "<unavailable>"
    if getattr(t, "is_nested", False):
        # An unlabeled NESTED output is an unsupported tensor
        # variant constructed inside forward (protocol-invisible
        # constructors like torch.nested.nested_tensor are never
        # logged), NOT a pre-bound-function escape -- the escape
        # remedy can never fix it (R65: typed-misdiagnosis
        # successor of the round-4 raw crash).
        raise OutputAttributionError(
            "TorchLens could not attribute a model output tensor to any "
            f"traced op (output address {output_address!r}, "
            f"shape={shape_text}, dtype={t.dtype}): the output is a NESTED "
            "tensor constructed inside forward(), an unsupported tensor "
            "variant TorchLens cannot log. Remedy: build the nested tensor "
            "outside the traced region, or pad to a dense tensor before "
            "the ops you want captured.",
            code="output_unsupported_tensor_variant",
            remedy=(
                "build the nested tensor outside the traced region, or "
                "pad to a dense tensor before the ops you want captured"
            ),
            output_address=output_address,
        )
    remedy = output_attribution_remedy(self_trace)
    raise OutputAttributionError(
        "TorchLens could not attribute a model output tensor to any traced op "
        f"(output address {output_address!r}, "
        f"shape={shape_text}, dtype={t.dtype}). This may indicate an opaque "
        "execution boundary or a pre-bound torch function that escaped wrapping. "
        f"Remedy: {remedy}.",
        code="output_attribution_failed",
        remedy=remedy,
        output_address=output_address,
    )


def _output_parent_promotion_amendment(
    self_trace: Trace, label_raw: str, t: torch.Tensor
) -> Any | None:
    """Register one output layer and build its output-parent promotion amendment.

    Returns ``None`` when the trace holds no op event for ``label_raw``. The caller
    appends the amendment, so the journal's append site stays in the caller's frame.
    """

    self_trace.output_layers.append(label_raw)
    event = self_trace.capture_events.op_event_by_label_raw.get(label_raw)
    if event is None:
        return None
    promoted_output, promoted_policy, promoted_matched, promoted_spec = (
        _promote_layers_to_save_output_parent(
            self_trace,
            event,
            t,
        )
    )
    return amend_output_parent_promotion(
        event.seq,
        label_raw,
        is_output_parent=True,
        output=promoted_output,
        policy=promoted_policy,
        predicate_matched=promoted_matched,
        capture_spec=promoted_spec,
    )


def extract_and_mark_outputs(
    self: TorchBackend,
    session: object,
    outputs: object,
) -> tuple[list[torch.Tensor], list[str]]:
    """Extract torch output tensors, deduplicate them, and mark output parents.

    Parameters
    ----------
    session:
        Active trace.
    outputs:
        Raw model output object returned by the captured forward pass.

    Returns
    -------
    tuple[list[torch.Tensor], list[str]]
        Output tensors and their display addresses.
    """

    self_trace = cast("Trace", session)
    output_entries = list(_walk_output_tensors_with_paths(outputs))
    # r35 I1 (subsumes r33 R32-B1): stamp the POSITIVE model-output losslessness
    # proof -- exact root kind, recursively supported children, encodable literal
    # leaves, and a tensor-leaf/typed-path bijection (duplicate paths and any BFS
    # fallback break the proof). The runnable producer refuses any save whose
    # output is not PROVED lossless (refuse-unless-proved), closing every lossy
    # cardinality/depth: bare one-tensor sets, nested sets, opaque tensor
    # holders, set subclasses, and multi-tensor collapses alike. Ordinary
    # analysis capture is unaffected by the stamp.
    self_trace._runnable.output_losslessness = runnable_output_losslessness(outputs, output_entries)
    output_tensors_w_addresses_all = _collect_output_addresses(self_trace, outputs, output_entries)
    output_tensors, output_tensor_addresses = _dedupe_output_addresses(
        output_tensors_w_addresses_all
    )

    attributable_output_tensors: list[torch.Tensor] = []
    attributable_output_tensor_addresses: list[str] = []
    for t, output_address in zip(output_tensors, output_tensor_addresses):
        _label_raw = _tl.get_tensor_label(t)
        if _label_raw is None:
            if _is_direct_registered_buffer_output(self_trace, t):
                # Late-logged in postprocess so untouched buffer outputs get
                # real source nodes without leaking labels onto model state.
                attributable_output_tensors.append(t)
                attributable_output_tensor_addresses.append(output_address)
                continue
            _label_raw = _model_input_output_label(self_trace, t)
        if _label_raw is None:
            if getattr(self_trace, "_raw_transform_escape_detected", False):
                continue
            _raise_output_attribution_error(self_trace, t, output_address)
        attributable_output_tensors.append(t)
        attributable_output_tensor_addresses.append(output_address)
        if self_trace._options_mode in {"exhaustive", "predicate"}:
            amendment = _output_parent_promotion_amendment(self_trace, _label_raw, t)
            if amendment is not None:
                self_trace.capture_events.append_amendment(amendment)

    return attributable_output_tensors, attributable_output_tensor_addresses


def finalize_forward_session(
    self: TorchBackend,
    session: object,
    trace_state: RawGraphWorkspace,
) -> None:
    """Run torch post-forward reconciliation before output extraction."""
    reconcile_buffer_writes(cast("Trace", session), trace_state)
