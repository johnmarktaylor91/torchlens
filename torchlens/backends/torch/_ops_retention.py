"""Save budgets, predicate payloads, and lookback retention."""

import dataclasses
import time
import warnings
from collections import deque
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

import torch

from ...capture.predicates import (
    build_op_record_context,
)
from ...data_classes.op import (
    _dtype_or_none,
    _memory_or_none,
    _shape_or_none,
    apply_transform,
    train_mode_tripwire_armed,
    validate_train_mode_transform_output,
)
from ...fastlog._storage_resolver import _resolve_storage
from ...fastlog.exceptions import PredicateError
from ...fastlog.types import (
    CaptureSpec,
    ModuleStackFrame,
    RecordContext,
    StorageIntent,
)
from ...intervention.selectors import (
    BaseSelector,
)
from ...ir.op_record import amend_lookback_retention
from ...ir.predicate import RetroactiveCaptureDecision
from ...ir.refs import TensorRef
from ...utils.tensor_utils import (
    get_memory_amount_from_metadata,
    safe_copy,
    safe_to,
)
from ._tl import (
    mark_detached_saved_activation,
)

if TYPE_CHECKING:
    from ...data_classes.trace import Trace

from .ops import (
    _RetainedLookbackPayload,
)

if TYPE_CHECKING:
    from .ops import (
        _RetainedLookbackCandidate,
        _retention_device,
    )

__all__ = (
    "_admit_save_budget",
    "_commit_save_budget",
    "_save_predicate_activation_fields",
    "_stream_predicate_payloads",
    "_module_stack_frames_from_fields",
    "_append_trace_predicate_context",
    "_trace_followed_by_candidate_selector",
    "_retain_lookback_candidate",
    "_copy_lookback_payload",
    "_apply_retroactive_decision",
    "_replace_event_with_retained_payload",
    "_build_trace_predicate_context",
)


def _admit_save_budget(
    trace: "Trace",
    tensor: torch.Tensor,
    fields_dict: dict[str, Any],
    *,
    target_device: torch.device,
    retain_in_ram: bool,
    site: str = "primary",
) -> Any:
    """Pre-admit a source-sized retained payload before any copy allocation.

    Parameters
    ----------
    trace:
        Active trace carrying the accountant.
    tensor:
        Live source tensor whose copy would be retained.
    fields_dict:
        Current operation fields used to name the admission site.
    target_device:
        Projected device of the retained payload.
    retain_in_ram:
        Whether this storage route keeps a RAM payload.
    site:
        Accountant admission site: ``"primary"`` for the per-op retained copy,
        ``"lookback_window"`` for a bounded retroactive-save window copy.

    Returns
    -------
    Any
        Opaque reservation reconciled after allocation, or ``None``.
    """

    budget = getattr(trace, "_save_budget_accountant", None)
    if budget is None or not retain_in_ram:
        return None
    label = fields_dict.get("_layer_label_raw") or fields_dict.get("_label_raw") or "<unlabeled>"
    shape = tuple(tensor.shape)
    num_bytes = get_memory_amount_from_metadata(tensor, shape, tensor.dtype)
    return budget.admit(str(label), target_device, int(num_bytes), site=site)


def _commit_save_budget(
    trace: "Trace",
    fields_dict: dict[str, Any],
    reservation: Any,
) -> None:
    """Reconcile admission against alias-aware retained physical storage.

    Parameters
    ----------
    trace:
        Active trace, carrying the per-capture accountant.
    fields_dict:
        Operation fields populated with retained payloads.
    reservation:
        Opaque pre-allocation reservation returned by :func:`_admit_save_budget`.

    Returns
    -------
    None
        Reconciles the accountant, which raises when added transform storage crosses
        the budget.

    Notes
    -----
    Storage identity, not logical field identity, is charged. An identity transform
    therefore counts once. A transform's output size cannot be known before user
    code runs; any storage beyond the source-sized admission is charged here.
    """

    budget = getattr(trace, "_save_budget_accountant", None)
    if budget is None or reservation is None:
        return
    budget.commit(
        reservation,
        (fields_dict.get("out"), fields_dict.get("transformed_out")),
    )


def _iter_tree_tensors(*roots: Any) -> Iterator[torch.Tensor]:
    """Yield every tensor leaf reachable through builtin containers."""

    stack: list[Any] = list(roots)
    while stack:
        value = stack.pop()
        if isinstance(value, torch.Tensor):
            yield value
        elif isinstance(value, dict):
            stack.extend(value.values())
        elif isinstance(value, (list, tuple, set, frozenset)):
            stack.extend(value)


def _admit_saved_args_budget(
    trace: "Trace",
    fields_dict: dict[str, Any],
    t_args: Any,
    t_kwargs: Any,
) -> "list[Any] | None":
    """Admit projected ``save_arg_values`` bytes BEFORE the clones allocate.

    r8 R34 (sol 2): both exhaustive paths cloned every tensor argument first
    and charged only afterward, so a large argument snapshot could allocate
    well beyond the configured budget before the accountant noticed --
    violating the pre-admission half of the save-budget contract for exactly
    the retention family it exists to bound. The source tensors and alias
    graph are available before the clone: project alias-deduped physical
    bytes per device and admit them through the standard ``primary`` site
    (rolling back sibling reservations if one device refuses); the
    post-clone commit reconciles the estimate alias-aware as usual.
    """

    budget = getattr(trace, "_save_budget_accountant", None)
    if budget is None:
        return None
    from ..._save_budget import _retained_storage_identities

    per_device: dict[Any, int] = {}
    seen_identities: set[tuple[Any, ...]] = set()
    for tensor in _iter_tree_tensors(t_args, t_kwargs):
        for identity, num_bytes in _retained_storage_identities(tensor):
            if identity in seen_identities:
                continue
            seen_identities.add(identity)
            per_device[tensor.device] = per_device.get(tensor.device, 0) + num_bytes
    if not per_device:
        return []
    label = fields_dict.get("_layer_label_raw") or fields_dict.get("_label_raw") or "<unlabeled>"
    reservations: list[Any] = []
    try:
        for device, num_bytes in per_device.items():
            reservation = budget.admit(str(label), device, num_bytes, site="primary")
            if reservation is not None:
                reservations.append(reservation)
    except BaseException:
        # Release earlier sibling-device reservations: an empty-payload
        # commit refunds the estimate and charges nothing.
        for reservation in reservations:
            budget.commit(reservation, ())
        raise
    return reservations


def _commit_saved_args_budget(
    trace: "Trace",
    fields_dict: dict[str, Any],
    reservations: "list[Any] | None",
) -> None:
    """Reconcile saved-args reservations against the cloned snapshots."""

    budget = getattr(trace, "_save_budget_accountant", None)
    if budget is None or reservations is None:
        return
    cloned = tuple(
        _iter_tree_tensors(fields_dict.get("saved_args"), fields_dict.get("saved_kwargs"))
    )
    by_device: dict[str, list[torch.Tensor]] = {}
    for tensor in cloned:
        by_device.setdefault(str(tensor.device), []).append(tensor)
    committed_devices: set[str] = set()
    for reservation in reservations:
        device_key = str(reservation.device)
        committed_devices.add(device_key)
        budget.commit(reservation, tuple(by_device.get(device_key, ())))
    # Clones that landed on a device no reservation covered (should not
    # happen -- safe copies preserve device -- but never leave retained
    # storage unledgered).
    label = fields_dict.get("_layer_label_raw") or fields_dict.get("_label_raw") or "<unlabeled>"
    for device_key, tensors in by_device.items():
        if device_key not in committed_devices:
            budget.charge_retained(str(label), tuple(tensors))


def _charge_saved_args_budget(
    trace: "Trace",
    fields_dict: dict[str, Any],
) -> None:
    """Charge ``save_arg_values`` argument snapshots against the save budget.

    Parameters
    ----------
    trace:
        Active trace, carrying the per-capture accountant.
    fields_dict:
        Operation fields whose ``saved_args``/``saved_kwargs`` deep copies were
        just built.

    Notes
    -----
    The argument snapshots are RAM retained for replay exactly like activation
    payloads, but were invisible to the accountant: a ``save_arg_values``
    capture could retain an unbounded second copy of every tensor argument
    without ever tripping the budget. Charges are alias-aware and
    release-credited, so an argument snapshot deduplicating onto
    already-charged storage costs nothing extra.
    """

    budget = getattr(trace, "_save_budget_accountant", None)
    if budget is None:
        return
    tensors: list[torch.Tensor] = []
    stack: list[Any] = [fields_dict.get("saved_args"), fields_dict.get("saved_kwargs")]
    while stack:
        value = stack.pop()
        if isinstance(value, torch.Tensor):
            tensors.append(value)
        elif isinstance(value, dict):
            stack.extend(value.values())
        elif isinstance(value, (list, tuple, set, frozenset)):
            stack.extend(value)
    if not tensors:
        return
    label = fields_dict.get("_layer_label_raw") or fields_dict.get("_label_raw") or "<unlabeled>"
    budget.charge_retained(str(label), tuple(tensors))


def _save_predicate_activation_fields(
    trace: "Trace",
    fields_dict: dict[str, Any],
    tensor: torch.Tensor,
    spec: CaptureSpec,
    ctx: RecordContext,
    activation_transform: Callable[..., Any] | None,
) -> None:
    """Save a ``trace(save=...)`` activation through the shared storage resolver.

    Parameters
    ----------
    trace:
        Active trace.
    fields_dict:
        Mutable operation fields for the current output.
    tensor:
        Live tensor selected by the predicate.
    spec:
        Resolved capture spec for this output.
    ctx:
        Predicate context used for transform error messages.
    activation_transform:
        Optional activation transform configured for the trace.
    """

    options = getattr(trace, "_predicate_save_options", None)
    streaming = None if options is None else options.streaming
    intent = StorageIntent(
        in_ram=streaming is None or streaming.bundle_path is None or streaming.retain_in_memory,
        on_disk=streaming is not None and streaming.bundle_path is not None,
    )
    budget_reservation = _admit_save_budget(
        trace,
        tensor,
        fields_dict,
        target_device=_retention_device(tensor, spec.device),
        retain_in_ram=intent.in_ram,
    )
    (
        ram_payload,
        disk_payload,
        transformed_ram_payload,
        transformed_disk_payload,
    ) = _resolve_storage(
        tensor,
        spec,
        intent,
        activation_transform=activation_transform,
        save_raw_activations=getattr(trace, "save_raw_activations", True),
        ctx=ctx,
        kind="activation",
    )
    metadata_tensor = ram_payload
    if metadata_tensor is None:
        metadata_tensor = disk_payload
    if metadata_tensor is None:
        metadata_tensor = tensor
    fields_dict["shape"] = tuple(metadata_tensor.shape)
    fields_dict["dtype"] = metadata_tensor.dtype
    fields_dict["activation_memory"] = get_memory_amount_from_metadata(
        metadata_tensor,
        fields_dict["shape"],
        fields_dict["dtype"],
    )
    fields_dict["out"] = ram_payload
    if isinstance(ram_payload, torch.Tensor):
        mark_detached_saved_activation(
            tensor,
            ram_payload,
            fields_dict.get("_layer_label_raw"),
        )
    fields_dict["transformed_out"] = transformed_ram_payload
    transformed_metadata = transformed_ram_payload
    if transformed_metadata is None:
        transformed_metadata = transformed_disk_payload
    fields_dict["transformed_out_shape"] = _shape_or_none(transformed_metadata)
    fields_dict["transformed_out_dtype"] = _dtype_or_none(transformed_metadata)
    fields_dict["transformed_activation_memory"] = _memory_or_none(transformed_metadata)
    fields_dict["has_saved_activation"] = True
    _commit_save_budget(trace, fields_dict, budget_reservation)
    _stream_predicate_payloads(
        trace,
        fields_dict,
        disk_payload=disk_payload,
        transformed_disk_payload=transformed_disk_payload,
    )
    out_sink = getattr(trace, "_out_sink", None)
    if out_sink is not None and isinstance(ram_payload, torch.Tensor):
        out_sink(fields_dict["_label_raw"], ram_payload)


def _stream_predicate_payloads(
    trace: "Trace",
    fields_dict: dict[str, Any],
    *,
    disk_payload: torch.Tensor | None,
    transformed_disk_payload: torch.Tensor | None,
) -> None:
    """Write predicate-selected disk payloads during forward capture.

    Parameters
    ----------
    trace:
        Active trace whose writer receives payload blobs.
    fields_dict:
        Mutable operation fields receiving pending blob ids.
    disk_payload:
        Detached raw payload for disk storage.
    transformed_disk_payload:
        Detached transformed payload for disk storage.
    """

    writer = getattr(trace, "_out_writer", None)
    if writer is None:
        return
    label = fields_dict["_label_raw"]
    for payload, pending_field, kind in (
        (disk_payload, "_pending_blob_id", "out"),
        (transformed_disk_payload, "_pending_transformed_out_blob_id", "transformed_out"),
    ):
        if payload is None:
            continue
        blob_id = writer.next_blob_id()
        fields_dict[pending_field] = blob_id
        writer.submit_blob(blob_id, payload, kind=kind, label=label)


def _module_stack_frames_from_fields(fields_dict: dict[str, Any]) -> tuple[ModuleStackFrame, ...]:
    """Project exhaustive op fields into predicate module-stack frames.

    Parameters
    ----------
    fields_dict:
        Live exhaustive op field mapping.

    Returns
    -------
    tuple[ModuleStackFrame, ...]
        Module frames suitable for ``RecordContext.module_stack``.
    """

    frames: list[ModuleStackFrame] = []
    for module_address, module_pass in fields_dict.get("modules", ()):
        frames.append(
            ModuleStackFrame(
                address=str(module_address),
                module_type="",
                module_id=0,
                pass_index=int(module_pass),
            )
        )
    return tuple(frames)


def _append_trace_predicate_context(trace: "Trace", ctx: RecordContext) -> None:
    """Append a trace selective-save context to the bounded runtime window.

    Parameters
    ----------
    trace:
        Active trace.
    ctx:
        Context to append.

    Returns
    -------
    None
        Mutates trace-owned predicate runtime state.
    """

    history_size = int(getattr(trace, "_predicate_history_size", 8))
    if history_size == 0:
        return
    history = getattr(trace, "_predicate_history", None)
    if history is None:
        history = deque()
        trace._predicate_history = history
    history.append(ctx)
    while len(history) > history_size:
        history.popleft()


def _trace_followed_by_candidate_selector(trace: "Trace") -> BaseSelector | None:
    """Return the candidate selector for ``candidate & followed_by(successor)``."""

    from ...ir.selector_eval import split_followed_by_conjunction

    options = getattr(trace, "_predicate_save_options", None)
    predicate = None if options is None else options.keep_op
    split = split_followed_by_conjunction(predicate)
    if split is None:
        return None
    return split[1]


def _retain_lookback_candidate(
    trace: "Trace",
    ctx: RecordContext,
    fields_dict: dict[str, Any],
    tensor: torch.Tensor,
) -> None:
    """Retain one candidate payload in the bounded retroactive-save window."""

    if ctx.raw_label is None:
        return
    lookback = int(getattr(trace, "_predicate_lookback", 0))
    if lookback <= 0:
        return
    policy = str(getattr(trace, "_predicate_lookback_payload_policy", "metadata_only"))
    if policy == "metadata_only":
        return
    candidate_selector = _trace_followed_by_candidate_selector(trace)
    if candidate_selector is None or not candidate_selector(ctx):
        return
    payload = _copy_lookback_payload(trace, fields_dict, tensor, policy)
    candidates = getattr(trace, "_predicate_lookback_candidates", None)
    if candidates is None:
        candidates = deque()
        trace._predicate_lookback_candidates = candidates
    candidates.append(_RetainedLookbackCandidate(raw_label=ctx.raw_label, payload=payload))
    while len(candidates) > lookback:
        candidates.popleft()


def _copy_lookback_payload(
    trace: "Trace",
    fields_dict: dict[str, Any],
    tensor: torch.Tensor,
    policy: str,
) -> _RetainedLookbackPayload:
    """Copy a candidate tensor according to the lookback payload policy.

    Window copies are retained RAM bytes like any other saved activation, so
    they are admitted against the save budget before the copy allocates and
    reconciled after; window eviction and end-of-capture cleanup release the
    payloads, which credits the charge back through the accountant's release
    watchers. Promoted candidates stay retained and therefore stay charged.
    """

    detach = policy != "grad_connected"
    if policy == "disk_spilled":
        warnings.warn(
            "lookback_payload_policy='disk_spilled' currently retains the bounded candidate "
            "payload in memory before final bundle streaming.",
            RuntimeWarning,
            stacklevel=3,
        )
    budget_reservation = _admit_save_budget(
        trace,
        tensor,
        fields_dict,
        target_device=_retention_device(tensor, fields_dict["output_device"]),
        retain_in_ram=True,
        site="lookback_window",
    )
    # Single-transport retention (r8 b5 R35): the copy materializes directly
    # on the retention device; the follow-up move is a safety net for
    # transport-fallback results and stays a no-op on the common path.
    raw_out = safe_copy(tensor, detach, target_device=fields_dict["output_device"])
    if fields_dict["output_device"] not in [str(raw_out.device), "same"]:
        raw_out = safe_to(raw_out, fields_dict["output_device"])
    transformed_out = None
    if policy == "transformed" and trace.activation_transform is not None:
        transformed_out = apply_transform(
            label=fields_dict.get("_layer_label_raw"),
            raw_label=fields_dict.get("_label_raw"),
            func_name=fields_dict.get("func_name"),
            tensor=raw_out,
            transform=trace.activation_transform,
            transform_kind="activation",
            streaming_active=False,
        )
        validate_train_mode_transform_output(
            raw_tensor=raw_out,
            transformed_tensor=transformed_out,
            transform_kind="activation",
            tripwire_armed=train_mode_tripwire_armed(
                backward_ready=fields_dict.get(
                    "backward_ready", getattr(trace, "backward_ready", False)
                ),
                transform=trace.activation_transform,
            ),
            label=fields_dict.get("_layer_label_raw"),
        )
    store_raw = policy != "transformed" or getattr(trace, "save_raw_activations", True)
    raw_shape = tuple(raw_out.shape)
    raw_dtype = raw_out.dtype
    _commit_save_budget(
        trace,
        {"out": raw_out if store_raw else None, "transformed_out": transformed_out},
        budget_reservation,
    )
    return _RetainedLookbackPayload(
        raw_out=raw_out if store_raw else None,
        transformed_out=transformed_out,
        shape=raw_shape,
        dtype=raw_dtype,
        activation_memory=get_memory_amount_from_metadata(raw_out, raw_shape, raw_dtype),
        transformed_shape=_shape_or_none(transformed_out),
        transformed_dtype=_dtype_or_none(transformed_out),
        transformed_memory=_memory_or_none(transformed_out),
    )


def _apply_retroactive_decision(
    trace: "Trace",
    decision: RetroactiveCaptureDecision,
) -> None:
    """Mark retained candidate events as saved after a successor matches."""

    policy = str(getattr(trace, "_predicate_lookback_payload_policy", "metadata_only"))
    if policy == "metadata_only":
        raise PredicateError(
            "tl.followed_by(...) requires lookback_payload_policy other than "
            "'metadata_only' so candidate payloads are retained. "
            "Remedy: pass lookback_payload_policy='detached_raw' or another "
            "payload-retaining policy.",
            code="lookback_payload_policy_conflict",
        )
    candidates = getattr(trace, "_predicate_lookback_candidates", ())
    by_label = {
        candidate.raw_label: candidate
        for candidate in candidates
        if isinstance(candidate, _RetainedLookbackCandidate)
    }
    for raw_label in decision.target_raw_labels:
        candidate = by_label.get(raw_label)
        if candidate is None:
            warnings.warn(
                f"followed_by target {raw_label!r} was not retained in the payload lookback "
                "window; increase lookback.",
                RuntimeWarning,
                stacklevel=3,
            )
            continue
        if candidate.marked:
            continue
        candidate.marked = True
        _replace_event_with_retained_payload(trace, raw_label, candidate.payload)


def _replace_event_with_retained_payload(
    trace: "Trace",
    raw_label: str,
    payload: _RetainedLookbackPayload,
) -> None:
    """Replace a frozen op event with a saved-output version."""

    event = trace.capture_events.op_event_by_label_raw.get(raw_label)
    if event is None:
        return
    tensor_ref = dataclasses.replace(
        event.output.tensor,
        shape=payload.shape,
        dtype=str(payload.dtype),
        device=str(payload.raw_out.device)
        if payload.raw_out is not None
        else event.output.tensor.device,
        requires_grad=payload.raw_out.requires_grad
        if payload.raw_out is not None
        else event.output.tensor.requires_grad,
        memory=payload.activation_memory,
        payload=payload.raw_out,
        backend_handle_id=str(id(payload.raw_out)) if payload.raw_out is not None else None,
    )
    transformed_ref = None
    if payload.transformed_out is not None:
        transformed_ref = TensorRef(
            label_raw=raw_label,
            shape=payload.transformed_shape,
            dtype=str(payload.transformed_dtype),
            device=str(payload.transformed_out.device),
            requires_grad=payload.transformed_out.requires_grad,
            memory=payload.transformed_memory,
            payload=payload.transformed_out,
            blob_ref=None,
            backend_handle_id=str(id(payload.transformed_out)),
        )
    output_ref = dataclasses.replace(
        event.output,
        tensor=tensor_ref,
        transformed_tensor=transformed_ref,
        has_saved_activation=True,
    )
    trace.capture_events.append_amendment(
        amend_lookback_retention(event.seq, raw_label, output=output_ref, predicate_matched=True)
    )


def _build_trace_predicate_context(
    trace: "Trace",
    fields_dict: dict[str, Any],
    tensor: torch.Tensor,
    *,
    parent_labels: tuple[str, ...] | None = None,
    output_index: int | None = None,
    is_bottom_level_func: bool = True,
) -> RecordContext:
    """Build the shared trace predicate context for one op.

    Parameters
    ----------
    trace:
        Active trace.
    fields_dict:
        Live exhaustive op fields for the tensor.
    tensor:
        Output tensor being considered.

    Parameters
    ----------
    trace:
        Active trace.
    fields_dict:
        Live exhaustive op fields for the tensor.
    tensor:
        Output tensor being considered.
    parent_labels:
        Optional raw parent labels supplied by the caller.
    output_index:
        Optional output index supplied by the caller.
    is_bottom_level_func:
        Whether this is a bottom-level function output.

    Returns
    -------
    RecordContext
        Predicate context shared by save and intervention slots.
    """

    history = tuple(getattr(trace, "_predicate_history", ()))
    raw_label = fields_dict["_label_raw"]
    module_address = fields_dict.get("module")
    module_pass_index = None
    if isinstance(module_address, tuple) and len(module_address) == 2:
        module_address, module_pass_index = module_address
    module_address = None if module_address is None else str(module_address)
    module_pass_index = None if module_pass_index is None else int(module_pass_index)
    return build_op_record_context(
        kind="op",
        label=raw_label,
        raw_label=raw_label,
        raw_index=int(fields_dict["raw_index"]),
        layer_type=str(fields_dict["type"]),
        type_index=int(fields_dict["type_index"]),
        func_name=fields_dict.get("func_name"),
        parent_labels=tuple(fields_dict.get("parents", ()))
        if parent_labels is None
        else parent_labels,
        tensor=tensor,
        output_index=fields_dict.get("multi_output_index")
        if output_index is None
        else output_index,
        is_bottom_level_func=is_bottom_level_func,
        module_stack=_module_stack_frames_from_fields(fields_dict),
        history=history,
        op_counts=dict(trace._raw_graph_ws.raw_layer_type_counter),
        pass_index=int(fields_dict.get("pass_index", 0)),
        event_index=int(fields_dict["raw_index"]),
        step_index=fields_dict.get("step_index"),
        capture_start_time=float(getattr(trace, "capture_start_time", time.time())),
        include_source_events=False,
        sample_id=None,
        address=module_address,
        module_type=None,
        module_pass_index=module_pass_index,
        is_transform=bool(fields_dict.get("is_transform", False)),
        transform_kind=fields_dict.get("transform_kind"),
    )
