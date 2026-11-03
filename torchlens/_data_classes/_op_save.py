"""Tensor save, grad logging, transform validation and copy logic for ``Op``.

Logic bodies live here as functions taking the record; ``Op`` keeps one thin
method per verb.
"""

from __future__ import annotations

import copy
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import torch

from .._constants import LAYER_PASS_LOG_FIELD_ORDER
from .._errors import InvalidArgumentError, UserTransformError
from .._io.format_errors import ArtifactError
from .._quantities import Bytes, Duration, Flops, as_bytes
from .._trace_core.op_store import row_clone_scope
from .._utils._torch_compat import tensor_version_or_none
from .._utils.tensor_utils import (
    SaveMode,
    copy_tensor_payload,
    get_memory_amount,
    get_memory_amount_from_metadata,
    safe_to,
)
from ._op_dedup import _dedup_cached_identity_out, _dedup_saved_activation_out
from ._op_grad_stream import stream_grad_blobs
from ._op_transforms import (
    apply_transform,
    train_mode_tripwire_armed,
    transform_error_message,
    validate_streaming_transform_output,
    validate_train_mode_transform_output,
)

if TYPE_CHECKING:
    from .op import Op
    from .trace import Trace

__all__ = [
    "save_activation",
]

_DEEPCOPY_IMMUTABLE_TYPES = (
    str,
    bytes,
    int,
    float,
    bool,
    type(None),
    torch.dtype,
    torch.device,
    Bytes,
    Duration,
    Flops,
)


_OUTPUT_NODE_REPLACED_FIELDS = frozenset(
    {
        "_edge_uses",
        "_label_raw",
        "_layer_label_raw",
        "_param_barcodes",
        "_param_logs",
        "activation_memory",
        "annotations",
        "arg_names",
        "atomic_module_call",
        "autograd_memory",
        "bytes_delta_at_call",
        "bytes_peak_at_call",
        "children",
        "code_context",
        "container_path",
        "container_spec",
        "dropped_edge_tensor_args",
        "dtype",
        "equivalence_class",
        "equivalent_ops",
        "func",
        "func_config",
        "func_duration",
        "func_name",
        "func_non_tensor_args",
        "func_rng_states",
        "grad_fn_class_name",
        "has_children",
        "has_out_variations",
        "has_output_descendant",
        "input_to_module_calls",
        "internal_source_parents",
        "is_edited",
        "edits",
        "io_role",
        "is_atomic_module",
        "is_buffer",
        "is_final_output",
        "is_input",
        "is_internal_source",
        "is_module_output",
        "is_output",
        "is_transform",
        "type",
        "module",
        "module_call_stack",
        "modules",
        "non_tensor_kwargs",
        "non_tensor_pos_args",
        "num_args_total",
        "num_autograd_tensors",
        "num_kwargs",
        "num_params",
        "num_params_frozen",
        "num_params_trainable",
        "num_passes",
        "num_pos_args",
        "out",
        "out_versions_by_child",
        "output_descendants",
        "output_of_module_calls",
        "output_of_modules",
        "param_memory",
        "param_shapes",
        "parent_arg_positions",
        "parent_param_ops",
        "parent_params",
        "parents",
        "pass_number",
        "raw_index",
        "recurrent_ops",
        "saved_args",
        "saved_kwargs",
        "shape",
        "transform_chain",
        "transform_config",
        "transform_fn_name",
        "transform_fn_qualname",
        "transform_fn_source",
        "transform_kind",
        "transformed_activation_memory",
        "transformed_out",
        "transformed_out_dtype",
        "transformed_out_shape",
        "unattributed_tensor_args",
        "var_names",
    }
)


_WARNED_REFERENCE_SAVE_MODE = False


def _copy_op_field_value(value: Any) -> Any:
    """Copy one Op field without deep-copy dispatch for immutable values.

    Parameters
    ----------
    value:
        Stored field value.

    Returns
    -------
    Any
        The original immutable value or an independent deep copy.
    """

    if isinstance(value, _DEEPCOPY_IMMUTABLE_TYPES):
        return value
    if isinstance(value, tuple) and all(
        isinstance(member, _DEEPCOPY_IMMUTABLE_TYPES) for member in value
    ):
        return value
    return copy.deepcopy(value)


def _shape_or_none(value: Any) -> tuple[int, ...] | None:
    """Return a tensor shape tuple, or ``None`` for non-tensor values."""

    return tuple(value.shape) if isinstance(value, torch.Tensor) else None


def _dtype_or_none(value: Any) -> torch.dtype | None:
    """Return a tensor dtype, or ``None`` for non-tensor values."""

    return value.dtype if isinstance(value, torch.Tensor) else None


def _memory_or_none(value: Any) -> Bytes | None:
    """Return tensor memory in bytes, or ``None`` for non-tensor values."""

    if not isinstance(value, torch.Tensor):
        return None
    return as_bytes(get_memory_amount_from_metadata(value, tuple(value.shape), value.dtype))


def _warn_reference_save_mode_once() -> None:
    """Emit the reference-mode mutation warning once per process."""

    global _WARNED_REFERENCE_SAVE_MODE
    if _WARNED_REFERENCE_SAVE_MODE:
        return
    warnings.warn(
        "save_mode='reference' stores source tensors by reference; reading a mutated "
        "saved tensor raises MutatedReferenceError.",
        UserWarning,
        stacklevel=3,
    )
    _WARNED_REFERENCE_SAVE_MODE = True


def _effective_activation_save_mode(
    trace: Trace | None,
    *,
    func_name: str | None,
    is_inplace: bool = False,
) -> SaveMode:
    """Return the effective saved-activation mode for one operation."""

    save_mode = cast(SaveMode, trace._options_save_mode if trace is not None else "copy")
    if save_mode not in {"copy", "reference", "view", "cpu_async"}:
        raise InvalidArgumentError(
            "save_mode must be one of 'copy', 'reference', 'view', or 'cpu_async'; "
            f"received {save_mode!r}",
            code="save_mode_invalid",
            remedy="set save_mode to 'copy', 'reference', 'view', or 'cpu_async'",
            argument="save_mode",
        )
    if save_mode == "reference":
        _warn_reference_save_mode_once()
        if is_inplace or _has_mutating_func_name(func_name):
            return "copy"
    return save_mode


def _has_mutating_func_name(func_name: str | None) -> bool:
    """Return whether ``func_name`` names a mutating call (the reference-mode fallback).

    ``"__add__".endswith("_")`` is true for EVERY dunder, so the trailing
    underscore marks a mutation only outside dunders. Among dunders only the
    augmented assignments (``__iadd__``...) and the item setters mutate. A
    non-mutating dunder (``a * b``, ``a + b``) returns a fresh tensor, which
    reference mode keeps as is: copying it would store a payload that is a
    dead end for autograd, so gradient probes from it (receptive and
    projective fields' ``method="gradient"``) would find no path downstream.
    """

    if func_name is None:
        return False
    if func_name.startswith("__") and func_name.endswith("__"):
        from ..backends.torch._ops_capture_records import (
            _is_inplace_augmented_assignment_dunder,
        )
        from ..backends.torch.ops import _SETTER_MUTATION_FUNC_NAMES

        return (
            _is_inplace_augmented_assignment_dunder(func_name)
            or func_name in _SETTER_MUTATION_FUNC_NAMES
        )
    return func_name.endswith("_")


def _stamp_reference_out(
    annotations: dict[str, Any], raw_out: torch.Tensor, save_mode: SaveMode
) -> None:
    """Store reference-mode mutation metadata for a saved output."""

    if save_mode != "reference":
        return
    annotations["save_mode"] = "reference"
    annotations["saved_out_version"] = tensor_version_or_none(raw_out)


def _copy_for_output(op: Op, *, _store: Any = None) -> Op:
    """Clone this op for immediate conversion into a synthetic output node.

    Parameters
    ----------
    op:
        The op record this function reads and writes.
    _store:
        Optional owning row store for the synthesized node.

    Returns
    -------
    Op
        Clone whose soon-to-be-replaced fields avoid unnecessary deep copies.
    """

    return op._copy_with_shallow_fields(_OUTPUT_NODE_REPLACED_FIELDS, _store=_store)


def _copy_with_shallow_fields(
    op: Op,
    extra_shallow_fields: frozenset[str],
    *,
    _store: Any = None,
) -> Op:
    """Clone this op while sharing fields the caller replaces immediately.

    Parameters
    ----------
    op:
        The op record this function reads and writes.
    extra_shallow_fields:
        Additional fields safe to share for this construction path.
    _store:
        Optional owning row store for the clone.

    Returns
    -------
    Op
        Selective-depth clone.
    """

    fields_dict = {}
    fields_not_to_deepcopy = {
        "func",
        "grad_fn_class_name",
        "grad_fn_handle",
        "source_trace",
        "func_rng_states",
        "saved_args",
        "saved_kwargs",
        "args_template",
        "kwargs_template",
        "parent_params",
        "out",
        "transformed_out",
        "transformed_grad",
        "out_versions_by_child",
        "container_spec",
    } | extra_shallow_fields

    # The whole-schema getattr loop is a ROW-CLONE read, not a set of
    # per-column dependencies: under a combined step audit these reads
    # are tagged category (d) via the clone scope (design-ppdag-v3
    # §2.4d) — legal on row-creating steps, a finding elsewhere. The
    # scope is one dict lookup when no audit is armed; clone WRITES
    # bypass interception entirely (adopt_row).
    with row_clone_scope(object.__getattribute__(op, "_core")):
        for field in LAYER_PASS_LOG_FIELD_ORDER:
            if field not in fields_not_to_deepcopy:
                fields_dict[field] = _copy_op_field_value(getattr(op, field, None))
            else:
                fields_dict[field] = getattr(op, field, None)
    copied_entry = type(op)(fields_dict, _store=_store)
    return copied_entry


def save_activation(
    op: Op,
    t: torch.Tensor,
    call_args: tuple[list[Any] | tuple[Any, ...], dict[str, Any]],
    save_args: bool,
    activation_transform: Callable[..., Any] | None = None,
) -> None:
    """Save the output tensor (and optionally args) for this operation.

    Flow:
    1. Clone the tensor via ``safe_copy`` (strips tl_ attributes to avoid
       logging the copy operation).
    2. Move to ``output_device`` if different from the tensor's current device.
    3. Apply ``activation_transform`` inside ``pause_logging()`` to prevent
       the transform's own tensor ops from being logged.
    4. Optionally deep-copy function args/kwargs via ``_recursive_safe_copy``.

    Args:
        op: The op record this function reads and writes.
        t: The output tensor of the operation.
        call_args: The operation's ``(t_args, t_kwargs)`` pair.
        save_args: Whether to deep-copy and store args/kwargs.
        activation_transform: Optional transform applied to the tensor
            before storing (e.g. detach, to-numpy, normalize).
    """

    from .._save_budget import SaveBudgetExceededError
    from ..backends.torch._tl import mark_detached_saved_activation

    t_args, t_kwargs = call_args
    trace = op._source_trace
    writer = getattr(trace, "_out_writer", None) if trace is not None else None
    try:
        save_mode = _effective_activation_save_mode(
            trace,
            func_name=op.func_name,
            is_inplace=bool(op.is_inplace),
        )
        budget = getattr(trace, "_save_budget_accountant", None)
        target_device = t.device
        if save_mode == "cpu_async":
            target_device = torch.device("cpu")
        elif op.output_device not in ("same", str(t.device)):
            target_device = torch.device(op.output_device)
        save_raw_activations = trace._options_save_raw_activations if trace is not None else True
        store_raw = save_raw_activations or activation_transform is None
        # Explorer P2 (same gate as the exhaustive hot path): a declared
        # summary reducer with raw retention off skips the transient
        # clone; the reducer sees a detached view of the live output.
        from .._ir.summary_role import is_summary_transform

        reduce_only = not store_raw and is_summary_transform(activation_transform)
        budget_reservation = (
            None
            if budget is None or reduce_only
            else budget.admit(
                op._layer_label_raw,
                target_device,
                get_memory_amount_from_metadata(t, tuple(t.shape), t.dtype),
            )
        )
        # Pre-copy identity probe (dedup-after-copy ordering): a hit
        # reuses the already-saved payload and skips the clone entirely.
        # Restricted to plain "copy" mode -- reference/view copies are
        # free and cpu_async has fence side effects.
        dedup_cached_out = (
            _dedup_cached_identity_out(trace, t, op.annotations, save_args)
            if store_raw and save_mode == "copy"
            else None
        )
        if reduce_only:
            raw_out = t
        elif dedup_cached_out is not None:
            raw_out = dedup_cached_out
        else:
            # Clone the tensor, optionally detaching from autograd graph.
            raw_out = copy_tensor_payload(
                t,
                detach_tensor=op.detach_saved_activations,
                save_mode=save_mode,
            )
            # Move to the user-requested output device if needed.
            if op.output_device not in [str(raw_out.device), "same"]:
                raw_out = safe_to(raw_out, op.output_device)
            _stamp_reference_out(op.annotations, raw_out, save_mode)

        op.shape = tuple(raw_out.shape)
        op.dtype = raw_out.dtype
        op.activation_memory = Bytes(get_memory_amount_from_metadata(raw_out, op.shape, op.dtype))

        if store_raw:
            if dedup_cached_out is None:
                raw_out = _dedup_saved_activation_out(
                    trace,
                    t,
                    raw_out,
                    op._layer_label_raw,
                    op.annotations,
                    save_args,
                )
            if isinstance(raw_out, torch.Tensor):
                mark_detached_saved_activation(t, raw_out, op._layer_label_raw)
        op._internal_set("out", raw_out if store_raw else None)

        op._internal_set("transformed_out", None)
        op.transformed_out_shape = None
        op.transformed_out_dtype = None
        op.transformed_activation_memory = None
        if activation_transform is not None:
            transformed_out = op._apply_transform(
                raw_out,
                activation_transform,
                transform_kind="activation",
                streaming_active=writer is not None,
            )
            if reduce_only:
                from .._ir.summary_role import ensure_summary_output_owns_storage

                transformed_out = ensure_summary_output_owns_storage(transformed_out, t)
            op._internal_set("transformed_out", transformed_out)
            op._validate_train_mode_transform_output(
                raw_out,
                op.transformed_out,
                transform_kind="activation",
                transform=activation_transform,
            )
            op._validate_streaming_transform_output(
                op.transformed_out,
                transform_kind="activation",
                streaming_active=writer is not None,
            )
            op.transformed_out_shape = _shape_or_none(op.transformed_out)
            op.transformed_out_dtype = _dtype_or_none(op.transformed_out)
            op.transformed_activation_memory = _memory_or_none(op.transformed_out)
        if budget is not None:
            if reduce_only:
                # No source-sized admission happened (no clone allocated);
                # charge the retained summary output directly.
                budget.charge_retained(str(op._layer_label_raw), (op.transformed_out,))
            else:
                budget.commit(budget_reservation, (op.out, op.transformed_out))
    except Exception as exc:
        if writer is not None:
            writer.abort(f"Failed while saving out for {op._streaming_label}: {exc}")
            if isinstance(exc, (SaveBudgetExceededError, UserTransformError)):
                raise
            raise ArtifactError(f"Streaming out save failed for {op._streaming_label}.") from exc
        raise

    op.has_saved_activation = True

    if trace is not None:
        out_sink = getattr(trace, "_out_sink", None)
        if out_sink is not None and isinstance(op.out, torch.Tensor):
            out_sink(op._streaming_label, op.out)

        if writer is not None and trace._wrapper_runtime_ws.in_exhaustive_pass:
            op._stream_tensor_blob(
                writer,
                tensor_field="out",
                pending_field="_pending_blob_id",
                kind="out",
            )
            op._stream_tensor_blob(
                writer,
                tensor_field="transformed_out",
                pending_field="_pending_transformed_out_blob_id",
                kind="transformed_out",
            )

    op._save_call_args(t_args, t_kwargs, save_args)


def _log_tensor_grad(
    op: Op,
    grad: torch.Tensor,
    copy_grad_payload: Callable[..., torch.Tensor],
    prebuilt: tuple[torch.Tensor | None, Any | None] | None = None,
) -> None:
    """Save the grad tensor for this layer's output.

    Called by the backward hook registered during the forward pass. The saved value is
    the gradient of this op's output after TorchLens's edit chain at this site
    (``tl.grad(site)`` edits run first), so ``.grad`` and an edit agree. The grad is
    ``detach().clone()``'d - a bare copy, not deep-copied - so it's independent of the
    autograd graph but cheap to store.

    Args:
        op: The op record this function reads and writes.
        copy_grad_payload: The backend's gradient payload copier
            (``tensor_tracking._copy_grad_payload``).
        grad: The grad tensor flowing back through this operation.
        prebuilt: Budget-charged ``(raw_payload, transformed_payload)`` pair already built
            (and transform-validated) by the event-sidecar path for this exact grad. When
            given, the slot REUSES those objects: no second clone, no second
            ``grad_transform`` execution, no uncharged retention (grind-r6 b5 R34-N1/R35-N1).
    """
    trace = op._source_trace
    raw_grad = grad
    op.grad_shape = tuple(raw_grad.shape)
    op.grad_dtype = raw_grad.dtype
    op.gradient_memory = Bytes(get_memory_amount(raw_grad))
    grad_transform = trace._options_grad_transform if trace is not None else None
    op._internal_set("transformed_grad", None)
    op.transformed_grad_shape = None
    op.transformed_grad_dtype = None
    op.transformed_gradient_memory = None
    writer = getattr(trace, "_out_writer", None) if trace is not None else None
    if prebuilt is not None:
        raw_payload, transformed_payload = prebuilt
        if grad_transform is not None:
            op._internal_set("transformed_grad", transformed_payload)
            op.transformed_grad_shape = _shape_or_none(op.transformed_grad)
            op.transformed_grad_dtype = _dtype_or_none(op.transformed_grad)
            op.transformed_gradient_memory = _memory_or_none(op.transformed_grad)
        op._internal_set("grad", raw_payload)
        op.has_grad = True
        stream_grad_blobs(op, trace, writer)
        return
    # Admit BEFORE the transform/clone allocate (r8 R34, fable F1): a
    # policy-DENIED label reaching this legacy slot (callable/selector
    # save_grads with _grad_op_nums_to_save == "all") used to retain a
    # full uncharged grad clone -- invisible to a tight save_budget.
    # Source-sized reservation, transform delta reconciled at commit,
    # exactly the primary-site contract.
    budget = getattr(trace, "_save_budget_accountant", None) if trace is not None else None
    grad_reservation = None
    if budget is not None:
        grad_reservation = budget.admit(
            str(getattr(op, "_layer_label_raw", "<grad>")),
            raw_grad.device,
            int(raw_grad.numel() * raw_grad.element_size()),
            site="primary",
        )
    if grad_transform is not None:
        op._internal_set(
            "transformed_grad",
            op._apply_transform(
                raw_grad,
                grad_transform,
                transform_kind="grad",
                streaming_active=writer is not None,
            ),
        )
        op._validate_train_mode_transform_output(
            raw_grad,
            op.transformed_grad,
            transform_kind="grad",
            transform=grad_transform,
        )
        op._validate_streaming_transform_output(
            op.transformed_grad,
            transform_kind="grad",
            streaming_active=writer is not None,
        )
        op.transformed_grad_shape = _shape_or_none(op.transformed_grad)
        op.transformed_grad_dtype = _dtype_or_none(op.transformed_grad)
        op.transformed_gradient_memory = _memory_or_none(op.transformed_grad)

    save_raw_gradients = trace._options_save_raw_grads if trace is not None else True
    store_raw = save_raw_gradients or grad_transform is None
    # Gradient payloads are ALWAYS genuine snapshots: under
    # save_mode="reference"/"view" an aliased payload would be silently
    # rewritten by later user mutation of the seed gradient or by
    # AccumulateGrad accumulating in place on the next backward, and no
    # wrapped in-place op exists on the grad path to disclose it. Route
    # through the same chokepoint the event-sidecar path uses.

    save_mode = cast(SaveMode, trace._options_save_mode if trace is not None else "copy")
    op._internal_set(
        "grad",
        copy_grad_payload(raw_grad, save_mode=save_mode) if store_raw else None,
    )
    op.has_grad = True
    if budget is not None and grad_reservation is not None:
        budget.commit(grad_reservation, (op.grad, op.transformed_grad))
    stream_grad_blobs(op, trace, writer)


def _apply_transform(
    op: Op,
    tensor: torch.Tensor,
    transform: Callable[..., Any],
    *,
    transform_kind: str,
    streaming_active: bool,
) -> Any:
    """Apply a user transform with logging paused and rich error context."""

    return apply_transform(
        label=op._streaming_label,
        raw_label=op._layer_label_raw,
        func_name=op.func_name,
        tensor=tensor,
        transform=transform,
        transform_kind=transform_kind,
        streaming_active=streaming_active,
    )


def _transform_error_message(
    op: Op,
    *,
    transform_kind: str,
    tensor: torch.Tensor,
    streaming_active: bool,
) -> str:
    """Build context for an out or grad transform failure."""

    return transform_error_message(
        label=op._streaming_label,
        raw_label=op._layer_label_raw,
        func_name=op.func_name,
        tensor=tensor,
        transform_kind=transform_kind,
        streaming_active=streaming_active,
    )


def _validate_train_mode_transform_output(
    op: Op,
    raw_tensor: torch.Tensor,
    output: Any,
    *,
    transform_kind: str,
    transform: Any = None,
) -> None:
    """Validate differentiability requirements for train-mode transform outputs."""

    trace = op._source_trace
    validate_train_mode_transform_output(
        raw_tensor=raw_tensor,
        transformed_tensor=output,
        transform_kind=transform_kind,
        tripwire_armed=train_mode_tripwire_armed(
            backward_ready=trace._options_backward_ready if trace is not None else False,
            transform=transform,
        ),
        label=op._streaming_label,
    )


def _validate_streaming_transform_output(
    op: Op,
    output: Any,
    *,
    transform_kind: str,
    streaming_active: bool,
) -> None:
    """Validate transformed tensors before streaming bundle finalization.

    Parameters
    ----------
    op:
        The op record this function reads and writes.
    output:
        Value returned by the user transform.
    transform_kind:
        Transform kind, either ``"out"`` or ``"grad"``.
    streaming_active:
        Whether a streaming bundle writer is active for this trace.

    Returns
    -------
    None
        Raises if streaming cannot serialize the transformed value.

    Raises
    ------
    ArtifactError
        If streaming is active and the transform returns a non-tensor or sparse tensor.
    """

    try:
        validate_streaming_transform_output(
            transformed_tensor=output,
            transform_kind=transform_kind,
            streaming_active=streaming_active,
            label=op._streaming_label,
        )
    except ArtifactError as exc:
        op._abort_streaming_writer(str(exc))
        raise


def _abort_streaming_writer(op: Op, message: str) -> None:
    """Abort the active streaming writer when one is attached.

    Parameters
    ----------
    op:
        The op record this function reads and writes.
    message:
        Reason written to the partial bundle marker.

    Returns
    -------
    None
        Mutates the writer state if present.
    """

    trace = op._source_trace
    writer = getattr(trace, "_out_writer", None) if trace is not None else None
    if writer is not None:
        writer.abort(message)


def _stream_tensor_blob(
    op: Op,
    writer: Any,
    *,
    tensor_field: str,
    pending_field: str,
    kind: str,
) -> None:
    """Stream one tensor field when present."""

    tensor = getattr(op, tensor_field)
    if tensor is None:
        return
    if not isinstance(tensor, torch.Tensor):
        if kind == "transformed_out":
            message = (
                "Streaming save requires activation_transform outputs to be torch.Tensor "
                f"instances, but layer {op._streaming_label} produced "
                f"{type(tensor).__name__}."
            )
        elif kind == "transformed_grad":
            message = (
                "Streaming save requires grad_transform outputs to be torch.Tensor "
                f"instances, but layer {op._streaming_label} produced "
                f"{type(tensor).__name__}."
            )
        else:
            message = (
                f"{tensor_field} expected a tensor for streaming, got {type(tensor).__name__}."
            )
        writer.abort(message)
        raise ArtifactError(message)
    blob_id = writer.next_blob_id()
    setattr(op, pending_field, blob_id)
    writer.submit_blob(
        blob_id,
        tensor,
        kind=kind,
        label=op._streaming_label,
    )
