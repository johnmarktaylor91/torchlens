"""Op: per-operation metadata for a single invocation of a layer.

Each Op records everything about one tensor operation in the
forward pass: the output tensor itself, the function that produced it,
its parents/children in the computation graph, module containment,
parameter usage, timing, RNG state, and more.

For recurrent models, the same "layer" may execute multiple times; each
execution is a separate Op with a distinct ``pass_index``.  The
aggregate view across ops is provided by :class:`Layer`.

Field categories (matching the LAYER_PASS_LOG_FIELD_ORDER in constants.py):

1. **General info** - raw/final labels, operation numbering, back-reference
   to the owning Trace.
2. **Label info** - human-readable labels in various formats (with/without
   pass qualifier, short form, etc.).
3. **Saved tensor info** - the tensor contents, shape, dtype, size, device
   transfer settings, out transform, and function arguments.
4. **Child tensor variations** - tracks per-child input values for
   validation replay (``out_versions_by_child`` stores RAW values
   because validation compares against ``saved_args``).
5. **Gradient info** - grad tensor and metadata (stored as a bare
   reference via ``log_tensor_grad``, not deep-copied).
6. **Function call info** - the applied function, call stack, timing,
   FLOPs, RNG state, arg metadata, grad_fn_handle, inplace flag.
7. **Param info** - which parameters were used, their shapes and sizes.
8. **Equivalence info** - loop-detection equivalence type and groups.
9. **Graph info** - parent/child/sibling/spouse edges, input/output
   ancestry, distances, buffer/internal-init status.
10. **Conditional info** - boolean branching metadata.
11. **Module info** - module entry/exit tracking, nesting depth,
    bottom-level submodule output status.
"""

import copy
import hashlib
import weakref
import warnings
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Optional,
    TYPE_CHECKING,
    Tuple,
    Union,
    cast,
    Literal,
)

import torch

import importlib

from .._deprecations import MISSING
from .._io import (
    FieldPolicy,
    TLSPEC_VERSION,
    TorchLensIOError,
    default_fill_state,
    read_tlspec_version,
)
from .._errors import MutatedReferenceError, TorchLensPostfuncError
from .._trace_state import TraceState
from .._training_validation import _NON_GRAD_DTYPES, TrainingModeConfigError
from ..constants import LAYER_PASS_LOG_FIELD_ORDER
from ..intervention.types import LAYER_PASS_LOG_FIELD_FORK_POLICY
from ..ir.refs import DeviceRef, DtypeRef
from ..intervention.errors import DirectActivationWriteWarning
from ..quantities import Bytes, Flops, Macs, as_bytes, as_duration, as_flops, as_macs
from .._state import pause_logging
from ._accessor_base import Accessor
from ..utils.tensor_utils import (
    SaveMode,
    concatenate_batch_tensors,
    get_memory_amount,
    get_memory_amount_from_metadata,
    is_functorch_wrapped_tensor,
    print_override,
    safe_copy,
    safe_to,
)
from ..utils.display import tensor_stats_summary

_LAYER_PASS_LOG_FIELD_ORDER_SET = frozenset(LAYER_PASS_LOG_FIELD_ORDER)
_DIRECT_WRITE_GUARDED_FIELDS = frozenset(
    {
        "out",
        "transformed_out",
        "grad",
        "transformed_grad",
        "interventions",
    }
)
_WARNED_REFERENCE_SAVE_MODE = False
_LAYER_PASS_LOG_DEFAULT_FILL: dict[str, Any] = {
    "_source_trace_ref": None,
    "out_ref": None,
    "grad_ref": None,
    "_pending_blob_id": None,
    "_pending_transformed_out_blob_id": None,
    "_pending_grad_blob_id": None,
    "_pending_transformed_grad_blob_id": None,
    "annotations": {},
    "autograd_memory": None,
    "num_autograd_tensors": None,
    "bytes_delta_at_call": None,
    "bytes_peak_at_call": None,
    "transformed_out": None,
    "transformed_out_shape": None,
    "transformed_out_dtype": None,
    "dtype_ref": None,
    "device_ref": None,
    "backend_address": None,
    "resolver_status": "resolved",
    "transformed_activation_memory": None,
    "visualizer_path": None,
    "transformed_grad": None,
    "transformed_grad_shape": None,
    "transformed_grad_dtype": None,
    "transformed_gradient_memory": None,
    "func_call_id": None,
    "container_path": (),
    "multi_output_name": None,
    "intervention_replaced": False,
    "interventions": [],
    "container_spec": None,
    "args_template": None,
    "kwargs_template": None,
    "_edge_uses": [],
    "is_orphan": False,
    "_address_normalized": None,
    "_construction_done": True,
}
_LAYER_PASS_LOG_DEFAULT_FILL = {
    **{field_name: None for field_name in LAYER_PASS_LOG_FIELD_ORDER},
    **_LAYER_PASS_LOG_DEFAULT_FILL,
}


def _recursive_safe_copy(val: Any) -> Any:
    """Deep-copy nested structures, cloning tensors instead of using copy.deepcopy (#44)."""
    if isinstance(val, torch.Tensor):
        return safe_copy(val)
    elif isinstance(val, (list, tuple)):
        return type(val)(_recursive_safe_copy(v) for v in val)
    elif isinstance(val, dict):
        return {k: _recursive_safe_copy(v) for k, v in val.items()}
    return safe_copy(val)


def _shape_or_none(value: Any) -> tuple[int, ...] | None:
    """Return a tensor shape tuple, or ``None`` for non-tensor values."""

    return tuple(value.shape) if isinstance(value, torch.Tensor) else None


def _dtype_or_none(value: Any) -> torch.dtype | None:
    """Return a tensor dtype, or ``None`` for non-tensor values."""

    return value.dtype if isinstance(value, torch.Tensor) else None


def _dtype_ref_or_none(value: Any) -> DtypeRef | None:
    """Return a neutral dtype reference for a dtype-like value."""

    return DtypeRef.from_value(value)


def _device_ref_from_metadata(out: Any, output_device: Any) -> DeviceRef | None:
    """Return a neutral device reference from payload or output-device metadata."""

    if isinstance(out, torch.Tensor):
        return DeviceRef.from_value(out.device)
    if output_device in (None, "same"):
        return None
    return DeviceRef.from_value(output_device)


def _memory_or_none(value: Any) -> Bytes | None:
    """Return tensor memory in bytes, or ``None`` for non-tensor values."""

    if not isinstance(value, torch.Tensor):
        return None
    return as_bytes(get_memory_amount_from_metadata(value, tuple(value.shape), value.dtype))


def _summarize_value(value: Any) -> str:
    """Return a compact, human-readable string for one argument value."""

    if isinstance(value, torch.Tensor):
        shape = "x".join(str(int(d)) for d in value.shape) or "scalar"
        return f"Tensor({shape}, {str(value.dtype).replace('torch.', '')})"
    text = repr(value)
    if len(text) > 60:
        text = text[:57] + "..."
    return text


class GradientRecord:
    """Saved gradient payload observed for one op during one backward pass.

    Parameters
    ----------
    owner:
        Op that received the gradient.
    ordinal:
        One-based local ordinal for this owner.
    backward_pass_index:
        One-based global backward pass number.
    grad:
        Saved raw gradient tensor, if retained.
    transformed_grad:
        Saved transformed gradient tensor, if retained.
    shape:
        Observed raw gradient shape.
    dtype:
        Observed raw gradient dtype string.
    memory:
        Observed raw gradient memory in bytes.
    timestamp:
        Event timestamp.
    """

    PORTABLE_STATE_SPEC: ClassVar[dict[str, FieldPolicy]] = {
        "owner": FieldPolicy.WEAKREF_STRIP,
        "ordinal": FieldPolicy.KEEP,
        "backward_pass_index": FieldPolicy.KEEP,
        "grad": FieldPolicy.BLOB,
        "transformed_grad": FieldPolicy.BLOB,
        "shape": FieldPolicy.KEEP,
        "dtype": FieldPolicy.KEEP,
        "memory": FieldPolicy.KEEP,
        "timestamp": FieldPolicy.KEEP,
    }

    def __init__(
        self,
        *,
        owner: Any,
        ordinal: int,
        backward_pass_index: int,
        grad: torch.Tensor | None,
        transformed_grad: Any | None,
        shape: tuple[int, ...] | None,
        dtype: str | None,
        memory: int | None,
        timestamp: float,
    ) -> None:
        self.owner = owner
        self.ordinal = ordinal
        self.backward_pass_index = backward_pass_index
        self.grad = grad
        self.transformed_grad = transformed_grad
        self.shape = shape
        self.dtype = dtype
        self.memory = Bytes(memory or 0)
        self.timestamp = timestamp

    @property
    def is_saved(self) -> bool:
        """Return whether this record retained a payload."""

        return self.grad is not None or self.transformed_grad is not None

    @property
    def transformed_grad_shape(self) -> tuple[int, ...] | None:
        """Return the transformed gradient payload shape, when tensor-like."""

        return _shape_or_none(self.transformed_grad)

    @property
    def transformed_grad_dtype(self) -> torch.dtype | None:
        """Return the transformed gradient payload dtype, when tensor-like."""

        return _dtype_or_none(self.transformed_grad)

    @property
    def transformed_gradient_memory(self) -> Bytes | None:
        """Return transformed gradient payload memory, when tensor-like."""

        return _memory_or_none(self.transformed_grad)


class GradientRecordAccessor(Accessor[GradientRecord]):
    """Accessor for per-owner gradient records."""

    def __init__(self, records: list[GradientRecord]) -> None:
        """Initialize from records in local ordinal order."""

        super().__init__({str(record.ordinal): record for record in records}, item_list=records)

    def for_pass(self, pass_index: int) -> GradientRecord:
        """Return the gradient record for a one-based backward pass number."""

        matches = [record for record in self._list if record.backward_pass_index == pass_index]
        if len(matches) == 1:
            return matches[0]
        if matches:
            raise ValueError(
                f"Multiple gradient records participated in pass {pass_index}; "
                "use positional indexing on .grads."
            )
        available = [record.backward_pass_index for record in self._list]
        raise KeyError(
            f"Gradient record for backward pass {pass_index} not found; participated in "
            f"passes {available}."
        )


def _summarize_call_args(saved_args: Any, non_tensor_pos_args: Any) -> str | None:
    """Build a human-readable summary of an Op's positional arguments.

    Prefers the fully-saved positional args when available; otherwise falls
    back to the captured non-tensor positional args. Returns ``None`` when no
    positional-argument information was captured.
    """

    source = saved_args if saved_args is not None else non_tensor_pos_args
    if source is None:
        return None
    try:
        items = list(source)
    except TypeError:
        return _summarize_value(source)
    if not items:
        return ""
    return ", ".join(_summarize_value(item) for item in items)


def _summarize_call_kwargs(saved_kwargs: Any, non_tensor_kwargs: Any) -> str | None:
    """Build a human-readable summary of an Op's keyword arguments.

    Prefers the fully-saved keyword args when available; otherwise falls back to
    the captured non-tensor keyword args. Returns ``None`` when no
    keyword-argument information was captured.
    """

    source = saved_kwargs if saved_kwargs is not None else non_tensor_kwargs
    if source is None:
        return None
    if isinstance(source, dict):
        pairs = source.items()
    else:
        try:
            pairs = list(source)  # type: ignore[assignment]
        except TypeError:
            return _summarize_value(source)
    rendered = []
    for pair in pairs:
        try:
            key, value = pair
        except (TypeError, ValueError):
            rendered.append(_summarize_value(pair))
            continue
        rendered.append(f"{key}={_summarize_value(value)}")
    if not rendered:
        return ""
    return ", ".join(rendered)


def _resolve_container_type(
    type_module: str | None, type_qualname: str | None
) -> type | str | None:
    """Resolve a stored container type reference to a runtime class.

    Falls back to the qualified name string when the class cannot be imported
    (e.g. after ``.tlspec`` load with the defining module unavailable). Returns
    ``None`` when no type reference was captured.
    """

    if type_qualname is None:
        return None
    if type_module:
        try:
            module = importlib.import_module(type_module)
            obj: Any = module
            for part in type_qualname.split("."):
                obj = getattr(obj, part)
            if isinstance(obj, type):
                return obj
        except (ImportError, AttributeError):
            pass
    if type_module:
        return f"{type_module}.{type_qualname}"
    return type_qualname


def apply_transform(
    *,
    label: str | None,
    tensor: torch.Tensor,
    transform: Callable[..., Any],
    transform_kind: str,
    streaming_active: bool = False,
    raw_label: str | None = None,
    func_name: str | None = None,
) -> Any:
    """Apply a user transform with logging paused and contextual errors.

    Parameters
    ----------
    label:
        Raw layer label for error context, or ``None`` when unavailable.
    tensor:
        Raw tensor passed to the user transform.
    transform:
        Callable applied to ``tensor``.
    transform_kind:
        Transform kind, either ``"out"`` or ``"grad"``.
    streaming_active:
        Whether a streaming bundle writer is active.
    raw_label:
        Raw layer label for error context when it differs from ``label``.
    func_name:
        Function name for error context.

    Returns
    -------
    Any
        Value returned by ``transform``.
    """

    try:
        with pause_logging():
            return transform(tensor)
    except Exception as exc:
        raise TorchLensPostfuncError(
            transform_error_message(
                label=label,
                raw_label=raw_label,
                func_name=func_name,
                tensor=tensor,
                transform_kind=transform_kind,
                streaming_active=streaming_active,
            )
        ) from exc


def transform_error_message(
    *,
    label: str | None,
    raw_label: str | None = None,
    func_name: str | None = None,
    tensor: torch.Tensor,
    transform_kind: str,
    streaming_active: bool,
) -> str:
    """Build context for an out or grad transform failure.

    Parameters
    ----------
    label:
        Raw layer label for error context, or ``None`` when unavailable.
    raw_label:
        Raw layer label for error context when it differs from ``label``.
    func_name:
        Function name for error context.
    tensor:
        Raw tensor passed to the transform.
    transform_kind:
        Transform kind, either ``"out"`` or ``"grad"``.
    streaming_active:
        Whether a streaming bundle writer is active.

    Returns
    -------
    str
        Contextual error message.
    """

    return (
        f"{transform_kind}_transform raised for layer {label} "
        f"(raw={raw_label or label}, func={func_name}, "
        f"shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
        f"streaming_active={streaming_active})."
    )


def validate_train_mode_transform_output(
    *,
    raw_tensor: torch.Tensor,
    transformed_tensor: Any,
    transform_kind: str,
    backward_ready: bool,
    label: str | None = None,
) -> None:
    """Validate differentiability requirements for train-mode transform outputs.

    Parameters
    ----------
    raw_tensor:
        Raw tensor passed to the transform.
    transformed_tensor:
        Value returned by the transform.
    transform_kind:
        Transform kind, either ``"out"`` or ``"grad"``.
    backward_ready:
        Whether TorchLens is preserving autograd graph connectivity.
    label:
        Raw layer label for error context, or ``None`` when unavailable.

    Returns
    -------
    None
        Raises if the transformed value violates train-mode requirements.
    """

    if not backward_ready or not raw_tensor.requires_grad:
        return
    if not isinstance(transformed_tensor, torch.Tensor):
        raise TrainingModeConfigError(
            f"{transform_kind}_transform must return a torch.Tensor while backward_ready=True "
            f"for layer {label}."
        )
    if transformed_tensor.dtype in _NON_GRAD_DTYPES:
        raise TrainingModeConfigError(
            f"backward_ready=True with non-grad dtype {transformed_tensor.dtype} on layer "
            f"{label}. Integer and bool dtypes cannot propagate grads."
        )
    if not transformed_tensor.requires_grad or (
        transformed_tensor.grad_fn is None and transformed_tensor is not raw_tensor
    ):
        raise TrainingModeConfigError(
            f"{transform_kind}_transform returned a tensor disconnected from the autograd "
            "graph (grad_fn is None) while backward_ready=True. The transformed out "
            "must remain differentiable."
        )


def validate_streaming_transform_output(
    *,
    transformed_tensor: Any,
    transform_kind: str,
    streaming_active: bool,
    label: str | None = None,
) -> None:
    """Validate transformed tensors before streaming bundle finalization.

    Parameters
    ----------
    transformed_tensor:
        Value returned by the user transform.
    transform_kind:
        Transform kind, either ``"out"`` or ``"grad"``.
    streaming_active:
        Whether a streaming bundle writer is active for this trace.
    label:
        Raw layer label for error context, or ``None`` when unavailable.

    Returns
    -------
    None
        Raises if streaming cannot serialize the transformed value.
    """

    if not streaming_active:
        return
    if not isinstance(transformed_tensor, torch.Tensor):
        raise TorchLensIOError(
            f"Streaming save requires {transform_kind}_transform outputs to be "
            f"torch.Tensor instances, but layer {label} produced "
            f"{type(transformed_tensor).__name__}."
        )
    if transformed_tensor.layout != torch.strided:
        raise TorchLensIOError(
            f"Streaming save does not support sparse {transform_kind}_transform outputs "
            f"for layer {label}."
        )


def _set_saved_out_metadata(entry: "Op", tensor: torch.Tensor) -> None:
    """Refresh saved output metadata from a replacement tensor.

    Parameters
    ----------
    entry:
        Layer pass whose saved output metadata should match ``tensor``.
    tensor:
        Saved output tensor.

    Returns
    -------
    None
        Metadata fields are updated through internal setters.
    """

    shape = tuple(tensor.shape)
    dtype = tensor.dtype
    entry._internal_set("shape", shape)
    entry._internal_set("dtype", dtype)
    entry._internal_set("dtype_ref", DtypeRef.from_value(dtype))
    entry._internal_set("device_ref", DeviceRef.from_value(tensor.device))
    entry._internal_set(
        "activation_memory",
        Bytes(get_memory_amount_from_metadata(tensor, shape, dtype)),
    )
    entry._internal_set("has_saved_activation", True)
    entry._internal_set("transformed_out_shape", _shape_or_none(entry.transformed_out))
    entry._internal_set("transformed_out_dtype", _dtype_or_none(entry.transformed_out))
    entry._internal_set("transformed_activation_memory", _memory_or_none(entry.transformed_out))


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
    trace: "Trace | None",
    *,
    func_name: str | None,
    is_inplace: bool = False,
) -> SaveMode:
    """Return the effective saved-activation mode for one operation."""

    save_mode = cast(SaveMode, getattr(trace, "save_mode", "copy"))
    if save_mode not in {"copy", "reference", "view", "cpu_async"}:
        raise ValueError("save_mode must be one of 'copy', 'reference', 'view', or 'cpu_async'")
    if save_mode == "reference":
        _warn_reference_save_mode_once()
        if is_inplace or (func_name is not None and func_name.endswith("_")):
            return "copy"
    return save_mode


def _stamp_reference_out(
    annotations: dict[str, Any], raw_out: torch.Tensor, save_mode: SaveMode
) -> None:
    """Store reference-mode mutation metadata for a saved output."""

    if save_mode != "reference":
        return
    annotations["save_mode"] = "reference"
    annotations["saved_out_version"] = getattr(raw_out, "_version", None)


def _validate_reference_out_not_mutated(state: dict[str, Any]) -> None:
    """Raise if a reference-mode saved output has changed since capture."""

    annotations = state.get("annotations") or {}
    if annotations.get("save_mode") != "reference":
        return
    out = state.get("out")
    if not isinstance(out, torch.Tensor):
        return
    saved_version = annotations.get("saved_out_version")
    current_version = getattr(out, "_version", None)
    if saved_version is not None and current_version != saved_version:
        label = state.get("label") or state.get("layer_label") or state.get("_label_raw")
        raise MutatedReferenceError(
            f"saved reference out for op {label!r} was mutated after capture "
            f"(saved _version={saved_version}, current _version={current_version}). "
            "Use save_mode='copy' for isolated saved tensors."
        )


def _tensor_content_hash(value: torch.Tensor) -> str:
    """Return a CPU content hash for a tensor.

    Parameters
    ----------
    value:
        Tensor to hash.

    Returns
    -------
    str
        SHA-256 digest.
    """

    if is_functorch_wrapped_tensor(value):
        return f"functorch_wrapped_tensor:{id(value)}"

    with pause_logging():
        tensor = safe_copy(value, detach_tensor=True).cpu().contiguous()
        if tensor.dtype is torch.bfloat16:
            tensor = tensor.to(torch.float32)
        payload = tensor.numpy().tobytes()
    hasher = hashlib.sha256()
    hasher.update(repr((tuple(tensor.shape), str(tensor.dtype))).encode("utf-8"))
    hasher.update(payload)
    return hasher.hexdigest()


def _dedup_saved_activation_out(
    trace: "Trace | None",
    source_tensor: torch.Tensor,
    raw_out: torch.Tensor,
    label: str,
    annotations: dict[str, Any],
    save_arg_values: bool,
) -> torch.Tensor:
    """Return a deduplicated saved activation payload when configured.

    Parameters
    ----------
    trace:
        Trace that owns the per-pass dedup caches.
    source_tensor:
        Live output tensor before ``safe_copy`` created ``raw_out``.
    raw_out:
        Copied activation payload.
    label:
        Raw layer label for the saved output.
    annotations:
        Mutable annotation dictionary for the saved output.
    save_arg_values:
        Whether argument values are being saved. Argument snapshots consume
        independent payloads, so activation dedup is disabled in that mode.

    Returns
    -------
    torch.Tensor
        Either ``raw_out`` or a previously saved payload for the same live
        source tensor.
    """

    if trace is None or save_arg_values or raw_out.is_meta:
        return raw_out

    mode = getattr(trace, "_out_dedup_mode", "identity")
    if mode == "none":
        return raw_out

    if mode == "content":
        hash_cache = getattr(trace, "_out_hash_cache", None)
        if hash_cache is None:
            hash_cache = {}
            setattr(trace, "_out_hash_cache", hash_cache)
        out_hash = _tensor_content_hash(raw_out)
        if out_hash in hash_cache:
            annotations["dedup_out_hash"] = out_hash
            annotations["dedup_reference_label"] = hash_cache[out_hash][0]
            return hash_cache[out_hash][1]
        hash_cache[out_hash] = (label, raw_out)
        return raw_out

    identity_cache = getattr(trace, "_out_identity_cache", None)
    if identity_cache is None:
        identity_cache = {}
        setattr(trace, "_out_identity_cache", identity_cache)

    source_key = id(source_tensor)
    cached = identity_cache.get(source_key)
    if cached is not None:
        cached_source, cached_label, cached_out = cached
        if cached_source is source_tensor:
            annotations["dedup_source_id"] = source_key
            annotations["dedup_reference_label"] = cached_label
            return cached_out

    identity_cache[source_key] = (source_tensor, label, raw_out)
    return raw_out


if TYPE_CHECKING:
    import pandas as pd

    from .._io.lazy import LazyActivationRef
    from .func_call_location import FuncCallLocation
    from .layer import Layer
    from .layer import OpAccessor
    from .module import Module
    from .param import Param
    from .trace import Trace


class Op:
    """Metadata for a single tensor operation (one pass of one layer).

    Constructed from a dict whose keys must exactly match
    ``LAYER_PASS_LOG_FIELD_ORDER`` (enforced at init time).  Every
    attribute is set explicitly (not via a loop) so that IDE
    autocompletion works.

    Notable design points:

    * ``_tracing_finished`` mirrors the owning Trace's flag. Methods like
      ``__str__`` branch on it to show raw vs final labels.
    * ``source_trace`` is a direct reference to the owning Trace.
      This creates a circular reference (Trace -> layer_list -> entry ->
      source_trace -> Trace) that is broken by ``cleanup()``.
    * ``fx_qualpath`` and ``fx_call_index`` expose metadata that mirrors
      ``torch.fx.symbolic_trace`` naming conventions, computed independently
      using TorchLens rules.  ``fx_qualpath`` is not a lookup key, so
      ``trace[label]`` does not accept it.  Combine
      ``fx_qualpath.replace(".", "_")`` with ``fx_call_index`` when an
      FX-style name form is needed.
    """

    PORTABLE_STATE_SPEC: dict[str, FieldPolicy] = {
        "_label_raw": FieldPolicy.KEEP,
        "_layer_label_raw": FieldPolicy.KEEP,
        "step_index": FieldPolicy.KEEP,
        "raw_index": FieldPolicy.KEEP,
        "ordinal_index": FieldPolicy.KEEP,
        "source_trace": FieldPolicy.DROP,
        "_source_trace_ref": FieldPolicy.WEAKREF_STRIP,
        "_tracing_finished": FieldPolicy.KEEP,
        "_construction_done": FieldPolicy.DROP,
        "_is_in_conditional_body": FieldPolicy.KEEP,
        "label": FieldPolicy.KEEP,
        "label_short": FieldPolicy.KEEP,
        "layer_label": FieldPolicy.KEEP,
        "layer_label_short": FieldPolicy.KEEP,
        "type": FieldPolicy.KEEP,
        "type_index": FieldPolicy.KEEP,
        "pass_index": FieldPolicy.KEEP,
        "num_passes": FieldPolicy.KEEP,
        "lookup_keys": FieldPolicy.KEEP,
        "out": FieldPolicy.BLOB,
        "transformed_out": FieldPolicy.BLOB,
        "has_saved_activation": FieldPolicy.KEEP,
        "output_device": FieldPolicy.KEEP,
        "activation_transform": FieldPolicy.DROP,
        "annotations": FieldPolicy.KEEP,
        "interventions": FieldPolicy.DROP,
        "intervention_replaced": FieldPolicy.KEEP,
        "detach_saved_activations": FieldPolicy.KEEP,
        "has_saved_args": FieldPolicy.KEEP,
        "saved_args": FieldPolicy.BLOB_RECURSIVE,
        "saved_kwargs": FieldPolicy.BLOB_RECURSIVE,
        "args_template": FieldPolicy.DROP,
        "kwargs_template": FieldPolicy.DROP,
        "input_ops": FieldPolicy.DROP,
        "input_activations": FieldPolicy.DROP,
        "input_shapes": FieldPolicy.DROP,
        "input_dtypes": FieldPolicy.DROP,
        "input_memory": FieldPolicy.DROP,
        "num_inputs": FieldPolicy.DROP,
        "shape": FieldPolicy.KEEP,
        "transformed_out_shape": FieldPolicy.KEEP,
        "dtype": FieldPolicy.KEEP,
        "dtype_ref": FieldPolicy.KEEP,
        "transformed_out_dtype": FieldPolicy.KEEP,
        "device_ref": FieldPolicy.KEEP,
        "backend_address": FieldPolicy.KEEP,
        "resolver_status": FieldPolicy.KEEP,
        "activation_memory": FieldPolicy.KEEP,
        "transformed_activation_memory": FieldPolicy.KEEP,
        "visualizer_path": FieldPolicy.KEEP,
        "autograd_memory": FieldPolicy.KEEP,
        "num_autograd_tensors": FieldPolicy.KEEP,
        "bytes_delta_at_call": FieldPolicy.KEEP,
        "bytes_peak_at_call": FieldPolicy.KEEP,
        "has_out_variations": FieldPolicy.KEEP,
        "out_versions_by_child": FieldPolicy.BLOB_RECURSIVE,
        "grad": FieldPolicy.BLOB,
        "transformed_grad": FieldPolicy.BLOB,
        "save_grads": FieldPolicy.KEEP,
        "has_grad": FieldPolicy.KEEP,
        "grad_shape": FieldPolicy.KEEP,
        "transformed_grad_shape": FieldPolicy.KEEP,
        "grad_dtype": FieldPolicy.KEEP,
        "transformed_grad_dtype": FieldPolicy.KEEP,
        "gradient_memory": FieldPolicy.KEEP,
        "transformed_gradient_memory": FieldPolicy.KEEP,
        "func": FieldPolicy.DROP,
        "func_call_id": FieldPolicy.KEEP,
        "func_name": FieldPolicy.KEEP,
        "func_qualname": FieldPolicy.KEEP,
        "code_context": FieldPolicy.KEEP,
        "func_duration": FieldPolicy.KEEP,
        "flops_forward": FieldPolicy.KEEP,
        "flops_backward": FieldPolicy.KEEP,
        "func_rng_states": FieldPolicy.BLOB_RECURSIVE,
        "func_autocast_state": FieldPolicy.KEEP,
        "arg_names": FieldPolicy.KEEP,
        "num_args_total": FieldPolicy.KEEP,
        "num_pos_args": FieldPolicy.KEEP,
        "num_kwargs": FieldPolicy.KEEP,
        "non_tensor_pos_args": FieldPolicy.KEEP,
        "non_tensor_kwargs": FieldPolicy.KEEP,
        "func_non_tensor_args": FieldPolicy.KEEP,
        "is_inplace": FieldPolicy.KEEP,
        "grad_fn_class_name": FieldPolicy.KEEP,
        "grad_fn_class_qualname": FieldPolicy.KEEP,
        "grad_fn_object_id": FieldPolicy.KEEP,
        "grad_fn_handle": FieldPolicy.DROP,
        "grad_fn": FieldPolicy.DROP,
        "in_multi_output": FieldPolicy.KEEP,
        "multi_output_index": FieldPolicy.KEEP,
        "multi_output_name": FieldPolicy.KEEP,
        "container_path": FieldPolicy.KEEP,
        "container_spec": FieldPolicy.KEEP,
        "is_transform": FieldPolicy.KEEP,
        "transform_kind": FieldPolicy.KEEP,
        "transform_chain": FieldPolicy.KEEP,
        "transform_config": FieldPolicy.KEEP,
        "transform_fn_name": FieldPolicy.KEEP,
        "transform_fn_qualname": FieldPolicy.KEEP,
        "transform_fn_source": FieldPolicy.KEEP,
        "unattributed_tensor_args": FieldPolicy.KEEP,
        "parent_params": FieldPolicy.KEEP,
        "_param_barcodes": FieldPolicy.KEEP,
        "parent_param_ops": FieldPolicy.KEEP,
        "_param_logs": FieldPolicy.KEEP,
        "param_shapes": FieldPolicy.KEEP,
        "num_params": FieldPolicy.KEEP,
        "num_params_trainable": FieldPolicy.KEEP,
        "num_params_frozen": FieldPolicy.KEEP,
        "param_memory": FieldPolicy.KEEP,
        "equivalence_class": FieldPolicy.KEEP,
        "equivalent_ops": FieldPolicy.KEEP,
        "recurrent_ops": FieldPolicy.KEEP,
        "parents": FieldPolicy.KEEP,
        "parent_arg_positions": FieldPolicy.KEEP,
        "_edge_uses": FieldPolicy.KEEP,
        "root_ancestors": FieldPolicy.KEEP,
        "children": FieldPolicy.KEEP,
        "has_children": FieldPolicy.KEEP,
        "is_input": FieldPolicy.KEEP,
        "has_input_ancestor": FieldPolicy.KEEP,
        "input_ancestors": FieldPolicy.KEEP,
        "min_distance_from_input": FieldPolicy.KEEP,
        "max_distance_from_input": FieldPolicy.KEEP,
        "is_output": FieldPolicy.KEEP,
        "is_output_parent": FieldPolicy.KEEP,
        "is_final_output": FieldPolicy.KEEP,
        "has_output_descendant": FieldPolicy.KEEP,
        "output_descendants": FieldPolicy.KEEP,
        "is_orphan": FieldPolicy.KEEP,
        "min_distance_to_output": FieldPolicy.KEEP,
        "max_distance_to_output": FieldPolicy.KEEP,
        "io_role": FieldPolicy.KEEP,
        "is_buffer": FieldPolicy.KEEP,
        "address": FieldPolicy.KEEP,
        "buffer_pass": FieldPolicy.KEEP,
        "buffer_source": FieldPolicy.KEEP,
        "buffer_write_kind": FieldPolicy.KEEP,
        "buffer_value_changed": FieldPolicy.KEEP,
        "buffer_replay_validated": FieldPolicy.KEEP,
        "buffer_source_func_name": FieldPolicy.KEEP,
        "is_internal_source": FieldPolicy.KEEP,
        "has_internal_source_ancestor": FieldPolicy.KEEP,
        "internal_source_parents": FieldPolicy.KEEP,
        "internal_source_ancestors": FieldPolicy.KEEP,
        "is_internal_sink": FieldPolicy.KEEP,
        "is_terminal_bool": FieldPolicy.KEEP,
        "is_terminal_conditional_bool": FieldPolicy.KEEP,
        "conditional_context_kind": FieldPolicy.KEEP,
        "conditional_wrapper_kind": FieldPolicy.KEEP,
        "terminal_conditional_id": FieldPolicy.KEEP,
        "is_scalar_bool": FieldPolicy.KEEP,
        "bool_value": FieldPolicy.KEEP,
        "in_conditionals": FieldPolicy.KEEP,
        "terminal_bool_for": FieldPolicy.KEEP,
        "is_in_conditional_body": FieldPolicy.KEEP,
        "conditional_branch_stack": FieldPolicy.KEEP,
        "conditional_branch_depth": FieldPolicy.KEEP,
        "conditional_entry_children": FieldPolicy.KEEP,
        "conditional_then_children": FieldPolicy.KEEP,
        "conditional_elif_children": FieldPolicy.KEEP,
        "conditional_else_children": FieldPolicy.KEEP,
        "conditional_arm_children": FieldPolicy.KEEP,
        "module": FieldPolicy.KEEP,
        "_address_normalized": FieldPolicy.KEEP,
        "modules": FieldPolicy.KEEP,
        "fx_qualpath": FieldPolicy.KEEP,
        "fx_call_index": FieldPolicy.KEEP,
        "module_call_stack": FieldPolicy.KEEP,
        "module_entry_arg_keys": FieldPolicy.KEEP,
        "input_to_module_calls": FieldPolicy.KEEP,
        "output_of_modules": FieldPolicy.KEEP,
        "output_of_module_calls": FieldPolicy.KEEP,
        "is_module_output": FieldPolicy.KEEP,
        "is_atomic_module": FieldPolicy.KEEP,
        "atomic_module_call": FieldPolicy.KEEP,
        "func_config": FieldPolicy.BLOB_RECURSIVE,
        "out_ref": FieldPolicy.DROP,
        "grad_ref": FieldPolicy.DROP,
        "_grad_records": FieldPolicy.BLOB_RECURSIVE,
        "_pending_blob_id": FieldPolicy.DROP,
        "_pending_transformed_out_blob_id": FieldPolicy.DROP,
        "_pending_grad_blob_id": FieldPolicy.DROP,
        "_pending_transformed_grad_blob_id": FieldPolicy.DROP,
    }
    FIELD_FORK_POLICY = LAYER_PASS_LOG_FIELD_FORK_POLICY
    DEFAULT_FILL_STATE = _LAYER_PASS_LOG_DEFAULT_FILL
    _construction_done: bool = False

    def __getattribute__(self, name: str) -> Any:
        """Materialize lazy grads and reject finalized unsaved predicate outs."""

        if name == "grad":
            state = object.__getattribute__(self, "__dict__")
            records = state.get("_grad_records")
            if records:
                saved = [record for record in records if record.grad is not None]
                if len(saved) == 1:
                    grad = saved[0].grad
                    if isinstance(grad, torch.Tensor) and state.get("grad") is None:
                        object.__getattribute__(self, "_internal_set")("grad", grad)
                    return grad
                if len(saved) > 1:
                    label = (
                        state.get("label") or state.get("layer_label") or state.get("_label_raw")
                    )
                    passes = [record.backward_pass_index for record in saved]
                    raise ValueError(
                        f"op {label} has gradients saved from multiple backward passes {passes}; "
                        "use op.grads[...] / op.grad_for(bwd=k)."
                    )
            grad = state.get("grad")
            if grad is None and state.get("grad_ref") is not None:
                return object.__getattribute__(self, "materialize_grad")()
            return grad
        if name == "out":
            state = object.__getattribute__(self, "__dict__")
            out = state.get("out")
            source_ref = state.get("_source_trace_ref")
            source_trace = None if source_ref is None else source_ref()
            if (
                out is None
                and state.get("out_ref") is not None
                and getattr(source_trace, "_predicate_save_options", None) is not None
            ):
                return object.__getattribute__(self, "materialize_out")()
            if (
                state.get("_tracing_finished")
                and not state.get("has_saved_activation", False)
                and getattr(source_trace, "_predicate_save_options", None) is not None
            ):
                label = state.get("label") or state.get("layer_label") or state.get("_label_raw")
                raise ValueError(
                    f"op {label} was not saved; no saved payload is available. "
                    "Re-run with save=... to retain this activation."
                )
            if not getattr(source_trace, "_postprocessing_active", False):
                _validate_reference_out_not_mutated(state)
        return object.__getattribute__(self, name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Mark owning logs dirty when user code directly writes guarded fields.

        Parameters
        ----------
        name:
            Attribute being written.
        value:
            New attribute value.
        """

        construction_done = self.__dict__.get("_construction_done", False)
        if construction_done and name in _DIRECT_WRITE_GUARDED_FIELDS:
            owner = self.__dict__.get("_source_trace_ref")
            trace = owner() if owner is not None else None
            if trace is not None:
                object.__setattr__(trace, "_has_direct_writes", True)
                object.__setattr__(trace, "state", TraceState.DIRECT_WRITE_DIRTY)
                if not getattr(trace, "_warned_direct_write", False):
                    warnings.warn(
                        "DirectActivationWriteWarning: direct Op out writes "
                        "are not recipe edits; replay/rerun propagation will overlay them.",
                        DirectActivationWriteWarning,
                        stacklevel=2,
                    )
                    object.__setattr__(trace, "_warned_direct_write", True)
        object.__setattr__(self, name, value)

    def _internal_set(self, attr: str, value: Any) -> None:
        """Set an attribute without marking the owner dirty.

        Parameters
        ----------
        attr:
            Attribute name to set.
        value:
            Value to assign.
        """

        object.__setattr__(self, attr, value)

    def _append_tensor_from(self, other: "Op", field_name: str) -> None:
        """Append one tensor field from another pass along batch dimension 0.

        Parameters
        ----------
        other:
            New chunk pass with a compatible tensor field.
        field_name:
            Tensor attribute name to concatenate.
        """

        current_value = getattr(self, field_name)
        other_value = getattr(other, field_name)
        if isinstance(current_value, torch.Tensor) and isinstance(other_value, torch.Tensor):
            self._internal_set(field_name, concatenate_batch_tensors(current_value, other_value))

    def __init__(self, fields_dict: Dict[str, Any]) -> None:
        """Initialise from a complete fields dictionary.

        Args:
            fields_dict: Dict with values for all fields defined in
                ``LAYER_PASS_LOG_FIELD_ORDER``.  Missing or extra keys
                raise ``ValueError``.
        """
        # Attributes are set explicitly (not via loop) for IDE autocompletion.
        object.__setattr__(self, "_construction_done", False)

        # Validate that fields_dict has exactly the expected keys:
        if "_address_normalized" not in fields_dict:
            fields_dict["_address_normalized"] = None
        if "fx_qualpath" not in fields_dict:
            fields_dict["fx_qualpath"] = None
        if "fx_call_index" not in fields_dict:
            fields_dict["fx_call_index"] = 0
        if "ordinal_index" not in fields_dict:
            fields_dict["ordinal_index"] = -1
        if "grad_fn" not in fields_dict:
            fields_dict["grad_fn"] = None
        if fields_dict.get("dtype_ref") is None:
            fields_dict["dtype_ref"] = _dtype_ref_or_none(fields_dict.get("dtype"))
        if fields_dict.get("device_ref") is None:
            fields_dict["device_ref"] = _device_ref_from_metadata(
                fields_dict.get("out"), fields_dict.get("output_device")
            )
        if fields_dict.get("backend_address") is None:
            fields_dict["backend_address"] = fields_dict.get("address")
        if fields_dict.get("resolver_status") is None:
            fields_dict["resolver_status"] = "resolved"
        if "save_gradients" in fields_dict and "save_grads" not in fields_dict:
            fields_dict["save_grads"] = fields_dict.pop("save_gradients")
        for derived_field in (
            "input_ops",
            "input_activations",
            "input_shapes",
            "input_dtypes",
            "input_memory",
            "num_inputs",
        ):
            if derived_field not in fields_dict:
                fields_dict[derived_field] = None
        fields_dict_key_set = set(fields_dict.keys())
        if fields_dict_key_set != _LAYER_PASS_LOG_FIELD_ORDER_SET:
            error_str = "Error initializing Op:"
            missing_fields = _LAYER_PASS_LOG_FIELD_ORDER_SET - fields_dict_key_set
            extra_fields = fields_dict_key_set - _LAYER_PASS_LOG_FIELD_ORDER_SET
            if len(missing_fields) > 0:
                error_str += f"\n\t- Missing fields {', '.join(missing_fields)}"
            if len(extra_fields) > 0:
                error_str += f"\n\t- Extra fields {', '.join(extra_fields)}"
            raise ValueError(error_str)

        # General info:
        self._label_raw = fields_dict["_label_raw"]
        self._layer_label_raw = fields_dict["_layer_label_raw"]
        self.step_index = fields_dict["step_index"]
        self.raw_index = fields_dict["raw_index"]
        self.ordinal_index = fields_dict["ordinal_index"]
        # Store as weakref to break circular reference (Trace -> layer_list -> entry -> Trace).
        _sml = fields_dict["source_trace"]
        self._source_trace_ref = weakref.ref(_sml) if _sml is not None else None
        self._tracing_finished = fields_dict["_tracing_finished"]

        # Label info:
        self.layer_label = fields_dict["layer_label"]
        self.layer_label_short = fields_dict["layer_label_short"]
        self.label = fields_dict["label"]
        self.label_short = fields_dict["label_short"]
        self.type = fields_dict["type"]
        self.type_index = fields_dict["type_index"]
        self.pass_index = fields_dict["pass_index"]
        self.num_passes = fields_dict["num_passes"]
        self.lookup_keys = fields_dict["lookup_keys"]

        # Saved tensor info:
        self.out = fields_dict["out"]
        self.transformed_out = fields_dict["transformed_out"]
        self.has_saved_activation = fields_dict["has_saved_activation"]
        self.output_device = fields_dict["output_device"]
        self.activation_transform = fields_dict["activation_transform"]
        self.annotations: Dict[str, Any] = fields_dict["annotations"]
        self.interventions = fields_dict["interventions"]
        self.intervention_replaced = fields_dict["intervention_replaced"]
        self.detach_saved_activations = fields_dict["detach_saved_activations"]
        self.has_saved_args = fields_dict["has_saved_args"]
        self.saved_args = fields_dict["saved_args"]
        self.saved_kwargs = fields_dict["saved_kwargs"]
        self.args_template = fields_dict["args_template"]
        self.kwargs_template = fields_dict["kwargs_template"]
        self.shape = fields_dict["shape"]
        self.transformed_out_shape = fields_dict["transformed_out_shape"]
        self.dtype = fields_dict["dtype"]
        self.dtype_ref: DtypeRef | None = fields_dict["dtype_ref"]
        self.transformed_out_dtype = fields_dict["transformed_out_dtype"]
        self.device_ref: DeviceRef | None = fields_dict["device_ref"]
        self.backend_address: str | None = fields_dict["backend_address"]
        self.resolver_status: str = fields_dict["resolver_status"]
        self.activation_memory: Bytes | None = as_bytes(fields_dict["activation_memory"])
        self.transformed_activation_memory: Bytes | None = as_bytes(
            fields_dict["transformed_activation_memory"]
        )
        self.visualizer_path: str | None = fields_dict["visualizer_path"]
        self.autograd_memory: Bytes | None = as_bytes(fields_dict["autograd_memory"])
        self.num_autograd_tensors: Optional[int] = fields_dict["num_autograd_tensors"]
        self.bytes_delta_at_call: Bytes | None = as_bytes(fields_dict["bytes_delta_at_call"])
        self.bytes_peak_at_call: Bytes | None = as_bytes(fields_dict["bytes_peak_at_call"])

        # Child tensor variation tracking - stores the raw tensor values that
        # each child operation received as input.  Must store RAW values (not
        # postprocessed) because validation compares these against saved_args.
        self.has_out_variations = fields_dict["has_out_variations"]
        self.out_versions_by_child = fields_dict["out_versions_by_child"]

        # Saved grad info - grad is stored as a bare clone (not deep-copied)
        # via log_tensor_grad().  grad is populated by a backward hook.
        self.grad = fields_dict["grad"]
        self.transformed_grad = fields_dict["transformed_grad"]
        self.save_grads = fields_dict["save_grads"]
        self.has_grad = fields_dict["has_grad"]
        self.grad_shape = fields_dict["grad_shape"]
        self.transformed_grad_shape = fields_dict["transformed_grad_shape"]
        self.grad_dtype = fields_dict["grad_dtype"]
        self.transformed_grad_dtype = fields_dict["transformed_grad_dtype"]
        self.gradient_memory: Bytes | None = as_bytes(fields_dict["gradient_memory"])
        self.transformed_gradient_memory: Bytes | None = as_bytes(
            fields_dict["transformed_gradient_memory"]
        )

        # Function call info:
        self.func = fields_dict["func"]
        self.func_call_id = fields_dict["func_call_id"]
        self.func_name = fields_dict["func_name"]
        self.func_qualname = fields_dict["func_qualname"]
        self.code_context: List["FuncCallLocation"] = fields_dict["code_context"]
        self.func_duration = as_duration(fields_dict["func_duration"])
        self.flops_forward = as_flops(fields_dict["flops_forward"])
        self.flops_backward = as_flops(fields_dict["flops_backward"])
        self.func_rng_states = fields_dict["func_rng_states"]
        self.func_autocast_state = fields_dict["func_autocast_state"]
        self.arg_names = fields_dict["arg_names"]
        self.num_args_total = fields_dict["num_args_total"]
        self.num_pos_args = fields_dict["num_pos_args"]
        self.num_kwargs = fields_dict["num_kwargs"]
        self.non_tensor_pos_args = fields_dict["non_tensor_pos_args"]
        self.non_tensor_kwargs = fields_dict["non_tensor_kwargs"]
        self.func_non_tensor_args = fields_dict["func_non_tensor_args"]
        self.is_inplace = fields_dict["is_inplace"]
        self.grad_fn_class_name = fields_dict["grad_fn_class_name"]
        self.grad_fn_class_qualname = fields_dict["grad_fn_class_qualname"]
        self.grad_fn_object_id = fields_dict["grad_fn_object_id"]
        self.grad_fn_handle = fields_dict["grad_fn_handle"]
        self.grad_fn = fields_dict["grad_fn"]
        self.in_multi_output = fields_dict["in_multi_output"]
        self.multi_output_index = fields_dict["multi_output_index"]
        self.multi_output_name = fields_dict["multi_output_name"]
        self.container_path = fields_dict["container_path"]
        self.container_spec = fields_dict["container_spec"]
        self.is_transform = fields_dict["is_transform"]
        self.transform_kind = fields_dict["transform_kind"]
        self.transform_chain = fields_dict["transform_chain"]
        self.transform_config = fields_dict["transform_config"]
        self.transform_fn_name = fields_dict["transform_fn_name"]
        self.transform_fn_qualname = fields_dict["transform_fn_qualname"]
        self.transform_fn_source = fields_dict["transform_fn_source"]
        self.unattributed_tensor_args = fields_dict["unattributed_tensor_args"]

        # Param info:
        self.parent_params = fields_dict["parent_params"]
        self._param_barcodes = fields_dict["_param_barcodes"]
        self.parent_param_ops = fields_dict["parent_param_ops"]
        self._param_logs: List["Param"] = fields_dict["_param_logs"]
        self.param_shapes = fields_dict["param_shapes"]
        self.num_params = fields_dict["num_params"]
        self.num_params_trainable = fields_dict["num_params_trainable"]
        self.num_params_frozen = fields_dict["num_params_frozen"]
        self.param_memory: Bytes = Bytes(fields_dict["param_memory"] or 0)

        # Loop-detection equivalence info:
        # equivalence_class groups structurally identical operations
        # (same func + same param barcodes).  equivalent_ops holds a
        # DIRECT reference to the Trace-level set for this type.
        # recurrent_ops is populated by loop_detection.py for layers
        # that are different ops of the same recurrent layer.
        self.equivalence_class = fields_dict["equivalence_class"]
        self.equivalent_ops = fields_dict["equivalent_ops"]
        self.recurrent_ops = fields_dict["recurrent_ops"]

        # Graph info:
        self.parents = fields_dict["parents"]
        self.parent_arg_positions = fields_dict["parent_arg_positions"]
        self._edge_uses = fields_dict["_edge_uses"]
        self.root_ancestors = fields_dict["root_ancestors"]
        self.children = fields_dict["children"]
        self.has_children = fields_dict["has_children"]
        self.is_input = fields_dict["is_input"]
        self.has_input_ancestor = fields_dict["has_input_ancestor"]
        self.input_ancestors = fields_dict["input_ancestors"]
        self.min_distance_from_input = fields_dict["min_distance_from_input"]
        self.max_distance_from_input = fields_dict["max_distance_from_input"]
        self.is_output = fields_dict["is_output"]
        self.is_output_parent = fields_dict["is_output_parent"]
        self.is_final_output = fields_dict["is_final_output"]
        self.has_output_descendant = fields_dict["has_output_descendant"]
        self.output_descendants = fields_dict["output_descendants"]
        self.is_orphan: bool = fields_dict["is_orphan"]
        self.min_distance_to_output = fields_dict["min_distance_to_output"]
        self.max_distance_to_output = fields_dict["max_distance_to_output"]
        self.io_role = fields_dict["io_role"]
        self.is_buffer = fields_dict["is_buffer"]
        self.address = fields_dict["address"]
        self.buffer_pass = fields_dict["buffer_pass"]
        self.buffer_source = fields_dict["buffer_source"]
        self.buffer_write_kind = fields_dict["buffer_write_kind"]
        self.buffer_value_changed = fields_dict["buffer_value_changed"]
        self.buffer_replay_validated = fields_dict["buffer_replay_validated"]
        self.buffer_source_func_name = fields_dict["buffer_source_func_name"]
        self.is_internal_source = fields_dict["is_internal_source"]
        self.has_internal_source_ancestor = fields_dict["has_internal_source_ancestor"]
        self.internal_source_parents = fields_dict["internal_source_parents"]
        self.internal_source_ancestors = fields_dict["internal_source_ancestors"]
        self.is_internal_sink = fields_dict["is_internal_sink"]

        # Conditional info
        self.is_terminal_bool = fields_dict["is_terminal_bool"]
        self.is_terminal_conditional_bool = fields_dict["is_terminal_conditional_bool"]
        self.conditional_context_kind = fields_dict["conditional_context_kind"]
        self.conditional_wrapper_kind = fields_dict["conditional_wrapper_kind"]
        self.terminal_conditional_id = fields_dict["terminal_conditional_id"]
        self.is_scalar_bool = fields_dict["is_scalar_bool"]
        self.bool_value = fields_dict["bool_value"]
        self.in_conditionals = fields_dict["in_conditionals"]
        self.terminal_bool_for = fields_dict["terminal_bool_for"]
        self.is_in_conditional_body = fields_dict["is_in_conditional_body"]
        self.conditional_branch_stack = fields_dict["conditional_branch_stack"]
        self.conditional_branch_depth = fields_dict["conditional_branch_depth"]
        self.conditional_entry_children = fields_dict["conditional_entry_children"]
        self.conditional_then_children = fields_dict["conditional_then_children"]
        self.conditional_elif_children = fields_dict["conditional_elif_children"]
        self.conditional_else_children = fields_dict["conditional_else_children"]
        self.conditional_arm_children = fields_dict["conditional_arm_children"]

        # Module info
        self.module = fields_dict["module"]
        self._address_normalized = fields_dict["_address_normalized"]
        self.modules = fields_dict["modules"]
        self.fx_qualpath: Optional[str] = fields_dict["fx_qualpath"]
        self.fx_call_index: int = fields_dict["fx_call_index"]
        self.module_call_stack = fields_dict["module_call_stack"]
        self.module_entry_arg_keys = fields_dict["module_entry_arg_keys"]
        self.input_to_module_calls = fields_dict["input_to_module_calls"]
        self.output_of_modules = fields_dict["output_of_modules"]
        self.output_of_module_calls = fields_dict["output_of_module_calls"]
        self.is_module_output = fields_dict["is_module_output"]
        self.is_atomic_module = fields_dict["is_atomic_module"]
        self.atomic_module_call = fields_dict["atomic_module_call"]

        # Function config - lightweight hyperparameters always captured.
        self.func_config = fields_dict["func_config"]

        self.out_ref: Optional["LazyActivationRef"] = None
        self.grad_ref: Optional["LazyActivationRef"] = None
        self._pending_blob_id: Optional[str] = None
        self._pending_transformed_out_blob_id: Optional[str] = None
        self._pending_grad_blob_id: Optional[str] = None
        self._pending_transformed_grad_blob_id: Optional[str] = None
        self._grad_records: list[GradientRecord] = []
        object.__setattr__(self, "_construction_done", True)

    @property
    def layer_type(self) -> str:
        """Return the operation type token used by existing internal callers."""

        return cast(str, self.type)

    @layer_type.setter
    def layer_type(self, value: str) -> None:
        """Set the operation type token through the legacy internal name."""

        self.type = value

    @property
    def macs_forward(self) -> Optional[Macs]:
        """Forward MACs (multiply-accumulate ops). 1 MAC = 2 FLOPs."""
        return as_macs(self.flops_forward // 2 if self.flops_forward is not None else None)

    @property
    def macs_backward(self) -> Optional[Macs]:
        """Backward MACs (multiply-accumulate ops). 1 MAC = 2 FLOPs."""
        return as_macs(self.flops_backward // 2 if self.flops_backward is not None else None)

    @property
    def flops_total(self) -> Flops:
        """Approximate total FLOPs for this Op.

        Returns
        -------
        Flops
            Forward plus backward FLOPs, treating unknown halves as zero.
        """

        return Flops((self.flops_forward or 0) + (self.flops_backward or 0))

    @property
    def macs_total(self) -> Macs:
        """Approximate total MACs for this Op.

        Returns
        -------
        Macs
            Forward plus backward MACs.
        """

        return Macs(self.flops_total // 2)

    @property
    def param_names(self) -> list[str]:
        """Return short names of parameters consumed by this Op.

        Returns
        -------
        list[str]
            Parameter names in consumed-parameter order.
        """

        return [param.name for param in self._param_logs]

    @property
    def param_dtypes(self) -> list[torch.dtype]:
        """Return dtypes of parameters consumed by this Op.

        Returns
        -------
        list[torch.dtype]
            Parameter dtypes in consumed-parameter order.
        """

        return [param.dtype for param in self._param_logs]

    @property
    def num_param_tensors_trainable(self) -> int:
        """Number of trainable parameter tensors consumed by this Op."""

        return sum(1 for param in self._param_logs if param.is_trainable)

    @property
    def num_param_tensors_frozen(self) -> int:
        """Number of frozen parameter tensors consumed by this Op."""

        return sum(1 for param in self._param_logs if not param.is_trainable)

    @property
    def has_trainable_params(self) -> bool:
        """Whether this Op consumes at least one trainable parameter."""

        return self.num_params_trainable > 0

    @property
    def has_frozen_params(self) -> bool:
        """Whether this Op consumes at least one frozen parameter."""

        return self.num_params_frozen > 0

    @property
    def is_compute_op(self) -> bool:
        """Whether this Op executed a torch function rather than a boundary sentinel."""

        return not (self.is_input or self.is_output or self.is_buffer)

    @property
    def fx_label(self) -> str | None:
        """Return a torch.fx-style label for this Op when module context is available."""

        if self.fx_qualpath is None:
            return None
        return f"{self.fx_qualpath.replace('.', '_')}_{self.fx_call_index}"

    @property
    def trace(self) -> "Trace":
        """Alias for the owning Trace back-reference."""

        return self.source_trace

    @property
    def grad_fn_cls(self) -> type[Any] | None:
        """Return the live grad_fn_handle class when the autograd object is retained.

        Returns
        -------
        type[Any] | None
            Runtime grad_fn_handle class, or ``None`` when the object is unavailable.
        """

        grad_fn_handle = self.grad_fn_handle
        return None if grad_fn_handle is None else type(grad_fn_handle)

    @property
    def grads(self) -> GradientRecordAccessor:
        """Per-pass gradient records saved for this Op."""

        trace = self.source_trace
        if getattr(trace, "backend", "torch") == "jax":
            raise ValueError(
                "JAX traces do not expose op.grads or saved_grad_ops because they do "
                "not capture true backward graphs. Use trace.derived_grads for "
                "leaf-level derived gradients."
            )
        return GradientRecordAccessor(self.__dict__.setdefault("_grad_records", []))

    def grad_for(self, *, bwd: int) -> torch.Tensor:
        """Return the saved gradient tensor for one backward pass.

        Parameters
        ----------
        bwd:
            One-based backward pass number.

        Returns
        -------
        torch.Tensor
            Saved raw gradient tensor for the requested pass.
        """

        record = self.grads.for_pass(bwd)
        if record.grad is None:
            raise ValueError(
                f"op {self.label} has no saved gradient payload for backward pass {bwd}."
            )
        return record.grad

    def _record_gradient(
        self,
        *,
        backward_pass_index: int,
        grad: torch.Tensor | None,
        transformed_grad: Any | None,
        shape: tuple[int, ...] | None,
        dtype: str | None,
        memory: int | None,
        timestamp: float,
    ) -> GradientRecord:
        """Append or replace the projected gradient record for one pass.

        Parameters
        ----------
        backward_pass_index:
            One-based global backward pass number.
        grad:
            Saved raw gradient tensor, if retained.
        transformed_grad:
            Saved transformed gradient tensor, if retained.
        shape:
            Observed raw gradient shape.
        dtype:
            Observed raw gradient dtype string.
        memory:
            Observed raw gradient memory in bytes.
        timestamp:
            Event timestamp.

        Returns
        -------
        GradientRecord
            Appended record.
        """

        records = self.__dict__.setdefault("_grad_records", [])
        records[:] = [
            record for record in records if record.backward_pass_index != backward_pass_index
        ]
        record = GradientRecord(
            owner=self,
            ordinal=len(records) + 1,
            backward_pass_index=backward_pass_index,
            grad=grad,
            transformed_grad=transformed_grad,
            shape=shape,
            dtype=dtype,
            memory=memory,
            timestamp=timestamp,
        )
        records.append(record)
        return record

    @property
    def grad_fn_label(self) -> str | None:
        """Stable GradFn label for this Op, or ``None`` when no GradFn was captured.

        Stored form of the cross-class reference: ``grad_fn`` resolves the
        TorchLens GradFn record, while ``grad_fn_label`` is the portable label.
        """

        grad_fn = self.grad_fn
        if grad_fn is None:
            return None
        return cast("Optional[str]", getattr(grad_fn, "label", None))

    @property
    def layer(self) -> "Layer":
        """Parent Layer record resolved via ``self.trace.layers[self.layer_label]``.

        Raises
        ------
        KeyError
            When the parent Layer label cannot be resolved on the owning Trace.
        """

        trace = self._source_trace_or_error()
        return cast("Layer", trace.layers[self.layer_label])

    @property
    def args_summary(self) -> str | None:
        """Human-readable summary of this Op's positional arguments.

        The Op-level source for ``Layer.args_summary``. Returns ``None`` when no
        positional-argument information was captured.
        """

        return _summarize_call_args(self.saved_args, self.non_tensor_pos_args)

    @property
    def kwargs_summary(self) -> str | None:
        """Human-readable summary of this Op's keyword arguments.

        Op-level source for ``Layer.kwargs_summary``. Returns ``None`` when no
        keyword-argument information was captured.
        """

        return _summarize_call_kwargs(self.saved_kwargs, self.non_tensor_kwargs)

    @property
    def multi_output_type(self) -> type | str | None:
        """Python class of the multi-output container this Op came from.

        Resolves ``container_spec.type_module``/``type_qualname`` to a runtime
        class when importable, falling back to the qualified name string when the
        class cannot be imported (e.g. after ``.tlspec`` load). ``None`` when this
        Op is not from a multi-output container.
        """

        if not self.in_multi_output:
            return None
        spec = self.container_spec
        if spec is None:
            return None
        return _resolve_container_type(
            getattr(spec, "type_module", None),
            getattr(spec, "type_qualname", None),
        )

    @property
    def is_module_input(self) -> bool:
        """Whether this Op's output feeds into at least one ModuleCall as an input.

        The Op itself is OUTSIDE the module (the upstream producer). Equivalent to
        ``bool(self.input_to_module_calls)``. Direction-of-data-flow framing:
        inputs come FROM outside the module.
        """

        return bool(self.input_to_module_calls)

    @property
    def atomic_module_call_label(self) -> str | None:
        """Stable ModuleCall label for an atomic-module output Op, else ``None``.

        Stored form of the cross-class reference; ``atomic_module_call`` resolves
        the ModuleCall record.
        """

        return cast("Optional[str]", self.atomic_module_call)

    @property
    def atomic_module_address(self) -> str | None:
        """Module address (no ``:N``) for an atomic-module output Op, else ``None``.

        Derived from ``atomic_module_call_label`` by stripping the ``:N`` pass
        suffix; pairs with the ``atomic_module`` resolver.
        """

        label = self.atomic_module_call_label
        if label is None:
            return None
        return label.rsplit(":", 1)[0]

    @property
    def atomic_module(self) -> "Module | None":
        """Module record resolved from ``atomic_module_address``, or ``None``.

        Returns ``None`` when this Op is not an atomic-module output or when the
        owning Trace is unavailable.
        """

        address = self.atomic_module_address
        if address is None:
            return None
        trace = self.source_trace
        if trace is None:
            return None
        try:
            return cast("Module", trace.modules[address])
        except (KeyError, TypeError):
            return None

    @property
    def has_parents(self) -> bool:
        """Whether this layer has any parent layers."""
        return len(self.parents) > 0

    @property
    def num_parents(self) -> int:
        """Number of distinct parent Ops feeding this Op."""

        return len(self.parents)

    @property
    def num_children(self) -> int:
        """Number of distinct child Ops fed by this Op."""

        return len(self.children)

    @property
    def siblings(self) -> list[str]:
        """Layers sharing at least one parent (excluding output layers)."""
        ml = self.source_trace
        if ml is None:
            return []
        my_label = self.layer_label if self._tracing_finished else self._label_raw
        siblings = []
        seen = {my_label}
        for parent_label in self.parents:
            parent = ml[parent_label]
            for child_label in parent.children:
                if child_label not in seen:
                    seen.add(child_label)
                    child = ml[child_label]
                    if not child.is_output:
                        siblings.append(child_label)
        return siblings

    @property
    def has_siblings(self) -> bool:
        """Whether this layer shares parents with other layers."""
        return len(self.siblings) > 0

    @property
    def co_parents(self) -> list[str]:
        """Layers sharing at least one child (excluding output layers)."""
        ml = self.source_trace
        if ml is None:
            return []
        my_label = self.layer_label if self._tracing_finished else self._label_raw
        spouses = []
        seen = {my_label}
        for child_label in self.children:
            child = ml[child_label]
            for parent_label in child.parents:
                if parent_label not in seen:
                    seen.add(parent_label)
                    parent = ml[parent_label]
                    if not parent.is_output:
                        spouses.append(parent_label)
        return spouses

    @property
    def has_co_parents(self) -> bool:
        """Whether this layer shares children with other layers."""
        return len(self.co_parents) > 0

    @property
    def is_in_conditional(self) -> bool:
        """Whether this op participates in any conditional role."""

        return bool(self.in_conditionals)

    @property
    def is_in_conditional_evaluation(self) -> bool:
        """Whether this op computes a conditional arm condition."""

        return any(role.role == "evaluation" for role in self.in_conditionals or [])

    @property
    def is_in_conditional_body(self) -> bool:
        """Whether this op is in a conditional arm body."""

        if self.has_output_descendant and not self.conditional_entry_children:
            return False
        return bool(self.__dict__.get("_is_in_conditional_body", False)) or any(
            role.role == "body" for role in self.in_conditionals or []
        )

    @is_in_conditional_body.setter
    def is_in_conditional_body(self, value: bool) -> None:
        """Set the cached conditional-body predicate used during postprocessing."""

        self.__dict__["_is_in_conditional_body"] = value

    @is_in_conditional_body.deleter
    def is_in_conditional_body(self) -> None:
        """Delete the cached conditional-body predicate during cleanup."""

        self.__dict__.pop("_is_in_conditional_body", None)

    @property
    def conditional_depth(self) -> int:
        """Number of distinct conditionals this op participates in."""

        return len({role.conditional_id for role in self.in_conditionals or []})

    @property
    def uses_params(self) -> bool:
        """Whether this operation used model parameters."""
        return len(self._param_barcodes) > 0

    @property
    def num_param_tensors(self) -> int:
        """Number of parameter tensors used by this operation."""
        return len(self._param_barcodes)

    @property
    def input_ops(self) -> "OpAccessor":
        """Accessor over graph-parent Op records in ``parents`` order."""

        from .layer import OpAccessor

        trace = self._source_trace_or_error()
        parent_ops = {}
        for parent_index, parent_label in enumerate(self.parents, start=1):
            try:
                parent_ops[parent_index] = trace.ops[parent_label]
            except KeyError:
                if parent_label in trace.layer_dict_all_keys:
                    parent_ops[parent_index] = trace.layer_dict_all_keys[parent_label]
        return OpAccessor(parent_ops)

    @input_ops.deleter
    def input_ops(self) -> None:
        """Ignore cleanup deletion for derived input Op access."""

    @property
    def input_activations(self) -> tuple[torch.Tensor | None, ...]:
        """Saved parent activations consumed by this Op, as references.

        Returned tensors are not copied. Mutating them mutates TorchLens saved
        state. For in-place-modified parents, the per-child version consumed by
        this Op is returned when available.
        """

        trace = self._source_trace_or_error()
        activations: list[torch.Tensor | None] = []
        child_labels = (
            self.layer_label,
            self.label,
            self._label_raw,
            self._layer_label_raw,
        )
        for parent_label in self.parents:
            parent: Op | None
            try:
                parent = trace.ops[parent_label]
            except KeyError:
                parent = cast("Op | None", trace.layer_dict_all_keys.get(parent_label))
            if parent is None:
                activations.append(None)
                continue
            if not parent.has_saved_activation:
                activations.append(None)
                continue
            child_versions = getattr(parent, "out_versions_by_child", {}) or {}
            consumed = next(
                (child_versions[label] for label in child_labels if label in child_versions),
                parent.out,
            )
            activations.append(consumed if isinstance(consumed, torch.Tensor) else None)
        return tuple(activations)

    @input_activations.deleter
    def input_activations(self) -> None:
        """Ignore cleanup deletion for derived input activations."""

    @property
    def input_shapes(self) -> tuple[Any | None, ...]:
        """Shapes of saved parent activations in ``parents`` order."""

        return tuple(
            None if activation is None else activation.shape
            for activation in self.input_activations
        )

    @input_shapes.deleter
    def input_shapes(self) -> None:
        """Ignore cleanup deletion for derived input shapes."""

    @property
    def input_dtypes(self) -> tuple[torch.dtype | None, ...]:
        """Dtypes of saved parent activations in ``parents`` order."""

        return tuple(
            None if activation is None else activation.dtype
            for activation in self.input_activations
        )

    @input_dtypes.deleter
    def input_dtypes(self) -> None:
        """Ignore cleanup deletion for derived input dtypes."""

    @property
    def input_memory(self) -> Bytes:
        """Sum of activation bytes across saved graph-parent Ops."""

        return Bytes(
            sum(
                int(getattr(parent, "activation_memory", 0) or 0)
                for parent in self.input_ops.values()
                if parent.has_saved_activation
            )
        )

    @input_memory.deleter
    def input_memory(self) -> None:
        """Ignore cleanup deletion for derived input memory."""

    @property
    def num_inputs(self) -> int:
        """Number of graph-parent input Ops."""

        return len(self.parents)

    @num_inputs.deleter
    def num_inputs(self) -> None:
        """Ignore cleanup deletion for derived input count."""

    @property
    def in_submodule(self) -> bool:
        """Whether this operation was computed inside a submodule."""
        return self.module is not None

    @property
    def module_call_depth(self) -> int:
        """Depth of module nesting for this operation."""
        return len(self.modules)

    @property
    def input_to_modules(self) -> list[str]:
        """Module addresses (no ``:N``) whose input this Op's output fed.

        Derived from ``input_to_module_calls`` by stripping the ``:N`` pass suffix
        and de-duplicating, mirroring the ``output_of_modules`` /
        ``output_of_module_calls`` split. Use ``input_to_module_calls`` for the
        ModuleCall-label list.
        """

        seen: dict[str, None] = {}
        for call_label in self.input_to_module_calls:
            address = str(call_label).rsplit(":", 1)[0]
            seen.setdefault(address, None)
        return list(seen)

    @property
    def gradient_transform(self) -> Callable[..., Any] | None:
        """Transform used for this Op's saved gradient, or ``None`` when unset.

        Mirrors ``activation_transform`` for the backward side. Reads the
        trace-level ``grad_transform`` that was applied to this Op's gradient.
        """

        trace = self.source_trace
        if trace is None:
            return None
        return cast("Optional[Callable[..., Any]]", getattr(trace, "grad_transform", None))

    @property
    def is_buffer_source(self) -> bool:
        """Whether this Op represents a buffer boundary (overwrites a buffer).

        Glossary name for the stored ``is_buffer`` flag.
        """

        return bool(self.is_buffer)

    @property
    def has_saved_gradient(self) -> bool:
        """Whether this Op's gradient was saved.

        Glossary name for the backward-side saved predicate; mirrors
        ``has_saved_activation``.
        """

        return bool(self.has_grad)

    @property
    def tensor(self) -> Any:
        """Alias for the raw saved out."""

        return self.out

    @property
    def ops(self) -> tuple["Op", ...]:
        """Tuple containing this pass for aggregate-compatible iteration.

        Returns
        -------
        tuple[Op, ...]
            Single-entry tuple containing this pass log.
        """

        return (self,)

    @property
    def _streaming_label(self) -> str:
        """Best available label for sink/writer callbacks during or after postprocess.

        Returns
        -------
        str
            Pass-qualified label when available, otherwise the current layer label.
        """

        for candidate in (
            self.label,
            self.layer_label,
            self._layer_label_raw,
            self._label_raw,
        ):
            if candidate is not None:
                return str(candidate)
        return "<unknown>"

    @property
    def source_trace(self) -> "Trace":
        """Back-reference to the owning Trace (stored as weakref)."""
        ref = self.__dict__.get("_source_trace_ref")
        if ref is None:
            return None  # type: ignore[return-value]
        obj = ref()
        return cast("Trace", obj)

    @source_trace.setter
    def source_trace(self, value: "Trace | None") -> None:
        """Set the owning Trace back-reference.

        Parameters
        ----------
        value:
            Owning model log, or ``None`` to clear the reference.
        """
        self._source_trace_ref = weakref.ref(value) if value is not None else None

    def _source_trace_or_error(self) -> "Trace":
        """Return the owning Trace, or raise a detached-log error.

        Returns
        -------
        Trace
            Source Trace that owns this operation log.

        Raises
        ------
        AttributeError
            If this operation log is detached from its source Trace.
        """

        ref = self.__dict__.get("_source_trace_ref")
        source = ref() if ref is not None else None
        if source is None or getattr(source, "_loaded_from_bundle", False):
            raise AttributeError(
                "This Op is detached from its source Trace "
                "(perhaps loaded from disk or after cleanup). "
                "Use trace.do(label, transform) directly."
            )
        return cast("Trace", source)

    def do(
        self,
        transform: Any,
        *,
        model: Any = None,
        x: Any = None,
        engine: Any = MISSING,
        confirm_mutation: Any = MISSING,
        strict: Any = MISSING,
        intervention: Any = None,
    ) -> "Trace":
        """Apply an intervention to this op through the owning Trace.

        Parameters
        ----------
        transform:
            Transform or hook to apply to this operation's output.
        model:
            Model required when ``engine="rerun"``.
        x:
            Input required when ``engine="rerun"``.
        engine:
            ``"auto"``, ``"replay"``, ``"rerun"``, or ``"set_only"``.
        confirm_mutation:
            Suppress root mutation warnings when intentionally mutating.
        strict:
            Whether selector and propagation checks should raise.
        intervention:
            Grouped intervention options.

        Returns
        -------
        Trace
            Source Trace after applying the intervention.
        """

        return self._source_trace_or_error().do(
            self.layer_label,
            transform,
            model=model,
            x=x,
            engine=engine,
            confirm_mutation=confirm_mutation,
            strict=strict,
            intervention=intervention,
        )

    def set(
        self,
        value: Any,
        *,
        strict: bool = False,
        confirm_mutation: bool = False,
    ) -> "Trace":
        """Set this op's out recipe through the owning Trace.

        Parameters
        ----------
        value:
            Static replacement value or one-shot callable.
        strict:
            Whether site resolution should reject non-portable selectors.
        confirm_mutation:
            Suppress root mutation warnings when intentionally mutating.

        Returns
        -------
        Trace
            Source Trace with a stale intervention recipe.
        """

        return self._source_trace_or_error().set(
            self.layer_label,
            value,
            strict=strict,
            confirm_mutation=confirm_mutation,
        )

    def attach_hooks(
        self,
        hook: Any = None,
        *extra_hooks: Any,
        strict: bool = False,
        prepend: bool = False,
        confirm_mutation: bool = False,
    ) -> Any:
        """Attach sticky hooks to this op through the owning Trace.

        Parameters
        ----------
        hook:
            Hook or helper to attach to this operation.
        *extra_hooks:
            Additional hooks to compose on this operation in left-to-right order.
        strict:
            Whether site resolution should reject non-portable selectors.
        prepend:
            Whether new sticky hooks should run before existing sticky hooks.
        confirm_mutation:
            Suppress root mutation warnings when intentionally mutating.

        Returns
        -------
        Any
            Trace or scoped removable hook handle, matching ``Trace.attach_hooks``.
        """

        return self._source_trace_or_error().attach_hooks(
            self.layer_label,
            hook,
            *extra_hooks,
            strict=strict,
            prepend=prepend,
            confirm_mutation=confirm_mutation,
        )

    def materialize_out(
        self,
        *,
        map_location: str | torch.device = "cpu",
    ) -> torch.Tensor:
        """Materialize this layer's saved out from a lazy bundle ref.

        Parameters
        ----------
        map_location:
            Target device for the materialized tensor.

        Returns
        -------
        torch.Tensor
            Materialized out tensor.

        Raises
        ------
        TorchLensIOError
            If no out ref is available for this layer.

        Examples
        --------
        >>> import torchlens as tl
        >>> trace = tl.load("demo_bundle", lazy=True)
        >>> tensor = trace["linear_1_1"].materialize_out()
        >>> tensor.shape
        torch.Size([2, 3])
        """

        current_out = self.__dict__.get("out")
        if isinstance(current_out, torch.Tensor):
            return current_out
        if self.out_ref is None:
            raise TorchLensIOError("no out_ref to materialize from")
        self._internal_set("out", self.out_ref.materialize(map_location=map_location))
        return cast(torch.Tensor, self.out)

    def materialize_grad(
        self,
        *,
        map_location: str | torch.device = "cpu",
    ) -> torch.Tensor:
        """Materialize this layer's saved grad from a lazy bundle ref.

        Parameters
        ----------
        map_location:
            Target device for the materialized tensor.

        Returns
        -------
        torch.Tensor
            Materialized grad tensor.

        Raises
        ------
        TorchLensIOError
            If no grad ref is available for this layer.

        Examples
        --------
        >>> import torchlens as tl
        >>> trace = tl.load("demo_bundle", lazy=True)
        >>> grad = trace["linear_1_1"].materialize_grad()
        >>> grad.shape
        torch.Size([2, 3])
        """

        grad = self.__dict__.get("grad")
        if isinstance(grad, torch.Tensor):
            return grad
        if self.grad_ref is None:
            raise TorchLensIOError("no grad_ref to materialize from")
        self._internal_set("grad", self.grad_ref.materialize(map_location=map_location))
        return cast(torch.Tensor, self.grad)

    def __getstate__(self) -> Dict[str, Any]:
        """Return pickle state with weakrefs stripped."""
        state = self.__dict__.copy()
        state["_source_trace_ref"] = None
        state.pop("_facets_cache", None)
        state["func"] = None
        state["grad_fn_handle"] = None
        state["grad_fn_handle"] = None
        state["tlspec_version"] = TLSPEC_VERSION
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore pickle state produced by ``__getstate__``."""
        version = read_tlspec_version(state, cls_name=type(self).__name__)
        legacy_thread_keys = (
            "_module_boundary_thread_output",
            "_module_boundary_threads_inputs",
            "module_entry_exit_threads_inputs",
        )
        dropped = False
        for key in legacy_thread_keys:
            if key in state:
                state.pop(key)
                dropped = True
        if dropped and version < 3:
            from .._io import _warn_legacy_thread_fields_dropped

            _warn_legacy_thread_fields_dropped()
        old_key_map = {
            "tensor_label_raw": "_label_raw",
            "operation_num": "op_num",
            "activation": "out",
            "transformed_activation": "transformed_out",
            "has_saved_activation": "has_saved_activation",
            "activation_transform": "activation_transform",
            "activation_shape": "shape",
            "transformed_activation_shape": "transformed_out_shape",
            "activation_dtype": "dtype",
            "transformed_activation_dtype": "transformed_out_dtype",
            "memory": "activation_memory",
            "activation_memory": "activation_memory",
            "transformed_activation_memory": "transformed_activation_memory",
            "in_multi_output": "in_multi_output",
            "iterable_output_index": "multi_output_index",
            "grad_fn_object": "grad_fn_handle",
            "corresponding_grad_fn": "grad_fn_handle",
            "is_input_layer": "is_input",
            "is_output_layer": "is_output",
            "is_output_ancestor": "has_output_descendant",
            "is_buffer_layer": "is_buffer",
            "internally_initialized": "is_internal_source",
            "internally_terminated": "is_internal_sink",
            "parent_param_barcodes": "_param_barcodes",
            "module_passes_entered": "input_to_module_calls",
            "input_to_modules": "input_to_module_calls",
            "modules_exited": "output_of_modules",
            "module_passes_exited": "output_of_module_calls",
            "is_leaf_module_output": "is_atomic_module",
            "leaf_module_pass": "atomic_module_call",
            "activation_ref": "out_ref",
            "gradient_ref": "grad_ref",
            "edge_uses": "_edge_uses",
            "min_distance_from_output": "min_distance_to_output",
            "max_distance_from_output": "max_distance_to_output",
        }
        for old_key, new_key in old_key_map.items():
            if new_key not in state and old_key in state:
                state[new_key] = state.pop(old_key)
        default_fill_state(
            state,
            defaults=self.DEFAULT_FILL_STATE,
        )
        if state.get("dtype_ref") is None:
            state["dtype_ref"] = _dtype_ref_or_none(state.get("dtype"))
        if state.get("device_ref") is None:
            state["device_ref"] = _device_ref_from_metadata(
                state.get("out"), state.get("output_device")
            )
        if state.get("backend_address") is None:
            state["backend_address"] = state.get("address")
        if state.get("resolver_status") is None:
            state["resolver_status"] = "resolved"
        for field_name in (
            "activation_memory",
            "transformed_activation_memory",
            "autograd_memory",
            "gradient_memory",
            "transformed_gradient_memory",
            "param_memory",
            "bytes_delta_at_call",
            "bytes_peak_at_call",
        ):
            if state.get(field_name) is not None:
                state[field_name] = Bytes(state[field_name])
        if state.get("func_duration") is not None:
            state["func_duration"] = as_duration(state["func_duration"])
        for field_name in ("flops_forward", "flops_backward"):
            if state.get(field_name) is not None:
                state[field_name] = Flops(state[field_name])
        object.__setattr__(self, "_construction_done", False)
        state.pop("source_trace", None)
        self.__dict__.update(state)
        object.__setattr__(self, "_construction_done", bool(state.get("_construction_done", True)))

    # ********************************************
    # *********** User-Facing Functions **********
    # ********************************************

    @property
    def facets(self) -> Any:
        """Return the lazy semantic facet view for this Op."""

        cache = self.__dict__.get("_facets_cache")
        if cache is None:
            from ..semantic import FacetView

            cache = FacetView(self)
            self.__dict__["_facets_cache"] = cache
        return cache

    @facets.deleter
    def facets(self) -> None:
        """Drop the cached semantic facet view for this Op."""

        self.__dict__.pop("_facets_cache", None)

    # ********************************************
    # ************* Logging Functions ************
    # ********************************************

    def copy(self) -> "Op":
        """Return a selective-depth copy of this entry.

        Most fields are ``copy.deepcopy``'d so the clone is fully independent.
        However, certain fields are shallow-copied (shared by reference) because:

        * ``func``, ``grad_fn_class_name`` - function objects, immutable/shared.
        * ``source_trace`` - must point to the same Trace instance.
        * ``func_rng_states`` - large state dicts, not mutated after capture.
        * ``saved_args``, ``saved_kwargs`` - may contain large tensors;
          deep-copying them is expensive and unnecessary.
        * ``parent_params`` - references to nn.Parameters, must stay shared.
        * ``out``, ``out_versions_by_child`` - large tensors;
          shared references are safe since they're replaced (not mutated).

        Returns:
            A new Op (or subclass) with the same field values.
        """
        fields_dict = {}
        fields_not_to_deepcopy = [
            "func",
            "grad_fn_class_name",
            "grad_fn_handle",
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
        ]
        for field in LAYER_PASS_LOG_FIELD_ORDER:
            if field not in fields_not_to_deepcopy:
                fields_dict[field] = copy.deepcopy(getattr(self, field, None))
            else:
                fields_dict[field] = getattr(self, field, None)
        copied_entry = type(self)(fields_dict)
        return copied_entry

    def save_activation(
        self,
        t: torch.Tensor,
        t_args: Union[List[Any], Tuple[Any, ...]],
        t_kwargs: Dict[str, Any],
        save_arg_values: bool,
        activation_transform: Optional[Callable[..., Any]] = None,
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
            t: The output tensor of the operation.
            t_args: Positional arguments passed to the operation.
            t_kwargs: Keyword arguments passed to the operation.
            save_arg_values: Whether to deep-copy and store args/kwargs.
            activation_transform: Optional transform applied to the tensor
                before storing (e.g. detach, to-numpy, normalize).
        """
        trace = self.source_trace
        writer = getattr(trace, "_out_writer", None) if trace is not None else None
        try:
            save_mode = _effective_activation_save_mode(
                trace,
                func_name=self.func_name,
                is_inplace=bool(self.is_inplace),
            )
            # Clone the tensor, optionally detaching from autograd graph.
            raw_out = safe_copy(
                t,
                self.detach_saved_activations,
                save_mode=save_mode,
            )
            # Move to the user-requested output device if needed.
            if self.output_device not in [str(raw_out.device), "same"]:
                raw_out = safe_to(raw_out, self.output_device)
            _stamp_reference_out(self.annotations, raw_out, save_mode)

            self.shape = tuple(raw_out.shape)
            self.dtype = raw_out.dtype
            self.activation_memory = Bytes(
                get_memory_amount_from_metadata(raw_out, self.shape, self.dtype)
            )

            save_raw_activations = getattr(trace, "save_raw_activations", True)
            store_raw = save_raw_activations or activation_transform is None
            if store_raw:
                raw_out = _dedup_saved_activation_out(
                    trace,
                    t,
                    raw_out,
                    self._layer_label_raw,
                    self.annotations,
                    save_arg_values,
                )
            self._internal_set("out", raw_out if store_raw else None)

            self._internal_set("transformed_out", None)
            self.transformed_out_shape = None
            self.transformed_out_dtype = None
            self.transformed_activation_memory = None
            if activation_transform is not None:
                self._internal_set(
                    "transformed_out",
                    self._apply_transform(
                        raw_out,
                        activation_transform,
                        transform_kind="activation",
                        streaming_active=writer is not None,
                    ),
                )
                self._validate_train_mode_transform_output(
                    raw_out,
                    self.transformed_out,
                    transform_kind="activation",
                )
                self._validate_streaming_transform_output(
                    self.transformed_out,
                    transform_kind="activation",
                    streaming_active=writer is not None,
                )
                self.transformed_out_shape = _shape_or_none(self.transformed_out)
                self.transformed_out_dtype = _dtype_or_none(self.transformed_out)
                self.transformed_activation_memory = _memory_or_none(self.transformed_out)
        except Exception as exc:
            if writer is not None:
                writer.abort(f"Failed while saving out for {self._streaming_label}: {exc}")
                if isinstance(exc, TorchLensPostfuncError):
                    raise
                raise TorchLensIOError(
                    f"Streaming out save failed for {self._streaming_label}."
                ) from exc
            raise

        self.has_saved_activation = True

        if trace is not None:
            out_sink = getattr(trace, "_out_sink", None)
            if out_sink is not None and isinstance(self.out, torch.Tensor):
                out_sink(self._streaming_label, self.out)

            if writer is not None and getattr(trace, "_in_exhaustive_pass", False):
                self._stream_tensor_blob(
                    writer,
                    tensor_field="out",
                    pending_field="_pending_blob_id",
                    kind="out",
                )
                self._stream_tensor_blob(
                    writer,
                    tensor_field="transformed_out",
                    pending_field="_pending_transformed_out_blob_id",
                    kind="transformed_out",
                )

        # Tensor args and kwargs:
        if save_arg_values:
            self.has_saved_args = True
            self._internal_set("saved_args", [_recursive_safe_copy(arg) for arg in t_args])
            self._internal_set(
                "saved_kwargs",
                {k: _recursive_safe_copy(v) for k, v in t_kwargs.items()},
            )
        else:
            self._internal_set("saved_args", None)
            self._internal_set("saved_kwargs", None)

    def log_tensor_grad(self, grad: torch.Tensor) -> None:
        """Save the grad tensor for this layer's output.

        Called by the backward hook registered during the forward pass.
        The grad is ``detach().clone()``'d - a bare copy, not deep-copied -
        so it's independent of the autograd graph but cheap to store.

        Args:
            grad: The grad tensor flowing back through this operation.
        """
        trace = self.source_trace
        raw_grad = grad
        self.grad_shape = tuple(raw_grad.shape)
        self.grad_dtype = raw_grad.dtype
        self.gradient_memory = Bytes(get_memory_amount(raw_grad))
        grad_transform = getattr(trace, "grad_transform", None)
        self._internal_set("transformed_grad", None)
        self.transformed_grad_shape = None
        self.transformed_grad_dtype = None
        self.transformed_gradient_memory = None
        writer = getattr(trace, "_out_writer", None) if trace is not None else None
        if grad_transform is not None:
            self._internal_set(
                "transformed_grad",
                self._apply_transform(
                    raw_grad,
                    grad_transform,
                    transform_kind="grad",
                    streaming_active=writer is not None,
                ),
            )
            self._validate_train_mode_transform_output(
                raw_grad,
                self.transformed_grad,
                transform_kind="grad",
            )
            self._validate_streaming_transform_output(
                self.transformed_grad,
                transform_kind="grad",
                streaming_active=writer is not None,
            )
            self.transformed_grad_shape = _shape_or_none(self.transformed_grad)
            self.transformed_grad_dtype = _dtype_or_none(self.transformed_grad)
            self.transformed_gradient_memory = _memory_or_none(self.transformed_grad)

        save_raw_gradients = getattr(trace, "save_raw_gradients", True)
        store_raw = save_raw_gradients or grad_transform is None
        save_mode = cast(SaveMode, getattr(trace, "save_mode", "copy"))
        self._internal_set(
            "grad",
            safe_copy(raw_grad, detach_tensor=True, save_mode=save_mode) if store_raw else None,
        )
        self.has_grad = True
        if writer is not None and getattr(trace, "_defer_streaming_bundle_finalization", False):
            self._stream_tensor_blob(
                writer,
                tensor_field="grad",
                pending_field="_pending_grad_blob_id",
                kind="grad",
            )
            self._stream_tensor_blob(
                writer,
                tensor_field="transformed_grad",
                pending_field="_pending_transformed_grad_blob_id",
                kind="transformed_grad",
            )

    def _apply_transform(
        self,
        tensor: torch.Tensor,
        transform: Callable[..., Any],
        *,
        transform_kind: str,
        streaming_active: bool,
    ) -> Any:
        """Apply a user transform with logging paused and rich error context."""

        return apply_transform(
            label=self._streaming_label,
            raw_label=self._layer_label_raw,
            func_name=self.func_name,
            tensor=tensor,
            transform=transform,
            transform_kind=transform_kind,
            streaming_active=streaming_active,
        )

    def _transform_error_message(
        self,
        *,
        transform_kind: str,
        tensor: torch.Tensor,
        streaming_active: bool,
    ) -> str:
        """Build context for an out or grad transform failure."""

        return transform_error_message(
            label=self._streaming_label,
            raw_label=self._layer_label_raw,
            func_name=self.func_name,
            tensor=tensor,
            transform_kind=transform_kind,
            streaming_active=streaming_active,
        )

    def _validate_train_mode_transform_output(
        self,
        raw_tensor: torch.Tensor,
        output: Any,
        *,
        transform_kind: str,
    ) -> None:
        """Validate differentiability requirements for train-mode transform outputs."""

        trace = self.source_trace
        validate_train_mode_transform_output(
            raw_tensor=raw_tensor,
            transformed_tensor=output,
            transform_kind=transform_kind,
            backward_ready=getattr(trace, "backward_ready", False),
            label=self._streaming_label,
        )

    def _validate_streaming_transform_output(
        self,
        output: Any,
        *,
        transform_kind: str,
        streaming_active: bool,
    ) -> None:
        """Validate transformed tensors before streaming bundle finalization.

        Parameters
        ----------
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
        TorchLensIOError
            If streaming is active and the transform returns a non-tensor or sparse tensor.
        """

        try:
            validate_streaming_transform_output(
                transformed_tensor=output,
                transform_kind=transform_kind,
                streaming_active=streaming_active,
                label=self._streaming_label,
            )
        except TorchLensIOError as exc:
            self._abort_streaming_writer(str(exc))
            raise

    def _abort_streaming_writer(self, message: str) -> None:
        """Abort the active streaming writer when one is attached.

        Parameters
        ----------
        message:
            Reason written to the partial bundle marker.

        Returns
        -------
        None
            Mutates the writer state if present.
        """

        trace = self.source_trace
        writer = getattr(trace, "_out_writer", None) if trace is not None else None
        if writer is not None:
            writer.abort(message)

    def _stream_tensor_blob(
        self,
        writer: Any,
        *,
        tensor_field: str,
        pending_field: str,
        kind: str,
    ) -> None:
        """Stream one tensor field when present."""

        tensor = getattr(self, tensor_field)
        if tensor is None:
            return
        if not isinstance(tensor, torch.Tensor):
            if kind == "transformed_out":
                message = (
                    "Streaming save requires activation_transform outputs to be torch.Tensor "
                    f"instances, but layer {self._streaming_label} produced "
                    f"{type(tensor).__name__}."
                )
            elif kind == "transformed_grad":
                message = (
                    "Streaming save requires grad_transform outputs to be torch.Tensor "
                    f"instances, but layer {self._streaming_label} produced "
                    f"{type(tensor).__name__}."
                )
            else:
                message = (
                    f"{tensor_field} expected a tensor for streaming, got {type(tensor).__name__}."
                )
            writer.abort(message)
            raise TorchLensIOError(message)
        blob_id = writer.next_blob_id()
        setattr(self, pending_field, blob_id)
        writer.write_blob(
            blob_id,
            tensor,
            kind=kind,
            label=self._streaming_label,
        )

    # ********************************************
    # ************* Fetcher Functions ************
    # ********************************************

    def get_children(self) -> list["Op"]:
        """Return child Op objects for this pass.

        Returns
        -------
        list[Op]
            Child ops resolved through the owning model log.
        """
        return [self.source_trace[child_label] for child_label in self.children]

    def get_parents(self) -> list["Op"]:
        """Return parent Op objects for this pass.

        Returns
        -------
        list[Op]
            Parent ops resolved through the owning model log.
        """
        return [self.source_trace[parent_label] for parent_label in self.parents]

    def show(
        self,
        method: Literal["auto", "heatmap", "channels", "rgb", "hist"] = "auto",
        **kwargs: Any,
    ) -> Any:
        """Display this pass's saved out.

        Parameters
        ----------
        method:
            Display method. ``"auto"`` chooses from tensor shape.
        **kwargs:
            Forwarded to the tensor display helper.

        Returns
        -------
        Any
            Matplotlib figure when plotting is available, otherwise a text
            fallback explaining why no plot was produced.
        """

        from ..viz._tensor_display import show_tensor

        return show_tensor(self, method=method, **kwargs)

    @property
    def params(self) -> Any:
        """Access parameter metadata by address, short name, or index."""
        from .param import ParamAccessor

        param_dict = {pl.address: pl for pl in self._param_logs}
        return ParamAccessor(param_dict)

    def to_pandas(self) -> "pd.DataFrame":
        """Export this Op as a one-row pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            One-row DataFrame ordered by ``LAYER_PASS_LOG_FIELD_ORDER``.
        """

        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "pandas is required for this feature. Install with `pip install torchlens[tabular]`."
            ) from e

        row = {field_name: getattr(self, field_name) for field_name in LAYER_PASS_LOG_FIELD_ORDER}
        return pd.DataFrame([row], columns=LAYER_PASS_LOG_FIELD_ORDER)

    # ********************************************
    # ************* Built-in Methods *************
    # ********************************************

    def __str__(self) -> str:
        if self._tracing_finished:
            return self._str_after_pass()
        else:
            return self._str_during_pass()

    def _str_during_pass(self) -> str:
        """Return a human-readable summary of this tensor entry while the forward pass is still in progress."""
        s = f"Tensor {self._label_raw} (layer {self._layer_label_raw}) (PASS NOT FINISHED):"
        s += f"\n\tPass: {self.pass_index}"
        s += f"\n\tTensor info: shape {self.shape}, dtype {self.dtype}"
        s += f"\n\tComputed from params: {self.uses_params}"
        s += f"\n\tComputed in modules: {self.modules}"
        s += f"\n\tOutput of modules: {self.output_of_module_calls}"
        if self.is_atomic_module:
            s += " (bottom-level submodule output)"
        else:
            s += " (not bottom-level submodule output)"
        s += "\n\tFamily info:"
        s += f"\n\t\tParents: {self.parents}"
        s += f"\n\t\tChildren: {self.children}"
        s += f"\n\t\tSpouses: {self.co_parents}"
        s += f"\n\t\tSiblings: {self.siblings}"
        s += (
            f"\n\t\tOriginal Ancestors: {self.root_ancestors} "
            f"(min dist {self.min_distance_from_input} nodes, max dist {self.max_distance_from_input} nodes)"
        )
        s += f"\n\t\tInput Ancestors: {self.input_ancestors}"
        s += f"\n\t\tInternal Ancestors: {self.internal_source_ancestors}"
        s += (
            f"\n\t\tOutput Descendents: {self.output_descendants} "
            f"(min dist {self.min_distance_to_output} nodes, max dist {self.max_distance_to_output} nodes)"
        )
        if self.out is not None:
            s += f"\n\tTensor contents: \n{print_override(self.out, '__str__')}"
        return s

    def _str_after_pass(self) -> str:
        """Return a human-readable summary of this tensor entry after the forward pass has completed."""
        if self.num_passes > 1:
            pass_str = f" (pass {self.pass_index}/{self.num_passes}), "
        else:
            pass_str = ", "
        sml = self.source_trace
        num_ops = sml.num_ops if sml is not None else "?"
        s = f"Layer {self.layer_label}{pass_str}operation {self.step_index}/{num_ops}:"
        s += f"\n\tOutput tensor: shape={self.shape}, dype={self.dtype}, size={self.activation_memory}"
        if not self.has_saved_activation:
            s += " (not saved)"
        s += self._tensor_contents_str_helper()
        s += self._tensor_family_str_helper()
        if len(self.param_shapes) > 0:
            params_shapes_str = ", ".join(str(param_shape) for param_shape in self.param_shapes)
            s += (
                f"\n\tParams: Computed from params with shape {params_shapes_str}; "
                f"{self.num_params} params total ({self.param_memory})"
            )
        else:
            s += "\n\tParams: no params used"
        if self.module is None:
            module_str = "\n\tComputed inside module: not computed inside a module"
        else:
            module_str = f"\n\tComputed inside module: {self.module}"
        if not self.is_input:
            s += f"\n\tFunction: {self.func_name} (grad_fn_handle: {self.grad_fn_class_name}) {module_str}"
            if self.func_config:
                config_str = ", ".join(f"{k}={v}" for k, v in self.func_config.items())
                s += f"\n\tConfig: {config_str}"
            s += f"\n\tTime elapsed: {self.func_duration: .3E}s"
        if len(self.output_of_modules) > 0:
            output_of_modules_str = ", ".join(self.output_of_modules)
            s += f"\n\tOutput of modules: {output_of_modules_str}"
        else:
            s += "\n\tOutput of modules: none"
        if self.is_atomic_module:
            s += f"\n\tOutput of bottom-level module: {self.atomic_module_call}"
        lookup_keys_str = ", ".join([str(key) for key in self.lookup_keys])
        s += f"\n\tLookup keys: {lookup_keys_str}"

        return s

    def _tensor_contents_str_helper(self) -> str:
        """Returns short, readable string for the tensor contents."""
        if self.out is None:
            return ""
        else:
            s = ""
            s += f"\n\t\t{tensor_stats_summary(self.out)}"
            tensor_size_shown = 8
            # Use logged shape, not live tensor shape (#45)
            saved_shape = self.shape if self.shape is not None else self.out.shape
            # Slice first, then clone only the small slice (#73)
            if len(saved_shape) == 0:
                tensor_slice = self.out.detach().clone()
            elif len(saved_shape) == 1:
                num_dims = min(tensor_size_shown, saved_shape[0])
                tensor_slice = self.out[0:num_dims].detach().clone()
            elif len(saved_shape) == 2:
                num_dims = min(tensor_size_shown, saved_shape[-2], saved_shape[-1])
                tensor_slice = self.out[0:num_dims, 0:num_dims].detach().clone()
            else:
                num_dims = min(tensor_size_shown, saved_shape[-2], saved_shape[-1])
                tensor_slice = self.out.data
                for _ in range(len(saved_shape) - 2):
                    tensor_slice = tensor_slice[0]
                tensor_slice = tensor_slice[0:num_dims, 0:num_dims].detach().clone()
            tensor_slice.requires_grad = False
            s += f"\n\t\t{str(tensor_slice)}"
            if (len(saved_shape) > 0) and (max(saved_shape) > tensor_size_shown):
                s += "..."
        return s

    def _tensor_family_str_helper(self) -> str:
        """Return a formatted string summarising parent, child, sibling, spouse, and ancestor relationships."""
        s = "\n\tRelated Layers:"
        if len(self.parents) > 0:
            s += "\n\t\t- parent layers: " + ", ".join(self.parents)
        else:
            s += "\n\t\t- no parent layers"

        if len(self.children) > 0:
            s += "\n\t\t- child layers: " + ", ".join(self.children)
        else:
            s += "\n\t\t- no child layers"

        if len(self.siblings) > 0:
            s += "\n\t\t- shares parents with layers: " + ", ".join(self.siblings)
        else:
            s += "\n\t\t- shares parents with no other layers"

        if len(self.co_parents) > 0:
            s += "\n\t\t- shares children with layers: " + ", ".join(self.co_parents)
        else:
            s += "\n\t\t- shares children with no other layers"

        if self.has_input_ancestor:
            s += "\n\t\t- descendent of input layers: " + ", ".join(self.input_ancestors)
        else:
            s += "\n\t\t- tensor was created de novo inside the model (not computed from input)"

        if self.has_output_descendant:
            s += "\n\t\t- ancestor of output layers: " + ", ".join(self.output_descendants)
        else:
            s += "\n\t\t- tensor is not an ancestor of the model output; it terminates within the model"

        return s

    def __repr__(self) -> str:
        return self.__str__()


# Backward-compatible alias: TensorLog was the original name for
# Op before the Layer aggregate class was introduced in PR #92.
TensorLog = Op
