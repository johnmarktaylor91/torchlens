"""Unified predicate context for capture filtering and selector adapters.

Also the capture decisions a predicate returns and the records a capture
retains (``CaptureSpec``, ``ActivationRecord``, ``GradRecord`` and kin),
published from ``torchlens.recording.types``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from typing_extensions import Self

from .._errors import InvalidArgumentError
from .refs import DeviceRef, DtypeRef

if TYPE_CHECKING:
    import torch

    from .._utils.tensor_utils import SaveMode

EventKind = Literal["op", "module_enter", "module_exit", "input", "buffer"]
_DEFERRED_VALUE_FIELDS = frozenset({"tensor_requires_grad", "is_scalar_bool", "bool_value"})


class MLXValueUnavailableError(RuntimeError):
    """Raised when MLX lazy evaluation makes a predicate value unavailable.

    Examples
    --------
    >>> issubclass(tl.errors.MLXValueUnavailableError, RuntimeError)
    True
    >>> try:
    ...     raise tl.errors.MLXValueUnavailableError("needs mx.eval")
    ... except RuntimeError as e:
    ...     print(type(e).__name__)
    MLXValueUnavailableError"""


class _DeferredValue:
    """Sentinel for MLX value-dependent fields that would require ``mx.eval``.

    MLX guarantees shape, dtype, and device at call time. It does not guarantee
    value-dependent predicate fields without forcing lazy evaluation, so
    ``tensor_requires_grad``, ``is_scalar_bool``, and ``bool_value`` may carry
    this sentinel under MLX. User predicates raise on use; internal projections
    must coerce it to ``None`` before storing metadata.
    """

    __slots__ = ()

    def __bool__(self) -> bool:
        """Raise because the deferred value is not available in-flight."""

        raise MLXValueUnavailableError(_deferred_value_message())

    def __eq__(self, other: object) -> bool:
        """Raise because equality would consume a deferred value."""

        raise MLXValueUnavailableError(_deferred_value_message())

    def __lt__(self, other: object) -> bool:
        """Raise because ordering would consume a deferred value."""

        raise MLXValueUnavailableError(_deferred_value_message())

    def __le__(self, other: object) -> bool:
        """Raise because ordering would consume a deferred value."""

        raise MLXValueUnavailableError(_deferred_value_message())

    def __gt__(self, other: object) -> bool:
        """Raise because ordering would consume a deferred value."""

        raise MLXValueUnavailableError(_deferred_value_message())

    def __ge__(self, other: object) -> bool:
        """Raise because ordering would consume a deferred value."""

        raise MLXValueUnavailableError(_deferred_value_message())

    def __hash__(self) -> int:
        """Raise because hashing would consume a deferred value."""

        raise MLXValueUnavailableError(_deferred_value_message())

    def __deepcopy__(self, memo: dict[int, object]) -> Self:
        """Return this singleton during dataclass serialization."""

        return self

    def __reduce__(self) -> str:
        """Return the module-level singleton name for pickle round-trips."""

        return "_DEFERRED_VALUE"

    def __repr__(self) -> str:
        """Return a diagnostic representation that does not consume the value."""

        return "_DEFERRED_VALUE"


def _deferred_value_message() -> str:
    """Return the standard MLX deferred-value error message."""

    return (
        "MLX lazy evaluation cannot provide this value-dependent RecordContext field "
        "during in-flight predicate evaluation without forcing mx.eval. Use shape, dtype, "
        "or tensor_device fields, or run this value-dependent predicate on the PyTorch backend."
    )


_DEFERRED_VALUE = _DeferredValue()


def is_deferred_value(value: object) -> bool:
    """Return whether ``value`` is the MLX deferred-value sentinel."""

    return value is _DEFERRED_VALUE


def coerce_deferred_value(value: Any) -> Any:
    """Return ``None`` for the MLX deferred-value sentinel."""

    return None if is_deferred_value(value) else value


@dataclass(frozen=True, slots=True)
class ModuleStackFrame:
    """One frame in the active module stack.

    Examples
    --------
    >>> seen = []
    >>> recording = tl.record(model, x, save=lambda ctx: seen.append(ctx) or False)
    >>> frame = next(c for c in seen if c.kind == "op").module_stack[-1]
    >>> isinstance(frame, ModuleStackFrame)
    True
    >>> frame.address, frame.module_type, frame.pass_number
    ('0', 'Conv2d', 1)"""

    address: str
    module_type: str
    module_id: int
    pass_number: int


@dataclass(frozen=True, slots=True)
class RecordContext:
    """Predicate input schema for one chronological capture event.

    Attributes
    ----------
    kind
        Event category: operation, module entry/exit, input, or buffer.
    label
        In-flight capture label passed to the current predicate invocation. Final
        postprocessed labels do not exist yet: a normal PyTorch op uses a raw
        spelling such as ``"relu_1_2_raw"``. A compatibility retry for a
        non-match may instead use the prefix alias ``"relu_1"``; predicates must
        therefore not assume this field contains the later final label
        ``"relu_1_2"``.
    raw_label
        Authoritative full raw capture label, such as ``"relu_1_2_raw"``, or
        ``None`` when the backend cannot supply one.
    pass_index
        One-based forward-pass index within a multi-pass recording.
    event_index
        Chronological event index within the active capture.
    step_index
        Chronological operation index, or ``None`` for events outside the op stream.
    layer_type
        Normalized TorchLens operation or source type.
    type_index
        One-based occurrence index within ``layer_type``.
    raw_index
        One-based operation index used by raw graph labels.
    func_name
        Backend function name, when the event represents a function call.
    address
        Nearest active module address.
    module_type
        Nearest active module class name.
    module_pass_number
        Call index for the nearest active module.
    module_stack
        Active module frames from outermost to innermost.
    recent_events
        Bounded chronological lookback including configured source events.
    recent_ops
        Operation-only view of the bounded lookback.
    parent_labels
        Parent labels exposed by the active predicate adapter.
    input_output_address
        Structural address for an input or output boundary event.
    shape
        Output tensor shape, when tensor metadata is observable.
    dtype
        Backend-neutral output dtype reference.
    tensor_device
        Backend-neutral output device reference.
    tensor_requires_grad
        Whether the output requires gradients. MLX may provide a deferred-value
        sentinel that raises when consumed because resolving it would force evaluation.
    output_index
        Position of this tensor within a multi-output operation.
    is_bottom_level_func
        Whether the event came from a leaf decorated function call.
    time_since_pass_start
        Elapsed wall-clock seconds since capture began.
    sample_id
        Optional identifier supplied by a batched predicate caller.
    label_raw
        Non-optional compatibility spelling of ``raw_label``; it contains the
        same full raw label or ``""`` when unavailable. It is never the final label.
    label_prefix
        Short raw compatibility alias, for example ``"relu_1"``.
    func_call_id
        Stable identifier for the decorated function invocation.
    parent_labels_raw
        Full raw parent labels when the backend records them separately.
    is_output_parent
        Whether this event directly parents a model output.
    backend_requires_isolation
        Whether predicate evaluation must be isolated for backend safety.
    is_scalar_bool
        Whether the output is a scalar boolean. MLX may defer this value.
    bool_value
        Scalar boolean value when safely observable. MLX may defer this value.
    is_transform
        Whether this event is a captured transform boundary.
    transform_kind
        Backend-neutral transform name when ``is_transform`` is true.
    window_miss
        Whether a history-dependent selector exceeded the retained lookback window.
    output_of_module_calls
        Labels of the module calls whose output this event is. Empty for events
        that are not a module-call output; a provisional op carries ``()`` until
        the module-exit hook resolves its membership.

    Examples
    --------
    >>> seen = []
    >>> def spy(ctx):
    ...     seen.append(ctx)
    ...     return False
    >>> recording = tl.record(model, x, save=spy)
    >>> ctx = next(c for c in seen if c.kind == "op")
    >>> isinstance(ctx, tl.recording.RecordContext)
    True
    >>> ctx.layer_type, ctx.shape, ctx.pass_number
    ('conv2d', (1, 4, 8, 8), 1)
    """

    kind: EventKind | str
    label: str
    raw_label: str | None
    pass_number: int
    event_index: int
    step_index: int | None
    layer_type: str | None
    type_index: int | None
    raw_index: int | None
    func_name: str | None
    address: str | None
    module_type: str | None
    module_pass_number: int | None
    module_stack: tuple[Any, ...]
    recent_events: tuple[RecordContext, ...]
    recent_ops: tuple[RecordContext, ...]
    parent_labels: tuple[str, ...]
    input_output_address: str | None
    shape: tuple[int, ...] | None
    dtype: DtypeRef | None
    tensor_device: DeviceRef | None
    tensor_requires_grad: bool | None | _DeferredValue
    output_index: int | None
    is_bottom_level_func: bool | None
    time_since_pass_start: float
    sample_id: str | int | None = None
    label_raw: str = ""
    label_prefix: str = ""
    func_call_id: int | None = None
    parent_labels_raw: tuple[str, ...] = ()
    is_output_parent: bool = False
    backend_requires_isolation: bool = False
    is_scalar_bool: bool | None | _DeferredValue = None
    bool_value: bool | None | _DeferredValue = None
    is_transform: bool = False
    transform_kind: str | None = None
    window_miss: bool = False
    output_of_module_calls: tuple[str, ...] = ()

    def __getattr__(self, name: str) -> Any:
        """Raise a schema-specific error for unknown predicate fields."""

        from ..recording.exceptions import RecordContextFieldError  # deferred: vocabulary closure

        raise RecordContextFieldError(name)

    def __repr__(self) -> str:
        """Return a bounded one-line context summary.

        The dataclass auto-repr embedded ``recent_events``/``recent_ops`` --
        each itself a ``RecordContext`` carrying its own lookback -- so one
        repr grew ~34x per saved site and OOMed an 8 GiB box at four toy-MLP
        sites (lovely bug 1, CRITICAL). Lookback tuples render as COUNTS;
        recursion is structurally impossible here.
        """

        kind = getattr(self.kind, "value", self.kind)
        shape = "?" if self.shape is None else "(" + ",".join(str(d) for d in self.shape) + ")"
        return (
            f"RecordContext(kind={kind!s}, label={self.label!r}, "
            f"pass={self.pass_number}, event={self.event_index}, shape={shape}, "
            f"recent_events={len(self.recent_events)}, recent_ops={len(self.recent_ops)})"
        )


@dataclass(frozen=True, slots=True)
class RetroactiveCaptureDecision:
    """Decision that saves already-emitted candidates from a lookback window.

    Parameters
    ----------
    target_raw_labels:
        Raw labels for candidate ops to mark as saved.
    spec:
        Capture spec to apply to each target.
    reason:
        Diagnostic reason for the retroactive decision.
    """

    target_raw_labels: tuple[str, ...]
    spec: Any
    reason: str = "followed_by"


def _distinct_label_index_keys(label: str, raw_label: str | None) -> tuple[str, ...]:
    """Return the distinct label keys (primary, then raw alias) that index one record."""
    if raw_label is None or raw_label == label:
        return (label,)
    return (label, raw_label)


@dataclass(frozen=True, slots=True)
class CaptureSpec:
    """Capture policy returned by predicate callbacks.

    Parameters
    ----------
    save_out:
        Whether tensor payloads should be retained for this event.
    save_metadata:
        Whether non-payload metadata should be retained for this event.
    save_grad:
        Whether the in-RAM tensor clone should stay attached to autograd.
    device:
        Optional target device for retained payloads.
    dtype:
        Optional target dtype for retained payloads.
    save_mode:
        Tensor retention mode for saved payloads.

    Examples
    --------
    >>> spec = CaptureSpec(save_grad=True)
    >>> spec.save_out, spec.save_grad, spec.save_mode
    (True, True, 'copy')

    """

    save_out: bool = True
    save_metadata: bool = True
    save_grad: bool = False
    device: torch.device | str | None = None
    dtype: torch.dtype | None = None
    save_mode: SaveMode = "copy"

    def __post_init__(self) -> None:
        """Normalize and validate capture save-mode settings."""

        from .._utils.tensor_utils import SAVE_MODES  # deferred: vocabulary closure

        if self.save_mode not in SAVE_MODES:
            raise InvalidArgumentError(
                "save_mode must be one of 'copy', 'reference', 'view', or 'cpu_async'; "
                f"received {self.save_mode!r}",
                code="save_mode_invalid",
                remedy="set save_mode to 'copy', 'reference', 'view', or 'cpu_async'",
                argument="save_mode",
            )
        if self.save_mode == "view" and not self.save_grad:
            object.__setattr__(self, "save_grad", True)


CaptureDecision = bool | CaptureSpec | None
PredicateDecision = CaptureDecision | RetroactiveCaptureDecision


@dataclass(frozen=True, slots=True)
class StorageIntent:
    """Resolved storage destinations for a capture decision."""

    in_ram: bool
    on_disk: bool


@dataclass(frozen=True, slots=True)
class ActivationRecord:
    """One retained fastlog event.

    Parameters
    ----------
    ctx:
        Frozen record context produced for the underlying event.
    spec:
        Resolved capture policy for this record.
    ram_payload:
        Raw out copy retained in memory, or ``None`` when not stored
        either because the record is metadata-only or the caller opted out
        via ``save_raw_activations=False``.
    disk_payload:
        Raw out copy persisted to disk, or ``None`` when no disk
        target is active or the caller opted out via
        ``save_raw_activations=False``.
    transformed_ram_payload:
        Output of ``activation_transform`` retained in memory. ``None`` when
        no transform is configured for the recording.
    transformed_disk_payload:
        Output of ``activation_transform`` persisted to disk. ``None`` when
        no transform is configured for the recording.
    metadata:
        Auxiliary record metadata, including disk blob entries when present.
    recorded_at:
        Wall-clock time the record was created.

    Examples
    --------
    >>> activation = tl.record(model, x, save=tl.func("relu")).records[0]
    >>> type(activation).__name__, tuple(activation.ram_payload.shape)
    ('ActivationRecord', (1, 4, 8, 8))

    """

    ctx: RecordContext
    spec: CaptureSpec
    ram_payload: torch.Tensor | None = None
    disk_payload: torch.Tensor | None = None
    transformed_ram_payload: torch.Tensor | None = None
    transformed_disk_payload: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    recorded_time: float = field(default_factory=time.time)

    def __repr__(self) -> str:
        """Return a bounded one-line record summary.

        Payloads render as metadata descriptors (dtype/shape/storage), never
        contents: the dataclass auto-repr printed full tensor payloads AND
        the recursive ``ctx`` lookback (lovely bug 1, CRITICAL).
        """

        stores = [
            name
            for name, payload in (
                ("ram", self.ram_payload),
                ("disk", self.disk_payload),
                ("transformed_ram", self.transformed_ram_payload),
                ("transformed_disk", self.transformed_disk_payload),
            )
            if payload is not None
        ]
        payload = self.ram_payload if self.ram_payload is not None else self.disk_payload
        if payload is None:
            descriptor = "metadata_only"
        else:
            dtype = str(payload.dtype).replace("torch.", "")
            descriptor = f"{dtype}[{','.join(str(d) for d in payload.shape)}]"
        return (
            f"ActivationRecord(label={self.ctx.label!r}, pass={self.ctx.pass_number}, "
            f"payload={descriptor}, stored={'+'.join(stores) if stores else 'none'})"
        )


@dataclass(frozen=True, slots=True)
class GradRecordContext:
    """Predicate input schema for one fastlog backward gradient event.

    Parameters
    ----------
    label:
        Label assigned to the autograd node during the backward walk.
    layer_label:
        Forward fastlog label joined by ``grad_fn_handle`` identity, when available.
    op_label:
        Alias of the joined forward operation label for selector parity.
    module_stack:
        Forward module stack captured for the joined operation.
    has_forward_op:
        Whether this backward node corresponds to a predicate-mode forward op.
    has_op:
        Whether this backward node has a joined forward op.

    Examples
    --------
    >>> net = nn.Sequential(nn.Linear(4, 3), nn.ReLU())
    >>> recording = tl.record(net, torch.randn(2, 4), save=tl.func("relu"),
    ...                       capture=tl.options.CaptureOptions(backward_ready=True, save_grads=True))
    >>> out = recording.output
    >>> _ = recording.backward(out.sum(), save_grads=True)
    >>> context = recording.grad_records[0].ctx
    >>> type(context).__name__, context.grad_kind in ("grad_input", "grad_output")
    ('GradRecordContext', True)

    """

    label: str
    grad_fn_class_name: str
    kind: str
    backward_call_index: int
    grad_kind: Literal["grad_input", "grad_output"]
    grad_input_index: int | None = None
    grad_output_index: int | None = None
    layer_label: str | None = None
    op_label: str | None = None
    module_stack: tuple[Any, ...] = ()
    has_forward_op: bool = False
    has_op: bool = False
    pass_number: int | None = None
    order: int | None = None
    event_index: int | None = None
    shape: tuple[int, ...] | None = None
    dtype: torch.dtype | None = None
    tensor_device: torch.device | None = None

    @property
    def effective_label(self) -> str:
        """Return the forward label when joined, otherwise the grad-fn label."""

        return self.layer_label or self.label


@dataclass(frozen=True, slots=True)
class GradRecord:
    """One retained fastlog gradient event.

    Examples
    --------
    >>> net = nn.Sequential(nn.Linear(4, 3), nn.ReLU())
    >>> recording = tl.record(net, torch.randn(2, 4), save=tl.func("relu"),
    ...                       capture=tl.options.CaptureOptions(backward_ready=True, save_grads=True))
    >>> out = recording.output
    >>> _ = recording.backward(out.sum(), save_grads=True)
    >>> type(recording.grad_records[0]).__name__
    'GradRecord'
    """

    ctx: GradRecordContext
    spec: CaptureSpec
    ram_payload: torch.Tensor | None = None
    disk_payload: torch.Tensor | None = None
    transformed_ram_payload: torch.Tensor | None = None
    transformed_disk_payload: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    recorded_time: float = field(default_factory=time.time)


@dataclass(frozen=True, slots=True)
class PredicateFailure:
    """One captured predicate exception."""

    event_index: int
    kind: EventKind
    label: str
    traceback: str


# Preserve public type identity and pickle paths through the host's re-exports.
CaptureSpec.__module__ = "torchlens.recording.types"
StorageIntent.__module__ = "torchlens.recording.types"
ActivationRecord.__module__ = "torchlens.recording.types"
GradRecordContext.__module__ = "torchlens.recording.types"
GradRecord.__module__ = "torchlens.recording.types"
PredicateFailure.__module__ = "torchlens.recording.types"
