"""Unified predicate context for capture filtering and selector adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from typing_extensions import Self

from .refs import DeviceRef, DtypeRef

EventKind = Literal["op", "module_enter", "module_exit", "input", "buffer"]
_DEFERRED_VALUE_FIELDS = frozenset({"tensor_requires_grad", "is_scalar_bool", "bool_value"})


class MLXValueUnavailableError(RuntimeError):
    """Raised when MLX lazy evaluation makes a predicate value unavailable."""


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
    """One frame in the active module stack."""

    address: str
    module_type: str
    module_id: int
    pass_index: int


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
    module_pass_index
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
    """

    kind: EventKind | str
    label: str
    raw_label: str | None
    pass_index: int
    event_index: int
    step_index: int | None
    layer_type: str | None
    type_index: int | None
    raw_index: int | None
    func_name: str | None
    address: str | None
    module_type: str | None
    module_pass_index: int | None
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

        from ..fastlog.exceptions import RecordContextFieldError

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
            f"pass={self.pass_index}, event={self.event_index}, shape={shape}, "
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
