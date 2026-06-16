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
    """Predicate input schema for one chronological capture event."""

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
    recent_events: tuple["RecordContext", ...]
    recent_ops: tuple["RecordContext", ...]
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

    def __getattr__(self, name: str) -> Any:
        """Raise a schema-specific error for unknown predicate fields."""

        from ..fastlog.exceptions import RecordContextFieldError

        raise RecordContextFieldError(name)


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
