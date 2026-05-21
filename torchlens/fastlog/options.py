"""Grouped options for fastlog predicate recording."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Final, Literal, Mapping

from .._deprecations import MISSING, MissingType, warn_deprecated_alias
from ..options import StreamingOptions
from ..types import ActivationPostfunc, GradientPostfunc
from .types import CaptureSpec, GradRecordContext, RecordContext

CaptureDecision = bool | CaptureSpec | None
PredicateFn = Callable[[RecordContext], CaptureDecision]
GradPredicateFn = Callable[[GradRecordContext], CaptureDecision]
PredicateErrorMode = Literal["auto", "accumulate", "fail-fast"]

_RECORDING_FIELDS: Final[tuple[str, ...]] = (
    "keep_op",
    "keep_module",
    "default_op",
    "default_module",
    "history_size",
    "include_source_events",
    "max_predicate_failures",
    "on_predicate_error",
    "streaming",
    "random_seed",
    "out_transform",
    "save_raw_outs",
    "keep_grad",
    "default_grad",
    "gradient_transform",
    "save_raw_gradients",
)


def _resolve_recording_option(
    field_name: str,
    supplied_value: Any,
    default_value: Any,
    specified_fields: set[str],
) -> Any:
    """Resolve an option field while tracking explicit caller presence."""

    if supplied_value is MISSING:
        return default_value
    specified_fields.add(field_name)
    return supplied_value


@dataclass(frozen=True, slots=True, init=False)
class RecordingOptions:
    """Grouped options for one fastlog predicate recording session."""

    keep_op: PredicateFn | None = None
    keep_module: PredicateFn | None = None
    default_op: bool | CaptureSpec = False
    default_module: bool | CaptureSpec = False
    history_size: int = 8
    include_source_events: bool = False
    max_predicate_failures: int = 32
    on_predicate_error: PredicateErrorMode = "auto"
    streaming: StreamingOptions | None = None
    random_seed: int | None = None
    out_transform: ActivationPostfunc | None = None
    save_raw_outs: bool = True
    keep_grad: GradPredicateFn | bool | CaptureSpec | None = None
    default_grad: bool | CaptureSpec = False
    gradient_transform: GradientPostfunc | None = None
    save_raw_gradients: bool = True
    _specified_fields: frozenset[str] = field(
        default_factory=frozenset,
        init=False,
        repr=False,
        compare=False,
    )

    def __init__(
        self,
        keep_op: PredicateFn | None | MissingType = MISSING,
        keep_module: PredicateFn | None | MissingType = MISSING,
        default_op: bool | CaptureSpec | MissingType = MISSING,
        default_module: bool | CaptureSpec | MissingType = MISSING,
        history_size: int | MissingType = MISSING,
        include_source_events: bool | MissingType = MISSING,
        max_predicate_failures: int | MissingType = MISSING,
        on_predicate_error: PredicateErrorMode | MissingType = MISSING,
        streaming: StreamingOptions | None | MissingType = MISSING,
        random_seed: int | None | MissingType = MISSING,
        out_transform: ActivationPostfunc | None | MissingType = MISSING,
        save_raw_outs: bool | MissingType = MISSING,
        keep_grad: GradPredicateFn | bool | CaptureSpec | None | MissingType = MISSING,
        default_grad: bool | CaptureSpec | MissingType = MISSING,
        gradient_transform: GradientPostfunc | None | MissingType = MISSING,
        save_raw_gradients: bool | MissingType = MISSING,
        *,
        out_postfunc: ActivationPostfunc | None | MissingType = MISSING,
        gradient_postfunc: GradientPostfunc | None | MissingType = MISSING,
    ) -> None:
        """Initialize a frozen recording option bundle."""

        if out_postfunc is not MISSING:
            if out_transform is not MISSING:
                raise TypeError(
                    "kwarg out_postfunc deprecated, use out_transform; do not pass both"
                )
            warn_deprecated_alias("out_postfunc", "out_transform")
            out_transform = out_postfunc
        if gradient_postfunc is not MISSING:
            if gradient_transform is not MISSING:
                raise TypeError(
                    "kwarg gradient_postfunc aliases gradient_transform; do not pass both"
                )
            gradient_transform = gradient_postfunc
        specified_fields: set[str] = set()
        values: dict[str, Any] = {
            "keep_op": _resolve_recording_option("keep_op", keep_op, None, specified_fields),
            "keep_module": _resolve_recording_option(
                "keep_module", keep_module, None, specified_fields
            ),
            "default_op": _resolve_recording_option(
                "default_op", default_op, False, specified_fields
            ),
            "default_module": _resolve_recording_option(
                "default_module", default_module, False, specified_fields
            ),
            "history_size": _resolve_recording_option(
                "history_size", history_size, 8, specified_fields
            ),
            "include_source_events": _resolve_recording_option(
                "include_source_events", include_source_events, False, specified_fields
            ),
            "max_predicate_failures": _resolve_recording_option(
                "max_predicate_failures", max_predicate_failures, 32, specified_fields
            ),
            "on_predicate_error": _resolve_recording_option(
                "on_predicate_error", on_predicate_error, "auto", specified_fields
            ),
            "streaming": _resolve_recording_option("streaming", streaming, None, specified_fields),
            "random_seed": _resolve_recording_option(
                "random_seed", random_seed, None, specified_fields
            ),
            "out_transform": _resolve_recording_option(
                "out_transform", out_transform, None, specified_fields
            ),
            "save_raw_outs": _resolve_recording_option(
                "save_raw_outs", save_raw_outs, True, specified_fields
            ),
            "keep_grad": _resolve_recording_option("keep_grad", keep_grad, None, specified_fields),
            "default_grad": _resolve_recording_option(
                "default_grad", default_grad, False, specified_fields
            ),
            "gradient_transform": _resolve_recording_option(
                "gradient_transform", gradient_transform, None, specified_fields
            ),
            "save_raw_gradients": _resolve_recording_option(
                "save_raw_gradients", save_raw_gradients, True, specified_fields
            ),
        }
        _validate_recording_values(values)
        for field_name in _RECORDING_FIELDS:
            object.__setattr__(self, field_name, values[field_name])
        object.__setattr__(self, "_specified_fields", frozenset(specified_fields))

    def as_dict(self) -> dict[str, Any]:
        """Return the option values as a plain dictionary."""

        return {field_name: getattr(self, field_name) for field_name in _RECORDING_FIELDS}

    @property
    def out_postfunc(self) -> ActivationPostfunc | None:
        """Deprecated alias for ``out_transform``."""

        warn_deprecated_alias("out_postfunc", "out_transform")
        return self.out_transform

    def is_field_explicit(self, field_name: str) -> bool:
        """Return whether a field was explicitly supplied by the caller."""

        return field_name in self._specified_fields

    @classmethod
    def from_values(
        cls,
        values: Mapping[str, Any],
        specified_fields: frozenset[str],
    ) -> "RecordingOptions":
        """Build an instance from already-resolved field values."""

        _validate_recording_values(values)
        instance = object.__new__(cls)
        for field_name in _RECORDING_FIELDS:
            object.__setattr__(instance, field_name, values[field_name])
        object.__setattr__(instance, "_specified_fields", specified_fields)
        return instance


def _validate_recording_values(values: Mapping[str, Any]) -> None:
    """Validate scalar recording option values."""

    history_size = values["history_size"]
    max_predicate_failures = values["max_predicate_failures"]
    on_predicate_error = values["on_predicate_error"]
    out_transform = values["out_transform"]
    save_raw_outs = values["save_raw_outs"]
    keep_grad = values["keep_grad"]
    default_grad = values["default_grad"]
    gradient_transform = values["gradient_transform"]
    save_raw_gradients = values["save_raw_gradients"]
    if not isinstance(history_size, int) or not 0 <= history_size <= 1024:
        raise ValueError("history_size must be an integer in [0, 1024]")
    if not isinstance(max_predicate_failures, int) or max_predicate_failures < 0:
        raise ValueError("max_predicate_failures must be a non-negative integer")
    if on_predicate_error not in {"auto", "accumulate", "fail-fast"}:
        raise ValueError("on_predicate_error must be 'auto', 'accumulate', or 'fail-fast'")
    if out_transform is not None and not callable(out_transform):
        raise ValueError("out_transform must be callable or None")
    if not isinstance(save_raw_outs, bool):
        raise ValueError("save_raw_outs must be a bool")
    if (
        keep_grad is not None
        and not isinstance(keep_grad, (bool, CaptureSpec))
        and not callable(keep_grad)
    ):
        raise ValueError("keep_grad must be callable, bool, CaptureSpec, or None")
    if not isinstance(default_grad, (bool, CaptureSpec)):
        raise ValueError("default_grad must be bool or CaptureSpec")
    if gradient_transform is not None and not callable(gradient_transform):
        raise ValueError("gradient_transform must be callable or None")
    if not isinstance(save_raw_gradients, bool):
        raise ValueError("save_raw_gradients must be a bool")


def merge_recording_options(
    *,
    recording: RecordingOptions | None,
    keep_op: PredicateFn | None | MissingType = MISSING,
    keep_module: PredicateFn | None | MissingType = MISSING,
    default_op: bool | CaptureSpec | MissingType = MISSING,
    default_module: bool | CaptureSpec | MissingType = MISSING,
    history_size: int | MissingType = MISSING,
    include_source_events: bool | MissingType = MISSING,
    max_predicate_failures: int | MissingType = MISSING,
    on_predicate_error: PredicateErrorMode | MissingType = MISSING,
    streaming: StreamingOptions | None | MissingType = MISSING,
    random_seed: int | None | MissingType = MISSING,
    out_transform: ActivationPostfunc | None | MissingType = MISSING,
    save_raw_outs: bool | MissingType = MISSING,
    keep_grad: GradPredicateFn | bool | CaptureSpec | None | MissingType = MISSING,
    default_grad: bool | CaptureSpec | MissingType = MISSING,
    gradient_transform: GradientPostfunc | None | MissingType = MISSING,
    save_raw_gradients: bool | MissingType = MISSING,
    out_postfunc: ActivationPostfunc | None | MissingType = MISSING,
    gradient_postfunc: GradientPostfunc | None | MissingType = MISSING,
) -> RecordingOptions:
    """Merge flat recording kwargs into a grouped options object."""

    if out_postfunc is not MISSING:
        if out_transform is not MISSING:
            raise TypeError("kwarg out_postfunc deprecated, use out_transform; do not pass both")
        warn_deprecated_alias("out_postfunc", "out_transform")
        out_transform = out_postfunc
    if gradient_postfunc is not MISSING:
        if gradient_transform is not MISSING:
            raise TypeError("kwarg gradient_postfunc aliases gradient_transform; do not pass both")
        gradient_transform = gradient_postfunc
    base_values = recording.as_dict() if recording is not None else RecordingOptions().as_dict()
    specified_fields = (
        set(recording._specified_fields) if recording is not None else set()  # noqa: SLF001
    )
    incoming = {
        "keep_op": keep_op,
        "keep_module": keep_module,
        "default_op": default_op,
        "default_module": default_module,
        "history_size": history_size,
        "include_source_events": include_source_events,
        "max_predicate_failures": max_predicate_failures,
        "on_predicate_error": on_predicate_error,
        "streaming": streaming,
        "random_seed": random_seed,
        "out_transform": out_transform,
        "save_raw_outs": save_raw_outs,
        "keep_grad": keep_grad,
        "default_grad": default_grad,
        "gradient_transform": gradient_transform,
        "save_raw_gradients": save_raw_gradients,
    }
    for field_name, value in incoming.items():
        if value is MISSING:
            continue
        if field_name in specified_fields:
            raise ValueError(f"Recording option {field_name!r} was specified twice")
        base_values[field_name] = value
        specified_fields.add(field_name)
    return RecordingOptions.from_values(base_values, frozenset(specified_fields))
