"""Grouped options for fastlog predicate recording."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Final, Literal, get_args

from .._deprecations import MISSING, MissingType
from .._errors import InvalidArgumentError
from ..intervention.predicates import InterventionPredicate
from ..ir.predicate import RetroactiveCaptureDecision
from ..options import StreamingOptions
from ..types import ActivationPostfunc, GradientPostfunc
from .types import CaptureSpec, GradRecordContext, RecordContext

CaptureDecision = bool | CaptureSpec | None
PredicateDecision = CaptureDecision | RetroactiveCaptureDecision
PredicateFn = Callable[[RecordContext], PredicateDecision]
HaltPredicateFn = Callable[[RecordContext], bool]
GradPredicateFn = Callable[[GradRecordContext], CaptureDecision]
PredicateErrorMode = Literal["auto", "accumulate", "fail-fast"]
ForwardErrorMode = Literal["raise", "attach_partial", "return_partial"]
LookbackPayloadPolicy = Literal[
    "metadata_only",
    "detached_raw",
    "transformed",
    "grad_connected",
    "disk_spilled",
]

LOOKBACK_PAYLOAD_POLICIES: Final[tuple[str, ...]] = get_args(LookbackPayloadPolicy)
"""Runtime authority for the lookback payload policy vocabulary.

Derived from the canonical :data:`LookbackPayloadPolicy` literal so validation
and the literal can never drift apart; consumers import this tuple rather than
re-spelling the policy strings.
"""

_RECORDING_FIELDS: Final[tuple[str, ...]] = (
    "keep_op",
    "default_op",
    "default_module",
    "history_size",
    "lookback",
    "lookback_payload_policy",
    "include_source_events",
    "intervene",
    "halt",
    "max_predicate_failures",
    "on_predicate_error",
    "on_forward_error",
    "streaming",
    "random_seed",
    "activation_transform",
    "save_raw_activations",
    "save_grads",
    "default_grad",
    "grad_transform",
    "save_raw_gradients",
    "echo",
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

    keep_op: PredicateFn | None
    default_op: bool | CaptureSpec
    default_module: bool | CaptureSpec
    history_size: int
    lookback: int
    lookback_payload_policy: LookbackPayloadPolicy
    include_source_events: bool
    intervene: InterventionPredicate | None
    halt: HaltPredicateFn | None
    max_predicate_failures: int
    on_predicate_error: PredicateErrorMode
    on_forward_error: ForwardErrorMode
    streaming: StreamingOptions | None
    random_seed: int | None
    activation_transform: ActivationPostfunc | None
    save_raw_activations: bool
    save_grads: GradPredicateFn | bool | CaptureSpec | None
    default_grad: bool | CaptureSpec
    grad_transform: GradientPostfunc | None
    save_raw_gradients: bool
    #: Normalized live-narration options (``torchlens.options.EchoOptions``)
    #: or ``None``; typed ``Any`` here because fastlog (L2) never imports
    #: the narration engine -- seams read the armed session off the trace.
    echo: Any | None
    _specified_fields: frozenset[str] = field(
        default_factory=frozenset,
        init=False,
        repr=False,
        compare=False,
    )

    def __init__(
        self,
        keep_op: PredicateFn | None | MissingType = MISSING,
        default_op: bool | CaptureSpec | MissingType = MISSING,
        default_module: bool | CaptureSpec | MissingType = MISSING,
        history_size: int | MissingType = MISSING,
        lookback: int | MissingType = MISSING,
        lookback_payload_policy: LookbackPayloadPolicy | MissingType = MISSING,
        include_source_events: bool | MissingType = MISSING,
        intervene: InterventionPredicate | None | MissingType = MISSING,
        halt: HaltPredicateFn | None | MissingType = MISSING,
        max_predicate_failures: int | MissingType = MISSING,
        on_predicate_error: PredicateErrorMode | MissingType = MISSING,
        on_forward_error: ForwardErrorMode | MissingType = MISSING,
        streaming: StreamingOptions | None | MissingType = MISSING,
        random_seed: int | None | MissingType = MISSING,
        activation_transform: ActivationPostfunc | None | MissingType = MISSING,
        save_raw_activations: bool | MissingType = MISSING,
        save_grads: GradPredicateFn | bool | CaptureSpec | None | MissingType = MISSING,
        default_grad: bool | CaptureSpec | MissingType = MISSING,
        grad_transform: GradientPostfunc | None | MissingType = MISSING,
        save_raw_gradients: bool | MissingType = MISSING,
        echo: Any | None | MissingType = MISSING,
    ) -> None:
        """Initialize a frozen recording option bundle."""

        specified_fields: set[str] = set()
        values: dict[str, Any] = {
            "keep_op": _resolve_recording_option("keep_op", keep_op, None, specified_fields),
            "default_op": _resolve_recording_option(
                "default_op", default_op, False, specified_fields
            ),
            "default_module": _resolve_recording_option(
                "default_module", default_module, False, specified_fields
            ),
            "history_size": _resolve_recording_option(
                "history_size", history_size, 8, specified_fields
            ),
            "lookback": _resolve_recording_option("lookback", lookback, 0, specified_fields),
            "lookback_payload_policy": _resolve_recording_option(
                "lookback_payload_policy",
                lookback_payload_policy,
                "metadata_only",
                specified_fields,
            ),
            "include_source_events": _resolve_recording_option(
                "include_source_events", include_source_events, False, specified_fields
            ),
            "intervene": _resolve_recording_option("intervene", intervene, None, specified_fields),
            "halt": _resolve_recording_option("halt", halt, None, specified_fields),
            "max_predicate_failures": _resolve_recording_option(
                "max_predicate_failures", max_predicate_failures, 32, specified_fields
            ),
            "on_predicate_error": _resolve_recording_option(
                "on_predicate_error", on_predicate_error, "auto", specified_fields
            ),
            "on_forward_error": _resolve_recording_option(
                "on_forward_error", on_forward_error, "raise", specified_fields
            ),
            "streaming": _resolve_recording_option("streaming", streaming, None, specified_fields),
            "random_seed": _resolve_recording_option(
                "random_seed", random_seed, None, specified_fields
            ),
            "activation_transform": _resolve_recording_option(
                "activation_transform", activation_transform, None, specified_fields
            ),
            "save_raw_activations": _resolve_recording_option(
                "save_raw_activations", save_raw_activations, True, specified_fields
            ),
            "save_grads": _resolve_recording_option(
                "save_grads", save_grads, None, specified_fields
            ),
            "default_grad": _resolve_recording_option(
                "default_grad", default_grad, False, specified_fields
            ),
            "grad_transform": _resolve_recording_option(
                "grad_transform", grad_transform, None, specified_fields
            ),
            "save_raw_gradients": _resolve_recording_option(
                "save_raw_gradients", save_raw_gradients, True, specified_fields
            ),
            "echo": _resolve_recording_option("echo", echo, None, specified_fields),
        }
        _validate_recording_values(values)
        for field_name in _RECORDING_FIELDS:
            object.__setattr__(self, field_name, values[field_name])
        object.__setattr__(self, "_specified_fields", frozenset(specified_fields))

    def as_dict(self) -> dict[str, Any]:
        """Return the option values as a plain dictionary."""

        return {field_name: getattr(self, field_name) for field_name in _RECORDING_FIELDS}

    def is_field_explicit(self, field_name: str) -> bool:
        """Return whether a field was explicitly supplied by the caller."""

        return field_name in self._specified_fields

    @classmethod
    def from_values(
        cls: type[RecordingOptions],
        values: Mapping[str, Any],
        specified_fields: frozenset[str],
    ) -> RecordingOptions:
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
    lookback = values["lookback"]
    lookback_payload_policy = values["lookback_payload_policy"]
    intervene = values["intervene"]
    halt = values["halt"]
    max_predicate_failures = values["max_predicate_failures"]
    on_predicate_error = values["on_predicate_error"]
    on_forward_error = values["on_forward_error"]
    activation_transform = values["activation_transform"]
    save_raw_activations = values["save_raw_activations"]
    save_grads = values["save_grads"]
    default_grad = values["default_grad"]
    grad_transform = values["grad_transform"]
    save_raw_gradients = values["save_raw_gradients"]
    if not isinstance(history_size, int) or not 0 <= history_size <= 1024:
        raise InvalidArgumentError(
            f"history_size must be an integer in [0, 1024]; received {history_size!r}",
            code="history_size_invalid",
            remedy="pass an integer history_size between 0 and 1024",
            argument="history_size",
        )
    if not isinstance(lookback, int) or not 0 <= lookback <= 1024:
        raise InvalidArgumentError(
            f"lookback must be an integer in [0, 1024]; received {lookback!r}",
            code="lookback_invalid",
            remedy="pass an integer lookback between 0 and 1024",
            argument="lookback",
        )
    if lookback_payload_policy not in LOOKBACK_PAYLOAD_POLICIES:
        allowed = ", ".join(repr(policy) for policy in LOOKBACK_PAYLOAD_POLICIES[:-1])
        raise InvalidArgumentError(
            f"lookback_payload_policy must be one of {allowed}, "
            f"or {LOOKBACK_PAYLOAD_POLICIES[-1]!r}; "
            f"received {lookback_payload_policy!r}",
            code="lookback_payload_policy_invalid",
            remedy="choose a documented lookback payload policy",
            argument="lookback_payload_policy",
        )
    # R64: the record surface carries its OWN codes (recording_*) so one
    # documented code never maps to two catchable builtins -- the tl.trace
    # twins keep TypeError lineage under the unprefixed codes, this surface
    # keeps its historical ValueError lineage (the F1/F3/F4 split pattern).
    if intervene is not None and not callable(intervene):
        raise InvalidArgumentError(
            f"intervene must be callable or None; received {type(intervene).__name__}",
            code="recording_intervention_predicate_type_invalid",
            remedy="pass tl.when(...), another predicate, or None",
            argument="intervene",
        )
    if halt is not None and not callable(halt):
        raise InvalidArgumentError(
            f"halt must be callable or None; received {type(halt).__name__}",
            code="recording_halt_predicate_type_invalid",
            remedy="pass a halt predicate or None",
            argument="halt",
        )
    if not isinstance(max_predicate_failures, int) or max_predicate_failures < 0:
        raise InvalidArgumentError(
            f"max_predicate_failures must be a non-negative integer; "
            f"received {max_predicate_failures!r}",
            code="max_predicate_failures_invalid",
            remedy="pass a non-negative integer max_predicate_failures",
            argument="max_predicate_failures",
        )
    if on_predicate_error not in {"auto", "accumulate", "fail-fast"}:
        raise InvalidArgumentError(
            "on_predicate_error must be 'auto', 'accumulate', or 'fail-fast'; "
            f"received {on_predicate_error!r}",
            code="on_predicate_error_invalid",
            remedy="pass on_predicate_error='auto', 'accumulate', or 'fail-fast'",
            argument="on_predicate_error",
        )
    if on_forward_error not in {"raise", "attach_partial", "return_partial"}:
        raise InvalidArgumentError(
            "on_forward_error must be 'raise', 'attach_partial', or 'return_partial'; "
            f"received {on_forward_error!r}",
            code="on_forward_error_invalid",
            remedy="pass on_forward_error='raise', 'attach_partial', or 'return_partial'",
            argument="on_forward_error",
        )
    if activation_transform is not None and not callable(activation_transform):
        raise InvalidArgumentError(
            f"activation_transform must be callable or None; "
            f"received {type(activation_transform).__name__}",
            code="recording_option_type_invalid",
            remedy="pass a callable activation_transform or None",
            argument="activation_transform",
        )
    if not isinstance(save_raw_activations, bool):
        raise InvalidArgumentError(
            f"save_raw_activations must be a bool; received {type(save_raw_activations).__name__}",
            code="recording_option_type_invalid",
            remedy="pass save_raw_activations=True or False",
            argument="save_raw_activations",
        )
    if (
        save_grads is not None
        and not isinstance(save_grads, (bool, CaptureSpec))
        and not callable(save_grads)
    ):
        raise InvalidArgumentError(
            f"save_grads must be callable, bool, CaptureSpec, or None; "
            f"received {type(save_grads).__name__}",
            code="recording_option_type_invalid",
            remedy="pass a predicate, bool, CaptureSpec, or None as save_grads",
            argument="save_grads",
        )
    if not isinstance(default_grad, (bool, CaptureSpec)):
        raise InvalidArgumentError(
            f"default_grad must be bool or CaptureSpec; received {type(default_grad).__name__}",
            code="recording_option_type_invalid",
            remedy="pass a bool or CaptureSpec default_grad",
            argument="default_grad",
        )
    if grad_transform is not None and not callable(grad_transform):
        raise InvalidArgumentError(
            f"grad_transform must be callable or None; received {type(grad_transform).__name__}",
            code="recording_option_type_invalid",
            remedy="pass a callable grad_transform or None",
            argument="grad_transform",
        )
    if not isinstance(save_raw_gradients, bool):
        raise InvalidArgumentError(
            f"save_raw_gradients must be a bool; received {type(save_raw_gradients).__name__}",
            code="recording_option_type_invalid",
            remedy="pass save_raw_gradients=True or False",
            argument="save_raw_gradients",
        )


def merge_recording_options(
    *,
    recording: RecordingOptions | None,
    keep_op: PredicateFn | None | MissingType = MISSING,
    default_op: bool | CaptureSpec | MissingType = MISSING,
    default_module: bool | CaptureSpec | MissingType = MISSING,
    history_size: int | MissingType = MISSING,
    lookback: int | MissingType = MISSING,
    lookback_payload_policy: LookbackPayloadPolicy | MissingType = MISSING,
    include_source_events: bool | MissingType = MISSING,
    intervene: InterventionPredicate | None | MissingType = MISSING,
    halt: HaltPredicateFn | None | MissingType = MISSING,
    max_predicate_failures: int | MissingType = MISSING,
    on_predicate_error: PredicateErrorMode | MissingType = MISSING,
    on_forward_error: ForwardErrorMode | MissingType = MISSING,
    streaming: StreamingOptions | None | MissingType = MISSING,
    random_seed: int | None | MissingType = MISSING,
    activation_transform: ActivationPostfunc | None | MissingType = MISSING,
    save_raw_activations: bool | MissingType = MISSING,
    save_grads: GradPredicateFn | bool | CaptureSpec | None | MissingType = MISSING,
    default_grad: bool | CaptureSpec | MissingType = MISSING,
    grad_transform: GradientPostfunc | None | MissingType = MISSING,
    save_raw_gradients: bool | MissingType = MISSING,
    echo: Any | None | MissingType = MISSING,
) -> RecordingOptions:
    """Merge flat recording kwargs into a grouped options object."""

    base_values = recording.as_dict() if recording is not None else RecordingOptions().as_dict()
    specified_fields = (
        set(recording._specified_fields) if recording is not None else set()  # noqa: SLF001
    )
    incoming = {
        "keep_op": keep_op,
        "default_op": default_op,
        "default_module": default_module,
        "history_size": history_size,
        "lookback": lookback,
        "lookback_payload_policy": lookback_payload_policy,
        "include_source_events": include_source_events,
        "intervene": intervene,
        "halt": halt,
        "max_predicate_failures": max_predicate_failures,
        "on_predicate_error": on_predicate_error,
        "on_forward_error": on_forward_error,
        "streaming": streaming,
        "random_seed": random_seed,
        "activation_transform": activation_transform,
        "save_raw_activations": save_raw_activations,
        "save_grads": save_grads,
        "default_grad": default_grad,
        "grad_transform": grad_transform,
        "save_raw_gradients": save_raw_gradients,
        "echo": echo,
    }
    for field_name, value in incoming.items():
        if value is MISSING:
            continue
        if field_name in specified_fields:
            raise InvalidArgumentError(
                f"Recording option {field_name!r} was specified twice",
                code="recording_option_duplicate",
                remedy="pass each recording option exactly once",
                argument=field_name,
            )
        base_values[field_name] = value
        specified_fields.add(field_name)
    return RecordingOptions.from_values(base_values, frozenset(specified_fields))
