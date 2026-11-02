"""Grouped options for fastlog predicate recording."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Final, Literal, get_args

from .._deprecations import MISSING, MissingType
from .._errors import InvalidArgumentError
from .._ir.predicate import RetroactiveCaptureDecision
from ..intervention.predicates import InterventionPredicate
from ..options import StorageOptions
from ..types import _ActivationPostfunc, _GradientPostfunc
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
    "include_source_events",
    "intervene",
    "max_predicate_failures",
    "on_predicate_error",
    "on_forward_error",
    "default_grad",
    "postprocess",
)
"""The public ``RecordingOptions`` fields, in declaration order."""

_RESOLVED_ONLY_FIELDS: Final[tuple[str, ...]] = (
    "lookback",
    "lookback_payload_policy",
    "halt",
    "streaming",
    "seed",
    "activation_transform",
    "save_raw_activations",
    "save_grads",
    "grad_transform",
    "save_raw_grads",
    "echo",
    "detach_saved_activations",
)
"""Recorder slots with no ``RecordingOptions`` field.

Each reaches the recorder through another public door: the capture knobs ride
``capture=CaptureOptions(...)`` (``lookback``, ``lookback_payload_policy``,
``seed``, ``activation_transform``, ``save_raw_activations``, ``save_grads``,
``grad_transform``, ``save_raw_grads``, ``detach_saved_activations``), the stop
predicate is the door's ``stop_after=``, narration is ``narrate=`` and the disk
sink is ``storage=``.
"""


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
    """Grouped recorder options for one ``tl.record`` session.

    These are the knobs that belong to the recorder itself. Capture knobs ride
    ``capture=CaptureOptions(...)``, the stop predicate is the door's
    ``stop_after=``, narration is ``narrate=`` and the disk sink is ``storage=``.
    """

    keep_op: PredicateFn | None
    default_op: bool | CaptureSpec
    default_module: bool | CaptureSpec
    history_size: int
    include_source_events: bool
    intervene: InterventionPredicate | None
    max_predicate_failures: int
    on_predicate_error: PredicateErrorMode
    on_forward_error: ForwardErrorMode
    default_grad: bool | CaptureSpec
    #: Post-capture finalization pass: ``"none"`` (the default) or the
    #: ``validate_postprocess`` vocabulary (``tl.record(recording=...)``).
    postprocess: str
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
        include_source_events: bool | MissingType = MISSING,
        intervene: InterventionPredicate | None | MissingType = MISSING,
        max_predicate_failures: int | MissingType = MISSING,
        on_predicate_error: PredicateErrorMode | MissingType = MISSING,
        on_forward_error: ForwardErrorMode | MissingType = MISSING,
        default_grad: bool | CaptureSpec | MissingType = MISSING,
        postprocess: str | MissingType = MISSING,
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
            "include_source_events": _resolve_recording_option(
                "include_source_events", include_source_events, False, specified_fields
            ),
            "intervene": _resolve_recording_option("intervene", intervene, None, specified_fields),
            "max_predicate_failures": _resolve_recording_option(
                "max_predicate_failures", max_predicate_failures, 32, specified_fields
            ),
            "on_predicate_error": _resolve_recording_option(
                "on_predicate_error", on_predicate_error, "auto", specified_fields
            ),
            "on_forward_error": _resolve_recording_option(
                "on_forward_error", on_forward_error, "raise", specified_fields
            ),
            "default_grad": _resolve_recording_option(
                "default_grad", default_grad, False, specified_fields
            ),
            "postprocess": _resolve_recording_option(
                "postprocess", postprocess, "none", specified_fields
            ),
        }
        _validate_recording_values(values)
        for field_name in _RECORDING_FIELDS:
            object.__setattr__(self, field_name, values[field_name])
        object.__setattr__(self, "_specified_fields", frozenset(specified_fields))

    def as_dict(self) -> dict[str, Any]:
        """Return the option values as a plain dictionary."""

        return {field_name: getattr(self, field_name) for field_name in _RECORDING_FIELDS}

    def _is_field_explicit(self, field_name: str) -> bool:
        """Return whether a field was explicitly supplied by the caller."""

        return field_name in self._specified_fields

    @classmethod
    def _from_values(
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


@dataclass(frozen=True, slots=True)
class _RecordingConfig:
    """The recorder's resolved configuration: every slot one predicate session reads.

    Private. Built by the Recorder from the public ``RecordingOptions`` fields plus
    the values the other doors carry (``capture=``, ``stop_after=``, ``narrate=``,
    ``storage=``), and stored on the trace as ``_predicate_save_options`` for the
    capture seams that read it. The field names are the recorder's historical
    slot names, so every reader is unchanged.
    """

    keep_op: PredicateFn | None = None
    default_op: bool | CaptureSpec = False
    default_module: bool | CaptureSpec = False
    history_size: int = 8
    lookback: int = 0
    lookback_payload_policy: LookbackPayloadPolicy = "metadata_only"
    include_source_events: bool = False
    intervene: InterventionPredicate | None = None
    halt: HaltPredicateFn | None = None
    max_predicate_failures: int = 32
    on_predicate_error: PredicateErrorMode = "auto"
    on_forward_error: ForwardErrorMode = "raise"
    streaming: StorageOptions | None = None
    seed: int | None = None
    activation_transform: _ActivationPostfunc | None = None
    save_raw_activations: bool = True
    save_grads: GradPredicateFn | bool | CaptureSpec | None = None
    default_grad: bool | CaptureSpec = False
    grad_transform: _GradientPostfunc | None = None
    save_raw_grads: bool = True
    #: Normalized live-narration options (``torchlens.options.NarrationOptions``)
    #: or ``None``; typed ``Any`` here because fastlog (L2) never imports
    #: the narration engine -- seams read the armed session off the trace.
    echo: Any | None = None
    #: Whether a retained activation whose capture spec does not ask for
    #: ``save_grad`` is detached from the autograd graph. ``True`` (the
    #: ``record`` default) stores detached copies; ``False`` stores
    #: graph-connected copies wherever the source tensor carries autograd
    #: history, as ``tl.trace``'s default save does. A spec with
    #: ``save_grad=True`` stays attached either way. Disk payloads are always
    #: detached inspection copies.
    detach_saved_activations: bool = True
    postprocess: str = "none"

    def __post_init__(self) -> None:
        """Validate every slot with the recorder's historical refusals."""

        values = self.as_dict()
        _validate_recording_values(values)
        _validate_resolved_recording_values(values)

    def as_dict(self) -> dict[str, Any]:
        """Return every slot as a plain dictionary."""

        return {
            field_name: getattr(self, field_name)
            for field_name in (*_RECORDING_FIELDS, *_RESOLVED_ONLY_FIELDS)
        }


def _resolve_recording_config(**values: Any) -> _RecordingConfig:
    """Build the resolved config; a ``MISSING`` value falls to the slot default."""

    return _RecordingConfig(
        **{name: value for name, value in values.items() if value is not MISSING}
    )


def _validate_recording_values(values: Mapping[str, Any]) -> None:
    """Validate the public recording option values."""

    history_size = values["history_size"]
    intervene = values["intervene"]
    max_predicate_failures = values["max_predicate_failures"]
    on_predicate_error = values["on_predicate_error"]
    on_forward_error = values["on_forward_error"]
    default_grad = values["default_grad"]
    if not isinstance(history_size, int) or not 0 <= history_size <= 1024:
        raise InvalidArgumentError(
            f"history_size must be an integer in [0, 1024]; received {history_size!r}",
            code="history_size_invalid",
            remedy="pass an integer history_size between 0 and 1024",
            argument="history_size",
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
    if not isinstance(default_grad, (bool, CaptureSpec)):
        raise InvalidArgumentError(
            f"default_grad must be bool or CaptureSpec; received {type(default_grad).__name__}",
            code="recording_option_type_invalid",
            remedy="pass a bool or CaptureSpec default_grad",
            argument="default_grad",
        )


def _validate_resolved_recording_values(values: Mapping[str, Any]) -> None:
    """Validate the slots the recorder resolves from the other doors."""

    lookback = values["lookback"]
    lookback_payload_policy = values["lookback_payload_policy"]
    halt = values["halt"]
    activation_transform = values["activation_transform"]
    save_raw_activations = values["save_raw_activations"]
    save_grads = values["save_grads"]
    grad_transform = values["grad_transform"]
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
    if halt is not None and not callable(halt):
        raise InvalidArgumentError(
            f"halt must be callable or None; received {type(halt).__name__}",
            code="recording_halt_predicate_type_invalid",
            remedy="pass a halt predicate or None",
            argument="halt",
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
    if grad_transform is not None and not callable(grad_transform):
        raise InvalidArgumentError(
            f"grad_transform must be callable or None; received {type(grad_transform).__name__}",
            code="recording_option_type_invalid",
            remedy="pass a callable grad_transform or None",
            argument="grad_transform",
        )
    for flag_name in ("save_raw_grads", "detach_saved_activations"):
        flag = values[flag_name]
        if not isinstance(flag, bool):
            raise InvalidArgumentError(
                f"{flag_name} must be a bool; received {type(flag).__name__}",
                code="recording_option_type_invalid",
                remedy=f"pass {flag_name}=True or False",
                argument=flag_name,
            )
