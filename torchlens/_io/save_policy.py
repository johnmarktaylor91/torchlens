"""Save-level payload policy: include-flag resolution + write-side coherence.

Lane A08 (WT1 A-IV items 19 + 22; weightsfree W3), split out of
``_io/bundle.py`` per the file-size ratchet's split-preferred rule. Owns the
per-level include-flag defaults/requirements tables, the explicit-conflict
refusals (``save_payload_level_conflict``), and the save-side M-C2 mirror
that refuses to WRITE a structure-only artifact its own load would refuse.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .._errors import InvalidArgumentError
from . import TorchLensIOError

if TYPE_CHECKING:
    from ..data_classes.trace import Trace


def _refuse_incoherent_structure_only_write(scrubbed_state: dict[str, Any]) -> None:
    """Refuse to WRITE a structure-only artifact its own load would refuse.

    Save-side mirror of the M-C2 load coherence gate (W3: "save-side coherence
    validation before writing"): a scrubbed structure-only state carrying any
    retained value payload would persist a poison artifact -- every future
    load refuses ``artifact_structure_only_incoherent`` -- so the save refuses
    typed instead, before any bytes land. The load-side gate in
    ``_io/forgery_validation.py`` stays strict and unchanged.
    """

    if scrubbed_state.get("structure_only") is not True:
        return
    from .forgery_validation import _OP_VALUE_PAYLOAD_FIELDS

    for entry in scrubbed_state.get("layer_list") or ():
        for field_name in _OP_VALUE_PAYLOAD_FIELDS:
            if getattr(entry, field_name, None) is not None:
                label = getattr(entry, "label", "<unknown>")
                raise TorchLensIOError(
                    "Refusing to write a structure-only artifact that its own "
                    f"load would refuse: op {label!r} retains a value payload "
                    f"in {field_name!r} under the structure-only marker "
                    "(M-C2). This is a capture/scrub defect, not a user "
                    "error; re-capture and report it if it persists."
                )


#: Per-level defaults for the four payload include flags when the caller omits
#: them (``None``). ``portable`` keeps the historical signature defaults;
#: ``audit``/``runnable`` are the payload-free levels;
#: ``executable_with_callables`` exists to ship re-execution payloads.
_LEVEL_INCLUDE_FLAG_DEFAULTS: dict[str, dict[str, bool]] = {
    "portable": {
        "include_outs": True,
        "include_grads": True,
        "include_saved_args": False,
        "include_rng_states": False,
    },
    "audit": {
        "include_outs": False,
        "include_grads": False,
        "include_saved_args": False,
        "include_rng_states": False,
    },
    "runnable": {
        "include_outs": False,
        "include_grads": False,
        "include_saved_args": False,
        "include_rng_states": False,
    },
    "executable_with_callables": {
        "include_outs": True,
        "include_grads": True,
        "include_saved_args": True,
        "include_rng_states": True,
    },
}

#: Include-flag values each level REQUIRES. An EXPLICIT contradicting value is
#: a typed refusal (WT1 A-IV item 19), never a silent override: the historical
#: behavior forced these assignments after the fact, so
#: ``level="executable_with_callables", include_saved_args=False`` shipped raw
#: input tensors against an explicit opt-out, and ``level="audit",
#: include_outs=True`` silently saved less than asked.
_LEVEL_INCLUDE_FLAG_REQUIREMENTS: dict[str, dict[str, bool]] = {
    "audit": _LEVEL_INCLUDE_FLAG_DEFAULTS["audit"],
    "runnable": _LEVEL_INCLUDE_FLAG_DEFAULTS["runnable"],
    "executable_with_callables": {
        "include_saved_args": True,
        "include_rng_states": True,
    },
}


def _resolve_include_flags(
    trace: Trace,
    *,
    save_level: str,
    include_outs: bool | None,
    include_grads: bool | None,
    include_saved_args: bool | None,
    include_rng_states: bool | None,
    include_buffer_values: bool | None,
) -> tuple[bool, bool, bool, bool, bool]:
    """Resolve omitted payload include flags and refuse explicit conflicts.

    Returns the five resolved boolean flags in signature order. Raises
    :class:`InvalidArgumentError` (code ``save_payload_level_conflict``) when
    an EXPLICIT flag contradicts a level requirement or the structure-only
    capture mode -- the two historical silent-override holes (WT1 A-IV items
    19 and 22; M(weightsfree) W3).
    """

    structure_only = bool(getattr(trace, "structure_only", False))
    if structure_only and save_level == "executable_with_callables":
        # Item 22: this level's required saved-args/RNG payloads land under
        # the structure-only marker and the M-C2 load coherence gate refuses
        # them, so the save used to write an artifact that could NEVER load.
        raise InvalidArgumentError(
            "level='executable_with_callables' cannot save a structure-only "
            "capture: the level ships saved-args/RNG payloads so the artifact "
            "can re-execute, but a structure-only capture retains no values "
            "and its loads refuse value payloads (M-C2). The historical save "
            "wrote a poison artifact here.",
            code="save_payload_level_conflict",
            remedy=(
                "save with level='portable' or level='audit', or re-capture "
                "without structure_only for an executable artifact"
            ),
            arguments=("level",),
        )
    requested: dict[str, bool | None] = {
        "include_outs": include_outs,
        "include_grads": include_grads,
        "include_saved_args": include_saved_args,
        "include_rng_states": include_rng_states,
    }
    requirements = _LEVEL_INCLUDE_FLAG_REQUIREMENTS.get(save_level, {})
    for flag_name, required_value in requirements.items():
        explicit = requested[flag_name]
        if explicit is not None and explicit != required_value:
            if required_value:
                problem = (
                    f"level={save_level!r} requires {flag_name}=True (the level "
                    "exists to ship re-execution payloads), so an explicit "
                    f"{flag_name}=False contradicts it. The historical save "
                    "silently overrode the opt-out and shipped the payloads "
                    "anyway."
                )
                remedy = (
                    f"omit {flag_name}, or save with level='portable' and "
                    f"{flag_name}=False to opt out of these payloads"
                )
            else:
                problem = (
                    f"level={save_level!r} is a payload-free level, so an "
                    f"explicit {flag_name}=True contradicts it. The historical "
                    "save silently dropped the requested payloads."
                )
                remedy = f"omit {flag_name}, or save with level='portable' to include payloads" + (
                    " (runnable payload families use include_weights= / include_activations=)"
                    if save_level == "runnable"
                    else ""
                )
            raise InvalidArgumentError(
                problem,
                code="save_payload_level_conflict",
                remedy=remedy,
                arguments=(flag_name, "level"),
            )
    defaults = _LEVEL_INCLUDE_FLAG_DEFAULTS[save_level]
    resolved = {
        flag_name: (defaults[flag_name] if value is None else value)
        for flag_name, value in requested.items()
    }
    if structure_only and include_buffer_values is True:
        raise InvalidArgumentError(
            "include_buffer_values=True cannot be used with a structure-only "
            "capture: pre-forward buffer values are training-data-derived "
            "state a weights-free artifact must not carry (W3), and the M-C2 "
            "load coherence gate refuses value payloads under the marker.",
            code="save_payload_level_conflict",
            remedy="omit include_buffer_values (structure-only saves drop the channel)",
            arguments=("include_buffer_values",),
        )
    resolved_buffer_values = (
        (not structure_only) if include_buffer_values is None else include_buffer_values
    )
    return (
        resolved["include_outs"],
        resolved["include_grads"],
        resolved["include_saved_args"],
        resolved["include_rng_states"],
        resolved_buffer_values,
    )
