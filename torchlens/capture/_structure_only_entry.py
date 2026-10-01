"""Structure-only Layer-0 entry contract (L7a sec 1.3/2.2).

Extracted from ``user_funcs`` by the wave-0 governance sweep (file-size
ratchet): the ONE entry chokepoint that refuses option combinations and
value-payload requests a structure-only capture cannot honor. All codes
here are DOCUMENTED-UNSTABLE pending naming-session/S2 ratification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .._errors import InvalidArgumentError, StructureOnlyOptionConflictError
from ..options import CaptureOptions


@dataclass(frozen=True)
class _StructureOnlyEntryFacts:
    """Resolved entry facts the structure-only Layer-0 contract checks read."""

    layers_to_save: Any
    save_predicate: Any
    halt: Any
    streaming_options: Any
    lookback_payload_policy: str
    raise_on_nan_value: bool
    track_nonfinite_value: bool
    intervention_ready: bool
    should_save_grads: bool


def _enforce_structure_only_entry_contract(
    capture_options: CaptureOptions,
    facts: _StructureOnlyEntryFacts,
) -> str | list[Any] | None:
    """Enforce the structure-only Layer-0 entry contract (L7a sec 1.3/2.2).

    Option COMBINATIONS that need tensor values refuse typed with the one
    ``structure_only_option_conflict`` code; explicit value-PAYLOAD requests
    refuse typed with ``structure_only_values_unsupported`` (the capability
    table's ``value_payloads`` row enforced at the earliest surface). A clean
    call resolves to a metadata-only save plan (returns ``None`` as the
    effective ``layers_to_save``): structure-only capture never retains value
    payloads, by contract, so the non-explicit ``layers_to_save`` default
    degrades to metadata-only rather than silently recording values.

    All codes here are DOCUMENTED-UNSTABLE pending naming-session/S2
    ratification.

    Returns
    -------
    str | list[Any] | None
        The effective metadata-only ``layers_to_save`` value (always ``None``).
    """

    _refuse_structure_only_conflicts(facts)
    _refuse_structure_only_payload_selections(capture_options, facts)
    _refuse_structure_only_option_payloads(capture_options, facts)
    return None


def _refuse_structure_only_conflicts(facts: _StructureOnlyEntryFacts) -> None:
    """The three option COMBINATIONS that need tensor values refuse typed."""

    halt = facts.halt
    conflict_remedy = (
        "drop the conflicting option or run a real capture (tl.trace without "
        "structure_only). Structure-only capture has no tensor values to test, "
        "mutate, or replay."
    )
    if facts.raise_on_nan_value:
        raise StructureOnlyOptionConflictError(
            "structure_only=True cannot combine with raise_on_nan=True: a "
            "structure-only capture has no tensor values for a nonfinite "
            "predicate to test",
            code="structure_only_option_conflict",
            remedy=conflict_remedy,
            arguments=("structure_only", "raise_on_nan"),
        )
    if facts.track_nonfinite_value:
        raise StructureOnlyOptionConflictError(
            "structure_only=True cannot combine with track_nonfinite=True: a "
            "structure-only capture has no tensor values for a per-op "
            "finiteness check to examine",
            code="structure_only_option_conflict",
            remedy=conflict_remedy,
            arguments=("structure_only", "track_nonfinite"),
        )
    if facts.intervention_ready:
        raise StructureOnlyOptionConflictError(
            "structure_only=True cannot combine with intervention_ready=True: "
            "runnable eligibility disables the plain escape belt and runnable "
            "save is refused wholesale under the structure-only contract, so "
            "the combination must be unreachable rather than quietly belt-less. "
            "This is the capability table's runnable_ready_composition row; "
            "its named flip event is the L7b declared late-bind amendment "
            "(state slots declared at capture, values bound at run time)",
            code="structure_only_option_conflict",
            remedy=(
                "capture the real model with intervention_ready=True (without "
                "structure_only) for a runnable artifact today, or drop "
                "intervention_ready to keep the structure-only capture"
            ),
            arguments=("structure_only", "intervention_ready"),
        )
    if halt is not None:
        from ..intervention.selectors import BaseSelector as _BaseSelector
        from ..ir.selector_eval import first_selector_kind_outside
        from ..postprocess._selective_save import _STATIC_SELECTOR_KINDS

        halt_value_suspect: str | None
        if not isinstance(halt, _BaseSelector):
            halt_value_suspect = "bare callable (value use unprovable)"
        else:
            halt_value_suspect = first_selector_kind_outside(halt, allowed=_STATIC_SELECTOR_KINDS)
        if halt_value_suspect is not None:
            raise StructureOnlyOptionConflictError(
                "structure_only=True requires a provably value-free halt= "
                f"predicate; received {halt_value_suspect!r}. A value-touching "
                "halt predicate would select the recorded graph by values the "
                "capture does not record",
                code="structure_only_option_conflict",
                remedy=(
                    "use structured value-free selectors (tl.func, tl.in_module, "
                    "label selectors, and their & | ~ compositions) as halt=, or "
                    "run a real capture"
                ),
                arguments=("structure_only", "halt"),
            )


_STRUCTURE_ONLY_VALUES_REMEDY = (
    "drop the payload-requesting option: structure-only capture records "
    "structure and shape/dtype hypotheses, never tensor values. Run a real "
    "capture (tl.trace without structure_only) to record values."
)


def _refuse_values(problem: str, *option_names: str) -> None:
    """Raise the typed structure-only value-payload refusal."""

    raise InvalidArgumentError(
        problem,
        code="structure_only_values_unsupported",
        remedy=_STRUCTURE_ONLY_VALUES_REMEDY,
        argument=option_names[0],
        arguments=option_names,
    )


def _refuse_structure_only_payload_selections(
    capture_options: CaptureOptions, facts: _StructureOnlyEntryFacts
) -> None:
    """Explicit activation/gradient payload SELECTIONS refuse typed."""

    if facts.save_predicate is not None:
        _refuse_values(
            "structure_only=True cannot honor a save= payload selection; "
            "activations are never recorded under the structure-only contract",
            "save",
        )
    if capture_options.is_field_explicit("layers_to_save") and facts.layers_to_save not in (
        "none",
        None,
        [],
    ):
        _refuse_values(
            "structure_only=True cannot honor an explicit layers_to_save "
            "payload selection; activations are never recorded under the "
            "structure-only contract",
            "layers_to_save",
        )
    if capture_options.is_field_explicit("save_grads") and facts.should_save_grads:
        _refuse_values(
            "structure_only=True cannot honor save_grads: gradient payloads "
            "are values and backward capture is refused under the "
            "structure-only contract",
            "save_grads",
        )
    streaming_options = facts.streaming_options
    if streaming_options.bundle_path is not None or streaming_options.out_callback is not None:
        _refuse_values(
            "structure_only=True cannot stream activation payloads to disk or "
            "callbacks; there are no value payloads to stream",
            "storage",
        )


def _refuse_structure_only_option_payloads(
    capture_options: CaptureOptions, facts: _StructureOnlyEntryFacts
) -> None:
    """Payload-recording OPTION values refuse typed."""

    if capture_options.is_field_explicit("save_arg_values") and capture_options.save_arg_values:
        _refuse_values(
            "structure_only=True cannot record non-tensor argument VALUES as "
            "payloads via save_arg_values",
            "save_arg_values",
        )
    if capture_options.is_field_explicit("layer_visualizers") and capture_options.layer_visualizers:
        _refuse_values(
            "structure_only=True cannot run payload-consuming layer "
            "visualizers; they require tensor values",
            "layer_visualizers",
        )
    if capture_options.is_field_explicit("save_raw_input") and capture_options.save_raw_input:
        _refuse_values(
            "structure_only=True cannot retain the raw input payload",
            "save_raw_input",
        )
    if capture_options.is_field_explicit("save_raw_output") and capture_options.save_raw_output:
        _refuse_values(
            "structure_only=True cannot retain the raw output payload",
            "save_raw_output",
        )
    if capture_options.is_field_explicit("output_style") and capture_options.output_style:
        _refuse_values(
            "structure_only=True cannot decode output VALUES via output_style",
            "output_style",
        )
    if facts.lookback_payload_policy != "metadata_only":
        _refuse_values(
            "structure_only=True supports only the metadata-only lookback "
            "window; retroactive payload retention records values",
            "lookback_payload_policy",
        )
