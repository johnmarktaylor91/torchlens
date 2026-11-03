"""Non-vacuous replay validation for the TensorFlow eager backend."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import numpy as np

from ..._validation_status import ValidationReplaySource, ValidationStatus
from .._constants import CONSTANT_PARENT_PAYLOAD_REASON, CONSTANT_PRODUCER_LEAF_REASON
from .._invariants import run_metadata_invariants
from .._validation_shared import float_replay_tolerances, ops_by_label as _ops_by_label
from ._raw_op_replay import _import_tensorflow, _replay_raw_op
from .op_callback_capture import TFOpCapture

TFOpClass = Literal[
    "value",
    "source",
    "pure-resource-read",
    "structural-value",
    "annotation",
    "effect-denied",
    "control-region",
]

_PURE_REPLAY_ALLOWLIST = frozenset(
    {
        "AddV2",
        "AvgPool",
        "BatchMatMulV2",
        "BiasAdd",
        "BroadcastTo",
        "Cast",
        "ConcatV2",
        "Conv2D",
        "DepthwiseConv2dNative",
        "Einsum",
        "Elu",
        "Erf",
        "Erfc",
        "Exp",
        "ExpandDims",
        "Fill",
        "GatherV2",
        "Greater",
        "Less",
        "Log",
        "MatMul",
        "Maximum",
        "MaxPool",
        "Mean",
        "Minimum",
        "Mul",
        "Neg",
        "Pad",
        "Pow",
        "Prod",
        "RealDiv",
        "Relu",
        "Relu6",
        "Reshape",
        "Rsqrt",
        "SelectV2",
        "Sigmoid",
        "Sin",
        "Slice",
        "Softmax",
        "Sqrt",
        "SquaredDifference",
        "Sub",
        "Sum",
        "Squeeze",
        "Tanh",
        "Tile",
        "Transpose",
    }
)

#: Literal producers (SPEC section 2, rule 2): ops whose every input is a host literal or
#: shape, so the op has no graph parents and its output is a constant of the forward.
#: ``Fill`` is also a variable-initializer op type; the classification keeps the
#: initializer-contamination failure for a ``Fill`` nothing in the forward consumed and
#: admits only a ``Fill`` whose output feeds a materialized op (``_is_literal_producer``).
_LITERAL_PRODUCER_OPS = frozenset({"Fill"})

#: Validation decision reasons this backend emits. All are members of the
#: closed ``VALIDATION_REASONS`` list in ``backends/_backend_protocol.py``
#: (``replay_match`` is the validated-decision spelling, not a failure reason).
_REASON_REPLAYED = "replay_match"
_REASON_REPLAY_MISMATCH = "replay_mismatch"
_REASON_PERTURBATION = "perturbation_insensitive"
_REASON_NO_FUNCTION = "no_replayable_function"
_REASON_EFFECT = "effectful_region"
_REASON_UNCLASSIFIED = "op_unclassified"
_REASON_INCONSISTENT = "capture_inconsistent"
_REASON_CONSTANT_LEAF = CONSTANT_PRODUCER_LEAF_REASON
_REASON_CONSTANT_PERTURB = CONSTANT_PARENT_PAYLOAD_REASON
_VALUE_OPS = _PURE_REPLAY_ALLOWLIST | frozenset({"Gelu", "Pack"})
_STRUCTURAL_VALUE_OPS = frozenset({"Shape", "Size", "Rank", "StridedSlice"})
_ANNOTATION_OPS = frozenset({"Identity", "StopGradient"})
_PURE_RESOURCE_READ_OPS = frozenset({"ReadVariableOp"})
_SOURCE_OPS = frozenset({"Const", "Placeholder", "constant"})
_EFFECT_DENIED_OPS = frozenset(
    {
        "Assert",
        "AssignAddVariableOp",
        "AssignSubVariableOp",
        "DestroyResourceOp",
        "NoOp",
        "PrintV2",
        "RandomStandardNormal",
        "StatefulPartitionedCall",
    }
)
_CONTROL_REGION_OPS = frozenset(
    {"If", "StatelessIf", "While", "StatelessWhile", "Case", "Switch", "Merge"}
    | {"StatefulPartitionedCall"}
)


@dataclass(frozen=True)
class TFValidationCounts:
    """TensorFlow validation outcome counts.

    Parameters
    ----------
    replayed_node_count
        Number of pure nodes replayed and perturbation-checked.
    pure_unverified_node_count
        Number of cleanly classified pure value nodes outside the replay
        allowlist or missing replayable payloads.
    effect_region_node_count
        Number of stateful, resource-read, control, structural, or annotation
        nodes excluded from pure per-op replay.
    failed_node_count
        Number of validation failures.
    """

    replayed_node_count: int = 0
    pure_unverified_node_count: int = 0
    effect_region_node_count: int = 0
    failed_node_count: int = 0


@dataclass(frozen=True)
class TFValidationResult:
    """Detailed TensorFlow validation result.

    Parameters
    ----------
    counts
        Replay and failure counts.
    failures
        Stable failure reason strings.
    classes
        Per-op class assignment keyed by raw label.
    replayed_histogram
        Replayed op-type histogram.
    pure_unverified_histogram
        Pure but non-replayed op-type histogram.
    effect_region_histogram
        Effect or non-value op-type histogram.
    """

    counts: TFValidationCounts
    failures: tuple[str, ...]
    classes: dict[str, TFOpClass]
    replayed_histogram: Counter[str]
    pure_unverified_histogram: Counter[str]
    effect_region_histogram: Counter[str]
    decisions: tuple[Mapping[str, Any], ...] = ()
    unverified_reason_counts: Mapping[str, int] = field(default_factory=dict)
    exempted_reason_counts: Mapping[str, int] = field(default_factory=dict)
    first_failure: tuple[str, str, str] | None = None


def validate_tf_trace(trace: Any, *, validate_metadata: bool = True) -> ValidationStatus:
    """Validate a TensorFlow trace with the P4 non-vacuous tripwire.

    Parameters
    ----------
    trace
        TensorFlow trace produced by the eager op-callback backend.
    validate_metadata
        Whether backend-neutral metadata invariants should also run.

    Returns
    -------
    ValidationStatus
        Failed, passed, or honestly unverified replay-validation status.
    """

    del validate_metadata
    result = validate_tf_trace_detailed(trace)
    source: ValidationReplaySource = (
        "loaded" if getattr(trace, "_loaded_from_bundle", False) else "live"
    )
    status = ValidationStatus.from_replay_counts(
        backend="tf",
        source=source,
        num_replayed_ops=result.counts.replayed_node_count,
        num_unverified_ops=0,
        num_pure_unverified_ops=result.counts.pure_unverified_node_count,
        num_effect_region_ops=result.counts.effect_region_node_count,
        num_failed_ops=result.counts.failed_node_count,
        payload_load_status=getattr(trace, "payload_load_status", None),
        unverified_reason_counts=result.unverified_reason_counts,
        exempted_reason_counts=result.exempted_reason_counts,
        decisions=result.decisions,
    )
    if status.state == "failed" and result.first_failure is not None:
        label, op_type, reason = result.first_failure
        status = replace(
            status,
            message=(
                f"Replay validation failed at {label} ({op_type}): {reason}; "
                f"{result.counts.failed_node_count} failing op(s), "
                f"{result.counts.replayed_node_count} replayed."
            ),
        )
    trace._tf_validation_result = result
    trace._validation_replay_status = status
    return status


def validate_tf_trace_detailed(trace: Any) -> TFValidationResult:
    """Run TensorFlow validation and return detailed counts and diagnostics.

    Parameters
    ----------
    trace
        TensorFlow trace produced by the eager op-callback backend.

    Returns
    -------
    TFValidationResult
        Detailed validation outcome.
    """

    replayed_histogram: Counter[str] = Counter()
    pure_unverified_histogram: Counter[str] = Counter()
    effect_region_histogram: Counter[str] = Counter()

    ops_by_label = _ops_by_label(trace)
    captures_by_label = _captures_by_label(trace)
    failures, classes = _structural_failures(trace, ops_by_label, captures_by_label)
    if failures:
        return _structural_failure_result(failures, classes, ops_by_label)

    replayed_count = 0
    pure_unverified_count = 0
    effect_region_count = 0
    decisions_list: list[Mapping[str, Any]] = []
    unverified_reasons: Counter[str] = Counter()
    exempted_reasons: Counter[str] = Counter()
    for label, op in ops_by_label.items():
        if label != getattr(op, "_label_raw", None):
            continue
        op_class = classes[label]
        op_type = str(getattr(op, "func_name", ""))
        op_label = str(getattr(op, "label", label))
        if op_class == "source":
            if op_type == "constant":
                _record_constant_leaf(op_label, decisions_list, exempted_reasons)
            continue
        if op_class != "value":
            if op_class in {"pure-resource-read", "effect-denied", "control-region"}:
                effect_region_count += 1
                effect_region_histogram[op_type] += 1
                unverified_reasons[_REASON_EFFECT] += 1
                decisions_list.append(
                    _decision(op_label, op_type, "replay", "unverified", _REASON_EFFECT)
                )
            continue
        capture = captures_by_label.get(label)
        if op_type not in _PURE_REPLAY_ALLOWLIST or capture is None:
            pure_unverified_count += 1
            pure_unverified_histogram[op_type] += 1
            unverified_reasons[_REASON_NO_FUNCTION] += 1
            decisions_list.append(
                _decision(op_label, op_type, "replay", "unverified", _REASON_NO_FUNCTION)
            )
            continue
        outcome = _replay_and_perturb_op(op=op, capture=capture, ops_by_label=ops_by_label)
        if outcome == _REASON_REPLAYED:
            replayed_count += 1
            replayed_histogram[op_type] += 1
            decisions_list.append(_decision(op_label, op_type, "replay", "validated", outcome))
            continue
        failures.append(f"replay_or_perturbation_failed:{label}:{op_type}:{outcome}")
        phase = "perturbation" if outcome == _REASON_PERTURBATION else "replay"
        decisions_list.append(_decision(op_label, op_type, phase, "failed", outcome))

    return _detailed_result(
        counts=TFValidationCounts(
            replayed_node_count=replayed_count,
            pure_unverified_node_count=pure_unverified_count,
            effect_region_node_count=effect_region_count,
            failed_node_count=len(failures),
        ),
        failures=failures,
        classes=classes,
        histograms=(replayed_histogram, pure_unverified_histogram, effect_region_histogram),
        decision_tallies=(decisions_list, unverified_reasons, exempted_reasons),
    )


def _record_constant_leaf(
    op_label: str, decisions_list: list[Mapping[str, Any]], exempted_reasons: Counter[str]
) -> None:
    """Record a closure constant as an exempted leaf (constants rule item 3)."""

    # Constants rule item 3: a closure constant is a leaf whose
    # saved payload is the record; nothing upstream to perturb.
    decisions_list.extend(_constant_leaf_decisions(op_label))
    exempted_reasons[_REASON_CONSTANT_LEAF] += 1
    exempted_reasons[_REASON_CONSTANT_PERTURB] += 1


def _detailed_result(
    *,
    counts: TFValidationCounts,
    failures: list[str],
    classes: dict[str, TFOpClass],
    histograms: tuple[Counter[str], Counter[str], Counter[str]],
    decision_tallies: tuple[list[Mapping[str, Any]], Counter[str], Counter[str]],
) -> TFValidationResult:
    """Assemble the detailed result from the loop's tallies (decisions, unverified, exempted)."""

    replayed_histogram, pure_unverified_histogram, effect_region_histogram = histograms
    decisions_list, unverified_reasons, exempted_reasons = decision_tallies
    decisions = tuple(decisions_list)
    return TFValidationResult(
        counts=counts,
        failures=tuple(failures),
        classes=classes,
        replayed_histogram=replayed_histogram,
        pure_unverified_histogram=pure_unverified_histogram,
        effect_region_histogram=effect_region_histogram,
        decisions=decisions,
        unverified_reason_counts=dict(unverified_reasons),
        exempted_reason_counts=dict(exempted_reasons),
        first_failure=_first_failure(decisions),
    )


def _structural_failures(
    trace: Any, ops_by_label: Mapping[str, Any], captures_by_label: Mapping[str, Any]
) -> tuple[list[str], dict[str, TFOpClass]]:
    """Classify every op and check the capture's self-consistency.

    Returns
    -------
    tuple[list[str], dict[str, TFOpClass]]
        The structural failure strings (empty when the capture is sound) and
        the op class by raw label.
    """

    failures: list[str] = []
    classes: dict[str, TFOpClass] = {}
    init_labels = set(getattr(trace, "_tf_init_op_labels", ()))
    _validate_classification(
        ops_by_label=ops_by_label,
        init_labels=init_labels,
        classes=classes,
        failures=failures,
    )
    _validate_self_consistency(
        trace=trace,
        ops_by_label=ops_by_label,
        captures_by_label=captures_by_label,
        failures=failures,
    )
    return failures, classes


def _structural_failure_result(
    failures: list[str], classes: dict[str, TFOpClass], ops_by_label: Mapping[str, Any]
) -> TFValidationResult:
    """The result of a capture whose structure failed before any replay ran."""

    decisions = tuple(_structural_failure_decision(item, ops_by_label) for item in failures)
    return TFValidationResult(
        counts=TFValidationCounts(failed_node_count=len(failures)),
        failures=tuple(failures),
        classes=classes,
        replayed_histogram=Counter(),
        pure_unverified_histogram=Counter(),
        effect_region_histogram=Counter(),
        decisions=decisions,
        first_failure=_first_failure(decisions),
    )


def _decision(
    op_label: str, func_name: str, phase: str, decision: str, reason: str
) -> Mapping[str, Any]:
    """One per-op validation decision in the torch record shape."""

    return {
        "op_label": op_label,
        "func_name": func_name,
        "phase": phase,
        "decision": decision,
        "reason": reason,
    }


def _constant_leaf_decisions(op_label: str) -> tuple[Mapping[str, Any], ...]:
    """The two named leaf decisions of a ``constant`` source op (constants rule item 3)."""

    replay = dict(_decision(op_label, "constant", "replay", "exempted", _REASON_CONSTANT_LEAF))
    replay["justification"] = "closure constant: its saved payload is the record"
    perturb = dict(
        _decision(op_label, "constant", "perturbation", "exempted", _REASON_CONSTANT_PERTURB)
    )
    perturb["justification"] = "closure constant: no upstream op to perturb"
    return (replay, perturb)


def _structural_failure_decision(item: str, ops_by_label: Mapping[str, Any]) -> Mapping[str, Any]:
    """Decision for one classification or self-consistency failure string.

    Failure strings are ``<kind>:<raw label>:<detail...>``; the op label is
    resolved through the op index so the decision names the public label.
    """

    kind, _, rest = item.partition(":")
    raw_label, _, detail = rest.partition(":")
    op = ops_by_label.get(raw_label)
    op_label = str(getattr(op, "label", raw_label)) if op is not None else raw_label
    func_name = str(getattr(op, "func_name", "")) if op is not None else detail
    structural = {"unclassified_op", "initializer_contamination"}
    reason = _REASON_UNCLASSIFIED if kind in structural else _REASON_INCONSISTENT
    record = dict(_decision(op_label, func_name, "metadata", "failed", reason))
    record["justification"] = item
    return record


def _first_failure(decisions: tuple[Mapping[str, Any], ...]) -> tuple[str, str, str] | None:
    """Label, func name and reason of the first failed decision, if any."""

    for record in decisions:
        if record.get("decision") == "failed":
            return (
                str(record.get("op_label")),
                str(record.get("func_name")),
                str(record.get("reason")),
            )
    return None


def validate_tf_trace_status(trace: Any, *, validate_metadata: bool = True) -> ValidationStatus:
    """The protocol's ``validate``: always a ``ValidationStatus``, never a bool.

    Parameters
    ----------
    trace
        TensorFlow trace produced by the eager op-callback backend.
    validate_metadata
        Whether backend-neutral metadata invariants run first.

    Returns
    -------
    ValidationStatus
        ``unavailable`` for a loaded trace whose runtime captures were
        stripped; otherwise the replay result with per-op ``decisions`` and,
        on failure, a message naming the first offending op.
    """

    status = trace.validation_replay_status
    if not status.is_available:
        trace._validation_replay_status = status
        return status
    # Direct read of the load-provenance marker: present only on a loaded
    # trace, so its absence IS the typed "live" reading (the live-reading seam).
    try:
        loaded_from_bundle = bool(trace._loaded_from_bundle)
    except AttributeError:
        loaded_from_bundle = False
    if loaded_from_bundle and not getattr(trace, "_tf_op_captures", ()):
        # Portable save strips the runtime callback captures; a loaded trace
        # has nothing to replay, which is "unavailable", never "failed".
        stripped = ValidationStatus.unavailable_loaded_runtime_stripped(
            backend="tf", payload_load_status=getattr(trace, "payload_load_status", None)
        )
        trace._validation_replay_status = stripped
        return stripped
    source: ValidationReplaySource = "loaded" if loaded_from_bundle else "live"
    try:
        if validate_metadata:
            run_metadata_invariants(trace)
        return validate_tf_trace(trace, validate_metadata=validate_metadata)
    except Exception as exc:  # noqa: BLE001 - the tripwire reports, never raises
        failed = ValidationStatus.result(
            passed=False,
            backend="tf",
            source=source,
            payload_load_status=getattr(trace, "payload_load_status", None),
            num_failed_ops=1,
            decisions=(
                _decision("<trace>", "", "metadata", "failed", "metadata_invariant_exception"),
            ),
        )
        failed = replace(
            failed,
            message=f"Replay validation failed before replay: {type(exc).__name__}: {exc}",
        )
        trace._validation_replay_status = failed
        return failed


def replay_allowlist() -> tuple[str, ...]:
    """Return the TensorFlow pure-op raw replay allowlist.

    Returns
    -------
    tuple[str, ...]
        Sorted replayable TensorFlow op types.
    """

    return tuple(sorted(_PURE_REPLAY_ALLOWLIST))


def _validate_classification(
    *,
    ops_by_label: Mapping[str, Any],
    init_labels: set[str],
    classes: dict[str, TFOpClass],
    failures: list[str],
) -> None:
    """Classify every materialized op exactly once and fail closed.

    Parameters
    ----------
    ops_by_label
        Materialized ops keyed by raw and public labels.
    init_labels
        Raw labels marked as initializer contamination during capture.
    classes
        Mutable class mapping to populate.
    failures
        Mutable failure list.

    Returns
    -------
    None
        Mutates ``classes`` and ``failures``.
    """

    for label, op in ops_by_label.items():
        if label != getattr(op, "_label_raw", None):
            continue
        op_type = str(getattr(op, "func_name", ""))
        if label in init_labels and not _is_literal_producer(op):
            failures.append(f"initializer_contamination:{label}:{op_type}")
            continue
        op_class = _classify_op(op)
        if op_class is None:
            failures.append(f"unclassified_op:{label}:{op_type}")
            continue
        classes[label] = op_class
    _propagate_control_region_classes(ops_by_label=ops_by_label, classes=classes)


def _propagate_control_region_classes(
    *,
    ops_by_label: Mapping[str, Any],
    classes: dict[str, TFOpClass],
) -> None:
    """Mark ops downstream of control regions as unverified regions.

    Parameters
    ----------
    ops_by_label
        Materialized ops keyed by raw and public labels.
    classes
        Mutable class mapping to update.

    Returns
    -------
    None
        Mutates ``classes`` in place.
    """

    def _raw_parent_labels(op: Any) -> tuple[str, ...]:
        """Return the op's parents resolved to raw label space.

        Classes are keyed by RAW labels while grouped edges hold final
        pass-qualified labels, so parents resolve through the op index back
        to their raw capture identity before region membership tests.
        """

        resolved: list[str] = []
        for parent in getattr(op, "parents", ()):
            parent_text = str(parent)
            parent_op = ops_by_label.get(parent_text)
            parent_raw = getattr(parent_op, "_label_raw", None) if parent_op is not None else None
            resolved.append(parent_raw if isinstance(parent_raw, str) else parent_text)
        return tuple(resolved)

    changed = True
    while changed:
        changed = False
        region_labels = {
            label for label, op_class in classes.items() if op_class == "control-region"
        }
        for label, op in ops_by_label.items():
            if label != getattr(op, "_label_raw", None):
                continue
            if classes.get(label) in {"source", "control-region"}:
                continue
            if any(parent in region_labels for parent in _raw_parent_labels(op)):
                classes[label] = "control-region"
                changed = True


def _is_literal_producer(op: Any) -> bool:
    """Whether ``op`` is a forward literal producer rather than initializer contamination.

    A literal producer (SPEC section 2, rule 2) has no graph parents and its
    output is consumed by at least one materialized op of the forward. An
    initializer ``Fill`` feeds only ``AssignVariableOp`` (never materialized),
    so it has no children and stays a contamination failure.
    """

    op_type = str(getattr(op, "func_name", ""))
    if op_type not in _LITERAL_PRODUCER_OPS:
        return False
    if tuple(getattr(op, "parents", ()) or ()):
        return False
    return bool(tuple(getattr(op, "children", ()) or ()))


def _classify_op(op: Any) -> TFOpClass | None:
    """Classify one TensorFlow op for validation accounting.

    Parameters
    ----------
    op
        Materialized TorchLens op.

    Returns
    -------
    TFOpClass | None
        Exactly one validation class, or ``None`` for fail-closed unknown ops.
    """

    op_type = str(getattr(op, "func_name", ""))
    if bool(getattr(op, "is_input", False)) or op_type == "input":
        return "source"
    if op_type in _SOURCE_OPS:
        return "source"
    if op_type in _PURE_RESOURCE_READ_OPS:
        return "pure-resource-read"
    if op_type in _STRUCTURAL_VALUE_OPS:
        return "structural-value"
    if op_type in _ANNOTATION_OPS:
        return "annotation"
    if op_type.startswith("__inference_") or op_type.startswith("region:"):
        return "control-region"
    if op_type in _CONTROL_REGION_OPS:
        return "control-region"
    if op_type in _EFFECT_DENIED_OPS:
        return "effect-denied"
    if op_type in _VALUE_OPS:
        return "value"
    return None


def _validate_self_consistency(
    *,
    trace: Any,
    ops_by_label: Mapping[str, Any],
    captures_by_label: Mapping[str, TFOpCapture],
    failures: list[str],
) -> None:
    """Validate callback output and consumed-input producer consistency.

    Parameters
    ----------
    trace
        TensorFlow trace with side-channel records.
    ops_by_label
        Materialized ops keyed by raw and public labels.
    captures_by_label
        Callback captures keyed by raw label.
    failures
        Mutable failure list.

    Returns
    -------
    None
        Mutates ``failures`` on coverage gaps.
    """

    unresolved = tuple(getattr(trace, "_tf_unresolved_producers", ()))
    if unresolved:
        failures.extend(
            f"unresolved_producer:{item.consumer_op_type}:{item.input_index}" for item in unresolved
        )
    produced_labels: set[str] = set()
    source_labels = _source_labels(ops_by_label)
    for capture in captures_by_label.values():
        if capture.label_raw not in ops_by_label:
            failures.append(f"captured_output_missing_materialized_op:{capture.label_raw}")
        produced_labels.add(capture.label_raw)
    for label, op in ops_by_label.items():
        if label != getattr(op, "_label_raw", None):
            continue
        if bool(getattr(op, "is_input", False)) or label in source_labels:
            continue
        if label not in captures_by_label:
            failures.append(f"materialized_op_missing_capture:{label}")
    for capture in captures_by_label.values():
        expected_graph_parents: set[str] = set()
        for input_record in capture.inputs:
            graph_parent = input_record.producer_label_raw or input_record.source_label_raw
            if graph_parent is not None:
                expected_graph_parents.add(graph_parent)
        for input_record in capture.inputs:
            producer_label = input_record.producer_label_raw
            source_label = input_record.source_label_raw
            if producer_label is not None and producer_label not in produced_labels:
                failures.append(
                    "input_producer_missing_prior_callback_output:"
                    f"{capture.label_raw}:{producer_label}:{input_record.input_index}"
                )
            if producer_label is None and source_label is None and input_record.source_kind is None:
                failures.append(
                    f"input_missing_producer_or_source:{capture.label_raw}:"
                    f"{input_record.input_index}"
                )
            if source_label is not None and source_label not in source_labels:
                failures.append(
                    f"input_source_label_missing:{capture.label_raw}:{source_label}:"
                    f"{input_record.input_index}"
                )
        op = ops_by_label.get(capture.label_raw)
        if op is None:
            continue
        # Capture records speak RAW label space. Recurrence grouping rewrites graph edges to final
        # pass-qualified labels, so parents are resolved back to raw space before the conservation
        # comparison; an unresolvable parent keeps its literal label and fails closed.
        graph_parents: set[str] = set()
        for parent in getattr(op, "parents", ()):
            parent_text = str(parent)
            parent_op = ops_by_label.get(parent_text)
            parent_raw = getattr(parent_op, "_label_raw", None) if parent_op is not None else None
            graph_parents.add(parent_raw if isinstance(parent_raw, str) else parent_text)
        if expected_graph_parents != graph_parents:
            failures.append(
                f"graph_parent_edges_not_conserved:{capture.label_raw}:"
                f"{sorted(expected_graph_parents)!r}:{sorted(graph_parents)!r}"
            )


def _source_labels(ops_by_label: Mapping[str, Any]) -> set[str]:
    """Raw labels of the trace's source ops: inputs and constants."""

    source_labels: set[str] = set()
    for label, op in ops_by_label.items():
        if label != getattr(op, "_label_raw", None):
            continue
        if bool(getattr(op, "is_input", False)) or getattr(op, "func_name", None) in {
            "input",
            "constant",
        }:
            source_labels.add(label)
    return source_labels


def _replay_and_perturb_op(
    *,
    op: Any,
    capture: TFOpCapture,
    ops_by_label: Mapping[str, Any],
) -> str:
    """Replay one pure TensorFlow op and run the parent perturbation tripwire.

    Parameters
    ----------
    op
        Materialized TorchLens op.
    capture
        Callback capture for ``op``.
    ops_by_label
        Materialized ops keyed by raw and public labels.

    Returns
    -------
    str
        ``replay_match`` when replay matches and at least one parent
        perturbation changes the output when value parents exist; else the
        reason it did not (``replay_mismatch`` or ``perturbation_insensitive``).
    """

    try:
        inputs = _rebuild_inputs(capture, ops_by_label, replacements={})
        replayed = _replay_raw_op(capture, inputs)
        saved = _saved_payload(op)
        if not _payloads_close(replayed, saved):
            return _REASON_REPLAY_MISMATCH
        if _parent_perturbations_change_output(
            capture=capture,
            ops_by_label=ops_by_label,
            saved_output=saved,
        ):
            return _REASON_REPLAYED
        return _REASON_PERTURBATION
    except Exception:
        return _REASON_REPLAY_MISMATCH


def _rebuild_inputs(
    capture: TFOpCapture,
    ops_by_label: Mapping[str, Any],
    replacements: Mapping[str, Any],
) -> list[Any]:
    """Rebuild callback inputs from recorded parents and typed sources.

    Parameters
    ----------
    capture
        Callback capture to rebuild.
    ops_by_label
        Materialized ops keyed by raw and public labels.
    replacements
        Optional replacement tensors keyed by raw producer/source label.

    Returns
    -------
    list[Any]
        TensorFlow tensors in callback input order.
    """

    tf = _import_tensorflow()
    inputs: list[Any] = []
    for input_record in sorted(capture.inputs, key=lambda item: item.input_index):
        label = input_record.producer_label_raw or input_record.source_label_raw
        if label is not None and label in replacements:
            inputs.append(replacements[label])
            continue
        if input_record.producer_label_raw is not None:
            parent = ops_by_label.get(input_record.producer_label_raw)
            if parent is None:
                raise LookupError(input_record.producer_label_raw)
            inputs.append(_to_tf_tensor(tf, _saved_payload(parent)))
            continue
        if input_record.source_label_raw is not None:
            source = ops_by_label.get(input_record.source_label_raw)
            if source is None:
                raise LookupError(input_record.source_label_raw)
            inputs.append(_to_tf_tensor(tf, _saved_payload(source)))
            continue
        if input_record.source_kind is not None:
            inputs.append(_to_tf_tensor(tf, _tensor_to_numpy(input_record.tensor)))
            continue
        raise LookupError(f"unresolved input {capture.label_raw}:{input_record.input_index}")
    return inputs


def _parent_perturbations_change_output(
    *,
    capture: TFOpCapture,
    ops_by_label: Mapping[str, Any],
    saved_output: Any,
) -> bool:
    """Return whether parent perturbation changes a replayed output.

    Parameters
    ----------
    capture
        Callback capture to perturb.
    ops_by_label
        Materialized ops keyed by raw and public labels.
    saved_output
        Recorded output payload.

    Returns
    -------
    bool
        True when at least one value parent perturbation changes output. Ops
        with no recorded value parents are accepted only when no perturbation is
        possible by construction.
    """

    parent_labels = tuple(
        dict.fromkeys(
            label
            for item in capture.inputs
            if (label := (item.producer_label_raw or item.source_label_raw)) is not None
        )
    )
    if not parent_labels:
        return True
    attempted = False
    tf = _import_tensorflow()
    for label in parent_labels:
        parent = ops_by_label.get(label)
        if parent is None:
            return False
        original = _to_tf_tensor(tf, _saved_payload(parent))
        for candidate in _perturb_candidates(tf, original):
            attempted = True
            try:
                replayed = _replay_raw_op(
                    capture,
                    _rebuild_inputs(capture, ops_by_label, replacements={label: candidate}),
                )
            except Exception:
                continue
            if not _payloads_close(replayed, saved_output):
                return True
    return not attempted


def _perturb_candidates(tf: Any, tensor: Any) -> tuple[Any, ...]:
    """Return deterministic TensorFlow perturbation candidates.

    Parameters
    ----------
    tf
        Imported TensorFlow module.
    tensor
        TensorFlow tensor to perturb.

    Returns
    -------
    tuple[Any, ...]
        Candidate tensors with matching dtype.
    """

    array = _tensor_to_numpy(tensor)
    dtype = getattr(tensor, "dtype", None)
    candidates: tuple[Any, ...]
    if np.issubdtype(array.dtype, np.bool_):
        candidates = (np.logical_not(array),)
    elif np.issubdtype(array.dtype, np.integer):
        candidates = (array + 1, np.zeros_like(array), np.invert(array))  # ~x flips the sign
    elif np.issubdtype(array.dtype, np.floating):
        magnitude = np.max(np.abs(array)).item() + 1.0 if array.size else 1.0
        ramp = np.arange(array.size, dtype=np.float64).reshape(array.shape)
        ramp = (ramp + 1.0) * (magnitude / max(array.size, 1))
        candidates = (array + magnitude, array - magnitude, np.zeros_like(array))
        candidates = (*candidates, array + ramp.astype(array.dtype, copy=False))
    else:
        return ()
    return tuple(tf.convert_to_tensor(candidate, dtype=dtype) for candidate in candidates)


def _payloads_close(left: Any, right: Any) -> bool:
    """Return whether two TensorFlow payloads match within backend tolerance.

    Parameters
    ----------
    left
        Left payload or TensorFlow tensor.
    right
        Right payload or TensorFlow tensor.

    Returns
    -------
    bool
        True when shape, dtype, and values match.
    """

    left_array = _tensor_to_numpy(left)
    right_array = _tensor_to_numpy(right)
    if left_array.shape != right_array.shape or left_array.dtype != right_array.dtype:
        return False
    if np.issubdtype(left_array.dtype, np.bool_) or np.issubdtype(left_array.dtype, np.integer):
        return bool(np.array_equal(left_array, right_array))
    if np.issubdtype(left_array.dtype, np.floating) or np.issubdtype(
        left_array.dtype,
        np.complexfloating,
    ):
        # equal_nan matches the jax/mlx/paddle sibling oracles and torch's tensor_nanequal doctrine:
        # an identical NaN pattern is agreement, NaN-vs-number still fails elementwise. Omitting it
        # false-FAILED every legitimately NaN-bearing TF payload. Complex payloads take the same
        # component-eps band as mlx/jax (R17 consistency sweep: bit-exact complex here false-failed
        # legitimate replay jitter); ``np.finfo`` reports component precision for complex dtypes.
        rtol, atol = float_replay_tolerances(np.finfo(left_array.dtype))
        return bool(np.allclose(left_array, right_array, rtol=rtol, atol=atol, equal_nan=True))
    return bool(np.array_equal(left_array, right_array))


def _tensor_to_numpy(value: Any) -> np.ndarray:
    """Convert a TensorFlow tensor or payload to a NumPy array.

    Parameters
    ----------
    value
        TensorFlow tensor, NumPy array, or scalar payload.

    Returns
    -------
    np.ndarray
        NumPy representation.
    """

    numpy_method = getattr(value, "numpy", None)
    if callable(numpy_method):
        value = numpy_method()
    return np.asarray(value)


def _to_tf_tensor(tf: Any, value: Any) -> Any:
    """Convert a payload to a TensorFlow tensor.

    Parameters
    ----------
    tf
        Imported TensorFlow module.
    value
        Payload value.

    Returns
    -------
    Any
        TensorFlow tensor.
    """

    return tf.convert_to_tensor(value)


def _saved_payload(op: Any) -> Any:
    """Return an op's saved output payload or raise on missing data.

    Parameters
    ----------
    op
        Materialized TorchLens op.

    Returns
    -------
    Any
        Saved output payload.
    """

    if not bool(getattr(op, "has_saved_activation", False)):
        raise ValueError(f"op has no saved activation: {getattr(op, 'label', op)!r}")
    return op.out


def _captures_by_label(trace: Any) -> dict[str, TFOpCapture]:
    """Return TensorFlow callback captures keyed by raw labels.

    Parameters
    ----------
    trace
        Materialized TensorFlow trace.

    Returns
    -------
    dict[str, TFOpCapture]
        Captures keyed by raw output label.
    """

    return {
        capture.label_raw: capture
        for capture in getattr(trace, "_tf_op_captures", ())
        if isinstance(capture, TFOpCapture)
    }


__all__ = [
    "TFValidationCounts",
    "TFValidationResult",
    "replay_allowlist",
    "validate_tf_trace",
    "validate_tf_trace_detailed",
]
