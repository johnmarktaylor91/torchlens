"""Non-vacuous replay validation for the TensorFlow eager backend."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from ... import _state
from ...validation.status import ValidationReplaySource, ValidationReplayStatus
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

_INIT_OP_TYPES = frozenset(
    {
        "StatelessRandomUniformV2",
        "RandomUniform",
        "Fill",
        "VarHandleOp",
        "AssignVariableOp",
    }
)
_PURE_REPLAY_ALLOWLIST = frozenset(
    {
        "AddV2",
        "AvgPool",
        "BatchMatMulV2",
        "BiasAdd",
        "Cast",
        "ConcatV2",
        "Conv2D",
        "Einsum",
        "MatMul",
        "MaxPool",
        "Mean",
        "Mul",
        "Neg",
        "RealDiv",
        "Relu",
        "Reshape",
        "Rsqrt",
        "Softmax",
        "Sqrt",
        "SquaredDifference",
        "Sub",
        "Squeeze",
        "Tanh",
        "Transpose",
    }
)
_VALUE_OPS = _PURE_REPLAY_ALLOWLIST | frozenset(
    {
        "Elu",
        "Erf",
        "Exp",
        "ExpandDims",
        "GatherV2",
        "Gelu",
        "Greater",
        "Less",
        "Log",
        "Maximum",
        "Minimum",
        "Pack",
        "Pad",
        "Pow",
        "Prod",
        "Sigmoid",
        "Sin",
        "Slice",
        "Squeeze",
        "Sum",
        "Tile",
    }
)
_STRUCTURAL_VALUE_OPS = frozenset({"Shape", "Size", "Rank", "StridedSlice"})
_ANNOTATION_OPS = frozenset({"Identity", "StopGradient"})
_PURE_RESOURCE_READ_OPS = frozenset({"ReadVariableOp"})
_SOURCE_OPS = frozenset({"Const", "Placeholder"})
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


def validate_tf_trace(trace: Any, *, validate_metadata: bool = True) -> ValidationReplayStatus:
    """Validate a TensorFlow trace with the P4 non-vacuous tripwire.

    Parameters
    ----------
    trace
        TensorFlow trace produced by the eager op-callback backend.
    validate_metadata
        Whether backend-neutral metadata invariants should also run.

    Returns
    -------
    ValidationReplayStatus
        Failed, passed, or honestly unverified replay-validation status.
    """

    del validate_metadata
    result = validate_tf_trace_detailed(trace)
    source: ValidationReplaySource = (
        "loaded" if getattr(trace, "_loaded_from_bundle", False) else "live"
    )
    status = ValidationReplayStatus.from_replay_counts(
        backend="tf",
        source=source,
        replayed_node_count=result.counts.replayed_node_count,
        unverified_node_count=0,
        pure_unverified_node_count=result.counts.pure_unverified_node_count,
        effect_region_node_count=result.counts.effect_region_node_count,
        failed_node_count=result.counts.failed_node_count,
        payload_load_status=getattr(trace, "payload_load_status", None),
    )
    setattr(trace, "_tf_validation_result", result)
    setattr(trace, "_validation_replay_status", status)
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

    failures: list[str] = []
    classes: dict[str, TFOpClass] = {}
    replayed_histogram: Counter[str] = Counter()
    pure_unverified_histogram: Counter[str] = Counter()
    effect_region_histogram: Counter[str] = Counter()

    ops_by_label = _ops_by_label(trace)
    captures_by_label = _captures_by_label(trace)
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
    if failures:
        return TFValidationResult(
            counts=TFValidationCounts(failed_node_count=len(failures)),
            failures=tuple(failures),
            classes=classes,
            replayed_histogram=replayed_histogram,
            pure_unverified_histogram=pure_unverified_histogram,
            effect_region_histogram=effect_region_histogram,
        )

    replayed_count = 0
    pure_unverified_count = 0
    effect_region_count = 0
    for label, op in ops_by_label.items():
        if label != getattr(op, "_label_raw", None):
            continue
        op_class = classes[label]
        op_type = str(getattr(op, "func_name", ""))
        if op_class != "value":
            if op_class in {"pure-resource-read", "effect-denied", "control-region"}:
                effect_region_count += 1
                effect_region_histogram[op_type] += 1
            continue
        capture = captures_by_label.get(label)
        if op_type not in _PURE_REPLAY_ALLOWLIST or capture is None:
            pure_unverified_count += 1
            pure_unverified_histogram[op_type] += 1
            continue
        if _replay_and_perturb_op(op=op, capture=capture, ops_by_label=ops_by_label):
            replayed_count += 1
            replayed_histogram[op_type] += 1
        else:
            failures.append(f"replay_or_perturbation_failed:{label}:{op_type}")

    return TFValidationResult(
        counts=TFValidationCounts(
            replayed_node_count=replayed_count,
            pure_unverified_node_count=pure_unverified_count,
            effect_region_node_count=effect_region_count,
            failed_node_count=len(failures),
        ),
        failures=tuple(failures),
        classes=classes,
        replayed_histogram=replayed_histogram,
        pure_unverified_histogram=pure_unverified_histogram,
        effect_region_histogram=effect_region_histogram,
    )


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
        if label in init_labels or op_type in _INIT_OP_TYPES and label in init_labels:
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
            if any(parent in region_labels for parent in getattr(op, "parents", ())):
                classes[label] = "control-region"
                changed = True


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
    source_labels: set[str] = set()
    for label, op in ops_by_label.items():
        if label != getattr(op, "_label_raw", None):
            continue
        if bool(getattr(op, "is_input", False)) or getattr(op, "func_name", None) == "input":
            source_labels.add(label)
    for capture in captures_by_label.values():
        if capture.label_raw not in ops_by_label:
            failures.append(f"captured_output_missing_materialized_op:{capture.label_raw}")
        produced_labels.add(capture.label_raw)
    for label, op in ops_by_label.items():
        if label != getattr(op, "_label_raw", None):
            continue
        if bool(getattr(op, "is_input", False)):
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
        graph_parents = set(getattr(op, "parents", ()))
        if expected_graph_parents != graph_parents:
            failures.append(
                f"graph_parent_edges_not_conserved:{capture.label_raw}:"
                f"{sorted(expected_graph_parents)!r}:{sorted(graph_parents)!r}"
            )


def _replay_and_perturb_op(
    *,
    op: Any,
    capture: TFOpCapture,
    ops_by_label: Mapping[str, Any],
) -> bool:
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
    bool
        True when replay matches and at least one parent perturbation changes
        the output when value parents exist.
    """

    try:
        inputs = _rebuild_inputs(capture, ops_by_label, replacements={})
        replayed = _replay_raw_op(capture, inputs)
        saved = _saved_payload(op)
        if not _payloads_close(replayed, saved):
            return False
        return _parent_perturbations_change_output(
            capture=capture,
            ops_by_label=ops_by_label,
            saved_output=saved,
        )
    except Exception:
        return False


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


def _replay_raw_op(capture: TFOpCapture, inputs: Sequence[Any]) -> Any:
    """Replay one TensorFlow raw op from rebuilt callback inputs.

    Parameters
    ----------
    capture
        Callback capture containing op type and attrs.
    inputs
        Rebuilt TensorFlow inputs.

    Returns
    -------
    Any
        Replayed TensorFlow output tensor.
    """

    dispatch: dict[str, Callable[[TFOpCapture, Sequence[Any]], Any]] = {
        "AddV2": lambda item, args: _raw(item).AddV2(x=args[0], y=args[1]),
        "AvgPool": _replay_pool,
        "BatchMatMulV2": _replay_batch_matmul_v2,
        "BiasAdd": _replay_bias_add,
        "Cast": _replay_cast,
        "ConcatV2": _replay_concat_v2,
        "Conv2D": _replay_conv2d,
        "Einsum": _replay_einsum,
        "MatMul": _replay_matmul,
        "MaxPool": _replay_pool,
        "Mean": _replay_mean,
        "Mul": lambda item, args: _raw(item).Mul(x=args[0], y=args[1]),
        "Neg": lambda item, args: _raw(item).Neg(x=args[0]),
        "RealDiv": lambda item, args: _raw(item).RealDiv(x=args[0], y=args[1]),
        "Relu": lambda item, args: _raw(item).Relu(features=args[0]),
        "Reshape": lambda item, args: _raw(item).Reshape(tensor=args[0], shape=args[1]),
        "Rsqrt": lambda item, args: _raw(item).Rsqrt(x=args[0]),
        "Softmax": lambda item, args: _raw(item).Softmax(logits=args[0]),
        "Sqrt": lambda item, args: _raw(item).Sqrt(x=args[0]),
        "SquaredDifference": lambda item, args: _raw(item).SquaredDifference(
            x=args[0],
            y=args[1],
        ),
        "Sub": lambda item, args: _raw(item).Sub(x=args[0], y=args[1]),
        "Squeeze": lambda item, args: _raw(item).Squeeze(
            input=args[0],
            axis=_optional_int_list(item.attrs.get("squeeze_dims")),
        ),
        "Tanh": lambda item, args: _raw(item).Tanh(x=args[0]),
        "Transpose": lambda item, args: _raw(item).Transpose(x=args[0], perm=args[1]),
    }
    replay = dispatch.get(capture.op_type)
    if replay is None:
        raise ValueError(f"op is not raw-op replayable: {capture.op_type}")
    with _state.pause_logging():
        return replay(capture, inputs)


def _raw(_capture: TFOpCapture) -> Any:
    """Return TensorFlow raw ops module.

    Parameters
    ----------
    _capture
        Unused capture, accepted for dispatch lambdas.

    Returns
    -------
    Any
        ``tf.raw_ops``.
    """

    return _import_tensorflow().raw_ops


def _replay_batch_matmul_v2(capture: TFOpCapture, inputs: Sequence[Any]) -> Any:
    """Replay ``BatchMatMulV2``.

    Parameters
    ----------
    capture
        Callback capture.
    inputs
        Rebuilt inputs.

    Returns
    -------
    Any
        Replayed output.
    """

    return _raw(capture).BatchMatMulV2(
        x=inputs[0],
        y=inputs[1],
        adj_x=bool(capture.attrs.get("adj_x", False)),
        adj_y=bool(capture.attrs.get("adj_y", False)),
    )


def _replay_bias_add(capture: TFOpCapture, inputs: Sequence[Any]) -> Any:
    """Replay ``BiasAdd``.

    Parameters
    ----------
    capture
        Callback capture.
    inputs
        Rebuilt inputs.

    Returns
    -------
    Any
        Replayed output.
    """

    data_format = capture.attrs.get("data_format")
    kwargs = {} if data_format is None else {"data_format": _attr_str(data_format)}
    return _raw(capture).BiasAdd(value=inputs[0], bias=inputs[1], **kwargs)


def _replay_cast(capture: TFOpCapture, inputs: Sequence[Any]) -> Any:
    """Replay ``Cast``.

    Parameters
    ----------
    capture
        Callback capture.
    inputs
        Rebuilt inputs.

    Returns
    -------
    Any
        Replayed output.
    """

    return _raw(capture).Cast(
        x=inputs[0],
        DstT=capture.attrs["DstT"],
        Truncate=bool(capture.attrs.get("Truncate", False)),
    )


def _replay_concat_v2(capture: TFOpCapture, inputs: Sequence[Any]) -> Any:
    """Replay ``ConcatV2``.

    Parameters
    ----------
    capture
        Callback capture.
    inputs
        Rebuilt inputs.

    Returns
    -------
    Any
        Replayed output.
    """

    del capture
    return _import_tensorflow().raw_ops.ConcatV2(values=list(inputs[:-1]), axis=inputs[-1])


def _replay_conv2d(capture: TFOpCapture, inputs: Sequence[Any]) -> Any:
    """Replay ``Conv2D``.

    Parameters
    ----------
    capture
        Callback capture.
    inputs
        Rebuilt inputs.

    Returns
    -------
    Any
        Replayed output.
    """

    return _raw(capture).Conv2D(
        input=inputs[0],
        filter=inputs[1],
        strides=list(capture.attrs["strides"]),
        padding=_attr_str(capture.attrs["padding"]),
        explicit_paddings=list(capture.attrs.get("explicit_paddings", [])),
        data_format=_attr_str(capture.attrs.get("data_format", "NHWC")),
        dilations=list(capture.attrs.get("dilations", [1, 1, 1, 1])),
        use_cudnn_on_gpu=bool(capture.attrs.get("use_cudnn_on_gpu", True)),
    )


def _replay_einsum(capture: TFOpCapture, inputs: Sequence[Any]) -> Any:
    """Replay ``Einsum``.

    Parameters
    ----------
    capture
        Callback capture.
    inputs
        Rebuilt inputs.

    Returns
    -------
    Any
        Replayed output.
    """

    return _raw(capture).Einsum(
        inputs=list(inputs),
        equation=_attr_str(capture.attrs["equation"]),
    )


def _replay_matmul(capture: TFOpCapture, inputs: Sequence[Any]) -> Any:
    """Replay ``MatMul``.

    Parameters
    ----------
    capture
        Callback capture.
    inputs
        Rebuilt inputs.

    Returns
    -------
    Any
        Replayed output.
    """

    return _raw(capture).MatMul(
        a=inputs[0],
        b=inputs[1],
        transpose_a=bool(capture.attrs.get("transpose_a", False)),
        transpose_b=bool(capture.attrs.get("transpose_b", False)),
    )


def _replay_mean(capture: TFOpCapture, inputs: Sequence[Any]) -> Any:
    """Replay ``Mean``.

    Parameters
    ----------
    capture
        Callback capture.
    inputs
        Rebuilt inputs.

    Returns
    -------
    Any
        Replayed output.
    """

    return _raw(capture).Mean(
        input=inputs[0],
        axis=inputs[1],
        keep_dims=bool(capture.attrs.get("keep_dims", False)),
    )


def _replay_pool(capture: TFOpCapture, inputs: Sequence[Any]) -> Any:
    """Replay ``MaxPool`` or ``AvgPool``.

    Parameters
    ----------
    capture
        Callback capture.
    inputs
        Rebuilt inputs.

    Returns
    -------
    Any
        Replayed output.
    """

    raw_op = getattr(_raw(capture), capture.op_type)
    return raw_op(
        input=inputs[0],
        ksize=list(capture.attrs["ksize"]),
        strides=list(capture.attrs["strides"]),
        padding=_attr_str(capture.attrs["padding"]),
        explicit_paddings=list(capture.attrs.get("explicit_paddings", [])),
        data_format=_attr_str(capture.attrs.get("data_format", "NHWC")),
    )


def _attr_str(value: Any) -> str:
    """Return a TensorFlow string attr as text.

    Parameters
    ----------
    value
        TensorFlow callback attr value.

    Returns
    -------
    str
        Decoded text attr.
    """

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _optional_int_list(value: Any) -> list[int]:
    """Return a TensorFlow optional int-list attr.

    Parameters
    ----------
    value
        Callback attr value.

    Returns
    -------
    list[int]
        Empty list for omitted attrs, otherwise the integer list.
    """

    if value is None:
        return []
    return [int(item) for item in value]


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
        candidates = (array + 1, np.zeros_like(array))
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
    if np.issubdtype(left_array.dtype, np.floating):
        return bool(np.allclose(left_array, right_array, rtol=1e-5, atol=1e-6))
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
    return getattr(op, "out")


def _ops_by_label(trace: Any) -> dict[str, Any]:
    """Return materialized trace operations keyed by labels.

    Parameters
    ----------
    trace
        Materialized TensorFlow trace.

    Returns
    -------
    dict[str, Any]
        Operations keyed by raw, layer, and public labels.
    """

    result: dict[str, Any] = {}
    for op in getattr(trace, "layer_list", ()):
        for label in (
            getattr(op, "_label_raw", None),
            getattr(op, "layer_label", None),
            getattr(op, "label", None),
        ):
            if isinstance(label, str):
                result[label] = op
    return result


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


def _import_tensorflow() -> Any:
    """Import TensorFlow lazily.

    Returns
    -------
    Any
        Imported TensorFlow module.
    """

    import tensorflow as tf

    return tf


__all__ = [
    "TFValidationCounts",
    "TFValidationResult",
    "replay_allowlist",
    "validate_tf_trace",
    "validate_tf_trace_detailed",
]
