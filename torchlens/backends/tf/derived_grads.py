"""TensorFlow derived-gradient replay (leaf + exact op-level intermediates).

TensorFlow previews cannot capture true backward graphs, so derived gradients
follow the sibling preview pattern: after the trace is finalized, one auxiliary
forward replay runs under a persistent ``tf.GradientTape`` with the op-callback
session installed. Every float op output is ``tape.watch``-ed live at callback
time (spike-verified on TF 2.21: watching an output before its consumers run
records the downstream ops), the replay output must match the captured output
or the surface refuses, and only unambiguous 1:1 signature matches attach
``status="exact"`` intermediate records.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ...data_classes.derived_grad import (
    DerivedGradAccessor,
    DerivedGradRecord,
    IntermediateDerivedGradAccessor,
    IntermediateDerivedGradRecord,
)
from ...data_classes.trace import Trace
from ...ir.refs import DtypeRef
from .._finalize import mirror_param_derived_grads, stable_callable_name as _callable_identity
from .._validation_shared import float_replay_tolerances
from ..registry import BackendUnsupportedError
from .modules import TFModuleTree
from .op_callback_capture import TFEagerCaptureSession, TFOpCapture


@dataclass(frozen=True)
class GradOptions:
    """TensorFlow derived-gradient preview options.

    Parameters
    ----------
    loss_fn
        Optional callable mapping raw output to scalar loss.
    input_grad_argnums
        Positional input argument indexes to differentiate. ``None`` means all
        floating TensorFlow tensor positional inputs.
    intermediate_grads
        Whether to request exact op-level intermediate derived gradients.
    max_intermediate_grads
        Hard cap on attached intermediate records.
    """

    loss_fn: Callable[[Any], Any] | None = None
    input_grad_argnums: tuple[int, ...] | None = None
    intermediate_grads: bool = False
    max_intermediate_grads: int = 64

    def __init__(
        self,
        *,
        loss_fn: Callable[[Any], Any] | None = None,
        input_grad_argnums: Sequence[int] | None = None,
        intermediate_grads: bool = False,
        max_intermediate_grads: int = 64,
    ) -> None:
        """Initialize TensorFlow derived-gradient options."""

        if not isinstance(max_intermediate_grads, int) or max_intermediate_grads < 1:
            raise ValueError("max_intermediate_grads must be an integer >= 1")
        object.__setattr__(self, "loss_fn", loss_fn)
        normalized = None if input_grad_argnums is None else tuple(input_grad_argnums)
        object.__setattr__(self, "input_grad_argnums", normalized)
        object.__setattr__(self, "intermediate_grads", bool(intermediate_grads))
        object.__setattr__(self, "max_intermediate_grads", max_intermediate_grads)


@dataclass(frozen=True)
class TFGradLeaf:
    """One TensorFlow tensor leaf differentiated by the derived-gradient replay.

    Parameters
    ----------
    path
        Public derived-gradient record path.
    source
        Source group, either ``"inputs"`` or ``"params"``.
    argnum
        Positional model input argnum for inputs, or ``-1`` for params.
    input_argnum
        Input-relative argument index for input gradients.
    tensor
        Live TensorFlow tensor or resource variable to differentiate.
    """

    path: str
    source: str
    argnum: int
    input_argnum: int | None
    tensor: Any


@dataclass(frozen=True)
class TFIntermediateSignature:
    """Stable key used to match capture-time and replay-time intermediates.

    Parameters
    ----------
    func_call_id
        Stable callback-stream raw index.
    op_name
        Captured TensorFlow operation type.
    parent_labels
        Raw labels of tensor parents.
    module_stack
        Module-call stack labels such as ``"encoder:1"``.
    """

    func_call_id: int
    op_name: str
    parent_labels: tuple[str, ...]
    module_stack: tuple[str, ...]


@dataclass(frozen=True)
class TFIntermediateCandidate:
    """Replay-time intermediate observed by the output tap.

    Parameters
    ----------
    signature
        Stable match key.
    label_raw
        Replay-stream raw label.
    value
        Replay-time TensorFlow tensor.
    grad
        Gradient of the replay loss with respect to ``value``.
    """

    signature: TFIntermediateSignature
    label_raw: str
    value: Any
    grad: Any | None


def attach_tf_derived_grads(
    *,
    tf: Any,
    trace: Trace,
    callable_obj: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    captured_output: Any,
    grad_options: GradOptions,
    module_tree: TFModuleTree | None,
) -> None:
    """Compute and attach TensorFlow leaf and optional intermediate derived grads.

    Parameters
    ----------
    tf
        Imported TensorFlow module.
    trace
        Finalized TensorFlow trace receiving derived-gradient accessors.
    callable_obj
        Captured TensorFlow callable.
    args
        Positional call arguments used for capture.
    kwargs
        Keyword call arguments used for capture.
    captured_output
        Raw output from the captured forward call.
    grad_options
        TensorFlow derived-gradient options.
    module_tree
        Object-module discovery tree when object-module attribution is active.

    Returns
    -------
    None
        ``trace.derived_grads`` and optionally
        ``trace.intermediate_derived_grads`` are populated.
    """

    input_argnums = _normalize_tf_input_grad_argnums(
        tf=tf,
        args=args,
        input_grad_argnums=grad_options.input_grad_argnums,
    )
    leaves = _tf_input_grad_leaves(tf, args, input_argnums)
    param_leaves = _tf_param_grad_leaves(trace)
    all_leaves = (*leaves, *param_leaves)
    if not all_leaves and not grad_options.intermediate_grads:
        raise ValueError(
            "TensorFlow derived gradients require at least one floating tensor "
            "input or module parameter."
        )
    previous_call_counts = dict(module_tree.call_counts) if module_tree is not None else None
    previous_forward_args = (
        dict(module_tree.forward_args_by_call) if module_tree is not None else None
    )
    candidates: list[tuple[TFOpCapture, tuple[str, ...]]] = []
    tape = tf.GradientTape(persistent=True, watch_accessed_variables=False)
    try:
        if module_tree is not None:
            module_tree.call_counts.clear()
            module_tree.forward_args_by_call.clear()
        replay_session = TFEagerCaptureSession(
            tf=tf,
            callable_obj=callable_obj,
            args=args,
            kwargs=kwargs,
            module_tree=module_tree,
            save_payloads=False,
            output_tap=_output_tap(tf, tape, candidates, grad_options),
        )
        with tape:
            for leaf in all_leaves:
                tape.watch(leaf.tensor)
            replay_result = replay_session.run()
            loss = (
                grad_options.loss_fn(replay_result.output)
                if grad_options.loss_fn
                else replay_result.output
            )
        if not _is_scalar_float_tf_tensor(loss, tf):
            raise ValueError(
                "TensorFlow derived gradients require loss_fn(raw_output) to be a "
                "scalar float tensor unless the traced output is already scalar."
            )
        if not _tf_trees_close(tf, replay_result.output, captured_output):
            raise ValueError(
                "TensorFlow derived gradient run raw output diverged from captured "
                "raw output; refusing to expose trace.derived_grads."
            )
        intermediate_accessor = IntermediateDerivedGradAccessor()
        if grad_options.intermediate_grads:
            intermediate_accessor = _records_for_intermediate_tf_grads(
                tf=tf,
                tape=tape,
                trace=trace,
                loss=loss,
                candidates=candidates,
                grad_options=grad_options,
            )
        leaf_grads: Sequence[Any | None]
        if all_leaves:
            leaf_grads = tape.gradient(
                loss,
                [leaf.tensor for leaf in all_leaves],
                unconnected_gradients=tf.UnconnectedGradients.NONE,
            )
        else:
            leaf_grads = ()
        records = _records_for_tf_leaf_grads(
            leaves=all_leaves,
            grads=leaf_grads,
            provenance={
                "backend": "tf",
                "kind": "derived_gradient",
                "mechanism": "tf.GradientTape",
                "loss_fn": _callable_identity(grad_options.loss_fn),
                "persistent_tape": True,
            },
        )
    finally:
        del tape
        if module_tree is not None and previous_call_counts is not None:
            module_tree.call_counts.clear()
            module_tree.call_counts.update(previous_call_counts)
        if module_tree is not None and previous_forward_args is not None:
            module_tree.forward_args_by_call.clear()
            module_tree.forward_args_by_call.update(previous_forward_args)
    trace.derived_grads = DerivedGradAccessor(records)
    if grad_options.intermediate_grads:
        trace.intermediate_derived_grads = intermediate_accessor
    mirror_param_derived_grads(trace, records)


def _output_tap(
    tf: Any,
    tape: Any,
    candidates: list[tuple[TFOpCapture, tuple[str, ...]]],
    grad_options: GradOptions,
) -> Callable[[TFOpCapture, tuple[str, ...]], None]:
    """Build the per-output replay tap that watches float intermediates.

    Parameters
    ----------
    tf
        Imported TensorFlow module.
    tape
        Active persistent gradient tape.
    candidates
        Accumulator receiving ``(capture, module_stack)`` pairs.
    grad_options
        TensorFlow derived-gradient options.

    Returns
    -------
    Callable[[TFOpCapture, tuple[str, ...]], None]
        Tap invoked by the replay capture session for every op output.
    """

    def tap(capture: TFOpCapture, module_stack: tuple[str, ...]) -> None:
        """Watch one replay op output and record it as a match candidate."""

        if not grad_options.intermediate_grads:
            return
        dtype = getattr(capture.output_tensor, "dtype", None)
        if dtype is None or not bool(getattr(dtype, "is_floating", False)):
            return
        tape.watch(capture.output_tensor)
        candidates.append((capture, module_stack))

    return tap


def _records_for_intermediate_tf_grads(
    *,
    tf: Any,
    tape: Any,
    trace: Trace,
    loss: Any,
    candidates: Sequence[tuple[TFOpCapture, tuple[str, ...]]],
    grad_options: GradOptions,
) -> IntermediateDerivedGradAccessor:
    """Build exact TensorFlow op-level derived-gradient records.

    Parameters
    ----------
    tf
        Imported TensorFlow module.
    tape
        Persistent tape that recorded the replay forward.
    trace
        Finalized trace whose saved ops define the attachment set.
    loss
        Replay-time scalar loss recorded on ``tape``.
    candidates
        Tap-recorded replay outputs with module stacks.
    grad_options
        TensorFlow derived-gradient options.

    Returns
    -------
    IntermediateDerivedGradAccessor
        Exact records keyed by pass-qualified op label.
    """

    replay_values = [capture.output_tensor for capture, _stack in candidates]
    replay_grads: Sequence[Any | None]
    if replay_values:
        replay_grads = tape.gradient(
            loss,
            replay_values,
            unconnected_gradients=tf.UnconnectedGradients.NONE,
        )
    else:
        replay_grads = ()
    replay_groups: dict[TFIntermediateSignature, list[TFIntermediateCandidate]] = defaultdict(list)
    for (capture, module_stack), grad in zip(candidates, replay_grads, strict=True):
        signature = TFIntermediateSignature(
            func_call_id=_capture_raw_index(capture.label_raw),
            op_name=capture.op_type,
            parent_labels=_capture_parent_labels(capture),
            module_stack=module_stack,
        )
        replay_groups[signature].append(
            TFIntermediateCandidate(
                signature=signature,
                label_raw=capture.label_raw,
                value=capture.output_tensor,
                grad=grad,
            )
        )
    trace_groups = _tf_trace_intermediate_signatures(trace)
    records: dict[str, IntermediateDerivedGradRecord] = {}
    for signature, ops in trace_groups.items():
        if len(ops) != 1:
            continue
        replay_candidates = replay_groups.get(signature, [])
        if len(replay_candidates) != 1:
            continue
        candidate = replay_candidates[0]
        if candidate.grad is None:
            continue
        op = ops[0]
        records[op.label] = IntermediateDerivedGradRecord(
            op_label=op.label,
            layer_label=op.layer_label,
            aval=(
                f"Tensor(shape={tuple(getattr(candidate.grad, 'shape', ()))}, "
                f"dtype={getattr(candidate.grad, 'dtype', None)})"
            ),
            dtype_ref=DtypeRef(backend="tf", name=str(getattr(candidate.grad, "dtype", ""))),
            grad=candidate.grad,
            provenance={
                "backend": "tf",
                "kind": "intermediate_derived_gradient",
                "mechanism": "tf.GradientTape_op_callback_replay",
                "loss_id": _callable_identity(grad_options.loss_fn),
                "save_predicate_id": "trace.saved_ops",
                "status": "exact",
                "match_key": "func_call_id+op_type+parents+module_stack",
                "max_intermediate_grads": grad_options.max_intermediate_grads,
            },
        )
    if len(records) > grad_options.max_intermediate_grads:
        raise BackendUnsupportedError(
            "TensorFlow intermediate derived gradients are capped at "
            f"{grad_options.max_intermediate_grads} attached records; got "
            f"{len(records)}."
        )
    return IntermediateDerivedGradAccessor(records)


def _tf_trace_intermediate_signatures(
    trace: Trace,
) -> dict[TFIntermediateSignature, list[Any]]:
    """Group finalized TensorFlow ops by replay-match signature.

    Parameters
    ----------
    trace
        Finalized TensorFlow trace.

    Returns
    -------
    dict[TFIntermediateSignature, list[Any]]
        Trace ops grouped by stable replay key.
    """

    groups: dict[TFIntermediateSignature, list[Any]] = defaultdict(list)
    # Replay-side signatures speak RAW label space (capture input records).
    # Recurrence grouping rewrites ``op.parents`` to final labels -- the
    # pass-qualified ``op.label`` for a multi-pass referenced layer, but the
    # BARE ``op.layer_label`` (torch parity) for a single-pass one -- so
    # parents are resolved back to raw space before signature construction.
    # Both final spellings must resolve: the pass-qualified key always, and
    # the bare key too for single-pass ops (unambiguous there; omitted for
    # multi-pass ops, where the bare label would collide across passes and
    # ``op.parents`` never uses it anyway). Without this every grouped
    # intermediate would silently fail to match its replay candidate.
    final_to_raw: dict[str, str] = {}
    for op in getattr(trace, "layer_list", ()):
        raw_label = getattr(op, "_label_raw", None)
        if not isinstance(raw_label, str):
            continue
        label = getattr(op, "label", None)
        if isinstance(label, str):
            final_to_raw[label] = raw_label
        layer_label = getattr(op, "layer_label", None)
        if isinstance(layer_label, str) and int(getattr(op, "num_passes", 1)) == 1:
            final_to_raw[layer_label] = raw_label
    for op in getattr(trace, "layer_list", ()):
        if bool(getattr(op, "is_input", False)) or not bool(
            getattr(op, "has_saved_activation", False)
        ):
            continue
        if not _is_float_dtype_text(str(getattr(op, "dtype", ""))):
            continue
        signature = TFIntermediateSignature(
            func_call_id=int(getattr(op, "func_call_id", 0)),
            op_name=str(getattr(op, "func_name", "")),
            parent_labels=tuple(
                dict.fromkeys(
                    final_to_raw.get(str(parent), str(parent))
                    for parent in getattr(op, "parents", ())
                )
            ),
            module_stack=tuple(str(module) for module in getattr(op, "modules", ())),
        )
        groups[signature].append(op)
    return groups


def _capture_raw_index(label_raw: str) -> int:
    """Return the callback-stream raw index encoded in a raw label.

    Parameters
    ----------
    label_raw
        Raw label such as ``"relu_1_6_raw"``.

    Returns
    -------
    int
        Raw stream index, or ``0`` when unparseable.
    """

    stem = label_raw[: -len("_raw")] if label_raw.endswith("_raw") else label_raw
    _prefix, _sep, index_text = stem.rpartition("_")
    return int(index_text) if index_text.isdigit() else 0


def _capture_parent_labels(capture: TFOpCapture) -> tuple[str, ...]:
    """Return raw parent labels recorded for one replay op capture.

    Parameters
    ----------
    capture
        Replay op capture.

    Returns
    -------
    tuple[str, ...]
        Producer or source raw labels in input order, deduplicated.
    """

    labels: list[str] = []
    for input_capture in capture.inputs:
        label = input_capture.producer_label_raw or input_capture.source_label_raw
        if label is not None and label not in labels:
            labels.append(label)
    return tuple(labels)


def _normalize_tf_input_grad_argnums(
    *,
    tf: Any,
    args: Sequence[Any],
    input_grad_argnums: Sequence[int] | None,
) -> tuple[int, ...]:
    """Normalize TensorFlow derived-gradient input argnums.

    Parameters
    ----------
    tf
        Imported TensorFlow module.
    args
        Positional model inputs.
    input_grad_argnums
        User-requested argnums, or ``None`` for all float tensor inputs.

    Returns
    -------
    tuple[int, ...]
        Validated positional input argnums.
    """

    if input_grad_argnums is None:
        return tuple(
            index
            for index, value in enumerate(args)
            if any(
                _is_float_tf_tensor(tensor, tf)
                for _path, tensor in _iter_tf_tensors_with_paths(value, tf)
            )
        )
    normalized = tuple(int(index) for index in input_grad_argnums)
    for index in normalized:
        if index < 0 or index >= len(args):
            raise ValueError(f"TensorFlow input_grad_argnums contains out-of-range argnum {index}.")
    return normalized


def _tf_input_grad_leaves(
    tf: Any,
    args: Sequence[Any],
    input_argnums: Sequence[int],
) -> tuple[TFGradLeaf, ...]:
    """Return differentiable TensorFlow input leaves.

    Parameters
    ----------
    tf
        Imported TensorFlow module.
    args
        Positional model inputs.
    input_argnums
        Positional argnums to differentiate.

    Returns
    -------
    tuple[TFGradLeaf, ...]
        Floating tensor input leaves.
    """

    leaves: list[TFGradLeaf] = []
    for argnum in input_argnums:
        for path, tensor in _iter_tf_tensors_with_paths(args[argnum], tf):
            if not _is_float_tf_tensor(tensor, tf):
                continue
            suffix = ".".join(str(part) for part in path)
            record_path = f"inputs.{argnum}" if suffix == "" else f"inputs.{argnum}.{suffix}"
            leaves.append(
                TFGradLeaf(
                    path=record_path,
                    source="inputs",
                    argnum=argnum,
                    input_argnum=argnum,
                    tensor=tensor,
                )
            )
    return tuple(leaves)


def _tf_param_grad_leaves(trace: Trace) -> tuple[TFGradLeaf, ...]:
    """Return differentiable TensorFlow parameter leaves from ``trace.params``.

    Parameters
    ----------
    trace
        Finalized TensorFlow trace.

    Returns
    -------
    tuple[TFGradLeaf, ...]
        Parameter leaves with stable ``params.<address>`` paths.
    """

    leaves: list[TFGradLeaf] = []
    for address, param in trace.params.items():
        variable = _underlying_tf_variable(getattr(param, "_param_ref", None))
        if variable is None or not _is_float_dtype_text(str(getattr(variable, "dtype", ""))):
            continue
        leaves.append(
            TFGradLeaf(
                path=f"params.{address}",
                source="params",
                argnum=-1,
                input_argnum=None,
                tensor=variable,
            )
        )
    return tuple(leaves)


def _underlying_tf_variable(value: Any) -> Any | None:
    """Return the TensorFlow resource variable behind a param reference.

    Keras 3 wraps backend state in ``keras.Variable`` objects whose ``.value``
    is the live ``tf.Variable``; raw TensorFlow modules expose ``tf.Variable``
    directly.

    Parameters
    ----------
    value
        Candidate variable reference.

    Returns
    -------
    Any | None
        Watchable TensorFlow variable or tensor, or ``None``.
    """

    if value is None:
        return None
    inner = getattr(value, "value", None)
    if inner is not None and not callable(inner) and hasattr(inner, "dtype"):
        return inner
    return value if hasattr(value, "dtype") else None


def _records_for_tf_leaf_grads(
    *,
    leaves: Sequence[TFGradLeaf],
    grads: Sequence[Any | None],
    provenance: Mapping[str, Any],
) -> dict[str, DerivedGradRecord]:
    """Build derived-gradient records for TensorFlow leaves.

    Parameters
    ----------
    leaves
        Differentiated leaves.
    grads
        Gradients returned by ``tape.gradient``.
    provenance
        Shared provenance metadata.

    Returns
    -------
    dict[str, DerivedGradRecord]
        Records keyed by stable leaf path. ``None`` gradients are skipped.
    """

    records: dict[str, DerivedGradRecord] = {}
    for leaf, grad in zip(leaves, grads, strict=True):
        if grad is None:
            continue
        records[leaf.path] = DerivedGradRecord(
            path=leaf.path,
            source=leaf.source,
            argnum=leaf.argnum,
            input_argnum=leaf.input_argnum,
            aval=(
                f"Tensor(shape={tuple(getattr(grad, 'shape', ()))}, "
                f"dtype={getattr(grad, 'dtype', None)})"
            ),
            dtype_ref=DtypeRef(backend="tf", name=str(getattr(grad, "dtype", ""))),
            grad=grad,
            provenance=provenance,
        )
    return records


def _iter_tf_tensors_with_paths(
    value: Any,
    tf: Any,
    path: tuple[Any, ...] = (),
) -> list[tuple[tuple[Any, ...], Any]]:
    """Return TensorFlow tensor leaves with container paths.

    Parameters
    ----------
    value
        Candidate nested value.
    tf
        Imported TensorFlow module.
    path
        Path accumulated so far.

    Returns
    -------
    list[tuple[tuple[Any, ...], Any]]
        Path and tensor pairs.
    """

    if _is_tf_tensor_like(value, tf):
        return [(path, value)]
    if isinstance(value, (list, tuple)):
        leaves: list[tuple[tuple[Any, ...], Any]] = []
        for index, item in enumerate(value):
            leaves.extend(_iter_tf_tensors_with_paths(item, tf, (*path, index)))
        return leaves
    if isinstance(value, dict):
        leaves = []
        for key, item in value.items():
            leaves.extend(_iter_tf_tensors_with_paths(item, tf, (*path, key)))
        return leaves
    return []


def _is_tf_tensor_like(value: Any, tf: Any) -> bool:
    """Return whether ``value`` is a TensorFlow tensor or variable.

    Parameters
    ----------
    value
        Candidate value.
    tf
        Imported TensorFlow module.

    Returns
    -------
    bool
        True for tensors and variables.
    """

    tensor_type = getattr(tf, "Tensor", None)
    variable_type = getattr(tf, "Variable", None)
    return bool(
        (tensor_type is not None and isinstance(value, tensor_type))
        or (variable_type is not None and isinstance(value, variable_type))
    )


def _is_float_tf_tensor(value: Any, tf: Any) -> bool:
    """Return whether ``value`` is a floating TensorFlow tensor.

    Parameters
    ----------
    value
        Candidate value.
    tf
        Imported TensorFlow module.

    Returns
    -------
    bool
        True when ``value`` is a floating TensorFlow tensor or variable.
    """

    return _is_tf_tensor_like(value, tf) and _is_float_dtype_text(str(getattr(value, "dtype", "")))


def _is_float_dtype_text(dtype: str) -> bool:
    """Return whether a dtype string names a floating dtype.

    Parameters
    ----------
    dtype
        Backend dtype string.

    Returns
    -------
    bool
        True for float and bfloat dtypes.
    """

    lowered = dtype.lower()
    return "float" in lowered or "bfloat" in lowered


def _is_scalar_float_tf_tensor(value: Any, tf: Any) -> bool:
    """Return whether ``value`` is a scalar floating TensorFlow tensor.

    Parameters
    ----------
    value
        Candidate loss value.
    tf
        Imported TensorFlow module.

    Returns
    -------
    bool
        True for float tensors with shape ``()`` or ``(1,)``.
    """

    if not _is_float_tf_tensor(value, tf):
        return False
    shape = tuple(int(dim) for dim in getattr(value, "shape", ()))
    return shape == () or shape == (1,)


def _tf_trees_close(tf: Any, left: Any, right: Any) -> bool:
    """Return whether two TensorFlow output trees are numerically close.

    Parameters
    ----------
    tf
        Imported TensorFlow module.
    left
        Replay output tree.
    right
        Captured output tree.

    Returns
    -------
    bool
        True when container structure, shapes, dtypes, and values match.
    """

    if _is_tf_tensor_like(left, tf) or _is_tf_tensor_like(right, tf):
        return (
            _is_tf_tensor_like(left, tf)
            and _is_tf_tensor_like(right, tf)
            and _tf_values_close(left, right)
        )
    if isinstance(left, tuple) and isinstance(right, tuple) and len(left) == len(right):
        return all(
            _tf_trees_close(tf, l_item, r_item) for l_item, r_item in zip(left, right, strict=True)
        )
    if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        return all(
            _tf_trees_close(tf, l_item, r_item) for l_item, r_item in zip(left, right, strict=True)
        )
    if isinstance(left, dict) and isinstance(right, dict) and set(left) == set(right):
        return all(_tf_trees_close(tf, left[key], right[key]) for key in left)
    return bool(left == right)


def _tf_values_close(left: Any, right: Any) -> bool:
    """Return whether two TensorFlow tensors are close.

    Parameters
    ----------
    left
        Replay tensor.
    right
        Captured tensor.

    Returns
    -------
    bool
        True when shape, dtype, and values match within preview tolerances.
    """

    if tuple(getattr(left, "shape", ())) != tuple(getattr(right, "shape", ())):
        return False
    if str(getattr(left, "dtype", "")) != str(getattr(right, "dtype", "")):
        return False
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if _is_float_dtype_text(str(getattr(left, "dtype", ""))):
        # Per-dtype ULP-derived bands (ported paddle/mlx validation-oracle
        # derivation): the former dtype-blind fp32 pair (rtol 1e-5 /
        # atol 1e-6) blessed fp64 corruption ~4.5e9 of its own ULPs and TOTAL
        # corruption of every element below 1e-6, while false-failing
        # legitimate one-ULP fp16 storage rounding.
        try:
            finfo = np.finfo(left_array.dtype)
        except (TypeError, ValueError):
            # Extended floats NumPy has no finfo row for (e.g. tf.bfloat16
            # payloads whose ml_dtypes registration is unavailable) compare
            # exactly: never derive a band from the wrong dtype's finfo.
            return bool(np.array_equal(left_array, right_array))
        rtol, atol = float_replay_tolerances(finfo)
        return bool(np.allclose(left_array, right_array, rtol=rtol, atol=atol, equal_nan=True))
    return bool(np.array_equal(left_array, right_array))


__all__ = [
    "GradOptions",
    "TFGradLeaf",
    "TFIntermediateSignature",
    "attach_tf_derived_grads",
]
