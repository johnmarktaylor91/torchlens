"""TensorFlow replay adapter: thin wrappers over the TensorFlow validator's replay.

Each function of :class:`._contract.ReplayAdapter` is implemented on top of the
per-op raw-op replay that TensorFlow validation already runs
(:func:`..tf._raw_op_replay._replay_raw_op`). The validator's dispatch table is
an explicit allowlist; the few extra kernels here (:data:`_EXTRA_KERNEL_OPS`)
replay op types the validator classifies but does not replay. ``close`` is the
validator's own comparison, imported.

Op identities are closed: a replayable op's identity is its TensorFlow op type
(one of :data:`ALLOWLIST`); every other recorded op gets the identity
``tf.inert``. An inert recipe never executes: it only rebuilds the runtime
capture so the unchanged validator sees the whole capture list after load.

Parent values follow the contract: one framework value per
:attr:`OpRecipe.inputs` occurrence, in callback input order, repeats kept.
Literal inputs (axes, shapes, factory constants) are recorded as data and are
not parent occurrences. Framework imports stay inside functions.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from ... import _state
from .._validation_shared import ops_by_label
from ..tf._helper_hooks import tf_helper_hook
from ..tf._raw_op_replay import _replay_raw_op
from ..tf.op_callback_capture import TFInputCapture, TFOpCapture, TFUnresolvedProducer
from ..tf.validation import (
    _CONTROL_REGION_OPS,
    _EFFECT_DENIED_OPS,
    _payloads_close,
    _tensor_to_numpy,
)
from ._contract import InputOccurrence, OpRecipe, ReplayRefusal, validate_recipe

BACKEND = "tf"
INERT_IDENTITY = "tf.inert"

#: Op types the TensorFlow validator classifies but does not replay; the
#: adapter replays them with the kernels in :func:`_extra_kernels`.
_EXTRA_KERNEL_OPS = frozenset(
    {"Gelu", "Identity", "Pack", "Rank", "Shape", "Size", "StopGradient", "StridedSlice"}
)

#: Raw-op kernels the validator's own dispatch holds (kept equal by a test).
_VALIDATOR_KERNEL_OPS = frozenset(
    {
        "AddV2", "AvgPool", "BatchMatMulV2", "BiasAdd", "BroadcastTo", "Cast", "ConcatV2",
        "Conv2D", "DepthwiseConv2dNative", "Einsum", "Elu", "Erf", "Erfc", "Exp", "ExpandDims",
        "Fill", "GatherV2", "Greater", "Less", "Log", "MatMul", "MaxPool", "Maximum", "Mean",
        "Minimum", "Mul", "Neg", "Pad", "Pow", "Prod", "RealDiv", "Relu", "Relu6", "Reshape",
        "Rsqrt", "SelectV2", "Sigmoid", "Sin", "Slice", "Softmax", "Sqrt", "SquaredDifference",
        "Squeeze", "Sub", "Sum", "Tanh", "Tile", "Transpose",
    }
)  # fmt: skip

KERNEL_OPS = _VALIDATOR_KERNEL_OPS | _EXTRA_KERNEL_OPS
ALLOWLIST: frozenset[str] = KERNEL_OPS | {INERT_IDENTITY}
"""Closed set of recipe op identities this adapter accepts."""

_RANDOM_PREFIXES = (
    "Random",
    "StatelessRandom",
    "StatelessTruncated",
    "TruncatedNormal",
    "Multinomial",
    "StatelessMultinomial",
    "ParameterizedTruncatedNormal",
)
_RANDOM_REASON = "random op: replaying its draw needs random-state capture (the rng lane, X2)"
_RESOURCE_REASON = (
    "resource read: a ReadVariableOp's value is the variable (a parameter source), "
    "not a function of recorded parents"
)
_MAX_LITERAL_ELEMENTS = 65536
_REQUIRED_ATTRS = ("tf_op_type", "tf_attrs", "output_index", "init_op", "input_sources")

# --- lookup ----------------------------------------------------------------


def _index(trace: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return ``(ops by raw and public label, captures by raw label)`` for ``trace``.

    Built per call and never cached, so the adapter holds no module state and a
    capture list replaced by :func:`bind_recipes` is always seen.
    """

    captures = getattr(trace, "_tf_op_captures", ())
    ops = dict(ops_by_label(trace))
    by_raw = {c.label_raw: c for c in captures if isinstance(c, TFOpCapture)}
    return ops, by_raw


def _resolve(trace: Any, op: Any) -> tuple[Any, Any]:
    """Return ``(op record, capture or None)`` for an op record, capture or label."""

    ops, by_raw = _index(trace)
    if isinstance(op, str):
        record = ops.get(op)
        if record is None:
            raise ReplayRefusal("unknown_label", "no op with this label in the trace", op_label=op)
        op = record
    raw = getattr(op, "_label_raw", None) or getattr(op, "label_raw", None)
    record = ops.get(raw, op) if isinstance(raw, str) else op
    return record, by_raw.get(raw) if isinstance(raw, str) else None


def _public_label(ops: Mapping[str, Any], raw: str) -> str:
    """Public label of the op whose raw label is ``raw`` (``raw`` when unknown)."""

    record = ops.get(raw)
    label = getattr(record, "label", None) if record is not None else None
    return label if isinstance(label, str) else raw


# --- can_replay ------------------------------------------------------------


def can_replay(trace: Any, op: Any) -> str | None:
    """``None`` when ``op`` replays from parent values, else the reason it cannot."""

    record, capture = _resolve(trace, op)
    if capture is None:
        return _no_capture_reason(record)
    return _capture_reason(capture)


def _no_capture_reason(record: Any) -> str:
    """Reason for an op that has no runtime capture."""

    if bool(getattr(record, "is_input", False)) or getattr(record, "func_name", None) in {
        "input",
        "constant",
    }:
        return "source op: inputs and closure constants are leaves, not replayed"
    return "no runtime capture for this op (a loaded trace needs bind_recipes first)"


def _type_reason(op_type: str) -> str | None:
    """Reason a TensorFlow op type is not replayable, or ``None``."""

    if op_type.startswith(_RANDOM_PREFIXES):
        return _RANDOM_REASON
    if op_type.startswith(("__inference_", "region:")) or op_type in _CONTROL_REGION_OPS:
        return f"opaque or control-flow region {op_type!r}: no per-op replay exists inside it"
    if op_type == "ReadVariableOp":
        return _RESOURCE_REASON
    if op_type in _EFFECT_DENIED_OPS:
        return f"stateful effect op {op_type!r}: replay would repeat a side effect"
    if op_type not in KERNEL_OPS:
        return f"unlisted raw op {op_type!r}: no replay kernel in the closed allowlist"
    return None


def _capture_reason(capture: Any) -> str | None:
    """Reason a capture cannot replay, or ``None``."""

    reason = _type_reason(capture.op_type)
    if reason is not None:
        return reason
    if capture.output_index != 0:
        return f"output slot {capture.output_index} of a multi-output op: kernels are single-output"
    if any(isinstance(value, _OpaqueAttr) for value in capture.attrs.values()):
        return "an attribute of this op was not recorded as data"
    for item in capture.inputs:
        if item.producer_label_raw is not None or item.source_label_raw is not None:
            continue
        if item.source_kind is None:
            return f"input {item.input_index} has no producer and no source"
        if _literal_array(item.tensor) is None:
            return f"literal input {item.input_index} is not available as data"
    return None


# --- forward and vjp -------------------------------------------------------


def forward(trace: Any, op: Any, parent_values: Sequence[Any]) -> tuple[Any, ...]:
    """Re-run ``op`` from ``parent_values`` through the validator's raw-op replay.

    Returns one TensorFlow tensor per output slot. The replay runs eagerly, so a
    surrounding ``tf.GradientTape`` differentiates it.
    """

    capture = _require_replayable(trace, op)
    return (_dispatch(capture, _assemble_inputs(capture, parent_values)),)


def vjp(
    trace: Any, op: Any, parent_values: Sequence[Any], cotangents: Sequence[Any]
) -> tuple[Any | None, ...]:
    """TensorFlow's own derivative of ``op``: a ``GradientTape`` over its single raw op.

    Returns one cotangent per parent occurrence, ``None`` for a non-float parent,
    an unconnected parent, or when every output cotangent is ``None``. Repeated
    parents are distinct occurrences: each gets its own gradient.
    """

    import tensorflow as tf  # deferred: optional dependency

    capture = _require_replayable(trace, op)
    if len(cotangents) != 1:
        raise ReplayRefusal(
            "arity", f"expected 1 cotangent, got {len(cotangents)}", op_label=capture.label_raw
        )
    occurrences = [tf.identity(tf.convert_to_tensor(value)) for value in parent_values]
    if cotangents[0] is None:
        return tuple(None for _ in occurrences)
    watched = [value for value in occurrences if value.dtype.is_floating or value.dtype.is_complex]
    with tf.GradientTape(persistent=False, watch_accessed_variables=False) as tape:
        for value in watched:
            tape.watch(value)
        (output,) = forward(trace, capture, occurrences)
    if not watched:
        return tuple(None for _ in occurrences)
    seed = tf.convert_to_tensor(cotangents[0], dtype=output.dtype)
    grads = iter(tape.gradient(output, watched, output_gradients=seed))
    by_id = {id(value): _dense(tf, next(grads)) for value in watched}
    return tuple(by_id.get(id(value)) for value in occurrences)


def _dense(tf: Any, grad: Any) -> Any:
    """Densify an ``IndexedSlices`` gradient (``GatherV2`` params)."""

    if grad is None:
        return None
    return tf.convert_to_tensor(grad)


def _require_replayable(trace: Any, op: Any) -> Any:
    """Return the op's capture, or raise :class:`ReplayRefusal` naming why it cannot replay."""

    if _is_capture(op):
        capture = op
    else:
        record, capture = _resolve(trace, op)
        if capture is None:
            label = getattr(record, "label", None)
            raise ReplayRefusal("cannot_replay", _no_capture_reason(record), op_label=label)
    reason = _capture_reason(capture)
    if reason is not None:
        raise ReplayRefusal("cannot_replay", reason, op_label=capture.label_raw)
    return capture


def _is_capture(value: Any) -> bool:
    """Whether ``value`` is a TensorFlow callback capture."""

    return isinstance(value, TFOpCapture)


def _assemble_inputs(capture: Any, parent_values: Sequence[Any]) -> list[Any]:
    """Callback inputs in input order: parent values for graph parents, literals otherwise."""

    import tensorflow as tf  # deferred: optional dependency

    ordered = sorted(capture.inputs, key=lambda item: item.input_index)
    parents = [i for i in ordered if i.producer_label_raw or i.source_label_raw]
    if len(parents) != len(parent_values):
        raise ReplayRefusal(
            "arity",
            f"takes {len(parents)} parent value(s), got {len(parent_values)}",
            op_label=capture.label_raw,
        )
    values = iter(parent_values)
    inputs: list[Any] = []
    for item in ordered:
        if item.producer_label_raw or item.source_label_raw:
            inputs.append(tf.convert_to_tensor(next(values)))
        else:
            inputs.append(tf.convert_to_tensor(_literal_array(item.tensor)))
    return inputs


def _dispatch(capture: Any, inputs: list[Any]) -> Any:
    """Run the validator's kernel for ``capture``, or this module's extra kernel."""

    if capture.op_type in _VALIDATOR_KERNEL_OPS:
        return _replay_raw_op(capture, inputs)
    with _state.pause_logging():
        return _extra_kernels()[capture.op_type](capture, inputs)


def _extra_kernels() -> dict[str, Callable[[Any, Sequence[Any]], Any]]:
    """Kernels for op types the validator classifies but does not replay."""

    import tensorflow as tf  # deferred: optional dependency

    raw = tf.raw_ops

    def _out_type(item: Any) -> Any:
        value = item.attrs.get("out_type")
        return tf.int32 if value is None else tf.dtypes.as_dtype(value)

    def _mask(item: Any, name: str) -> int:
        return int(item.attrs.get(name) or 0)

    return {
        "Gelu": lambda c, a: raw.Gelu(
            features=a[0], approximate=bool(c.attrs.get("approximate", False))
        ),
        "Identity": lambda c, a: raw.Identity(input=a[0]),
        "Pack": lambda c, a: raw.Pack(values=list(a), axis=int(c.attrs.get("axis") or 0)),
        "Rank": lambda c, a: raw.Rank(input=a[0]),
        "Shape": lambda c, a: raw.Shape(input=a[0], out_type=_out_type(c)),
        "Size": lambda c, a: raw.Size(input=a[0], out_type=_out_type(c)),
        "StopGradient": lambda c, a: raw.StopGradient(input=a[0]),
        "StridedSlice": lambda c, a: raw.StridedSlice(
            input=a[0],
            begin=a[1],
            end=a[2],
            strides=a[3],
            begin_mask=_mask(c, "begin_mask"),
            end_mask=_mask(c, "end_mask"),
            ellipsis_mask=_mask(c, "ellipsis_mask"),
            new_axis_mask=_mask(c, "new_axis_mask"),
            shrink_axis_mask=_mask(c, "shrink_axis_mask"),
        ),
    }


# --- array helpers ---------------------------------------------------------


def zeros_like(value: Any) -> Any:
    """TensorFlow zeros with ``value``'s shape and dtype."""

    import tensorflow as tf  # deferred: optional dependency

    return tf.zeros_like(tf.convert_to_tensor(value))


def add(left: Any, right: Any) -> Any:
    """Elementwise TensorFlow sum (gradient accumulation); densifies ``IndexedSlices``."""

    import tensorflow as tf  # deferred: optional dependency

    return tf.math.add(tf.convert_to_tensor(left), tf.convert_to_tensor(right))


def apply_helper(trace: Any, helper: Any, value: Any) -> Any:
    """Apply a built-in edit helper to ``value`` with TensorFlow ops (finished-trace edits).

    Delegates to the live-edit helper table, so ``fork().do`` and ``intervene=``
    compute the same replacement; an unsupported helper refuses typed there.
    """

    import tensorflow as tf  # deferred: optional dependency

    del trace
    return tf_helper_hook(helper, tf)(tf.convert_to_tensor(value))


def to_host(value: Any) -> np.ndarray:
    """Host NumPy copy of a TensorFlow value (the validator's own conversion)."""

    return np.array(_tensor_to_numpy(value), copy=True)


def close(expected: Any, actual: Any) -> bool:
    """The TensorFlow validator's payload comparison, imported unchanged."""

    return bool(_payloads_close(actual, expected))


# --- literals and attrs as data --------------------------------------------


class _OpaqueAttr:
    """Stand-in for an attribute that was not recordable as data (never replayed)."""

    def __init__(self, type_name: str) -> None:
        self.type_name = type_name

    def __repr__(self) -> str:
        return f"<unrecorded TensorFlow attr of type {self.type_name}>"


def _literal_array(tensor: Any) -> np.ndarray | None:
    """Host array of a literal input tensor, or ``None`` when it has no data value."""

    if tensor is None:
        return None
    dtype = getattr(tensor, "dtype", None)
    if dtype is not None and getattr(dtype, "name", "") in {"resource", "variant"}:
        return None
    try:
        array = _tensor_to_numpy(tensor)
    except Exception:  # noqa: BLE001 - a graph tensor has no value; that is the answer
        return None
    return None if array.dtype == object else array


def _encode_attr(value: Any) -> Any:
    """Encode one TensorFlow callback attr as recipe data (essential type switch)."""

    if value is None or type(value) in (bool, int, float, str):
        return value
    if isinstance(value, bytes):
        return {"tf": "bytes", "hex": value.hex()}
    if isinstance(value, (list, tuple)):
        items = [_encode_attr(item) for item in value]
        return items if isinstance(value, list) else tuple(items)
    enum = getattr(value, "as_datatype_enum", None)
    if isinstance(enum, int):
        return {"tf": "dtype", "enum": enum}
    if isinstance(value, np.generic):
        return _encode_attr(value.item())
    return {"tf": "opaque", "type": type(value).__name__}


def _decode_attr(tf: Any, node: Any) -> Any:
    """Inverse of :func:`_encode_attr`."""

    if isinstance(node, list):
        return [_decode_attr(tf, item) for item in node]
    if isinstance(node, tuple):
        return tuple(_decode_attr(tf, item) for item in node)
    if not isinstance(node, dict):
        return node
    kind = node.get("tf")
    if kind == "bytes":
        return bytes.fromhex(str(node["hex"]))
    if kind == "dtype":
        return tf.dtypes.as_dtype(int(node["enum"]))
    if kind == "opaque":
        return _OpaqueAttr(str(node.get("type")))
    raise ReplayRefusal("type", f"unknown TensorFlow attr encoding {node!r}")


def _encode_literal(item: Any) -> dict[str, Any]:
    """Data form of one non-parent callback input."""

    if item.source_kind is None:
        return {"source_kind": None, "array": None}
    array = _literal_array(item.tensor)
    if array is None or array.size > _MAX_LITERAL_ELEMENTS or array.dtype.kind not in "biufc":
        return {"source_kind": item.source_kind, "array": None}
    flat = array.reshape(-1).tolist()
    return {
        "source_kind": item.source_kind,
        "array": {"dtype": array.dtype.str, "shape": tuple(array.shape), "data": tuple(flat)},
    }


def _decode_literal(tf: Any, node: Mapping[str, Any]) -> Any:
    """Rebuild a literal input tensor from its data form (``None`` when not recorded)."""

    array = node.get("array")
    if array is None:
        return None
    dtype = np.dtype(str(array["dtype"]))
    if dtype.kind not in "biufc":
        raise ReplayRefusal("type", f"literal dtype {dtype.str!r} is not numeric")
    host = np.array(list(array["data"]), dtype=dtype).reshape(tuple(int(d) for d in array["shape"]))
    return tf.convert_to_tensor(host)


# --- recipe and bind_recipes -----------------------------------------------


def recipe(trace: Any, op: Any) -> OpRecipe | None:
    """Data-only recipe of ``op``; ``None`` for an op without a runtime capture."""

    import tensorflow as tf  # deferred: optional dependency

    _record, capture = _resolve(trace, op)
    if capture is None:
        return None
    ops, _ = _index(trace)
    occurrences: list[InputOccurrence] = []
    sources: list[tuple[int, Any, str]] = []
    literals: list[tuple[tuple[str | int, ...], Any]] = []
    for item in sorted(capture.inputs, key=lambda entry: entry.input_index):
        parent_raw = item.producer_label_raw or item.source_label_raw
        if parent_raw is None:
            literals.append(((item.input_index,), _encode_literal(item)))
            continue
        role = "producer" if item.producer_label_raw else "source"
        occurrences.append(InputOccurrence(_public_label(ops, parent_raw), (item.input_index,)))
        sources.append((item.input_index, item.source_kind, role))
    identity = capture.op_type if capture.op_type in KERNEL_OPS else INERT_IDENTITY
    init_labels = set(getattr(trace, "_tf_init_op_labels", ()) or ())
    attrs = {
        "tf_op_type": capture.op_type,
        "tf_attrs": {str(k): _encode_attr(v) for k, v in capture.attrs.items()},
        "output_index": int(capture.output_index),
        "init_op": capture.label_raw in init_labels,
        "input_sources": tuple(sources),
    }
    if capture.op_type == "ReadVariableOp":
        from ..tf._variable_reads import persisted_read

        if (read := persisted_read(trace, capture)) is not None:
            attrs["variable_read"] = read
    built = OpRecipe(
        backend=BACKEND,
        op_identity=identity,
        inputs=tuple(occurrences),
        literals=tuple(literals),
        attrs=attrs,
        output_slots=(_public_label(ops, capture.label_raw),),
        rng={"op_type": capture.op_type} if capture.op_type.startswith(_RANDOM_PREFIXES) else None,
        framework_version=str(tf.__version__),
    )
    validate_recipe(built)
    return built


def bind_recipes(trace: Any, recipes: Mapping[str, OpRecipe]) -> None:
    """Rebuild ``trace``'s runtime capture list from recipes plus its loaded payloads.

    Sets ``_tf_op_captures``, ``_tf_init_op_labels`` and ``_tf_unresolved_producers``
    so the unchanged TensorFlow validator replays the trace. Refuses a recipe for
    another backend, an identity outside :data:`ALLOWLIST`, a label the trace does
    not hold, or an op type that disagrees with the recorded op, with
    :class:`ReplayRefusal`, before anything is rebuilt.
    """

    import tensorflow as tf  # deferred: optional dependency

    ops, _ = _index(trace)
    order = {id(op): position for position, op in enumerate(trace._layer_list)}
    rebuilt: list[tuple[int, Any]] = []
    init_labels: list[str] = []
    unresolved: list[Any] = []
    reads: dict[str, Any] = {}
    for label, item in recipes.items():
        record, op_type = _checked_record(ops, label, item)
        capture = TFOpCapture(
            label_raw=str(record._label_raw),
            op_type=op_type,
            attrs={k: _decode_attr(tf, v) for k, v in dict(item.attrs["tf_attrs"]).items()},
            output_index=int(item.attrs["output_index"]),
            inputs=_rebuilt_inputs(tf, ops, item),
            output_tensor=_payload_tensor(tf, record),
        )
        rebuilt.append((order.get(id(record), len(order)), capture))
        if item.attrs.get("init_op"):
            init_labels.append(capture.label_raw)
        if "variable_read" in item.attrs:
            reads[capture.label_raw] = item.attrs["variable_read"]
        unresolved.extend(
            TFUnresolvedProducer(f"{op_type.lower()}_pending", op_type, entry.input_index, None)
            for entry in capture.inputs
            if entry.producer_label_raw is None
            and entry.source_label_raw is None
            and entry.source_kind is None
        )
    rebuilt.sort(key=lambda pair: pair[0])
    from ..tf._variable_reads import restore_variable_reads

    restore_variable_reads(trace, reads)
    trace._tf_op_captures = tuple(capture for _, capture in rebuilt)
    trace._tf_init_op_labels = tuple(init_labels)
    trace._tf_unresolved_producers = tuple(unresolved)
    # A cached replay status describes the capture list just replaced (for a
    # loaded trace: "stripped"); drop it so the next validation recomputes.
    trace._validation_replay_status = None


def _checked_record(ops: Mapping[str, Any], label: str, item: OpRecipe) -> tuple[Any, str]:
    """Return ``(op record, op type)`` for one recipe, refusing an inconsistent one."""

    validate_recipe(item)
    attrs = item.attrs
    if not all(key in attrs for key in _REQUIRED_ATTRS) or not isinstance(attrs["tf_attrs"], dict):
        raise ReplayRefusal("shape", "TensorFlow recipe attrs are incomplete", op_label=label)
    if item.backend != BACKEND:
        raise ReplayRefusal("backend", f"recipe for {item.backend!r}", op_label=label)
    if item.op_identity not in ALLOWLIST:
        raise ReplayRefusal("unknown_op", repr(item.op_identity), op_label=label)
    record = ops.get(label)
    if record is None or not isinstance(getattr(record, "_label_raw", None), str):
        raise ReplayRefusal("slots", "the trace holds no op with this label", op_label=label)
    op_type = item.attrs.get("tf_op_type")
    inert = item.op_identity == INERT_IDENTITY
    if not isinstance(op_type, str) or (op_type in KERNEL_OPS) == inert:
        raise ReplayRefusal("unknown_op", f"op type {op_type!r}", op_label=label)
    if not inert and op_type != item.op_identity:
        raise ReplayRefusal("unknown_op", "identity and op type differ", op_label=label)
    if op_type != str(getattr(record, "func_name", "")):
        raise ReplayRefusal("slots", f"recorded op is not a {op_type}", op_label=label)
    return record, op_type


def _rebuilt_inputs(tf: Any, ops: Mapping[str, Any], item: OpRecipe) -> tuple[Any, ...]:
    """Callback input records from a recipe's occurrences, sources and literals."""

    label = item.output_slots[0]
    roles = {int(index): (kind, role) for index, kind, role in item.attrs["input_sources"]}
    records: list[tuple[int, Any]] = []
    for occurrence in item.inputs:
        index = int(occurrence.arg_path[0])
        parent = ops.get(occurrence.parent_label)
        parent_raw = getattr(parent, "_label_raw", None) if parent is not None else None
        if not isinstance(parent_raw, str) or index not in roles:
            raise ReplayRefusal("slots", f"parent {occurrence.parent_label!r}", op_label=label)
        kind, role = roles[index]
        is_source = role == "source"
        tensor = _payload_tensor(tf, parent)
        records.append(
            (index, TFInputCapture(index, None if is_source else parent_raw, kind,
                                   parent_raw if is_source else None, tensor, None))
        )  # fmt: skip
    for path, literal in item.literals:
        index = int(path[0])
        tensor = _decode_literal(tf, literal)
        kind = literal.get("source_kind")
        records.append((index, TFInputCapture(index, None, kind, None, tensor, None)))
    indexes = [index for index, _ in records]
    if len(set(indexes)) != len(indexes):
        raise ReplayRefusal("slots", "two inputs share one index", op_label=label)
    return tuple(record for _, record in sorted(records, key=lambda pair: pair[0]))


def _payload_tensor(tf: Any, record: Any) -> Any:
    """The op's loaded payload as a TensorFlow tensor, or ``None`` when not saved."""

    if not bool(getattr(record, "has_saved_activation", False)):
        return None
    return tf.convert_to_tensor(to_host(record.out))


__all__ = (
    "ALLOWLIST",
    "BACKEND",
    "INERT_IDENTITY",
    "KERNEL_OPS",
    "add",
    "bind_recipes",
    "can_replay",
    "close",
    "forward",
    "recipe",
    "to_host",
    "vjp",
    "zeros_like",
)
