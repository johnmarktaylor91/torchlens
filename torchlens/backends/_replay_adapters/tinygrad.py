"""tinygrad replay adapter: per-op replay, derivative and recipe over the validator's replay.

The tinygrad validator replays each captured UOp by swapping the sources
recorded as parents for the parents' saved payloads and realizing the result
(``TinygradBackend._replay_uop_from_trace_graph``). This module exposes that
replay through the contract in :mod:`._contract`:

- :func:`forward` builds the same replayed UOp from given parent values but
  does not realize it, so tinygrad can differentiate through it.
- :func:`vjp` applies tinygrad's own derivative rule for the one replayed
  UOp (the rule ``tinygrad.gradient.compute_gradient`` applies per node),
  giving one cotangent per parent occurrence, repeated parents kept apart.
- :func:`recipe` describes an op as data: the UOp op name from a closed
  allowlist, ordered parent occurrences (``arg_path`` is the UOp source
  position), the UOp's dtype and arg structurally encoded, and every source
  that is not a parent (shape vectors, scalar constants) as an encoded UOp
  tree. :func:`bind_recipes` rebuilds the validator's capture list from
  recipes plus loaded payloads; the validator itself is unchanged.
- :func:`close` is the validator's own payload comparison.

An op with no recorded parents (a parameter buffer, a constant, a realized
random-init weight) replays to its saved payload in the validator; its
recipe is :data:`SAVED_VALUE_IDENTITY` and carries no UOp sources.

Framework imports stay inside functions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..._utils._record_group_table import record_member
from ..tinygrad._captures import TinygradUOpCapture
from ..tinygrad._raw_labels import _final_to_raw_labels
from ..tinygrad._rules import CONSTANT_KIND_ANNOTATION, RNG_UOP_NAMES
from ..tinygrad._uop_graph import _uop_name
from ..tinygrad.backend import (
    _host_tensor,
    _payloads_close,
    _saved_single_output,
    _source_matches_payload,
)
from ._contract import InputOccurrence, OpRecipe, ReplayRefusal

__tl_layer__ = "L2"

BACKEND = "tinygrad"

SAVED_VALUE_IDENTITY = "tinygrad.saved_value"
"""Recipe identity of an op the validator replays to its saved payload."""

_UOP_IDENTITY_PREFIX = "tinygrad.uop."
_VALUE_NODE = "VALUE"

#: UOp ops a recorded op may be replayed as. Closed: anything else is refused.
REPLAY_UOP_OPS: frozenset[str] = frozenset(
    {
        # elementwise
        "EXP2", "LOG2", "SIN", "SQRT", "RECIPROCAL", "NEG", "TRUNC",
        "ADD", "MUL", "SUB", "FDIV", "POW", "MAX", "CDIV", "CMOD",
        "FLOORDIV", "FLOORMOD", "SHL", "SHR", "XOR", "OR", "AND",
        "CMPLT", "CMPNE", "CMPEQ", "WHERE", "MULACC", "CAST", "BITCAST",
        # buffer boundaries
        "CONTIGUOUS", "CONTIGUOUS_BACKWARD", "DETACH", "COPY",
        # movement and reduction
        "RESHAPE", "PERMUTE", "EXPAND", "PAD", "SHRINK", "FLIP", "REDUCE",
    }
)  # fmt: skip

#: UOp ops a non-parent source may be built from when encoded as data.
LITERAL_UOP_OPS: frozenset[str] = frozenset({"CONST", "STACK", "DEVICE"})

#: The closed op-identity allowlist ``decode_recipe`` checks against.
ALLOWLIST: frozenset[str] = frozenset(
    {SAVED_VALUE_IDENTITY} | {_UOP_IDENTITY_PREFIX + name for name in REPLAY_UOP_OPS}
)

_RNG_REASON = (
    "random op: tinygrad random-state replay is not built yet, so a value drawn "
    "inside the forward cannot be replayed"
)


class _NotData(ValueError):
    """A UOp field has no data-only encoding."""


@dataclass(frozen=True)
class _TraceIndex:
    """Label maps and captures of one trace, rebuilt when either changes."""

    layer_list: Any
    captures: Any
    ops_by_raw: dict[str, Any]
    final_to_raw: dict[str, str]
    raw_to_final: dict[str, str]
    captures_by_raw: dict[str, Any]


_INDEX_CACHE: list[_TraceIndex] = []


def _index(trace: Any) -> _TraceIndex:
    """The label maps for ``trace``, cached on the identity of its op list and captures."""

    layer_list = getattr(trace, "_layer_list", ())
    captures = getattr(trace, "tinygrad_uop_captures", None)
    if _INDEX_CACHE:
        cached = _INDEX_CACHE[0]
        if cached.layer_list is layer_list and cached.captures is captures:
            return cached
    ops_by_raw = {getattr(op, "_label_raw", ""): op for op in layer_list}
    index = _TraceIndex(
        layer_list=layer_list,
        captures=captures,
        ops_by_raw=ops_by_raw,
        final_to_raw=_final_to_raw_labels(ops_by_raw),
        raw_to_final={raw: getattr(op, "label", raw) for raw, op in ops_by_raw.items()},
        captures_by_raw={c.label_raw: c for c in (captures or ())},
    )
    _INDEX_CACHE[:] = [index]
    return index


def _occurrences(capture: Any) -> tuple[tuple[int, str], ...]:
    """Parent occurrences ``(source position, raw parent label)`` in position order."""

    return tuple(sorted(capture.parent_arg_positions))


def _saved(op: Any) -> Any:
    return _saved_single_output(op)


def _matches(source_uop: Any, value: Any) -> bool:
    """The validator's rule for replacing a UOp source with a parent value."""

    return bool(_source_matches_payload(source_uop, value))


# --- contract functions ------------------------------------------------------


def can_replay(trace: Any, op: Any) -> str | None:
    """``None`` when ``op`` replays through this adapter, else the reason it cannot."""

    index = _index(trace)
    if not index.captures:
        return (
            "the trace holds no tinygrad replay captures (a loaded trace needs "
            "bind_recipes first; an audit-only trace has no payloads)"
        )
    capture = index.captures_by_raw.get(getattr(op, "_label_raw", None))
    if capture is None:
        if getattr(op, "is_input_layer", False) or not getattr(op, "parents", ()):
            return "input source: a value of the capture, not a replayed op"
        return "no tinygrad replay capture was recorded for this op"
    if not capture.parent_arg_positions:
        return None
    return _uop_refusal(capture, op, index)


def _uop_refusal(capture: Any, op: Any, index: _TraceIndex) -> str | None:
    """The refusal for a captured UOp with parents, or ``None``."""

    uop = capture.uop
    name = _uop_name(uop)
    if name in RNG_UOP_NAMES:
        return _RNG_REASON
    if name not in REPLAY_UOP_OPS:
        return f"unlisted tinygrad UOp op {name}"
    graph = _graph_positions(op, index)
    if tuple(sorted(graph.items())) != _occurrences(capture):
        return "the op's parent positions no longer match its capture"
    try:
        _encode_attrs(uop)
        fixed = [uop.src[pos] for pos in _fixed_positions(capture, index)]
    except _NotData as error:
        return f"tinygrad UOp {name} has a field with no data encoding: {error}"
    except ValueError as error:
        return f"a parent payload is not available for replay: {error}"
    if any(_uop_name(node) in RNG_UOP_NAMES for source in fixed for node in source.toposort()):
        return _RNG_REASON
    return None


def _graph_positions(op: Any, index: _TraceIndex) -> dict[int, str]:
    """The op's recorded parent positions, labels resolved to raw capture labels."""

    positions = record_member(op, "parent_arg_positions", {}).get("args", {})
    return {
        position: index.final_to_raw.get(label, label)
        for position, label in positions.items()
        if isinstance(label, str)
    }


def _require_capture(trace: Any, op: Any) -> Any:
    reason = can_replay(trace, op)
    if reason is not None:
        raise ValueError(f"tinygrad op {getattr(op, 'label', op)!r} cannot be replayed: {reason}")
    return _index(trace).captures_by_raw[op._label_raw]


def _replay_uop(capture: Any, values: Sequence[Any]) -> Any:
    """The captured UOp with parent sources swapped for ``values`` (validator rule)."""

    occurrences = _occurrences(capture)
    if len(values) != len(occurrences):
        raise ValueError(
            f"tinygrad op {capture.label_raw} takes {len(occurrences)} parent values, "
            f"got {len(values)}"
        )
    src = list(capture.uop.src)
    for (position, _label), value in zip(occurrences, values, strict=True):
        if _matches(src[position], value):
            src[position] = value.uop
    return capture.uop.replace(src=tuple(src))


def forward(trace: Any, op: Any, parent_values: Sequence[Any]) -> tuple[Any, ...]:
    """Replay ``op`` from ``parent_values`` (one per occurrence); unrealized, differentiable."""

    from tinygrad import Tensor

    capture = _require_capture(trace, op)
    if not capture.parent_arg_positions:
        if parent_values:
            raise ValueError(f"tinygrad op {capture.label_raw} has no parents")
        return (capture.payload_snapshot,)
    return (Tensor(_replay_uop(capture, parent_values)),)


def vjp(
    trace: Any,
    op: Any,
    parent_values: Sequence[Any],
    cotangents: Sequence[Any],
) -> tuple[Any | None, ...]:
    """tinygrad's derivative of the one replayed UOp: one cotangent per parent occurrence.

    ``None`` for an occurrence that is not differentiable: a non-float parent,
    a bool or integer output, a parent the validator keeps as a constant
    source, or a source tinygrad's rule gives no gradient (a ``WHERE`` condition).
    """

    from tinygrad import Tensor

    capture = _require_capture(trace, op)
    if len(cotangents) != 1:
        raise ValueError("a tinygrad op has one output slot; pass one cotangent")
    occurrences = _occurrences(capture)
    if not occurrences:
        return ()
    replay = _replay_uop(capture, parent_values)
    cotangent = cotangents[0]
    if cotangent is None or not _is_float(replay.dtype):
        return (None,) * len(occurrences)
    local = _local_gradients(replay, cotangent.uop)
    grads: list[Any | None] = []
    for (position, _label), value in zip(occurrences, parent_values, strict=True):
        grad = local[position]
        usable = grad is not None and replay.src[position] is value.uop
        grads.append(Tensor(grad) if usable and _is_float(value.dtype) else None)
    return tuple(grads)


def _local_gradients(root: Any, root_grad: Any) -> tuple[Any | None, ...]:
    """Per-source gradients of one UOp under tinygrad's own derivative rules."""

    from tinygrad import gradient as tg_gradient

    rules = getattr(tg_gradient, "pm_gradient", None)
    if rules is None:
        raise ValueError("this tinygrad has no per-UOp derivative rules (pm_gradient)")
    local = rules.rewrite(root, ctx=root_grad)
    if local is None or len(local) != len(root.src):
        raise ValueError(f"tinygrad has no derivative rule for UOp {_uop_name(root)}")
    return tuple(local)


def _is_float(dtype: Any) -> bool:
    from tinygrad import dtypes

    return bool(dtypes.is_float(dtype))


def zeros_like(value: Any) -> Any:
    """tinygrad zeros with ``value``'s shape, dtype and device."""

    return value.zeros_like()


def add(left: Any, right: Any) -> Any:
    """tinygrad elementwise sum."""

    return left + right


def to_host(value: Any) -> Any:
    """A NumPy copy of a tinygrad value."""

    import numpy as np

    return np.asarray(value.numpy())


def close(expected: Any, actual: Any) -> bool:
    """The tinygrad validator's payload comparison (shape, dtype, dtype-aware tolerance)."""

    return bool(_payloads_close(expected, actual))


# --- recipes -----------------------------------------------------------------


def recipe(trace: Any, op: Any) -> OpRecipe | None:
    """The data-only recipe for ``op``, or ``None`` when :func:`can_replay` refuses it."""

    if can_replay(trace, op) is not None:
        return None
    index = _index(trace)
    capture = index.captures_by_raw[op._label_raw]
    label = index.raw_to_final.get(capture.label_raw, capture.label_raw)
    occurrences = _occurrences(capture)
    if not occurrences:
        return OpRecipe(
            backend=BACKEND,
            op_identity=SAVED_VALUE_IDENTITY,
            inputs=(),
            attrs={"uop_op": capture.op_name},
            output_slots=(label,),
            framework_version=_framework_version(),
        )
    inputs = tuple(
        InputOccurrence(index.raw_to_final.get(raw, raw), (position,))
        for position, raw in occurrences
    )
    parents = [(position, index.ops_by_raw[raw]) for position, raw in occurrences]
    return OpRecipe(
        backend=BACKEND,
        op_identity=_UOP_IDENTITY_PREFIX + _uop_name(capture.uop),
        inputs=inputs,
        literals=tuple(((pos,), tree) for pos, tree in _literal_sources(capture, index).items()),
        attrs=_encode_attrs(capture.uop),
        output_slots=(label,),
        param_refs=_param_refs(parents),
        constant_refs=_constant_refs(parents, index),
        rng=None,
        framework_version=_framework_version(),
    )


def _fixed_positions(capture: Any, index: _TraceIndex) -> list[int]:
    """Source positions the validator keeps as captured (not swapped for a parent value)."""

    swapped = {
        position
        for position, raw in _occurrences(capture)
        if _matches(capture.uop.src[position], _saved(index.ops_by_raw[raw]))
    }
    return [position for position in range(len(capture.uop.src)) if position not in swapped]


def _literal_sources(capture: Any, index: _TraceIndex) -> dict[int, Any]:
    """Encoded trees for every source the validator keeps as captured.

    A source built only from constants is stored as its UOp structure; any
    other fixed source (a realized constant buffer) is stored by value.
    """

    return {
        position: _encode_source(capture.uop.src[position])
        for position in _fixed_positions(capture, index)
    }


def _param_refs(parents: Sequence[tuple[int, Any]]) -> tuple[tuple[tuple[int], str], ...]:
    """``(arg_path, address)`` for parent occurrences that are a parameter's own value."""

    refs = []
    for position, parent in parents:
        barcodes = tuple(getattr(parent, "_param_barcodes", ()) or ())
        if getattr(parent, "parents", ()) or len(barcodes) != 1:
            continue
        refs.append(((position,), str(barcodes[0]).removeprefix("tinygrad:")))
    return tuple(refs)


def _constant_refs(
    parents: Sequence[tuple[int, Any]], index: _TraceIndex
) -> tuple[tuple[tuple[int], str], ...]:
    """``(arg_path, label)`` for parent occurrences classified as constants."""

    return tuple(
        ((position,), index.raw_to_final.get(parent._label_raw, parent._label_raw))
        for position, parent in parents
        if CONSTANT_KIND_ANNOTATION in (getattr(parent, "annotations", None) or {})
    )


def _framework_version() -> str:
    from importlib import metadata

    try:
        return metadata.version("tinygrad")
    except metadata.PackageNotFoundError:
        return ""


def bind_recipes(trace: Any, recipes: Mapping[str, OpRecipe]) -> None:
    """Rebuild ``trace.tinygrad_uop_captures`` from recipes and the trace's saved payloads.

    Every op that is not an input source needs a recipe; a missing, foreign,
    unlisted or inconsistent recipe raises :class:`ReplayRefusal` before
    the trace is changed.
    """

    ops_by_raw = {getattr(op, "_label_raw", ""): op for op in trace._layer_list}
    final_to_raw = _final_to_raw_labels(ops_by_raw)
    by_raw: dict[str, Any] = {}
    for label, item in recipes.items():
        raw = _check_recipe(label, item, final_to_raw, ops_by_raw)
        by_raw[raw] = TinygradUOpCapture(*_capture_fields(item, raw, ops_by_raw, final_to_raw))
    for raw, op in ops_by_raw.items():
        if raw not in by_raw and getattr(op, "parents", ()):
            raise ReplayRefusal("slots", "op has no recipe", op_label=op.label)
    trace.tinygrad_uop_captures = tuple(by_raw[raw] for raw in ops_by_raw if raw in by_raw)


def _check_recipe(
    label: str, item: OpRecipe, final_to_raw: Mapping[str, str], ops_by_raw: Mapping[str, Any]
) -> str:
    """The raw label ``item`` binds to, after its identity and slots are checked."""

    if item.backend != BACKEND:
        raise ReplayRefusal("backend", f"recipe is for {item.backend!r}", op_label=label)
    if item.op_identity not in ALLOWLIST:
        raise ReplayRefusal("unknown_op", f"{item.op_identity!r} is unlisted", op_label=label)
    if item.output_slots != (label,):
        raise ReplayRefusal("slots", "a tinygrad op has exactly its own slot", op_label=label)
    raw = final_to_raw.get(label)
    if raw is None or raw not in ops_by_raw:
        raise ReplayRefusal("slots", "no op in the trace has this label", op_label=label)
    return raw


def _capture_fields(
    item: OpRecipe, raw: str, ops_by_raw: Mapping[str, Any], final_to_raw: Mapping[str, str]
) -> tuple[Any, ...]:
    """``TinygradUOpCapture`` fields rebuilt from one recipe."""

    label = item.output_slots[0]
    payload = _saved(ops_by_raw[raw])
    if item.op_identity == SAVED_VALUE_IDENTITY:
        if item.inputs:
            raise ReplayRefusal("shape", "a saved value has no inputs", op_label=label)
        return (raw, payload.uop, str(item.attrs.get("uop_op", "")), (), (), payload)
    occurrences = []
    for occurrence in item.inputs:
        path = occurrence.arg_path
        parent_raw = final_to_raw.get(occurrence.parent_label)
        if len(path) != 1 or type(path[0]) is not int or parent_raw not in ops_by_raw:
            raise ReplayRefusal("shape", "bad parent occurrence", op_label=label)
        occurrences.append((path[0], parent_raw))
    uop = _rebuild_uop(item, occurrences, ops_by_raw)
    occurrences.sort()
    return (
        raw,
        uop,
        _uop_name(uop),
        tuple(parent for _pos, parent in occurrences),
        tuple(occurrences),
        payload,
    )


def _rebuild_uop(
    item: OpRecipe, occurrences: Sequence[tuple[int, str]], ops_by_raw: Mapping[str, Any]
) -> Any:
    """The replay UOp: literal sources decoded, parent sources set to saved parent payloads."""

    label = item.output_slots[0]
    attrs = item.attrs
    name = item.op_identity.removeprefix(_UOP_IDENTITY_PREFIX)
    n_src = attrs.get("n_src")
    if attrs.get("uop_op") != name or type(n_src) is not int or n_src < 0:
        raise ReplayRefusal("shape", "attrs disagree with the op identity", op_label=label)
    src: list[Any] = [None] * n_src
    try:
        for path, tree in item.literals:
            if len(path) != 1 or type(path[0]) is not int or not 0 <= path[0] < n_src:
                raise _NotData(f"bad literal position {path!r}")
            src[path[0]] = _decode_source(tree)
        for position, parent_raw in occurrences:
            if not 0 <= position < n_src:
                raise _NotData(f"bad parent position {position}")
            if src[position] is None:
                src[position] = _saved(ops_by_raw[parent_raw]).uop
        if any(source is None for source in src):
            raise _NotData("a source position has neither a literal nor a parent")
        from tinygrad.uop.ops import UOp

        return UOp(
            _decode_ops(name), _decode_dtype(attrs["dtype"]), tuple(src), _decode_arg(attrs["arg"])
        )
    except (_NotData, KeyError, TypeError, AttributeError) as error:
        raise ReplayRefusal("shape", f"cannot rebuild the UOp: {error}", op_label=label) from error


# --- structural encoding -------------------------------------------------------


def _encode_attrs(uop: Any) -> dict[str, Any]:
    if uop.tag is not None:
        raise _NotData("tagged UOp")
    return {
        "uop_op": _uop_name(uop),
        "dtype": _encode_dtype(uop.dtype),
        "arg": _encode_arg(uop.arg),
        "n_src": len(uop.src),
    }


def _encode_source(source: Any) -> dict[str, Any]:
    """A fixed source as its literal UOp structure, else as its captured value."""

    try:
        return _encode_uop(source)
    except _NotData:
        pass
    from tinygrad import Tensor

    value = Tensor(source)
    if not isinstance(value.device, str):
        raise _NotData("fixed source on a multi-device tensor")
    return {
        "uop_op": _VALUE_NODE,
        "dtype": _encode_dtype(value.dtype),
        "shape": tuple(int(n) for n in value.shape),
        "device": value.device,
        "data": value.tolist(),
    }


def _decode_source(tree: Any) -> Any:
    """Inverse of :func:`_encode_source`."""

    if not isinstance(tree, Mapping) or tree.get("uop_op") != _VALUE_NODE:
        return _decode_uop(tree)
    shape, device = tree.get("shape"), tree.get("device")
    if not isinstance(shape, (tuple, list)) or not isinstance(device, str):
        raise _NotData("malformed fixed source value")
    value = _host_tensor(tree.get("data"), _decode_dtype(tree.get("dtype")), device)
    if tuple(value.shape) != tuple(shape):
        raise _NotData("fixed source value does not have its recorded shape")
    return value.uop


def _encode_uop(uop: Any, depth: int = 0) -> dict[str, Any]:
    """A literal source subgraph as nested data (closed op set, bounded depth)."""

    name = _uop_name(uop)
    if name not in LITERAL_UOP_OPS:
        raise _NotData(f"source UOp {name} is not a literal")
    if depth > 16:
        raise _NotData("literal source is too deep")
    return {**_encode_attrs(uop), "src": tuple(_encode_uop(s, depth + 1) for s in uop.src)}


def _decode_uop(tree: Any, depth: int = 0) -> Any:
    from tinygrad.uop.ops import UOp

    if not isinstance(tree, Mapping) or depth > 16:
        raise _NotData("literal source is not a bounded mapping")
    name = tree.get("uop_op")
    src = tree.get("src")
    if (
        name not in LITERAL_UOP_OPS
        or not isinstance(src, (tuple, list))
        or len(src) != tree.get("n_src")
    ):
        raise _NotData(f"literal source {name!r} is malformed or not allowed")
    children = tuple(_decode_uop(child, depth + 1) for child in src)
    return UOp(_decode_ops(name), _decode_dtype(tree["dtype"]), children, _decode_arg(tree["arg"]))


def _encode_dtype(dtype: Any) -> dict[str, Any]:
    from tinygrad.dtype import INVERSE_DTYPES_DICT, PtrDType

    if isinstance(dtype, PtrDType):
        raise _NotData("pointer dtype")
    return {"scalar": INVERSE_DTYPES_DICT[dtype.scalar().name], "count": int(dtype.count)}


def _decode_dtype(node: Any) -> Any:
    from tinygrad.dtype import DTYPES_DICT, dtypes

    if not isinstance(node, Mapping) or type(node.get("count")) is not int:
        raise _NotData("malformed dtype")
    name = node.get("scalar")
    if name not in DTYPES_DICT and name not in ("void", "weakint"):
        raise _NotData(f"unknown dtype {name!r}")
    base = getattr(dtypes, name)
    return base if node["count"] == 1 else base.vec(node["count"])


def _encode_arg(arg: Any) -> Any:
    """A UOp arg as data: scalars, ``Ops`` members, dtypes and tuples of these."""

    from tinygrad.dtype import ConstFloat, DType
    from tinygrad.uop import Ops

    if arg is None or type(arg) in (bool, int, str):
        return arg
    if type(arg) in (float, ConstFloat):
        return {"kind": "float", "hex": float(arg).hex(), "const": type(arg) is ConstFloat}
    if isinstance(arg, Ops):
        return {"kind": "ops", "name": arg.name}
    if isinstance(arg, DType):
        return {"kind": "dtype", "dtype": _encode_dtype(arg)}
    if type(arg) is tuple:
        return tuple(_encode_arg(item) for item in arg)
    raise _NotData(f"UOp arg of type {type(arg).__name__}")


def _decode_arg(node: Any) -> Any:
    from tinygrad.dtype import ConstFloat

    if node is None or type(node) in (bool, int, str):
        return node
    if isinstance(node, (tuple, list)):
        return tuple(_decode_arg(item) for item in node)
    if not isinstance(node, Mapping):
        raise _NotData(f"malformed arg {type(node).__name__}")
    kind = node.get("kind")
    if kind == "float" and isinstance(node.get("hex"), str):
        value = float.fromhex(node["hex"])
        return ConstFloat(value) if node.get("const") is True else value
    if kind == "ops":
        return _decode_ops(node.get("name"))
    if kind == "dtype":
        return _decode_dtype(node.get("dtype"))
    raise _NotData(f"unknown arg kind {kind!r}")


def _decode_ops(name: Any) -> Any:
    from tinygrad.uop import Ops

    if name not in REPLAY_UOP_OPS and name not in LITERAL_UOP_OPS and name not in _ARG_OPS:
        raise _NotData(f"UOp op {name!r} is not allowed")
    return getattr(Ops, name)


#: ``Ops`` members that appear inside args (a ``REDUCE``'s reduction op).
_ARG_OPS: frozenset[str] = frozenset({"ADD", "MUL", "MAX"})
