"""Private Paddle replay adapter (wave B contract, ``_replay_adapters._contract``).

Thin wrappers over the replay the Paddle validator already runs: every
recorded Paddle call keeps a ``PaddleOpCapture`` whose ``func`` is the
original unwrapped callable and whose argument templates mark each tensor
leaf with its producer label. ``forward`` re-runs that callable with the
given parent values substituted at the template's tensor paths (the same
substitution ``paddle/validation.py`` ``_replace_template_paths`` performs),
outside ``paddle.no_grad`` so Paddle can differentiate the result. ``vjp`` is
``paddle.grad`` over that single call. ``close`` is the validator's own
``_payloads_close``.

Recipes are data only: the op identity is a key into :data:`ALLOWLIST`
(built from the wrapper inventory's curated capture sets), tensor leaves are
input occurrences, parameter references or factory holes, every other leaf is
a literal with Paddle-specific values (dtypes, slices, NumPy scalars) tagged
structurally. ``bind_recipes`` rebuilds the capture list the unchanged
validator reads, failing closed (a gap capture naming the op) wherever a
recipe or a parameter value is missing.

Paddle closure constants are recorded ops with labels, so their consumers
list them as input occurrences; ``constant_refs`` stays empty.
"""

from __future__ import annotations

import inspect
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ... import _state
from ..._utils._record_group_table import record_member
from ..paddle import wrappers
from ..paddle._param_writes import PARAMETER_WRITE_MARKER
from ..paddle.backend import PaddleOpCapture, TensorLeafCapture
from ..paddle.constants_rule import PADDLE_CONSTANT_OP, is_constant_capture
from ..paddle.conv_layers import PADDLE_CONV_LAYER_OPS, paddle_conv_layer_replay
from ..paddle.validation import (
    RebuiltPaddleInputs,
    _is_tensor_marker,
    _parameter_replay_copy,
    _payloads_close,
    _replace_template_paths,
    _value_at_path,
)
from ..paddle.wrappers import (
    _FUNCTIONAL_CORE_OPS,
    _TENSOR_CORE_METHODS,
    _TOP_LEVEL_CORE_OPS,
    _classify_candidate,
)
from ._contract import InputOccurrence, OpRecipe, ReplayRefusal

BACKEND = "paddle"

_REASON_NO_CAPTURE = "graph-only entry: no Paddle capture record for this op"
_REASON_INPUT = "graph-only entry: model input"
_REASON_RANDOM = "random op without RNG support: dropout in training mode with p > 0"
_REASON_INTERVENED = "intervened call: the recorded value is a user replacement"

#: Literal tags (single-key mappings) for Paddle values that are not plain data.
_TAG_DTYPE = "__paddle_dtype__"
_TAG_NP_DTYPE = "__numpy_dtype__"
_TAG_NP_SCALAR = "__numpy_scalar__"
_TAG_SLICE = "__slice__"
_TAG_ELLIPSIS = "__ellipsis__"
_TAGS = frozenset({_TAG_DTYPE, _TAG_NP_DTYPE, _TAG_NP_SCALAR, _TAG_SLICE, _TAG_ELLIPSIS})

#: Skeleton node kinds in ``attrs["call"]``.
_NODE_LITERAL = "L"
_NODE_INPUT = "I"
_NODE_PARAM = "P"
_NODE_HOLE = "H"


def _probe() -> None:
    """A plain function standing in for a candidate in the inventory classifier."""


def _capturable(owner_name: str, prefix: str, names: set[str]) -> frozenset[str]:
    """Op names the wrapper inventory captures for one owner."""
    return frozenset(
        f"{prefix}{name}"
        for name in names
        if _classify_candidate(owner_name, name, _probe, names) == "capture"
    )


#: Closed set of Paddle op identities a recipe may name.
ALLOWLIST: frozenset[str] = (
    _capturable("paddle", "", _TOP_LEVEL_CORE_OPS)
    | _capturable("paddle.nn.functional", "functional.", _FUNCTIONAL_CORE_OPS)
    | _capturable("paddle.Tensor", "tensor.", _TENSOR_CORE_METHODS)
    | frozenset(PADDLE_CONV_LAYER_OPS.values())
    | frozenset({PADDLE_CONSTANT_OP})
)


@dataclass(frozen=True)
class _Site:
    """One recorded Paddle call: its capture and its emitted output slots."""

    capture: Any
    slot_labels: tuple[str, ...]
    slot_paths: tuple[tuple[Any, ...], ...]


# --- locating an op's capture ------------------------------------------------


def _raw_label(op: Any) -> str | None:
    """Raw (capture-identity) label of an op or capture."""
    for name in ("label_raw", "_label_raw", "label"):
        value = getattr(op, name, None)
        if isinstance(value, str):
            return value
    return None


def _ops(trace: Any) -> tuple[Any, ...]:
    """The trace's materialized ops in order."""
    return tuple(getattr(trace, "_layer_list", ()) or ())


def _alias_paths(capture: Any) -> frozenset[tuple[Any, ...]]:
    """Output paths the call returned as an alias of an input (no op)."""
    return frozenset(
        tuple(note.get("output_path", ())) for note in getattr(capture, "alias_annotations", ())
    )


def _emitted_paths(capture: Any) -> tuple[tuple[Any, ...], ...]:
    """Output paths that became ops, in emission order."""
    aliases = _alias_paths(capture)
    return tuple(
        tuple(path) for path in getattr(capture, "output_leaf_paths", ()) if path not in aliases
    )


def _site_for_capture(trace: Any, capture: Any) -> _Site | None:
    """The output slots of one capture, or ``None`` when its first op is missing."""
    first_label = str(capture.label_raw)
    ops = _ops(trace)
    first = next((op for op in ops if getattr(op, "_label_raw", None) == first_label), None)
    paths = _emitted_paths(capture)
    if first is None or not paths:
        return None
    if len(paths) == 1:
        return _Site(capture, (first_label,), paths)
    call_id = record_member(first, "func_call_id", None)
    by_path = {
        tuple(record_member(op, "container_path", ()) or ()): str(op._label_raw)
        for op in ops
        if call_id is not None and record_member(op, "func_call_id", None) == call_id
    }
    if any(path not in by_path for path in paths):
        return None
    return _Site(capture, tuple(by_path[path] for path in paths), paths)


def _site(trace: Any, op: Any) -> _Site | None:
    """The recorded call that emitted ``op`` (an ``Op`` or a capture)."""
    label = _raw_label(op)
    for capture in getattr(trace, "_paddle_op_captures", ()) or ():
        if capture is op or str(getattr(capture, "label_raw", "")) == label:
            return _site_for_capture(trace, capture)
    for capture in getattr(trace, "_paddle_op_captures", ()) or ():
        site = _site_for_capture(trace, capture)
        if site is not None and label in site.slot_labels:
            return site
    return None


# --- template walking --------------------------------------------------------


def _template_leaves(capture: Any) -> Iterator[tuple[tuple[Any, ...], Any]]:
    """Every tensor marker of the capture's templates with its template path."""
    return _leaves_of(
        tuple(getattr(capture, "args_template", ())), dict(getattr(capture, "kwargs_template", {}))
    )


def _leaves_of(
    args: tuple[Any, ...], kwargs: Mapping[str, Any]
) -> Iterator[tuple[tuple[Any, ...], Any]]:
    """Every tensor marker of argument templates with its template path."""

    def walk(value: Any, path: tuple[Any, ...]) -> Iterator[tuple[tuple[Any, ...], Any]]:
        """Yield tensor markers below ``value``."""
        if _is_tensor_marker(value):
            yield path, value
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                yield from walk(item, (*path, index))
        elif isinstance(value, dict):
            for key, item in value.items():
                yield from walk(item, (*path, key))

    for index, item in enumerate(args):
        yield from walk(item, ("args", index))
    for key, item in kwargs.items():
        yield from walk(item, ("kwargs", str(key)))


def _occurrences(capture: Any) -> tuple[tuple[str, tuple[Any, ...]], ...]:
    """``(parent label, template path)`` per labeled tensor leaf, repeats kept."""
    return tuple(
        (marker["label"], path)
        for path, marker in _template_leaves(capture)
        if isinstance(marker.get("label"), str)
    )


def _arg_path(template_path: tuple[Any, ...]) -> tuple[Any, ...]:
    """Contract ``ArgPath`` (index or keyword, then flatten path) of a template path."""
    return tuple(template_path[1:])


def _template_path(arg_path: Sequence[Any]) -> tuple[Any, ...]:
    """Template path (``("args", i, ...)`` or ``("kwargs", k, ...)``) of an ``ArgPath``."""
    head = arg_path[0] if arg_path else None
    root = "args" if isinstance(head, int) else "kwargs"
    return (root, *arg_path)


def _call_arguments(
    capture: Any, replacements: Mapping[tuple[Any, ...], Any]
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """The call's arguments: given parent values, replay copies of parameters."""
    param_inputs: Mapping[tuple[Any, ...], Any] = getattr(capture, "param_inputs", None) or {}
    params = {path: _parameter_replay_copy(value) for path, value in param_inputs.items()}
    holes = {
        path: None
        for path, marker in _template_leaves(capture)
        if marker.get("label") is None and path not in params
    }
    rebuilt = RebuiltPaddleInputs(True, (), {}, None, {}, {}, params)
    return _replace_template_paths(
        tuple(getattr(capture, "args_template", ())),
        dict(getattr(capture, "kwargs_template", {})),
        {**holes, **dict(replacements)},
        rebuilt,
    )


# --- contract functions ------------------------------------------------------


def _trace_gap(trace: Any) -> str | None:
    """A trace-level capture gap (a parameter written outside any wrapped op)."""
    for marker in tuple(getattr(trace, "_paddle_capture_gap_markers", ()) or ()):
        if str(marker).startswith(PARAMETER_WRITE_MARKER):
            return f"opaque region: {marker}"
    return None


def _dropout_is_random(capture: Any) -> bool:
    """Whether a captured dropout call draws random numbers."""
    if str(getattr(capture, "op_name", "")).rsplit(".", 1)[-1] != "dropout":
        return False
    args = tuple(getattr(capture, "args_template", ()))
    kwargs = dict(getattr(capture, "kwargs_template", {}))
    try:
        bound = inspect.signature(capture.func).bind(*args, **kwargs)
    except (TypeError, ValueError):
        return True
    bound.apply_defaults()
    p = bound.arguments.get("p", 0.5)
    return bool(bound.arguments.get("training", True)) and not (
        isinstance(p, (int, float)) and p == 0
    )


def can_replay(trace: Any, op: Any) -> str | None:
    """``None``, or why this op cannot be replayed through the Paddle validator's replay."""
    if bool(record_member(op, "is_input", False)):
        return _REASON_INPUT
    site = _site(trace, op)
    if site is None:
        return _REASON_NO_CAPTURE
    gap = _trace_gap(trace)
    if gap is not None:
        return gap
    capture = site.capture
    markers = tuple(getattr(capture, "capture_gap_markers", ()))
    if markers:
        return f"opaque region: {markers[0]}"
    if is_constant_capture(capture):
        return None
    if str(capture.op_name) not in ALLOWLIST or not callable(capture.func):
        return f"unlisted raw op: {capture.op_name!r}"
    if _dropout_is_random(capture):
        return _REASON_RANDOM
    return None


def _require(trace: Any, op: Any) -> _Site:
    """The op's site, raising when the op cannot be replayed."""
    reason = can_replay(trace, op)
    site = _site(trace, op)
    if reason is not None or site is None:
        raise ReplayRefusal("cannot_replay", f"{reason}", op_label=_raw_label(op))
    return site


def forward(trace: Any, op: Any, parent_values: Sequence[Any]) -> tuple[Any, ...]:
    """Re-run this op's call with ``parent_values`` (one per input occurrence).

    Returns one value per output slot. Runs outside ``paddle.no_grad``, so a
    parent with ``stop_gradient=False`` makes the outputs differentiable.
    """

    site = _require(trace, op)
    capture = site.capture
    if is_constant_capture(capture):
        if parent_values:
            raise ReplayRefusal("arity", "a Paddle closure constant takes no parent values")
        label = site.slot_labels[0]
        return (next(o for o in _ops(trace) if getattr(o, "_label_raw", None) == label).out,)
    occurrences = _occurrences(capture)
    if len(parent_values) != len(occurrences):
        raise ReplayRefusal(
            "arity",
            f"Paddle op {site.slot_labels[0]!r} takes {len(occurrences)} parent values, "
            f"got {len(parent_values)}",
        )
    replacements = {
        path: value for (_label, path), value in zip(occurrences, parent_values, strict=True)
    }
    with _state.pause_logging():
        args, kwargs = _call_arguments(capture, replacements)
        output = capture.func(*args, **kwargs)
    return tuple(_value_at_path(output, path) for path in site.slot_paths)


def _is_float(value: Any) -> bool:
    """Whether a Paddle tensor has a floating or complex dtype."""
    import paddle

    return bool(paddle.is_floating_point(value) or paddle.is_complex(value))


def _differentiable_leaf(value: Any) -> Any:
    """A fresh graph leaf with ``value``'s data (one per occurrence)."""
    leaf = value.detach()
    leaf.stop_gradient = False
    return leaf


def vjp(
    trace: Any, op: Any, parent_values: Sequence[Any], cotangents: Sequence[Any]
) -> tuple[Any | None, ...]:
    """Paddle's own derivative of this op: one cotangent per parent occurrence.

    ``None`` marks an occurrence that is not differentiable (integer or bool)
    or that the selected outputs do not depend on.
    """

    import paddle

    with _state.pause_logging(), paddle.set_grad_enabled(True):
        float_mask = tuple(_is_float(value) for value in parent_values)
        leaves = tuple(
            _differentiable_leaf(value) if is_float else value
            for value, is_float in zip(parent_values, float_mask, strict=True)
        )
        outputs = forward(trace, op, leaves)
        if len(cotangents) != len(outputs):
            raise ReplayRefusal(
                "arity", f"expected {len(outputs)} cotangents, got {len(cotangents)}"
            )
        pairs = [
            (out, cot)
            for out, cot in zip(outputs, cotangents, strict=True)
            if cot is not None and not bool(out.stop_gradient)
        ]
        inputs = [leaf for leaf, is_float in zip(leaves, float_mask, strict=True) if is_float]
        if not pairs or not inputs:
            return tuple(None for _ in parent_values)
        grads = iter(
            paddle.grad(
                outputs=[out for out, _ in pairs],
                inputs=inputs,
                grad_outputs=[cot for _, cot in pairs],
                allow_unused=True,
            )
        )
    return tuple(next(grads) if is_float else None for is_float in float_mask)


def zeros_like(value: Any) -> Any:
    """Paddle zeros with ``value``'s shape and dtype."""
    import paddle

    with _state.pause_logging():
        return paddle.zeros_like(value)


def add(left: Any, right: Any) -> Any:
    """Paddle elementwise sum (gradient accumulation)."""
    import paddle

    with _state.pause_logging():
        return paddle.add(left, right)


def to_host(value: Any) -> Any:
    """Host NumPy copy of a Paddle value."""
    with _state.pause_logging():
        return np.asarray(value.numpy())


def close(expected: Any, actual: Any) -> bool:
    """The Paddle validator's own payload comparison (replayed value first, as it calls it)."""
    return bool(_payloads_close(actual, expected))


# --- recipes -----------------------------------------------------------------


def _encode_literal(value: Any) -> Any:
    """Encode one non-tensor template leaf as recipe data (essential type switch)."""
    if value is None or type(value) in (bool, int, float, complex, str):
        return value
    if isinstance(value, (tuple, list)):
        items = [_encode_literal(item) for item in value]
        return tuple(items) if isinstance(value, tuple) else items
    if type(value) is dict:
        if any(type(key) is not str or key in _TAGS for key in value):
            raise ReplayRefusal("type", "literal mapping with a non-str or reserved key")
        return {key: _encode_literal(item) for key, item in value.items()}
    if value is Ellipsis:
        return {_TAG_ELLIPSIS: True}
    if isinstance(value, slice):
        return {_TAG_SLICE: _encode_literal((value.start, value.stop, value.step))}
    if isinstance(value, np.generic):
        return {_TAG_NP_SCALAR: (str(value.dtype), _encode_literal(value.item()))}
    if isinstance(value, np.dtype):
        return {_TAG_NP_DTYPE: str(value)}
    dtype_name = _paddle_dtype_name(value)
    if dtype_name is not None:
        return {_TAG_DTYPE: dtype_name}
    raise ReplayRefusal("type", f"literal of type {type(value).__name__}")


def _paddle_dtype_name(value: Any) -> str | None:
    """The ``paddle.<name>`` spelling of a Paddle dtype object, else ``None``."""
    import paddle

    for name in ("bool", "uint8", "int8", "int16", "int32", "int64", "float16", "bfloat16",
                 "float32", "float64", "complex64", "complex128"):  # fmt: skip
        if value is getattr(paddle, name, None):
            return name
    return None


def _decode_literal(node: Any) -> Any:
    """Rebuild one literal from recipe data (inverse of :func:`_encode_literal`)."""
    if isinstance(node, tuple):
        return tuple(_decode_literal(item) for item in node)
    if isinstance(node, list):
        return [_decode_literal(item) for item in node]
    if not isinstance(node, Mapping):
        return node
    if len(node) == 1 and next(iter(node)) in _TAGS:
        ((tag, payload),) = node.items()
        if tag == _TAG_ELLIPSIS:
            return Ellipsis
        if tag == _TAG_SLICE:
            return slice(*_decode_literal(payload))
        if tag == _TAG_NP_SCALAR:
            return np.dtype(payload[0]).type(payload[1])
        if tag == _TAG_NP_DTYPE:
            return np.dtype(payload)
        return _decode_paddle_dtype(payload)
    return {key: _decode_literal(item) for key, item in node.items()}


def _decode_paddle_dtype(name: Any) -> Any:
    """The Paddle dtype object for a recorded ``paddle.<name>``."""
    import paddle

    value = getattr(paddle, str(name), None) if isinstance(name, str) else None
    if value is None or _paddle_dtype_name(value) != name:
        raise ReplayRefusal("type", f"unknown Paddle dtype {name!r}")
    return value


def _skeleton(capture: Any, value: Any, path: tuple[Any, ...], parts: dict[str, list[Any]]) -> Any:
    """Skeleton node for one template value; fills ``parts`` (essential type switch)."""
    if _is_tensor_marker(value):
        param_inputs = getattr(capture, "param_inputs", None) or {}
        label = value.get("label")
        if isinstance(label, str):
            parts["inputs"].append(InputOccurrence(label, _arg_path(path)))
            return _NODE_INPUT
        address = _param_address(capture, path) if path in param_inputs else None
        if address is not None:
            parts["params"].append((_arg_path(path), address))
            return _NODE_PARAM
        return _NODE_HOLE
    has_tensor = any(True for _ in _leaves_of((value,), {}))
    if isinstance(value, (tuple, list)) and has_tensor:
        items = [_skeleton(capture, item, (*path, i), parts) for i, item in enumerate(value)]
        return {"tuple" if isinstance(value, tuple) else "list": items}
    if isinstance(value, dict) and has_tensor:
        if any(type(key) is not str for key in value):
            raise ReplayRefusal("type", "tensor container with a non-str key")
        return {"dict": {k: _skeleton(capture, v, (*path, k), parts) for k, v in value.items()}}
    parts["literals"].append((_arg_path(path), _encode_literal(value)))
    return _NODE_LITERAL


def _param_address(capture: Any, path: tuple[Any, ...]) -> str | None:
    """The recorded parameter address of the tensor leaf at ``path``."""
    for leaf in getattr(capture, "tensor_inputs", ()):
        if tuple(leaf.path) == path:
            return getattr(leaf, "param_address", None)
    return None


def _framework_version() -> str:
    """The installed Paddle version."""
    import paddle

    return str(getattr(paddle, "__version__", ""))


def recipe(trace: Any, op: Any) -> OpRecipe | None:
    """Data-only description of the op's call, or ``None`` (see :func:`recipe_refusal`)."""
    return _recipe_or_reason(trace, op)[0]


def recipe_refusal(trace: Any, op: Any) -> str | None:
    """Why :func:`recipe` returns ``None`` for this op, or ``None`` when it has one."""
    return _recipe_or_reason(trace, op)[1]


def _recipe_or_reason(trace: Any, op: Any) -> tuple[OpRecipe | None, str | None]:
    """The op's recipe, or the reason it has none."""
    reason = can_replay(trace, op)
    site = _site(trace, op)
    if reason is not None or site is None:
        return None, reason or _REASON_NO_CAPTURE
    capture = site.capture
    if getattr(capture, "intervention", None) is not None:
        return None, _REASON_INTERVENED
    parts: dict[str, list[Any]] = {"inputs": [], "params": [], "literals": []}
    try:
        call = {
            "args": tuple(
                _skeleton(capture, item, ("args", index), parts)
                for index, item in enumerate(getattr(capture, "args_template", ()))
            ),
            "kwargs": {
                str(key): _skeleton(capture, item, ("kwargs", str(key)), parts)
                for key, item in dict(getattr(capture, "kwargs_template", {})).items()
            },
        }
        attrs = {
            "call": call,
            "output_paths": tuple(tuple(path) for path in capture.output_leaf_paths),
            "aliases": tuple(_encode_literal(dict(n)) for n in capture.alias_annotations),
        }
    except (TypeError, ReplayRefusal) as exc:
        return None, f"argument not data-encodable: {exc}"
    return (
        OpRecipe(
            backend=BACKEND,
            op_identity=str(capture.op_name),
            inputs=tuple(parts["inputs"]),
            literals=tuple(parts["literals"]),
            attrs=attrs,
            output_slots=site.slot_labels,
            param_refs=tuple(parts["params"]),
            rng=None,
            framework_version=_framework_version(),
        ),
        None,
    )


# --- binding recipes back into a capture list --------------------------------


def _resolve_callable(op_identity: str) -> Any:
    """The original unwrapped Paddle callable for an allowlisted op identity."""
    if op_identity not in ALLOWLIST:
        raise ReplayRefusal("unknown_op", f"{op_identity!r} is not in the paddle allowlist")
    if op_identity == PADDLE_CONSTANT_OP:
        return None
    if op_identity in PADDLE_CONV_LAYER_OPS.values():
        return paddle_conv_layer_replay
    paddle, functional, tensor_cls = wrappers._import_paddle()
    owner: Any = paddle
    name = op_identity
    if op_identity.startswith("functional."):
        owner, name = functional, op_identity.removeprefix("functional.")
    elif op_identity.startswith("tensor."):
        owner, name = tensor_cls, op_identity.removeprefix("tensor.")
    original = wrappers._REGISTRY._originals.get((owner, name))
    return original if original is not None else getattr(owner, name)


def _rebuild_node(
    node: Any, path: tuple[Any, ...], refs: Mapping[str, Mapping[tuple[Any, ...], Any]]
) -> Any:
    """Rebuild one template value from its skeleton node."""
    if node == _NODE_LITERAL:
        return _decode_literal(refs["literals"][_arg_path(path)])
    if node == _NODE_INPUT:
        return {"kind": "tensor", "label": refs["inputs"][_arg_path(path)]}
    if node in (_NODE_PARAM, _NODE_HOLE):
        return {"kind": "tensor", "label": None}
    if isinstance(node, Mapping) and len(node) == 1:
        ((kind, body),) = node.items()
        if kind in ("tuple", "list"):
            items = [_rebuild_node(item, (*path, i), refs) for i, item in enumerate(body)]
            return tuple(items) if kind == "tuple" else items
        if kind == "dict":
            return {key: _rebuild_node(item, (*path, key), refs) for key, item in body.items()}
    raise ReplayRefusal("shape", f"bad call skeleton node at {path!r}")


def _param_value(trace: Any, address: str, payloads: Mapping[str, Any]) -> Any | None:
    """Parameter ``address``'s value: a loader-supplied payload, else the live parameter."""
    if address in payloads:
        import paddle

        with _state.pause_logging():
            return paddle.to_tensor(np.asarray(payloads[address]))
    logs = getattr(trace, "_param_logs", None)
    try:
        param = logs[address] if logs is not None else None
    except (KeyError, IndexError, TypeError):
        return None
    return getattr(param, "value", None) if param is not None else None


def live_param_payloads(trace: Any) -> dict[str, Any]:
    """``{parameter address: host array}`` for every parameter the live captures read.

    The values a saver persists and a loader hands back as
    ``bind_recipes(..., param_payloads=...)`` (a loaded trace keeps no parameter bytes).
    """
    payloads: dict[str, Any] = {}
    for capture in getattr(trace, "_paddle_op_captures", ()) or ():
        values = getattr(capture, "param_inputs", None) or {}
        for leaf in getattr(capture, "tensor_inputs", ()):
            address = getattr(leaf, "param_address", None)
            if address is not None and leaf.path in values and address not in payloads:
                payloads[address] = to_host(values[leaf.path])
    return payloads


def _bind_one(trace: Any, item: OpRecipe, payloads: Mapping[str, Any]) -> Any:
    """Rebuild one ``PaddleOpCapture`` from a recipe (fail-closed on missing params)."""
    label = item.output_slots[0] if item.output_slots else None
    if item.backend != BACKEND:
        raise ReplayRefusal("backend", f"recipe is for {item.backend!r}", op_label=label)
    func = _resolve_callable(item.op_identity)
    refs = {
        "literals": {tuple(path): value for path, value in item.literals},
        "inputs": {tuple(occ.arg_path): occ.parent_label for occ in item.inputs},
    }
    call = item.attrs.get("call", {"args": (), "kwargs": {}})
    args = tuple(_rebuild_node(n, ("args", i), refs) for i, n in enumerate(call["args"]))
    kwargs = {str(k): _rebuild_node(n, ("kwargs", str(k)), refs) for k, n in call["kwargs"].items()}
    params = {_template_path(path): address for path, address in item.param_refs}
    param_inputs: dict[tuple[Any, ...], Any] = {}
    markers: list[str] = []
    for path, address in params.items():
        value = _param_value(trace, address, payloads)
        if value is None:
            markers.append(f"parameter value unavailable at load at {path!r}: {address}")
        else:
            param_inputs[path] = value
    tensor_inputs = tuple(
        TensorLeafCapture(path, marker.get("label"), params[path] if path in param_inputs else None)
        for path, marker in _leaves_of(args, kwargs)
    )
    return PaddleOpCapture(
        func=func,
        op_name=item.op_identity,
        label_raw=str(label),
        args_template=args,
        kwargs_template=kwargs,
        tensor_inputs=tensor_inputs,
        output_leaf_paths=tuple(tuple(p) for p in item.attrs.get("output_paths", ((),))),
        producer_labels=frozenset(occ.parent_label for occ in item.inputs),
        alias_annotations=tuple(dict(note) for note in item.attrs.get("aliases", ())),
        capture_gap_markers=tuple(markers),
        param_inputs=param_inputs,
    )


def _gap_capture(label: str) -> Any:
    """A capture the validator fails at its coverage check, naming ``label``."""
    return PaddleOpCapture(
        func=None,
        op_name="unbound",
        label_raw=label,
        args_template=(),
        kwargs_template={},
        tensor_inputs=(),
        output_leaf_paths=((),),
        producer_labels=frozenset(),
        capture_gap_markers=(f"no replay recipe bound for op {label}",),
    )


def bind_recipes(
    trace: Any,
    recipes: Mapping[str, OpRecipe],
    *,
    param_payloads: Mapping[str, Any] | None = None,
) -> None:
    """Rebuild ``trace._paddle_op_captures`` from recipes plus the trace's payloads.

    ``param_payloads`` maps parameter addresses to saved arrays (from
    :func:`live_param_payloads` at save); it is read before the live parameter,
    which a loaded trace does not have. Every non-input op the recipes do not
    cover gets a gap capture, so the unchanged validator fails naming it
    instead of skipping it.
    """
    payloads = dict(param_payloads or {})

    by_first: dict[str, OpRecipe] = {}
    for key, item in recipes.items():
        if not item.output_slots or item.output_slots[0] != key:
            raise ReplayRefusal("slots", f"recipe keyed {key!r} starts elsewhere", op_label=key)
        by_first[key] = item
    covered = {slot for item in by_first.values() for slot in item.output_slots}
    captures: list[Any] = []
    for op in _ops(trace):
        label = getattr(op, "_label_raw", None)
        if not isinstance(label, str) or bool(record_member(op, "is_input", False)):
            continue
        if label in by_first:
            captures.append(_bind_one(trace, by_first[label], payloads))
        elif label not in covered:
            captures.append(_gap_capture(label))
    trace._paddle_op_captures = captures
    trace._paddle_alias_annotations = [n for c in captures for n in c.alias_annotations]
    trace._paddle_capture_gap_markers = [m for c in captures for m in c.capture_gap_markers]


__all__ = [
    "ALLOWLIST",
    "BACKEND",
    "add",
    "bind_recipes",
    "can_replay",
    "close",
    "forward",
    "live_param_payloads",
    "recipe",
    "recipe_refusal",
    "to_host",
    "vjp",
    "zeros_like",
]
