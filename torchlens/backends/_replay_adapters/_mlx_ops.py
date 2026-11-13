"""Op identity, capture lookup and template walk for the MLX replay adapter.

The walk mirrors ``mlx/_replay_args._rebuild_node`` exactly: one labeled
array leaf, or ``REPLAY_SLOT``, consumes one recorded parent label in
flatten order, and containers are exact ``tuple``/``list``/``dict`` only.
MLX itself is imported inside functions.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterator, Sequence
from typing import Any

from .._validation_shared import ops_by_label
from ..mlx import wrappers as w
from ..mlx._capture_records import _is_mlx_array, _ReplaySlot
from ..mlx._replay_args import _saved_payload
from ._contract import ReplayRefusal

BACKEND = "mlx"

_WRAPPER_QUALNAME_TAIL = "wrap_attr.<locals>.wrapper"
_RANDOM_LAYERS = frozenset({"Dropout", "Dropout2d", "Dropout3d"})

# --------------------------------------------------------------- op identity


def _original(owner: Any, name: str) -> Any:
    """The callable the wrapper registry wraps at ``owner.name`` (unwrapped)."""

    attr = inspect.getattr_static(owner, name, None)
    if isinstance(attr, property):
        attr = attr.fget
    qualname = getattr(attr, "__qualname__", None)
    if isinstance(qualname, str) and qualname.endswith(_WRAPPER_QUALNAME_TAIL):
        attr = getattr(attr, "__wrapped__", attr)
    return attr


def _owners() -> dict[str, tuple[Any, tuple[str, ...]]]:
    """Identity kind to ``(owner, attribute names)`` for the function-like ops."""

    import mlx.core as mx
    import mlx.nn as nn

    activations = w.mlx_activations_module(nn)
    nn_names: tuple[str, ...] = ("relu", "gelu", "sigmoid", "tanh", "softmax", "silu")
    nn_names += w.mlx_activation_function_names(activations) if activations else ()
    array_names = (*w.MLX_ARRAY_DUNDER_OPS, *w.MLX_ARRAY_METHOD_OPS, *w.MLX_ARRAY_PROPERTY_OPS)
    return {
        "core": (mx, w.MLX_CORE_FUNCTION_OPS),
        "fast": (getattr(mx, "fast", None), w.MLX_FAST_FUNCTION_OPS),
        "array": (mx.array, array_names),
        "nn": (nn, tuple(dict.fromkeys(nn_names))),
    }


def _layer_classes() -> dict[str, type[Any]]:
    """Allowlisted ``mlx.nn`` layer classes by class name."""

    import mlx.nn as nn

    classes = {name: getattr(nn, name, None) for name in w.MLX_LAYER_CLASS_OPS}
    activations = w.mlx_activations_module(nn)
    if activations is not None:
        for cls in w.mlx_activation_layer_classes(activations, nn):
            classes.setdefault(cls.__name__, cls)
    return {name: cls for name, cls in classes.items() if isinstance(cls, type)}


def _function_index() -> dict[int, str]:
    """``id(original callable) -> identity`` over the function-like ops.

    Rebuilt per call: the module keeps no state between traces.
    """

    index: dict[int, str] = {}
    for kind, (owner, names) in _owners().items():
        for name in names if owner is not None else ():
            func = _original(owner, name)
            if func is not None:
                index.setdefault(id(func), f"{kind}.{name}")
    return index


def op_allowlist() -> frozenset[str]:
    """The closed set of MLX op identities a recipe may name."""

    names = {"constant", "input"}
    for kind, (owner, attrs) in _owners().items():
        if owner is not None:
            names.update(f"{kind}.{name}" for name in attrs if _original(owner, name))
    names.update(f"layer.{name}" for name in _layer_classes())
    return frozenset(names)


def _identity(capture: Any) -> str | None:
    """The allowlist identity of a capture's callable, or ``None`` when unlisted."""

    if capture.func is None:
        return "constant" if capture.op_name == "constant" else None
    module = capture.args[0] if capture.args else None
    cls = _layer_classes().get(type(module).__name__)
    if cls is not None and type(module) is cls and capture.func is _original(cls, "__call__"):
        return f"layer.{cls.__name__}"
    return _function_index().get(id(capture.func))


def _resolve(identity: str) -> tuple[Any, type[Any] | None]:
    """The callable (and layer class) an allowlisted identity names."""

    if identity in ("constant", "input"):
        return None, None
    kind, _, name = identity.partition(".")
    if kind == "layer":
        cls = _layer_classes()[name]
        return _original(cls, "__call__"), cls
    owner, names = _owners()[kind]
    if name not in names:
        raise ReplayRefusal("unknown_op", f"op identity {identity!r} is not allowlisted")
    return _original(owner, name), None


# ------------------------------------------------------------ capture lookup


def _captures(trace: Any) -> list[Any]:
    """The trace's runtime replay captures."""

    return list(getattr(trace, "_mlx_op_captures", None) or ())


def _capture_index(trace: Any) -> dict[str, Any]:
    """Raw output label to capture, built from the current capture list."""

    captures = getattr(trace, "_mlx_op_captures", None) or ()
    return {label: capture for capture in captures for label in capture.labels_raw}


def _label_of(op: Any) -> str:
    """Raw label of an op record, a capture or a label string."""

    if isinstance(op, str):
        return op
    labels = getattr(op, "labels_raw", None)
    if labels:
        return str(labels[0])
    label = getattr(op, "_label_raw", None) or getattr(op, "label", None)
    if not isinstance(label, str):
        raise ReplayRefusal("unknown_label", f"no label on {type(op).__name__}")
    return label


def _capture_for(trace: Any, op: Any) -> Any:
    """The capture that produced ``op`` (raises when none was recorded)."""

    if hasattr(op, "labels_raw") and hasattr(op, "func"):
        return op
    label = _label_of(op)
    capture = _capture_index(trace).get(label)
    if capture is None:
        ops = _ops_by_label(trace)
        alias = getattr(ops.get(label), "_label_raw", None)
        capture = _capture_index(trace).get(alias) if isinstance(alias, str) else None
    if capture is None:
        raise ReplayRefusal("unknown_label", "no replay capture recorded", op_label=label)
    return capture


def _is_input(trace: Any, op: Any) -> bool:
    """Whether ``op`` (a record, capture or label) is a model input op."""

    if hasattr(op, "labels_raw") and hasattr(op, "func"):
        return False
    record = _ops_by_label(trace).get(op) if isinstance(op, str) else op
    return bool(getattr(record, "is_input", False))


def _input_payload(trace: Any, op: Any) -> Any:
    """The saved payload of an input op."""

    return _saved_payload(trace, _ops_by_label(trace), _label_of(op))


def _ops_by_label(trace: Any) -> dict[str, Any]:
    """The validator's label-to-op index."""

    return ops_by_label(trace)


# ------------------------------------------------------------- template walk


def _walk(
    node: Any,
    cursor: Iterator[str | None],
    path: tuple[Any, ...],
    leaf: Callable[[Any, str | None, tuple[Any, ...]], Any],
    other: Callable[[Any, tuple[Any, ...]], Any],
) -> Any:
    """Walk one template exactly as ``mlx._replay_args._rebuild_node`` does.

    ``leaf`` receives each array or ``REPLAY_SLOT`` with the recorded label
    the walk consumes for it; ``other`` receives every non-container leaf.
    Containers are exact ``tuple``/``list``/``dict`` only (a module stays one
    leaf), so walk order and label alignment match the validator's.
    """

    if _is_mlx_array(node) or isinstance(node, _ReplaySlot):
        return leaf(node, next(cursor, None), path)
    if type(node) is tuple:
        return tuple(_walk(v, cursor, (*path, i), leaf, other) for i, v in enumerate(node))
    if type(node) is list:
        return [_walk(v, cursor, (*path, i), leaf, other) for i, v in enumerate(node)]
    if type(node) is dict:
        return {k: _walk(v, cursor, (*path, k), leaf, other) for k, v in node.items()}
    return other(node, path)


def _walk_call(
    capture: Any,
    leaf: Callable[[Any, str | None, tuple[Any, ...]], Any],
    other: Callable[[Any, tuple[Any, ...]], Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Walk every positional and keyword template of a capture."""

    arg_labels = capture.arg_leaf_labels
    args = tuple(
        _walk(
            value,
            iter(arg_labels[i] if i < len(arg_labels) else ()),
            (i,),
            leaf,
            other,
        )
        for i, value in enumerate(capture.args)
    )
    kwargs = {
        key: _walk(value, iter(capture.kwarg_leaf_labels.get(key, ())), (key,), leaf, other)
        for key, value in capture.kwargs.items()
    }
    return args, kwargs


def _occurrences(capture: Any) -> list[tuple[str, tuple[Any, ...]]]:
    """``(parent label, argument path)`` per labeled leaf, in walk order."""

    found: list[tuple[str, tuple[Any, ...]]] = []

    def leaf(node: Any, label: str | None, path: tuple[Any, ...]) -> Any:
        if label is not None:
            found.append((label, path))
        return node

    _walk_call(capture, leaf, lambda node, path: node)
    return found


def _bind_values(
    capture: Any, parent_values: Sequence[Any]
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """The capture's call material with each occurrence bound to its value."""

    values = iter(parent_values)
    missing = object()

    def leaf(node: Any, label: str | None, path: tuple[Any, ...]) -> Any:
        if label is None:
            if isinstance(node, _ReplaySlot):
                raise ReplayRefusal("slots", "template slot has no recorded parent label")
            return node
        value = next(values, missing)
        if value is missing:
            raise ReplayRefusal("arity", f"too few parent values for {capture.op_name!r}")
        return value

    args, kwargs = _walk_call(capture, leaf, lambda node, path: node)
    if next(values, missing) is not missing:
        raise ReplayRefusal("arity", f"too many parent values for {capture.op_name!r}")
    return args, kwargs
