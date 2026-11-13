"""Data-only recipes for MLX replay captures, and binding them back.

A recipe describes one capture with data only: the op identity (a key of
the adapter's ``ALLOWLIST``), the argument skeleton (containers and leaf
kinds, in ``attrs["args"]``/``attrs["kwargs"]``), plain literals, parameter
addresses, structurally encoded special leaves (dtypes, slices, ``nn``
layer modules rebuilt from their class, data attributes and parameter
addresses) and the emit-time leaf-label fingerprint the validator's
coverage check compares. :func:`bind_recipes` rebuilds the capture list
from recipes, the trace's saved op payloads and parameter payloads keyed by
address (the ``param_payloads`` a loader saved from
:func:`live_param_payloads`); the unchanged validator then replays it. MLX itself is imported inside functions.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from .._finalize import normalize_op_module_calls
from ..mlx._capture_records import REPLAY_SLOT, MLXOpCapture, _ReplaySlot
from ..mlx.validation import _constant_source_labels
from ._contract import InputOccurrence, OpRecipe, ReplayRefusal
from ._mlx_ops import (
    _RANDOM_LAYERS,
    BACKEND,
    _capture_for,
    _captures,
    _function_index,
    _identity,
    _is_input,
    _label_of,
    _layer_classes,
    _ops_by_label,
    _resolve,
    _walk_call,
    op_allowlist,
)

_DTYPE_NAMES = frozenset(
    {
        "bool_",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "int8",
        "int16",
        "int32",
        "int64",
        "float16",
        "float32",
        "float64",
        "bfloat16",
        "complex64",
    }
)
_MAX_ARRAY_LITERAL = 4096
_PLAIN_TYPES = (type(None), bool, int, float, complex, str)

# ------------------------------------------------------------------- recipes


_NOT_DATA = "not_data"


def _not_data(detail: str) -> ReplayRefusal:
    """The refusal for a capture leaf with no data-only encoding."""

    return ReplayRefusal(_NOT_DATA, detail)


class _Leaf:
    """One encoded skeleton leaf, kept apart from walked containers."""

    __slots__ = ("payload",)

    def __init__(self, payload: Any) -> None:
        self.payload = payload


def _tag(node: Any) -> Any:
    """Turn a walked skeleton into tagged data: one-key mappings or leaf markers."""

    if isinstance(node, _Leaf):
        return node.payload
    if type(node) is tuple:
        return {"tuple": tuple(_tag(v) for v in node)}
    if type(node) is list:
        return {"list": tuple(_tag(v) for v in node)}
    return {"dict": {k: _tag(v) for k, v in node.items()}}


def _is_plain(value: Any) -> bool:
    """Whether ``value`` is a plain data literal (scalars in exact containers)."""

    if type(value) in _PLAIN_TYPES:
        return not isinstance(value, str) or len(value) < 65_536
    if type(value) in (tuple, list):
        return all(_is_plain(v) for v in value)
    if type(value) is dict:
        return all(type(k) is str and _is_plain(v) for k, v in value.items())
    return False


def _dtype_name(value: Any) -> str | None:
    """The ``mlx.core`` attribute name of a dtype object, or ``None``."""

    import mlx.core as mx

    if not isinstance(value, mx.Dtype):
        return None
    name = str(value).rsplit(".", 1)[-1]
    name = "bool_" if name == "bool" else name
    return name if name in _DTYPE_NAMES and getattr(mx, name, None) == value else None


def _array_literal(value: Any) -> dict[str, Any]:
    """A small unlabeled array as data (dtype, shape, flat values)."""

    import numpy as np

    dtype = _dtype_name(value.dtype)
    if dtype is None or dtype == "bfloat16" or value.size > _MAX_ARRAY_LITERAL:
        raise _not_data(f"unlabeled {value.dtype} array of {value.size} values has no data form")
    host = np.array(value)
    return {"dtype": dtype, "shape": tuple(host.shape), "data": tuple(host.ravel().tolist())}


def _join(owner: str, key: Any) -> str:
    """Parameter address of ``key`` under module address ``owner``."""

    return str(key) if owner in ("", "self") else f"{owner}.{key}"


class _RecipeBuilder:
    """Collects one capture's occurrences, literals and references while walking."""

    def __init__(self, trace: Any, capture: Any) -> None:
        self.trace = trace
        self.capture = capture
        self.inputs: list[InputOccurrence] = []
        self.literals: list[tuple[tuple[Any, ...], Any]] = []
        self.param_refs: list[tuple[tuple[Any, ...], str]] = []
        self.constant_refs: list[tuple[tuple[Any, ...], str]] = []
        self.payloads: dict[str, Any] = {}
        self.params = list(getattr(trace, "_param_logs", None) or ())
        self.addresses = {a for p in self.params for a in (p.address, *p.all_addresses)}
        self.constants = _constant_labels(trace)
        self.owner = _owner_address(trace, capture)

    def leaf(self, node: Any, label: str | None, path: tuple[Any, ...]) -> Any:
        """Encode an array or slot leaf: an occurrence, a parameter or a literal."""

        if label is not None:
            self.inputs.append(InputOccurrence(label, path))
            if label in self.constants:
                self.constant_refs.append((path, label))
            return _Leaf("in")
        if isinstance(node, _ReplaySlot):
            raise _not_data("template slot has no recorded parent label")
        address = self._direct_param_address(node)
        if address is not None:
            self.param_refs.append((path, address))
            self.payloads[address] = node
            return _Leaf("param")
        return _Leaf({"array": _array_literal(node)})

    def other(self, node: Any, path: tuple[Any, ...]) -> Any:
        """Encode a non-array leaf: a plain literal or a structural special."""

        if type(node) in _PLAIN_TYPES:
            self.literals.append((path, node))
            return _Leaf("lit")
        return _Leaf(self._special(node, path))

    def _special(self, node: Any, path: tuple[Any, ...]) -> dict[str, Any]:
        """Dtype, slice, ellipsis or layer-module leaves as tagged data."""

        dtype = _dtype_name(node)
        if dtype is not None:
            return {"dtype": dtype}
        if node is Ellipsis:
            return {"ellipsis": None}
        if isinstance(node, slice):
            parts = (node.start, node.stop, node.step)
            if not all(p is None or type(p) is int for p in parts):
                raise _not_data("slice with non-integer bounds")
            return {"slice": parts}
        identity = _identity(self.capture) if self.capture.func is not None else None
        if path == (0,) and identity is not None and identity.startswith("layer."):
            return {"module": self.module_spec(node, self.owner)}
        raise _not_data(f"argument leaf of type {type(node).__name__} has no data form")

    def _direct_param_address(self, node: Any) -> str | None:
        """Address of a parameter passed straight to an op (unique shape match)."""

        if id(node) not in getattr(self.trace, "_mlx_param_ids", frozenset()):
            return None
        shape, dtype = tuple(node.shape), str(node.dtype)
        matches = [p for p in self.params if tuple(p.shape) == shape and str(p.dtype) == dtype]
        owned = [p for p in matches if p.module_address == self.owner]
        candidates = owned or matches
        if len(candidates) != 1:
            raise _not_data(f"parameter leaf address ambiguous ({len(candidates)} candidates)")
        return str(candidates[0].address)

    def module_spec(self, module: Any, address: str) -> dict[str, Any]:
        """A layer module as data: class, data attributes, items by address."""

        cls = _layer_classes().get(type(module).__name__)
        if cls is None or type(module) is not cls:
            raise _not_data(f"module class {type(module).__name__} is not allowlisted")
        return {
            "class": cls.__name__,
            "state": {key: _encode_attr(value) for key, value in vars(module).items()},
            "items": {key: self._item(value, _join(address, key)) for key, value in module.items()},
        }

    def _item(self, value: Any, address: str) -> Any:
        """One module item (parameter, child module, container or literal)."""

        import mlx.core as mx
        import mlx.nn as nn

        if isinstance(value, mx.array):
            if address in self.addresses:
                self.payloads[address] = value
                return {"param": address}
            return {"array": _array_literal(value)}
        if isinstance(value, nn.Module):
            return {"module": self.module_spec(value, address)}
        if _is_plain(value):
            return {"lit": value}
        if type(value) in (tuple, list):
            kind = "tuple" if type(value) is tuple else "list"
            return {kind: tuple(self._item(v, _join(address, i)) for i, v in enumerate(value))}
        if type(value) is dict:
            return {"dict": {k: self._item(v, _join(address, k)) for k, v in value.items()}}
        raise _not_data(f"module item of type {type(value).__name__} has no data form")


def _encode_attr(value: Any) -> dict[str, Any]:
    """One module instance attribute as tagged data."""

    if _is_plain(value):
        return {"v": value}
    if type(value) in (set, frozenset) and _is_plain(sorted(value, key=repr)):
        return {"set": tuple(sorted(value, key=repr))}
    dtype = _dtype_name(value)
    if dtype is not None:
        return {"dtype": dtype}
    identity = _function_index().get(id(value)) if callable(value) else None
    if identity is not None:
        return {"fn": identity}
    raise _not_data(f"module attribute of type {type(value).__name__} has no data form")


def _constant_labels(trace: Any) -> frozenset[str]:
    """Raw labels of the trace's constant source ops."""

    return _constant_source_labels(trace)


def _owner_address(trace: Any, capture: Any) -> str:
    """Innermost module address of the capture's first output op."""

    if not capture.labels_raw:
        return ""
    op = _ops_by_label(trace).get(capture.labels_raw[0])
    calls = normalize_op_module_calls(getattr(op, "modules", ()) or ())
    return calls[-1][0] if calls else ""


def _framework_version() -> str:
    """Installed MLX version."""

    import mlx.core as mx

    return str(getattr(mx, "__version__", ""))


def _recipe_or_reason(trace: Any, op: Any) -> OpRecipe | str:
    """The recipe for ``op``, or why it has none."""

    if _is_input(trace, op):
        label = _label_of(op)
        attrs = {"op_name": "input", "args": (), "kwargs": {}}
        return OpRecipe(BACKEND, "input", (), attrs=attrs, output_slots=(label,))
    capture = _capture_for(trace, op)
    identity = _identity(capture)
    if identity is None:
        return f"unlisted op {capture.op_name!r}"
    builder = _RecipeBuilder(trace, capture)
    try:
        args, kwargs = _walk_call(capture, builder.leaf, builder.other)
    except ReplayRefusal as exc:
        if exc.reason != _NOT_DATA:
            raise
        return str(exc)
    module = capture.args[0] if identity.startswith("layer.") else None
    random = identity.partition(".")[2] in _RANDOM_LAYERS and getattr(module, "training", False)
    return OpRecipe(
        backend=BACKEND,
        op_identity=identity,
        inputs=tuple(builder.inputs),
        literals=tuple(builder.literals),
        attrs={
            "op_name": capture.op_name,
            "args": tuple(_tag(a) for a in args),
            "kwargs": {k: _tag(v) for k, v in kwargs.items()},
            "arg_leaf_labels": tuple(tuple(s) for s in capture.arg_leaf_labels),
            "kwarg_leaf_labels": {k: tuple(s) for k, s in capture.kwarg_leaf_labels.items()},
            "interventions": tuple(tuple(i) for i in capture.interventions),
        },
        output_slots=tuple(capture.labels_raw),
        param_refs=tuple(builder.param_refs),
        constant_refs=tuple(builder.constant_refs),
        rng={"source": "mx.random", "replayable": False} if random else None,
        framework_version=_framework_version(),
    )


def recipe(trace: Any, op: Any) -> OpRecipe | None:
    """Data-only description of ``op``'s capture, or ``None`` when it has none."""

    result = _recipe_or_reason(trace, op)
    return result if isinstance(result, OpRecipe) else None


def live_param_payloads(trace: Any) -> dict[str, Any]:
    """``{parameter address: array}`` for every parameter the live captures hold.

    A live trace's templates keep parameter values; this collects them under
    the addresses :func:`recipe` writes, as the payloads a loader saves and
    passes back to :func:`bind_recipes` as ``param_payloads``.
    """

    payloads: dict[str, Any] = {}
    for capture in _captures(trace):
        if _identity(capture) is None or capture.func is None:
            continue
        builder = _RecipeBuilder(trace, capture)
        try:
            _walk_call(capture, builder.leaf, builder.other)
        except ReplayRefusal as exc:
            if exc.reason != _NOT_DATA:
                raise
            continue
        payloads.update(builder.payloads)
    return payloads


# ------------------------------------------------------------------ binding


class _Binder:
    """Rebuilds one capture's templates from a recipe and parameter payloads."""

    def __init__(self, rec: OpRecipe, params: Mapping[str, Any]) -> None:
        self.rec = rec
        self.params = params
        self.literals = dict(rec.literals)
        self.param_refs = dict(rec.param_refs)
        self.in_paths: list[tuple[Any, ...]] = []

    def node(self, node: Any, path: tuple[Any, ...]) -> Any:
        """Rebuild one skeleton node at ``path``."""

        if node == "in":
            self.in_paths.append(path)
            return REPLAY_SLOT
        if node == "lit":
            return self._lookup(self.literals, path, "literal")
        if node == "param":
            return self.param(self._lookup(self.param_refs, path, "parameter"))
        if not isinstance(node, Mapping) or len(node) != 1:
            raise self.error("skeleton node is not a one-key mapping")
        ((kind, body),) = node.items()
        return self._tagged(kind, body, path)

    def _tagged(self, kind: str, body: Any, path: tuple[Any, ...]) -> Any:
        """Rebuild a container or special leaf (essential kind switch)."""

        if kind in ("tuple", "list"):
            items = [self.node(v, (*path, i)) for i, v in enumerate(body)]
            return tuple(items) if kind == "tuple" else items
        if kind == "dict":
            return {k: self.node(v, (*path, k)) for k, v in body.items()}
        if kind in ("dtype", "array", "slice", "ellipsis"):
            return _special_value(kind, body, self.error)
        if kind == "module":
            return self.module(body)
        raise self.error(f"unknown skeleton kind {kind!r}")

    def _lookup(self, table: Mapping[Any, Any], path: tuple[Any, ...], what: str) -> Any:
        """The recipe entry at ``path`` (refused when missing)."""

        if path not in table:
            raise self.error(f"no {what} recorded at argument path {path!r}")
        return table[path]

    def param(self, address: str) -> Any:
        """The parameter payload for ``address`` (refused when absent)."""

        if address not in self.params:
            raise self.error(f"no parameter payload for {address!r}")
        return self.params[address]

    def module(self, spec: Mapping[str, Any]) -> Any:
        """Rebuild an allowlisted layer module from its data spec."""

        import mlx.nn as nn

        cls: Any = _layer_classes().get(spec.get("class", ""))
        if cls is None:
            raise self.error(f"module class {spec.get('class')!r} is not allowlisted")
        module = cls.__new__(cls)
        nn.Module.__init__(module)
        for key, value in spec["state"].items():
            module.__dict__[key] = _decode_attr(value, self.error)
        for key, value in spec["items"].items():
            dict.__setitem__(module, key, self._item(value))
        return module

    def _item(self, node: Mapping[str, Any]) -> Any:
        """Rebuild one module item."""

        ((kind, body),) = node.items()
        if kind == "param":
            return self.param(body)
        if kind == "module":
            return self.module(body)
        if kind == "lit":
            return body
        if kind in ("tuple", "list"):
            items = [self._item(v) for v in body]
            return tuple(items) if kind == "tuple" else items
        if kind == "dict":
            return {k: self._item(v) for k, v in body.items()}
        return _special_value(kind, body, self.error)

    def error(self, detail: str) -> ReplayRefusal:
        """A refusal naming this recipe's op."""

        slots = self.rec.output_slots
        return ReplayRefusal("shape", detail, op_label=slots[0] if slots else None)


def _special_value(kind: str, body: Any, error: Callable[[str], Exception]) -> Any:
    """Rebuild a dtype, array literal, slice or ellipsis leaf."""

    import mlx.core as mx
    import numpy as np

    if kind == "ellipsis":
        return Ellipsis
    if kind == "slice":
        return slice(*body)
    name = body if kind == "dtype" else body.get("dtype")
    if name not in _DTYPE_NAMES:
        raise error(f"dtype {name!r} is not allowlisted")
    if kind == "dtype":
        return getattr(mx, name)
    host = np.array(body["data"], dtype=np.dtype(name.rstrip("_"))).reshape(body["shape"])
    return mx.array(host, dtype=getattr(mx, name))


def _decode_attr(node: Mapping[str, Any], error: Callable[[str], Exception]) -> Any:
    """Rebuild one module instance attribute."""

    ((kind, body),) = node.items()
    if kind == "v":
        return body
    if kind == "set":
        return set(body)
    if kind == "dtype":
        return _special_value("dtype", body, error)
    if kind == "fn":
        if body not in op_allowlist():
            raise error(f"function {body!r} is not allowlisted")
        return _resolve(body)[0]
    raise error(f"unknown module attribute kind {kind!r}")


def _capture_from_recipe(rec: OpRecipe, params: Mapping[str, Any]) -> Any:
    """One ``MLXOpCapture`` rebuilt from a recipe (refused when inconsistent)."""

    if rec.backend != BACKEND or rec.op_identity not in op_allowlist():
        raise ReplayRefusal("unknown_op", f"{rec.backend}:{rec.op_identity} is not allowlisted")
    func, _cls = _resolve(rec.op_identity)
    binder = _Binder(rec, params)
    attrs = rec.attrs
    args = tuple(binder.node(node, (i,)) for i, node in enumerate(attrs.get("args", ())))
    kwargs = {k: binder.node(node, (k,)) for k, node in dict(attrs.get("kwargs", {})).items()}
    if binder.in_paths != [occ.arg_path for occ in rec.inputs]:
        raise binder.error("input occurrences differ from the argument skeleton")
    return MLXOpCapture(
        labels_raw=tuple(rec.output_slots),
        op_name=str(attrs.get("op_name", "")),
        func=func,
        args=args,
        kwargs=kwargs,
        arg_leaf_labels=tuple(tuple(s) for s in attrs.get("arg_leaf_labels", ())),
        kwarg_leaf_labels={
            k: tuple(s) for k, s in dict(attrs.get("kwarg_leaf_labels", {})).items()
        },
        interventions=tuple((int(i), str(h)) for i, h in attrs.get("interventions", ())),
    )


def _as_mlx(value: Any) -> Any:
    """An MLX array for a saved parameter payload (NumPy values are converted)."""

    import mlx.core as mx

    return value if isinstance(value, mx.array) else mx.array(value)


def _fingerprint(capture: Any) -> tuple[Any, ...]:
    """The replay-inventory row the MLX backend writes for a capture."""

    return (
        capture.op_name,
        capture.labels_raw,
        capture.arg_leaf_labels,
        tuple(sorted((key, labels) for key, labels in capture.kwarg_leaf_labels.items())),
        capture.interventions,
    )


def bind_recipes(
    trace: Any,
    recipes: Mapping[str, OpRecipe],
    *,
    param_payloads: Mapping[str, Any] | None = None,
) -> None:
    """Rebuild ``trace._mlx_op_captures`` from recipes and loaded payloads.

    Parameter values come from ``param_payloads`` (address to MLX or NumPy
    array; NumPy values become MLX arrays of the same dtype); op inputs replay from the trace's saved op payloads, as in live
    validation. When the trace has no replay inventory (a loaded trace), one
    is written from the rebuilt captures. Every recipe is checked before the
    capture list is replaced, so a refused recipe leaves the trace unchanged.
    Declared intervention hooks cannot be rebuilt from data, so such ops
    fail validation closed (the validator's own rule).
    """

    params = {key: _as_mlx(value) for key, value in (param_payloads or {}).items()}
    order = {
        getattr(op, "_label_raw", None): index
        for index, op in enumerate(getattr(trace, "_layer_list", ()) or ())
    }
    ordered = sorted(recipes.values(), key=lambda r: order.get(r.output_slots[0], len(order)))
    captures = [_capture_from_recipe(rec, params) for rec in ordered if rec.op_identity != "input"]
    trace._mlx_op_captures = captures
    if getattr(trace, "_mlx_replay_inventory", None) is None:
        trace._mlx_replay_inventory = [_fingerprint(capture) for capture in captures]
