"""The shared constants rule, Paddle side (closure constants and literal producers).

A constant is a Paddle tensor the forward used that is neither an input leaf,
a registered parameter of the captured layer tree, nor the output of a
recorded op. Two kinds exist:

* a closure constant: a tensor built before the traced call (a closure cell,
  a global the traced code names, a default argument, a plain layer attribute).
  The capture records it as a source op with ``func_name="constant"``, no
  parents and a saved payload; the consuming op lists it as a parent.
* a literal producer: a recorded op with no tensor inputs (``paddle.ones``,
  ``paddle.full``, ...). It stays an op with zero parents; validation replays
  it from its saved arguments.

The rule's safety property: a tensor is recorded as a closure constant only
when it is PROVEN to exist before the forward, by finding it on a path the
traced code can reach (:func:`find_closure_tensors`). A tensor created during
the forward by an operation TorchLens did not wrap is never found there, so it
stays an untraced input and validation still fails on it.

This module imports no Paddle at import time.
"""

from __future__ import annotations

import functools
import types
from collections.abc import Callable, Iterable, Mapping
from typing import Any

#: ``func_name`` of a recorded closure constant (SPEC 2 rule 1).
PADDLE_CONSTANT_OP = "constant"

#: Recorded ops that take no tensor input and build a value from literals.
#: ``*_like`` factories are excluded: they read a tensor argument's shape.
LITERAL_PRODUCER_OPS: frozenset[str] = frozenset(
    {"arange", "eye", "full", "linspace", "ones", "to_tensor", "zeros"}
)

#: How deep the reachability walk follows containers, helper functions and
#: layer attributes. Anything deeper is not proven and stays a capture gap.
_MAX_WALK_DEPTH = 4

#: Layer attributes the walk never treats as constant sources: parameters are
#: their own leaf kind, registered buffers are recorded as ``buffer`` sources
#: (``source_leaves``),
#: and sublayers are walked as layers.
_SKIPPED_LAYER_ATTRIBUTES = frozenset({"_parameters", "_buffers", "_sub_layers"})


def is_constant_capture(capture: Any) -> bool:
    """Whether ``capture`` is a closure-constant record minted by this rule.

    Parameters
    ----------
    capture:
        A ``PaddleOpCapture``.

    Returns
    -------
    bool
        True only for records with ``op_name == "constant"`` and no callable.
    """

    return getattr(capture, "op_name", None) == PADDLE_CONSTANT_OP and (
        getattr(capture, "func", None) is None
    )


def is_literal_producer_capture(capture: Any) -> bool:
    """Whether ``capture`` is a literal producer (no tensor inputs at all).

    Parameters
    ----------
    capture:
        A ``PaddleOpCapture``.

    Returns
    -------
    bool
        True for an allowlisted factory with an empty ``tensor_inputs``.
    """

    base_name = str(getattr(capture, "op_name", "")).rsplit(".", 1)[-1]
    return base_name in LITERAL_PRODUCER_OPS and not tuple(getattr(capture, "tensor_inputs", ()))


def constant_labels(captures: Iterable[Any]) -> frozenset[str]:
    """Raw labels of every closure constant recorded in ``captures``.

    Parameters
    ----------
    captures:
        The trace's ``_paddle_op_captures``.

    Returns
    -------
    frozenset[str]
        Raw labels of the constant records.
    """

    return frozenset(str(capture.label_raw) for capture in captures if is_constant_capture(capture))


def constant_layer_labels(trace: Any) -> frozenset[str]:
    """Public labels of every op the constants rule marks on a Paddle trace.

    Closure constants and literal producers both count. This is the capture-side
    answer Paddle validation uses; the record carries the shared rule's stamp
    as ``Op.is_constant``.

    Parameters
    ----------
    trace:
        A live Paddle trace.

    Returns
    -------
    frozenset[str]
        Labels (``layer_label`` where resolvable, else raw) of constant ops.
    """

    from .validation import _ops_by_label

    ops_by_label = _ops_by_label(trace)
    labels: set[str] = set()
    for capture in tuple(trace._paddle_op_captures):
        if not (is_constant_capture(capture) or is_literal_producer_capture(capture)):
            continue
        op = ops_by_label.get(str(capture.label_raw))
        label = getattr(op, "layer_label", None) if op is not None else None
        labels.add(label if isinstance(label, str) else str(capture.label_raw))
    return frozenset(labels)


def find_closure_tensors(
    root: Any,
    tensor_type: type[Any],
    layer_type: type[Any] | None,
    *,
    exclude_ids: Iterable[int] = (),
) -> dict[int, Any]:
    """Find tensors the traced callable can reach without running it.

    The walk visits the root's closure cells, the globals its code names,
    default arguments, bound ``self``, ``functools.partial`` pieces, helper
    functions it names (recursively), and for a layer every plain attribute of
    the layer and its sublayers plus the ``forward`` methods' closures and
    named globals. Registered parameters, buffers and modules are skipped.

    Parameters
    ----------
    root:
        The traced model (a ``paddle.nn.Layer``) or callable.
    tensor_type:
        ``paddle.Tensor``.
    layer_type:
        ``paddle.nn.Layer``, or ``None`` when unavailable.
    exclude_ids:
        Object ids never returned (registered parameters, input leaves).

    Returns
    -------
    dict[int, Any]
        Found tensors keyed by ``id``. The caller keeps this mapping alive for
        the capture so the ids cannot be reused by new tensors.
    """

    walker = _ReachabilityWalk(tensor_type, layer_type, frozenset(exclude_ids))
    walker.visit(root, 0)
    return walker.found


class _ReachabilityWalk:
    """Bounded walk over what a callable can reach before it runs."""

    def __init__(
        self, tensor_type: type[Any], layer_type: type[Any] | None, exclude: frozenset[int]
    ) -> None:
        self.tensor_type = tensor_type
        self.layer_type = layer_type
        self.exclude = exclude
        self.found: dict[int, Any] = {}
        self._seen: set[int] = set()

    def visit(self, value: Any, depth: int) -> None:
        """Visit one value (dispatch by kind)."""

        if depth > _MAX_WALK_DEPTH or id(value) in self._seen:
            return
        if isinstance(value, self.tensor_type):
            if id(value) not in self.exclude:
                self.found[id(value)] = value
            return
        if isinstance(value, (types.ModuleType, type, str, bytes, int, float, complex, bool)):
            return
        self._seen.add(id(value))
        if self.layer_type is not None and isinstance(value, self.layer_type):
            self._visit_layer(value, depth)
        elif isinstance(value, functools.partial):
            self._visit_many((value.func, *value.args, *value.keywords.values()), depth)
        elif isinstance(value, types.MethodType):
            self.visit(value.__self__, depth + 1)
            self.visit(value.__func__, depth + 1)
        elif isinstance(value, types.FunctionType):
            self._visit_function(value, depth)
        elif isinstance(value, Mapping):
            self._visit_many(value.values(), depth)
        elif isinstance(value, (list, tuple, set, frozenset)):
            self._visit_many(value, depth)

    def _visit_many(self, values: Iterable[Any], depth: int) -> None:
        """Visit each item of ``values`` one level deeper."""

        for item in values:
            self.visit(item, depth + 1)

    def _visit_layer(self, layer: Any, depth: int) -> None:
        """Visit a layer's attributes and registered sub-layers."""

        for name, attribute in vars(layer).items():
            if name in _SKIPPED_LAYER_ATTRIBUTES:
                continue
            self.visit(attribute, depth + 1)
        sub_layers = vars(layer).get("_sub_layers", {})
        if isinstance(sub_layers, Mapping):
            for sub_layer in sub_layers.values():
                self.visit(sub_layer, depth + 1)
        forward = getattr(type(layer), "forward", None)
        if isinstance(forward, types.FunctionType):
            self._visit_function(forward, depth)

    def _visit_function(self, function: Callable[..., Any], depth: int) -> None:
        """Visit a function's closure cells and default arguments."""

        code = getattr(function, "__code__", None)
        for cell in getattr(function, "__closure__", None) or ():
            try:
                self.visit(cell.cell_contents, depth + 1)
            except ValueError:  # empty cell
                continue
        self._visit_many(getattr(function, "__defaults__", None) or (), depth)
        self._visit_many((getattr(function, "__kwdefaults__", None) or {}).values(), depth)
        wrapped = getattr(function, "__wrapped__", None)
        if wrapped is not None:
            self.visit(wrapped, depth + 1)
        if code is None:
            return
        function_globals = getattr(function, "__globals__", {})
        for name in _code_names(code):
            if name in function_globals:
                self.visit(function_globals[name], depth + 1)


def _code_names(code: types.CodeType) -> set[str]:
    """Global names a code object and its nested code objects refer to."""

    names = set(code.co_names)
    for constant in code.co_consts:
        if isinstance(constant, types.CodeType):
            names |= _code_names(constant)
    return names
