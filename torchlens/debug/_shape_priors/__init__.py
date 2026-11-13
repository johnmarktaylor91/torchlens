"""Framework shape priors behind the neutral input-shape search.

The search controller (:mod:`torchlens.debug._shape_search`) owns the loop, the
probe budget, the bounds, verification with ``tl.trace`` and the
:class:`~torchlens.debug._infer_report.GuessInputShapeReport` it returns. A
prior module answers only framework questions. Torch keeps its own priors in
:mod:`torchlens.debug._infer_input_shape`; every other backend has one module
here, named in :data:`PRIOR_MODULES` and looked up by backend name.

A prior module exposes these module-level functions (framework imports happen
inside the functions, never at module import):

``FRAMEWORK: str``
    Top-level package name of the framework (``"paddle"``); a callable that
    names no framework at all goes to the one prior whose framework is
    imported (:func:`imported_priors`).
``handles(model) -> bool``
    Whether ``model`` is this framework's model object. Must not import the
    framework: check ``sys.modules`` first and return ``False`` when the
    framework was never imported.
``lazy_reason(model) -> str | None``
    Why the model cannot be probed without initializing lazy or unbuilt layers
    (``None`` when it can). The search never initializes them in the user's
    object.
``candidates(model, request) -> Sequence[ShapeCandidate]``
    First-layer shape priors in the framework's native layout, best first.
    Empty when the model has no readable first layer.
``make_input(shape, dtype, value_range, seed) -> Any``
    One native input tensor of ``shape`` and ``dtype`` filled per
    ``value_range`` (``("uniform", lo, hi)`` or ``("randint", lo, hi)``),
    deterministic for ``seed`` and without touching the global RNG beyond
    what :func:`state_scope` restores.
``forward(model, example) -> Any``
    One plain forward pass (no capture), realized where the framework is lazy.
``classify_error(exc) -> ProbeError``
    Sort a forward exception into shape, dtype, device or blocker, with the
    framework's suggested dtype and measured feature counts when the message
    names them.
``executed_constraint(trace) -> ExecutedConstraint | None``
    The first constraining op the verification trace actually executed.
``state_scope(model) -> ContextManager[None]``
    Save the model's mode flags and the framework RNG, switch to inference
    mode, and restore both on exit (also on error).

Search arguments arrive in :class:`SearchRequest`; ``input_dtype`` stays the
caller's native dtype object (or ``None``) and the prior interprets it.
"""

from __future__ import annotations

import importlib
import re
import sys
from collections.abc import Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from types import MappingProxyType, ModuleType
from typing import Any, Literal, Protocol

ValueRange = tuple[str, float, float]
"""Synthetic value recipe: ``("uniform", lo, hi)`` or ``("randint", lo, hi)``."""

ProbeErrorKind = Literal["shape", "dtype", "device", "blocker"]

#: Backend name to prior module (relative to this package). Torch is absent on
#: purpose: its priors stay in ``_infer_input_shape``.
PRIOR_MODULES: MappingProxyType[str, str] = MappingProxyType(
    {
        "paddle": "paddle",
        "tinygrad": "tinygrad",
        "tf": "keras",
        "mlx": "mlx",
        "jax": "jax",
    }
)


@dataclass(frozen=True)
class SearchRequest:
    """Caller arguments the search passes to a prior, already validated."""

    batch_size: int
    input_dtype: Any
    channels: int | None
    spatial_rank: int | Literal["auto"]
    seq_len: int | None
    min_size: int
    max_size: int
    preferred_sizes: tuple[int, ...]


@dataclass(frozen=True)
class ShapeCandidate:
    """One first-layer prior: a shape to probe and what pins it.

    Attributes
    ----------
    kind:
        ``"linear"``, ``"conv"``, ``"embedding"``, ``"sequence"`` or
        ``"unknown"``; the search only side-searches ``"conv"``.
    shape:
        Shape to probe first, batch dimension included.
    dtype:
        Native dtype object of the framework.
    value_range:
        Synthetic value recipe.
    layout:
        Native layout string, e.g. ``"NC"``, ``"NCHW"``, ``"NHWC"``,
        ``"NCL"``, ``"NLC"``, ``"NCDHW"``, ``"NDHWC"`` or ``"NL"`` (tokens).
    spatial_axes:
        Axes the search may resize (square, equal sides) when the first probe
        fails with a shape error.
    flexible_dims:
        Axes the prior believes tolerate other sizes (batch excluded); the
        search keeps only the ones a grown probe confirms.
    determined:
        ``True`` when a weight shape pins every non-flexible axis, so a
        working probe is the unique shape up to the flexible axes. ``False``
        when the prior guessed an axis (rank, spatial side, sequence length)
        and the working shape is one candidate among others.
    constraining_module:
        Module address that pinned the prior, when known.
    constraining_op:
        Layer or op type that pinned the prior, when known.
    strategy:
        Strategy name reported when this candidate wins.
    min_side:
        Smallest spatial side the side search may try.
    """

    kind: str
    shape: tuple[int, ...]
    dtype: Any
    value_range: ValueRange
    layout: str
    spatial_axes: tuple[int, ...] = ()
    flexible_dims: tuple[int, ...] = ()
    determined: bool = False
    constraining_module: str | None = None
    constraining_op: str | None = None
    strategy: str = "layer_prior"
    min_side: int = 1


@dataclass(frozen=True)
class ProbeError:
    """A classified forward-probe exception."""

    kind: ProbeErrorKind
    message: str
    suggested_dtype: Any = None
    got_features: int | None = None
    target_features: int | None = None


@dataclass(frozen=True)
class ExecutedConstraint:
    """The first constraining op a verification trace executed."""

    kind: str
    module: str | None
    op: str | None
    source_line: str | None
    input_shape: tuple[int, ...] | None


class ShapePriors(Protocol):
    """Typing view of a prior module (modules satisfy protocols in mypy)."""

    def handles(self, model: Any) -> bool:
        """Whether ``model`` belongs to this framework (no framework import)."""

    def lazy_reason(self, model: Any) -> str | None:
        """Why probing would initialize lazy layers, or ``None``."""

    def candidates(self, model: Any, request: SearchRequest) -> Sequence[ShapeCandidate]:
        """First-layer priors in native layout, best first."""

    def make_input(
        self, shape: tuple[int, ...], dtype: Any, value_range: ValueRange, seed: int
    ) -> Any:
        """One deterministic native input tensor."""

    def forward(self, model: Any, example: Any) -> Any:
        """One plain realized forward pass."""

    def classify_error(self, exc: Exception) -> ProbeError:
        """Sort a forward exception."""

    def executed_constraint(self, trace: Any) -> ExecutedConstraint | None:
        """The first constraining op the trace executed."""

    def state_scope(self, model: Any) -> AbstractContextManager[None]:
        """Save and restore model mode flags and framework RNG."""


def resolve_priors(model: Any) -> tuple[str, ShapePriors] | None:
    """Find the prior module whose ``handles`` accepts ``model``.

    A prior module that is not present yet (a sibling lane has not landed it)
    is skipped; any other import error propagates.

    Parameters
    ----------
    model:
        The caller's model object.

    Returns
    -------
    tuple[str, ShapePriors] | None
        ``(backend name, prior module)``, or ``None`` when no prior handles it.
    """

    for backend_name, module_name in PRIOR_MODULES.items():
        qualified = f"{__name__}.{module_name}"
        try:
            module: ModuleType = importlib.import_module(qualified)
        except ModuleNotFoundError as exc:
            if exc.name == qualified:
                continue
            raise
        priors: ShapePriors = module
        if priors.handles(model):
            return backend_name, priors
    return None


def imported_priors() -> list[tuple[str, ShapePriors]]:
    """Prior modules whose framework (``FRAMEWORK``) is already imported."""

    found: list[tuple[str, ShapePriors]] = []
    for backend_name, module_name in PRIOR_MODULES.items():
        qualified = f"{__name__}.{module_name}"
        try:
            module = importlib.import_module(qualified)
        except ModuleNotFoundError as exc:
            if exc.name == qualified:
                continue
            raise
        framework = getattr(module, "FRAMEWORK", None)
        if isinstance(framework, str) and framework in sys.modules:
            found.append((backend_name, module))
    return found


#: Most objects :func:`walk_references` visits.
MAX_WALK_NODES = 512
#: Deepest attribute nesting :func:`walk_references` follows.
MAX_WALK_DEPTH = 4


def walk_references(root: Any) -> list[tuple[str, Any]]:
    """Objects reachable from ``root`` in definition order, bounded and cycle-safe.

    Follows instance ``__dict__`` entries, list, tuple and dict items, function
    closure cells and the globals a function names, bound-method ``__self__``
    and ``functools.partial`` targets. Modules, classes and tensors (anything
    with ``shape`` and ``dtype``) are yielded but not entered. Used by priors for frameworks whose models are plain callables.

    Parameters
    ----------
    root:
        The caller's model object or function.

    Returns
    -------
    list[tuple[str, Any]]
        ``(path, object)`` pairs, ``root`` first as ``""``.
    """

    seen: set[int] = set()
    found: list[tuple[str, Any]] = []
    stack: list[tuple[str, Any, int]] = [("", root, 0)]
    while stack and len(found) < MAX_WALK_NODES:
        path, obj, depth = stack.pop(0)
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        found.append((path, obj))
        if depth < MAX_WALK_DEPTH and not isinstance(obj, (ModuleType, type, str, bytes)):
            stack.extend((_join(path, name), child, depth + 1) for name, child in _children(obj))
    return found


def _join(path: str, name: str) -> str:
    """Join a walk path and a child name."""

    return f"{path}.{name}" if path else name


def _children(obj: Any) -> list[tuple[str, Any]]:
    """Direct references of one object for :func:`walk_references`."""

    if hasattr(obj, "shape") and hasattr(obj, "dtype"):
        return []  # A tensor is a leaf: its internals are framework state.
    if isinstance(obj, (list, tuple)):
        return [(str(index), item) for index, item in enumerate(obj)]
    if isinstance(obj, dict):
        return [(str(key), value) for key, value in obj.items() if isinstance(key, str)]
    if getattr(obj, "__code__", None) is not None:
        return _function_items(obj)
    for attribute in ("__self__", "func"):
        target = getattr(obj, attribute, None)
        if target is not None and callable(target) and not isinstance(obj, ModuleType):
            return [(attribute, target), *_instance_items(obj)]
    return _instance_items(obj)


def _function_items(function: Any) -> list[tuple[str, Any]]:
    """A function's non-empty closure cells and the module globals its code names."""

    code = function.__code__
    cells = zip(code.co_freevars, getattr(function, "__closure__", None) or (), strict=False)
    closure = [(name, _cell_value(cell)) for name, cell in cells]
    globals_ = getattr(function, "__globals__", {})
    named = [(name, globals_[name]) for name in code.co_names if name in globals_]
    return [(name, value) for name, value in closure + named if value is not None]


def _instance_items(obj: Any) -> list[tuple[str, Any]]:
    """An object's ``__dict__`` items, plus its ``__call__`` when user-defined."""

    try:
        items = list(vars(obj).items())
    except TypeError:
        return []
    call = type(obj).__dict__.get("__call__")
    if call is not None and getattr(call, "__code__", None) is not None:
        items.append(("__call__", call))
    return items


def _cell_value(cell: Any) -> Any:
    """A closure cell's contents, or ``None`` when empty."""

    try:
        return cell.cell_contents
    except ValueError:
        return None


# ---------------------------------------------------------------- shared prior helpers
#: Matrix-multiply mismatch texts: ``(AxB and CxD)`` and tinygrad's ``((B) != (C))``.
_MATMUL_MISMATCH = (
    re.compile(r"\((\d+)x(\d+) and (\d+)x(\d+)\)"),
    re.compile(r"\(\((\d+)\) != \((\d+)\)\)"),
    re.compile(r"K_lhs:(\d+) != K_rhs:(\d+)"),
    re.compile(r"width_:(\d+) != dim_b\.height_:(\d+)"),
    re.compile(r"cannot dot \((?:\d+, )*(\d+),?\) and \((\d+)(?:, \d+)*,?\)"),
)


def integer_range(high: int) -> ValueRange:
    """Integer value recipe ``[0, high)`` (token ids below a vocabulary size)."""

    return ("randint", 0.0, float(max(1, high)))


def parse_feature_mismatch(message: str) -> tuple[int | None, int | None]:
    """``(got, target)`` feature counts from a matrix-multiply mismatch, else ``None``s."""

    four = _MATMUL_MISMATCH[0].search(message)
    if four is not None:
        return int(four.group(2)), int(four.group(3))
    for pattern in _MATMUL_MISMATCH[1:]:
        two = pattern.search(message)
        if two is not None:
            return int(two.group(1)), int(two.group(2))
    return None, None


def framework_of(obj: Any) -> str | None:
    """Top-level package that defines ``obj`` (a module, a class or an instance)."""

    if isinstance(obj, ModuleType):
        name = obj.__name__
    elif isinstance(obj, type):
        name = obj.__module__
    else:
        name = type(obj).__module__
    return name.split(".", 1)[0] if isinstance(name, str) else None


def generic_ladder(request: SearchRequest, dtype: Any) -> list[ShapeCandidate]:
    """Shapes for a callable no weight constrains; a win here is never determined."""

    batch = request.batch_size
    shapes = ((batch, 8), (8,), (batch, 2, 4), (batch, 4), (batch, 16), (batch, 3, 16, 16))
    layouts = {1: "N", 2: "NC", 3: "NLC", 4: "NCHW"}
    return [
        ShapeCandidate(
            "unknown",
            shape,
            dtype,
            ("uniform", 0.0, 1.0),
            layouts[len(shape)],
            strategy="generic_probe",
        )
        for shape in shapes
    ]


def closure_matrix_candidates(
    tensors: Sequence[tuple[str, tuple[int, ...]]], request: SearchRequest, dtype: Any
) -> list[ShapeCandidate]:
    """``(batch, rows)`` candidates from rank-2 tensors a function closes over."""

    seen: set[int] = set()
    found: list[ShapeCandidate] = []
    for path, shape in tensors:
        if len(shape) != 2 or shape[0] in seen:
            continue
        seen.add(shape[0])
        found.append(
            ShapeCandidate(
                "matmul",
                (request.batch_size, shape[0]),
                dtype,
                ("uniform", 0.0, 1.0),
                "NC",
                constraining_module=path or None,
                constraining_op="closure weight",
                strategy="closure_weight_prior",
            )
        )
    return found


def constraint_from_trace(
    trace: Any, kinds: Sequence[tuple[str, str]]
) -> ExecutedConstraint | None:
    """The first op whose ``func_name`` contains a needle of ``kinds`` (``(needle, kind)``).

    The input shape is the op's first parent's shape when the trace exposes it.
    """

    for op in getattr(trace, "ops", ()) or ():
        func_name = str(getattr(op, "func_name", "") or "").lower()
        kind = next((kind for needle, kind in kinds if needle in func_name), None)
        if kind is None:
            continue
        return ExecutedConstraint(
            kind=kind,
            module=_op_text(op, "module_address", "module"),
            op=_op_text(op, "layer_label", "label"),
            source_line=_op_text(op, "source_line"),
            input_shape=_first_parent_shape(trace, op),
        )
    return None


def _op_text(op: Any, *names: str) -> str | None:
    """The first non-empty string attribute of ``op`` among ``names``."""

    for name in names:
        value = getattr(op, name, None)
        if value:
            return str(getattr(value, "address", value))
    return None


def _first_parent_shape(trace: Any, op: Any) -> tuple[int, ...] | None:
    """The shape of ``op``'s first parent, when the trace can resolve it."""

    parents = list(getattr(op, "parents", ()) or ())
    if not parents:
        return None
    try:
        parent = trace[parents[0]] if isinstance(parents[0], str) else parents[0]
        shape = getattr(parent, "shape", None) or getattr(parent, "tensor_shape", None)
        return tuple(int(dim) for dim in shape) if shape is not None else None
    except (KeyError, TypeError, ValueError, IndexError):
        return None


__all__ = [
    "PRIOR_MODULES",
    "ExecutedConstraint",
    "ProbeError",
    "ProbeErrorKind",
    "SearchRequest",
    "ShapeCandidate",
    "ShapePriors",
    "ValueRange",
    "closure_matrix_candidates",
    "constraint_from_trace",
    "framework_of",
    "generic_ladder",
    "imported_priors",
    "integer_range",
    "parse_feature_mismatch",
    "resolve_priors",
    "walk_references",
]
