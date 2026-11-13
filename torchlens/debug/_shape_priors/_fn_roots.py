"""Shared helpers for function roots and array-framework priors.

A plain Python function carries no framework tag, so the Keras, MLX and JAX
priors attribute one by the names its code resolves to (module objects and
arrays in its globals and closure). A function that names no framework is
claimed by none of them, and the search reports it as unknown rather than
guessing a framework.

Function roots have no first layer. Their candidates are bounded shape
hypotheses ordered by the array dimensions the function closes over (or the
parameters a ``functools.partial`` binds), each checked by the framework's
cheapest probe. A surviving hypothesis is a working candidate, never a
determined shape: :attr:`ShapeCandidate.determined` stays ``False``.
"""

from __future__ import annotations

import functools
import types
from collections.abc import Callable, Iterator, Sequence
from typing import Any

from .._infer_op_facts import _shape_op_module, _shape_op_source_line, _shape_tuple
from . import ExecutedConstraint, SearchRequest, ShapeCandidate, ValueRange

#: Hypotheses probed per function root (abstract or lazy probes, outside the
#: search's own forward budget).
MAX_HYPOTHESES = 32
#: Working hypotheses returned per function root.
MAX_FUNCTION_CANDIDATES = 4
#: Generic feature widths tried after the dimensions the function closes over.
_GENERIC_WIDTHS: tuple[int, ...] = (4, 8, 16, 3, 32, 64, 2, 1, 128, 256, 512, 768)
#: Image channel counts tried for channel-last rank-4 hypotheses.
_IMAGE_CHANNELS: tuple[int, ...] = (1, 3)
_UNIFORM: ValueRange = ("uniform", 0.0, 1.0)

#: Lower-cased op-name fragments that mark a constraining op, by kind.
_CONSTRAINT_FRAGMENTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("conv", ("conv",)),
    ("linear", ("dot_general", "matmul", "addmm", "linear", "dense", "batchmatmul")),
    ("embedding", ("embedding", "gather", "take")),
)


def _code_names(code: types.CodeType) -> Iterator[str]:
    """Every global and free name a code object (or a nested one) reads."""

    yield from code.co_names
    yield from code.co_freevars
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            yield from _code_names(const)


def _closure_values(fn: types.FunctionType) -> dict[str, Any]:
    """The function's free variables that hold a value."""

    values: dict[str, Any] = {}
    for name, cell in zip(fn.__code__.co_freevars, fn.__closure__ or (), strict=False):
        try:
            values[name] = cell.cell_contents
        except ValueError:  # an empty cell: the variable is not bound yet
            continue
    return values


def referenced_values(fn: Any) -> list[Any]:
    """Objects a function root reads: globals, closure cells and bound arguments.

    Parameters
    ----------
    fn:
        A plain function or a ``functools.partial`` of one.

    Returns
    -------
    list[Any]
        Resolved objects in a stable order; empty for other callables.
    """

    if isinstance(fn, functools.partial):
        bound = [*fn.args, *fn.keywords.values()]
        return bound + referenced_values(fn.func)
    if not isinstance(fn, types.FunctionType):
        return []
    closure = _closure_values(fn)
    found: list[Any] = []
    for name in dict.fromkeys(_code_names(fn.__code__)):
        if name in closure:
            found.append(closure[name])
        elif name in fn.__globals__:
            found.append(fn.__globals__[name])
    return found


def _owner_module(value: Any) -> str:
    """Dotted module that owns ``value`` (a module's own name for modules)."""

    if isinstance(value, types.ModuleType):
        return value.__name__
    return str(getattr(type(value), "__module__", "") or "")


def names_framework(fn: Any, prefixes: Sequence[str]) -> bool:
    """Whether a function root reads a module or object of the framework.

    Parameters
    ----------
    fn:
        A plain function or a ``functools.partial`` of one.
    prefixes:
        Top-level package names of the framework, e.g. ``("jax", "flax")``.

    Returns
    -------
    bool
        ``True`` when any resolved name belongs to one of the packages.
    """

    for value in _leaves(referenced_values(fn)):
        owner = _owner_module(value)
        if any(owner == prefix or owner.startswith(prefix + ".") for prefix in prefixes):
            return True
    return False


def _leaves(values: Sequence[Any], depth: int = 0) -> Iterator[Any]:
    """Values plus the members of nested dicts, lists and tuples (bounded depth)."""

    for value in values:
        yield value
        if depth >= 3:
            continue
        if isinstance(value, dict):
            yield from _leaves(list(value.values()), depth + 1)
        elif isinstance(value, (list, tuple)):
            yield from _leaves(list(value), depth + 1)


def array_dims(fn: Any) -> tuple[int, ...]:
    """Distinct dimensions of the arrays a function root reads, in reading order.

    For each array of rank two or more the leading dimension comes first (the
    contraction side of ``x @ w``), then the rest.

    Parameters
    ----------
    fn:
        A plain function or a ``functools.partial`` of one.

    Returns
    -------
    tuple[int, ...]
        Positive dimensions, duplicates removed.
    """

    dims: list[int] = []
    for value in _leaves(referenced_values(fn)):
        if isinstance(value, types.ModuleType):
            continue
        shape = _shape_tuple(getattr(value, "shape", None))
        if shape:
            dims.extend(dim for dim in shape if dim > 0)
    return tuple(dict.fromkeys(dims))


def hypotheses(dims: Sequence[int], request: SearchRequest) -> list[tuple[int, ...]]:
    """Bounded input-shape hypotheses for a function root, best first.

    Parameters
    ----------
    dims:
        Dimensions read from the function's arrays (tried first as widths).
    request:
        The validated search arguments (batch size, sizes, overrides).

    Returns
    -------
    list[tuple[int, ...]]
        At most :data:`MAX_HYPOTHESES` shapes: ``(batch, width)`` rows, then
        channel-last square images, then unbatched vectors.
    """

    batch = request.batch_size
    widths = [w for w in (*dims, *_GENERIC_WIDTHS) if request.min_size <= w <= request.max_size]
    if request.seq_len is not None:
        widths.insert(0, request.seq_len)
    channels = (request.channels,) if request.channels else _IMAGE_CHANNELS
    sides = [s for s in request.preferred_sizes if request.min_size <= s <= request.max_size]
    shapes: list[tuple[int, ...]] = [(batch, w) for w in widths]
    shapes += [(batch, s, s, c) for c in channels for s in sides]
    shapes += [(w,) for w in widths]
    return list(dict.fromkeys(shapes))[:MAX_HYPOTHESES]


def function_candidates(
    dims: Sequence[int],
    request: SearchRequest,
    dtype: Any,
    works: Callable[[tuple[int, ...]], bool],
) -> list[ShapeCandidate]:
    """Working, undetermined candidates for a function root.

    Parameters
    ----------
    dims:
        Dimensions read from the function's arrays.
    request:
        The validated search arguments.
    dtype:
        Native float dtype for the probe.
    works:
        The framework's cheap probe: ``True`` when a hypothesis runs.

    Returns
    -------
    list[ShapeCandidate]
        At most :data:`MAX_FUNCTION_CANDIDATES` candidates, all with
        ``determined=False``.
    """

    found: list[ShapeCandidate] = []
    for shape in hypotheses(dims, request):
        if len(found) >= MAX_FUNCTION_CANDIDATES:
            break
        if works(shape):
            found.append(_function_candidate(shape, dtype))
    return found


def _function_candidate(shape: tuple[int, ...], dtype: Any) -> ShapeCandidate:
    """One function-root candidate; every non-batch axis may be flexible."""

    image = len(shape) == 4
    return ShapeCandidate(
        kind="unknown",
        shape=shape,
        dtype=dtype,
        value_range=_UNIFORM,
        layout="NHWC" if image else ("NC" if len(shape) == 2 else "C"),
        spatial_axes=(1, 2) if image else (),
        flexible_dims=tuple(range(1, len(shape))) if len(shape) > 1 else (0,),
        determined=False,
        strategy="function_probe",
    )


def fill_array(shape: tuple[int, ...], value_range: ValueRange, seed: int, np_dtype: str) -> Any:
    """A deterministic NumPy array from a private generator (global RNG untouched).

    Parameters
    ----------
    shape:
        Array shape.
    value_range:
        ``("uniform", lo, hi)`` or ``("randint", lo, hi)``.
    seed:
        Generator seed.
    np_dtype:
        NumPy dtype name for the result.

    Returns
    -------
    Any
        A ``numpy.ndarray``.
    """

    import numpy as np

    generator = np.random.default_rng(seed)
    kind, low, high = value_range
    if kind == "randint":
        return generator.integers(int(low), max(int(high), int(low) + 1), size=shape).astype(
            np_dtype
        )
    return generator.uniform(float(low), float(high), size=shape).astype(np_dtype)


def _constraint_kind(func_name: str) -> str | None:
    """Kind of a constraining op name, or ``None``."""

    for kind, fragments in _CONSTRAINT_FRAGMENTS:
        if any(fragment in func_name for fragment in fragments):
            return kind
    return None


def executed_constraint(trace: Any) -> ExecutedConstraint | None:
    """The first linear, convolution or embedding op a trace executed.

    Parameters
    ----------
    trace:
        A verification trace from any array backend.

    Returns
    -------
    ExecutedConstraint | None
        The first constraining op, or ``None`` when none ran.
    """

    for op in getattr(trace, "ops", None) or ():
        func_name = str(getattr(op, "func_name", "") or "").lower()
        kind = _constraint_kind(func_name)
        if kind is None:
            continue
        shapes = (_shape_tuple(shape) for shape in getattr(op, "input_shapes", ()) or ())
        return ExecutedConstraint(
            kind=kind,
            module=_shape_op_module(op),
            op=func_name,
            source_line=_shape_op_source_line(op),
            input_shape=next((shape for shape in shapes if shape is not None), None),
        )
    return None


def parse_ints(text: str) -> tuple[int, ...]:
    """Integers in a comma-separated shape string such as ``"2, 5"``."""

    return tuple(
        int(part) for part in text.replace(" ", "").split(",") if part.lstrip("-").isdigit()
    )
