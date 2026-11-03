"""Per-unit index-set mapping for causal attention windows.

The set walks carry a box of indices per tensor, so unlike the axis-level
descriptor they keep the query and key coordinates of an attention tensor
together. That is what lets them narrow a causal hop exactly where the
descriptor (one output axis per input axis) can only report a whole-extent
upper bound. Both directions of :mod:`._engine_causal`'s relation live here.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from ._engine_causal import CausalSpec, causal_spec, resolve_per_parent
from ._index_set import _AxisSets, _IndexSet
from ._rules import _RuleResult

if TYPE_CHECKING:
    from .._data_classes.op import Op

EXTENDED_KINDS = frozenset({"per_parent", "causal_window"})
"""Rule kinds the set walks resolve here before their standard dispatch."""


def _whole_envelope(op: Op, sets: _AxisSets) -> _AxisSets:
    """Return the sound inexact whole-extent fallback for a constrained box."""

    if not any(item is not None for item in sets):
        return (None,) * len(op.shape)
    return tuple(_IndexSet.interval(0, int(extent) - 1, exact=False) for extent in op.shape)


def map_extended_to_parent(
    map_to_parent: Callable[[Op, Op, _AxisSets, _RuleResult], tuple[_AxisSets, bool]],
    op: Op,
    parent: Op,
    output_sets: _AxisSets,
    result: _RuleResult,
) -> tuple[_AxisSets, bool]:
    """Resolve a per-parent or causal result on the receptive set walk."""

    if result.kind == "per_parent":
        return map_to_parent(op, parent, output_sets, resolve_per_parent(result, parent))
    causal = map_causal_to_parent(op, parent, output_sets, result)
    return causal if causal is not None else (_whole_envelope(parent, output_sets), False)


def map_extended_to_child(
    map_to_child: Callable[[Op, Op, _AxisSets, _RuleResult, str], tuple[_AxisSets, bool]],
    parent: Op,
    child: Op,
    parent_sets: _AxisSets,
    result: _RuleResult,
    rule_name: str,
) -> tuple[_AxisSets, bool]:
    """Resolve a per-parent or causal result on the projective set walk."""

    if result.kind == "per_parent":
        return map_to_child(
            parent, child, parent_sets, resolve_per_parent(result, parent), rule_name
        )
    causal = map_causal_to_child(parent, child, parent_sets, result)
    return causal if causal is not None else (_whole_envelope(child, parent_sets), False)


def _bounds(index_set: _IndexSet | None, extent: int) -> tuple[int, int] | None:
    """Return ``(minimum, maximum)`` of a set, the whole extent for ``None``, or ``None`` if empty."""

    if index_set is None:
        return 0, extent - 1
    if index_set.is_empty:
        return None
    minimum, maximum = index_set.minimum, index_set.maximum
    assert minimum is not None and maximum is not None
    return minimum, maximum


def _uniform_windows(spec: CausalSpec, query_min: int, query_max: int) -> bool:
    """Whether every query in the range sees the same clipped key prefix."""

    return query_min == query_max or query_min + spec.offset >= spec.key_extent - 1


def _full_axes(spec: CausalSpec, shape: tuple[int, ...], sets: list[_IndexSet | None]) -> bool:
    """Fill the whole-extent axes of a mapped frame; return their joint exactness."""

    exact = True
    for axis, flag in spec.full_axes.items():
        sets[axis] = _IndexSet.interval(0, int(shape[axis]) - 1, exact=flag)
        exact = exact and flag
    return exact


def map_causal_to_parent(
    op: Op, parent: Op, output_sets: _AxisSets, result: _RuleResult
) -> tuple[_AxisSets, bool] | None:
    """Map a child box onto one causal parent (receptive direction)."""

    spec = causal_spec(result, op, parent)
    if spec is None:
        return None
    rank = len(parent.shape)
    mapped: list[_IndexSet | None] = [
        None if axis in spec.full_axes or axis == spec.key_axis else output_sets[axis]
        for axis in range(rank)
    ]
    exact = _full_axes(spec, tuple(int(extent) for extent in parent.shape), mapped) and spec.exact
    query_bounds = _bounds(output_sets[spec.query_axis], spec.query_extent)
    if query_bounds is None:
        return tuple(_IndexSet.empty() for _ in range(rank)), True
    query_min, query_max = query_bounds
    if spec.mode == "self":
        key_bounds = _bounds(output_sets[spec.key_axis], spec.key_extent)
        if key_bounds is None or key_bounds[0] > query_max + spec.offset:
            return tuple(_IndexSet.empty() for _ in range(rank)), True
        exact = exact and key_bounds[1] <= query_min + spec.offset
    upper = min(query_max + spec.offset, spec.key_extent - 1)
    mapped[spec.key_axis] = _IndexSet.empty() if upper < 0 else _IndexSet.interval(0, upper)
    if spec.mode == "self":
        mapped[spec.query_axis] = output_sets[spec.query_axis]
    exact = exact and _uniform_windows(spec, query_min, query_max)
    return tuple(mapped), exact


def map_causal_to_child(
    parent: Op, child: Op, parent_sets: _AxisSets, result: _RuleResult
) -> tuple[_AxisSets, bool] | None:
    """Transpose a parent box onto one causal child (projective direction)."""

    spec = causal_spec(result, child, parent)
    if spec is None:
        return None
    rank = len(child.shape)
    child_shape = tuple(int(extent) for extent in child.shape)
    mapped: list[_IndexSet | None] = [
        None if axis in spec.full_axes else parent_sets[axis] for axis in range(rank)
    ]
    exact = spec.exact
    for axis, flag in spec.full_axes.items():
        if parent_sets[axis] is not None:
            mapped[axis] = _IndexSet.interval(0, child_shape[axis] - 1, exact=flag)
            exact = exact and flag
    key_bounds = _bounds(parent_sets[spec.key_axis], spec.key_extent)
    if key_bounds is None:
        return tuple(_IndexSet.empty() for _ in range(rank)), True
    key_min, key_max = key_bounds
    if spec.mode == "cross":
        lower = max(key_min - spec.offset, 0)
        if lower > spec.query_extent - 1:
            return tuple(_IndexSet.empty() for _ in range(rank)), True
        mapped[spec.query_axis] = _IndexSet.interval(lower, spec.query_extent - 1)
        return tuple(mapped), exact and (key_min == key_max or key_max <= spec.offset)
    query_bounds = _bounds(parent_sets[spec.query_axis], spec.query_extent)
    if query_bounds is None or key_min > query_bounds[1] + spec.offset:
        return tuple(_IndexSet.empty() for _ in range(rank)), True
    query_min, query_max = query_bounds
    exact = exact and key_max <= query_min + spec.offset
    upper = min(query_max + spec.offset, child_shape[spec.key_axis] - 1)
    mapped[spec.query_axis] = parent_sets[spec.query_axis]
    mapped[spec.key_axis] = _IndexSet.interval(0, upper)
    return tuple(mapped), exact and _uniform_windows(spec, query_min, query_max)


__all__ = [
    "EXTENDED_KINDS",
    "map_causal_to_child",
    "map_causal_to_parent",
    "map_extended_to_child",
    "map_extended_to_parent",
]
