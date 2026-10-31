"""Exact per-unit projective queries via membership transposition."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from fractions import Fraction
from math import ceil, floor
from typing import TYPE_CHECKING, Any, cast

from . import _engine
from ._engine_forward import _ProjectiveFieldSolution
from ._engine_geometry import _Affine, _as_tuple, _Mapped, _select_full_axes, _transpose_mapped
from ._errors import AmbiguousTargetError, ReceptiveFieldError
from ._path import forward_index_image
from ._query import (
    _AxisSets,
    _build_box,
    _call_index_callback,
    _call_interval_callback,
    _distinct_terminals,
    _IndexSet,
    _initial_axis_sets,
    _normalize_unit,
    _query_graph,
    _query_memo_scope,
    _run_memoized_walk,
    _shared_reach,
    _TerminalState,
    _validate_descriptor_for_query,
    _WalkKey,
    _WalkStep,
    map_transposed_convolution_index_set,
)
from ._rules import _RuleResult
from ._types import ReceptiveField, ReceptiveFieldBox, ReceptiveFieldDirection

if TYPE_CHECKING:
    from .._data_classes.op import Op


_TRANSPOSE_CANDIDATE_BUDGET = 4096


def box_for_source_unit(
    solution: _ProjectiveFieldSolution,
    op: Op,
    unit: Sequence[int],
    *,
    target: Op | str | None = None,
    clip: bool = True,
    complete_unit: Sequence[int] | None = None,
) -> ReceptiveFieldBox:
    """Compute one source unit's projective support by forward index-set propagation.

    Parameters
    ----------
    solution:
        Target-anchored projective descriptor solution.
    op:
        Source operation whose unit is selected.
    unit:
        Coordinates over the source's derived windowed axes.
    target:
        Optional target operation, exact target role, or operation label.
    clip:
        Whether returned bounds are intersected with captured target extents.
    complete_unit:
        Optional complete source index used internally to route structural axes.

    Returns
    -------
    ReceptiveFieldBox
        Target-space support hull. Candidate-budget collapse is reported as ``UPPER_BOUND``.
    """

    descriptors = solution.per_op.get(op.label)
    if not descriptors:
        raise ReceptiveFieldError(
            f"No projective-field solution is available from source {op.label!r}."
        )
    descriptor = _select_target_descriptor(descriptors, target)
    _validate_descriptor_for_query(descriptor)
    coordinates = _normalize_unit(op, descriptor, unit)
    initial = _initial_axis_sets(op, descriptor, coordinates, complete_unit=complete_unit)
    operations, by_reference = _query_graph(op.source_trace, solution)
    target_label = descriptor.input_op_label
    reaching = _shared_reach(
        "projective",
        solution,
        target_label,
        lambda: _labels_reaching(target_label, operations, by_reference),
    )
    active = reaching if op.label in reaching else frozenset()
    # The walk enumerates paths, so it revisits each operation once per path
    # through it, re-deriving per-operation facts that are linear in that
    # operation's parents. Memoize them for this one query (or for every query
    # of an enclosing shared scope).
    with _query_memo_scope():
        terminals = _walk_to_target(op, initial, target_label, active, by_reference, True, {})
    if not terminals:
        raise ReceptiveFieldError(
            f"Source {op.label!r} has no live path to target {target_label!r}."
        )
    box = _build_box(op, descriptor, coordinates, terminals, clip=clip)
    return replace(
        box,
        direction=ReceptiveFieldDirection.PROJECTIVE,
        unit_shape=tuple(op.shape),
    )


def _select_target_descriptor(
    descriptors: Mapping[str, ReceptiveField], target: Op | str | None
) -> ReceptiveField:
    """Resolve one projective result descriptor without choosing silently."""

    if target is None:
        if len(descriptors) != 1:
            roles = ", ".join(descriptors)
            raise AmbiguousTargetError(
                f"Select one reachable target; available target roles: {roles}."
            )
        return next(iter(descriptors.values()))
    identity = target if isinstance(target, str) else target.label
    for role, descriptor in descriptors.items():
        if identity in {role, descriptor.input_op_label}:
            return descriptor
    raise ReceptiveFieldError(f"Target {identity!r} is not reachable from this source.")


def _labels_reaching(
    target_label: str,
    operations: tuple[Op, ...],
    by_reference: Mapping[str, Op],
) -> frozenset[str]:
    """Return the labels of the target and every operation with a path to it."""

    parents: dict[str, set[str]] = {item.label: set() for item in operations}
    for item in operations:
        for parent in item.parents:
            if parent in by_reference:
                parents[item.label].add(by_reference[parent].label)
    reachable = {target_label}
    stack = [target_label]
    while stack:
        label = stack.pop()
        unseen = parents.get(label, set()) - reachable
        reachable.update(unseen)
        stack.extend(unseen)
    return frozenset(reachable)


def _walk_to_target(
    op: Op,
    input_sets: _AxisSets,
    target_label: str,
    active: frozenset[str],
    by_reference: Mapping[str, Op],
    exact: bool,
    memo: dict[_WalkKey, tuple[_TerminalState, ...]],
) -> tuple[_TerminalState, ...]:
    """Transpose a joint axis-set state toward one target.

    ``memo`` collapses the path enumeration to distinct
    ``(operation, axis-set state)`` states for the duration of one query. It is
    plain memoization of a pure function, so the cached tuple is exactly what
    a fresh walk would rebuild; see ``_query._walk_to_input``. The hops run on
    ``_query._run_memoized_walk``'s explicit stack, so depth is not bounded by
    the interpreter's recursion limit.
    """

    def body(item: Op, sets: _AxisSets, item_exact: bool) -> _WalkStep:
        """Start one projective hop from ``item`` toward this query's target."""
        return _walk_target_body(item, sets, target_label, active, by_reference, item_exact)

    return _run_memoized_walk(body, op, input_sets, exact, memo)


def _walk_target_body(
    op: Op,
    input_sets: _AxisSets,
    target_label: str,
    active: frozenset[str],
    by_reference: Mapping[str, Op],
    exact: bool,
) -> _WalkStep:
    """Transpose one joint axis-set state across a single forward hop.

    Yields one request per live child branch and receives that branch's
    terminals; see ``_query._run_memoized_walk``.
    """

    if op.label == target_label:
        clipped = tuple(
            None if item is None else item.clipped(int(op.shape[axis]))
            for axis, item in enumerate(input_sets)
        )
        sets_exact = all(item is None or item.exact for item in input_sets)
        return (_TerminalState(input_sets, clipped, exact and sets_exact),)
    terminals: list[_TerminalState] = []
    children = tuple(
        by_reference[label]
        for label in op.children
        if label in by_reference and by_reference[label].label in active
    )
    for child in children:
        result, rule_name = _engine._rule_result(child)
        child_sets, hop_exact = _map_to_child(op, child, input_sets, result, rule_name)
        if isinstance(result.values.get("concatenate_axis"), int) and any(
            item is not None and item.is_empty for item in child_sets
        ):
            continue
        bounded = (
            child_sets
            if child.label == target_label
            else tuple(
                None if item is None else item.clipped(int(child.shape[axis]))
                for axis, item in enumerate(child_sets)
            )
        )
        terminals.extend(
            (
                yield (
                    child,
                    bounded,
                    exact and hop_exact and all(item is None or item.exact for item in child_sets),
                )
            )
        )
    return _distinct_terminals(terminals)


def _map_to_child(
    parent: Op,
    child: Op,
    parent_sets: _AxisSets,
    result: _RuleResult,
    rule_name: str,
) -> tuple[_AxisSets, bool]:
    """Transpose one sealed local backward rule onto a child branch."""

    if result.kind in {"data_dependent", "unknown", "unsupported", "dissolve", "piecewise"}:
        return _whole_child_envelope(child, parent_sets), False
    if result.unit_box is not None and not (
        result.map_index_set is not None or result.map_index_set_forward is not None
    ):
        return _rule_envelope(parent, child, parent_sets, result), False
    if result.kind == "window":
        return _map_window_forward(parent, child, parent_sets, result)
    if result.kind == "window_edges":
        return _map_window_edges_forward(parent, child, parent_sets, result)
    if result.kind == "full":
        return _map_full_forward(parent, child, parent_sets, result)
    if result.kind == "axis_map":
        return _map_axis_forward(parent, child, parent_sets, result)
    if result.kind == "passthrough":
        return _map_passthrough_forward(parent, child, parent_sets, result), True
    _ = rule_name
    return _whole_child_envelope(child, parent_sets), False


def _map_window_forward(
    parent: Op, child: Op, parent_sets: _AxisSets, result: _RuleResult
) -> tuple[_AxisSets, bool]:
    """Transpose one standard affine window using shared feasible-tap arithmetic."""

    kernels = _as_tuple(result.values["kernel"])
    rank = len(kernels)
    strides = _as_tuple(result.values.get("stride", 1), rank)
    paddings = _as_tuple(result.values.get("padding", 0), rank)
    dilations = _as_tuple(result.values.get("dilation", 1), rank)
    mapped = list(_map_passthrough_forward(parent, child, parent_sets))
    parent_start = len(parent.shape) - rank
    child_start = len(child.shape) - rank
    exact = bool(result.values.get("exact", True))
    for local_axis in range(rank):
        source_set = parent_sets[parent_start + local_axis]
        if source_set is None:
            continue
        if result.map_index_set_forward is not None:
            image, callback_exact = _call_index_callback(
                result.map_index_set_forward, local_axis, source_set
            )
        elif result.map_index_set is not None or result.map_interval is not None:
            mapping = _Mapped(
                _Affine(Fraction(strides[local_axis]), Fraction(-paddings[local_axis])),
                _Affine(
                    Fraction(strides[local_axis]),
                    Fraction(
                        -paddings[local_axis] + dilations[local_axis] * (kernels[local_axis] - 1)
                    ),
                ),
                exact=exact,
            )
            image, callback_exact = _membership_image(
                source_set, child_start + local_axis, child, result, mapping, local_axis
            )
        else:
            image, callback_exact = map_transposed_convolution_index_set(
                local_axis,
                source_set,
                kernel=kernels,
                stride=strides,
                padding=paddings,
                dilation=dilations,
                input_extent=tuple(int(value) for value in child.shape[-rank:]),
            )
            callback_exact = callback_exact and exact
        mapped[child_start + local_axis] = image
        exact = exact and callback_exact and image.exact
    channel_dependency = str(result.values.get("channel_dependency", "full_exact"))
    parent_channel_axis = parent_start - 1
    child_channel_axis = child_start - 1
    if (
        parent_channel_axis >= 0
        and child_channel_axis >= 0
        and parent_sets[parent_channel_axis] is not None
        and channel_dependency != "pointwise"
    ):
        channel_exact = channel_dependency == "full_exact"
        mapped[child_channel_axis] = _IndexSet.interval(
            0, int(child.shape[child_channel_axis]) - 1, exact=channel_exact
        )
        exact = exact and channel_exact
    return tuple(mapped), exact


def _map_window_edges_forward(
    parent: Op, child: Op, parent_sets: _AxisSets, result: _RuleResult
) -> tuple[_AxisSets, bool]:
    """Transpose callback-defined edge maps through membership queries."""

    raw_edges = result.values.get("per_axis_edges")
    if not isinstance(raw_edges, Sequence) or isinstance(raw_edges, (str, bytes)):
        return _whole_child_envelope(child, parent_sets), False
    rank = len(raw_edges)
    parent_start = len(parent.shape) - rank
    child_start = len(child.shape) - rank
    mapped = list(_map_passthrough_forward(parent, child, parent_sets))
    exact = True
    for local_axis, raw_axis in enumerate(raw_edges):
        source_set = parent_sets[parent_start + local_axis]
        if source_set is None:
            continue
        try:
            lo, hi = cast(Sequence[Sequence[object]], raw_axis)
            mapping = _Mapped(
                _Affine(Fraction(cast(Any, lo[0])), Fraction(cast(Any, lo[1]))),
                _Affine(Fraction(cast(Any, hi[0])), Fraction(cast(Any, hi[1]))),
                exact=bool(result.values.get("exact", False)),
            )
        except (IndexError, TypeError, ValueError, ZeroDivisionError):
            return _whole_child_envelope(child, parent_sets), False
        if result.map_index_set_forward is not None:
            image, callback_exact = _call_index_callback(
                result.map_index_set_forward, local_axis, source_set
            )
        elif result.map_index_set is not None or result.map_interval is not None:
            image, callback_exact = _membership_image(
                source_set, child_start + local_axis, child, result, mapping, local_axis
            )
        else:
            image, callback_exact = _affine_membership_image(
                source_set, mapping, int(child.shape[child_start + local_axis])
            )
        mapped[child_start + local_axis] = image
        exact = exact and callback_exact and image.exact
    return tuple(mapped), exact


def _membership_image(
    source_set: _IndexSet,
    child_axis: int,
    child: Op,
    result: _RuleResult,
    mapping: _Mapped,
    local_axis: int,
) -> tuple[_IndexSet, bool]:
    """Compute one callback-backed image from the sealed backward oracle."""

    candidates = _candidate_envelope(source_set, mapping, int(child.shape[child_axis]))
    if _index_set_size(candidates) > _TRANSPOSE_CANDIDATE_BUDGET:
        return _IndexSet(candidates.progressions, exact=False), False

    def backward_map(candidate_set: _IndexSet) -> tuple[_IndexSet, bool]:
        """Invoke the rule's sealed backward callback for one candidate set."""

        if result.map_index_set is not None:
            return _call_index_callback(result.map_index_set, local_axis, candidate_set)
        assert result.map_interval is not None
        return _call_interval_callback(result.map_interval, local_axis, candidate_set)

    return forward_index_image(source_set, candidates, backward_map)


def _affine_membership_image(
    source_set: _IndexSet, mapping: _Mapped, child_extent: int
) -> tuple[_IndexSet, bool]:
    """Transpose a callback-free two-edge window by exact integer membership.

    The affine candidate envelope alone loses lattice structure: a point-edge
    map such as a step-2 slice admits at most one child per source index, yet
    interval inversion plus floor/ceil widening reports a nonempty child range
    for off-lattice sources. Every candidate is therefore tested against the
    window's own integer backward relation (``ceil(lo(c)) <= s <= floor(hi(c))``)
    so the image contains exactly the children whose windows reach the source
    set. Over-budget candidate ranges fall back to the inexact envelope.

    Parameters
    ----------
    source_set:
        Parent indices whose forward image is requested.
    mapping:
        Sealed backward affine edge map for this axis.
    child_extent:
        Child-axis extent used for clipping.

    Returns
    -------
    tuple[_IndexSet, bool]
        Membership-proven child indices and the hop's exactness.
    """

    candidates = _candidate_envelope(source_set, mapping, child_extent)
    if _index_set_size(candidates) > _TRANSPOSE_CANDIDATE_BUDGET:
        return _IndexSet(candidates.progressions, exact=False), False

    def backward_map(candidate_set: _IndexSet) -> tuple[_IndexSet, bool]:
        """Return the integer taps of each candidate child's affine window."""

        intervals: list[_IndexSet] = []
        for candidate in candidate_set.values():
            bounds = sorted(
                (
                    mapping.lo.a * candidate + mapping.lo.b,
                    mapping.hi.a * candidate + mapping.hi.b,
                )
            )
            start = ceil(bounds[0])
            stop = floor(bounds[1])
            intervals.append(_IndexSet.empty() if start > stop else _IndexSet.interval(start, stop))
        return _IndexSet.union(intervals), mapping.exact

    return forward_index_image(source_set, candidates, backward_map)


def _candidate_envelope(source_set: _IndexSet, mapping: _Mapped, child_extent: int) -> _IndexSet:
    """Return a clipped affine envelope containing all candidate child indices."""

    if source_set.is_empty:
        return _IndexSet.empty(exact=source_set.exact)
    transposed = _transpose_mapped(mapping, child_extent)
    minimum = source_set.minimum
    maximum = source_set.maximum
    assert minimum is not None and maximum is not None
    values = tuple(
        affine.a * coordinate + affine.b
        for affine in (transposed.lo, transposed.hi)
        for coordinate in (minimum, maximum)
    )
    start = max(0, floor(min(values)))
    stop = min(child_extent - 1, ceil(max(values)))
    return _IndexSet.interval(start, stop, exact=source_set.exact)


def _index_set_size(index_set: _IndexSet) -> int:
    """Return the represented candidate count without materializing it."""

    return sum(progression.count for progression in index_set.progressions)


def _map_passthrough_forward(
    parent: Op,
    child: Op,
    parent_sets: _AxisSets,
    result_spec: _RuleResult | None = None,
) -> _AxisSets:
    """Map identity, reduction, broadcast, and concatenation relations to a child."""

    parent_to_child: Mapping[int, int] | None = None
    if result_spec is not None:
        surviving = result_spec.values.get("surviving_parent_axes")
        if isinstance(surviving, Sequence) and not isinstance(surviving, (str, bytes)):
            surviving_axes = tuple(int(axis) for axis in surviving)
            if len(child.shape) == len(surviving_axes):
                parent_to_child = {
                    parent_axis: child_axis for child_axis, parent_axis in enumerate(surviving_axes)
                }
        if parent_to_child is None:
            parent_to_child = _engine._passthrough_axis_map(child, parent, result_spec)
    if parent_to_child is None:
        offset = len(child.shape) - len(parent.shape)
        parent_to_child = {axis: axis + offset for axis in range(len(parent.shape))}
    result: list[_IndexSet | None] = [None] * len(child.shape)
    for parent_axis, source_set in enumerate(parent_sets):
        if source_set is None:
            continue
        child_axis = parent_to_child.get(parent_axis, -1)
        if child_axis < 0 or child_axis >= len(child.shape):
            continue
        if int(parent.shape[parent_axis]) == 1 and int(child.shape[child_axis]) != 1:
            result[child_axis] = _IndexSet.interval(0, int(child.shape[child_axis]) - 1)
        else:
            result[child_axis] = source_set
    if result_spec is None:
        return tuple(result)
    concat_axis = result_spec.values.get("concatenate_axis")
    if not isinstance(concat_axis, int):
        return tuple(result)
    source_axis = next(
        (axis for axis, child_axis in parent_to_child.items() if child_axis == concat_axis),
        None,
    )
    offsets = _engine._concatenation_offsets(child, parent, result_spec)
    if bool(result_spec.values.get("stack", False)):
        result[concat_axis] = _IndexSet.from_values(offsets)
    elif source_axis is not None and parent_sets[source_axis] is not None:
        source_set = cast(_IndexSet, parent_sets[source_axis])
        result[concat_axis] = _IndexSet.from_values(
            (source_index + offset for offset in offsets for source_index in source_set.values()),
            exact=source_set.exact,
        )
    return tuple(result)


def _map_axis_forward(
    parent: Op, child: Op, parent_sets: _AxisSets, result: _RuleResult
) -> tuple[_AxisSets, bool]:
    """Transpose an explicit child-axis to parent-axis mapping.

    Surviving sliced axes apply the inverse of their recorded exact affine
    (``child = (parent - start) / step``, dropping coordinates off the slice
    lattice); a scalar-selected parent axis whose constrained source set
    excludes the recorded index prunes the whole forward influence to empty.
    Both were silently ignored before (disputed-r2 b6/R20-2/R20-3), so a
    projective query through a rank-changing getitem served unshifted,
    over-covering images.
    """

    raw = result.values.get("out_to_parent_axis", {})
    if not isinstance(raw, Mapping):
        return _whole_child_envelope(child, parent_sets), True
    raw_edges = result.values.get("out_axis_edges", {})
    edges = raw_edges if isinstance(raw_edges, Mapping) else {}
    raw_indices = result.values.get("selected_parent_indices", {})
    selected_indices = raw_indices if isinstance(raw_indices, Mapping) else {}
    raw_selected = result.values.get("selected_parent_axes", ())
    selected_axes = (
        tuple(int(axis) for axis in raw_selected)
        if isinstance(raw_selected, Sequence) and not isinstance(raw_selected, (str, bytes))
        else ()
    )
    mapped: list[_IndexSet | None] = [None] * len(child.shape)
    exact = True
    for child_axis, parent_axis in raw.items():
        if isinstance(child_axis, int) and isinstance(parent_axis, int):
            if 0 <= child_axis < len(mapped) and 0 <= parent_axis < len(parent_sets):
                source_set = parent_sets[parent_axis]
                edge = edges.get(child_axis)
                if source_set is not None and edge is not None:
                    step, start = int(edge[0]), int(edge[1])
                    source_set = _IndexSet.from_values(
                        (
                            (value - start) // step
                            for value in source_set.values()
                            if (value - start) % step == 0 and value >= start
                        ),
                        exact=source_set.exact,
                    )
                mapped[child_axis] = source_set
    for parent_axis in selected_axes:
        if not 0 <= parent_axis < len(parent_sets):
            continue
        source_set = parent_sets[parent_axis]
        if source_set is None:
            continue
        index = selected_indices.get(parent_axis)
        if isinstance(index, int):
            if index not in set(source_set.values()):
                # The constrained source coordinates never take the selected
                # index: nothing flows through this getitem at all.
                return tuple(_IndexSet.empty() for _ in child.shape), exact
        else:
            # Unknown selected index over a constrained axis: the forward
            # image may be empty, so the non-pruned claim is an upper bound.
            exact = False
    return tuple(mapped), exact


def _map_full_forward(
    parent: Op, child: Op, parent_sets: _AxisSets, result: _RuleResult
) -> tuple[_AxisSets, bool]:
    """Transpose exact whole-parent-axis dependence to the whole child extent."""

    mapped = list(_map_passthrough_forward(parent, child, parent_sets, result))
    selected = _select_full_axes(result.values.get("axes"), parent, child)
    surviving = result.values.get("surviving_parent_axes")
    parent_to_child = (
        {
            int(parent_axis): child_axis
            for child_axis, parent_axis in enumerate(cast(Sequence[int], surviving))
        }
        if isinstance(surviving, Sequence)
        and not isinstance(surviving, (str, bytes))
        and len(child.shape) == len(surviving)
        else {
            axis: axis + len(child.shape) - len(parent.shape) for axis in range(len(parent.shape))
        }
    )
    for parent_axis in selected:
        child_axis = parent_to_child.get(parent_axis, -1)
        if parent_sets[parent_axis] is not None and 0 <= child_axis < len(child.shape):
            mapped[child_axis] = _IndexSet.interval(0, int(child.shape[child_axis]) - 1)
    exact = bool(result.values.get("exact", True))
    return tuple(mapped), exact


def _rule_envelope(parent: Op, child: Op, parent_sets: _AxisSets, result: _RuleResult) -> _AxisSets:
    """Return the best affine envelope available for a unit-box-only rule."""

    if result.kind == "window" or result.kind == "window_edges":
        envelope_result = replace(result, unit_box=None)
        if result.kind == "window":
            mapped, _ = _map_window_forward(parent, child, parent_sets, envelope_result)
        else:
            mapped, _ = _map_window_edges_forward(parent, child, parent_sets, envelope_result)
        return tuple(
            None if item is None else _IndexSet(item.progressions, exact=False) for item in mapped
        )
    return _whole_child_envelope(child, parent_sets)


def _whole_child_envelope(child: Op, parent_sets: _AxisSets) -> _AxisSets:
    """Return a sound inexact whole-child fallback for constrained state."""

    if not any(item is not None for item in parent_sets):
        return (None,) * len(child.shape)
    return tuple(_IndexSet.interval(0, int(extent) - 1, exact=False) for extent in child.shape)


__all__: list[str] = []
