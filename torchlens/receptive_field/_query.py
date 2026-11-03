"""Exact per-unit receptive-field queries over the captured operation DAG."""

from __future__ import annotations

import threading
from collections.abc import Callable, Generator, Iterable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
from fractions import Fraction
from math import ceil, floor
from typing import TYPE_CHECKING, Any, TypeAlias, cast

from . import _engine
from ._engine_geometry import _as_tuple, _select_full_axes
from ._errors import AmbiguousInputError, ReceptiveFieldError
from ._index_set import (
    _AxisSets,
    _IndexSet,
    _Progression,
    _progression_sumset,
)
from ._query_causal import EXTENDED_KINDS, map_extended_to_parent
from ._query_maps import (
    _call_interval_callback,
    _coerce_index_set,
)
from ._reshape_blocks import apply_blocks
from ._rules import _RuleResult
from ._types import (
    ReceptiveField,
    ReceptiveFieldAxis,
    ReceptiveFieldBox,
    ReceptiveFieldBoxAxis,
    ReceptiveFieldStatus,
)

if TYPE_CHECKING:
    from .._data_classes.op import Op


class _SharedQueryState(threading.local):
    """Per-thread facts shared by the per-unit queries inside one :func:`_shared_query_scope`.

    Each query derives the solved operations, their reference index and the
    endpoint's path slice from the trace and solution alone, and its walk asks
    the engine for every operation's local rule result. Those are pure functions
    of the trace, the solution and the rule-registry epoch (which the engine's
    rule memo re-checks), so queries issued back to back over one solution can
    share them. Outside a scope nothing is cached and every query derives its
    own, exactly as before.
    """

    def __init__(self) -> None:
        """Start this thread with no open scope."""

        self.depth = 0
        self.graphs: dict[int, tuple[object, object, tuple[Op, ...], dict[str, Op]]] = {}
        self.reach: dict[tuple[str, int, str], tuple[object, frozenset[str]]] = {}


_SHARED_QUERY_STATE = _SharedQueryState()


@contextmanager
def _shared_query_scope() -> Iterator[None]:
    """Share graph facts and rule results across the per-unit queries run inside it.

    The always-on metadata invariants query one box per solved operation; on a
    chain of ``n`` operations each query re-derived ``O(n)`` graph facts and
    ``O(n)`` rule results. Inside this scope they are derived once per solution
    (graph facts) and once per operation (rule results, through one enclosing
    engine memo scope). Every query still walks its own path, so the boxes are
    unchanged.
    """

    state = _SHARED_QUERY_STATE
    with _engine._geometry_memo():
        state.depth += 1
        try:
            yield
        finally:
            state.depth -= 1
            if not state.depth:
                state.graphs.clear()
                state.reach.clear()


def _query_graph(trace: Any, solution: Any) -> tuple[tuple[Op, ...], dict[str, Op]]:
    """Return the solved operations in trace order and their label index."""

    state = _SHARED_QUERY_STATE
    if state.depth:
        cached = state.graphs.get(id(solution))
        if cached is not None and cached[0] is solution and cached[1] is trace:
            return cached[2], cached[3]
    operations = tuple(item for item in trace._layer_list if item.label in solution.per_op)
    by_reference = {
        reference: item
        for item in operations
        for reference in (item.label, item.layer_label, item._layer_label_raw)
    }
    if state.depth:
        state.graphs[id(solution)] = (solution, trace, operations, by_reference)
    return operations, by_reference


def _shared_reach(
    kind: str, solution: Any, label: str, derive: Callable[[], frozenset[str]]
) -> frozenset[str]:
    """Return ``derive()``, shared per ``(kind, solution, label)`` inside a scope."""

    state = _SHARED_QUERY_STATE
    if not state.depth:
        return derive()
    key = (kind, id(solution), label)
    cached = state.reach.get(key)
    if cached is not None and cached[0] is solution:
        return cached[1]
    value = derive()
    state.reach[key] = (solution, value)
    return value


def _query_memo_scope() -> AbstractContextManager[None]:
    """Return the per-query engine memo scope, or the enclosing shared one."""

    return nullcontext() if _SHARED_QUERY_STATE.depth else _engine._geometry_memo()


@dataclass(frozen=True)
class _TerminalState:
    """One reverse-walk path that reached the requested model input."""

    theoretical: _AxisSets
    clipped: _AxisSets
    exact: bool


def box_for_unit(
    solution: _engine._ReceptiveFieldSolution,
    op: Op,
    unit: Sequence[int],
    *,
    input_op: Op | str | None = None,
    source: Op | None = None,
    clip: bool = True,
    complete_unit: Sequence[int] | None = None,
) -> ReceptiveFieldBox:
    """Compute one per-unit receptive-field box by reverse index-set propagation.

    Parameters
    ----------
    solution:
        T3 whole-graph solution for the target's trace.
    op:
        Target operation.
    unit:
        Coordinates over the target's derived windowed output axes, ordered
        by ascending output axis (the target's own grid order).
    input_op:
        Optional input operation or exact IO role/operation label.
    source:
        Optional source operation for a source-seeded solution. Its output grid is the
        coordinate space of the returned box.
    clip:
        Whether ``clipped_*`` bounds are intersected with captured input extents.
    complete_unit:
        Optional complete target index used internally to route structural axes.

    """
    if input_op is not None and source is not None:
        raise TypeError("input and source cannot be supplied together.")
    descriptors = solution.per_op.get(op.label)
    if not descriptors:
        endpoint = source.label if source is not None else op.label
        raise ReceptiveFieldError(
            f"No receptive-field solution is available from {endpoint} to {op.label}."
        )
    descriptor = _select_descriptor(descriptors, source if source is not None else input_op)
    _validate_descriptor_for_query(descriptor)
    coordinates = _normalize_unit(op, descriptor, unit)
    initial = _initial_axis_sets(op, descriptor, coordinates, complete_unit=complete_unit)
    operations, by_reference = _query_graph(op.source_trace, solution)
    ancestry = _shared_reach(
        "receptive",
        solution,
        descriptor.input_op_label,
        lambda: _ancestor_labels(descriptor.input_op_label, operations, by_reference),
    )
    # The walk enumerates paths, so it revisits each operation once per path
    # through it, re-deriving per-operation facts that are linear in that
    # operation's parents. Memoize them for this one query (or for every query
    # of an enclosing shared scope).
    with _query_memo_scope():
        terminals = _walk_to_input(
            op,
            initial,
            descriptor.input_op_label,
            ancestry,
            by_reference,
            True,
            {},
        )
    if not terminals:
        raise ReceptiveFieldError(
            f"The target {op.label} has no live path to {descriptor.input_op_label}."
        )
    return _build_box(op, descriptor, coordinates, terminals, clip=clip)


def _select_descriptor(
    descriptors: Mapping[str, ReceptiveField], selected: Op | str | None
) -> ReceptiveField:
    """Resolve one input descriptor, rejecting ambiguous convenience queries."""
    if selected is None:
        if len(descriptors) != 1:
            roles = ", ".join(descriptors)
            raise AmbiguousInputError(f"Select one reachable input; available IO roles: {roles}.")
        return next(iter(descriptors.values()))
    identity = selected if isinstance(selected, str) else selected.label
    for role, descriptor in descriptors.items():
        if identity in {role, descriptor.input_op_label}:
            return descriptor
    raise ReceptiveFieldError(f"Input {identity!r} is not reachable from this target.")


def _validate_descriptor_for_query(
    descriptor: ReceptiveField, *, require_windowed: bool = True
) -> None:
    """Reject descriptors whose derived grid cannot support a geometric query.

    A single-unit query needs a windowed axis; a set query narrows whole-extent
    axes as it walks, so it only needs a solved descriptor (``require_windowed``).
    """
    if descriptor.axes is None or descriptor.status in {
        ReceptiveFieldStatus.DATA_DEPENDENT,
        ReceptiveFieldStatus.UNKNOWN,
        ReceptiveFieldStatus.UNSUPPORTED,
    }:
        raise ReceptiveFieldError("Per-unit geometry is unavailable; use .probe_grad() instead.")
    if require_windowed and not descriptor.axis_roles.windowed_axes:
        raise ReceptiveFieldError("No windowed axes are available; use .probe_grad() instead.")
    if any(kind == "unknown" for kind in descriptor.axis_roles.axis_kinds):
        raise ReceptiveFieldError("The derived layout is ambiguous; use .probe_grad() instead.")


def _windowed_axes_in_unit_order(descriptor: ReceptiveField) -> tuple[ReceptiveFieldAxis, ...]:
    """Return windowed descriptor axes in the unit's coordinate order.

    A ``unit`` tuple addresses the seed operation's OWN output grid, so its
    coordinates are ordered by ascending ``output_axis`` — the order the
    windowed axes appear in that operation's shape. Descriptor axes are stored
    by ascending ``input_axis`` instead, and any axis permutation between the
    two endpoints (transpose/permute/movedim) makes those orders differ, so
    every producer and consumer of a windowed unit tuple must route through
    this one ordering (R20-1: zipping coordinates in descriptor order silently
    transposed exact boxes).
    """
    assert descriptor.axes is not None
    windowed = tuple(axis for axis in descriptor.axes if axis.kind == "windowed")
    return tuple(sorted(windowed, key=lambda axis: cast(int, axis.output_axis)))


def _normalize_unit(op: Op, descriptor: ReceptiveField, unit: Sequence[int]) -> tuple[int, ...]:
    """Validate or resolve target windowed-axis coordinates.

    Coordinates are ordered by ascending output axis (the seed operation's own
    grid order; see :func:`_windowed_axes_in_unit_order`).

    Geometric queries (``.at()``) require in-range non-negative coordinates:
    a coordinate ``< 0`` is rejected as out of bounds, it is NOT Python-wrapped.
    This differs deliberately from the gradient/validation path
    (``_gradient._normalize_unit``), which wraps negative complete-index
    coordinates. The two contracts are distinct and are not unified here.
    """
    assert descriptor.axes is not None
    output_axes = tuple(
        cast(int, axis.output_axis) for axis in _windowed_axes_in_unit_order(descriptor)
    )
    coordinates = tuple(unit)
    if len(coordinates) != len(output_axes):
        raise ReceptiveFieldError(
            f"unit requires {len(output_axes)} windowed coordinates; got {len(coordinates)}."
        )
    for coordinate, output_axis in zip(coordinates, output_axes, strict=True):
        if isinstance(coordinate, bool) or not isinstance(coordinate, int):
            raise ReceptiveFieldError("unit coordinates must be integers.")
        if coordinate < 0 or coordinate >= int(op.shape[output_axis]):
            raise ReceptiveFieldError(
                f"unit coordinate {coordinate} is out of bounds for output axis {output_axis}."
            )
    return coordinates


def _initial_axis_sets(
    op: Op,
    descriptor: ReceptiveField,
    coordinates: tuple[int, ...],
    *,
    complete_unit: Sequence[int] | None = None,
) -> _AxisSets:
    """Seed singleton sets on derived axes and optional structural coordinates."""
    assert descriptor.axes is not None
    result: list[_IndexSet | None] = [None] * len(op.shape)
    if complete_unit is not None:
        if len(complete_unit) != len(op.shape):
            raise ReceptiveFieldError("complete_unit rank does not match the target operation.")
        result = [_IndexSet.singleton(int(coordinate)) for coordinate in complete_unit]
    for axis, coordinate in zip(_windowed_axes_in_unit_order(descriptor), coordinates, strict=True):
        assert axis.output_axis is not None
        existing = result[axis.output_axis]
        singleton = _IndexSet.singleton(coordinate)
        result[axis.output_axis] = (
            singleton if existing is None else _IndexSet.union((existing, singleton))
        )
    return tuple(result)


def _ancestor_labels(
    input_label: str, operations: tuple[Op, ...], by_reference: Mapping[str, Op]
) -> frozenset[str]:
    """Return operations reachable forward from one requested model input."""
    children: dict[str, set[str]] = {item.label: set() for item in operations}
    for item in operations:
        for parent in item.parents:
            if parent in by_reference:
                children[by_reference[parent].label].add(item.label)
    reachable = {input_label}
    stack = [input_label]
    while stack:
        label = stack.pop()
        unseen = children.get(label, set()) - reachable
        reachable.update(unseen)
        stack.extend(unseen)
    return frozenset(reachable)


def _distinct_terminals(terminals: Iterable[_TerminalState]) -> tuple[_TerminalState, ...]:
    """Collapse repeated terminal states produced by distinct paths.

    Path enumeration reaches the same terminal state once per path, so a
    merge-heavy graph grows the terminal tuple exponentially while adding no
    information. Every consumer -- the emptiness test in the two query entry
    points and all four aggregations in :func:`_build_box` (two unions, one
    ``any``, one ``all``) -- is insensitive to both multiplicity and order, and
    first-occurrence order is preserved here, so the resolved box is unchanged.
    """
    return tuple(dict.fromkeys(terminals))


_WalkKey: TypeAlias = tuple[str, _AxisSets, bool]
_WalkStep: TypeAlias = Generator[
    tuple["Op", _AxisSets, bool], tuple[_TerminalState, ...], tuple[_TerminalState, ...]
]


def _run_memoized_walk(
    body: Callable[[Op, _AxisSets, bool], _WalkStep],
    op: Op,
    sets: _AxisSets,
    exact: bool,
    memo: dict[_WalkKey, tuple[_TerminalState, ...]],
) -> tuple[_TerminalState, ...]:
    """Drive a hop-by-hop graph walk on an explicit stack.

    ``body(op, sets, exact)`` is a generator for one hop: it yields one
    ``(next_op, next_sets, next_exact)`` request per branch, receives that
    branch's terminal tuple back, and returns the hop's own terminal tuple.
    This driver is the recursion ``f(op, sets, exact) = memo or body(...)``
    with every pending hop kept on a list instead of the interpreter stack,
    so the walk's depth is bounded by memory rather than the recursion limit.
    Branches are visited in yield order and each finishes before the next
    starts, so ``memo`` gains its ``(label, sets, exact)`` keys and the
    terminal tuples keep their first-occurrence order exactly as the
    recursion produced them.
    """
    key: _WalkKey = (op.label, sets, exact)
    cached = memo.get(key)
    if cached is not None:
        return cached
    stack: list[tuple[_WalkKey, _WalkStep]] = [(key, body(op, sets, exact))]
    sent: tuple[_TerminalState, ...] | None = None
    while True:
        frame_key, step = stack[-1]
        try:
            request = next(step) if sent is None else step.send(sent)
        except StopIteration as finished:
            stack.pop()
            memo[frame_key] = finished.value
            if not stack:
                return finished.value
            sent = finished.value
            continue
        next_op, next_sets, next_exact = request
        next_key: _WalkKey = (next_op.label, next_sets, next_exact)
        cached = memo.get(next_key)
        if cached is not None:
            sent = cached
            continue
        stack.append((next_key, body(next_op, next_sets, next_exact)))
        sent = None


def _walk_to_input(
    op: Op,
    output_sets: _AxisSets,
    input_label: str,
    ancestry: frozenset[str],
    by_reference: Mapping[str, Op],
    exact: bool,
    memo: dict[_WalkKey, tuple[_TerminalState, ...]],
) -> tuple[_TerminalState, ...]:
    """Map a joint axis-set state to one requested model input.

    The walk enumerates paths, so a merge-heavy graph re-derives the same
    ``(operation, axis-set state)`` subwalk once per path that reaches it --
    exponential in the number of merges. ``memo`` caches that pure subresult
    for the duration of one query. It is plain memoization of a pure function:
    the cached tuple is exactly what a fresh walk would rebuild. Terminal
    de-duplication is a separate step; see :func:`_distinct_terminals`. The
    hops run on :func:`_run_memoized_walk`'s explicit stack, so a model of any
    depth walks without reaching the interpreter's recursion limit.
    """

    def body(item: Op, sets: _AxisSets, item_exact: bool) -> _WalkStep:
        """Start one receptive hop from ``item`` toward this query's input."""
        return _walk_body(item, sets, input_label, ancestry, by_reference, item_exact)

    return _run_memoized_walk(body, op, output_sets, exact, memo)


def _walk_body(
    op: Op,
    output_sets: _AxisSets,
    input_label: str,
    ancestry: frozenset[str],
    by_reference: Mapping[str, Op],
    exact: bool,
) -> _WalkStep:
    """Map one joint axis-set state across a single reverse hop.

    Yields one request per live parent branch and receives that branch's
    terminals; see :func:`_run_memoized_walk`.
    """
    if op.label == input_label:
        clipped = tuple(
            None if item is None else item.clipped(int(op.shape[axis]))
            for axis, item in enumerate(output_sets)
        )
        sets_exact = all(item is None or item.exact for item in output_sets)
        return (_TerminalState(output_sets, clipped, exact and sets_exact),)
    result, _ = _engine._rule_result(op)
    parents = tuple(
        by_reference[label]
        for label in op.parents
        if label in by_reference and by_reference[label].label in ancestry
    )
    terminals: list[_TerminalState] = []
    for parent in parents:
        parent_sets, hop_exact = _map_to_parent(op, parent, output_sets, result)
        if isinstance(result.values.get("concatenate_axis"), int) and any(
            item is not None and item.is_empty for item in parent_sets
        ):
            continue
        bounded = (
            parent_sets
            if parent.label == input_label
            else tuple(
                None if item is None else item.clipped(int(parent.shape[axis]))
                for axis, item in enumerate(parent_sets)
            )
        )
        terminals.extend(
            (
                yield (
                    parent,
                    bounded,
                    exact and hop_exact and all(item is None or item.exact for item in parent_sets),
                )
            )
        )
    return _distinct_terminals(terminals)


def _map_to_parent(
    op: Op, parent: Op, output_sets: _AxisSets, result: _RuleResult
) -> tuple[_AxisSets, bool]:
    """Apply one local rule result to one parent branch."""
    if result.kind in EXTENDED_KINDS:
        return map_extended_to_parent(_map_to_parent, op, parent, output_sets, result)
    if result.kind == "window":
        return _map_window(op, parent, output_sets, result)
    if result.kind == "window_edges":
        return _map_window_edges(op, parent, output_sets, result)
    if result.kind == "full":
        mapped = list(_map_passthrough(op, parent, output_sets, result))
        for axis in _select_full_axes(result.values.get("axes"), parent, op):
            mapped[axis] = _IndexSet.interval(0, int(parent.shape[axis]) - 1)
        return tuple(mapped), bool(result.values.get("exact", True))
    if result.kind == "axis_map":
        return _map_axis_mapping(op, parent, output_sets, result)
    if result.kind == "passthrough":
        return _map_passthrough(op, parent, output_sets, result), True
    return _map_composed_envelope(op, parent, output_sets), False


def _map_window(
    op: Op, parent: Op, output_sets: _AxisSets, result: _RuleResult
) -> tuple[_AxisSets, bool]:
    """Map standard or callback-defined window coordinates to a parent tensor."""
    kernels = _as_tuple(result.values["kernel"])
    rank = len(kernels)
    strides = _as_tuple(result.values.get("stride", 1), rank)
    paddings = _as_tuple(result.values.get("padding", 0), rank)
    dilations = _as_tuple(result.values.get("dilation", 1), rank)
    parent_sets = list(_map_passthrough(op, parent, output_sets))
    parent_start = len(parent.shape) - rank
    output_start = len(op.shape) - rank
    exact = True if result.map_index_set is not None else bool(result.values.get("exact", True))
    for local_axis in range(rank):
        output_set = output_sets[output_start + local_axis]
        if output_set is None:
            continue
        if result.map_index_set is not None:
            mapped, callback_exact = _call_index_callback(
                result.map_index_set, local_axis, output_set
            )
        elif result.map_interval is not None:
            mapped, callback_exact = _call_interval_callback(
                result.map_interval, local_axis, output_set
            )
        else:
            mapped = _standard_window_index_set(
                output_set,
                kernel=kernels[local_axis],
                stride=strides[local_axis],
                padding=paddings[local_axis],
                dilation=dilations[local_axis],
            )
            callback_exact = True
        parent_sets[parent_start + local_axis] = mapped
        exact = exact and callback_exact
    channel_dependency = str(result.values.get("channel_dependency", "full_exact"))
    channel_axis = parent_start - 1
    if channel_axis >= 0 and channel_dependency != "pointwise":
        channel_exact = channel_dependency == "full_exact"
        parent_sets[channel_axis] = _IndexSet.interval(
            0, int(parent.shape[channel_axis]) - 1, exact=channel_exact
        )
        exact = exact and channel_exact
    return tuple(parent_sets), exact


def _standard_window_index_set(
    output_set: _IndexSet, *, kernel: int, stride: int, padding: int, dilation: int
) -> _IndexSet:
    """Map a standard affine window while retaining dilation residue structure.

    A one-progression output set whose window image is again one progression
    (every stride-1 window over an interval) is mapped in closed form; anything
    else enumerates. Both yield the same canonical set.
    """
    progressions = output_set.progressions
    if len(progressions) == 1 and kernel > 0 and stride > 0 and dilation > 0:
        item = progressions[0]
        image = _progression_sumset(
            _Progression(stride * item.start - padding, stride * item.step, item.count),
            _Progression(0, dilation, kernel),
        )
        if image is not None:
            return _IndexSet((image,), output_set.exact)
    values = (
        stride * output_index - padding + dilation * tap
        for output_index in output_set.values()
        for tap in range(kernel)
    )
    return _IndexSet.from_values(values, exact=output_set.exact)


def _map_passthrough(
    op: Op,
    parent: Op,
    output_sets: _AxisSets,
    result_spec: _RuleResult | None = None,
) -> _AxisSets:
    """Map identity, reduction, broadcast, and concatenation coordinates to one parent."""

    parent_to_child: Mapping[int, int] | None = None
    if result_spec is not None:
        surviving = result_spec.values.get("surviving_parent_axes")
        if isinstance(surviving, Sequence) and not isinstance(surviving, (str, bytes)):
            surviving_axes = tuple(int(axis) for axis in surviving)
            if len(op.shape) == len(surviving_axes):
                parent_to_child = {
                    parent_axis: child_axis for child_axis, parent_axis in enumerate(surviving_axes)
                }
        if parent_to_child is None:
            parent_to_child = _engine._passthrough_axis_map(op, parent, result_spec)
    if parent_to_child is None:
        offset = len(op.shape) - len(parent.shape)
        parent_to_child = {axis: axis + offset for axis in range(len(parent.shape))}
    result: list[_IndexSet | None] = [None] * len(parent.shape)
    for parent_axis in range(len(parent.shape)):
        output_axis = parent_to_child.get(parent_axis, -1)
        if output_axis < 0 or output_axis >= len(output_sets):
            continue
        output_set = output_sets[output_axis]
        if output_set is None:
            continue
        if int(parent.shape[parent_axis]) == 1 and int(op.shape[output_axis]) != 1:
            result[parent_axis] = _IndexSet.singleton(0)
        else:
            result[parent_axis] = output_set
    if result_spec is None:
        return tuple(result)
    concat_axis = result_spec.values.get("concatenate_axis")
    if not isinstance(concat_axis, int):
        return tuple(result)
    offsets = _engine._concatenation_offsets(op, parent, result_spec)
    selected = output_sets[concat_axis]
    if selected is None:
        return tuple(result)
    if bool(result_spec.values.get("stack", False)):
        routed = any(index in offsets for index in selected.values())
        if not routed:
            empty_axis = next((axis for axis, value in enumerate(result) if value is not None), 0)
            result[empty_axis] = _IndexSet.empty()
        return tuple(result)
    routed_parent_axis = next(
        (axis for axis, child_axis in parent_to_child.items() if child_axis == concat_axis),
        None,
    )
    if routed_parent_axis is None:
        return tuple(result)
    extent = int(parent.shape[routed_parent_axis])
    values = (
        child_index - offset
        for offset in offsets
        for child_index in selected.values()
        if offset <= child_index < offset + extent
    )
    result[routed_parent_axis] = _IndexSet.from_values(values, exact=selected.exact)
    return tuple(result)


def _map_axis_mapping(
    op: Op, parent: Op, output_sets: _AxisSets, result: _RuleResult
) -> tuple[_AxisSets, bool]:
    """Apply an explicit output-axis to parent-axis mapping.

    Surviving sliced axes apply their recorded exact affine
    (``parent = step * out + start``; the historical path copied index sets
    through unshifted, dropping every slice offset — disputed-r2 b6/R20-2).
    Scalar-selected parent axes narrow to their recorded singleton index;
    a selected axis with no recorded index and extent > 1 falls back to its
    conservative full extent and makes the hop non-exact (R20-3: the
    conservative bound must be visible in the box's exactness claim).
    """
    raw = result.values.get("out_to_parent_axis", {})
    if not isinstance(raw, Mapping):
        return (None,) * len(parent.shape), True
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
    mapped: list[_IndexSet | None] = [None] * len(parent.shape)
    exact = True
    for output_axis, parent_axis in raw.items():
        if isinstance(output_axis, int) and isinstance(parent_axis, int):
            if 0 <= output_axis < len(output_sets) and 0 <= parent_axis < len(mapped):
                output_set = output_sets[output_axis]
                edge = edges.get(output_axis)
                if output_set is not None and edge is not None:
                    step, start = int(edge[0]), int(edge[1])
                    output_set = _IndexSet.from_values(
                        (step * value + start for value in output_set.values()),
                        exact=output_set.exact,
                    )
                mapped[parent_axis] = output_set
    for parent_axis in selected_axes:
        if not 0 <= parent_axis < len(mapped):
            continue
        index = selected_indices.get(parent_axis)
        extent = int(parent.shape[parent_axis])
        if isinstance(index, int) and 0 <= index < extent:
            mapped[parent_axis] = _IndexSet.singleton(index)
        elif extent == 1:
            mapped[parent_axis] = _IndexSet.singleton(0)
        else:
            mapped[parent_axis] = _IndexSet.interval(0, extent - 1, exact=False)
            exact = False
    return apply_blocks(result, output_sets, op, parent, mapped, exact, forward=False)


def _map_window_edges(
    op: Op, parent: Op, output_sets: _AxisSets, result: _RuleResult
) -> tuple[_AxisSets, bool]:
    """Lift an endpoint-monotone two-edge result into an integer-set envelope."""
    raw_edges = result.values.get("per_axis_edges")
    if not isinstance(raw_edges, Sequence):
        return _map_composed_envelope(op, parent, output_sets), False
    mapped = list(_map_passthrough(op, parent, output_sets))
    rank = len(raw_edges)
    callbacks_exact = True
    for local_axis, raw_axis in enumerate(raw_edges):
        output_set = output_sets[len(op.shape) - rank + local_axis]
        if output_set is None:
            continue
        if result.map_index_set is not None:
            index_set, callback_exact = _call_index_callback(
                result.map_index_set, local_axis, output_set
            )
            mapped[len(parent.shape) - rank + local_axis] = index_set
            callbacks_exact = callbacks_exact and callback_exact
            continue
        if result.map_interval is not None:
            index_set, callback_exact = _call_interval_callback(
                result.map_interval, local_axis, output_set
            )
            mapped[len(parent.shape) - rank + local_axis] = index_set
            callbacks_exact = callbacks_exact and callback_exact
            continue
        lo, hi = cast(Sequence[Sequence[object]], raw_axis)
        values = output_set.values()
        starts = [
            Fraction(cast(Any, lo[0])) * value + Fraction(cast(Any, lo[1])) for value in values
        ]
        stops = [
            Fraction(cast(Any, hi[0])) * value + Fraction(cast(Any, hi[1])) for value in values
        ]
        mapped[len(parent.shape) - rank + local_axis] = _IndexSet.interval(
            floor(min(starts)), ceil(max(stops))
        )
    if result.map_index_set or result.map_interval:
        return tuple(mapped), callbacks_exact
    return tuple(mapped), bool(result.values.get("exact", False))


def _map_composed_envelope(op: Op, parent: Op, output_sets: _AxisSets) -> _AxisSets:
    """Return a sound whole-parent fallback for every constrained output axis."""
    if not any(item is not None for item in output_sets):
        return (None,) * len(parent.shape)
    return tuple(_IndexSet.interval(0, int(extent) - 1, exact=False) for extent in parent.shape)


def _call_index_callback(callback: Any, axis: int, output_set: _IndexSet) -> tuple[_IndexSet, bool]:
    """Invoke and normalize a rule's progression-set callback result."""
    raw, exact = callback(axis, output_set)
    mapped = _coerce_index_set(raw, exact=bool(exact) and output_set.exact)
    return mapped, bool(exact)


def _build_box(
    op: Op,
    descriptor: ReceptiveField,
    unit: tuple[int, ...],
    terminals: tuple[_TerminalState, ...],
    *,
    clip: bool,
) -> ReceptiveFieldBox:
    """Aggregate terminal path sets into the public three-level bound model."""
    assert descriptor.axes is not None
    theoretical_sets = tuple(
        _union_optional(terminal.theoretical[axis] for terminal in terminals)
        for axis in range(len(descriptor.input_shape))
    )
    clipped_sets = tuple(
        _union_optional(terminal.clipped[axis] for terminal in terminals)
        for axis in range(len(descriptor.input_shape))
    )
    path_nonempty = any(
        all(
            item is not None and not item.is_empty
            for item in (terminal.clipped[axis] for axis in descriptor.axis_roles.windowed_axes)
        )
        for terminal in terminals
    )
    exact = all(terminal.exact for terminal in terminals)
    box_axes: list[ReceptiveFieldBoxAxis] = []
    was_clipped = False
    for axis, (axis_descriptor, extent) in enumerate(
        zip(descriptor.axes, descriptor.input_shape, strict=True)
    ):
        if axis_descriptor.kind == "pointwise":
            # A surviving pointwise axis is identity-geometry by construction
            # (non-identity composed geometry degrades to a "full" envelope in
            # _engine_descriptor, disputed-r3 F1/F2, and the full case below
            # folds its inexactness), so the same-index claim is sound and the
            # box's exactness stays the query-time derivation from windowed
            # terminals — the static descriptor ``exact`` flag is a
            # conservative composition artifact shared with windowed axes and
            # deliberately NOT folded here.
            box_axes.append(
                ReceptiveFieldBoxAxis(
                    axis,
                    "pointwise",
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    sparse_possible=axis_descriptor.sparse_possible,
                )
            )
        elif axis_descriptor.kind == "full":
            narrowed = theoretical_sets[axis]
            if narrowed is not None and not narrowed.is_empty:
                # The query walk produced a real per-axis index set (for
                # example a scalar-selected getitem axis narrowed to its
                # recorded singleton). Serve the walk's tight bounds instead
                # of the descriptor's conservative whole extent.
                narrowed_minimum = narrowed.minimum
                narrowed_maximum = narrowed.maximum
                assert narrowed_minimum is not None and narrowed_maximum is not None
                box_axes.append(
                    ReceptiveFieldBoxAxis(
                        axis,
                        "full",
                        None,
                        None,
                        narrowed_minimum,
                        narrowed_maximum + 1,
                        max(narrowed_minimum, 0),
                        min(narrowed_maximum + 1, extent),
                        sparse_possible=axis_descriptor.sparse_possible,
                    )
                )
            else:
                # No walk narrowing: the whole-extent claim is only as exact
                # as the descriptor's own axis evidence. A conservative full
                # axis (for example an unresolved scalar selection over
                # extent > 1) must be visible in the box's exactness claim —
                # the historical box derived `exact` from windowed terminals
                # alone, so this dishonest-exact class was tripwire-BLIND
                # (disputed-r2 b6/R20-3).
                exact = exact and axis_descriptor.exact
                box_axes.append(
                    ReceptiveFieldBoxAxis(
                        axis,
                        "full",
                        None,
                        None,
                        0,
                        extent,
                        0,
                        extent,
                        sparse_possible=axis_descriptor.sparse_possible,
                    )
                )
        elif axis_descriptor.kind == "windowed":
            theoretical = theoretical_sets[axis]
            actual = clipped_sets[axis]
            index_start = None if theoretical is None else theoretical.minimum
            maximum = None if theoretical is None else theoretical.maximum
            index_stop = None if maximum is None else maximum + 1
            actual_start = None if actual is None else actual.minimum
            actual_maximum = None if actual is None else actual.maximum
            actual_stop = None if actual_maximum is None else actual_maximum + 1
            if not clip:
                actual_start, actual_stop = index_start, index_stop
            was_clipped = was_clipped or (
                clip and (actual_start, actual_stop) != (index_start, index_stop)
            )
            box_axes.append(
                ReceptiveFieldBoxAxis(
                    axis,
                    "windowed",
                    None if index_start is None else Fraction(index_start),
                    None if maximum is None else Fraction(maximum),
                    index_start,
                    index_stop,
                    actual_start,
                    actual_stop,
                    sparse_possible=axis_descriptor.sparse_possible,
                )
            )
        else:
            box_axes.append(
                ReceptiveFieldBoxAxis(
                    axis,
                    "unknown",
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    sparse_possible=axis_descriptor.sparse_possible,
                )
            )
    empty = not path_nonempty
    covers_input = not empty and all(
        axis.kind == "pointwise" or (axis.clipped_start == 0 and axis.clipped_stop == extent)
        for axis, extent in zip(box_axes, descriptor.input_shape, strict=True)
    )
    status = (
        ReceptiveFieldStatus.WHOLE_INPUT
        if exact and descriptor.status is ReceptiveFieldStatus.WHOLE_INPUT
        else ReceptiveFieldStatus.EXACT
        if exact
        else ReceptiveFieldStatus.UPPER_BOUND
    )
    return ReceptiveFieldBox(
        op_label=op.label,
        io_role=descriptor.io_role,
        unit=unit,
        axes=tuple(box_axes),
        input_shape=descriptor.input_shape,
        status=status,
        exact=exact,
        clipped=was_clipped,
        empty=empty,
        covers_input=covers_input,
    )


def _union_optional(sets: Iterable[_IndexSet | None]) -> _IndexSet | None:
    """Union constrained sets while preserving an unconstrained ``None`` result."""
    materialized = tuple(item for item in sets if item is not None)
    return None if not materialized else _IndexSet.union(materialized)


__all__: list[str] = []
