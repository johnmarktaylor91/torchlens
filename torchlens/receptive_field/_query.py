"""Exact per-unit receptive-field queries over the captured operation DAG."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from math import ceil, floor, gcd, ulp
from typing import TYPE_CHECKING, Any, TypeAlias, cast

from . import _engine
from ._engine_geometry import _as_tuple, _select_full_axes
from ._errors import AmbiguousInputError, ReceptiveFieldError
from ._rules import _RuleResult
from ._types import (
    ReceptiveField,
    ReceptiveFieldBox,
    ReceptiveFieldBoxAxis,
    ReceptiveFieldStatus,
)


if TYPE_CHECKING:
    from ..data_classes.op import Op


_PROGRESSION_BUDGET = 16


@dataclass(frozen=True, order=True)
class _Progression:
    """Finite arithmetic progression of integer indices."""

    start: int
    step: int
    count: int

    def __post_init__(self) -> None:
        """Validate canonical progression parameters."""
        if self.step <= 0 or self.count <= 0:
            raise ValueError("progression step and count must be positive")

    @property
    def stop(self) -> int:
        """Return the inclusive final index."""
        return self.start + self.step * (self.count - 1)

    def values(self) -> range:
        """Return a lazy range over all represented indices."""
        return range(self.start, self.stop + 1, self.step)


@dataclass(frozen=True)
class _IndexSet:
    """Bounded union of arithmetic progressions with honest collapse metadata."""

    progressions: tuple[_Progression, ...]
    exact: bool = True

    @classmethod
    def empty(cls, *, exact: bool = True) -> _IndexSet:
        """Construct an empty index set with the requested exactness."""
        return cls((), exact)

    @classmethod
    def singleton(cls, value: int) -> _IndexSet:
        """Construct a singleton index set."""
        return cls((_Progression(value, 1, 1),))

    @classmethod
    def interval(cls, start: int, stop: int, *, exact: bool = True) -> _IndexSet:
        """Construct an inclusive integer interval."""
        if stop < start:
            return cls.empty(exact=exact)
        return cls((_Progression(start, 1, stop - start + 1),), exact)

    @classmethod
    def from_values(cls, values: Iterable[int], *, exact: bool = True) -> _IndexSet:
        """Compress values into at most sixteen runs, else an inexact dense hull."""
        ordered = sorted(set(values))
        if not ordered:
            return cls.empty(exact=exact)
        runs: list[_Progression] = []
        run_start = ordered[0]
        run_step = 1
        run_count = 1
        for index, value in enumerate(ordered[1:], start=1):
            difference = value - ordered[index - 1]
            if run_count == 1:
                run_step = difference
                run_count = 2
            elif difference == run_step:
                run_count += 1
            else:
                runs.append(_Progression(run_start, run_step, run_count))
                run_start = value
                run_step = 1
                run_count = 1
        runs.append(_Progression(run_start, run_step, run_count))
        if len(runs) > _PROGRESSION_BUDGET:
            return cls.interval(ordered[0], ordered[-1], exact=False)
        return cls(tuple(runs), exact)

    @property
    def is_empty(self) -> bool:
        """Return whether no indices are represented."""
        return not self.progressions

    @property
    def minimum(self) -> int | None:
        """Return the smallest index, or ``None`` for an empty set."""
        return None if self.is_empty else min(item.start for item in self.progressions)

    @property
    def maximum(self) -> int | None:
        """Return the largest index, or ``None`` for an empty set."""
        return None if self.is_empty else max(item.stop for item in self.progressions)

    def values(self) -> tuple[int, ...]:
        """Materialize represented indices in sorted order."""
        return tuple(sorted({value for item in self.progressions for value in item.values()}))

    def clipped(self, extent: int) -> _IndexSet:
        """Intersect this set with ``[0, extent)`` without changing exactness."""
        return _IndexSet.from_values(
            (value for value in self.values() if 0 <= value < extent), exact=self.exact
        )

    @classmethod
    def union(cls, sets: Iterable[_IndexSet]) -> _IndexSet:
        """Union progression sets and enforce the global progression budget."""
        materialized = tuple(sets)
        return cls.from_values(
            (value for index_set in materialized for value in index_set.values()),
            exact=all(index_set.exact for index_set in materialized),
        )


_AxisSets: TypeAlias = tuple[_IndexSet | None, ...]


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
    input: Op | str | None = None,
    source: Op | None = None,
    clip: bool = True,
) -> ReceptiveFieldBox:
    """Compute one per-unit receptive-field box by reverse index-set propagation.

    Parameters
    ----------
    solution:
        T3 whole-graph solution for the target's trace.
    op:
        Target operation.
    unit:
        Coordinates over the target's derived windowed output axes.
    input:
        Optional input operation or exact IO role/operation label.
    source:
        Optional source operation for a source-seeded solution. Its output grid is the
        coordinate space of the returned box.
    clip:
        Whether ``clipped_*`` bounds are intersected with captured input extents.

    """
    if input is not None and source is not None:
        raise TypeError("input and source cannot be supplied together.")
    descriptors = solution.per_op.get(op.label)
    if not descriptors:
        endpoint = source.label if source is not None else op.label
        raise ReceptiveFieldError(
            f"No receptive-field solution is available from {endpoint} to {op.label}."
        )
    descriptor = _select_descriptor(descriptors, source if source is not None else input)
    _validate_descriptor_for_query(descriptor)
    coordinates = _normalize_unit(op, descriptor, unit)
    initial = _initial_axis_sets(op, descriptor, coordinates)
    trace = op.source_trace
    operations = tuple(item for item in trace.layer_list if item.label in solution.per_op)
    by_reference = {
        reference: item
        for item in operations
        for reference in (item.label, item.layer_label, item._layer_label_raw)
    }
    ancestry = _ancestor_labels(descriptor.input_op_label, operations, by_reference)
    terminals = _walk_to_input(
        op,
        initial,
        descriptor.input_op_label,
        ancestry,
        by_reference,
        True,
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


def _validate_descriptor_for_query(descriptor: ReceptiveField) -> None:
    """Reject descriptors whose derived grid cannot support a geometric unit query."""
    if descriptor.axes is None or descriptor.status in {
        ReceptiveFieldStatus.DATA_DEPENDENT,
        ReceptiveFieldStatus.UNKNOWN,
        ReceptiveFieldStatus.UNSUPPORTED,
    }:
        raise ReceptiveFieldError("Per-unit geometry is unavailable; use .gradient() instead.")
    if not descriptor.layout.windowed_axes:
        raise ReceptiveFieldError("No windowed axes are available; use .gradient() instead.")
    if any(kind == "unknown" for kind in descriptor.layout.axis_kinds):
        raise ReceptiveFieldError("The derived layout is ambiguous; use .gradient() instead.")


def _normalize_unit(op: Op, descriptor: ReceptiveField, unit: Sequence[int]) -> tuple[int, ...]:
    """Validate or resolve target windowed-axis coordinates."""
    assert descriptor.axes is not None
    output_axes = tuple(
        cast(int, axis.output_axis) for axis in descriptor.axes if axis.kind == "windowed"
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
    op: Op, descriptor: ReceptiveField, coordinates: tuple[int, ...]
) -> _AxisSets:
    """Seed singleton sets on the target axes named by the derived layout."""
    assert descriptor.axes is not None
    result: list[_IndexSet | None] = [None] * len(op.shape)
    for axis, coordinate in zip(
        (item for item in descriptor.axes if item.kind == "windowed"), coordinates, strict=True
    ):
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


def _walk_to_input(
    op: Op,
    output_sets: _AxisSets,
    input_label: str,
    ancestry: frozenset[str],
    by_reference: Mapping[str, Op],
    exact: bool,
) -> tuple[_TerminalState, ...]:
    """Recursively map a joint axis-set state to one requested model input."""
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
        bounded = (
            parent_sets
            if parent.label == input_label
            else tuple(
                None if item is None else item.clipped(int(parent.shape[axis]))
                for axis, item in enumerate(parent_sets)
            )
        )
        terminals.extend(
            _walk_to_input(
                parent,
                bounded,
                input_label,
                ancestry,
                by_reference,
                exact and hop_exact and all(item is None or item.exact for item in parent_sets),
            )
        )
    return tuple(terminals)


def _map_to_parent(
    op: Op, parent: Op, output_sets: _AxisSets, result: _RuleResult
) -> tuple[_AxisSets, bool]:
    """Apply one local rule result to one parent branch."""
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
        return _map_axis_mapping(parent, output_sets, result), True
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
    """Map a standard affine window while retaining dilation residue structure."""
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


def _map_axis_mapping(parent: Op, output_sets: _AxisSets, result: _RuleResult) -> _AxisSets:
    """Apply an explicit output-axis to parent-axis mapping."""
    raw = result.values.get("out_to_parent_axis", {})
    if not isinstance(raw, Mapping):
        return (None,) * len(parent.shape)
    mapped: list[_IndexSet | None] = [None] * len(parent.shape)
    for output_axis, parent_axis in raw.items():
        if isinstance(output_axis, int) and isinstance(parent_axis, int):
            if 0 <= output_axis < len(output_sets) and 0 <= parent_axis < len(mapped):
                mapped[parent_axis] = output_sets[output_axis]
    return tuple(mapped)


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


def _call_interval_callback(
    callback: Any, axis: int, output_set: _IndexSet
) -> tuple[_IndexSet, bool]:
    """Lift a monotone interval callback across each progression hull."""
    results: list[_IndexSet] = []
    exact = output_set.exact
    for progression in output_set.progressions:
        starts = progression.values() if progression.step > 1 else (progression.start,)
        for start in starts:
            stop = start if progression.step > 1 else progression.stop
            raw, interval_exact = callback(axis, (start, stop))
            results.append(_coerce_index_set(raw, exact=bool(interval_exact)))
            exact = exact and bool(interval_exact)
    combined = _IndexSet.union(results)
    return _IndexSet(combined.progressions, combined.exact and exact), exact


def _coerce_index_set(raw: object, *, exact: bool) -> _IndexSet:
    """Normalize callback values into the walk's bounded set representation."""
    if isinstance(raw, _IndexSet):
        return _IndexSet(raw.progressions, raw.exact and exact)
    if isinstance(raw, range):
        return _IndexSet.from_values(raw, exact=exact)
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        return _IndexSet.from_values((int(value) for value in raw), exact=exact)
    raise TypeError("map_index_set/map_interval callbacks must return an integer index set")


def map_transposed_convolution_index_set(
    axis: int,
    output_set: _IndexSet,
    *,
    kernel: int | Sequence[int],
    stride: int | Sequence[int],
    padding: int | Sequence[int],
    dilation: int | Sequence[int],
    input_extent: int | Sequence[int],
) -> tuple[_IndexSet, bool]:
    """Map transposed-convolution outputs by authoritative feasible-tap enumeration.

    The gcd test is only a necessary pre-filter. Every surviving output/tap pair must
    also produce an integral, in-range input index.
    """
    kernels = _parameter_tuple(kernel)
    strides = _parameter_tuple(stride, len(kernels))
    paddings = _parameter_tuple(padding, len(kernels))
    dilations = _parameter_tuple(dilation, len(kernels))
    extents = _parameter_tuple(input_extent, len(kernels))
    kernel_value = kernels[axis]
    stride_value = strides[axis]
    padding_value = paddings[axis]
    dilation_value = dilations[axis]
    extent = extents[axis]
    feasible: list[int] = []
    divisor = gcd(stride_value, dilation_value)
    for output_index in output_set.values():
        shifted = output_index + padding_value
        if shifted % divisor:
            continue
        for tap in range(kernel_value):
            numerator = shifted - dilation_value * tap
            if numerator % stride_value:
                continue
            input_index = numerator // stride_value
            if 0 <= input_index < extent:
                feasible.append(input_index)
    return _IndexSet.from_values(feasible, exact=output_set.exact), True


def map_adaptive_pool_index_set(
    axis: int,
    output_set: _IndexSet,
    *,
    input_extent: int | Sequence[int],
    output_extent: int | Sequence[int],
) -> tuple[_IndexSet, bool]:
    """Map adaptive-pooling bins exactly using floor/ceil endpoint rules."""
    inputs = _parameter_tuple(input_extent)
    outputs = _parameter_tuple(output_extent, len(inputs))
    values = (
        input_index
        for output_index in output_set.values()
        for input_index in range(
            floor(Fraction(output_index * inputs[axis], outputs[axis])),
            ceil(Fraction((output_index + 1) * inputs[axis], outputs[axis])),
        )
    )
    return _IndexSet.from_values(values, exact=output_set.exact), True


def map_interpolation_index_set(
    axis: int,
    output_set: _IndexSet,
    *,
    mode: str,
    input_extent: int | Sequence[int],
    output_extent: int | Sequence[int],
    align_corners: bool | None = None,
    scale_factor: int | float | Fraction | Sequence[int | float | Fraction] | None = None,
    recompute_scale_factor: bool | None = None,
) -> tuple[_IndexSet, bool]:
    """Map supported interpolation modes with branch-correct rational source indices."""
    inputs = _parameter_tuple(input_extent)
    outputs = _parameter_tuple(output_extent, len(inputs))
    scales = _fraction_tuple(scale_factor, len(inputs)) if scale_factor is not None else None
    source_scale = (
        Fraction(inputs[axis], outputs[axis])
        if scales is None or recompute_scale_factor is True
        else Fraction(1, 1) / scales[axis]
    )
    linear_modes = {"linear", "bilinear", "trilinear", "bicubic"}
    nearest_modes = {"nearest", "nearest-exact"}
    if mode not in linear_modes | nearest_modes:
        return _IndexSet.interval(0, inputs[axis] - 1, exact=False), False
    support: list[int] = []
    exact = output_set.exact
    for output_index in output_set.values():
        candidates: tuple[int, ...]
        source = _interpolation_source(
            output_index,
            inputs[axis],
            outputs[axis],
            align_corners=align_corners,
            supplied_scale=None if scales is None else scales[axis],
            recompute_scale_factor=recompute_scale_factor,
        )
        if mode == "nearest":
            phase = Fraction(output_index) * source_scale
            candidates = (floor(phase),)
        elif mode == "nearest-exact":
            phase = (Fraction(output_index) + Fraction(1, 2)) * source_scale
            candidates = (floor(phase),)
        elif source.denominator == 1:
            phase = source
            candidates = (int(source),)
        elif mode == "bicubic":
            phase = source
            base = floor(source)
            candidates = (base - 1, base, base + 1, base + 2)
        else:
            phase = source
            candidates = (floor(source), ceil(source))
        phase_float = float(phase)
        boundary = round(phase_float)
        if phase.denominator != 1 and abs(phase_float - boundary) <= ulp(phase_float):
            radius = 2 if mode == "bicubic" else 1
            candidates += tuple(range(boundary - radius, boundary + radius + 1))
            exact = False
        support.extend(min(max(candidate, 0), inputs[axis] - 1) for candidate in candidates)
    return _IndexSet.from_values(support, exact=exact), exact


def _interpolation_source(
    output_index: int,
    input_extent: int,
    output_extent: int,
    *,
    align_corners: bool | None,
    supplied_scale: Fraction | None,
    recompute_scale_factor: bool | None,
) -> Fraction:
    """Return the branch-correct linear-family source coordinate."""
    if align_corners:
        if output_extent == 1:
            return Fraction(0)
        return Fraction(output_index * (input_extent - 1), output_extent - 1)
    scale = (
        Fraction(input_extent, output_extent)
        if supplied_scale is None or recompute_scale_factor is True
        else Fraction(1, 1) / supplied_scale
    )
    return (Fraction(output_index) + Fraction(1, 2)) * scale - Fraction(1, 2)


def _parameter_tuple(value: int | Sequence[int], rank: int | None = None) -> tuple[int, ...]:
    """Normalize callback integer parameters without accepting strings."""
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        result = tuple(int(item) for item in value)
    else:
        result = (int(cast(Any, value)),)
    if rank is not None and len(result) == 1:
        result *= rank
    if rank is not None and len(result) != rank:
        raise ValueError("callback parameter rank mismatch")
    return result


def _fraction_tuple(
    value: int | float | Fraction | Sequence[int | float | Fraction], rank: int
) -> tuple[Fraction, ...]:
    """Normalize supplied interpolation scales using exact rational conversion."""
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        result = tuple(Fraction(item) for item in value)
    else:
        result = (Fraction(cast(Any, value)),) * rank
    if len(result) != rank:
        raise ValueError("scale_factor rank mismatch")
    return result


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
            for item in (terminal.clipped[axis] for axis in descriptor.layout.windowed_axes)
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
            box_axes.append(
                ReceptiveFieldBoxAxis(axis, "pointwise", None, None, None, None, None, None)
            )
        elif axis_descriptor.kind == "full":
            box_axes.append(ReceptiveFieldBoxAxis(axis, "full", None, None, 0, extent, 0, extent))
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
                )
            )
        else:
            box_axes.append(
                ReceptiveFieldBoxAxis(axis, "unknown", None, None, None, None, None, None)
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
