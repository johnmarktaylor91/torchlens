"""Target-anchored projective-field descriptor solver."""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from types import MappingProxyType
from typing import TYPE_CHECKING

from ._engine import (
    _concatenation_offsets,
    _geometry_memo,
    _graph_revision,
    _merge_states,
    _operation_is_live,
    _passthrough_axis_map,
    _rule_result,
    _topological_operations,
)
from ._engine_descriptor import _descriptor
from ._engine_geometry import (
    _Affine,
    _as_tuple,
    _AxisState,
    _compose,
    _Dissolved,
    _Full,
    _identity_map,
    _InputState,
    _Mapped,
    _select_full_axes,
    _transpose_mapped,
)
from ._errors import ReceptiveFieldConfigurationError
from ._path import ancestor_labels, resolve_graph_point
from ._rules import _rf_rules_epoch, _RuleResult
from ._types import ReceptiveField, ReceptiveFieldDirection, ReceptiveFieldStatus

if TYPE_CHECKING:
    from ..data_classes.op import Op
    from ..data_classes.trace import Trace


_TARGET_SOLUTION_CACHE_SIZE = 8


@dataclass(frozen=True)
class _ProjectiveFieldSolution:
    """Immutable target-anchored projective descriptor solution."""

    registry_epoch: int
    graph_revision: tuple[object, ...]
    target_labels: tuple[str, ...]
    descriptors: Mapping[tuple[str, str], ReceptiveField]
    per_op: Mapping[str, Mapping[str, ReceptiveField]]
    states: Mapping[tuple[str, str], _InputState]


def solve_projective(trace: Trace, target_ops: Iterable[Op | str]) -> _ProjectiveFieldSolution:
    """Solve projective descriptors from every target ancestor into target space.

    Parameters
    ----------
    trace:
        Captured TorchLens trace.
    target_ops:
        Non-empty iterable of target operations or exact pass-qualified labels.

    Returns
    -------
    _ProjectiveFieldSolution
        Immutable solution containing one descriptor per source and reachable target.

    Raises
    ------
    ValueError
        If no target is supplied or target result keys are ambiguous.
    ReceptiveFieldError
        If a target does not resolve in ``trace``.
    """

    targets = _canonical_targets(trace, target_ops)
    target_labels = tuple(target.label for target in targets)
    epoch = _rf_rules_epoch()
    revision = _graph_revision(trace)
    caches = trace.__dict__.get("_rf_directional_solutions")
    if not isinstance(caches, dict):
        caches = {}
        trace.__dict__["_rf_directional_solutions"] = caches
    cache = caches.get("target")
    if not isinstance(cache, OrderedDict):
        cache = OrderedDict()
        caches["target"] = cache

    cached = cache.get(target_labels)
    if cached is not None:
        cached_epoch, cached_revision, cached_solution = cached
        if cached_epoch == epoch and cached_revision == revision:
            cache.move_to_end(target_labels)
            return cached_solution

    with _geometry_memo():
        solution = _solve_projective_uncached(trace, targets, epoch, revision)
    cache[target_labels] = (epoch, revision, solution)
    cache.move_to_end(target_labels)
    while len(cache) > _TARGET_SOLUTION_CACHE_SIZE:
        cache.popitem(last=False)
    return solution


def _canonical_targets(trace: Trace, target_ops: Iterable[Op | str]) -> tuple[Op, ...]:
    """Resolve, deduplicate, and trace-order one target set."""

    resolved = {resolve_graph_point(trace, target).label for target in target_ops}
    if not resolved:
        raise ReceptiveFieldConfigurationError(
            "Projective descriptor solving requires at least one target operation."
        )
    targets = tuple(
        op for op in trace.layer_list if op.label in resolved and _operation_is_live(op)
    )
    keys = [target.io_role or target.label for target in targets]
    if len(keys) != len(set(keys)):
        raise ReceptiveFieldConfigurationError("Projective targets must have distinct result keys.")
    return targets


def _solve_projective_uncached(
    trace: Trace,
    targets: tuple[Op, ...],
    epoch: int,
    graph_revision: tuple[object, ...],
) -> _ProjectiveFieldSolution:
    """Run the reverse dynamic program for one canonical target set."""

    operations = tuple(op for op in trace.layer_list if _operation_is_live(op))
    by_reference = {
        reference: op
        for op in operations
        for reference in (op.label, op.layer_label, op._layer_label_raw)
    }
    active_labels = frozenset(
        label for target in targets for label in ancestor_labels(trace, target)
    )
    target_by_label = {target.label: target for target in targets}
    states_by_op: dict[str, dict[str, _InputState]] = {}

    for op in reversed(_topological_operations(operations, by_reference)):
        if op.label not in active_labels:
            continue
        branches: dict[str, list[_InputState]] = defaultdict(list)
        target = target_by_label.get(op.label)
        if target is not None:
            seed = _seed_target(target)
            branches[seed.io_role].append(seed)

        children = tuple(
            by_reference[reference]
            for reference in op.children
            if reference in by_reference and by_reference[reference].label in active_labels
        )
        # This reverse program visits each edge from the PARENT side, so one
        # child's local rule and concatenation layout are re-requested once per
        # parent -- quadratic on a wide fan-in concatenation, since each miss is
        # itself linear in the child's parents. The enclosing ``_geometry_memo``
        # scope opened by ``solve_projective`` collapses both to once per child.
        for child in children:
            result, rule_name = _rule_result(child)
            for role, child_state in states_by_op.get(child.label, {}).items():
                branches[role].append(_transpose_rule(child, op, child_state, result, rule_name))

        states_by_op[op.label] = {
            role: _merge_states(op, role_states, "projective_union")
            for role, role_states in branches.items()
        }

    descriptors: dict[tuple[str, str], ReceptiveField] = {}
    per_op: dict[str, Mapping[str, ReceptiveField]] = {}
    flattened_states: dict[tuple[str, str], _InputState] = {}
    for op in operations:
        op_descriptors: dict[str, ReceptiveField] = {}
        for role, state in states_by_op.get(op.label, {}).items():
            descriptor = replace(
                _descriptor(op, state),
                direction=ReceptiveFieldDirection.PROJECTIVE,
                # ``op.shape`` is legitimately ``None`` for a non-tensor-valued
                # op (e.g. a JAX region boundary/projection pseudo-op whose
                # captured output is a tuple of values, not one tensor) that
                # is merely a pass-through ancestor on the walk to a real
                # geometric target. Treat it as the op.py-documented
                # shapeless default (``()``, zero spatial axes) instead of
                # crashing on ``tuple(None)``.
                unit_shape=tuple(op.shape) if op.shape is not None else (),
            )
            descriptors[(op.label, role)] = descriptor
            flattened_states[(op.label, role)] = state
            op_descriptors[role] = descriptor
        per_op[op.label] = MappingProxyType(op_descriptors)

    return _ProjectiveFieldSolution(
        registry_epoch=epoch,
        graph_revision=graph_revision,
        target_labels=tuple(target.label for target in targets),
        descriptors=MappingProxyType(descriptors),
        per_op=MappingProxyType(per_op),
        states=MappingProxyType(flattened_states),
    )


def _seed_target(op: Op) -> _InputState:
    """Create identity geometry at one projective result target."""

    shape = tuple(op.shape)
    role = op.io_role or op.label
    identity = _identity_map()
    return _InputState(
        input_op_label=op.label,
        io_role=role,
        input_shape=shape,
        axes=tuple(_AxisState(identity, axis, "unknown", op.label) for axis in range(len(shape))),
        taint=None,
        notes=(),
        rule="target_identity",
    )


def _transpose_rule(
    child: Op,
    parent: Op,
    state: _InputState,
    result: _RuleResult,
    rule_name: str,
) -> _InputState:
    """Transpose one child's local rule onto one parent branch."""

    notes = state.notes + ((f"{child.label}: {result.note}",) if result.note else ())
    if result.kind in {"data_dependent", "unknown", "unsupported"}:
        status = {
            "data_dependent": ReceptiveFieldStatus.DATA_DEPENDENT,
            "unknown": ReceptiveFieldStatus.UNKNOWN,
            "unsupported": ReceptiveFieldStatus.UNSUPPORTED,
        }[result.kind]
        if result.kind == "data_dependent":
            notes += (f"{child.label}: data-dependent forward routing has no static transpose",)
        return replace(state, axes=None, taint=status, notes=notes, rule=rule_name)
    if state.taint is not None or state.axes is None:
        return replace(state, notes=notes, rule=rule_name)
    if result.kind == "window":
        return _transpose_window(child, parent, state, result, rule_name, notes)
    if result.kind == "window_edges":
        return _transpose_window_edges(child, parent, state, result, rule_name, notes)
    if result.kind == "full":
        return _transpose_full(child, parent, state, result, rule_name, notes)
    if result.kind == "axis_map":
        return _transpose_axis_map(child, parent, state, result, rule_name, notes)
    if result.kind == "dissolve":
        axes = tuple(
            replace(
                axis,
                geometry=_Dissolved(),
                output_axis=None,
                kind="unknown",
                provenance=child.label,
            )
            for axis in state.axes
        )
        return replace(state, axes=axes, notes=notes, rule=rule_name)
    if result.kind == "piecewise":
        degradation = (
            f"{child.label}: transposed piecewise relation widened to the whole target extent"
        )
        axes = tuple(
            replace(
                axis,
                geometry=_Full(exact=False),
                output_axis=None,
                kind="full",
                provenance=child.label,
            )
            for axis in state.axes
        )
        return replace(state, axes=axes, notes=notes + (degradation,), rule=rule_name)
    return _transpose_passthrough(child, parent, state, result, rule_name, notes)


def _transpose_passthrough(
    child: Op,
    parent: Op,
    state: _InputState,
    result: _RuleResult,
    rule_name: str,
    notes: tuple[str, ...],
    *,
    child_to_parent: Mapping[int, int] | None = None,
) -> _InputState:
    """Transpose identity and broadcast relations with explicit axis alignment."""

    assert state.axes is not None
    if child_to_parent is None:
        parent_to_child = _passthrough_axis_map(child, parent, result)
        if parent_to_child is not None:
            child_to_parent = {
                child_axis: parent_axis for parent_axis, child_axis in parent_to_child.items()
            }
    if child_to_parent is None:
        return replace(
            state,
            axes=None,
            taint=ReceptiveFieldStatus.UNKNOWN,
            notes=notes + (f"{child.label}: rank-changing passthrough lacks an explicit axis map",),
            rule=rule_name,
        )
    concat_axis = result.values.get("concatenate_axis")
    concat_offsets = _concatenation_offsets(child, parent, result)
    axes: list[_AxisState] = []
    for axis in state.axes:
        geometry = axis.geometry
        child_axis = axis.output_axis
        if isinstance(geometry, _Dissolved):
            axes.append(
                replace(
                    axis,
                    geometry=_Full(exact=False),
                    output_axis=None,
                    kind="full",
                    provenance=child.label,
                )
            )
            continue
        if child_axis is None or isinstance(geometry, _Full):
            axes.append(axis)
            continue
        parent_axis = child_to_parent.get(child_axis)
        if parent_axis is None or parent_axis < 0 or parent_axis >= len(parent.shape):
            axes.append(
                replace(
                    axis,
                    geometry=_Full(exact=True),
                    output_axis=None,
                    kind="full",
                    provenance=child.label,
                )
            )
            continue
        broadcast = parent.shape[parent_axis] == 1 and child.shape[child_axis] != 1
        if broadcast:
            axes.append(
                replace(
                    axis,
                    geometry=_Full(exact=True),
                    output_axis=None,
                    kind="full",
                    provenance=child.label,
                )
            )
            continue
        local = _identity_map()
        if child_axis == concat_axis and concat_offsets:
            local = _Mapped(
                _Affine(Fraction(1), Fraction(min(concat_offsets))),
                _Affine(Fraction(1), Fraction(max(concat_offsets))),
                exact=len(concat_offsets) == 1 and axis.kind != "windowed",
                aligned=len(concat_offsets) == 1,
                sparse=len(concat_offsets) > 1,
            )
        composed = _compose(geometry, local)
        if isinstance(concat_axis, int) and isinstance(composed, _Mapped):
            composed = replace(composed, exact=False)
        axes.append(
            replace(
                axis,
                geometry=composed,
                output_axis=parent_axis,
                kind=axis.kind if axis.kind != "unknown" else "pointwise",
                provenance=child.label if axis.kind == "unknown" else axis.provenance,
            )
        )
    return replace(state, axes=tuple(axes), notes=notes, rule=rule_name)


def _transpose_window(
    child: Op,
    parent: Op,
    state: _InputState,
    result: _RuleResult,
    rule_name: str,
    notes: tuple[str, ...],
) -> _InputState:
    """Transpose a standard kernel/stride/padding/dilation recurrence."""

    kernels = _as_tuple(result.values["kernel"])
    rank = len(kernels)
    strides = _as_tuple(result.values.get("stride", 1), rank)
    paddings = _as_tuple(result.values.get("padding", 0), rank)
    dilations = _as_tuple(result.values.get("dilation", 1), rank)
    exact = bool(result.values.get("exact", True))
    local_maps = tuple(
        _Mapped(
            _Affine(Fraction(stride), Fraction(-padding)),
            _Affine(Fraction(stride), Fraction(-padding + dilation * (kernel - 1))),
            exact=exact,
            sparse=dilation > 1,
        )
        for kernel, stride, padding, dilation in zip(
            kernels, strides, paddings, dilations, strict=True
        )
    )
    return _transpose_window_maps(
        child,
        parent,
        state,
        local_maps,
        rule_name,
        notes,
        channel_dependency=str(result.values.get("channel_dependency", "full_exact")),
    )


def _transpose_window_edges(
    child: Op,
    parent: Op,
    state: _InputState,
    result: _RuleResult,
    rule_name: str,
    notes: tuple[str, ...],
) -> _InputState:
    """Transpose registered raw two-edge maps or taint malformed metadata."""

    raw_edges = result.values.get("per_axis_edges")
    if not isinstance(raw_edges, Sequence) or isinstance(raw_edges, (str, bytes)):
        return _malformed_window_state(child, state, rule_name, notes)
    exact = bool(result.values.get("exact", False))
    maps: list[_Mapped] = []
    for raw_axis in raw_edges:
        try:
            lo, hi = raw_axis
            maps.append(
                _Mapped(
                    _Affine(Fraction(lo[0]), Fraction(lo[1])),
                    _Affine(Fraction(hi[0]), Fraction(hi[1])),
                    exact=exact,
                )
            )
        except (IndexError, TypeError, ValueError, ZeroDivisionError):
            return _malformed_window_state(child, state, rule_name, notes)
    return _transpose_window_maps(
        child,
        parent,
        state,
        tuple(maps),
        rule_name,
        notes,
        preserve_non_window_axes=bool(result.values.get("preserve_non_window_axes", False)),
    )


def _malformed_window_state(
    child: Op,
    state: _InputState,
    rule_name: str,
    notes: tuple[str, ...],
) -> _InputState:
    """Return an UNKNOWN state for malformed transposed window metadata."""

    return replace(
        state,
        axes=None,
        taint=ReceptiveFieldStatus.UNKNOWN,
        notes=notes + (f"{child.label}: malformed window_edges rule result",),
        rule=rule_name,
    )


def _transpose_window_maps(
    child: Op,
    parent: Op,
    state: _InputState,
    local_maps: tuple[_Mapped, ...],
    rule_name: str,
    notes: tuple[str, ...],
    *,
    preserve_non_window_axes: bool = False,
    channel_dependency: str = "full_exact",
) -> _InputState:
    """Transpose local window maps and mirror semantic axis inference."""

    assert state.axes is not None
    spatial_rank = len(local_maps)
    parent_rank = len(parent.shape)
    child_rank = len(child.shape)
    if spatial_rank == 0 or parent_rank < spatial_rank or child_rank < spatial_rank:
        return replace(
            state,
            axes=None,
            taint=ReceptiveFieldStatus.UNKNOWN,
            notes=notes + (f"{child.label}: window rank is inconsistent with captured shapes",),
            rule=rule_name,
        )
    parent_spatial_start = parent_rank - spatial_rank
    child_spatial_start = child_rank - spatial_rank
    child_channel_axis = child_spatial_start - 1
    parent_channel_axis = parent_spatial_start - 1
    transposed = tuple(
        _transpose_mapped(mapping, int(child.shape[child_spatial_start + index]))
        for index, mapping in enumerate(local_maps)
    )
    axes: list[_AxisState] = []
    for axis in state.axes:
        geometry = axis.geometry
        child_axis = axis.output_axis
        if child_axis is None or isinstance(geometry, (_Full, _Dissolved)):
            axes.append(axis)
        elif child_axis >= child_spatial_start:
            local_index = child_axis - child_spatial_start
            axes.append(
                replace(
                    axis,
                    geometry=_compose(geometry, transposed[local_index]),
                    output_axis=parent_spatial_start + local_index,
                    kind="windowed",
                    provenance=child.label,
                )
            )
        elif (
            child_axis == child_channel_axis
            and parent_channel_axis >= 0
            and not preserve_non_window_axes
            and channel_dependency != "pointwise"
        ):
            axes.append(
                replace(
                    axis,
                    geometry=_Full(exact=channel_dependency == "full_exact"),
                    output_axis=None,
                    kind="full",
                    provenance=child.label,
                )
            )
        else:
            parent_axis = child_axis - (child_rank - parent_rank)
            if parent_axis < 0 or parent_axis >= parent_rank:
                axes.append(
                    replace(
                        axis,
                        geometry=_Full(exact=False),
                        output_axis=None,
                        kind="full",
                        provenance=child.label,
                    )
                )
            else:
                axes.append(
                    replace(
                        axis,
                        geometry=_compose(geometry, _identity_map()),
                        output_axis=parent_axis,
                        kind="pointwise",
                        provenance=child.label,
                    )
                )
    if any(not mapping.exact for mapping in transposed):
        notes += (f"{child.label}: transposed window relation is an upper-bound envelope",)
    batch_axis = state.batch_axis
    if batch_axis is None and parent_channel_axis > 0:
        candidates = [
            index
            for index, axis in enumerate(axes)
            if axis.output_axis is not None
            and axis.output_axis < parent_channel_axis
            and axis.kind == "pointwise"
        ]
        if len(candidates) == 1:
            batch_axis = candidates[0]
    return replace(
        state,
        axes=tuple(axes),
        notes=notes,
        rule=rule_name,
        batch_axis=batch_axis,
    )


def _transpose_full(
    child: Op,
    parent: Op,
    state: _InputState,
    result: _RuleResult,
    rule_name: str,
    notes: tuple[str, ...],
) -> _InputState:
    """Transpose selected whole-parent-axis dependence into target-space fullness."""

    assert state.axes is not None
    selected = _select_full_axes(result.values.get("axes"), parent, child)
    exact = bool(result.values.get("exact", True))
    parent_rank = len(parent.shape)
    child_rank = len(child.shape)
    surviving = result.values.get("surviving_parent_axes")
    child_to_parent: Mapping[int, int] | None = None
    if isinstance(surviving, Sequence) and not isinstance(surviving, (str, bytes)):
        surviving_axes = tuple(int(axis) for axis in surviving)
        if child_rank == parent_rank:
            child_to_parent = {axis: axis for axis in range(child_rank)}
        elif child_rank == len(surviving_axes):
            child_to_parent = dict(enumerate(surviving_axes))
    passthrough = _transpose_passthrough(
        child,
        parent,
        state,
        result,
        rule_name,
        notes,
        child_to_parent=child_to_parent,
    )
    if passthrough.axes is None and all(axis.output_axis is None for axis in state.axes):
        passthrough = replace(state, notes=notes, rule=rule_name)
    elif passthrough.axes is None and selected == set(range(parent_rank)):
        fallback_axes = tuple(
            replace(
                axis,
                geometry=_Full(exact=exact),
                output_axis=None,
                kind="full",
                provenance=child.label,
            )
            for axis in state.axes
        )
        return replace(state, axes=fallback_axes, notes=notes, rule=rule_name)
    if passthrough.axes is None:
        # A partial-axes kind="full" rule transposed onto a rank-mismatched
        # parent (for example the computed-weight branch of F.linear) has no
        # derivable axis map and no explicit surviving_parent_axes obligation.
        # Degrade fail-closed to UNKNOWN instead of crashing a public
        # validation or table query with a bare assertion.
        return replace(
            state,
            axes=None,
            taint=ReceptiveFieldStatus.UNKNOWN,
            notes=notes
            + (f"{child.label}: rank-changing partial-full relation lacks an explicit axis map",),
            rule=rule_name,
        )
    axes: list[_AxisState] = []
    batch_axis = passthrough.batch_axis
    for axis_index, (old_axis, mapped_axis) in enumerate(
        zip(state.axes, passthrough.axes, strict=True)
    ):
        child_axis = old_axis.output_axis
        parent_axis = (
            None
            if child_axis is None
            else child_to_parent.get(child_axis)
            if child_to_parent
            else child_axis
        )
        if parent_axis in selected:
            axes.append(
                replace(
                    mapped_axis,
                    geometry=_Full(exact=exact),
                    output_axis=None,
                    kind="full",
                    provenance=child.label,
                )
            )
            if parent_axis == 0 and len(parent.shape) > 1:
                batch_axis = axis_index
        else:
            axes.append(mapped_axis)
    if not exact:
        notes = passthrough.notes + (
            f"{child.label}: transposed whole-extent relation remains an upper bound",
        )
    else:
        notes = passthrough.notes
    return replace(passthrough, axes=tuple(axes), notes=notes, batch_axis=batch_axis)


def _transpose_axis_map(
    child: Op,
    parent: Op,
    state: _InputState,
    result: _RuleResult,
    rule_name: str,
    notes: tuple[str, ...],
) -> _InputState:
    """Transpose an exact child-axis to parent-axis remapping."""

    raw_mapping = result.values.get("out_to_parent_axis", {})
    if not isinstance(raw_mapping, Mapping):
        return replace(
            state,
            axes=None,
            taint=ReceptiveFieldStatus.UNKNOWN,
            notes=notes + (f"{child.label}: malformed axis_map rule result",),
            rule=rule_name,
        )
    mapping = {int(child_axis): int(parent_axis) for child_axis, parent_axis in raw_mapping.items()}
    raw_edges = result.values.get("out_axis_edges", {})
    out_axis_edges = raw_edges if isinstance(raw_edges, Mapping) else {}
    assert state.axes is not None
    axes: list[_AxisState] = []
    for axis in state.axes:
        if axis.output_axis is None:
            axes.append(axis)
        elif axis.output_axis not in mapping:
            axes.append(
                replace(
                    axis,
                    geometry=_Full(exact=int(child.shape[axis.output_axis]) == 1),
                    output_axis=None,
                    kind="full",
                    provenance=child.label,
                )
            )
        else:
            geometry = axis.geometry
            kind = axis.kind if axis.kind != "unknown" else "pointwise"
            provenance = child.label if axis.kind == "unknown" else axis.provenance
            edge = out_axis_edges.get(axis.output_axis)
            if edge is not None and isinstance(geometry, _Mapped):
                # Walking child->parent, the new frame coordinate ``p`` maps
                # to the old child coordinate ``(p - start) / step``; without
                # this inverse-affine composition every slice offset was
                # dropped from the transposed relation (disputed-r2 b6/R20-2).
                step, start = int(edge[0]), int(edge[1])
                inverse_slope = Fraction(1, step)
                local = _Mapped(
                    _Affine(inverse_slope, Fraction(-start, step)),
                    _Affine(inverse_slope, Fraction(-start, step)),
                    sparse=inverse_slope.denominator != 1,
                )
                geometry = _compose(geometry, local)
                if kind == "pointwise":
                    kind = "windowed"
                    provenance = child.label
            axes.append(
                replace(
                    axis,
                    geometry=geometry,
                    output_axis=mapping[axis.output_axis],
                    kind=kind,
                    provenance=provenance,
                )
            )
    return _degrade_for_selected_axes(child, parent, state, result, rule_name, notes, axes)


def _degrade_for_selected_axes(
    child: Op,
    parent: Op,
    state: _InputState,
    result: _RuleResult,
    rule_name: str,
    notes: tuple[str, ...],
    axes: list[_AxisState],
) -> _InputState:
    """Widen a transposed axis-map state whose scalar-select drops real extent.

    A scalar-selected parent axis of extent > 1 makes the whole transposed
    image CONDITIONAL on the source's dropped coordinate: sources at the
    recorded index project as mapped, every other source projects NOWHERE.
    The descriptor lattice has no per-axis slot for the dropped parent axis,
    so the sound presentation is an upper-bound envelope — every surviving
    claim loses exactness. When the dropped axis is the parent's leading
    axis (the batch heuristic shared with :func:`_transpose_full`) the
    select also rebases batch indices across the boundary, so the known
    target batch axis widens to a whole-extent full claim, DECLARING the
    coupling geometrically instead of serving an identity pointwise claim
    that misses the true mapping (disputed-r3 F2).
    """

    raw_selected = result.values.get("selected_parent_axes", ())
    selected = (
        tuple(int(axis) for axis in raw_selected)
        if isinstance(raw_selected, Sequence) and not isinstance(raw_selected, (str, bytes))
        else ()
    )
    oversized = tuple(
        axis for axis in selected if 0 <= axis < len(parent.shape) and int(parent.shape[axis]) > 1
    )
    if not oversized:
        return replace(state, axes=tuple(axes), notes=notes, rule=rule_name)
    widened: list[_AxisState] = []
    for index, axis in enumerate(axes):
        if (
            index == state.batch_axis
            and 0 in oversized
            and len(parent.shape) > 1
            and not isinstance(axis.geometry, _Dissolved)
        ):
            widened.append(
                replace(
                    axis,
                    geometry=_Full(exact=False),
                    output_axis=None,
                    kind="full",
                    provenance=child.label,
                )
            )
        elif isinstance(axis.geometry, (_Mapped, _Full)):
            widened.append(replace(axis, geometry=replace(axis.geometry, exact=False)))
        else:
            widened.append(axis)
    notes += (
        f"{child.label}: a scalar-selected parent axis restricts which sources project;"
        " unselected sources project nowhere, so these bounds are an upper-bound envelope",
    )
    return replace(state, axes=tuple(widened), notes=notes, rule=rule_name)


__all__: list[str] = []
