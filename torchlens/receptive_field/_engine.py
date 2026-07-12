"""Whole-graph geometric receptive-field solver.

The engine deliberately contains no operation-family registrations.  It interprets the
local geometry emitted by :mod:`torchlens.receptive_field._rules` and composes that geometry
over the captured DAG using exact rational arithmetic.
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict, deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from types import MappingProxyType
from typing import TYPE_CHECKING

from .._io import FieldPolicy
from ..capture.arg_positions import _normalize_func_name
from ._engine_descriptor import _descriptor
from ._engine_geometry import (
    _Affine,
    _AxisState,
    _Dissolved,
    _Full,
    _InputState,
    _Mapped,
    _as_tuple,
    _compose,
    _constant_map,
    _endpoint_chord,
    _identity_map,
    _join_axis_kinds,
    _select_full_axes,
    _unique_notes,
)
from ._rules import ReceptiveFieldRuleContext, _RF_RULES, _RuleResult, _rf_rules_epoch
from ._types import (
    ReceptiveField,
    ReceptiveFieldStatus,
)


if TYPE_CHECKING:
    from ..data_classes.op import Op
    from ..data_classes.trace import Trace


_TAINT_ORDER = {
    ReceptiveFieldStatus.UNKNOWN: 0,
    ReceptiveFieldStatus.DATA_DEPENDENT: 1,
    ReceptiveFieldStatus.UNSUPPORTED: 2,
}

_SOURCE_SOLUTION_CACHE_SIZE = 8


@dataclass(frozen=True)
class _ReceptiveFieldSolution:
    """Immutable whole-trace geometric solution and its cache identity."""

    registry_epoch: int
    graph_revision: tuple[object, ...]
    descriptors: Mapping[tuple[str, str], ReceptiveField]
    per_op: Mapping[str, Mapping[str, ReceptiveField]]
    states: Mapping[tuple[str, str], _InputState]


def _is_input_seed(op: Op) -> bool:
    """Return whether an operation is a default model-input seed."""

    return op.is_input


def solve(trace: Trace) -> _ReceptiveFieldSolution:
    """Solve geometric receptive fields for every operation and reachable model input.

    Parameters
    ----------
    trace:
        Captured TorchLens trace.

    Returns
    -------
    _ReceptiveFieldSolution
        Trace-wide immutable solution. Repeated calls return the trace-owned cached object
        while both the rule-registry epoch and structural graph revision remain unchanged.
    """

    _ensure_trace_cache_policy(trace)
    epoch = _rf_rules_epoch()
    graph_revision = _graph_revision(trace)
    cached = trace.__dict__.get("_receptive_field_solution")
    if (
        isinstance(cached, _ReceptiveFieldSolution)
        and cached.registry_epoch == epoch
        and cached.graph_revision == graph_revision
    ):
        return cached

    solution = _solve_uncached(trace, epoch, graph_revision, _is_input_seed)
    trace.__dict__["_receptive_field_solution"] = solution
    return solution


def solve_from(trace: Trace, source: Op) -> _ReceptiveFieldSolution:
    """Solve geometric receptive fields seeded at one captured source operation.

    Parameters
    ----------
    trace:
        Captured TorchLens trace.
    source:
        Operation whose output grid is the receptive-field source.

    Returns
    -------
    _ReceptiveFieldSolution
        Immutable solution containing descriptors for the source and its descendants.

    Raises
    ------
    ValueError
        If ``source`` does not belong to ``trace``.
    """

    if source.source_trace is not trace or not _operation_is_live(source):
        raise ValueError("Receptive-field source operation does not belong to the supplied trace.")

    _ensure_trace_cache_policy(trace)
    epoch = _rf_rules_epoch()
    graph_revision = _graph_revision(trace)
    cache = trace.__dict__.get("_rf_source_solutions")
    if not isinstance(cache, OrderedDict):
        cache = OrderedDict()
        trace.__dict__["_rf_source_solutions"] = cache

    cached = cache.get(source.label)
    if cached is not None:
        cached_epoch, cached_revision, cached_solution = cached
        if cached_epoch == epoch and cached_revision == graph_revision:
            cache.move_to_end(source.label)
            return cached_solution

    solution = _solve_uncached(trace, epoch, graph_revision, lambda op: op.label == source.label)
    cache[source.label] = (epoch, graph_revision, solution)
    cache.move_to_end(source.label)
    while len(cache) > _SOURCE_SOLUTION_CACHE_SIZE:
        cache.popitem(last=False)
    return solution


def _ensure_trace_cache_policy(trace: Trace) -> None:
    """Declare the dynamically owned trace cache as runtime-only portable state.

    Parameters
    ----------
    trace:
        Trace class whose portable state policy owns the runtime cache.
    """

    type(trace).PORTABLE_STATE_SPEC.setdefault("_receptive_field_solution", FieldPolicy.DROP)
    type(trace).PORTABLE_STATE_SPEC.setdefault("_rf_source_solutions", FieldPolicy.DROP)


def lookup(trace: Trace, op: Op | str) -> Mapping[str, ReceptiveField]:
    """Look up all per-input descriptors for one operation.

    Parameters
    ----------
    trace:
        Trace owning the operation.
    op:
        Operation object or its exact pass-qualified label.

    Returns
    -------
    collections.abc.Mapping
        Insertion-ordered mapping from model-input ``io_role`` to descriptor.
    """

    label = op if isinstance(op, str) else op.label
    return solve(trace).per_op.get(label, MappingProxyType({}))


def _solve_uncached(
    trace: Trace,
    epoch: int,
    graph_revision: tuple[object, ...],
    seed_predicate: Callable[[Op], bool] = _is_input_seed,
) -> _ReceptiveFieldSolution:
    """Compute a fresh trace-wide solution.

    Parameters
    ----------
    trace:
        Captured trace.
    epoch:
        Registry epoch recorded on the solution.
    graph_revision:
        Structural graph fingerprint recorded on the solution.
    seed_predicate:
        Predicate selecting operations that terminate the reverse solve. The default selects
        model-input operations and preserves the input receptive-field behavior.

    Returns
    -------
    _ReceptiveFieldSolution
        Newly computed immutable solution.
    """

    operations = tuple(op for op in trace.layer_list if _operation_is_live(op))
    by_reference = {
        reference: op
        for op in operations
        for reference in (op.label, op.layer_label, op._layer_label_raw)
    }
    states_by_op: dict[str, dict[str, _InputState]] = {}

    for op in _topological_operations(operations, by_reference):
        if seed_predicate(op):
            seed = _seed_input(op)
            states_by_op[op.label] = {seed.io_role: seed}
            continue

        parents = tuple(by_reference[label] for label in op.parents if label in by_reference)
        parent_states = tuple(states_by_op.get(parent.label, {}) for parent in parents)
        result, rule_name = _rule_result(op)
        branches: dict[str, list[_InputState]] = defaultdict(list)
        for parent, states in zip(parents, parent_states):
            for role, state in states.items():
                branches[role].append(_apply_rule(op, parent, state, result, rule_name))

        states_by_op[op.label] = {
            role: _merge_states(op, role_states, rule_name)
            for role, role_states in branches.items()
        }

    descriptors: dict[tuple[str, str], ReceptiveField] = {}
    per_op: dict[str, Mapping[str, ReceptiveField]] = {}
    flattened_states: dict[tuple[str, str], _InputState] = {}
    for op in operations:
        op_descriptors: dict[str, ReceptiveField] = {}
        for role, state in states_by_op.get(op.label, {}).items():
            descriptor = _descriptor(op, state)
            descriptors[(op.label, role)] = descriptor
            flattened_states[(op.label, role)] = state
            op_descriptors[role] = descriptor
        per_op[op.label] = MappingProxyType(op_descriptors)

    return _ReceptiveFieldSolution(
        registry_epoch=epoch,
        graph_revision=graph_revision,
        descriptors=MappingProxyType(descriptors),
        per_op=MappingProxyType(per_op),
        states=MappingProxyType(flattened_states),
    )


def _seed_input(op: Op) -> _InputState:
    """Create identity geometry for a model-input operation.

    Parameters
    ----------
    op:
        Captured model-input operation.

    Returns
    -------
    _InputState
        Unclassified identity seed.
    """

    shape = tuple(op.shape)
    role = op.io_role or op.label
    identity = _Mapped(_Affine(Fraction(1), Fraction(0)), _Affine(Fraction(1), Fraction(0)))
    return _InputState(
        input_op_label=op.label,
        io_role=role,
        input_shape=shape,
        axes=tuple(_AxisState(identity, axis, "unknown", op.label) for axis in range(len(shape))),
        taint=None,
        notes=(),
        rule="input_identity",
    )


def _rule_result(op: Op) -> tuple[_RuleResult, str]:
    """Evaluate the registered local rule for one operation.

    Parameters
    ----------
    op:
        Captured operation.

    Returns
    -------
    tuple[_RuleResult, str]
        Opaque local result and normalized rule name.
    """

    if op.func_name in {None, "none"}:
        return ReceptiveFieldRuleContext(op).passthrough(), "graph_identity"
    name = _normalize_func_name(op.func_name)
    rule = _RF_RULES.get(name)
    context = ReceptiveFieldRuleContext(op)
    if rule is None:
        return context.unsupported(f"{op.label}: no receptive-field rule for {name}"), name
    return rule(context), name


def _apply_rule(
    op: Op,
    parent: Op,
    state: _InputState,
    result: _RuleResult,
    rule_name: str,
) -> _InputState:
    """Apply one local rule to one parent branch state.

    Parameters
    ----------
    op:
        Child operation.
    parent:
        Parent operation for this branch.
    state:
        Parent's state for one model input.
    result:
        Registered local rule result.
    rule_name:
        Normalized operation name.

    Returns
    -------
    _InputState
        Child-coordinate branch state.
    """

    notes = state.notes + ((f"{op.label}: {result.note}",) if result.note else ())
    if result.kind in {"data_dependent", "unknown", "unsupported"}:
        status = {
            "data_dependent": ReceptiveFieldStatus.DATA_DEPENDENT,
            "unknown": ReceptiveFieldStatus.UNKNOWN,
            "unsupported": ReceptiveFieldStatus.UNSUPPORTED,
        }[result.kind]
        return replace(state, axes=None, taint=status, notes=notes, rule=rule_name)

    if state.taint is not None:
        if result.kind == "full" and _select_full_axes(
            result.values.get("axes"), parent, op
        ) == set(range(len(parent.shape))):
            recovered = tuple(
                _AxisState(_Full(exact=False), None, "full", op.label) for _ in state.input_shape
            )
            recovery_note = (
                f"{op.label}: whole-extent rule recovered {state.taint.value} ancestry as an "
                "upper bound"
            )
            return replace(
                state,
                axes=recovered,
                taint=None,
                notes=notes + (recovery_note,),
                rule=rule_name,
            )
        return replace(state, notes=notes, rule=rule_name)

    if state.axes is None:
        return replace(state, notes=notes, rule=rule_name)
    if result.kind == "window":
        return _apply_window(op, parent, state, result, rule_name, notes)
    if result.kind == "window_edges":
        return _apply_window_edges(op, parent, state, result, rule_name, notes)
    if result.kind == "full":
        return _apply_full(op, parent, state, result, rule_name, notes)
    if result.kind == "axis_map":
        mapping = result.values.get("out_to_parent_axis", {})
        return _apply_axis_map(op, parent, state, mapping, rule_name, notes)
    if result.kind == "dissolve":
        axes = tuple(
            replace(
                axis, geometry=_Dissolved(), output_axis=None, kind="unknown", provenance=op.label
            )
            for axis in state.axes
        )
        return replace(state, axes=axes, notes=notes, rule=rule_name)
    if result.kind == "piecewise":
        degradation = f"{op.label}: piecewise descriptor collapsed to a whole-extent upper bound"
        axes = tuple(
            replace(
                axis,
                geometry=_Full(exact=False),
                output_axis=None,
                kind="full",
                provenance=op.label,
            )
            for axis in state.axes
        )
        return replace(state, axes=axes, notes=notes + (degradation,), rule=rule_name)
    return _apply_passthrough(op, parent, state, rule_name, notes)


def _apply_passthrough(
    op: Op,
    parent: Op,
    state: _InputState,
    rule_name: str,
    notes: tuple[str, ...],
) -> _InputState:
    """Compose identity or broadcast geometry for a parent branch."""

    assert state.axes is not None
    parent_rank = len(parent.shape)
    child_rank = len(op.shape)
    offset = child_rank - parent_rank
    axes: list[_AxisState] = []
    for axis in state.axes:
        if isinstance(axis.geometry, _Dissolved):
            axes.append(
                replace(axis, geometry=_Full(exact=False), kind="full", provenance=op.label)
            )
            continue
        if axis.output_axis is None or isinstance(axis.geometry, _Full):
            axes.append(axis)
            continue
        child_axis = axis.output_axis + offset
        if child_axis < 0 or child_axis >= child_rank:
            axes.append(replace(axis, geometry=_Full(exact=False), output_axis=None, kind="full"))
            continue
        broadcast = parent.shape[axis.output_axis] == 1 and op.shape[child_axis] != 1
        local = _constant_map() if broadcast else _identity_map()
        geometry = _compose(axis.geometry, local)
        kind = axis.kind if axis.kind != "unknown" else "pointwise"
        axes.append(
            replace(
                axis,
                geometry=geometry,
                output_axis=child_axis,
                kind=kind,
                provenance=op.label if axis.kind == "unknown" else axis.provenance,
            )
        )
    return replace(state, axes=tuple(axes), notes=notes, rule=rule_name)


def _apply_window(
    op: Op,
    parent: Op,
    state: _InputState,
    result: _RuleResult,
    rule_name: str,
    notes: tuple[str, ...],
) -> _InputState:
    """Compose a standard kernel/stride/padding/dilation local recurrence."""

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
        for kernel, stride, padding, dilation in zip(kernels, strides, paddings, dilations)
    )
    return _compose_window_maps(op, parent, state, local_maps, rule_name, notes)


def _apply_window_edges(
    op: Op,
    parent: Op,
    state: _InputState,
    result: _RuleResult,
    rule_name: str,
    notes: tuple[str, ...],
) -> _InputState:
    """Compose raw registered two-edge maps."""

    raw_edges = result.values.get("per_axis_edges")
    if not isinstance(raw_edges, Sequence) or isinstance(raw_edges, (str, bytes)):
        return replace(
            state,
            axes=None,
            taint=ReceptiveFieldStatus.UNKNOWN,
            notes=notes + (f"{op.label}: malformed window_edges rule result",),
            rule=rule_name,
        )
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
            return replace(
                state,
                axes=None,
                taint=ReceptiveFieldStatus.UNKNOWN,
                notes=notes + (f"{op.label}: malformed window_edges rule result",),
                rule=rule_name,
            )
    return _compose_window_maps(op, parent, state, tuple(maps), rule_name, notes)


def _compose_window_maps(
    op: Op,
    parent: Op,
    state: _InputState,
    local_maps: tuple[_Mapped, ...],
    rule_name: str,
    notes: tuple[str, ...],
) -> _InputState:
    """Compose window maps and derive axis roles from their registered semantics."""

    assert state.axes is not None
    spatial_rank = len(local_maps)
    parent_rank = len(parent.shape)
    child_rank = len(op.shape)
    if spatial_rank == 0 or parent_rank < spatial_rank or child_rank < spatial_rank:
        return replace(
            state,
            axes=None,
            taint=ReceptiveFieldStatus.UNKNOWN,
            notes=notes + (f"{op.label}: window rank is inconsistent with captured shapes",),
            rule=rule_name,
        )
    parent_spatial_start = parent_rank - spatial_rank
    child_spatial_start = child_rank - spatial_rank
    channel_axis = parent_spatial_start - 1
    axes: list[_AxisState] = []
    for axis in state.axes:
        geometry = axis.geometry
        parent_axis = axis.output_axis
        if parent_axis is None or isinstance(geometry, (_Full, _Dissolved)):
            axes.append(axis)
        elif parent_axis >= parent_spatial_start:
            local_index = parent_axis - parent_spatial_start
            axes.append(
                replace(
                    axis,
                    geometry=_compose(geometry, local_maps[local_index]),
                    output_axis=child_spatial_start + local_index,
                    kind="windowed",
                    provenance=op.label,
                )
            )
        elif parent_axis == channel_axis:
            axes.append(
                replace(
                    axis,
                    geometry=_Full(exact=True),
                    output_axis=None,
                    kind="full",
                    provenance=op.label,
                )
            )
        else:
            child_axis = parent_axis + (child_rank - parent_rank)
            axes.append(
                replace(
                    axis,
                    geometry=_compose(geometry, _identity_map()),
                    output_axis=child_axis,
                    kind="pointwise",
                    provenance=op.label,
                )
            )
    batch_axis = state.batch_axis
    if batch_axis is None and channel_axis > 0:
        candidates = [
            index
            for index, axis in enumerate(axes)
            if axis.output_axis is not None
            and axis.output_axis < channel_axis
            and axis.kind == "pointwise"
        ]
        if len(candidates) == 1:
            batch_axis = candidates[0]
    return replace(state, axes=tuple(axes), notes=notes, rule=rule_name, batch_axis=batch_axis)


def _apply_full(
    op: Op,
    parent: Op,
    state: _InputState,
    result: _RuleResult,
    rule_name: str,
    notes: tuple[str, ...],
) -> _InputState:
    """Apply whole-extent dependence on selected parent axes."""

    assert state.axes is not None
    selected = _select_full_axes(result.values.get("axes"), parent, op)
    exact = bool(result.values.get("exact", True))
    passthrough = _apply_passthrough(op, parent, state, rule_name, notes)
    assert passthrough.axes is not None
    axes = []
    for old_axis, mapped_axis in zip(state.axes, passthrough.axes):
        if old_axis.output_axis in selected:
            axes.append(
                replace(
                    mapped_axis,
                    geometry=_Full(exact=exact),
                    output_axis=None,
                    kind="full",
                    provenance=op.label,
                )
            )
        else:
            axes.append(mapped_axis)
    return replace(passthrough, axes=tuple(axes))


def _apply_axis_map(
    op: Op,
    parent: Op,
    state: _InputState,
    raw_mapping: object,
    rule_name: str,
    notes: tuple[str, ...],
) -> _InputState:
    """Apply an exact registered output-to-parent axis remapping."""

    if not isinstance(raw_mapping, Mapping):
        return replace(
            state,
            axes=None,
            taint=ReceptiveFieldStatus.UNKNOWN,
            notes=notes + (f"{op.label}: malformed axis_map rule result",),
            rule=rule_name,
        )
    inverse = {int(parent_axis): int(out_axis) for out_axis, parent_axis in raw_mapping.items()}
    assert state.axes is not None
    axes = []
    for axis in state.axes:
        if axis.output_axis is None:
            axes.append(axis)
        elif axis.output_axis not in inverse:
            axes.append(replace(axis, geometry=_Full(exact=False), output_axis=None, kind="full"))
        else:
            axes.append(
                replace(
                    axis,
                    output_axis=inverse[axis.output_axis],
                    kind=axis.kind if axis.kind != "unknown" else "pointwise",
                    provenance=op.label if axis.kind == "unknown" else axis.provenance,
                )
            )
    return replace(state, axes=tuple(axes), notes=notes, rule=rule_name)


def _merge_states(op: Op, branches: Sequence[_InputState], rule_name: str) -> _InputState:
    """Union all parent branches for one model input at a merge operation."""

    if len(branches) == 1:
        return branches[0]
    taints = [branch.taint for branch in branches if branch.taint is not None]
    notes = _unique_notes(*(branch.notes for branch in branches))
    if taints:
        taint = max(taints, key=lambda status: _TAINT_ORDER[status])
        return replace(
            branches[0], axes=None, taint=taint, notes=notes, rule=rule_name, merge_seen=True
        )
    if any(branch.axes is None for branch in branches):
        return replace(
            branches[0],
            axes=None,
            taint=ReceptiveFieldStatus.UNKNOWN,
            notes=notes,
            rule=rule_name,
            merge_seen=True,
        )

    axis_count = len(branches[0].input_shape)
    merged_axes: list[_AxisState] = []
    merge_notes: list[str] = []
    for axis_index in range(axis_count):
        axis_branches = [branch.axes[axis_index] for branch in branches if branch.axes is not None]
        merged_axis, note = _merge_axes(op, axis_branches)
        merged_axes.append(merged_axis)
        if note is not None:
            merge_notes.append(note)
    batch_axes = {branch.batch_axis for branch in branches}
    batch_axis = batch_axes.pop() if len(batch_axes) == 1 else None
    return replace(
        branches[0],
        axes=tuple(merged_axes),
        taint=None,
        notes=notes + tuple(merge_notes),
        rule=rule_name,
        batch_axis=batch_axis,
        merge_seen=True,
    )


def _merge_axes(op: Op, branches: Sequence[_AxisState]) -> tuple[_AxisState, str | None]:
    """Union one input axis across affine branches with a sound chord envelope."""

    first = branches[0]
    geometries = [branch.geometry for branch in branches]
    if any(isinstance(geometry, _Full) for geometry in geometries):
        tainted_recovery = any(
            isinstance(geometry, _Full) and not geometry.exact for geometry in geometries
        )
        return (
            replace(
                first,
                geometry=_Full(exact=not tainted_recovery),
                output_axis=None,
                kind="full",
                provenance=op.label,
            ),
            None,
        )
    if any(isinstance(geometry, _Dissolved) for geometry in geometries):
        return (
            replace(
                first,
                geometry=_Full(exact=False),
                output_axis=None,
                kind="full",
                provenance=op.label,
            ),
            f"{op.label}: dissolved merge recovered as a whole-extent upper bound",
        )

    mapped = [geometry for geometry in geometries if isinstance(geometry, _Mapped)]
    output_axes = {branch.output_axis for branch in branches}
    output_axis = output_axes.pop() if len(output_axes) == 1 else None
    if output_axis is None:
        return (
            replace(
                first,
                geometry=_Full(exact=False),
                output_axis=None,
                kind="full",
                provenance=op.label,
            ),
            f"{op.label}: incompatible axis mappings widened to a whole-extent upper bound",
        )

    same_slopes = len({(item.lo.a, item.hi.a) for item in mapped}) == 1
    centers = {(item.lo.b + item.hi.b) / 2 for item in mapped}
    aligned = same_slopes and len(centers) == 1
    if same_slopes:
        lo = _Affine(mapped[0].lo.a, min(item.lo.b for item in mapped))
        hi = _Affine(mapped[0].hi.a, max(item.hi.b for item in mapped))
    else:
        extent = int(op.shape[output_axis])
        lo = _endpoint_chord(mapped, extent, lower=True)
        hi = _endpoint_chord(mapped, extent, lower=False)
    exact = all(item.exact for item in mapped) and aligned
    geometry = _Mapped(
        lo,
        hi,
        exact=exact,
        aligned=aligned and all(item.aligned for item in mapped),
        sparse=any(item.sparse for item in mapped) or not aligned,
    )
    note = None
    if not aligned:
        note = f"{op.label}: branch grids are misaligned; using a sound endpoint-chord envelope"
    kind = _join_axis_kinds(branches)
    return replace(first, geometry=geometry, output_axis=output_axis, kind=kind), note


def _topological_operations(
    operations: tuple[Op, ...], by_reference: Mapping[str, Op]
) -> tuple[Op, ...]:
    """Return stable forward topological order, appending disconnected components."""

    order = {op.label: index for index, op in enumerate(operations)}
    indegree = {op.label: sum(parent in by_reference for parent in op.parents) for op in operations}
    children: dict[str, list[str]] = defaultdict(list)
    for op in operations:
        for parent in op.parents:
            if parent in by_reference:
                children[by_reference[parent].label].append(op.label)
    queue = deque(op.label for op in operations if indegree[op.label] == 0)
    result: list[Op] = []
    while queue:
        label = queue.popleft()
        result.append(by_reference[label])
        for child in sorted(children[label], key=order.__getitem__):
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    seen = {op.label for op in result}
    result.extend(op for op in operations if op.label not in seen)
    return tuple(result)


def _graph_revision(trace: Trace) -> tuple[object, ...]:
    """Build a stable structural fingerprint that invalidates cache after graph edits."""

    return tuple(
        (
            op.label,
            tuple(op.parents),
            tuple(op.children),
            tuple(op.shape),
            op.io_role,
            op.func_name,
        )
        for op in trace.layer_list
        if _operation_is_live(op)
    )


def _operation_is_live(op: Op) -> bool:
    """Return whether an operation has not been cleared by trace removal."""

    return isinstance(getattr(op, "label", None), str)


__all__: list[str] = []
