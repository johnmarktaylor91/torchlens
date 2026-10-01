"""Whole-graph geometric receptive-field solver.

The engine deliberately contains no operation-family registrations.  It interprets the
local geometry emitted by :mod:`torchlens.receptive_field._rules` and composes that geometry
over the captured DAG using exact rational arithmetic.
"""

from __future__ import annotations

import threading
from collections import OrderedDict, defaultdict, deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from types import MappingProxyType
from typing import TYPE_CHECKING

import torch

from ..backends import TORCH_BACKEND_NAME
from ..capture.arg_positions import (
    VARIADIC_TENSOR_ARG_FUNCS,
    _normalize_func_name,
    _schema_arg_is_parent_candidate,
)
from ._engine_descriptor import _descriptor
from ._engine_geometry import (
    _Affine,
    _as_tuple,
    _AxisState,
    _compose,
    _constant_map,
    _Dissolved,
    _endpoint_chord,
    _Full,
    _identity_map,
    _InputState,
    _join_axis_kinds,
    _Mapped,
    _select_full_axes,
    _unique_notes,
)
from ._errors import ReceptiveFieldConfigurationError
from ._rules import _RF_RULES, ReceptiveFieldRuleContext, _rf_rules_epoch, _RuleResult
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


@dataclass(frozen=True)
class _BranchState:
    """Geometry state plus whether ancestry reaches only through a control edge."""

    state: _InputState
    geometry_neutral: bool


@dataclass(frozen=True)
class _SchemaOperandSlots:
    """ATen schema slots partitioned into data operands and metadata-only positions."""

    operand_positions: frozenset[int]
    operand_names: frozenset[str]
    seen_positions: frozenset[int]
    seen_names: frozenset[str]
    int_list_positions: frozenset[int]


# Positive entries only (bounded by the aten operator universe); misses live
# in the bounded FIFO companion below so an unrecognized func name (loaded
# traces, custom ops, non-aten labels) neither grows this dict without bound
# NOR re-enters the C++ operator registry on every edge of every solve
# (~12x slower than a cached read, measured grind-p3).
_SCHEMA_OPERAND_SLOTS_CACHE: dict[str, _SchemaOperandSlots] = {}
_SCHEMA_OPERAND_MISS_NAMES: dict[str, None] = {}
_SCHEMA_OPERAND_MISS_NAMES_MAX_ENTRIES = 1024
_SCHEMA_METADATA_ONLY_BASE_TYPES = frozenset(
    {
        "int",
        "SymInt",
        "bool",
        "str",
        "Device",
        "Generator",
        "AnyEnumType",
        "ScalarType",
        "Layout",
        "MemoryFormat",
        "Dimname",
        "QScheme",
        "Stream",
    }
)


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

    epoch = _rf_rules_epoch()
    graph_revision = _graph_revision(trace)
    cached = trace.__dict__.get("_receptive_field_solution")
    if (
        isinstance(cached, _ReceptiveFieldSolution)
        and cached.registry_epoch == epoch
        and cached.graph_revision == graph_revision
    ):
        return cached

    with _geometry_memo():
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
        raise ReceptiveFieldConfigurationError(
            "Receptive-field source operation does not belong to the supplied trace."
        )

    epoch = _rf_rules_epoch()
    graph_revision = _graph_revision(trace)
    caches = trace.__dict__.get("_rf_directional_solutions")
    if not isinstance(caches, dict):
        caches = {}
        trace.__dict__["_rf_directional_solutions"] = caches
    cache = caches.get("source")
    if not isinstance(cache, OrderedDict):
        cache = OrderedDict()
        caches["source"] = cache

    cached = cache.get(source.label)
    if cached is not None:
        cached_epoch, cached_revision, cached_solution = cached
        if cached_epoch == epoch and cached_revision == graph_revision:
            cache.move_to_end(source.label)
            return cached_solution

    with _geometry_memo():
        solution = _solve_uncached(
            trace, epoch, graph_revision, lambda op: op.label == source.label
        )
    cache[source.label] = (epoch, graph_revision, solution)
    cache.move_to_end(source.label)
    while len(cache) > _SOURCE_SOLUTION_CACHE_SIZE:
        cache.popitem(last=False)
    return solution


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
    states_by_op: dict[str, dict[str, _BranchState]] = {}

    for op in _topological_operations(operations, by_reference):
        if seed_predicate(op):
            seed = _seed_input(op)
            states_by_op[op.label] = {
                seed.io_role: _BranchState(state=seed, geometry_neutral=False)
            }
            continue

        parents = tuple(by_reference[label] for label in op.parents if label in by_reference)
        parent_states = tuple(states_by_op.get(parent.label, {}) for parent in parents)
        result, rule_name = _rule_result(op)
        # Group edge-use records once per operation, and classify each parent edge
        # once rather than once per (parent, io_role): both are functions of the
        # (op, parent) pair alone, so a wide fan-in operation paid O(parents x
        # records x roles) for an answer that does not vary with the role.
        records_by_parent_label = _edge_records_by_parent_label(op) if parents else {}
        branches: dict[str, list[_BranchState]] = defaultdict(list)
        for parent, states in zip(parents, parent_states):
            if not states:
                continue
            edge_neutral = _edge_is_geometry_neutral(op, parent, records_by_parent_label)
            for role, branch in states.items():
                branches[role].append(
                    _BranchState(
                        state=_apply_rule(op, parent, branch.state, result, rule_name),
                        geometry_neutral=(branch.geometry_neutral or edge_neutral),
                    )
                )

        states_by_op[op.label] = {
            role: _merge_branch_states(op, role_states, rule_name)
            for role, role_states in branches.items()
        }

    descriptors: dict[tuple[str, str], ReceptiveField] = {}
    per_op: dict[str, Mapping[str, ReceptiveField]] = {}
    flattened_states: dict[tuple[str, str], _InputState] = {}
    for op in operations:
        op_branches = states_by_op.get(op.label, {})
        if op_branches and all(branch.geometry_neutral for branch in op_branches.values()):
            continue
        op_descriptors: dict[str, ReceptiveField] = {}
        for role, branch in op_branches.items():
            if branch.geometry_neutral:
                continue
            state = branch.state
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


class _GeometryMemo:
    """Per-traversal memo for the per-operation facts a walk re-derives.

    A per-unit walk enumerates paths, so it revisits one operation once per path
    through it, and each visit re-derives facts that depend on the operation
    alone. Two of them are linear in the operation's parent count, which makes a
    wide fan-in concatenation (DenseNet, dense multi-branch blocks) quadratic per
    visit:

    ``rule_results``
        ``_rule_result``. Both solvers already reuse one result object across
        every branch of one operation, so reusing it across repeated visits is
        the same sharing, not new sharing: ``_RuleResult`` is frozen, its
        ``values`` mapping is immutable, and its optional callbacks are pure
        functions over immutable captured configuration. A miss re-derives
        ``in_shapes`` through ``Op.input_activations``, which resolves every
        parent through the trace accessor.
    ``concatenation_starts``
        The per-position concatenation slice starts, which
        ``_concatenation_offsets`` otherwise rebuilds from ``op.input_shapes``
        once per *parent* of every visited concatenation.

    Both are keyed by ``id(op)`` and validated against the retained operation
    object, so a recycled id can never alias. ``rule_results`` additionally
    validates ``_rf_rules_epoch()``, so a rule registered mid-traversal is never
    served a stale result.
    """

    __slots__ = ("concatenation_starts", "rule_results")

    def __init__(self) -> None:
        """Create empty memo tables."""

        self.rule_results: dict[int, tuple[Op, int, tuple[_RuleResult, str]]] = {}
        self.concatenation_starts: dict[tuple[int, int], tuple[Op, tuple[int, ...]]] = {}


class _GeometryMemoState(threading.local):
    """Per-thread stack of active geometry memo scopes, innermost last.

    Thread-local rather than a module global: receptive-field queries read an
    already-captured trace, so unlike capture they are plausibly issued from more
    than one thread, and a shared stack's push/pop could interleave into an
    unbalanced state.
    """

    def __init__(self) -> None:
        """Start this thread with no active scope."""

        self.stack: list[_GeometryMemo] = []


_GEOMETRY_MEMO_STATE = _GeometryMemoState()


def _active_geometry_memo() -> _GeometryMemo | None:
    """Return this thread's innermost active memo scope, or ``None``.

    Returns
    -------
    _GeometryMemo | None
        Innermost scope, or ``None`` when no traversal scope is open -- in which
        case every memoized helper falls back to deriving its value.
    """

    stack = _GEOMETRY_MEMO_STATE.stack
    return stack[-1] if stack else None


class _geometry_memo:
    """Scope per-operation geometry memoization to one solve or per-unit walk.

    Reentrant: nested scopes push their own tables, so an inner walk never
    outlives its own memo or reads an outer one.
    """

    def __enter__(self) -> None:
        """Push a fresh memo scope."""

        _GEOMETRY_MEMO_STATE.stack.append(_GeometryMemo())

    def __exit__(self, *exc_info: object) -> None:
        """Pop this memo scope, including on the exception path."""

        _GEOMETRY_MEMO_STATE.stack.pop()


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

    scope = _active_geometry_memo()
    memo = None if scope is None else scope.rule_results
    if memo is not None:
        epoch = _rf_rules_epoch()
        cached = memo.get(id(op))
        if cached is not None and cached[0] is op and cached[1] == epoch:
            return cached[2]
    computed = _rule_result_uncached(op)
    if memo is not None:
        memo[id(op)] = (op, epoch, computed)
    return computed


def _rule_result_uncached(op: Op) -> tuple[_RuleResult, str]:
    """Evaluate one operation's registered local rule, bypassing the memo.

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
    if rule is None:
        # Some preview backends capture a generic implementation-call name as
        # func_name that loses the actual operation identity -- tinygrad
        # reconstructs its graph from the UOp DAG, so an elementwise binary op
        # (add, mul, ...) carries its Python lambda wrapper's own name
        # ("<lambda>") as func_name, with the real semantic category only on
        # layer_type. Fall back to layer_type before reporting unsupported, so
        # RF rules keyed by canonical op name (e.g. "add") still dispatch for
        # those ops; this only ever widens dispatch (torch/mlx/tf/jax/paddle
        # already resolve on the first, unchanged lookup).
        fallback_name = _normalize_func_name(str(getattr(op, "layer_type", "") or ""))
        if fallback_name != name:
            fallback_rule = _RF_RULES.get(fallback_name)
            if fallback_rule is not None:
                rule = fallback_rule
                name = fallback_name
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
        return _apply_axis_map(op, parent, state, result, rule_name, notes)
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
    return _apply_passthrough(op, parent, state, result, rule_name, notes)


def _apply_passthrough(
    op: Op,
    parent: Op,
    state: _InputState,
    result: _RuleResult,
    rule_name: str,
    notes: tuple[str, ...],
    *,
    parent_to_child: Mapping[int, int] | None = None,
) -> _InputState:
    """Compose identity or broadcast geometry using an explicit axis correspondence."""

    assert state.axes is not None
    child_rank = len(op.shape)
    if parent_to_child is None:
        parent_to_child = _passthrough_axis_map(op, parent, result)
    if parent_to_child is None:
        return replace(
            state,
            axes=None,
            taint=ReceptiveFieldStatus.UNKNOWN,
            notes=notes + (f"{op.label}: rank-changing passthrough lacks an explicit axis map",),
            rule=rule_name,
        )
    concat_axis = result.values.get("concatenate_axis")
    concat_offsets = _concatenation_offsets(op, parent, result)
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
        parent_axis = axis.output_axis
        child_axis = parent_to_child.get(parent_axis)
        if child_axis is None or child_axis < 0 or child_axis >= child_rank:
            axes.append(replace(axis, geometry=_Full(exact=False), output_axis=None, kind="full"))
            continue
        broadcast = parent.shape[parent_axis] == 1 and op.shape[child_axis] != 1
        if broadcast:
            local = _constant_map()
        elif child_axis == concat_axis and concat_offsets:
            starts = tuple(-offset for offset in concat_offsets)
            local = _Mapped(
                _Affine(Fraction(1), Fraction(min(starts))),
                _Affine(Fraction(1), Fraction(max(starts))),
                exact=len(starts) == 1 and axis.kind != "windowed",
                aligned=len(starts) == 1,
                sparse=len(starts) > 1,
            )
        else:
            local = _identity_map()
        geometry = _compose(axis.geometry, local)
        if isinstance(concat_axis, int) and isinstance(geometry, _Mapped):
            geometry = replace(geometry, exact=False)
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


def _passthrough_axis_map(op: Op, parent: Op, result: _RuleResult) -> Mapping[int, int] | None:
    """Resolve a parent-to-child map for one passthrough-like relation.

    Parameters
    ----------
    op:
        Child operation.
    parent:
        Parent branch being composed.
    result:
        Registered local rule result.

    Returns
    -------
    collections.abc.Mapping[int, int] | None
        Explicit parent-to-child correspondence, or ``None`` when ambiguous.
    """

    parent_rank = len(parent.shape)
    child_rank = len(op.shape)
    raw_axis_maps = result.values.get("parent_to_child_axes")
    if isinstance(raw_axis_maps, Mapping):
        parent_references = (parent.label, parent.layer_label, parent._layer_label_raw)
        raw_axis_map = next(
            (
                raw_axis_maps[reference]
                for reference in parent_references
                if reference in raw_axis_maps
            ),
            raw_axis_maps.get("default"),
        )
        if isinstance(raw_axis_map, Mapping):
            try:
                axis_map = {
                    int(parent_axis): int(child_axis)
                    for parent_axis, child_axis in raw_axis_map.items()
                }
            except (TypeError, ValueError):
                axis_map = {}
            if all(
                0 <= parent_axis < parent_rank and 0 <= child_axis < child_rank
                for parent_axis, child_axis in axis_map.items()
            ) and (axis_map or not raw_axis_map):
                return axis_map
    concat_axis = result.values.get("concatenate_axis")
    if isinstance(concat_axis, int) and bool(result.values.get("stack", False)):
        return {
            parent_axis: parent_axis if parent_axis < concat_axis else parent_axis + 1
            for parent_axis in range(parent_rank)
        }
    if parent_rank == child_rank:
        return {axis: axis for axis in range(parent_rank)}
    if result.values.get("axis_alignment") == "trailing":
        offset = child_rank - parent_rank
        return {axis: axis + offset for axis in range(parent_rank)}
    return None


def _concatenation_offsets(op: Op, parent: Op, result: _RuleResult) -> tuple[int, ...]:
    """Return slice starts for every occurrence of a concatenated parent.

    Parameters
    ----------
    op:
        Concatenation operation.
    parent:
        Parent branch being composed.
    result:
        Concatenation rule result.

    Returns
    -------
    tuple[int, ...]
        Captured child-axis slice starts for the parent occurrence(s).
    """

    raw_axis = result.values.get("concatenate_axis")
    if not isinstance(raw_axis, int):
        return ()
    parent_references = {parent.label, parent.layer_label, parent._layer_label_raw}
    if bool(result.values.get("stack", False)):
        return tuple(index for index, label in enumerate(op.parents) if label in parent_references)
    starts = _concatenation_starts_by_position(op, raw_axis)
    return tuple(start for start, label in zip(starts, op.parents) if label in parent_references)


def _concatenation_starts_by_position(op: Op, raw_axis: int) -> tuple[int, ...]:
    """Return one concatenation's child-axis slice start for every parent position.

    The running offset depends on the operation and axis alone, never on which
    parent is being composed, but ``_concatenation_offsets`` is called once per
    parent -- so recomputing it there made a wide fan-in concatenation resolve
    every parent's activation through the trace accessor once per parent. Under a
    :class:`_geometry_memo` scope this is derived once per operation instead.

    Parameters
    ----------
    op:
        Concatenation operation.
    raw_axis:
        Concatenation axis, already normalized against the output rank.

    Returns
    -------
    tuple[int, ...]
        Slice start at each ``op.parents`` position, truncated to the shorter of
        ``op.parents`` and ``op.input_shapes`` exactly as the per-parent scan was.
    """

    scope = _active_geometry_memo()
    memo = None if scope is None else scope.concatenation_starts
    key = (id(op), raw_axis)
    if memo is not None:
        cached = memo.get(key)
        if cached is not None and cached[0] is op:
            return cached[1]
    positions: list[int] = []
    offset = 0
    for _label, shape in zip(op.parents, op.input_shapes, strict=False):
        positions.append(offset)
        if shape is None:
            continue
        # torch.cat accepts a one-dimensional empty tensor at any concat axis.
        if tuple(shape) == (0,):
            continue
        offset += int(shape[raw_axis])
    starts = tuple(positions)
    if memo is not None:
        memo[key] = (op, starts)
    return starts


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
    return _compose_window_maps(
        op,
        parent,
        state,
        local_maps,
        rule_name,
        notes,
        channel_dependency=str(result.values.get("channel_dependency", "full_exact")),
    )


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
    return _compose_window_maps(
        op,
        parent,
        state,
        tuple(maps),
        rule_name,
        notes,
        preserve_non_window_axes=bool(result.values.get("preserve_non_window_axes", False)),
    )


def _compose_window_maps(
    op: Op,
    parent: Op,
    state: _InputState,
    local_maps: tuple[_Mapped, ...],
    rule_name: str,
    notes: tuple[str, ...],
    *,
    preserve_non_window_axes: bool = False,
    channel_dependency: str = "full_exact",
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
        elif (
            parent_axis == channel_axis
            and not preserve_non_window_axes
            and channel_dependency != "pointwise"
        ):
            axes.append(
                replace(
                    axis,
                    geometry=_Full(exact=channel_dependency == "full_exact"),
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
    parent_rank = len(parent.shape)
    child_rank = len(op.shape)
    surviving = result.values.get("surviving_parent_axes")
    parent_to_child: Mapping[int, int] | None = None
    if isinstance(surviving, Sequence) and not isinstance(surviving, (str, bytes)):
        surviving_axes = tuple(int(axis) for axis in surviving)
        if child_rank == parent_rank:
            parent_to_child = {axis: axis for axis in range(parent_rank)}
        elif child_rank == len(surviving_axes):
            parent_to_child = {
                parent_axis: child_axis for child_axis, parent_axis in enumerate(surviving_axes)
            }
    passthrough = _apply_passthrough(
        op,
        parent,
        state,
        result,
        rule_name,
        notes,
        parent_to_child=parent_to_child,
    )
    if passthrough.axes is None and selected == set(range(parent_rank)):
        fallback_axes = tuple(
            replace(
                axis,
                geometry=_Full(exact=exact),
                output_axis=None,
                kind="full",
                provenance=op.label,
            )
            if axis.output_axis is not None
            else axis
            for axis in state.axes
        )
        return replace(state, axes=fallback_axes, notes=notes, rule=rule_name)
    if passthrough.axes is None:
        # A partial-axes kind="full" rule applied across a rank-mismatched
        # parent (for example an input-derived computed-weight branch) has no
        # derivable axis map and no explicit surviving_parent_axes obligation.
        # Degrade fail-closed to UNKNOWN instead of crashing a public
        # validation or table query with a bare assertion.
        return replace(
            state,
            axes=None,
            taint=ReceptiveFieldStatus.UNKNOWN,
            notes=notes
            + (f"{op.label}: rank-changing partial-full relation lacks an explicit axis map",),
            rule=rule_name,
        )
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
    result: _RuleResult,
    rule_name: str,
    notes: tuple[str, ...],
) -> _InputState:
    """Apply an exact registered output-to-parent axis remapping."""

    raw_mapping = result.values.get("out_to_parent_axis", {})
    if not isinstance(raw_mapping, Mapping):
        return replace(
            state,
            axes=None,
            taint=ReceptiveFieldStatus.UNKNOWN,
            notes=notes + (f"{op.label}: malformed axis_map rule result",),
            rule=rule_name,
        )
    inverse = {int(parent_axis): int(out_axis) for out_axis, parent_axis in raw_mapping.items()}
    selected = result.values.get("selected_parent_axes", ())
    selected_axes = (
        {int(axis) for axis in selected}
        if isinstance(selected, Sequence) and not isinstance(selected, (str, bytes))
        else set()
    )
    raw_edges = result.values.get("out_axis_edges", {})
    out_axis_edges = raw_edges if isinstance(raw_edges, Mapping) else {}
    assert state.axes is not None
    axes = []
    for axis in state.axes:
        if axis.output_axis is None:
            axes.append(axis)
        elif axis.output_axis not in inverse:
            axes.append(
                replace(
                    axis,
                    geometry=_Full(
                        exact=axis.output_axis in selected_axes
                        and int(parent.shape[axis.output_axis]) == 1
                    ),
                    output_axis=None,
                    kind="full",
                )
            )
        else:
            new_output_axis = inverse[axis.output_axis]
            geometry = axis.geometry
            kind = axis.kind if axis.kind != "unknown" else "pointwise"
            provenance = op.label if axis.kind == "unknown" else axis.provenance
            edge = out_axis_edges.get(new_output_axis)
            if edge is not None and isinstance(geometry, _Mapped):
                # A surviving sliced axis carries the exact affine
                # ``parent = step * out + start``; composing it here is what
                # keeps the descriptor's window offsets in the CURRENT frame
                # (the historical rank-changing path dropped the slice offset
                # under an exact claim; disputed-r2 b6/R20-2).
                step, start = int(edge[0]), int(edge[1])
                # A strided slice keeps only every ``step``-th parent index:
                # the true support has holes, so the axis must disclose
                # ``sparse_possible`` (round-3 b6-fable R20-1).
                local = _Mapped(
                    _Affine(Fraction(step), Fraction(start)),
                    _Affine(Fraction(step), Fraction(start)),
                    sparse=abs(step) != 1,
                )
                geometry = _compose(geometry, local)
                if kind == "pointwise":
                    # A non-identity coordinate map is no longer a pointwise
                    # identity; present it as the windowed affine it is.
                    kind = "windowed"
                    provenance = op.label
            axes.append(
                replace(
                    axis,
                    geometry=geometry,
                    output_axis=new_output_axis,
                    kind=kind,
                    provenance=provenance,
                )
            )
    return replace(state, axes=tuple(axes), notes=notes, rule=rule_name)


def _schema_type_base(type_text: str) -> str:
    """Strip optional/list wrappers from a rendered backend schema type."""

    text = type_text.strip()
    while True:
        if text.endswith("?"):
            text = text[:-1].strip()
            continue
        if text.endswith("[]"):
            text = text[:-2].strip()
            continue
        for wrapper in ("Optional[", "List["):
            if text.startswith(wrapper) and text.endswith("]"):
                text = text[len(wrapper) : -1].strip()
                break
        else:
            return text


def _schema_type_is_metadata_only(type_text: str) -> bool:
    """Return whether a schema type can carry only shape/control metadata."""

    return _schema_type_base(type_text) in _SCHEMA_METADATA_ONLY_BASE_TYPES


def _schema_type_is_int_list(type_text: str) -> bool:
    """Return whether a schema type is a variadic integer size list."""

    text = type_text.strip()
    if text.startswith("Optional[") and text.endswith("]"):
        text = text[len("Optional[") : -1].strip()
    if text.endswith("?"):
        text = text[:-1].strip()
    return text in {"List[int]", "List[SymInt]", "int[]", "SymInt[]"}


def _compute_schema_operand_slots(canonical: str) -> _SchemaOperandSlots | None:
    """Derive conservative data-versus-metadata slots from an ATen packet."""

    packet = getattr(torch.ops.aten, canonical, None)
    # ``torch.ops.aten`` is a live namespace INSTANCE: plain Python attribute
    # names ("name", "__module__", ...) resolve to non-packet objects before
    # any operator lookup, and ``.strip("_")`` maps spellings like ``_name_``
    # onto them. Anything without a callable ``overloads`` is "no schema"
    # (fail closed), never a raw AttributeError out of a public engine.
    overloads_method = getattr(packet, "overloads", None)
    if packet is None or not callable(overloads_method):
        return None
    try:
        overload_names = list(overloads_method())
    except Exception:
        return None
    operand_positions: set[int] = set()
    operand_names: set[str] = set()
    seen_positions: set[int] = set()
    seen_names: set[str] = set()
    int_list_positions: set[int] = set()
    found_schema = False
    for overload_name in overload_names:
        schema = getattr(getattr(packet, overload_name, None), "_schema", None)
        if schema is None:
            continue
        found_schema = True
        for index, schema_arg in enumerate(getattr(schema, "arguments", ()) or ()):
            arg_name = getattr(schema_arg, "name", None)
            seen_positions.add(index)
            if isinstance(arg_name, str):
                seen_names.add(arg_name)
            if not _schema_arg_is_parent_candidate(schema_arg):
                continue
            type_text = str(getattr(schema_arg, "type", ""))
            if _schema_type_is_int_list(type_text):
                int_list_positions.add(index)
            if not _schema_type_is_metadata_only(type_text):
                operand_positions.add(index)
                if isinstance(arg_name, str):
                    operand_names.add(arg_name)
    if not found_schema:
        return None
    return _SchemaOperandSlots(
        operand_positions=frozenset(operand_positions),
        operand_names=frozenset(operand_names),
        seen_positions=frozenset(seen_positions),
        seen_names=frozenset(seen_names),
        int_list_positions=frozenset(int_list_positions),
    )


def _schema_edge_is_metadata_only(func_name: str, arg_kind: str, arg_path: object) -> bool:
    """Return whether an edge is schema-proven shape/control metadata, failing closed."""

    if _normalize_func_name(func_name) in VARIADIC_TENSOR_ARG_FUNCS:
        return False
    canonical = func_name.strip("_")
    slots = _SCHEMA_OPERAND_SLOTS_CACHE.get(canonical)
    if slots is None:
        if canonical in _SCHEMA_OPERAND_MISS_NAMES:
            return False
        computed_slots = _compute_schema_operand_slots(canonical)
        if computed_slots is None:
            # Bounded negative cache: torch does not memoize a FAILED
            # ``torch.ops.aten`` lookup, so an uncached miss pays the C++
            # registry round trip per (edge, arg) on every solve.
            while len(_SCHEMA_OPERAND_MISS_NAMES) >= _SCHEMA_OPERAND_MISS_NAMES_MAX_ENTRIES:
                _SCHEMA_OPERAND_MISS_NAMES.pop(next(iter(_SCHEMA_OPERAND_MISS_NAMES)))
            _SCHEMA_OPERAND_MISS_NAMES[canonical] = None
            return False
        _SCHEMA_OPERAND_SLOTS_CACHE[canonical] = computed_slots
        slots = computed_slots
    if not isinstance(arg_path, tuple) or not arg_path:
        return False
    top = arg_path[0]
    if arg_kind == "keyword" and isinstance(top, str):
        normalized = _normalize_func_name(top)
        if any(_normalize_func_name(name) == normalized for name in slots.operand_names):
            return False
        return any(_normalize_func_name(name) == normalized for name in slots.seen_names)
    if arg_kind != "positional" or not isinstance(top, int):
        return False
    if top in slots.operand_positions:
        return False
    if top in slots.seen_positions:
        return True
    return any(position <= top for position in slots.int_list_positions)


def _edge_record_parent_label(record: object) -> str | None:
    """Return one edge record's parent label across current and legacy shapes."""

    label = getattr(record, "parent_label", None)
    if isinstance(label, str):
        return label
    if isinstance(record, tuple) and record and isinstance(record[0], str):
        return record[0]
    return None


def _edge_records_by_parent_label(op: Op) -> dict[str | None, tuple[object, ...]]:
    """Group one operation's edge-use records by their recorded parent label.

    ``_edge_is_geometry_neutral`` otherwise rescans every ``edge_uses`` record
    for every parent, which is quadratic on a wide fan-in operation. Grouping
    once per operation makes each parent's lookup a dict hit. Records with no
    recoverable parent label are grouped under ``None``, exactly reproducing the
    scan's ``None in parent_references`` match for an operation whose parent has
    an unset raw layer label.

    Parameters
    ----------
    op:
        Captured operation whose edge-use records are grouped.

    Returns
    -------
    dict[str | None, tuple[object, ...]]
        Mapping from recorded parent label to that label's records, in
        ``edge_uses`` order.
    """

    grouped: dict[str | None, list[object]] = {}
    for record in getattr(op, "edge_uses", ()):
        grouped.setdefault(_edge_record_parent_label(record), []).append(record)
    return {label: tuple(records) for label, records in grouped.items()}


def _edge_is_geometry_neutral(
    op: Op,
    parent: Op,
    records_by_parent_label: Mapping[str | None, tuple[object, ...]] | None = None,
) -> bool:
    """Return whether one parent reaches ``op`` only through scalar control metadata.

    Parameters
    ----------
    op:
        Captured consuming operation.
    parent:
        Captured parent operation whose edge into ``op`` is classified.
    records_by_parent_label:
        Optional prebuilt grouping from :func:`_edge_records_by_parent_label` for
        ``op``. Supplying it avoids rescanning ``op.edge_uses`` per parent; the
        classification is unchanged because it depends only on the *set* of
        matching records, never on their relative order.

    Returns
    -------
    bool
        Whether the edge carries only scalar shape/control metadata.
    """

    if records_by_parent_label is None:
        records_by_parent_label = _edge_records_by_parent_label(op)
    matched: list[object] = []
    seen_references: set[str | None] = set()
    for reference in (parent.label, parent.layer_label, parent._layer_label_raw):
        if reference in seen_references:
            continue
        seen_references.add(reference)
        matched.extend(records_by_parent_label.get(reference, ()))
    records = tuple(matched)
    if not records:
        return False
    edge_kinds = tuple(getattr(record, "edge_use", None) for record in records)
    if all(kind == "control" for kind in edge_kinds):
        return True
    if tuple(parent.shape) or getattr(op.source_trace, "backend", None) != TORCH_BACKEND_NAME:
        return False
    for record, edge_kind in zip(records, edge_kinds):
        if edge_kind == "control":
            continue
        arg_kind = getattr(record, "arg_kind", None)
        arg_path = getattr(record, "arg_path", None)
        if not isinstance(arg_kind, str) or not _schema_edge_is_metadata_only(
            op.func_name, arg_kind, arg_path
        ):
            return False
    return True


def _merge_branch_states(op: Op, branches: Sequence[_BranchState], rule_name: str) -> _BranchState:
    """Union data-bearing branches without letting scalar control ancestry erase them."""

    if len(branches) == 1:
        return branches[0]

    data_branches = [branch.state for branch in branches if not branch.geometry_neutral]
    geometry_neutral = not data_branches
    if geometry_neutral:
        data_branches = [branch.state for branch in branches]
    elif len(data_branches) == 1:
        state = data_branches[0]
        neutral_note = f"{op.label}: scalar/control-only ancestry contributes no spatial geometry"
        return _BranchState(
            state=replace(
                state,
                notes=_unique_notes(state.notes, (neutral_note,)),
                rule=rule_name,
                merge_seen=True,
            ),
            geometry_neutral=False,
        )

    taints = [branch.taint for branch in data_branches if branch.taint is not None]
    notes = _unique_notes(*(branch.notes for branch in data_branches))
    if taints:
        taint = max(taints, key=lambda status: _TAINT_ORDER[status])
        return _BranchState(
            state=replace(
                data_branches[0],
                axes=None,
                taint=taint,
                notes=notes,
                rule=rule_name,
                merge_seen=True,
            ),
            geometry_neutral=geometry_neutral,
        )
    if any(branch.axes is None for branch in data_branches):
        return _BranchState(
            state=replace(
                data_branches[0],
                axes=None,
                taint=ReceptiveFieldStatus.UNKNOWN,
                notes=notes,
                rule=rule_name,
                merge_seen=True,
            ),
            geometry_neutral=geometry_neutral,
        )

    axis_count = len(data_branches[0].input_shape)
    merged_axes: list[_AxisState] = []
    merge_notes: list[str] = []
    for axis_index in range(axis_count):
        axis_branches = [
            branch.axes[axis_index] for branch in data_branches if branch.axes is not None
        ]
        merged_axis, note = _merge_axes(op, axis_branches)
        merged_axes.append(merged_axis)
        if note is not None:
            merge_notes.append(note)
    batch_axes = {branch.batch_axis for branch in data_branches}
    batch_axis = batch_axes.pop() if len(batch_axes) == 1 else None
    return _BranchState(
        state=replace(
            data_branches[0],
            axes=tuple(merged_axes),
            taint=None,
            notes=notes + tuple(merge_notes),
            rule=rule_name,
            batch_axis=batch_axis,
            merge_seen=True,
        ),
        geometry_neutral=geometry_neutral,
    )


def _merge_states(op: Op, branches: Sequence[_InputState], rule_name: str) -> _InputState:
    """Union ordinary geometry branches for receptive and projective solvers."""

    wrapped = tuple(_BranchState(state=branch, geometry_neutral=False) for branch in branches)
    return _merge_branch_states(op, wrapped, rule_name).state


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


def _snapshot_value(value: object) -> str:
    """Return a mutation-proof, type-tagged token for one recorded argument value."""

    try:
        return f"{type(value).__name__}:{value!r}"
    except Exception:
        # Best-effort fallback: identity keeps the token stable for the same
        # object and still invalidates when the object is replaced outright.
        return f"{type(value).__name__}:@{id(value):x}"


def _snapshot_mapping(mapping: object) -> object:
    """Return an order-insensitive by-value token for one recorded mapping field."""

    if isinstance(mapping, Mapping):
        return tuple(sorted((str(key), _snapshot_value(value)) for key, value in mapping.items()))
    return _snapshot_value(mapping)


def _geometry_args_snapshot(op: Op) -> tuple[object, ...]:
    """Snapshot the recorded operation arguments that RF rules consume, by value.

    Geometry rules read kernel/stride/padding/dilation/output-size/etc. from
    ``op.func_config`` (``RuleContext.cfg``) with fallbacks into the captured
    non-tensor arguments (``RuleContext.arg``). The staleness fingerprint must
    capture BOTH surfaces by value: holding live references would compare a
    mutated dict against itself and never invalidate, so an in-place geometry
    change that preserves shape and topology would keep serving stale frozen
    descriptors from the trace-level solution cache while per-unit ``.at()``
    queries recompute fresh from the live values -- the two public RF paths
    would disagree.
    """

    return (
        _snapshot_mapping(getattr(op, "func_config", None)),
        tuple(_snapshot_value(value) for value in getattr(op, "non_tensor_pos_args", ()) or ()),
        _snapshot_mapping(getattr(op, "non_tensor_kwargs", None)),
    )


def _edge_uses_snapshot(op: Op) -> tuple[str, ...]:
    """Snapshot semantic edge records consumed by control-branch classification."""

    return tuple(_snapshot_value(record) for record in getattr(op, "edge_uses", ()) or ())


def _graph_revision(trace: Trace) -> tuple[object, ...]:
    """Build a stable structural fingerprint that invalidates cache after graph edits.

    Includes topology (parents/children), shape, role, function identity, and a
    by-value snapshot of the geometry arguments RF rules read, so an argument
    change with unchanged shape+topology still bumps the revision.

    ``op.shape`` is legitimately ``None`` for a non-tensor-valued op -- e.g. a
    JAX region boundary/projection pseudo-op, whose captured ``output`` is a
    tuple of values rather than one tensor (``_tensor_ref`` reports
    ``shape=None`` for exactly this "not a tensor" case). ``None`` is itself a
    stable, hashable fingerprint component, so it is included as-is instead of
    calling ``tuple()`` on it.
    """

    return tuple(
        (
            op.label,
            tuple(op.parents),
            tuple(op.children),
            tuple(op.shape) if op.shape is not None else None,
            op.io_role,
            op.func_name,
            _geometry_args_snapshot(op),
            _edge_uses_snapshot(op),
        )
        for op in trace.layer_list
        if _operation_is_live(op)
    )


def _operation_is_live(op: Op) -> bool:
    """Return whether an operation has not been cleared by trace removal."""

    return isinstance(getattr(op, "label", None), str)


__all__: list[str] = []
