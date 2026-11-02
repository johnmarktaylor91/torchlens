"""Index-set seeded geometric queries: one walk for a whole block of units.

The per-unit entry points (:func:`~torchlens.receptive_field._query.box_for_unit`
and :func:`~torchlens.receptive_field._forward_query.box_for_source_unit`) seed
one coordinate over the windowed axes. This module is their SET sibling: the
seed is one ``_IndexSet`` (or ``None``) per output axis of the seed operation,
so an axis-aligned block of units walks the executed DAG once, with the same
rules, the same memo discipline and the same hull builder. The per-unit
functions are untouched; their behaviour is pinned by the field tests.

The sibling exists for the unit-level influence engine
(:mod:`torchlens.receptive_field._influence`), which decomposes a selection mask
into rectangles and folds one set query per rectangle back into a mask.
Nothing here is public.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, cast

from . import _engine
from ._engine_forward import _ProjectiveFieldSolution
from ._errors import ReceptiveFieldError
from ._forward_query import _labels_reaching, _select_target_descriptor, _walk_to_target
from ._query import (
    _ancestor_labels,
    _AxisSets,
    _build_box,
    _IndexSet,
    _query_graph,
    _query_memo_scope,
    _select_descriptor,
    _shared_reach,
    _TerminalState,
    _union_optional,
    _validate_descriptor_for_query,
    _walk_to_input,
    _WalkKey,
    _windowed_axes_in_unit_order,
)
from ._types import ReceptiveField, ReceptiveFieldBox, ReceptiveFieldDirection

if TYPE_CHECKING:
    from .._data_classes.op import Op

_WalkMemo = dict[_WalkKey, tuple[_TerminalState, ...]]


@dataclass(frozen=True)
class _IndexSetBox:
    """Engine answer for an index-set seed: the hull box plus the propagated sets.

    ``box`` is the same three-level hull a per-unit query builds (its ``unit``
    is the seed block's lower corner over the windowed axes). ``clipped_sets``
    is the union over terminal paths of the clipped per-axis sets in the far
    endpoint's index space (``None`` = unconstrained). It keeps a seed that is
    restricted on a pointwise axis (one batch row, one channel) restricted
    there, where the hull's full-slice convention would widen it.
    """

    box: ReceptiveFieldBox
    clipped_sets: _AxisSets


def _seed_axis_sets(op: Op, descriptor: ReceptiveField, seed: _AxisSets) -> _AxisSets:
    """Validate an index-set seed over the seed operation's output axes.

    The set overload of ``_query._initial_axis_sets``: the caller supplies
    one ``_IndexSet`` or ``None`` per output axis. Every windowed axis must
    carry a non-empty in-range set; other axes may be ``None`` (unconstrained,
    the per-unit convention) or a set, which the walk routes exactly as
    ``complete_unit`` coordinates are routed.
    """
    assert descriptor.axes is not None
    if len(seed) != len(op.shape):
        raise ReceptiveFieldError("seed rank does not match the seed operation.")
    windowed = {cast(int, axis.output_axis) for axis in _windowed_axes_in_unit_order(descriptor)}
    for axis, index_set in enumerate(seed):
        if index_set is None:
            if axis in windowed:
                raise ReceptiveFieldError(f"seed leaves windowed output axis {axis} unconstrained.")
            continue
        if index_set.is_empty:
            raise ReceptiveFieldError(f"seed is empty on output axis {axis}.")
        minimum, maximum = index_set.minimum, index_set.maximum
        assert minimum is not None and maximum is not None
        if minimum < 0 or maximum >= int(op.shape[axis]):
            raise ReceptiveFieldError(
                f"seed indices {minimum}..{maximum} are out of bounds for output axis {axis}."
            )
    return tuple(seed)


def _index_set_box(
    op: Op,
    descriptor: ReceptiveField,
    initial: _AxisSets,
    terminals: tuple[_TerminalState, ...],
    *,
    clip: bool,
) -> _IndexSetBox:
    """Fold a set query's terminals into the hull box plus the propagated sets.

    No terminal (no live path from the seed block) is an ``empty`` box, not an
    error: for a set query "nothing reached" is an answer.
    """
    assert descriptor.axes is not None
    corner = tuple(
        cast(int, cast(_IndexSet, initial[cast(int, axis.output_axis)]).minimum)
        for axis in _windowed_axes_in_unit_order(descriptor)
    )
    rank = len(descriptor.input_shape)
    if not terminals:
        terminals = (
            _TerminalState(
                theoretical=(None,) * rank,
                clipped=tuple(_IndexSet.empty() for _ in range(rank)),
                exact=True,
            ),
        )
    box = _build_box(op, descriptor, corner, terminals, clip=clip)
    clipped_sets = tuple(
        _union_optional(terminal.clipped[axis] for terminal in terminals) for axis in range(rank)
    )
    return _IndexSetBox(box=box, clipped_sets=clipped_sets)


def box_for_units(
    solution: _engine._ReceptiveFieldSolution,
    op: Op,
    seed: _AxisSets,
    *,
    input_op: Op | str | None = None,
    source: Op | None = None,
    clip: bool = True,
    memo: _WalkMemo | None = None,
) -> _IndexSetBox:
    """Compute the receptive hull of an axis-aligned block of target units.

    The set sibling of ``_query.box_for_unit``: same solution, same
    descriptor selection and validation, same reverse walk, one seed block
    instead of one coordinate. ``memo`` lets the rectangles of one query share
    the walk memo.
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
    initial = _seed_axis_sets(op, descriptor, seed)
    operations, by_reference = _query_graph(op.source_trace, solution)
    ancestry = _shared_reach(
        "receptive",
        solution,
        descriptor.input_op_label,
        lambda: _ancestor_labels(descriptor.input_op_label, operations, by_reference),
    )
    with _query_memo_scope():
        terminals = _walk_to_input(
            op,
            initial,
            descriptor.input_op_label,
            ancestry,
            by_reference,
            True,
            {} if memo is None else memo,
        )
    return _index_set_box(op, descriptor, initial, terminals, clip=clip)


def box_for_source_units(
    solution: _ProjectiveFieldSolution,
    op: Op,
    seed: _AxisSets,
    *,
    target: Op | str | None = None,
    clip: bool = True,
    memo: _WalkMemo | None = None,
) -> _IndexSetBox:
    """Compute the projective support of an axis-aligned block of source units.

    The set sibling of ``_forward_query.box_for_source_unit``: same
    target-anchored solution, same descriptor selection and validation, same
    forward transposition walk, one seed block instead of one coordinate.
    """
    descriptors = solution.per_op.get(op.label)
    if not descriptors:
        raise ReceptiveFieldError(
            f"No projective-field solution is available from source {op.label!r}."
        )
    descriptor = _select_target_descriptor(descriptors, target)
    _validate_descriptor_for_query(descriptor)
    initial = _seed_axis_sets(op, descriptor, seed)
    operations, by_reference = _query_graph(op.source_trace, solution)
    target_label = descriptor.input_op_label
    reaching = _shared_reach(
        "projective",
        solution,
        target_label,
        lambda: _labels_reaching(target_label, operations, by_reference),
    )
    active = reaching if op.label in reaching else frozenset()
    with _query_memo_scope():
        terminals = _walk_to_target(
            op, initial, target_label, active, by_reference, True, {} if memo is None else memo
        )
    result = _index_set_box(op, descriptor, initial, terminals, clip=clip)
    box = replace(
        result.box, direction=ReceptiveFieldDirection.PROJECTIVE, unit_shape=tuple(op.shape)
    )
    return _IndexSetBox(box=box, clipped_sets=result.clipped_sets)


__all__: list[str] = []
