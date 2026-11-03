"""Unit-level influence as a selection: the ENGINE behind the field views' ``.of()``.

One ACT term, :class:`~torchlens._selection.graph._InfluenceTerm` (built by
``op.projective_field.of(units, within=...)`` and
``op.receptive_field.of(units, within=...)``, or by the private builder
:func:`torchlens._selection.graph._influence`), turns a SET of units (any resolved ACT
selection) into the set of units it can reach (projective) or the set of
units that can reach it (receptive) at a family of target sites. The result
is an ordinary resolved selection, so ``| & - ~``, ``do()`` and
``trace.subgraph`` need no change.

Geometric method (default). The seed mask at each seed site is decomposed
into axis-aligned rectangles (greedy, budget ``_RECTANGLE_BUDGET``; past the
budget the bounding rectangle stands in and the relation drops to
``upper_bound``). Each rectangle is one index-set seed for the field engine's
set sibling (:mod:`~torchlens.receptive_field._set_query`): one walk per
rectangle, the walk memo shared per (seed site, target site) pair. The
resulting hull becomes a mask over the target site; rectangles and seed
sites fold per target through ``_mask_union`` and ``_join_relation``. The
field status folds worst-of in the order EXACT < WHOLE_INPUT < UPPER_BOUND;
``DATA_DEPENDENT``, ``UNKNOWN`` and ``UNSUPPORTED`` descriptors never yield a
geometric mask and refuse typed (``influence_geometry_unavailable``),
pointing at the gradient method.

Gradient method (``method="gradient"``). One autograd probe per
(seed unit, target site) through the shipped per-unit probes, the autograd
graph retained so the trace stays probe-able; supports OR per target site.
Seeds are never summed into one backward (sign cancellation can hide a true
influence). The probe count is bounded by ``probe_budget``
(refusal ``influence_probe_budget``) and stated before any probe runs.

Disclosure. Every entry states ``provenance.basis`` (``geometric`` or
``gradient``) and ``provenance.field_status`` (the folded geometric status,
``empirical`` for probes, ``none`` for an all-empty seed);
``provenance.source`` reads ``influence(<direction>)``, plus the probe count
and ``cross_batch_influence`` for the gradient method. Batch is a pointwise
axis: a seed on one batch row stays on that row unless a batch-coupling op
widens it, and the status says so.

The term and its builder live in :mod:`torchlens._selection.graph` so the field
view can build a query without importing this module; this module registers
the resolver, and the ``receptive_field`` package imports it.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import replace
from typing import Any

import torch

from .._selection.graph import _ExecutedDag, _InfluenceTerm, _resolve_region
from .._selection.mask import _Mask, _mask_empty, _mask_from_dense, _mask_from_slices, _mask_union
from .._selection.selection import (
    Selection,
    Site,
    _act_entry,
    _join_relation,
    _site_shape,
    _unresolvable,
    register_term_resolver,
)
from . import _engine
from ._engine_forward import solve_projective
from ._errors import ReceptiveFieldError, ReceptiveFieldUnavailableError
from ._gradient import gradient_for_unit
from ._gradient_forward import projective_gradient_for_unit, projective_supports_by_row_sweep
from ._path import between_labels
from ._query import _AxisSets, _IndexSet, _validate_descriptor_for_query
from ._set_query import _IndexSetBox, box_for_source_units, box_for_units
from ._types import GradReceptiveField, ReceptiveField, ReceptiveFieldStatus

__all__: list[str] = []

_RECTANGLE_BUDGET = 64
_STATUS_RANK = {
    ReceptiveFieldStatus.EXACT: 0,
    ReceptiveFieldStatus.WHOLE_INPUT: 1,
    ReceptiveFieldStatus.UPPER_BOUND: 2,
}
_SOURCE_PREFIX = "influence("

_Rectangle = tuple[tuple[int, int], ...]


# ---------------------------------------------------------------------------
# The term and its producer (private).
# ---------------------------------------------------------------------------


def _source_string(direction: str, **extra: Any) -> str:
    """Spell the free-text disclosure carried in ``provenance.source``."""

    tail = "".join(f", {key}={value}" for key, value in extra.items())
    return f"{_SOURCE_PREFIX}{direction}{tail})"


def _influence_entry(
    op: Any, mask: _Mask, relation: str, source: str, disclosure: tuple[str, str]
) -> Site:
    """Build one influence entry; ``disclosure`` is its ``(basis, field_status)``."""

    basis, field_status = disclosure
    entry = _act_entry(op, mask, relation, source)
    provenance = replace(entry.provenance, basis=basis, field_status=field_status)
    return replace(entry, provenance=provenance)


# ---------------------------------------------------------------------------
# Resolution: shared plumbing.
# ---------------------------------------------------------------------------


def _resolve_influence_term(node: _InfluenceTerm, trace: Any) -> Selection:
    """Resolve one influence query against a trace (geometric or gradient)."""

    graph = _ExecutedDag.from_trace(trace)
    seed = _resolve_region(node.seed, trace)
    targets = _target_ops(graph, trace, node)
    seeds = tuple((_site_op(graph, entry), entry) for entry in seed if entry._mask.form != "empty")
    if not seeds:
        # G9 rule: an all-empty seed reaches nothing, disclosed as zero-mask
        # entries at every target (family kept), never as an error.
        source = _source_string(node.direction)
        entries = [
            _influence_entry(
                op, _mask_empty(_site_shape(op)), "exact", source, (node.method, "none")
            )
            for op in targets
        ]
        return Selection(trace, "ACT", entries)
    if node.method == "gradient":
        return _gradient_influence(trace, seeds, targets, node)
    return _geometric_influence(trace, seeds, targets, node)


def _site_op(graph: _ExecutedDag, entry: Site) -> Any:
    """Map one resolved ACT entry to its executed op (internal invariant)."""

    label = graph.by_site.get(entry.key)
    if label is None:
        raise RuntimeError(
            f"influence resolved site {entry.key!r} on this trace but the executed-DAG "
            "index has no such node; the site and graph universes drifted (internal invariant)."
        )
    return graph.ops[label]


def _target_ops(graph: _ExecutedDag, trace: Any, node: _InfluenceTerm) -> tuple[Any, ...]:
    """Resolve the target family: ``within`` regions or the view default endpoints."""

    if node.within is None:
        if node.direction == "projective":
            return tuple(trace.output_ops)
        return tuple(op for op in trace._layer_list if bool(getattr(op, "is_input", False)))
    labels: set[str] = set()
    for operand in node.within:
        for entry in _resolve_region(operand, trace):
            labels.add(_site_op(graph, entry).label)
    return tuple(graph.ops[label] for label in graph.execution_ordered(frozenset(labels)))


def _worst_status(statuses: Iterable[ReceptiveFieldStatus]) -> str:
    """Fold field statuses worst-of (EXACT < WHOLE_INPUT < UPPER_BOUND)."""

    collected = tuple(statuses)
    if not collected:
        return "none"
    return max(collected, key=lambda status: _STATUS_RANK[status]).value


# ---------------------------------------------------------------------------
# Geometric method.
# ---------------------------------------------------------------------------


def _geometric_influence(
    trace: Any,
    seeds: tuple[tuple[Any, Site], ...],
    targets: tuple[Any, ...],
    node: _InfluenceTerm,
) -> Selection:
    """Fold one set query per (seed site, rectangle, target site) into entries."""

    entries: list[Site] = []
    for target in targets:
        shape = _site_shape(target)
        folded = _mask_empty(shape)
        relation = "exact"
        statuses: list[ReceptiveFieldStatus] = []
        for seed_op, entry in seeds:
            mask, pair_relation, pair_statuses = _geometric_pair(
                trace, seed_op, entry._mask, target, node.direction
            )
            folded = _mask_union(folded, mask)
            relation = _join_relation(relation, pair_relation)
            statuses.extend(pair_statuses)
        source = _source_string(node.direction)
        entries.append(
            _influence_entry(
                target, folded, relation, source, ("geometric", _worst_status(statuses))
            )
        )
    return Selection(trace, "ACT", entries)


def _geometric_pair(
    trace: Any,
    seed_op: Any,
    mask: _Mask,
    target: Any,
    direction: str,
) -> tuple[_Mask, str, tuple[ReceptiveFieldStatus, ...]]:
    """Propagate one seed site's mask to one target site (geometric)."""

    shape = _site_shape(target)
    if target is seed_op:
        return mask, "exact", (ReceptiveFieldStatus.EXACT,)
    pair = _pair_descriptor(trace, seed_op, target, direction)
    if pair is None:
        return _mask_empty(shape), "exact", ()
    descriptor, query = pair
    _refuse_ungeometric(descriptor, seed_op, target)
    if tuple(descriptor.input_shape) != shape:
        raise _unresolvable(
            "mask_shape_mismatch",
            f"influence geometry space {tuple(descriptor.input_shape)!r} does not match "
            f"site {target.label!r} output space {shape!r}.",
            code="selection_unresolvable",
            site=target.label,
        )
    rectangles, decomposed_exactly = _rectangles(mask)
    relation = "exact" if decomposed_exactly else "upper_bound"
    folded = _mask_empty(shape)
    statuses: list[ReceptiveFieldStatus] = []
    memo: dict[Any, Any] = {}
    for rectangle in rectangles:
        result = query(_rectangle_seed(seed_op, descriptor, rectangle), memo)
        folded = _mask_union(folded, _mask_from_box(result, descriptor, shape))
        relation = _join_relation(relation, _box_relation(result))
        statuses.append(result.box.status)
    return folded, relation, tuple(statuses)


def _pair_descriptor(trace: Any, seed_op: Any, target: Any, direction: str) -> Any:
    """Return ``(descriptor, query)`` for one (seed, target) pair, or ``None`` if unreachable.

    ``query(seed_sets, memo)`` runs the engine's set sibling for the pair on
    the very descriptor returned here (the one the caller checks for geometry
    and shape); an unreachable pair (no descriptor) means no influence, an
    exact answer.
    """

    if direction == "projective":
        projective = solve_projective(trace, (target,))
        found = _descriptor_for(projective.per_op.get(seed_op.label), target.label)
        if found is None:
            return None
        return found, _source_units_query(projective, seed_op, found)
    if bool(getattr(target, "is_input", False)):
        solution = _engine.solve(trace)
    else:
        solution = _engine.solve_from(trace, target)
    found = _descriptor_for(solution.per_op.get(seed_op.label), target.label)
    if found is None:
        return None
    return found, _units_query(solution, seed_op, found)


def _units_query(solution: Any, seed_op: Any, descriptor: ReceptiveField) -> Any:
    """Bind the receptive set query for one (seed, far endpoint) descriptor."""

    return lambda sets, memo: box_for_units(solution, seed_op, descriptor, sets, memo=memo)


def _source_units_query(solution: Any, seed_op: Any, descriptor: ReceptiveField) -> Any:
    """Bind the projective set query for one (seed, target) descriptor."""

    return lambda sets, memo: box_for_source_units(solution, seed_op, descriptor, sets, memo=memo)


def _descriptor_for(descriptors: Any, far_label: str) -> ReceptiveField | None:
    """Pick the descriptor whose far endpoint is ``far_label``."""

    if not descriptors:
        return None
    for descriptor in descriptors.values():
        if descriptor.input_op_label == far_label:
            return descriptor
    return None


def _refuse_ungeometric(descriptor: ReceptiveField, seed_op: Any, target: Any) -> None:
    """Refuse typed when the pair's descriptor cannot support per-unit geometry."""

    try:
        _validate_descriptor_for_query(descriptor, require_windowed=False)
    except ReceptiveFieldError as error:
        raise _unresolvable(
            "influence_geometry_unavailable",
            f"influence geometry between {seed_op.label!r} and {target.label!r} is "
            f"unavailable (field status {descriptor.status.value!r}): the field engine has no "
            "per-unit index geometry along this path (data-dependent routing such as "
            "index selection has none). For an empirical answer pass method='gradient' on a "
            f"trace captured with capture={_GRADIENT_CAPTURE}.",
            code="influence_geometry_unavailable",
            seed_site=seed_op.label,
            target_site=target.label,
            field_status=descriptor.status.value,
        ) from error


def _rectangles(mask: _Mask) -> tuple[tuple[_Rectangle, ...], bool]:
    """Decompose a mask into axis-aligned rectangles (half-open bounds per axis).

    ``whole`` and unit-step ``slices`` masks are one rectangle. A dense mask is
    decomposed greedily: the lexicographically first selected cell starts a
    block that extends axis by axis while the whole slab stays selected. Past
    ``_RECTANGLE_BUDGET`` blocks the bounding rectangle stands in and the
    second item is ``False`` (the caller drops the relation to upper_bound).
    """
    # Complexity (C): the greedy slab extension is one essential loop; splitting hides the budget.

    shape = mask.shape
    if mask.form == "whole":
        return (tuple((0, extent) for extent in shape),), True
    if mask.form == "slices":
        assert mask.slice_bounds is not None
        if all(step == 1 for _, _, step in mask.slice_bounds):
            return (tuple((start, stop) for start, stop, _ in mask.slice_bounds),), True
    dense = mask._dense_ro()
    remaining = dense.clone()
    blocks: list[_Rectangle] = []
    while True:
        selected = remaining.nonzero()
        if selected.numel() == 0:
            return tuple(blocks), True
        if len(blocks) >= _RECTANGLE_BUDGET:
            return (_bounding_rectangle(dense),), False
        block = _grow_block(remaining, tuple(int(value) for value in selected[0]))
        blocks.append(block)
        remaining[tuple(slice(start, stop) for start, stop in block)] = False


def _grow_block(remaining: torch.Tensor, start: tuple[int, ...]) -> _Rectangle:
    """Extend a block from ``start`` axis by axis while the whole slab is selected."""

    stop = [coordinate + 1 for coordinate in start]
    for axis, extent in enumerate(remaining.shape):
        while stop[axis] < extent:
            probe = tuple(
                slice(stop[axis], stop[axis] + 1)
                if other == axis
                else slice(start[other], stop[other])
                for other in range(len(start))
            )
            if not bool(remaining[probe].all()):
                break
            stop[axis] += 1
    return tuple(zip(start, stop, strict=True))


def _bounding_rectangle(dense: torch.Tensor) -> _Rectangle:
    """Return the per-axis bounding rectangle of a dense mask's selected cells."""

    selected = dense.nonzero()
    return tuple(
        (int(selected[:, axis].min()), int(selected[:, axis].max()) + 1)
        for axis in range(dense.dim())
    )


def _rectangle_seed(seed_op: Any, descriptor: ReceptiveField, rectangle: _Rectangle) -> _AxisSets:
    """Turn one rectangle over the seed op's output into an engine seed.

    Windowed axes always carry their interval; an axis the rectangle covers
    completely is left unconstrained (``None``, the per-unit convention) so a
    whole-axis seed reproduces the per-unit hull exactly; a partially covered
    pointwise or full axis carries its interval and is routed like a
    ``complete_unit`` coordinate.
    """

    assert descriptor.axes is not None
    windowed = {axis.output_axis for axis in descriptor.axes if axis.kind == "windowed"}
    seed: list[_IndexSet | None] = []
    for axis, ((start, stop), extent) in enumerate(zip(rectangle, seed_op.shape, strict=True)):
        if axis not in windowed and start == 0 and stop == int(extent):
            seed.append(None)
        else:
            seed.append(_IndexSet.interval(start, stop - 1))
    return tuple(seed)


def _box_relation(result: _IndexSetBox) -> str:
    """Grade one hull exactly as the shipped box lift does."""

    box = result.box
    return "exact" if (box.exact and not box.sparse_possible) else "upper_bound"


def _mask_from_box(
    result: _IndexSetBox, descriptor: ReceptiveField, shape: tuple[int, ...]
) -> _Mask:
    """Turn one set-query answer into a mask over the target site.

    Windowed and full axes take the hull's slices (the box lift's
    convention); a pointwise axis takes the propagated index set when the
    walk constrained it, so a seed restricted to one batch row or channel
    stays restricted there.
    """
    # Complexity (C): the three axis kinds (windowed, full, pointwise) are the rule itself.

    assert descriptor.axes is not None
    if result.box.empty:
        return _mask_empty(shape)
    slices = result.box.slices()
    per_axis: list[slice | tuple[int, ...]] = []
    for axis, axis_descriptor in enumerate(descriptor.axes):
        propagated = result.clipped_sets[axis]
        if axis_descriptor.kind == "pointwise" and propagated is not None:
            values = propagated.clipped(shape[axis]).values()
            if not values:
                return _mask_empty(shape)
            contiguous = values[-1] - values[0] + 1 == len(values)
            per_axis.append(slice(values[0], values[-1] + 1) if contiguous else values)
        else:
            per_axis.append(slices[axis])
    if all(isinstance(item, slice) for item in per_axis):
        return _mask_from_slices(shape, tuple(item for item in per_axis if isinstance(item, slice)))
    return _mask_from_dense(shape, _product_mask(shape, per_axis))


def _product_mask(
    shape: tuple[int, ...], per_axis: Sequence[slice | tuple[int, ...]]
) -> torch.Tensor:
    """Materialize the Cartesian product of per-axis index choices as a dense mask."""

    dense = torch.zeros(shape, dtype=torch.bool)
    indexers = []
    for axis, item in enumerate(per_axis):
        index = (
            torch.arange(*item.indices(shape[axis]))
            if isinstance(item, slice)
            else torch.tensor(item, dtype=torch.long)
        )
        view = [1] * len(shape)
        view[axis] = -1
        indexers.append(index.reshape(view))
    dense[tuple(indexers)] = True
    return dense


# ---------------------------------------------------------------------------
# Gradient method.
# ---------------------------------------------------------------------------


def _refuse_over_probe_budget(seed_units: int, target_sites: int, probe_budget: int) -> None:
    """Refuse typed, before any probe runs, when the gradient method would exceed its budget."""

    total = seed_units * target_sites
    if total > probe_budget:
        raise _unresolvable(
            "influence_probe_budget",
            f"the gradient method would run {total} autograd probes ({seed_units} seed "
            f"unit(s) x {target_sites} target site(s)), over probe_budget={probe_budget}. "
            "Narrow the seed or the targets, raise probe_budget=, or use the geometric method.",
            code="influence_probe_budget",
            probes=total,
            probe_budget=probe_budget,
        )


_GRADIENT_CAPTURE = 'tl.options.CaptureOptions(backward_ready=True, save_mode="reference")'


def _gradient_influence(
    trace: Any,
    seeds: tuple[tuple[Any, Site], ...],
    targets: tuple[Any, ...],
    node: _InfluenceTerm,
) -> Selection:
    """OR one autograd probe per (seed unit, target site) into exact entries."""
    # Complexity (C): the per-probe fold with its typed shape check is one cohesive step.

    probes = tuple(
        (seed_op, tuple(int(value) for value in coordinates))
        for seed_op, entry in seeds
        for coordinates in entry._mask._dense_ro().nonzero()
    )
    total = len(probes) * len(targets)
    _refuse_over_probe_budget(len(probes), len(targets), node.probe_budget)
    shapes = {target.label: _site_shape(target) for target in targets}
    supports = {
        target.label: torch.zeros(shapes[target.label], dtype=torch.bool) for target in targets
    }
    cross_batch = dict.fromkeys(shapes, False)
    units_by_seed: dict[str, tuple[Any, list[tuple[int, ...]]]] = {}
    for seed_op, unit in probes:
        units_by_seed.setdefault(seed_op.label, (seed_op, []))[1].append(unit)
    # Every probe retains the autograd graph: freeing it on the last probe
    # would make the next gradient query on the same trace fail, and the
    # saved tensors live as long as the armed trace does anyway.
    for seed_op, units in units_by_seed.values():
        for target in targets:
            for support, crossed in _probe_units(seed_op, units, target, node):
                if tuple(support.shape) != shapes[target.label]:
                    raise _unresolvable(
                        "mask_shape_mismatch",
                        f"gradient support shape {tuple(support.shape)!r} does not match site "
                        f"{target.label!r} output space {shapes[target.label]!r}.",
                        code="selection_unresolvable",
                        site=target.label,
                    )
                supports[target.label] |= support
                cross_batch[target.label] = cross_batch[target.label] or crossed
    entries = [
        _influence_entry(
            target,
            _mask_from_dense(shapes[target.label], supports[target.label]),
            "exact",
            _source_string(
                node.direction, cross_batch_influence=cross_batch[target.label], probes=total
            ),
            ("gradient", "empirical"),
        )
        for target in targets
    ]
    return Selection(trace, "ACT", entries)


def _probe_units(
    seed_op: Any, units: Sequence[tuple[int, ...]], target: Any, node: _InfluenceTerm
) -> list[tuple[torch.Tensor, bool]]:
    """Probe every seed unit against one target site, sweeping rows when needed.

    A projective probe linearizes the captured graph a second time, which a
    fused kernel with no second derivative (fused attention) refuses. Then one
    ordinary backward pass per target element answers every seed unit at once,
    so the sweep runs when the target site has at most ``probe_budget``
    elements; otherwise the linearization refusal stands.
    """

    try:
        return [_probe_pair(seed_op, unit, target, node.direction) for unit in units]
    except ReceptiveFieldUnavailableError as refusal:
        linearization_failed = isinstance(refusal.__cause__, RuntimeError)
        if node.direction != "projective" or not linearization_failed:
            raise
        if _site_numel(target) > node.probe_budget:
            raise
    swept = projective_supports_by_row_sweep(seed_op, units, target)
    return [(result.support_mask.bool(), bool(result.cross_batch_influence)) for result in swept]


def _site_numel(target: Any) -> int:
    """Number of elements in a target site's output space."""

    numel = 1
    for size in _site_shape(target):
        numel *= int(size)
    return numel


def _pair_connected(seed_op: Any, target: Any, direction: str) -> bool:
    """Whether a directed seed-to-target path exists in the captured graph.

    The geometric method answers an unreachable pair with an empty mask (no
    influence), so the gradient method answers it the same way instead of
    raising; a default target list routinely holds outputs the seed never
    reaches, such as a cache entry from an earlier block.
    """

    trace = seed_op.source_trace
    if trace is None or target.source_trace is not trace:
        return False
    source, sink = (seed_op, target) if direction == "projective" else (target, seed_op)
    return sink.label in between_labels(trace, source, sink)


def _probe_pair(
    seed_op: Any, unit: tuple[int, ...], target: Any, direction: str
) -> tuple[torch.Tensor, bool]:
    """Run one per-unit autograd probe between a seed unit and a target site."""

    if target is seed_op:
        support = torch.zeros(tuple(seed_op.shape), dtype=torch.bool)
        support[unit] = True
        return support, False
    if not _pair_connected(seed_op, target, direction):
        return torch.zeros(_site_shape(target), dtype=torch.bool), False
    if direction == "projective":
        result = projective_gradient_for_unit(seed_op, unit, target=target, retain_graph=True)
    else:
        result = gradient_for_unit(seed_op, unit, source=target, retain_graph=True)
    if not isinstance(result, GradReceptiveField):
        raise RuntimeError("a single-endpoint probe returned a mapping (internal invariant).")
    return result.support_mask.bool(), bool(result.cross_batch_influence)


register_term_resolver(_InfluenceTerm, _resolve_influence_term)
