"""Graph-structural selection producers (L6 producer wave, graph slice).

Selects by WHERE OPS SIT IN THE EXECUTED DAG, not by value or by name. Two
producers, both returning :class:`~torchlens.selection.Selection` queries
that compose with the full ``| & - ~`` algebra and resolve explicitly
against one trace:

- ``neighborhood(of, hops, direction)`` — every op within N dataflow hops
  of a seed region (S3): the seed's touched sites plus everything reachable
  through at most ``hops`` parent/child edges in the requested direction.

- ``between(sources, sinks)`` — the executed sub-DAG carrying influence
  from the source region to the sink region (S4): exactly the ops lying on
  at least one directed source-to-sink path, endpoints included. The same
  region, presented as a graph view with an explicit boundary instead of a
  Selection, is ``trace.between(sources, sinks)`` (one idea, two binding
  modes — see :mod:`torchlens.trace_slice`).

THE GENERAL GRAPH-QUERY SUBSTRATE: both producers are thin pure functions
over :class:`_TraceGraph`, a frozen queryable view of one trace's executed
DAG — nodes are pass-qualified op labels, edges the recorded parent->child
dataflow, and ``graph.ops`` hands back the full typed ``Op`` record for any
node (func name, module stack, shapes). A future graph-MOTIF producer
(subgraph isomorphism over node/edge predicates) is one more pure function
over this same object producing label sets that materialize through the
same ``_materialize_labels`` door — an extension, not a rewrite. The
adjacency itself is built by the ONE shipped executed-DAG indexer
(:func:`torchlens.receptive_field._path._graph_indexes`), shared verbatim
with the influence-geometry path machinery.

Membership claims are STRUCTURAL facts about THIS capture's executed DAG,
so every resolved entry is a whole-site mask with
``provenance.relation="exact"``. When a seed region was itself produced
inexactly (an upper-bound receptive-field hull), the hop claim stays exact
RELATIVE TO the seed's touched-site family; the seed's own inexactness is
disclosed on the seed, not silently re-graded here.

Every spelling here ships DOCUMENTED-UNSTABLE pending naming-session
ratification (provisional-name protocol). Producers are
ACT-kind only: PARAM/EDGE operands refuse ``selection_kind_incompatible``
(parameters and edge occurrences are not nodes of the op DAG; a
param-aware graph query is a named possibility, not a promise).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .selection import (
    ResolvedSelection,
    Selection,
    SelectionError,
    _act_entry,
    _lift,
    _mask_whole,
    _site_shape,
    _unresolvable,
    _WholeSiteTerm,
    register_term_resolver,
)

__all__ = [
    "between",
    "neighborhood",
]

_DIRECTIONS = ("both", "upstream", "downstream")


# ---------------------------------------------------------------------------
# The graph-query substrate.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _TraceGraph:
    """One trace's executed DAG as a frozen, queryable plain graph.

    Nodes are pass-qualified op labels (``Op.label``); ``parents`` /
    ``children`` are the directed dataflow adjacency; ``ops`` maps every
    node to its full typed ``Op`` record; ``order`` is execution order and
    ``by_site`` indexes nodes by the selection-algebra site key
    ``(layer_label, pass_index)``. Queries are pure functions over this
    object — the seam a future motif matcher plugs into.
    """

    ops: Mapping[str, Any] = field(repr=False)
    parents: Mapping[str, frozenset[str]] = field(repr=False)
    children: Mapping[str, frozenset[str]] = field(repr=False)
    order: tuple[str, ...] = field(repr=False)
    by_site: Mapping[tuple[str, int], str] = field(repr=False)

    @classmethod
    def from_trace(cls, trace: Any) -> _TraceGraph:
        """Build the graph view from one finished trace's executed ops."""

        from .receptive_field._path import _graph_indexes

        by_label, parents, children = _graph_indexes(trace)
        order = tuple(
            op.label for op in trace.layer_list if isinstance(getattr(op, "label", None), str)
        )
        by_site: dict[tuple[str, int], str] = {}
        for label, op in by_label.items():
            layer_label = getattr(op, "layer_label", None) or label
            pass_index = getattr(op, "pass_index", 1) or 1
            by_site[(layer_label, pass_index)] = label
        return cls(
            ops=by_label,
            parents=parents,
            children=children,
            order=order,
            by_site=by_site,
        )

    def execution_ordered(self, labels: frozenset[str]) -> tuple[str, ...]:
        """Return the given node labels in execution order."""

        return tuple(label for label in self.order if label in labels)


def _n_hop_labels(
    graph: _TraceGraph, seeds: frozenset[str], hops: int, direction: str
) -> frozenset[str]:
    """Return every node within ``hops`` edges of ``seeds`` (seeds included).

    Bounded breadth-first traversal over the requested adjacency:
    ``upstream`` follows parent edges (toward the inputs), ``downstream``
    follows child edges (toward the outputs), ``both`` follows either at
    every step.
    """

    reached = set(seeds)
    frontier = set(seeds)
    for _ in range(hops):
        advanced: set[str] = set()
        for label in frontier:
            if direction in ("both", "upstream"):
                advanced |= graph.parents.get(label, frozenset())
            if direction in ("both", "downstream"):
                advanced |= graph.children.get(label, frozenset())
        frontier = advanced - reached
        if not frontier:
            break
        reached |= frontier
    return frozenset(reached)


def _between_label_set(
    graph: _TraceGraph, sources: frozenset[str], sinks: frozenset[str]
) -> frozenset[str]:
    """Return nodes on at least one directed source-to-sink path.

    Equal to (descendants of any source) ∩ (ancestors of any sink): a node
    in both sets witnesses a concrete source->node->sink path. Endpoints
    are included; the set is empty when no directed path exists (emptiness
    is disclosure, never an error).
    """

    from .receptive_field._path import _reachable_labels

    descendants: set[str] = set()
    for label in sources:
        descendants |= _reachable_labels(label, graph.children)
    ancestors: set[str] = set()
    for label in sinks:
        ancestors |= _reachable_labels(label, graph.parents)
    return frozenset(descendants & ancestors)


def _edge_partition(
    graph: _TraceGraph, members: frozenset[str]
) -> tuple[
    tuple[tuple[str, str], ...],
    tuple[tuple[str, str], ...],
    tuple[tuple[str, str], ...],
]:
    """Partition the dataflow edges touching a member set.

    Returns ``(internal, boundary_in, boundary_out)`` as deterministic
    execution-ordered ``(parent_label, child_label)`` tuples: edges with
    both ends inside, edges entering from an outside parent, and edges
    leaving to an outside child. The two boundary tuples ARE the slice's
    explicit dangling-edge declaration — external dependencies are named,
    never silently dropped.
    """

    position = {label: index for index, label in enumerate(graph.order)}

    def _ordered(labels: frozenset[str]) -> list[str]:
        """Sort parent labels by execution position (deterministic edge order)."""

        return sorted(labels, key=lambda label: position.get(label, len(position)))

    internal: list[tuple[str, str]] = []
    boundary_in: list[tuple[str, str]] = []
    boundary_out: list[tuple[str, str]] = []
    for child in graph.order:
        if child in members:
            for parent in _ordered(graph.parents.get(child, frozenset())):
                if parent in members:
                    internal.append((parent, child))
                else:
                    boundary_in.append((parent, child))
        else:
            for parent in _ordered(graph.parents.get(child, frozenset())):
                if parent in members:
                    boundary_out.append((parent, child))
    return tuple(internal), tuple(boundary_in), tuple(boundary_out)


# ---------------------------------------------------------------------------
# Endpoint operands: lifting + resolution to node-label sets.
# ---------------------------------------------------------------------------


def _lift_region(value: Any, producer: str, operand: str) -> Any:
    """Lift one endpoint/seed operand to an ACT selection.

    Accepts a site label string (whole-site; a bare layer label on a
    multi-pass layer is the all-passes Layer spelling), or anything
    selection-shaped (``Op``, ``Layer``, receptive-field regions, other
    Selections). Non-ACT kinds refuse typed; ``None`` is never a region.
    """

    if isinstance(value, str):
        if not value:
            raise ValueError(f"{producer} `{operand}` site label must be a non-empty string.")
        return Selection(_WholeSiteTerm(site_label=value, pass_index=None), kind="ACT")
    lifted = _lift(value)
    if lifted is None:
        raise ValueError(
            f"{producer} `{operand}` must be a site label string or a "
            "selection-shaped producer implementing __selection__; got "
            f"{type(value).__name__}."
        )
    kind = lifted.kind
    if kind != "ACT":
        raise SelectionError(
            f"{producer} queries the executed op DAG, so `{operand}` must be "
            f"an ACT selection; got a {kind} selection. (Parameters and edge "
            "occurrences are not nodes of the op DAG; a param-aware graph "
            "query is a named possibility, not a promise.)",
            code="selection_kind_incompatible",
            left_kind="ACT",
            right_kind=kind,
            operator=operand,
        )
    return lifted


def _lift_region_group(value: Any, producer: str, operand: str) -> tuple[Any, ...]:
    """Lift one-or-many endpoint operands (list/tuple/set fan in)."""

    if isinstance(value, (list, tuple, set, frozenset)):
        collected = tuple(value)
        if not collected:
            raise ValueError(f"{producer} `{operand}` must name at least one region.")
        return tuple(
            _lift_region(item, producer, f"{operand}[{index}]")
            for index, item in enumerate(collected)
        )
    return (_lift_region(value, producer, operand),)


def _resolve_region(operand: Any, trace: Any) -> ResolvedSelection:
    """Resolve one lifted endpoint operand against the resolution trace."""

    if isinstance(operand, ResolvedSelection):
        if operand._trace is not trace:
            raise SelectionError(
                "a graph-producer endpoint is bound to a different trace; "
                "re-resolve it against this trace first.",
                code="selection_trace_mismatch",
            )
        return operand
    return operand.resolve(trace)


def _region_labels(
    graph: _TraceGraph, resolved: ResolvedSelection, producer: str
) -> frozenset[str]:
    """Map a resolved region's touched-site family to graph node labels.

    Family semantics: element masks never shrink a graph region — a
    touched site is a touched node. Zero-mask entries therefore still
    seed/bound the traversal (two-level denotation, family level).
    """

    labels: set[str] = set()
    for entry in resolved:
        label = graph.by_site.get(entry.site_key)
        if label is None:
            raise RuntimeError(
                f"{producer} resolved site {entry.site_key!r} on this trace but "
                "the executed-DAG index has no such node; the site and graph "
                "universes drifted (internal invariant)."
            )
        labels.add(label)
    return frozenset(labels)


def _materialize_labels(
    graph: _TraceGraph, trace: Any, labels: frozenset[str], source: str
) -> ResolvedSelection:
    """Materialize a node-label set as whole-site ACT entries (exact).

    The ONE door from substrate label sets to the selection algebra —
    neighborhood, between, and any future motif matcher all exit through
    it. Sites without a usable output index space refuse typed
    (``no_index_space`` / ``non_tensor_site``), exactly like the shipped
    selector lift: silently excluding a member would misreport the family.
    """

    entries = [
        _act_entry(graph.ops[label], _mask_whole(_site_shape(graph.ops[label])), "exact", source)
        for label in graph.execution_ordered(labels)
    ]
    return ResolvedSelection(trace, "ACT", entries)


# ---------------------------------------------------------------------------
# AST terms + resolvers.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _SliceMembersTerm:
    """AST leaf for a TraceSlice's member family (explicit site set).

    The slice's algebra lift: a QUERY over the member sites by
    ``(layer_label, pass_index)`` address (exactly how Op/Layer lifts
    work), so a slice built on one trace composes and re-resolves on a
    same-architecture trace (a fork) by site name. A member absent from
    the resolution trace refuses ``site_not_in_trace``.
    """

    sites: tuple[tuple[str, int], ...]
    display: str

    def __repr__(self) -> str:
        """Return the compact disclosure (member count, never the full set)."""

        return f"slice_members(n={len(self.sites)}, from={self.display})"


def _resolve_slice_members_term(node: _SliceMembersTerm, trace: Any) -> ResolvedSelection:
    """Resolve a slice's member family against one trace (whole-site exact)."""

    graph = _TraceGraph.from_trace(trace)
    labels: set[str] = set()
    for site in node.sites:
        label = graph.by_site.get(site)
        if label is None:
            raise _unresolvable(
                "site_not_in_trace",
                f"slice member site {site!r} is not a site of this trace.",
                site=repr(site),
            )
        labels.add(label)
    return _materialize_labels(graph, trace, frozenset(labels), node.display)


@dataclass(frozen=True)
class _NeighborhoodTerm:
    """AST leaf for the n-hop neighborhood producer."""

    seed: Any
    hops: int
    direction: str

    def __repr__(self) -> str:
        """Return the compact constructor-shaped disclosure."""

        return f"neighborhood(of={self.seed!r}, hops={self.hops}, direction={self.direction!r})"


@dataclass(frozen=True)
class _BetweenTerm:
    """AST leaf for the source-to-sink influence-region producer."""

    sources: tuple[Any, ...]
    sinks: tuple[Any, ...]

    def __repr__(self) -> str:
        """Return the compact constructor-shaped disclosure."""

        sources = ", ".join(repr(operand) for operand in self.sources)
        sinks = ", ".join(repr(operand) for operand in self.sinks)
        return f"between(sources=[{sources}], sinks=[{sinks}])"


def _resolve_neighborhood_term(node: _NeighborhoodTerm, trace: Any) -> ResolvedSelection:
    """Resolve one n-hop neighborhood against the trace's executed DAG."""

    graph = _TraceGraph.from_trace(trace)
    seeds = _region_labels(graph, _resolve_region(node.seed, trace), "neighborhood")
    labels = _n_hop_labels(graph, seeds, node.hops, node.direction)
    source = f"neighborhood(hops={node.hops}, direction={node.direction!r})"
    return _materialize_labels(graph, trace, labels, source)


def _resolve_between_term(node: _BetweenTerm, trace: Any) -> ResolvedSelection:
    """Resolve one influence region against the trace's executed DAG."""

    labels, graph = resolve_between_labels(trace, node.sources, node.sinks)
    return _materialize_labels(graph, trace, labels, "between(sources -> sinks)")


def resolve_between_labels(
    trace: Any, sources: tuple[Any, ...], sinks: tuple[Any, ...]
) -> tuple[frozenset[str], _TraceGraph]:
    """Resolve lifted endpoint groups to the between-region label set.

    The shared machinery behind BOTH spellings of the one idea: the
    ``tl.between`` producer's resolver and the ``trace.between`` slice
    presenter call this same function (internal seam).
    """

    graph = _TraceGraph.from_trace(trace)
    source_labels: set[str] = set()
    for operand in sources:
        source_labels |= _region_labels(graph, _resolve_region(operand, trace), "between")
    sink_labels: set[str] = set()
    for operand in sinks:
        sink_labels |= _region_labels(graph, _resolve_region(operand, trace), "between")
    return _between_label_set(graph, frozenset(source_labels), frozenset(sink_labels)), graph


# ---------------------------------------------------------------------------
# Producers.
# ---------------------------------------------------------------------------


def neighborhood(
    of: Any,
    hops: int = 1,
    *,
    direction: str = "both",
) -> Selection:
    """Select every op within N dataflow hops of a seed region.

    Bounded traversal over THIS capture's executed DAG from the seed's
    touched sites: ``hops=0`` is the seed family itself, each further hop
    crosses one recorded parent/child dataflow edge. ``direction`` is one
    of ``'both'`` (default), ``'upstream'`` (parent edges, toward the
    inputs), or ``'downstream'`` (child edges, toward the outputs). ``of``
    accepts a site label string (bare layer labels on multi-pass layers
    mean all passes), an ``Op``/``Layer``, or any ACT selection — element
    masks never shrink the seed (a touched site is a touched node).
    Membership is a structural fact about this capture, so entries are
    whole-site masks with ``provenance.relation="exact"``. Resolution
    refuses typed on unknown seed sites (``site_not_in_trace``) and on
    member sites with no output index space (``non_tensor_site`` /
    ``no_index_space``). DOCUMENTED-UNSTABLE spelling.
    """

    if isinstance(hops, bool) or not isinstance(hops, int) or hops < 0:
        raise ValueError(f"neighborhood `hops` must be a non-negative int; got {hops!r}.")
    if direction not in _DIRECTIONS:
        raise ValueError(
            f"neighborhood `direction` must be one of {_DIRECTIONS}; got {direction!r}."
        )
    return Selection(
        _NeighborhoodTerm(
            seed=_lift_region(of, "neighborhood", "of"), hops=hops, direction=direction
        ),
        kind="ACT",
    )


def between(sources: Any, sinks: Any) -> Selection:
    """Select the executed sub-DAG carrying influence from sources to sinks.

    Exactly the ops lying on at least one directed source-to-sink dataflow
    path in THIS capture's executed DAG, endpoints included — the
    circuit-slice primitive. ``sources`` and ``sinks`` each accept one
    region or a list of regions (site label strings, ``Op``/``Layer``
    handles, or any ACT selection); the region is
    (descendants of any source) ∩ (ancestors of any sink). No directed
    path resolves to the EMPTY selection — emptiness is disclosure, never
    an error (the raising flavor is the influence-geometry
    ``require_path``). Entries are whole-site masks with
    ``provenance.relation="exact"``. The same region as a graph VIEW with
    an explicit boundary is ``trace.between(sources, sinks)``.
    DOCUMENTED-UNSTABLE spelling.
    """

    return Selection(
        _BetweenTerm(
            sources=_lift_region_group(sources, "between", "sources"),
            sinks=_lift_region_group(sinks, "between", "sinks"),
        ),
        kind="ACT",
    )


register_term_resolver(_NeighborhoodTerm, _resolve_neighborhood_term)
register_term_resolver(_BetweenTerm, _resolve_between_term)
register_term_resolver(_SliceMembersTerm, _resolve_slice_members_term)
