"""Run-fold legality over indexed flow adjacency, de-quadratic (B2).

This module owns the v2 run-fold legality grammar (moved verbatim in
semantics from ``auto_collapse.py``) plus the collapse memo's B2 build item
(item 7): the legality checks read a per-graph :class:`FlowAdjacency` index
built ONCE per :class:`~._condensed_flow.ChildCondensedFlowGraph` instead of
re-scanning ``graph.edges`` per candidate window per member (the measured
24,165-call quadratic on densenet201-class sibling fans), and
:func:`longest_uniform_legal_run` replaces the historical grow-every-window
enumeration with one signature pass per component plus a longest-first
legality scan — output-identical to the historical "longest legal uniform
window from each start" selection.

``legality_check_count`` is the deterministic de-quadratic instrument: tests
pin that the number of legality checks scales linearly, not quadratically,
in sibling-fan width (a wall-clock-free gate).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from ._collapse_signatures import MemberFingerprintCache, _exterior_bindings_consistent

if TYPE_CHECKING:
    from ._condensed_flow import ChildCondensedFlowGraph

#: Minimum member count for a "+N more" repeat fold.
RUN_FOLD_MIN_LENGTH = 3

#: Monotone count of legality-grammar checks (test instrument, never reset
#: by library code; read via :func:`legality_check_count`).
_LEGALITY_CHECKS = 0


def legality_check_count() -> int:
    """Return the process-lifetime count of run-fold legality checks."""

    return _LEGALITY_CHECKS


class FlowAdjacency:
    """Deduplicated adjacency index over one child-condensed flow graph.

    Built once per graph object and memoized weakly, so every legality
    check over the same sibling component costs ``O(sum deg(member))``
    instead of ``O(len(graph.edges))``.
    """

    __slots__ = ("edge_set", "flow_index", "in_neighbors", "out_neighbors")

    def __init__(self, graph: ChildCondensedFlowGraph) -> None:
        """Index ``graph``'s flow order and deduplicated edge adjacency."""

        self.flow_index: dict[str, int] = {
            address: index for index, address in enumerate(graph.flow_children)
        }
        self.edge_set: frozenset[tuple[str, str]] = frozenset(graph.edges)
        out_neighbors: dict[str, dict[str, None]] = {}
        in_neighbors: dict[str, dict[str, None]] = {}
        for source, target in self.edge_set:
            out_neighbors.setdefault(source, {})[target] = None
            in_neighbors.setdefault(target, {})[source] = None
        self.out_neighbors: dict[str, tuple[str, ...]] = {
            node: tuple(targets) for node, targets in out_neighbors.items()
        }
        self.in_neighbors: dict[str, tuple[str, ...]] = {
            node: tuple(sources) for node, sources in in_neighbors.items()
        }


#: Attribute slot for the memoized per-graph adjacency index. The graph is a
#: frozen dataclass whose Mapping fields make it unhashable (no weak-dict
#: keying), so the index rides the instance dict via ``object.__setattr__``
#: -- identity-keyed and lifetime-tied by construction, and the graph's
#: frozen fields guarantee the index can never go stale.
_ADJACENCY_ATTR = "_torchlens_flow_adjacency"


def _flow_adjacency(graph: ChildCondensedFlowGraph) -> FlowAdjacency:
    """Return the memoized adjacency index for ``graph``."""

    cached = getattr(graph, _ADJACENCY_ATTR, None)
    if cached is not None:
        return cast("FlowAdjacency", cached)
    adjacency = FlowAdjacency(graph)
    object.__setattr__(graph, _ADJACENCY_ATTR, adjacency)
    return adjacency


def _run_fold_is_legal(
    addresses: tuple[str, ...],
    graph: ChildCondensedFlowGraph | None,
) -> bool:
    """Return whether a candidate run satisfies the v2 legality grammar.

    Parameters
    ----------
    addresses:
        Candidate run addresses in flow order.
    graph:
        Child-condensed flow graph for the run's parent.

    Returns
    -------
    bool
        True for legal chain intervals or legal parallel-fan bundles.
    """

    global _LEGALITY_CHECKS
    _LEGALITY_CHECKS += 1
    if len(addresses) < RUN_FOLD_MIN_LENGTH or graph is None:
        return False
    adjacency = _flow_adjacency(graph)
    if not _run_is_flow_consecutive(addresses, adjacency):
        return False
    return _run_fold_is_chain_interval(addresses, adjacency) or _run_fold_is_parallel_fan(
        addresses,
        adjacency,
    )


def _coerce_adjacency(graph: ChildCondensedFlowGraph | FlowAdjacency) -> FlowAdjacency:
    """Return ``graph`` as a :class:`FlowAdjacency`, indexing it if needed."""

    if isinstance(graph, FlowAdjacency):
        return graph
    return _flow_adjacency(graph)


def _run_is_flow_consecutive(
    addresses: tuple[str, ...],
    adjacency: ChildCondensedFlowGraph | FlowAdjacency,
) -> bool:
    """Return whether ``addresses`` are adjacent in graph flow-child order."""

    flow_index = _coerce_adjacency(adjacency).flow_index
    first = flow_index.get(addresses[0])
    if first is None:
        return False
    return all(
        flow_index.get(address) == first + offset for offset, address in enumerate(addresses)
    )


def _run_fold_is_chain_interval(
    addresses: tuple[str, ...],
    adjacency: ChildCondensedFlowGraph | FlowAdjacency,
) -> bool:
    """Return whether a run satisfies the chain-interval legality contract.

    True when members form one path with one external entry, one external
    exit, and no flagged interior boundary crossing. Reads the adjacency
    index; behavior matches the historical full-edge-scan implementation
    exactly (the scan deduplicated edges through ``set(graph.edges)``, which
    the index bakes in).
    """

    adjacency = _coerce_adjacency(adjacency)
    member_set = set(addresses)
    internal_edges = {
        (source, target)
        for source in addresses
        for target in adjacency.out_neighbors.get(source, ())
        if target in member_set and target != source
    }
    expected_edges = set(zip(addresses[:-1], addresses[1:], strict=True))
    connector_nodes = _chain_connector_nodes(addresses, adjacency)
    if connector_nodes is None:
        return False
    direct_expected_edges = expected_edges & internal_edges
    if internal_edges - direct_expected_edges:
        return False
    entries = [
        (source, target)
        for target in addresses
        for source in adjacency.in_neighbors.get(target, ())
        if source not in member_set and source not in connector_nodes
    ]
    if len(entries) != 1 or entries[0][1] != addresses[0]:
        return False
    exits = [
        (source, target)
        for source in addresses
        for target in adjacency.out_neighbors.get(source, ())
        if target not in member_set and target not in connector_nodes
    ]
    return not (len(exits) != 1 or exits[0][0] != addresses[-1])


def _chain_connector_nodes(
    addresses: tuple[str, ...],
    adjacency: ChildCondensedFlowGraph | FlowAdjacency,
) -> set[str] | None:
    """Return external one-hop connectors for a chain run if it forms a path.

    Returns the external connector nodes used between members, or ``None``
    if any consecutive pair is not connected by exactly one path step.
    """

    adjacency = _coerce_adjacency(adjacency)
    member_set = set(addresses)
    edge_set = adjacency.edge_set
    connectors: set[str] = set()
    for left, right in zip(addresses[:-1], addresses[1:], strict=True):
        if (left, right) in edge_set:
            continue
        left_targets = adjacency.out_neighbors.get(left, ())
        pair_connectors = {
            target
            for target in left_targets
            if target not in member_set and (target, right) in edge_set
        }
        paired_connectors = {
            (target, paired)
            for target in left_targets
            for paired in (_paired_external_connector(target),)
            if target not in member_set and paired is not None and (paired, right) in edge_set
        }
        if paired_connectors:
            pair_connectors.update(connector for pair in paired_connectors for connector in pair)
        if len(pair_connectors) != 1:
            if len(pair_connectors) != 2 or not any(
                _paired_external_connector(connector) in pair_connectors
                for connector in pair_connectors
            ):
                return None
        connectors.update(pair_connectors)
    return connectors


def _paired_external_connector(node: str) -> str | None:
    """Return the source/sink counterpart for an external connector node.

    Parameters
    ----------
    node:
        Condensed external connector node name.

    Returns
    -------
    str | None
        Paired connector name, or ``None`` when ``node`` is not an external
        source/sink sentinel.
    """

    if node.startswith("external_sink:"):
        return f"external_source:{node.removeprefix('external_sink:')}"
    if node.startswith("external_source:"):
        return f"external_sink:{node.removeprefix('external_source:')}"
    return None


def _run_fold_is_parallel_fan(
    addresses: tuple[str, ...],
    adjacency: ChildCondensedFlowGraph | FlowAdjacency,
) -> bool:
    """Return whether a run satisfies the parallel-fan legality contract.

    True when members have no mutual edges and identical external source
    and sink sets.
    """

    adjacency = _coerce_adjacency(adjacency)
    member_set = set(addresses)
    source_sets: list[frozenset[str]] = []
    sink_sets: list[frozenset[str]] = []
    for address in addresses:
        sources: set[str] = set()
        sinks: set[str] = set()
        for target in adjacency.out_neighbors.get(address, ()):
            if target in member_set:
                return False
            sinks.add(target)
        for source in adjacency.in_neighbors.get(address, ()):
            if source in member_set:
                return False
            sources.add(source)
        source_sets.append(frozenset(sources))
        sink_sets.append(frozenset(sinks))
    return (
        bool(source_sets[0])
        and bool(sink_sets[0])
        and all(sources == source_sets[0] for sources in source_sets[1:])
        and all(sinks == sink_sets[0] for sinks in sink_sets[1:])
    )


def uniform_prefix_length(
    candidate: tuple[str, ...],
    fingerprints: MemberFingerprintCache,
) -> int:
    """Return the longest member-uniform prefix length of ``candidate``.

    Uniformity is monotone in window length: every member must share the
    first member's structural signature AND merge its exterior bindings
    consistently into the fold's shared frame (in address order), so the
    first failing member bounds every longer window. One signature pass per
    component (B2) replaces the historical per-window recomputation.
    """

    if not candidate:
        return 0
    first_signature = fingerprints.signature(candidate[0])
    merged: dict[str, int] = {}
    length = 0
    for address in candidate:
        if fingerprints.signature(address) != first_signature:
            break
        if not _exterior_bindings_consistent(merged, fingerprints.bindings(address)):
            break
        length += 1
    return length


def longest_uniform_legal_run(
    candidate: tuple[str, ...],
    graph: ChildCondensedFlowGraph | None,
    fingerprints: MemberFingerprintCache,
) -> tuple[str, ...]:
    """Return the longest legal, member-uniform prefix run of ``candidate``.

    Output-identical to the historical grow-every-window enumeration (the
    longest window ``w`` with ``RUN_FOLD_MIN_LENGTH <= w <= len(candidate)``
    that is both legal and uniform), but pays one uniformity pass over the
    component and typically ONE legality check: uniformity is monotone, so
    the scan starts at the uniform prefix bound and walks down, stopping at
    the first legal width. Legality is NOT monotone (entries/exits shift
    with the window), which is why the scan tests each width rather than
    bisecting.
    """

    if graph is None or not candidate:
        return ()
    # Exact head pre-filter (the measured dominant scan waste): when the
    # first member has NO in-edges at all, no window starting here is ever
    # legal at ANY width -- the chain form needs exactly one external entry
    # INTO the first member (zero head in-edges means entries are absent or
    # target an interior member, both illegal), and the fan form needs the
    # first member's external source set to be non-empty. Width-independent,
    # so the whole downward scan is skipped, not approximated.
    if not _flow_adjacency(graph).in_neighbors.get(candidate[0]):
        return ()
    bound = min(len(candidate), uniform_prefix_length(candidate, fingerprints))
    for width in range(bound, RUN_FOLD_MIN_LENGTH - 1, -1):
        run = candidate[:width]
        if _run_fold_is_legal(run, graph):
            return run
    return ()
