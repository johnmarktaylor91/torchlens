"""Sibling-ordering resolution and verification for the forward DOT pipeline.

Extracted verbatim from ``_render_dot.py`` (renderer-thinning pass): these
helpers decide whether sibling ordering is in scope for a render, verify the
injected rank chains against a plain-layout baseline, and rewrite the DOT
source with the surviving chains. They consume :class:`RenderIR`-level
ordering constraints and never touch Trace state.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
import warnings
from collections import defaultdict
from typing import TYPE_CHECKING, cast

from ..errors._base import TorchLensWarning
from ..utils.display import user_stacklevel
from . import _render_utils
from ._render_common import (
    _SIBLING_ORDER_WARNING_EMITTED,
    SIBLING_ORDER_EPSILON,
    SIBLING_ORDER_NODE_CAP,
    SIBLING_ORDER_STRETCH_CAP,
    CapturedForwardEdge,
    CollapseFn,
    PlainLayout,
    SiblingOrderChain,
    SiblingOrderDecision,
    strict_collapse_checks_enabled,
)
from ._render_flow import (
    _assert_sibling_backstops,
    _filter_sibling_chains_to_rendered_nodes,
    _flow_span,
    _inject_sibling_rank_groups,
)

if TYPE_CHECKING:
    from typing import Any

    from .._literals import VisInterventionModeLiteral
    from ..data_classes.module import Module
    from .render_ir import RenderIROrderingConstraint


def _should_order_siblings(
    *,
    order_siblings: bool,
    engine: str,
    vis_mode: str,
    num_nodes: int,
    module: Module | str | None,
    vis_intervention_mode: VisInterventionModeLiteral,
    collapse_fn: CollapseFn | None,
    vis_call_depth: int,
) -> bool:
    """Return whether sibling ordering is in scope for this render."""

    return (
        order_siblings
        and engine == "dot"
        and vis_mode == "unrolled"
        and num_nodes <= SIBLING_ORDER_NODE_CAP
        and module is None
        and vis_intervention_mode == "node_mark"
        and vis_call_depth >= 1000
    )


def _queue_sibling_rank_group(
    module_edge_dict: dict[str, Any],
    top_level_rank_groups: list[SiblingOrderChain],
    chain: SiblingOrderChain,
) -> None:
    """Queue a sibling rank group in the cluster dictionary."""

    if chain.lca_key == -1:
        top_level_rank_groups.append(chain)
    else:
        module_edge_dict[cast(str, chain.lca_key)]["rank_groups"].append(chain)


def _verify_and_apply_sibling_ordering(
    source: str,
    chains: tuple[SiblingOrderChain | RenderIROrderingConstraint, ...],
    captured_edges: list[CapturedForwardEdge],
    rankdir: str,
) -> tuple[str, SiblingOrderDecision]:
    """Verify sibling rank chains and return final DOT source."""

    baseline_source = _strip_sibling_rank_groups(source)
    baseline = _layout_dot_plain(baseline_source, rankdir, captured_edges)
    rendered_chains = _filter_sibling_chains_to_rendered_nodes(
        cast(tuple[SiblingOrderChain, ...], chains), baseline.nodes
    )
    chains = _filter_sibling_chains_to_member_cluster(rendered_chains, baseline_source)
    if not chains:
        return baseline_source, _sibling_order_decision((), (), {})
    if chains != rendered_chains:
        # A dropped chain may still sit in ``source``; rebuild from the baseline.
        source = _inject_sibling_rank_groups(baseline_source, chains)
    injected = _layout_dot_plain(source, rankdir, captured_edges)
    if baseline.nodes and not injected.nodes:
        raise subprocess.SubprocessError("dot -Tplain produced no layout for the ordered graph")
    _assert_sibling_backstops(baseline, injected, chains, captured_edges)

    ratios = {
        _sibling_chain_key(chain): _sibling_chain_stretch_ratio(
            chain, captured_edges, baseline, injected
        )
        for chain in chains
    }
    survivors = tuple(
        chain for chain in chains if ratios[_sibling_chain_key(chain)] <= SIBLING_ORDER_STRETCH_CAP
    )
    current_source = (
        source if survivors == chains else _inject_sibling_rank_groups(baseline_source, survivors)
    )
    current_layout = (
        injected
        if survivors == chains
        else _layout_dot_plain(current_source, rankdir, captured_edges)
    )

    for _ in range(2):
        bad_chains = tuple(
            chain
            for chain in survivors
            if _sibling_chain_stretch_ratio(chain, captured_edges, baseline, current_layout)
            > SIBLING_ORDER_STRETCH_CAP
        )
        if not bad_chains:
            return current_source, _sibling_order_decision(chains, survivors, ratios)
        survivors = tuple(chain for chain in survivors if chain not in bad_chains)
        current_source = _inject_sibling_rank_groups(baseline_source, survivors)
        current_layout = _layout_dot_plain(current_source, rankdir, captured_edges)
    return current_source, _sibling_order_decision(chains, survivors, ratios)


# r-b7 R42-9: one shared TORCHLENS_COLLAPSE_STRICT parser (_render_common).
_strict_sibling_order_checks_enabled = strict_collapse_checks_enabled


def _warn_sibling_order_fallback_once(exc: BaseException) -> None:
    """Warn once when sibling-order verification is skipped in production.

    Parameters
    ----------
    exc:
        Verification failure that triggered the fallback.
    """

    global _SIBLING_ORDER_WARNING_EMITTED
    if _SIBLING_ORDER_WARNING_EMITTED:
        return
    _SIBLING_ORDER_WARNING_EMITTED = True
    warnings.warn(
        TorchLensWarning(
            "Sibling-order verification failed; rendering the plain layout without the "
            f"optional sibling-order post-pass. ({type(exc).__name__}: {exc}) "
            "Remedy: pass order_siblings=False to skip the pass, or report the Graphviz "
            "version and model if the plain layout looks wrong",
            code="sibling_order_fallback",
        ),
        stacklevel=user_stacklevel(),
    )


# Line shapes of the DOT that python-graphviz emits: one statement per line.
_DOT_GRAPH_OPEN = re.compile(r"^\s*(?:strict\s+)?(?:di)?graph\b[^\[]*\{\s*$")
_DOT_SUBGRAPH_OPEN = re.compile(r'^\s*subgraph\s+("(?:[^"\\]|\\.)*"|[^\s{]+)\s*\{\s*$')
_DOT_ANONYMOUS_OPEN = re.compile(r"^\s*(?:subgraph\s*)?\{\s*$")
_DOT_CLOSE = re.compile(r"^\s*\}\s*$")
_DOT_ID = re.compile(r'"(?:[^"\\]|\\.)*"|[^\s\[\];{}=]+')
_DOT_KEYWORDS = frozenset({"graph", "node", "edge"})
# Sentinel for a node referenced from two clusters where neither contains the other.
_AMBIGUOUS_CLUSTER = "\0ambiguous"


def _filter_sibling_chains_to_member_cluster(
    chains: tuple[SiblingOrderChain, ...],
    baseline_source: str,
) -> tuple[SiblingOrderChain, ...]:
    """Keep sibling chains whose rank group sits in its members' own cluster.

    Graphviz does not support a ``rank=same`` set whose members live in a
    different cluster than the set itself: dot 2.43 warns "already in a
    rankset, deleted from cluster" and pulls the node out of its cluster, and
    dot 16 fails with "trouble in init_rank" or crashes. A chain survives only
    when every target's innermost cluster is the cluster its group is emitted
    into (``lca_key``), or all targets and the group are top-level. When the
    baseline DOT cannot be parsed, every chain is dropped (invariant 11).

    Parameters
    ----------
    chains:
        Candidate sibling chains.
    baseline_source:
        DOT source with every sibling-order rank group stripped.

    Returns
    -------
    tuple[SiblingOrderChain, ...]
        Chains that are safe to emit, in input order.
    """

    if not chains:
        return chains
    innermost = _dot_innermost_clusters(baseline_source)
    if innermost is None:
        return ()
    return tuple(
        chain
        for chain in chains
        if all(innermost.get(target) == _sibling_group_cluster(chain) for target in chain.targets)
    )


def _sibling_group_cluster(chain: SiblingOrderChain) -> str | None:
    """Return the DOT cluster name a chain's rank group is emitted into."""

    if chain.lca_key == -1:
        return None
    return f"cluster_{cast(str, chain.lca_key).replace(':', '_pass')}"


def _dot_innermost_clusters(source: str) -> dict[str, str | None] | None:
    """Map each DOT node to the innermost cluster that contains it.

    A node belongs to every subgraph that names it, so its innermost cluster
    is the deepest cluster path among its references. Top-level nodes map to
    ``None``; nodes named from two unrelated clusters map to a sentinel that
    matches no group. Returns ``None`` when the braces do not balance.
    """

    paths = _dot_node_cluster_paths(source)
    if paths is None:
        return None
    innermost: dict[str, str | None] = {}
    for name, node_paths in paths.items():
        deepest = max(node_paths, key=len)
        if any(path != deepest[: len(path)] for path in node_paths):
            innermost[name] = _AMBIGUOUS_CLUSTER
        else:
            innermost[name] = deepest[-1] if deepest else None
    return innermost


def _dot_node_cluster_paths(source: str) -> dict[str, set[tuple[str, ...]]] | None:
    """Return the cluster paths at which each node is referenced in ``source``."""

    stack: list[str | None] = []
    paths: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    for line in source.splitlines():
        subgraph_match = _DOT_SUBGRAPH_OPEN.match(line)
        if subgraph_match is not None:
            name = _unquote_dot_id(subgraph_match.group(1))
            stack.append(name if name.startswith("cluster") else None)
        elif _DOT_GRAPH_OPEN.match(line) or _DOT_ANONYMOUS_OPEN.match(line):
            stack.append(None)
        elif _DOT_CLOSE.match(line):
            if not stack:
                return None
            stack.pop()
        elif stack:
            cluster_path = tuple(name for name in stack if name is not None)
            for node_name in _dot_statement_node_refs(line):
                paths[node_name].add(cluster_path)
    return None if stack else dict(paths)


def _dot_statement_node_refs(line: str) -> tuple[str, ...]:
    """Return the node names a node or edge statement line references."""

    text = line.strip()
    first = _DOT_ID.match(text)
    if first is None or text.startswith("//"):
        return ()
    rest = text[first.end() :].lstrip()
    if rest.startswith("=") or (first.group(0) in _DOT_KEYWORDS and rest.startswith("[")):
        return ()
    if not rest.startswith("->"):
        return (_unquote_dot_id(first.group(0)),) if not rest or rest.startswith("[") else ()
    second = _DOT_ID.match(rest[2:].lstrip())
    if second is None:
        return ()
    return _unquote_dot_id(first.group(0)), _unquote_dot_id(second.group(0))


def _unquote_dot_id(token: str) -> str:
    """Return a DOT identifier without its quotes and escapes."""

    if len(token) >= 2 and token.startswith('"') and token.endswith('"'):
        return token[1:-1].replace('\\"', '"')
    return token


def _sibling_chain_key(chain: SiblingOrderChain) -> tuple[str, tuple[str, ...]]:
    """Return a stable key for decision reporting."""

    return chain.source_name, chain.targets


def _sibling_order_decision(
    chains: tuple[SiblingOrderChain, ...],
    survivors: tuple[SiblingOrderChain, ...],
    ratios: dict[tuple[str, tuple[str, ...]], float],
) -> SiblingOrderDecision:
    """Build a sibling-order decision record."""

    return SiblingOrderDecision(
        candidate_count=len(chains),
        survivor_count=len(survivors),
        ratios=ratios,
        surviving_keys=tuple(_sibling_chain_key(chain) for chain in survivors),
    )


def _layout_dot_plain(
    source: str,
    rankdir: str,
    captured_edges: list[CapturedForwardEdge],
) -> PlainLayout:
    """Run ``dot -Tplain`` and parse coordinates and real-edge spans."""

    real_edges = {(edge.tail_name, edge.head_name) for edge in captured_edges}
    with tempfile.NamedTemporaryFile("w", suffix=".dot", delete=False) as source_file:
        source_file.write(source)
        source_path = source_file.name
    try:
        proc = _render_utils.run_bounded_subprocess(
            ["dot", "-Tplain", source_path],
            text=True,
            timeout=120,
        )
    finally:
        os.remove(source_path)

    nodes: dict[str, tuple[float, float]] = {}
    pending_edges: list[tuple[str, str]] = []
    for line in proc.stdout.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "node" and len(parts) >= 4:
            nodes[parts[1]] = (float(parts[2]), float(parts[3]))
        elif parts[0] == "edge" and len(parts) >= 4:
            edge_key = (parts[1], parts[2])
            if edge_key in real_edges:
                pending_edges.append(edge_key)

    edge_spans: dict[tuple[str, str], float] = {}
    for edge_key in pending_edges:
        if edge_key[0] in nodes and edge_key[1] in nodes:
            edge_spans[edge_key] = _flow_span(nodes[edge_key[0]], nodes[edge_key[1]], rankdir)
    return PlainLayout(nodes=nodes, edge_spans=edge_spans)


def _sibling_chain_stretch_ratio(
    chain: SiblingOrderChain,
    captured_edges: list[CapturedForwardEdge],
    baseline: PlainLayout,
    candidate: PlainLayout,
) -> float:
    """Return the local incident-edge stretch ratio for ``chain``."""

    local_nodes = {chain.source_name, *chain.targets}
    ratios: list[float] = []
    for edge in captured_edges:
        edge_key = (edge.tail_name, edge.head_name)
        if edge.tail_name not in local_nodes and edge.head_name not in local_nodes:
            continue
        if edge_key not in baseline.edge_spans or edge_key not in candidate.edge_spans:
            continue
        ratios.append(
            candidate.edge_spans[edge_key]
            / max(SIBLING_ORDER_EPSILON, baseline.edge_spans[edge_key])
        )
    return max(ratios, default=1.0)


def _strip_sibling_rank_groups(source: str) -> str:
    """Remove TorchLens sibling-order rank-group blocks from DOT source."""

    lines = source.splitlines()
    stripped: list[str] = []
    skipping = False
    for line in lines:
        if "tl:sibling-order:start" in line:
            skipping = True
            continue
        if "tl:sibling-order:end" in line:
            skipping = False
            continue
        if not skipping:
            stripped.append(line)
    return "\n".join(stripped) + "\n"


__all__ = [
    "_dot_innermost_clusters",
    "_filter_sibling_chains_to_member_cluster",
    "_layout_dot_plain",
    "_queue_sibling_rank_group",
    "_should_order_siblings",
    "_sibling_chain_key",
    "_sibling_chain_stretch_ratio",
    "_sibling_order_decision",
    "_strict_sibling_order_checks_enabled",
    "_strip_sibling_rank_groups",
    "_verify_and_apply_sibling_ordering",
    "_warn_sibling_order_fallback_once",
]
