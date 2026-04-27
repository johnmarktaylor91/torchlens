"""Cluster construction helpers for the multi-trace bundle renderer.

Private helper module: not part of the public API. Kept separate from
``multi_trace/visualization.py`` because the cluster code is the largest
self-contained chunk of bundle-specific orchestration and the renderer
is easier to read when these helpers live in their own file.

These helpers convert a :class:`TraceBundle`'s supergraph into the data
the cluster emitter needs:

* a per-cluster list of supergraph nodes
* a parent->child cluster map for nesting
* a display title per cluster (with ModelLog-equivalent
  ``@<module>:N`` pass suffixes when the underlying module is multi-pass)
* a "cluster has any traversal" flag per cluster
* the max nesting depth (1-based, matches
  ``rendering._get_max_nesting_depth``)

The actual Graphviz emission stays in ``visualization.py`` so the
renderer can keep all rendering decisions in one place.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

if TYPE_CHECKING:  # pragma: no cover - typing-only
    from .bundle import TraceBundle
    from .topology import SupergraphNode


def _split_leaf_pass(component: str) -> Tuple[str, Optional[str]]:
    """Split ``"name:pass"`` into ``(name, pass)``; ``"name"`` -> ``(name, None)``."""

    if ":" in component:
        base, _, suffix = component.rpartition(":")
        return base, suffix or None
    return component, None


def safe_dot_id(name: str) -> str:
    """Return a Graphviz-safe identifier for arbitrary node/cluster names.

    Graphviz allows fairly liberal node names but quotes characters like
    ``:``/``.``/spaces in the DOT source.  We rewrite to a pure
    ``[A-Za-z0-9_]`` form so generated DOT is easy to inspect in tests.
    """

    return "".join(c if c.isalnum() or c == "_" else "_" for c in name)


def _supergraph_node_module_chain(node: "SupergraphNode") -> List[str]:
    """Return the cluster-path chain for a supergraph node.

    Reads ``containing_modules`` from the first contributing trace's
    ``LayerLog`` -- the same source the supergraph uses to populate
    ``module_path``.  Falls back to a single-element chain derived from
    ``module_path`` if ``containing_modules`` is unavailable for any
    reason (e.g. a synthetic LayerLog from a test fixture).

    The returned list elements are full module-pass addresses (e.g.
    ``"level21:1"``, ``"level21.level11:1"``).  Each element is used
    verbatim as a cluster key so two passes of the same module live in
    distinct clusters, matching ``rendering.py``'s unrolled-mode
    behaviour.
    """

    if not node.layer_refs:
        return []
    first_trace = node.traces[0] if node.traces else next(iter(node.layer_refs))
    layer = node.layer_refs[first_trace]
    chain = getattr(layer, "containing_modules", None)
    if isinstance(chain, list) and chain:
        return [str(item) for item in chain if item is not None]
    if node.module_path:
        return [str(node.module_path)]
    return []


def _consensus_num_passes(bundle: "TraceBundle", node_names: List[str]) -> Optional[int]:
    """Return the consensus ``num_passes`` for a cluster, or ``None`` if mixed.

    For each direct supergraph node in the cluster, sample
    ``num_passes`` from every traversing trace's LayerLog.  If every
    sample agrees on a single value > 1 we return it -- the cluster is
    consistently multi-pass and merits an ``(xN)`` suffix in the
    rolled-mode title.  Anything else (single-pass everywhere, divergent
    across traces, or accessor-missing) returns ``None`` so the caller
    falls back to the plain title.
    """

    if not node_names:
        return None

    sg = bundle._supergraph  # type: ignore[attr-defined]
    seen: Set[int] = set()
    for node_name in node_names:
        node = sg.nodes.get(node_name)
        if node is None:
            continue
        for layer in node.layer_refs.values():
            np_attr = getattr(layer, "num_passes", None)
            if not isinstance(np_attr, int) or np_attr <= 0:
                return None
            seen.add(np_attr)
    if len(seen) != 1:
        return None
    consensus = next(iter(seen))
    if consensus <= 1:
        return None
    return consensus


def collect_cluster_metadata(
    bundle: "TraceBundle",
) -> Tuple[
    Dict[str, List[str]],  # cluster key -> direct nodes
    Dict[str, Set[str]],  # cluster key -> direct child cluster keys
    Dict[str, str],  # cluster key -> display title
    Dict[str, bool],  # cluster key -> traversed by any trace
    int,  # 1-based max nesting depth
]:
    """Walk the supergraph and gather everything the cluster renderer needs.

    Returns a tuple of:

    * ``cluster_nodes`` -- map cluster key -> list of supergraph node
      names that live directly inside that leaf cluster.  ``""`` collects
      root nodes (no module).
    * ``cluster_children`` -- map cluster key -> set of direct-child
      cluster keys.  ``""`` maps to the set of top-level clusters.
    * ``cluster_title`` -- map cluster key -> display title used in the
      ``@<title>`` cluster label.  Mirrors ``rendering.py`` rolled-mode
      conventions: single-pass modules drop the ``:N`` suffix (so ``fc1``
      not ``fc1:1``); multi-pass modules show ``base (xN)`` when every
      traversing trace agrees on the pass count, falling back to the
      plain base name when traces diverge.
    * ``cluster_has_traversal`` -- map cluster key -> True iff any trace
      populated a node inside the cluster.  Bundle clusters always have
      traversal because the supergraph wouldn't have created an empty
      cluster, but the dict is exposed for parity with
      ``rendering._setup_subgraphs_recurse`` and to leave room for
      future "skip empty branches" behaviour.
    * ``max_nesting_depth`` -- 1-based depth of the deepest cluster
      (i.e. the number of levels in the deepest chain).  Mirrors the
      value returned by ``rendering._get_max_nesting_depth`` so the
      shared ``compute_module_penwidth`` formula produces matching
      values for ModelLog and bundle clusters at the same depth.
    """

    sg = bundle._supergraph  # type: ignore[attr-defined]

    cluster_nodes: Dict[str, List[str]] = defaultdict(list)
    cluster_children: Dict[str, Set[str]] = defaultdict(set)
    cluster_has_traversal: Dict[str, bool] = defaultdict(bool)

    # Track how many distinct pass-suffixes appear for each base module
    # address.  ModelLog shows ``@fc1`` for single-pass modules and
    # ``@fc1:N`` for unrolled-mode multi-pass.  The bundle operates on
    # rolled-equivalent supergraph nodes so it primarily emits ``@fc1``
    # or ``@fc1 (xN)``; the suffix-set is still tracked because divergent
    # topologies can put different passes of the same module in distinct
    # supergraph clusters (cf. NestedModules' level21 called 3x).
    base_to_passes: Dict[str, Set[Optional[str]]] = defaultdict(set)
    cluster_keys: Set[str] = set()

    max_depth = 1  # mirrors ``rendering._get_max_nesting_depth`` floor
    for name in sg.topological_order:
        node = sg.nodes[name]
        chain = _supergraph_node_module_chain(node)
        if not chain:
            cluster_nodes[""].append(name)
            continue
        leaf_key = chain[-1]
        cluster_nodes[leaf_key].append(name)

        # Walk the chain registering each level's parent->child relation
        # and collecting pass info to drive the title format.
        prev_key = ""
        for depth, key in enumerate(chain):
            cluster_has_traversal[key] = True
            cluster_keys.add(key)
            base_addr, pass_suffix = _split_leaf_pass(key)
            base_to_passes[base_addr].add(pass_suffix)
            cluster_children[prev_key].add(key)
            prev_key = key
            max_depth = max(max_depth, depth + 1)

    cluster_title: Dict[str, str] = {}
    for key in cluster_keys:
        base_addr, pass_suffix = _split_leaf_pass(key)
        # Decide on ``base`` vs ``base:N`` first (the "is it multi-pass at
        # the unrolled level" question), then decorate with ``(xN)`` if the
        # cluster's underlying LayerLogs are themselves multi-pass.
        if pass_suffix is None:
            title_base = key
        else:
            distinct_pass_suffixes = {p for p in base_to_passes[base_addr] if p is not None}
            if len(distinct_pass_suffixes) > 1:
                # Multiple passes show up at the supergraph level
                # (rolled-style equivalent of unrolled per-pass clusters).
                title_base = key
            else:
                title_base = base_addr

        # Append a ``(xN)`` suffix when every traversing LayerLog agrees
        # on the same pass count > 1 -- the rolled-mode signal that the
        # cluster represents N invocations of the same module.
        consensus = _consensus_num_passes(bundle, cluster_nodes.get(key, []))
        if consensus is not None:
            title_base = f"{title_base} (x{consensus})"

        cluster_title[key] = title_base

    return (
        dict(cluster_nodes),
        dict(cluster_children),
        cluster_title,
        dict(cluster_has_traversal),
        max_depth,
    )
