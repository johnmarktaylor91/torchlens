"""Nested module-cluster (region) emission for the forward DOT pipeline.

Extracted verbatim from ``_render_dot.py`` (renderer-thinning pass): these
helpers turn the decision-complete :class:`RenderIR` region records plus the
legacy module-cluster payloads into nested Graphviz subgraphs, including the
empty-subtree pruning and rank-group parity tripwire.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import TYPE_CHECKING, cast

import graphviz

from ._render_common import (
    GRADIENT_ARROW_COLOR,
    ContainerClusterSpec,
    SiblingOrderChain,
)
from ._render_flow import (
    _emit_container_cluster,
    _emit_sibling_rank_group,
    _get_max_call_depth,
)
from ._render_leaf import _collapsed_module_rolling_suffix
from ._render_utils import compute_module_penwidth, make_module_cluster_attrs

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Any

    from ..data_classes.internal_types import VisualizationOverrides
    from ..data_classes.trace import Trace
    from .render_ir import RenderIROrderingConstraint


def _queue_container_clusters(
    module_cluster_dict: dict[str, Any],
    clusters: Sequence[ContainerClusterSpec],
) -> None:
    """Queue container clusters into their owning module cluster payload."""

    for cluster in clusters:
        if cluster.owner_key == -1:
            continue
        module_cluster_dict[cast(str, cluster.owner_key)]["container_clusters"].append(cluster)


def _setup_combined_special_clusters(
    graphviz_graph: graphviz.Digraph,
    module_cluster_dict: dict[str, Any],
) -> None:
    """Render non-module combined graph clusters.

    Parameters
    ----------
    graphviz_graph:
        Graphviz graph being rendered.
    module_cluster_dict:
        Shared module cluster accumulator.
    """

    cluster_data = module_cluster_dict.get("__intervening__")
    if cluster_data is None:
        return
    with graphviz_graph.subgraph(name="cluster___intervening__") as subgraph:
        subgraph.attr(
            label="intervening grad_fns",
            color=GRADIENT_ARROW_COLOR,
            fontcolor="black",
            style="dashed",
        )
        for node_args in cluster_data.get("nodes", []):
            subgraph.node(**node_args)
        for edge_dict in cluster_data.get("edges", []):
            subgraph.edge(**edge_dict)


def _setup_subgraphs(
    self: Trace,
    graphviz_graph: graphviz.Digraph,
    vis_mode: str,
    module_edge_dict: dict[str, Any],
    overrides: VisualizationOverrides | None = None,
    top_level_rank_groups: Sequence[SiblingOrderChain | RenderIROrderingConstraint] = (),
    regions: Sequence[Any] = (),
    container_ink: Mapping[str, str] | None = None,
) -> None:
    """Build nested Graphviz subgraphs for module clusters.

    Creates the module hierarchy as nested Graphviz subgraphs (clusters),
    placing edges into the appropriate depth level.  Uses a BFS-like
    approach: starts from top-level modules, builds each subgraph via
    ``_setup_subgraphs_recurse``, and pushes child modules onto a stack.

    In **unrolled** mode, each module pass is a separate subgraph (keyed by
    ``"module_addr:call_index"``).  In **rolled** mode, all ops share one
    subgraph (keyed by ``"module_addr"``).

    Subgraph names are prefixed with ``"cluster_"`` (Graphviz convention to
    draw a border box around them).

    Args:
        graphviz_graph: The top-level Graphviz Digraph.
        vis_mode: ``'rolled'`` or ``'unrolled'``.
        module_edge_dict: Dict mapping each module cluster name to
            ``{"edges": [...], "has_input_ancestor": bool}``.
        overrides: Graphviz attribute overrides for module subgraphs.
        container_ink: Skin ink for container clusters (``_theme_cluster_attrs``).
    """
    if "self" not in self.modules:
        return
    region_styles = {
        region.key: dict(region.style) for region in regions if region.kind == "module"
    }
    if vis_mode == "unrolled":
        module_submodule_dict = defaultdict(list)
        for call_label, mpl in self.modules._pass_dict.items():
            module_submodule_dict[call_label] = list(mpl.call_children)
        subgraphs = list(self.modules["self"].ops[0].call_children)
    else:
        module_submodule_dict = defaultdict(list)
        for ml in self.modules:
            if ml.address != "self":
                module_submodule_dict[ml.address] = list(ml.call_children)
        subgraphs = list(self.modules["self"].call_children)

    # Get the max module nesting depth:

    max_call_depth = _get_max_call_depth(subgraphs, module_edge_dict, module_submodule_dict)

    # deque: list.pop(0) shifted the whole queue per module, Theta(M^2) on
    # flat module-heavy graphs before Graphviz even ran (R29, b4 sol MED).
    subgraph_stack: deque[list[str]] = deque([subgraph] for subgraph in subgraphs)
    call_depth = 0
    emitted_rank_groups = 0
    while len(subgraph_stack) > 0:
        parent_graph_list = subgraph_stack.popleft()
        emitted_rank_groups += _setup_subgraphs_recurse(
            self,
            graphviz_graph,
            parent_graph_list,
            module_edge_dict,
            module_submodule_dict,
            subgraph_stack,
            call_depth,
            max_call_depth,
            vis_mode,
            overrides,  # type: ignore[arg-type]
            region_styles,
            container_ink,
        )
    for chain in top_level_rank_groups:
        _emit_sibling_rank_group(graphviz_graph, cast(SiblingOrderChain, chain))
        emitted_rank_groups += 1
    queued_rank_groups = len(top_level_rank_groups) + sum(
        len(data.get("rank_groups", [])) for data in module_edge_dict.values()
    )
    _assert_rank_group_parity(queued_rank_groups, emitted_rank_groups)


def _assert_rank_group_parity(queued: int, emitted: int) -> None:
    """Raise when queued sibling rank groups were not all emitted.

    T9 (grind-p3): a raise, not an assert — this guard runs on the DEFAULT
    ``draw()`` path and ``python -O`` strips asserts, which would let a
    dropped rank group silently reorder rendered siblings.

    Parameters
    ----------
    queued:
        Sibling rank groups queued across the top level and every module
        cluster.
    emitted:
        Sibling rank groups actually emitted into the Graphviz graph.
    """

    if queued != emitted:
        raise RuntimeError(
            f"sibling rank-group emission mismatch: queued {queued} != "
            f"emitted {emitted}; a dropped rank group would silently reorder "
            "rendered siblings"
        )


def _module_subtree_payload_empty(
    module_edge_dict: dict,
    module_submodule_dict: dict,
    subgraph_name_w_pass: str,
    vis_mode: str,
) -> bool:
    """Return whether a module cluster AND all its descendants would be empty.

    r-b6 R19-3: when collapse="max" condenses a module's ops into a segment
    OUTSIDE it, the module's whole subtree accumulates no nodes, edges, or
    rank groups — emitting its cluster produces a labeled dashed husk that
    both misplaces the ops and fabricates a "no input ancestor" claim about
    nothing. The descent branch opens clusters before reaching the leaf
    guard, so emptiness has to be decided for the SUBTREE up front.
    """

    payload_key = (
        subgraph_name_w_pass if vis_mode == "unrolled" else subgraph_name_w_pass.split(":")[0]
    )
    payload = module_edge_dict[payload_key]
    if payload.get("nodes") or payload.get("edges") or payload.get("rank_groups"):
        return False
    return all(
        _module_subtree_payload_empty(module_edge_dict, module_submodule_dict, child, vis_mode)
        for child in module_submodule_dict.get(subgraph_name_w_pass, ())
    )


def _setup_subgraphs_recurse(
    self: Trace,
    starting_subgraph: graphviz.Digraph,
    parent_graph_list: list[str],
    module_edge_dict: dict[str, Any],
    module_submodule_dict: dict[str, list[str]],
    subgraph_stack: deque[list[str]],
    call_depth: int,
    max_call_depth: int,
    vis_mode: str,
    overrides: VisualizationOverrides,
    region_styles: Mapping[str, Mapping[str, str]],
    container_ink: Mapping[str, str] | None = None,
) -> int:
    """Recursively build a single branch of the module subgraph hierarchy.

    Walks down ``parent_graph_list`` (a path from root to leaf module),
    creating nested Graphviz context managers at each level.  When the
    leaf is reached, adds all accumulated edges and pushes child modules
    onto ``subgraph_stack`` for later processing.

    Module border width scales inversely with nesting depth (deeper modules
    get thinner borders) to provide visual hierarchy.

    Args:
        starting_subgraph: The parent Graphviz subgraph to nest into.
        parent_graph_list: Path of module names from root to current target.
        module_edge_dict: Dict mapping each cluster to its edges.
        module_submodule_dict: Dict mapping each cluster to its subclusters.
        subgraph_stack: BFS work queue for remaining branches.
        call_depth: Current position in ``parent_graph_list``.
        max_call_depth: Maximum depth across all branches (for penwidth scaling).
        vis_mode: ``'rolled'`` or ``'unrolled'``.
        overrides: Graphviz attribute overrides.
        region_styles: Resolved module-region attributes keyed by cluster name.
        container_ink: Skin ink for container clusters.
    """
    subgraph_name_w_pass = parent_graph_list[call_depth]
    subgraph_module = subgraph_name_w_pass.split(":")[0]
    if vis_mode == "unrolled":
        cluster_name = f"cluster_{subgraph_name_w_pass.replace(':', '_pass')}"
        subgraph_name = subgraph_name_w_pass
    elif vis_mode == "rolled":
        cluster_name = f"cluster_{subgraph_module}"
        subgraph_name = subgraph_module
    else:
        raise ValueError("vis_mode must be 'rolled' or 'unrolled'")
    sg_ml = self.modules[subgraph_module]
    module_type = sg_ml.class_name
    if (sg_ml.num_calls > 1) and (vis_mode == "unrolled"):
        subgraph_title = subgraph_name_w_pass
    elif (sg_ml.num_calls > 1) and (vis_mode == "rolled"):
        subgraph_title = (
            f"{subgraph_module} (x{sg_ml.num_calls}"
            f"{_collapsed_module_rolling_suffix(self, subgraph_module)})"
        )
    else:
        subgraph_title = subgraph_module

    if call_depth < len(parent_graph_list) - 1:  # we haven't gotten to the bottom yet, keep going.
        if _module_subtree_payload_empty(
            module_edge_dict, module_submodule_dict, parent_graph_list[-1], vis_mode
        ):
            # r-b6 R19-3 (+ r5 empty-duplicate residual): each queued path
            # exists ONLY to nest its FINAL element -- the intermediates were
            # already emitted by earlier queue entries. Checking the current
            # node's subtree (which includes its own already-emitted payload)
            # never pruned these descents, so every hidden-member path
            # re-opened its ancestor clusters as empty duplicate
            # ``subgraph cluster_X { }`` blocks, one per member consulted.
            # Prune on the path TAIL instead: an empty tail contributes
            # nothing, so nothing may be opened.
            return 0
        with starting_subgraph.subgraph(name=cluster_name) as s:
            return _setup_subgraphs_recurse(
                self,
                s,
                parent_graph_list,
                module_edge_dict,
                module_submodule_dict,
                subgraph_stack,
                call_depth + 1,
                max_call_depth,
                vis_mode,
                overrides,
                region_styles,
                container_ink,
            )

    else:  # Leaf of this branch: create the subgraph and add all edges.
        emitted_rank_groups = 0
        cluster_payload = module_edge_dict[subgraph_name]
        # r-b6 R19-3: an empty cluster is never emitted, whatever the module's
        # layer count. When collapse="max" condenses a module's ops into a
        # segment OUTSIDE it, the historical num_layers>1 carve-out still
        # emitted the husk — a labeled, dashed ("no input ancestor") box with
        # ZERO contents, both misplacing the ops and fabricating a
        # disconnection claim about nothing. Module containment for such ops
        # is disclosed on the segment label instead (R19-5).
        if _module_subtree_payload_empty(
            module_edge_dict, module_submodule_dict, subgraph_name_w_pass, vis_mode
        ):
            return emitted_rank_groups
        with starting_subgraph.subgraph(name=cluster_name) as s:
            # Penwidth + cluster attrs come from ``_render_utils`` so the
            # bundle renderer in ``multi_trace/visualization.py`` can build
            # equivalent clusters with the same formula and label format.
            pen_width = compute_module_penwidth(call_depth, max_call_depth)
            if cluster_payload["has_input_ancestor"]:
                line_style = "solid"
            else:
                line_style = "dashed"

            # Module-address-derived titles can contain arbitrary user text
            # (e.g. an ``nn.ModuleDict`` key like ``"score & rank"``), so the
            # title must go through ``html_escape`` like any other
            # user-provided string -- do not assume it is HTML-safe.
            module_args = dict(region_styles.get(subgraph_name, {}))
            if not module_args:
                module_args = make_module_cluster_attrs(
                    title=subgraph_title,
                    module_type=module_type,
                    line_style=line_style,
                    penwidth=pen_width,
                )
                for arg_name, arg_val in overrides.module.items():  # type: ignore[union-attr]
                    if callable(arg_val):
                        module_args[arg_name] = str(arg_val(self, subgraph_name))
                    else:
                        module_args[arg_name] = str(arg_val)
            s.attr(**module_args)
            for chain in cluster_payload.get("rank_groups", []):
                _emit_sibling_rank_group(s, chain)
                emitted_rank_groups += 1
            subgraph_nodes = cluster_payload.get("nodes", [])
            for node_args in subgraph_nodes:
                s.node(**node_args)
            for container_cluster in cluster_payload.get("container_clusters", []):
                _emit_container_cluster(
                    s, cast(ContainerClusterSpec, container_cluster), ink=container_ink
                )
            subgraph_edges = cluster_payload["edges"]
            for edge_dict in subgraph_edges:
                s.edge(**edge_dict)
            subgraph_children = module_submodule_dict[subgraph_name_w_pass]
            for subgraph_child in subgraph_children:  # it's weird but have to go in reverse order.
                subgraph_stack.append(parent_graph_list[:] + [subgraph_child])
        return emitted_rank_groups


__all__ = [
    "_assert_rank_group_parity",
    "_module_subtree_payload_empty",
    "_queue_container_clusters",
    "_setup_combined_special_clusters",
    "_setup_subgraphs",
    "_setup_subgraphs_recurse",
]
