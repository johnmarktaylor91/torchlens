"""Graphviz visualization for :class:`TraceBundle` objects.

This module provides :func:`show_bundle_graph`, the canonical entry
point for rendering a multi-trace bundle, plus three styling modes:

* ``divergence`` -- per-node mean pairwise distance, sequential ``Reds``
  colormap.  Use when the bundle is shared-topology and the question is
  "which nodes change the most across traces?".
* ``swarm`` -- per-node coverage fraction, sequential ``viridis``
  colormap.  Use for divergent-topology bundles where the question is
  "which nodes did most/all traces visit?".
* ``group_color`` -- categorical colouring by trace-group membership
  (``tab10`` palette).  Requires ``bundle.groups`` to be non-empty;
  multi-group nodes get a neutral light-grey shade rather than a blended
  palette colour.

Implementation factoring:

* ``torchlens/visualization/_render_utils.py`` holds the renderer-agnostic
  Graphviz primitives (file-format dispatch, layout direction, HTML
  escaping, module-cluster styling primitives).  Both this file and
  ``rendering.py`` go through that module.
* ``torchlens/multi_trace/_bundle_clusters.py`` builds the pass-aware
  module cluster hierarchy from the supergraph.
* ``torchlens/multi_trace/_bundle_styling.py`` holds the colormap tables,
  per-node aggregation, and per-node fill/font/label resolution per mode.

This file is the bundle-side orchestrator: it composes the helpers,
constructs the Graphviz Digraph, and dispatches to file output.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Callable,
    Literal,
    Optional,
    Union,
)

import graphviz
import torch

from ..utils.display import in_notebook
from ..visualization._render_utils import (
    compute_module_penwidth,
    direction_to_rankdir,
    html_escape,
    make_module_cluster_attrs,
    render_dot_to_file,
    strip_known_extension,
)
from ._bundle_clusters import collect_cluster_metadata, safe_dot_id
from ._bundle_styling import (
    ModeLiteral,
    TAB10_HEX,
    compute_node_styles,
    resolve_edge_color,
    resolve_mode,
)

if TYPE_CHECKING:  # pragma: no cover - typing-only
    from .bundle import TraceBundle


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

DirectionLiteral = Literal["bottomup", "topdown", "leftright"]
FileFormatLiteral = Literal["pdf", "png", "svg", "jpg", "jpeg", "bmp", "tif", "tiff"]
VisModeLiteral = Literal["unrolled", "rolled"]


def show_bundle_graph(
    bundle: "TraceBundle",
    vis_outpath: Optional[str] = None,
    mode: ModeLiteral = "auto",
    *,
    metric: Union[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = "cosine",
    show_coverage: bool = False,
    vis_opt: VisModeLiteral = "unrolled",
    vis_format: FileFormatLiteral = "pdf",
    vis_orientation: DirectionLiteral = "bottomup",
    save_only: bool = False,
    direction: Literal["forward", "backward", "both"] = "forward",
    return_dot: bool = False,
) -> Optional[str]:
    """Render a :class:`TraceBundle` as a Graphviz graph.

    Parameters
    ----------
    bundle:
        The :class:`TraceBundle` to render.
    vis_outpath:
        Output path. Extension is inferred from ``vis_format`` and any
        recognised extension on ``vis_outpath`` is stripped first. If
        ``None`` (default), writes to ``"bundle_graph"`` in the current
        working directory -- mirrors ``show_model_graph``'s default.
    mode:
        Styling mode: ``'auto'`` (picks divergence for shared topology,
        swarm for divergent), ``'divergence'``, ``'swarm'``, or
        ``'group_color'``.
    metric:
        Distance metric for divergence mode. Accepts the same strings as
        :func:`torchlens.multi_trace.metrics.resolve_metric` (``cosine``,
        ``relative_l2``, ``pearson``) or a callable
        ``(Tensor, Tensor) -> Tensor``.
    show_coverage:
        For swarm mode, append a ``coverage: NN%`` row to each node's
        label. Ignored in other modes.
    vis_opt:
        Layout flavour, ``'unrolled'`` (default) or ``'rolled'``.
        Currently advisory -- the bundle renderer always treats each
        supergraph node as a single rendered node; the kwarg is reserved
        for future parity with ``show_model_graph``.  Rolled bundle
        rendering is logged as a deferred follow-up in
        ``.project-context/todos.md`` under the Multi-trace V2 section.
    vis_format:
        Output file format (``pdf``/``png``/``svg``/...).
    vis_orientation:
        Layout direction: ``'bottomup'`` (default), ``'topdown'``, or
        ``'leftright'``. Mirrors ``show_model_graph``'s ``direction``.
    save_only:
        If ``True``, write the output file but skip opening a viewer.
    direction:
        Reserved for symmetry with ``show_model_graph``'s forward/backward
        mode toggle. Bundle visualization currently only renders forward
        graphs; ``'backward'`` raises ``ValueError``.  Backward bundle
        topology is logged as a deferred follow-up in
        ``.project-context/todos.md`` under the Multi-trace V2 section.
    return_dot:
        Internal/test hook -- if ``True``, return the generated DOT
        source instead of writing to disk. Used by tests to inspect
        styling without spawning Graphviz.

    Returns
    -------
    str | None
        The DOT source when ``return_dot=True``; otherwise ``None``.

    Raises
    ------
    ValueError
        If ``mode='group_color'`` and ``bundle.groups`` is empty, if the
        mode literal is unknown, or if ``direction`` is anything other
        than ``'forward'``.

    Notes
    -----
    Module clusters are rendered with ModelLog-style aesthetics: per-
    depth penwidth scaling (outermost thickest, deepest thinnest, using
    the same min/max constants as ``show_model_graph``), separate
    clusters for each pass of multi-pass modules, and ``@module:N``
    pass-suffix labels when the supergraph contains more than one pass
    of the underlying base module.  Single-pass modules drop the suffix
    just like ModelLog does.  The bundle's supergraph collapses canonical
    nodes by ``(containing_module, func_name)`` fingerprint, so divergent
    topologies naturally end up in distinct clusters per pass without
    special-case handling.
    """

    # ----- arg validation -------------------------------------------------
    if mode not in ("auto", "divergence", "swarm", "group_color"):
        raise ValueError(
            f"mode must be one of 'auto', 'divergence', 'swarm', or 'group_color'; got {mode!r}"
        )
    if vis_opt not in ("unrolled", "rolled"):
        raise ValueError(f"vis_opt must be 'unrolled' or 'rolled'; got {vis_opt!r}")
    if direction != "forward":
        raise ValueError(
            "show_bundle_graph currently only supports direction='forward'; "
            "backward graph rendering is not part of the multi-trace "
            "visualization phase."
        )

    if mode == "group_color" and not bundle.groups:
        raise ValueError(
            "mode='group_color' requires bundle.groups to be non-empty; "
            "construct the bundle with `tl.bundle(traces, names=..., "
            "groups={'a': ['n1'], 'b': ['n2']})`."
        )
    if mode == "group_color" and len(bundle.groups) > len(TAB10_HEX):
        raise ValueError(
            f"mode='group_color' supports up to {len(TAB10_HEX)} groups; "
            f"got {len(bundle.groups)}. Reduce the group count or extend "
            "the palette."
        )

    resolved_mode = resolve_mode(bundle, mode)

    # ----- per-node aggregation + styling --------------------------------
    # Stable, dict-insertion-order indexing; ``bundle.groups`` is a copy
    # but the underlying dict preserves insertion order in CPython.
    group_index = (
        {gname: i for i, gname in enumerate(bundle.groups)}
        if resolved_mode == "group_color"
        else {}
    )
    node_styles, node_labels, _, _, _ = compute_node_styles(
        bundle,
        resolved_mode=resolved_mode,
        metric=metric,
        show_coverage=show_coverage,
        group_index=group_index,
    )

    # ----- DOT graph build ------------------------------------------------
    sg = bundle._supergraph  # type: ignore[attr-defined]
    rankdir = direction_to_rankdir(vis_orientation)

    n_traces = len(bundle.names)
    n_nodes = len(sg.nodes)
    # HTML-style labels are bracketed by ``<>`` so any literal ``<``/``>``
    # inside the body must be escaped or graphviz refuses to parse the
    # label. The caption is the only place we historically embed an
    # arrow ("auto -> swarm"); escape it via the standard HTML entities.
    caption_lines = [
        "<B>TraceBundle</B>",
        f"{n_traces} traces, {n_nodes} supergraph nodes",
        f"mode={html_escape(resolved_mode)}",
    ]
    if mode == "auto" and resolved_mode != "auto":
        caption_lines[-1] = f"mode=auto -&gt; {html_escape(resolved_mode)}"
    if resolved_mode == "group_color":
        caption_lines.append("groups: " + html_escape(", ".join(sorted(bundle.groups))))
    graph_caption = "<" + "<BR ALIGN='LEFT'/>".join(caption_lines) + "<BR ALIGN='LEFT'/>>"

    dot = graphviz.Digraph(
        name="trace_bundle",
        comment="Computational graph for a TorchLens TraceBundle",
        format=vis_format,
    )
    dot.graph_attr.update(
        {
            "rankdir": rankdir,
            "label": graph_caption,
            "labelloc": "t",
            "labeljust": "left",
            "ordering": "out",
            "compound": "true",
        }
    )
    dot.node_attr.update({"ordering": "out", "shape": "box", "style": "filled"})

    # ----- cluster construction ------------------------------------------
    (
        cluster_nodes,
        cluster_children,
        cluster_title,
        cluster_has_traversal,
        max_nesting_depth,
    ) = collect_cluster_metadata(bundle)

    # ROOT nodes: those whose cluster key is "" (no module).
    for name in cluster_nodes.get("", []):
        fill, font = node_styles[name]
        dot.node(
            safe_dot_id(name),
            label=node_labels[name],
            fillcolor=fill,
            fontcolor=font,
            color=fill,
        )

    # Nested clusters: walk depth-first, attaching nodes to the right level.
    # Penwidth and style mirror rendering._setup_subgraphs_recurse so bundle
    # clusters read identically to single-trace clusters at typical depths.
    def _emit_subgraph(parent_dot: graphviz.Digraph, cluster_key: str, depth: int) -> None:
        title = cluster_title.get(cluster_key, cluster_key)
        cluster_name = f"cluster_{safe_dot_id(cluster_key)}"
        # Match ``rendering.py`` semantics: ``has_input_ancestor=True`` ->
        # ``filled,solid``, otherwise ``filled,dashed``.  At the bundle
        # level we treat "any traversal at this cluster" as the positive
        # signal.  A cluster the supergraph created with no traversal is
        # impossible by construction, but the dict is exposed so future
        # filtering (e.g. "skip clusters all traces avoided") slots in.
        line_style = "solid" if cluster_has_traversal.get(cluster_key, False) else "dashed"
        # ``compute_module_penwidth`` accepts 0-based depth; 1-based
        # max_nesting_depth.  Depth 0 -> max penwidth, deepest -> min.
        penwidth = compute_module_penwidth(depth, max_nesting_depth)
        cluster_args = make_module_cluster_attrs(
            title=title,
            module_type=None,  # supergraph doesn't preserve module class
            line_style=line_style,
            penwidth=penwidth,
        )
        with parent_dot.subgraph(name=cluster_name) as sub:
            sub.attr(**cluster_args)
            # Preserve a neutral border colour cue so module clusters
            # remain visually distinct against white-background themes.
            sub.attr(color="#888888")
            for child_key in sorted(cluster_children.get(cluster_key, set())):
                _emit_subgraph(sub, child_key, depth + 1)
            for node_name in cluster_nodes.get(cluster_key, []):
                fill, font = node_styles[node_name]
                sub.node(
                    safe_dot_id(node_name),
                    label=node_labels[node_name],
                    fillcolor=fill,
                    fontcolor=font,
                    color=fill,
                )

    for top_cluster_key in sorted(cluster_children.get("", set())):
        _emit_subgraph(dot, top_cluster_key, 0)

    # Edges: in group_color mode we colour edges by the dominant group of
    # the traces that traversed the edge; in other modes we use a faded
    # neutral so coloured nodes stay the visual focus.
    for (parent, child), trace_set in sg.edges.items():
        edge_color = resolve_edge_color(
            trace_set,
            resolved_mode=resolved_mode,
            bundle=bundle,
            group_index=group_index,
        )
        dot.edge(
            safe_dot_id(parent),
            safe_dot_id(child),
            color=edge_color,
            penwidth="1.5",
        )

    if return_dot:
        return dot.source

    # ----- output ---------------------------------------------------------
    outpath = vis_outpath if vis_outpath is not None else "bundle_graph"
    outpath = strip_known_extension(outpath)

    if in_notebook() and not save_only:  # pragma: no cover - env-dependent
        from IPython.display import display

        display(dot)

    timeout_warning = (
        f"Graphviz render timed out for TraceBundle graph "
        f"({n_nodes} nodes). DOT source preserved on disk."
    )
    render_dot_to_file(
        dot,
        outpath,
        vis_format,
        save_only,
        timeout_warning=timeout_warning,
    )
    return None


__all__ = ["show_bundle_graph"]
