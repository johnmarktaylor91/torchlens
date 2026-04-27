"""Graphviz visualization for :class:`TraceBundle` objects.

This module provides :func:`show_bundle_graph`, the canonical entry point
for rendering a multi-trace bundle, plus three styling modes:

* ``divergence`` -- per-node mean pairwise distance, sequential ``Reds``
  colormap. Use when the bundle is shared-topology and the question is
  "which nodes change the most across traces?".
* ``swarm`` -- per-node coverage fraction, sequential ``viridis``
  colormap. Use for divergent-topology bundles where the question is
  "which nodes did most/all traces visit?".
* ``group_color`` -- categorical colouring by trace-group membership
  (``tab10`` palette). Requires ``bundle.groups`` to be non-empty;
  multi-group nodes get a neutral light-grey shade rather than a blended
  palette colour.

The renderer is deliberately compact and self-contained -- it builds a
fresh :class:`graphviz.Digraph` from the bundle's :class:`Supergraph`
rather than reusing the ModelLog-coupled ``rendering.render_graph``
machinery, which is too entangled with single-trace state to be safely
re-pointed at a multi-trace input within the scope of this dispatch.
What it DOES share with ``rendering.py`` is the file-format dispatch +
view orchestration, factored into ``visualization/_render_utils.py``.

Module clusters are derived from the supergraph's ``module_path`` strings
(``"a.b.c"``-shaped), giving users the same hierarchical layout cue they
get from ``show_model_graph`` even though the underlying ``ModuleLog``
hierarchy is not first-class on a Supergraph.
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    Union,
)

import graphviz
import torch

from ..utils.display import in_notebook
from ..visualization._render_utils import (
    render_dot_to_file,
    strip_known_extension,
)
from .metrics import is_scalar_like, relative_l1_scalar, resolve_metric

if TYPE_CHECKING:  # pragma: no cover - typing-only
    from .bundle import TraceBundle
    from .topology import Supergraph, SupergraphNode


# ---------------------------------------------------------------------------
# Colormap definitions
# ---------------------------------------------------------------------------
#
# All palettes are hardcoded as hex strings to avoid pulling matplotlib (or
# any other plotting library) into torchlens's required dependency set.
# Sampled at 10 stops for sequential maps; tab10 is the canonical
# 10-category categorical palette. White / very light grey is reserved as
# the "no-data / neutral" shade.

# matplotlib's ``Reds`` colormap sampled at 10 evenly-spaced stops.
# Light end is near-white so unchanged nodes fade into the background.
_REDS_HEX: Tuple[str, ...] = (
    "#fff5f0",
    "#fee0d2",
    "#fcbba1",
    "#fc9272",
    "#fb6a4a",
    "#ef3b2c",
    "#cb181d",
    "#a50f15",
    "#67000d",
    "#4a000a",
)

# matplotlib's ``viridis`` colormap sampled at 10 evenly-spaced stops.
# Monotonic luminance, colorblind-safe.
_VIRIDIS_HEX: Tuple[str, ...] = (
    "#440154",
    "#482878",
    "#3e4989",
    "#31688e",
    "#26828e",
    "#1f9e89",
    "#35b779",
    "#6ece58",
    "#b5de2b",
    "#fde725",
)

# matplotlib's ``tab10`` categorical palette (RGB hex). Used for
# group_color mode. Capacity is 10 distinct groups.
_TAB10_HEX: Tuple[str, ...] = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)

# Designated neutral shade for nodes traversed by multiple groups in
# group_color mode. Picked to read clearly against any tab10 colour.
_NEUTRAL_GREY = "#d9d9d9"

# Edge colour for divergence/swarm modes (faded so coloured nodes pop).
_EDGE_COLOR_NEUTRAL = "#cccccc"

# Default text colour for high-saturation node fills (improves contrast on
# dark Reds / dark viridis stops). Heuristic: switch to white once the
# fill darkens past ~halfway through the colormap.
_LIGHT_TEXT = "#ffffff"
_DARK_TEXT = "#000000"


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

ModeLiteral = Literal["auto", "divergence", "swarm", "group_color"]
DirectionLiteral = Literal["bottomup", "topdown", "leftright"]
FileFormatLiteral = Literal["pdf", "png", "svg", "jpg", "jpeg", "bmp", "tif", "tiff"]
VisModeLiteral = Literal["unrolled", "rolled"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_mode(bundle: "TraceBundle", mode: ModeLiteral) -> ModeLiteral:
    """Resolve ``mode='auto'`` to a concrete styling mode.

    Shared topology -> divergence (every node is universal so swarm would
    paint everything the same colour). Divergent topology -> swarm
    (coverage frequency carries information).
    """

    if mode != "auto":
        return mode
    return "divergence" if bundle.is_shared_topology else "swarm"


def _direction_to_rankdir(direction: str) -> str:
    """Translate a vis-direction literal into a Graphviz ``rankdir``."""

    if direction == "bottomup":
        return "BT"
    if direction == "leftright":
        return "LR"
    if direction == "topdown":
        return "TB"
    raise ValueError(
        f"vis_direction must be one of 'bottomup', 'topdown', or 'leftright'; got {direction!r}"
    )


def _sample_colormap(palette: Tuple[str, ...], frac: float) -> str:
    """Return the palette colour closest to fractional position ``frac``.

    ``frac`` is clamped to ``[0, 1]``; ``frac == 0`` -> first stop, ``frac
    == 1`` -> last stop. Linear interpolation between stops would be more
    accurate but adds zero perceptual benefit for rendered Graphviz
    fills, so we round to the nearest stop.
    """

    if not palette:
        return _NEUTRAL_GREY
    if frac <= 0.0:
        return palette[0]
    if frac >= 1.0:
        return palette[-1]
    idx = int(round(frac * (len(palette) - 1)))
    return palette[idx]


def _is_dark_swatch(palette: Tuple[str, ...], frac: float) -> bool:
    """Heuristic: is the colour at ``frac`` dark enough to need light text?"""

    if not palette:
        return False
    if frac <= 0.0:
        return False
    # Past the midpoint of either palette the swatches are dark.
    return frac >= 0.55


def _per_node_distance(
    bundle: "TraceBundle",
    metric: Union[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
) -> Dict[str, Optional[float]]:
    """Compute mean pairwise distance per supergraph node.

    Returns a dict ``{node_name: distance_or_None}``. Nodes traversed by
    fewer than two traces, or missing stored activations, map to
    ``None`` so the caller can render them with a neutral shade.
    """

    metric_fn = resolve_metric(metric)
    sg: "Supergraph" = bundle._supergraph  # type: ignore[attr-defined]
    out: Dict[str, Optional[float]] = {}

    for name in sg.topological_order:
        node = sg.nodes[name]
        tensors: List[torch.Tensor] = []
        for trace_name in node.traces:
            layer = node.layer_refs[trace_name]
            has = getattr(layer, "has_saved_activations", False)
            value = getattr(layer, "activation", None) if has else None
            if isinstance(value, torch.Tensor):
                tensors.append(value)
        if len(tensors) < 2:
            out[name] = None
            continue
        distances: List[float] = []
        # Pairwise mean: mirror the contract in TraceBundle.most_changed.
        for i in range(len(tensors)):
            for j in range(i + 1, len(tensors)):
                ti, tj = tensors[i], tensors[j]
                if is_scalar_like(ti) or is_scalar_like(tj):
                    d = relative_l1_scalar(ti, tj)
                else:
                    try:
                        d = metric_fn(ti, tj)
                    except (RuntimeError, ValueError):
                        # Shape disagreement etc. -- fall back to scalar
                        # form so the mode still produces something
                        # rather than crashing the entire render.
                        d = relative_l1_scalar(ti, tj)
                distances.append(float(d.detach().item()))
        if not distances:
            out[name] = None
            continue
        out[name] = sum(distances) / len(distances)
    return out


def _per_node_coverage(bundle: "TraceBundle") -> Dict[str, float]:
    """Fraction of bundled traces that traversed each supergraph node."""

    sg: "Supergraph" = bundle._supergraph  # type: ignore[attr-defined]
    n = max(len(bundle.names), 1)
    return {name: len(node.traces) / n for name, node in sg.nodes.items()}


def _per_node_groups(
    bundle: "TraceBundle",
) -> Dict[str, frozenset]:
    """Set of group names that traversed each supergraph node.

    Returns ``frozenset`` so the values are hashable and easy to compare;
    a node with no group memberships maps to the empty frozenset.
    """

    groups = bundle.groups
    if not groups:
        return {}

    # Reverse the trace_name -> group(s) mapping so we can ask "which
    # groups does this set of traces belong to?".
    trace_to_groups: Dict[str, Set[str]] = defaultdict(set)
    for group_name, members in groups.items():
        for trace_name in members:
            trace_to_groups[trace_name].add(group_name)

    sg: "Supergraph" = bundle._supergraph  # type: ignore[attr-defined]
    out: Dict[str, frozenset] = {}
    for name, node in sg.nodes.items():
        node_groups: Set[str] = set()
        for trace_name in node.traces:
            node_groups.update(trace_to_groups.get(trace_name, set()))
        out[name] = frozenset(node_groups)
    return out


def _cluster_path(module_path: Optional[str]) -> List[str]:
    """Split a dotted module path into the prefix chain used for clusters.

    Trailing pass-suffixes (``"a.b:1"``) are dropped because Graphviz
    cluster names need to be stable across traces.
    """

    if not module_path:
        return []
    # Strip any ``":N"`` suffix on the leaf (pass index).
    head = module_path.split(":")[0]
    parts = [p for p in head.split(".") if p]
    return parts


def _safe_dot_id(name: str) -> str:
    """Return a Graphviz-safe identifier for arbitrary node/cluster names.

    Graphviz allows fairly liberal node names but quotes characters like
    ``:``/``.``/spaces in the DOT source. We rewrite to a pure
    ``[A-Za-z0-9_]`` form so generated DOT is easy to inspect in tests.
    """

    return "".join(c if c.isalnum() or c == "_" else "_" for c in name)


def _build_module_clusters(
    bundle: "TraceBundle",
) -> Dict[str, List[str]]:
    """Map cluster path (dotted prefix) -> list of node names in it.

    Empty-prefix nodes (no module path) live at the root and don't get
    wrapped in a cluster.
    """

    sg: "Supergraph" = bundle._supergraph  # type: ignore[attr-defined]
    clusters: Dict[str, List[str]] = defaultdict(list)
    for name in sg.topological_order:
        node = sg.nodes[name]
        path = _cluster_path(node.module_path)
        # We attach each node to its leaf cluster only; ancestor clusters
        # are reconstructed lazily when laying out subgraph nesting.
        cluster_key = ".".join(path)
        clusters[cluster_key].append(name)
    return clusters


def _walk_clusters(
    cluster_paths: List[str],
) -> List[List[str]]:
    """Return the list of dotted-prefix paths needed to render clusters.

    For each leaf path we emit every ancestor (prefix) too, so deeply
    nested clusters get their parents created. Ordering preserves the
    first-seen order of leaves for stable DOT output.
    """

    seen: Set[str] = set()
    ordered: List[List[str]] = []
    for leaf in cluster_paths:
        if not leaf:
            continue
        parts = leaf.split(".")
        for depth in range(1, len(parts) + 1):
            prefix = ".".join(parts[:depth])
            if prefix in seen:
                continue
            seen.add(prefix)
            ordered.append(parts[:depth])
    return ordered


def _node_label_lines(
    name: str,
    node: "SupergraphNode",
    *,
    coverage: Optional[float] = None,
    distance: Optional[float] = None,
    show_coverage: bool = False,
) -> List[str]:
    """Build the multi-line label for a supergraph node.

    First line is the node name, second line is the op_type, and (when
    requested) we append a coverage / divergence line. Kept text-only --
    we let Graphviz draw it as an HTML-like table so it stays readable.
    """

    lines = [_html_escape(name)]
    if node.op_type:
        lines.append(f"<I>{_html_escape(node.op_type)}</I>")
    if show_coverage and coverage is not None:
        pct = int(round(coverage * 100))
        lines.append(f"<FONT POINT-SIZE='10'>coverage: {pct}%</FONT>")
    if distance is not None:
        lines.append(f"<FONT POINT-SIZE='10'>div: {distance:.3g}</FONT>")
    return lines


def _html_escape(value: str) -> str:
    """Escape the three Graphviz HTML-label specials."""

    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _format_node_html(lines: List[str]) -> str:
    """Wrap label lines into a Graphviz HTML-like label string."""

    return "<" + "<BR/>".join(lines) + ">"


# ---------------------------------------------------------------------------
# Mode dispatch
# ---------------------------------------------------------------------------


def _style_node_divergence(
    distances: Dict[str, Optional[float]],
    name: str,
) -> Tuple[str, str]:
    """Return (fill_color, font_color) for a node in divergence mode."""

    val = distances.get(name)
    if val is None:
        return _NEUTRAL_GREY, _DARK_TEXT
    # Normalise to [0, 1] using the bundle-wide max so scales stay
    # comparable. Skip nodes with None when computing the max.
    valid = [v for v in distances.values() if v is not None]
    max_val = max(valid) if valid else 0.0
    frac = (val / max_val) if max_val > 0 else 0.0
    fill = _sample_colormap(_REDS_HEX, frac)
    font = _LIGHT_TEXT if _is_dark_swatch(_REDS_HEX, frac) else _DARK_TEXT
    return fill, font


def _style_node_swarm(
    coverage_map: Dict[str, float],
    name: str,
) -> Tuple[str, str]:
    """Return (fill_color, font_color) for a node in swarm mode."""

    cov = coverage_map.get(name, 0.0)
    fill = _sample_colormap(_VIRIDIS_HEX, cov)
    font = _LIGHT_TEXT if _is_dark_swatch(_VIRIDIS_HEX, cov) else _DARK_TEXT
    return fill, font


def _style_node_group_color(
    group_map: Dict[str, frozenset],
    group_index: Dict[str, int],
    name: str,
) -> Tuple[str, str]:
    """Return (fill_color, font_color) for a node in group_color mode."""

    groups = group_map.get(name, frozenset())
    if not groups:
        return _NEUTRAL_GREY, _DARK_TEXT
    if len(groups) > 1:
        # Multi-group node -- neutral shade rather than blending.
        return _NEUTRAL_GREY, _DARK_TEXT
    only_group = next(iter(groups))
    idx = group_index.get(only_group, 0)
    fill = _TAB10_HEX[idx % len(_TAB10_HEX)]
    return fill, _LIGHT_TEXT


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


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
        for future parity with ``show_model_graph``.
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
        graphs; ``'backward'`` raises ``ValueError``.
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
    if mode == "group_color" and len(bundle.groups) > len(_TAB10_HEX):
        raise ValueError(
            f"mode='group_color' supports up to {len(_TAB10_HEX)} groups; "
            f"got {len(bundle.groups)}. Reduce the group count or extend "
            "the palette."
        )

    resolved_mode = _resolve_mode(bundle, mode)

    # ----- per-node styling data -----------------------------------------
    distances: Dict[str, Optional[float]] = {}
    coverage_map: Dict[str, float] = {}
    group_map: Dict[str, frozenset] = {}
    group_index: Dict[str, int] = {}

    if resolved_mode == "divergence":
        distances = _per_node_distance(bundle, metric)
    if resolved_mode == "swarm" or show_coverage:
        coverage_map = _per_node_coverage(bundle)
    if resolved_mode == "group_color":
        group_map = _per_node_groups(bundle)
        # Stable, dict-insertion-order indexing; bundle.groups is a copy
        # but the underlying dict preserves insertion order in CPython.
        group_index = {gname: i for i, gname in enumerate(bundle.groups)}

    # ----- DOT graph build ------------------------------------------------
    sg: "Supergraph" = bundle._supergraph  # type: ignore[attr-defined]
    rankdir = _direction_to_rankdir(vis_orientation)

    n_traces = len(bundle.names)
    n_nodes = len(sg.nodes)
    # HTML-style labels are bracketed by ``<>`` so any literal ``<``/``>``
    # inside the body must be escaped or graphviz refuses to parse the
    # label. The caption is the only place we historically embed an
    # arrow ("auto -> swarm"); escape it via the standard HTML entities.
    caption_lines = [
        "<B>TraceBundle</B>",
        f"{n_traces} traces, {n_nodes} supergraph nodes",
        f"mode={_html_escape(resolved_mode)}",
    ]
    if mode == "auto" and resolved_mode != "auto":
        caption_lines[-1] = f"mode=auto -&gt; {_html_escape(resolved_mode)}"
    if resolved_mode == "group_color":
        caption_lines.append("groups: " + _html_escape(", ".join(sorted(bundle.groups))))
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

    # Build cluster lookup (leaf cluster path -> node names).
    leaf_cluster_to_nodes = _build_module_clusters(bundle)

    # Pre-compute per-node fill / font colours so cluster nesting is the
    # only remaining concern.
    node_styles: Dict[str, Tuple[str, str]] = {}
    node_labels: Dict[str, str] = {}
    for name in sg.topological_order:
        node = sg.nodes[name]
        if resolved_mode == "divergence":
            fill, font = _style_node_divergence(distances, name)
        elif resolved_mode == "swarm":
            fill, font = _style_node_swarm(coverage_map, name)
        else:  # group_color
            fill, font = _style_node_group_color(group_map, group_index, name)
        node_styles[name] = (fill, font)
        label_lines = _node_label_lines(
            name,
            node,
            coverage=coverage_map.get(name) if coverage_map else None,
            distance=distances.get(name) if distances else None,
            show_coverage=show_coverage and resolved_mode == "swarm",
        )
        node_labels[name] = _format_node_html(label_lines)

    # Build the cluster hierarchy lazily: for each cluster path we open a
    # subgraph and emit its nodes; nested clusters open inside their
    # parents. We do this by sorting leaves by depth and walking each
    # ancestor chain.

    # Map dotted-prefix -> set of direct child dotted-prefixes (one
    # level deeper) and direct nodes that live in that prefix.
    prefix_children: Dict[str, Set[str]] = defaultdict(set)
    prefix_nodes: Dict[str, List[str]] = defaultdict(list)
    for leaf, names_at_leaf in leaf_cluster_to_nodes.items():
        prefix_nodes[leaf].extend(names_at_leaf)
        if not leaf:
            continue
        parts = leaf.split(".")
        for depth in range(1, len(parts) + 1):
            child = ".".join(parts[:depth])
            parent = ".".join(parts[: depth - 1])
            prefix_children[parent].add(child)

    # ROOT nodes: those whose leaf cluster is "" (no module path).
    root_nodes = list(prefix_nodes.get("", []))
    for name in root_nodes:
        fill, font = node_styles[name]
        dot.node(
            _safe_dot_id(name),
            label=node_labels[name],
            fillcolor=fill,
            fontcolor=font,
            color=fill,
        )

    # Nested clusters: walk depth-first, attaching nodes to the right level.
    def _emit_subgraph(parent_dot: graphviz.Digraph, prefix: str) -> None:
        cluster_label = prefix.split(".")[-1] if prefix else ""
        cluster_name = f"cluster_{_safe_dot_id(prefix)}"
        with parent_dot.subgraph(name=cluster_name) as sub:
            sub.attr(
                label=f"<<B>@{_html_escape(cluster_label)}</B>>",
                labelloc="b",
                style="filled",
                fillcolor="white",
                color="#888888",
                penwidth="2",
            )
            for child_prefix in sorted(prefix_children.get(prefix, set())):
                _emit_subgraph(sub, child_prefix)
            for node_name in prefix_nodes.get(prefix, []):
                fill, font = node_styles[node_name]
                sub.node(
                    _safe_dot_id(node_name),
                    label=node_labels[node_name],
                    fillcolor=fill,
                    fontcolor=font,
                    color=fill,
                )

    for top_prefix in sorted(prefix_children.get("", set())):
        _emit_subgraph(dot, top_prefix)

    # Edges. In group_color mode we colour edges by the dominant group of
    # the traces that traversed the edge; in other modes we use a faded
    # neutral so coloured nodes pop.
    for (parent, child), trace_set in sg.edges.items():
        edge_color = _resolve_edge_color(
            trace_set,
            resolved_mode=resolved_mode,
            bundle=bundle,
            group_index=group_index,
        )
        dot.edge(
            _safe_dot_id(parent),
            _safe_dot_id(child),
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


def _resolve_edge_color(
    trace_set: Set[str],
    *,
    resolved_mode: ModeLiteral,
    bundle: "TraceBundle",
    group_index: Dict[str, int],
) -> str:
    """Pick a stroke colour for a supergraph edge given the traversing traces.

    In ``group_color`` mode an edge gets the colour of the group that
    traversed it (or neutral grey if multiple groups did). In divergence
    and swarm modes we keep edges neutral so the coloured nodes stay the
    visual focus.
    """

    if resolved_mode != "group_color":
        return _EDGE_COLOR_NEUTRAL

    groups = bundle.groups
    if not groups or not group_index:
        return _EDGE_COLOR_NEUTRAL

    trace_to_groups: Dict[str, Set[str]] = defaultdict(set)
    for group_name, members in groups.items():
        for trace_name in members:
            trace_to_groups[trace_name].add(group_name)

    edge_groups: Set[str] = set()
    for trace_name in trace_set:
        edge_groups.update(trace_to_groups.get(trace_name, set()))

    if len(edge_groups) == 1:
        only = next(iter(edge_groups))
        return _TAB10_HEX[group_index.get(only, 0) % len(_TAB10_HEX)]
    return _EDGE_COLOR_NEUTRAL


__all__ = ["show_bundle_graph"]
