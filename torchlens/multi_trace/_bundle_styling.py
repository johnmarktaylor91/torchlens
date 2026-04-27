"""Per-node styling helpers for the multi-trace bundle renderer.

Private helper module: not part of the public API. Holds the colormap
tables and per-node aggregation/styling logic so
``multi_trace/visualization.py`` can stay focused on Graphviz orchestration.

The public entry point :func:`compute_node_styles` returns the per-node
fill colour, font colour, and label-line list for a given mode -- a clean
seam between bundle-aware aggregation (here) and Graphviz emission (in
``visualization.py``).
"""

from __future__ import annotations

from collections import defaultdict
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    Union,
)

import torch

from ..visualization._render_utils import format_node_html, html_escape
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
REDS_HEX: Tuple[str, ...] = (
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
VIRIDIS_HEX: Tuple[str, ...] = (
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
TAB10_HEX: Tuple[str, ...] = (
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
NEUTRAL_GREY = "#d9d9d9"

# Edge colour for divergence/swarm modes (faded so coloured nodes pop).
EDGE_COLOR_NEUTRAL = "#cccccc"

# Default text colour for high-saturation node fills (improves contrast on
# dark Reds / dark viridis stops). Heuristic: switch to white once the
# fill darkens past ~halfway through the colormap.
LIGHT_TEXT = "#ffffff"
DARK_TEXT = "#000000"


ModeLiteral = Literal["auto", "divergence", "swarm", "group_color"]


# ---------------------------------------------------------------------------
# Colormap sampling
# ---------------------------------------------------------------------------


def sample_colormap(palette: Tuple[str, ...], frac: float) -> str:
    """Return the palette colour closest to fractional position ``frac``.

    ``frac`` is clamped to ``[0, 1]``; ``frac == 0`` -> first stop, ``frac
    == 1`` -> last stop. Linear interpolation between stops would be more
    accurate but adds zero perceptual benefit for rendered Graphviz
    fills, so we round to the nearest stop.
    """

    if not palette:
        return NEUTRAL_GREY
    if frac <= 0.0:
        return palette[0]
    if frac >= 1.0:
        return palette[-1]
    idx = int(round(frac * (len(palette) - 1)))
    return palette[idx]


def is_dark_swatch(palette: Tuple[str, ...], frac: float) -> bool:
    """Heuristic: is the colour at ``frac`` dark enough to need light text?"""

    if not palette:
        return False
    if frac <= 0.0:
        return False
    # Past the midpoint of either palette the swatches are dark.
    return frac >= 0.55


# ---------------------------------------------------------------------------
# Mode resolution + per-node aggregation
# ---------------------------------------------------------------------------


def resolve_mode(bundle: "TraceBundle", mode: ModeLiteral) -> ModeLiteral:
    """Resolve ``mode='auto'`` to a concrete styling mode.

    Shared topology -> divergence (every node is universal so swarm would
    paint everything the same colour). Divergent topology -> swarm
    (coverage frequency carries information).
    """

    if mode != "auto":
        return mode
    return "divergence" if bundle.is_shared_topology else "swarm"


def _layer_tensor(
    layer: Any,
    *,
    field: str,
    has_field: str,
) -> Optional[torch.Tensor]:
    """Pull a tensor off a single-pass LayerLog without crashing on multi-pass.

    LayerLog raises ``ValueError`` rather than ``AttributeError`` for fields
    that depend on a specific pass number; multi-pass LayerLogs reach us via
    ``SupergraphNode.layer_refs`` and we treat them as "no tensor available"
    rather than letting the divergence/swarm pipeline crash. Multi-pass
    activation rendering is tracked in todos.md (Multi-trace V2).
    """

    try:
        has = getattr(layer, has_field, False)
    except (ValueError, AttributeError):
        return None
    if not has:
        return None
    try:
        value = getattr(layer, field, None)
    except (ValueError, AttributeError):
        return None
    if isinstance(value, torch.Tensor):
        return value
    return None


def per_node_distance(
    bundle: "TraceBundle",
    metric: Union[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
) -> Dict[str, Optional[float]]:
    """Compute mean pairwise distance per supergraph node.

    Returns a dict ``{node_name: distance_or_None}``.  Nodes traversed by
    fewer than two traces, or missing stored activations (including
    multi-pass layers whose activations live on per-pass ``passes[k]``
    accessors), map to ``None`` so the caller can render them with a
    neutral shade.
    """

    metric_fn = resolve_metric(metric)
    sg: "Supergraph" = bundle._supergraph  # type: ignore[attr-defined]
    out: Dict[str, Optional[float]] = {}

    for name in sg.topological_order:
        node = sg.nodes[name]
        tensors: List[torch.Tensor] = []
        for trace_name in node.traces:
            tensor = _layer_tensor(
                node.layer_refs[trace_name],
                field="activation",
                has_field="has_saved_activations",
            )
            if tensor is not None:
                tensors.append(tensor)
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


def per_node_coverage(bundle: "TraceBundle") -> Dict[str, float]:
    """Fraction of bundled traces that traversed each supergraph node."""

    sg: "Supergraph" = bundle._supergraph  # type: ignore[attr-defined]
    n = max(len(bundle.names), 1)
    return {name: len(node.traces) / n for name, node in sg.nodes.items()}


def per_node_groups(bundle: "TraceBundle") -> Dict[str, frozenset]:
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


# ---------------------------------------------------------------------------
# Per-node styling
# ---------------------------------------------------------------------------


def _style_node_divergence(
    distances: Dict[str, Optional[float]],
    name: str,
) -> Tuple[str, str]:
    """Return (fill_color, font_color) for a node in divergence mode."""

    val = distances.get(name)
    if val is None:
        return NEUTRAL_GREY, DARK_TEXT
    valid = [v for v in distances.values() if v is not None]
    max_val = max(valid) if valid else 0.0
    frac = (val / max_val) if max_val > 0 else 0.0
    fill = sample_colormap(REDS_HEX, frac)
    font = LIGHT_TEXT if is_dark_swatch(REDS_HEX, frac) else DARK_TEXT
    return fill, font


def _style_node_swarm(
    coverage_map: Dict[str, float],
    name: str,
) -> Tuple[str, str]:
    """Return (fill_color, font_color) for a node in swarm mode."""

    cov = coverage_map.get(name, 0.0)
    fill = sample_colormap(VIRIDIS_HEX, cov)
    font = LIGHT_TEXT if is_dark_swatch(VIRIDIS_HEX, cov) else DARK_TEXT
    return fill, font


def _style_node_group_color(
    group_map: Dict[str, frozenset],
    group_index: Dict[str, int],
    name: str,
) -> Tuple[str, str]:
    """Return (fill_color, font_color) for a node in group_color mode."""

    groups = group_map.get(name, frozenset())
    if not groups:
        return NEUTRAL_GREY, DARK_TEXT
    if len(groups) > 1:
        # Multi-group node -- neutral shade rather than blending.
        return NEUTRAL_GREY, DARK_TEXT
    only_group = next(iter(groups))
    idx = group_index.get(only_group, 0)
    fill = TAB10_HEX[idx % len(TAB10_HEX)]
    return fill, LIGHT_TEXT


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

    lines = [html_escape(name)]
    if node.op_type:
        lines.append(f"<I>{html_escape(node.op_type)}</I>")
    if show_coverage and coverage is not None:
        pct = int(round(coverage * 100))
        lines.append(f"<FONT POINT-SIZE='10'>coverage: {pct}%</FONT>")
    if distance is not None:
        lines.append(f"<FONT POINT-SIZE='10'>div: {distance:.3g}</FONT>")
    return lines


def compute_node_styles(
    bundle: "TraceBundle",
    *,
    resolved_mode: ModeLiteral,
    metric: Union[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
    show_coverage: bool,
    group_index: Dict[str, int],
) -> Tuple[
    Dict[str, Tuple[str, str]],  # node name -> (fill, font)
    Dict[str, str],  # node name -> HTML label string
    Dict[str, Optional[float]],  # raw distances (divergence) for caption use
    Dict[str, float],  # coverage map (swarm + show_coverage)
    Dict[str, frozenset],  # group memberships (group_color)
]:
    """Compute per-node Graphviz fill/font colours and HTML labels.

    Returns the resolved node styling along with the raw mode-specific
    aggregations -- they're kept available so the caller can wire any
    of them into legend/caption strings later if needed.
    """

    sg: "Supergraph" = bundle._supergraph  # type: ignore[attr-defined]

    distances: Dict[str, Optional[float]] = {}
    coverage_map: Dict[str, float] = {}
    group_map: Dict[str, frozenset] = {}

    if resolved_mode == "divergence":
        distances = per_node_distance(bundle, metric)
    if resolved_mode == "swarm" or show_coverage:
        coverage_map = per_node_coverage(bundle)
    if resolved_mode == "group_color":
        group_map = per_node_groups(bundle)

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
        node_labels[name] = format_node_html(label_lines)

    return node_styles, node_labels, distances, coverage_map, group_map


# ---------------------------------------------------------------------------
# Edge styling
# ---------------------------------------------------------------------------


def resolve_edge_color(
    trace_set: Set[str],
    *,
    resolved_mode: ModeLiteral,
    bundle: "TraceBundle",
    group_index: Dict[str, int],
) -> str:
    """Pick a stroke colour for a supergraph edge given the traversing traces.

    In ``group_color`` mode an edge gets the colour of the group that
    traversed it (or neutral grey if multiple groups did).  In divergence
    and swarm modes we keep edges neutral so the coloured nodes stay the
    visual focus.
    """

    if resolved_mode != "group_color":
        return EDGE_COLOR_NEUTRAL

    groups = bundle.groups
    if not groups or not group_index:
        return EDGE_COLOR_NEUTRAL

    trace_to_groups: Dict[str, Set[str]] = defaultdict(set)
    for group_name, members in groups.items():
        for trace_name in members:
            trace_to_groups[trace_name].add(group_name)

    edge_groups: Set[str] = set()
    for trace_name in trace_set:
        edge_groups.update(trace_to_groups.get(trace_name, set()))

    if len(edge_groups) == 1:
        only = next(iter(edge_groups))
        return TAB10_HEX[group_index.get(only, 0) % len(TAB10_HEX)]
    return EDGE_COLOR_NEUTRAL
