"""Bundle diff renderer for paired TorchLens traces."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, Literal

import graphviz
import torch

from .._errors import InvalidArgumentError
from ..bundle._deltas import site_deltas
from ._bundle_diff_marks import (
    DiffStyle,
    _aria_label,
    _caption,
    _node_border_color,
    _node_id,
    _node_label,
    _node_penwidth,
    _node_title,
    _site_label,
)
from ._render_utils import (
    _add_svg_accessibility,
    direction_to_rankdir,
    merge_edge_style,
    merge_node_style,
    render_dot_to_file,
    strip_known_extension,
)
from .themes import resolve_theme, theme_edge_attrs, theme_graph_attrs, theme_node_attrs

if TYPE_CHECKING:
    from .._data_classes.trace import Trace
    from ..bundle import Bundle
    from ..bundle._deltas import SiteDeltas


DiffLayout = Literal["paired"]
DiffTensorField = Literal["out", "grad"]

#: Style ``Bundle.draw(graph="diff")`` sets around its ``bundle_diff`` call.
_DIFF_STYLE: ContextVar[DiffStyle | None] = ContextVar("tl_bundle_diff_style", default=None)


@contextmanager
def diff_style(options: Mapping[str, Any]) -> Iterator[None]:
    """Draw the ``bundle_diff`` calls inside with ``Bundle.draw``'s style keywords.

    ``options`` carries ``direction`` (``None`` keeps the two columns),
    ``graph_overrides``, ``node_overrides``, ``edge_overrides`` and
    ``show_legend``.
    """

    direction = options["direction"]
    style = DiffStyle(
        rankdir=direction_to_rankdir(direction) if direction is not None else "TB",
        graph_overrides=options["graph_overrides"] or {},
        node_overrides=options["node_overrides"],
        edge_overrides=options["edge_overrides"],
        show_legend=options["show_legend"],
    )
    token = _DIFF_STYLE.set(style)
    try:
        yield
    finally:
        _DIFF_STYLE.reset(token)


def bundle_diff(
    bundle: Bundle,
    *,
    metric: str | Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = "relative_l2",
    layout: DiffLayout = "paired",
    left: str | Trace | None = None,
    right: str | Trace | None = None,
    baseline: str | Trace | None = None,
    on: DiffTensorField = "out",
    path: str = "bundle_diff",
    open_viewer: bool = False,
    file_format: str = "svg",
    theme: str = "paper",
    max_pairs: int | None = 16,
    include_unmatched: bool = False,
) -> str:
    """Render a two-column bundle diff SVG using aligned node pairs.

    Parameters
    ----------
    bundle:
        Bundle containing at least two traces.
    metric:
        Per-site tensor metric (the baseline comparison ``Bundle.compare`` runs).
    layout:
        Layout strategy. Only ``"paired"`` is currently supported.
    left:
        Left member name or log. Defaults to the first bundle member.
    right:
        Right member name or log. Defaults to the second bundle member.
    baseline:
        Baseline member for the per-node distances. Defaults to ``left``.
    on:
        Tensor field compared at each site.
    path:
        Output path, with or without a recognised extension.
    open_viewer:
        ``True`` opens the written picture in a viewer; the default never does.
    file_format:
        Output format. Any Graphviz-supported format (e.g., ``"svg"``,
        ``"pdf"``, ``"png"``). Accessibility metadata is only embedded for
        ``"svg"``.
    theme:
        Visualization theme preset.
    max_pairs:
        Maximum aligned pairs to draw. ``None`` draws every pair.
    include_unmatched:
        Whether to add unpaired nodes with gray borders.

    Returns
    -------
    str
        Graphviz DOT source for the rendered bundle diff.

    Examples
    --------
    >>> # requires: graphviz
    >>> import os, tempfile
    >>> bundle = tl.Bundle({"clean": tl.trace(model, x), "again": tl.trace(model, x)})
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     dot = bundle_diff(bundle, path=os.path.join(tmp, "diff"), open_viewer=False)
    >>> "digraph" in dot
    True
    """

    if layout != "paired":
        raise InvalidArgumentError(
            f"bundle_diff layout must be 'paired'; received {layout!r}",
            code="bundle_diff_layout_invalid",
            remedy="pass layout='paired'",
            argument="layout",
        )

    left_name, right_name = _resolve_side_names(bundle, left=left, right=right)
    baseline_ref = baseline if baseline is not None else left_name

    baseline_name = bundle._baseline_or_raise(baseline_ref)
    sides = sorted({left_name, right_name} - {baseline_name})
    # Gate only the pairs this view renders (baseline against each side), so an
    # unrelated third member does not refuse a two-member diff. One node per op:
    # each pass of a multi-pass layer is coloured by its own pass's distance.
    bundle._require_comparable("delta_map", pairs=[(baseline_name, side) for side in sides])
    deltas = site_deltas(bundle, metric, baseline_name, on, [baseline_name, *sides])
    pairs = _select_pairs(
        bundle._aligned_pairs(left_name, right_name),
        names=(left_name, right_name),
        deltas=deltas,
        max_pairs=max_pairs,
    )
    dot = _build_dot(
        pairs=pairs,
        bundle=bundle,
        left_name=left_name,
        right_name=right_name,
        metric_name=_metric_display_name(metric),
        tensor_field=on,
        deltas=deltas,
        theme_name=theme,
        include_unmatched=include_unmatched,
    )
    outpath = strip_known_extension(path)
    source = render_dot_to_file(dot, outpath, file_format, not open_viewer)
    if file_format == "svg":
        _add_svg_accessibility(
            f"{outpath}.{file_format}",
            _aria_label(left_name, right_name, _metric_display_name(metric), on),
        )
    return source


def _resolve_side_names(
    bundle: Bundle,
    *,
    left: str | Trace | None,
    right: str | Trace | None,
) -> tuple[str, str]:
    """Resolve the left and right bundle member names.

    Parameters
    ----------
    bundle:
        Bundle being rendered.
    left:
        Optional left member name or log.
    right:
        Optional right member name or log.

    Returns
    -------
    tuple[str, str]
        Resolved ``(left_name, right_name)``.
    """

    names = bundle.names
    if len(names) < 2 and (left is None or right is None):
        raise InvalidArgumentError(
            "bundle_diff requires at least two bundle members",
            code="bundle_diff_members_invalid",
            remedy="diff a bundle with at least two members, or name both sides",
        )
    left_name = _resolve_member_name(bundle, left if left is not None else names[0])
    right_name = _resolve_member_name(bundle, right if right is not None else names[1])
    if left_name == right_name:
        raise InvalidArgumentError(
            "bundle_diff requires distinct left and right members",
            code="bundle_diff_members_invalid",
            remedy="pass two different member names for left and right",
        )
    return left_name, right_name


def _resolve_member_name(bundle: Bundle, member: str | Trace) -> str:
    """Resolve a member reference inside a bundle.

    Parameters
    ----------
    bundle:
        Bundle being queried.
    member:
        Member name or Trace reference.

    Returns
    -------
    str
        Resolved member name.
    """

    if isinstance(member, str):
        if member not in bundle:
            raise KeyError(f"Unknown bundle member {member!r}.")
        return member
    for name, candidate in bundle.members.items():
        if candidate is member:
            return name
    raise KeyError("Trace is not a member of this Bundle.")


def _build_dot(
    *,
    pairs: list[tuple[Any, Any]],
    bundle: Bundle,
    left_name: str,
    right_name: str,
    metric_name: str,
    tensor_field: DiffTensorField,
    deltas: SiteDeltas,
    theme_name: str,
    include_unmatched: bool,
) -> graphviz.Digraph:
    """Build the Graphviz object for a paired bundle diff.

    Parameters
    ----------
    pairs:
        Aligned layer pairs across the two members.
    bundle:
        Bundle being rendered.
    left_name:
        Left member name.
    right_name:
        Right member name.
    metric_name:
        Human-readable metric name.
    tensor_field:
        Tensor field used for comparison.
    deltas:
        Per-site metric values from the baseline comparison.
    theme_name:
        Visualization theme name.
    include_unmatched:
        Whether to draw nodes without aligned counterparts.

    Returns
    -------
    graphviz.Digraph
        Configured Graphviz graph.
    """

    theme = resolve_theme(theme_name)
    dot = graphviz.Digraph(
        name="TorchLens_Bundle_Diff",
        comment="TorchLens bundle diff",
        format="svg",
    )
    style = _DIFF_STYLE.get() or DiffStyle()
    caption = _caption(left_name, right_name, metric_name, tensor_field)
    dot.graph_attr.update(_diff_graph_attrs(theme, caption, style))
    dot.node_attr.update(theme_node_attrs(theme, font_size=10))
    dot.edge_attr.update(theme_edge_attrs(theme, font_size=8))

    paired_left_ids = {id(left_layer) for left_layer, _right_layer in pairs}
    paired_right_ids = {id(right_layer) for _left_layer, right_layer in pairs}
    left_unmatched = _unmatched_layers(bundle[left_name], paired_left_ids, include_unmatched)
    right_unmatched = _unmatched_layers(bundle[right_name], paired_right_ids, include_unmatched)
    left_layers = [left_layer for left_layer, _right_layer in pairs] + left_unmatched
    right_layers = [right_layer for _left_layer, right_layer in pairs] + right_unmatched
    values = _pair_delta_values(
        pairs=pairs, left_name=left_name, right_name=right_name, deltas=deltas
    )
    max_delta = max(values) if values else 0.0
    _add_side_cluster(
        dot,
        layers=left_layers,
        side="clean",
        member_name=left_name,
        compared_member_name=left_name,
        deltas=deltas,
        max_delta=max_delta,
        unmatched_layer_ids={id(layer) for layer in left_unmatched},
        node_overrides=style.node_overrides,
    )
    _add_side_cluster(
        dot,
        layers=right_layers,
        side="intervention",
        member_name=right_name,
        compared_member_name=right_name,
        deltas=deltas,
        max_delta=max_delta,
        unmatched_layer_ids={id(layer) for layer in right_unmatched},
        node_overrides=style.node_overrides,
    )
    for side, layers in (("clean", left_layers), ("intervention", right_layers)):
        _add_side_edges(dot, layers=layers, side=side, edge_overrides=style.edge_overrides)
    _add_pair_alignment(dot, pairs)
    if style.show_legend is not False:
        _add_legend(dot)
    return dot


def _diff_graph_attrs(theme: Any, caption: str, style: DiffStyle) -> dict[str, str]:
    """Return the diff graph's attributes: theme defaults, layout, caption, then overrides."""

    # dpi is raster-only now (vizmech), and this graph renders SVG: the
    # historical ``dpi=72`` workaround (pinning viewBox units to layout
    # coords) is the identity value, so passing the format simply drops it.
    graph_attrs = theme_graph_attrs(theme, font_size=18, dpi=72, fileformat="svg")
    graph_attrs.update(
        {
            "rankdir": style.rankdir,
            "compound": "true",
            # One global ranking across clusters. Without it, dot pulls every
            # node named in a ``rank=same`` pair subgraph (_add_pair_alignment)
            # OUT of its side cluster ("already in a rankset, deleted from
            # cluster"), drops both side clusters as empty, and lays the pairs
            # out as loose node pairs: the column frames and member titles
            # vanish and the white baseline column reads as an unlabeled
            # duplicate of the marked one.
            "newrank": "true",
            "splines": "ortho",
            "nodesep": "0.32",
            "ranksep": "0.34",
            # Let graphviz auto-size to the natural aspect ratio. A hard
            # ``size: 12,8!`` combined with ``ratio: fill`` forces a 1.5:1
            # box; when the natural graph aspect is taller (e.g. 1.69:1 for
            # the default paired layout) the bottom-positioned caption gets
            # pushed outside the SVG viewBox and clipped.
            "label": caption,
            "labelloc": "b",
            "labeljust": "l",
            "fontname": "Helvetica",
            # The caption and the column titles take the page's text ink, so
            # the dark skin's caption is not drawn black on a dark page.
            "fontcolor": theme.default_font,
        }
    )
    graph_attrs.update({str(key): str(value) for key, value in style.graph_overrides.items()})
    return graph_attrs


def _unmatched_layers(member: Any, paired_ids: set[int], include_unmatched: bool) -> list[Any]:
    """Return ``member``'s layers with no aligned counterpart, or ``[]`` when not drawn."""

    if not include_unmatched:
        return []
    return [layer for layer in member._layer_list if id(layer) not in paired_ids]


def _metric_display_name(metric: str | Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> str:
    """Return a stable display name for a bundle-diff metric.

    Parameters
    ----------
    metric:
        Metric name or callable passed to ``bundle_diff``.

    Returns
    -------
    str
        Human-readable metric name.
    """

    if isinstance(metric, str):
        return metric
    return getattr(metric, "__name__", metric.__class__.__name__)


def _pair_delta_values(
    *,
    pairs: list[tuple[Any, Any]],
    left_name: str,
    right_name: str,
    deltas: SiteDeltas,
) -> list[float]:
    """Return member-specific delta values for color normalization.

    Parameters
    ----------
    pairs:
        Aligned op pairs.
    left_name:
        Left member name.
    right_name:
        Right member name.
    deltas:
        Per-site metric values.

    Returns
    -------
    list[float]
        Non-negative delta values for both rendered members, each read at that
        member's own op.
    """

    values = [
        deltas.value(op, member_name)
        for left_layer, right_layer in pairs
        for op, member_name in ((left_layer, left_name), (right_layer, right_name))
    ]
    return [value for value in values if value >= 0.0]


def _select_pairs(
    pairs: list[tuple[Any, Any]],
    *,
    names: tuple[str, str],
    deltas: SiteDeltas,
    max_pairs: int | None,
) -> list[tuple[Any, Any]]:
    """Select high-signal aligned pairs for a compact hero diff.

    Parameters
    ----------
    pairs:
        Candidate aligned pairs.
    names:
        The ``(left, right)`` member names.
    deltas:
        Per-site metric values.
    max_pairs:
        Maximum pair count, or ``None`` for all pairs.

    Returns
    -------
    list[tuple[Any, Any]]
        Selected pairs in original trace order.
    """

    left_name, right_name = names
    if max_pairs is None or len(pairs) <= max_pairs:
        return pairs
    if max_pairs < 1:
        raise InvalidArgumentError(
            f"max_pairs must be at least 1 or None; received {max_pairs!r}",
            code="max_pairs_invalid",
            remedy="pass max_pairs >= 1 or None",
            argument="max_pairs",
        )
    scored: list[tuple[float, int, tuple[Any, Any]]] = []
    for index, (left_layer, right_layer) in enumerate(pairs):
        value = max(deltas.value(left_layer, left_name), deltas.value(right_layer, right_name))
        scored.append((value, index, (left_layer, right_layer)))
    chosen_indexes = {
        index
        for _value, index, _pair in sorted(scored, key=lambda row: row[0], reverse=True)[:max_pairs]
    }
    return [pair for index, pair in enumerate(pairs) if index in chosen_indexes]


def _add_side_cluster(
    dot: graphviz.Digraph,
    *,
    layers: list[Any],
    side: str,
    member_name: str,
    compared_member_name: str,
    deltas: SiteDeltas,
    max_delta: float,
    unmatched_layer_ids: set[int],
    node_overrides: Mapping[str, Any] | None = None,
) -> None:
    """Add one clean or intervention cluster to the diff graph.

    Parameters
    ----------
    dot:
        Graphviz graph to mutate.
    layers:
        Layers to render in this side.
    side:
        Stable side key used in Graphviz node IDs.
    member_name:
        Bundle member represented by this side.
    compared_member_name:
        Member used for delta color values.
    deltas:
        Per-site metric values.
    max_delta:
        Maximum delta value used for normalization.
    unmatched_layer_ids:
        ``id(layer)`` values whose aligned counterpart is missing.
    node_overrides:
        Node attributes keyed by op label, applied over the defaults.
    """

    with dot.subgraph(name=f"cluster_{side}") as subgraph:
        subgraph.attr(
            label=f"{member_name}",
            labelloc="t",
            color="#222222",
            penwidth="2",
            style="rounded",
        )
        for layer in layers:
            value = deltas.value(layer, compared_member_name)
            base = {
                "label": _node_label(layer, value),
                "tooltip": _node_title(member_name, layer, value),
                "fillcolor": _delta_color(value, max_delta),
                "color": _node_border_color(layer, unmatched_layer_ids),
                "fontcolor": "#111111",
                "shape": "box",
                "style": "filled,rounded",
                "penwidth": _node_penwidth(layer),
            }
            attrs = merge_node_style(base, node_overrides, _site_label(layer), layer)
            subgraph.node(_node_id(side, layer), **attrs)


def _add_side_edges(
    dot: graphviz.Digraph,
    *,
    layers: list[Any],
    side: str,
    edge_overrides: Mapping[tuple[str, str], Any] | None = None,
) -> None:
    """Add intra-side parent-child edges among displayed nodes.

    Parameters
    ----------
    dot:
        Graphviz graph to mutate.
    layers:
        Layers displayed for one side.
    side:
        Stable side key used in Graphviz node IDs.
    edge_overrides:
        Edge attributes keyed by ``(source, target)`` op labels.

    Returns
    -------
    None
        ``dot`` is mutated in place.
    """

    visible = {_site_label(layer): layer for layer in layers}
    for layer in layers:
        target_id = _node_id(side, layer)
        for parent_label in getattr(layer, "parents", []) or []:
            parent = visible.get(str(parent_label))
            if parent is not None:
                key = (_site_label(parent), _site_label(layer))
                base = {"color": "#707070", "arrowsize": "0.5"}
                attrs = merge_edge_style(base, edge_overrides, key, (parent, layer))
                dot.edge(_node_id(side, parent), target_id, **attrs)


def _add_pair_alignment(dot: graphviz.Digraph, pairs: list[tuple[Any, Any]]) -> None:
    """Add same-rank constraints for aligned layer pairs.

    Parameters
    ----------
    dot:
        Graphviz graph to mutate.
    pairs:
        Aligned layer pairs.

    Returns
    -------
    None
        ``dot`` is mutated in place.
    """

    for left_layer, right_layer in pairs:
        left_id = _node_id("clean", left_layer)
        right_id = _node_id("intervention", right_layer)
        with dot.subgraph() as rank:
            rank.attr(rank="same")
            rank.node(left_id)
            rank.node(right_id)
        dot.edge(left_id, right_id, style="invis", weight="8")


def _add_legend(dot: graphviz.Digraph) -> None:
    """Add a compact color legend to the diff graph.

    Parameters
    ----------
    dot:
        Graphviz graph to mutate.

    Returns
    -------
    None
        ``dot`` is mutated in place.
    """

    with dot.subgraph(name="cluster_delta_legend") as legend:
        legend.attr(label="delta_map", color="#D0D0D0", style="rounded", labelloc="t")
        legend.node(
            "legend_low",
            label="zero",
            shape="box",
            style="filled,rounded",
            fillcolor="#FFFFFF",
            color="#222222",
        )
        legend.node(
            "legend_mid",
            label="mid",
            shape="box",
            style="filled,rounded",
            fillcolor=_interpolate(0.5),
            color="#222222",
        )
        legend.node(
            "legend_high",
            label="max",
            shape="box",
            style="filled,rounded",
            fillcolor="#C93F3F",
            color="#222222",
        )
        legend.edge("legend_low", "legend_mid", style="invis")
        legend.edge("legend_mid", "legend_high", style="invis")


def _delta_color(value: float, max_delta: float) -> str:
    """Map a non-negative delta value to a white-to-red color.

    Parameters
    ----------
    value:
        Delta value.
    max_delta:
        Maximum delta used for normalization.

    Returns
    -------
    str
        Hex color. White at zero delta, progressively red as the delta
        approaches ``max_delta``.
    """

    if max_delta <= 0.0:
        return "#FFFFFF"
    normalized = max(0.0, min(1.0, value / max_delta))
    return _interpolate(normalized)


def _interpolate(fraction: float) -> str:
    """Return the white-to-red delta ramp's colour at ``fraction``.

    Thin alias of the ONE colormap interpolation home
    (:func:`torchlens._visualization._encoding.interpolate_hex`).
    """

    from ._encoding import interpolate_hex  # deferred: cycle _visualization

    return interpolate_hex(("#FFFFFF", "#C93F3F"), fraction)


__all__ = ["bundle_diff"]
