"""Graphviz construction helpers for ``tl._visualization.show_bundle_graph``.

These walk a bundle's forward supergraph (and, when present, its backward
graph) and add nodes, module clusters and edges to a Graphviz digraph. The
public entry point and its argument checks stay in
:mod:`torchlens._user_public_impls`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .themes import VisualizationTheme


def _members_note(traces: Any, all_members: frozenset[str] | None) -> str | None:
    """Return the member list for a node or edge, or ``None`` when every member runs it.

    The picture names members only where they differ, so a bundle whose
    members share most of their graph is not covered in repeated member lists.
    """

    names = [str(name) for name in traces]
    if all_members is not None and set(names) == all_members:
        return None
    return ",".join(sorted(names))


def _bundle_node_style(node: Any, theme: VisualizationTheme) -> dict[str, str]:
    """Return the node's shape and fill as ``draw`` paints them under ``theme``.

    Shape says what the node is (oval: one function call, input and output
    included; box: a leaf module; cylinder: a buffer); fill says its role
    (input, output, weights, buffer), from the first member that ran it.
    """

    from ._render_nodes import _get_node_bg_color
    from .node_spec import NodeSpec
    from .themes import apply_theme_to_spec

    layer = next(iter(getattr(node, "layer_refs", {}).values()), None)
    shape, fill = "oval", "white"
    if layer is not None:
        if getattr(layer, "is_buffer", False):
            shape = "cylinder"
        elif getattr(layer, "is_atomic_module", False) and getattr(layer, "modules", None):
            shape = "box"
        try:
            fill = _get_node_bg_color(getattr(layer, "trace", None), layer)  # type: ignore[arg-type]
        except AttributeError:
            fill = "white"
    themed = apply_theme_to_spec(NodeSpec(["x"], shape=shape, fillcolor=fill), theme)
    style = {
        "shape": shape,
        "style": "filled,solid",
        "fillcolor": str(themed.fillcolor),
        "color": str(themed.color),
        "fontcolor": str(themed.fontcolor),
    }
    if ":" in style["fillcolor"]:
        style["gradientangle"] = "0"
    return style


def _bundle_node_display_label(
    graph_node_label: str,
    node: Any,
    vis_mode: str,
    all_members: frozenset[str] | None = None,
) -> str:
    """Return a compact Graphviz label for a bundle supergraph node.

    Parameters
    ----------
    graph_node_label:
        Canonical supergraph node name.
    node:
        Supergraph node-like object.
    vis_mode:
        Bundle visualization mode.

    all_members:
        Every member name; a node every member ran carries no member list.

    Returns
    -------
    str
        Display label.
    """

    traces = _members_note(getattr(node, "traces", []), all_members)
    mode_suffix = " rolled" if vis_mode == "rolled" else ""
    op_type = getattr(node, "op_type", "") or "op"
    label = f"{graph_node_label}\n{op_type}{mode_suffix}"
    return label if traces is None else f"{label}\n[{traces}]"


def _bundle_module_groups(bundle: Any) -> dict[str, list[str]]:
    """Return bundle supergraph nodes grouped by representative module path.

    Parameters
    ----------
    bundle:
        Bundle with a ``supergraph`` accessor.

    Returns
    -------
    dict[str, list[str]]
        Module path to canonical node names.
    """

    groups: dict[str, list[str]] = {}
    for graph_node_label in bundle.supergraph.topological_order:
        node = bundle.supergraph.nodes[graph_node_label]
        module_path = getattr(node, "module_path", None)
        if module_path:
            groups.setdefault(str(module_path), []).append(graph_node_label)
    return groups


def _add_bundle_forward_nodes(
    dot: Any,
    bundle: Any,
    vis_mode: str,
    node_styles: dict[str, Any] | None,
    theme: VisualizationTheme | None = None,
) -> None:
    """Add forward supergraph nodes to a Graphviz digraph.

    Parameters
    ----------
    dot:
        Graphviz digraph.
    bundle:
        Bundle to render.
    vis_mode:
        Bundle visualization mode.
    node_styles:
        Optional per-node style overrides.
    theme:
        Resolved skin; ``None`` uses the default ``torchlens`` skin.
        Module captions read ``dot``'s rank direction, so they land top left
        on the page.

    Returns
    -------
    None
        ``dot`` is mutated in place.
    """

    from ._render_utils import (
        compute_module_penwidth,
        make_module_cluster_attrs,
        merge_node_style,
    )
    from .themes import THEME_PRESETS, _theme_cluster_attrs

    theme = theme if theme is not None else THEME_PRESETS["torchlens"]
    rankdir = dot.graph_attr.get("rankdir")
    all_members = frozenset(str(name) for name in bundle.names)
    module_groups = _bundle_module_groups(bundle)
    grouped_nodes = {node for nodes in module_groups.values() for node in nodes}
    for module_path, node_names in module_groups.items():
        first_node = bundle.supergraph.nodes[node_names[0]]
        cluster_name = "cluster_bundle_" + "".join(
            char if char.isalnum() else "_" for char in module_path
        )
        with dot.subgraph(name=cluster_name) as subgraph:
            cluster_attrs = make_module_cluster_attrs(
                title=module_path,
                module_type=getattr(first_node, "module_type", None),
                line_style="solid",
                penwidth=compute_module_penwidth(0, 1),
                rankdir=rankdir,
            )
            subgraph.attr(**{**cluster_attrs, **_theme_cluster_attrs(theme)})
            for graph_node_label in node_names:
                node = bundle.supergraph.nodes[graph_node_label]
                attrs = merge_node_style(
                    _bundle_node_style(node, theme), node_styles, graph_node_label, node
                )
                subgraph.node(
                    f"fwd_{graph_node_label}",
                    label=_bundle_node_display_label(graph_node_label, node, vis_mode, all_members),
                    **attrs,
                )
    for graph_node_label in bundle.supergraph.topological_order:
        if graph_node_label in grouped_nodes:
            continue
        node = bundle.supergraph.nodes[graph_node_label]
        attrs = merge_node_style(
            _bundle_node_style(node, theme), node_styles, graph_node_label, node
        )
        dot.node(
            f"fwd_{graph_node_label}",
            label=_bundle_node_display_label(graph_node_label, node, vis_mode, all_members),
            **attrs,
        )


def _add_bundle_forward_edges(
    dot: Any,
    bundle: Any,
    edge_styles: dict[tuple[str, str], Any] | None,
    theme: VisualizationTheme | None = None,
) -> None:
    """Add forward supergraph edges to a Graphviz digraph.

    Parameters
    ----------
    dot:
        Graphviz digraph.
    bundle:
        Bundle to render.
    edge_styles:
        Optional per-edge style overrides.
    theme:
        Resolved skin; ``None`` uses the default ``torchlens`` skin.

    Returns
    -------
    None
        ``dot`` is mutated in place.

    An edge every member runs carries no label; the others name their members.
    """

    from ._render_utils import merge_edge_style
    from .themes import THEME_PRESETS

    theme = theme if theme is not None else THEME_PRESETS["torchlens"]
    all_members = frozenset(str(name) for name in bundle.names)
    ink = theme.edge.get("color", theme.default_border)
    base_style = {"color": ink, "fontcolor": theme.edge.get("fontcolor", ink)}
    for edge_key, traces in bundle.supergraph.edges.items():
        attrs = merge_edge_style(base_style, edge_styles, edge_key, {"traces": traces})
        note = _members_note(traces, all_members)
        if note is not None:
            attrs = {"label": note, **attrs}
        dot.edge(f"fwd_{edge_key[0]}", f"fwd_{edge_key[1]}", **attrs)


def _add_bundle_backward_graph(dot: Any, bundle: Any) -> None:
    """Add per-member backward graph clusters to a Graphviz digraph.

    Parameters
    ----------
    dot:
        Graphviz digraph.
    bundle:
        Bundle to render.

    Returns
    -------
    None
        ``dot`` is mutated in place.
    """

    for member_name, member in bundle.members.items():
        with dot.subgraph(name=f"cluster_backward_{member_name}") as subgraph:
            subgraph.attr(label=f"{member_name} backward", color="#7A3E9D")
            grad_fns = list(getattr(getattr(member, "autograd", None), "grad_fns", []))
            if not grad_fns:
                subgraph.node(
                    f"bwd_{member_name}_empty",
                    label="no backward graph",
                    shape="box",
                    style="dashed",
                    color="#7A3E9D",
                )
                continue
            visible_ids = {grad_fn_handle.grad_fn_object_id for grad_fn_handle in grad_fns}
            for grad_fn_handle in grad_fns:
                subgraph.node(
                    f"bwd_{member_name}_{grad_fn_handle.grad_fn_object_id}",
                    label=str(
                        getattr(
                            grad_fn_handle,
                            "label",
                            getattr(grad_fn_handle, "name", "grad_fn_handle"),
                        )
                    ),
                    shape="box",
                    style="filled,rounded",
                    fillcolor="#F4E8FA",
                    color="#7A3E9D",
                )
            for grad_fn_handle in grad_fns:
                for next_id in getattr(grad_fn_handle, "next_grad_fn_ids", []):
                    if next_id in visible_ids:
                        subgraph.edge(
                            f"bwd_{member_name}_{grad_fn_handle.grad_fn_object_id}",
                            f"bwd_{member_name}_{next_id}",
                            color="#7A3E9D",
                        )


def _member_row_notes(bundle: Any) -> dict[str, list[str]]:
    """Return per-member label lines from MEMBER-kind rows (``episode_member``)."""

    from ..bundle._relations import MemberRelationRow

    notes: dict[str, list[str]] = {}
    for row in bundle.relations:
        if isinstance(row, MemberRelationRow) and row.member is not None:
            params = row.params
            notes.setdefault(str(row.member), []).append(
                f"{row.kind} {params.get('episode_id')} step {params.get('at_step')} "
                f"({params.get('role')})"
            )
    return notes


def _relation_edge_label(row: Any) -> tuple[str, bool]:
    """Return ``(label, checked)`` for one pair row: kind, label, basis, coverage, choice."""

    params = row.params
    lines = [str(row.kind)]
    if params.get("label"):
        lines.append(str(params["label"]))
    checked = bool(params.get("token_spans")) and params.get("basis") in {"verified", "matched"}
    if row.kind == "fed_by":
        spans = params.get("token_spans") or []
        basis = params.get("basis", "declared") if spans else "no spans"
        coverage = params.get("coverage")
        lines.append(basis if coverage is None else f"{basis}, coverage {coverage:.2f}")
    if row.kind == "selected_by" and "chosen" in params:
        lines.append(f"chose {params['chosen']!r}")
    return "\n".join(lines), checked


def _add_bundle_member_graph(
    dot: Any,
    bundle: Any,
    node_styles: dict[str, Any] | None,
    edge_styles: dict[tuple[str, str], Any] | None,
    theme: VisualizationTheme | None = None,
) -> int:
    """Add one node per member and one edge per pair relation row.

    Edges run from the earlier member to the later one (the direction text
    and time flow: ``draft -> critique`` for "critique fed_by draft"; peer
    rows such as ``alternative_of`` draw without an arrowhead),
    labelled with the kind, the row's free ``label``, on ``fed_by`` its
    basis and coverage, and on ``selected_by`` the ``chosen`` decision. Solid edges carry token spans TorchLens checked
    (``verified``/``matched``); every other row is dashed. MEMBER-kind rows
    (``episode_member``) annotate their member's node. Overrides key nodes
    by member name and edges by ``(earlier, later)``.

    Returns
    -------
    int
        The number of opaque rows (unknown namespaced kinds) left undrawn:
        their endpoints are unreadable without their kind's schema.
    """

    from ..bundle._compare_gate import _ORDERING_RELATION_KINDS
    from ..bundle._relations import MemberRelationRow
    from ._render_utils import merge_edge_style, merge_node_style
    from .themes import THEME_PRESETS

    theme = theme if theme is not None else THEME_PRESETS["torchlens"]
    ink = theme.edge.get("color", theme.default_border)
    base_node = {
        "shape": "box",
        "style": "filled,rounded",
        "fillcolor": theme.default_fill,
        "color": theme.default_border,
        "fontcolor": theme.default_font,
    }
    notes = _member_row_notes(bundle)
    # Index ids, not names: a member name such as "a:b" would read as a DOT port.
    node_ids = {name: f"member_{index}" for index, name in enumerate(bundle.names)}
    for name in bundle.names:
        member = bundle[name]
        lines = [name, str(getattr(member, "model_class_name", "") or "")]
        lines.extend(notes.get(name, []))
        attrs = merge_node_style(base_node, node_styles, name, member)
        dot.node(node_ids[name], label="\n".join(line for line in lines if line), **attrs)
    opaque = 0
    for row in bundle.relations:
        if not isinstance(row, MemberRelationRow):
            opaque += 1
            continue
        if row.from_member is None:
            continue
        label, checked = _relation_edge_label(row)
        base_edge = {"color": ink, "fontcolor": theme.edge.get("fontcolor", ink)}
        base_edge["style"] = "solid" if checked else "dashed"
        if row.kind not in _ORDERING_RELATION_KINDS:
            base_edge["dir"] = "none"  # peers (alternative_of) have no direction
        edge_key = (str(row.to_member), str(row.from_member))
        attrs = merge_edge_style(base_edge, edge_styles, edge_key, row)
        dot.edge(node_ids[edge_key[0]], node_ids[edge_key[1]], label=label, **attrs)
    return opaque


def bundle_header_label(
    bundle: Any,
    header_lines: list[str],
    theme: VisualizationTheme,
    *,
    show_legend: bool,
    draws_ops: bool,
) -> str:
    """Return the bundle picture's header label, with the key under it when asked.

    The header and key form one top-left block as on ``draw``. The key lists
    the marks the op picture draws (from the same styles the nodes were given)
    and says that unlabelled nodes and arrows run in every member.
    """

    from ._header_block import header_block_label, key_swatches
    from ._legend import LegendRow, LegendSection, drawn_legend_roles
    from ._render_utils import html_escape

    body = "".join(f"{html_escape(line)}<br align='left'/>" for line in header_lines)
    caption = f"<<FONT COLOR='{theme.default_font}'>{body}</FONT>>"
    if not show_legend:
        return caption
    swatches: tuple[Any, ...] = ()
    if draws_ops:
        nodes = bundle.supergraph.nodes.values()
        node_attrs = [_bundle_node_style(node, theme) for node in nodes]
        swatches = key_swatches(
            theme,
            drawn_legend_roles(node_attrs, theme),
            has_module=any(getattr(node, "module_path", None) for node in nodes)
            or any(attrs["shape"] == "box" for attrs in node_attrs),
        )
    note = LegendSection(
        title="members",
        rows=(LegendRow(text="a node or arrow with no member list runs in every member"),),
    )
    return header_block_label(caption, swatches, (note,), theme)


@dataclass(frozen=True)
class BundlePictureOptions:
    """The checked ``tl._visualization.show_bundle_graph`` arguments the bundle picture needs."""

    outpath: str
    vis_mode: str
    direction: str
    rankdir: str
    graph_overrides: dict[str, Any] | None
    node_overrides: dict[str, Any] | None
    edge_overrides: dict[tuple[str, str], Any] | None
    save_only: bool
    fileformat: str
    nodes: str
    theme_name: str
    show_legend: bool


def render_bundle_graph(bundle: Any, options: BundlePictureOptions) -> str | None:
    """Build and render the bundle picture behind ``tl._visualization.show_bundle_graph``.

    Arguments are documented on :func:`torchlens._visualization.show_bundle_graph`, which
    checks them first.

    Returns
    -------
    str | None
        DOT source.
    """

    import graphviz

    from ._render_utils import render_dot_to_file, strip_known_extension
    from .themes import (
        resolve_theme,
        theme_edge_attrs,
        theme_graph_attrs,
        theme_node_attrs,
    )

    dot = graphviz.Digraph(
        name="TorchLens_Bundle",
        comment="TorchLens bundle graph",
        format=options.fileformat,
    )
    theme = resolve_theme(options.theme_name)
    draws_forward = options.direction in {"forward", "both", "overlay"}
    title = "members" if options.nodes == "members" else f"{options.vis_mode}, {options.direction}"
    graph_attrs = {
        "rankdir": options.rankdir,
        "labelloc": "t",
        "labeljust": "left",
        "compound": "true",
        **theme_graph_attrs(theme, fileformat=options.fileformat),
    }
    dot.graph_attr["rankdir"] = options.rankdir
    dot.node_attr.update(theme_node_attrs(theme))
    dot.edge_attr.update(theme_edge_attrs(theme))

    header_lines = [f"TorchLens bundle graph ({title})"]
    if options.nodes == "members":
        opaque = _add_bundle_member_graph(
            dot, bundle, options.node_overrides, options.edge_overrides, theme
        )
        if opaque:
            header_lines.append(f"{opaque} opaque relation row(s) not drawn")
    elif draws_forward:
        _add_bundle_forward_nodes(dot, bundle, options.vis_mode, options.node_overrides, theme)
        _add_bundle_forward_edges(dot, bundle, options.edge_overrides, theme)
    if options.nodes == "ops" and options.direction in {"backward", "both", "overlay"}:
        _add_bundle_backward_graph(dot, bundle)
    graph_attrs["label"] = bundle_header_label(
        bundle,
        header_lines,
        theme,
        show_legend=options.show_legend,
        draws_ops=options.nodes == "ops" and draws_forward,
    )
    graph_attrs.update({key: str(value) for key, value in (options.graph_overrides or {}).items()})
    dot.graph_attr.update(graph_attrs)
    outpath = strip_known_extension(options.outpath)
    source = render_dot_to_file(dot, outpath, options.fileformat, options.save_only)
    if options.fileformat == "svg" and options.show_legend and Path(f"{outpath}.svg").is_file():
        # Embed the key's swatch images, so the saved SVG stands alone.
        from ._svg_compose import _inline_svg_file_local_images

        _inline_svg_file_local_images(f"{outpath}.svg")
    return source
