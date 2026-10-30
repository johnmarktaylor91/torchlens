"""NodeSpec helpers for TorchLens graph visualization labels.

The NodeSpec VALUE vocabulary (dataclass, callback aliases, style constants)
lives in :mod:`torchlens._vocab.node_spec` after the ratified V2 relocation
(architecture memo 3.3, C01 item 6) -- it is consumed by strata below
PRESENT. This module re-exports it unchanged and keeps the render BEHAVIOR
helpers, which are L7 code (they reach into the resolver, replay cone, and
render node internals and may not ride a vocabulary module under Rule V3).
"""

from __future__ import annotations

from html import escape
from typing import TYPE_CHECKING, Any, cast

from .._vocab.node_spec import (  # noqa: F401
    INTERVENTION_CONE_COLOR,
    INTERVENTION_HOOK_BORDER_COLOR,
    INTERVENTION_HOOK_FILL_COLOR,
    INTERVENTION_OVERRIDE_KEYS,
    INTERVENTION_SITE_COLOR,
    BackwardNodeSpecFn,
    CollapsedNodeSpecFn,
    NodeSpec,
    NodeSpecFn,
)
from ..utils._multipass_access import get_multipass_attr

if TYPE_CHECKING:
    from ..data_classes.layer import Layer
    from ..data_classes.op import Op
    from ..data_classes.trace import Trace
    from .themes import VisualizationTheme

__tl_layer__ = "L7"

__all__ = [
    "BackwardNodeSpecFn",
    "CollapsedNodeSpecFn",
    "INTERVENTION_CONE_COLOR",
    "INTERVENTION_HOOK_BORDER_COLOR",
    "INTERVENTION_HOOK_FILL_COLOR",
    "INTERVENTION_OVERRIDE_KEYS",
    "INTERVENTION_SITE_COLOR",
    "NodeSpec",
    "NodeSpecFn",
    "graphviz_graph_overrides",
    "intervention_graph_override",
    "intervention_site_and_cone_labels",
    "intervention_sites_for_log",
    "make_intervention_node_spec_fn",
    "render_lines_to_html",
]


def _annotation_image_path_for_node(trace: Trace, node: Any) -> str | None:
    """Return a user annotation image path for a rendered node.

    One of the three record-derived image mechanisms in the closed 2.4(i)
    image-origin predicate (``_encoding.is_record_derived_image_node``).
    Moved here from ``_render_nodes`` (S5 territory; ratchet offload).

    Returns
    -------
    str | None
        Image path stored in ``annotations["user"]["image"]``, if present.
    """

    from ._render_nodes import BoundaryNode, _layer_log_for_node

    if isinstance(node, BoundaryNode):
        return None
    candidates: list[Any] = [node]
    try:
        candidates.append(_layer_log_for_node(trace, node))
    except ValueError:
        pass
    for candidate in candidates:
        annotations = getattr(candidate, "annotations", None)
        if not isinstance(annotations, dict):
            continue
        user_annotations = annotations.get("user")
        if not isinstance(user_annotations, dict):
            continue
        image = user_annotations.get("image")
        if isinstance(image, str) and image:
            return image
    return None


def render_lines_to_html(lines: list[str]) -> str:
    """Render plain-text node rows as a Graphviz HTML-like table label.

    The first row is bolded as the node title; subsequent rows render plain.
    Edit rows (:class:`._edit_marks.EditRow`) render last-row style: bold, in
    the edit purple, after a 4 pt gap.

    Parameters
    ----------
    lines:
        Plain-text row contents. Special HTML characters are escaped.

    Returns
    -------
    str
        A string suitable for Graphviz ``label=<...>`` syntax.
    """

    from ._edit_marks import EDIT_GAP_ROW, EditRow, edit_row_html

    rows = []
    gap_drawn = False
    for index, line in enumerate(lines):
        if isinstance(line, EditRow):
            # An edit row: a 4 pt gap above the first one (no rule), then the
            # bold purple delta line.
            if not gap_drawn and index > 0:
                rows.append(EDIT_GAP_ROW)
                gap_drawn = True
            rows.append(
                f'<TR><TD ALIGN="CENTER">{edit_row_html(line, line.color, line.size)}</TD></TR>'
            )
            continue
        text = escape(str(line), quote=False)
        if index == 0:
            text = f"<B>{text}</B>"
        rows.append(f'<TR><TD ALIGN="CENTER">{text}</TD></TR>')
    return (
        '<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="2">'
        + "".join(rows)
        + "</TABLE>>"
    )


def intervention_graph_override(
    graph_overrides: dict[str, Any] | None,
    key: str,
    default: Any,
) -> Any:
    """Return an intervention-specific visualization override value.

    Parameters
    ----------
    graph_overrides:
        Graph override dictionary supplied by the user.
    key:
        Intervention-specific key to look up.
    default:
        Fallback value.

    Returns
    -------
    Any
        Override value when present, otherwise ``default``.
    """

    if graph_overrides is None:
        return default
    return graph_overrides.get(key, default)


def graphviz_graph_overrides(graph_overrides: dict[str, Any] | None) -> dict[str, Any]:
    """Return graph overrides excluding TorchLens intervention style keys.

    Parameters
    ----------
    graph_overrides:
        Graph override dictionary supplied by the user.

    Returns
    -------
    dict[str, Any]
        Overrides suitable for Graphviz graph attributes.
    """

    if graph_overrides is None:
        return {}
    return {
        key: value
        for key, value in graph_overrides.items()
        if key not in INTERVENTION_OVERRIDE_KEYS
    }


def intervention_sites_for_log(trace: Trace) -> list[Op]:
    """Resolve intervention spec targets to layer-pass records.

    Parameters
    ----------
    trace:
        Log whose intervention spec should be inspected.

    Returns
    -------
    list[Op]
        Distinct intervention sites in execution order.
    """

    spec = getattr(trace, "_intervention_spec", None)
    if spec is None:
        return []
    targets: list[Any] = []
    targets.extend(getattr(spec, "targets", ()) or ())
    targets.extend(
        value_spec.site_target for value_spec in getattr(spec, "target_value_specs", ()) or ()
    )
    targets.extend(hook_spec.site_target for hook_spec in getattr(spec, "hook_specs", ()) or ())
    if not targets:
        return []

    from ..intervention.resolver import resolve_sites

    by_label: dict[str, Op] = {}
    for target in targets:
        table = resolve_sites(trace, target, max_fanout=max(1, len(trace.layer_list)))
        for site in table:
            forward_site = _forward_site(site)
            # Key by the pass-qualified label (op.label, e.g. relu_1_1:2) so
            # recurrent passes stay DISTINCT sites. The old aggregate layer_label
            # collapsed :1/:2/:3 into one site, so resolving pass 2 later colored
            # every pass. Fall back to layer_label only when no pass label exists.
            site_label = get_multipass_attr(forward_site, "label", None, multipass=None)
            if not isinstance(site_label, str):
                site_label = getattr(forward_site, "layer_label", None)
            if isinstance(site_label, str):
                by_label.setdefault(site_label, cast("Op", forward_site))
    execution_order = {
        _site_sort_key(op): index for index, op in enumerate(getattr(trace, "layer_list", ()))
    }
    return sorted(by_label.values(), key=lambda site: execution_order.get(_site_sort_key(site), 0))


def _forward_site(site: Any) -> Any:
    """Return the forward op a resolved intervention site marks, at its own pass.

    A GradFn differentiates one pass: resolve that pass, not the multi-pass
    Layer its layer-label ``op`` names. Other sites mark their ``op`` (a hook
    site) or themselves.
    """

    from ..data_classes.grad_fn import GradFn
    from ..ir.selector_eval import _grad_fn_forward_op

    if isinstance(site, GradFn):
        return _grad_fn_forward_op(site)
    return getattr(site, "op", site)


def _site_sort_key(op: Any) -> str:
    """Return an op's pass-qualified label for stable per-pass site ordering."""

    label = get_multipass_attr(op, "label", None, multipass=None)
    if isinstance(label, str):
        return label
    return str(getattr(op, "layer_label", ""))


def intervention_site_and_cone_labels(
    trace: Trace,
    *,
    show_cone: bool,
) -> tuple[set[str], set[str]]:
    """Return intervention site and cone label sets for visualization.

    Parameters
    ----------
    trace:
        Log whose intervention spec should be inspected.
    show_cone:
        Whether to include downstream cone members.

    Returns
    -------
    tuple[set[str], set[str]]
        Site labels and cone labels. Cone labels exclude site labels.
    """

    sites = intervention_sites_for_log(trace)
    site_labels = {site.layer_label for site in sites}
    if not sites or not show_cone:
        return site_labels, set()

    from ..intervention.replay import cone_of_effect

    cone = cone_of_effect(trace, sites)
    cone_labels = {site.layer_label for site in cone}
    return site_labels, cone_labels - site_labels


def _cone_pass_labels(trace: Trace, sites: list[Op], site_pass_labels: set[str]) -> set[str]:
    """Return the pass-qualified labels downstream of ``sites``, sites excluded."""

    from ..intervention.replay import cone_of_effect

    return {_site_sort_key(op) for op in cone_of_effect(trace, sites)} - site_pass_labels


def _is_downstream(
    layer_log: Any,
    target: Any,
    pass_label: Any,
    *,
    cone_pass_labels: set[str],
    cone_labels: set[str],
) -> bool:
    """Return whether a drawn node lies downstream of the edit.

    One pass of an unrolled layer (``target`` is its rendered Op) is downstream
    only if that pass is. A rolled node matches the aggregate cone by its layer
    label, short label or any per-pass call label.
    """

    if target is not layer_log and isinstance(pass_label, str):
        return pass_label in cone_pass_labels
    cone_match_labels = {
        str(getattr(layer_log, "layer_label", "")),
        str(getattr(layer_log, "layer_label_short", "")),
    }
    cone_match_labels.update(str(label) for label in getattr(layer_log, "call_labels", ()) or ())
    return bool(cone_match_labels & cone_labels)


def _with_svg_class(extra_attrs: dict[str, str], name: str) -> dict[str, str]:
    """Return ``extra_attrs`` with ``name`` added to its Graphviz ``class`` list."""

    classes = str(extra_attrs.get("class", "")).split()
    if name not in classes:
        classes.append(name)
    return {**extra_attrs, "class": " ".join(classes)}


def make_intervention_node_spec_fn(
    trace: Trace,
    *,
    show_cone: bool,
    graph_overrides: dict[str, Any] | None,
    user_node_spec_fn: NodeSpecFn | None,
    theme: VisualizationTheme | None = None,
) -> NodeSpecFn | None:
    """Build a node callback that marks edited nodes and tags downstream ones.

    An edited node keeps its fill, shape and dash and gets the edit purple
    ring (2.5 pt) and a last label row naming the edit (``Δ zero_ablate``),
    with the full spelling in its tooltip. With ``show_cone`` a node the edit
    changed downstream keeps its normal outline and only gains the
    ``tl_downstream`` SVG class; the renderer lays the translucent
    highlighter under it and its out-edges after layout
    (:mod:`._edit_highlighter`).

    Parameters
    ----------
    trace:
        Log whose intervention spec should be visualized.
    show_cone:
        Whether downstream nodes should be tagged for the highlighter.
    graph_overrides:
        Graph override dictionary, including optional intervention style keys.
    user_node_spec_fn:
        Existing user callback to run after TorchLens intervention styling.
    theme:
        Resolved skin; the dark skin draws the edit in its lighter purple.

    Returns
    -------
    NodeSpecFn | None
        Combined callback, or the original callback when there is no
        intervention overlay to apply.
    """

    from ._edit_marks import (
        DOWNSTREAM_NODE_CLASS,
        EDIT_RING_PENWIDTH,
        EDITED_NODE_CLASS,
        edit_ink,
        edit_marks_by_site,
        edit_marks_for_node,
        edit_rows,
        edit_tooltip,
    )

    # Build BOTH a pass-qualified site set (for unrolled per-pass Op nodes) and an
    # aggregate site set (for rolled multi-pass Layer nodes). cone_of_effect
    # traverses pass-qualified, but this overlay deliberately AGGREGATES the
    # cone to layer_label (rolled nodes are layer-level), reusing the stable
    # public helper's layer-label sets.
    site_ops = intervention_sites_for_log(trace)
    site_pass_labels: set[str] = set()
    site_agg_labels: set[str] = set()
    site_keys_by_layer: dict[str, list[str]] = {}
    for site_op in site_ops:
        pass_label = get_multipass_attr(site_op, "label", None, multipass=None)
        if isinstance(pass_label, str):
            site_pass_labels.add(pass_label)
        agg_label = getattr(site_op, "layer_label", None)
        if isinstance(agg_label, str):
            site_agg_labels.add(agg_label)
            site_keys_by_layer.setdefault(agg_label, []).append(_site_sort_key(site_op))
    _, cone_labels = intervention_site_and_cone_labels(trace, show_cone=show_cone)
    if not site_pass_labels and not site_agg_labels and not cone_labels:
        return user_node_spec_fn
    cone_pass_labels = (
        _cone_pass_labels(trace, site_ops, site_pass_labels) if cone_labels and site_ops else set()
    )
    from ._render_nodes import _SPEC_SLOT_RENDERED_NODE

    marks = edit_marks_by_site(trace, site_ops)
    label_size = theme.typography.base_size if theme is not None else 14.0
    site_color = str(
        intervention_graph_override(graph_overrides, "intervention_site_color", edit_ink(theme))
    )
    site_penwidth = float(
        intervention_graph_override(
            graph_overrides, "intervention_site_penwidth", EDIT_RING_PENWIDTH
        )
    )

    def intervention_node_spec_fn(layer_log: Layer, default_spec: NodeSpec) -> NodeSpec:
        """Apply intervention styling before any user node-spec callback."""

        # The slot hands an unrolled node its aggregate Layer; the rendered
        # per-pass Op rides the slot's context, so read the pass from it.
        rendered = _SPEC_SLOT_RENDERED_NODE.get()
        target = layer_log if rendered is None else rendered
        # A per-pass Op resolves a pass-qualified ``label`` -> match the
        # pass-qualified site set so a single-pass intervention colors ONLY its own
        # pass in unrolled mode. A rolled aggregate Layer has no single pass
        # (``label`` would trip the multi-pass tripwire, so get_multipass_attr
        # returns None) -> match its layer_label against the aggregate set, coloring
        # the one rolled node when any of its passes is a site.
        pass_label = get_multipass_attr(target, "label", None, multipass=None)
        node_agg_label = str(getattr(layer_log, "layer_label", ""))
        if isinstance(pass_label, str):
            is_site = pass_label in site_pass_labels
        else:
            is_site = node_agg_label in site_agg_labels

        is_downstream = _is_downstream(
            layer_log,
            target,
            pass_label,
            cone_pass_labels=cone_pass_labels,
            cone_labels=cone_labels,
        )

        spec = default_spec
        if is_site:
            node_marks = edit_marks_for_node(marks, target, site_keys_by_layer)
            tooltip = edit_tooltip(node_marks)
            spec = spec.replace(
                color=site_color,
                penwidth=site_penwidth,
                lines=[*spec.lines, *edit_rows(node_marks, site_color, label_size)],
                tooltip=f"{spec.tooltip}; {tooltip}" if spec.tooltip else tooltip,
                extra_attrs=_with_svg_class(spec.extra_attrs, EDITED_NODE_CLASS),
            )
        elif is_downstream:
            spec = spec.replace(
                extra_attrs=_with_svg_class(spec.extra_attrs, DOWNSTREAM_NODE_CLASS),
            )

        if user_node_spec_fn is None:
            return spec
        user_result = user_node_spec_fn(layer_log, spec)
        return spec if user_result is None else user_result

    return intervention_node_spec_fn
