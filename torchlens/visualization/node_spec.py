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

    Parameters
    ----------
    lines:
        Plain-text row contents. Special HTML characters are escaped.

    Returns
    -------
    str
        A string suitable for Graphviz ``label=<...>`` syntax.
    """

    rows = []
    for index, line in enumerate(lines):
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
            forward_site = getattr(site, "op", site)
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


def make_intervention_node_spec_fn(
    trace: Trace,
    *,
    show_cone: bool,
    graph_overrides: dict[str, Any] | None,
    user_node_spec_fn: NodeSpecFn | None,
) -> NodeSpecFn | None:
    """Build a node callback that applies intervention site/cone styling.

    Parameters
    ----------
    trace:
        Log whose intervention spec should be visualized.
    show_cone:
        Whether downstream cone members should be styled.
    graph_overrides:
        Graph override dictionary, including optional intervention style keys.
    user_node_spec_fn:
        Existing user callback to run after TorchLens intervention styling.

    Returns
    -------
    NodeSpecFn | None
        Combined callback, or the original callback when there is no
        intervention overlay to apply.
    """

    # Build BOTH a pass-qualified site set (for unrolled per-pass Op nodes) and an
    # aggregate site set (for rolled multi-pass Layer nodes). cone_of_effect
    # traverses pass-qualified, but this overlay deliberately AGGREGATES the
    # cone to layer_label (rolled nodes are layer-level), reusing the stable
    # public helper's layer-label sets.
    site_ops = intervention_sites_for_log(trace)
    site_pass_labels: set[str] = set()
    site_agg_labels: set[str] = set()
    for site_op in site_ops:
        pass_label = get_multipass_attr(site_op, "label", None, multipass=None)
        if isinstance(pass_label, str):
            site_pass_labels.add(pass_label)
        agg_label = getattr(site_op, "layer_label", None)
        if isinstance(agg_label, str):
            site_agg_labels.add(agg_label)
    _, cone_labels = intervention_site_and_cone_labels(trace, show_cone=show_cone)
    if not site_pass_labels and not site_agg_labels and not cone_labels:
        return user_node_spec_fn

    site_color = str(
        intervention_graph_override(
            graph_overrides, "intervention_site_color", INTERVENTION_SITE_COLOR
        )
    )
    cone_color = str(
        intervention_graph_override(
            graph_overrides, "intervention_cone_color", INTERVENTION_CONE_COLOR
        )
    )
    site_penwidth = float(
        intervention_graph_override(graph_overrides, "intervention_site_penwidth", 3.0)
    )
    cone_penwidth = float(
        intervention_graph_override(graph_overrides, "intervention_cone_penwidth", 1.75)
    )

    def intervention_node_spec_fn(layer_log: Layer, default_spec: NodeSpec) -> NodeSpec:
        """Apply intervention styling before any user node-spec callback."""

        # A per-pass Op resolves a pass-qualified ``label`` -> match the
        # pass-qualified site set so a single-pass intervention colors ONLY its own
        # pass in unrolled mode. A rolled aggregate Layer has no single pass
        # (``label`` would trip the multi-pass tripwire, so get_multipass_attr
        # returns None) -> match its layer_label against the aggregate set, coloring
        # the one rolled node when any of its passes is a site.
        pass_label = get_multipass_attr(layer_log, "label", None, multipass=None)
        node_agg_label = str(getattr(layer_log, "layer_label", ""))
        if isinstance(pass_label, str):
            is_site = pass_label in site_pass_labels
        else:
            is_site = node_agg_label in site_agg_labels

        cone_match_labels = {
            node_agg_label,
            str(getattr(layer_log, "layer_label_short", "")),
        }
        cone_match_labels.update(
            str(label) for label in getattr(layer_log, "call_labels", ()) or ()
        )

        spec = default_spec
        if is_site:
            spec = spec.replace(color=site_color, penwidth=site_penwidth)
        elif cone_match_labels & cone_labels:
            spec = spec.replace(color=cone_color, penwidth=cone_penwidth)

        if user_node_spec_fn is None:
            return spec
        user_result = user_node_spec_fn(layer_log, spec)
        return spec if user_result is None else user_result

    return intervention_node_spec_fn
