"""NodeSpec helpers for TorchLens graph visualization labels."""

from __future__ import annotations

from dataclasses import dataclass, field, replace as dataclass_replace
from html import escape
from typing import TYPE_CHECKING, Any, Callable, cast

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..data_classes.layer import Layer
    from ..data_classes.op import Op
    from ..data_classes.trace import Trace

INTERVENTION_SITE_COLOR = "#FF00FF"
INTERVENTION_CONE_COLOR = "#FFB3FF"
INTERVENTION_HOOK_FILL_COLOR = "#FFE6FF"
INTERVENTION_HOOK_BORDER_COLOR = "#CC00CC"
INTERVENTION_OVERRIDE_KEYS = frozenset(
    {
        "intervention_site_color",
        "intervention_cone_color",
        "intervention_site_penwidth",
        "intervention_cone_penwidth",
        "intervention_hook_fillcolor",
        "intervention_hook_color",
        "intervention_hook_penwidth",
    }
)


@dataclass
class NodeSpec:
    """Graphviz node attributes produced by TorchLens before user customization.

    The dataclass is intentionally mutable because visualization callbacks are
    user ergonomics APIs: mutating and returning the supplied default spec is a
    natural pattern for small display tweaks.

    Attributes
    ----------
    lines:
        Plain-text rows to render in the node label.
    shape:
        Graphviz node shape.
    fillcolor:
        Optional fill color.
    fontcolor:
        Optional font color.
    style:
        Graphviz node style.
    color:
        Optional border color.
    penwidth:
        Optional border width.
    tooltip:
        Optional node tooltip.
    image:
        Optional image path to embed in the node.
    extra_attrs:
        Additional Graphviz node attributes.
    """

    lines: list[str]
    shape: str = "box"
    fillcolor: str | None = None
    fontcolor: str | None = None
    style: str = "filled,rounded"
    color: str | None = None
    penwidth: float | None = None
    tooltip: str | None = None
    image: str | None = None
    extra_attrs: dict[str, str] = field(default_factory=dict)

    def replace(self, **kwargs: Any) -> "NodeSpec":
        """Return a copy of this spec with selected fields replaced.

        Parameters
        ----------
        **kwargs:
            Dataclass fields to replace.

        Returns
        -------
        NodeSpec
            A copied ``NodeSpec`` with the requested field changes.
        """

        return dataclass_replace(self, **kwargs)


NodeSpecFn = Callable[["Layer", NodeSpec], NodeSpec | None]


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


def intervention_sites_for_log(trace: "Trace") -> list["Op"]:
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
            layer_label = getattr(forward_site, "layer_label", None)
            if layer_label is not None:
                by_label.setdefault(layer_label, cast("Op", forward_site))
    execution_order = {
        layer.layer_label: index for index, layer in enumerate(getattr(trace, "layer_list", ()))
    }
    return sorted(by_label.values(), key=lambda site: execution_order.get(site.layer_label, 0))


def intervention_site_and_cone_labels(
    trace: "Trace",
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
    trace: "Trace",
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

    site_labels, cone_labels = intervention_site_and_cone_labels(trace, show_cone=show_cone)
    if not site_labels and not cone_labels:
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

    def intervention_node_spec_fn(layer_log: "Layer", default_spec: NodeSpec) -> NodeSpec:
        """Apply intervention styling before any user node-spec callback."""

        matching_labels = {
            str(getattr(layer_log, "layer_label", "")),
            str(getattr(layer_log, "layer_label", "")),
        }
        call_labels = getattr(layer_log, "call_indexs", None)
        if call_labels is None:
            call_labels = ()
        for call_index in call_labels:
            base = str(getattr(layer_log, "layer_label", ""))
            matching_labels.add(f"{base}:{call_index}")

        spec = default_spec
        if matching_labels & site_labels:
            spec = spec.replace(color=site_color, penwidth=site_penwidth)
        elif matching_labels & cone_labels:
            spec = spec.replace(color=cone_color, penwidth=cone_penwidth)

        if user_node_spec_fn is None:
            return spec
        user_result = user_node_spec_fn(layer_log, spec)
        return spec if user_result is None else user_result

    return intervention_node_spec_fn
