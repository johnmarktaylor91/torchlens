"""NodeSpec helpers for TorchLens graph visualization labels."""

from __future__ import annotations

from dataclasses import dataclass, field, replace as dataclass_replace
from html import escape
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..data_classes.layer_log import LayerLog
    from ..data_classes.layer_pass_log import LayerPassLog
    from ..data_classes.model_log import ModelLog

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


NodeSpecFn = Callable[["LayerLog", NodeSpec], NodeSpec | None]


def render_lines_to_html(lines: list[str]) -> str:
    """Render plain-text node rows as a Graphviz HTML-like table label.

    Parameters
    ----------
    lines:
        Plain-text row contents. Special HTML characters are escaped.

    Returns
    -------
    str
        A string suitable for Graphviz ``label=<...>`` syntax.
    """

    rows = [f'<TR><TD ALIGN="CENTER">{escape(str(line), quote=False)}</TD></TR>' for line in lines]
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


def intervention_sites_for_log(model_log: "ModelLog") -> list["LayerPassLog"]:
    """Resolve intervention spec targets to layer-pass records.

    Parameters
    ----------
    model_log:
        Log whose intervention spec should be inspected.

    Returns
    -------
    list[LayerPassLog]
        Distinct intervention sites in execution order.
    """

    spec = getattr(model_log, "_intervention_spec", None)
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

    by_label: dict[str, LayerPassLog] = {}
    for target in targets:
        table = resolve_sites(model_log, target, max_fanout=max(1, len(model_log.layer_list)))
        for site in table:
            by_label.setdefault(site.layer_label, site)
    execution_order = {
        layer.layer_label: index for index, layer in enumerate(getattr(model_log, "layer_list", ()))
    }
    return sorted(by_label.values(), key=lambda site: execution_order.get(site.layer_label, 0))


def intervention_site_and_cone_labels(
    model_log: "ModelLog",
    *,
    show_cone: bool,
) -> tuple[set[str], set[str]]:
    """Return intervention site and cone label sets for visualization.

    Parameters
    ----------
    model_log:
        Log whose intervention spec should be inspected.
    show_cone:
        Whether to include downstream cone members.

    Returns
    -------
    tuple[set[str], set[str]]
        Site labels and cone labels. Cone labels exclude site labels.
    """

    sites = intervention_sites_for_log(model_log)
    site_labels = {site.layer_label for site in sites}
    if not sites or not show_cone:
        return site_labels, set()

    from ..intervention.replay import cone_of_effect

    cone = cone_of_effect(model_log, sites)
    cone_labels = {site.layer_label for site in cone}
    return site_labels, cone_labels - site_labels


def make_intervention_node_spec_fn(
    model_log: "ModelLog",
    *,
    show_cone: bool,
    graph_overrides: dict[str, Any] | None,
    user_node_spec_fn: NodeSpecFn | None,
) -> NodeSpecFn | None:
    """Build a node callback that applies intervention site/cone styling.

    Parameters
    ----------
    model_log:
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

    site_labels, cone_labels = intervention_site_and_cone_labels(model_log, show_cone=show_cone)
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

    def intervention_node_spec_fn(layer_log: "LayerLog", default_spec: NodeSpec) -> NodeSpec:
        """Apply intervention styling before any user node-spec callback."""

        matching_labels = {
            str(getattr(layer_log, "layer_label", "")),
            str(getattr(layer_log, "layer_label_no_pass", "")),
        }
        pass_labels = getattr(layer_log, "pass_nums", None)
        if pass_labels is None:
            pass_labels = ()
        for pass_num in pass_labels:
            base = str(getattr(layer_log, "layer_label_no_pass", ""))
            matching_labels.add(f"{base}:{pass_num}")

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
