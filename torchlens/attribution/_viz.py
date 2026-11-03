"""The ``color_by=`` resolution of attribution results on ``Trace.draw``."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from torchlens.attribution._result import (
    _REDUCERS,
    _attribution_painter,
    _overlay_refusal,
)


def _graph_site_values(graph: Any, graph_value: str) -> dict[str, list[float]]:
    """Each site's node values: influence of feature and error nodes, or feature activations."""

    by_site: dict[str, list[float]] = {}
    if graph_value == "influence":
        from torchlens.attribution.graph._neuronpedia import node_influence

        influence = node_influence(graph).tolist()
        for index, node in enumerate(graph.nodes):
            if node.kind in ("feature", "error"):
                by_site.setdefault(node.site, []).append(influence[index])
        return by_site
    activations = graph.feature_activations.detach().cpu().tolist()
    for index in graph.node_indices("feature"):
        by_site.setdefault(graph.nodes[index].site, []).append(activations[index])
    return by_site


def _graph_entries(trace: Any, graph: Any, graph_value: str, reduce: str) -> dict[str, float]:
    """Reduce an attribution graph's node values to one scalar per site of ``trace``.

    Raises
    ------
    AttributionError
        ``attribution_overlay_graph_mismatch`` when a site of the graph is
        neither a module output nor a layer label of ``trace``.
    """

    known: set[str] = set()
    for label in trace.layer_labels:
        known.add(label)
        known.update(trace[label].module_context.output_of_modules)
    by_site = _graph_site_values(graph, graph_value)
    unknown = sorted(site for site in by_site if site not in known)
    if unknown:
        raise _overlay_refusal(
            f"The attribution graph holds nodes at {unknown[:3]!r}, which this trace has no "
            "module output or layer label for, so the graph was not built from this trace. "
            "Remedy: draw the trace the graph was built from, "
            "trace.draw(color_by=graph).",
            code="attribution_overlay_graph_mismatch",
        )
    return {
        site: _REDUCERS[reduce](torch.tensor(values, dtype=torch.float64))
        for site, values in by_site.items()
    }


def _graph_value_for(source: Any) -> str | None:
    """``"influence"`` for an attribution graph source, ``None`` otherwise."""

    return "influence" if getattr(type(source), "_color_by_kind", None) == "graph" else None


def attribution_color_by(trace: Any, source: Any) -> Callable[[Any], float | None]:
    """Build the ``color_by`` callable that paints attribution values onto a graph.

    This is what ``trace.draw(color_by=<attribution result>)`` resolves to.
    The callable maps rendered nodes to scalar magnitudes: nodes that are
    the output of an attributed module (or match an attributed layer label)
    receive that layer's ``abs_sum`` value, and every other node returns
    ``None`` (drawn unencoded, disclosed by the legend's ``n/a`` note). A
    module fired several times paints its per-layer TOTAL on every node of
    that module; per-pass splits are not claimed. Another reduction is a
    mapping the caller computes and passes to ``color_by`` directly.

    An :class:`~torchlens.attribution.graph.AttributionGraph` source paints
    each site holding feature or error nodes with the ``abs_sum`` of its
    nodes' influence on the logits (circuit-tracer's rule, the quantity
    :meth:`AttributionGraph.to_neuronpedia` prunes by). Token and logit nodes
    paint nothing.

    Parameters
    ----------
    trace
        Trace the picture is drawn on; keys are validated against it.
    source
        One layer-scoped :class:`AttributionResult` (``extra['layer']``
        present), an iterable of them, a mapping from
        ``model.named_modules()`` name or trace layer label to a result,
        tensor, or real scalar, or an :class:`AttributionGraph` built from
        ``trace``.

    Raises
    ------
    AttributionError
        If a key matches neither a module output nor a layer label on
        ``trace``; ``attribution_overlay_graph_mismatch`` for a graph whose
        sites are not on ``trace``.
    """

    reduce = "abs_sum"
    painted = _graph_value_for(source)
    if painted is not None:
        source = _graph_entries(trace, source, painted, reduce)
    return _attribution_painter(trace, source, reduce)


__all__ = ["attribution_color_by"]
