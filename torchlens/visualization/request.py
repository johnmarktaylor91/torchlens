"""Resolved visualization requests and output targets."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Literal, Mapping

from .._literals import (
    BufferVisibilityLiteral,
    CollapseLiteral,
    FoldRepeatsLiteral,
    VisDirectionLiteral,
    VisInterventionModeLiteral,
    VisModeLiteral,
    VisNodeModeLiteral,
    VisNodePlacementLiteral,
    VisRendererLiteral,
)

ShowContainersLiteral = Literal[False, "labels", "cluster", "collapsed", "auto", "nodes"]


@dataclass(frozen=True)
class RenderTarget:
    """Output-only destination for a rendered visualization.

    Parameters
    ----------
    outpath:
        Rendered artifact path without the selected file extension.
    fileformat:
        Graphviz output format.
    save_only:
        Whether interactive display should be suppressed.
    viewer:
        Whether a local viewer should be opened after rendering.
    renderer_name:
        Requested renderer implementation.
    """

    outpath: str = "modelgraph"
    fileformat: str = "pdf"
    save_only: bool = False
    viewer: bool = True
    renderer_name: VisRendererLiteral = "graphviz"


@dataclass(frozen=True)
class ResolvedRenderRequest:
    """Frozen, renderer-semantic closure for one ``Trace.draw`` request.

    This type also replaces the former narrow ``RenderContext`` used by the
    collapse subsystem.  Defaults retain that context's standalone behavior.

    Parameters
    ----------
    vis_mode:
        Render granularity.
    show_buffer_layers:
        Normalized buffer visibility policy.
    show_containers:
        Container presentation policy.
    engine:
        Requested node-placement engine.
    skip_fn:
        Optional predicate for hiding rendered nodes.
    """

    vis_mode: VisModeLiteral = "unrolled"
    show_buffer_layers: BufferVisibilityLiteral = "meaningful"
    show_containers: ShowContainersLiteral = False
    engine: VisNodePlacementLiteral = "dot"
    skip_fn: Callable[[Any], bool] | None = None
    vis_call_depth: int = 1000
    module: Any = None
    node_mode: VisNodeModeLiteral = "default"
    node_spec_fn: Callable[..., Any] | None = None
    collapsed_node_spec_fn: Callable[..., Any] | None = None
    collapse_fn: Callable[[Any], bool] | None = None
    collapse: CollapseLiteral = "none"
    fold_repeats: FoldRepeatsLiteral = None
    graph_overrides: Mapping[str, Any] | None = None
    edge_overrides: Mapping[str, Any] | None = None
    grad_edge_overrides: Mapping[str, Any] | None = None
    module_overrides: Mapping[str, Any] | None = None
    overrides: Any = None
    theme: str = "torchlens"
    intervention_mode: VisInterventionModeLiteral = "node_mark"
    show_cone: bool = True
    code_panel: Any = False
    node_overlay: Any = None
    node_label_fields: tuple[str, ...] | None = None
    show_legend: bool = False
    font_size: int | None = None
    dpi: int | None = None
    for_paper: bool = False
    return_graph: bool = False
    order_siblings: bool = True
    container_max_inline: int = 12
    show_input_transform_summary: bool = False
    show_orphans: bool = False
    direction: VisDirectionLiteral = "bottomup"

    def with_resolved_collapse(
        self,
        collapse_fn: Callable[[Any], bool] | None,
    ) -> "ResolvedRenderRequest":
        """Return this request with its resolved collapse predicate.

        Parameters
        ----------
        collapse_fn:
            Predicate selected from the request's public collapse options.

        Returns
        -------
        ResolvedRenderRequest
            Frozen request carrying the resolved predicate.
        """

        return replace(self, collapse_fn=collapse_fn)

    def __hash__(self) -> int:
        """Hash the collapse-planning subset of this request.

        Returns
        -------
        int
            Stable hash for existing collapse-plan caches.  The complete
            request may carry mutable Graphviz override mappings.
        """

        return hash(
            (
                self.vis_mode,
                self.show_buffer_layers,
                self.show_containers,
                self.engine,
                self.skip_fn,
            )
        )


# Kept as an import-compatible name while internal consumers migrate to the
# complete request vocabulary.
RenderContext = ResolvedRenderRequest
