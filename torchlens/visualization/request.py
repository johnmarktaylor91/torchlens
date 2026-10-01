"""Resolved visualization requests and output targets."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any, Literal

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
    graph_name:
        Optional backend graph identifier.
    graph_comment:
        Optional backend graph comment.
    timeout:
        Maximum layout-execution time in seconds.
    """

    outpath: str = "modelgraph"
    fileformat: str = "pdf"
    save_only: bool = False
    viewer: bool = True
    renderer_name: VisRendererLiteral = "graphviz"
    graph_name: str | None = None
    graph_comment: str | None = None
    timeout: int = 120


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
    # Tri-state (L5 channel core): None = AUTO -- no legend unless an
    # encoding channel is active, then a channel-only disclosure legend.
    # True/False keep their historical meanings; explicit False is honored
    # even with channels active (a deliberate act).
    show_legend: bool | None = None
    # Encoding channel core (L5, DOCUMENTED-UNSTABLE until ratified):
    # ``color_by``/``size_by``/``scale`` are the raw user sources;
    # ``encoding`` carries the resolved per-draw EncodingState (all active
    # channels). Presentation-only: NONE joins __hash__ (the
    # collapse-planning subset is unchanged -- channels must never affect the
    # collapse plan).
    color_by: Any = None
    size_by: Any = None
    scale: Any = None
    stack_by: Any = None
    encoding: Any = None
    # Checked suppression (L5 M4, DOCUMENTED-UNSTABLE spelling): True shows
    # every constructor arg; False (default) suppresses args the equality
    # check proves redundant against this trace's captured shapes.
    show_redundant_args: bool = False
    # Saved-for-backward annotation (DOCUMENTED-UNSTABLE spelling): True adds
    # a label row on every op whose grad_fn retained tensors for backward,
    # from the captured autograd_memory/num_autograd_tensors measurements.
    show_saved_for_backward: bool = False
    font_size: int | None = None
    dpi: int | None = None
    for_paper: bool = False
    return_graph: bool = False
    order_siblings: bool = True
    container_max_inline: int = 12
    show_input_transform_summary: bool = False
    show_orphans: bool = False
    direction: VisDirectionLiteral = "bottomup"
    # Declarative user-named pattern folding (F11, collapse memo D11 / K5;
    # DOCUMENTED-UNSTABLE spelling): resolved PatternSpec records, () = off.
    # Joins __hash__: pattern chips change what renders, so a pattern-bearing
    # request must never share cached plans with a bare one (plan-cache
    # identity is part of the D11 contract).
    fold_patterns: tuple[Any, ...] = ()

    def with_resolved_collapse(
        self,
        collapse_fn: Callable[[Any], bool] | None,
    ) -> ResolvedRenderRequest:
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
                self.fold_patterns,
            )
        )


# Kept as an import-compatible name while internal consumers migrate to the
# complete request vocabulary.
RenderContext = ResolvedRenderRequest
