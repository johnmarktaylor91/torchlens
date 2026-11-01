"""The one ``Bundle.draw`` door: the bundle graph and the member diff picture."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, cast

import torch

if TYPE_CHECKING:
    from .._data_classes.trace import Trace
    from .._literals import VisDirectionLiteral
    from . import Bundle

BundleGraphLiteral = Literal["forward", "backward", "both", "overlay", "diff"]


def bundle_draw(
    bundle: Bundle,
    path: str | None,
    *,
    graph: BundleGraphLiteral,
    nodes: Literal["ops", "members"],
    rolled: bool,
    direction: VisDirectionLiteral,
    theme: str | None,
    show_legend: bool | None,
    file_format: str | None,
    open_viewer: bool,
    graph_overrides: dict[str, Any] | None,
    node_overrides: dict[str, Any] | None,
    edge_overrides: dict[tuple[str, str], Any] | None,
    members: tuple[str | Trace, str | Trace] | None,
    metric: str | Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    on: Literal["out", "grad"],
) -> str:
    """Render a bundle picture and return its DOT source (``Bundle.draw``).

    ``graph="diff"`` draws two members side by side with their aligned
    sites coloured by ``metric`` distance (the former ``show_diff``); every
    other ``graph`` value draws the bundle's union graph, or with
    ``nodes="members"`` the member relation graph.
    """

    if graph == "diff":
        from .._visualization.bundle_diff import bundle_diff

        left, right = members if members is not None else (None, None)
        return bundle_diff(
            bundle,
            metric=metric,
            left=left,
            right=right,
            on=on,
            path=path if path is not None else "bundle_diff",
            vis_save_only=not open_viewer,
            file_format=file_format if file_format is not None else "svg",
            theme=theme if theme is not None else "paper",
        )

    from .._user_public_impls import show_bundle_graph

    dot = show_bundle_graph(
        bundle,
        path=path if path is not None else "bundle_modelgraph",
        vis_mode="rolled" if rolled else "unrolled",
        signal=graph,
        vis_direction=direction,
        vis_graph_overrides=graph_overrides,
        vis_node_overrides=node_overrides,
        vis_edge_overrides=edge_overrides,
        vis_save_only=not open_viewer,
        file_format=file_format if file_format is not None else "pdf",
        nodes=nodes,
        theme=theme if theme is not None else "torchlens",
        show_legend=show_legend,
    )
    return cast("str", dot)
