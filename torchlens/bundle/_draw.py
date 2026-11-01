"""The one ``Bundle.draw`` door: the bundle graph and the member diff picture."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, cast

import torch

from .._errors import InvalidArgumentError

if TYPE_CHECKING:
    from .._data_classes.trace import Trace
    from .._literals import VisDirectionLiteral
    from . import Bundle

BundleGraphLiteral = Literal["forward", "backward", "combined", "diff"]
_PASS_SIGNALS = {"forward": "forward", "backward": "backward", "combined": "both"}


def bundle_draw(
    bundle: Bundle,
    path: str | None,
    *,
    graph: BundleGraphLiteral,
    nodes: Literal["ops", "members"],
    rolled: bool,
    direction: VisDirectionLiteral | None,
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

    from .._visualization._render_utils import _KNOWN_EXTS, collision_safe_outpath

    if graph not in ("diff", *_PASS_SIGNALS):
        raise InvalidArgumentError(
            f"graph must be 'forward', 'backward', 'combined' or 'diff'; received {graph!r}",
            code="bundle_draw_graph_invalid",
            remedy="pass graph='forward', 'backward', 'combined' or 'diff'",
            argument="graph",
        )
    default_format = "svg" if graph == "diff" else "pdf"
    stem, fmt = _split_outpath(path, file_format, default_format, _KNOWN_EXTS)
    if stem is None:
        stem = collision_safe_outpath(
            "bundle_diff" if graph == "diff" else "bundle_modelgraph", fmt
        )
    if graph == "diff":
        from .._visualization.bundle_diff import bundle_diff

        left, right = members if members is not None else (None, None)
        return bundle_diff(
            bundle,
            metric=metric,
            left=left,
            right=right,
            on=on,
            path=stem,
            vis_save_only=not open_viewer,
            file_format=fmt,
            theme=theme if theme is not None else "paper",
        )

    from .._user_public_impls import show_bundle_graph

    dot = show_bundle_graph(
        bundle,
        path=stem,
        vis_mode="rolled" if rolled else "unrolled",
        signal=_PASS_SIGNALS[graph],
        vis_direction=direction if direction is not None else "bottomup",
        vis_graph_overrides=graph_overrides,
        vis_node_overrides=node_overrides,
        vis_edge_overrides=edge_overrides,
        vis_save_only=not open_viewer,
        file_format=fmt,
        nodes=nodes,
        theme=theme if theme is not None else "torchlens",
        show_legend=show_legend,
    )
    return cast("str", dot)


def _split_outpath(
    path: str | None, file_format: str | None, default_format: str, known: tuple[str, ...]
) -> tuple[str | None, str]:
    """Return ``(stem, format)``: the path's extension picks the format unless one is given."""

    if path is None:
        return None, file_format if file_format is not None else default_format
    parts = str(path).split(".")
    if len(parts) > 1 and parts[-1].lower() in known:
        return ".".join(parts[:-1]), file_format if file_format is not None else parts[-1].lower()
    return str(path), file_format if file_format is not None else default_format
