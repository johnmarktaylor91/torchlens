"""The one ``Bundle.draw`` door: the bundle graph and the member diff picture."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

from .._errors import InvalidArgumentError

if TYPE_CHECKING:
    from . import Bundle

BundleGraphLiteral = Literal["forward", "backward", "combined", "diff"]
_PASS_SIGNALS = {"forward": "forward", "backward": "backward", "combined": "both"}


def bundle_draw(bundle: Bundle, path: str | None, **options: Any) -> str:
    """Render a bundle picture and return its DOT source (``Bundle.draw``).

    ``options`` are ``Bundle.draw``'s keywords. ``graph="diff"`` draws two
    members side by side with their aligned sites coloured by ``metric``
    distance (the former ``show_diff``); every other ``graph`` value draws
    the bundle's union graph, or with ``nodes="members"`` the member
    relation graph.
    """

    from .._visualization._render_utils import _KNOWN_EXTS, collision_safe_outpath

    graph = options["graph"]
    if graph not in ("diff", *_PASS_SIGNALS):
        raise InvalidArgumentError(
            f"graph must be 'forward', 'backward', 'combined' or 'diff'; received {graph!r}",
            code="bundle_draw_graph_invalid",
            remedy="pass graph='forward', 'backward', 'combined' or 'diff'",
            argument="graph",
        )
    default_format = "svg" if graph == "diff" else "pdf"
    stem, fmt = _split_outpath(path, options["file_format"], default_format, _KNOWN_EXTS)
    if stem is None:
        stem = collision_safe_outpath(
            "bundle_diff" if graph == "diff" else "bundle_modelgraph", fmt
        )
    if graph == "diff":
        return _draw_diff(bundle, stem, fmt, options)
    return _draw_graph(bundle, stem, fmt, options)


def _draw_diff(bundle: Bundle, stem: str, fmt: str, options: dict[str, Any]) -> str:
    """The ``graph="diff"`` picture: ``bundle_diff`` over the two ``members``."""

    from .._visualization.bundle_diff import bundle_diff

    members = options["members"]
    left, right = members if members is not None else (None, None)
    theme = options["theme"]
    return bundle_diff(
        bundle,
        metric=options["metric"],
        left=left,
        right=right,
        on=options["on"],
        path=stem,
        vis_save_only=not options["open_viewer"],
        file_format=fmt,
        theme=theme if theme is not None else "paper",
    )


def _draw_graph(bundle: Bundle, stem: str, fmt: str, options: dict[str, Any]) -> str:
    """The union (or member relation) graph through ``show_bundle_graph``."""

    from .._user_public_graph_impls import show_bundle_graph

    direction = options["direction"]
    theme = options["theme"]
    dot = show_bundle_graph(
        bundle,
        path=stem,
        vis_mode="rolled" if options["rolled"] else "unrolled",
        signal=_PASS_SIGNALS[options["graph"]],
        vis_direction=direction if direction is not None else "bottomup",
        vis_graph_overrides=options["graph_overrides"],
        vis_node_overrides=options["node_overrides"],
        vis_edge_overrides=options["edge_overrides"],
        vis_save_only=not options["open_viewer"],
        file_format=fmt,
        nodes=options["nodes"],
        theme=theme if theme is not None else "torchlens",
        show_legend=options["show_legend"],
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
