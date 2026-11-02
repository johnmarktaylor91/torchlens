"""The one ``Bundle.draw`` door (the bundle graph and the member diff picture) and ``Bundle.show``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from .._errors import ArgumentTypeError, InvalidArgumentError

if TYPE_CHECKING:
    from ..viz._drawing import Drawing, DrawTarget
    from . import Bundle

BundleGraphLiteral = Literal["forward", "backward", "combined", "diff"]
_PASS_SIGNALS = {"forward": "forward", "backward": "backward", "combined": "both"}

#: ``Bundle.draw``'s keywords, named by the refusal for any other keyword.
BUNDLE_DRAW_KEYWORDS = (
    "path",
    "graph",
    "nodes",
    "rolled",
    "direction",
    "theme",
    "show_legend",
    "file_format",
    "open_viewer",
    "graph_overrides",
    "node_overrides",
    "edge_overrides",
    "members",
    "metric",
    "on",
)
_SHOW_KINDS = (None, "graph", "strip", "repr", "html")


def bundle_draw(
    bundle: Bundle, path: str | None, unsupported: dict[str, Any], **options: Any
) -> Drawing:
    """Render a bundle picture and return it as a :class:`Drawing` (``Bundle.draw``).

    ``options`` are ``Bundle.draw``'s keywords; ``unsupported`` collects any
    other keyword, which refuses (``Trace.draw``'s other keywords are added one
    by one as the bundle renderer supports them). ``graph="diff"`` draws two
    members side by side with their aligned sites coloured by ``metric``
    distance; every other ``graph`` value draws the bundle's union graph, or
    with ``nodes="members"`` the member relation graph.
    """

    from .._visualization._render_utils import (
        _KNOWN_EXTS,
        announce_written_graph,
        collision_safe_outpath,
    )
    from ..viz._drawing import resolve_draw_target

    if unsupported:
        raise ArgumentTypeError(
            f"Bundle.draw does not take {', '.join(sorted(unsupported))}",
            code="bundle_draw_keyword_unsupported",
            remedy=f"use the keywords Bundle.draw supports: {', '.join(BUNDLE_DRAW_KEYWORDS)}",
            argument=sorted(unsupported)[0],
        )
    graph = options["graph"]
    if graph not in ("diff", *_PASS_SIGNALS):
        raise InvalidArgumentError(
            f"graph must be 'forward', 'backward', 'combined' or 'diff'; received {graph!r}",
            code="bundle_draw_graph_invalid",
            remedy="pass graph='forward', 'backward', 'combined' or 'diff'",
            argument="graph",
        )
    default_format = "svg" if graph == "diff" else "pdf"
    target = resolve_draw_target(path, options["file_format"], _KNOWN_EXTS, default_format)
    outpath = target.path
    if outpath is None:
        outpath = collision_safe_outpath(
            "bundle_diff" if graph == "diff" else "bundle_modelgraph", target.file_format
        )
    drawn = _draw_diff if graph == "diff" else _draw_graph
    dot = drawn(bundle, target, outpath, options)
    if target.path is None:
        announce_written_graph(f"{outpath}.{target.file_format}")
    policy = {"graph": graph, "nodes": options["nodes"], "rolled": options["rolled"]}
    return target.finish(dot, outpath, policy=policy)


def _draw_diff(bundle: Bundle, target: DrawTarget, outpath: str, options: dict[str, Any]) -> str:
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
        path=target.stem or outpath,
        vis_save_only=not options["open_viewer"],
        file_format=target.file_format,
        theme=theme if theme is not None else "paper",
    )


def _draw_graph(bundle: Bundle, target: DrawTarget, outpath: str, options: dict[str, Any]) -> str:
    """The union (or member relation) graph."""

    from .._visualization._bundle_graph import BundlePictureOptions, render_bundle_graph
    from .._visualization._render_utils import direction_to_rankdir

    nodes = options["nodes"]
    if nodes not in ("ops", "members"):
        raise ValueError(f"nodes must be 'ops' or 'members'; received {nodes!r}.")
    if nodes == "members" and options["graph"] != "forward":
        raise ValueError("nodes='members' draws the relation graph; leave graph='forward'.")
    direction = options["direction"]
    theme = options["theme"]
    picture = BundlePictureOptions(
        outpath=target.stem or outpath,
        vis_mode="rolled" if options["rolled"] else "unrolled",
        direction=_PASS_SIGNALS[options["graph"]],
        rankdir=direction_to_rankdir(direction if direction is not None else "bottomup"),
        graph_overrides=options["graph_overrides"],
        node_overrides=options["node_overrides"],
        edge_overrides=options["edge_overrides"],
        save_only=not options["open_viewer"],
        fileformat=target.file_format,
        nodes=nodes,
        theme_name=theme if theme is not None else "torchlens",
        show_legend=options["show_legend"] is True,
    )
    return render_bundle_graph(bundle, picture)


def bundle_show(
    bundle: Bundle, kind: str | None, kwargs: dict[str, Any]
) -> Drawing | dict[str, Drawing | str]:
    """``Bundle.show``: the merged graph by default, or a member-keyed strip."""

    if kind not in _SHOW_KINDS:
        raise InvalidArgumentError(
            f"Bundle.show kind must be None, 'graph', 'strip', 'repr' or 'html'; received {kind!r}",
            code="bundle_show_kind_invalid",
            remedy="pass kind=None (the merged graph), 'graph', 'strip', 'repr' or 'html'",
            argument="kind",
        )
    if kind in (None, "graph"):
        return bundle.draw(**kwargs)
    base_outpath = kwargs.get("path")
    results: dict[str, Drawing | str] = {}
    for name, member in bundle._members.items():
        if kind == "repr":
            results[name] = repr(member)
        elif kind == "html":
            results[name] = member._repr_html_()
        else:
            member_kwargs = dict(kwargs)
            if isinstance(base_outpath, str):
                member_kwargs["path"] = f"{base_outpath}_{name}"
            results[name] = member.draw(**member_kwargs)
    return results
