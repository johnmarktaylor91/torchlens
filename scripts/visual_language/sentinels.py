"""The sentinel strip: a fixed set of renders that together show the look of TorchLens.

Every visual change is checked against all of it, before and after, side by side (the
review mode in ``review.py``). The fourteen rows are fixed: ids are stable, a row is never
renumbered, and a row whose subject does not exist yet is listed as pending with its
reason instead of being invented. Row 1 is drawn once per skin; every other row is drawn
in the default look (TorchLens's own draw defaults, not the teaching slides' conventions).

Like ``slides.py`` this module imports no torch, so tests and the coverage check read it
cheaply. Fixture names resolve through ``models.FIXTURES`` and
``sentinel_models.SENTINEL_FIXTURES``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from scripts.visual_language.slides import Panel, Slide

#: Only what every sentinel render needs to be written as an SVG file and not opened.
OUTPUT: Mapping[str, Any] = {"file_format": "svg", "vis_save_only": True}
SKINS = ("torchlens", "paper", "dark", "colorblind", "high_contrast")


def _panel(name: str, fixture: str, label: str, call: str = "draw", **kw: Any) -> Panel:
    """A default-look panel: TorchLens's own draw defaults plus SVG output, nothing else."""

    kwargs = {**OUTPUT, **kw.pop("kwargs", {})}
    return Panel(name, fixture, call=call, kwargs=kwargs, label=label, conventions=False, **kw)


@dataclass(frozen=True)
class Sentinel:
    """One row of the strip: what is drawn, how, and why it is in the set."""

    id: str
    slug: str
    models: tuple[str, ...]
    why: str
    panels: tuple[Panel, ...] = ()
    #: ``"active"`` rows render; ``"pending"`` rows name what is missing in ``reason``.
    status: str = "active"
    reason: str = ""
    #: Members of the row's subject not drawn yet, each with its reason.
    pending_parts: Mapping[str, str] = field(default_factory=dict)
    #: Words shown with the row (a stand-in spelling, say); empty for most rows.
    note: str = ""


_GRAD = {"save_grads": "all"}
_EDIT = {"intervention_ready": True, "save_arg_values": True}

SENTINELS: tuple[Sentinel, ...] = (
    Sentinel(
        "SN01",
        "anatomy",
        ("Tiny",),
        "The anatomy, and the only place skins and directions are compared cleanly.",
        (
            _panel("a", "Tiny", "show_legend=True", kwargs={"show_legend": True}),
            _panel("b", "Tiny", 'direction="topdown"', kwargs={"direction": "topdown"}),
            _panel("c", "Tiny", 'direction="leftright"', kwargs={"direction": "leftright"}),
            _panel("d", "Tiny", 'direction="rightleft"', kwargs={"direction": "rightleft"}),
        ),
    ),
    Sentinel(
        "SN02",
        "residual-norm",
        ("ResidualNorm",),
        "Module boxes and depth widths, dashed not-from-input, buffers, box3d collapse.",
        (
            _panel("a", "ResidualNorm", "default"),
            _panel("b", "ResidualNorm", "vis_call_depth=1", kwargs={"vis_call_depth": 1}),
            _panel("c", "ResidualNorm", 'collapse="max"', kwargs={"collapse": "max"}),
        ),
    ),
    Sentinel(
        "SN03",
        "recurrent",
        ("RNNCellSeq", "LoopShapes", "Loop"),
        "Unrolled against rolled, (xN), In/Out labels, back-edges, stack_by.",
        (
            _panel("a", "RNNCellSeq", "default (unrolled)"),
            _panel("b", "RNNCellSeq", 'vis_mode="rolled"', kwargs={"vis_mode": "rolled"}),
            _panel("c", "LoopShapes", 'vis_mode="rolled"', kwargs={"vis_mode": "rolled"}),
            _panel("d", "Loop", "stack_by=True", kwargs={"stack_by": True}),
        ),
    ),
    Sentinel(
        "SN04",
        "branches",
        ("BranchConv",),
        "Fan-in mid-arrow labels, sibling order, IF/ELSE arms, multiplicity.",
        (
            _panel("a", "BranchConv", "default"),
            _panel("b", "BranchConv", "order_siblings=False", kwargs={"order_siblings": False}),
        ),
    ),
    Sentinel(
        "SN05",
        "transformer",
        ("MiniTransformer",),
        "Attention topology, argument slots on matmul, +N more folding.",
        (
            _panel("a", "MiniTransformer", "default"),
            _panel("b", "MiniTransformer", "fold_repeats=True", kwargs={"fold_repeats": True}),
            _panel(
                "c",
                "MiniTransformer",
                'lens "transformer"',
                call="lens",
                kwargs={"lens": "transformer"},
            ),
        ),
    ),
    Sentinel(
        "SN06",
        "multi-output",
        ("MultiOut",),
        "Element nodes of multi-output ops and the container modes.",
        (
            _panel("a", "MultiOut", "default"),
            _panel("b", "MultiOut", 'show_containers="nodes"', kwargs={"show_containers": "nodes"}),
            _panel(
                "c",
                "MultiOut",
                'show_containers="collapsed"',
                kwargs={"show_containers": "collapsed"},
            ),
        ),
    ),
    Sentinel(
        "SN07",
        "intervention",
        ("ResidualNorm",),
        "Edit site, cone, the as_node diamond, the surgery lens and the surgery diff.",
        (
            _panel(
                "a",
                "ResidualNorm",
                'vis_intervention_mode="as_node"',
                intervene="zero_relu",
                kwargs={"vis_intervention_mode": "as_node"},
            ),
            _panel("b", "ResidualNorm", "render_surgery", call="surgery", capture=_EDIT),
            _panel(
                "c", "ResidualNorm", "surgery_diff(...).draw()", call="surgery_diff", capture=_EDIT
            ),
        ),
    ),
    Sentinel(
        "SN08",
        "bundle",
        ("ResidualNorm",),
        "The bundle graph and the bundle diff of clean, intervened and successor traces.",
        (
            _panel("a", "ResidualNorm", "show_bundle_graph", call="bundle_graph", capture=_EDIT),
            _panel("b", "ResidualNorm", "bundle_diff", call="bundle_diff", capture=_EDIT),
        ),
        pending_parts={"relation page": "the bundle relation page (11.30) is not built yet"},
    ),
    Sentinel(
        "SN09",
        "gradients",
        ("Grad", "HigherOrder"),
        "Backward palette, grad-of-grad, combined ties.",
        (
            _panel(
                "a",
                "Grad",
                "draw_backward()",
                call="draw_backward",
                capture=_GRAD,
                prep="log_backward",
            ),
            _panel(
                "b",
                "Grad",
                "draw_combined()",
                call="draw_combined",
                capture=_GRAD,
                prep="log_backward",
            ),
            _panel(
                "c",
                "HigherOrder",
                "draw_backward() of a double backward",
                call="draw_backward",
                capture=_GRAD,
                prep="higher_order",
            ),
        ),
    ),
    Sentinel(
        "SN10",
        "measurements",
        ("Sizes", "NaNMidway", "Nonfinite"),
        "color_by (rank), size_by dims, the nan overlay, debug lens nonfinite motifs.",
        (
            _panel(
                "a",
                "Sizes",
                "color_by bytes (rank)",
                kwargs={"color_by": "@bytes_rank", "show_legend": True},
            ),
            _panel("b", "Sizes", 'size_by="dims"', kwargs={"size_by": "dims", "show_legend": True}),
            _panel("c", "NaNMidway", 'node_overlay="nan"', kwargs={"node_overlay": "nan"}),
            _panel(
                "d",
                "Nonfinite",
                'lens "debug"',
                call="lens",
                capture={"module_filter": "@unsaved_abs"},
                kwargs={"lens": "debug"},
            ),
        ),
    ),
    Sentinel(
        "SN11",
        "image-io",
        ("ImageNet",),
        "Input thumbnails and decoded output rows.",
        (
            _panel(
                "a",
                "ImageNet",
                "show_input_transform_summary=True",
                capture={"transform": "@to_tensor", "output_transform": "@decode"},
                kwargs={"show_input_transform_summary": True},
            ),
        ),
    ),
    Sentinel(
        "SN12",
        "degenerate",
        ("SingleOp", "ScalarOut"),
        "The smallest graphs: one op, a scalar output.",
        (
            _panel("a", "SingleOp", "one op"),
            _panel("b", "ScalarOut", "scalar output"),
        ),
    ),
    Sentinel(
        "SN13",
        "first-picture",
        ("BranchConv",),
        "The first picture a new user sees: the one-call facade, default collapse auto.",
        (_panel("a", "BranchConv", "quickstart.render(model, x)", call="quickstart_render"),),
        note="pending name: tl.render (naming sprint, defaults.8); renders "
        "torchlens._quickstart.render today",
    ),
    Sentinel(
        "SN14",
        "side-surfaces",
        ("MiniTransformer",),
        "The non-graph look of TorchLens.",
        (_panel("a", "MiniTransformer", "tl.summary(model, x)", call="summary"),),
        pending_parts={
            "notebook card": "HTML, not DOT; the deck has no HTML or raster comparison yet",
            "Op.show": "a raster figure; the deck has no raster comparison yet",
            "tviz picture": "HTML or raster; the deck has no comparison for it yet",
            "receptive field show": "a raster figure; the deck has no raster comparison yet",
            "attribution overlay": "needs an attribution source the deck does not build yet",
        },
    ),
)


def sentinel_ids() -> tuple[str, ...]:
    """The strip's row ids, in order."""

    return tuple(row.id for row in SENTINELS)


def _rule(row: Sentinel) -> str:
    return f"{row.why} {row.note}".strip()


def sentinel_slides() -> tuple[Slide, ...]:
    """The strip as deck slides: row 1 once per skin, every other active row once."""

    out: list[Slide] = []
    for row in SENTINELS:
        if row.status != "active":
            continue
        title = f"Sentinel {int(row.id[2:])}: {row.slug}"
        if row.id == "SN01":
            for skin in SKINS:
                panels = tuple(
                    Panel(
                        p.name,
                        p.fixture,
                        call=p.call,
                        kwargs={**p.kwargs, "theme": skin},
                        label=f"{p.label}, {skin}",
                        conventions=False,
                    )
                    for p in row.panels
                )
                out.append(
                    Slide(
                        f"sentinel-01-{skin}",
                        f"{title} ({skin})",
                        _rule(row),
                        panels=panels,
                        layout="grid",
                    )
                )
            continue
        out.append(
            Slide(
                f"sentinel-{row.id[2:]}-{row.slug}",
                title,
                _rule(row),
                panels=row.panels,
                layout="grid",
            )
        )
    return tuple(out)
