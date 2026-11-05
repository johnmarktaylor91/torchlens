"""Label-aware node geometry for the ``size_by`` channel (render-IR post-pass).

Graphviz treats an encoded width/height as a MINIMUM (``fixedsize=false``):
a node is never smaller than what its label needs, so a size computed without
the labels lets the label, not the value, set the node's size. This pass makes
the value the node's size: once every node's label is resolved, it estimates
the node size each label needs (the label box, padded the way Graphviz pads
it and widened for the node's shape, e.g. an ellipse needs sqrt(2) times its
label box), then picks ONE base box for every size-encoded node, the smallest
that still holds every label once each node is scaled by its value. Every
node is that base scaled by its normalized value, so the largest-value node
has ``SIZE_BY_MAX_AREA_MULT`` times the area of the smallest. Equal values
give equal node sizes whatever the label lengths, and because the estimate
never falls short of Graphviz's own measure, Graphviz never grows a node
past its encoded size. The cost: one long label on a small-value node
widens every encoded node, since the base must hold it.

Leaf module by design: its only package import is the leaf ``_label_width``.
"""

from __future__ import annotations

import html
import math
import re
from collections.abc import Iterable
from dataclasses import replace
from typing import Any

from ._label_width import text_width_em

# ---------------------------------------------------------------------------
# SIZE channel constants (the default area-only mapping).
# ---------------------------------------------------------------------------

#: Closed ``scale=`` vocabulary. Log is REJECTED by design: it
#: flattens 512-vs-4096 and defeats the size motif.
SIZE_SCALE_VOCABULARY = ("sqrt", "linear")

#: Max clamp in value space: encoded node area never exceeds this multiple of
#: the default node area. PROVISIONAL value per the design memo (4.0), to be
#: tuned once at a rendered-candidates review; the min clamp is free
#: (fixedsize=false -- the box only ever GROWS from the label's natural size).
SIZE_BY_MAX_AREA_MULT = 4.0

#: Graphviz's own default node geometry in inches (dot default width=0.75,
#: height=0.5). Emitted sizes are MINIMUMS scaled from this baseline with the
#: default aspect ratio preserved (default mapping: area-only,
#: conservative -- no per-rank axis rules that could mislead about which axis
#: is which).
DEFAULT_NODE_WIDTH_IN = 0.75
DEFAULT_NODE_HEIGHT_IN = 0.5


def non_batch_numel(shape: tuple[int, ...]) -> float:
    """Default size mapping: numel of the non-batch shape (batch = dim 0).

    Rank <= 1 resolves to 1 (default box) -- conservative: a rank-1 dim may
    or may not be a batch axis, and the mapping never misleads about which
    axis is which.
    """

    if len(shape) <= 1:
        return 1.0
    numel = 1.0
    for dim in shape[1:]:
        numel *= max(1, dim)
    return numel


#: Graphviz's default node font size in points (no TorchLens theme sets one).
DEFAULT_FONT_PT = 14.0
#: Row height in ems plus the label table's 2 pt cell padding on each side.
ROW_HEIGHT_EM = 1.2
CELL_PADDING_PT = 2.0
#: Graphviz default node margins (inches, each side) plus a box3d allowance.
MARGIN_X_IN = 0.11 + 0.06
MARGIN_Y_IN = 0.055 + 0.06
#: Graphviz draws each extra periphery 4 pt outside the last, on every side.
PERIPHERY_GAP_IN = 4.0 / 72.0

#: Shapes whose node box is the padded label box itself.
_BOX_SHAPES = frozenset(
    {
        "box",
        "rect",
        "rectangle",
        "square",
        "box3d",
        "plaintext",
        "plain",
        "none",
        "note",
        "tab",
        "folder",
        "component",
        "underline",
        "Msquare",
    }
)
#: Ellipse-family shapes: Graphviz fits the label box inside, sqrt(2) wider and taller.
_ELLIPSE_SHAPES = frozenset({"ellipse", "oval", "circle", "egg", "doublecircle", "Mcircle"})
#: Other polygons: the ellipse fit, widened by 1 / cos(pi / sides) as Graphviz does.
_POLYGON_SIDES = {
    "triangle": 3,
    "invtriangle": 3,
    "diamond": 4,
    "Mdiamond": 4,
    "trapezium": 4,
    "invtrapezium": 4,
    "parallelogram": 4,
    "pentagon": 5,
    "house": 5,
    "invhouse": 5,
    "hexagon": 6,
    "septagon": 7,
    "octagon": 8,
    "doubleoctagon": 8,
    "tripleoctagon": 8,
}
_EXTRA_PERIPHERIES = {"doublecircle": 1, "doubleoctagon": 1, "tripleoctagon": 2}
#: Fit factor for a shape outside the tables: a diamond's, the widest four-sided fit.
UNKNOWN_SHAPE_FACTOR = 2.0


_MARKUP = re.compile(r"<[^>]*>")


def _visible_text(line: Any) -> str:
    """Return the text Graphviz draws for one label row (tags and entities removed)."""

    return html.unescape(_MARKUP.sub("", str(line)))


def estimate_label_box(lines: Iterable[Any], font_pt: float) -> tuple[float, float]:
    """Return a conservative padded (width, height) in inches for a node label.

    The first row draws bold (the node title); widths come from
    :func:`._label_width.text_width_em`, which never falls short of Graphviz's
    measure for the node font.
    """

    rows = [_visible_text(line) for line in lines] or [""]
    longest = max(text_width_em(row, bold=index == 0) for index, row in enumerate(rows))
    width_pt = longest * font_pt + 2 * CELL_PADDING_PT
    height_pt = len(rows) * (ROW_HEIGHT_EM * font_pt + 2 * CELL_PADDING_PT)
    return (width_pt / 72.0 + 2 * MARGIN_X_IN, height_pt / 72.0 + 2 * MARGIN_Y_IN)


def shape_fit_factor(shape: str | None) -> float:
    """Return how much wider and taller than its padded label box a node of ``shape`` is.

    ``None`` is Graphviz's default node shape, the ellipse.
    """

    if shape is None:
        return math.sqrt(2.0)
    if shape in _BOX_SHAPES:
        return 1.0
    if shape in _ELLIPSE_SHAPES:
        return math.sqrt(2.0)
    sides = _POLYGON_SIDES.get(shape)
    if sides is None:
        return UNKNOWN_SHAPE_FACTOR
    return math.sqrt(2.0) / math.cos(math.pi / sides)


def node_size_needed(box: tuple[float, float], shape: str | None) -> tuple[float, float]:
    """Return the smallest (width, height) a node of ``shape`` needs around ``box``."""

    factor = shape_fit_factor(shape)
    extra = 2 * PERIPHERY_GAP_IN * _EXTRA_PERIPHERIES.get(shape or "", 0)
    return (box[0] * factor + extra, box[1] * factor + extra)


def growth_factor(fraction: float, max_area_mult: float) -> float:
    """Return the linear scale that grows a box's area by ``1 + fraction * (mult - 1)``."""

    return math.sqrt(1.0 + fraction * (max_area_mult - 1.0))


def shared_base(
    needs: dict[str, tuple[float, float]], fractions: dict[str, float], max_area_mult: float
) -> tuple[float, float]:
    """Return the smallest base box that holds every label once scaled by its value."""

    growths = {key: growth_factor(fractions[key], max_area_mult) for key in needs}
    return (
        max(need[0] / growths[key] for key, need in needs.items()),
        max(need[1] / growths[key] for key, need in needs.items()),
    )


def scaled_size(
    reference: tuple[float, float], fraction: float, max_area_mult: float
) -> tuple[float, float]:
    """Scale the reference box so its area grows linearly with ``fraction``."""

    factor = growth_factor(fraction, max_area_mult)
    return (round(reference[0] * factor, 3), round(reference[1] * factor, 3))


def _rewrite_attrs(attrs: dict[str, Any], provisional: tuple[float, float], final: Any) -> bool:
    """Rewrite channel-written width/height in place; keep user overrides."""

    if attrs.get("width") != str(provisional[0]) or attrs.get("height") != str(provisional[1]):
        return False
    attrs.update({"width": str(final[0]), "height": str(final[1])})
    return True


def _rewrite_node(node: Any, provisional: tuple[float, float], final: Any) -> tuple[Any, bool]:
    """Return ``node`` with its channel size attrs rewritten, and whether any were."""

    rewritten = False
    calls = []
    for call in node.node_calls:
        attrs = dict(call.attrs)
        if call.kind == "node" and _rewrite_attrs(attrs, provisional, final):
            call = replace(call, attrs=tuple(attrs.items()))
            rewritten = True
        calls.append(call)
    owned = []
    for owner, args in node.owned_node_args:
        args = dict(args)
        rewritten = _rewrite_attrs(args, provisional, final) or rewritten
        owned.append((owner, args))
    if not rewritten:
        return node, False
    return replace(node, node_calls=tuple(calls), owned_node_args=tuple(owned)), True


def _node_shape(node: Any) -> str | None:
    """Return the Graphviz shape a render-IR node is drawn with (``None``: Graphviz default)."""

    spec = node.node_spec
    if spec is not None:
        return str(spec.extra_attrs.get("shape", spec.shape))
    for call in node.node_calls:
        shape = dict(call.attrs).get("shape")
        if shape is not None:
            return str(shape)
    for _owner, args in node.owned_node_args:
        if "shape" in args:
            return str(args["shape"])
    return None


def layout_encoded_sizes(nodes: tuple[Any, ...], state: Any, context: Any) -> tuple[Any, ...]:
    """Return render-IR nodes with label-aware ``size_by`` geometry.

    Parameters
    ----------
    nodes:
        Resolved render-IR nodes (their specs carry the final label rows).
    state:
        The draw's encoding state, or ``None`` (nodes pass through). Segment
        capsules in ``state.aggregates`` share the base and receive their
        final size through ``state.final_sizes``.
    context:
        Render request; ``font_size`` (when set) is the node font size.
    """

    if state is None or not state.size_fractions:
        return nodes

    font_pt = float(getattr(context, "font_size", None) or DEFAULT_FONT_PT)
    labels: dict[str, tuple[Any, ...]] = {}
    shapes: dict[str, str | None] = {}
    for node in nodes:
        key = state.unit_keys.get(node.name)
        if key in state.size_fractions and key in state.sizes:
            spec = node.node_spec
            labels[key] = tuple(spec.lines) if spec is not None else tuple(node.label_spans)
            shapes[key] = _node_shape(node)
    for target in getattr(state.aggregates, "segments", ()) or ():
        if target.key in state.size_fractions:
            # Segment capsules are emitted as boxes (``_render_nodes``).
            labels[target.key] = target.label_lines
            shapes[target.key] = "box"
    if not labels:
        return nodes
    needs = {
        key: node_size_needed(estimate_label_box(lines, font_pt), shapes[key])
        for key, lines in labels.items()
    }
    base = shared_base(needs, state.size_fractions, SIZE_BY_MAX_AREA_MULT)
    for key in needs:
        state.final_sizes[key] = scaled_size(base, state.size_fractions[key], SIZE_BY_MAX_AREA_MULT)
    out = []
    for node in nodes:
        key = state.unit_keys.get(node.name)
        if key in labels and key in state.sizes:
            node, _ = _rewrite_node(node, state.sizes[key], state.final_sizes[key])
        out.append(node)
    return tuple(out)
