"""Label-aware node geometry for the ``size_by`` channel (render-IR post-pass).

Graphviz treats an encoded width/height as a MINIMUM (``fixedsize=false``):
a node is never smaller than its label. Sized from Graphviz's bare default
box (0.75 x 0.5 in), every multi-row TorchLens label outgrows the minimum and
the label, not the value, sets the node's size. This pass makes the value
set the size: once every node's label is resolved, it estimates the largest
label box among the size-encoded nodes and scales that reference by the
node's normalized value, so the smallest-value node is exactly large enough
for the largest label and the largest-value node has
``SIZE_BY_MAX_AREA_MULT`` times its area. Every encoded node therefore
outgrows its own label, area stays monotone in the value, and no label is
clipped (the attributes remain minimums).

Leaf module by design: no runtime imports from the visualization package.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import replace
from typing import Any

# ---------------------------------------------------------------------------
# SIZE channel constants (wave 1; D4 DEFAULT-APPLIED -- memo 3.1/3.2 C2).
# ---------------------------------------------------------------------------

#: Closed ``scale=`` vocabulary (slate 3.2). Log is REJECTED by design: it
#: flattens 512-vs-4096 and defeats the size motif.
SIZE_SCALE_VOCABULARY = ("sqrt", "linear")

#: Max clamp in value space: encoded node area never exceeds this multiple of
#: the default node area. PROVISIONAL value per the design memo (4.0), to be
#: tuned once at the D4 rendered-candidates session; the min clamp is free
#: (fixedsize=false -- the box only ever GROWS from the label's natural size).
SIZE_BY_MAX_AREA_MULT = 4.0

#: Graphviz's own default node geometry in inches (dot default width=0.75,
#: height=0.5). Emitted sizes are MINIMUMS scaled from this baseline with the
#: default aspect ratio preserved (D4 default mapping C2: area-only,
#: conservative -- no per-rank axis rules that could mislead about which axis
#: is which).
DEFAULT_NODE_WIDTH_IN = 0.75
DEFAULT_NODE_HEIGHT_IN = 0.5


def non_batch_numel(shape: tuple[int, ...]) -> float:
    """D4 default mapping C2: numel of the non-batch shape (batch = dim 0).

    Rank <= 1 resolves to 1 (default box) -- conservative: a rank-1 dim may
    or may not be a batch axis, and C2 never misleads about which axis is
    which.
    """

    if len(shape) <= 1:
        return 1.0
    numel = 1.0
    for dim in shape[1:]:
        numel *= max(1, dim)
    return numel


#: Graphviz's default node font size in points (no TorchLens theme sets one).
DEFAULT_FONT_PT = 14.0
#: Conservative average glyph advance for the node font, in ems. Erring wide
#: only makes encoded nodes a little larger; it can never clip a label.
CHAR_WIDTH_EM = 0.6
#: Row height in ems plus the label table's 2 pt cell padding on each side.
ROW_HEIGHT_EM = 1.2
CELL_PADDING_PT = 2.0
#: Graphviz default node margins (inches, each side) plus a box3d allowance.
MARGIN_X_IN = 0.11 + 0.06
MARGIN_Y_IN = 0.055 + 0.06


def estimate_label_box(lines: Iterable[Any], font_pt: float) -> tuple[float, float]:
    """Return a conservative (width, height) in inches for a node label."""

    rows = [str(line) for line in lines] or [""]
    longest = max(len(row) for row in rows)
    width_pt = longest * CHAR_WIDTH_EM * font_pt + 2 * CELL_PADDING_PT
    height_pt = len(rows) * (ROW_HEIGHT_EM * font_pt + 2 * CELL_PADDING_PT)
    return (width_pt / 72.0 + 2 * MARGIN_X_IN, height_pt / 72.0 + 2 * MARGIN_Y_IN)


def scaled_size(
    reference: tuple[float, float], fraction: float, max_area_mult: float
) -> tuple[float, float]:
    """Scale the reference box so its area grows linearly with ``fraction``."""

    factor = math.sqrt(1.0 + fraction * (max_area_mult - 1.0))
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


def layout_encoded_sizes(nodes: tuple[Any, ...], state: Any, context: Any) -> tuple[Any, ...]:
    """Return render-IR nodes with label-aware ``size_by`` geometry.

    Parameters
    ----------
    nodes:
        Resolved render-IR nodes (their specs carry the final label rows).
    state:
        The draw's encoding state, or ``None`` (nodes pass through). Segment
        capsules in ``state.aggregates`` join the reference and receive their
        final size through ``state.final_sizes``.
    context:
        Render request; ``font_size`` (when set) is the node font size.
    """

    if state is None or not state.size_fractions:
        return nodes

    font_pt = float(getattr(context, "font_size", None) or DEFAULT_FONT_PT)
    labels: dict[str, tuple[Any, ...]] = {}
    for node in nodes:
        key = state.unit_keys.get(node.name)
        if key in state.size_fractions and key in state.sizes:
            spec = node.node_spec
            labels[key] = tuple(spec.lines) if spec is not None else tuple(node.label_spans)
    for target in getattr(state.aggregates, "segments", ()) or ():
        if target.key in state.size_fractions:
            labels[target.key] = target.label_lines
    if not labels:
        return nodes
    boxes = [estimate_label_box(lines, font_pt) for lines in labels.values()]
    reference = (max(box[0] for box in boxes), max(box[1] for box in boxes))
    for key in labels:
        state.final_sizes[key] = scaled_size(
            reference, state.size_fractions[key], SIZE_BY_MAX_AREA_MULT
        )
    out = []
    for node in nodes:
        key = state.unit_keys.get(node.name)
        if key in labels and key in state.sizes:
            node, _ = _rewrite_node(node, state.sizes[key], state.final_sizes[key])
        out.append(node)
    return tuple(out)
