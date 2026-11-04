"""Rolled-view recurrence edges: node spacing, self-loop ports and pass labels.

A rolled draw shows a repeated op once, so the edges that carry a value from
one pass to the next become loops. This module holds the geometry that makes
such a loop read as one: the wider node separation (dot draws a self-loop half
as wide as the node gap), the ports that take a self-loop from the node's
output side back to its input side, and the label table that sits a pass
label against its edge. It imports nothing from the edge renderer, so the
graph shell can share the spacing constant without an import cycle.
"""

from __future__ import annotations

from ._render_utils import html_escape
from ._typography import edge_annotation_pt

#: Node separation of a rolled draw, in inches (dot default 0.25): a self-loop
#: is half of it wide, so a recurrence draws as an obvious loop.
ROLLED_NODESEP = "0.5"
_PASS_LABEL_EDGE_GAP = 2  # points between a pass label and the edge it sits beside
_SELF_LOOP_LABEL_HGAP = 8  # points of blank spacer on the far side of a pass label

#: Self-loop ports by rank direction: the loop leaves the node's downstream
#: side and comes back on its upstream side, so it draws as a tall D-shaped
#: arrow beside the node instead of Graphviz's default small curl.
_SELF_LOOP_PORTS: dict[str, tuple[str, str]] = {
    "BT": ("ne", "se"),
    "TB": ("se", "ne"),
    "LR": ("ne", "nw"),
    "RL": ("nw", "ne"),
}


def _self_loop_port_attrs(rankdir: str) -> dict[str, str]:
    """Return the tail and head ports that make a recurrence self-loop read as a loop.

    A self-loop with no ports is a small curl on one side of the node that is
    easy to miss next to a stack of straight arrows; anchoring it from the
    output side back to the input side spans the node's height, so the loop
    is unmistakable (JMT, 3 Nov 2026).
    """

    tail_port, head_port = _SELF_LOOP_PORTS.get(rankdir, _SELF_LOOP_PORTS["BT"])
    return {"tailport": tail_port, "headport": head_port}


def _html_combined_recurrence_label(top: str, bottom: str | None = None) -> str:
    """Return an HTML combined recurrence label that sits against its edge.

    Used for a node's self-loop arc, a merged recurrence back-edge, and merged
    buffer-edge annotations. Graphviz routes the edge along a midpoint label's
    left side, so the narrow left spacer keeps the text on its edge and the
    wide right one keeps it off the next node; ``bottom=None`` gives one line.

    ``top``/``bottom`` are escaped individually (not after joining) so the
    literal ``<BR/>`` line-break tag inserted between them is preserved.
    """

    top = html_escape(top)
    text = top if bottom is None else f"{top}<BR/>{html_escape(bottom)}"
    return (
        '<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="0">'
        f'<TR><TD WIDTH="{_PASS_LABEL_EDGE_GAP}"></TD>'
        f'<TD><FONT POINT-SIZE="{edge_annotation_pt()}">{text}</FONT></TD>'
        f'<TD WIDTH="{_SELF_LOOP_LABEL_HGAP}"></TD></TR></TABLE>>'
    )
