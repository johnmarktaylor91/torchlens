"""Ports that make a rolled recurrence read as a loop.

A rolled drawing folds a repeated op into one node, so the flow from one
pass into the next becomes either a self-loop (a one-op body) or a back
edge from the body's last op to its first (a longer body).  Without ports
a self-loop is a small curl and a back edge is a second line almost
parallel to the forward edge, so neither reads as a cycle.  A self-loop now
leaves the output side and returns to the input side; a back edge leaves
and re-enters on one flank; both read as one arc around the body (JMT, 3
Nov 2026).
"""

from __future__ import annotations

__all__ = ["loop_edge_attrs", "pass_label_placement", "self_loop_port_attrs"]

# self-loop: tail port (output side), head port (input side) per rank direction
_LOOP_PORTS: dict[str, tuple[str, str]] = {
    "BT": ("ne", "se"),
    "TB": ("se", "ne"),
    "LR": ("ne", "nw"),
    "RL": ("nw", "ne"),
}
# back edge between two body nodes: the two inner corners on one flank (the tail's
# corner nearest the head, the head's corner nearest the tail), so dot draws a short,
# shallow arc past the middle node instead of a wide loop around both boxes (compared
# flank centres, inner corners and no ports in bottom-up and left-to-right on the README
# toy, 3 Nov 2026: inner corners halve the arc's depth and keep it clear of the body).
_BACK_EDGE_PORTS: dict[str, tuple[str, str]] = {
    "BT": ("se", "ne"),
    "TB": ("ne", "se"),
    "LR": ("sw", "se"),
    "RL": ("se", "sw"),
}
# return arc's head/tail label placement (labeldistance, labelangle) per rank direction
_BACK_EDGE_LABEL_PLACEMENT: dict[str, tuple[str, str]] = {
    "BT": ("2", "-45"),
    "TB": ("2", "-45"),
    "LR": ("2", "-45"),
    "RL": ("2", "-45"),
}


def pass_label_placement(rankdir: str) -> dict[str, str]:
    """Return the head/tail label placement for a return arc's ``In``/``Out`` labels.

    The arc leaves the tail and enters the head at inner corners (``_BACK_EDGE_PORTS``),
    so each label is placed relative to the arc's end tangent to sit beside its own
    node's corner, clear of the box and the curve.
    """

    distance, angle = _BACK_EDGE_LABEL_PLACEMENT.get(rankdir, _BACK_EDGE_LABEL_PLACEMENT["BT"])
    return {"labeldistance": distance, "labelangle": angle}


def self_loop_port_attrs(rankdir: str) -> dict[str, str]:
    """Return the tail and head ports for a recurrence self-loop."""

    tail_port, head_port = _LOOP_PORTS.get(rankdir, _LOOP_PORTS["BT"])
    return {"tailport": tail_port, "headport": head_port}


def loop_edge_attrs(is_self_loop: bool, is_back_edge: bool, rankdir: str) -> dict[str, str]:
    """Return the DOT attributes that draw a rolled recurrence edge as a loop.

    Parameters
    ----------
    is_self_loop:
        The edge's tail and head are the same rendered node.
    is_back_edge:
        The edge carries a value from a later-placed op back to an earlier
        one of the same loop body (an earlier pass feeding a later pass).
    rankdir:
        Graphviz rank direction of the drawing.

    Returns
    -------
    dict[str, str]
        Empty for an ordinary edge.  A self-loop gets its ports.  A back edge
        leaves and enters on one flank of the body and carries
        ``constraint=false``, so dot ranks the body by its forward edges alone
        and bows the return around that flank instead of drawing a second
        line beside the forward edge.
    """

    if is_self_loop:
        return self_loop_port_attrs(rankdir)
    if is_back_edge:
        tail_port, head_port = _BACK_EDGE_PORTS.get(rankdir, _BACK_EDGE_PORTS["BT"])
        return {"tailport": tail_port, "headport": head_port, "constraint": "false"}
    return {}
