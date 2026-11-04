"""Ports that make a rolled recurrence read as a loop.

A rolled drawing folds a repeated op into one node, so the flow from one
pass into the next becomes either a self-loop (a one-op body) or a back
edge from the body's last op to its first (a longer body).  Without ports
a self-loop is a small curl and a back edge is a second line almost
parallel to the forward edge, so neither reads as a cycle.  A self-loop now
leaves the output side and returns to the input side; a back edge leaves
and re-enters on one flank; both read as one arc around the body (JMT, 3
Nov 2026).

Older Graphviz builds cannot always route those ports: 2.43, which CI and current
Ubuntu ship, reports ``libpath/shortest.c:324: triangulation failed`` for the README
loop's left-to-right return arc under the bundled font's metrics, while newer builds
route it.  When the installed build reports such a router failure for the text an image
is rendered from, the image is laid out again without the edges' compass ports
(:func:`without_compass_ports`), and that layout is kept only when it routes cleanly
(:func:`reports_route_failure`); the DOT a drawing returns or saves keeps its ports.
"""

from __future__ import annotations

import re

__all__ = [
    "loop_edge_attrs",
    "pass_label_spacing",
    "reports_route_failure",
    "self_loop_port_attrs",
    "without_compass_ports",
]

# self-loop: tail port (output side), head port (input side) per rank direction
_LOOP_PORTS: dict[str, tuple[str, str]] = {
    "BT": ("ne", "se"),
    "TB": ("se", "ne"),
    "LR": ("ne", "nw"),
    "RL": ("nw", "ne"),
}
#: an edge's ``tailport``/``headport`` compass attribute, as the graphviz package writes it
_PORT_ATTR = r'(?:tail|head)port="?(?:n|ne|e|se|s|sw|w|nw|c|_)"?'
_COMPASS_PORT = re.compile(rf"[ \t]{_PORT_ATTR}(?=[ \t\]])|(?<=\[){_PORT_ATTR}(?:[ \t]+|(?=\]))")
#: Graphviz's spline-router (pathplan) failures: the edge was routed by a fallback
_ROUTE_FAILURES = (
    "triangulation failed",
    "source point not in any triangle",
    "destination point not in any triangle",
    "cannot find triangle path",
    "Pshortestpath failed",
)
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


def pass_label_spacing(is_back_edge: bool) -> dict[str, str]:
    """Return the length a straight edge needs for its head/tail pass labels.

    Graphviz reserves no space for head/tail labels.  A pass-labelled straight edge
    keeps the two ranks its midpoint label used to reserve, so ``Out 2`` fits between
    the nodes sideways and nearby loops keep the room the label geometry gate was
    measured with.  A return arc is ``constraint=false`` and takes no length from ranks.
    """

    return {} if is_back_edge else {"minlen": "2"}


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


def reports_route_failure(stderr: str | bytes | None) -> bool:
    """Return whether Graphviz's ``stderr`` reports that its spline router failed."""

    if not stderr:
        return False
    text = stderr.decode("utf-8", "replace") if isinstance(stderr, bytes) else stderr
    return any(failure in text for failure in _ROUTE_FAILURES)


def without_compass_ports(source: str) -> str:
    """Return DOT ``source`` with the compass ports of its edge statements dropped.

    Only ``tailport``/``headport`` attributes on edge lines go; node statements,
    labels and ``node:port`` endpoints are untouched.
    """

    return "\n".join(_drop_ports(line) if "->" in line else line for line in source.split("\n"))


def _drop_ports(line: str) -> str:
    """Return one edge line without its compass ports (adjacent ports need a second pass)."""

    dropped = 1
    while dropped:
        line, dropped = _COMPASS_PORT.subn("", line)
    return line
