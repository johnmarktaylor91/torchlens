"""Display text and Graphviz HTML for conditional branch labels.

A branch label names one arm of a Python conditional (THEN, ELIF n, ELSE,
FALLTHROUGH) or the kind of its test (IF, ELIF). This module turns a stored
branch kind into that word, orders arm kinds, and wraps the word in the padded
HTML table drawn on an edge. It holds no trace logic, so the edge labelers and
the tests can share it without importing the whole leaf renderer.
"""

from __future__ import annotations

from ._typography import edge_emphasis_pt

__all__ = [
    "BRANCH_EDGE_LABEL_PAD",
    "_branch_kind_sort_key",
    "_format_branch_edge_label_html",
    "_format_branch_kind_text",
]

#: Drawn text and sort rank of the non-elif arm kinds; the fall-through arm never reads as ELSE.
_BRANCH_KIND_TEXT = {"then": "THEN", "else": "ELSE", "fallthrough": "FALLTHROUGH"}
_BRANCH_KIND_RANK = {"then": 0, "else": 10**6, "fallthrough": 10**6 + 1}


def _format_branch_kind_text(branch_kind: str) -> str:
    """Format a branch-kind token as display text.

    Args:
        branch_kind:
            Stored branch kind: ``"then"``, ``"elif_N"``, ``"else"`` or ``"fallthrough"``.

    Returns
    -------
    str
        Display label such as ``"THEN"`` or ``"ELIF 1"``.

    Raises
    ------
    ValueError
        If ``branch_kind`` is not recognized.
    """
    if branch_kind in _BRANCH_KIND_TEXT:
        return _BRANCH_KIND_TEXT[branch_kind]
    if branch_kind.startswith("elif_"):
        return f"ELIF {int(branch_kind.split('_', 1)[1])}"
    raise ValueError(f"Unrecognized branch kind: {branch_kind}")


def _branch_kind_sort_key(branch_kind: str) -> int:
    """Return an ordering key for branch kinds.

    Args:
        branch_kind:
            Stored branch kind: ``"then"``, ``"elif_N"``, ``"else"`` or ``"fallthrough"``.

    Returns
    -------
    int
        Sort rank for the branch kind.
    """
    if branch_kind.startswith("elif_"):
        return int(branch_kind.split("_", 1)[1])
    return _BRANCH_KIND_RANK.get(branch_kind, 10**9)


#: Transparent margin (points) on each side of a branch-edge label (IF, THEN, ELSE, ELIF).
#: dot reserves the label's box beside the edge midpoint but draws the spline right along the
#: box's edge, so an unpadded word gets crossed by its own line; the side cells push the text
#: clear of the line on whichever side dot places the label (any ``rankdir``).
BRANCH_EDGE_LABEL_PAD = 8


def _format_branch_edge_label_html(label_text: str) -> str:
    """Wrap plain branch-label text in the Graphviz HTML used by TorchLens.

    The word sits in the middle cell of a borderless one-row table whose outer cells are
    ``BRANCH_EDGE_LABEL_PAD`` points wide, so the edge line passes beside the word instead
    of through it. ``_pass_marks._BRANCH_LABEL`` recognizes this shape.

    Args:
        label_text:
            Plain text to display on the edge.

    Returns
    -------
    str
        Graphviz HTML edge-label string.
    """
    pad = f'<TD WIDTH="{BRANCH_EDGE_LABEL_PAD}"></TD>'
    word = f'<FONT POINT-SIZE="{edge_emphasis_pt()}"><b><u>{label_text}</u></b></FONT>'
    return (
        '<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="0">'
        f"<TR>{pad}<TD>{word}</TD>{pad}</TR></TABLE>>"
    )
