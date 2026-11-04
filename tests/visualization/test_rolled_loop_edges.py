"""A rolled loop body of more than one op draws its return as an arc, and the branch entry edge says IF.

The README toy negates its input when the sum is negative, then applies ``relu(fc(x))``
three times.  Rolled, the body is two nodes; the edge from relu back to fc must leave and re-enter
on the body's right flank and not constrain the ranking, so dot bows one arc around the
body instead of drawing a second line beside the forward edge.
"""

from __future__ import annotations

import os
import tempfile

import torch
from torch import nn

import torchlens as tl
from torchlens._visualization._loop_edges import loop_edge_attrs
from torchlens._visualization._pass_marks import _BRANCH_LABEL
from torchlens._visualization._render_leaf import (
    BRANCH_EDGE_LABEL_PAD,
    _format_branch_edge_label_html,
)
from torchlens.options import CaptureOptions


class Loop(nn.Module):
    """Flip the sign when the sum is negative, then apply fc and relu three times."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.sum() < 0:
            x = -x
        for _ in range(3):
            x = torch.relu(self.fc(x))
        return x


def _dot(rolled: bool, direction: str = "bottomup") -> str:
    torch.manual_seed(0)
    trace = tl.trace(Loop(), torch.randn(1, 4), capture=CaptureOptions(save_source_context=True))
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            return trace.draw(
                rolled=rolled,
                direction=direction,
                path=os.path.join(tmpdir, "loop"),
                file_format="dot",
            ).dot
    finally:
        trace.cleanup()


def _edge_lines(dot_source: str, tail: str, head: str) -> list[str]:
    return [
        line
        for line in dot_source.splitlines()
        if "->" in line and line.split("->")[0].strip().startswith(tail) and head in line
    ]


def test_loop_edge_attrs_distinguish_self_loop_back_edge_and_plain_edge() -> None:
    assert loop_edge_attrs(True, False, "BT") == {"tailport": "ne", "headport": "se"}
    assert loop_edge_attrs(False, True, "BT") == {
        "tailport": "se",
        "headport": "ne",
        "constraint": "false",
    }
    assert loop_edge_attrs(False, True, "TB") == {
        "tailport": "ne",
        "headport": "se",
        "constraint": "false",
    }
    assert loop_edge_attrs(False, True, "LR") == {
        "tailport": "sw",
        "headport": "se",
        "constraint": "false",
    }
    assert loop_edge_attrs(False, True, "RL") == {
        "tailport": "se",
        "headport": "sw",
        "constraint": "false",
    }
    assert loop_edge_attrs(False, False, "BT") == {}


def test_rolled_two_op_loop_body_returns_on_an_unconstrained_arc() -> None:
    dot_source = _dot(rolled=True)
    back = _edge_lines(dot_source, "relu_", "linear_")
    forward = _edge_lines(dot_source, "linear_", "relu_")
    assert len(back) == 1 and len(forward) == 1, dot_source
    assert "tailport=se" in back[0], back[0]
    assert "headport=ne" in back[0], back[0]
    assert "xlabel=" in back[0] and " label=" not in back[0], back[0]
    assert "constraint=false" in back[0] or 'constraint="false"' in back[0], back[0]
    assert "In 2-3" in back[0] and "Out 1-2" in back[0], back[0]
    assert "tailport" not in forward[0] and "constraint" not in forward[0], forward[0]


def test_unrolled_loop_body_edges_carry_no_loop_ports() -> None:
    dot_source = _dot(rolled=False)
    assert "tailport" not in dot_source and "constraint=false" not in dot_source


def test_branch_entry_edge_is_labelled_if_in_both_modes() -> None:
    for rolled in (False, True):
        dot_source = _dot(rolled=rolled)
        entry = _edge_lines(dot_source, "input_1", "sum_")
        assert len(entry) == 1, dot_source
        assert "<u>IF</u>" in entry[0], entry[0]
        then = _edge_lines(dot_source, "input_1", "neg_")
        assert len(then) == 1 and "THEN" in then[0], then


def test_branch_edge_label_is_padded_clear_of_its_edge() -> None:
    """IF/THEN words sit in a one-row table with a transparent cell on each side."""
    label = _format_branch_edge_label_html("IF")
    pad = f'<TD WIDTH="{BRANCH_EDGE_LABEL_PAD}"></TD>'
    assert BRANCH_EDGE_LABEL_PAD >= 6
    assert label.startswith(
        '<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="0"><TR>'
    )
    assert label.count(pad) == 2 and label.index(pad) < label.index("<u>IF</u>") < label.rindex(pad)
    assert _BRANCH_LABEL.fullmatch(label), label
    for rolled in (True, False):
        for direction in ("bottomup", "topdown"):
            (if_edge,) = _edge_lines(_dot(rolled, direction), "input_1", "sum_")
            assert (
                f'{pad}<TD><FONT POINT-SIZE="' in if_edge
                and "<u>IF</u></b></FONT></TD>" + pad in if_edge
            ), if_edge
            (then_edge,) = _edge_lines(_dot(rolled, direction), "input_1", "neg_")
            assert "<u>THEN</u></b></FONT></TD>" + pad in then_edge, then_edge


def test_rolled_back_edge_sideways_leaves_and_enters_from_below() -> None:
    """Left-to-right, the return arc hangs under the body between the inner bottom corners;
    top ports made dot route the curve under relu anyway while hooking both ends upward."""
    (back,) = _edge_lines(_dot(rolled=True, direction="leftright"), "relu_", "linear_")
    assert "tailport=sw" in back and "headport=se" in back, back
    assert "xlabel=" in back and " label=" not in back, back
    assert "constraint=false" in back or 'constraint="false"' in back, back


def test_rolled_loop_head_hides_the_redundant_in_features_like_its_body() -> None:
    """fc's rolled node has two parents (entry and return edge) with one shape, so the
    ``in_features`` row is as redundant there as on every other Linear node."""
    for direction in ("bottomup", "leftright"):
        dot_source = _dot(rolled=True, direction=direction)
        assert "in_features" not in dot_source, direction
