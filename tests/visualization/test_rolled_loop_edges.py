"""A rolled loop body of more than one op draws its return as an arc, and the branch entry edge says IF.

The README toy negates its input when the sum is negative, then applies ``relu(fc(x))``
three times.  Rolled, the body is two nodes; the edge from relu back to fc must leave
relu's output side, return to fc's input side and not constrain the ranking, so dot
draws one tall arc around the body instead of a second line beside the forward edge.
"""

from __future__ import annotations

import os
import tempfile

import torch
from torch import nn

import torchlens as tl
from torchlens._visualization._loop_edges import loop_edge_attrs
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


def _dot(rolled: bool) -> str:
    torch.manual_seed(0)
    trace = tl.trace(Loop(), torch.randn(1, 4), capture=CaptureOptions(save_source_context=True))
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            return trace.draw(
                rolled=rolled, path=os.path.join(tmpdir, "loop"), file_format="dot"
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
        "tailport": "ne",
        "headport": "se",
        "constraint": "false",
    }
    assert loop_edge_attrs(False, True, "TB") == {
        "tailport": "se",
        "headport": "ne",
        "constraint": "false",
    }
    assert loop_edge_attrs(False, False, "BT") == {}


def test_rolled_two_op_loop_body_returns_on_an_unconstrained_arc() -> None:
    dot_source = _dot(rolled=True)
    back = _edge_lines(dot_source, "relu_", "linear_")
    forward = _edge_lines(dot_source, "linear_", "relu_")
    assert len(back) == 1 and len(forward) == 1, dot_source
    assert 'tailport="ne"' in back[0] or "tailport=ne" in back[0], back[0]
    assert 'headport="se"' in back[0] or "headport=se" in back[0], back[0]
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
