"""Rolled ``In``/``Out`` pass labels sit at the node they describe, clear of everything.

Restored behaviour (JMT, 3 Nov 2026): ``Out N`` is the tail label of an edge (the passes
it leaves its source on) and ``In N`` its head label (the passes it enters its
destination on), on straight edges and on a rolled loop's return arc alike.  A merged
midpoint label had floated those ranges between two nodes, belonging to neither.
Graphviz reserves no room for head/tail labels and moves both ends of an edge with one
``labeldistance``/``labelangle``, so the renderer chooses each edge's pair after layout
(``torchlens/_visualization/_pass_label_placement.py``), and a labelled straight edge
keeps the two ranks of length its midpoint label used to reserve.  These tests pin that
per direction on the README loop toy with the label-geometry audit the rolled gate uses.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest
import torch
from support.label_geometry import audit_gv_source
from torch import nn

import torchlens as tl
from torchlens._visualization._pass_label_placement import (
    choose_pass_label_placements,
    has_pass_labels,
)

#: how far (pt) a pass label may sit from its own node and still read as that node's
_NEAR_NODE_PT = 14.0
_DIRECTIONS = ("bottomup", "topdown", "leftright", "rightleft")


class _Loop(nn.Module):
    """The README loop: a sign-flip branch, then two passes of fc2(relu(fc1(x)))."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.sum() < 0:
            x = -x
        for _ in range(2):
            x = self.fc2(torch.relu(self.fc1(x)))
        return x


def _rolled_dot(direction: str, tmp_path: Path) -> str:
    torch.manual_seed(2)
    model = _Loop()
    x = torch.randn(1, 4)
    trace = tl.trace(model, x)
    try:
        return trace.draw(
            rolled=True, direction=direction, file_format="dot", path=str(tmp_path / direction)
        ).dot
    finally:
        trace.cleanup()


def _edge(dot: str, tail: str, head: str) -> str:
    (line,) = [
        line
        for line in dot.splitlines()
        if re.match(rf"\s*{re.escape(tail)} -> {re.escape(head)} \[", line)
    ]
    return line


def _label(line: str, kind: str) -> str | None:
    match = re.search(rf"(?<![a-z]){kind}=<<TABLE.*?>([^<>]+)</FONT>", line)
    return None if match is None else match.group(1)


@pytest.mark.parametrize("direction", _DIRECTIONS)
def test_pass_labels_sit_at_their_nodes_in_every_direction(direction: str, tmp_path: Path) -> None:
    dot = _rolled_dot(direction, tmp_path)
    arc = _edge(dot, "linear_2_6", "linear_1_4")
    entry = _edge(dot, "neg_1_3", "linear_1_4")
    exit_ = _edge(dot, "linear_2_6", "output_1")

    assert (_label(arc, "taillabel"), _label(arc, "headlabel")) == ("Out 1", "In 2"), arc
    assert _label(arc, "label") is None and _label(arc, "xlabel") is None, arc
    assert (_label(entry, "headlabel"), _label(entry, "taillabel")) == ("In 1", None), entry
    assert (_label(exit_, "taillabel"), _label(exit_, "headlabel")) == ("Out 2", None), exit_
    for line in (entry, exit_):
        assert "minlen=2" in line, line
    assert "minlen" not in arc, arc

    result = audit_gv_source(dot)
    assert result.hard_violation_count == 0, result.describe_violations(direction)
    pass_labels = [rec for rec in result.labels if rec["kind"] in ("head", "tail")]
    assert len(pass_labels) == 4, pass_labels
    for rec in pass_labels:
        gap = rec["metrics"]["gap_to_own_endpoint_node"]
        assert gap is not None and gap <= _NEAR_NODE_PT, (direction, rec["text"], gap)


def test_chooser_moves_a_colliding_default_pair_clear(tmp_path: Path) -> None:
    """Sideways, Graphviz's default spot for the arc's ``Out 1`` crosses the arc."""

    dot = _rolled_dot("leftright", tmp_path)
    plain = re.sub(r" labelangle=\S+ labeldistance=[^\s\]]+", "", dot)
    assert audit_gv_source(plain).hard_violation_count >= 1
    body = plain.splitlines(keepends=True)
    layout = subprocess.run(
        ["dot", "-Tjson"], input=plain, capture_output=True, text=True, check=True
    ).stdout
    chosen = "".join(choose_pass_label_placements(body, layout))
    assert "labelangle=" in _edge(chosen, "linear_2_6", "linear_1_4")
    assert audit_gv_source(chosen).hard_violation_count == 0


def test_chooser_leaves_graphs_without_pass_labels_alone() -> None:
    body = [
        "digraph g {\n",
        "\ta -> b [headlabel=<<TABLE><TR><TD>arg 1</TD></TR></TABLE>>]\n",
        "}\n",
    ]
    assert not has_pass_labels(body)
    assert choose_pass_label_placements(body, "{}") == body
    labelled = ["\ta -> b [headlabel=<<TABLE><TR><TD><FONT>In 2</FONT></TD></TR></TABLE>>]\n"]
    assert has_pass_labels(labelled)
    assert choose_pass_label_placements(labelled, "not json") == labelled
