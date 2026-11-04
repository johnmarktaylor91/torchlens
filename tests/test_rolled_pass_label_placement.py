"""Rolled ``In``/``Out`` pass labels sit at the node they describe, clear of everything.

Restored behaviour (JMT, 3 Nov 2026): ``Out N`` is the tail label of an edge (the passes
it leaves its source on) and ``In N`` its head label (the passes it enters its
destination on), on straight edges and on a rolled loop's return arc alike.  A merged
midpoint label had floated those ranges between two nodes, belonging to neither.
Graphviz reserves no room for head/tail labels and moves both ends of an edge with one
``labeldistance``/``labelangle``, so the renderer chooses each edge's pair after layout
(``torchlens/_visualization/_pass_label_placement.py``), and a labelled straight edge
keeps the two ranks of length its midpoint label used to reserve.  These tests pin that
per direction on the README loop toy with the label-geometry audit the rolled gate uses,
and check that each label touches the end of its own edge and is nearer that edge
than any other edge at its node (JMT, 4 Nov 2026: a label beside the wrong edge
reads as that edge's).
"""

from __future__ import annotations

import json
import math
import re
import subprocess
from pathlib import Path

import pytest
import torch
from support.label_geometry import audit_gv_source
from torch import nn

import torchlens as tl
from torchlens._visualization._fonts import graphviz_font_env
from torchlens._visualization._pass_label_placement import (
    _beziers,
    _label_lines,
    _sample_bezier,
    _shift_label,
    choose_pass_label_placements,
    has_pass_labels,
)

#: how far (pt) a pass label may sit from its own edge's end and still touch it
_NEAR_END_PT = 12.0
#: how much nearer (pt) a pass label must be to its own edge than to any other edge at its node
_ATTACH_MARGIN_PT = 4.0
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
    attachments = _attachments(dot)
    assert sorted(text for text, *_ in attachments) == ["In 1", "In 2", "Out 1", "Out 2"]
    for text, end_gap, own_gap, rival_gap in attachments:
        assert end_gap <= _NEAR_END_PT, (direction, text, end_gap)
        assert rival_gap >= own_gap + _ATTACH_MARGIN_PT, (direction, text, own_gap, rival_gap)


def _attachments(dot: str) -> list[tuple[str, float, float, float]]:
    """Per pass label: its gap to its own edge's end, to that edge's last 30 pt, and
    to the nearest other edge at the same node (within 60 pt of the end)."""

    layout = json.loads(
        subprocess.run(
            ["dot", "-Tjson"],
            input=dot,
            capture_output=True,
            text=True,
            check=True,
            env=graphviz_font_env(),
        ).stdout
    )
    names = {obj["_gvid"]: obj["name"] for obj in layout["objects"]}
    edges = [
        (
            names[edge["tail"]],
            names[edge["head"]],
            edge,
            [p for ctrl in _beziers(edge.get("_draw_")) for p in _sample_bezier(ctrl)],
        )
        for edge in layout["edges"]
    ]
    found = []
    for tail, head, edge, samples in edges:
        for kind, node in (("head", head), ("tail", tail)):
            texts = [op["text"] for op in edge.get(f"_{kind[0]}ldraw_", []) if op["op"] == "T"]
            if not texts or not texts[0].startswith(("In ", "Out ")):
                continue
            boxes = _label_lines(edge[f"_{kind[0]}ldraw_"])
            end = _end_point(edge["pos"], head=kind == "head")
            own = [p for p in samples if math.dist(p, end) <= 30.0]
            rivals = [
                p
                for other_tail, other_head, other, other_samples in edges
                if other is not edge and node in (other_tail, other_head)
                for p in other_samples
                if math.dist(p, end) <= 60.0
            ]
            found.append(
                (
                    texts[0],
                    min(_box_point_gap(b, end) for b in boxes),
                    min(_box_point_gap(b, p) for b in boxes for p in own),
                    min((_box_point_gap(b, p) for b in boxes for p in rivals), default=math.inf),
                )
            )
    return found


def _end_point(pos: str, *, head: bool) -> tuple[float, float]:
    pieces = pos.split(";")
    tokens = (pieces[-1] if head else pieces[0]).split()
    flag = "e," if head else "s,"
    marked = [t[2:] for t in tokens if t.startswith(flag)]
    plain = [t for t in tokens if not t.startswith(("e,", "s,"))]
    x, y = (float(v) for v in (marked[0] if marked else plain[-1 if head else 0]).split(","))
    return x, y


def _box_point_gap(box: tuple[float, float, float, float], point: tuple[float, float]) -> float:
    dx = max(box[0] - point[0], 0.0, point[0] - box[2])
    dy = max(box[1] - point[1], 0.0, point[1] - box[3])
    return math.hypot(dx, dy)


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


def _head_text_centre(source: str) -> tuple[float, float]:
    """The head label's text centre relative to its node (a wider label can grow the
    drawing's bounding box, which moves the whole drawing)."""

    layout = json.loads(
        subprocess.run(
            ["dot", "-Tjson"], input=source, capture_output=True, text=True, check=True
        ).stdout
    )
    (edge,) = layout["edges"]
    ((x0, y0, x1, y1),) = _label_lines(edge["_hldraw_"])
    (node_x, node_y) = next(
        (float(v) for v in obj["pos"].split(",")) for obj in layout["objects"] if obj["name"] == "b"
    )
    return (x0 + x1) / 2 - node_x, (y0 + y1) / 2 - node_y


@pytest.mark.parametrize("shift", [(3.5, -2.0), (-6.0, 4.5), (0.0, -7.0)])
def test_padding_moves_one_label_by_exactly_its_shift(shift: tuple[float, float]) -> None:
    """A label the shared pair leaves astray moves alone, by blank table padding."""

    line = (
        '\ta -> b [headlabel=<<TABLE BORDER="0" CELLPADDING="2"><TR><TD>'
        '<FONT POINT-SIZE="11">In 2</FONT></TD></TR></TABLE>> labelangle=60 labeldistance=2]\n'
    )
    graph = "digraph g {\n\trankdir=BT\n\ta [shape=box]\n\tb [shape=box]\n%s}\n"
    x0, y0 = _head_text_centre(graph % line)
    x1, y1 = _head_text_centre(graph % _shift_label(line, "headlabel", shift))
    assert abs(x1 - x0 - shift[0]) <= 0.51 and abs(y1 - y0 - shift[1]) <= 0.51, (x1 - x0, y1 - y0)


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


@pytest.mark.parametrize("failure", ["layout error", "no binary"])
def test_a_failed_placement_layout_leaves_the_labels_unplaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """If the extra ``-Tjson`` layout fails, the drawing keeps its labels, unplaced."""

    from torchlens._visualization import _render_utils
    from torchlens.errors import GraphvizUnavailableError

    real = _render_utils.run_bounded_subprocess

    def failing_json(cmd: list[str], **kwargs: object) -> object:
        if "-Tjson" not in cmd:
            return real(cmd, **kwargs)  # type: ignore[arg-type]
        if failure == "no binary":
            raise GraphvizUnavailableError("no dot", executable=cmd[0])
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(_render_utils, "run_bounded_subprocess", failing_json)
    dot = _rolled_dot("bottomup", tmp_path)
    arc = _edge(dot, "linear_2_6", "linear_1_4")
    assert (_label(arc, "taillabel"), _label(arc, "headlabel")) == ("Out 1", "In 2")
    assert "labelangle=" not in arc
