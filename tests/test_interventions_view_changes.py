"""The interventions view shows what an edit changed, in plain words.

``trace.draw(view="interventions")`` outlines the edited op in purple and every
other op whose saved value differs from the clean run (the unedited trace the
fork came from) in light purple, with the largest absolute change. Upstream and
unchanged ops stay plain. Without clean values the caption says so in one line.
The header's capture-status lines about the edit move to the census sidecar;
a plain draw of the same fork keeps them.
"""

from __future__ import annotations

import gc
import math
import re
from pathlib import Path

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens._visualization._edit_change import CHANGED_INK, _value_change

_CAPTURE = tl.options.CaptureOptions(replayable=True, save_args=True)


class _Mlp(nn.Module):
    """fc1 -> relu -> fc2 -> relu -> out."""

    def __init__(self) -> None:
        """Build the layers."""

        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(16, 16)
        self.act2 = nn.ReLU()
        self.out = nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the MLP."""

        return self.out(self.act2(self.fc2(self.act(self.fc1(x)))))


def _node(source: str, name: str) -> str:
    """Return one node's DOT statement."""

    match = re.search(rf'^\t"?{re.escape(name)}(?:pass1)?"? \[.*?\]\n', source, re.M | re.S)
    names = re.findall(r'^\t"?(\w+)"? \[', source, re.M)
    assert match is not None, f"no node {name} in the DOT; nodes: {names}"
    return match.group(0)


def _edited_pair() -> tuple[tl.Trace, tl.Trace]:
    """Return ``(clean, fork)`` with the first relu zeroed in the fork."""

    torch.manual_seed(0)
    clean = tl.trace(_Mlp().eval(), torch.randn(2, 8), capture=_CAPTURE)
    fork = clean.fork()
    fork.do(fork["relu_1_2:1"], tl.zero_ablate())
    return clean, fork


def test_downstream_changes_are_outlined_with_their_largest_change(tmp_path: Path) -> None:
    """Changed downstream ops get the light purple outline; upstream ops stay plain."""

    clean, fork = _edited_pair()
    graph = fork.draw(view="interventions", path=str(tmp_path / "view"), file_format="svg")
    source = graph.dot
    for name in ("linear_2_3", "linear_3_5"):
        node = _node(source, name)
        assert CHANGED_INK in node, node
        assert "changed by up to" in node, node
    for name in ("linear_1_1", "input_1"):
        node = _node(source, name)
        assert CHANGED_INK not in node and "changed by up to" not in node, node
    edited = _node(source, "relu_1_2")
    assert "changed by up to" not in edited
    assert "penwidth=3" in edited
    expected = float((fork["linear_2_3:1"].out - clean["linear_2_3:1"].out).abs().max())
    assert f"changed by up to {expected:.3g}" in _node(source, "linear_2_3")
    assert "light purple: changed by the edit" in source


def test_header_drops_status_jargon_and_census_keeps_it(tmp_path: Path) -> None:
    """The view's header is plain; the sidecar keeps the full record."""

    _clean, fork = _edited_pair()
    source = fork.draw(view="interventions", path=str(tmp_path / "view"), file_format="svg").dot
    for jargon in ("capture outcome", "INTERVENED capture", "COUNTERFACTUAL", "solid border"):
        assert jargon not in source, jargon
    assert "exit values were substituted" not in source
    assert "1 edit:" in source
    census = (tmp_path / "view.census.txt").read_text()
    assert "INTERVENED capture" in census
    assert "ops whose saved value changed: 4" in census
    assert "exit values were substituted" in census


def test_plain_draw_of_the_fork_keeps_its_header_lines(tmp_path: Path) -> None:
    """Only the view moves the status lines; a plain draw still shows them."""

    _clean, fork = _edited_pair()
    source = fork.draw(path=str(tmp_path / "plain"), file_format="svg").dot
    assert "INTERVENED capture" in source


def test_missing_clean_values_are_said_in_one_line(tmp_path: Path) -> None:
    """With the unedited trace gone, only the edit is marked and the caption says why."""

    clean, fork = _edited_pair()
    del clean
    gc.collect()
    source = fork.draw(view="interventions", path=str(tmp_path / "view"), file_format="svg").dot
    assert "changed ops not shown: the unedited trace is not available" in source
    assert CHANGED_INK not in source
    assert "light purple" not in source


def test_unchanged_downstream_values_stay_plain(tmp_path: Path) -> None:
    """An edit that changes nothing marks only its own site."""

    torch.manual_seed(0)
    clean = tl.trace(_Mlp().eval(), torch.randn(2, 8), capture=_CAPTURE)
    fork = clean.fork()
    fork.do(fork["relu_1_2:1"], tl.scale(1.0))
    source = fork.draw(view="interventions", path=str(tmp_path / "view"), file_format="svg").dot
    assert CHANGED_INK not in source
    assert "no other op's saved value changed" in source
    del clean


@pytest.mark.parametrize(
    ("edited", "clean", "changed", "largest"),
    [
        (torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.5]), True, 0.5),
        (torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.0]), False, None),
        (torch.tensor([math.nan, 1.0]), torch.tensor([math.nan, 1.0]), False, 0.0),
        (torch.tensor([math.nan, 1.0]), torch.tensor([0.0, 1.0]), True, math.inf),
        (torch.tensor([1, 2]), torch.tensor([1, 5]), True, 3.0),
        (torch.zeros(2), torch.zeros(3), True, None),
        (torch.zeros(2), torch.zeros(2, dtype=torch.float64), True, None),
        (None, torch.zeros(2), False, None),
    ],
)
def test_value_change_rules(
    edited: object, clean: object, changed: bool, largest: float | None
) -> None:
    """Equal values, matching NaN, shape or dtype changes and missing values."""

    assert _value_change(edited, clean) == (changed, largest)
