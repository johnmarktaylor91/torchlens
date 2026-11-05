"""``Trace.draw(show_arg_labels=)`` and text-free nodes.

Argument-position labels (``arg 0``, ``arg (0, 1)``) mark the edges into a multi-input
operation by default; ``show_arg_labels=False`` draws none. A node spec whose ``lines`` are
empty renders an empty label instead of an empty HTML table, which Graphviz rejects.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl


class TwoBranchCat(nn.Module):
    """Two linear branches concatenated: ``cat`` has two parents, so its edges get arg labels."""

    def __init__(self) -> None:
        super().__init__()
        self.left = nn.Linear(4, 3)
        self.right = nn.Linear(4, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.left(x), self.right(x)], dim=1)


@pytest.fixture
def cat_trace() -> tl.Trace:
    torch.manual_seed(0)
    return tl.trace(TwoBranchCat(), torch.randn(2, 4))


def test_arg_labels_present_by_default(cat_trace: tl.Trace, tmp_path: Path) -> None:
    dot = str(cat_trace.draw(str(tmp_path / "g"), file_format="gv").dot)
    assert "arg (0, 0)" in dot and "arg (0, 1)" in dot


def test_show_arg_labels_false_draws_none(cat_trace: tl.Trace, tmp_path: Path) -> None:
    dot = str(cat_trace.draw(str(tmp_path / "g"), file_format="gv", show_arg_labels=False).dot)
    assert "arg (" not in dot
    # Every other edge attribute is untouched: the two cat edges are still there.
    assert dot.count("-> cat_1_") == 2


def test_show_arg_labels_option_round_trip() -> None:
    options = tl.options.VisualizationOptions()
    assert options.show_arg_labels is True
    assert tl.options.VisualizationOptions(show_arg_labels=False).show_arg_labels is False
    with pytest.raises(tl.errors.InvalidArgumentError):
        tl.options.VisualizationOptions(show_arg_labels="no")  # type: ignore[arg-type]


def test_show_arg_labels_false_leaves_default_edge_count(
    cat_trace: tl.Trace, tmp_path: Path
) -> None:
    plain = str(cat_trace.draw(str(tmp_path / "a"), file_format="gv").dot)
    quiet = str(cat_trace.draw(str(tmp_path / "b"), file_format="gv", show_arg_labels=False).dot)
    assert plain.count("->") == quiet.count("->")


def test_empty_lines_node_spec_renders_an_empty_label(cat_trace: tl.Trace, tmp_path: Path) -> None:
    def dots(layer, spec):  # noqa: ANN001 - the documented callback contract
        spec.lines = []
        spec.shape = "circle"
        return spec

    dot = str(
        cat_trace.draw(
            str(tmp_path / "g"), file_format="gv", node_spec_fn=dots, show_arg_labels=False
        ).dot
    )
    node_lines = [
        line
        for line in dot.splitlines()
        if "label=" in line
        and "->" not in line
        and not line.strip().startswith(("graph ", "node ", "edge "))
    ]
    assert node_lines and all('label=""' in line for line in node_lines), node_lines
    assert "<TABLE" not in dot
    if shutil.which("dot") is None:
        pytest.skip("Graphviz is not installed")
    # Graphviz accepts the source: an empty HTML table used to be a syntax error.
    cat_trace.draw(str(tmp_path / "g.svg"), node_spec_fn=dots, show_arg_labels=False)
    assert (tmp_path / "g.svg").exists()
