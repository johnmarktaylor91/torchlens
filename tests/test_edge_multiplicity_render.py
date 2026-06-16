"""Regression tests for repeated parent edge rendering."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch

import torchlens as tl


class _AddTwice(torch.nn.Module):
    """Model whose input feeds both positional slots of ``add``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``x + x``.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Sum of the tensor with itself.
        """

        return x + x


class _CatTwice(torch.nn.Module):
    """Model whose input appears twice inside a ``cat`` sequence argument."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``torch.cat([x, x])``.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Concatenated tensor.
        """

        return torch.cat([x, x], dim=0)


class _SubTwice(torch.nn.Module):
    """Model whose input feeds both positional slots of non-commutative ``sub``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``torch.sub(x, x)``.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Difference of the tensor with itself.
        """

        return torch.sub(x, x)


class _SingleUse(torch.nn.Module):
    """Model with ordinary single-use dataflow edges."""

    def __init__(self) -> None:
        """Initialize the linear layer."""

        super().__init__()
        self.linear = torch.nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a single-use linear/relu graph.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Activated linear output.
        """

        return torch.relu(self.linear(x))


def _render_dot(model: torch.nn.Module, tmp_path: Path, name: str) -> str:
    """Trace and render a model to Graphviz DOT source.

    Parameters
    ----------
    model:
        Model to trace.
    tmp_path:
        Temporary output directory.
    name:
        Output file stem.

    Returns
    -------
    str
        DOT source returned by ``Trace.draw``.
    """

    trace = tl.trace(model, torch.randn(2, 3))
    return str(
        trace.draw(
            vis_outpath=str(tmp_path / name),
            vis_fileformat="svg",
            vis_save_only=True,
            order_siblings=False,
        )
    )


def _dataflow_edge_lines(dot_source: str) -> list[str]:
    """Return non-overlay dataflow edge lines from DOT source.

    Parameters
    ----------
    dot_source:
        Graphviz DOT source.

    Returns
    -------
    list[str]
        Edge lines excluding invisible sibling-ordering helper edges.
    """

    return [
        line.strip()
        for line in dot_source.splitlines()
        if " -> " in line and "tl:sibling-order" not in line
    ]


def _matching_edge_lines(dot_source: str, target_func: str) -> list[str]:
    """Return input-to-target edge lines for a rendered operation.

    Parameters
    ----------
    dot_source:
        Graphviz DOT source.
    target_func:
        Rendered operation name prefix, such as ``"add"``.

    Returns
    -------
    list[str]
        Matching DOT edge lines.
    """

    pattern = re.compile(rf"^input_1pass1 -> {re.escape(target_func)}_1_1pass1\b")
    return [line for line in _dataflow_edge_lines(dot_source) if pattern.search(line)]


@pytest.mark.smoke
def test_add_same_parent_arg_slots_render_two_unlabeled_edges(tmp_path: Path) -> None:
    """``x + x`` renders one arrow per arg-slot occurrence without labels."""

    dot_source = _render_dot(_AddTwice(), tmp_path, "add_twice")
    edges = _matching_edge_lines(dot_source, "add")

    assert len(edges) == 2
    assert all("label=" not in edge and "headlabel=" not in edge for edge in edges)


@pytest.mark.smoke
def test_cat_same_parent_sequence_slots_render_two_unlabeled_edges(tmp_path: Path) -> None:
    """``torch.cat([x, x])`` renders one arrow per sequence-slot occurrence."""

    dot_source = _render_dot(_CatTwice(), tmp_path, "cat_twice")
    edges = _matching_edge_lines(dot_source, "cat")

    assert len(edges) == 2
    assert all("label=" not in edge and "headlabel=" not in edge for edge in edges)


@pytest.mark.smoke
def test_single_use_edges_render_once(tmp_path: Path) -> None:
    """Ordinary single-use render edges are not overdrawn."""

    dot_source = _render_dot(_SingleUse(), tmp_path, "single_use")
    edge_counts: dict[tuple[str, str], int] = {}
    for edge in _dataflow_edge_lines(dot_source):
        tail, rest = edge.split(" -> ", 1)
        head = rest.split(" ", 1)[0]
        edge_counts[(tail, head)] = edge_counts.get((tail, head), 0) + 1

    assert edge_counts
    assert all(count == 1 for count in edge_counts.values())


@pytest.mark.smoke
def test_non_commutative_same_parent_arg_slots_render_labeled_edges(tmp_path: Path) -> None:
    """``torch.sub(x, x)`` renders two per-slot-labeled arrows."""

    dot_source = _render_dot(_SubTwice(), tmp_path, "sub_twice")
    edges = _matching_edge_lines(dot_source, "sub")

    assert len(edges) == 2
    assert any("arg 0" in edge for edge in edges)
    assert any("arg 1" in edge for edge in edges)
