"""Tests for structured ModuleCall call-tree accessors."""

from __future__ import annotations

from io import StringIO

import torch
from torch import nn

import torchlens as tl
from torchlens.data_classes import ModuleCall


class Leaf(nn.Module):
    """Leaf module used in a nested call-tree fixture."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a pointwise nonlinearity.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Rectified tensor.
        """

        return torch.relu(x)


class Branch(nn.Module):
    """Intermediate module containing one child leaf."""

    def __init__(self) -> None:
        """Initialize the nested leaf."""

        super().__init__()
        self.leaf = Leaf()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the leaf and add a residual tensor.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Branch output.
        """

        return self.leaf(x) + x


class CallTreeModel(nn.Module):
    """Multi-level module fixture for call-tree tests."""

    def __init__(self) -> None:
        """Initialize the branch."""

        super().__init__()
        self.branch = Branch()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the nested branch and one top-level operation.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Model output.
        """

        return self.branch(x) * 2


def _trace() -> tl.Trace:
    """Build the call-tree fixture trace.

    Returns
    -------
    tl.Trace
        Trace with a three-level ModuleCall tree.
    """

    return tl.trace(CallTreeModel(), torch.randn(2, 3), layers_to_save="all")


def _printed_call_labels(trace: tl.Trace) -> list[str]:
    """Return call labels rendered by ``show_call_tree``.

    Parameters
    ----------
    trace:
        Trace whose call tree should be rendered.

    Returns
    -------
    list[str]
        Display labels in printed tree order.
    """

    output = StringIO()
    trace.show_call_tree(file=output)
    labels: list[str] = []
    for line in output.getvalue().splitlines():
        cleaned = line.replace("|   ", "").replace("    ", "")
        cleaned = cleaned.replace("|-- ", "").replace("`-- ", "")
        cleaned = cleaned.replace("\u251c\u2500\u2500 ", "")
        cleaned = cleaned.replace("\u2514\u2500\u2500 ", "")
        labels.append(cleaned.split(" ", 1)[0])
    return labels


def test_trace_call_tree_matches_show_call_tree_order() -> None:
    """Structured call tree has the same node order as the display helper."""

    trace = _trace()

    structured_labels = [call.call_label for call in trace.call_tree.walk_descendants()]

    assert structured_labels == _printed_call_labels(trace)


def test_call_tree_nodes_are_module_calls() -> None:
    """Trace, Module, and ModuleCall accessors expose ModuleCall roots."""

    trace = _trace()
    root = trace.call_tree

    assert isinstance(root, ModuleCall)
    assert trace.root_module.call_tree is root
    assert trace.modules["branch"].call_tree is trace.module_calls["branch:1"]
    assert trace.module_calls["branch:1"].call_tree is trace.module_calls["branch:1"]


def test_call_tree_depth_is_nested() -> None:
    """Structured call tree reports the expected descendant depth."""

    trace = _trace()

    assert trace.call_tree.call_label == "self:1"
    assert trace.call_tree.max_descendant_depth >= 2
    assert [child.call_label for child in trace.call_tree.walk_descendants()] == [
        "self:1",
        "branch:1",
        "branch.leaf:1",
    ]
