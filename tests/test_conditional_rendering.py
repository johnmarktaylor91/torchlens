"""Graphviz rendering coverage for conditional edge labels."""

from __future__ import annotations

import os
import tempfile

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens._data_classes.trace import Trace
from torchlens._utils._torch_compat import HAS_CODE_POSITIONS
from torchlens.options import CaptureOptions


class SimpleIfElseModel(nn.Module):
    """Minimal model with one ``if``/``else`` branch."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model forward pass.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Branch-selected output tensor.
        """
        if x.mean() > 0:
            y = torch.relu(x)
        else:
            y = torch.sigmoid(x)
        return y


class ElifLadderModel(nn.Module):
    """Model with one ``if``/``elif``/``elif``/``else`` ladder."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model forward pass.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Branch-selected output tensor.
        """
        if x.mean() < -0.5:
            y = torch.relu(x)
        elif x.mean() < 0.0:
            y = torch.sigmoid(x)
        elif x.mean() < 0.5:
            y = torch.tanh(x)
        else:
            y = torch.square(x)
        return y


class BasicTernaryModel(nn.Module):
    """Minimal model using a ternary expression."""

    def __init__(self) -> None:
        """Initialise the two ternary branches."""
        super().__init__()
        self.then_branch = nn.Linear(2, 2)
        self.else_branch = nn.Linear(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model forward pass.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Branch-selected output tensor.
        """
        return self.then_branch(x) if x.mean() > 0 else self.else_branch(x)


class BranchEntryWithArgLabelModel(nn.Module):
    """Model whose branch-entry edge also needs an argument-position label."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model forward pass.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor from a non-commutative op inside each branch.
        """
        baseline = torch.neg(x)
        if x.mean() > 0:
            return torch.sub(baseline, x)
        return torch.sub(x, baseline)


class RolledMixedArmModel(nn.Module):
    """Model whose rolled arm-entry edge appears in both THEN and ELSE ops."""

    def __init__(self) -> None:
        """Initialise the repeated parent and branch layers."""
        super().__init__()
        self.parent = nn.Linear(4, 4)
        self.branch = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the recurrent forward pass.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Final recurrent output.
        """
        for call_index in range(1, 5):
            x = self.parent(x)
            branch_offset = 1.0 if call_index % 2 == 1 else -1.0
            branch_marker = (x.mean() * 0) + branch_offset
            if branch_marker > 0:
                x = self.branch(x)
            else:
                x = self.branch(x)
        return x


class LoopRepeatedIfModel(nn.Module):
    """Evaluate the same source ``if`` independently on every loop iteration."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run four evaluations of one source-level condition."""

        for index in range(4):
            marker = (x.mean() * 0) + (1 if index % 2 == 0 else -1)
            if marker > 0:
                x = torch.relu(x)
            else:
                x = torch.sigmoid(x)
        return x


def _render_dot_source(
    model: nn.Module,
    x: torch.Tensor,
    rolled: bool = False,
) -> tuple[str, Trace]:
    """Render a model graph and return the DOT source plus model log.

    Parameters
    ----------
    model:
        Model to execute.
    x:
        Input tensor for the forward pass.
    vis_mode:
        Rendering mode, either ``"unrolled"`` or ``"rolled"``.

    Returns
    -------
    Tuple[str, Trace]
        Rendered DOT source and the populated model log.
    """
    trace = tl.trace(model, x, capture=CaptureOptions(save_source_context=True))
    with tempfile.TemporaryDirectory() as tmpdir:
        outpath = os.path.join(tmpdir, "conditional_render")
        dot_source = trace.draw(
            rolled=rolled,
            path=outpath,
            file_format="dot",
        ).dot
    return dot_source, trace


def _get_only_event_id(trace: Trace) -> int:
    """Return the lone conditional id from a model log.

    Parameters
    ----------
    trace:
        Logged model execution.

    Returns
    -------
    int
        Dense conditional id.
    """
    assert len(trace.graph.conditional_records) == 1
    return trace.graph.conditional_records[0].id


def _find_edge_line(dot_source: str, parent_label: str, child_label: str) -> str:
    """Return the DOT line for one rendered edge.

    Parameters
    ----------
    dot_source:
        Graphviz DOT source.
    parent_label:
        Source layer label from the model log.
    child_label:
        Destination layer label from the model log.

    Returns
    -------
    str
        Matching DOT line.
    """
    parent_names = {parent_label.replace(":", "pass"), parent_label.split(":")[0]}
    child_names = {child_label.replace(":", "pass"), child_label.split(":")[0]}
    for line in dot_source.splitlines():
        if "->" not in line:
            continue
        if any(parent_name in line for parent_name in parent_names) and any(
            child_name in line for child_name in child_names
        ):
            return line
    raise AssertionError(f"Could not find edge line for {parent_label!r} -> {child_label!r}")


def _find_node_line(dot_source: str, layer_label: str) -> str:
    """Return the DOT line that declares one rendered node.

    Parameters
    ----------
    dot_source:
        Graphviz DOT source.
    layer_label:
        Layer label from the model log.

    Returns
    -------
    str
        Matching DOT line.
    """
    names = {layer_label.replace(":", "pass"), layer_label.split(":")[0]}
    for line in dot_source.splitlines():
        if "->" not in line and line.strip().split(" ", 1)[0] in names:
            return line
    raise AssertionError(f"Could not find node line for {layer_label!r}")


def test_elif_ladder_tests_name_if_then_elif_on_the_test_node() -> None:
    """The leading test node reads ``IF:`` and each later test ``ELIF:``.

    Distinct from the arm-entry labels (THEN/ELIF N/ELSE) on edges: the test kind sits on
    the test itself, and the edges entering each condition subgraph carry no ``IF`` label.
    """
    dot_source, trace = _render_dot_source(ElifLadderModel(), torch.tensor([[0.25]]))
    try:
        bool_layers = trace.graph.conditional_records[0].bool_layers
        for index, bool_label in enumerate(bool_layers):
            node_line = _find_node_line(dot_source, bool_label)
            expected = "IF:" if index == 0 else "ELIF:"
            assert expected in node_line, f"{bool_label} -> expected {expected}: {node_line}"
            if index == 0:
                assert "ELIF" not in node_line
        for parent_label, child_label in trace.graph.conditional_branch_edges:
            edge_line = _find_edge_line(dot_source, parent_label, child_label)
            assert "<u>IF</u>" not in edge_line and "<u>ELIF</u>" not in edge_line
    finally:
        trace.cleanup()


# Kept: edited on the C-100 side (C-091/C-098) after the shrink removed it.
def test_loop_repeated_if_tests_remain_if() -> None:
    """Repeated evaluations of one source ``if`` never become an ELIF ladder."""

    dot_source, trace = _render_dot_source(LoopRepeatedIfModel(), torch.ones(2, 2))
    try:
        event = trace.graph.conditional_records[0]
        assert len(event.bool_layers) == 4
        assert all(
            {op.conditional_context_kind for op in trace.layers[bool_label].ops.values()}
            == {"if_test"}
            for bool_label in event.bool_layers
        )
        node_lines = [_find_node_line(dot_source, label) for label in event.bool_layers]
        assert all("IF:" in line and "ELIF" not in line for line in node_lines)
    finally:
        trace.cleanup()


@pytest.mark.skipif(
    not HAS_CODE_POSITIONS,
    reason=(
        "PEP 657 bytecode column positions unavailable; same-line ternary arm "
        "attribution deliberately fails closed on this runtime"
    ),
)
def test_basic_ternary_graphviz_labels_then_and_else_edges() -> None:
    """Ternary rendering uses THEN/ELSE labels for ``ifexp`` arms."""
    positive_dot, positive_log = _render_dot_source(BasicTernaryModel(), torch.ones(2, 2))
    negative_dot, negative_log = _render_dot_source(BasicTernaryModel(), -torch.ones(2, 2))

    try:
        positive_conditional_id = _get_only_event_id(positive_log)
        negative_conditional_id = _get_only_event_id(negative_log)
        assert positive_log.graph.conditional_records[0].kind == "ifexp"
        assert negative_log.graph.conditional_records[0].kind == "ifexp"

        then_parent, then_child = positive_log.graph.conditional_arm_entry_edges[
            (positive_conditional_id, "then")
        ][0]
        else_parent, else_child = negative_log.graph.conditional_arm_entry_edges[
            (negative_conditional_id, "else")
        ][0]

        then_line = _find_edge_line(positive_dot, then_parent, then_child)
        else_line = _find_edge_line(negative_dot, else_parent, else_child)

        assert "THEN" in then_line
        assert "ELSE" in else_line
    finally:
        positive_log.cleanup()
        negative_log.cleanup()


def test_rolled_mixed_arm_graphviz_shows_composite_call_label() -> None:
    """Rolled rendering shows a composite THEN/ELSE label when ops diverge."""
    dot_source, trace = _render_dot_source(
        RolledMixedArmModel(),
        torch.ones(1, 4),
        rolled=True,
    )

    try:
        conditional_id = _get_only_event_id(trace)
        parent_label, child_label = trace.graph.conditional_arm_entry_edges[
            (conditional_id, "then")
        ][0]
        edge_line = _find_edge_line(dot_source, parent_label, child_label)

        assert "THEN(1,3) / ELSE(2,4)" in edge_line
    finally:
        trace.cleanup()
