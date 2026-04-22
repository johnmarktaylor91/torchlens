"""Graphviz rendering coverage for conditional edge labels."""

from __future__ import annotations

import os
import tempfile
from typing import Tuple

import pytest
import torch
import torch.nn as nn

from torchlens import log_forward_pass
from torchlens.data_classes.model_log import ModelLog


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
    """Model whose rolled arm-entry edge appears in both THEN and ELSE passes."""

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
        for pass_num in range(1, 5):
            x = self.parent(x)
            branch_offset = 1.0 if pass_num % 2 == 1 else -1.0
            branch_marker = (x.mean() * 0) + branch_offset
            if branch_marker > 0:
                x = self.branch(x)
            else:
                x = self.branch(x)
        return x


def _render_dot_source(
    model: nn.Module,
    x: torch.Tensor,
    vis_mode: str = "unrolled",
) -> Tuple[str, ModelLog]:
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
    Tuple[str, ModelLog]
        Rendered DOT source and the populated model log.
    """
    model_log = log_forward_pass(model, x, save_source_context=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        outpath = os.path.join(tmpdir, "conditional_render")
        dot_source = model_log.render_graph(
            vis_mode=vis_mode,
            vis_outpath=outpath,
            vis_save_only=True,
            vis_fileformat="dot",
        )
    return dot_source, model_log


def _get_only_event_id(model_log: ModelLog) -> int:
    """Return the lone conditional id from a model log.

    Parameters
    ----------
    model_log:
        Logged model execution.

    Returns
    -------
    int
        Dense conditional id.
    """
    assert len(model_log.conditional_events) == 1
    return model_log.conditional_events[0].id


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


@pytest.mark.smoke
def test_simple_if_else_graphviz_labels_then_and_else_edges() -> None:
    """Simple ``if``/``else`` rendering shows THEN and ELSE labels on the right edges."""
    positive_dot, positive_log = _render_dot_source(SimpleIfElseModel(), torch.ones(2, 2))
    negative_dot, negative_log = _render_dot_source(SimpleIfElseModel(), -torch.ones(2, 2))

    try:
        positive_conditional_id = _get_only_event_id(positive_log)
        negative_conditional_id = _get_only_event_id(negative_log)
        then_parent, then_child = positive_log.conditional_arm_edges[
            (positive_conditional_id, "then")
        ][0]
        else_parent, else_child = negative_log.conditional_arm_edges[
            (negative_conditional_id, "else")
        ][0]

        then_line = _find_edge_line(positive_dot, then_parent, then_child)
        else_line = _find_edge_line(negative_dot, else_parent, else_child)

        assert "THEN" in then_line
        assert "ELSE" in else_line
    finally:
        positive_log.cleanup()
        negative_log.cleanup()


@pytest.mark.smoke
def test_elif_ladder_graphviz_labels_elif_and_else_edges() -> None:
    """Elif ladder rendering shows ``ELIF 1``, ``ELIF 2``, and ``ELSE`` labels."""
    expected_cases = [
        ("elif_1", torch.tensor([[-0.25]]), "ELIF 1"),
        ("elif_2", torch.tensor([[0.25]]), "ELIF 2"),
        ("else", torch.tensor([[1.0]]), "ELSE"),
    ]

    for branch_kind, x, label_text in expected_cases:
        dot_source, model_log = _render_dot_source(ElifLadderModel(), x)
        try:
            conditional_id = _get_only_event_id(model_log)
            parent_label, child_label = model_log.conditional_arm_edges[
                (conditional_id, branch_kind)
            ][0]
            edge_line = _find_edge_line(dot_source, parent_label, child_label)
            assert label_text in edge_line
        finally:
            model_log.cleanup()


def test_basic_ternary_graphviz_labels_then_and_else_edges() -> None:
    """Ternary rendering uses THEN/ELSE labels for ``ifexp`` arms."""
    positive_dot, positive_log = _render_dot_source(BasicTernaryModel(), torch.ones(2, 2))
    negative_dot, negative_log = _render_dot_source(BasicTernaryModel(), -torch.ones(2, 2))

    try:
        positive_conditional_id = _get_only_event_id(positive_log)
        negative_conditional_id = _get_only_event_id(negative_log)
        assert positive_log.conditional_events[0].kind == "ifexp"
        assert negative_log.conditional_events[0].kind == "ifexp"

        then_parent, then_child = positive_log.conditional_arm_edges[
            (positive_conditional_id, "then")
        ][0]
        else_parent, else_child = negative_log.conditional_arm_edges[
            (negative_conditional_id, "else")
        ][0]

        then_line = _find_edge_line(positive_dot, then_parent, then_child)
        else_line = _find_edge_line(negative_dot, else_parent, else_child)

        assert "THEN" in then_line
        assert "ELSE" in else_line
    finally:
        positive_log.cleanup()
        negative_log.cleanup()


def test_branch_entry_with_arg_label_keeps_semantic_and_argument_labels_separate() -> None:
    """Branch-entry edges retain the branch label and move arg labels to the head/x label."""
    dot_source, model_log = _render_dot_source(BranchEntryWithArgLabelModel(), torch.ones(2, 2))

    try:
        conditional_id = _get_only_event_id(model_log)
        parent_label, child_label = model_log.conditional_arm_edges[(conditional_id, "then")][0]
        edge_line = _find_edge_line(dot_source, parent_label, child_label)

        assert 'label=<<FONT POINT-SIZE="18"><b><u>THEN</u></b></FONT>>' in edge_line
        assert (
            "headlabel=<<FONT POINT-SIZE='10'><b>arg" in edge_line
            or "xlabel=<<FONT POINT-SIZE='10'><b>arg" in edge_line
        )
    finally:
        model_log.cleanup()


def test_rolled_mixed_arm_graphviz_shows_composite_pass_label() -> None:
    """Rolled rendering shows a composite THEN/ELSE label when passes diverge."""
    dot_source, model_log = _render_dot_source(
        RolledMixedArmModel(),
        torch.ones(1, 4),
        vis_mode="rolled",
    )

    try:
        conditional_id = _get_only_event_id(model_log)
        parent_label, child_label = model_log.conditional_arm_edges[(conditional_id, "then")][0]
        edge_line = _find_edge_line(dot_source, parent_label, child_label)

        assert "THEN(1,3) / ELSE(2,4)" in edge_line
    finally:
        model_log.cleanup()
