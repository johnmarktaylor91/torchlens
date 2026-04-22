"""PYTEST_DONT_REWRITE

Coverage for Phase 5 multi-pass LayerLog conditional aggregation.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from torchlens import log_forward_pass
from torchlens.data_classes.layer_log import LayerLog
from torchlens.data_classes.model_log import ModelLog


class AlternatingBranchBlock(nn.Module):
    """Block whose tensor-driven branch alternates by loop pass."""

    def __init__(self) -> None:
        """Initialise the shared branch module."""
        super().__init__()
        self.shared = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor, pass_num: int) -> torch.Tensor:
        """Run one alternating branch step.

        Parameters
        ----------
        x:
            Input tensor for this pass.
        pass_num:
            One-indexed loop pass number.

        Returns
        -------
        torch.Tensor
            Shared-module output from the selected branch.
        """
        branch_offset = 1.0 if pass_num % 2 == 1 else -1.0
        branch_marker = (x.mean() * 0) + branch_offset
        if branch_marker > 0:
            y = self.shared(x)
        else:
            y = self.shared(x)
        return y


class AlternatingRecurrentIfModel(nn.Module):
    """Recurrent model whose branch choice alternates across passes."""

    def __init__(self) -> None:
        """Initialise the repeating block."""
        super().__init__()
        self.block = AlternatingBranchBlock()

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
            x = self.block(x, pass_num)
        return x


class RolledMixedArmModel(nn.Module):
    """Model whose rolled arm-entry edge appears in both THEN and ELSE."""

    def __init__(self) -> None:
        """Initialise the repeating parent and branch layers."""
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


class LoopedIfAlternatingModel(nn.Module):
    """Loop model with one conditional that alternates between two arms."""

    def __init__(self) -> None:
        """Initialise the shared branch module."""
        super().__init__()
        self.shared = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the alternating loop.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Final loop output.
        """
        for i in range(4):
            branch_offset = 1.0 if i % 2 == 0 else -1.0
            branch_marker = (x.mean() * 0) + branch_offset
            if branch_marker > 0:
                x = self.shared(x)
            else:
                x = self.shared(x)
        return x


class RecurrentNoConditionModel(nn.Module):
    """Simple recurrent baseline with no conditional branches."""

    def __init__(self) -> None:
        """Initialise the repeated linear layer."""
        super().__init__()
        self.shared = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the non-conditional recurrent forward pass.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Final recurrent output.
        """
        for _ in range(3):
            x = self.shared(x)
        return x


def _log_model(model: nn.Module) -> ModelLog:
    """Log a model forward pass with full layer capture.

    Parameters
    ----------
    model:
        Model to execute.

    Returns
    -------
    ModelLog
        Fully postprocessed model log.
    """
    x = torch.ones(1, 4)
    return log_forward_pass(model, x, layers_to_save="all")


def _get_only_event(model_log: ModelLog) -> int:
    """Return the only conditional event id in a model log.

    Parameters
    ----------
    model_log:
        Postprocessed model log.

    Returns
    -------
    int
        Dense conditional id.
    """
    assert len(model_log.conditional_events) == 1
    return model_log.conditional_events[0].id


def _find_multi_pass_linear_layer(model_log: ModelLog) -> LayerLog:
    """Find the unique multi-pass linear LayerLog.

    Parameters
    ----------
    model_log:
        Postprocessed model log.

    Returns
    -------
    LayerLog
        Repeated linear layer entry.
    """
    matching_layers = [
        layer
        for layer in model_log.layer_logs.values()
        if layer.func_name == "linear" and layer.num_passes > 1
    ]
    assert len(matching_layers) == 1
    return matching_layers[0]


def _assert_sorted_unique_pass_lists(
    pass_map: Dict[Tuple[str, str, int, str], List[int]],
) -> None:
    """Assert every ``conditional_edge_passes`` value is sorted and unique.

    Parameters
    ----------
    pass_map:
        Rolled edge-pass mapping from the model log.
    """
    for pass_nums in pass_map.values():
        assert pass_nums == sorted(pass_nums)
        assert len(pass_nums) == len(set(pass_nums))


def test_alternating_recurrent_if_model_merges_layerlog_conditionals() -> None:
    """Multi-pass LayerLog stores both branch signatures and pass unions."""
    model_log = _log_model(AlternatingRecurrentIfModel())
    conditional_id = _get_only_event(model_log)
    linear_layer = _find_multi_pass_linear_layer(model_log)

    assert linear_layer.in_cond_branch is True
    assert linear_layer.conditional_branch_stacks == [
        [(conditional_id, "then")],
        [(conditional_id, "else")],
    ]
    assert linear_layer.conditional_branch_stack_passes == {
        ((conditional_id, "then"),): [1, 3],
        ((conditional_id, "else"),): [2, 4],
    }
    assert len(linear_layer.conditional_branch_stacks) >= 2

    parent_layer = next(
        layer
        for layer in model_log.layer_logs.values()
        if conditional_id in layer.cond_branch_children_by_cond
        and "then" in layer.cond_branch_children_by_cond[conditional_id]
        and "else" in layer.cond_branch_children_by_cond[conditional_id]
    )
    assert parent_layer.cond_branch_children_by_cond[conditional_id]["then"] == [
        linear_layer.layer_label
    ]
    assert parent_layer.cond_branch_children_by_cond[conditional_id]["else"] == [
        linear_layer.layer_label
    ]


def test_rolled_mixed_arm_model_records_per_arm_edge_passes() -> None:
    """Rolled mixed-arm edges retain sorted pass lists for both branches."""
    model_log = _log_model(RolledMixedArmModel())
    conditional_id = _get_only_event(model_log)

    then_edges = model_log.conditional_arm_edges[(conditional_id, "then")]
    else_edges = model_log.conditional_arm_edges[(conditional_id, "else")]

    assert len(then_edges) == 2
    assert len(else_edges) == 2

    parent_label, child_label = then_edges[0]
    parent_no_pass = model_log[parent_label].layer_label_no_pass
    child_no_pass = model_log[child_label].layer_label_no_pass

    assert model_log.conditional_edge_passes[
        (parent_no_pass, child_no_pass, conditional_id, "then")
    ] == [1, 3]
    assert model_log.conditional_edge_passes[
        (parent_no_pass, child_no_pass, conditional_id, "else")
    ] == [2, 4]
    _assert_sorted_unique_pass_lists(model_log.conditional_edge_passes)


def test_looped_if_alternating_model_has_exactly_two_signatures() -> None:
    """Looped alternating condition aggregates to exactly two stack signatures."""
    model_log = _log_model(LoopedIfAlternatingModel())
    conditional_id = _get_only_event(model_log)
    linear_layer = _find_multi_pass_linear_layer(model_log)

    assert linear_layer.conditional_branch_stacks == [
        [(conditional_id, "then")],
        [(conditional_id, "else")],
    ]
    assert linear_layer.conditional_branch_stack_passes == {
        ((conditional_id, "then"),): [1, 3],
        ((conditional_id, "else"),): [2, 4],
    }
    assert len(linear_layer.conditional_branch_stacks) == 2


def test_non_conditional_recurrent_model_keeps_empty_aggregate_views() -> None:
    """Non-conditional recurrent aggregation preserves empty conditional views."""
    model_log = _log_model(RecurrentNoConditionModel())
    linear_layer = _find_multi_pass_linear_layer(model_log)

    assert model_log.conditional_events == []
    assert linear_layer.in_cond_branch is False
    assert linear_layer.conditional_branch_stacks == [[]]
    assert linear_layer.conditional_branch_stack_passes == {(): [1, 2, 3]}
    assert linear_layer.cond_branch_children_by_cond == {}
    assert linear_layer.cond_branch_start_children == []
    assert linear_layer.cond_branch_then_children == []
    assert linear_layer.cond_branch_elif_children == {}
    assert linear_layer.cond_branch_else_children == []
