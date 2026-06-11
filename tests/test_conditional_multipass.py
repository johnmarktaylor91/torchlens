"""PYTEST_DONT_REWRITE

Coverage for Phase 5 multi-pass Layer conditional aggregation.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from torchlens import trace as trace_fn
from torchlens.data_classes.layer import Layer
from torchlens.data_classes.trace import Trace


class AlternatingBranchBlock(nn.Module):
    """Block whose tensor-driven branch alternates by loop pass."""

    def __init__(self) -> None:
        """Initialise the shared branch module."""
        super().__init__()
        self.shared = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor, call_index: int) -> torch.Tensor:
        """Run one alternating branch step.

        Parameters
        ----------
        x:
            Input tensor for this pass.
        call_index:
            One-indexed loop pass number.

        Returns
        -------
        torch.Tensor
            Shared-module output from the selected branch.
        """
        branch_offset = 1.0 if call_index % 2 == 1 else -1.0
        branch_marker = (x.mean() * 0) + branch_offset
        if branch_marker > 0:
            y = self.shared(x)
        else:
            y = self.shared(x)
        return y


class AlternatingRecurrentIfModel(nn.Module):
    """Recurrent model whose branch choice alternates across ops."""

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
        for call_index in range(1, 5):
            x = self.block(x, call_index)
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
        for call_index in range(1, 5):
            x = self.parent(x)
            branch_offset = 1.0 if call_index % 2 == 1 else -1.0
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


def _log_model(model: nn.Module) -> Trace:
    """Log a model forward pass with full layer capture.

    Parameters
    ----------
    model:
        Model to execute.

    Returns
    -------
    Trace
        Fully postprocessed model log.
    """
    x = torch.ones(1, 4)
    return trace_fn(model, x, layers_to_save="all")


def _get_only_event(trace: Trace) -> int:
    """Return the only conditional event id in a model log.

    Parameters
    ----------
    trace:
        Postprocessed model log.

    Returns
    -------
    int
        Dense conditional id.
    """
    assert len(trace.conditional_records) == 1
    return trace.conditional_records[0].id


def _find_multi_pass_linear_layer(trace: Trace) -> Layer:
    """Find the unique multi-pass linear Layer.

    Parameters
    ----------
    trace:
        Postprocessed model log.

    Returns
    -------
    Layer
        Repeated linear layer entry.
    """
    matching_layers = [
        layer
        for layer in trace.layer_logs.values()
        if layer.func_name == "linear" and layer.num_passes > 1
    ]
    assert len(matching_layers) == 1
    return matching_layers[0]


def _assert_sorted_unique_pass_lists(
    pass_map: Dict[Tuple[str, str, int, str], List[int]],
) -> None:
    """Assert every ``conditional_edge_call_indices`` value is sorted and unique.

    Parameters
    ----------
    pass_map:
        Rolled edge-pass mapping from the model log.
    """
    for call_indexs in pass_map.values():
        assert call_indexs == sorted(call_indexs)
        assert len(call_indexs) == len(set(call_indexs))


def test_alternating_recurrent_if_model_merges_layerlog_conditionals() -> None:
    """Multi-pass Layer stores both branch signatures and pass unions."""
    trace = _log_model(AlternatingRecurrentIfModel())
    conditional_id = _get_only_event(trace)
    linear_layer = _find_multi_pass_linear_layer(trace)

    assert linear_layer.is_in_conditional_body is True
    assert linear_layer.conditional_role_stacks == [
        [(conditional_id, "then")],
        [(conditional_id, "else")],
    ]
    assert linear_layer.conditional_branch_stack_ops == {
        ((conditional_id, "then"),): [1, 3],
        ((conditional_id, "else"),): [2, 4],
    }
    assert len(linear_layer.conditional_role_stacks) >= 2

    parent_layer = next(
        layer
        for layer in trace.layer_logs.values()
        if conditional_id in layer.conditional_arm_children
        and "then" in layer.conditional_arm_children[conditional_id]
        and "else" in layer.conditional_arm_children[conditional_id]
    )
    assert parent_layer.conditional_arm_children[conditional_id]["then"] == [
        linear_layer.layer_label
    ]
    assert parent_layer.conditional_arm_children[conditional_id]["else"] == [
        linear_layer.layer_label
    ]


def test_rolled_mixed_arm_model_records_per_arm_edge_ops() -> None:
    """Rolled mixed-arm edges retain sorted pass lists for both branches."""
    trace = _log_model(RolledMixedArmModel())
    conditional_id = _get_only_event(trace)

    then_edges = trace.conditional_arm_entry_edges[(conditional_id, "then")]
    else_edges = trace.conditional_arm_entry_edges[(conditional_id, "else")]

    assert len(then_edges) == 2
    assert len(else_edges) == 2

    parent_label, child_label = then_edges[0]
    parent_no_pass = trace[parent_label].layer_label
    child_no_pass = trace[child_label].layer_label

    assert trace.conditional_edge_call_indices[
        (parent_no_pass, child_no_pass, conditional_id, "then")
    ] == [
        1,
        3,
    ]
    assert trace.conditional_edge_call_indices[
        (parent_no_pass, child_no_pass, conditional_id, "else")
    ] == [
        2,
        4,
    ]
    _assert_sorted_unique_pass_lists(trace.conditional_edge_call_indices)


def test_looped_if_alternating_model_has_exactly_two_signatures() -> None:
    """Looped alternating condition aggregates to exactly two stack signatures."""
    trace = _log_model(LoopedIfAlternatingModel())
    conditional_id = _get_only_event(trace)
    linear_layer = _find_multi_pass_linear_layer(trace)

    assert linear_layer.conditional_role_stacks == [
        [(conditional_id, "then")],
        [(conditional_id, "else")],
    ]
    assert linear_layer.conditional_branch_stack_ops == {
        ((conditional_id, "then"),): [1, 3],
        ((conditional_id, "else"),): [2, 4],
    }
    assert len(linear_layer.conditional_role_stacks) == 2


def test_non_conditional_recurrent_model_keeps_empty_aggregate_views() -> None:
    """Non-conditional recurrent aggregation preserves empty conditional views."""
    trace = _log_model(RecurrentNoConditionModel())
    linear_layer = _find_multi_pass_linear_layer(trace)

    assert trace.conditional_records == []
    assert linear_layer.is_in_conditional_body is False
    assert linear_layer.conditional_role_stacks == [[]]
    assert linear_layer.conditional_branch_stack_ops == {(): [1, 2, 3]}
    assert linear_layer.conditional_arm_children == {}
    assert linear_layer.conditional_entry_children == []
    assert linear_layer.conditional_then_children == []
    assert linear_layer.conditional_elif_children == {}
    assert linear_layer.conditional_else_children == []


class SecondPassOnlyThenModel(nn.Module):
    """Model whose multi-pass parent only enters the THEN arm on pass 2.

    Regression model for COND-THEN-MULTIPASS: the rolled ``Layer`` for the
    repeated parent linear must reflect THEN-arm children contributed by pass
    2 even though pass 1 contributes none (the old merge kept only the first
    pass's conditional child views).
    """

    def __init__(self) -> None:
        """Initialise the repeated parent and the THEN-arm branch layer."""
        super().__init__()
        self.parent = nn.Linear(4, 4)
        self.branch = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run two parent passes; only the second triggers the THEN arm.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Final recurrent output.
        """
        for call_index in range(1, 3):
            x = self.parent(x)
            branch_marker = (x.mean() * 0) + (1.0 if call_index == 2 else -1.0)
            if branch_marker > 0:
                x = x + self.branch(x)
        return x


def test_then_children_from_second_pass_survive_layer_merge() -> None:
    """THEN children seen only on pass 2 must appear on the rolled Layer."""
    trace = _log_model(SecondPassOnlyThenModel())
    conditional_id = _get_only_event(trace)
    parent_layer = _find_multi_pass_linear_layer(trace)

    branch_layers = [
        layer
        for layer in trace.layer_logs.values()
        if layer.func_name == "linear" and layer.num_passes == 1
    ]
    assert len(branch_layers) == 1
    branch_label = branch_layers[0].layer_label

    # Pass 1 contributes no conditional children; pass 2 contributes the
    # THEN-arm entries.  (``ops`` is keyed by 1-based pass index via items().)
    pass_ops = dict(parent_layer.ops.items())
    assert pass_ops[1].conditional_arm_children == {}
    pass_two_then_children = [
        child_label.rsplit(":", 1)[0] if child_label.rsplit(":", 1)[-1].isdigit() else child_label
        for child_label in pass_ops[2].conditional_arm_children[conditional_id]["then"]
    ]
    assert branch_label in pass_two_then_children

    # The rolled aggregate must reflect pass 2, not just pass 1.
    assert branch_label in parent_layer.conditional_arm_children[conditional_id]["then"]
    assert branch_label in parent_layer.conditional_then_children
    assert parent_layer.conditional_elif_children == {}
    assert parent_layer.conditional_else_children == []
