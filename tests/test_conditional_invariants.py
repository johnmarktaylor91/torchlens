"""PYTEST_DONT_REWRITE

Targeted corruption tests for Phase 6 conditional metadata invariants.
"""

from __future__ import annotations

from typing import Iterable

import pytest
import torch
import torch.nn as nn

from torchlens import MetadataInvariantError, check_metadata_invariants, trace as trace_fn
from torchlens.data_classes.layer_log import Layer
from torchlens.data_classes.op_log import Op
from torchlens.data_classes.model_log import ConditionalEvent, Trace


class SimpleIfElseModel(nn.Module):
    """Minimal model with one tensor-driven ``if``/``else`` branch."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the conditional forward pass.

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
    """Model with one flattened ``if``/``elif``/``elif``/``else`` ladder."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the ladder forward pass.

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


class NestedIfModel(nn.Module):
    """Model with nested tensor-driven ``if`` statements."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the nested conditional forward pass.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Final branch-selected output tensor.
        """

        if x.mean() > 0:
            y = torch.relu(x)
            if y.mean() > 0:
                z = torch.sigmoid(y)
            else:
                z = torch.tanh(y)
        else:
            z = torch.square(x)
        return z


class AlternatingBranchBlock(nn.Module):
    """Shared block whose taken branch alternates by recurrent pass."""

    def __init__(self) -> None:
        """Initialise the shared linear layer."""

        super().__init__()
        self.shared = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor, call_index: int) -> torch.Tensor:
        """Run one alternating branch step.

        Parameters
        ----------
        x:
            Input tensor for this pass.
        call_index:
            One-indexed recurrent pass number.

        Returns
        -------
        torch.Tensor
            Shared-layer output from the selected branch.
        """

        branch_offset = 1.0 if call_index % 2 == 1 else -1.0
        branch_marker = (x.mean() * 0) + branch_offset
        if branch_marker > 0:
            y = self.shared(x)
        else:
            y = self.shared(x)
        return y


class AlternatingRecurrentIfModel(nn.Module):
    """Recurrent model whose tensor-driven branch alternates across ops."""

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
            Final recurrent output tensor.
        """

        for call_index in range(1, 5):
            x = self.block(x, call_index)
        return x


def _log_model(model: nn.Module, x: torch.Tensor) -> Trace:
    """Capture a full ``Trace`` for a small test model.

    Parameters
    ----------
    model:
        Model to execute.
    x:
        Input tensor for the forward pass.

    Returns
    -------
    Trace
        Fully postprocessed model log.
    """

    return trace_fn(model, x, layers_to_save="all")


def _get_only_event(trace: Trace) -> ConditionalEvent:
    """Return the only ``ConditionalEvent`` in a model log.

    Parameters
    ----------
    trace:
        Logged model execution.

    Returns
    -------
    ConditionalEvent
        The lone conditional event.
    """

    assert len(trace.conditional_records) == 1
    return trace.conditional_records[0]


def _get_only_terminal_bool(trace: Trace) -> Op:
    """Return the only terminal scalar bool layer in a model log.

    Parameters
    ----------
    trace:
        Logged model execution.

    Returns
    -------
    Op
        The only terminal scalar bool layer.
    """

    bool_layers = [
        layer for layer in trace.layer_list if layer.is_terminal_bool and layer.is_scalar_bool
    ]
    assert len(bool_layers) == 1
    return bool_layers[0]


def _find_multi_pass_linear_layer(trace: Trace) -> Layer:
    """Return the unique repeated linear ``Layer`` in a recurrent test log.

    Parameters
    ----------
    trace:
        Logged recurrent model execution.

    Returns
    -------
    Layer
        Multi-pass linear layer aggregate.
    """

    matching_layers = [
        layer_log
        for layer_log in trace.layer_logs.values()
        if layer_log.func_name == "linear" and layer_log.num_passes > 1
    ]
    assert len(matching_layers) == 1
    return matching_layers[0]


def _assert_invariant_error(trace: Trace, substrings: Iterable[str]) -> None:
    """Assert that invariant validation fails with all expected substrings.

    Parameters
    ----------
    trace:
        Corrupted model log to validate.
    substrings:
        Message fragments that must appear in the raised error.
    """

    with pytest.raises(MetadataInvariantError) as exc_info:
        check_metadata_invariants(trace)
    message = str(exc_info.value)
    for substring in substrings:
        assert substring in message


def _sync_layer_log_child_views(layer_log: Layer) -> None:
    """Recompute ``Layer`` derived child views from its primary structure.

    Parameters
    ----------
    layer_log:
        Aggregate layer entry whose derived conditional views should match
        its current ``conditional_arm_children`` mapping.
    """

    then_children: list[str] = []
    elif_children: dict[int, list[str]] = {}
    else_children: list[str] = []
    for branch_children in layer_log.conditional_arm_children.values():
        for child_label in branch_children.get("then", []):
            if child_label not in then_children:
                then_children.append(child_label)
        for branch_kind, child_labels in branch_children.items():
            if not branch_kind.startswith("elif_"):
                continue
            elif_index = int(branch_kind.split("_", 1)[1])
            expected_children = elif_children.setdefault(elif_index, [])
            for child_label in child_labels:
                if child_label not in expected_children:
                    expected_children.append(child_label)
        for child_label in branch_children.get("else", []):
            if child_label not in else_children:
                else_children.append(child_label)
    layer_log.conditional_then_children = then_children
    layer_log.conditional_elif_children = elif_children
    layer_log.conditional_else_children = else_children


def test_clean_conditional_log_ops_all_invariants() -> None:
    """Clean conditional metadata ops the full invariant validator."""

    trace = _log_model(ElifLadderModel(), torch.tensor([[0.25]]))
    try:
        assert check_metadata_invariants(trace) is True
    finally:
        trace.cleanup()


def test_invariant_1_arm_edges_bidirectional_consistency() -> None:
    """Invariant 1 fails when a parent child-list no longer matches arm edges."""

    trace = _log_model(SimpleIfElseModel(), torch.ones(2, 3))
    try:
        event = _get_only_event(trace)
        parent_label, child_label = trace.conditional_arm_entry_edges[(event.id, "then")][0]
        parent_layer = trace.layer_logs[parent_label]
        parent_layer.conditional_arm_children[event.id]["then"] = [
            label
            for label in parent_layer.conditional_arm_children[event.id]["then"]
            if label != child_label
        ]
        _assert_invariant_error(trace, ("Invariant 1", "conditional_arm_entry_edges"))
    finally:
        trace.cleanup()


def test_invariant_2_derived_child_views_match_primary_structures() -> None:
    """Invariant 2 fails when a derived layer child view is corrupted."""

    trace = _log_model(SimpleIfElseModel(), torch.ones(2, 3))
    try:
        event = _get_only_event(trace)
        parent_label, _child_label = trace.conditional_arm_entry_edges[(event.id, "then")][0]
        parent_layer = trace[parent_label]
        parent_layer.conditional_then_children = []
        _assert_invariant_error(trace, ("Invariant 2", "conditional_then_children"))
    finally:
        trace.cleanup()


def test_invariant_3_child_labels_exist_in_trace() -> None:
    """Invariant 3 fails when a conditional child label does not exist."""

    trace = _log_model(SimpleIfElseModel(), torch.ones(2, 3))
    try:
        parent_label = trace.conditional_branch_edges[0][0]
        missing_label = "missing_bool_layer"
        trace[parent_label].conditional_entry_children.append(missing_label)
        trace.conditional_branch_edges.append((parent_label, missing_label))
        _assert_invariant_error(trace, ("Invariant 3", missing_label))
    finally:
        trace.cleanup()


def test_invariant_4_bool_classification_fields_agree() -> None:
    """Invariant 4 fails when ``is_terminal_conditional_bool`` disagrees with the context kind."""

    trace = _log_model(SimpleIfElseModel(), torch.ones(2, 3))
    try:
        bool_layer = _get_only_terminal_bool(trace)
        bool_layer.is_terminal_conditional_bool = False
        _assert_invariant_error(trace, ("Invariant 4", "is_terminal_conditional_bool"))
    finally:
        trace.cleanup()


def test_invariant_5_all_conditional_ids_resolve_to_events() -> None:
    """Invariant 5 fails when a referenced cond_id is missing from events."""

    trace = _log_model(SimpleIfElseModel(), torch.ones(2, 3))
    try:
        bool_layer = _get_only_terminal_bool(trace)
        bool_layer.terminal_conditional_id = 999
        _assert_invariant_error(trace, ("Invariant 5", "cond_id 999"))
    finally:
        trace.cleanup()


def test_invariant_6_parent_child_stacks_are_prefix_related() -> None:
    """Invariant 6 fails on a non-prefix stack reordering across one graph edge."""

    trace = _log_model(NestedIfModel(), torch.ones(2, 3))
    try:
        violating_child: Op | None = None
        for layer in trace.layer_list:
            if len(layer.conditional_branch_stack) != 2:
                continue
            for parent_label in layer.parents:
                parent_layer = trace[parent_label]
                if len(parent_layer.conditional_branch_stack) == 1:
                    violating_child = layer
                    break
            if violating_child is not None:
                break

        assert violating_child is not None
        violating_child.conditional_branch_stack = list(
            reversed(violating_child.conditional_branch_stack)
        )
        _assert_invariant_error(trace, ("Invariant 6", "non-prefix conditional stacks"))
    finally:
        trace.cleanup()


def test_invariant_7_elif_keys_are_contiguous() -> None:
    """Invariant 7 fails when ``ConditionalEvent`` elif keys skip an index."""

    trace = _log_model(ElifLadderModel(), torch.tensor([[0.25]]))
    try:
        event = _get_only_event(trace)
        event.branch_ranges["elif_3"] = event.branch_ranges.pop("elif_1")
        _assert_invariant_error(trace, ("Invariant 7", "branch_ranges"))
    finally:
        trace.cleanup()


def test_invariant_8_event_bool_layers_point_back_to_the_event() -> None:
    """Invariant 8 fails when ``ConditionalEvent.bool_layers`` names the wrong layer."""

    trace = _log_model(SimpleIfElseModel(), torch.ones(2, 3))
    try:
        event = _get_only_event(trace)
        non_bool_label = next(
            layer.layer_label for layer in trace.layer_list if not layer.is_terminal_bool
        )
        event.bool_layers.append(non_bool_label)
        _assert_invariant_error(trace, ("Invariant 8", non_bool_label))
    finally:
        trace.cleanup()


def test_invariant_9_layerlog_stack_aggregates_match_pass_logs() -> None:
    """Invariant 9 fails when ``Layer`` stack-pass aggregation is corrupted."""

    trace = _log_model(AlternatingRecurrentIfModel(), torch.ones(1, 4))
    try:
        layer_log = _find_multi_pass_linear_layer(trace)
        first_signature = next(iter(layer_log.conditional_branch_stack_ops))
        layer_log.conditional_branch_stack_ops[first_signature] = [999]
        _assert_invariant_error(trace, ("Invariant 9", "conditional_branch_stack_ops"))
    finally:
        trace.cleanup()


def test_invariant_10_conditional_edge_call_indices_reject_invalid_pass() -> None:
    """Invariant 10 fails when an edge-pass entry names an invalid pass."""

    trace = _log_model(AlternatingRecurrentIfModel(), torch.ones(1, 4))
    try:
        edge_key = next(iter(trace.conditional_edge_call_indices))
        existing_ops = trace.conditional_edge_call_indices[edge_key]
        trace.conditional_edge_call_indices[edge_key] = sorted(existing_ops + [0])
        _assert_invariant_error(trace, ("Invariant 10", "pass 0"))
    finally:
        trace.cleanup()


def test_invariant_11_transient_bool_key_is_removed() -> None:
    """Invariant 11 fails when a pass log still carries ``_bool_conditional_key``."""

    trace = _log_model(SimpleIfElseModel(), torch.ones(2, 3))
    try:
        bool_layer = _get_only_terminal_bool(trace)
        bool_layer._bool_conditional_key = ("fake.py", 1, 2, 3)
        _assert_invariant_error(trace, ("Invariant 11", "_bool_conditional_key"))
    finally:
        trace.cleanup()


def test_invariant_12_layerlog_children_union_is_exact() -> None:
    """Invariant 12 fails when ``Layer.conditional_arm_children`` loses a child."""

    trace = _log_model(AlternatingRecurrentIfModel(), torch.ones(1, 4))
    try:
        layer_log = _find_multi_pass_linear_layer(trace)
        only_event = _get_only_event(trace)
        layer_log.conditional_arm_children[only_event.id]["then"] = []
        _sync_layer_log_child_views(layer_log)
        _assert_invariant_error(trace, ("Invariant 1", "conditional_arm_children"))
    finally:
        trace.cleanup()


def test_invariant_13_legacy_if_view_is_bidirectionally_consistent() -> None:
    """Invariant 13 fails when ``conditional_branch_edges`` lacks a node back-reference."""

    trace = _log_model(SimpleIfElseModel(), torch.ones(2, 3))
    try:
        bool_label = _get_only_event(trace).bool_layers[0]
        parent_label = next(
            layer.layer_label
            for layer in trace.layer_list
            if bool_label not in layer.conditional_entry_children
        )
        trace.conditional_branch_edges.append((parent_label, bool_label))
        _assert_invariant_error(trace, ("Invariant 13", "conditional_branch_edges"))
    finally:
        trace.cleanup()
