"""PYTEST_DONT_REWRITE

Integration coverage for the Step 5 conditional 5a-5f pipeline.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import chain
from typing import Dict, List, Tuple

import pytest
import torch
import torch.nn as nn

from torchlens import log_forward_pass
from torchlens.data_classes.layer_pass_log import LayerPassLog
from torchlens.data_classes.model_log import ConditionalEvent, ModelLog


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
            Branch-selected tensor output.
        """

        if x.mean() > 0:
            y = torch.relu(x)
        else:
            y = torch.sigmoid(x)
        return y


class ElifLadderModel(nn.Module):
    """Model with a flattened ``if``/``elif``/``elif``/``else`` ladder."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model forward pass.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Branch-selected tensor output.
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


class AssertNotBranchModel(nn.Module):
    """Model whose bool is consumed by ``assert`` instead of control flow."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model forward pass.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Activated output tensor.
        """

        assert x.mean() > 0
        y = torch.relu(x)
        return y


class SaveSourceContextFalseModel(nn.Module):
    """``if``/``else`` model used with ``save_source_context=False``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model forward pass.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Branch-selected tensor output.
        """

        if x.mean() > 0:
            y = torch.relu(x)
        else:
            y = torch.sigmoid(x)
        return y


def _log_model(
    model: nn.Module,
    x: torch.Tensor,
    save_source_context: bool = True,
) -> ModelLog:
    """Log a forward pass for a small inline test model.

    Parameters
    ----------
    model:
        Model to execute.
    x:
        Input tensor.
    save_source_context:
        Whether rich source loading is enabled during capture.

    Returns
    -------
    ModelLog
        Postprocessed model log.
    """

    return log_forward_pass(model, x, save_source_context=save_source_context)


def _get_only_event(model_log: ModelLog) -> ConditionalEvent:
    """Return the lone conditional event from a model log.

    Parameters
    ----------
    model_log:
        Postprocessed model log.

    Returns
    -------
    ConditionalEvent
        The only materialized conditional event.
    """

    assert len(model_log.conditional_events) == 1
    return model_log.conditional_events[0]


def _get_terminal_bool_layers(model_log: ModelLog) -> List[LayerPassLog]:
    """Return terminal scalar bool layers from a model log.

    Parameters
    ----------
    model_log:
        Postprocessed model log.

    Returns
    -------
    List[LayerPassLog]
        Terminal scalar bool layers in execution order.
    """

    return [
        layer
        for layer in model_log.layer_list
        if layer.is_terminal_bool_layer and layer.is_scalar_bool
    ]


def _find_single_layer(model_log: ModelLog, func_name: str) -> LayerPassLog:
    """Find the unique layer with the given function name.

    Parameters
    ----------
    model_log:
        Postprocessed model log.
    func_name:
        Function name to match.

    Returns
    -------
    LayerPassLog
        Matching layer.
    """

    matching_layers = [layer for layer in model_log.layer_list if layer.func_name == func_name]
    assert len(matching_layers) == 1, (
        f"Expected one {func_name!r} layer, found {len(matching_layers)}"
    )
    return matching_layers[0]


def _assert_derived_views_consistent(model_log: ModelLog) -> None:
    """Assert derived conditional views match the primary data structures.

    Parameters
    ----------
    model_log:
        Postprocessed model log.
    """

    expected_then_edges = [
        (parent_label, child_label)
        for (conditional_id, branch_kind), edge_list in model_log.conditional_arm_edges.items()
        if branch_kind == "then"
        for parent_label, child_label in edge_list
    ]
    expected_elif_edges = [
        (conditional_id, int(branch_kind.split("_")[1]), parent_label, child_label)
        for (conditional_id, branch_kind), edge_list in model_log.conditional_arm_edges.items()
        if branch_kind.startswith("elif_")
        for parent_label, child_label in edge_list
    ]
    expected_else_edges = [
        (conditional_id, parent_label, child_label)
        for (conditional_id, branch_kind), edge_list in model_log.conditional_arm_edges.items()
        if branch_kind == "else"
        for parent_label, child_label in edge_list
    ]

    assert model_log.conditional_then_edges == expected_then_edges
    assert model_log.conditional_elif_edges == expected_elif_edges
    assert model_log.conditional_else_edges == expected_else_edges

    for pass_nums in model_log.conditional_edge_passes.values():
        assert pass_nums == sorted(pass_nums)

    for layer in model_log.layer_list:
        expected_then_children = sorted(
            set(
                chain.from_iterable(
                    branch_children.get("then", [])
                    for branch_children in layer.cond_branch_children_by_cond.values()
                )
            )
        )
        expected_else_children = sorted(
            set(
                chain.from_iterable(
                    branch_children.get("else", [])
                    for branch_children in layer.cond_branch_children_by_cond.values()
                )
            )
        )
        expected_elif_children: Dict[int, List[str]] = {}
        grouped_elif_children: Dict[int, set[str]] = defaultdict(set)
        for branch_children in layer.cond_branch_children_by_cond.values():
            for branch_kind, child_labels in branch_children.items():
                if not branch_kind.startswith("elif_"):
                    continue
                elif_index = int(branch_kind.split("_", 1)[1])
                grouped_elif_children[elif_index].update(child_labels)
        for elif_index, child_labels in sorted(grouped_elif_children.items()):
            expected_elif_children[elif_index] = sorted(child_labels)

        assert layer.cond_branch_then_children == expected_then_children
        assert layer.cond_branch_elif_children == expected_elif_children
        assert layer.cond_branch_else_children == expected_else_children


@pytest.mark.smoke
def test_simple_if_else_model_step5_pipeline() -> None:
    """Simple ``if``/``else`` logs events plus THEN and ELSE arm attribution."""

    positive_log = _log_model(SimpleIfElseModel(), torch.ones(2, 2))
    negative_log = _log_model(SimpleIfElseModel(), -torch.ones(2, 2))

    positive_event = _get_only_event(positive_log)
    negative_event = _get_only_event(negative_log)

    assert positive_event.kind == "if_chain"
    assert negative_event.kind == "if_chain"
    assert set(positive_event.branch_ranges) == {"then", "else"}
    assert set(negative_event.branch_ranges) == {"then", "else"}

    positive_bool = _get_terminal_bool_layers(positive_log)
    negative_bool = _get_terminal_bool_layers(negative_log)
    assert len(positive_bool) == 1
    assert len(negative_bool) == 1
    assert positive_bool[0].bool_context_kind == "if_test"
    assert positive_bool[0].bool_is_branch is True
    assert positive_bool[0].bool_conditional_id == 0
    assert negative_bool[0].bool_context_kind == "if_test"
    assert negative_bool[0].bool_is_branch is True
    assert negative_bool[0].bool_conditional_id == 0

    assert positive_log.conditional_branch_edges
    assert negative_log.conditional_branch_edges
    assert (0, "then") in positive_log.conditional_arm_edges
    assert (0, "else") in negative_log.conditional_arm_edges

    relu_layer = _find_single_layer(positive_log, "relu")
    sigmoid_layer = _find_single_layer(negative_log, "sigmoid")
    assert relu_layer.conditional_branch_stack == [(0, "then")]
    assert sigmoid_layer.conditional_branch_stack == [(0, "else")]

    assert all(pass_nums == [1] for pass_nums in positive_log.conditional_edge_passes.values())
    assert all(pass_nums == [1] for pass_nums in negative_log.conditional_edge_passes.values())

    _assert_derived_views_consistent(positive_log)
    _assert_derived_views_consistent(negative_log)


@pytest.mark.smoke
def test_elif_ladder_model_step5_pipeline() -> None:
    """Elif ladder materializes one event with all four arm ranges."""

    branch_cases: List[Tuple[torch.Tensor, str, str]] = [
        (torch.full((2, 2), -1.0), "relu", "then"),
        (torch.full((2, 2), -0.25), "sigmoid", "elif_1"),
        (torch.full((2, 2), 0.25), "tanh", "elif_2"),
        (torch.full((2, 2), 1.0), "square", "else"),
    ]

    observed_branch_kinds = set()
    for x, func_name, branch_kind in branch_cases:
        model_log = _log_model(ElifLadderModel(), x)
        event = _get_only_event(model_log)

        assert event.kind == "if_chain"
        assert set(event.branch_ranges) == {"then", "elif_1", "elif_2", "else"}
        assert (0, branch_kind) in model_log.conditional_arm_edges
        assert all(pass_nums == [1] for pass_nums in model_log.conditional_edge_passes.values())

        target_layer = _find_single_layer(model_log, func_name)
        assert target_layer.conditional_branch_stack == [(0, branch_kind)]
        observed_branch_kinds.add(branch_kind)

        _assert_derived_views_consistent(model_log)

    assert observed_branch_kinds == {"then", "elif_1", "elif_2", "else"}


def test_assert_not_branch_model_step5_pipeline() -> None:
    """Assert consumers are classified but do not materialize branch metadata."""

    model_log = _log_model(AssertNotBranchModel(), torch.ones(2, 2))
    bool_layers = _get_terminal_bool_layers(model_log)

    assert len(bool_layers) == 1
    assert bool_layers[0].bool_context_kind == "assert"
    assert bool_layers[0].bool_is_branch is False
    assert bool_layers[0].bool_conditional_id is None
    assert model_log.conditional_events == []
    assert model_log.conditional_arm_edges == {}
    assert model_log.conditional_branch_edges == []

    _assert_derived_views_consistent(model_log)


def test_save_source_context_false_still_attributes_branches() -> None:
    """Conditional classification and attribution still run without source loading."""

    model_log = _log_model(
        SaveSourceContextFalseModel(),
        torch.ones(2, 2),
        save_source_context=False,
    )
    event = _get_only_event(model_log)
    bool_layers = _get_terminal_bool_layers(model_log)
    relu_layer = _find_single_layer(model_log, "relu")

    assert event.kind == "if_chain"
    assert set(event.branch_ranges) == {"then", "else"}
    assert len(bool_layers) == 1
    assert bool_layers[0].bool_context_kind == "if_test"
    assert bool_layers[0].bool_is_branch is True
    assert bool_layers[0].bool_conditional_id == 0
    assert (0, "then") in model_log.conditional_arm_edges
    assert relu_layer.conditional_branch_stack == [(0, "then")]

    _assert_derived_views_consistent(model_log)
