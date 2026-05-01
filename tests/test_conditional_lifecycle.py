"""Lifecycle coverage for conditional-label rename, cleanup, and export wiring."""

from types import SimpleNamespace
from typing import Dict, Iterator, List, Optional

import pandas as pd
import pytest
import torch
import torch.nn as nn

from torchlens import log_forward_pass
from torchlens._deprecations import _WARNED_DEPRECATIONS
from torchlens.data_classes.cleanup import _remove_log_entry_references
from torchlens.data_classes.model_log import ConditionalEvent, ModelLog
from torchlens.postprocess.labeling import (
    _rename_model_history_layer_names,
    _replace_layer_names_for_layer_entry,
)


class _StubModelLog:
    """Minimal stand-in for lifecycle helper tests."""

    def __init__(
        self,
        layer_list: Optional[List[SimpleNamespace]] = None,
        layer_logs: Optional[Dict[str, SimpleNamespace]] = None,
    ) -> None:
        """Initialize the stub log.

        Parameters
        ----------
        layer_list:
            Surviving pass-level layer entries.
        layer_logs:
            Aggregate no-pass layer entries.
        """
        self.layer_list = layer_list or []
        self.layer_logs = layer_logs or {}
        self._pass_finished = True

        self.input_layers: List[str] = []
        self.output_layers: List[str] = []
        self.buffer_layers: List[str] = []
        self.internally_initialized_layers: List[str] = []
        self.internally_terminated_layers: List[str] = []
        self.internally_terminated_bool_layers: List[str] = []
        self.layers_with_saved_activations: List[str] = []
        self.layers_with_saved_gradients: List[str] = []
        self._layers_where_internal_branches_merge_with_input: List[str] = []

        self.layers_with_params: Dict[str, List[str]] = {}
        self.equivalent_operations: Dict[str, set] = {}

        self.conditional_branch_edges = []
        self.conditional_then_edges = []
        self.conditional_elif_edges = []
        self.conditional_else_edges = []
        self.conditional_arm_edges = {}
        self.conditional_edge_passes = {}
        self.conditional_events: List[ConditionalEvent] = []

        self._raw_to_final_layer_labels: Dict[str, str] = {}
        self._module_build_data = {"module_layer_argnames": {}}

    def __iter__(self) -> Iterator[SimpleNamespace]:
        """Iterate over surviving pass-level entries."""
        return iter(self.layer_list)


class _TinyModel(nn.Module):
    """Small model used to exercise ``to_pandas()`` on a real ModelLog."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a minimal forward pass.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Activated output tensor.
        """
        return torch.relu(x + 1)


def _make_conditional_event(bool_layers: List[str]) -> ConditionalEvent:
    """Build a small ``ConditionalEvent`` fixture.

    Parameters
    ----------
    bool_layers:
        Bool-layer labels referenced by the event.

    Returns
    -------
    ConditionalEvent
        Event with stable dummy metadata.
    """
    return ConditionalEvent(
        id=0,
        kind="if_chain",
        source_file="test_file.py",
        function_qualname="Tiny.forward",
        function_span=(1, 10),
        if_stmt_span=(3, 6),
        test_span=(3, 0, 3, 8),
        branch_ranges={"then": (3, 9, 4, 12), "else": (5, 9, 6, 12)},
        branch_test_spans={"then": (3, 0, 3, 8)},
        nesting_depth=0,
        parent_conditional_id=None,
        parent_branch_kind=None,
        bool_layers=bool_layers,
    )


def _make_layer_stub(
    layer_label: str, layer_label_no_pass: Optional[str] = None
) -> SimpleNamespace:
    """Create a minimal layer-like object for rename and cleanup tests.

    Parameters
    ----------
    layer_label:
        Layer label stored on the stub.
    layer_label_no_pass:
        Pass-stripped label. Defaults to ``layer_label`` with any pass suffix removed.

    Returns
    -------
    SimpleNamespace
        Layer-like object with the fields touched by the lifecycle helpers.
    """
    label_no_pass = layer_label_no_pass or layer_label.split(":", 1)[0]
    return SimpleNamespace(
        layer_label=layer_label,
        layer_label_no_pass=label_no_pass,
        tensor_label_raw=layer_label,
        parent_layers=[],
        root_ancestors=[],
        child_layers=[],
        input_ancestors=[],
        output_descendants=[],
        internally_initialized_parents=[],
        internally_initialized_ancestors=[],
        cond_branch_start_children=[],
        cond_branch_then_children=[],
        cond_branch_elif_children={},
        cond_branch_else_children=[],
        cond_branch_children_by_cond={},
        equivalent_operations=set(),
        recurrent_group=[],
        parent_layer_arg_locs={"args": {}, "kwargs": {}},
        children_tensor_versions={},
    )


def test_conditional_labels_rename_across_lifecycle_surfaces() -> None:
    """Step 11 rename rewrites every new conditional-label surface."""
    mapping = {
        "raw_parent": "linear_1_1",
        "raw_bool": "gt_1_2",
        "raw_child_then": "relu_1_3",
        "raw_child_elif": "sigmoid_1_4",
        "raw_child_else": "add_1_5",
    }
    parent_layer = _make_layer_stub("raw_parent")
    parent_layer.cond_branch_start_children = ["raw_bool"]
    parent_layer.cond_branch_then_children = ["raw_child_then"]
    parent_layer.cond_branch_elif_children = {1: ["raw_child_elif"]}
    parent_layer.cond_branch_else_children = ["raw_child_else"]
    parent_layer.cond_branch_children_by_cond = {
        0: {
            "then": ["raw_child_then"],
            "elif_1": ["raw_child_elif"],
            "else": ["raw_child_else"],
        }
    }

    model_log = _StubModelLog(layer_list=[parent_layer])
    model_log._raw_to_final_layer_labels = mapping
    model_log.conditional_branch_edges = [("raw_bool", "raw_parent")]
    model_log.conditional_then_edges = [("raw_parent", "raw_child_then")]
    model_log.conditional_elif_edges = [(0, 1, "raw_parent", "raw_child_elif")]
    model_log.conditional_else_edges = [(0, "raw_parent", "raw_child_else")]
    model_log.conditional_arm_edges = {
        (0, "then"): [("raw_parent", "raw_child_then")],
        (0, "elif_1"): [("raw_parent", "raw_child_elif")],
        (0, "else"): [("raw_parent", "raw_child_else")],
    }
    model_log.conditional_edge_passes = {
        ("raw_parent", "raw_child_then", 0, "then"): [1],
        ("raw_parent", "raw_child_elif", 0, "elif_1"): [1],
        ("raw_parent", "raw_child_else", 0, "else"): [1],
    }
    model_log.conditional_events = [_make_conditional_event(["raw_bool"])]

    _replace_layer_names_for_layer_entry(model_log, parent_layer)
    _rename_model_history_layer_names(model_log)

    assert parent_layer.cond_branch_start_children == ["gt_1_2"]
    assert parent_layer.cond_branch_then_children == ["relu_1_3"]
    assert parent_layer.cond_branch_elif_children == {1: ["sigmoid_1_4"]}
    assert parent_layer.cond_branch_else_children == ["add_1_5"]
    assert parent_layer.cond_branch_children_by_cond == {
        0: {
            "then": ["relu_1_3"],
            "elif_1": ["sigmoid_1_4"],
            "else": ["add_1_5"],
        }
    }
    assert model_log.conditional_branch_edges == [("gt_1_2", "linear_1_1")]
    assert model_log.conditional_arm_edges == {
        (0, "then"): [("linear_1_1", "relu_1_3")],
        (0, "elif_1"): [("linear_1_1", "sigmoid_1_4")],
        (0, "else"): [("linear_1_1", "add_1_5")],
    }
    assert model_log.conditional_edge_passes == {
        ("linear_1_1", "relu_1_3", 0, "then"): [1],
        ("linear_1_1", "sigmoid_1_4", 0, "elif_1"): [1],
        ("linear_1_1", "add_1_5", 0, "else"): [1],
    }
    assert model_log.conditional_events[0].bool_layers == ["gt_1_2"]


def test_conditional_cleanup_scrubs_removed_labels() -> None:
    """Conditional cleanup removes deleted labels and prunes empty containers."""
    parent_pass = _make_layer_stub("parent:1", "parent")
    parent_pass.cond_branch_start_children = ["removed_child:2", "kept_start:1"]
    parent_pass.cond_branch_then_children = ["removed_child:2", "kept_child:1"]
    parent_pass.cond_branch_elif_children = {1: ["removed_child:2"]}
    parent_pass.cond_branch_else_children = ["removed_child:2"]
    parent_pass.cond_branch_children_by_cond = {
        0: {
            "then": ["removed_child:2", "kept_child:1"],
            "else": ["removed_child:2"],
        }
    }

    parent_layer = _make_layer_stub("parent", "parent")
    parent_layer.cond_branch_start_children = ["removed_child", "kept_start"]
    parent_layer.cond_branch_then_children = ["removed_child", "kept_child"]
    parent_layer.cond_branch_elif_children = {1: ["removed_child"]}
    parent_layer.cond_branch_else_children = ["removed_child"]
    parent_layer.cond_branch_children_by_cond = {
        0: {
            "then": ["removed_child", "kept_child"],
            "else": ["removed_child"],
        }
    }
    parent_layer.conditional_branch_stack_passes = {((0, "then"),): [1, 2]}

    model_log = _StubModelLog(layer_list=[parent_pass], layer_logs={"parent": parent_layer})
    model_log.conditional_branch_edges = [
        ("removed_child:2", "parent:1"),
        ("kept_start:1", "parent:1"),
    ]
    model_log.conditional_then_edges = [
        ("parent:1", "removed_child:2"),
        ("parent:1", "kept_child:1"),
    ]
    model_log.conditional_elif_edges = [(0, 1, "parent:1", "removed_child:2")]
    model_log.conditional_else_edges = [(0, "parent:1", "removed_child:2")]
    model_log.conditional_arm_edges = {
        (0, "then"): [("parent:1", "removed_child:2"), ("parent:1", "kept_child:1")],
        (0, "else"): [("parent:1", "removed_child:2")],
    }
    model_log.conditional_edge_passes = {
        ("parent", "removed_child", 0, "then"): [2],
        ("parent", "kept_child", 0, "then"): [1],
    }
    model_log.conditional_events = [_make_conditional_event(["removed_child:2", "kept_bool:1"])]

    _remove_log_entry_references(model_log, "removed_child:2")

    assert model_log.conditional_branch_edges == [("kept_start:1", "parent:1")]
    assert model_log.conditional_arm_edges == {
        (0, "then"): [("parent:1", "kept_child:1")],
    }
    assert model_log.conditional_edge_passes == {
        ("parent", "kept_child", 0, "then"): [1],
    }
    assert model_log.conditional_events[0].bool_layers == ["kept_bool:1"]

    assert parent_pass.cond_branch_start_children == ["kept_start:1"]
    assert parent_pass.cond_branch_then_children == ["kept_child:1"]
    assert parent_pass.cond_branch_elif_children == {}
    assert parent_pass.cond_branch_else_children == []
    assert parent_pass.cond_branch_children_by_cond == {0: {"then": ["kept_child:1"]}}

    assert parent_layer.cond_branch_start_children == ["kept_start"]
    assert parent_layer.cond_branch_then_children == ["kept_child"]
    assert parent_layer.cond_branch_elif_children == {}
    assert parent_layer.cond_branch_else_children == []
    assert parent_layer.cond_branch_children_by_cond == {0: {"then": ["kept_child"]}}
    assert parent_layer.conditional_branch_stack_passes == {((0, "then"),): [1, 2]}


def test_to_pandas_exports_conditional_columns() -> None:
    """`to_pandas()` exposes the Phase 3 conditional export columns."""
    model_log = log_forward_pass(_TinyModel(), torch.ones(1, 3), layers_to_save="all")
    target_layer = next(
        layer for layer in model_log.layer_list if layer.layer_type not in {"input", "output"}
    )
    target_layer.func_config = {"alpha": 1}
    target_layer.bool_is_branch = True
    target_layer.bool_context_kind = "if_test"
    target_layer.bool_wrapper_kind = "bool_cast"
    target_layer.bool_conditional_id = 7
    target_layer.conditional_branch_depth = 2
    target_layer.conditional_branch_stack = [(0, "then"), (1, "elif_1")]
    target_layer.cond_branch_then_children = ["then_child"]
    target_layer.cond_branch_elif_children = {1: ["elif_child"]}
    target_layer.cond_branch_else_children = ["else_child"]

    layer_df = model_log.to_pandas()

    assert isinstance(layer_df, pd.DataFrame)
    for column_name in [
        "func_config",
        "bool_is_branch",
        "bool_context_kind",
        "bool_wrapper_kind",
        "bool_conditional_id",
        "conditional_branch_depth",
        "conditional_branch_stack",
        "cond_branch_then_children",
        "cond_branch_elif_children",
        "cond_branch_else_children",
    ]:
        assert column_name in layer_df.columns

    target_row = layer_df.loc[layer_df["layer_label"] == target_layer.layer_label].iloc[0]
    assert bool(target_row["bool_is_branch"]) is True
    assert target_row["bool_context_kind"] == "if_test"
    assert target_row["bool_wrapper_kind"] == "bool_cast"
    assert int(target_row["bool_conditional_id"]) == 7
    assert int(target_row["conditional_branch_depth"]) == 2
    assert target_row["conditional_branch_stack"] == "cond_0:then,cond_1:elif_1"
    assert target_row["cond_branch_then_children"] == ["then_child"]
    assert target_row["cond_branch_elif_children"] == {1: ["elif_child"]}
    assert target_row["cond_branch_else_children"] == ["else_child"]
    assert target_row["func_config"] == {"alpha": 1}


def test_conditional_edge_legacy_aliases_warn() -> None:
    """Legacy conditional edge views warn while projecting canonical arm edges."""

    model_log = ModelLog("Tiny")
    _WARNED_DEPRECATIONS.clear()
    model_log.conditional_arm_edges = {
        (0, "then"): [("parent", "then_child")],
        (0, "elif_1"): [("parent", "elif_child")],
        (0, "else"): [("parent", "else_child")],
    }

    with pytest.warns(DeprecationWarning):
        assert model_log.conditional_then_edges == [("parent", "then_child")]
    with pytest.warns(DeprecationWarning):
        assert model_log.conditional_elif_edges == [(0, 1, "parent", "elif_child")]
    with pytest.warns(DeprecationWarning):
        assert model_log.conditional_else_edges == [(0, "parent", "else_child")]
