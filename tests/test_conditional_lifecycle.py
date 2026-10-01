"""Lifecycle coverage for conditional-label rename, cleanup, and export wiring."""

from collections.abc import Iterator
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

pd = pytest.importorskip("pandas")

import torchlens as tl  # noqa: E402
from torchlens.data_classes.cleanup import _remove_log_entry_references  # noqa: E402
from torchlens.data_classes.trace import ConditionalEvent, Trace  # noqa: E402
from torchlens.postprocess.labeling import (  # noqa: E402
    _rename_model_history_layer_names,
    _replace_layer_names_for_layer_entry,
)


class _StubTrace:
    """Minimal stand-in for lifecycle helper tests."""

    def __init__(
        self,
        layer_list: list[SimpleNamespace] | None = None,
        layer_logs: dict[str, SimpleNamespace] | None = None,
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
        self._tracing_finished = True

        self.input_layers: list[str] = []
        self.output_layers: list[str] = []
        self.buffer_layers: list[str] = []
        self.internal_source_ops: list[str] = []
        self.internal_sink_ops: list[str] = []
        self.internally_terminated_bool_ops: list[str] = []
        self.saved_ops: list[str] = []
        self.saved_grad_ops: list[str] = []
        self._layers_where_internal_branches_merge_with_input: list[str] = []

        self.layers_with_params: dict[str, list[str]] = {}
        self.op_equivalence_classes: dict[str, set] = {}

        self.conditional_branch_edges = []
        self.conditional_arm_entry_edges = {}
        self.conditional_edge_call_indices = {}
        self.conditional_records: list[ConditionalEvent] = []

        self._raw_to_final_layer_labels: dict[str, str] = {}
        self._raw_to_final_parent_layer_labels: dict[str, str] = {}
        self._raw_to_final_op_labels: dict[str, str] = {}
        from torchlens.ir.workspaces import (
            ModuleCaptureWorkspace,
            RawGraphWorkspace,
            WrapperRuntimeWorkspace,
        )

        self._raw_graph_ws = RawGraphWorkspace()
        self._module_capture_ws = ModuleCaptureWorkspace(
            module_build_data={"module_layer_argnames": {}}
        )
        self._wrapper_runtime_ws = WrapperRuntimeWorkspace()

    def __iter__(self) -> Iterator[SimpleNamespace]:
        """Iterate over surviving pass-level entries."""
        return iter(self.layer_list)


class _TinyModel(nn.Module):
    """Small model used to exercise ``to_pandas()`` on a real Trace."""

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


class _ReluThenAdd(nn.Module):
    """Small model with a stable single-pass ``relu`` layer label."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a relu followed by an add.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            ReLU out plus one.
        """

        return torch.relu(x) + 1


def _make_conditional_event(bool_layers: list[str]) -> ConditionalEvent:
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
        call_depth=0,
        parent_conditional_id=None,
        parent_branch_kind=None,
        bool_layers=bool_layers,
    )


def _make_layer_stub(label: str, layer_label: str | None = None) -> SimpleNamespace:
    """Create a minimal layer-like object for rename and cleanup tests.

    Parameters
    ----------
    label:
        Layer label stored on the stub.
    layer_label:
        Pass-stripped label. Defaults to ``layer_label`` with any pass suffix removed.

    Returns
    -------
    SimpleNamespace
        Layer-like object with the fields touched by the lifecycle helpers.
    """
    label_no_pass = layer_label or label.split(":", 1)[0]
    return SimpleNamespace(
        label=label,
        layer_label=label_no_pass,
        _label_raw=label,
        parents=[],
        root_ancestors=[],
        children=[],
        input_ancestors=[],
        output_descendants=[],
        internal_source_parents=[],
        internal_source_ancestors=[],
        conditional_entry_children=[],
        conditional_then_children=[],
        conditional_elif_children={},
        conditional_else_children=[],
        conditional_arm_children={},
        op_equivalence_classes=set(),
        recurrent_ops=[],
        parent_arg_positions={"args": {}, "kwargs": {}},
        out_versions_by_child={},
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
    parent_layer.conditional_entry_children = ["raw_bool"]
    parent_layer.conditional_then_children = ["raw_child_then"]
    parent_layer.conditional_elif_children = {1: ["raw_child_elif"]}
    parent_layer.conditional_else_children = ["raw_child_else"]
    parent_layer.conditional_arm_children = {
        0: {
            "then": ["raw_child_then"],
            "elif_1": ["raw_child_elif"],
            "else": ["raw_child_else"],
        }
    }

    trace = _StubTrace(layer_list=[parent_layer])
    trace._raw_to_final_layer_labels = mapping
    trace._raw_to_final_parent_layer_labels = mapping
    trace._raw_to_final_op_labels = mapping
    trace.conditional_branch_edges = [("raw_bool", "raw_parent")]
    trace.conditional_arm_entry_edges = {
        (0, "then"): [("raw_parent", "raw_child_then")],
        (0, "elif_1"): [("raw_parent", "raw_child_elif")],
        (0, "else"): [("raw_parent", "raw_child_else")],
    }
    trace.conditional_edge_call_indices = {
        ("raw_parent", "raw_child_then", 0, "then"): [1],
        ("raw_parent", "raw_child_elif", 0, "elif_1"): [1],
        ("raw_parent", "raw_child_else", 0, "else"): [1],
    }
    trace.conditional_records = [_make_conditional_event(["raw_bool"])]

    _replace_layer_names_for_layer_entry(trace, parent_layer)
    _rename_model_history_layer_names(trace)

    assert parent_layer.conditional_entry_children == ["gt_1_2"]
    assert parent_layer.conditional_then_children == ["relu_1_3"]
    assert parent_layer.conditional_elif_children == {1: ["sigmoid_1_4"]}
    assert parent_layer.conditional_else_children == ["add_1_5"]
    assert parent_layer.conditional_arm_children == {
        0: {
            "then": ["relu_1_3"],
            "elif_1": ["sigmoid_1_4"],
            "else": ["add_1_5"],
        }
    }
    assert trace.conditional_branch_edges == [("gt_1_2", "linear_1_1")]
    assert trace.conditional_arm_entry_edges == {
        (0, "then"): [("linear_1_1", "relu_1_3")],
        (0, "elif_1"): [("linear_1_1", "sigmoid_1_4")],
        (0, "else"): [("linear_1_1", "add_1_5")],
    }
    assert trace.conditional_edge_call_indices == {
        ("linear_1_1", "relu_1_3", 0, "then"): [1],
        ("linear_1_1", "sigmoid_1_4", 0, "elif_1"): [1],
        ("linear_1_1", "add_1_5", 0, "else"): [1],
    }
    assert trace.conditional_records[0].bool_layers == ["gt_1_2"]


def test_conditional_cleanup_scrubs_removed_labels() -> None:
    """Conditional cleanup removes deleted labels and prunes empty containers."""
    parent_pass = _make_layer_stub("parent:1", "parent")
    parent_pass.conditional_entry_children = ["removed_child:2", "kept_start:1"]
    parent_pass.conditional_then_children = ["removed_child:2", "kept_child:1"]
    parent_pass.conditional_elif_children = {1: ["removed_child:2"]}
    parent_pass.conditional_else_children = ["removed_child:2"]
    parent_pass.conditional_arm_children = {
        0: {
            "then": ["removed_child:2", "kept_child:1"],
            "else": ["removed_child:2"],
        }
    }

    parent_layer = _make_layer_stub("parent", "parent")
    parent_layer.conditional_entry_children = ["removed_child", "kept_start"]
    parent_layer.conditional_then_children = ["removed_child", "kept_child"]
    parent_layer.conditional_elif_children = {1: ["removed_child"]}
    parent_layer.conditional_else_children = ["removed_child"]
    parent_layer.conditional_arm_children = {
        0: {
            "then": ["removed_child", "kept_child"],
            "else": ["removed_child"],
        }
    }
    parent_layer.conditional_branch_stack_ops = {((0, "then"),): [1, 2]}

    trace = _StubTrace(layer_list=[parent_pass], layer_logs={"parent": parent_layer})
    trace.conditional_branch_edges = [
        ("removed_child:2", "parent:1"),
        ("kept_start:1", "parent:1"),
    ]
    trace.conditional_arm_entry_edges = {
        (0, "then"): [("parent:1", "removed_child:2"), ("parent:1", "kept_child:1")],
        (0, "else"): [("parent:1", "removed_child:2")],
    }
    trace.conditional_edge_call_indices = {
        ("parent", "removed_child", 0, "then"): [2],
        ("parent", "kept_child", 0, "then"): [1],
    }
    trace.conditional_records = [_make_conditional_event(["removed_child:2", "kept_bool:1"])]

    _remove_log_entry_references(trace, "removed_child:2")

    assert trace.conditional_branch_edges == [("kept_start:1", "parent:1")]
    assert trace.conditional_arm_entry_edges == {
        (0, "then"): [("parent:1", "kept_child:1")],
    }
    assert trace.conditional_edge_call_indices == {
        ("parent", "kept_child", 0, "then"): [1],
    }
    assert trace.conditional_records[0].bool_layers == ["kept_bool:1"]

    assert parent_pass.conditional_entry_children == ["kept_start:1"]
    assert parent_pass.conditional_then_children == ["kept_child:1"]
    assert parent_pass.conditional_elif_children == {}
    assert parent_pass.conditional_else_children == []
    assert parent_pass.conditional_arm_children == {0: {"then": ["kept_child:1"]}}

    assert parent_layer.conditional_entry_children == ("kept_start",)
    assert parent_layer.conditional_then_children == ("kept_child",)
    assert parent_layer.conditional_elif_children == {}
    assert parent_layer.conditional_else_children == ()
    assert parent_layer.conditional_arm_children == {0: {"then": ["kept_child"]}}
    assert parent_layer.conditional_branch_stack_ops == {((0, "then"),): [1, 2]}


def test_batch_remove_log_entries_accepts_finished_layer_objects() -> None:
    """Batch removal scrubs before clearing a finished trace's aggregate layer."""

    trace = tl.trace(_ReluThenAdd(), torch.tensor([-1.0, 2.0]))
    relu_layer = trace["relu_1_1"]

    trace._batch_remove_log_entries([relu_layer], remove_references=True)

    assert not hasattr(relu_layer, "conditional_entry_children")


def test_to_pandas_exports_conditional_columns() -> None:
    """`to_pandas()` exposes the Phase 3 conditional export columns."""
    trace = tl.trace(
        _TinyModel(),
        torch.ones(1, 3),
        capture=tl.options.CaptureOptions(layers_to_save="all"),
    )
    target_layer = next(
        layer for layer in trace.layer_list if layer.layer_type not in {"input", "output"}
    )
    target_layer.func_config = {"alpha": 1}
    target_layer.is_terminal_conditional_bool = True
    target_layer.conditional_context_kind = "if_test"
    target_layer.conditional_wrapper_kind = "bool_cast"
    target_layer.terminal_conditional_id = 7
    target_layer.conditional_branch_depth = 2
    target_layer.conditional_branch_stack = [(0, "then"), (1, "elif_1")]
    target_layer.conditional_then_children = ["then_child"]
    target_layer.conditional_elif_children = {1: ["elif_child"]}
    target_layer.conditional_else_children = ["else_child"]

    layer_df = trace.to_pandas()

    assert isinstance(layer_df, pd.DataFrame)
    for column_name in [
        "func_config",
        "is_terminal_conditional_bool",
        "conditional_context_kind",
        "conditional_wrapper_kind",
        "terminal_conditional_id",
        "conditional_branch_depth",
        "conditional_branch_stack",
        "conditional_then_children",
        "conditional_elif_children",
        "conditional_else_children",
    ]:
        assert column_name in layer_df.columns

    target_row = layer_df.loc[layer_df["layer_label"] == target_layer.layer_label].iloc[0]
    assert bool(target_row["is_terminal_conditional_bool"]) is True
    assert target_row["conditional_context_kind"] == "if_test"
    assert target_row["conditional_wrapper_kind"] == "bool_cast"
    assert int(target_row["terminal_conditional_id"]) == 7
    assert int(target_row["conditional_branch_depth"]) == 2
    assert target_row["conditional_branch_stack"] == "cond_0:then,cond_1:elif_1"
    assert target_row["conditional_then_children"] == ("then_child",)
    assert target_row["conditional_elif_children"] == {1: ["elif_child"]}
    assert target_row["conditional_else_children"] == ("else_child",)
    assert target_row["func_config"] == {"alpha": 1}


def test_conditional_edge_legacy_aliases_removed() -> None:
    """The legacy conditional edge views are deleted; canonical arm edges stay."""

    trace = Trace("Tiny")
    trace.conditional_arm_entry_edges = {
        (0, "then"): [("parent", "then_child")],
        (0, "elif_1"): [("parent", "elif_child")],
        (0, "else"): [("parent", "else_child")],
    }

    assert not hasattr(type(trace), "conditional_then_entry_edges")
    assert not hasattr(type(trace), "conditional_elif_entry_edges")
    assert not hasattr(type(trace), "conditional_else_entry_edges")
    assert trace.conditional_arm_entry_edges[(0, "then")] == [("parent", "then_child")]
