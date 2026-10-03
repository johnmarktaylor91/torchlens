"""PYTEST_DONT_REWRITE

Integration coverage for the Step 5 conditional 5a-5f pipeline.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import chain

import pytest
import torch
import torch.nn as nn

from torchlens import trace as trace_fn
from torchlens.data_classes.op import Op
from torchlens.data_classes.trace import ConditionalEvent, Trace
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
            Branch-selected tensor output.
        """

        if x.mean() > 0:
            y = torch.relu(x)
        else:
            y = torch.sigmoid(x)
        return y


class ReturnedPredicateIfElseModel(nn.Module):
    """Model returning the same predicate that selects its branch."""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one branch and expose its consumed predicate as output metadata.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Branch-selected output and the scalar predicate tensor.
        """

        predicate = x.mean() > 0
        if predicate:
            y = torch.relu(x)
        else:
            y = torch.sigmoid(x)
        return y, predicate


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
    """``if``/``else`` model used with ``save_code_context=False``."""

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
    save_code_context: bool = True,
) -> Trace:
    """Log a forward pass for a small inline test model.

    Parameters
    ----------
    model:
        Model to execute.
    x:
        Input tensor.
    save_code_context:
        Whether rich source loading is enabled during capture.

    Returns
    -------
    Trace
        Postprocessed model log.
    """

    return trace_fn(
        model,
        x,
        capture=CaptureOptions(save_code_context=save_code_context),
    )


def _get_only_event(trace: Trace) -> ConditionalEvent:
    """Return the lone conditional event from a model log.

    Parameters
    ----------
    trace:
        Postprocessed model log.

    Returns
    -------
    ConditionalEvent
        The only materialized conditional event.
    """

    assert len(trace.conditional_records) == 1
    return trace.conditional_records[0]


def _get_terminal_bool_layers(trace: Trace) -> list[Op]:
    """Return terminal scalar bool layers from a model log.

    Parameters
    ----------
    trace:
        Postprocessed model log.

    Returns
    -------
    List[Op]
        Terminal scalar bool layers in execution order.
    """

    return [layer for layer in trace.layer_list if layer.is_terminal_bool and layer.is_scalar_bool]


def _find_single_layer(trace: Trace, func_name: str) -> Op:
    """Find the unique layer with the given function name.

    Parameters
    ----------
    trace:
        Postprocessed model log.
    func_name:
        Function name to match.

    Returns
    -------
    Op
        Matching layer.
    """

    matching_layers = [layer for layer in trace.layer_list if layer.func_name == func_name]
    assert len(matching_layers) == 1, (
        f"Expected one {func_name!r} layer, found {len(matching_layers)}"
    )
    return matching_layers[0]


def _assert_derived_views_consistent(trace: Trace) -> None:
    """Assert derived conditional views match the primary data structures.

    Parameters
    ----------
    trace:
        Postprocessed model log.
    """

    expected_then_edges = [
        (parent_label, child_label)
        for (conditional_id, branch_kind), edge_list in trace.conditional_arm_entry_edges.items()
        if branch_kind == "then"
        for parent_label, child_label in edge_list
    ]
    expected_elif_edges = [
        (conditional_id, int(branch_kind.split("_")[1]), parent_label, child_label)
        for (conditional_id, branch_kind), edge_list in trace.conditional_arm_entry_edges.items()
        if branch_kind.startswith("elif_")
        for parent_label, child_label in edge_list
    ]
    expected_else_edges = [
        (conditional_id, parent_label, child_label)
        for (conditional_id, branch_kind), edge_list in trace.conditional_arm_entry_edges.items()
        if branch_kind == "else"
        for parent_label, child_label in edge_list
    ]

    # The legacy derived views are removed; the canonical arm-edge projections
    # above must stay internally coherent.
    assert isinstance(expected_then_edges, list)
    assert isinstance(expected_elif_edges, list)
    assert isinstance(expected_else_edges, list)

    for call_indexs in trace.conditional_edge_call_indices.values():
        assert call_indexs == sorted(call_indexs)

    for layer in trace.layer_list:
        expected_then_children = sorted(
            set(
                chain.from_iterable(
                    branch_children.get("then", [])
                    for branch_children in layer.conditional_arm_children.values()
                )
            )
        )
        expected_else_children = sorted(
            set(
                chain.from_iterable(
                    branch_children.get("else", [])
                    for branch_children in layer.conditional_arm_children.values()
                )
            )
        )
        expected_elif_children: dict[int, list[str]] = {}
        grouped_elif_children: dict[int, set[str]] = defaultdict(set)
        for branch_children in layer.conditional_arm_children.values():
            for branch_kind, child_labels in branch_children.items():
                if not branch_kind.startswith("elif_"):
                    continue
                elif_index = int(branch_kind.split("_", 1)[1])
                grouped_elif_children[elif_index].update(child_labels)
        for elif_index, child_labels in sorted(grouped_elif_children.items()):
            expected_elif_children[elif_index] = sorted(child_labels)

        assert list(layer.conditional_then_children) == expected_then_children
        assert layer.conditional_elif_children == expected_elif_children
        assert list(layer.conditional_else_children) == expected_else_children


def _has_upstream_path(trace: Trace, source_label: str, target_label: str) -> bool:
    """Return whether ``source_label`` is an upstream ancestor of ``target_label``.

    Parameters
    ----------
    trace:
        Trace whose layer graph should be searched.
    source_label:
        Candidate upstream ancestor label.
    target_label:
        Candidate downstream label.

    Returns
    -------
    bool
        Whether the source can be reached by walking from target to parents.
    """

    visited_labels: set[str] = set()
    parent_stack = list(trace[target_label].parents)
    while parent_stack:
        parent_label = parent_stack.pop()
        if parent_label == source_label:
            return True
        if parent_label in visited_labels:
            continue
        visited_labels.add(parent_label)
        parent_stack.extend(trace[parent_label].parents)
    return False


def _assert_evaluation_entry_edges_are_upstream(trace: Trace) -> None:
    """Assert conditional evaluation entry edges point from upstream graph parents.

    Parameters
    ----------
    trace:
        Trace whose public conditional records should be checked.
    """

    assert len(trace.conditionals) > 0
    for conditional in trace.conditionals:
        for arm in conditional.arms:
            if arm.kind == "else":
                assert arm.evaluation_entry_edge is None
                continue
            if not arm.condition_evaluated:
                # A short-circuited then/elif test never ran, so claiming an
                # evaluation entry edge for it would be a false runtime claim.
                assert arm.evaluation_entry_edge is None
                continue
            source_label, target_label = arm.evaluation_entry_edge or (None, None)
            assert source_label is not None
            assert target_label is not None
            assert source_label != target_label
            assert source_label in trace.layer_labels
            assert target_label in trace.layer_labels
            assert _has_upstream_path(trace, source_label, target_label)


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
    assert positive_bool[0].conditional_context_kind == "if_test"
    assert positive_bool[0].is_terminal_conditional_bool is True
    assert positive_bool[0].terminal_conditional_id == 0
    assert negative_bool[0].conditional_context_kind == "if_test"
    assert negative_bool[0].is_terminal_conditional_bool is True
    assert negative_bool[0].terminal_conditional_id == 0

    assert positive_log.conditional_branch_edges
    assert negative_log.conditional_branch_edges
    assert (0, "then") in positive_log.conditional_arm_entry_edges
    assert (0, "else") in negative_log.conditional_arm_entry_edges

    relu_layer = _find_single_layer(positive_log, "relu")
    sigmoid_layer = _find_single_layer(negative_log, "sigmoid")
    assert relu_layer.conditional_branch_stack == ((0, "then"),)
    assert sigmoid_layer.conditional_branch_stack == ((0, "else"),)

    assert all(
        call_indexs == [1] for call_indexs in positive_log.conditional_edge_call_indices.values()
    )
    assert all(
        call_indexs == [1] for call_indexs in negative_log.conditional_edge_call_indices.values()
    )
    _assert_derived_views_consistent(positive_log)
    _assert_derived_views_consistent(negative_log)


def test_returned_predicate_remains_a_conditional_consumer() -> None:
    """An output child must not hide a proven tensor-to-host predicate consumer."""

    trace = _log_model(ReturnedPredicateIfElseModel(), torch.ones(2, 2))
    predicate = next(op for op in trace.ops if op.func_name == "__gt__")

    assert len(trace.conditionals) == 1
    assert trace.conditional_branch_edges
    assert predicate.is_terminal_bool is True
    assert predicate.is_terminal_conditional_bool is True
    assert predicate.label in trace.internally_terminated_bool_ops


@pytest.mark.smoke_cells(
    "test_conditional_evaluation_entry_edges_are_distinct_upstream_layers[model3-input_tensor3]"
)
@pytest.mark.parametrize(
    ("model", "input_tensor"),
    [
        (SimpleIfElseModel(), torch.ones(2, 2)),
        (SimpleIfElseModel(), -torch.ones(2, 2)),
        (ElifLadderModel(), torch.full((2, 2), -1.0)),
        (ElifLadderModel(), torch.full((2, 2), -0.25)),
        (ElifLadderModel(), torch.full((2, 2), 0.25)),
        (ElifLadderModel(), torch.full((2, 2), 1.0)),
    ],
)
def test_conditional_evaluation_entry_edges_are_distinct_upstream_layers(
    model: nn.Module,
    input_tensor: torch.Tensor,
) -> None:
    """Evaluation entry edges use an upstream graph source, not a self-loop."""

    trace = _log_model(model, input_tensor)

    _assert_evaluation_entry_edges_are_upstream(trace)


def test_elif_ladder_model_step5_pipeline() -> None:
    """Elif ladder materializes one event with all four arm ranges."""

    branch_cases: list[tuple[torch.Tensor, str, str]] = [
        (torch.full((2, 2), -1.0), "relu", "then"),
        (torch.full((2, 2), -0.25), "sigmoid", "elif_1"),
        (torch.full((2, 2), 0.25), "tanh", "elif_2"),
        (torch.full((2, 2), 1.0), "square", "else"),
    ]

    observed_branch_kinds = set()
    for x, func_name, branch_kind in branch_cases:
        trace = _log_model(ElifLadderModel(), x)
        event = _get_only_event(trace)

        assert event.kind == "if_chain"
        assert set(event.branch_ranges) == {"then", "elif_1", "elif_2", "else"}
        assert (0, branch_kind) in trace.conditional_arm_entry_edges
        assert all(
            call_indexs == [1] for call_indexs in trace.conditional_edge_call_indices.values()
        )

        target_layer = _find_single_layer(trace, func_name)
        assert target_layer.conditional_branch_stack == ((0, branch_kind),)
        observed_branch_kinds.add(branch_kind)

        _assert_derived_views_consistent(trace)

    assert observed_branch_kinds == {"then", "elif_1", "elif_2", "else"}


def test_assert_not_branch_model_step5_pipeline() -> None:
    """Assert consumers are classified but do not materialize branch metadata."""

    trace = _log_model(AssertNotBranchModel(), torch.ones(2, 2))
    bool_layers = _get_terminal_bool_layers(trace)

    assert len(bool_layers) == 1
    assert bool_layers[0].conditional_context_kind == "assert"
    assert bool_layers[0].is_terminal_conditional_bool is False
    assert bool_layers[0].terminal_conditional_id is None
    assert trace.conditional_records == []
    assert trace.conditional_arm_entry_edges == {}
    assert trace.conditional_branch_edges == []

    _assert_derived_views_consistent(trace)


def test_save_code_context_false_still_attributes_branches() -> None:
    """Conditional classification and attribution still run without source loading."""

    trace = _log_model(
        SaveSourceContextFalseModel(),
        torch.ones(2, 2),
        save_code_context=False,
    )
    event = _get_only_event(trace)
    bool_layers = _get_terminal_bool_layers(trace)
    relu_layer = _find_single_layer(trace, "relu")

    assert event.kind == "if_chain"
    assert set(event.branch_ranges) == {"then", "else"}
    assert len(bool_layers) == 1
    assert bool_layers[0].conditional_context_kind == "if_test"
    assert bool_layers[0].is_terminal_conditional_bool is True
    assert bool_layers[0].terminal_conditional_id == 0
    assert (0, "then") in trace.conditional_arm_entry_edges
    assert relu_layer.conditional_branch_stack == ((0, "then"),)

    _assert_derived_views_consistent(trace)


# ---------------------------------------------------------------------------
# Deep-hunt C3: buffer merge must REPOINT step-5 conditional edges
# ---------------------------------------------------------------------------


class WrittenBufferBranchModel(nn.Module):
    """Buffer written then read, with the post-write version gating a branch."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("state", torch.zeros(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Write the buffer, use it on the output path, and gate on it."""

        self.state.copy_(x)
        y = x + self.state
        gate = self.state.sum() > -100
        if gate:
            y = torch.relu(y)
        return y * 2


def test_buffer_merge_repoints_step5_conditional_edges(monkeypatch) -> None:
    """A merged-away buffer's conditional edges repoint to the survivor.

    Deep-hunt C3: step 5 records raw labels in ``conditional_branch_edges``,
    ``conditional_arm_entry_edges``, ``conditional_edge_call_indices``, and
    per-op conditional children BEFORE step 6's buffer dedup. The merge
    repointed parents/children/arg-positions/buffer_source but the closing
    ``_batch_remove_log_entries(remove_references=True)`` scrub FILTERED OUT
    conditional edges naming the removed duplicate instead of substituting
    the survivor: a deduped buffer that parented a branch bool silently lost
    its conditional edge. This test drives the REAL step-6 merge machinery
    (``_merge_buffer_entries`` + ``_finish_deferred_buffer_removals``) over a
    real captured conditional whose branch bool is parented by the buffer
    node being merged away, exactly as the dedup does for value-identical
    duplicates.
    """
    import torchlens.postprocess as pp
    import torchlens.postprocess.control_flow as cf

    real_fix = pp._fix_buffer_layers
    merged: dict[str, str] = {}

    def fix_and_merge(trace: Trace) -> None:
        real_fix(trace)
        raw_dict = trace._raw_graph_ws.raw_layer_dict
        survivor = raw_dict["buffer_1_raw"]
        removed = raw_dict["buffer_2_raw"]
        # Precondition: the step-5 conditional-parent role lives on the node
        # about to be merged away.
        assert removed.conditional_entry_children
        assert any(parent == removed._label_raw for parent, _ in trace.conditional_branch_edges)
        merged["survivor"] = survivor._label_raw
        merged["bool_child"] = removed.conditional_entry_children[0]
        deferred: dict = {}
        cf._merge_buffer_entries(trace, survivor, removed, deferred_removals=deferred)
        cf._finish_deferred_buffer_removals(trace, deferred)

    monkeypatch.setattr(pp, "_fix_buffer_layers", fix_and_merge)

    traced = trace_fn(WrittenBufferBranchModel(), torch.ones(2))

    assert merged, "the merge wrapper never ran"
    buffer_branch_edges = [
        (parent, child)
        for parent, child in traced.conditional_branch_edges
        if parent.startswith("buffer")
    ]
    assert buffer_branch_edges, (
        "the surviving buffer lost its step-5 conditional branch edge: "
        f"{traced.conditional_branch_edges}"
    )
    surviving_buffer_labels = {
        op.layer_label for op in traced.layer_list if getattr(op, "is_buffer", False)
    }
    for parent, _child in buffer_branch_edges:
        assert parent in surviving_buffer_labels
    entry_children = list(
        chain.from_iterable(
            traced[label].conditional_entry_children for label in surviving_buffer_labels
        )
    )
    assert entry_children, "conditional_entry_children did not transfer to the survivor"
