"""PYTEST_DONT_REWRITE

Integration coverage for the Step 5 conditional 5a-5f pipeline.
"""

from __future__ import annotations

from itertools import chain

import pytest
import torch
import torch.nn as nn

from torchlens import trace as trace_fn
from torchlens._data_classes.trace import Trace
from torchlens._utils._record_group_table import record_member
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
) -> Trace:
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
    Trace
        Postprocessed model log.
    """

    return trace_fn(
        model,
        x,
        capture=CaptureOptions(save_source_context=save_source_context),
    )


# Kept by shrink-r4 (per-id subset rule): sole coverage of _postprocess/control_flow.py.
def test_returned_predicate_remains_a_conditional_consumer() -> None:
    """An output child must not hide a proven tensor-to-host predicate consumer."""

    trace = _log_model(ReturnedPredicateIfElseModel(), torch.ones(2, 2))
    predicate = next(op for op in trace.ops if op.func_name == "__gt__")

    assert len(trace.graph.conditionals) == 1
    assert trace.graph.conditional_branch_edges
    assert predicate.control_flow.is_terminal_bool is True
    assert predicate.control_flow.is_terminal_conditional_bool is True
    assert predicate.label in trace.internally_terminated_bool_ops


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


@pytest.mark.smoke
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
    import torchlens._postprocess as pp
    import torchlens._postprocess.control_flow as cf

    real_fix = pp.control_flow._fix_buffer_layers
    merged: dict[str, str] = {}

    def fix_and_merge(trace: Trace) -> None:
        real_fix(trace)
        raw_dict = trace._raw_graph_ws.raw_layer_dict
        survivor = raw_dict["buffer_1_raw"]
        removed = raw_dict["buffer_2_raw"]
        # Precondition: the step-5 conditional-parent role lives on the node
        # about to be merged away.
        assert removed.control_flow.entry_children
        assert any(
            parent == removed._label_raw for parent, _ in trace.graph.conditional_branch_edges
        )
        merged["survivor"] = survivor._label_raw
        merged["bool_child"] = removed.control_flow.entry_children[0]
        deferred: dict = {}
        cf._merge_buffer_entries(trace, survivor, removed, deferred_removals=deferred)
        cf._finish_deferred_buffer_removals(trace, deferred)

    monkeypatch.setattr(pp.control_flow, "_fix_buffer_layers", fix_and_merge)

    traced = trace_fn(WrittenBufferBranchModel(), torch.ones(2))

    assert merged, "the merge wrapper never ran"
    buffer_branch_edges = [
        (parent, child)
        for parent, child in traced.graph.conditional_branch_edges
        if parent.startswith("buffer")
    ]
    assert buffer_branch_edges, (
        "the surviving buffer lost its step-5 conditional branch edge: "
        f"{traced.graph.conditional_branch_edges}"
    )
    surviving_buffer_labels = {
        op.layer_label for op in traced._layer_list if record_member(op, "is_buffer", False)
    }
    for parent, _child in buffer_branch_edges:
        assert parent in surviving_buffer_labels
    entry_children = list(
        chain.from_iterable(
            traced[label].control_flow.entry_children for label in surviving_buffer_labels
        )
    )
    assert entry_children, "conditional_entry_children did not transfer to the survivor"
