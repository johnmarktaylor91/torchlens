"""Append rerun state semantics regression tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.intervention.errors import AppendBatchDependenceError


class _AppendSemanticsModel(nn.Module):
    """Small model for append state semantics."""

    def __init__(self) -> None:
        """Initialize the linear layer."""

        super().__init__()
        self.linear = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a linear layer followed by relu.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Relu output.
        """

        return torch.relu(self.linear(x))


def _first_batch_out(trace: tl.Trace) -> torch.Tensor:
    """Return the first saved output with a batch dimension.

    Parameters
    ----------
    trace:
        Trace to inspect.

    Returns
    -------
    torch.Tensor
        First batched saved output.
    """

    for layer in trace.layer_list:
        if isinstance(layer.out, torch.Tensor) and layer.out.ndim > 0:
            return layer.out
    raise AssertionError("no batched output found")


def test_is_appended_resets_on_non_append_rerun() -> None:
    """A replacement rerun clears previous append state."""

    model = _AppendSemanticsModel().eval()
    trace = tl.trace(model, torch.randn(2, 3), intervention_ready=True)
    trace.rerun(model, torch.randn(1, 3), append=True)

    assert trace.is_appended is True
    assert trace._append_sequence_id == 1

    trace.rerun(model, torch.randn(4, 3))

    assert trace.is_appended is False
    assert trace._append_sequence_id == 0
    assert trace.append_history == []
    assert _first_batch_out(trace).shape[0] == 4


def test_repeated_appends_increment_sequence_id_and_history() -> None:
    """Repeated append reruns increment sequence ids and keep provenance."""

    model = _AppendSemanticsModel().eval()
    trace = tl.trace(model, torch.randn(2, 3), intervention_ready=True)

    trace.rerun(model, torch.randn(1, 3), append=True)
    trace.rerun(model, torch.randn(3, 3), append=True)

    assert trace.is_appended is True
    assert trace._append_sequence_id == 2
    assert [record["append_sequence_id"] for record in trace.append_history] == [1, 2]
    assert [record["chunk_size"] for record in trace.append_history] == [1, 3]
    assert _first_batch_out(trace).shape[0] == 6


def test_intervention_save_append_state_consistency(tmp_path: Path) -> None:
    """Saved intervention metadata mirrors in-memory append state."""

    model = _AppendSemanticsModel().eval()
    trace = tl.trace(model, torch.randn(2, 3), intervention_ready=True)
    trace.rerun(model, torch.randn(1, 3), append=True)

    spec_path = tmp_path / "append_state.tlspec"
    trace.save_intervention(spec_path, level="audit")
    spec = tl.load_intervention_spec(spec_path)
    append_state = spec.metadata["append_state"]

    assert append_state["is_appended"] is True
    assert append_state["append_sequence_id"] == 1
    assert append_state["append_history"][0]["chunk_size"] == 1
    assert append_state["state_history"][-1]["op"] == "append"


def test_append_train_mode_no_helper_grad_message() -> None:
    """Grad append without helper opt-in raises an actionable error."""

    model = _AppendSemanticsModel()
    trace = tl.trace(
        model,
        torch.randn(2, 3),
        intervention_ready=True,
        save_grads=True,
        backward_ready=True,
    )
    trace.log_backward(trace[trace.output_layers[0]].out.sum(), retain_graph=True)

    with pytest.raises(AppendBatchDependenceError, match="batch-independent helper"):
        trace.rerun(model, torch.randn(1, 3), append=True)
