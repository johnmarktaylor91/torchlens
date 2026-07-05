"""Golden tests for torch replay-validation decisions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

import torchlens as tl
from torchlens.validation.status import ValidationReplayStatus


GOLDEN_PATH = Path(__file__).with_name("validation_decisions.json")


class TinyFeedForward(nn.Module):
    """Small feed-forward model with representative elementwise and linear ops."""

    def __init__(self) -> None:
        """Initialize the model."""

        super().__init__()
        self.linear = nn.Linear(4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a feed-forward pass.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Model output.
        """

        return torch.relu(self.linear(x)).sum(dim=1)


class TinyBatchNorm(nn.Module):
    """BatchNorm model that exercises registered buffer capture."""

    def __init__(self) -> None:
        """Initialize the model."""

        super().__init__()
        self.bn = nn.BatchNorm1d(4)
        self.bn.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a BatchNorm-backed pass.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Model output.
        """

        return self.bn(x) + self.bn.running_mean


class TinyRecurrent(nn.Module):
    """Small recurrent model with repeated module calls."""

    def __init__(self) -> None:
        """Initialize the model."""

        super().__init__()
        self.cell = nn.GRUCell(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run three recurrent steps.

        Parameters
        ----------
        x:
            Input sequence with shape ``(steps, batch, features)``.

        Returns
        -------
        torch.Tensor
            Final hidden state.
        """

        h = torch.zeros(x.shape[1], 3, dtype=x.dtype)
        for step in range(x.shape[0]):
            h = self.cell(x[step], h)
        return h


class TinyIntervention(nn.Module):
    """Small model used for validating edited traces."""

    def __init__(self) -> None:
        """Initialize the model."""

        super().__init__()
        self.linear = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a pass with an editable ReLU site.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Model output.
        """

        return torch.sigmoid(torch.relu(self.linear(x)))


def _trace_output(trace: Any) -> torch.Tensor:
    """Return the first tensor output saved on a trace.

    Parameters
    ----------
    trace:
        TorchLens trace.

    Returns
    -------
    torch.Tensor
        Detached output tensor.
    """

    return trace[trace.output_layers[0]].out.detach().clone()


def _status_for_trace(trace: Any, outputs: list[torch.Tensor]) -> ValidationReplayStatus:
    """Validate a trace and return its cached status.

    Parameters
    ----------
    trace:
        TorchLens trace to validate.
    outputs:
        Ground-truth output tensors.

    Returns
    -------
    ValidationReplayStatus
        Cached torch validation status.
    """

    result = trace.validate_forward_pass(outputs)
    if isinstance(result, ValidationReplayStatus):
        return result
    return trace.validation_replay_status


def _seeded_status_for_trace(
    seed: int,
    trace: Any,
    outputs: list[torch.Tensor],
) -> ValidationReplayStatus:
    """Validate a trace after pinning perturbation RNG.

    Parameters
    ----------
    seed:
        RNG seed for replay perturbations.
    trace:
        TorchLens trace to validate.
    outputs:
        Ground-truth output tensors.

    Returns
    -------
    ValidationReplayStatus
        Cached torch validation status.
    """

    torch.manual_seed(seed)
    return _status_for_trace(trace, outputs)


def _case_summary(status: ValidationReplayStatus) -> dict[str, Any]:
    """Build the JSON-stable golden payload for one case.

    Parameters
    ----------
    status:
        Validation status to summarize.

    Returns
    -------
    dict[str, Any]
        Golden payload.
    """

    return {
        "state": status.state,
        "replayed_node_count": status.replayed_node_count,
        "unverified_node_count": status.unverified_node_count,
        "failed_node_count": status.failed_node_count,
        "unverified_reason_counts": dict(status.unverified_reason_counts),
        "exempted_reason_counts": dict(status.exempted_reason_counts),
        "decisions": list(status.decisions),
    }


def build_validation_decision_snapshot() -> dict[str, Any]:
    """Build the complete validation decision snapshot.

    Returns
    -------
    dict[str, Any]
        Snapshot keyed by stable case name.
    """

    torch.manual_seed(11)
    ff = TinyFeedForward().eval()
    x_ff = torch.randn(2, 4)
    ff_trace = tl.trace(ff, x_ff, layers_to_save="all", save_arg_values=True)

    torch.manual_seed(12)
    bn = TinyBatchNorm().eval()
    x_bn = torch.randn(3, 4)
    bn_trace = tl.trace(bn, x_bn, layers_to_save="all", save_arg_values=True)

    torch.manual_seed(13)
    recurrent = TinyRecurrent().eval()
    x_recurrent = torch.randn(3, 2, 3)
    recurrent_trace = tl.trace(
        recurrent,
        x_recurrent,
        layers_to_save="all",
        save_arg_values=True,
        random_seed=13,
    )

    torch.manual_seed(14)
    intervention = TinyIntervention().eval()
    x_intervention = torch.randn(2, 3)
    clean = tl.trace(intervention, x_intervention, layers_to_save="all", save_arg_values=True)
    edited = clean.fork("zero_relu")
    relu_pass = next(layer for layer in edited.layer_list if layer.func_name == "relu")
    edited.set(tl.func("relu"), torch.zeros_like(relu_pass.out), confirm_mutation=True)
    edited.rerun(intervention, x_intervention)

    return {
        "tiny_feed_forward": _case_summary(
            _seeded_status_for_trace(101, ff_trace, [_trace_output(ff_trace)])
        ),
        "tiny_batch_norm": _case_summary(
            _seeded_status_for_trace(102, bn_trace, [_trace_output(bn_trace)])
        ),
        "tiny_recurrent": _case_summary(
            _seeded_status_for_trace(103, recurrent_trace, [_trace_output(recurrent_trace)])
        ),
        "tiny_intervention": _case_summary(
            _seeded_status_for_trace(104, edited, [_trace_output(edited)])
        ),
    }


def test_validation_decision_snapshot_matches_golden() -> None:
    """Ensure torch validation decisions stay behavior-locked."""

    expected = json.loads(GOLDEN_PATH.read_text())
    assert build_validation_decision_snapshot() == expected
