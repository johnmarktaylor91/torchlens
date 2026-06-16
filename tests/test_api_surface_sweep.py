"""Tests for the public ``tl.sweep`` convenience constructor."""

from __future__ import annotations

import torch
from torch import nn

import torchlens as tl


class _ReluModel(nn.Module):
    """Small model with a sweepable ReLU site."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a simple nonlinear forward pass.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            ReLU output.
        """

        return torch.relu(x)


def _relu_op(trace: tl.Trace) -> tl.Op:
    """Return the captured ReLU op from a sweep trace.

    Parameters
    ----------
    trace:
        Trace containing the sweep intervention.

    Returns
    -------
    tl.Op
        Captured ReLU operation.
    """

    return next(op for op in trace.layer_list if op.layer_type == "relu")


def test_sweep_returns_bundle_with_one_trace_per_value() -> None:
    """``tl.sweep`` should return one Bundle member per swept value."""

    bundle = tl.sweep(_ReluModel(), torch.tensor([-1.0, 2.0]), param="relu", values=[0.0, 3.0])

    assert isinstance(bundle, tl.Bundle)
    assert len(bundle) == 2
    assert bundle.names == ["sweep_0", "sweep_1"]


def test_sweep_applies_each_replacement_value() -> None:
    """Each swept trace should include the replacement applied at the target site."""

    bundle = tl.sweep(_ReluModel(), torch.tensor([-1.0, 2.0]), param="relu", values=[0.0, 3.0])

    first = bundle["sweep_0"]
    second = bundle["sweep_1"]
    torch.testing.assert_close(first.reconstruct_output(), torch.tensor([0.0, 0.0]))
    torch.testing.assert_close(second.reconstruct_output(), torch.tensor([3.0, 3.0]))
    assert _relu_op(first).intervention_replaced
    assert _relu_op(second).intervention_replaced


def test_sweep_accepts_selector_and_custom_names() -> None:
    """Selectors and explicit Bundle names should be forwarded."""

    bundle = tl.sweep(
        _ReluModel(),
        torch.tensor([-1.0, 2.0]),
        param=tl.func("relu"),
        values=[torch.tensor([4.0, 5.0])],
        names=["relu_value"],
    )

    assert bundle.names == ["relu_value"]
    torch.testing.assert_close(bundle["relu_value"].reconstruct_output(), torch.tensor([4.0, 5.0]))
