"""Tests for rerun preserving selective save scope."""

from __future__ import annotations

import torch
from torch import nn

import torchlens as tl


class ThreeOpModel(nn.Module):
    """Small model with several selectable operation sites."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a deterministic three-op computation.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Model output.
        """

        y = x + 1
        z = torch.relu(y)
        return z * 3


def _saved_op_labels(trace: tl.Trace) -> list[str]:
    """Return labels for Ops with saved activations.

    Parameters
    ----------
    trace:
        Trace to inspect.

    Returns
    -------
    list[str]
        Saved Op labels in execution order.
    """

    return [op.label for op in trace.layer_list if op.has_saved_activation]


def test_rerun_preserves_static_layers_to_save_subset() -> None:
    """Rerun does not save every Op after a static selective capture."""

    x = torch.randn(2, 3)
    log = tl.trace(ThreeOpModel(), x, layers_to_save=["relu"], save_arg_values=True)
    saved_before = _saved_op_labels(log)
    num_ops_before = len(log.layer_list)

    log.rerun(ThreeOpModel(), x + 1)

    assert saved_before
    assert _saved_op_labels(log) == saved_before
    assert len(saved_before) < num_ops_before


def test_rerun_preserves_predicate_save_subset() -> None:
    """Rerun keeps predicate-selected saves scoped to matching Ops."""

    x = torch.randn(2, 3)
    log = tl.trace(ThreeOpModel(), x, save=tl.func("relu"))
    saved_before = _saved_op_labels(log)
    num_ops_before = len(log.layer_list)

    log.rerun(ThreeOpModel(), x + 1)

    assert saved_before
    assert _saved_op_labels(log) == saved_before
    assert len(saved_before) < num_ops_before
