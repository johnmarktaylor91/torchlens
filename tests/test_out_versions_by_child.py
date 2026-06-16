"""Regression tests for fast-pass child-version snapshots."""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl


class FastPassAliasMutationModel(nn.Module):
    """Model whose in-place alias mutation requires a per-child parent version."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass with an unsaved parent consumed before mutation.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Mutated parent multiplied downstream.
        """

        parent = x + 1
        view = parent.view_as(parent)
        view.add_(2)
        return parent * 3


def test_fast_selective_save_rebuilds_out_versions_for_child_lookup() -> None:
    """Fast selective save repopulates child versions after clearing stale payloads."""

    model = FastPassAliasMutationModel()
    x = torch.ones(2, 4)

    trace = tl.trace(model, x, layers_to_save=["mul"], save_arg_values=True)

    parent = trace["add_1_1"]
    assert trace.capture_mode == "fast"
    assert trace._replay_arg_version_data_complete
    assert not parent.has_saved_activation
    assert parent.has_out_variations
    assert torch.equal(
        parent.out_versions_by_child["viewas_1_2"],
        torch.full((2, 4), 2.0),
    )


def test_fast_selective_save_without_arg_values_reports_versions_incomplete() -> None:
    """Fast selective save without arg snapshots leaves consumers explicitly blocked."""

    model = FastPassAliasMutationModel()
    x = torch.ones(2, 4)

    trace = tl.trace(model, x, layers_to_save=["mul"], save_arg_values=False)

    assert trace.capture_mode == "fast"
    assert not trace._replay_arg_version_data_complete
    assert all(not op.out_versions_by_child for op in trace.layer_list)
    with pytest.raises(ValueError, match="child-version snapshots"):
        trace.validate_forward_pass([model(x).detach().clone()], validate_metadata=False)
