"""Regression tests for selective-save child-version snapshots."""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.backends.torch.aliasing import detect_torch_alias_contract
from torchlens.ir.intervention import FunctionEventInput


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


def test_selective_save_rebuilds_out_versions_for_child_lookup() -> None:
    """Selective save repopulates child versions after clearing stale payloads."""

    model = FastPassAliasMutationModel()
    x = torch.ones(2, 4)

    trace = tl.trace(
        model,
        x,
        capture=tl.options.CaptureOptions(layers_to_save=["mul"], save_arg_values=True),
    )

    parent = trace["add_1_1"]
    assert trace._replay_arg_version_data_complete
    assert not parent.has_saved_activation
    assert parent.has_out_variations
    assert torch.equal(
        parent.out_versions_by_child["viewas_1_2"],
        torch.full((2, 4), 2.0),
    )


def test_selective_save_without_arg_values_reports_versions_incomplete() -> None:
    """Selective save without arg snapshots leaves consumers explicitly blocked."""

    model = FastPassAliasMutationModel()
    x = torch.ones(2, 4)

    trace = tl.trace(
        model,
        x,
        capture=tl.options.CaptureOptions(layers_to_save=["mul"], save_arg_values=False),
    )

    assert not trace._replay_arg_version_data_complete
    assert all(not op.out_versions_by_child for op in trace.layer_list)
    with pytest.raises(ValueError, match="child-version snapshots"):
        trace.validate_forward_pass([model(x).detach().clone()], validate_metadata=False)


def test_alias_contract_output_traversal_matches_nested_input_shapes() -> None:
    """Output alias traversal follows the same nested container shapes as input traversal."""

    x = torch.randn(2, 3)
    semantics = detect_torch_alias_contract(
        FunctionEventInput(
            func=torch.relu,
            func_name="probe",
            func_qualname=None,
            args=(x,),
            kwargs={},
            raw_output=({"outer": [x]},),
            arg_copies=(x.clone(),),
            kwarg_copies={},
            module_stack=(),
            is_bottom_level_func=True,
            func_call_id=1,
            expected_output_count=1,
        )
    )

    assert semantics.mutated_input_positions == ()
    assert semantics.aliased_output_inputs == (0,)
