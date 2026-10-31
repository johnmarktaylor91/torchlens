"""Buffer operands for substitution (R2-25, N-IV-9): BatchNorm-stat and state what-ifs.

``trace.do(tl.params("<buffer address>"), edit)`` substitutes the buffer's
initial version (the value its consumers read) and replays the cone. Each row
is checked against a deep copy of the model whose buffer was actually changed,
and the live buffer is checked unchanged.
"""

from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.errors import SelectionError

_CAPTURE = tl.options.CaptureOptions(intervention_ready=True)
_CONFIRM = tl.options.InterventionOptions(confirm_mutation=True)


def _batchnorm_model() -> nn.Sequential:
    """Linear -> eval BatchNorm with non-trivial running stats -> Linear."""

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4), nn.Linear(4, 2)).eval()
    with torch.no_grad():
        model[1].running_mean.copy_(torch.randn(4))
        model[1].running_var.copy_(torch.rand(4) + 0.5)
    return model


_X = torch.randn(5, 3, generator=torch.Generator().manual_seed(1))


def _edited_output(model: nn.Module, selection: object, edit: object) -> torch.Tensor:
    """Run one buffer what-if through TorchLens and return the model output."""

    trace = tl.trace(model, _X, capture=_CAPTURE)
    edited = trace.do(selection, edit, intervention=_CONFIRM)
    return edited[edited.output_layers[0]].out.detach()


def test_running_mean_what_if_matches_a_changed_model() -> None:
    """Doubling ``running_mean`` equals a model whose running mean was doubled."""

    model = _batchnorm_model()
    before = model[1].running_mean.clone()
    got = _edited_output(model, tl.params("1.running_mean"), tl.scale(2.0))
    changed = copy.deepcopy(model)
    with torch.no_grad():
        changed[1].running_mean.mul_(2.0)
        want = changed(_X)
        clean = model(_X)
    torch.testing.assert_close(got, want, rtol=0.0, atol=1e-6)
    assert not torch.allclose(got, clean)
    assert torch.equal(model[1].running_mean, before)


def test_masked_running_var_what_if() -> None:
    """A mask restricts the substitution to the selected buffer elements."""

    model = _batchnorm_model()
    mask = torch.tensor([True, False, False, True])
    got = _edited_output(model, tl.params("1.running_var", mask), tl.scale(3.0))
    changed = copy.deepcopy(model)
    with torch.no_grad():
        changed[1].running_var[mask] *= 3.0
        want = changed(_X)
    torch.testing.assert_close(got, want, rtol=0.0, atol=1e-6)


class _StateCell(nn.Module):
    """A module whose registered state buffer is rewritten during forward."""

    def __init__(self) -> None:
        """Register a zero state."""

        super().__init__()
        self.register_buffer("h", torch.zeros(3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Two state updates; the output reads the final state."""

        self.h = torch.tanh(self.h + x)
        self.h = torch.tanh(self.h + x)
        return self.h * 2.0


def test_state_buffer_what_if_threads_through_writes() -> None:
    """Substituting a rewritten buffer's initial value propagates through its writes."""

    model = _StateCell()
    x = torch.tensor([0.1, -0.2, 0.3])
    trace = tl.trace(model, x, capture=_CAPTURE)
    edited = trace.do(tl.params("h"), tl.replace_with(torch.ones(3)), intervention=_CONFIRM)
    got = edited[edited.output_layers[0]].out.detach()
    want = torch.tanh(torch.tanh(torch.ones(3) + x) + x) * 2.0
    torch.testing.assert_close(got, want, rtol=0.0, atol=1e-6)


def test_buffer_mask_shape_mismatch_refuses_typed() -> None:
    """A mask over the wrong shape refuses instead of broadcasting."""

    trace = tl.trace(_batchnorm_model(), _X, capture=_CAPTURE)
    with pytest.raises(SelectionError) as info:
        trace.do(
            tl.params("1.running_mean", torch.ones(3, dtype=torch.bool)),
            tl.scale(2.0),
            intervention=_CONFIRM,
        )
    assert info.value.fields["code"] == "param_substitution_occurrence_underivable"
