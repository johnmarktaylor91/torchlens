"""The guarded fast rerun applies module targets and refuses function targets, typed.

Confirming tests for an intended contract, not a behaviour change. The fast
live engine applies a staged ``tl.module(address)`` intervention at the real
module boundary, so ``run(inputs=..., fast=True)`` equals the same edit made
with a plain eager forward hook. A ``tl.func(...)`` target has no module
boundary to hook and refuses with ``fast_rerun_target_unsupported``. The
transactional ``run(inputs=...)`` never applies a staged spec and refuses with
``run_staged_spec_unapplied``. A fork whose hooks were attached after capture
refuses the fast door with ``fast_rerun_graph_unsteered`` until one legacy
rerun has recorded the edit in its graph.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.errors import RunCapabilityUnavailableError
from torchlens.intervention.errors import EngineDispatchError
from torchlens.options import CaptureOptions


class _Mlp(nn.Module):
    """Two linear layers with a ReLU between them."""

    def __init__(self) -> None:
        """Build the layers."""

        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the MLP.

        Parameters
        ----------
        x:
            Batch of shape ``(n, 8)``.

        Returns
        -------
        torch.Tensor
            Output of shape ``(n, 4)``.
        """

        return self.fc2(self.relu(self.fc1(x)))


def _model() -> _Mlp:
    """Return a seeded eval-mode MLP."""

    torch.manual_seed(0)
    return _Mlp().eval()


def _inputs(seed: int) -> torch.Tensor:
    """Return a seeded batch.

    Parameters
    ----------
    seed:
        Generator seed.

    Returns
    -------
    torch.Tensor
        Batch of shape ``(3, 8)``.
    """

    return torch.randn(3, 8, generator=torch.Generator().manual_seed(seed))


def _eager_zeroed_fc1(model: _Mlp, x: torch.Tensor) -> torch.Tensor:
    """Run the model with ``fc1``'s output zeroed by a plain forward hook.

    Parameters
    ----------
    model:
        Model to run.
    x:
        Input batch.

    Returns
    -------
    torch.Tensor
        The intervened output.
    """

    handle = model.fc1.register_forward_hook(lambda _m, _a, out: torch.zeros_like(out))
    try:
        with torch.no_grad():
            return model(x)
    finally:
        handle.remove()


def test_fast_door_applies_a_module_target_exactly() -> None:
    """A capture-time ``tl.module`` edit replays on new inputs, equal to the eager hook."""

    model = _model()
    trace = tl.trace(model, _inputs(0), intervene=tl.when(tl.module("fc1"), tl.zero_ablate()))
    for seed in (1, 2):
        x = _inputs(seed)
        result = trace.run(inputs=x, fast=True)
        assert torch.equal(result.output.detach(), _eager_zeroed_fc1(model, x))

    with pytest.raises(EngineDispatchError) as refused:
        trace.run(inputs=_inputs(1))
    assert refused.value.fields["code"] == "run_staged_spec_unapplied"


def test_fast_door_on_a_fork_applies_a_module_target_after_one_recapture() -> None:
    """A fork's post-hoc module edit refuses the fast door until a legacy rerun records it."""

    model = _model()
    ready = tl.trace(model, _inputs(0), capture=CaptureOptions(replayable=True))
    fork = ready.fork()
    fork.attach_hooks(tl.module("fc1"), tl.zero_ablate(), confirm_mutation=True)

    with pytest.raises(RunCapabilityUnavailableError) as unsteered:
        fork.run(inputs=_inputs(1), fast=True)
    assert unsteered.value.fields["detection_stage"] == "fast_rerun_graph_unsteered"

    fork.run(model, _inputs(0))
    x = _inputs(3)
    assert torch.equal(fork.run(inputs=x, fast=True).output.detach(), _eager_zeroed_fc1(model, x))


def test_fast_door_refuses_a_function_target_typed() -> None:
    """A ``tl.func`` target has no module boundary for the fast engine and refuses."""

    model = _model()
    ready = tl.trace(model, _inputs(0), capture=CaptureOptions(replayable=True))
    fork = ready.fork()
    fork.attach_hooks(tl.func("relu"), tl.zero_ablate(), confirm_mutation=True)

    with pytest.raises(RunCapabilityUnavailableError) as refused:
        fork.run(inputs=_inputs(1), fast=True)
    assert refused.value.fields["code"] == "run_capability_unavailable"
    assert refused.value.fields["detection_stage"] == "fast_rerun_target_unsupported"
