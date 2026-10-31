"""A gradient edit on a replay fork refuses with a typed code, and its remedy runs.

Replay recomputes forward values only, so ``fork.do(site, tl.zero_grad())`` (and every
other backward verb) has no gradient to edit there. The refusal names the verb and the
site, changes nothing, and points at the live door that does edit gradients.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._errors import InvalidArgumentError
from torchlens.intervention.errors import ReplayPreconditionError


class _Net(nn.Module):
    """``l2(l1(x))``: the gradient edit sits on ``l1``'s output."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.l1 = nn.Linear(4, 4)
        self.l2 = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.l2(self.l1(x))


def _identity(g: torch.Tensor, *, ctx: Any) -> torch.Tensor:
    return g


_VERBS: dict[str, Callable[[], Any]] = {
    "zero_grad": lambda: tl.zero_grad(),
    "scale_grad": lambda: tl.scale_grad(0.5),
    "clip_grad": lambda: tl.clip_grad(1.0),
    "add_grad_noise": lambda: tl.add_grad_noise(0.1, seed=0),
    "clamp_grad": lambda: tl.clamp_grad(-1.0, 1.0),
    "hook_grad": lambda: tl.hook_grad(_identity),
}

_X = torch.randn(2, 4, generator=torch.Generator().manual_seed(3))


def _ready(model: nn.Module) -> Any:
    return tl.trace(model, _X, capture=tl.options.CaptureOptions(intervention_ready=True))


@pytest.mark.parametrize("verb", sorted(_VERBS))
def test_fork_do_gradient_edit_refuses_and_the_remedy_runs(verb: str) -> None:
    model = _Net()
    fork = _ready(model).fork()
    before = {label: fork[label].out.detach().clone() for label in ("linear_1_1", "linear_2_2")}
    with pytest.raises(InvalidArgumentError) as excinfo:
        fork.do("linear_1_1", _VERBS[verb]())
    fields = excinfo.value.fields
    assert fields["code"] == "replay_gradient_edit_unsupported"
    assert fields["verb"] == verb
    assert fields["site"] == "linear_1_1"
    assert "linear_1_1" in str(excinfo.value) and verb in str(excinfo.value)
    # Nothing changed: values, and no sticky hook left to re-fire on the next push.
    for label, value in before.items():
        assert torch.equal(fork[label].out, value)
    with pytest.raises(ReplayPreconditionError):
        fork.push()
    # The remedy, spelled as the refusal names it, edits the gradient live.
    assert "tl.module('l1')" in fields["remedy"]
    trace = tl.trace(
        model,
        _X,
        capture=tl.options.CaptureOptions(backward_ready=True),
        intervene=tl.when(tl.module("l1"), _VERBS[verb]()),
    )
    trace.backward(trace[trace.output_layers[0]].out.sum())
    assert model.l1.weight.grad is not None
    if verb == "zero_grad":
        assert torch.count_nonzero(model.l1.weight.grad) == 0


def test_push_replay_hooks_gradient_edit_refuses() -> None:
    """The other forward-only door, ``push(replay=ReplayOptions(hooks=...))``, refuses alike."""

    replay = tl.options.ReplayOptions(hooks={tl.label("linear_1_1"): tl.zero_grad()})
    with pytest.raises(InvalidArgumentError) as excinfo:
        _ready(_Net()).push(replay=replay)
    assert excinfo.value.fields["code"] == "replay_gradient_edit_unsupported"
