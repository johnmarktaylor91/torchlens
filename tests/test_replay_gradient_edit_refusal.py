"""A gradient site on a replay fork refuses with a typed code, and its remedy runs.

Replay recomputes forward values only, so ``fork.do(tl.grad(site), edit)`` (any edit:
the site, not the edit, makes it a gradient edit) has no gradient to edit there. The
refusal names the site, changes nothing, and points at the rerun door that arms the
edit for the next backward.
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


def _zero(g: torch.Tensor, *, ctx: Any) -> torch.Tensor:
    return torch.zeros_like(g)


_EDITS: dict[str, Callable[[], Any]] = {
    "zero_ablate": lambda: tl.zero_ablate(),
    "scale": lambda: tl.scale(0.5),
    "clip_norm": lambda: tl.clip_norm(1.0),
    "add_noise": lambda: tl.add_noise(0.1, seed=0),
    "clamp": lambda: tl.clamp(low=-1.0, high=1.0),
    "callable": lambda: _zero,
}

_X = torch.randn(2, 4, generator=torch.Generator().manual_seed(3))


def _ready(model: nn.Module) -> Any:
    return tl.trace(model, _X, capture=tl.options.CaptureOptions(intervention_ready=True))


@pytest.mark.parametrize("edit", sorted(_EDITS))
def test_fork_do_gradient_site_refuses_and_the_remedy_runs(edit: str) -> None:
    model = _Net()
    fork = _ready(model).fork()
    before = {label: fork[label].out.detach().clone() for label in ("linear_1_1", "linear_2_2")}
    recipe_before = fork.recipe
    with pytest.raises(InvalidArgumentError) as excinfo:
        fork.do(tl.grad("linear_1_1"), _EDITS[edit]())
    fields = excinfo.value.fields
    assert fields["code"] == "replay_gradient_edit_unsupported"
    message = str(excinfo.value)
    assert "linear_1_1" in message
    assert "model= and x=" in message and "engine='set_only'" in message
    # Nothing changed: values, recipe, and no sticky hook left to re-fire on the next push.
    for label, value in before.items():
        assert torch.equal(fork[label].out, value)
    assert fork.recipe == recipe_before
    with pytest.raises(ReplayPreconditionError):
        fork.push()
    # The remedy, spelled as the refusal names it: rerun with the edit armed, then backward.
    result = fork.do(
        tl.grad("linear_1_1"),
        _EDITS[edit](),
        model=model,
        inputs=_X,
        loss=lambda t: t[t.output_layers[0]].out.sum(),
    )
    assert result.num_backward_passes == 1
    assert model.l1.weight.grad is not None
    if edit in ("zero_ablate", "callable"):
        assert torch.count_nonzero(model.l1.weight.grad) == 0


def test_push_replay_hooks_gradient_site_refuses() -> None:
    """The other forward-only door, ``push(replay=ReplayOptions(hooks=...))``, refuses alike."""

    replay = tl.options.ReplayOptions(hooks={tl.grad(tl.label("linear_1_1")): tl.zero_ablate()})
    with pytest.raises(InvalidArgumentError) as excinfo:
        _ready(_Net()).push(replay=replay)
    assert excinfo.value.fields["code"] == "replay_gradient_edit_unsupported"
