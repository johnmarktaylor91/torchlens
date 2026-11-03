"""Intervention honesty: ``mean_ablate(dim=)`` fails closed and
``is_batch_independent`` is DERIVED (edits memo rows 0a/0b, decisions D16/D18/D19).

Pins, in both failure directions of the one line that shipped wrong:

- A1b: an unknown ``dim=`` token refuses typed at construction, so no typo can
  flip ``is_batch_independent`` open and permit an unsound append.
- A1c: ``mean_ablate(source=tensor)`` derives ``is_batch_independent=True`` (the
  fill value never reads the traced batch; append is permitted), while the
  self-mean derives ``False`` (couples batch rows; append refuses).
- 0b: ``source=``/``from_=`` both passed refuses typed instead of silently
  preferring ``source=``.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._errors import InvalidArgumentError
from torchlens.intervention.errors import AppendBatchDependenceError
from torchlens.options import ReplayOptions


class _TwoLinear(nn.Module):
    """Small real-module stack for append-gate pins."""

    def __init__(self) -> None:
        super().__init__()
        self.lin1 = nn.Linear(4, 4)
        self.lin2 = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(torch.relu(self.lin1(x)))


def _traced_model() -> tuple[nn.Module, torch.Tensor, tl.Trace]:
    torch.manual_seed(0)
    model = _TwoLinear().eval()
    x = torch.randn(3, 4)
    log = tl.trace(model, x, capture=tl.options.CaptureOptions(replayable=True))
    return model, x, log


def test_append_refuses_self_mean_and_permits_tensor_source() -> None:
    """The append gate honors the DERIVED flag in both directions."""

    model, x, log = _traced_model()
    log.attach_hooks(tl.func("relu"), tl.mean_ablate(dim=None), confirm_mutation=True)
    log.run(model, x)
    with pytest.raises(AppendBatchDependenceError):
        log.run(model, torch.randn(2, 4), replay=ReplayOptions(append=True))

    model2, x2, log2 = _traced_model()
    log2.attach_hooks(
        tl.func("relu"), tl.mean_ablate(source=torch.ones(4), dim=None), confirm_mutation=True
    )
    log2.run(model2, x2)
    n_before = log2["relu_1_2"].out.shape[0]
    appended = log2.run(model2, torch.randn(2, 4), replay=ReplayOptions(append=True))
    assert appended["relu_1_2"].out.shape[0] == n_before + 2


def test_scramble_source_and_from_both_passed_refuses() -> None:
    """0b: both spellings passed is a conflict, never a silent preference."""

    with pytest.raises(InvalidArgumentError) as excinfo:
        tl.intervention.scramble_elements(torch.ones(3), from_=torch.zeros(3))
    assert excinfo.value.fields["code"] == "intervention_source_conflict"
    assert "source=" in str(excinfo.value)
    assert "from_=" in str(excinfo.value)


def test_typo_can_never_flip_batch_independent_open() -> None:
    """A1b belt: no constructible spec with a bad token exists, so the append
    gate can never see ``is_batch_independent=True`` from a typo."""

    with pytest.raises(InvalidArgumentError):
        tl.mean_ablate(dim="btach")
    # The only constructible self spellings derive fail-closed flags:
    assert tl.mean_ablate(dim=None).is_batch_independent is False
    assert tl.mean_ablate(dim=0).is_batch_independent is False
