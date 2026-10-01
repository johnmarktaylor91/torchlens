"""Intervention honesty: ``mean_ablate(over=)`` fails closed and
``batch_independent`` is DERIVED (edits memo rows 0a/0b, decisions D16/D18/D19).

Pins, in both failure directions of the one line that shipped wrong:

- A1b: an unknown ``over=`` token refuses typed at construction, so no typo can
  flip ``batch_independent`` open and permit an unsound append.
- A1c: ``mean_ablate(source=tensor)`` derives ``batch_independent=True`` (the
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

pytestmark = pytest.mark.smoke


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
    log = tl.trace(model, x, capture=tl.options.CaptureOptions(intervention_ready=True))
    return model, x, log


@pytest.mark.parametrize("token", ["btach", "banana", "batch", "batch_mean", ""])
def test_unknown_over_token_refuses_typed(token: str) -> None:
    """A1b: any token outside the closed vocabulary refuses at construction."""

    with pytest.raises(InvalidArgumentError) as excinfo:
        tl.mean_ablate(over=token)
    assert excinfo.value.fields["code"] == "intervention_over_invalid"
    assert repr(token) in str(excinfo.value)
    # The refusal teaches the supported spellings.
    assert "'self'" in str(excinfo.value)
    assert "source=" in str(excinfo.value)


@pytest.mark.parametrize("token", [3, (0, 1), None])
def test_non_string_over_refuses_typed(token: object) -> None:
    """Axis-valued ``over=`` is not implemented; accepting it would be the
    audit-only-label defect again. Refuse until the axis-aware family ships."""

    with pytest.raises(InvalidArgumentError) as excinfo:
        tl.mean_ablate(over=token)  # type: ignore[arg-type]
    assert excinfo.value.fields["code"] == "intervention_over_invalid"


def test_typo_can_never_flip_batch_independent_open() -> None:
    """A1b belt: no constructible spec with a bad token exists, so the append
    gate can never see ``batch_independent=True`` from a typo."""

    with pytest.raises(InvalidArgumentError):
        tl.mean_ablate(over="btach")
    # The only constructible spellings derive fail-closed flags:
    assert tl.mean_ablate().batch_independent is False
    assert tl.mean_ablate(over="self").batch_independent is False


def test_tensor_source_derives_batch_independent_true() -> None:
    """A1c: an external tensor source never reads the traced batch."""

    spec = tl.mean_ablate(source=torch.ones(8))
    assert spec.batch_independent is True


def test_self_mean_derives_batch_independent_false() -> None:
    """A1c inverse: the self-mean couples batch rows."""

    assert tl.mean_ablate().batch_independent is False


def test_append_refuses_self_mean_and_permits_tensor_source() -> None:
    """The append gate honors the DERIVED flag in both directions."""

    model, x, log = _traced_model()
    log.attach_hooks(tl.func("relu"), tl.mean_ablate(), confirm_mutation=True)
    log.run(model, x)
    with pytest.raises(AppendBatchDependenceError):
        log.run(model, torch.randn(2, 4), replay=ReplayOptions(append=True))

    model2, x2, log2 = _traced_model()
    log2.attach_hooks(tl.func("relu"), tl.mean_ablate(source=torch.ones(4)), confirm_mutation=True)
    log2.run(model2, x2)
    n_before = log2["relu_1_2"].out.shape[0]
    appended = log2.run(model2, torch.randn(2, 4), replay=ReplayOptions(append=True))
    assert appended["relu_1_2"].out.shape[0] == n_before + 2


def test_mean_ablate_fires_with_tensor_source_mean() -> None:
    """The tensor-source spelling computes the SOURCE mean, not the self mean."""

    model, x, log = _traced_model()
    source = torch.full((5,), 7.0)
    fork = log.fork()
    fork.do(tl.func("relu"), tl.mean_ablate(source=source))
    patched = fork["relu_1_2"].out
    assert torch.allclose(patched, torch.full_like(patched, 7.0))


def test_scramble_source_and_from_both_passed_refuses() -> None:
    """0b: both spellings passed is a conflict, never a silent preference."""

    with pytest.raises(InvalidArgumentError) as excinfo:
        tl.intervention.scramble_elements(torch.ones(3), from_=torch.zeros(3))
    assert excinfo.value.fields["code"] == "intervention_source_conflict"
    assert "source=" in str(excinfo.value)
    assert "from_=" in str(excinfo.value)


def test_scramble_single_spelling_still_accepted() -> None:
    """Either spelling alone keeps working after the refusal lands."""

    by_source = tl.intervention.scramble_elements(torch.ones(3), seed=1)
    by_from = tl.intervention.scramble_elements(from_=torch.ones(3), seed=1)
    assert torch.equal(by_source.args[0], by_from.args[0])
