"""Intervention honesty: ``mean_ablate(dim=)`` fails closed and
``batch_independent`` is DERIVED (edits memo rows 0a/0b, decisions D16/D18/D19).

Pins, in both failure directions of the one line that shipped wrong:

- A1b: an unknown ``dim=`` token refuses typed at construction, so no typo can
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


@pytest.mark.parametrize("token", ["btach", "banana", "batch_mean", "", "self", "all"])
def test_unknown_dim_token_refuses_typed(token: str) -> None:
    """A1b: any token outside the dim= vocabulary refuses at construction."""

    with pytest.raises(InvalidArgumentError) as excinfo:
        tl.mean_ablate(dim=token)  # type: ignore[arg-type]
    assert excinfo.value.fields["code"] == "intervention_over_invalid"
    assert repr(token) in str(excinfo.value)
    # The refusal teaches the supported spellings.
    assert "dim=None" in str(excinfo.value)
    assert "source=" in str(excinfo.value)


@pytest.mark.parametrize("token", ["batch", "position", "feature"])
def test_semantic_axis_name_refuses_until_evidence(token: str) -> None:
    """A semantic axis is never guessed from rank: the ladder refuses typed."""

    with pytest.raises(InvalidArgumentError) as excinfo:
        tl.mean_ablate(dim=token)  # type: ignore[arg-type]
    assert excinfo.value.fields["code"] == "axis_semantics_unknown"


@pytest.mark.parametrize("token", [1.5, True, (), [0, "x"]])
def test_non_axis_dim_refuses_typed(token: object) -> None:
    """Only None, an int or a tuple of ints name axes; anything else refuses."""

    with pytest.raises(InvalidArgumentError) as excinfo:
        tl.mean_ablate(dim=token)  # type: ignore[arg-type]
    assert excinfo.value.fields["code"] == "intervention_over_invalid"


def test_typo_can_never_flip_batch_independent_open() -> None:
    """A1b belt: no constructible spec with a bad token exists, so the append
    gate can never see ``batch_independent=True`` from a typo."""

    with pytest.raises(InvalidArgumentError):
        tl.mean_ablate(dim="btach")
    # The only constructible self spellings derive fail-closed flags:
    assert tl.mean_ablate(dim=None).batch_independent is False
    assert tl.mean_ablate(dim=0).batch_independent is False


def test_tensor_source_derives_batch_independent_true() -> None:
    """A1c: an external tensor source never reads the traced batch."""

    spec = tl.mean_ablate(source=torch.ones(8), dim=None)
    assert spec.batch_independent is True


def test_self_mean_derives_batch_independent_false() -> None:
    """A1c inverse: the self-mean couples batch rows."""

    assert tl.mean_ablate(dim=None).batch_independent is False


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


def test_mean_ablate_fires_with_tensor_source_mean() -> None:
    """The tensor-source spelling computes the SOURCE mean, not the self mean."""

    model, x, log = _traced_model()
    source = torch.full((5,), 7.0)
    fork = log.fork()
    fork.do(tl.func("relu"), tl.mean_ablate(source=source, dim=None))
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
