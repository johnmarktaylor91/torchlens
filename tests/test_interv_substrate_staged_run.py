"""C03 honesty gate: ``run(inputs=...)`` refuses typed when a spec is staged.

Surgery memo build item 1 (the measured silent wrong answer): the unified
provider is a fresh verified execution and never installs the staged
intervention spec, while the legacy ``run(model, x)`` surface does. Before
this gate, ``run(inputs=...)`` on a trace carrying staged sticky hooks
silently returned an un-intervened verified run.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.intervention.errors import EngineDispatchError


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def _capture(model: nn.Module) -> tl.Trace:
    torch.manual_seed(0)
    return tl.trace(model, torch.randn(2, 4), save=tl.func("relu"))


@pytest.mark.smoke
def test_run_inputs_refuses_with_staged_hooks() -> None:
    """A staged sticky hook makes run(inputs=) a typed refusal, not a silent drop."""

    model = _TinyModel()
    log = _capture(model)
    log.attach_hooks(tl.func("relu"), tl.scale(0.0), confirm_mutation=True)
    with pytest.raises(EngineDispatchError) as excinfo:
        log.run(inputs=torch.randn(2, 4))
    assert excinfo.value.fields["code"] == "run_staged_spec_unapplied"
    assert "remedy" in excinfo.value.fields
    # The teaching message names both honest spellings.
    message = str(excinfo.value)
    assert "intervene=" in message
    assert "clear_hooks" in message


@pytest.mark.smoke
def test_run_inputs_refuses_with_staged_value_replacement() -> None:
    """A staged set() value replacement also trips the gate."""

    model = _TinyModel()
    log = _capture(model)
    log.set("relu_1_2", torch.zeros(2, 4), confirm_mutation=True)
    with pytest.raises(EngineDispatchError) as excinfo:
        log.run(inputs=torch.randn(2, 4))
    assert excinfo.value.fields["code"] == "run_staged_spec_unapplied"


@pytest.mark.smoke
def test_run_inputs_allowed_after_hooks_detached() -> None:
    """clear_hooks() (the taught remedy) re-opens the unified surface."""

    model = _TinyModel()
    log = _capture(model)
    log.attach_hooks(tl.func("relu"), tl.scale(0.0), confirm_mutation=True)
    log.clear_hooks(confirm_mutation=True)
    result = log.run(inputs=torch.randn(2, 4))
    assert result is not None


@pytest.mark.smoke
def test_run_inputs_allowed_without_staged_spec() -> None:
    """No staged spec: the unified surface is untouched by the gate."""

    model = _TinyModel()
    log = _capture(model)
    result = log.run(inputs=torch.randn(2, 4))
    assert result is not None


@pytest.mark.smoke
def test_legacy_run_still_applies_staged_spec() -> None:
    """The legacy run(model, x) surface keeps INSTALLING the staged spec.

    This is the other half of the honesty story: the refusal exists because
    the two surfaces genuinely differ. A staged zero-scale hook on relu must
    change the fresh legacy-rerun activations.
    """

    model = _TinyModel()
    log = _capture(model)
    x2 = torch.randn(2, 4)
    log.attach_hooks(tl.func("relu"), tl.scale(0.0), confirm_mutation=True)
    log.run(model, x2)
    relu_out = log["relu_1_2"].out
    assert torch.count_nonzero(relu_out) == 0
