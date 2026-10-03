"""Runnable save refuses intervention-replaced captures (deephunt F1).

An intervened capture's archived provenance (activations, taken path) reflects
the REPLACED computation, but the sparse runnable descriptor records the
original callables, so a replay would silently recompute the UN-intervened
function. The producer preflight must refuse with a named diagnostic instead of
emitting an artifact whose replay output comes from a different computation
than its provenance.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.errors import RunnablePreflightError
from torchlens.options import CaptureOptions

_CAPTURE = CaptureOptions(
    intervention_ready=True,
    capture_container_structure=True,
    cache=False,
)


class ReluModel(nn.Module):
    """Two-linear model with a relu interior an intervention can replace."""

    def __init__(self) -> None:
        """Initialize the two linear maps."""

        super().__init__()
        self.fc1 = nn.Linear(4, 3)
        self.fc2 = nn.Linear(3, 2)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Run linear -> relu -> linear."""

        return self.fc2(torch.relu(self.fc1(value)))


def test_runnable_save_refuses_intervention_replaced_capture(tmp_path: Path) -> None:
    """An intervention-replaced capture must refuse level='runnable' save typed."""

    model = ReluModel()
    value = torch.randn(2, 4)
    trace = tl.trace(
        model,
        value,
        intervene=tl.when(tl.func("relu"), tl.zero_ablate()),
        capture=_CAPTURE,
    )
    assert any(bool(getattr(op, "intervention_replaced", False)) for op in trace.layer_list)
    with pytest.raises(RunnablePreflightError) as excinfo:
        tl.save(trace, tmp_path / "intervened.tlspec", level="runnable")
    assert excinfo.value.fields.get("code") == "sparse_preflight_failed"
    diagnostics = str(excinfo.value.fields.get("diagnostics"))
    assert "user_intervention_not_replayable" in diagnostics
    assert "relu" in diagnostics


def test_runnable_save_allows_zero_match_intervention(tmp_path: Path) -> None:
    """A selector that fired on ZERO sites leaves the plain capture runnable."""

    model = ReluModel()
    value = torch.randn(2, 4)
    with pytest.warns(UserWarning, match="matched zero sites"):
        trace = tl.trace(
            model,
            value,
            intervene=tl.when(tl.func("sigmoid"), tl.zero_ablate()),
            capture=_CAPTURE,
        )
    assert not any(bool(getattr(op, "intervention_replaced", False)) for op in trace.layer_list)
    tl.save(trace, tmp_path / "plain.tlspec", level="runnable")
    loaded = tl.load(tmp_path / "plain.tlspec")
    assert loaded.readiness is not None
