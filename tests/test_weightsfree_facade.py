"""Rung-4 summary facade (weightsfree memo D12 / face 2 / item 17).

A read-only facade may auto-select the exact structure-only option state
ONLY for an all-meta model plus an unambiguous input plan; it stamps the
SAME marker, admission record, and evidence envelope as the explicit power
path, with the auto-set visibly recorded in provenance. A bare summary with
no input evidence refuses rather than guessing.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl

pytestmark = pytest.mark.smoke


class Toy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def _meta_model() -> nn.Module:
    with torch.device("meta"):
        model = Toy()
    model.eval()
    return model


def test_facade_auto_selects_with_declared_input_size() -> None:
    report = tl.summary(_meta_model(), input_size=(2, 8))
    text = report.lower()
    assert "structure-only" in text or "hypothes" in text
    assert "8" in report  # geometry rendered


def test_facade_accepts_explicit_meta_inputs() -> None:
    report = tl.summary(_meta_model(), torch.empty(2, 8, device="meta"))
    assert "hypothes" in report.lower() or "structure-only" in report.lower()


def test_bare_facade_refuses_rather_than_guessing() -> None:
    with pytest.raises(Exception) as excinfo:
        tl.summary(_meta_model())
    assert excinfo.value.fields["code"] == "weightsfree_facade_input_evidence_required"
    assert "input_size" in str(excinfo.value)


def test_facade_never_triggers_on_real_models() -> None:
    """An all-real model takes the ordinary summary path (values measured)."""

    model = Toy()
    model.eval()
    report = tl.summary(model, input_size=(2, 8))
    assert "hypothes" not in report.lower()
