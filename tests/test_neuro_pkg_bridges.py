"""Lane F22 bridge repairs (neuro MEMO items 6 and 9).

Pins:

- The legacy ``bridge.rsatoolbox.dataset`` is a DELEGATION to the neuro
  per-site code path (D2): disclosed ``pool="flatten"``, the ``"neuroid"``
  channel label retired, presentation-order recoverability preserved
  (``tl_presentation_index`` plus the legacy integer ``presentation``
  column for existing readers), final-output compat intact, and the core
  eligibility gate live on explicit site requests.
- The three brain_score bridge defects stay dead: a partially saved trace
  no longer crashes the default sweep with ``PayloadUnavailableError``
  (the documented ValueError contract actually fires for explicit
  requests), default sweeps run through the core stimulus-indexed filter
  (buffer rows are never scored), and the ``sites=`` vocabulary accepts
  module dotted paths.
"""

from __future__ import annotations

import warnings

import pytest
import torch
from torch import nn

import torchlens as tl


class _BNModel(nn.Module):
    """Toy model whose BatchNorm mints buffer sites on save-everything."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3)
        self.bn = nn.BatchNorm2d(4)
        self.fc = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.bn(self.conv(x)))
        return self.fc(hidden.mean(dim=(2, 3)))


def _full_trace(n_stimuli: int = 6) -> tl.Trace:
    """Save-everything trace with buffer sites present."""

    torch.manual_seed(0)
    model = _BNModel().eval()
    x = torch.randn(n_stimuli, 3, 8, 8)
    return tl.trace(model, x, save="all")


def _partial_trace(n_stimuli: int = 6) -> tl.Trace:
    """Partially saved trace (the call that failed before the fix)."""

    torch.manual_seed(0)
    model = _BNModel().eval()
    x = torch.randn(n_stimuli, 3, 8, 8)
    return tl.trace(model, x, save=tl.func("relu"))


@pytest.mark.heavy
def test_legacy_dataset_extraction_source_requires_site() -> None:
    """Item 6: site=None on a file-route source refuses typed."""

    pytest.importorskip("rsatoolbox")
    from torchlens.bridge import rsatoolbox as bridge
    from torchlens.neuro._handoff import NeuroHandoffError

    torch.manual_seed(0)
    with pytest.raises(NeuroHandoffError) as excinfo:
        bridge.dataset(
            tl.dataset_extraction.LoadedExtraction.__new__(tl.dataset_extraction.LoadedExtraction)
        )
    assert excinfo.value.fields["code"] == "neuro_source_invalid"


@pytest.mark.heavy
def test_brain_score_default_sweep_survives_partial_trace() -> None:
    """Item 9: default sites= on a partially saved trace works (T4 fixture rule)."""

    from torchlens.bridge.brain_score import per_layer

    log = _partial_trace()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = per_layer(log, lambda out, layer=None: float(out.mean()))
    assert list(scores) == ["relu_1_3"]


@pytest.mark.heavy
def test_brain_score_explicit_contracts() -> None:
    """Item 9: documented ValueError fires; module dotted paths resolve."""

    from torchlens.bridge.brain_score import per_layer

    partial = _partial_trace()
    with pytest.raises(ValueError, match="does not have a saved tensor out") as excinfo:
        per_layer(partial, lambda out, layer=None: 0.0, sites=["conv2d_1_1"])
    # The DOCUMENTED contract, not the payload-read crash it used to be.
    assert type(excinfo.value) is ValueError

    full = _full_trace()
    scores = per_layer(full, lambda out, layer=None: float(out.mean()), sites=["bn"])
    assert len(scores) == 1

    with pytest.raises(ValueError, match="not stimulus-indexed|buffer overwrite"):
        per_layer(full, lambda out, layer=None: 0.0, sites=["buffer_1"])
