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

import numpy as np
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
    return tl.trace(model, x, capture=tl.options.CaptureOptions(layers_to_save="all"))


def _partial_trace(n_stimuli: int = 6) -> tl.Trace:
    """Partially saved trace (the call that failed before the fix)."""

    torch.manual_seed(0)
    model = _BNModel().eval()
    x = torch.randn(n_stimuli, 3, 8, 8)
    return tl.trace(model, x, save=tl.func("relu"))


@pytest.mark.heavy
def test_legacy_dataset_final_output_compat_and_disclosure() -> None:
    """Item 6: site=None keeps final-output behavior with honest descriptors."""

    rsatoolbox = pytest.importorskip("rsatoolbox")
    log = _full_trace()
    from torchlens.bridge import rsatoolbox as bridge

    dataset = bridge.dataset(log)
    assert isinstance(dataset, rsatoolbox.data.Dataset)
    assert dataset.measurements.shape == (6, 2)
    assert dataset.descriptors["pool"] == "flatten"
    assert dataset.descriptors["site"] == "<final_output>"
    assert "neuroid" not in dataset.channel_descriptors
    assert "feature_index" in dataset.channel_descriptors
    obs = dataset.obs_descriptors
    assert list(obs["presentation"]) == list(range(6))
    assert list(obs["tl_presentation_index"]) == list(range(6))


@pytest.mark.heavy
def test_legacy_dataset_site_route_gates_eligibility() -> None:
    """Item 6: explicit sites run the core gate; buffers refuse."""

    pytest.importorskip("rsatoolbox")
    log = _full_trace()
    from torchlens.bridge import rsatoolbox as bridge

    dataset = bridge.dataset(log, "relu_1_3")
    assert dataset.descriptors["site"] == "relu_1_3"
    assert dataset.measurements.shape[0] == 6
    with pytest.raises(ValueError, match="buffer overwrite"):
        bridge.dataset(log, "buffer_1")


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
def test_brain_score_default_sweep_filters_buffers() -> None:
    """Item 9: buffer rows are never scored; one summarized disclosure."""

    from torchlens.bridge.brain_score import per_layer
    from torchlens.errors._base import TorchLensWarning

    log = _full_trace()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scores = per_layer(log, lambda out, layer=None: float(out.mean()))
    assert not any("buffer" in label for label in scores)
    assert len(scores) == 6
    skip_warnings = [
        w for w in caught if isinstance(w.message, TorchLensWarning) and "skipped" in str(w.message)
    ]
    assert len(skip_warnings) == 1


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


@pytest.mark.heavy
def test_delegated_dataset_measurements_match_neuro_route() -> None:
    """One code path: the bridge and neuro.datasets return identical bytes."""

    pytest.importorskip("rsatoolbox")
    log = _full_trace()
    from torchlens.bridge import rsatoolbox as bridge
    from torchlens.neuro._datasets import datasets

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plural = datasets(log, ["relu_1_3"])
    single = bridge.dataset(log, "relu_1_3")
    assert np.array_equal(single.measurements, plural["relu_1_3"].measurements)
