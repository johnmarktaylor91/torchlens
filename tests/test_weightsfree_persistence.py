"""Persistence matrix (memo sec 8.1 item 9 / W3 / D21).

Real- and meta-substrate buffer traces round-trip with no payload; the
envelope survives; a loaded trace is HYPOTHESIS again (the discharge
registry is session-only); forged payloads still fail the M-C2/M-C3
coherence gates.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.options import CaptureOptions

pytestmark = pytest.mark.smoke


class BufferedToy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(4)
        self.fc = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.bn(x))


def _roundtrip(trace, tmp_path):
    path = tmp_path / "trace.tlspec"
    tl.save(trace, str(path))
    return tl.load(str(path))


def test_meta_buffer_trace_roundtrips_value_free(tmp_path) -> None:
    """Defect L6 closure, meta form: the save no longer dies and the load
    passes its own coherence gates with zero payloads."""

    with torch.device("meta"):
        model = BufferedToy()
    model.eval()
    trace = tl.trace(
        model, torch.empty(2, 4, device="meta"), capture=CaptureOptions(structure_only=True)
    )
    loaded = _roundtrip(trace, tmp_path)
    assert loaded.structure_only is True
    assert loaded.structure_evidence is not None
    assert loaded.structure_evidence["substrate"] == "meta"
    for layer in loaded.layer_list:
        assert getattr(layer, "out", None) is None


def test_real_buffer_trace_roundtrips_value_free(tmp_path) -> None:
    """Defect L6 closure, real form: the shipped save-then-cannot-load bug on
    buffer-holding structure-only captures is fixed on BOTH substrates."""

    model = BufferedToy()
    model.eval()
    trace = tl.trace(model, torch.randn(2, 4), capture=CaptureOptions(structure_only=True))
    loaded = _roundtrip(trace, tmp_path)
    assert loaded.structure_only is True
    assert loaded.structure_evidence is not None
    assert loaded.structure_evidence["substrate"] == "real"
    for layer in loaded.layer_list:
        assert getattr(layer, "out", None) is None


def test_loaded_trace_is_hypothesis_again(tmp_path) -> None:
    """D21: the discharge registry is session-only; a corroborated trace
    loads back as HYPOTHESIS and the persisted envelope says discharge=absent."""

    from torchlens.capture.structure_only import StructureClaimStatus, claim_status_for

    torch.manual_seed(0)
    model = BufferedToy()
    model.eval()
    trace = tl.trace(model, torch.randn(2, 4), capture=CaptureOptions(structure_only=True))
    real = tl.trace(model, torch.randn(2, 4))
    assert trace.discharge_against(real).verdict.value == "corroborated"
    assert claim_status_for(trace) is StructureClaimStatus.CORROBORATED
    loaded = _roundtrip(trace, tmp_path)
    assert claim_status_for(loaded) is StructureClaimStatus.HYPOTHESIS
    assert loaded.structure_evidence["discharge"] == "absent"


def test_forged_envelope_fails_validation(tmp_path) -> None:
    """A forged envelope refuses fail-closed (C07 validation, M-C2 class).

    The envelope persists inside the pickled metadata blob; a forged
    substrate token is refused by the load-side validator with the typed
    ``artifact_structure_evidence_invalid`` code, and an envelope planted on
    a non-structure-only capture refuses as incoherent.
    """

    from torchlens._io.forgery_validation import _validate_structure_evidence

    model = BufferedToy()
    model.eval()
    trace = tl.trace(model, torch.randn(2, 4), capture=CaptureOptions(structure_only=True))
    loaded = _roundtrip(trace, tmp_path)
    # Forge 1: an out-of-vocabulary substrate token.
    loaded.structure_evidence = dict(loaded.structure_evidence, substrate="warp")
    with pytest.raises(Exception) as excinfo:
        _validate_structure_evidence(loaded)
    assert excinfo.value.fields["code"] == "artifact_structure_evidence_invalid"
    # Forge 2: values_available=True contradicts the mode the envelope discloses.
    loaded.structure_evidence = dict(
        loaded.structure_evidence, substrate="real", values_available=True
    )
    with pytest.raises(Exception) as excinfo:
        _validate_structure_evidence(loaded)
    assert excinfo.value.fields["code"] == "artifact_structure_evidence_invalid"
    # Forge 3: an envelope on a values-bearing capture is incoherent.
    ordinary = tl.trace(BufferedToy().eval(), torch.randn(2, 4))
    ordinary.structure_evidence = trace.structure_evidence
    with pytest.raises(Exception) as excinfo:
        _validate_structure_evidence(ordinary)
    assert excinfo.value.fields["code"] == "artifact_structure_evidence_invalid"
