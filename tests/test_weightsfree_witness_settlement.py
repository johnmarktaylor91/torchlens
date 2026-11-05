"""The completeness witness never verifies a value-free capture.

A clean dispatch census proves that every aten dispatch had an owner; it says
nothing about tensor values. A structure-only capture (meta or real substrate)
therefore settles ``capture_verified=None`` under the witness, exactly as it
does without it, while ``completeness_witness_verified`` keeps the census
result. The weights-free settlement invariant stays a tripwire: an incoherent
value-free settlement still raises.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import pytest
import torch
from _stale_holders import OpaqueCallable
from torch import nn

import torchlens as tl
from torchlens import _state
from torchlens._capture._weightsfree_admission import enforce_settlement_invariants
from torchlens._capture_honesty import honesty_preamble_lines
from torchlens._errors import WeightsfreeIntegrityError
from torchlens.backends.torch import _completeness_finalize, rescue
from torchlens.backends.torch.wrappers import unwrap_torch, wrap_torch
from torchlens.options import CaptureOptions


class _LinearRelu(nn.Module):
    """One linear layer and a relu."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the layer and the relu."""

        return torch.relu(self.fc(x))


class _StaleReluAfterLinear(nn.Module):
    """A relu bound before TorchLens wrapped torch, between two traced ops.

    The stale relu feeds a traced op, so no module-output boundary can credit it.
    """

    def __init__(self, stale_relu: Callable[..., Any]) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)
        self.stale_relu = stale_relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the layer, the unwrapped relu the witness cannot attribute, then tanh."""

        return torch.tanh(self.stale_relu(self.fc(x)))


@pytest.fixture
def _witness_armed() -> Iterator[None]:
    """Arm the completeness witness for one test and restore the prior modes."""

    saved_escape = _state._escape_detector_mode
    saved_witness = _state._completeness_witness_mode
    unwrap_torch()
    wrap_torch(completeness_witness=True)
    yield
    unwrap_torch()
    wrap_torch(escape_detector=saved_escape, completeness_witness=saved_witness)


def _structure_only_trace(device: str) -> tl.Trace:
    """Capture the model structure-only on ``device`` ("meta" or "cpu")."""

    with torch.device(device):
        model = _LinearRelu()
    return tl.trace(
        model.eval(),
        torch.empty(2, 4, device=device),
        capture=CaptureOptions(structure_only=True),
    )


# Kept: sole branch coverage of _capture/structure_only.py, _capture_honesty.py, _data_classes/_trace_groups_capture.py; a sole mutation kill.
@pytest.mark.usefixtures("_witness_armed")
@pytest.mark.parametrize(
    "device",
    [
        "meta",
    ],
)
def test_witness_leaves_a_value_free_capture_unverified(device: str) -> None:
    """Witness on plus a structure-only capture: no error, capture_verified is None."""

    trace = _structure_only_trace(device)

    assert trace.options.structure_only is True
    assert trace.health.completeness_mode == "shadow"
    assert trace.health.completeness_verified is True
    assert trace.provenance.verified is None
    assert trace.provenance.verification_reason is None
    preamble = honesty_preamble_lines(trace)
    assert preamble[0].endswith("verified=not_recorded")
    assert any("structure-only (value-free) capture is never verified" in line for line in preamble)
    assert not any("is not armed" in line for line in preamble)


# Kept: sole branch coverage of backends/torch/_completeness_dispatch_names.py; a sole mutation kill.
@pytest.mark.filterwarnings("ignore:TorchLens completeness witness observed:UserWarning")
@pytest.mark.filterwarnings("ignore:TorchLens found tensor arguments with no graph:UserWarning")
@pytest.mark.usefixtures("_witness_armed")
@pytest.mark.parametrize("device", ["cpu"])
def test_witness_gap_still_ceilings_a_value_free_capture(
    device: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unaccounted dispatch still settles False: the carve-out never launders a ceiling."""

    # Keep the primary capture: the rescue re-run would replace its verdict.
    monkeypatch.setattr(rescue, "_escape_signal", lambda trace: None)
    # A custom callable object: capture preparation never rebinds it, so the
    # pristine relu still escapes the wrappers.
    stale_relu = OpaqueCallable(_state._decorated_to_orig.get(id(torch.relu), torch.relu))
    with torch.device(device):
        model = _StaleReluAfterLinear(stale_relu)
    trace = tl.trace(
        model.eval(),
        torch.empty(2, 4, device=device),
        capture=CaptureOptions(structure_only=True),
    )

    assert trace.health.completeness_verified is False
    assert trace.provenance.verified is False
    assert trace.provenance.verification_reason == "dispatch_witness_unaccounted_ops"


# Kept: sole branch coverage of _capture/_structure_evidence.py, _capture/finalize/epilogue.py.
@pytest.mark.usefixtures("_witness_armed")
def test_incoherent_value_free_settlement_still_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the carve-out removed the witness settles True and the invariant fires."""

    # The finalizer reads the predicate from its own module, so patch it there.
    monkeypatch.setattr(_completeness_finalize, "_is_value_free_capture", lambda trace: False)

    with pytest.raises(WeightsfreeIntegrityError) as excinfo:
        _structure_only_trace("meta")

    assert excinfo.value.fields["code"] == "structure_only_settlement_incoherent"
    assert excinfo.value.fields["invariant"] == "capture_verified is True on a value-free capture"


# Kept: sole branch coverage of _capture/_weightsfree_admission.py; a sole mutation kill.
def test_settlement_invariant_rejects_a_verified_value_free_trace() -> None:
    """The invariant itself is unchanged: a value-free trace claiming True is refused."""

    trace = _structure_only_trace("meta")
    trace._provenance_verified = True

    with pytest.raises(WeightsfreeIntegrityError) as excinfo:
        enforce_settlement_invariants(trace)

    assert excinfo.value.fields["invariant"] == "capture_verified is True on a value-free capture"


# Kept: a sole mutation kill.
@pytest.mark.usefixtures("_witness_armed")
@pytest.mark.parametrize(
    "device",
    [
        "meta",
    ],
)
def test_attribute_held_original_relu_is_rebound_and_witnessed(device: str) -> None:
    """The stale-relu case with the bare original, as it was first written.

    Capture preparation rebinds the held original to its wrapper, so the relu
    is captured in the one forward and the witness accounts for every dispatch.
    """

    stale_relu = _state._decorated_to_orig.get(id(torch.relu), torch.relu)
    with torch.device(device):
        model = _StaleReluAfterLinear(stale_relu)
    trace = tl.trace(
        model.eval(),
        torch.empty(2, 4, device=device),
        capture=CaptureOptions(structure_only=True),
    )

    assert "relu" in [op.func_name for op in trace.ops]
    assert trace._rescue_rerun is None
    assert trace.health.completeness_verified is True
    assert model.stale_relu is stale_relu
