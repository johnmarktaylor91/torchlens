"""Dead-model persistent-buffer universe for NON-TENSOR-STATE models (r77 F2).

Round-76 (free-roam) found the r75 F2 gc-independent-save closure one model class short:
``snapshot_capture_state`` deliberately returns ``None`` when ANY ``state_dict()`` value is
non-tensor (``get_extra_state()``, packed/quantized entries), and the dead-model leg of
``_add_persistent_buffer_slot_drafts`` keyed its whole rebuild on that value snapshot with a
silent early return. A model with benign extra state whose source object died before the
save therefore DECLARED A DIFFERENT STATE UNIVERSE than the live lane -- never-forward-used
persistent buffers (BatchNorm's ``num_batches_tracked`` in eval mode) were dropped -- so an
honest tensor-only ``load_state_dict`` refused with ``state_unexpected_key``. The r74-F2
over-refusal class surviving for non-tensor-state models, violating the contract sentence
"a runnable save never requires the SOURCE MODEL to still be alive."

r77 derives the persistent-buffer NAME universe (plus per-slot geometry) from a dedicated
capture-time record (``snapshot_persistent_buffer_universe``) that walks ``state_dict()``
names against ``named_parameters``/``named_buffers`` and SURVIVES extra state, so the dead
lane declares exactly the live lane's universe. The embedded-state ceiling is untouched:
an extra-state model still cannot report ``verified`` (missing comparison basis --
pre-existing, coherent, both lanes). When no capture-time record exists either
(``state_dict()`` is not a mapping at the capture boundary), the dead save refuses LOUDLY
and TYPED (``TorchLensIOError``, mirroring the ``include_weights=True`` lane) -- never a
silent under-declaration, never a wrong bind.
"""

from __future__ import annotations

import gc
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.errors import TorchLensIOError
from torchlens.options import CaptureOptions

_CAPTURE = CaptureOptions(intervention_ready=True, capture_container_structure=True, cache=False)


class ExtraStateBN(nn.Module):
    """BatchNorm carrier with benign non-tensor extra state (the r76 repro)."""

    def __init__(self) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize; in eval mode ``num_batches_tracked`` is never forward-used."""

        return self.bn(x)

    def get_extra_state(self) -> dict[str, str]:
        """Return benign non-tensor extra state."""

        return {"note": "hello"}

    def set_extra_state(self, state: dict[str, str]) -> None:
        """Accept the extra state (round-trip no-op)."""


class PlainBN(nn.Module):
    """Zero-collateral control: the same BN without extra state (r75 F2 lane)."""

    def __init__(self) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize."""

        return self.bn(x)


class NonMappingStateDict(nn.Module):
    """Pathological control: ``state_dict()`` returns a non-mapping at capture time."""

    def __init__(self) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize."""

        return self.bn(x)

    def state_dict(self, *args: object, **kwargs: object) -> list[str]:  # type: ignore[override]
        """Return a name list instead of a mapping (unknowable state universe)."""

        return [
            "bn.weight",
            "bn.bias",
            "bn.running_mean",
            "bn.running_var",
            "bn.num_batches_tracked",
        ]


def _x() -> torch.Tensor:
    """Return a fixed probe input."""

    torch.manual_seed(0)
    return torch.randn(2, 4, 8, 8)


def _slot_universe(path: Path) -> list[str]:
    """Return the sorted declared state-slot name universe of a saved artifact."""

    descriptor = tl.load(path).__dict__["_runnable_descriptor"]
    return sorted(
        slot.state_binding.state_dict_name
        for slot in descriptor.tensor_slots
        if slot.state_binding is not None
    )


@pytest.mark.smoke
def test_r77_dead_extra_state_universe_matches_live_and_binds(tmp_path: Path) -> None:
    """RED-now-fixed: dead-save universe == live-save universe; honest bind succeeds.

    Pre-fix the dead save silently dropped ``bn.num_batches_tracked`` and the honest
    tensor-only ``load_state_dict`` refused ``state_unexpected_key``. The run verdict
    stays the pre-existing extra-state ceiling (no ``verified`` without a capture-state
    comparison basis) IDENTICALLY in both lanes -- lane parity, never a wrong bind.
    """

    x = _x()

    live_model = ExtraStateBN().eval()
    tensor_only_sd = {
        name: value.clone()
        for name, value in live_model.state_dict().items()
        if isinstance(value, torch.Tensor)
    }
    live_trace = tl.trace(live_model, x, capture=_CAPTURE)
    live_path = tmp_path / "live.tlspec"
    live_trace.save(live_path, level="runnable", include_weights=False)
    live_universe = _slot_universe(live_path)
    assert "bn.num_batches_tracked" in live_universe

    dead_model = ExtraStateBN().eval()
    dead_trace = tl.trace(dead_model, x, capture=_CAPTURE)
    del dead_model
    gc.collect()
    dead_path = tmp_path / "dead.tlspec"
    dead_trace.save(dead_path, level="runnable", include_weights=False)
    dead_universe = _slot_universe(dead_path)

    assert dead_universe == live_universe
    assert "bn.num_batches_tracked" in dead_universe

    live_loaded = tl.load(live_path)
    live_loaded.load_state_dict(tensor_only_sd)
    live_result = live_loaded.run(inputs=x.clone())

    dead_loaded = tl.load(dead_path)
    dead_loaded.load_state_dict(tensor_only_sd)
    dead_result = dead_loaded.run(inputs=x.clone())

    # Lane parity: the pre-existing extra-state verdict ceiling is identical in both
    # lanes; the dead lane never diverges into refusal or a differing verdict class.
    assert dead_result.report.path_faithfulness is live_result.report.path_faithfulness
    assert dead_result.report.poisoned == live_result.report.poisoned


@pytest.mark.smoke
def test_r77_plain_bn_dead_universe_unchanged(tmp_path: Path) -> None:
    """Zero collateral: the tensor-only-state dead lane (r75 F2) declares identically."""

    x = _x()

    live_trace = tl.trace(PlainBN().eval(), x, capture=_CAPTURE)
    live_path = tmp_path / "live.tlspec"
    live_trace.save(live_path, level="runnable", include_weights=False)

    dead_model = PlainBN().eval()
    dead_trace = tl.trace(dead_model, x, capture=_CAPTURE)
    del dead_model
    gc.collect()
    dead_path = tmp_path / "dead.tlspec"
    dead_trace.save(dead_path, level="runnable", include_weights=False)

    live_universe = _slot_universe(live_path)
    assert _slot_universe(dead_path) == live_universe
    assert "bn.num_batches_tracked" in live_universe


@pytest.mark.smoke
def test_r77_unknown_universe_refuses_loudly(tmp_path: Path) -> None:
    """No capture-time record at all: the dead save refuses TYPED, never under-declares.

    A ``state_dict()`` that is not a mapping defeats both the value snapshot and the
    r77 universe record, so the declared slot universe is UNKNOWABLE; the save refuses
    with ``TorchLensIOError`` (mirroring the ``include_weights=True`` lane) instead of
    silently declaring a smaller universe.
    """

    x = _x()
    model = NonMappingStateDict().eval()
    trace = tl.trace(model, x, capture=_CAPTURE)
    del model
    gc.collect()
    with pytest.raises(TorchLensIOError, match="persistent-buffer state universe"):
        trace.save(tmp_path / "unknown.tlspec", level="runnable", include_weights=False)
