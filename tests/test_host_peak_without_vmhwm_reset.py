"""Host peak tests guard on the high-water reset itself, and its absence stays covered.

The CPU peak tests need the kernel to reset the resident high-water mark
(``VmHWM``) when ``5`` is written to ``/proc/self/clear_refs``, a Linux
feature newer than some cluster kernels. On such a kernel the file exists and
is writable, but the write fails, so a guard that only asked whether the file
was writable ran the tests without the capability they measure. The guard is
now the product's own probe, and the product's behaviour without the reset
(``resident_basis`` ``"process_lifetime"``, scoped growth left out of the
forward peak) is driven here on every host.
"""

from __future__ import annotations

import os

import pytest
import torch
from _fresh_plant import (
    PLANT_BYTES,
    VMHWM_RESET_UNSUPPORTED,
    needs_vmhwm_reset,
    touch_fresh_plant,
    vmhwm_reset_supported,
)
from torch import nn

import torchlens as tl
from torchlens._capture import _forward_helpers, peak_memory
from torchlens.options import CaptureOptions

MARGIN = 2**20
HOST = CaptureOptions(measure_host_peak_memory=True)


class _Planting(nn.Module):
    """Writes and frees a fresh plant inside its forward."""

    def __init__(self) -> None:
        """Build the one linear layer the forward returns through."""

        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Touch and release the plant, then apply the layer."""

        touch_fresh_plant().close()
        return self.fc(x)


def test_reset_guard_follows_the_product_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """The guard reads the product's reset outcome and names the missing capability."""

    vmhwm_reset_supported.cache_clear()
    monkeypatch.setattr(peak_memory, "reset_peak_rss", lambda: False)
    try:
        assert vmhwm_reset_supported() is False
    finally:
        vmhwm_reset_supported.cache_clear()
    assert "VmHWM reset unsupported on this kernel" in VMHWM_RESET_UNSUPPORTED
    assert needs_vmhwm_reset.kwargs["reason"] == VMHWM_RESET_UNSUPPORTED


@pytest.mark.skipif(
    not os.path.exists("/proc/self/status"),
    reason="reads the resident set from /proc/self/status (Linux only)",
)
def test_capture_without_the_reset_reports_process_lifetime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the reset the pair says process lifetime and the freed plant is not claimed."""

    monkeypatch.setattr(_forward_helpers, "reset_peak_rss", lambda: False)
    trace = tl.trace(_Planting(), torch.randn(2, 4), capture=HOST)
    try:
        pair = trace.cost.forward_peak_memory_pair
        assert pair is not None
        assert pair["resident_basis"] == "process_lifetime"
        assert pair["resident"] is not None
        peak = trace.cost.forward_peak_memory
        assert peak is None or int(peak) < PLANT_BYTES - MARGIN
    finally:
        trace.cleanup()
