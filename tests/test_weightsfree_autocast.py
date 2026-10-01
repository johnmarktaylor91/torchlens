"""W1-AC null-autocast shim rows (memo 8.1 item 11 / D18 / defect L8).

Under structure-only meta admission ONLY, exactly the call (meta device AND
``enabled=False``) maps to ``nullcontext`` — semantically null: disabled
autocast performs no dtype coercion and no numerics exist on meta, yet torch
validates the device string even for ``enabled=False`` (measured on torch
2.13: ``RuntimeError: unsupported scalarType``), which blocks the flagship
llama family on the declared transformers 4.x band. ``enabled=True`` on meta
still raises; ordinary captures and plain code see the shipped ``torch.autocast``.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.options import CaptureOptions

pytestmark = pytest.mark.smoke


class AutocastGuardModel(nn.Module):
    """The HF 4.x LlamaRotaryEmbedding shape: a disabled autocast scope
    around a value-free computation."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device_type = x.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            y = self.fc(x)
        return torch.relu(y)


def _torch_raises_on_meta_disabled_autocast() -> bool:
    try:
        with torch.autocast(device_type="meta", enabled=False):
            pass
    except RuntimeError:
        return True
    return False


def test_shim_unlocks_disabled_autocast_on_admitted_meta() -> None:
    """The L8 unlock: the exact null call works inside an admitted forward."""

    if not _torch_raises_on_meta_disabled_autocast():
        pytest.skip("this torch build accepts meta autocast; the shim is inert")
    with torch.device("meta"):
        model = AutocastGuardModel()
    model.eval()
    trace = tl.trace(
        model, torch.empty(2, 4, device="meta"), capture=CaptureOptions(structure_only=True)
    )
    assert trace.outcome.status.value == "complete"
    funcs = [layer.func_name for layer in trace.layer_list]
    assert "linear" in funcs and "relu" in funcs


def test_enabled_autocast_on_meta_still_refuses() -> None:
    """``enabled=True`` on meta is NOT nullified: torch's own refusal stands,
    classified into the substrate family by W1-CLS (torch/amp provenance)."""

    class EnabledAutocast(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            with torch.autocast(device_type="meta", enabled=True):
                return self.fc(x)

    with torch.device("meta"):
        model = EnabledAutocast()
    model.eval()
    with pytest.raises(Exception) as excinfo:
        tl.trace(
            model, torch.empty(2, 4, device="meta"), capture=CaptureOptions(structure_only=True)
        )
    code = getattr(excinfo.value, "fields", {}).get("code")
    assert code == "structure_only_substrate_mismatch" or isinstance(excinfo.value, RuntimeError)


def test_shim_is_scoped_to_the_admitted_forward() -> None:
    """Outside an admitted capture the shipped torch.autocast is untouched."""

    if not _torch_raises_on_meta_disabled_autocast():
        pytest.skip("this torch build accepts meta autocast; nothing to scope")
    original = torch.autocast
    with torch.device("meta"):
        model = AutocastGuardModel()
    model.eval()
    tl.trace(model, torch.empty(2, 4, device="meta"), capture=CaptureOptions(structure_only=True))
    assert torch.autocast is original
    with pytest.raises(RuntimeError), torch.autocast(device_type="meta", enabled=False):
        pass


def test_real_capture_autocast_untouched() -> None:
    """Ordinary captures never see the shim: the disabled-autocast scope on a
    real device runs through the shipped class."""

    model = AutocastGuardModel()
    model.eval()
    trace = tl.trace(model, torch.randn(2, 4))
    assert trace.outcome.status.value == "complete"
