"""Deployment envelope (lane F37): bitsandbytes 4/8-bit capture on CPU.

Quantized capture is DISCLOSED-DEGRADED, never silently wrong: outputs match
the bare forward bit-for-bit at fp32 tolerance, the quantized-module warning
fires at entry, the compat row reads pass/warning, and the validation
tripwires keep their honest verdicts on out-of-contract quantization-state
reads (8-bit raises the metadata invariant; 4-bit replay reports False).
The resilient state restore lets validation COMPLETE instead of dying in
bitsandbytes' load_state_dict.
"""

from __future__ import annotations

import warnings

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens._deploy_env import restore_state_dict_resilient
from torchlens.compat import report

bnb = pytest.importorskip("bitsandbytes")


def _quantized_net(kind: str) -> nn.Module:
    class QNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            if kind == "8bit":
                self.fc1 = bnb.nn.Linear8bitLt(16, 32, has_fp16_weights=False)
                self.fc2 = bnb.nn.Linear8bitLt(32, 4, has_fp16_weights=False)
            else:
                self.fc1 = bnb.nn.Linear4bit(16, 32, compute_dtype=torch.float32)
                self.fc2 = bnb.nn.Linear4bit(32, 4, compute_dtype=torch.float32)
            self.act = nn.ReLU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc2(self.act(self.fc1(x)))

    torch.manual_seed(0)
    return QNet().eval().to("cpu")


@pytest.mark.heavy
@pytest.mark.parametrize("kind", ["8bit", "4bit"])
def test_quantized_capture_has_value_parity_and_discloses(kind: str) -> None:
    """Captured outputs equal the bare forward; degradation is disclosed at entry."""

    model = _quantized_net(kind)
    x = torch.randn(2, 16)
    with torch.no_grad():
        reference = model(x)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trace = tl.trace(model, x)
    assert any("quantized" in str(w.message).lower() for w in caught)
    final_ops = [op for op in trace.ops.values() if op.is_output_parent]
    assert final_ops and torch.allclose(final_ops[-1].out, reference, atol=1e-6)


def test_bitsandbytes_compat_row_reads_pass_warning() -> None:
    """Detection reads pass/warning (verified capture, disclosed degradation)."""

    row = report(_quantized_net("8bit"), torch.randn(1, 16)).row("bitsandbytes_8bit_4bit")
    assert (row.detected, row.status, row.severity) == (True, "pass", "warning")
    clean_row = report(nn.Linear(4, 2), torch.randn(1, 4)).row("bitsandbytes_8bit_4bit")
    assert (clean_row.detected, clean_row.severity) == (False, "ok")


@pytest.mark.heavy
@pytest.mark.filterwarnings("ignore:TorchLens detected quantized submodules")
@pytest.mark.filterwarnings("ignore:TorchLens found tensor arguments with no graph")
def test_quantized_validation_completes_with_honest_verdicts() -> None:
    """The resilient restore lets validation finish; the tripwires stay armed.

    8-bit forwards read quantization state (state.CB/SCB) that is genuinely
    outside the known-sources contract, so the metadata invariant HONESTLY
    raises; 4-bit's custom-function replay reports False. Neither dies in
    bitsandbytes' load_state_dict anymore.
    """

    from torchlens.errors import MetadataInvariantError

    x = torch.randn(2, 16)
    with pytest.raises(MetadataInvariantError):
        tl.validate(_quantized_net("8bit"), x, scope="forward")
    assert tl.validate(_quantized_net("4bit"), x, scope="forward") is False


class _RefusesRoundTrip(nn.Module):
    """Module whose load_state_dict refuses its own state dict (bnb-shaped)."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 3)

    def load_state_dict(self, state_dict, strict=True, assign=False):  # type: ignore[override]
        raise RuntimeError("loading a quantized checkpoint is not supported")


def test_resilient_restore_copies_in_place_when_load_refuses() -> None:
    """Fallback restores by exact-slot copy when load_state_dict refuses."""

    model = _RefusesRoundTrip()
    saved = {k: v.detach().clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        model.linear.weight.add_(1.0)
    restore_state_dict_resilient(model, saved)
    assert torch.equal(model.linear.weight, saved["linear.weight"])


def test_resilient_restore_reraises_on_unrestorable_drift() -> None:
    """A saved key with no live slot and a drifted current value re-raises."""

    model = _RefusesRoundTrip()
    saved = {k: v.detach().clone() for k, v in model.state_dict().items()}
    saved["linear.weight.sidecar"] = torch.ones(2)  # slotless, absent live
    with pytest.raises(RuntimeError, match="quantized checkpoint"):
        restore_state_dict_resilient(model, saved)
