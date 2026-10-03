"""A06 capture-options truth: AMP-scaled captured gradients are disclosed.

WALKTHROUGH list-A row 11 (AMP sub-item, riding A06): gradients captured from
a backward run under ``torch.amp.GradScaler`` carry the loss scale (~2**16 at
the default ``init_scale``) -- TorchLens records the gradients that really
flowed, and the scaler unscales only leaf ``param.grad``, never the
intermediate gradients ``gradient_flow_audit`` reads -- so the audit
false-flagged EVERY op exploding. The audit now takes
``grad_scale=scaler.get_scale()`` (disclosed in ``df.attrs``), refuses junk
scales typed (``grad_scale_invalid``), and an all-exploding result carries a
deterministic AMP hint instead of silence.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.options import CaptureOptions

SCALE = 65536.0  # torch.amp.GradScaler default init_scale = 2**16


class ThreeStep(nn.Module):
    """fc1 -> relu -> fc2."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


def _scaled_backward_trace() -> tl.Trace:
    torch.manual_seed(20260826)
    log = tl.trace(
        ThreeStep(),
        torch.randn(2, 4),
        capture=CaptureOptions(save_grads=True, backward_ready=True),
        save_mode="reference",
    )
    loss = list(log)[-1].out.sum() * SCALE  # what GradScaler.scale(loss) does
    loss.backward()
    return log


def test_scaled_backward_false_flags_and_hints() -> None:
    """Without the kwarg every op explodes and the AMP hint is disclosed."""

    pytest.importorskip("pandas")
    from torchlens.debug import gradient_flow_audit

    frame = gradient_flow_audit(_scaled_backward_trace())
    measurable = len(frame) - int(frame.attrs.get("unavailable", 0))
    assert measurable > 0
    assert frame.attrs["exploding"] == measurable
    assert "GradScaler" in frame.attrs["all_exploding_hint"]
    assert "grad_scale" not in frame.attrs


def test_grad_scale_audits_in_unscaled_units() -> None:
    """grad_scale=scaler.get_scale() clears the false flags and is disclosed."""

    pytest.importorskip("pandas")
    from torchlens.debug import gradient_flow_audit

    frame = gradient_flow_audit(_scaled_backward_trace(), grad_scale=SCALE)
    assert frame.attrs["exploding"] == 0, (
        "unscaled toy gradients must not flag exploding once the disclosed "
        "loss scale is divided out"
    )
    assert frame.attrs["grad_scale"] == SCALE
    assert "all_exploding_hint" not in frame.attrs


@pytest.mark.parametrize("bad_scale", [0.0, -1.0, float("inf"), float("nan")])
def test_grad_scale_junk_refuses_typed(bad_scale: float) -> None:
    """Non-positive / non-finite scales refuse with grad_scale_invalid."""

    pytest.importorskip("pandas")
    from torchlens._errors import InvalidArgumentError
    from torchlens.debug import gradient_flow_audit

    log = _scaled_backward_trace()
    with pytest.raises(InvalidArgumentError) as excinfo:
        gradient_flow_audit(log, grad_scale=bad_scale)
    assert excinfo.value.fields["code"] == "grad_scale_invalid"


def test_unscaled_backward_stays_clean_without_hint() -> None:
    """A plain backward neither flags everything nor emits the AMP hint."""

    pytest.importorskip("pandas")
    from torchlens.debug import gradient_flow_audit

    torch.manual_seed(20260826)
    log = tl.trace(
        ThreeStep(),
        torch.randn(2, 4),
        capture=CaptureOptions(save_grads=True, backward_ready=True),
        save_mode="reference",
    )
    list(log)[-1].out.sum().backward()
    frame = gradient_flow_audit(log)
    assert frame.attrs["exploding"] == 0
    assert "all_exploding_hint" not in frame.attrs
