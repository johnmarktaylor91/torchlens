"""The two round-2 options codes that shipped without a provoking test.

``record_capture_option_unsupported``: ``tl.record`` refuses a ``CaptureOptions`` field
the recorder has no door for instead of ignoring it. ``capture_nonfinite_detected``:
``on_nonfinite='warn'`` tracks non-finite outputs and warns once after the capture.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.errors import ArgumentTypeError, TorchLensWarning


class _NaN(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(x - 10.0)


def test_record_refuses_capture_fields_it_does_not_read() -> None:
    """A CaptureOptions field outside the record door's set refuses typed."""

    with pytest.raises(ArgumentTypeError) as info:
        tl.record(
            nn.Linear(2, 2),
            torch.randn(1, 2),
            save=tl.func("linear"),
            capture=tl.options.CaptureOptions(structure_only=True),
        )
    assert info.value.fields["code"] == "record_capture_option_unsupported"
    assert "structure_only" in str(info.value)


def test_on_nonfinite_warn_emits_one_capture_nonfinite_detected_warning() -> None:
    """``on_nonfinite='warn'`` completes the capture and warns once with the code."""

    with pytest.warns(TorchLensWarning) as caught:
        trace = tl.trace(
            _NaN(), torch.zeros(2), capture=tl.options.CaptureOptions(on_nonfinite="warn")
        )
    codes = [
        w.message.fields.get("code") for w in caught if isinstance(w.message, TorchLensWarning)
    ]
    assert codes.count("capture_nonfinite_detected") == 1
    assert len(trace) >= 1
