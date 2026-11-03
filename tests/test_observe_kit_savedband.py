"""Observe items 7-8: saved-band decomposition + trace-global first-save identity.

The parameter phantom: most of the bytes autograd retains on real models are
PARAMETER storages already counted in the parameter band, so a naive
parameter+autograd_saved stack double-plots them. The storage-class
decomposition (saved_parameter / saved_buffer / saved_activation, keyed on
``untyped_storage().data_ptr()`` on BOTH sides) and the trace-global
first-save counter (``newly_saved_bytes``) make the stack honest. The
replacement oracles here compute truth a DIFFERENT way: torch's own
``saved_tensors_hooks`` is the independent storage-identity oracle.
"""

from __future__ import annotations

import torch
from torch import nn

import torchlens as tl


class _MLP(nn.Module):
    """Two linears around a tanh: params + activations both get saved."""

    def __init__(self) -> None:
        super().__init__()
        self.l1 = nn.Linear(8, 8)
        self.l2 = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Linear -> tanh -> linear."""

        return self.l2(torch.tanh(self.l1(x)))


class _BufferScale(nn.Module):
    """Multiplies by a registered buffer so autograd saves BUFFER storage."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("scale", torch.full((4,), 2.0))
        self.linear = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The mul's backward saves ``scale`` (a buffer storage)."""

        return self.linear(x) * self.scale


def _traced_bands(model: nn.Module, x: torch.Tensor) -> tuple[tl.Trace, dict[str, dict[str, int]]]:
    """Capture with backward_ready and return (trace, per-raw-label bands)."""

    captured = tl.trace(
        model,
        x,
        capture=tl.options.CaptureOptions(backward_ready=True, save_mode="reference"),
    )
    return captured, dict(captured.__dict__.get("_autograd_saved_bands", {}))


def test_param_saved_fixed_while_activation_saved_scales_with_batch() -> None:
    """The decomposition is USEFUL, not merely correct: param slice is fixed.

    'N MB held for backward, of which M MB scales with your batch' is the
    sentence the split buys: the param-saved slice must be byte-identical
    across batch sizes while activation-saved grows.
    """

    def _totals(batch: int) -> tuple[int, int]:
        model = _MLP()
        captured, bands = _traced_bands(model, torch.randn(batch, 8, requires_grad=True))
        try:
            return (
                sum(band["saved_parameter"] for band in bands.values()),
                sum(band["saved_activation"] for band in bands.values()),
            )
        finally:
            captured.cleanup()

    small_param, small_act = _totals(2)
    large_param, large_act = _totals(8)
    assert small_param == large_param
    assert large_act > small_act
