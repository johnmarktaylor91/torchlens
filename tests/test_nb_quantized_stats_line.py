"""Quantized tensors print a reason token on every stats line.

A quantized payload holds integer storage plus a scale and zero point, so
the kernel computes no moments for it. The line must say so with the
ruled token instead of a generic dtype refusal, on every surface that
prints the line: ``glance``, the record repr and the notebook card.
"""

from __future__ import annotations

import torch
from torch import nn

import torchlens as tl
from torchlens.notebook.cards import _stats_line
from torchlens.stats import glance, tensor_stats
from torchlens.stats._stats_render import render_core_line

TOKEN = "quantized: stats not computed"


def _per_tensor() -> torch.Tensor:
    return torch.quantize_per_tensor(torch.randn(4, 4), 0.1, 0, torch.qint8)


def _per_channel() -> torch.Tensor:
    return torch.quantize_per_channel(
        torch.randn(3, 4),
        torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64),
        torch.tensor([0, 0, 0]),
        0,
        torch.quint8,
    )


def test_glance_prints_the_quantized_token() -> None:
    """Per-tensor and per-channel quantization both print the token."""

    for tensor in (_per_tensor(), _per_channel()):
        line = glance(tensor)
        assert TOKEN in line, line
        assert "unsupported dtype" not in line
        # Identity stays on the line: dtype, shape and element count.
        assert str(tensor.dtype).removeprefix("torch.") in line
        assert f"n={tensor.numel()}" in line


def test_kernel_reason_and_unicode_style_carry_the_token() -> None:
    """Both byte-checked renderings carry the same token."""

    stats = tensor_stats(_per_tensor())
    assert TOKEN in render_core_line(stats, style="ascii")
    assert TOKEN in render_core_line(stats, style="unicode")


def test_other_unsupported_dtypes_keep_their_reason() -> None:
    """Only quantized payloads change; other unsupported dtypes say why."""

    fp8 = torch.zeros(4).to(torch.float8_e4m3fn)
    line = glance(fp8)
    assert "unsupported dtype torch.float8_e4m3fn" in line
    assert TOKEN not in line


class _Quantizes(nn.Module):
    """Model whose output is a quantized tensor."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize a relu output."""
        return torch.quantize_per_tensor(torch.relu(x), 0.1, 0, torch.qint8)


def test_record_repr_and_card_print_the_token() -> None:
    """The captured record's repr and its notebook card print the token."""

    trace = tl.trace(_Quantizes(), torch.randn(3, 3))
    label = next(label for label in trace.layer_labels if "quantize" in label)
    layer = trace[label]
    assert TOKEN in repr(layer)
    assert TOKEN in _stats_line(layer.out)
    assert TOKEN in layer._repr_html_()
