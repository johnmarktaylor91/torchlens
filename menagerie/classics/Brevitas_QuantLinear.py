"""Brevitas QuantLinear: quantization-aware linear layer.

Source: Brevitas, a PyTorch research library for quantization-aware training
from Xilinx Research Labs.

Brevitas quantized layers expose independently configurable input, weight, bias,
and output quantizers.  This compact reconstruction keeps the load-bearing
QuantLinear behavior: learned affine weights pass through signed low-bit
fake-quantization, activations pass through unsigned low-bit fake-quantization,
and the output can be quantized again to model QAT/export graphs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _ste_round(x: torch.Tensor) -> torch.Tensor:
    """Round with a straight-through estimator.

    Parameters
    ----------
    x:
        Tensor to round.

    Returns
    -------
    torch.Tensor
        Rounded tensor with identity gradient.
    """

    return x + (torch.round(x) - x).detach()


class UniformFakeQuant(nn.Module):
    """Uniform affine fake quantizer."""

    def __init__(self, bit_width: int, signed: bool) -> None:
        """Initialize a fake quantizer.

        Parameters
        ----------
        bit_width:
            Number of quantization bits.
        signed:
            Whether to use a signed integer range.
        """

        super().__init__()
        self.bit_width = bit_width
        self.signed = signed
        if signed:
            self.qmin = -(2 ** (bit_width - 1))
            self.qmax = 2 ** (bit_width - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2**bit_width - 1
        self.log_scale = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply uniform fake quantization.

        Parameters
        ----------
        x:
            Floating-point tensor.

        Returns
        -------
        torch.Tensor
            Fake-quantized tensor.
        """

        learned = F.softplus(self.log_scale) + 1e-4
        observed = x.detach().abs().amax().clamp_min(1e-4)
        scale = torch.maximum(learned, observed / max(abs(self.qmin), abs(self.qmax)))
        q = torch.clamp(_ste_round(x / scale), self.qmin, self.qmax)
        return q * scale


class QuantLinearCompact(nn.Module):
    """Compact Brevitas-style QuantLinear block."""

    def __init__(self, in_features: int = 16, out_features: int = 8, bit_width: int = 4) -> None:
        """Initialize QuantLinear.

        Parameters
        ----------
        in_features:
            Input feature count.
        out_features:
            Output feature count.
        bit_width:
            Quantization bit width for input, weight, and output quantizers.
        """

        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.05)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.input_quant = UniformFakeQuant(bit_width, signed=False)
        self.weight_quant = UniformFakeQuant(bit_width, signed=True)
        self.output_quant = UniformFakeQuant(bit_width, signed=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a fake-quantized linear projection.

        Parameters
        ----------
        x:
            Floating-point input tensor.

        Returns
        -------
        torch.Tensor
            Fake-quantized output tensor.
        """

        q_input = self.input_quant(x)
        q_weight = self.weight_quant(self.weight)
        out = F.linear(q_input, q_weight, self.bias)
        return self.output_quant(out)


def build_Brevitas_QuantLinear() -> nn.Module:
    """Build compact Brevitas QuantLinear.

    Returns
    -------
    nn.Module
        Random-init quantized linear layer.
    """

    return QuantLinearCompact()


def example_input() -> torch.Tensor:
    """Return a small activation batch.

    Returns
    -------
    torch.Tensor
        Example activation tensor.
    """

    return torch.rand(2, 16)


MENAGERIE_ENTRIES = [
    (
        "Brevitas QuantLinear (QAT fake-quantized linear layer)",
        "build_Brevitas_QuantLinear",
        "example_input",
        "2018",
        "E7",
    )
]
