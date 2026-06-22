"""QSD-Transformer quantized spike-driven Transformer.

Paper: Quantized Spike-driven Transformer, Qiu et al. 2025.

The distinctive primitive is quantized spike-driven self-attention (Q-SDSA):
low-bit quantized projections produce binary spike activity, information-enhanced
LIF dynamics rectify the spike distribution, and spike-form Q/K/V interact by
masking and addition rather than vanilla dot-product softmax attention.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def fake_quantize(x: torch.Tensor, bits: int = 3) -> torch.Tensor:
    """Symmetrically fake-quantize activations to a low-bit grid.

    Parameters
    ----------
    x:
        Input activations.
    bits:
        Quantization bit width.

    Returns
    -------
    torch.Tensor
        Fake-quantized activations.
    """

    levels = 2 ** (bits - 1) - 1
    scale = x.detach().abs().amax().clamp_min(1e-6) / levels
    return torch.round(x / scale).clamp(-levels, levels) * scale


class IELIF(nn.Module):
    """Information-enhanced LIF spike activation."""

    def __init__(self, dim: int, threshold: float = 0.5) -> None:
        """Initialize adaptive threshold parameters."""

        super().__init__()
        self.threshold = nn.Parameter(torch.full((dim,), threshold))
        self.info_gate = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert membrane values to rectified spike rates.

        Parameters
        ----------
        x:
            Membrane activations.

        Returns
        -------
        torch.Tensor
            Straight-through binary spike tensor.
        """

        membrane = torch.sigmoid(self.info_gate(x)) * x
        soft_spike = torch.sigmoid(8.0 * (membrane - self.threshold))
        hard_spike = (soft_spike > 0.5).float()
        return hard_spike + soft_spike - soft_spike.detach()


class QSDSelfAttention(nn.Module):
    """Quantized spike-driven self-attention."""

    def __init__(self, dim: int = 32, heads: int = 4) -> None:
        """Initialize quantized projections and LIF units."""

        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.lif = IELIF(dim * 3)
        self.out = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply no-softmax Q-SDSA to patch tokens.

        Parameters
        ----------
        x:
            Patch token sequence.

        Returns
        -------
        torch.Tensor
            Spike-driven mixed token sequence.
        """

        batch, tokens, _ = x.shape
        qkv = self.lif(fake_quantize(self.qkv(x))).view(batch, tokens, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        key_mask = (k.sum(dim=1, keepdim=True) > 0).float()
        sparse_additive_context = (v * key_mask).sum(dim=1, keepdim=True)
        query_mask = (q > 0).float()
        out = query_mask * sparse_additive_context + v
        return self.out(out.reshape(batch, tokens, -1))


class QSDTransformer(nn.Module):
    """Compact quantized spike-driven vision Transformer."""

    def __init__(self, dim: int = 32) -> None:
        """Initialize the patch stem and transformer block."""

        super().__init__()
        self.patch = nn.Conv2d(3, dim, kernel_size=4, stride=4)
        self.attn = QSDSelfAttention(dim)
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 2), IELIF(dim * 2), nn.Linear(dim * 2, dim))
        self.head = nn.Linear(dim, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify a compact image.

        Parameters
        ----------
        x:
            RGB image tensor.

        Returns
        -------
        torch.Tensor
            Class logits.
        """

        tokens = self.patch(x).flatten(2).transpose(1, 2)
        tokens = tokens + self.attn(self.norm(tokens))
        tokens = tokens + self.mlp(self.norm(tokens))
        return self.head(tokens.mean(dim=1))


def build() -> nn.Module:
    """Build compact QSD-Transformer.

    Returns
    -------
    nn.Module
        Random-init compact QSD-Transformer.
    """

    return QSDTransformer()


def example_input() -> torch.Tensor:
    """Return an image tensor.

    Returns
    -------
    torch.Tensor
        Compact RGB image.
    """

    return torch.randn(1, 3, 16, 16)


MENAGERIE_ENTRIES = [
    ("qsd_transformer", "build", "example_input", "2025", "spiking/quantized-transformer")
]
