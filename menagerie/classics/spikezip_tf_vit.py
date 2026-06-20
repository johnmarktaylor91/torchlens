"""SpikeZIP-TF: Spike-Equivalent Vision Transformer (conversion-based SNN).

Lin et al., "SpikeZIP-TF: Conversion is All You Need for Transformer-based SNN".
Paper: https://arxiv.org/abs/2406.03470

SpikeZIP-TF is an ANN->SNN *conversion* method that makes a quantized ViT and
its spiking counterpart EXACTLY equivalent.  The distinctive trick is replacing
each ANN operator by a spike-equivalent one driven by a multi-level
(quantized) spiking neuron (ST-BIF+): the accumulated spike output over T
timesteps equals the quantized-ReLU (Q-ReLU) output of the source ANN.  It
introduces Spike-Equivalent Self-Attention (SESA), Spike-Softmax and
Spike-LayerNorm while keeping ANN/SNN operator equivalence.

This faithful random-init reimplementation keeps the *family-distinctive*
structure: a standard ViT (patch embed -> N transformer blocks -> head) whose
ReLU/GELU activations are replaced by MULTI-LEVEL spiking activations (the
quantized ST-BIF+ stand-in) and whose attention is the spike-equivalent
self-attention.  Unlike the directly-trained spiking ViT, the time dimension is
the quantization level rather than an explicit unrolled axis, so we keep the
graph as a single static ViT with spiking (quantized) activations everywhere a
ReLU/GELU would be -- exactly the "ViT with activations swapped for spiking
operators" reading.  Sizes are tiny so the graph renders quickly.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from menagerie.classics._snn_neurons import SpikingActivation


class _PatchEmbed(nn.Module):
    """Conv patch embedding -> flatten to tokens."""

    def __init__(self, in_ch: int = 3, img_size: int = 32, patch: int = 8, dim: int = 96) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)
        self.num_tokens = (img_size // patch) ** 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # (B, D, h, w)
        return x.flatten(2).transpose(1, 2)  # (B, N, D)


class _SpikeEquivalentAttention(nn.Module):
    """Spike-Equivalent Self-Attention (SESA): attention with spiking activations.

    Standard scaled-dot-product attention but the softmax is replaced by a
    spike-softmax stand-in (positive multi-level spiking activation, normalised)
    and the projections feed through multi-level spiking activations.
    """

    def __init__(self, dim: int = 96, heads: int = 3, levels: int = 4) -> None:
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.spike_attn = SpikingActivation(threshold=0.25, levels=levels)
        self.spike_proj = SpikingActivation(threshold=0.5, levels=levels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, N, hd)
        scores = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim**-0.5)
        # spike-softmax stand-in: multi-level positive spiking + row-normalise
        attn = self.spike_attn(scores)
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-6)
        out = torch.matmul(attn, v)  # (B, H, N, hd)
        out = out.transpose(1, 2).reshape(b, n, self.dim)
        out = self.spike_proj(self.proj(out))
        return out


class _SpikeEquivalentMLP(nn.Module):
    """MLP block with the GELU replaced by a multi-level spiking activation."""

    def __init__(self, dim: int = 96, hidden: int = 192, levels: int = 4) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = SpikingActivation(threshold=0.5, levels=levels)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class _SpikeEquivalentBlock(nn.Module):
    """ViT block: LayerNorm + SESA + LayerNorm + spiking MLP (residual)."""

    def __init__(self, dim: int = 96, heads: int = 3, hidden: int = 192, levels: int = 4) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _SpikeEquivalentAttention(dim, heads, levels)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _SpikeEquivalentMLP(dim, hidden, levels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SpikeZipTFViT(nn.Module):
    """Spike-equivalent ViT (SpikeZIP-TF style, multi-level spiking activations)."""

    def __init__(
        self,
        in_ch: int = 3,
        img_size: int = 32,
        patch: int = 8,
        dim: int = 96,
        depth: int = 2,
        heads: int = 3,
        hidden: int = 192,
        num_classes: int = 10,
        levels: int = 4,
    ) -> None:
        super().__init__()
        self.patch_embed = _PatchEmbed(in_ch, img_size, patch, dim)
        n = self.patch_embed.num_tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n + 1, dim))
        self.blocks = nn.ModuleList(
            [_SpikeEquivalentBlock(dim, heads, hidden, levels) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(x)  # (B, N, D)
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1) + self.pos_embed
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)
        return self.head(tokens[:, 0])


def build_spikezip_tf_vit() -> nn.Module:
    """Build the SpikeZIP-TF spike-equivalent ViT (random init, multi-level spikes)."""
    return SpikeZipTFViT(
        in_ch=3,
        img_size=32,
        patch=8,
        dim=96,
        depth=2,
        heads=3,
        hidden=192,
        num_classes=10,
        levels=4,
    )


def example_input() -> torch.Tensor:
    """Example RGB image ``(1, 3, 32, 32)`` for the spike-equivalent ViT."""
    return torch.randn(1, 3, 32, 32)


MENAGERIE_ENTRIES = [
    (
        "SpikeZIP-TF (spike-equivalent ViT, multi-level spiking activations)",
        "build_spikezip_tf_vit",
        "example_input",
        "2024",
        "DC",
    ),
]
