"""QKFormer: spiking transformer with Q-K Token Attention (no softmax).

Zhou et al. (2023/2024), "QKFormer: Hierarchical Spiking Transformer using
Q-K Attention".  NeurIPS 2024.  arXiv:2403.16552.
Source: https://github.com/zhouchenlin2096/QKFormer

Distinctive primitives:
  1. QK TOKEN ATTENTION (spiking linear attention): spiking neurons produce binary
     (0/1) spike outputs from Q and K projections.  Attention is computed as
     softmax-free dot product of spike-Q and spike-K: Attn = spike(Q) @ spike(K)^T
     (no softmax, no value projection -- QK-only attention).  The output is
     spike_Q * (spike_K^T @ V) / N (linear attention form via associativity).
  2. SURROGATE-GRADIENT SPIKES: neurons threshold at 1.0 with heaviside in forward,
     surrogate gradient in backward.  For the atlas we use a simple
     piecewise-linear surrogate (SpikingActivation-like: clamp to [0,1]).
  3. HIERARCHICAL SPIKING VISION TRANSFORMER: patch embedding, multiple stages
     with downsampling (spiking convolution) and QK spiking attention blocks.

Compact config: d_model=32, n_heads=2, T=1 (single timestep), img_size=8, patch=2.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================
# Surrogate-gradient spike activation
# ==============================================================


class SpikeAct(torch.autograd.Function):
    """Heaviside spike in forward; piecewise-linear surrogate in backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return (x >= 1.0).float()

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> torch.Tensor:
        (x,) = ctx.saved_tensors
        # Surrogate: rectangular window in [-0.5, 0.5] around threshold
        sg = (x.abs() < 0.5).float()
        return grad * sg


class SpikingNeuron(nn.Module):
    """Leaky integrate-and-fire neuron with surrogate gradient spike."""

    def __init__(self) -> None:
        super().__init__()
        self.thresh = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: any shape -> binary spike (0 or 1) same shape."""
        return SpikeAct.apply(x)


# ==============================================================
# QK Spiking Attention
# ==============================================================


class QKSpikingAttention(nn.Module):
    """Q-K Token Attention: spike(Q) @ spike(K)^T -- softmax-free spiking attention.

    Linear attention form: output_i = sum_j spike_Q_i * spike_K_j * V_j / N
    Equivalently:  output = spike_Q * (spike_K^T @ V) / N
    (same as standard linear attention but with binary Q, K).
    """

    def __init__(self, d_model: int = 32, n_heads: int = 2) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        # Batch normalisation before spiking (common in spiking transformers)
        self.bn_q = nn.BatchNorm1d(d_model)
        self.bn_k = nn.BatchNorm1d(d_model)
        self.bn_v = nn.BatchNorm1d(d_model)
        self.proj_q = nn.Linear(d_model, d_model, bias=False)
        self.proj_k = nn.Linear(d_model, d_model, bias=False)
        self.proj_v = nn.Linear(d_model, d_model, bias=False)
        self.spike = SpikingNeuron()
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, d_model) -> (B, N, d_model)"""
        B, N, d = x.shape
        H = self.n_heads
        S = self.d_head

        # BN expects (B, C) or (B, C, L); flatten to (B*N, d) for BN
        xf = x.reshape(B * N, d)
        q = self.spike(self.bn_q(self.proj_q(xf))).view(B, N, H, S)
        k = self.spike(self.bn_k(self.proj_k(xf))).view(B, N, H, S)
        v = self.bn_v(self.proj_v(xf)).view(B, N, H, S)

        # Linear attention: (B, H, S, S) context matrix = k^T @ v
        # q: (B, N, H, S) -> (B, H, N, S), k/v same
        q = q.permute(0, 2, 1, 3)  # (B, H, N, S)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        # Context: sum_j k_j^T v_j = (B, H, S, S)
        kv = k.transpose(-2, -1) @ v  # (B, H, S, S)
        # Output: q @ kv / N
        out = (q @ kv) / N  # (B, H, N, S)
        out = out.permute(0, 2, 1, 3).reshape(B, N, d)
        return self.out(out)


# ==============================================================
# QKFormer block
# ==============================================================


class QKFormerBlock(nn.Module):
    """QKFormer block: BN + QK spiking attention + BN + spiking FFN."""

    def __init__(self, d_model: int = 32, n_heads: int = 2) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm1d(d_model)
        self.attn = QKSpikingAttention(d_model, n_heads)
        self.bn2 = nn.BatchNorm1d(d_model)
        self.ff1 = nn.Linear(d_model, d_model * 2, bias=False)
        self.ff2 = nn.Linear(d_model * 2, d_model, bias=False)
        self.spike = SpikingNeuron()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, d_model)"""
        B, N, d = x.shape
        # Attention
        h = x.reshape(B * N, d)
        h = self.spike(self.bn1(h)).view(B, N, d)
        x = x + self.attn(h)
        # FFN
        h = x.reshape(B * N, d)
        h = self.spike(self.bn2(h)).view(B, N, d)
        h = self.spike(self.ff1(h))  # spiking FFN
        h = self.ff2(h)
        x = x + h
        return x


class QKFormerImageNet(nn.Module):
    """Compact QKFormer hierarchical spiking ViT.

    Patch embed -> [QKFormerBlock] -> [DownsampleBlock + QKFormerBlock] -> cls head.
    Simplified: 2 stages.
    """

    def __init__(
        self,
        img_size: int = 8,
        patch_size: int = 2,
        d_model: int = 32,
        n_heads: int = 2,
        n_classes: int = 10,
        n_blocks: int = 2,
    ) -> None:
        super().__init__()
        n_patches = (img_size // patch_size) ** 2
        d_in = 3 * patch_size * patch_size
        self.patch_size = patch_size
        self.img_size = img_size
        self.n_patches = n_patches
        # Patch embedding
        self.patch_emb = nn.Conv2d(
            3, d_model, kernel_size=patch_size, stride=patch_size, bias=False
        )
        self.patch_bn = nn.BatchNorm2d(d_model)
        # Stage 1 blocks
        self.stage1 = nn.ModuleList([QKFormerBlock(d_model, n_heads) for _ in range(n_blocks)])
        # Downsample (halve spatial dims)
        d2 = d_model * 2
        self.down = nn.Conv2d(d_model, d2, kernel_size=2, stride=2, bias=False)
        self.down_bn = nn.BatchNorm2d(d2)
        # Stage 2 blocks
        self.stage2 = nn.ModuleList([QKFormerBlock(d2, n_heads) for _ in range(n_blocks)])
        # Classification head
        self.cls = nn.Linear(d2, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, img_size, img_size) -> (B, n_classes)"""
        B = x.shape[0]
        # Patch embed
        h = F.silu(self.patch_bn(self.patch_emb(x)))  # (B, d, P, P)
        _, d, Ph, Pw = h.shape
        h = h.flatten(2).transpose(1, 2)  # (B, n_patches, d)
        # Stage 1
        for blk in self.stage1:
            h = blk(h)
        # Downsample
        h = h.transpose(1, 2).view(B, d, Ph, Pw)  # (B, d, Ph, Pw)
        h = F.silu(self.down_bn(self.down(h)))  # (B, d2, Ph//2, Pw//2)
        d2, Ph2, Pw2 = h.shape[1], h.shape[2], h.shape[3]
        h = h.flatten(2).transpose(1, 2)  # (B, n_patches//4, d2)
        # Stage 2
        for blk in self.stage2:
            h = blk(h)
        # Global average pooling + classify
        h = h.mean(dim=1)  # (B, d2)
        return self.cls(h)


def build_qkformer_imagenet() -> nn.Module:
    return QKFormerImageNet(
        img_size=8, patch_size=2, d_model=32, n_heads=2, n_classes=10, n_blocks=2
    ).eval()


def example_input() -> torch.Tensor:
    """(2, 3, 8, 8) -- batch=2, 3 channels, 8x8 image."""
    return torch.randn(2, 3, 8, 8)


MENAGERIE_ENTRIES = [
    (
        "QKFormer (hierarchical spiking ViT with Q-K token attention: binary spike linear attention, no softmax)",
        "build_qkformer_imagenet",
        "example_input",
        "2024",
        "DC",
    ),
]
