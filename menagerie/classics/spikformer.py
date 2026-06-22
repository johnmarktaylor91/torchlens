"""Spikformer: spiking Vision Transformer with Spiking Self-Attention (SSA).

Zhou et al., "Spikformer: When Spiking Neural Network Meets Transformer",
ICLR 2023.
Paper: https://arxiv.org/abs/2209.15425
Source: https://github.com/ZK-Zhou/spikformer

Spikformer is the original spike-driven Vision Transformer that replaces the
conventional softmax self-attention with **Spiking Self-Attention (SSA)** -- a
softmax-free, low-power attention mechanism where Q, K, V are binary spike
trains produced by LIF neurons, and attention is computed via spike-form
matrix multiplications.  The architecture consists of:

  - **SPS (Spiking Patch Splitting):** a four-stage Conv-BN-LIF front end that
    progressively downsamples the image and produces binary spiking patch tokens
    of width ``D``.
  - **SSA block:** spiking Q, K, V from LIF(BN(Linear)); attention computed as
    ``LIF((Q K^T / scale) V)`` -- matrix products of binary spike trains, no
    softmax, energy-efficient.  Multi-head, with a spiking output projection.
  - **Spiking MLP block:** LIF(Linear) -> LIF(Linear), addition-only.
  - Residual connections (membrane-potential shortcuts), averaged over T steps
    for the final linear classification head.

The CIFAR-10 published configuration uses 4 transformer blocks, embed dim 384,
4 attention heads, input 32x32.  This compact proxy uses 2 blocks and embed 48
(4 heads, mlp_ratio 4) to keep draw < 60s; the entry name documents the nominal
4-block / 384-dim config.

Distinct from Spike-Driven Transformer V3 (SDT-v3): Spikformer uses the
standard ``Q K^T V`` attention form (with spikes), while SDT / E-SpikeFormer
uses the linear-attention-equivalent ``(K^T V)`` form for O(N) complexity.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ============================================================
# Surrogate-gradient LIF spiking neuron (traceable Heaviside)
# ============================================================


class _SurrogateSpike(torch.autograd.Function):
    """Heaviside spike (forward), fast-sigmoid surrogate gradient (backward)."""

    @staticmethod
    def forward(ctx, v: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(v)
        return (v >= 0).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (v,) = ctx.saved_tensors
        sg = 1.0 / (1.0 + 10.0 * v.abs()) ** 2
        return grad_output * sg


def spike_fn(v_minus_thresh: torch.Tensor) -> torch.Tensor:
    """Emit a spike where the membrane potential reaches threshold."""
    return _SurrogateSpike.apply(v_minus_thresh)


class LIF(nn.Module):
    """Leaky-integrate-and-fire neuron over a leading time dimension ``(T, ...)``."""

    def __init__(self, thresh: float = 1.0, decay: float = 0.5) -> None:
        super().__init__()
        self.thresh = thresh
        self.decay = decay

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        time = x.shape[0]
        v = torch.zeros_like(x[0])
        spikes = []
        for t in range(time):
            v = self.decay * v + x[t]
            s = spike_fn(v - self.thresh)
            v = v * (1 - s)
            spikes.append(s)
        return torch.stack(spikes, dim=0)


# ============================================================
# Spiking Patch Splitting (SPS) stem
# ============================================================


class _ConvBNLIF(nn.Module):
    """Conv -> BN -> LIF block applied per timestep (shared weights over T)."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.lif = LIF()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        time, b = x.shape[0], x.shape[1]
        x = self.bn(self.conv(x.flatten(0, 1)))
        x = x.reshape(time, b, *x.shape[1:])
        return self.lif(x)


class SPS(nn.Module):
    """Spiking Patch Splitting: 4 Conv-BN-LIF + max-pool blocks -> spiking tokens.

    Channels follow schedule ``D/8, D/4, D/2, D``; two 2x2 max-pools give 4x
    total spatial downsampling.
    """

    def __init__(self, in_ch: int = 3, embed_dim: int = 48) -> None:
        super().__init__()
        c1, c2, c3, c4 = max(1, embed_dim // 8), max(2, embed_dim // 4), embed_dim // 2, embed_dim
        self.block1 = _ConvBNLIF(in_ch, c1)
        self.block2 = _ConvBNLIF(c1, c2)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.block3 = _ConvBNLIF(c2, c3)
        self.pool3 = nn.MaxPool2d(2, stride=2)
        self.block4 = _ConvBNLIF(c3, c4)
        self.embed_dim = embed_dim

    def _pool(self, x: torch.Tensor, pool: nn.Module) -> torch.Tensor:
        time, b = x.shape[0], x.shape[1]
        x = pool(x.flatten(0, 1))
        return x.reshape(time, b, *x.shape[1:])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self._pool(x, self.pool2)
        x = self.block3(x)
        x = self._pool(x, self.pool3)
        x = self.block4(x)
        time, b, c, h, w = x.shape
        return x.reshape(time, b, c, h * w).transpose(2, 3)  # (T, B, N, D)


# ============================================================
# Spiking Self-Attention (SSA)
# ============================================================


class _SpikeLinear(nn.Module):
    """LIF(BN(Linear)) per timestep over (T, B, N, C)."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.bn = nn.BatchNorm1d(out_dim)
        self.lif = LIF()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        time, b, n, _ = x.shape
        y = self.linear(x)
        y = self.bn(y.reshape(-1, y.shape[-1])).reshape(time, b, n, -1)
        return self.lif(y)


class SSA(nn.Module):
    """Spiking Self-Attention (SSA).

    The Spikformer SSA distinctive mechanism: Q, K, V are binary spike trains
    from LIF(BN(Linear)); attention is the standard Q K^T V form but fully on
    binary spikes (no softmax) -- this is the direct spike-form attention, as
    opposed to the SDT-v3 linear-attention associativity trick.  A LIF fires on
    the attention output before the projection.
    """

    def __init__(self, dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.q = _SpikeLinear(dim, dim)
        self.k = _SpikeLinear(dim, dim)
        self.v = _SpikeLinear(dim, dim)
        self.attn_lif = LIF()
        self.proj = _SpikeLinear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        time, b, n, _ = x.shape
        q = self.q(x).reshape(time, b, n, self.num_heads, self.head_dim).transpose(2, 3)
        k = self.k(x).reshape(time, b, n, self.num_heads, self.head_dim).transpose(2, 3)
        v = self.v(x).reshape(time, b, n, self.num_heads, self.head_dim).transpose(2, 3)
        # Spike-form attention: Q K^T * scale -> LIF -> V
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (T, B, H, N, N)
        attn = self.attn_lif(attn)
        out = attn @ v  # (T, B, H, N, head_dim)
        out = out.transpose(2, 3).reshape(time, b, n, self.num_heads * self.head_dim)
        return self.proj(out)


class SpikingMLP(nn.Module):
    """Spiking channel-MLP block: LIF(Linear) -> LIF(Linear), addition-only."""

    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.fc1 = _SpikeLinear(dim, hidden)
        self.fc2 = _SpikeLinear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.fc1(x))


class SpikformerBlock(nn.Module):
    """One Spikformer encoder block: SSA + spiking MLP, membrane-shortcut residuals."""

    def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.attn = SSA(dim, num_heads=num_heads)
        self.mlp = SpikingMLP(dim, dim * mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


# ============================================================
# Spikformer model (parametric)
# ============================================================


class Spikformer(nn.Module):
    """Spikformer: spiking ViT with SSA (softmax-free spike self-attention).

    The compact proxy uses embed_dim=48, depth=2 to keep the unrolled graph
    tractable.  The CIFAR-10 published config is depth=4, embed_dim=384,
    num_heads=8, documented in the entry name.
    """

    def __init__(
        self,
        embed_dim: int = 48,
        depth: int = 2,
        num_heads: int = 4,
        mlp_ratio: int = 4,
        num_classes: int = 10,
        in_ch: int = 3,
        timesteps: int = 2,
    ) -> None:
        super().__init__()
        self.timesteps = timesteps
        self.patch_embed = SPS(in_ch=in_ch, embed_dim=embed_dim)
        self.blocks = nn.ModuleList(
            [
                SpikformerBlock(embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )
        self.head_lif = LIF()
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W) -> repeat over T timesteps
        x = x.unsqueeze(0).repeat(self.timesteps, 1, 1, 1, 1)
        tokens = self.patch_embed(x)  # (T, B, N, D)
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.head_lif(tokens)
        pooled = tokens.mean(dim=2)  # (T, B, D)
        logits = self.head(pooled)  # (T, B, num_classes)
        return logits.mean(dim=0)  # (B, num_classes)


def build_spikformer_cifar10() -> nn.Module:
    """Build Spikformer CIFAR-10 proxy (compact: embed=48, depth=2; nominal: embed=384, depth=4)."""
    return Spikformer(
        embed_dim=48, depth=2, num_heads=4, mlp_ratio=4, num_classes=10, in_ch=3, timesteps=2
    )


def example_input() -> torch.Tensor:
    """Example RGB image tensor ``(1, 3, 32, 32)``; model repeats over T internally."""
    return torch.randn(1, 3, 32, 32)


MENAGERIE_ENTRIES = [
    (
        "Spikformer CIFAR-10 (4-block 384-dim SSA spiking ViT)",
        "build_spikformer_cifar10",
        "example_input",
        "2023",
        "DC",
    ),
]
