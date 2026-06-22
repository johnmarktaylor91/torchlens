"""SNN-ViT with Saccadic Spike Self-Attention (Spiking Vision Transformer).

Wang et al., "Spiking Vision Transformer with Saccadic Attention", ICLR 2025.
Paper: https://arxiv.org/abs/2502.12677

A spiking Vision Transformer whose distinctive contribution is the Saccadic
Spike Self-Attention (SSSA).  The vanilla self-attention used by ANN ViTs is
mismatched to spatio-temporal spike trains (degraded spatial relevance, weak
temporal interaction).  SSSA fixes this with two saccade-inspired pieces:

  * SPATIAL (saccadic relevance): Query/Key are LIF spike trains; their
    relevance is assessed from a *spike-distribution* statistic rather than a
    dense softmax dot-product, yielding linear-complexity attention.
  * TEMPORAL (saccadic sampling): a small per-timestep gating of the value
    stream emulates the eye's saccadic re-sampling across the T spiking steps.

This faithful random-init reimplementation keeps the distinctive SSSA structure:
spike-form Q/K/V via LIF neurons, softmax-free additive/multiplicative attention
with a spike neuron after the attention product, and a leading TIME axis that
TorchLens unrolls.  Sizes are kept tiny so the unrolled-over-time graph renders
quickly; the dynamics are identical at any T / token count.

Architecture (per published SNN-ViT, scaled down):
  Spiking Patch Splitting (SPS): Conv stem -> BN -> LIF, producing spike tokens
  -> N x [ SSSA block (saccadic spike self-attention) + spiking MLP ]
  -> temporal-average pool -> linear classification head.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from menagerie.classics._snn_neurons import LIFNeuron, spike_fn


class _SpikingPatchSplit(nn.Module):
    """Spiking Patch Splitting (SPS) stem: conv embed + BN + LIF -> spike tokens.

    Input image (B, C, H, W) is repeated over T steps and embedded into spike
    tokens of shape (T, B, N, D).
    """

    def __init__(
        self, in_ch: int = 3, embed_dim: int = 64, img_size: int = 32, patch: int = 8, time: int = 2
    ) -> None:
        super().__init__()
        self.time = time
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch, stride=patch)
        self.bn = nn.BatchNorm2d(embed_dim)
        self.lif = LIFNeuron(beta=0.9, threshold=1.0)
        self.num_tokens = (img_size // patch) ** 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        feat = self.bn(self.proj(x))  # (B, D, h, w)
        b, d, h, w = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)  # (B, N, D)
        # broadcast over time and spike
        tokens_t = tokens.unsqueeze(0).expand(self.time, -1, -1, -1)  # (T, B, N, D)
        spikes = self.lif(tokens_t)  # (T, B, N, D)
        return spikes


class _SaccadicSpikeSelfAttention(nn.Module):
    """Saccadic Spike Self-Attention (SSSA): softmax-free spike attention.

    Q, K, V are produced as spike trains by LIF neurons.  Spatial relevance is
    computed by a linear-complexity spike attention K^T V then Q(K^T V) (the
    associativity trick that gives SSSA its linear cost), followed by a spiking
    neuron on the attention output.  A learned per-timestep saccadic gate
    modulates the value stream (temporal saccade sampling).
    """

    def __init__(self, dim: int = 64, heads: int = 4, time: int = 2) -> None:
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.time = time
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.q_lif = LIFNeuron(beta=0.9, threshold=1.0)
        self.k_lif = LIFNeuron(beta=0.9, threshold=1.0)
        self.v_lif = LIFNeuron(beta=0.9, threshold=1.0)
        self.attn_lif = LIFNeuron(beta=0.9, threshold=1.0)
        # saccadic temporal gate: one scalar gate per timestep per head
        self.saccade_gate = nn.Parameter(torch.ones(time, heads))
        self.proj = nn.Linear(dim, dim)
        self.proj_lif = LIFNeuron(beta=0.9, threshold=1.0)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, B, N, D) -> (T, B, H, N, head_dim)
        t, b, n, _ = x.shape
        return x.reshape(t, b, n, self.heads, self.head_dim).permute(0, 1, 3, 2, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, B, N, D)
        q = self.q_lif(self.q_proj(x))
        k = self.k_lif(self.k_proj(x))
        v = self.v_lif(self.v_proj(x))
        qh = self._heads(q)  # (T, B, H, N, hd)
        kh = self._heads(k)
        vh = self._heads(v)
        # saccadic temporal gate on value stream: (T, H) -> (T,1,H,1,1)
        gate = self.saccade_gate.view(self.time, 1, self.heads, 1, 1)
        vh = vh * gate
        # linear-complexity spike attention: K^T V then Q (K^T V)
        kv = torch.matmul(kh.transpose(-2, -1), vh)  # (T, B, H, hd, hd)
        out = torch.matmul(qh, kv)  # (T, B, H, N, hd)
        # scale + spiking neuron (softmax-free)
        out = out * (self.head_dim**-0.5)
        out = self.attn_lif(out)
        # merge heads
        t, b, hh, n, hd = out.shape
        out = out.permute(0, 1, 3, 2, 4).reshape(t, b, n, self.dim)
        out = self.proj_lif(self.proj(out))
        return out


class _SpikingMLP(nn.Module):
    """Spiking MLP block: Linear -> LIF -> Linear -> LIF."""

    def __init__(self, dim: int = 64, hidden: int = 128) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.lif1 = LIFNeuron(beta=0.9, threshold=1.0)
        self.fc2 = nn.Linear(hidden, dim)
        self.lif2 = LIFNeuron(beta=0.9, threshold=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lif1(self.fc1(x))
        x = self.lif2(self.fc2(x))
        return x


class _SSSABlock(nn.Module):
    """One SNN-ViT block: residual SSSA + residual spiking MLP."""

    def __init__(self, dim: int = 64, heads: int = 4, mlp_hidden: int = 128, time: int = 2) -> None:
        super().__init__()
        self.attn = _SaccadicSpikeSelfAttention(dim, heads, time)
        self.mlp = _SpikingMLP(dim, mlp_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class SNNViTSaccadic(nn.Module):
    """Spiking Vision Transformer with Saccadic Spike Self-Attention."""

    def __init__(
        self,
        in_ch: int = 3,
        img_size: int = 32,
        patch: int = 8,
        embed_dim: int = 64,
        depth: int = 2,
        heads: int = 4,
        mlp_hidden: int = 128,
        num_classes: int = 10,
        time: int = 2,
    ) -> None:
        super().__init__()
        self.time = time
        self.sps = _SpikingPatchSplit(in_ch, embed_dim, img_size, patch, time)
        self.blocks = nn.ModuleList(
            [_SSSABlock(embed_dim, heads, mlp_hidden, time) for _ in range(depth)]
        )
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        tokens = self.sps(x)  # (T, B, N, D)
        for blk in self.blocks:
            tokens = blk(tokens)
        # temporal + token mean pool, then classify
        pooled = tokens.mean(dim=0).mean(dim=1)  # (B, D)
        return self.head(pooled)


def build_snn_vit_saccadic() -> nn.Module:
    """Build the SNN-ViT with Saccadic Spike Self-Attention (random init)."""
    return SNNViTSaccadic(
        in_ch=3,
        img_size=32,
        patch=8,
        embed_dim=64,
        depth=2,
        heads=4,
        mlp_hidden=128,
        num_classes=10,
        time=2,
    )


def example_input() -> torch.Tensor:
    """Example RGB image ``(1, 3, 32, 32)``; the stem repeats it over T=2 steps."""
    return torch.randn(1, 3, 32, 32)


MENAGERIE_ENTRIES = [
    (
        "SNN-ViT (spiking vision transformer with saccadic spike self-attention)",
        "build_snn_vit_saccadic",
        "example_input",
        "2025",
        "DC",
    ),
]
