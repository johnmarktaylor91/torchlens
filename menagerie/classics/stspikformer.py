"""ST-SpikFormer / STAtten: Spiking Transformer with Spatial-Temporal Attention.

Lee et al., "Spiking Transformer with Spatial-Temporal Attention", CVPR 2025.
Paper: https://arxiv.org/abs/2409.19764
Source: https://github.com/Intelligent-Computing-Lab-Yale (STAtten)

Existing spiking transformers (Spikformer, Spike-Driven Transformer) run
self-attention *independently per timestep*, so they capture spatial (token)
relationships but neglect the temporal dependencies inherent to spike trains.
STAtten introduces **Spatial-Temporal Attention**: self-attention is computed
*jointly* across the spatial (token) and temporal (timestep) dimensions, so each
spike token attends to tokens at other timesteps as well as other spatial
positions -- all with Leaky-Integrate-and-Fire (LIF) spiking neurons and the
same ``O(T N D^2)`` complexity as spatial-only attention.

Faithful (random-init) reimplementation of the architectural skeleton:

  - **Spiking patch embedding:** Conv-BN-LIF stem with max-pool downsampling,
    producing ``N`` spiking patch tokens of width ``D`` per timestep (the
    Spikformer SPS lineage).
  - **Spatial-Temporal spiking attention block:** spiking ``Q, K, V`` from
    LIF(Linear); the ``T`` per-timestep token sets are concatenated into a
    single combined ``T*N`` space-time token set, and attention is computed over
    that joint set as ``LIF((Q (K^T V)) * scale)`` (softmax-free spike-driven
    linear attention), then split back per timestep.  This is the "cross-manner"
    / block-wise joint space-time computation of STAtten.
  - **Spiking channel-MLP** block + membrane-shortcut residuals.
  - Classifier averages the per-timestep logits.

The LIF neuron is the standard leaky-integrate-and-fire unit (decay + integrate
+ threshold spike via a traceable surrogate Heaviside + reset).  ``T`` is kept
small so the unrolled-time graph renders quickly; the dynamics are identical at
any horizon.
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

    def __init__(self, thresh: float = 1.0, decay: float = 0.5, reset: float = 0.0) -> None:
        super().__init__()
        self.thresh = thresh
        self.decay = decay
        self.reset = reset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        time = x.shape[0]
        v = torch.zeros_like(x[0])
        spikes = []
        for t in range(time):
            v = self.decay * v + x[t]
            s = spike_fn(v - self.thresh)
            v = v * (1 - s) + self.reset * s
            spikes.append(s)
        return torch.stack(spikes, dim=0)


# ============================================================
# Spiking patch embedding (Conv-BN-LIF stem + max-pool)
# ============================================================


class _ConvBNLIF(nn.Module):
    """Conv -> BN -> LIF block, applied per timestep with shared weights."""

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


class SpikingPatchEmbed(nn.Module):
    """Spiking patch embedding: Conv-BN-LIF blocks with channels D/4, D/2, D + max-pool."""

    def __init__(self, in_ch: int = 3, embed_dim: int = 48) -> None:
        super().__init__()
        c1, c2, c3 = embed_dim // 4, embed_dim // 2, embed_dim
        self.block1 = _ConvBNLIF(in_ch, c1)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.block2 = _ConvBNLIF(c1, c2)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.block3 = _ConvBNLIF(c2, c3)
        self.embed_dim = embed_dim

    def _pool(self, x: torch.Tensor, pool: nn.Module) -> torch.Tensor:
        time, b = x.shape[0], x.shape[1]
        x = pool(x.flatten(0, 1))
        return x.reshape(time, b, *x.shape[1:])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self._pool(x, self.pool1)
        x = self.block2(x)
        x = self._pool(x, self.pool2)
        x = self.block3(x)
        time, b, c, h, w = x.shape
        return x.reshape(time, b, c, h * w).transpose(2, 3)  # (T, B, N, D)


# ============================================================
# Spatial-Temporal spiking attention (STAtten)
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


class STAtten(nn.Module):
    """Spatial-Temporal Attention: joint self-attention over space-time tokens.

    Spiking ``Q, K, V`` are produced per timestep, then the ``T`` timesteps are
    fused into a single combined ``T*N`` space-time token set so that attention
    mixes information *across* timesteps (the STAtten contribution) as well as
    across spatial positions.  The softmax-free spike-driven linear form
    ``LIF((Q (K^T V)) * scale)`` keeps the cost at ``O(T N D^2)``.
    """

    def __init__(self, dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.q = _SpikeLinear(dim, dim)
        self.k = _SpikeLinear(dim, dim)
        self.v = _SpikeLinear(dim, dim)
        self.proj = _SpikeLinear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        time, b, n, dim = x.shape
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        # Fuse the T timesteps into one combined space-time token set: (B, T*N, D)
        q = q.transpose(0, 1).reshape(b, time * n, dim)
        k = k.transpose(0, 1).reshape(b, time * n, dim)
        v = v.transpose(0, 1).reshape(b, time * n, dim)
        # Multi-head split: (B, H, T*N, head_dim)
        q = q.reshape(b, time * n, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(b, time * n, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(b, time * n, self.num_heads, self.head_dim).transpose(1, 2)
        # Spike-driven linear attention across the joint space-time set
        kv = k.transpose(-2, -1) @ v  # (B, H, head_dim, head_dim)
        out = (q @ kv) * self.scale  # (B, H, T*N, head_dim)
        out = out.transpose(1, 2).reshape(b, time * n, dim)
        # Split back per timestep: (T, B, N, D)
        out = out.reshape(b, time, n, dim).transpose(0, 1)
        return self.proj(out)


class SpikingMLP(nn.Module):
    """Spiking channel-MLP: LIF(Linear) -> LIF(Linear)."""

    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.fc1 = _SpikeLinear(dim, hidden)
        self.fc2 = _SpikeLinear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.fc1(x))


class STBlock(nn.Module):
    """One ST-SpikFormer block: STAtten + spiking MLP, membrane-shortcut residuals."""

    def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: int = 2) -> None:
        super().__init__()
        self.attn = STAtten(dim, num_heads=num_heads)
        self.mlp = SpikingMLP(dim, dim * mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


# ============================================================
# ST-SpikFormer model
# ============================================================


class STSpikFormer(nn.Module):
    """ST-SpikFormer: spiking transformer with spatial-temporal attention."""

    def __init__(
        self,
        embed_dim: int = 48,
        depth: int = 2,
        num_heads: int = 4,
        mlp_ratio: int = 2,
        num_classes: int = 10,
        in_ch: int = 3,
        timesteps: int = 4,
    ) -> None:
        super().__init__()
        self.timesteps = timesteps
        self.patch_embed = SpikingPatchEmbed(in_ch=in_ch, embed_dim=embed_dim)
        self.blocks = nn.ModuleList(
            [STBlock(embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)]
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
        return logits.mean(dim=0)  # average over timesteps -> (B, num_classes)


def build_stspikformer() -> nn.Module:
    """Build ST-SpikFormer with spatial-temporal spiking attention (compact proxy)."""
    return STSpikFormer(
        embed_dim=48, depth=2, num_heads=4, mlp_ratio=2, num_classes=10, in_ch=3, timesteps=4
    )


def build_stspikformer_temporal_attention() -> nn.Module:
    """Build ST-SpikFormer showcasing the temporal attention over T=4 timesteps.

    Uses T=4 timesteps (nominal config) so TorchLens unrolls the full
    space-time attention over 4 spiking steps -- the unrolled graph
    highlights the STAtten temporal interaction more clearly.  Same
    architecture as ``build_stspikformer``; named variant to expose the
    temporal-attention axis explicitly in the catalog.
    """
    return STSpikFormer(
        embed_dim=48, depth=1, num_heads=4, mlp_ratio=2, num_classes=10, in_ch=3, timesteps=4
    )


def example_input() -> torch.Tensor:
    """Example RGB image tensor ``(1, 3, 32, 32)``; the model repeats it over T internally."""
    return torch.randn(1, 3, 32, 32)


MENAGERIE_ENTRIES = [
    (
        "ST-SpikFormer (STAtten spatial-temporal spiking attention)",
        "build_stspikformer",
        "example_input",
        "2024",
        "DC",
    ),
    (
        "ST-SpikFormer temporal attention (space-time joint spiking attention, T=4)",
        "build_stspikformer_temporal_attention",
        "example_input",
        "2024",
        "DC",
    ),
]
