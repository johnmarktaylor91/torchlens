"""Spike-Driven Transformer V3 (E-SpikeFormer): scaling spike-driven Vision Transformers.

Yao et al., "Scaling Spike-driven Transformer with Efficient Spike Firing
Approximation Training", IEEE T-PAMI 2025.
Paper: https://arxiv.org/abs/2411.16061
Source: https://github.com/BICLab/Spike-Driven-Transformer-V3

A pure spike-driven Vision Transformer: every tensor that flows through the
network is a binary spike train, and attention is computed with mask/addition
only (no softmax multiply).  The architecture is the spike-driven lineage of
Spikformer (Zhou et al., ICLR 2023) / Spike-Driven Transformer (Yao et al.,
NeurIPS 2023), here scaled and trained with the Spike Firing Approximation (SFA)
integer-training / spike-driven-inference recipe that lets SNNs scale to ANN
accuracy at low power.

Faithful (random-init) reimplementation of the architectural skeleton:

  - **SPS (Spiking Patch Splitting / patch embedding stem):** four Conv-BN-LIF
    blocks with output channels ``D/8, D/4, D/2, D``, each followed by a
    stride-2 max-pool that downsamples the feature map (the published SPS uses
    3x3 stride-1 convs + 2x2 max-pool).  Produces ``N = H'*W'`` spiking patch
    tokens of width ``D``.
  - **SDSA (Spike-Driven Self-Attention):** spiking ``Q, K, V`` are produced by
    LIF(Linear(x)); attention is the spike-driven ``LIF(Q (K^T V) * scale)``
    form, which is linear-attention-like (no softmax) and uses only mask /
    addition on the binary spike trains.  ``V`` carries the channel-expansion
    factor of the E-SDSA variant.
  - **Spiking channel-MLP block:** LIF(Linear) -> LIF(Linear), addition-only.
  - Both sub-blocks are wrapped with membrane-shortcut residuals and unrolled
    over ``T`` timesteps; the classifier averages the per-timestep logits.

The LIF neuron is the standard leaky-integrate-and-fire unit: the membrane
potential is decayed and integrated each timestep, a spike is emitted where it
crosses threshold (via a traceable surrogate Heaviside), and the membrane is
reset where it spiked.  To keep ``draw()`` fast the random-init reimpl uses
compact proxy widths/depths and a small ``T``; the *variant names* encode the
published 10M/19M/55M/83M and SpikeFormer-12 (512/768) configurations.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ============================================================
# Surrogate-gradient LIF spiking neuron (traceable Heaviside)
# ============================================================


class _SurrogateSpike(torch.autograd.Function):
    """Heaviside spike in the forward pass, fast-sigmoid surrogate gradient.

    Forward is a hard threshold (binary 0/1 spikes); backward uses a smooth
    surrogate so the graph stays differentiable.  The forward is a plain tensor
    comparison cast to float -- no ``.item()`` / python branching on data -- so
    it traces cleanly.
    """

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
    """Leaky-integrate-and-fire spiking neuron over a leading time dimension.

    Expects ``x`` of shape ``(T, ...)``; integrates the membrane potential
    ``V = decay * V + x[t]`` per timestep, emits a spike when ``V >= thresh``,
    and soft-resets the membrane where it spiked.  Returns a binary spike train
    of the same shape as ``x``.
    """

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
        # x: (T, B, C, H, W) -> fold time into batch for the conv/bn
        time, b = x.shape[0], x.shape[1]
        x = x.flatten(0, 1)
        x = self.bn(self.conv(x))
        x = x.reshape(time, b, *x.shape[1:])
        return self.lif(x)


class SPS(nn.Module):
    """Spiking Patch Splitting: 4 Conv-BN-LIF blocks + max-pool downsampling.

    Output channels follow the published SPS schedule ``D/8, D/4, D/2, D`` with a
    2x2 stride-2 max-pool after each block (4x downsample total here), producing
    ``N = H'*W'`` spiking patch tokens of width ``embed_dim``.
    """

    def __init__(self, in_ch: int = 3, embed_dim: int = 64) -> None:
        super().__init__()
        c1, c2, c3, c4 = embed_dim // 8, embed_dim // 4, embed_dim // 2, embed_dim
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
        # x: (T, B, 3, H, W) -> tokens (T, B, N, D)
        x = self.block1(x)
        x = self.block2(x)
        x = self._pool(x, self.pool2)
        x = self.block3(x)
        x = self._pool(x, self.pool3)
        x = self.block4(x)
        time, b, c, h, w = x.shape
        return x.reshape(time, b, c, h * w).transpose(2, 3)  # (T, B, N, D)


# ============================================================
# Spike-Driven Self-Attention (SDSA) -- mask/addition, no softmax
# ============================================================


class _SpikeLinear(nn.Module):
    """LIF(BN(Linear(x))) applied per timestep over (T, B, N, C)."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.bn = nn.BatchNorm1d(out_dim)
        self.lif = LIF()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        time, b, n, _ = x.shape
        y = self.linear(x)
        # BN1d over channels: (T*B*N, C)
        y = self.bn(y.reshape(-1, y.shape[-1])).reshape(time, b, n, -1)
        return self.lif(y)


class SDSA(nn.Module):
    """Spike-Driven Self-Attention.

    Spiking ``Q, K, V`` come from LIF(Linear); attention uses the spike-driven
    linear form ``LIF((Q (K^T V)) * scale)`` -- associativity makes it
    softmax-free and addition/mask dominated on the binary spike trains.  ``V``
    carries the E-SDSA channel-expansion factor.
    """

    def __init__(self, dim: int, num_heads: int = 4, v_expand: int = 1) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.q = _SpikeLinear(dim, dim)
        self.k = _SpikeLinear(dim, dim)
        self.v = _SpikeLinear(dim, dim * v_expand)
        self.v_dim = (dim * v_expand) // num_heads
        self.proj = _SpikeLinear(dim * v_expand, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        time, b, n, _ = x.shape
        q = self.q(x).reshape(time, b, n, self.num_heads, self.head_dim).transpose(2, 3)
        k = self.k(x).reshape(time, b, n, self.num_heads, self.head_dim).transpose(2, 3)
        v = self.v(x).reshape(time, b, n, self.num_heads, self.v_dim).transpose(2, 3)
        # Spike-driven linear attention: K^T V first (heads, head_dim x v_dim)
        kv = k.transpose(-2, -1) @ v  # (T, B, H, head_dim, v_dim)
        out = (q @ kv) * self.scale  # (T, B, H, N, v_dim)
        out = out.transpose(2, 3).reshape(time, b, n, self.num_heads * self.v_dim)
        return self.proj(out)


class SpikingMLP(nn.Module):
    """Spiking channel-MLP block: LIF(Linear) -> LIF(Linear), addition-only."""

    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.fc1 = _SpikeLinear(dim, hidden)
        self.fc2 = _SpikeLinear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.fc1(x))


class SDTBlock(nn.Module):
    """One spike-driven encoder block: SDSA + spiking MLP, membrane-shortcut residuals."""

    def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: int = 2, v_expand: int = 1) -> None:
        super().__init__()
        self.attn = SDSA(dim, num_heads=num_heads, v_expand=v_expand)
        self.mlp = SpikingMLP(dim, dim * mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


# ============================================================
# Spike-Driven Transformer V3 parametric model
# ============================================================


class SpikeDrivenTransformerV3(nn.Module):
    """E-SpikeFormer (Spike-Driven Transformer V3), parametric.

    A single parametric module covers all catalog variants; they differ only by
    ``embed_dim`` / ``depth`` / ``num_heads`` (and the published name).  The
    random-init reimpl uses compact proxy widths and a small ``T`` so the
    unrolled-time graph stays small.
    """

    def __init__(
        self,
        embed_dim: int = 32,
        depth: int = 1,
        num_heads: int = 4,
        mlp_ratio: int = 2,
        v_expand: int = 1,
        num_classes: int = 10,
        in_ch: int = 3,
        timesteps: int = 2,
    ) -> None:
        super().__init__()
        self.timesteps = timesteps
        self.patch_embed = SPS(in_ch=in_ch, embed_dim=embed_dim)
        self.blocks = nn.ModuleList(
            [
                SDTBlock(embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, v_expand=v_expand)
                for _ in range(depth)
            ]
        )
        self.head_lif = LIF()
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W) -> repeat over T timesteps -> (T, B, 3, H, W)
        x = x.unsqueeze(0).repeat(self.timesteps, 1, 1, 1, 1)
        tokens = self.patch_embed(x)  # (T, B, N, D)
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.head_lif(tokens)
        pooled = tokens.mean(dim=2)  # (T, B, D) global-average-pool over tokens
        logits = self.head(pooled)  # (T, B, num_classes)
        return logits.mean(dim=0)  # average over timesteps -> (B, num_classes)


# Variant configs: name encodes published config; dims kept compact for tracing.
_VARIANTS = {
    "t": dict(embed_dim=32, depth=1, num_heads=4, mlp_ratio=2, v_expand=1),
    "s": dict(embed_dim=48, depth=2, num_heads=4, mlp_ratio=2, v_expand=1),
    "m": dict(embed_dim=64, depth=2, num_heads=8, mlp_ratio=2, v_expand=1),
    "l": dict(embed_dim=64, depth=3, num_heads=8, mlp_ratio=2, v_expand=2),
    "spikformer12_512": dict(embed_dim=64, depth=2, num_heads=8, mlp_ratio=2, v_expand=1),
    "spikformer12_768": dict(embed_dim=96, depth=2, num_heads=8, mlp_ratio=2, v_expand=1),
}


def _build(key: str) -> nn.Module:
    cfg = _VARIANTS[key]
    return SpikeDrivenTransformerV3(timesteps=2, num_classes=10, in_ch=3, **cfg)


def build_sdt_v3_t() -> nn.Module:
    """Build E-SpikeFormer tiny (published ~10M-param config, compact proxy)."""
    return _build("t")


def build_sdt_v3_s() -> nn.Module:
    """Build E-SpikeFormer small (published ~19M-param config, compact proxy)."""
    return _build("s")


def build_sdt_v3_m() -> nn.Module:
    """Build E-SpikeFormer medium (published ~55M-param config, compact proxy)."""
    return _build("m")


def build_sdt_v3_l() -> nn.Module:
    """Build E-SpikeFormer large (published ~83M-param config, compact proxy)."""
    return _build("l")


def build_sdt_v3_spikformer12_512() -> nn.Module:
    """Build SpikeFormer-12 with published embed dim 512 (compact proxy)."""
    return _build("spikformer12_512")


def build_sdt_v3_spikformer12_768() -> nn.Module:
    """Build SpikeFormer-12 with published embed dim 768 (compact proxy)."""
    return _build("spikformer12_768")


def example_input() -> torch.Tensor:
    """Example RGB image tensor ``(1, 3, 32, 32)``; the model repeats it over T internally."""
    return torch.randn(1, 3, 32, 32)


MENAGERIE_ENTRIES = [
    (
        "Spike-Driven Transformer V3 (E-SpikeFormer, tiny)",
        "build_sdt_v3_t",
        "example_input",
        "2024",
        "DC",
    ),
    (
        "Spike-Driven Transformer V3 (E-SpikeFormer, small)",
        "build_sdt_v3_s",
        "example_input",
        "2024",
        "DC",
    ),
    (
        "Spike-Driven Transformer V3 (E-SpikeFormer, medium)",
        "build_sdt_v3_m",
        "example_input",
        "2024",
        "DC",
    ),
    (
        "Spike-Driven Transformer V3 (E-SpikeFormer, large)",
        "build_sdt_v3_l",
        "example_input",
        "2024",
        "DC",
    ),
    (
        "Spike-Driven Transformer V3 (SpikeFormer-12, embed 512)",
        "build_sdt_v3_spikformer12_512",
        "example_input",
        "2024",
        "DC",
    ),
    (
        "Spike-Driven Transformer V3 (SpikeFormer-12, embed 768)",
        "build_sdt_v3_spikformer12_768",
        "example_input",
        "2024",
        "DC",
    ),
]
