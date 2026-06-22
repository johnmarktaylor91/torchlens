"""SpikingResformer: spiking residual CNN + Dual Spike Self-Attention (CVPR 2024).

Shi et al., "SpikingResformer: Bridging ResNet and Vision Transformer in
Spiking Neural Networks", CVPR 2024.
Paper: https://arxiv.org/abs/2403.14302
Source: https://github.com/xyshi2000/SpikingResformer

SpikingResformer bridges spiking residual CNNs and spiking transformers by
combining:

  * **Spiking Residual Stages:** standard residual blocks (Conv-BN-LIF shortcuts
    + Conv-BN-LIF main paths) that form a deep spiking CNN backbone -- similar
    to ResNet stages but with LIF neurons instead of ReLU.  Each stage halves
    the spatial resolution and doubles the channels.
  * **Dual Spike Self-Attention (DSSA):** a novel spiking self-attention that
    processes spatial patches with a *dual-path* spike-form attention.  The two
    paths compute spike Q/K via separate LIF(Linear) projections; spike V via a
    third projection; and the attention is merged as a weighted sum of the two
    spike-attention maps.  This gives richer spike-domain interaction than a
    single SDSA head while staying softmax-free.
  * The DSSA is inserted between (or after) the residual stages; a global average
    pool and linear head follow.

This faithful compact reimplementation uses:
  - 2 spiking residual stages (channels 16->32 for tiny, 32->64 for small)
  - 1 DSSA block after the residual stages
  - Input (1, 3, 32, 32), T=2 timesteps

Simplification: the published SpikingResformer has more stages and wider
channels; the compact proxy shows the distinctive primitives at tiny size.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ============================================================
# Surrogate-gradient LIF spiking neuron
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


def spike_fn(v: torch.Tensor) -> torch.Tensor:
    return _SurrogateSpike.apply(v)


class LIF(nn.Module):
    """Leaky-integrate-and-fire neuron over leading TIME axis ``(T, ...)``."""

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
# Spiking residual blocks (ResNet-style with LIF neurons)
# ============================================================


class _ConvBNLIF(nn.Module):
    """Conv2d -> BN -> LIF, applied per timestep (shared weights over T)."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.lif = LIF()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        time, b = x.shape[0], x.shape[1]
        y = self.bn(self.conv(x.flatten(0, 1)))
        y = y.reshape(time, b, *y.shape[1:])
        return self.lif(y)


class SpikingResBlock(nn.Module):
    """Spiking residual block: main path (Conv-BN-LIF x2) + shortcut.

    The shortcut uses a 1x1 conv when stride > 1 or channels change; otherwise
    identity.  Both paths are spike-domain (LIF outputs) enabling spike-domain
    addition as in SpikingResformer.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.path = nn.Sequential(
            _ConvBNLIF(in_ch, out_ch, stride=stride),
            _ConvBNLIF(out_ch, out_ch, stride=1),
        )
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, B, C, H, W)
        out = self.path(x)
        if self.shortcut is not None:
            time, b = x.shape[0], x.shape[1]
            sc = self.shortcut(x.flatten(0, 1))
            sc = sc.reshape(time, b, *sc.shape[1:])
        else:
            sc = x
        return out + sc


class SpikingResStage(nn.Module):
    """A stage of spiking residual blocks; first block may stride-2."""

    def __init__(self, in_ch: int, out_ch: int, n_blocks: int = 1, stride: int = 2) -> None:
        super().__init__()
        blocks = [SpikingResBlock(in_ch, out_ch, stride=stride)]
        for _ in range(n_blocks - 1):
            blocks.append(SpikingResBlock(out_ch, out_ch, stride=1))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


# ============================================================
# Dual Spike Self-Attention (DSSA)
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


class DSSA(nn.Module):
    """Dual Spike Self-Attention (DSSA).

    Two parallel spike Q/K paths compute independent spike-attention maps over
    the same V; the results are merged via addition.  This dual-path attention
    is the distinctive mechanism of SpikingResformer vs. single-path SDSA/SSA.
    Softmax-free: all operations are spike-domain matrix products + LIF.
    """

    def __init__(self, dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = max(1, dim // num_heads)
        self.scale = self.head_dim**-0.5
        # Dual paths: path A and path B
        self.qa = _SpikeLinear(dim, dim)
        self.ka = _SpikeLinear(dim, dim)
        self.qb = _SpikeLinear(dim, dim)
        self.kb = _SpikeLinear(dim, dim)
        # Shared V
        self.v = _SpikeLinear(dim, dim)
        self.proj = _SpikeLinear(dim, dim)

    def _attn_path(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Compute spike-linear attention: (K^T V) then Q(K^T V)."""
        time, b, n, _ = q.shape
        qh = q.reshape(time, b, n, self.num_heads, self.head_dim).transpose(2, 3)
        kh = k.reshape(time, b, n, self.num_heads, self.head_dim).transpose(2, 3)
        vh = v.reshape(time, b, n, self.num_heads, self.head_dim).transpose(2, 3)
        kv = kh.transpose(-2, -1) @ vh  # (T, B, H, head_dim, head_dim)
        out = (qh @ kv) * self.scale  # (T, B, H, N, head_dim)
        return out.transpose(2, 3).reshape(time, b, n, self.num_heads * self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = self.v(x)
        out_a = self._attn_path(self.qa(x), self.ka(x), v)
        out_b = self._attn_path(self.qb(x), self.kb(x), v)
        out = out_a + out_b  # dual-path merge
        return self.proj(out)


class _SpikingMLP(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.fc1 = _SpikeLinear(dim, hidden)
        self.fc2 = _SpikeLinear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.fc1(x))


class DSSABlock(nn.Module):
    """DSSA block: dual-path spike self-attention + spiking MLP, residual."""

    def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: int = 2) -> None:
        super().__init__()
        self.attn = DSSA(dim, num_heads=num_heads)
        self.mlp = _SpikingMLP(dim, dim * mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


# ============================================================
# SpikingResformer model (parametric)
# ============================================================


class SpikingResformer(nn.Module):
    """SpikingResformer: spiking residual stages + Dual Spike Self-Attention.

    A spiking ResNet backbone followed by DSSA transformer blocks.  The
    distinctive primitives: spiking residual blocks (Conv-BN-LIF + identity
    shortcut addition in spike domain) and the dual-path softmax-free DSSA.
    """

    def __init__(
        self,
        in_ch: int = 3,
        base_ch: int = 16,
        depth_stages: int = 2,
        n_dssa: int = 1,
        num_heads: int = 4,
        num_classes: int = 10,
        timesteps: int = 2,
    ) -> None:
        super().__init__()
        self.timesteps = timesteps
        # Spiking stem
        self.stem = _ConvBNLIF(in_ch, base_ch, stride=1)
        # Spiking residual stages: channels double, spatial halved each stage
        stages = []
        ch = base_ch
        for i in range(depth_stages):
            out_ch = ch * 2 if i > 0 else ch
            stages.append(SpikingResStage(ch, out_ch, n_blocks=1, stride=2))
            ch = out_ch
        self.stages = nn.Sequential(*stages)
        self.feat_dim = ch
        # DSSA blocks over flattened spatial tokens
        self.dssa_blocks = nn.ModuleList(
            [DSSABlock(ch, num_heads=num_heads) for _ in range(n_dssa)]
        )
        self.head_lif = LIF()
        self.head = nn.Linear(ch, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W) -> repeat over T
        x = x.unsqueeze(0).repeat(self.timesteps, 1, 1, 1, 1)
        x = self.stem(x)  # (T, B, base_ch, H, W)
        x = self.stages(x)  # (T, B, feat_dim, H', W')
        time, b, c, h, w = x.shape
        tokens = x.reshape(time, b, c, h * w).transpose(2, 3)  # (T, B, N, C)
        for blk in self.dssa_blocks:
            tokens = blk(tokens)
        tokens = self.head_lif(tokens)
        pooled = tokens.mean(dim=2)  # (T, B, C)
        logits = self.head(pooled)  # (T, B, num_classes)
        return logits.mean(dim=0)  # (B, num_classes)


def build_spikingresformer_tiny() -> nn.Module:
    """Build SpikingResformer tiny (compact: base_ch=16, 2 stages + 1 DSSA)."""
    return SpikingResformer(
        in_ch=3, base_ch=16, depth_stages=2, n_dssa=1, num_heads=4, num_classes=10, timesteps=2
    )


def build_spikingresformer_small() -> nn.Module:
    """Build SpikingResformer small (compact: base_ch=32, 2 stages + 1 DSSA)."""
    return SpikingResformer(
        in_ch=3, base_ch=32, depth_stages=2, n_dssa=1, num_heads=4, num_classes=10, timesteps=2
    )


def example_input() -> torch.Tensor:
    """Example RGB image tensor ``(1, 3, 32, 32)``; model repeats over T internally."""
    return torch.randn(1, 3, 32, 32)


MENAGERIE_ENTRIES = [
    (
        "SpikingResformer tiny (spiking residual stages + Dual Spike Self-Attention)",
        "build_spikingresformer_tiny",
        "example_input",
        "2024",
        "DC",
    ),
    (
        "SpikingResformer small (spiking residual stages + Dual Spike Self-Attention)",
        "build_spikingresformer_small",
        "example_input",
        "2024",
        "DC",
    ),
]
