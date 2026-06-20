"""Flow-Matching / Rectified-Flow vector-field networks.

TorchCFM (Tong et al., 2023, "Improving and generalizing flow-based generative
models with minibatch optimal transport", arXiv:2302.00482) and Rectified Flow
(Liu et al., ICLR 2023, "Flow Straight and Fast", arXiv:2209.03003) are *training
objectives* over an ODE whose drift is a learned, time-conditioned vector field
v_theta(t, x). The conditional-flow-matching variants (conditional / exact-OT /
Schrodinger-bridge / variance-preserving) and rectified flow all share the SAME
neural network -- they differ only in how the target velocity / coupling is
constructed during training, not in architecture.

This module captures the two canonical vector-field backbones that TorchCFM ships:

  * ``MLPVectorField`` -- the low-dimensional field used for 2D / tabular /
    single-cell experiments (``torchcfm.models.MLP``): time is appended as an
    extra input coordinate, then a stack of Linear + SELU layers regresses the
    velocity. Optional time-embedding variant matches ``time_varying=True``.

  * ``UNetVectorField`` -- the image field (``torchcfm.models.unet.UNetModel``,
    adapted from the guided-diffusion UNet of Dhariwal & Nichol): sinusoidal
    timestep embedding -> residual blocks with down/up sampling and a self-
    attention bottleneck, predicting a velocity image of the same shape as the
    input.  This is the field used for CIFAR-style flow-matching image generation.

References:
  https://github.com/atong01/conditional-flow-matching  (TorchCFM)
  https://github.com/gnobitab/RectifiedFlow
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# MLP vector field (torchcfm.models.MLP)
# ============================================================


class MLPVectorField(nn.Module):
    """Time-conditioned MLP velocity field v(t, x) for low-dimensional data.

    Faithful to ``torchcfm.models.models.MLP``: time t is concatenated as one
    extra input coordinate (``time_varying=True``), and a Linear/SELU stack maps
    R^{dim+1} -> R^{dim}.  This is the network used for the 2D toy, tabular, and
    single-cell flow-matching / rectified-flow experiments, identical across the
    conditional / exact-OT / Schrodinger-bridge / variance-preserving objectives.
    """

    def __init__(self, dim: int = 2, w: int = 64, time_varying: bool = True) -> None:
        super().__init__()
        self.time_varying = time_varying
        in_dim = dim + (1 if time_varying else 0)
        self.net = nn.Sequential(
            nn.Linear(in_dim, w),
            nn.SELU(),
            nn.Linear(w, w),
            nn.SELU(),
            nn.Linear(w, w),
            nn.SELU(),
            nn.Linear(w, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x packs the state coordinates and (optionally) time in the last column.
        return self.net(x)


# ============================================================
# UNet vector field (torchcfm.models.unet.UNetModel)
# ============================================================


def _timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Sinusoidal timestep embedding (guided-diffusion convention)."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half).to(
        timesteps.device
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class _ResBlock(nn.Module):
    """Residual block with an injected timestep embedding (guided-diffusion style)."""

    def __init__(self, in_ch: int, out_ch: int, emb_ch: int) -> None:
        super().__init__()
        self.in_layers = nn.Sequential(
            nn.GroupNorm(min(32, in_ch), in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(emb_ch, out_ch))
        self.out_layers = nn.Sequential(
            nn.GroupNorm(min(32, out_ch), out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.in_layers(x)
        h = h + self.emb_layers(emb)[:, :, None, None]
        h = self.out_layers(h)
        return h + self.skip(x)


class _AttnBlock(nn.Module):
    """Single-head spatial self-attention bottleneck block."""

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(min(32, ch), ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x)).reshape(b, 3, c, h * w)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        attn = torch.softmax(torch.einsum("bci,bcj->bij", q, k) / math.sqrt(c), dim=-1)
        out = torch.einsum("bij,bcj->bci", attn, v).reshape(b, c, h, w)
        return x + self.proj(out)


class UNetVectorField(nn.Module):
    """U-Net velocity field v(t, x) for image flow-matching / rectified flow.

    Compact faithful version of ``torchcfm.models.unet.UNetModel`` (guided-diffusion
    UNet): sinusoidal timestep embedding -> MLP, an encoder of timestep-conditioned
    residual blocks with strided downsampling, a self-attention bottleneck, and a
    symmetric decoder with skip connections, predicting a same-shape velocity image.
    """

    def __init__(
        self,
        in_channels: int = 3,
        model_channels: int = 32,
        out_channels: int = 3,
        channel_mult=(1, 2, 2),
    ) -> None:
        super().__init__()
        self.model_channels = model_channels
        emb_ch = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, emb_ch), nn.SiLU(), nn.Linear(emb_ch, emb_ch)
        )

        self.in_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)

        # Encoder
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        chs = []
        ch = model_channels
        for i, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            self.down_blocks.append(_ResBlock(ch, out_ch, emb_ch))
            ch = out_ch
            chs.append(ch)
            if i != len(channel_mult) - 1:
                self.downsamples.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
            else:
                self.downsamples.append(nn.Identity())

        # Bottleneck
        self.mid_block1 = _ResBlock(ch, ch, emb_ch)
        self.mid_attn = _AttnBlock(ch)
        self.mid_block2 = _ResBlock(ch, ch, emb_ch)

        # Decoder
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for i, mult in list(enumerate(channel_mult))[::-1]:
            skip_ch = chs[i]
            out_ch = model_channels * mult
            self.up_blocks.append(_ResBlock(ch + skip_ch, out_ch, emb_ch))
            ch = out_ch
            if i != 0:
                self.upsamples.append(nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1))
            else:
                self.upsamples.append(nn.Identity())

        self.out_norm = nn.GroupNorm(min(32, ch), ch)
        self.out_conv = nn.Conv2d(ch, out_channels, 3, padding=1)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        emb = self.time_embed(_timestep_embedding(t, self.model_channels))
        h = self.in_conv(x)
        skips = []
        for block, down in zip(self.down_blocks, self.downsamples):
            h = block(h, emb)
            skips.append(h)
            h = down(h)
        h = self.mid_block1(h, emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, emb)
        for block, up in zip(self.up_blocks, self.upsamples):
            skip = skips.pop()
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = block(torch.cat([h, skip], dim=1), emb)
            h = up(h)
        return self.out_conv(F.silu(self.out_norm(h)))


# ============================================================
# Wrappers so torchlens sees a single-tensor signature
# ============================================================


class _MLPWrap(nn.Module):
    """Wrap the MLP field; the example input already packs (x, t) in its columns."""

    def __init__(self, model: MLPVectorField) -> None:
        super().__init__()
        self.model = model

    def forward(self, xt: torch.Tensor) -> torch.Tensor:
        return self.model(xt)


class _UNetWrap(nn.Module):
    """Wrap the UNet field; derive a per-sample timestep from the input batch."""

    def __init__(self, model: UNetVectorField) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = torch.full((x.shape[0],), 0.5, device=x.device, dtype=x.dtype)
        return self.model(t, x)


# ============================================================
# Builders
# ============================================================


def build_mlp_field() -> nn.Module:
    """Build the low-dimensional MLP velocity field (dim=2, time-varying)."""
    return _MLPWrap(MLPVectorField(dim=2, w=64, time_varying=True))


def build_unet_field() -> nn.Module:
    """Build the image U-Net velocity field (guided-diffusion style)."""
    return _UNetWrap(UNetVectorField(in_channels=3, model_channels=32, out_channels=3))


def example_input_mlp() -> torch.Tensor:
    """A batch of 2D states with time appended as the last column: shape (8, 3)."""
    return torch.randn(8, 3)


def example_input_unet() -> torch.Tensor:
    """A small image batch ``(1, 3, 32, 32)`` for the U-Net field."""
    return torch.randn(1, 3, 32, 32)


MENAGERIE_ENTRIES = [
    (
        "Flow-Matching MLP vector field (TorchCFM low-dim velocity net)",
        "build_mlp_field",
        "example_input_mlp",
        "2023",
        "DC",
    ),
    (
        "Flow-Matching / Rectified-Flow U-Net vector field (image velocity net)",
        "build_unet_field",
        "example_input_unet",
        "2023",
        "DC",
    ),
]
