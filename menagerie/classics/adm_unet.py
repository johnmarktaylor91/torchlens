"""ADM guided-diffusion UNet, 2021.

Paper: Diffusion Models Beat GANs on Image Synthesis (Dhariwal and Nichol;
NeurIPS 2021). Source lineage: OpenAI guided-diffusion UNetModel.

Faithful compact random-init reconstruction of ADM's load-bearing primitives:
sinusoidal timestep embedding, residual blocks modulated by adaptive GroupNorm
scale/shift from the time embedding, attention blocks at selected resolutions,
and a symmetric UNet with skip concatenations.
"""

from __future__ import annotations

from collections.abc import Sequence
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def timestep_embedding(timesteps: Tensor, dim: int) -> Tensor:
    """Create sinusoidal diffusion timestep embeddings.

    Parameters
    ----------
    timesteps
        Integer or floating timestep tensor of shape ``(B,)``.
    dim
        Embedding dimension.

    Returns
    -------
    Tensor
        Sinusoidal embedding matrix.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / half
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class AdaGNResBlock(nn.Module):
    """ADM residual block with timestep-conditioned adaptive GroupNorm."""

    def __init__(
        self, in_ch: int, out_ch: int, emb_dim: int, up: bool = False, down: bool = False
    ) -> None:
        """Create an adaptive residual block.

        Parameters
        ----------
        in_ch
            Input channel count.
        out_ch
            Output channel count.
        emb_dim
            Timestep embedding dimension.
        up
            Whether to upsample before convolution.
        down
            Whether to downsample before convolution.
        """
        super().__init__()
        self.up = up
        self.down = down
        self.norm1 = nn.GroupNorm(4, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.emb = nn.Linear(emb_dim, 2 * out_ch)
        self.norm2 = nn.GroupNorm(4, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def _resize(self, x: Tensor) -> Tensor:
        """Apply the block's optional resolution change.

        Parameters
        ----------
        x
            Feature map.

        Returns
        -------
        Tensor
            Resized feature map.
        """
        if self.up:
            return F.interpolate(x, scale_factor=2, mode="nearest")
        if self.down:
            return F.avg_pool2d(x, 2)
        return x

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        """Apply adaptive residual processing.

        Parameters
        ----------
        x
            Input feature map.
        emb
            Timestep embedding.

        Returns
        -------
        Tensor
            Residual block output.
        """
        resized = self._resize(x)
        h = self.conv1(F.silu(self.norm1(resized)))
        scale, shift = self.emb(emb).unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
        h = self.conv2(F.silu(h))
        return self.skip(resized) + h


class AttentionBlock(nn.Module):
    """ADM spatial self-attention block."""

    def __init__(self, channels: int, heads: int = 2) -> None:
        """Create an attention block.

        Parameters
        ----------
        channels
            Feature channel count.
        heads
            Number of attention heads.
        """
        super().__init__()
        self.heads = heads
        self.norm = nn.GroupNorm(4, channels)
        self.qkv = nn.Conv1d(channels, 3 * channels, 1)
        self.proj = nn.Conv1d(channels, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Apply self-attention over spatial locations.

        Parameters
        ----------
        x
            Input feature map.

        Returns
        -------
        Tensor
            Attention-refined feature map.
        """
        b, c, h, w = x.shape
        y = self.norm(x).reshape(b, c, h * w)
        q, k, v = self.qkv(y).chunk(3, dim=1)
        head_dim = c // self.heads
        q = q.reshape(b, self.heads, head_dim, h * w).transpose(2, 3)
        k = k.reshape(b, self.heads, head_dim, h * w)
        v = v.reshape(b, self.heads, head_dim, h * w).transpose(2, 3)
        attn = torch.softmax(torch.matmul(q, k) / math.sqrt(head_dim), dim=-1)
        out = torch.matmul(attn, v).transpose(2, 3).reshape(b, c, h * w)
        return x + self.proj(out).reshape(b, c, h, w)


class ADMUNet(nn.Module):
    """Compact ADM UNet with AdaGN residual blocks and attention."""

    def __init__(self, channels: int = 16, emb_dim: int = 64) -> None:
        """Create the compact ADM UNet.

        Parameters
        ----------
        channels
            Base channel count.
        emb_dim
            Timestep embedding dimension.
        """
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4),
            nn.SiLU(),
            nn.Linear(emb_dim * 4, emb_dim),
        )
        self.in_conv = nn.Conv2d(3, channels, 3, padding=1)
        self.down1 = AdaGNResBlock(channels, channels, emb_dim)
        self.down2 = AdaGNResBlock(channels, channels * 2, emb_dim, down=True)
        self.attn = AttentionBlock(channels * 2)
        self.mid = AdaGNResBlock(channels * 2, channels * 2, emb_dim)
        self.up1 = AdaGNResBlock(channels * 4, channels, emb_dim, up=True)
        self.up2 = AdaGNResBlock(channels * 2, channels, emb_dim)
        self.out_norm = nn.GroupNorm(4, channels)
        self.out = nn.Conv2d(channels, 3, 3, padding=1)
        self.emb_dim = emb_dim

    def forward(self, x: Tensor, timesteps: Tensor) -> Tensor:
        """Predict diffusion noise for an image and timestep.

        Parameters
        ----------
        x
            Noisy image batch.
        timesteps
            Diffusion timestep indices.

        Returns
        -------
        Tensor
            Noise prediction.
        """
        emb = self.time_mlp(timestep_embedding(timesteps, self.emb_dim))
        h0 = self.in_conv(x)
        h1 = self.down1(h0, emb)
        h2 = self.attn(self.down2(h1, emb))
        mid = self.mid(h2, emb)
        up = self.up1(torch.cat([mid, h2], dim=1), emb)
        up = self.up2(torch.cat([up, h1], dim=1), emb)
        return self.out(F.silu(self.out_norm(up)))


def build() -> nn.Module:
    """Build the compact ADM UNet.

    Returns
    -------
    nn.Module
        Random-init ADM UNet.
    """
    return ADMUNet().eval()


def example_input() -> tuple[Tensor, Tensor]:
    """Return image and timestep inputs.

    Returns
    -------
    tuple[Tensor, Tensor]
        Example noisy image and timestep.
    """
    return torch.randn(1, 3, 32, 32), torch.tensor([10])


MENAGERIE_ENTRIES: Sequence[tuple[str, str, str, str, str]] = [
    ("adm_unet", "build", "example_input", "2021", "E7"),
]
