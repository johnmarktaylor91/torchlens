# FAITHFUL REIMPLEMENTATION from arXiv:2306.03110 (no public code) -- A/B codex
"""SwinRDM: SwinRNN+ forecaster with conditional diffusion super-resolution."""

from __future__ import annotations

import torch
from torch import nn


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Partition a feature map into non-overlapping windows.

    Parameters
    ----------
    x:
        Tensor of shape ``(batch, height, width, channels)``.
    window_size:
        Spatial window size.

    Returns
    -------
    torch.Tensor
        Windows of shape ``(num_windows * batch, window_size * window_size, channels)``.
    """
    batch, height, width, channels = x.shape
    x = x.view(
        batch, height // window_size, window_size, width // window_size, window_size, channels
    )
    return x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size * window_size, channels)


def window_reverse(
    windows: torch.Tensor, window_size: int, height: int, width: int, batch: int
) -> torch.Tensor:
    """Reverse window partitioning.

    Parameters
    ----------
    windows:
        Window tokens.
    window_size:
        Spatial window size.
    height:
        Feature-map height.
    width:
        Feature-map width.
    batch:
        Batch size.

    Returns
    -------
    torch.Tensor
        Feature map of shape ``(batch, height, width, channels)``.
    """
    channels = windows.shape[-1]
    x = windows.view(
        batch, height // window_size, width // window_size, window_size, window_size, channels
    )
    return x.permute(0, 1, 3, 2, 4, 5).reshape(batch, height, width, channels)


class SwinTransformerBlock(nn.Module):
    """Single-scale Swin Transformer block with optional shifted windows."""

    def __init__(self, channels: int, num_heads: int, window_size: int, shift: bool) -> None:
        """Initialize a Swin block.

        Parameters
        ----------
        channels:
            Feature channels.
        num_heads:
            Attention heads.
        window_size:
            Local attention window size.
        shift:
            Whether to use shifted windows.
        """
        super().__init__()
        self.window_size = window_size
        self.shift = shift
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * 4), nn.GELU(), nn.Linear(channels * 4, channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply shifted-window self-attention and an MLP.

        Parameters
        ----------
        x:
            Feature map ``(batch, channels, height, width)``.

        Returns
        -------
        torch.Tensor
            Updated feature map.
        """
        batch, channels, height, width = x.shape
        x_nhwc = x.permute(0, 2, 3, 1)
        shifted = torch.roll(
            x_nhwc, shifts=(-self.window_size // 2, -self.window_size // 2), dims=(1, 2)
        )
        if not self.shift:
            shifted = x_nhwc
        windows = window_partition(self.norm1(shifted), self.window_size)
        attended, _ = self.attn(windows, windows, windows, need_weights=False)
        merged = window_reverse(attended, self.window_size, height, width, batch)
        if self.shift:
            merged = torch.roll(
                merged, shifts=(self.window_size // 2, self.window_size // 2), dims=(1, 2)
            )
        x_nhwc = x_nhwc + merged
        x_nhwc = x_nhwc + self.mlp(self.norm2(x_nhwc))
        return x_nhwc.permute(0, 3, 1, 2)


class SwinRNNPlus(nn.Module):
    """Single-scale recurrent Swin forecaster with multi-layer aggregation."""

    def __init__(
        self,
        in_channels: int = 6,
        hidden_channels: int = 32,
        blocks: int = 6,
        window_size: int = 2,
    ) -> None:
        """Initialize SwinRNN+.

        Parameters
        ----------
        in_channels:
            Weather variables per frame.
        hidden_channels:
            Hidden channels.
        blocks:
            Number of Swin blocks in encoder and decoder.
        window_size:
            Window attention size.
        """
        super().__init__()
        self.in_channels = in_channels
        self.cube_embedding = nn.Conv3d(
            in_channels, hidden_channels, kernel_size=(3, 2, 2), padding=(1, 0, 0), stride=(1, 2, 2)
        )
        self.encoder = nn.ModuleList(
            [
                SwinTransformerBlock(
                    hidden_channels, num_heads=4, window_size=window_size, shift=bool(i % 2)
                )
                for i in range(blocks)
            ]
        )
        self.frame_embedding = nn.Conv2d(in_channels, hidden_channels, kernel_size=2, stride=2)
        self.decoder = nn.ModuleList(
            [
                SwinTransformerBlock(
                    hidden_channels, num_heads=4, window_size=window_size, shift=bool(i % 2)
                )
                for i in range(blocks)
            ]
        )
        self.aggregate = nn.Conv2d(hidden_channels * blocks, hidden_channels, kernel_size=1)
        self.predict = nn.ConvTranspose2d(hidden_channels, in_channels, kernel_size=2, stride=2)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """Forecast the next low-resolution weather frame.

        Parameters
        ----------
        history:
            Tensor ``(batch, time, channels, height, width)``.

        Returns
        -------
        torch.Tensor
            Next-frame forecast ``(batch, channels, height, width)``.
        """
        batch, _, channels, _, _ = history.shape
        cube = history.permute(0, 2, 1, 3, 4)
        hidden = self.cube_embedding(cube).mean(dim=2)
        for block in self.encoder:
            hidden = block(hidden)
        current = history[:, -1, :channels]
        decoded = hidden + self.frame_embedding(current)
        features: list[torch.Tensor] = []
        for block in self.decoder:
            decoded = block(decoded)
            features.append(decoded)
        aggregated = self.aggregate(torch.cat(features, dim=1))
        return self.predict(aggregated) + current


class ConditionalDiffusionSR(nn.Module):
    """Conditional diffusion denoiser for weather super-resolution."""

    def __init__(self, channels: int = 6, hidden_channels: int = 32, scale_factor: int = 2) -> None:
        """Initialize the super-resolution denoiser.

        Parameters
        ----------
        channels:
            Weather variables.
        hidden_channels:
            Hidden channels.
        scale_factor:
            Spatial upsampling factor.
        """
        super().__init__()
        self.scale_factor = scale_factor
        self.time_embed = nn.Sequential(
            nn.Linear(2, hidden_channels), nn.SiLU(), nn.Linear(hidden_channels, hidden_channels)
        )
        self.in_conv = nn.Conv2d(
            channels * 2 + hidden_channels, hidden_channels, kernel_size=3, padding=1
        )
        self.down = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=2, padding=1)
        self.mid = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
        )
        self.up = nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=2, stride=2)
        self.out = nn.Conv2d(hidden_channels * 2, channels, kernel_size=3, padding=1)

    def forward(
        self, low_res: torch.Tensor, diffusion_step: torch.Tensor, forecast_step: torch.Tensor
    ) -> torch.Tensor:
        """Denoise a high-resolution weather sample conditioned on low resolution.

        Parameters
        ----------
        low_res:
            Low-resolution forecast.
        diffusion_step:
            Normalized diffusion step.
        forecast_step:
            Normalized recurrent forecast step.

        Returns
        -------
        torch.Tensor
            High-resolution denoised forecast.
        """
        condition = torch.nn.functional.interpolate(
            low_res, scale_factor=self.scale_factor, mode="bilinear"
        )
        noisy = condition + 0.05 * torch.randn_like(condition)
        time = torch.stack([diffusion_step, forecast_step], dim=-1)
        time_features = self.time_embed(time)[:, :, None, None].expand(
            -1, -1, condition.shape[-2], condition.shape[-1]
        )
        hidden = torch.relu(self.in_conv(torch.cat([noisy, condition, time_features], dim=1)))
        down = torch.relu(self.down(hidden))
        mid = self.mid(down) + down
        up = torch.relu(self.up(mid))
        return self.out(torch.cat([up, hidden], dim=1)) + condition


class SwinRDM(nn.Module):
    """Two-stage SwinRNN+ and diffusion super-resolution forecaster."""

    def __init__(self, channels: int = 6) -> None:
        """Initialize SwinRDM.

        Parameters
        ----------
        channels:
            Weather variable channels.
        """
        super().__init__()
        self.forecaster = SwinRNNPlus(in_channels=channels)
        self.super_resolution = ConditionalDiffusionSR(channels=channels)

    def forward(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forecast low-resolution fields and refine them to high resolution.

        Parameters
        ----------
        history:
            Weather history ``(batch, time, channels, height, width)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Low-resolution forecast and high-resolution refinement.
        """
        low = self.forecaster(history)
        batch = history.shape[0]
        diffusion_step = torch.full((batch,), 0.5, device=history.device, dtype=history.dtype)
        forecast_step = torch.full((batch,), 1.0, device=history.device, dtype=history.dtype)
        high = self.super_resolution(low, diffusion_step, forecast_step)
        return low, high


def build_swinrdm() -> SwinRDM:
    """Build a tiny traceable SwinRDM.

    Returns
    -------
    SwinRDM
        Tiny SwinRDM model.
    """
    return SwinRDM()


def example_input_swinrdm() -> torch.Tensor:
    """Create a low-resolution weather history.

    Returns
    -------
    torch.Tensor
        Weather history tensor.
    """
    return torch.randn(1, 3, 6, 8, 8)


MENAGERIE_ZOO = "reimpl-pytorch"
MENAGERIE_ENTRIES = [("SwinRDM", "build_swinrdm", "example_input_swinrdm", 2023, "REIMPL")]
