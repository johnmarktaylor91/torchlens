"""Imaginaire vid2vid / world-consistent vid2vid generator reconstruction.

Paper: Mallya et al., 2020, "World-Consistent Video-to-Video Synthesis".

NVIDIA Imaginaire's vid2vid family uses semantic labels, optical-flow warping
of previous outputs, and guidance images to modulate SPADE residual generator
blocks.  This compact random-init model preserves the Multi-SPADE conditioning
and temporal flow-warp primitive.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class MultiSPADE(nn.Module):
    """SPADE normalization conditioned by labels, warped frame, and guidance."""

    def __init__(self, channels: int, cond_channels: int) -> None:
        """Initialize modulation convolutions."""

        super().__init__()
        self.norm = nn.InstanceNorm2d(channels, affine=False)
        self.shared = nn.Sequential(nn.Conv2d(cond_channels, channels, 3, padding=1), nn.ReLU())
        self.gamma = nn.Conv2d(channels, channels, 3, padding=1)
        self.beta = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """Apply spatially adaptive denormalization."""

        act = self.shared(cond)
        return self.norm(x) * (1 + self.gamma(act)) + self.beta(act)


class Vid2VidBlock(nn.Module):
    """Residual generator block with Multi-SPADE."""

    def __init__(self, channels: int, cond_channels: int) -> None:
        """Initialize two modulated convolutions."""

        super().__init__()
        self.spade1 = MultiSPADE(channels, cond_channels)
        self.spade2 = MultiSPADE(channels, cond_channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """Run one Multi-SPADE residual block."""

        y = self.conv1(F.leaky_relu(self.spade1(x, cond), 0.2))
        y = self.conv2(F.leaky_relu(self.spade2(y, cond), 0.2))
        return x + y


class ImaginaireVid2VidGenerator(nn.Module):
    """Compact vid2vid generator with flow-warped temporal conditioning."""

    def __init__(self, labels: int = 8, channels: int = 32) -> None:
        """Initialize label embedding, flow warper, and generator blocks."""

        super().__init__()
        self.label_embed = nn.Conv2d(labels, channels, 3, padding=1)
        self.guidance = nn.Conv2d(3, channels, 3, padding=1)
        self.prev = nn.Conv2d(3, channels, 3, padding=1)
        self.in_proj = nn.Conv2d(channels * 3, channels, 3, padding=1)
        self.blocks = nn.ModuleList([Vid2VidBlock(channels, channels * 3) for _ in range(2)])
        self.to_rgb = nn.Conv2d(channels, 3, 3, padding=1)

    def forward(self, labels: Tensor, prev_frame: Tensor, flow: Tensor, guidance: Tensor) -> Tensor:
        """Synthesize a frame from labels, warped previous output, and guidance."""

        batch, _, height, width = prev_frame.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, height, device=prev_frame.device),
            torch.linspace(-1, 1, width, device=prev_frame.device),
            indexing="ij",
        )
        grid = torch.stack([xx, yy], dim=-1).unsqueeze(0).expand(batch, -1, -1, -1)
        flow_grid = grid + flow.permute(0, 2, 3, 1) * 0.25
        warped = F.grid_sample(prev_frame, flow_grid, align_corners=True)
        cond = torch.cat(
            [self.label_embed(labels), self.prev(warped), self.guidance(guidance)], dim=1
        )
        x = F.relu(self.in_proj(cond))
        for block in self.blocks:
            x = block(x, cond)
        return torch.tanh(self.to_rgb(x))


def build() -> nn.Module:
    """Build the compact Imaginaire vid2vid generator."""

    return ImaginaireVid2VidGenerator().eval()


def example_input() -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return labels, previous frame, optical flow, and guidance image."""

    return (
        torch.randn(1, 8, 32, 32),
        torch.randn(1, 3, 32, 32),
        torch.randn(1, 2, 32, 32),
        torch.randn(1, 3, 32, 32),
    )


MENAGERIE_ENTRIES = [
    ("imaginaire_vid2vid_generator", "build", "example_input", "2020", "GEN"),
]
