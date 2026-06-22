"""Decoupled body-and-edge semantic segmenter.

Yu et al. (ECCV 2020), "Improving Semantic Segmentation via Decoupled Body
and Edge Supervision."  DecoupleSegNets explicitly split low-frequency object
body and high-frequency object edge information, then let the two branches
interact before producing semantic logits.  This compact reconstruction keeps
the load-bearing flow-field warping primitive: a learned body-flow branch warps
features with ``grid_sample`` and a residual edge branch is derived from the
pre-warp minus warped body feature.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    """Convolution, batch normalization, and ReLU."""

    def __init__(self, in_channels: int, out_channels: int, dilation: int = 1) -> None:
        """Initialize the block.

        Parameters
        ----------
        in_channels:
            Input channel count.
        out_channels:
            Output channel count.
        dilation:
            Convolution dilation.
        """

        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the block.

        Parameters
        ----------
        x:
            Input feature map.

        Returns
        -------
        torch.Tensor
            Output feature map.
        """

        return self.net(x)


class CompactDecoupledSegNet(nn.Module):
    """Compact body/edge decoupled segmentation model."""

    def __init__(self, classes: int = 7, width: int = 16) -> None:
        """Initialize the model.

        Parameters
        ----------
        classes:
            Number of segmentation classes.
        width:
            Feature width.
        """

        super().__init__()
        self.stem = ConvBNReLU(3, width)
        self.body = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            ConvBNReLU(width, width, dilation=2),
        )
        self.body_flow = nn.Conv2d(width * 2, 2, 3, padding=1)
        self.edge = nn.Sequential(
            ConvBNReLU(width, width),
            nn.Conv2d(width, 1, 1),
        )
        self.edge_embed = nn.Conv2d(1, width, 1)
        self.fuse = ConvBNReLU(width * 2, width)
        self.seg_head = nn.Conv2d(width, classes, 1)
        self.edge_head = nn.Conv2d(width, 1, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict semantic and edge logits.

        Parameters
        ----------
        x:
            RGB image tensor.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Semantic logits and edge logits.
        """

        feat = self.stem(F.avg_pool2d(x, 2))
        coarse_body = self.body(feat)
        flow = torch.tanh(self.body_flow(torch.cat([feat, coarse_body], dim=1)))
        height, width = feat.shape[-2:]
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=x.device),
            torch.linspace(-1.0, 1.0, width, device=x.device),
            indexing="ij",
        )
        base_grid = torch.stack([xx, yy], dim=-1).unsqueeze(0)
        warp_grid = base_grid + 0.1 * flow.permute(0, 2, 3, 1)
        body = F.grid_sample(coarse_body, warp_grid, align_corners=False)
        residual_edge = feat - body
        edge_logits = self.edge(residual_edge)
        edge_gate = torch.sigmoid(edge_logits)
        edge_feat = self.edge_embed(edge_gate)
        fused = self.fuse(torch.cat([body * (1.0 + edge_gate), edge_feat], dim=1))
        seg = F.interpolate(self.seg_head(fused), size=x.shape[2:], mode="bilinear")
        edge = F.interpolate(self.edge_head(fused) + edge_logits, size=x.shape[2:], mode="bilinear")
        return seg, edge


def build() -> nn.Module:
    """Build compact Decoupled SegNet.

    Returns
    -------
    nn.Module
        Random-init model in evaluation mode.
    """

    return CompactDecoupledSegNet().eval()


def example_input() -> torch.Tensor:
    """Return a small RGB image.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(1, 3, 48, 48)``.
    """

    return torch.randn(1, 3, 48, 48)


MENAGERIE_ENTRIES = [
    ("paddleseg_decoupled_segnet", "build", "example_input", "2020", "DC"),
]
