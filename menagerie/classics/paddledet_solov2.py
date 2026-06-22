"""SOLOv2 compact instance segmenter with dynamic mask kernels.

Paper: "SOLOv2: Dynamic and Fast Instance Segmentation" (Wang et al., 2020).

The faithful primitive is the decoupled SOLOv2 mask system: an FPN-like feature
trunk feeds a mask-feature branch, while a parallel location-conditioned kernel
branch predicts per-grid dynamic 1x1 kernels.  Each grid cell's kernel is then
applied to shared mask features to produce instance masks; a small matrix-NMS
style score decay is included to expose the parallel mask-overlap primitive.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    """Convolution, batch normalization, and SiLU activation."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        """Initialize the block.

        Parameters
        ----------
        in_ch:
            Input channel count.
        out_ch:
            Output channel count.
        stride:
            Convolution stride.
        """

        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolutional feature extraction.

        Parameters
        ----------
        x:
            Image feature tensor.

        Returns
        -------
        torch.Tensor
            Activated feature tensor.
        """

        return F.silu(self.bn(self.conv(x)))


class TinyFPN(nn.Module):
    """Small backbone plus lateral top-down FPN."""

    def __init__(self, channels: int = 32) -> None:
        """Initialize the FPN.

        Parameters
        ----------
        channels:
            Shared FPN channel width.
        """

        super().__init__()
        self.stem = ConvBNAct(3, channels)
        self.c3 = ConvBNAct(channels, channels, stride=2)
        self.c4 = ConvBNAct(channels, channels, stride=2)
        self.lat3 = nn.Conv2d(channels, channels, 1)
        self.lat4 = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return two pyramid levels.

        Parameters
        ----------
        x:
            Input image tensor.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Fine and coarse FPN tensors.
        """

        c2 = self.stem(x)
        c3 = self.c3(c2)
        c4 = self.c4(c3)
        p4 = self.lat4(c4)
        p3 = self.lat3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        return p3, p4


class SOLOv2Compact(nn.Module):
    """Compact SOLOv2 with dynamic kernel and mask-feature branches."""

    def __init__(self, channels: int = 32, grid: int = 4, num_classes: int = 5) -> None:
        """Initialize the compact SOLOv2 model.

        Parameters
        ----------
        channels:
            Internal feature width.
        grid:
            Spatial grid size for location-conditioned kernels.
        num_classes:
            Number of category logits per grid cell.
        """

        super().__init__()
        self.grid = grid
        self.backbone = TinyFPN(channels)
        self.mask_feat = nn.Sequential(
            ConvBNAct(channels, channels),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.kernel_head = nn.Sequential(
            ConvBNAct(channels + 2, channels),
            nn.Conv2d(channels, channels, 1),
        )
        self.cls_head = nn.Sequential(
            ConvBNAct(channels + 2, channels), nn.Conv2d(channels, num_classes, 1)
        )

    def _coord_grid(self, feat: torch.Tensor) -> torch.Tensor:
        """Build normalized coordinate channels for SOLO location conditioning.

        Parameters
        ----------
        feat:
            Feature tensor whose spatial size determines the coordinate grid.

        Returns
        -------
        torch.Tensor
            Broadcast coordinate tensor.
        """

        bsz, _, height, width = feat.shape
        yy = torch.linspace(-1.0, 1.0, height, device=feat.device, dtype=feat.dtype)
        xx = torch.linspace(-1.0, 1.0, width, device=feat.device, dtype=feat.dtype)
        yv, xv = torch.meshgrid(yy, xx, indexing="ij")
        coords = torch.stack((xv, yv), dim=0).unsqueeze(0)
        return coords.expand(bsz, -1, -1, -1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict class logits, dynamic masks, and matrix-NMS style scores.

        Parameters
        ----------
        x:
            Input image tensor.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Class logits, dynamic masks, and decayed mask scores.
        """

        p3, p4 = self.backbone(x)
        mask_features = self.mask_feat(p3)
        grid_feat = F.adaptive_avg_pool2d(p4, (self.grid, self.grid))
        grid_feat = torch.cat((grid_feat, self._coord_grid(grid_feat)), dim=1)
        kernels = self.kernel_head(grid_feat).flatten(2).transpose(1, 2)
        cls_logits = self.cls_head(grid_feat).flatten(2).transpose(1, 2)
        masks = torch.einsum("bnc,bchw->bnhw", kernels, mask_features)
        flat_masks = torch.sigmoid(masks).flatten(2)
        inter = torch.matmul(flat_masks, flat_masks.transpose(1, 2))
        areas = flat_masks.sum(dim=-1, keepdim=True)
        union = areas + areas.transpose(1, 2) - inter
        iou = inter / union.clamp_min(1e-6)
        decay = torch.exp(-2.0 * iou.max(dim=-1).values)
        return cls_logits, masks, decay


def build_paddledet_solov2() -> nn.Module:
    """Build compact random-init SOLOv2.

    Returns
    -------
    nn.Module
        SOLOv2 compact model.
    """

    return SOLOv2Compact().eval()


def example_input() -> torch.Tensor:
    """Create a small RGB image.

    Returns
    -------
    torch.Tensor
        Input tensor.
    """

    return torch.randn(1, 3, 64, 64)


MENAGERIE_ENTRIES = [
    ("paddledet_solov2", "build_paddledet_solov2", "example_input", "2020", "DC"),
]
