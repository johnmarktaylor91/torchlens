"""BSRGAN x4 super-resolution generator (RRDBNet: Residual-in-Residual Dense Blocks).

Zhang et al., ICCV 2021 -- "Designing a Practical Degradation Model for Deep Blind Image
Super-Resolution" (BSRGAN). arXiv:2103.14006. Source: github.com/cszn/BSRGAN.

BSRGAN's generator is the ESRGAN RRDBNet. Its DISTINCTIVE primitive is the
Residual-in-Residual Dense Block (RRDB):
  * a Residual Dense Block (RDB) = 5 conv layers where each conv sees the concatenation of
    ALL preceding feature maps (dense connectivity), with a residual scaling (x0.2) skip.
  * an RRDB = 3 stacked RDBs wrapped in another residual-scaling skip (residual-in-residual).
RRDBNet = conv_first -> N x RRDB -> trunk conv -> (pixel-shuffle / nearest upsample) x2
for x4 -> HR conv -> conv_last. No batch-norm (removed vs SRGAN, a key ESRGAN/BSRGAN choice).

Faithful compact reimpl: reduced channels (nf=32) and trunk depth (nb=4 RRDBs), x4 upscale
via two nearest-neighbor + conv stages (the BSRGAN repo's upsample path). Random init.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualDenseBlock_5C(nn.Module):
    """5-conv dense block: each conv consumes all preceding features, x0.2 residual scaling."""

    def __init__(self, nf: int = 32, gc: int = 16) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual-in-Residual Dense Block: 3 stacked RDBs + x0.2 residual-in-residual skip."""

    def __init__(self, nf: int = 32, gc: int = 16) -> None:
        super().__init__()
        self.rdb1 = ResidualDenseBlock_5C(nf, gc)
        self.rdb2 = ResidualDenseBlock_5C(nf, gc)
        self.rdb3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rdb3(self.rdb2(self.rdb1(x)))
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """BSRGAN/ESRGAN RRDBNet x4 generator (no batch-norm, nearest-upsample path)."""

    def __init__(
        self, in_nc: int = 3, out_nc: int = 3, nf: int = 32, nb: int = 4, gc: int = 16
    ) -> None:
        super().__init__()
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1)
        self.trunk = nn.Sequential(*[RRDB(nf, gc) for _ in range(nb)])
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1)
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.hr_conv = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.trunk(fea))
        fea = fea + trunk
        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode="nearest")))
        fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode="nearest")))
        out = self.conv_last(self.lrelu(self.hr_conv(fea)))
        return out


def build_bsrgan_rrdb_x4() -> nn.Module:
    """BSRGAN x4 RRDBNet super-resolution generator (compact: nf=32, nb=4)."""
    return RRDBNet().eval()


def example_input() -> torch.Tensor:
    """Low-resolution RGB image (1, 3, 32, 32) -> x4 -> (1, 3, 128, 128)."""
    return torch.randn(1, 3, 32, 32)


MENAGERIE_ENTRIES = [
    (
        "BSRGAN x4 generator (RRDBNet, residual-in-residual dense blocks)",
        "build_bsrgan_rrdb_x4",
        "example_input",
        "2021",
        "DC",
    ),
]
