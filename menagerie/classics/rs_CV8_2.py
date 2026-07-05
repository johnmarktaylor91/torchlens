# SOURCE: vendored from zhuhd15/synapse_pytorch @ 5b8eaf9
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

MENAGERIE_ZOO = "vendored-pytorch"


class ResUnetIsoBlock(nn.Module):
    """Vendored isotropic residual block from synapse_pytorch."""

    def __init__(self, in_planes: int, out_planes: int) -> None:
        """Initialize the isotropic residual block."""
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv3d(
                in_planes,
                out_planes,
                kernel_size=(3, 3, 3),
                stride=1,
                padding=(1, 1, 1),
                bias=False,
            ),
            nn.BatchNorm3d(out_planes),
            nn.ELU(alpha=1, inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv3d(
                out_planes,
                out_planes,
                kernel_size=(3, 3, 3),
                stride=1,
                padding=(1, 1, 1),
                bias=False,
            ),
            nn.BatchNorm3d(out_planes),
            nn.ELU(alpha=1, inplace=True),
            nn.Conv3d(
                out_planes,
                out_planes,
                kernel_size=(3, 3, 3),
                stride=1,
                padding=(1, 1, 1),
                bias=False,
            ),
            nn.BatchNorm3d(out_planes),
        )
        self.block3 = nn.ELU(alpha=1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the isotropic residual block."""
        residual = self.block1(x)
        out = residual + self.block2(residual)
        return self.block3(out)


class ResUnetAnisoBlock(nn.Module):
    """Vendored anisotropic residual block from synapse_pytorch."""

    def __init__(self, in_planes: int, out_planes: int) -> None:
        """Initialize the anisotropic residual block."""
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv3d(
                in_planes,
                out_planes,
                kernel_size=(1, 3, 3),
                stride=1,
                padding=(0, 1, 1),
                bias=False,
            ),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv3d(
                out_planes,
                out_planes,
                kernel_size=(1, 3, 3),
                stride=1,
                padding=(0, 1, 1),
                bias=False,
            ),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                out_planes,
                out_planes,
                kernel_size=(1, 3, 3),
                stride=1,
                padding=(0, 1, 1),
                bias=False,
            ),
            nn.BatchNorm3d(out_planes),
        )
        self.block3 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the anisotropic residual block."""
        residual = self.block1(x)
        out = residual + self.block2(residual)
        return self.block3(out)


class ResUnet(nn.Module):
    """Vendored CleftNet residual 3D U-Net."""

    def __init__(self, in_num: int = 1, out_num: int = 1, filters: list[int] | None = None) -> None:
        """Initialize the residual U-Net encoder and decoder."""
        super().__init__()
        self.filters = [4, 8, 12, 16, 20] if filters is None else filters
        self.layer_num = len(self.filters)
        self.aniso_num = 3

        self.down_c = nn.ModuleList(
            [ResUnetAnisoBlock(in_num, self.filters[0])]
            + [
                ResUnetAnisoBlock(self.filters[x], self.filters[x + 1])
                for x in range(self.aniso_num - 1)
            ]
            + [
                ResUnetIsoBlock(self.filters[x], self.filters[x + 1])
                for x in range(self.aniso_num - 1, self.layer_num - 2)
            ],
        )

        self.down_s = nn.ModuleList(
            [nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)) for _ in range(self.aniso_num)]
            + [
                nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
                for _ in range(self.aniso_num, self.layer_num - 1)
            ],
        )

        self.center = ResUnetIsoBlock(self.filters[-2], self.filters[-1])

        self.up_s = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Upsample(scale_factor=(2, 2, 2), mode="trilinear", align_corners=False),
                    nn.Conv3d(
                        self.filters[self.layer_num - 1 - x],
                        self.filters[self.layer_num - 2 - x],
                        kernel_size=(3, 3, 3),
                        stride=1,
                        padding=(1, 1, 1),
                        bias=True,
                    ),
                )
                for x in range(self.layer_num - self.aniso_num - 1)
            ]
            + [
                nn.Sequential(
                    nn.Upsample(scale_factor=(1, 2, 2), mode="trilinear", align_corners=False),
                    nn.Conv3d(
                        self.filters[self.layer_num - 1 - x],
                        self.filters[self.layer_num - 2 - x],
                        kernel_size=(1, 3, 3),
                        stride=1,
                        padding=(0, 1, 1),
                        bias=True,
                    ),
                )
                for x in range(1, self.aniso_num + 1)
            ],
        )

        self.up_c = nn.ModuleList(
            [
                ResUnetIsoBlock(
                    self.filters[self.layer_num - 2 - x], self.filters[self.layer_num - 2 - x]
                )
                for x in range(self.layer_num - self.aniso_num - 1)
            ]
            + [
                ResUnetAnisoBlock(
                    self.filters[self.layer_num - 2 - x], self.filters[self.layer_num - 2 - x]
                )
                for x in range(1, self.aniso_num)
            ]
            + [
                nn.Sequential(
                    ResUnetAnisoBlock(self.filters[0], self.filters[0]),
                    nn.Conv3d(
                        self.filters[0],
                        out_num,
                        kernel_size=(1, 3, 3),
                        stride=1,
                        padding=(0, 1, 1),
                        bias=True,
                    ),
                ),
            ],
        )

        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm3d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the CleftNet residual U-Net forward pass."""
        down_u: list[torch.Tensor | None] = [None] * (self.layer_num - 1)
        for index in range(self.layer_num - 1):
            down_u[index] = self.down_c[index](x)
            x = self.down_s[index](down_u[index])

        x = self.center(x)

        for index in range(self.layer_num - 1):
            skip = down_u[self.layer_num - 2 - index]
            if skip is None:
                raise RuntimeError("Missing residual U-Net skip tensor.")
            x = skip + self.up_s[index](x)
            x = F.relu(x)
            x = self.up_c[index](x)
            x = torch.sigmoid(x)
        return x


def build_cleftnet() -> ResUnet:
    """Build a traceable vendored CleftNet model."""
    return ResUnet().eval()


def example_input_cleftnet() -> torch.Tensor:
    """Return a sample CleftNet EM-volume input."""
    return torch.randn(1, 1, 8, 32, 32)


MENAGERIE_ENTRIES = [
    ("CleftNet", "build_cleftnet", "example_input_cleftnet", 2021, "CV8-CLEFTNET"),
]
