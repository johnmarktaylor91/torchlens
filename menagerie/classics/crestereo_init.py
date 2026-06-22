"""CREStereo initial-disparity compact reconstruction.

Paper/System: CREStereo cascaded recurrent stereo matching and IGEV-Stereo
(Xu et al., CVPR 2023).

This target emphasizes the initialization path: a learned geometry encoding
volume regresses a strong starting disparity before recurrent refinement.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class CREStereoInit(nn.Module):
    """Compact geometry-volume initial disparity regressor."""

    def __init__(self, channels: int = 24, max_disp: int = 5) -> None:
        """Initialize stereo feature encoder and geometry-volume head."""

        super().__init__()
        self.max_disp = max_disp
        self.encoder = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.gev = nn.Sequential(
            nn.Conv2d(max_disp + channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, 1, 3, padding=1),
        )
        self.context = nn.Conv2d(channels, channels, 1)

    def forward(self, inputs: tuple[Tensor, Tensor]) -> Tensor:
        """Regress an initial disparity from local correlation and context."""

        left_img, right_img = inputs
        left = self.encoder(left_img)
        right = self.encoder(right_img)
        corr = []
        for disp in range(self.max_disp):
            shifted = F.pad(right[..., :, disp:], (0, disp))
            corr.append((left * shifted).mean(dim=1, keepdim=True))
        volume = torch.cat([*corr, self.context(left)], dim=1)
        return self.gev(volume)


def build() -> nn.Module:
    """Build a compact random-init CREStereo initialization model."""

    return CREStereoInit().eval()


def example_input() -> tuple[Tensor, Tensor]:
    """Return a small stereo image pair."""

    return (torch.randn(1, 3, 24, 32), torch.randn(1, 3, 24, 32))


MENAGERIE_ENTRIES = [
    ("crestereo_init", "build", "example_input", "2023", "DC"),
]
