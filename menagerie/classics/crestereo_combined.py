"""CREStereo/combined geometry stereo compact reconstruction.

Paper/System: CREStereo cascaded recurrent stereo matching and IGEV-Stereo
combined geometry encoding volume (Xu et al., CVPR 2023).

The distinctive primitive is a combined stereo volume: local/all-pairs
correlation is fused with learned geometry/context features and iteratively
indexed by a recurrent updater to refine disparity.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class StereoUpdate(nn.Module):
    """ConvGRU-like disparity update from indexed geometry volume."""

    def __init__(self, channels: int = 24) -> None:
        """Initialize update gates and delta head."""

        super().__init__()
        self.gate = nn.Conv2d(channels + 1, channels, 3, padding=1)
        self.delta = nn.Conv2d(channels, 1, 3, padding=1)

    def forward(self, hidden: Tensor, disparity: Tensor) -> tuple[Tensor, Tensor]:
        """Update hidden state and disparity."""

        hidden = torch.tanh(self.gate(torch.cat([hidden, disparity], dim=1)))
        return hidden, disparity + self.delta(hidden)


class CREStereoCombined(nn.Module):
    """Compact recurrent stereo matcher with combined geometry encoding."""

    def __init__(self, channels: int = 24, max_disp: int = 4) -> None:
        """Initialize feature encoder, geometry fusion, and recurrent updater."""

        super().__init__()
        self.max_disp = max_disp
        self.encoder = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.context = nn.Conv2d(channels, channels, 3, padding=1)
        self.fuse = nn.Conv2d(max_disp + channels, channels, 1)
        self.init_disp = nn.Conv2d(channels, 1, 3, padding=1)
        self.update = StereoUpdate(channels)

    def _corr_volume(self, left: Tensor, right: Tensor) -> Tensor:
        """Build a compact local disparity correlation volume."""

        vols = []
        for disp in range(self.max_disp):
            shifted = F.pad(right[..., :, disp:], (0, disp))
            vols.append((left * shifted).mean(dim=1, keepdim=True))
        return torch.cat(vols, dim=1)

    def forward(self, inputs: tuple[Tensor, Tensor]) -> Tensor:
        """Estimate disparity from left/right stereo images."""

        left_img, right_img = inputs
        left = self.encoder(left_img)
        right = self.encoder(right_img)
        combined = self.fuse(torch.cat([self._corr_volume(left, right), self.context(left)], dim=1))
        disparity = self.init_disp(combined)
        hidden = combined
        for _ in range(3):
            hidden, disparity = self.update(hidden, disparity)
        return disparity


def build() -> nn.Module:
    """Build a compact random-init CREStereo combined-volume model."""

    return CREStereoCombined().eval()


def example_input() -> tuple[Tensor, Tensor]:
    """Return a small stereo image pair."""

    return (torch.randn(1, 3, 24, 32), torch.randn(1, 3, 24, 32))


MENAGERIE_ENTRIES = [
    ("crestereo_combined", "build", "example_input", "2023", "DC"),
]
