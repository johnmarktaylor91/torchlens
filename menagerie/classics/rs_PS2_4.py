# FAITHFUL REIMPLEMENTATION from a published architecture table (no public code)
#
# "21cmDeep" (candidate queue name) has no GitHub repo under this exact name (confirmed by repo/code
# search; PabloVD/21cmDeepLearning is a different, unrelated density-field-extraction tool). The
# candidate's own triage note identifies the closest concrete match as arXiv:2006.06236, Kwon, Hong &
# Park, "Deep-Learning Study of the 21-cm Differential Brightness Temperature During the Epoch of
# Reionization" (PASP 2021): a CNN that predicts the sliced-averaged neutral hydrogen fraction (x_HI)
# from 21-cm differential-brightness-temperature tomography maps. That paper's Table 1 / Fig. 5 give
# an EXACT layer-by-layer architecture (verified by recomputing every intermediate spatial dimension
# below), which is faithfully reimplemented here at RUNG 4 (no code exists anywhere to vendor/port):
#   Input (3, 200, 200)
#   Conv2D-1: 32 filters, 3x3, stride 1, valid  -> (32, 198, 198); BatchNorm+ReLU
#   MaxPool2D-1: 2x2, stride 2                  -> (32,  99,  99)
#   Conv2D-2: 32 filters, 3x3, stride 1, valid  -> (32,  97,  97); BatchNorm+ReLU
#   MaxPool2D-2: 2x2, stride 2                  -> (32,  48,  48)
#   Conv2D-3: 64 filters, 3x3, stride 1, valid  -> (64,  46,  46); BatchNorm+ReLU
#   MaxPool2D-3: 2x2, stride 2                  -> (64,  23,  23)
#   Flatten                                     -> 33856  (= 64*23*23, matches the paper's table)
#   FC-1 (He-uniform init) -> 64;  BatchNorm+ReLU
#   FC-2 (He-uniform init) -> 32;  BatchNorm+ReLU
#   FC-3 (linear)          -> 1   (predicted neutral hydrogen fraction x_HI)
# All conv/FC layers use ReLU activation (paper Sec. 3); FC weights use He-uniform initialization
# (paper Sec. 3, citing He et al. 2015) which `nn.init.kaiming_uniform_` implements directly.
# The staged build below uses a smaller (32x32) input than the paper's native 200x200 tomography
# slices purely to keep the trace-gated tiny-config model cheap; every layer, channel count, kernel
# size, stride, and ordering is unchanged from the table above (only the flattened FC-1 input width
# scales with the smaller spatial size).
from __future__ import annotations

import torch
from torch import Tensor, nn

MENAGERIE_ZOO = "reimpl-pytorch"


class TwentyOneCmDeepCNN(nn.Module):
    """CNN regressor from 21-cm brightness-temperature tomography maps to neutral hydrogen fraction.

    Faithful reimplementation of the architecture table in arXiv:2006.06236 (Kwon, Hong & Park 2021):
    3 conv+BN+ReLU+maxpool blocks (32, 32, 64 filters, all 3x3 valid convolutions, 2x2 maxpools) then
    a 3-layer FC regression head (64 -> 32 -> 1) with BatchNorm+ReLU between the first two FC layers.
    """

    def __init__(self, in_channels: int = 3, input_size: int = 32) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        with torch.no_grad():
            flat_dim = (
                self.features(torch.zeros(1, in_channels, input_size, input_size))
                .flatten(1)
                .shape[1]
            )
        self.regressor = nn.Sequential(
            nn.Linear(flat_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )
        self._he_uniform_init()

    def _he_uniform_init(self) -> None:
        for module in self.regressor:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, tomography_map: Tensor) -> Tensor:
        """tomography_map: (batch, 3, H, W) stacked 21-cm brightness-temperature slices."""
        features = self.features(tomography_map).flatten(1)
        return self.regressor(features)


def build_21cmdeep() -> nn.Module:
    model = TwentyOneCmDeepCNN(in_channels=3, input_size=32)
    model.eval()
    return model


def example_input_21cmdeep() -> Tensor:
    torch.manual_seed(0)
    return torch.randn(2, 3, 32, 32)  # (batch, 3 stacked brightness-temp slices, 32x32 map)


MENAGERIE_ENTRIES = [
    ("21cmDeep", "build_21cmdeep", "example_input_21cmdeep", 2021, MENAGERIE_ZOO),
]
