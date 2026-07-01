# SOURCE: vendored from SS47816/DriveSceneGen @ main (DriveSceneGen/scripts/train.py, DriveSceneGen/scripts/generation.py)
"""DriveSceneGen (RA-L 2024) -- generates diverse, realistic driving scenarios from scratch.

The paper's "novelty" is entirely in the DATA representation and pipeline, not the network
architecture: real-world Waymo Motion Dataset trajectories + map elements are rasterized into
RGB "pattern" images (dx/dy agent motion channels stacked with map geometry), a standard
denoising diffusion probabilistic model (DDPM) is trained to generate new pattern images from
noise, and a separate downstream vectorization stage (graph extraction, spline fitting -- not a
neural network) converts the generated raster back into vector map/agent trajectories. The
diffusion network itself, per the official repo's train.py, is `diffusers.UNet2DModel`
constructed and used completely unmodified (stock down/up ResNet blocks, no custom attention,
no architectural additions) -- so this is a real, unmodified library model (rung 1 material),
staged as a module because the diffusers UNet forward needs two positional inputs
(noisy sample tensor + integer diffusion timestep) rather than a single input tensor.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from diffusers import UNet2DModel

MENAGERIE_ZOO = "vendored-pytorch"


class DriveSceneGenUNetWrapper(nn.Module):
    """Wraps the real diffusers UNet2DModel exactly as constructed in
    DriveSceneGen/scripts/train.py, tiny-sized for tracing, returning a single tensor
    (UNet2DModel.forward natively returns a UNet2DOutput dataclass with `.sample`)."""

    def __init__(self):
        super().__init__()
        self.unet = UNet2DModel(
            sample_size=(32, 32),  # real repo uses (256, 256); shrunk for tracing
            in_channels=3,  # 3 for RGB pattern image, unmodified from repo
            out_channels=3,
            layers_per_block=1,  # real repo uses 2
            block_out_channels=(32, 64),  # real repo uses (64, 128, 256, 512)
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "UpBlock2D",
            ),
        )

    def forward(self, sample: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        return self.unet(sample, timestep, return_dict=False)[0]


def build_drivescenegen():
    return DriveSceneGenUNetWrapper()


def example_input_drivescenegen():
    return (torch.randn(1, 3, 32, 32), torch.LongTensor([10]))


MENAGERIE_ENTRIES = [
    (
        "DriveSceneGen",
        build_drivescenegen,
        example_input_drivescenegen,
        2024,
        MENAGERIE_ZOO,
    ),
]
