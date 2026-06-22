"""CPA camera-pose-aware diffusion transformer compact reconstruction.

Paper: Camera-pose-awareness Diffusion Transformer for Video Generation
(2024).

CPA augments a video diffusion transformer with Pluecker-ray camera-pose
embeddings, a sparse motion encoding module, and temporal attention injection.
The compact version keeps those camera-motion control primitives.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SparseMotionEncoding(nn.Module):
    """Encode camera intrinsics/extrinsics as sparse Pluecker motion tokens."""

    def __init__(self, dim: int = 48) -> None:
        """Initialize Pluecker ray and sparse-token projections."""

        super().__init__()
        self.ray = nn.Linear(6, dim)
        self.compress = nn.Linear(dim, dim)

    def forward(self, pluecker: Tensor) -> Tensor:
        """Return compact sparse camera-motion tokens."""

        rays = torch.sin(self.ray(pluecker))
        return self.compress(rays[:, ::2])


class CPADiffusionTransformer(nn.Module):
    """Compact camera-pose-aware video diffusion transformer."""

    def __init__(self, dim: int = 48) -> None:
        """Initialize latent, text, timestep, camera, and video transformer blocks."""

        super().__init__()
        self.latent = nn.Linear(16, dim)
        self.text = nn.Linear(12, dim)
        self.time = nn.Sequential(nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.camera = SparseMotionEncoding(dim)
        self.temporal_inject = nn.MultiheadAttention(dim, 4, batch_first=True)
        layer = nn.TransformerEncoderLayer(dim, 4, dim * 2, batch_first=True)
        self.video = nn.TransformerEncoder(layer, 1)
        self.noise = nn.Linear(dim, 16)

    def forward(self, inputs: tuple[Tensor, Tensor, Tensor, Tensor]) -> Tensor:
        """Predict video diffusion noise under camera-pose control."""

        latents, text, timestep, pluecker = inputs
        bsz, frames, patches, channels = latents.shape
        tokens = self.latent(latents).view(bsz, frames * patches, -1)
        tokens = tokens + self.text(text).unsqueeze(1) + self.time(timestep[:, None]).unsqueeze(1)
        cam_tokens = self.camera(pluecker)
        injected = self.temporal_inject(tokens, cam_tokens, cam_tokens, need_weights=False)[0]
        tokens = self.video(tokens + injected)
        return self.noise(tokens).view(bsz, frames, patches, channels)


def build() -> nn.Module:
    """Build a compact random-init CPA video diffusion transformer."""

    return CPADiffusionTransformer().eval()


def example_input() -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return video latents, text embedding, diffusion time, and Pluecker rays."""

    return (
        torch.randn(1, 3, 4, 16),
        torch.randn(1, 12),
        torch.tensor([0.25]),
        torch.randn(1, 8, 6),
    )


MENAGERIE_ENTRIES = [
    ("CPA", "build", "example_input", "2024", "DC"),
]
