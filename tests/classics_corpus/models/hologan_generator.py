"""HoloGAN: Unsupervised Learning of 3D Representations from Natural Images.

Nguyen-Phuoc et al. (Oxford / Adobe) 2019.  arXiv:1904.01326.
Source: https://github.com/thunguyenphuoc/HoloGAN

HoloGAN's distinctive primitives:
  - **3D feature volume**: the generator synthesizes a 3D latent feature volume
    (D x H x W x C) instead of a 2D feature map, enabling explicit 3D reasoning.
  - **Learned rigid-body transform**: a rotation/translation matrix (predicted from
    the camera-angle conditioning) is applied to the 3D feature volume via a
    spatial transformer (3D affine grid sampling).
  - **3D-to-2D projection**: the transformed 3D volume is collapsed along the depth
    axis (sum/mean projection) to produce a 2D feature map which is then upsampled
    to the final image by 2D conv blocks.
  - No explicit scene graph or NeRF rendering -- the key insight is that learning a
    3D representation with rigid-body transforms bakes equivariance to viewpoint.

Here we reproduce:
  1. Noise z -> 3D latent volume via a 3D transposed-conv stack.
  2. Affine 3D grid sampling to apply a rotation matrix derived from pose code.
  3. Sum-projection to 2D.
  4. 2D upsample conv blocks -> RGB.

Random init, CPU, very small spatial volumes for compact tracing.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PoseToTransform(nn.Module):
    """Maps a pose/rotation code to a 3x4 affine matrix for 3D volume rotation."""

    def __init__(self, pose_dim: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(pose_dim, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 12),  # 3x4 affine matrix
        )
        # Initialize to near-identity
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, 0.0)

    def forward(self, pose: torch.Tensor) -> torch.Tensor:
        B = pose.shape[0]
        theta = self.net(pose).view(B, 3, 4)
        # Add identity to keep near-identity transforms at init
        eye = torch.eye(3, 4, device=theta.device).unsqueeze(0)
        return theta + eye


class HoloGAN3DEncoder(nn.Module):
    """z -> 3D feature volume via 3D transposed convolutions."""

    def __init__(self, z_dim: int = 32, nf: int = 8, vol_size: int = 4) -> None:
        super().__init__()
        # Project z to a 1x1x1 spatial starting cube
        self.proj = nn.Linear(z_dim, nf * 8)
        # 3D transposed conv stack: 1->2->4  (vol_size=4 after 2 blocks)
        self.deconv1 = nn.ConvTranspose3d(nf * 8, nf * 4, 4, 1, 0)  # 1->4
        self.bn1 = nn.BatchNorm3d(nf * 4)
        self.deconv2 = nn.ConvTranspose3d(nf * 4, nf * 2, 3, 1, 1)  # 4->4 (same)
        self.bn2 = nn.BatchNorm3d(nf * 2)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        x = self.proj(z).view(B, -1, 1, 1, 1)  # (B, nf*8, 1, 1, 1)
        x = F.relu(self.bn1(self.deconv1(x)))  # (B, nf*4, 4, 4, 4)
        x = F.relu(self.bn2(self.deconv2(x)))  # (B, nf*2, 4, 4, 4)
        return x


class Rigid3DTransform(nn.Module):
    """Apply learned affine transform to a 3D feature volume via grid_sample."""

    def forward(self, vol: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        # vol: (B, C, D, H, W),  theta: (B, 3, 4)
        grid = F.affine_grid(theta, vol.shape, align_corners=True)
        return F.grid_sample(vol, grid, align_corners=True, mode="bilinear")


class Project3Dto2D(nn.Module):
    """Collapse depth dimension via sum-projection to get 2D feature map."""

    def forward(self, vol: torch.Tensor) -> torch.Tensor:
        # vol: (B, C, D, H, W) -> (B, C, H, W)
        return vol.sum(dim=2)


class HoloGANGenerator(nn.Module):
    """HoloGAN generator: z + pose -> 3D volume -> rigid transform -> project -> image."""

    def __init__(
        self,
        z_dim: int = 24,
        pose_dim: int = 8,
        nf: int = 8,
    ) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.pose_dim = pose_dim
        self.enc3d = HoloGAN3DEncoder(z_dim, nf)
        self.pose_to_tf = PoseToTransform(pose_dim)
        self.transform = Rigid3DTransform()
        self.project = Project3Dto2D()
        # 2D refinement + upsample: 4x4 -> 8x8 -> 16x16
        self.up1 = nn.Sequential(
            nn.Conv2d(nf * 2, nf * 2, 3, padding=1),
            nn.BatchNorm2d(nf * 2),
            nn.ReLU(True),
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(nf * 2, nf, 3, padding=1),
            nn.BatchNorm2d(nf),
            nn.ReLU(True),
        )
        self.to_rgb = nn.Sequential(
            nn.Conv2d(nf, 3, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, z_dim + pose_dim)
        z = x[:, : self.z_dim]
        pose = x[:, self.z_dim :]
        vol = self.enc3d(z)  # (B, nf*2, D, H, W)
        theta = self.pose_to_tf(pose)  # (B, 3, 4)
        vol = self.transform(vol, theta)  # (B, nf*2, D, H, W)
        feat = self.project(vol)  # (B, nf*2, H, W)
        feat = F.interpolate(feat, scale_factor=2.0, mode="bilinear", align_corners=False)
        feat = self.up1(feat)
        feat = F.interpolate(feat, scale_factor=2.0, mode="bilinear", align_corners=False)
        feat = self.up2(feat)
        return self.to_rgb(feat)


def build_hologan_generator() -> nn.Module:
    return HoloGANGenerator()


def example_input() -> torch.Tensor:
    # z (24) + pose code (8)
    return torch.randn(1, 24 + 8)


MENAGERIE_ENTRIES = [
    (
        "HoloGAN Generator (3D feature volume + rigid-body transform + depth projection)",
        "build_hologan_generator",
        "example_input",
        "2019",
        "DC",
    ),
]
