"""SPADE / GauGAN: Spatially-Adaptive Denormalization for Semantic Image Synthesis.

Park et al., "Semantic Image Synthesis with Spatially-Adaptive Normalization",
CVPR 2019. arXiv:1903.07291.
Source: https://github.com/NVlabs/SPADE

SPADE's key architectural contribution:
  SPATIALLY-ADAPTIVE DENORMALIZATION: per-pixel gamma/beta (scale/bias) are predicted
  from a segmentation map via small convolutional networks, then used to modulate
  normalized feature maps. This allows the segmentation map to control synthesis
  at every spatial location and every layer.

  x = gamma(seg) * norm(x) + beta(seg)

  where gamma and beta are spatially-varying tensors predicted from the seg map.
  This is the SPADE signature op -- compare to AdaIN (which is spatially uniform).

All models here use random init, tiny sizes, small channels.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Shared SPADE primitive
# ============================================================


class SPADELayer(nn.Module):
    """Spatially-Adaptive Denormalization (SPADE) layer.

    Predicts per-pixel gamma and beta from a segmentation map via 2 small convs.
    Applies: x = gamma(seg) * norm(x) + beta(seg).
    This is the signature op of SPADE/GauGAN.
    """

    def __init__(self, num_features: int, seg_channels: int, hidden: int = 16) -> None:
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        # Small conv network predicts gamma and beta from seg map
        self.seg_conv = nn.Conv2d(seg_channels, hidden, 3, 1, 1)
        self.gamma_conv = nn.Conv2d(hidden, num_features, 3, 1, 1)
        self.beta_conv = nn.Conv2d(hidden, num_features, 3, 1, 1)

    def forward(self, x: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W), seg: (B, num_classes, H, W) -- resize seg to match x spatial size
        seg_resized = F.interpolate(seg, size=x.shape[2:], mode="nearest")
        actv = F.relu(self.seg_conv(seg_resized))
        gamma = self.gamma_conv(actv)  # per-pixel scale (B, C, H, W)
        beta = self.beta_conv(actv)  # per-pixel shift (B, C, H, W)
        # Normalize then spatially-adaptive denormalize
        x_norm = self.norm(x)
        return x_norm * (1 + gamma) + beta


class SPADEResBlock(nn.Module):
    """Residual block using SPADE layers for both BN positions."""

    def __init__(self, ch: int, seg_channels: int) -> None:
        super().__init__()
        self.spade1 = SPADELayer(ch, seg_channels)
        self.conv1 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.spade2 = SPADELayer(ch, seg_channels)
        self.conv2 = nn.Conv2d(ch, ch, 3, 1, 1)

    def forward(self, x: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.relu(self.spade1(x, seg)))
        h = self.conv2(F.relu(self.spade2(h, seg)))
        return x + h


# ============================================================
# MODULE 13: spade_generator
# ============================================================


class SPADEGenerator(nn.Module):
    """SPADE Generator (Park et al. CVPR 2019).

    Takes one-hot segmentation map. For each residual block, computes per-pixel
    gamma/beta from the seg map via SPADE and applies: x = gamma*norm(x) + beta.
    Upsamples from learned constant to output image.
    """

    def __init__(self, seg_channels: int = 8, ch: int = 16) -> None:
        super().__init__()
        # Encode seg map to initial feature map
        self.seg_to_feat = nn.Conv2d(seg_channels, ch * 4, 3, 1, 1)
        # SPADE ResBlocks at different resolutions
        self.resblock1 = SPADEResBlock(ch * 4, seg_channels)
        self.resblock2 = SPADEResBlock(ch * 2, seg_channels)
        self.up1 = nn.ConvTranspose2d(ch * 4, ch * 2, 4, 2, 1)
        self.to_rgb = nn.Conv2d(ch * 2, 3, 3, 1, 1)

    def forward(self, seg: torch.Tensor) -> torch.Tensor:
        # seg: (B, seg_channels, H, W)
        x = self.seg_to_feat(seg)
        x = self.resblock1(x, seg)
        x = F.relu(self.up1(x))
        x = self.resblock2(x, seg)
        return torch.tanh(self.to_rgb(x))


def build_spade_generator() -> nn.Module:
    return SPADEGenerator(seg_channels=8, ch=16)


def example_spade_generator() -> torch.Tensor:
    return torch.randn(1, 8, 16, 16)


# ============================================================
# MODULE 14: spade_gaugan_generator
# ============================================================


class SPADEGauGANGenerator(nn.Module):
    """GauGAN Generator: SPADE + VAE noise injection at input.

    Full GauGAN: input is (seg_map, z_noise from VAE reparameterization).
    The noise z is projected and added to the initial features before SPADE blocks.
    This enables stochastic diverse synthesis (not just deterministic from seg map).
    """

    def __init__(self, seg_channels: int = 8, z_dim: int = 16, ch: int = 16) -> None:
        super().__init__()
        self.z_proj = nn.Linear(z_dim, ch * 4 * 4 * 4)  # project noise to feature space
        self.seg_to_feat = nn.Conv2d(seg_channels, ch * 4, 3, 1, 1)
        self.resblock1 = SPADEResBlock(ch * 4, seg_channels)
        self.up1 = nn.ConvTranspose2d(ch * 4, ch * 2, 4, 2, 1)
        self.resblock2 = SPADEResBlock(ch * 2, seg_channels)
        self.to_rgb = nn.Conv2d(ch * 2, 3, 3, 1, 1)

    def forward(self, seg: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # seg: (B, seg_channels, H, W), z: (B, z_dim)
        # VAE noise injection: project z and reshape to spatial features
        z_feat = self.z_proj(z).view(z.shape[0], -1, 4, 4)
        seg_feat = self.seg_to_feat(F.interpolate(seg, size=(4, 4), mode="nearest"))
        # Combine noise + seg features (GauGAN: stochastic + deterministic input)
        x = seg_feat + z_feat
        x = self.resblock1(x, seg)
        x = F.relu(self.up1(x))
        x = self.resblock2(x, seg)
        return torch.tanh(self.to_rgb(x))


def build_spade_gaugan_generator() -> nn.Module:
    return SPADEGauGANGenerator(seg_channels=8, z_dim=16, ch=16)


def example_spade_gaugan_generator() -> tuple:
    return (torch.randn(1, 8, 32, 32), torch.randn(1, 16))


# ============================================================
# MODULE 15: spade_encoder
# ============================================================


class SPADEEncoder(nn.Module):
    """SPADE/GauGAN VAE Encoder: image -> (mu, logvar).

    Used for KL-divergence loss during training (VAE branch of GauGAN).
    Architecture: conv-down stack -> global avg pool -> two linear heads for mu and logvar.
    """

    def __init__(self, z_dim: int = 16, ch: int = 16) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(ch, ch * 2, 4, 2, 1)
        self.conv3 = nn.Conv2d(ch * 2, ch * 4, 4, 2, 1)
        self.bn1 = nn.InstanceNorm2d(ch)
        self.bn2 = nn.InstanceNorm2d(ch * 2)
        self.bn3 = nn.InstanceNorm2d(ch * 4)
        # After global pool: linear heads for mu and logvar
        self.mu_head = nn.Linear(ch * 4, z_dim)
        self.logvar_head = nn.Linear(ch * 4, z_dim)

    def forward(self, x: torch.Tensor) -> tuple:
        # x: (B, 3, H, W)
        h = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        h = F.leaky_relu(self.bn2(self.conv2(h)), 0.2)
        h = F.leaky_relu(self.bn3(self.conv3(h)), 0.2)
        h = h.mean(dim=[2, 3])  # global average pool
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        return mu, logvar


def build_spade_encoder() -> nn.Module:
    return SPADEEncoder(z_dim=16, ch=16)


def example_spade_encoder() -> torch.Tensor:
    return torch.randn(1, 3, 32, 32)


# ============================================================
# MODULE 16: spade_gaugan_discriminator
# ============================================================


class SPADEGauGANDiscriminator(nn.Module):
    """SPADE/GauGAN PatchGAN Discriminator conditioned on seg map.

    Isola et al. PatchGAN: outputs a patch logit map (not a single scalar).
    Conditioned on segmentation map by concatenating image + seg as input.
    """

    def __init__(self, seg_channels: int = 8, ch: int = 16) -> None:
        super().__init__()
        in_ch = 3 + seg_channels  # concat image + seg
        self.conv1 = nn.Conv2d(in_ch, ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(ch, ch * 2, 4, 2, 1)
        self.bn2 = nn.InstanceNorm2d(ch * 2)
        self.conv3 = nn.Conv2d(ch * 2, ch * 4, 4, 1, 1)
        self.bn3 = nn.InstanceNorm2d(ch * 4)
        # Output: patch logit map (no global pool -> PatchGAN)
        self.conv_out = nn.Conv2d(ch * 4, 1, 4, 1, 1)

    def forward(self, image: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
        # Resize seg to image spatial size
        seg_r = F.interpolate(seg, size=image.shape[2:], mode="nearest")
        x = torch.cat([image, seg_r], dim=1)
        h = F.leaky_relu(self.conv1(x), 0.2)
        h = F.leaky_relu(self.bn2(self.conv2(h)), 0.2)
        h = F.leaky_relu(self.bn3(self.conv3(h)), 0.2)
        return self.conv_out(h)  # patch logit map


def build_spade_gaugan_discriminator() -> nn.Module:
    return SPADEGauGANDiscriminator(seg_channels=8, ch=16)


def example_spade_gaugan_discriminator() -> tuple:
    return (torch.randn(1, 3, 32, 32), torch.randn(1, 8, 32, 32))


# ============================================================
# MODULE 17: spade_multiscale_discriminator
# ============================================================


class SingleScalePatchGAN(nn.Module):
    """Single-scale PatchGAN discriminator (used inside multi-scale)."""

    def __init__(self, in_ch: int, ch: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(ch, ch * 2, 4, 2, 1)
        self.bn2 = nn.InstanceNorm2d(ch * 2)
        self.conv3 = nn.Conv2d(ch * 2, 1, 4, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.leaky_relu(self.conv1(x), 0.2)
        h = F.leaky_relu(self.bn2(self.conv2(h)), 0.2)
        return self.conv3(h)


class SPADEMultiScaleDiscriminator(nn.Module):
    """Multi-scale PatchGAN: runs discriminator at 2 different scales.

    SPADE uses a multi-scale discriminator that applies the same PatchGAN architecture
    to both the original resolution and a 2x downsampled version, returning logits
    from both scales. This allows the discriminator to see both global and local structure.
    """

    def __init__(self, seg_channels: int = 8, ch: int = 16) -> None:
        super().__init__()
        in_ch = 3 + seg_channels
        # Two discriminator heads at different scales
        self.disc_scale1 = SingleScalePatchGAN(in_ch, ch)  # original scale
        self.disc_scale2 = SingleScalePatchGAN(in_ch, ch)  # downsampled scale

    def forward(self, image: torch.Tensor, seg: torch.Tensor) -> list:
        seg_r = F.interpolate(seg, size=image.shape[2:], mode="nearest")
        x = torch.cat([image, seg_r], dim=1)
        # Scale 1: original resolution
        logit1 = self.disc_scale1(x)
        # Scale 2: downsample 2x
        x_down = F.avg_pool2d(x, kernel_size=3, stride=2, padding=1)
        logit2 = self.disc_scale2(x_down)
        return [logit1, logit2]


def build_spade_multiscale_discriminator() -> nn.Module:
    return SPADEMultiScaleDiscriminator(seg_channels=8, ch=16)


def example_spade_multiscale_discriminator() -> tuple:
    return (torch.randn(1, 3, 32, 32), torch.randn(1, 8, 32, 32))


# ============================================================
# MENAGERIE_ENTRIES
# ============================================================

MENAGERIE_ENTRIES = [
    (
        "spade_generator",
        "build_spade_generator",
        "example_spade_generator",
        "2019",
        "DC",
    ),
    (
        "spade_gaugan_generator",
        "build_spade_gaugan_generator",
        "example_spade_gaugan_generator",
        "2019",
        "DC",
    ),
    (
        "spade_encoder",
        "build_spade_encoder",
        "example_spade_encoder",
        "2019",
        "DC",
    ),
    (
        "spade_gaugan_discriminator",
        "build_spade_gaugan_discriminator",
        "example_spade_gaugan_discriminator",
        "2019",
        "DC",
    ),
    (
        "spade_multiscale_discriminator",
        "build_spade_multiscale_discriminator",
        "example_spade_multiscale_discriminator",
        "2019",
        "DC",
    ),
]
