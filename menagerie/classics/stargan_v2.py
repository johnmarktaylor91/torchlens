"""StarGAN-v2: Multi-domain image translation with style codes.

Choi et al., "StarGAN v2: Diverse Image Synthesis for Multiple Domains",
CVPR 2020. arXiv:1912.01865.
Source: https://github.com/clovaai/stargan-v2

StarGAN-v2 key contributions:
  1. STYLE CODE: a per-domain style code (a latent vector) replaces the domain one-hot
     label. Each domain has a style space; the style code captures domain-specific
     appearance variation.
  2. AdaIN MODULATION: style code modulates instance-normalized features via
     AdaIN (Adaptive Instance Normalization): x = sigma(style)*norm(x) + mu(style),
     where sigma, mu are predicted from the style code.
  3. MULTI-TASK DISCRIMINATOR: one discriminator with N output heads, one per domain.
  4. MAPPING NETWORK: z + domain_idx -> style code (learned domain-specific style space).
  5. STYLE ENCODER: image + domain_idx -> style code (for style reference transfer).

All models are compact random-init CPU models with tiny spatial sizes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Shared AdaIN primitive
# ============================================================


class AdaIN(nn.Module):
    """Adaptive Instance Normalization with style code.

    Signature op of StarGAN-v2: style_code -> (gamma, beta) via linear layers,
    then apply: x = gamma * InstanceNorm(x) + beta.
    """

    def __init__(self, num_features: int, style_dim: int) -> None:
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.gamma = nn.Linear(style_dim, num_features)
        self.beta = nn.Linear(style_dim, num_features)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W), style: (B, style_dim)
        gamma = self.gamma(style).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        beta = self.beta(style).unsqueeze(-1).unsqueeze(-1)
        return self.norm(x) * (1 + gamma) + beta


class AdaINResBlock(nn.Module):
    """Residual block with AdaIN modulation (StarGAN-v2 synthesis block)."""

    def __init__(self, ch: int, style_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.adain1 = AdaIN(ch, style_dim)
        self.adain2 = AdaIN(ch, style_dim)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.relu(self.adain1(x, style)))
        h = self.conv2(F.relu(self.adain2(h, style)))
        return x + h


# ============================================================
# MODULE 18: stargan_v2_generator
# ============================================================


class StarGANv2Generator(nn.Module):
    """StarGAN-v2 Generator: AdaIN-modulated ResBlk image translation.

    Encoder (conv-down) -> bottleneck ResBlks with AdaIN modulated by style_code
    -> decoder (conv-up). The style code replaces the domain label, allowing
    diverse synthesis within each domain.
    Input: (image (1,3,H,W), style_code (1, style_dim)).
    """

    def __init__(self, style_dim: int = 16, ch: int = 16) -> None:
        super().__init__()
        # Encoder: image -> features
        self.enc1 = nn.Conv2d(3, ch, 3, 1, 1)
        self.enc2 = nn.Conv2d(ch, ch * 2, 4, 2, 1)
        self.enc3 = nn.Conv2d(ch * 2, ch * 4, 4, 2, 1)
        # Bottleneck with AdaIN modulation
        self.bottleneck1 = AdaINResBlock(ch * 4, style_dim)
        self.bottleneck2 = AdaINResBlock(ch * 4, style_dim)
        # Decoder: features -> image
        self.dec1 = nn.ConvTranspose2d(ch * 4, ch * 2, 4, 2, 1)
        self.dec2 = nn.ConvTranspose2d(ch * 2, ch, 4, 2, 1)
        self.to_rgb = nn.Conv2d(ch, 3, 3, 1, 1)
        self.in1 = nn.InstanceNorm2d(ch * 2)
        self.in2 = nn.InstanceNorm2d(ch)

    def forward(self, image: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        # Encode
        h = F.leaky_relu(self.enc1(image), 0.2)
        h = F.leaky_relu(self.enc2(h), 0.2)
        h = F.leaky_relu(self.enc3(h), 0.2)
        # Bottleneck with AdaIN style modulation
        h = self.bottleneck1(h, style)
        h = self.bottleneck2(h, style)
        # Decode
        h = F.leaky_relu(self.in1(self.dec1(h)), 0.2)
        h = F.leaky_relu(self.in2(self.dec2(h)), 0.2)
        return torch.tanh(self.to_rgb(h))


def build_stargan_v2_generator() -> nn.Module:
    return StarGANv2Generator(style_dim=16, ch=16)


def example_stargan_v2_generator() -> tuple:
    return (torch.randn(1, 3, 32, 32), torch.randn(1, 16))


# ============================================================
# MODULE 19: stargan_v2_discriminator
# ============================================================


class StarGANv2Discriminator(nn.Module):
    """StarGAN-v2 Multi-task Discriminator with per-domain output heads.

    Architecture: shared conv-down backbone -> flatten -> N domain-specific linear heads.
    At inference, select the head for the target domain. This avoids the mode-collapse
    issue of a single conditional discriminator.
    Input: (image, domain_idx).
    """

    def __init__(self, num_domains: int = 4, ch: int = 16) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(ch, ch * 2, 4, 2, 1)
        self.conv3 = nn.Conv2d(ch * 2, ch * 4, 4, 2, 1)
        feat_dim = ch * 4 * 4 * 4
        # N domain-specific linear heads
        self.domain_heads = nn.ModuleList([nn.Linear(feat_dim, 1) for _ in range(num_domains)])
        self.num_domains = num_domains

    def forward(self, image: torch.Tensor, domain_idx: int) -> torch.Tensor:
        h = F.leaky_relu(self.conv1(image), 0.2)
        h = F.leaky_relu(self.conv2(h), 0.2)
        h = F.leaky_relu(self.conv3(h), 0.2)
        h_flat = h.view(h.shape[0], -1)
        # Select domain-specific head
        return self.domain_heads[domain_idx](h_flat)


def build_stargan_v2_discriminator() -> nn.Module:
    return StarGANv2Discriminator(num_domains=4, ch=16)


def example_stargan_v2_discriminator() -> tuple:
    return (torch.randn(1, 3, 32, 32), 0)


# ============================================================
# MODULE 20: stargan_v2_mapping_network
# ============================================================


class StarGANv2MappingNetwork(nn.Module):
    """StarGAN-v2 Mapping Network: z + domain -> style code.

    A shared MLP backbone maps z to a shared representation, then
    domain-specific linear heads project to domain style codes.
    Input: (z, domain_idx).
    """

    def __init__(self, z_dim: int = 16, style_dim: int = 16, num_domains: int = 4) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(z_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        # Domain-specific output heads
        self.domain_heads = nn.ModuleList([nn.Linear(64, style_dim) for _ in range(num_domains)])

    def forward(self, z: torch.Tensor, domain_idx: int) -> torch.Tensor:
        h = self.shared(z)
        return self.domain_heads[domain_idx](h)


def build_stargan_v2_mapping_network() -> nn.Module:
    return StarGANv2MappingNetwork(z_dim=16, style_dim=16, num_domains=4)


def example_stargan_v2_mapping_network() -> tuple:
    return (torch.randn(1, 16), 0)


# ============================================================
# MODULE 21: stargan_v2_style_encoder
# ============================================================


class StarGANv2StyleEncoder(nn.Module):
    """StarGAN-v2 Style Encoder: image + domain -> style code.

    Conv-down backbone -> global average pool -> domain-specific linear head.
    Used for reference-guided style transfer: extract style code from a reference image.
    Input: (image, domain_idx).
    """

    def __init__(self, style_dim: int = 16, num_domains: int = 4, ch: int = 16) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(ch, ch * 2, 4, 2, 1)
        self.conv3 = nn.Conv2d(ch * 2, ch * 4, 4, 2, 1)
        feat_dim = ch * 4
        # Domain-specific linear heads
        self.domain_heads = nn.ModuleList(
            [nn.Linear(feat_dim, style_dim) for _ in range(num_domains)]
        )

    def forward(self, image: torch.Tensor, domain_idx: int) -> torch.Tensor:
        h = F.leaky_relu(self.conv1(image), 0.2)
        h = F.leaky_relu(self.conv2(h), 0.2)
        h = F.leaky_relu(self.conv3(h), 0.2)
        h = h.mean(dim=[2, 3])  # global average pool
        return self.domain_heads[domain_idx](h)


def build_stargan_v2_style_encoder() -> nn.Module:
    return StarGANv2StyleEncoder(style_dim=16, num_domains=4, ch=16)


def example_stargan_v2_style_encoder() -> tuple:
    return (torch.randn(1, 3, 32, 32), 0)


# ============================================================
# MENAGERIE_ENTRIES
# ============================================================

MENAGERIE_ENTRIES = [
    (
        "stargan_v2_generator",
        "build_stargan_v2_generator",
        "example_stargan_v2_generator",
        "2020",
        "DC",
    ),
    (
        "stargan_v2_discriminator",
        "build_stargan_v2_discriminator",
        "example_stargan_v2_discriminator",
        "2020",
        "DC",
    ),
    (
        "stargan_v2_mapping_network",
        "build_stargan_v2_mapping_network",
        "example_stargan_v2_mapping_network",
        "2020",
        "DC",
    ),
    (
        "stargan_v2_style_encoder",
        "build_stargan_v2_style_encoder",
        "example_stargan_v2_style_encoder",
        "2020",
        "DC",
    ),
]
