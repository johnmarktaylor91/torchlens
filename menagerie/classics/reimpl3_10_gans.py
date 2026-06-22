"""Compact GAN-family classics for dependency-gated generators.

Mimicry provides reproducible DCGAN, SNGAN, SSGAN, WGAN-GP, conditional GAN,
InfoMax/InfoGAN, and SAGAN-style baselines. StyleGAN2 uses a mapping network
and modulated/demodulated convolutions; SWAGAN moves style generation into a
wavelet-frequency representation. These compact reconstructions preserve those
distinctive primitives for TorchLens graph rendering.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm

from menagerie.classics.reimpl3_10_core import SelfAttention2d


class DCGANGenerator(nn.Module):
    """DCGAN up-convolution generator."""

    def __init__(self, conditional: bool = False, spectral: bool = False) -> None:
        """Initialize generator layers.

        Parameters
        ----------
        conditional:
            Whether to concatenate class-conditioning channels.
        spectral:
            Whether to spectral-normalize convolutions for SNGAN variants.
        """

        super().__init__()
        self.conditional = conditional
        zdim = 18 if conditional else 16
        conv = spectral_norm if spectral else (lambda layer: layer)
        self.fc = nn.Linear(zdim, 32 * 4 * 4)
        self.c1 = conv(nn.Conv2d(32, 16, 3, padding=1))
        self.c2 = conv(nn.Conv2d(16, 8, 3, padding=1))
        self.out = conv(nn.Conv2d(8, 3, 3, padding=1))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Generate 32x32 RGB images.

        Parameters
        ----------
        z:
            Latent vector, optionally with two conditional channels appended.

        Returns
        -------
        torch.Tensor
            Generated image.
        """

        x = self.fc(z).view(z.shape[0], 32, 4, 4)
        x = F.interpolate(F.relu(x), scale_factor=2.0, mode="nearest")
        x = F.interpolate(F.relu(self.c1(x)), scale_factor=2.0, mode="nearest")
        x = F.interpolate(F.relu(self.c2(x)), scale_factor=2.0, mode="nearest")
        return torch.tanh(self.out(x))


class MimicryDiscriminator(nn.Module):
    """Mimicry-style convolutional discriminator with optional auxiliary heads."""

    def __init__(
        self, rotation: bool = False, infomax: bool = False, conditional: bool = False
    ) -> None:
        """Initialize discriminator.

        Parameters
        ----------
        rotation:
            Add SSGAN rotation-classification head.
        infomax:
            Add InfoGAN latent-code prediction head.
        conditional:
            Add projection-discriminator class conditioning.
        """

        super().__init__()
        self.rotation = rotation
        self.infomax = infomax
        self.conditional = conditional
        self.c1 = spectral_norm(nn.Conv2d(3, 8, 3, stride=2, padding=1))
        self.c2 = spectral_norm(nn.Conv2d(8, 16, 3, stride=2, padding=1))
        self.c3 = spectral_norm(nn.Conv2d(16, 32, 3, stride=2, padding=1))
        self.adv = spectral_norm(nn.Linear(32, 1))
        self.rot = nn.Linear(32, 4) if rotation else None
        self.code = nn.Linear(32, 4) if infomax else None
        self.embed = nn.Embedding(2, 32) if conditional else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Score images and append auxiliary predictions when configured.

        Parameters
        ----------
        x:
            RGB image tensor.  For conditional mode the first pixel encodes the
            compact class id in this dependency-free reconstruction.

        Returns
        -------
        torch.Tensor
            Adversarial and auxiliary logits.
        """

        y = F.leaky_relu(self.c1(x), 0.2)
        y = F.leaky_relu(self.c2(y), 0.2)
        feat = F.leaky_relu(self.c3(y), 0.2).mean(dim=(2, 3))
        adv = self.adv(feat)
        if self.embed is not None:
            label = x[:, 0, 0, 0].round().long().abs().clamp(0, 1)
            adv = adv + (self.embed(label) * feat).sum(dim=-1, keepdim=True)
        outs = [adv]
        if self.rot is not None:
            outs.append(self.rot(feat))
        if self.code is not None:
            outs.append(self.code(feat))
        return torch.cat(outs, dim=-1)


class SAGAN32Generator(DCGANGenerator):
    """Mimicry SAGAN generator with a non-local attention block."""

    def __init__(self) -> None:
        """Initialize spectral-normalized generator with attention."""

        super().__init__(False, True)
        self.attn = SelfAttention2d(16)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Generate image with mid-resolution self-attention.

        Parameters
        ----------
        z:
            Latent vector.

        Returns
        -------
        torch.Tensor
            Generated image.
        """

        x = self.fc(z).view(z.shape[0], 32, 4, 4)
        x = F.interpolate(F.relu(x), scale_factor=2.0, mode="nearest")
        x = F.interpolate(F.relu(self.attn(self.c1(x))), scale_factor=2.0, mode="nearest")
        x = F.interpolate(F.relu(self.c2(x)), scale_factor=2.0, mode="nearest")
        return torch.tanh(self.out(x))


class ModulatedConv2d(nn.Module):
    """StyleGAN2 modulated and demodulated convolution."""

    def __init__(self, in_channels: int, out_channels: int, style_dim: int) -> None:
        """Initialize modulated convolution parameters.

        Parameters
        ----------
        in_channels:
            Input channels.
        out_channels:
            Output channels.
        style_dim:
            Style vector dimension.
        """

        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, 3, 3) * 0.05)
        self.affine = nn.Linear(style_dim, in_channels)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """Apply per-sample modulated convolution.

        Parameters
        ----------
        x:
            Feature map.
        style:
            Style vector.

        Returns
        -------
        torch.Tensor
            Styled feature map.
        """

        batch, in_channels, height, width = x.shape
        mod = self.affine(style).view(batch, 1, in_channels, 1, 1) + 1.0
        weight = self.weight.unsqueeze(0) * mod
        demod = torch.rsqrt(weight.pow(2).sum(dim=(2, 3, 4), keepdim=True) + 1e-8)
        weight = (weight * demod).view(batch * self.weight.shape[0], in_channels, 3, 3)
        x = x.reshape(1, batch * in_channels, height, width)
        y = F.conv2d(x, weight, padding=1, groups=batch)
        return y.view(batch, self.weight.shape[0], height, width)


class StyleGAN2Tiny(nn.Module):
    """Tiny StyleGAN2 generator with mapping network and style modulation."""

    def __init__(self, wavelet: bool = False) -> None:
        """Initialize style synthesis network.

        Parameters
        ----------
        wavelet:
            Predict wavelet sub-bands as in SWAGAN.
        """

        super().__init__()
        self.wavelet = wavelet
        self.mapping = nn.Sequential(nn.Linear(16, 16), nn.LeakyReLU(0.2), nn.Linear(16, 16))
        self.const = nn.Parameter(torch.randn(1, 16, 4, 4))
        self.conv1 = ModulatedConv2d(16, 16, 16)
        self.conv2 = ModulatedConv2d(16, 8, 16)
        self.to_rgb = nn.Conv2d(8, 12 if wavelet else 3, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Generate image from latent vector.

        Parameters
        ----------
        z:
            Latent vector.

        Returns
        -------
        torch.Tensor
            RGB image.
        """

        style = self.mapping(z)
        x = self.const.expand(z.shape[0], -1, -1, -1)
        noise = torch.sin(torch.arange(16, device=z.device, dtype=z.dtype)).view(1, 1, 4, 4)
        x = F.leaky_relu(self.conv1(x, style) + noise, 0.2)
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x = F.leaky_relu(self.conv2(x, style), 0.2)
        coeff = self.to_rgb(x)
        if not self.wavelet:
            return torch.tanh(coeff)
        ll, lh, hl, hh = coeff.chunk(4, dim=1)
        top = torch.cat([ll + lh, hl + hh], dim=-1)
        bottom = torch.cat([ll - lh, hl - hh], dim=-1)
        return torch.tanh(torch.cat([top, bottom], dim=-2))


def build_mimicry_sngan_generator() -> nn.Module:
    """Build Mimicry SNGAN generator.

    Returns
    -------
    nn.Module
        Spectral-normalized DCGAN generator.
    """

    return DCGANGenerator(False, True)


def build_mimicry_ssgan_generator() -> nn.Module:
    """Build Mimicry SSGAN generator.

    Returns
    -------
    nn.Module
        SSGAN generator.
    """

    return DCGANGenerator(False, True)


def build_mimicry_cgan_pd_32() -> nn.Module:
    """Build conditional projection discriminator.

    Returns
    -------
    nn.Module
        Projection discriminator.
    """

    return MimicryDiscriminator(conditional=True)


def build_mimicry_dcgan_32() -> nn.Module:
    """Build DCGAN generator.

    Returns
    -------
    nn.Module
        DCGAN generator.
    """

    return DCGANGenerator()


def build_mimicry_infomax_gan_32() -> nn.Module:
    """Build InfoGAN/InfoMax discriminator.

    Returns
    -------
    nn.Module
        InfoGAN discriminator with code head.
    """

    return MimicryDiscriminator(infomax=True)


def build_mimicry_sagan_32() -> nn.Module:
    """Build Mimicry SAGAN generator.

    Returns
    -------
    nn.Module
        Attention generator.
    """

    return SAGAN32Generator()


def build_mimicry_sngan_32() -> nn.Module:
    """Build SNGAN discriminator.

    Returns
    -------
    nn.Module
        Spectral-normalized discriminator.
    """

    return MimicryDiscriminator()


def build_mimicry_ssgan_32() -> nn.Module:
    """Build SSGAN discriminator.

    Returns
    -------
    nn.Module
        Discriminator with rotation head.
    """

    return MimicryDiscriminator(rotation=True)


def build_mimicry_wgan_gp_32() -> nn.Module:
    """Build WGAN-GP critic.

    Returns
    -------
    nn.Module
        Critic without sigmoid output.
    """

    return MimicryDiscriminator()


def build_mimicry_wgangp_generator() -> nn.Module:
    """Build WGAN-GP generator.

    Returns
    -------
    nn.Module
        DCGAN-style generator.
    """

    return DCGANGenerator()


def build_stylegan2_lucidrains() -> nn.Module:
    """Build lucidrains-style compact StyleGAN2 generator.

    Returns
    -------
    nn.Module
        StyleGAN2 generator.
    """

    return StyleGAN2Tiny(False)


def build_swagan_generator() -> nn.Module:
    """Build SWAGAN wavelet generator.

    Returns
    -------
    nn.Module
        Wavelet StyleGAN generator.
    """

    return StyleGAN2Tiny(True)


def example_z() -> torch.Tensor:
    """Return latent vector.

    Returns
    -------
    torch.Tensor
        Latent tensor.
    """

    return torch.randn(1, 16)


def example_z_cond() -> torch.Tensor:
    """Return conditional latent vector.

    Returns
    -------
    torch.Tensor
        Latent plus one-hot condition.
    """

    return torch.randn(1, 18)


def example_img32() -> torch.Tensor:
    """Return 32x32 image.

    Returns
    -------
    torch.Tensor
        Image tensor.
    """

    return torch.randn(1, 3, 32, 32)


MENAGERIE_ENTRIES = [
    ("mimicry_sngan_generator", "build_mimicry_sngan_generator", "example_z", "2018", "gan/vision"),
    ("mimicry_ssgan_generator", "build_mimicry_ssgan_generator", "example_z", "2019", "gan/vision"),
    ("stylegan2_lucidrains", "build_stylegan2_lucidrains", "example_z", "2020", "gan/style"),
    ("swagan_generator", "build_swagan_generator", "example_z", "2021", "gan/wavelet"),
    ("mimicry_cgan_pd_32", "build_mimicry_cgan_pd_32", "example_img32", "2018", "gan/conditional"),
    ("mimicry_dcgan_32", "build_mimicry_dcgan_32", "example_z", "2015", "gan/vision"),
    (
        "mimicry_infomax_gan_32",
        "build_mimicry_infomax_gan_32",
        "example_img32",
        "2016",
        "gan/infomax",
    ),
    ("mimicry_sagan_32", "build_mimicry_sagan_32", "example_z", "2018", "gan/attention"),
    ("mimicry_sngan_32", "build_mimicry_sngan_32", "example_img32", "2018", "gan/spectral"),
    ("mimicry_ssgan_32", "build_mimicry_ssgan_32", "example_img32", "2019", "gan/self-supervised"),
    ("mimicry_wgan_gp_32", "build_mimicry_wgan_gp_32", "example_img32", "2017", "gan/wasserstein"),
    (
        "mimicry_wgangp_generator",
        "build_mimicry_wgangp_generator",
        "example_z",
        "2017",
        "gan/wasserstein",
    ),
]
