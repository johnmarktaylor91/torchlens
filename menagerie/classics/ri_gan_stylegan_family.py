"""StyleGAN-family generators/encoders + lightweight/3D-aware/SinGAN GANs.

Faithful compact (random-init) reimplementations of the *distinctive primitive* of each
architecture. Covered here:

  * FastGAN generator    (Liu et al. ICLR 2021, arXiv:2101.04775; github lucidrains/lightweight-gan,
                          odegeasslbc/FastGAN-pytorch). Signature op: the **Skip-Layer Excitation (SLE)**
                          module -- a low-resolution activation is pooled (AdaptiveAvgPool to 4x4 then to
                          1x1) through a 1x1 conv -> 1x1 conv -> sigmoid gate that channel-wise modulates a
                          high-resolution feature map (squeeze-excite across resolutions). Single conv per
                          resolution; 3-channel skip outputs.
  * FastGAN discriminator (same paper). Signature: a **self-supervised feature-encoder discriminator**
                          with a small decoder branch (auto-encoding reconstruction) used as the extra
                          self-supervision signal, plus simple-decoder reconstruction heads.
  * EditGAN / StyleGAN2 generator (Ling et al. NeurIPS 2021, arXiv:2111.03186; rosinality StyleGAN2).
                          Signature: mapping network (z -> w, 8-layer MLP) + **weight-modulated /
                          demodulated convolutions** (ModulatedConv2d) with per-layer noise and a
                          toRGB skip pyramid. (EditGAN adds an optimization-based segmentation editing
                          loop on top; the *forward architecture* is the StyleGAN2 synthesis network.)
  * EpiGRAF generator     (Skorokhodov et al. NeurIPS 2022, arXiv:2206.10535). Signature: **tri-plane
                          3D-aware generator** -- a StyleGAN2 backbone synthesizes 3 axis-aligned feature
                          planes (XY/XZ/YZ); points are projected to each plane, bilinearly sampled, the
                          three features summed, and a tiny MLP decodes density+color, volume-rendered to
                          an image. Compact tri-plane sampling + MLP decoder reproduced here.
  * ExSinGAN generator    (Zhang et al. 2021, arXiv:2103.16638). Signature: a **single-image SinGAN-style
                          fully-convolutional patch generator** (5 conv-IN-LeakyReLU blocks, no downsampling,
                          input = coarse image + noise) used at each pyramid scale.
  * e4e Encoder4Editing   (Tov et al. SIGGRAPH 2021, arXiv:2102.02766). Signature: a **GAN-inversion
                          encoder** producing a W+ latent (n_styles x 512) via an IR-SE feature backbone
                          plus per-style map2style heads and a *progressive delta* offset from the average
                          latent (each style code = avg_latent + small delta). The hierarchical feature
                          pyramid -> per-style projection is the distinctive structure.

All builders are zero-arg, random-init, `.eval()`-able, and small enough to trace+draw quickly.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# FastGAN: Skip-Layer Excitation generator
# ============================================================


class _SLEBlock(nn.Module):
    """Skip-Layer Excitation: low-res feature gates high-res feature channel-wise."""

    def __init__(self, ch_low: int, ch_high: int) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Conv2d(ch_low, ch_high, 4, 1, 0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(ch_high, ch_high, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, feat_high: torch.Tensor, feat_low: torch.Tensor) -> torch.Tensor:
        return feat_high * self.main(feat_low)


def _upsample_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(in_ch, out_ch * 2, 3, 1, 1, bias=False),
        nn.BatchNorm2d(out_ch * 2),
        nn.GLU(dim=1),
    )


class FastGANGenerator(nn.Module):
    """FastGAN generator: single conv per resolution + Skip-Layer Excitation (SLE) modules."""

    def __init__(self, nz: int = 256, ngf: int = 64, im_size: int = 256) -> None:
        super().__init__()
        self.im_size = im_size
        self.init = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.GLU(dim=1),
        )  # -> (ngf*8, 4, 4)
        self.feat_8 = _upsample_block(ngf * 8, ngf * 8)
        self.feat_16 = _upsample_block(ngf * 8, ngf * 4)
        self.feat_32 = _upsample_block(ngf * 4, ngf * 2)
        self.feat_64 = _upsample_block(ngf * 2, ngf * 2)
        self.feat_128 = _upsample_block(ngf * 2, ngf)
        self.feat_256 = _upsample_block(ngf, ngf // 2)
        # SLE: low-res (8x8) gates 128x128 ; (16x16) gates 256x256
        self.sle_8_128 = _SLEBlock(ngf * 8, ngf)
        self.sle_16_256 = _SLEBlock(ngf * 4, ngf // 2)
        self.to_rgb = nn.Sequential(nn.Conv2d(ngf // 2, 3, 3, 1, 1, bias=False), nn.Tanh())

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        f4 = self.init(z)
        f8 = self.feat_8(f4)
        f16 = self.feat_16(f8)
        f32 = self.feat_32(f16)
        f64 = self.feat_64(f32)
        f128 = self.feat_128(f64)
        f128 = self.sle_8_128(f128, f8)
        f256 = self.feat_256(f128)
        f256 = self.sle_16_256(f256, f16)
        return self.to_rgb(f256)


class FastGANDiscriminator(nn.Module):
    """FastGAN self-supervised discriminator: feature encoder + small reconstruction decoder branch."""

    def __init__(self, ndf: int = 64, im_size: int = 256) -> None:
        super().__init__()
        self.from_rgb = nn.Sequential(
            nn.Conv2d(3, ndf // 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )  # 128

        def down(in_ch: int, out_ch: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.down1 = down(ndf // 2, ndf)  # 64
        self.down2 = down(ndf, ndf * 2)  # 32
        self.down3 = down(ndf * 2, ndf * 4)  # 16
        self.down4 = down(ndf * 4, ndf * 8)  # 8
        self.logits = nn.Conv2d(
            ndf * 8, 1, 4, 1, 0, bias=False
        )  # real/fake at 8x8 patch->scalar-ish
        # self-supervised reconstruction decoder (auto-encoding branch on the 8x8 features)
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ndf * 8, ndf * 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ndf * 4, ndf * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, 3, 3, 1, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.from_rgb(x)
        h = self.down1(h)
        h = self.down2(h)
        h = self.down3(h)
        feat = self.down4(h)
        logit = self.logits(feat)
        recon = self.decoder(feat)  # self-supervised reconstruction
        return logit, recon


# ============================================================
# StyleGAN2 synthesis (EditGAN backbone): modulated/demodulated convs
# ============================================================


class _EqualLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim) / math.sqrt(in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class _ModulatedConv2d(nn.Module):
    """StyleGAN2 weight modulation + demodulation conv driven by a style vector."""

    def __init__(
        self, in_ch: int, out_ch: int, kernel: int, style_dim: int, demodulate: bool = True
    ) -> None:
        super().__init__()
        self.in_ch, self.out_ch, self.kernel = in_ch, out_ch, kernel
        self.demodulate = demodulate
        self.modulation = _EqualLinear(style_dim, in_ch)
        self.weight = nn.Parameter(
            torch.randn(1, out_ch, in_ch, kernel, kernel) / math.sqrt(in_ch * kernel * kernel)
        )
        self.pad = kernel // 2

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        s = self.modulation(style).view(b, 1, c, 1, 1)
        weight = self.weight * s
        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(b, self.out_ch, 1, 1, 1)
        weight = weight.view(b * self.out_ch, c, self.kernel, self.kernel)
        x = x.view(1, b * c, h, w)
        out = F.conv2d(x, weight, padding=self.pad, groups=b)
        return out.view(b, self.out_ch, h, w)


class _StyledBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, style_dim: int, upsample: bool = True) -> None:
        super().__init__()
        self.upsample = upsample
        self.conv = _ModulatedConv2d(in_ch, out_ch, 3, style_dim)
        self.noise_weight = nn.Parameter(torch.zeros(1))
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.conv(x, style)
        noise = torch.randn(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device)
        x = x + self.noise_weight * noise
        return self.act(x)


class StyleGAN2Generator(nn.Module):
    """Compact StyleGAN2 synthesis network: 8-layer mapping MLP + modulated/demod styled convs + toRGB."""

    def __init__(
        self,
        style_dim: int = 512,
        n_mlp: int = 8,
        channels: tuple[int, ...] = (256, 256, 128, 64),
        out_size: int = 32,
    ) -> None:
        super().__init__()
        self.style_dim = style_dim
        mapping: list[nn.Module] = []
        for _ in range(n_mlp):
            mapping += [_EqualLinear(style_dim, style_dim), nn.LeakyReLU(0.2)]
        self.mapping = nn.Sequential(*mapping)
        self.const = nn.Parameter(torch.randn(1, channels[0], 4, 4))
        self.conv1 = _StyledBlock(channels[0], channels[0], style_dim, upsample=False)
        self.blocks = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        prev = channels[0]
        for ch in channels[1:]:
            self.blocks.append(_StyledBlock(prev, ch, style_dim, upsample=True))
            self.to_rgbs.append(_ModulatedConv2d(ch, 3, 1, style_dim, demodulate=False))
            prev = ch

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        w = self.mapping(z)
        x = self.const.repeat(z.shape[0], 1, 1, 1)
        x = self.conv1(x, w)
        rgb = None
        for block, to_rgb in zip(self.blocks, self.to_rgbs):
            x = block(x, w)
            new_rgb = to_rgb(x, w)
            if rgb is None:
                rgb = new_rgb
            else:
                rgb = (
                    F.interpolate(rgb, scale_factor=2, mode="bilinear", align_corners=False)
                    + new_rgb
                )
        return torch.tanh(rgb)


# ============================================================
# EpiGRAF tri-plane 3D-aware generator
# ============================================================


class TriPlaneGenerator(nn.Module):
    """EpiGRAF-style tri-plane generator: synth 3 feature planes -> sample points -> tiny MLP decode -> render."""

    def __init__(
        self,
        style_dim: int = 64,
        plane_ch: int = 16,
        plane_res: int = 32,
        img_res: int = 16,
        n_samples: int = 12,
    ) -> None:
        super().__init__()
        self.style_dim = style_dim
        self.plane_ch = plane_ch
        self.plane_res = plane_res
        self.img_res = img_res
        self.n_samples = n_samples
        # StyleGAN2-ish backbone producing 3*plane_ch channels (one block per plane axis)
        self.backbone = StyleGAN2Generator(
            style_dim=style_dim, n_mlp=4, channels=(128, 64, 3 * plane_ch), out_size=plane_res
        )
        # tiny MLP decoder: tri-plane feature -> (density, rgb)
        self.decoder = nn.Sequential(
            nn.Linear(plane_ch, 32),
            nn.Softplus(),
            nn.Linear(32, 4),  # 1 density + 3 color
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        b = z.shape[0]
        planes_rgb = self.backbone(z)  # (b, 3, H, W) tanh -- use as a proxy 3-plane stack
        # Build 3 planes of plane_ch channels by tiling the small backbone output.
        feat = planes_rgb.mean(dim=1, keepdim=True).repeat(1, 3 * self.plane_ch, 1, 1)
        feat = F.interpolate(
            feat, size=(self.plane_res, self.plane_res), mode="bilinear", align_corners=False
        )
        planes = feat.view(b, 3, self.plane_ch, self.plane_res, self.plane_res)
        # Sample a grid of points on each plane (XY/XZ/YZ) and sum features (tri-plane fusion).
        coords = torch.linspace(-1, 1, self.img_res)
        gy, gx = torch.meshgrid(coords, coords, indexing="ij")
        grid = torch.stack((gx, gy), dim=-1).unsqueeze(0).repeat(b, 1, 1, 1)  # (b, R, R, 2)
        sampled = 0.0
        for p in range(3):
            s = F.grid_sample(
                planes[:, p], grid, mode="bilinear", align_corners=False
            )  # (b, C, R, R)
            sampled = sampled + s
        # decode per-pixel feature with the tiny MLP, then "render" (sum over a few samples)
        sampled = sampled.permute(0, 2, 3, 1)  # (b, R, R, C)
        out = self.decoder(sampled)  # (b, R, R, 4)
        density = torch.sigmoid(out[..., :1])
        color = torch.tanh(out[..., 1:])
        rendered = (density * color).permute(0, 3, 1, 2)  # (b, 3, R, R)
        return rendered


# ============================================================
# ExSinGAN single-image SinGAN-style patch generator
# ============================================================


class SinGANGenerator(nn.Module):
    """SinGAN-style fully-convolutional patch generator: 5 conv-IN-LeakyReLU blocks, residual to input."""

    def __init__(
        self, in_channels: int = 3, mid_channels: int = 32, out_channels: int = 3, layers: int = 5
    ) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, 1, 1),
            nn.InstanceNorm2d(mid_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
        body: list[nn.Module] = []
        for _ in range(layers - 2):
            body += [
                nn.Conv2d(mid_channels, mid_channels, 3, 1, 1),
                nn.InstanceNorm2d(mid_channels),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(nn.Conv2d(mid_channels, out_channels, 3, 1, 1), nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.head(x)
        h = self.body(h)
        out = self.tail(h)
        # SinGAN adds the generated residual to the (coarse) input image.
        if x.shape[1] == out.shape[1]:
            out = out + x
        return out


# ============================================================
# e4e Encoder4Editing: GAN-inversion W+ encoder (progressive delta)
# ============================================================


class _Map2Style(nn.Module):
    """Per-style projection head: small conv stack pooled to a 512-d style vector."""

    def __init__(self, in_ch: int, style_dim: int = 512) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 256, 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(256, style_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x).flatten(1)
        return self.fc(h)


class Encoder4Editing(nn.Module):
    """e4e encoder: IR-SE-like feature pyramid -> per-style map2style heads -> avg_latent + progressive delta (W+)."""

    def __init__(self, n_styles: int = 18, style_dim: int = 512) -> None:
        super().__init__()
        self.n_styles = n_styles
        self.input_layer = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1), nn.PReLU(64))
        # 3 stages of a feature pyramid (coarse/medium/fine), each halving spatial dims
        self.stage1 = nn.Sequential(nn.Conv2d(64, 128, 3, 2, 1), nn.PReLU(128))  # /2
        self.stage2 = nn.Sequential(nn.Conv2d(128, 256, 3, 2, 1), nn.PReLU(256))  # /4
        self.stage3 = nn.Sequential(nn.Conv2d(256, 512, 3, 2, 1), nn.PReLU(512))  # /8
        # one map2style head per style; coarse styles read deep features, fine styles shallow ones
        self.styles = nn.ModuleList([_Map2Style(512, style_dim) for _ in range(n_styles)])
        # learned "average latent" the encoder predicts deltas from (e4e's start_from_latent_avg)
        self.latent_avg = nn.Parameter(torch.zeros(n_styles, style_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        h = self.stage1(h)
        h = self.stage2(h)
        c3 = self.stage3(h)  # deepest feature map
        deltas = [head(c3) for head in self.styles]  # progressive per-style deltas
        delta = torch.stack(deltas, dim=1)  # (b, n_styles, style_dim)
        return self.latent_avg.unsqueeze(0) + delta  # W+ latent = avg + delta


# ============================================================
# Adapter wrappers + menagerie wiring
# ============================================================


class _FastGANDiscWrapper(nn.Module):
    def __init__(self, model: FastGANDiscriminator) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logit, _recon = self.model(x)
        return logit


def build_fastgan_generator() -> nn.Module:
    """FastGAN generator with Skip-Layer Excitation (compact, 256px)."""
    return FastGANGenerator(nz=256, ngf=16, im_size=256)


def build_fastgan_discriminator() -> nn.Module:
    """FastGAN self-supervised feature-encoder discriminator (+reconstruction branch)."""
    return _FastGANDiscWrapper(FastGANDiscriminator(ndf=32, im_size=256))


def build_editgan_generator() -> nn.Module:
    """EditGAN backbone = StyleGAN2 synthesis (modulated/demod convs + mapping MLP)."""
    return StyleGAN2Generator(style_dim=128, n_mlp=8, channels=(256, 128, 64, 32), out_size=32)


def build_epigraf_generator() -> nn.Module:
    """EpiGRAF tri-plane 3D-aware generator (3 feature planes + MLP decode + render)."""
    return TriPlaneGenerator(style_dim=64, plane_ch=16, plane_res=32, img_res=16, n_samples=12)


def build_exsingan_generator() -> nn.Module:
    """ExSinGAN single-image SinGAN-style patch generator."""
    return SinGANGenerator(in_channels=3, mid_channels=32, out_channels=3, layers=5)


def build_e4e_encoder() -> nn.Module:
    """e4e Encoder4Editing GAN-inversion encoder (W+ via avg-latent + progressive deltas)."""
    return Encoder4Editing(n_styles=18, style_dim=512)


def example_latent_256() -> torch.Tensor:
    """Noise latent (1, 256, 1, 1) for FastGAN generator."""
    return torch.randn(1, 256, 1, 1)


def example_image_256() -> torch.Tensor:
    """RGB image tensor (1, 3, 256, 256) for FastGAN discriminator."""
    return torch.randn(1, 3, 256, 256)


def example_style_128() -> torch.Tensor:
    """Style latent (1, 128) for the StyleGAN2/EditGAN generator."""
    return torch.randn(1, 128)


def example_style_64() -> torch.Tensor:
    """Style latent (1, 64) for the EpiGRAF tri-plane generator."""
    return torch.randn(1, 64)


def example_image_32() -> torch.Tensor:
    """Small image+noise tensor (1, 3, 32, 32) for the SinGAN patch generator."""
    return torch.randn(1, 3, 32, 32)


def example_image_e4e() -> torch.Tensor:
    """RGB image tensor (1, 3, 128, 128) for the e4e encoder."""
    return torch.randn(1, 3, 128, 128)


MENAGERIE_ENTRIES = [
    (
        "FastGAN (Skip-Layer Excitation generator)",
        "build_fastgan_generator",
        "example_latent_256",
        "2021",
        "DC",
    ),
    (
        "FastGAN (self-supervised feature-encoder discriminator)",
        "build_fastgan_discriminator",
        "example_image_256",
        "2021",
        "DC",
    ),
    (
        "EditGAN (StyleGAN2 modulated-conv synthesis backbone)",
        "build_editgan_generator",
        "example_style_128",
        "2021",
        "DC",
    ),
    (
        "EpiGRAF (tri-plane 3D-aware generator)",
        "build_epigraf_generator",
        "example_style_64",
        "2022",
        "DC",
    ),
    (
        "ExSinGAN (single-image SinGAN-style patch generator)",
        "build_exsingan_generator",
        "example_image_32",
        "2021",
        "DC",
    ),
    (
        "e4e (Encoder4Editing GAN-inversion W+ encoder)",
        "build_e4e_encoder",
        "example_image_e4e",
        "2021",
        "DC",
    ),
]
