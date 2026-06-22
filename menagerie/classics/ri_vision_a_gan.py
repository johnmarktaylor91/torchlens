"""StyleGAN-family generators/encoders + AOT-GAN inpainting (ALAE, AnyCost, AOT-GAN).

Three generative-vision families. Each reproduces its DISTINCTIVE primitive at compact
random-init scale (small resolution, reduced channels) for the architecture atlas.

  * ALAE -- Adversarial Latent Autoencoder (Pidhorskyi, Adjeroh & Doretto, CVPR 2020,
    arXiv:2004.04467, github podgorskiy/ALAE).
      - decoder/Generator: a StyleGAN-style synthesis network -- a per-resolution stack of
        blocks, each modulated by a style vector via Adaptive Instance Normalization (AdaIN),
        with a learned constant input and progressive upsampling.
      - encoder: the RECIPROCAL network that maps an image back into the SAME latent W-space
        the generator's styles live in (instance-norm statistics -> style codes), so the
        autoencoder is closed in latent space (the defining ALAE idea).

  * AnyCost GAN generator (Lin et al., CVPR 2021, arXiv:2103.03243, github mit-han-lab/anycost-gan).
      - A StyleGAN2 generator (mapping MLP z->w + modulated-conv synthesis with weight
        demodulation and a toRGB skip pyramid) that supports ELASTIC channels/resolution
        (sub-networks of varying width/depth share weights). We build the full StyleGAN2
        modulated-conv generator (the anycost search operates over channel multipliers of
        this same network).

  * AOT-GAN inpainting generator (Zeng et al., TVCG 2021, arXiv:2104.01431, github
    researchmm/AOT-GAN-for-Inpainting).
      - AOT block = Aggregated Contextual Transformations: split-transform-merge where the
        SAME features pass through several PARALLEL dilated convolutions at different rates
        ([1,2,4,8]), are concatenated and fused, then gated by a learned spatial mask
        ("layer attention") residual. Stacking AOT blocks enlarges the receptive field for
        large-hole inpainting. Input is a 4-channel masked image (RGB + mask).

All builders are zero-arg and return small random-init `.eval()`-able models. The W-space
modulation (AdaIN / StyleGAN2 modulated conv) and AOT dilated split-transform-merge are the
faithful primitives; weights are random (atlas bar = structure, not trained samples).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# ALAE: StyleGAN-style generator (AdaIN) + reciprocal W-space encoder
# ===========================================================================
class _AdaIN(nn.Module):
    """Adaptive Instance Norm: style vector -> per-channel scale & bias."""

    def __init__(self, channels: int, w_dim: int) -> None:
        super().__init__()
        self.norm = nn.InstanceNorm2d(channels)
        self.style = nn.Linear(w_dim, channels * 2)

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        h = self.style(w).unsqueeze(-1).unsqueeze(-1)
        gamma, beta = h.chunk(2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class _ALAEGenBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, w_dim: int, upsample: bool) -> None:
        super().__init__()
        self.upsample = upsample
        self.conv = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.adain = _AdaIN(out_c, w_dim)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.act(self.adain(self.conv(x), w))


class _ALAEGenerator(nn.Module):
    """ALAE decoder: learned constant -> AdaIN-modulated progressive synthesis -> toRGB."""

    def __init__(self, w_dim: int = 64, base: int = 32, n_blocks: int = 4) -> None:
        super().__init__()
        self.w_dim = w_dim
        self.const = nn.Parameter(torch.randn(1, base, 4, 4))
        chans = [base] + [max(8, base // (2**i)) for i in range(1, n_blocks + 1)]
        self.blocks = nn.ModuleList()
        for i in range(n_blocks):
            up = i > 0
            self.blocks.append(_ALAEGenBlock(chans[i], chans[i + 1], w_dim, up))
        self.to_rgb = nn.Conv2d(chans[-1], 3, 1)

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        x = self.const.expand(w.shape[0], -1, -1, -1)
        for blk in self.blocks:
            x = blk(x, w)
        return torch.tanh(self.to_rgb(x))


class _ALAEGenWrapper(nn.Module):
    """Maps a latent z (B, w_dim) through a mapping MLP into W then synthesizes an image."""

    def __init__(self, w_dim: int = 64) -> None:
        super().__init__()
        self.mapping = nn.Sequential(
            nn.Linear(w_dim, w_dim), nn.LeakyReLU(0.2, True), nn.Linear(w_dim, w_dim)
        )
        self.gen = _ALAEGenerator(w_dim=w_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        w = self.mapping(z)
        return self.gen(w)


class _ALAEEncoder(nn.Module):
    """ALAE encoder: image -> instance-statistics-based style codes back into W-space."""

    def __init__(self, w_dim: int = 64, base: int = 8, n_blocks: int = 4) -> None:
        super().__init__()
        self.w_dim = w_dim
        chans = [3] + [base * (2**i) for i in range(n_blocks)]
        self.blocks = nn.ModuleList()
        self.style_proj = nn.ModuleList()
        for i in range(n_blocks):
            self.blocks.append(
                nn.Sequential(
                    nn.Conv2d(chans[i], chans[i + 1], 3, stride=2, padding=1),
                    nn.InstanceNorm2d(chans[i + 1], affine=False),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )
            # each scale contributes a style code in W (reciprocal to the generator AdaIN)
            self.style_proj.append(nn.Linear(chans[i + 1] * 2, w_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = x.new_zeros(x.shape[0], self.w_dim)
        for blk, proj in zip(self.blocks, self.style_proj):
            x = blk(x)
            mean = x.mean(dim=(2, 3))
            std = x.std(dim=(2, 3))
            w = w + proj(torch.cat([mean, std], dim=1))  # accumulate style code into W
        return w


def build_alae_decoder_generator() -> nn.Module:
    """ALAE decoder/generator: StyleGAN-style AdaIN synthesis from a latent z."""
    return _ALAEGenWrapper(w_dim=64)


def build_alae_encoder() -> nn.Module:
    """ALAE encoder: image -> W-space style code (reciprocal to the generator)."""
    return _ALAEEncoder(w_dim=64)


def example_input_alae_decoder() -> torch.Tensor:
    """Latent z tensor (1, 64) for the ALAE generator."""
    return torch.randn(1, 64)


def example_input_alae_encoder() -> torch.Tensor:
    """Image tensor (1, 3, 64, 64) for the ALAE encoder."""
    return torch.randn(1, 3, 64, 64)


# ===========================================================================
# AnyCost GAN: StyleGAN2 modulated-conv generator (elastic channels)
# ===========================================================================
class _ModulatedConv(nn.Module):
    """StyleGAN2 modulated convolution with weight demodulation."""

    def __init__(self, in_c: int, out_c: int, w_dim: int, demodulate: bool = True) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.demodulate = demodulate
        self.weight = nn.Parameter(torch.randn(1, out_c, in_c, 3, 3) * 0.1)
        self.modulation = nn.Linear(w_dim, in_c)

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        b, c, h, wd = x.shape
        style = self.modulation(w).view(b, 1, c, 1, 1)
        weight = self.weight * style  # modulate
        if self.demodulate:
            d = torch.rsqrt(weight.pow(2).sum([2, 3, 4], keepdim=True) + 1e-8)
            weight = weight * d  # demodulate
        weight = weight.view(b * self.out_c, self.in_c, 3, 3)
        x = x.view(1, b * c, h, wd)
        out = F.conv2d(x, weight, padding=1, groups=b)
        return out.view(b, self.out_c, h, wd)


class _StyleGAN2Block(nn.Module):
    def __init__(self, in_c: int, out_c: int, w_dim: int, upsample: bool) -> None:
        super().__init__()
        self.upsample = upsample
        self.conv1 = _ModulatedConv(in_c, out_c, w_dim)
        self.conv2 = _ModulatedConv(out_c, out_c, w_dim)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.to_rgb = _ModulatedConv(out_c, 3, w_dim, demodulate=False)

    def forward(self, x: torch.Tensor, w: torch.Tensor, prev_rgb):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.act(self.conv1(x, w))
        x = self.act(self.conv2(x, w))
        rgb = self.to_rgb(x, w)
        if prev_rgb is not None:
            rgb = rgb + F.interpolate(
                prev_rgb, size=rgb.shape[2:], mode="bilinear", align_corners=False
            )
        return x, rgb


class _AnyCostGenerator(nn.Module):
    """StyleGAN2 generator (mapping MLP + modulated-conv synthesis + toRGB skip pyramid)."""

    def __init__(self, w_dim: int = 64, base: int = 64, n_blocks: int = 4) -> None:
        super().__init__()
        self.w_dim = w_dim
        self.mapping = nn.Sequential(
            nn.Linear(w_dim, w_dim),
            nn.LeakyReLU(0.2, True),
            nn.Linear(w_dim, w_dim),
            nn.LeakyReLU(0.2, True),
        )
        self.const = nn.Parameter(torch.randn(1, base, 4, 4))
        chans = [base] + [max(8, base // (2**i)) for i in range(1, n_blocks + 1)]
        self.blocks = nn.ModuleList(
            [
                _StyleGAN2Block(chans[i], chans[i + 1], w_dim, upsample=(i > 0))
                for i in range(n_blocks)
            ]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        w = self.mapping(z)  # z -> w (W-space)
        x = self.const.expand(z.shape[0], -1, -1, -1)
        rgb = None
        for blk in self.blocks:
            x, rgb = blk(x, w, rgb)
        return torch.tanh(rgb)


def build_anycost_gan_generator() -> nn.Module:
    """AnyCost StyleGAN2 generator (modulated conv, elastic-channel base network)."""
    return _AnyCostGenerator(w_dim=64)


def example_input_anycost() -> torch.Tensor:
    """Latent z tensor (1, 64) for the AnyCost generator."""
    return torch.randn(1, 64)


# ===========================================================================
# AOT-GAN: Aggregated Contextual Transformations inpainting generator
# ===========================================================================
class _AOTBlock(nn.Module):
    """Aggregated Contextual Transformations block: parallel multi-rate dilated convs + gate."""

    def __init__(self, channels: int, rates=(1, 2, 4, 8)) -> None:
        super().__init__()
        self.rates = rates
        split = channels // len(rates)
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, split, 3, padding=r, dilation=r), nn.ReLU(inplace=True)
                )
                for r in rates
            ]
        )
        self.fuse = nn.Conv2d(split * len(rates), channels, 3, padding=1)
        # learned spatial gate ("layer attention" mask) for the residual
        self.gate = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = [b(x) for b in self.branches]  # split-transform at different dilation rates
        merged = self.fuse(torch.cat(outs, dim=1))  # merge
        mask = torch.sigmoid(self.gate(x))  # spatial gating
        return x * (1 - mask) + merged * mask  # gated aggregated-context residual


class _AOTGenerator(nn.Module):
    """AOT-GAN inpainting generator: encoder -> stacked AOT blocks -> decoder."""

    def __init__(self, base: int = 16, block_num: int = 4, rates=(1, 2, 4, 8)) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4, base, 7, padding=3),
            nn.ReLU(True),
            nn.Conv2d(base, base * 2, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(base * 2, base * 4, 4, stride=2, padding=1),
            nn.ReLU(True),
        )
        self.middle = nn.Sequential(*[_AOTBlock(base * 4, rates) for _ in range(block_num)])
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base * 4, base * 2, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(base, 3, 7, padding=3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.middle(x)
        return torch.tanh(self.decoder(x))


def build_aotgan_inpainting() -> nn.Module:
    """AOT-GAN inpainting generator (aggregated contextual transformations, rates=[1,2,4,8])."""
    return _AOTGenerator(base=16, block_num=4, rates=(1, 2, 4, 8))


def example_input_aotgan() -> torch.Tensor:
    """Masked image: RGB + mask = 4-channel tensor (1, 4, 128, 128)."""
    return torch.randn(1, 4, 128, 128)


MENAGERIE_ENTRIES = [
    (
        "ALAE Generator (StyleGAN-style AdaIN latent-autoencoder decoder)",
        "build_alae_decoder_generator",
        "example_input_alae_decoder",
        "2020",
        "DC",
    ),
    (
        "ALAE Encoder (reciprocal W-space autoencoder encoder)",
        "build_alae_encoder",
        "example_input_alae_encoder",
        "2020",
        "DC",
    ),
    (
        "AnyCost GAN Generator (elastic StyleGAN2 modulated-conv generator)",
        "build_anycost_gan_generator",
        "example_input_anycost",
        "2021",
        "DC",
    ),
    (
        "AOT-GAN Inpainting (aggregated contextual transformations generator)",
        "build_aotgan_inpainting",
        "example_input_aotgan",
        "2021",
        "DC",
    ),
]
