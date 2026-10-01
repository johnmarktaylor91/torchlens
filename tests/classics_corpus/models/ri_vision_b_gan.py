"""Image-translation / generative-GAN family: CycleGAN, CUT, DCGAN, DF-GAN, DiGAN, DeepFillv2.

A compact, faithful atlas pack of the distinctive generator/discriminator cores from a
cluster of well-known GANs. Each entry reproduces the *distinctive primitive* of its paper
at small random-init scale (atlas bar = architecture visual, not trained weights).

Families (paper / repo):
  * CycleGAN ResNet generator + n-layer PatchGAN discriminator
    Zhu et al., ICCV 2017, arXiv:1703.10593 -- github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    Generator: c7s1-64, down x2 (stride-2 convs), N residual blocks (reflect-pad +
    instance-norm), up x2 (transposed convs), c7s1-3 + tanh. 6 vs 9 blocks share one core.
    Discriminator: 70x70 PatchGAN -- n stride-2 conv-IN-LReLU layers -> 1-channel patch map.
  * CUT (Contrastive Unpaired Translation)
    Park et al., ECCV 2020, arXiv:2007.15651 -- github.com/taesungp/contrastive-unpaired-translation
    Same ResNet generator as CycleGAN PLUS a PatchSampleF MLP head that projects sampled
    feature patches to a unit-norm embedding (the PatchNCE primitive). The discriminator is
    the same n-layer PatchGAN.
  * DCGAN generator (PyTorch GAN Zoo flavor)
    Radford et al., ICLR 2016, arXiv:1511.06434
    All-transposed-conv generator: z -> stack of ConvTranspose2d + BN + ReLU, tanh out.
  * DF-GAN generator (NetG)
    Tao et al., CVPR 2022, arXiv:2008.05865 -- github.com/tobran/DF-GAN
    Text-to-image: fc(noise) -> reshape, then a stack of UPBlocks each containing a
    *deep-fusion* DFBlock whose conv weights are AFFINE-modulated by the sentence embedding
    (predicted gamma/beta per channel). This affine text conditioning IS the distinctive bit.
  * DiGAN generator (implicit video GAN)
    Yu et al., ICLR 2022, arXiv:2202.10571 -- github.com/sihyun-yu/digan
    INR-based video generator: a content latent + a motion latent + Fourier-feature
    (x,y,t) coordinate grid -> MLP (modulated by the latent) -> RGB frame. Generating frames
    by querying an implicit neural representation at continuous space-time coords is the
    distinctive primitive.
  * DeepFillv2 gated free-form inpainting generator
    Yu et al., ICCV 2019, arXiv:1806.03589 -- github.com/JiahuiYu/generative_inpainting
    Coarse-to-fine inpainting with GATED CONVOLUTIONS (a learned soft mask gate per pixel:
    feature * sigmoid(gate)) plus a contextual-attention branch in the refinement stage.

All cores are width-reduced and traced forward-only on small inputs.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# CycleGAN / CUT: shared ResNet generator + PatchGAN discriminator
# ============================================================


class _ResnetBlock(nn.Module):
    """Reflection-padded residual block (conv-IN-ReLU-conv-IN + skip)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3),
            nn.InstanceNorm2d(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ResnetGenerator(nn.Module):
    """CycleGAN/CUT ResNet generator: c7s1, 2x down, N res blocks, 2x up, c7s1+tanh."""

    def __init__(self, in_ch: int = 3, out_ch: int = 3, ngf: int = 32, n_blocks: int = 9) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, ngf, 7),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
        ]
        # 2 downsampling blocks
        mult = 1
        for _ in range(2):
            layers += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, 3, stride=2, padding=1),
                nn.InstanceNorm2d(ngf * mult * 2),
                nn.ReLU(inplace=True),
            ]
            mult *= 2
        # N residual blocks
        for _ in range(n_blocks):
            layers.append(_ResnetBlock(ngf * mult))
        # 2 upsampling blocks
        for _ in range(2):
            layers += [
                nn.ConvTranspose2d(
                    ngf * mult, ngf * mult // 2, 3, stride=2, padding=1, output_padding=1
                ),
                nn.InstanceNorm2d(ngf * mult // 2),
                nn.ReLU(inplace=True),
            ]
            mult //= 2
        layers += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, out_ch, 7), nn.Tanh()]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class NLayerDiscriminator(nn.Module):
    """70x70 PatchGAN: n stride-2 conv-IN-LReLU layers -> 1-channel patch map."""

    def __init__(self, in_ch: int = 3, ndf: int = 32, n_layers: int = 3) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, ndf, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        mult = 1
        for n in range(1, n_layers):
            prev = mult
            mult = min(2**n, 8)
            layers += [
                nn.Conv2d(ndf * prev, ndf * mult, 4, stride=2, padding=1),
                nn.InstanceNorm2d(ndf * mult),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        prev = mult
        mult = min(2**n_layers, 8)
        layers += [
            nn.Conv2d(ndf * prev, ndf * mult, 4, stride=1, padding=1),
            nn.InstanceNorm2d(ndf * mult),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * mult, 1, 4, stride=1, padding=1),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class PatchSampleF(nn.Module):
    """CUT PatchNCE projection head: 2-layer MLP -> L2-normalized patch embeddings.

    Reproduces the PatchSampleF primitive: feature patches (here the flattened spatial
    locations of a feature map) are projected by a shared MLP and L2-normalized to give the
    contrastive embeddings used by the PatchNCE loss.
    """

    def __init__(self, in_ch: int = 64, nc: int = 64) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_ch, nc), nn.ReLU(inplace=True), nn.Linear(nc, nc))

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        b, c, h, w = feat.shape
        flat = feat.permute(0, 2, 3, 1).reshape(b * h * w, c)
        out = self.mlp(flat)
        return F.normalize(out, dim=1)


class _CUTDiscriminator(nn.Module):
    """CUT discriminator = the same n-layer PatchGAN as CycleGAN."""

    def __init__(self) -> None:
        super().__init__()
        self.disc = NLayerDiscriminator(in_ch=3, ndf=32, n_layers=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.disc(x)


class _CUTPatchNCEWrapper(nn.Module):
    """CUT generator encoder feature -> PatchSampleF head (the PatchNCE primitive).

    Takes an image, encodes it with the first few generator layers, and projects the feature
    patches through the PatchSampleF MLP -> unit-norm embeddings. This is the contrastive
    head used in CUT (the discriminator is the standard PatchGAN above).
    """

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 32, 7),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.head = PatchSampleF(in_ch=64, nc=64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)
        return self.head(feat)


# ============================================================
# DCGAN generator
# ============================================================


class DCGANGenerator(nn.Module):
    """Classic DCGAN all-transposed-conv generator: z -> 4x ConvT-BN-ReLU -> tanh."""

    def __init__(self, nz: int = 100, ngf: int = 32, nc: int = 3) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 2:  # (B, nz) -> (B, nz, 1, 1)
            z = z[:, :, None, None]
        return self.main(z)


# ============================================================
# DF-GAN generator: deep-fusion blocks with affine text conditioning
# ============================================================


class _DFAffine(nn.Module):
    """Affine modulation: predict per-channel gamma/beta from the sentence embedding."""

    def __init__(self, ch: int, cond_dim: int) -> None:
        super().__init__()
        self.gamma = nn.Sequential(
            nn.Linear(cond_dim, ch), nn.ReLU(inplace=True), nn.Linear(ch, ch)
        )
        self.beta = nn.Sequential(nn.Linear(cond_dim, ch), nn.ReLU(inplace=True), nn.Linear(ch, ch))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        g = self.gamma(c)[:, :, None, None]
        b = self.beta(c)[:, :, None, None]
        return g * x + b


class _DFBlock(nn.Module):
    """Deep-fusion block: two (affine -> ReLU -> conv) fusions, used inside each UPBlock."""

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int) -> None:
        super().__init__()
        self.aff1 = _DFAffine(in_ch, cond_dim)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.aff2 = _DFAffine(out_ch, cond_dim)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.relu(self.aff1(x, c)))
        h = self.conv2(F.relu(self.aff2(h, c)))
        return h + self.skip(x)


class DFGANGenerator(nn.Module):
    """DF-GAN NetG: fc(noise)->reshape, then UPBlocks of affine-modulated DFBlocks -> tanh."""

    def __init__(self, ngf: int = 32, nz: int = 100, cond_dim: int = 256, ch_size: int = 3) -> None:
        super().__init__()
        self.fc = nn.Linear(nz, ngf * 8 * 4 * 4)
        self.ngf = ngf
        chs = [ngf * 8, ngf * 8, ngf * 4, ngf * 2]
        self.blocks = nn.ModuleList(
            [_DFBlock(chs[i], chs[i + 1], cond_dim) for i in range(len(chs) - 1)]
        )
        self.to_rgb = nn.Sequential(nn.Conv2d(chs[-1], ch_size, 3, padding=1), nn.Tanh())

    def forward(self, noise: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        if c is None:  # bake a default sentence embedding so build is single-tensor traceable
            c = torch.zeros(noise.shape[0], 256, device=noise.device)
        x = self.fc(noise).view(noise.shape[0], self.ngf * 8, 4, 4)
        for blk in self.blocks:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            x = blk(x, c)
        return self.to_rgb(x)


class _DFGANWrapper(nn.Module):
    """DF-GAN generator forwardable from a single noise tensor (default text embedding)."""

    def __init__(self) -> None:
        super().__init__()
        self.netg = DFGANGenerator()

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        c = torch.zeros(noise.shape[0], 256, device=noise.device)
        return self.netg(noise, c)


# ============================================================
# DiGAN generator: implicit (INR) video generator over (x,y,t) coords
# ============================================================


class _FourierFeats(nn.Module):
    """Map (x,y,t) coords -> sin/cos Fourier features (the INR positional encoding)."""

    def __init__(self, n_freq: int = 8) -> None:
        super().__init__()
        freqs = 2.0 ** torch.arange(n_freq) * math.pi
        self.register_buffer("freqs", freqs)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        # coords: (..., 3)
        proj = coords[..., None] * self.freqs  # (..., 3, n_freq)
        proj = proj.flatten(-2)  # (..., 3*n_freq)
        return torch.cat([proj.sin(), proj.cos()], dim=-1)


class DiGANGenerator(nn.Module):
    """Implicit video GAN generator: latent-modulated MLP over (x,y,t) grid -> RGB frames.

    Distinctive primitive: a content latent z is mapped to MLP modulation weights; an (x,y,t)
    coordinate grid is Fourier-encoded and decoded by the modulated MLP to produce each frame.
    Implementing video as a continuous implicit neural representation queried at space-time
    coordinates is what makes this DiGAN rather than a conv generator.
    """

    def __init__(
        self, z_dim: int = 64, hidden: int = 64, n_freq: int = 8, hw: int = 16, t: int = 4
    ) -> None:
        super().__init__()
        self.hw, self.t = hw, t
        self.ff = _FourierFeats(n_freq)
        coord_dim = 3 * n_freq * 2
        self.mod = nn.Linear(z_dim, hidden)  # latent -> per-feature modulation
        self.mlp = nn.Sequential(
            nn.Linear(coord_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )
        self.to_rgb = nn.Linear(hidden, 3)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        b = z.shape[0]
        ys, xs, ts = torch.meshgrid(
            torch.linspace(-1, 1, self.hw),
            torch.linspace(-1, 1, self.hw),
            torch.linspace(-1, 1, self.t),
            indexing="ij",
        )
        coords = torch.stack([xs, ys, ts], dim=-1).reshape(-1, 3)  # (hw*hw*t, 3)
        feats = self.ff(coords)  # (N, coord_dim)
        h = self.mlp(feats)  # (N, hidden)
        mod = self.mod(z)  # (B, hidden)
        h = h[None] * (1.0 + mod[:, None])  # latent modulates the implicit field
        rgb = torch.tanh(self.to_rgb(h))  # (B, N, 3)
        rgb = rgb.view(b, self.hw, self.hw, self.t, 3).permute(0, 4, 3, 1, 2)
        return rgb  # (B, 3, T, H, W)


# ============================================================
# DeepFillv2: gated-conv free-form inpainting generator
# ============================================================


class GatedConv2d(nn.Module):
    """Gated convolution: feature * sigmoid(gate), both from separate conv branches."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.feat = nn.Conv2d(in_ch, out_ch, k, stride, padding, dilation)
        self.gate = nn.Conv2d(in_ch, out_ch, k, stride, padding, dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.elu(self.feat(x)) * torch.sigmoid(self.gate(x))


class _ContextualAttention(nn.Module):
    """Lightweight contextual-attention stand-in (self-attention over spatial patches)."""

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.q = nn.Conv2d(ch, ch // 2, 1)
        self.k = nn.Conv2d(ch, ch // 2, 1)
        self.v = nn.Conv2d(ch, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q = self.q(x).flatten(2).permute(0, 2, 1)  # (B, HW, c/2)
        k = self.k(x).flatten(2)  # (B, c/2, HW)
        v = self.v(x).flatten(2).permute(0, 2, 1)  # (B, HW, c)
        attn = torch.softmax(q @ k / math.sqrt(c // 2), dim=-1)
        out = (attn @ v).permute(0, 2, 1).view(b, c, h, w)
        return x + out


class DeepFillv2Generator(nn.Module):
    """Coarse-to-fine gated-conv inpainting generator with a contextual-attention branch.

    Input is a 4-channel image+mask (RGB + binary mask). Coarse stage = gated-conv encoder/
    decoder; refinement stage adds a contextual-attention branch. The GATED CONVOLUTION
    (learned soft mask gating) and contextual attention are the distinctive DeepFillv2 bits.
    """

    def __init__(self, cnum: int = 24) -> None:
        super().__init__()
        # Coarse network
        self.coarse = nn.Sequential(
            GatedConv2d(4, cnum, 5, 1, 2),
            GatedConv2d(cnum, cnum * 2, 3, 2, 1),
            GatedConv2d(cnum * 2, cnum * 2, 3, 1, 1),
            GatedConv2d(cnum * 2, cnum * 4, 3, 2, 1),
            GatedConv2d(cnum * 4, cnum * 4, 3, 1, 2, dilation=2),
            GatedConv2d(cnum * 4, cnum * 4, 3, 1, 4, dilation=4),
            nn.Upsample(scale_factor=2, mode="nearest"),
            GatedConv2d(cnum * 4, cnum * 2, 3, 1, 1),
            nn.Upsample(scale_factor=2, mode="nearest"),
            GatedConv2d(cnum * 2, cnum, 3, 1, 1),
        )
        self.coarse_out = nn.Conv2d(cnum, 3, 3, padding=1)
        # Refinement: contextual-attention branch + gated convs
        self.refine_in = GatedConv2d(3, cnum, 5, 1, 2)
        self.ctx_attn = _ContextualAttention(cnum)
        self.refine = nn.Sequential(
            GatedConv2d(cnum, cnum * 2, 3, 1, 1),
            GatedConv2d(cnum * 2, cnum, 3, 1, 1),
        )
        self.refine_out = nn.Conv2d(cnum, 3, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        coarse = torch.tanh(self.coarse_out(self.coarse(x)))
        r = self.refine_in(coarse)
        r = self.ctx_attn(r)
        r = self.refine(r)
        return torch.tanh(self.refine_out(r))


# ============================================================
# Menagerie wiring
# ============================================================


def build_cyclegan_resnet_6blocks_generator() -> nn.Module:
    """CycleGAN ResNet generator with 6 residual blocks (128x128 default config)."""
    return ResnetGenerator(n_blocks=6).eval()


def build_cyclegan_resnet_9blocks_generator() -> nn.Module:
    """CycleGAN ResNet generator with 9 residual blocks (256x256 default config)."""
    return ResnetGenerator(n_blocks=9).eval()


def build_cyclegan_resnet_generator() -> nn.Module:
    """CycleGAN default ResNet generator (9 residual blocks)."""
    return ResnetGenerator(n_blocks=9).eval()


def build_cyclegan_nlayer_discriminator() -> nn.Module:
    """CycleGAN/pix2pix 70x70 PatchGAN (n-layer) discriminator."""
    return NLayerDiscriminator(n_layers=3).eval()


def build_cut_resnet_generator() -> nn.Module:
    """CUT ResNet generator (same core as CycleGAN, 9 blocks)."""
    return ResnetGenerator(n_blocks=9).eval()


def build_cut_patchnce_discriminator() -> nn.Module:
    """CUT PatchNCE projection head (PatchSampleF MLP -> unit-norm patch embeddings)."""
    return _CUTPatchNCEWrapper().eval()


def build_dcgan_generator() -> nn.Module:
    """DCGAN transposed-conv generator (PyTorch GAN Zoo flavor)."""
    return DCGANGenerator().eval()


def build_dfgan_generator() -> nn.Module:
    """DF-GAN NetG with deep-fusion affine-text-conditioned blocks."""
    return _DFGANWrapper().eval()


def build_digan_generator() -> nn.Module:
    """DiGAN implicit (INR) video generator over an (x,y,t) coordinate grid."""
    return DiGANGenerator().eval()


def build_deepfillv2_gated_generator() -> nn.Module:
    """DeepFillv2 gated-conv free-form inpainting generator (with contextual attention)."""
    return DeepFillv2Generator().eval()


def example_input_image() -> torch.Tensor:
    """RGB image (1, 3, 64, 64) for the CycleGAN/CUT generators and PatchGAN disc."""
    return torch.randn(1, 3, 64, 64)


def example_input_noise() -> torch.Tensor:
    """Noise vector (1, 100) for DCGAN / DF-GAN generators."""
    return torch.randn(1, 100)


def example_input_digan() -> torch.Tensor:
    """Content latent (1, 64) for the DiGAN implicit video generator."""
    return torch.randn(1, 64)


def example_input_inpaint() -> torch.Tensor:
    """RGB+mask 4-channel image (1, 4, 64, 64) for DeepFillv2."""
    return torch.randn(1, 4, 64, 64)


MENAGERIE_ENTRIES = [
    (
        "CycleGAN ResNet generator (6 residual blocks)",
        "build_cyclegan_resnet_6blocks_generator",
        "example_input_image",
        "2017",
        "DC",
    ),
    (
        "CycleGAN ResNet generator (9 residual blocks)",
        "build_cyclegan_resnet_9blocks_generator",
        "example_input_image",
        "2017",
        "DC",
    ),
    (
        "CycleGAN ResNet generator (default)",
        "build_cyclegan_resnet_generator",
        "example_input_image",
        "2017",
        "DC",
    ),
    (
        "CycleGAN n-layer PatchGAN discriminator",
        "build_cyclegan_nlayer_discriminator",
        "example_input_image",
        "2017",
        "DC",
    ),
    (
        "CUT ResNet generator (contrastive unpaired translation)",
        "build_cut_resnet_generator",
        "example_input_image",
        "2020",
        "DC",
    ),
    (
        "CUT PatchNCE projection head (PatchSampleF)",
        "build_cut_patchnce_discriminator",
        "example_input_image",
        "2020",
        "DC",
    ),
    (
        "DCGAN transposed-conv generator",
        "build_dcgan_generator",
        "example_input_noise",
        "2016",
        "DC",
    ),
    (
        "DF-GAN generator (deep-fusion affine text conditioning)",
        "build_dfgan_generator",
        "example_input_noise",
        "2022",
        "DC",
    ),
    (
        "DiGAN implicit video generator (INR over (x,y,t))",
        "build_digan_generator",
        "example_input_digan",
        "2022",
        "DC",
    ),
    (
        "DeepFillv2 gated-conv inpainting generator",
        "build_deepfillv2_gated_generator",
        "example_input_inpaint",
        "2019",
        "DC",
    ),
]
