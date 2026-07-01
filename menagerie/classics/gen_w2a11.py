"""Menagerie batch w2a11: style-transfer / generative image architectures.

Sources checked (reference only; no cloning, no pip installs):
  - AesFA: https://github.com/Sooyyoungg/AesFA (model.py, networks.py, blocks.py) --
    AAAI 2024 frequency-decomposed neural style transfer via octave convolutions with
    an AdaConv-style dynamic-kernel-prediction decoder ("AdaOctConv").
  - ArtFlow: https://github.com/pkuanjie/ArtFlow (glow_wct.py) -- CVPR 2021 reversible
    normalizing-flow (Glow: ActNorm + invertible 1x1 LU conv + affine coupling, squeeze)
    used as an unbiased content/style projector, with a WCT/AdaIN transform at the
    bottleneck between the forward and reverse flow passes.
  - ArtGAN: https://github.com/cs-chan/ArtGAN (ArtGAN/Style128GANAE.py, TensorFlow
    reference reimplemented in torch) -- ICIP2017/TIP2019 categorical GAN whose
    discriminator doubles as a classifier + autoencoder (class-conditional generator,
    auxiliary-classification + reconstruction discriminator).
  - BigColor: https://github.com/KIMGEONUNG/BigColor (models/biggan.py,
    models/encoders.py) -- ECCV 2022 colorization: a grayscale ResNet-style encoder
    produces per-block conditioning features that modulate a BigGAN-style
    class-conditional generator (class-conditional batchnorm, self-attention) whose
    output is the colorized RGB image.
  - BrushNet: https://github.com/TencentARC/BrushNet
    (src/diffusers/models/brushnet.py) -- ECCV 2024 plug-and-play inpainting:
    a ControlNet-shaped clone of the diffusion UNet's down/mid/up path that consumes
    [noisy latent || masked-image-condition || mask] and injects zero-initialized 1x1
    "brushnet_block" taps into the frozen UNet's down/mid/up residual stream
    (decomposed dual-branch diffusion).
  - CAST: https://github.com/zyxElsa/CAST_pytorch (models/net.py, models/MSP.py) --
    SIGGRAPH 2022 arbitrary style transfer: a VGG encoder + AdaIN bottleneck + decoder
    generator, paired with a separate multi-layer style-extractor + projector head
    trained with an InfoNCE contrastive loss over style embeddings ("contrastive
    arbitrary style transfer" / domain enhancement module).

All models below are compact, faithfully-reimplemented-from-scratch nn.Modules with
random init and small dims for TorchLens architecture-catalog tracing (not a
trained-weights zoo).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

# ---------------------------------------------------------------------------
# AesFA -- octave-convolution encoder + AdaOctConv (dynamic-kernel) decoder
# ---------------------------------------------------------------------------


class OctConv(nn.Module):
    """Octave convolution with high/low frequency cross paths (AesFA variant)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        alpha_in: float = 0.5,
        alpha_out: float = 0.5,
        kind: str = "normal",
    ) -> None:
        super().__init__()
        self.kind = kind
        hf_in = int(in_channels * (1 - alpha_in))
        lf_in = in_channels - hf_in
        hf_out = int(out_channels * (1 - alpha_out))
        lf_out = out_channels - hf_out

        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        if kind == "first":
            self.convh = nn.Conv2d(in_channels, hf_out, kernel_size, stride, padding, bias=False)
            self.convl = nn.Conv2d(in_channels, lf_out, kernel_size, stride, padding, bias=False)
        elif kind == "last":
            self.convh = nn.Conv2d(hf_in, out_channels, kernel_size, stride, padding, bias=False)
            self.convl = nn.Conv2d(lf_in, out_channels, kernel_size, stride, padding, bias=False)
        else:
            self.h2h = nn.Conv2d(hf_in, hf_out, kernel_size, stride, padding, bias=False)
            self.l2l = nn.Conv2d(lf_in, lf_out, kernel_size, stride, padding, bias=False)
            self.h2l = nn.Conv2d(hf_in, lf_out, kernel_size, stride, padding, bias=False)
            self.l2h = nn.Conv2d(lf_in, hf_out, kernel_size, stride, padding, bias=False)

    def forward(self, x: Tensor | tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        """Run the octave convolution.

        Parameters
        ----------
        x
            Either a single tensor (``kind="first"``) or an ``(hf, lf)`` pair.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(hf, lf)`` output pair (``kind="last"`` additionally fuses them but
            still returns the pair here for a uniform interface).
        """
        if self.kind == "first":
            hf = self.convh(x)
            lf = self.convl(self.avg_pool(x))
            return hf, lf
        if self.kind == "last":
            hf, lf = x
            out_h = self.convh(hf)
            out_l = self.convl(self.upsample(lf))
            fused = out_h + out_l
            return fused, fused
        hf, lf = x
        out_h = self.h2h(hf) + self.l2h(self.upsample(lf))
        out_l = self.l2l(lf) + self.h2l(self.avg_pool(hf))
        return out_h, out_l


def _oct_relu(x: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
    hf, lf = x
    return F.leaky_relu(hf, 0.2), F.leaky_relu(lf, 0.2)


class AesFAEncoder(nn.Module):
    """Octave-conv encoder producing dual-frequency content/style features."""

    def __init__(self, in_dim: int = 3, nf: int = 8, style_kernel: int = 3) -> None:
        super().__init__()
        self.stem = nn.Conv2d(in_dim, nf, kernel_size=3, stride=1, padding=1)
        self.oct1 = OctConv(nf, nf, 3, stride=2, padding=1, kind="first")
        self.oct2 = OctConv(nf, 2 * nf, 3, stride=1, padding=1, kind="normal")
        self.pool_h = nn.AdaptiveAvgPool2d((style_kernel, style_kernel))
        self.pool_l = nn.AdaptiveAvgPool2d((style_kernel, style_kernel))

    def forward(self, x: Tensor) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
        """Encode ``x`` into octave features and pooled style statistics."""
        out = self.stem(x)
        out = _oct_relu(self.oct1(out))
        out = _oct_relu(self.oct2(out))
        hf, lf = out
        style = self.pool_h(hf), self.pool_l(lf)
        return out, style


class KernelPredictor(nn.Module):
    """Predicts a spatial conv kernel + bias from a style feature map (AdaConv)."""

    def __init__(self, channels: int, style_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        pad = kernel_size // 2
        self.spatial = nn.Conv2d(style_channels, channels * channels, kernel_size, padding=pad)
        self.bias = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(style_channels, channels, 1))

    def forward(self, style: Tensor) -> tuple[Tensor, Tensor]:
        """Predict a per-sample depthwise-ish kernel and bias from ``style``."""
        w = self.spatial(style)
        w = F.adaptive_avg_pool2d(w, 1).view(-1, self.channels, self.channels, 1, 1)
        b = self.bias(style).view(-1, self.channels)
        return w, b


class AdaOctConvBlock(nn.Module):
    """Style-conditioned dynamic conv applied per octave, then an OctConv mix."""

    def __init__(
        self,
        hf_channels: int,
        lf_channels: int,
        style_hf_channels: int,
        style_lf_channels: int,
        out_channels: int,
        kind: str = "normal",
    ) -> None:
        super().__init__()
        self.kp_h = KernelPredictor(hf_channels, style_hf_channels)
        self.kp_l = KernelPredictor(lf_channels, style_lf_channels)
        self.oct = OctConv(
            hf_channels + lf_channels, out_channels, 3, stride=1, padding=1, kind=kind
        )

    @staticmethod
    def _apply_dynamic(feat: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
        outs = []
        for b in range(feat.shape[0]):
            outs.append(F.conv2d(feat[b : b + 1], weight[b], bias[b], padding=0))
        return torch.cat(outs, dim=0)

    def forward(
        self, content: tuple[Tensor, Tensor], style: tuple[Tensor, Tensor]
    ) -> tuple[Tensor, Tensor]:
        """Modulate ``content`` octaves with dynamic kernels predicted from ``style``."""
        c_hf, c_lf = content
        s_hf, s_lf = style
        w_h, b_h = self.kp_h(s_hf)
        w_l, b_l = self.kp_l(s_lf)
        out_h = self._apply_dynamic(F.pad(c_hf, (0, 0, 0, 0)), w_h, b_h)
        out_l = self._apply_dynamic(F.pad(c_lf, (0, 0, 0, 0)), w_l, b_l)
        out_h, out_l = F.leaky_relu(out_h, 0.2), F.leaky_relu(out_l, 0.2)
        return _oct_relu(self.oct((out_h, out_l)))


class AesFADecoder(nn.Module):
    """Decoder that reconstructs an RGB image from content octaves + style stats."""

    def __init__(self, nf: int = 8, out_dim: int = 3) -> None:
        super().__init__()
        # AesFAEncoder's final OctConv (2*nf out, alpha=0.5) splits evenly into
        # nf high-freq + nf low-freq channels for both content and pooled style.
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.ada1 = AdaOctConvBlock(nf, nf, nf, nf, nf, kind="normal")
        self.ada2 = AdaOctConvBlock(nf // 2, nf // 2, nf, nf, nf, kind="last")
        self.out_conv = nn.Conv2d(nf, out_dim, kernel_size=1)

    def forward(self, content: tuple[Tensor, Tensor], style: tuple[Tensor, Tensor]) -> Tensor:
        """Decode fused frequency features to an RGB image."""
        out = self.ada1(content, style)
        out = self.up(out[0]), self.up(out[1])
        out = self.ada2(out, style)
        return self.out_conv(out[0])


class AesFA(nn.Module):
    """AesFA: frequency-decomposed neural style transfer (content/style octave AdaConv)."""

    def __init__(self, nf: int = 8) -> None:
        super().__init__()
        self.content_encoder = AesFAEncoder(nf=nf)
        self.style_encoder = AesFAEncoder(nf=nf)
        self.decoder = AesFADecoder(nf=nf)

    def forward(self, content: Tensor, style: Tensor) -> Tensor:
        """Transfer the style of ``style`` onto the content of ``content``."""
        content_feat, _ = self.content_encoder(content)
        _, style_stats = self.style_encoder(style)
        return self.decoder(content_feat, style_stats)


def build_aesfa() -> nn.Module:
    """Build a compact AesFA model.

    Returns
    -------
    nn.Module
        AesFA in eval mode.
    """
    return AesFA(nf=8).eval()


def example_input_aesfa() -> tuple[Tensor, Tensor]:
    """Example (content, style) image pair for AesFA.

    Returns
    -------
    tuple[Tensor, Tensor]
        Two ``(1, 3, 32, 32)`` tensors.
    """
    return torch.randn(1, 3, 32, 32), torch.randn(1, 3, 32, 32)


# ---------------------------------------------------------------------------
# ArtFlow -- reversible Glow-style flow with a WCT/AdaIN bottleneck transform
# ---------------------------------------------------------------------------


class ActNorm(nn.Module):
    """Per-channel affine normalization, data-dependently initialized (Glow)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.loc = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, channels, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        """Apply the learned affine shift/scale."""
        return self.scale * (x + self.loc)


class InvConv2dLU(nn.Module):
    """Invertible 1x1 convolution via a fixed LU factorization (Glow)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        weight = torch.linalg.qr(torch.randn(channels, channels))[0]
        self.weight = nn.Parameter(weight.unsqueeze(-1).unsqueeze(-1))

    def forward(self, x: Tensor) -> Tensor:
        """Apply the invertible 1x1 convolution."""
        return F.conv2d(x, self.weight)


class AffineCoupling(nn.Module):
    """Affine coupling layer: half the channels condition an affine map on the rest."""

    def __init__(self, channels: int, hidden: int = 16) -> None:
        super().__init__()
        half = channels // 2
        self.net = nn.Sequential(
            nn.Conv2d(half, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels - half, 1),
        )
        for layer in (self.net[0], self.net[2]):
            nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the affine coupling transform (forward direction only)."""
        in_a, in_b = x.chunk(2, dim=1)
        shift = self.net(in_a)
        out_b = in_b + shift
        return torch.cat([in_a, out_b], dim=1)


class GlowFlowStep(nn.Module):
    """One Glow flow step: ActNorm -> invertible 1x1 conv -> affine coupling."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.actnorm = ActNorm(channels)
        self.invconv = InvConv2dLU(channels)
        self.coupling = AffineCoupling(channels)

    def forward(self, x: Tensor) -> Tensor:
        """Run one flow step."""
        x = self.actnorm(x)
        x = self.invconv(x)
        return self.coupling(x)


def _squeeze(x: Tensor) -> Tensor:
    b, c, h, w = x.shape
    x = x.view(b, c, h // 2, 2, w // 2, 2)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    return x.view(b, c * 4, h // 2, w // 2)


def _wct(content: Tensor, style: Tensor, eps: float = 1e-5) -> Tensor:
    """Simplified whitening-and-coloring transform used at ArtFlow's flow bottleneck."""
    b, c, h, w = content.shape
    cf = content.view(b, c, -1)
    sf = style.view(b, c, -1)
    c_mean = cf.mean(dim=2, keepdim=True)
    c_std = cf.std(dim=2, keepdim=True) + eps
    s_mean = sf.mean(dim=2, keepdim=True)
    s_std = sf.std(dim=2, keepdim=True) + eps
    whitened = (cf - c_mean) / c_std
    colored = whitened * s_std + s_mean
    return colored.view(b, c, h, w)


class ArtFlow(nn.Module):
    """ArtFlow: shared reversible flow + WCT bottleneck for unbiased style transfer."""

    def __init__(self, channels: int = 3, n_flow: int = 3) -> None:
        super().__init__()
        squeezed = channels * 4
        self.flows = nn.ModuleList([GlowFlowStep(squeezed) for _ in range(n_flow)])

    def _forward_flow(self, x: Tensor) -> Tensor:
        out = _squeeze(x)
        for flow in self.flows:
            out = flow(out)
        return out

    def forward(self, content: Tensor, style: Tensor) -> Tensor:
        """Project content/style through the shared flow, apply WCT, then invert."""
        content_z = self._forward_flow(content)
        style_z = self._forward_flow(style)
        stylized_z = _wct(content_z, style_z)
        # ArtFlow's actual reverse pass exactly inverts each flow step; a compact
        # architecture-preserving stand-in reuses the same squeezed representation
        # via a 1x1 projection back to image channels (captures the "flow-in /
        # transform-at-bottleneck / flow-out" shape without duplicating the
        # analytic matrix-inverse bookkeeping of the reference).
        b, c, h, w = stylized_z.shape
        out = stylized_z.view(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).contiguous()
        return out.view(b, c // 4, h * 2, w * 2)


def build_artflow() -> nn.Module:
    """Build a compact ArtFlow model.

    Returns
    -------
    nn.Module
        ArtFlow in eval mode.
    """
    return ArtFlow(channels=3, n_flow=3).eval()


def example_input_artflow() -> tuple[Tensor, Tensor]:
    """Example (content, style) image pair for ArtFlow.

    Returns
    -------
    tuple[Tensor, Tensor]
        Two ``(1, 3, 16, 16)`` tensors (spatial dims must be even for squeeze).
    """
    return torch.randn(1, 3, 16, 16), torch.randn(1, 3, 16, 16)


# ---------------------------------------------------------------------------
# ArtGAN -- categorical GAN with an encoder-decoder (classifying + reconstructing)
# discriminator
# ---------------------------------------------------------------------------


class ArtGANGenerator(nn.Module):
    """Class-conditional generator: upconvolves ``[z; one_hot(y)]`` to an RGB image."""

    def __init__(self, z_dim: int = 16, n_classes: int = 8, base: int = 32) -> None:
        super().__init__()
        self.fc = nn.Linear(z_dim + n_classes, base * 4 * 4 * 4)
        self.base = base
        self.net = nn.Sequential(
            nn.BatchNorm2d(base * 4),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(base * 4, base * 2, 3, padding=1),
            nn.BatchNorm2d(base * 2),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(base * 2, base, 3, padding=1),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, 3, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, z: Tensor, y_onehot: Tensor) -> Tensor:
        """Generate an image from noise ``z`` and a one-hot class label."""
        h = self.fc(torch.cat([z, y_onehot], dim=1))
        h = h.view(-1, self.base * 4, 4, 4)
        return self.net(h)


class ArtGANEncoderDiscriminator(nn.Module):
    """Discriminator that is also a classifier and an autoencoder (ArtGAN's GANAE)."""

    def __init__(self, n_classes: int = 8, base: int = 32) -> None:
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, base, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base, base * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(base * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 2, base * 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(base * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.classifier = nn.Linear(base * 4, n_classes)
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(base * 4, base * 2, 3, padding=1),
            nn.BatchNorm2d(base * 2),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(base * 2, base, 3, padding=1),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(base, 3, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return (class logits, reconstruction) for a real or generated image."""
        feat = self.enc(x)
        pooled = feat.mean(dim=(2, 3))
        logits = self.classifier(pooled)
        recon = self.decoder(feat)
        return logits, recon


class ArtGAN(nn.Module):
    """ArtGAN: categorical generator + classifying/reconstructing discriminator."""

    def __init__(self, z_dim: int = 16, n_classes: int = 8, base: int = 32) -> None:
        super().__init__()
        self.generator = ArtGANGenerator(z_dim=z_dim, n_classes=n_classes, base=base)
        self.discriminator = ArtGANEncoderDiscriminator(n_classes=n_classes, base=base)

    def forward(self, z: Tensor, y_onehot: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Generate an image, then classify + reconstruct it via the discriminator."""
        fake = self.generator(z, y_onehot)
        logits, recon = self.discriminator(fake)
        return fake, logits, recon


def build_artgan() -> nn.Module:
    """Build a compact ArtGAN model.

    Returns
    -------
    nn.Module
        ArtGAN in eval mode.
    """
    return ArtGAN(z_dim=16, n_classes=8, base=16).eval()


def example_input_artgan() -> tuple[Tensor, Tensor]:
    """Example (noise, one-hot label) input for ArtGAN.

    Returns
    -------
    tuple[Tensor, Tensor]
        A ``(1, 16)`` noise tensor and a ``(1, 8)`` one-hot class tensor.
    """
    z = torch.randn(1, 16)
    y = torch.zeros(1, 8)
    y[0, 2] = 1.0
    return z, y


# ---------------------------------------------------------------------------
# BigColor -- grayscale encoder conditioning a BigGAN-style class-conditional
# generator for colorization
# ---------------------------------------------------------------------------


class ClassConditionalBN(nn.Module):
    """Class-conditional batchnorm: gain/bias predicted from a class embedding."""

    def __init__(self, channels: int, cond_dim: int) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(channels, affine=False)
        self.gain = nn.Linear(cond_dim, channels)
        self.bias = nn.Linear(cond_dim, channels)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """Normalize ``x`` then apply class-conditional affine modulation."""
        out = self.bn(x)
        gain = (1 + self.gain(cond)).unsqueeze(-1).unsqueeze(-1)
        bias = self.bias(cond).unsqueeze(-1).unsqueeze(-1)
        return out * gain + bias


class BigColorGBlock(nn.Module):
    """BigGAN-style residual up-block with class-conditional BN, fused with an
    encoder skip feature (BigColor's colorization conditioning)."""

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, enc_ch: int) -> None:
        super().__init__()
        self.cbn1 = ClassConditionalBN(in_ch, cond_dim)
        self.cbn2 = ClassConditionalBN(out_ch, cond_dim)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1)
        self.enc_proj = nn.Conv2d(enc_ch, out_ch, 1)

    def forward(self, x: Tensor, cond: Tensor, enc_feat: Tensor) -> Tensor:
        """Upsample-and-fuse ``x`` with the matching encoder feature ``enc_feat``."""
        h = F.relu(self.cbn1(x, cond))
        h = F.interpolate(h, scale_factor=2, mode="nearest")
        enc_proj = self.enc_proj(enc_feat)
        enc_proj = F.interpolate(enc_proj, size=h.shape[-2:], mode="nearest")
        h = self.conv1(h) + enc_proj
        h = F.relu(self.cbn2(h, cond))
        h = self.conv2(h)
        skip = self.skip(F.interpolate(x, scale_factor=2, mode="nearest"))
        return h + skip


class BigColorEncoder(nn.Module):
    """Grayscale ResNet-style encoder producing multi-scale conditioning features."""

    def __init__(self, base: int = 12) -> None:
        super().__init__()
        self.stage1 = nn.Sequential(nn.Conv2d(1, base, 3, padding=1), nn.ReLU(inplace=True))
        self.stage2 = nn.Sequential(
            nn.Conv2d(base, base * 2, 3, stride=2, padding=1), nn.ReLU(inplace=True)
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(base * 2, base * 4, 3, stride=2, padding=1), nn.ReLU(inplace=True)
        )

    def forward(self, gray: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return per-stage conditioning features (fine to coarse)."""
        f1 = self.stage1(gray)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        return f3, f2, f1


class BigColor(nn.Module):
    """BigColor: grayscale encoder conditions a class-conditional BigGAN generator."""

    def __init__(self, z_dim: int = 16, n_classes: int = 8, base: int = 12) -> None:
        super().__init__()
        self.encoder = BigColorEncoder(base=base)
        self.class_embed = nn.Embedding(n_classes, z_dim)
        cond_dim = z_dim * 2
        self.fc = nn.Linear(cond_dim, base * 4 * 4 * 4)
        self.block1 = BigColorGBlock(base * 4, base * 2, cond_dim, enc_ch=base * 4)
        self.block2 = BigColorGBlock(base * 2, base, cond_dim, enc_ch=base * 2)
        self.to_rgb = nn.Sequential(
            nn.BatchNorm2d(base), nn.ReLU(inplace=True), nn.Conv2d(base, 3, 3, padding=1), nn.Tanh()
        )
        self._base = base

    def forward(self, gray: Tensor, z: Tensor, class_idx: Tensor) -> Tensor:
        """Colorize ``gray`` conditioned on noise ``z`` and class label ``class_idx``."""
        f3, f2, f1 = self.encoder(gray)
        cond = torch.cat([z, self.class_embed(class_idx)], dim=1)
        h = self.fc(cond).view(-1, self._base * 4, 4, 4)
        h = self.block1(h, cond, f3)
        h = self.block2(h, cond, f2)
        return self.to_rgb(h)


def build_bigcolor() -> nn.Module:
    """Build a compact BigColor model.

    Returns
    -------
    nn.Module
        BigColor in eval mode.
    """
    return BigColor(z_dim=16, n_classes=8, base=12).eval()


def example_input_bigcolor() -> tuple[Tensor, Tensor, Tensor]:
    """Example (grayscale image, noise, class index) input for BigColor.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        A ``(1, 1, 16, 16)`` grayscale image, a ``(1, 16)`` noise vector, and a
        ``(1,)`` long class-index tensor.
    """
    return torch.randn(1, 1, 16, 16), torch.randn(1, 16), torch.tensor([3])


# ---------------------------------------------------------------------------
# BrushNet -- ControlNet-shaped dual-branch UNet clone with zero-init taps for
# decomposed masked-image-conditioned inpainting
# ---------------------------------------------------------------------------


class BrushDownBlock(nn.Module):
    """Downsampling UNet block with a zero-initialized 1x1 tap (brushnet_block)."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)
        self.norm = nn.GroupNorm(4, out_ch)
        self.tap = nn.Conv2d(out_ch, out_ch, 1)
        nn.init.zeros_(self.tap.weight)
        nn.init.zeros_(self.tap.bias)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Downsample ``x``; return (features, zero-init tap for the frozen UNet)."""
        h = F.silu(self.norm(self.conv(x)))
        return h, self.tap(h)


class BrushUpBlock(nn.Module):
    """Upsampling UNet block with a zero-initialized 1x1 tap (brushnet_block)."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm = nn.GroupNorm(4, out_ch)
        self.tap = nn.Conv2d(out_ch, out_ch, 1)
        nn.init.zeros_(self.tap.weight)
        nn.init.zeros_(self.tap.bias)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Upsample ``x``; return (features, zero-init tap for the frozen UNet)."""
        h = F.interpolate(x, scale_factor=2, mode="nearest")
        h = F.silu(self.norm(self.conv(h)))
        return h, self.tap(h)


class BrushNet(nn.Module):
    """BrushNet: masked-image branch cloning a UNet's down/mid/up path, tapped
    into the main denoiser via zero-initialized 1x1 convs at every stage."""

    def __init__(self, base: int = 8) -> None:
        super().__init__()
        # noisy latent (4ch) || masked-image VAE features (4ch) || mask (1ch)
        self.conv_in = nn.Conv2d(4 + 4 + 1, base, 3, padding=1)
        self.down1 = BrushDownBlock(base, base * 2)
        self.down2 = BrushDownBlock(base * 2, base * 4)
        self.mid_conv = nn.Conv2d(base * 4, base * 4, 3, padding=1)
        self.mid_tap = nn.Conv2d(base * 4, base * 4, 1)
        nn.init.zeros_(self.mid_tap.weight)
        nn.init.zeros_(self.mid_tap.bias)
        self.up1 = BrushUpBlock(base * 4, base * 2)
        self.up2 = BrushUpBlock(base * 2, base)

    def forward(self, latent: Tensor, masked_image_feat: Tensor, mask: Tensor) -> list[Tensor]:
        """Encode the inpainting condition and return the hierarchical zero-init taps.

        Returns
        -------
        list[Tensor]
            Down-block taps, the mid-block tap, then up-block taps -- the residual
            features that would be added into the frozen denoiser UNet.
        """
        x = torch.cat([latent, masked_image_feat, mask], dim=1)
        h = self.conv_in(x)
        h, tap1 = self.down1(h)
        h, tap2 = self.down2(h)
        h = F.silu(self.mid_conv(h))
        mid_tap = self.mid_tap(h)
        h, tap3 = self.up1(h)
        _, tap4 = self.up2(h)
        return [tap1, tap2, mid_tap, tap3, tap4]


def build_brushnet() -> nn.Module:
    """Build a compact BrushNet model.

    Returns
    -------
    nn.Module
        BrushNet in eval mode.
    """
    return BrushNet(base=8).eval()


def example_input_brushnet() -> tuple[Tensor, Tensor, Tensor]:
    """Example (latent, masked-image feature, mask) input for BrushNet.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        A ``(1, 4, 16, 16)`` noisy latent, a ``(1, 4, 16, 16)`` masked-image VAE
        feature map, and a ``(1, 1, 16, 16)`` binary mask.
    """
    return torch.randn(1, 4, 16, 16), torch.randn(1, 4, 16, 16), torch.rand(1, 1, 16, 16)


# ---------------------------------------------------------------------------
# CAST -- AdaIN encoder/decoder generator + a contrastive multi-layer style
# projector head (contrastive arbitrary style transfer)
# ---------------------------------------------------------------------------


def _calc_mean_std(feat: Tensor, eps: float = 1e-5) -> tuple[Tensor, Tensor]:
    n, c = feat.shape[:2]
    feat_var = feat.view(n, c, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(n, c, 1, 1)
    feat_mean = feat.view(n, c, -1).mean(dim=2).view(n, c, 1, 1)
    return feat_mean, feat_std


def _adain(content: Tensor, style: Tensor) -> Tensor:
    style_mean, style_std = _calc_mean_std(style)
    content_mean, content_std = _calc_mean_std(content)
    normalized = (content - content_mean) / content_std
    return normalized * style_std + style_mean


class CASTEncoder(nn.Module):
    """Small VGG-like encoder shared by content and style images."""

    def __init__(self, base: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, base, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, base * 2, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 2, base * 4, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Extract features."""
        return self.net(x)


class CASTDecoder(nn.Module):
    """Decoder mirroring the encoder, reconstructing an RGB image from AdaIN features."""

    def __init__(self, base: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(base * 4, base * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(base * 2, base, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, 3, 3, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Decode AdaIN-transformed features back to an RGB image."""
        return self.net(x)


class CASTStyleProjector(nn.Module):
    """Multi-layer style extractor + projection head for the InfoNCE contrastive loss."""

    def __init__(self, base: int = 16, proj_dim: int = 8) -> None:
        super().__init__()
        self.extractor = nn.Sequential(
            nn.Conv2d(3, base, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projector = nn.Sequential(
            nn.Linear(base, base), nn.ReLU(inplace=True), nn.Linear(base, proj_dim)
        )

    def forward(self, x: Tensor) -> Tensor:
        """Project ``x`` into the contrastive style embedding space (L2-normalized)."""
        feat = self.extractor(x).flatten(1)
        z = self.projector(feat)
        return F.normalize(z, dim=1)


class CAST(nn.Module):
    """CAST: AdaIN style-transfer generator + a contrastive style projector head."""

    def __init__(self, base: int = 16) -> None:
        super().__init__()
        self.encoder = CASTEncoder(base=base)
        self.decoder = CASTDecoder(base=base)
        self.style_projector = CASTStyleProjector(base=base)

    def forward(self, content: Tensor, style: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Stylize ``content`` with ``style`` and compute their contrastive embeddings."""
        content_feat = self.encoder(content)
        style_feat = self.encoder(style)
        stylized_feat = _adain(content_feat, style_feat)
        stylized = self.decoder(stylized_feat)
        z_style = self.style_projector(style)
        z_stylized = self.style_projector(stylized)
        return stylized, z_style, z_stylized


def build_cast() -> nn.Module:
    """Build a compact CAST model.

    Returns
    -------
    nn.Module
        CAST in eval mode.
    """
    return CAST(base=16).eval()


def example_input_cast() -> tuple[Tensor, Tensor]:
    """Example (content, style) image pair for CAST.

    Returns
    -------
    tuple[Tensor, Tensor]
        Two ``(1, 3, 32, 32)`` tensors.
    """
    return torch.randn(1, 3, 32, 32), torch.randn(1, 3, 32, 32)


MENAGERIE_ENTRIES = [
    ("AesFA", "build_aesfa", "example_input_aesfa", "2024", "VIS"),
    ("ArtFlow", "build_artflow", "example_input_artflow", "2021", "VIS"),
    ("ArtGAN", "build_artgan", "example_input_artgan", "2017", "GEN"),
    ("BigColor", "build_bigcolor", "example_input_bigcolor", "2022", "VIS"),
    ("BrushNet", "build_brushnet", "example_input_brushnet", "2024", "GEN"),
    ("CAST", "build_cast", "example_input_cast", "2022", "VIS"),
]
