"""StyleGAN Inversion Encoders, Hypernetworks, and Face-Restoration GANs.

Multiple related architectures sharing the StyleGAN2 latent space as a common prior.
All use compact channel counts and small spatial sizes (64x64) for fast tracing.

--- Architectures covered ---

pSp / pixel2style2pixel (Richardson et al., CVPR 2021, arXiv:2008.00951)
  Source: https://github.com/eladrich/pixel2style2pixel
  Distinctive primitive: map2style blocks -- small conv stacks that pool a feature-map
  pyramid level down to a single (1, 512) style vector; 18 such vectors form the W+
  latent code fed level-by-level to the StyleGAN2 generator.

ReStyle (Alaluf et al., ICCV 2021, arXiv:2104.02763)
  Source: https://github.com/yuval-alaluf/restyle-encoder
  Iterative residual inversion: at each step, encoder receives [input | current_recon]
  (6-channel) and predicts a delta latent; deltas accumulate over N steps.

GFPGAN v1 (Wang et al., CVPR 2021, arXiv:2101.04061)
  Source: https://github.com/TencentARC/GFPGAN
  Distinctive primitive: Channel-Split Spatial Feature Transform (CS-SFT) -- each
  decoder layer splits the feature map channel-wise into two halves; one half is
  modulated by affine (scale/shift) parameters predicted from the StyleGAN2 prior
  latent; the other half passes through unchanged.

GPEN (Yang et al., ICCV 2021, arXiv:2105.06070)
  Source: https://github.com/yangxy/GPEN
  Blind face restoration with GAN prior: U-Net encoder maps degraded face to
  noise/style codes; StyleGAN2-style generator decoder with per-layer style injection.

HyperInverter (Dinh et al., CVPR 2022, arXiv:2112.00719)
  Source: https://github.com/VinAIResearch/HyperInverter
  Hypernetwork predicts weight residuals (delta_w) for specific conv layers of a
  frozen StyleGAN2 generator from a ResNet-encoded image embedding.

HyperStyle (Alaluf et al., CVPR 2022, arXiv:2111.15666)
  Source: https://github.com/yuval-alaluf/hyperstyle
  Hypernetwork architecture that, given an image, predicts per-layer convolution
  weight offsets for a frozen StyleGAN2 generator; offsets are applied via a
  RefineBlock on each synthesis layer.

In-Domain GAN Inversion / IDInvert (Zhu et al., ECCV 2020, arXiv:2004.00049)
  Source: https://github.com/genforce/idinvert
  Domain-guided encoder: ResNet-like backbone maps an image to a W latent code
  (not W+); the latent lies "in-domain" so the generator can faithfully reconstruct it.

pix2pix3D (Wang et al., CVPR 2023, arXiv:2302.08509) -- conditional triplane generator
  Source: https://github.com/dunbar12138/pix2pix3D
  Label/segmentation-conditioned triplane synthesis: a 2-D semantic label map is
  processed by a UNet-like feature extractor; its features condition (via AdaIN) a
  triplane generator head that outputs a (C, H, W) triplane feature volume fed to an
  NeRF-style renderer.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared primitives
# ---------------------------------------------------------------------------


class ConvBnRelu(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(self.bn(self.conv(x)), 0.2)


class ResBlock(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(ch)
        self.bn2 = nn.BatchNorm2d(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn1(self.c1(x)))
        h = self.bn2(self.c2(h))
        return F.relu(h + x)


# ---------------------------------------------------------------------------
# map2style block (pSp key primitive)
# Small conv stack that takes an HxW feature map and outputs a (1, style_dim) vector.
# ---------------------------------------------------------------------------


class Map2Style(nn.Module):
    """Pool a spatial feature map to a single style vector (1, style_dim).

    In pSp each FPN level has one Map2Style block.  A progressive sequence of
    stride-2 convolutions collapses spatial dimensions; a final 1x1 produces
    the style vector.
    """

    def __init__(self, in_ch: int, style_dim: int = 32) -> None:
        super().__init__()
        mid = max(in_ch, style_dim)
        self.convs = nn.Sequential(
            nn.Conv2d(in_ch, mid, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(mid, mid, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(mid, style_dim),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.convs(feat)  # (B, style_dim)


# ---------------------------------------------------------------------------
# Tiny StyleGAN2-style synthesis block (shared by pSp decoder, GPEN, GFPGAN)
# ---------------------------------------------------------------------------


class StyleBlock(nn.Module):
    """Single StyleGAN2 synthesis layer: style modulation -> conv -> noise -> act."""

    def __init__(self, in_ch: int, out_ch: int, style_dim: int) -> None:
        super().__init__()
        self.style_affine = nn.Linear(style_dim, in_ch)
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.noise_strength = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(out_ch))
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # style modulation: scale each channel
        s = self.style_affine(w).unsqueeze(-1).unsqueeze(-1)  # (B, in_ch, 1, 1)
        h = x * (s + 1.0)
        h = self.conv(h)
        # inject noise
        noise = torch.randn(h.shape[0], 1, h.shape[2], h.shape[3], device=h.device)
        h = h + self.noise_strength * noise + self.bias.view(1, -1, 1, 1)
        return self.act(h)


# ===========================================================================
# 1. pSp — pixel2style2pixel
# ===========================================================================


class PspEncoder(nn.Module):
    """Feature-pyramid (FPN) encoder with per-level Map2Style blocks.

    Produces num_styles style vectors stacked as (B, num_styles, style_dim).
    """

    def __init__(
        self, in_ch: int = 3, base_ch: int = 16, style_dim: int = 32, num_styles: int = 4
    ) -> None:
        super().__init__()
        # Backbone: 3 strided stages
        self.stage1 = nn.Sequential(
            ConvBnRelu(in_ch, base_ch, k=7, s=2, p=3),
            ResBlock(base_ch),
        )  # /2
        self.stage2 = nn.Sequential(
            ConvBnRelu(base_ch, base_ch * 2, s=2),
            ResBlock(base_ch * 2),
        )  # /4
        self.stage3 = nn.Sequential(
            ConvBnRelu(base_ch * 2, base_ch * 4, s=2),
            ResBlock(base_ch * 4),
        )  # /8
        # FPN lateral connections
        self.lat1 = nn.Conv2d(base_ch, base_ch * 4, 1)
        self.lat2 = nn.Conv2d(base_ch * 2, base_ch * 4, 1)
        # Map2Style: one block per style level
        ch = base_ch * 4
        self.m2s = nn.ModuleList([Map2Style(ch, style_dim) for _ in range(num_styles)])
        self.num_styles = num_styles

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        # FPN top-down
        p3 = f3
        p2 = self.lat2(f2) + F.interpolate(p3, size=f2.shape[2:], mode="nearest")
        p1 = self.lat1(f1) + F.interpolate(p2, size=f1.shape[2:], mode="nearest")
        feats = [p3, p3, p2, p1]  # repeat coarse for more style levels
        feats = feats[: self.num_styles]
        styles = torch.stack([self.m2s[i](feats[i]) for i in range(self.num_styles)], dim=1)
        return styles  # (B, num_styles, style_dim)


class PspStyleDecoder(nn.Module):
    """Minimal StyleGAN2 synthesis decoder consuming a W+ stack."""

    def __init__(self, style_dim: int = 32, base_ch: int = 16, num_styles: int = 4) -> None:
        super().__init__()
        # Learned constant input at 4x4
        self.const = nn.Parameter(torch.randn(1, base_ch * 4, 4, 4))
        ch = base_ch * 4
        self.blocks = nn.ModuleList()
        self.ups = []
        for i in range(num_styles):
            self.blocks.append(StyleBlock(ch, ch, style_dim))
        self.to_rgb = nn.Conv2d(ch, 3, 1)

    def forward(self, styles: torch.Tensor) -> torch.Tensor:
        # styles: (B, num_styles, style_dim)
        B = styles.shape[0]
        h = self.const.expand(B, -1, -1, -1)
        for i, blk in enumerate(self.blocks):
            if i > 0:
                h = F.interpolate(h, scale_factor=2.0, mode="nearest")
            h = blk(h, styles[:, i, :])
        return torch.tanh(self.to_rgb(h))


class PspWrapper(nn.Module):
    """Full pSp pipeline: encode -> W+ styles -> decode to image."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = PspEncoder(in_ch=3, base_ch=16, style_dim=32, num_styles=4)
        self.decoder = PspStyleDecoder(style_dim=32, base_ch=16, num_styles=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        styles = self.encoder(x)  # (B, 4, 32)
        return self.decoder(styles)  # (B, 3, H', W')


def build_psp_encoder() -> nn.Module:
    return PspWrapper().eval()


def example_input_psp() -> torch.Tensor:
    return torch.randn(1, 3, 64, 64)


# ---------------------------------------------------------------------------
# GradualStyleEncoder alias (same arch -- pSp's encoder component standalone)
# ---------------------------------------------------------------------------


class GradualStyleEncoder(nn.Module):
    """pSp's GradualStyleEncoder: FPN backbone + map2style blocks.

    Returns a (B, num_styles, style_dim) W+ tensor — the 18 style codes
    (here reduced to 4 for compactness).
    """

    def __init__(self) -> None:
        super().__init__()
        self.enc = PspEncoder(in_ch=3, base_ch=16, style_dim=32, num_styles=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc(x)


def build_psp_gradualstyle() -> nn.Module:
    return GradualStyleEncoder().eval()


def example_input_psp_gradualstyle() -> torch.Tensor:
    return torch.randn(1, 3, 64, 64)


# ===========================================================================
# 2. ReStyle — iterative residual encoder
# ===========================================================================


class ReStyleEncoder(nn.Module):
    """ReStyle: image + current_recon -> delta W+ latent (one refinement step).

    Input is 6-channel (original || current reconstruction) at each step.
    The delta is accumulated: w_{t+1} = w_t + delta_t.
    """

    def __init__(self, style_dim: int = 32, num_styles: int = 4) -> None:
        super().__init__()
        # Encoder takes 6-channel (img concat recon)
        self.enc = PspEncoder(in_ch=6, base_ch=16, style_dim=style_dim, num_styles=num_styles)
        self.style_dim = style_dim
        self.num_styles = num_styles

    def forward(self, x: torch.Tensor, current_latent: torch.Tensor) -> torch.Tensor:
        # x: (B,3,H,W), current_latent: (B, num_styles, style_dim)
        # generate reconstruction from current latent (simplified: just upsample latent to image)
        recon = current_latent.mean(dim=1, keepdim=True)  # (B,1,style_dim)
        recon = recon.unsqueeze(-1).expand(-1, -1, x.shape[2], x.shape[3])
        recon_img = recon[:, :3, :, :]  # take first 3 channels as dummy reconstruction
        inp = torch.cat([x, recon_img], dim=1)  # (B, 6, H, W)
        delta = self.enc(inp)  # (B, num_styles, style_dim)
        return current_latent + delta


class ReStyleIterativeWrapper(nn.Module):
    """ReStyle iterative refinement: run N residual encoder steps."""

    def __init__(self, n_steps: int = 2) -> None:
        super().__init__()
        self.encoder = ReStyleEncoder(style_dim=32, num_styles=4)
        self.decoder = PspStyleDecoder(style_dim=32, base_ch=16, num_styles=4)
        self.n_steps = n_steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        latent = torch.zeros(B, 4, 32, device=x.device)
        for _ in range(self.n_steps):
            latent = self.encoder(x, latent)
        return self.decoder(latent)


def build_restyle_encoder() -> nn.Module:
    """Single ReStyle residual step (encoder only, no iterative loop)."""

    class _SingleStep(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.enc = ReStyleEncoder(style_dim=32, num_styles=4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            B = x.shape[0]
            latent = torch.zeros(B, 4, 32, device=x.device)
            return self.enc(x, latent)

    return _SingleStep().eval()


def build_restyle_iterative() -> nn.Module:
    return ReStyleIterativeWrapper(n_steps=2).eval()


def example_input_restyle() -> torch.Tensor:
    return torch.randn(1, 3, 64, 64)


# ===========================================================================
# 3. GFPGAN v1 — CS-SFT modulated U-Net + StyleGAN prior
# ===========================================================================


class CsSftLayer(nn.Module):
    """Channel-Split Spatial Feature Transform (CS-SFT).

    Splits feature tensor channel-wise into two halves.
    The *left* half (ch//2) is modulated by (scale, shift) from the StyleGAN prior.
    The *right* half passes unchanged. Both halves are concatenated and output.
    """

    def __init__(self, ch: int, style_dim: int) -> None:
        super().__init__()
        half = ch // 2
        self.scale_pred = nn.Linear(style_dim, half)
        self.shift_pred = nn.Linear(style_dim, half)

    def forward(self, feat: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # feat: (B, ch, H, W),  w: (B, style_dim)
        ch = feat.shape[1]
        half = ch // 2
        f_mod, f_pass = feat[:, :half], feat[:, half:]
        scale = self.scale_pred(w).unsqueeze(-1).unsqueeze(-1)  # (B, half, 1, 1)
        shift = self.shift_pred(w).unsqueeze(-1).unsqueeze(-1)
        f_mod = f_mod * scale + shift
        return torch.cat([f_mod, f_pass], dim=1)


class GfpganUNet(nn.Module):
    """GFPGAN-style U-Net with CS-SFT modulation in the decoder.

    Encoder: strided conv stages.
    Prior encoder: image -> W latent (compact ResNet-like path).
    Decoder: transposed conv with CS-SFT at each scale.
    """

    def __init__(self, in_ch: int = 3, base_ch: int = 16, style_dim: int = 32) -> None:
        super().__init__()
        ch = base_ch
        # Encoder
        self.enc1 = ConvBnRelu(in_ch, ch, k=7, s=1, p=3)
        self.enc2 = ConvBnRelu(ch, ch * 2, s=2)
        self.enc3 = ConvBnRelu(ch * 2, ch * 4, s=2)
        # Bottleneck
        self.bot = ResBlock(ch * 4)
        # StyleGAN prior encoder: compress image to W latent
        self.prior_enc = nn.Sequential(
            nn.Conv2d(in_ch, ch * 2, 4, stride=4, padding=0),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch * 2, style_dim),
        )
        # Decoder with CS-SFT modulation
        self.dec3 = nn.ConvTranspose2d(ch * 4, ch * 2, 4, stride=2, padding=1)
        self.sft3 = CsSftLayer(ch * 2, style_dim)
        self.dec2 = nn.ConvTranspose2d(ch * 4, ch, 4, stride=2, padding=1)  # skip concat
        self.sft2 = CsSftLayer(ch, style_dim)
        self.dec1 = nn.Conv2d(ch * 2, ch, 3, padding=1)  # skip concat
        self.to_rgb = nn.Conv2d(ch, 3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.prior_enc(x)  # (B, style_dim)
        # encode
        e1 = self.enc1(x)  # (B, ch, H, W)
        e2 = self.enc2(e1)  # (B, 2ch, H/2, W/2)
        e3 = self.enc3(e2)  # (B, 4ch, H/4, W/4)
        b = self.bot(e3)
        # decode with CS-SFT
        d3 = F.leaky_relu(self.dec3(b), 0.2)
        d3 = self.sft3(d3, w)
        d3 = torch.cat([d3, e2], dim=1)
        d2 = F.leaky_relu(self.dec2(d3), 0.2)
        d2 = self.sft2(d2, w)
        d2 = torch.cat([d2, e1], dim=1)
        d1 = F.leaky_relu(self.dec1(d2), 0.2)
        return torch.tanh(self.to_rgb(d1))


def build_gfpgan_v1() -> nn.Module:
    return GfpganUNet(in_ch=3, base_ch=16, style_dim=32).eval()


def example_input_gfpgan() -> torch.Tensor:
    return torch.randn(1, 3, 64, 64)


# ===========================================================================
# 4. GPEN — U-Net encoder + StyleGAN decoder with style/noise injection
# ===========================================================================


class GpenEncoder(nn.Module):
    """U-Net encoder mapping degraded face -> per-level noise codes."""

    def __init__(self, in_ch: int = 3, base_ch: int = 16) -> None:
        super().__init__()
        self.d1 = ConvBnRelu(in_ch, base_ch, k=7, s=1, p=3)
        self.d2 = ConvBnRelu(base_ch, base_ch * 2, s=2)
        self.d3 = ConvBnRelu(base_ch * 2, base_ch * 4, s=2)
        # style code from bottleneck
        self.style_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_ch * 4, base_ch * 4),
        )

    def forward(self, x: torch.Tensor):
        e1 = self.d1(x)
        e2 = self.d2(e1)
        e3 = self.d3(e2)
        style = self.style_fc(e3)
        return e1, e2, e3, style


class GpenDecoder(nn.Module):
    """StyleGAN2-style generator decoder with noise/style injection + U-Net skip."""

    def __init__(self, base_ch: int = 16, style_dim: int = 64) -> None:
        super().__init__()
        ch = base_ch * 4
        self.const = nn.Parameter(torch.randn(1, ch, 4, 4))
        self.b1 = StyleBlock(ch, ch, style_dim)
        self.b2 = StyleBlock(ch, base_ch * 2, style_dim)
        self.b3 = StyleBlock(base_ch * 2, base_ch, style_dim)
        self.to_rgb = nn.Conv2d(base_ch, 3, 1)
        # lateral projections for skip connections
        self.lat3 = nn.Conv2d(base_ch * 4, ch, 1)
        self.lat2 = nn.Conv2d(base_ch * 2, base_ch * 2, 1)

    def forward(self, skips, style: torch.Tensor) -> torch.Tensor:
        e1, e2, e3 = skips
        B = style.shape[0]
        h = self.const.expand(B, -1, -1, -1)
        h = self.b1(h, style)
        h = F.interpolate(h, size=e3.shape[2:], mode="nearest")
        h = h + self.lat3(e3)
        h = self.b2(h, style)
        h = F.interpolate(h, size=e2.shape[2:], mode="nearest")
        h = h + self.lat2(e2)
        h = self.b3(h, style)
        return torch.tanh(self.to_rgb(h))


class GpenBfr(nn.Module):
    """GPEN blind face restoration: U-Net encoder -> StyleGAN decoder."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = GpenEncoder(in_ch=3, base_ch=16)
        self.decoder = GpenDecoder(base_ch=16, style_dim=64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1, e2, e3, style = self.encoder(x)
        return self.decoder((e1, e2, e3), style)


def build_gpen_bfr() -> nn.Module:
    return GpenBfr().eval()


def example_input_gpen() -> torch.Tensor:
    return torch.randn(1, 3, 64, 64)


# ===========================================================================
# 5. HyperInverter — hypernetwork predicting weight residuals for StyleGAN
# ===========================================================================


class HyperInverterEncoder(nn.Module):
    """ResNet-like backbone mapping image to a global embedding."""

    def __init__(self, in_ch: int = 3, embed_dim: int = 64) -> None:
        super().__init__()
        self.stem = ConvBnRelu(in_ch, 16, k=7, s=2, p=3)
        self.rb1 = ResBlock(16)
        self.down1 = ConvBnRelu(16, 32, s=2)
        self.rb2 = ResBlock(32)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.rb1(self.stem(x))
        h = self.rb2(self.down1(h))
        h = self.pool(h).flatten(1)
        return self.fc(h)  # (B, embed_dim)


class WeightResidualPredictor(nn.Module):
    """Hypernetwork head: embedding -> weight residual tensor for one conv layer.

    Predicts a delta_W of shape (out_ch, in_ch, k, k) for a target conv.
    """

    def __init__(self, embed_dim: int, out_ch: int, in_ch: int, k: int = 3) -> None:
        super().__init__()
        target_size = out_ch * in_ch * k * k
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, target_size),
        )
        self.out_ch = out_ch
        self.in_ch = in_ch
        self.k = k

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        # emb: (B, embed_dim)
        delta = self.fc(emb)  # (B, out_ch*in_ch*k*k)
        return delta.view(-1, self.out_ch, self.in_ch, self.k, self.k)  # (B, out_ch, in_ch, k, k)


class FrozenStyleGanLayer(nn.Module):
    """Single conv layer of a (frozen) StyleGAN2 generator, with weight-residual support."""

    def __init__(self, in_ch: int = 16, out_ch: int = 16) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_ch, in_ch, 3, 3) * 0.02, requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(out_ch))

    def forward(self, x: torch.Tensor, delta_w=None) -> torch.Tensor:
        w = self.weight
        if delta_w is not None:
            # delta_w: (B, out_ch, in_ch, 3, 3) -- apply per-sample (loop over batch)
            # For tractable tracing: use batch mean delta
            w = w + delta_w.mean(dim=0)
        return F.leaky_relu(F.conv2d(x, w, self.bias, padding=1), 0.2)


class HyperInverterModel(nn.Module):
    """HyperInverter: image -> embedding -> weight residuals -> modulated generator."""

    def __init__(self) -> None:
        super().__init__()
        self.enc = HyperInverterEncoder(in_ch=3, embed_dim=64)
        # predict weight residuals for 2 generator layers
        self.hyp1 = WeightResidualPredictor(64, out_ch=16, in_ch=16, k=3)
        self.hyp2 = WeightResidualPredictor(64, out_ch=16, in_ch=16, k=3)
        # frozen generator layers
        self.gen1 = FrozenStyleGanLayer(16, 16)
        self.gen2 = FrozenStyleGanLayer(16, 16)
        self.const = nn.Parameter(torch.randn(1, 16, 8, 8))
        self.to_rgb = nn.Conv2d(16, 3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.enc(x)  # (B, 64)
        dw1 = self.hyp1(emb)  # (B, 16, 16, 3, 3)
        dw2 = self.hyp2(emb)
        B = x.shape[0]
        h = self.const.expand(B, -1, -1, -1)
        h = self.gen1(h, dw1)
        h = F.interpolate(h, scale_factor=2.0, mode="nearest")
        h = self.gen2(h, dw2)
        return torch.tanh(self.to_rgb(h))


def build_hyper_inverter() -> nn.Module:
    return HyperInverterModel().eval()


def example_input_hyper_inverter() -> torch.Tensor:
    return torch.randn(1, 3, 64, 64)


# ===========================================================================
# 6. HyperStyle — hypernetwork predicting per-layer conv weight offsets
# ===========================================================================


class RefineBlock(nn.Module):
    """HyperStyle RefineBlock: given image features, predict weight offsets for one
    StyleGAN conv layer.  Offsets delta_W are added to the frozen generator weights
    before convolution.
    """

    def __init__(self, feat_ch: int, gen_in: int, gen_out: int) -> None:
        super().__init__()
        k_size = gen_in * gen_out * 3 * 3
        self.offset_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feat_ch, k_size),
        )
        self.gen_in = gen_in
        self.gen_out = gen_out

    def forward(self, feat: torch.Tensor, base_w: torch.Tensor) -> torch.Tensor:
        # feat: (B, feat_ch, H, W)  ->  delta_w: (gen_out, gen_in, 3, 3)
        delta = self.offset_head(feat)  # (B, k)
        delta = delta.mean(dim=0).view(self.gen_out, self.gen_in, 3, 3)
        return base_w + delta  # modulated weight


class HyperStyleNet(nn.Module):
    """HyperStyle: feature extractor -> N RefineBlocks -> modulated synthesis."""

    def __init__(self) -> None:
        super().__init__()
        feat_ch = 32
        # Feature extractor (shared across layers)
        self.feat_ext = nn.Sequential(
            ConvBnRelu(3, feat_ch, k=7, s=2, p=3),
            ResBlock(feat_ch),
            ConvBnRelu(feat_ch, feat_ch, s=2),
            ResBlock(feat_ch),
        )
        # Two synthesis conv layers with their RefineBlocks
        gen_ch = 16
        self.base_w1 = nn.Parameter(torch.randn(gen_ch, gen_ch, 3, 3) * 0.02)
        self.base_w2 = nn.Parameter(torch.randn(gen_ch, gen_ch, 3, 3) * 0.02)
        self.refine1 = RefineBlock(feat_ch, gen_ch, gen_ch)
        self.refine2 = RefineBlock(feat_ch, gen_ch, gen_ch)
        self.const = nn.Parameter(torch.randn(1, gen_ch, 8, 8))
        self.to_rgb = nn.Conv2d(gen_ch, 3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.feat_ext(x)
        w1 = self.refine1(feats, self.base_w1)  # modulated weight
        w2 = self.refine2(feats, self.base_w2)
        B = x.shape[0]
        h = self.const.expand(B, -1, -1, -1)
        h = F.leaky_relu(F.conv2d(h, w1, padding=1), 0.2)
        h = F.interpolate(h, scale_factor=2.0, mode="nearest")
        h = F.leaky_relu(F.conv2d(h, w2, padding=1), 0.2)
        return torch.tanh(self.to_rgb(h))


def build_hyperstyle() -> nn.Module:
    return HyperStyleNet().eval()


def example_input_hyperstyle() -> torch.Tensor:
    return torch.randn(1, 3, 64, 64)


# ===========================================================================
# 7. IDInvert — In-Domain GAN Inversion encoder (image -> W latent)
# ===========================================================================


class IdInvertEncoder(nn.Module):
    """ResNet-style domain-guided encoder mapping image to a W latent vector.

    Unlike W+, a single W vector is shared across all generator layers,
    ensuring the inversion stays "in-domain."
    """

    def __init__(self, in_ch: int = 3, latent_dim: int = 64) -> None:
        super().__init__()
        self.stem = ConvBnRelu(in_ch, 16, k=7, s=2, p=3)
        self.layer1 = nn.Sequential(ResBlock(16), ConvBnRelu(16, 32, s=2))
        self.layer2 = nn.Sequential(ResBlock(32), ConvBnRelu(32, 64, s=2))
        self.layer3 = nn.Sequential(ResBlock(64), ConvBnRelu(64, 64, s=2))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.pool(h).flatten(1)
        return self.fc(h)  # (B, latent_dim) -- the W vector


class IdInvertWrapper(nn.Module):
    """IDInvert: encode to W, then decode through tiny StyleGAN generator."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = IdInvertEncoder(in_ch=3, latent_dim=32)
        self.decoder = PspStyleDecoder(style_dim=32, base_ch=16, num_styles=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.encoder(x)  # (B, 32) -- single W vector
        # broadcast to W+: same vector for each style level
        w_plus = w.unsqueeze(1).expand(-1, 4, -1)
        return self.decoder(w_plus)


def build_idinvert_encoder() -> nn.Module:
    return IdInvertWrapper().eval()


def example_input_idinvert() -> torch.Tensor:
    return torch.randn(1, 3, 64, 64)


# ===========================================================================
# 8. pix2pix3D — conditional triplane generator
# ===========================================================================


class SegCondExtractor(nn.Module):
    """UNet-lite feature extractor for label/segmentation map conditioning."""

    def __init__(self, in_ch: int = 4, base_ch: int = 16) -> None:
        super().__init__()
        self.d1 = ConvBnRelu(in_ch, base_ch, k=7, s=1, p=3)
        self.d2 = ConvBnRelu(base_ch, base_ch * 2, s=2)
        self.d3 = ConvBnRelu(base_ch * 2, base_ch * 4, s=2)
        self.u2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, stride=2, padding=1)
        self.u1 = nn.ConvTranspose2d(base_ch * 4, base_ch, 4, stride=2, padding=1)

    def forward(self, seg: torch.Tensor) -> tuple:
        e1 = self.d1(seg)
        e2 = self.d2(e1)
        e3 = self.d3(e2)
        d2 = F.leaky_relu(self.u2(e3), 0.2)
        d2 = torch.cat([d2, e2], dim=1)
        d1 = F.leaky_relu(self.u1(d2), 0.2)
        return d1, e3  # (coarse_feat, bottleneck_cond)


class AdaIN(nn.Module):
    """Adaptive Instance Normalization: normalize x, then shift/scale with style."""

    def __init__(self, ch: int, style_dim: int) -> None:
        super().__init__()
        self.norm = nn.InstanceNorm2d(ch, affine=False)
        self.fc = nn.Linear(style_dim, ch * 2)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        gamma_beta = self.fc(style)  # (B, 2*ch)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return gamma * self.norm(x) + beta


class TriplaneHead(nn.Module):
    """Generate triplane features (XY, XZ, YZ planes) conditioned on style.

    Each plane is an HxW spatial feature map.  The three planes are stacked
    along the channel dimension to form the triplane volume.
    """

    def __init__(self, style_dim: int, plane_ch: int = 16, spatial: int = 8) -> None:
        super().__init__()
        # Shared backbone, then 3 separate heads per plane
        self.backbone = nn.Sequential(
            ConvBnRelu(style_dim, plane_ch, k=3, s=1, p=1),
            ResBlock(plane_ch),
        )
        self.ada = AdaIN(plane_ch, style_dim)
        self.planes = nn.ModuleList([nn.Conv2d(plane_ch, plane_ch, 3, padding=1) for _ in range(3)])
        self.spatial = spatial
        self.style_dim = style_dim

    def forward(self, style: torch.Tensor) -> torch.Tensor:
        # style: (B, style_dim) -> reshape to spatial feature map
        B = style.shape[0]
        h = style.view(B, self.style_dim, 1, 1).expand(-1, -1, self.spatial, self.spatial)
        h = self.backbone(h)
        h = self.ada(h, style)
        # produce three planes
        tri = torch.cat([F.leaky_relu(p(h), 0.2) for p in self.planes], dim=1)
        return tri  # (B, 3*plane_ch, spatial, spatial)


class Pix2Pix3DTriplaneGenerator(nn.Module):
    """pix2pix3D: segmentation label -> condition -> triplane volume.

    Label map (4-channel one-hot) is processed by SegCondExtractor;
    bottleneck features are compressed to a style vector which conditions
    a TriplaneHead via AdaIN.  The output is a triplane feature volume
    usable by an NeRF renderer (renderer not included here).
    """

    def __init__(self) -> None:
        super().__init__()
        base_ch = 16
        style_dim = 32
        self.cond_ext = SegCondExtractor(in_ch=4, base_ch=base_ch)
        # compress bottleneck cond to style
        self.cond_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_ch * 4, style_dim),
        )
        self.triplane = TriplaneHead(style_dim=style_dim, plane_ch=base_ch, spatial=8)
        # upsample triplane to output resolution
        self.out_conv = nn.Conv2d(base_ch * 3, base_ch * 3, 3, padding=1)

    def forward(self, seg: torch.Tensor) -> torch.Tensor:
        # seg: (B, 4, H, W) label map (4 semantic classes as channels)
        feat, cond = self.cond_ext(seg)
        style = self.cond_fc(cond)  # (B, 32)
        tri = self.triplane(style)  # (B, 48, 8, 8)
        return F.leaky_relu(self.out_conv(tri), 0.2)


def build_pix2pix3d() -> nn.Module:
    return Pix2Pix3DTriplaneGenerator().eval()


def example_input_pix2pix3d() -> torch.Tensor:
    # 4-channel segmentation label map (one-hot style)
    return torch.randn(1, 4, 64, 64)


# ===========================================================================
# MENAGERIE_ENTRIES
# ===========================================================================

MENAGERIE_ENTRIES = [
    (
        "pSp (pixel2style2pixel FPN encoder with map2style W+ inversion)",
        "build_psp_encoder",
        "example_input_psp",
        "2021",
        "DC",
    ),
    (
        "pSp GradualStyleEncoder (map2style FPN -> 18 W+ style codes)",
        "build_psp_gradualstyle",
        "example_input_psp_gradualstyle",
        "2021",
        "DC",
    ),
    (
        "ReStyle encoder (iterative residual W+ inversion, one refinement step)",
        "build_restyle_encoder",
        "example_input_restyle",
        "2021",
        "DC",
    ),
    (
        "ReStyle iterative inverter (N-step residual refinement loop)",
        "build_restyle_iterative",
        "example_input_restyle",
        "2021",
        "DC",
    ),
    (
        "GFPGAN v1 (CS-SFT modulated U-Net + StyleGAN2 prior for face restoration)",
        "build_gfpgan_v1",
        "example_input_gfpgan",
        "2021",
        "DC",
    ),
    (
        "GPEN BFR-512 (GAN-prior U-Net encoder + StyleGAN2 decoder for blind face restoration)",
        "build_gpen_bfr",
        "example_input_gpen",
        "2021",
        "DC",
    ),
    (
        "HyperInverter (hypernetwork predicting StyleGAN weight residuals from image embedding)",
        "build_hyper_inverter",
        "example_input_hyper_inverter",
        "2022",
        "DC",
    ),
    (
        "HyperStyle hypernetwork (per-layer conv weight offsets for StyleGAN inversion)",
        "build_hyperstyle",
        "example_input_hyperstyle",
        "2022",
        "DC",
    ),
    (
        "IDInvert encoder (in-domain ResNet encoder mapping image to single W latent)",
        "build_idinvert_encoder",
        "example_input_idinvert",
        "2020",
        "DC",
    ),
    (
        "pix2pix3D triplane conditional generator (label-conditioned AdaIN triplane synthesis)",
        "build_pix2pix3d",
        "example_input_pix2pix3d",
        "2023",
        "DC",
    ),
]
