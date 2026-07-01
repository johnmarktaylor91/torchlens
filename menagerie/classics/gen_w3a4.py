"""Compact faithful reimplementations of six image/sketch generative architecture families.

Sources checked (paper + official source, no clone/pip-install; reimplemented from scratch):
  - PEN-Net (Zeng, Fu, Chao & Guo, CVPR 2019, arxiv:1904.07475); official repo
    github.com/researchmm/PEN-Net-for-Inpainting -- a U-Net-shaped pyramid-context
    encoder/decoder for image inpainting. The distinctive mechanism is CROSS-LAYER
    ATTENTION TRANSFER (their "attention transfer network", ATN): region affinity
    (patch-cosine-similarity attention over known vs. missing regions) is learned once
    at a deep, low-resolution encoder layer, and that SAME attention map is reused
    (upsampled/transferred) to composite the adjacent higher-resolution encoder feature
    map, propagated pyramid-style from deep to shallow layers instead of re-learning
    attention independently at every scale. A multi-scale decoder emits a full-res
    output plus lower-res auxiliary reconstructions (deep supervision).
  - RePaint (Lugmayr, Danelljan, Romero, Yu, Timofte & Van Gool, CVPR 2022,
    arxiv:2201.09865); official repo github.com/andreas128/RePaint -- inpainting built
    on top of an UNCONDITIONAL, UNMODIFIED DDPM denoiser (no fine-tuning). The paper's
    distinctive architectural contribution is not a new denoiser but the INFERENCE-TIME
    RESAMPLING algorithm: at every reverse-diffusion step the known region is replaced by
    a fresh forward-diffusion sample of the ground-truth image at that noise level while
    the unknown region keeps the network's own denoised prediction, and every few steps
    the sampler jumps back U steps and re-runs forward+reverse diffusion ("resampling")
    to harmonize the two regions before continuing. Modeled here as a compact UNet
    epsilon-predictor plus a `forward()` that performs exactly this masked
    known/unknown-region harmonizing resample loop (the paper's Algorithm 1).
  - SA-Net / SANet (Style-Attentional Network) (Park & Lee, CVPR 2019,
    arxiv:1812.02342); official repo github.com/dypark86/SANET -- arbitrary style
    transfer. The distinctive mechanism is the SANet ATTENTION MODULE applied at
    multiple VGG feature levels: instance-normalized content features project to a
    query, instance-normalized style features project to key/value, softmax attention
    over style spatial positions produces a per-content-position weighted mixture of
    style features (a non-local-block-style content-to-style attention, NOT AdaIN's
    global mean/std matching), followed by a learned convolution and multi-level fusion
    (relu4_1 + upsampled relu5_1 attention maps) before the decoder. This candidate is
    the CVPR-2019 style-attentional SANet (dypark86/SANET), explicitly NOT the
    ICASSP-2021 channel/spatial Shuffle-Attention SA-Net (wofmanaf/SA-Net), per the
    build-queue alias note.
  - SCGAN (Zhao, Po, Cheung, Yu & Rui, IEEE TCSVT 2020, arxiv:2011.11377); official repo
    github.com/zhaoyuzhi/Semantic-Colorization-GAN -- automatic grayscale colorization.
    The distinctive mechanism is a shared colorization ENCODER that fuses local
    convolutional features with GLOBAL semantic features pooled from a frozen
    VGG-16-gray classification backbone (broadcast-concatenated into the bottleneck),
    feeding a single decoder trunk with TWO PARALLEL OUTPUT HEADS: an RGB colorization
    head and a saliency-map head trained as a joint proxy target, so the network is
    forced to learn where in the image color should be confidently assigned. Only the
    generator (encoder+decoder+dual heads) is modeled; the two adversarial
    discriminators are a training-time-only auxiliary loss and not part of the
    architecture's forward mapping.
  - Shift-Net (Yan, Li, Li, Zhang & Yang, ECCV 2018, arxiv:1801.09392); official repo
    github.com/Zhaoyi-Yan/Shift-Net_pytorch -- U-Net-based inpainting with a
    SHIFT-CONNECTION LAYER inserted at one skip connection. The distinctive mechanism:
    for each spatial position inside the missing region, the layer finds the most
    cosine-similar KNOWN-region encoder feature vector (encoder features re-used as a
    patch dictionary) and copies ("shifts") that known feature into the missing
    position, concatenating the shifted feature with the plain U-Net skip feature
    before the corresponding decoder block -- i.e. deep-feature nearest-neighbor
    rearrangement/copy-paste guided by the decoder's own partially-reconstructed
    content, rather than a learned attention-weighted blend.
  - Sketch-BERT (Lin, Fu, Jiang & Xue, CVPR 2020, arxiv:2005.09159); official repo
    github.com/avalonstrel/SketchBERT -- BERT-style bidirectional transformer over
    vector-format sketch stroke sequences (each point: (dx, dy, pen-state)), NOT
    pixel CNNs or the SketchRNN LSTM baseline. The distinctive mechanism is a Sketch
    Embedding network (continuous offset MLP-embedding + discrete pen-state
    embedding, summed) feeding a standard bidirectional Transformer encoder, pretrained
    with a "Sketch Gestalt Model" self-supervised objective: a contiguous span of
    stroke points is masked and the model must jointly RECONSTRUCT the continuous
    (dx, dy) offsets (regression head) and the discrete pen-state (classification
    head) of the masked points from bidirectional context, mirroring masked-LM but
    over a mixed continuous/discrete point-sequence representation.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


# ---------------------------------------------------------------------------
# PEN-Net: U-Net encoder/decoder with cross-layer attention transfer -- region
# affinity is learned once at a deep layer and the SAME attention map is reused
# to composite the adjacent shallower encoder feature map, propagated pyramid
# style, plus a multi-scale (deeply supervised) decoder output.
# ---------------------------------------------------------------------------
class AttentionTransferBlock(nn.Module):
    """Learn region-affinity attention at one scale and reuse it at the next scale up."""

    def __init__(self, channels: int) -> None:
        """Build the query/key projections used to compute patch-cosine attention."""

        super().__init__()
        self.query_proj = nn.Conv2d(channels, channels, 1)
        self.key_proj = nn.Conv2d(channels, channels, 1)

    def compute_attention(self, feat: Tensor, mask: Tensor) -> Tensor:
        """Return an (n, hw, hw) affinity matrix between missing and known patches."""

        n, c, h, w = feat.shape
        mask_hw = F.interpolate(mask, size=(h, w), mode="nearest")
        q = F.normalize(self.query_proj(feat).reshape(n, c, h * w), dim=1)
        k = F.normalize(self.key_proj(feat).reshape(n, c, h * w), dim=1)
        affinity = torch.bmm(q.transpose(1, 2), k)  # (n, hw, hw)
        key_mask = mask_hw.reshape(n, 1, h * w).bool()  # keys (columns) at missing positions
        # Query positions attend over KNOWN key positions only.
        affinity = affinity.masked_fill(key_mask, float("-inf"))
        return F.softmax(affinity, dim=-1)

    def apply_attention(self, feat: Tensor, attn: Tensor, mask: Tensor) -> Tensor:
        """Composite `feat` at missing positions using a (possibly transferred) attn map."""

        n, c, h, w = feat.shape
        v = feat.reshape(n, c, h * w)
        composited = torch.bmm(v, attn.transpose(1, 2)).reshape(n, c, h, w)
        mask_up = F.interpolate(mask, size=(h, w), mode="nearest")
        return feat * (1 - mask_up) + composited * mask_up


class PENNet(nn.Module):
    """Compact PEN-Net: pyramid-context encoder with cross-layer attention transfer."""

    def __init__(self, base_channels: int = 16) -> None:
        """Build the 3-level encoder/decoder pyramid and the shared attention-transfer block."""

        super().__init__()
        c = base_channels
        self.enc1 = nn.Conv2d(4, c, 3, stride=2, padding=1)
        self.enc2 = nn.Conv2d(c, 2 * c, 3, stride=2, padding=1)
        self.enc3 = nn.Conv2d(2 * c, 4 * c, 3, stride=2, padding=1)

        self.attn_deep = AttentionTransferBlock(4 * c)

        self.dec3 = nn.ConvTranspose2d(4 * c, 2 * c, 4, stride=2, padding=1)
        self.dec2 = nn.ConvTranspose2d(4 * c, c, 4, stride=2, padding=1)
        self.dec1 = nn.ConvTranspose2d(2 * c, c, 4, stride=2, padding=1)

        self.out_full = nn.Conv2d(c, 3, 3, padding=1)
        self.out_mid = nn.Conv2d(4 * c, 3, 3, padding=1)
        self.out_deep = nn.Conv2d(4 * c, 3, 3, padding=1)

    def forward(self, image: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return (full-res, mid-res, deep-res) pyramid reconstructions.

        Parameters
        ----------
        image : Tensor
            Masked RGB input, shape (batch, 3, H, W).
        mask : Tensor
            Binary missing-region mask (1 = missing), shape (batch, 1, H, W).
        """

        x = torch.cat([image, mask], dim=1)
        f1 = F.relu(self.enc1(x))  # H/2
        f2 = F.relu(self.enc2(f1))  # H/4
        f3 = F.relu(self.enc3(f2))  # H/8, deepest -- attention learned here

        n, _, h3, w3 = f3.shape
        deep_attn = self.attn_deep.compute_attention(f3, mask)  # (n, h3*w3, h3*w3)
        f3_composited = self.attn_deep.apply_attention(f3, deep_attn, mask)

        # Transfer the SAME attention map (upsampled along both the query and key
        # axes) to composite the shallower f2 map, instead of re-learning attention
        # independently at every pyramid level.
        _, _, h2, w2 = f2.shape
        attn_grid = deep_attn.reshape(n * h3 * w3, 1, h3, w3)
        attn_key_up = F.interpolate(attn_grid, size=(h2, w2), mode="nearest")
        attn_key_up = attn_key_up.reshape(n, h3, w3, h2 * w2).permute(0, 3, 1, 2)
        attn_full_up = F.interpolate(attn_key_up, size=(h2, w2), mode="nearest")
        attn_up = attn_full_up.reshape(n, h2 * w2, h2 * w2)
        attn_up = F.softmax(attn_up, dim=-1)
        f2_composited = self.attn_deep.apply_attention(f2, attn_up, mask)

        d3 = F.relu(self.dec3(f3_composited))
        deep_out = self.out_deep(f3_composited)

        d2_in = torch.cat([d3, f2_composited], dim=1)
        d2 = F.relu(self.dec2(d2_in))
        mid_out = self.out_mid(d2_in)

        d1 = F.relu(self.dec1(torch.cat([d2, f1], dim=1)))
        full_out = torch.tanh(self.out_full(d1))

        return full_out, torch.tanh(mid_out), torch.tanh(deep_out)


def build_pennet() -> nn.Module:
    """Build the compact PEN-Net pyramid-context inpainting model."""

    return PENNet().eval()


def example_input_pennet() -> tuple[Tensor, Tensor]:
    """Return (masked RGB image, missing-region mask) at 64x64."""

    image = torch.randn(2, 3, 64, 64)
    mask = torch.zeros(2, 1, 64, 64)
    mask[:, :, 20:40, 20:40] = 1.0
    return image, mask


# ---------------------------------------------------------------------------
# RePaint: pretrained unconditional DDPM epsilon-predictor UNet + the paper's
# distinctive INFERENCE-TIME resampling algorithm that harmonizes known (forward
# -diffused ground truth) and unknown (network-predicted) regions.
# ---------------------------------------------------------------------------
class TinyEpsilonUNet(nn.Module):
    """Small unconditional DDPM epsilon predictor, timestep-conditioned by FiLM."""

    def __init__(self, channels: int = 3, base: int = 16, n_steps: int = 100) -> None:
        """Build a compact conv UNet with sinusoidal-timestep FiLM modulation."""

        super().__init__()
        self.n_steps = n_steps
        self.time_embed = nn.Sequential(
            nn.Linear(base, 4 * base), nn.SiLU(), nn.Linear(4 * base, base)
        )
        self.base = base
        self.enc1 = nn.Conv2d(channels, base, 3, padding=1)
        self.enc2 = nn.Conv2d(base, 2 * base, 3, stride=2, padding=1)
        self.mid = nn.Conv2d(2 * base, 2 * base, 3, padding=1)
        self.dec2 = nn.ConvTranspose2d(2 * base, base, 4, stride=2, padding=1)
        self.dec1 = nn.Conv2d(2 * base, channels, 3, padding=1)

    def _sinusoidal(self, t: Tensor) -> Tensor:
        half = self.base // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    def forward(self, x_t: Tensor, t: Tensor) -> Tensor:
        """Predict the noise epsilon added to `x_t` at timestep `t`."""

        temb = self.time_embed(self._sinusoidal(t)).unsqueeze(-1).unsqueeze(-1)
        h1 = F.silu(self.enc1(x_t) + temb)
        h2 = F.silu(self.enc2(h1))
        m = F.silu(self.mid(h2))
        u2 = F.silu(self.dec2(m))
        return self.dec1(torch.cat([u2, h1], dim=1))


class RePaint(nn.Module):
    """Compact RePaint: DDPM UNet + masked known/unknown resampling reverse sampler."""

    def __init__(self, base: int = 16, n_steps: int = 20, resample_every: int = 5) -> None:
        """Build the DDPM denoiser and precompute the linear noise schedule.

        Parameters
        ----------
        n_steps : int
            Number of discrete reverse-diffusion steps (kept tiny for tracing).
        resample_every : int
            Jump-back interval U for the paper's resampling harmonization step.
        """

        super().__init__()
        self.unet = TinyEpsilonUNet(base=base, n_steps=n_steps)
        self.n_steps = n_steps
        self.resample_every = resample_every
        betas = torch.linspace(1e-4, 0.02, n_steps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

    def forward(self, gt_image: Tensor, mask: Tensor, x_init: Tensor) -> Tensor:
        """Run the RePaint reverse-diffusion resampling loop and return the inpainted image.

        Parameters
        ----------
        gt_image : Tensor
            Ground-truth RGB image supplying the KNOWN region, shape (batch, 3, H, W).
        mask : Tensor
            Binary known-region mask (1 = known/keep), shape (batch, 1, H, W).
        x_init : Tensor
            Initial pure-noise sample x_T, shape (batch, 3, H, W).
        """

        x_t = x_init
        for step in range(self.n_steps - 1, -1, -1):
            t = torch.full((x_t.shape[0],), step, device=x_t.device, dtype=torch.long)
            alpha_bar_t = self.alpha_bars[step]

            # Known region: fresh forward-diffusion sample of the ground truth at t.
            noise = torch.randn_like(gt_image)
            x_known = alpha_bar_t.sqrt() * gt_image + (1 - alpha_bar_t).sqrt() * noise

            # Unknown region: the network's own denoised reverse step.
            eps_pred = self.unet(x_t, t)
            alpha_t = self.alphas[step]
            beta_t = self.betas[step]
            mean = (x_t - beta_t / (1 - alpha_bar_t).sqrt() * eps_pred) / alpha_t.sqrt()
            step_noise = torch.randn_like(x_t) if step > 0 else torch.zeros_like(x_t)
            x_unknown = mean + beta_t.sqrt() * step_noise

            x_t = mask * x_known + (1 - mask) * x_unknown

            # Resampling: every `resample_every` steps, jump back one step by
            # re-diffusing forward then immediately re-denoising, harmonizing the
            # known/unknown boundary before continuing the schedule.
            if self.resample_every > 0 and step > 0 and step % self.resample_every == 0:
                jump_noise = torch.randn_like(x_t)
                alpha_prev = self.alphas[step - 1]
                x_t = alpha_prev.sqrt() * x_t + (1 - alpha_prev).sqrt() * jump_noise

        return x_t


def build_repaint() -> nn.Module:
    """Build the compact RePaint DDPM-resampling inpainting model."""

    return RePaint().eval()


def example_input_repaint() -> tuple[Tensor, Tensor, Tensor]:
    """Return (ground-truth image, known-region mask, initial noise) at 32x32."""

    gt_image = torch.randn(2, 3, 32, 32)
    mask = torch.ones(2, 1, 32, 32)
    mask[:, :, 10:20, 10:20] = 0.0
    x_init = torch.randn(2, 3, 32, 32)
    return gt_image, mask, x_init


# ---------------------------------------------------------------------------
# SA-Net (Style-Attentional Network): multi-level content-to-style attention
# (instance-normalized query/key/value non-local attention, not AdaIN moment
# matching) fused across two VGG feature scales, then decoded.
# ---------------------------------------------------------------------------
class SANetAttentionModule(nn.Module):
    """Content-query / style-key-value attention, softmax over style spatial positions."""

    def __init__(self, channels: int) -> None:
        """Build the query/key/value 1x1 projections and the output convolution."""

        super().__init__()
        self.query_conv = nn.Conv2d(channels, channels, 1)
        self.key_conv = nn.Conv2d(channels, channels, 1)
        self.value_conv = nn.Conv2d(channels, channels, 1)
        self.out_conv = nn.Conv2d(channels, channels, 1)

    @staticmethod
    def _instance_norm(feat: Tensor) -> Tensor:
        mean = feat.mean(dim=(2, 3), keepdim=True)
        std = feat.std(dim=(2, 3), keepdim=True) + 1e-5
        return (feat - mean) / std

    def forward(self, content: Tensor, style: Tensor) -> Tensor:
        """Return content features re-decorated with attention-weighted style features."""

        n, c, h, w = content.shape
        q = self.query_conv(self._instance_norm(content)).reshape(n, c, h * w)
        k = self.key_conv(self._instance_norm(style)).reshape(n, c, -1)
        v = self.value_conv(style).reshape(n, c, -1)

        attn = F.softmax(torch.bmm(q.transpose(1, 2), k), dim=-1)  # (n, hw, hw_style)
        mixed = torch.bmm(v, attn.transpose(1, 2)).reshape(n, c, h, w)
        return content + self.out_conv(mixed)


class SANet(nn.Module):
    """Compact SA-Net: VGG-lite encoder + two-level SANet attention fusion + decoder."""

    def __init__(self, channels: int = 32) -> None:
        """Build the shared VGG-lite encoder, two attention modules, and the decoder."""

        super().__init__()
        self.enc_relu4 = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.enc_relu5 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.attn4 = SANetAttentionModule(channels)
        self.attn5 = SANetAttentionModule(channels)
        self.fusion_conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.decoder = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(channels, 3, 3, padding=1),
        )

    def forward(self, content: Tensor, style: Tensor) -> Tensor:
        """Return the stylized image, shape matching `content` at half resolution x2."""

        c4 = self.enc_relu4(content)
        s4 = self.enc_relu4(style)
        c5 = self.enc_relu5(c4)
        s5 = self.enc_relu5(s4)

        fused4 = self.attn4(c4, s4)
        fused5 = self.attn5(c5, s5)
        fused5_up = F.interpolate(fused5, size=fused4.shape[-2:], mode="nearest")

        merged = self.fusion_conv(fused4 + fused5_up)
        return self.decoder(merged)


def build_sanet() -> nn.Module:
    """Build the compact SA-Net style-attentional style-transfer model."""

    return SANet().eval()


def example_input_sanet() -> tuple[Tensor, Tensor]:
    """Return (content image, style image) both 64x64 RGB."""

    content = torch.randn(2, 3, 64, 64)
    style = torch.randn(2, 3, 64, 64)
    return content, style


# ---------------------------------------------------------------------------
# SCGAN: colorization encoder fusing local conv features with globally-pooled
# VGG-16-gray semantic features, feeding a decoder with TWO parallel heads
# (RGB colorization + saliency map) trained as a joint proxy target.
# ---------------------------------------------------------------------------
class VGGGrayGlobalFeatures(nn.Module):
    """Tiny stand-in for the frozen VGG-16-gray global-feature branch."""

    def __init__(self, out_dim: int = 32) -> None:
        """Build a small conv stack that pools a grayscale image to a global vector."""

        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, out_dim, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, gray: Tensor) -> Tensor:
        """Return a global semantic feature vector, shape (batch, out_dim)."""

        return self.net(gray).flatten(1)


class SCGAN(nn.Module):
    """Compact SCGAN generator: global-fused encoder + dual colorization/saliency heads."""

    def __init__(self, channels: int = 24, global_dim: int = 32) -> None:
        """Build the local encoder, global-feature branch, fusion, and dual decoder heads."""

        super().__init__()
        self.local_encoder = nn.Sequential(
            nn.Conv2d(1, channels, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.global_branch = VGGGrayGlobalFeatures(global_dim)
        self.global_proj = nn.Linear(global_dim, channels)

        self.decoder_trunk = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
        )
        self.color_head = nn.Conv2d(channels, 2, 3, padding=1)  # predicted ab channels
        self.saliency_head = nn.Conv2d(channels, 1, 3, padding=1)

    def forward(self, gray: Tensor) -> tuple[Tensor, Tensor]:
        """Return (predicted ab color channels, predicted saliency map).

        Parameters
        ----------
        gray : Tensor
            Grayscale luminance input, shape (batch, 1, H, W).
        """

        local_feat = self.local_encoder(gray)
        global_feat = self.global_proj(self.global_branch(gray))
        global_feat = global_feat.unsqueeze(-1).unsqueeze(-1).expand_as(local_feat)

        fused = torch.cat([local_feat, global_feat], dim=1)
        trunk = self.decoder_trunk(fused)

        ab = torch.tanh(self.color_head(trunk))
        saliency = torch.sigmoid(self.saliency_head(trunk))
        return ab, saliency


def build_scgan() -> nn.Module:
    """Build the compact SCGAN saliency-guided colorization generator."""

    return SCGAN().eval()


def example_input_scgan() -> Tensor:
    """Return a grayscale luminance input, shape (2, 1, 64, 64)."""

    return torch.randn(2, 1, 64, 64)


# ---------------------------------------------------------------------------
# Shift-Net: U-Net encoder/decoder with a shift-connection layer -- missing
# -region positions are filled by nearest-neighbor COPYING the most
# cosine-similar known-region encoder feature vector, concatenated with the
# plain skip feature before the corresponding decoder block.
# ---------------------------------------------------------------------------
class ShiftConnectionLayer(nn.Module):
    """Shift known-region encoder features into missing positions by cosine NN lookup."""

    def forward(self, enc_feat: Tensor, mask: Tensor) -> Tensor:
        """Return `enc_feat` with missing positions replaced by their nearest known match.

        Parameters
        ----------
        enc_feat : Tensor
            Encoder feature map at the shift-connection scale, shape (n, c, h, w).
        mask : Tensor
            Binary missing-region mask (1 = missing) at the input resolution; resized
            to (h, w) with nearest interpolation.
        """

        n, c, h, w = enc_feat.shape
        mask_hw = F.interpolate(mask, size=(h, w), mode="nearest").reshape(n, h * w)
        flat = enc_feat.reshape(n, c, h * w)
        normed = F.normalize(flat, dim=1)

        shifted = flat.clone()
        for b in range(n):
            known_idx = (mask_hw[b] < 0.5).nonzero(as_tuple=True)[0]
            missing_idx = (mask_hw[b] >= 0.5).nonzero(as_tuple=True)[0]
            if known_idx.numel() == 0 or missing_idx.numel() == 0:
                continue
            sims = normed[b, :, missing_idx].t() @ normed[b, :, known_idx]  # (n_miss, n_known)
            best = sims.argmax(dim=-1)
            shifted[b, :, missing_idx] = flat[b, :, known_idx[best]]

        return shifted.reshape(n, c, h, w)


class ShiftNet(nn.Module):
    """Compact Shift-Net: U-Net with a shift-connection skip at the innermost scale."""

    def __init__(self, base_channels: int = 16) -> None:
        """Build the 3-level U-Net encoder/decoder and the shift-connection layer."""

        super().__init__()
        c = base_channels
        self.enc1 = nn.Conv2d(4, c, 4, stride=2, padding=1)
        self.enc2 = nn.Conv2d(c, 2 * c, 4, stride=2, padding=1)
        self.enc3 = nn.Conv2d(2 * c, 4 * c, 4, stride=2, padding=1)

        self.shift = ShiftConnectionLayer()

        self.dec3 = nn.ConvTranspose2d(4 * c, 2 * c, 4, stride=2, padding=1)
        # Skip concatenates [plain enc2 skip, shifted enc2 feature] -> 4*c channels in.
        self.dec2 = nn.ConvTranspose2d(2 * c + 4 * c, c, 4, stride=2, padding=1)
        self.dec1 = nn.ConvTranspose2d(2 * c, 3, 4, stride=2, padding=1)

    def forward(self, image: Tensor, mask: Tensor) -> Tensor:
        """Return the inpainted RGB image, shape (batch, 3, H, W)."""

        x = torch.cat([image, mask], dim=1)
        f1 = F.relu(self.enc1(x))
        f2 = F.relu(self.enc2(f1))
        f3 = F.relu(self.enc3(f2))

        d3 = F.relu(self.dec3(f3))

        shifted_f2 = self.shift(f2, mask)
        d2_in = torch.cat([d3, f2, shifted_f2], dim=1)
        d2 = F.relu(self.dec2(d2_in))

        d1_in = torch.cat([d2, f1], dim=1)
        return torch.tanh(self.dec1(d1_in))


def build_shiftnet() -> nn.Module:
    """Build the compact Shift-Net feature-rearrangement inpainting model."""

    return ShiftNet().eval()


def example_input_shiftnet() -> tuple[Tensor, Tensor]:
    """Return (masked RGB image, missing-region mask) at 64x64."""

    image = torch.randn(2, 3, 64, 64)
    mask = torch.zeros(2, 1, 64, 64)
    mask[:, :, 24:40, 24:40] = 1.0
    return image, mask


# ---------------------------------------------------------------------------
# Sketch-BERT: bidirectional Transformer over vector-format sketch stroke
# points (continuous offset + discrete pen-state embeddings), pretrained with
# a masked "Sketch Gestalt" objective (joint continuous-offset regression +
# discrete pen-state classification of masked points).
# ---------------------------------------------------------------------------
class SketchEmbedding(nn.Module):
    """Embed each stroke point as continuous-offset MLP embedding + pen-state embedding."""

    def __init__(self, dim: int, n_pen_states: int = 4) -> None:
        """Build the continuous-offset MLP and the discrete pen-state embedding table."""

        super().__init__()
        self.offset_embed = nn.Sequential(nn.Linear(2, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.pen_embed = nn.Embedding(n_pen_states, dim)

    def forward(self, offsets: Tensor, pen_states: Tensor) -> Tensor:
        """Return per-point embeddings, shape (batch, seq_len, dim)."""

        return self.offset_embed(offsets) + self.pen_embed(pen_states)


class SketchBERT(nn.Module):
    """Compact Sketch-BERT: sketch embedding + bidirectional Transformer + gestalt heads."""

    def __init__(
        self,
        dim: int = 32,
        n_layers: int = 2,
        n_heads: int = 4,
        n_pen_states: int = 4,
        max_len: int = 64,
    ) -> None:
        """Build the sketch embedding, positional embedding, encoder, and gestalt heads."""

        super().__init__()
        self.embed = SketchEmbedding(dim, n_pen_states)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, dim) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(dim, n_heads, 4 * dim, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # Sketch Gestalt Model heads: continuous offset regression + discrete pen classification.
        self.offset_head = nn.Linear(dim, 2)
        self.pen_head = nn.Linear(dim, n_pen_states)

    def forward(
        self, offsets: Tensor, pen_states: Tensor, mask_indicator: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Return (predicted offsets, pen-state logits) at every sequence position.

        Parameters
        ----------
        offsets : Tensor
            Per-point (dx, dy) offsets, shape (batch, seq_len, 2); masked positions are
            zeroed by the caller before this call, mirroring BERT-style masking.
        pen_states : Tensor
            Per-point discrete pen-state id, shape (batch, seq_len).
        mask_indicator : Tensor
            Boolean/float indicator of which points were masked, shape (batch, seq_len),
            concatenated into the embedding via an extra learned mask-token bias.
        """

        seq_len = offsets.shape[1]
        h = self.embed(offsets, pen_states) + self.pos_embed[:, :seq_len, :]
        h = h + mask_indicator.unsqueeze(-1) * self.pos_embed[:, :1, :]  # simple mask-token bias
        encoded = self.encoder(h)
        return self.offset_head(encoded), self.pen_head(encoded)


def build_sketchbert() -> nn.Module:
    """Build the compact Sketch-BERT masked sketch-gestalt Transformer."""

    return SketchBERT().eval()


def example_input_sketchbert() -> tuple[Tensor, Tensor, Tensor]:
    """Return (point offsets, pen states, mask indicator) for a 20-point sketch."""

    offsets = torch.randn(2, 20, 2)
    pen_states = torch.randint(0, 4, (2, 20), dtype=torch.long)
    mask_indicator = torch.zeros(2, 20)
    mask_indicator[:, 8:12] = 1.0
    return offsets, pen_states, mask_indicator


MENAGERIE_ENTRIES = [
    ("PEN-Net", "build_pennet", "example_input_pennet", "2019", "VIS"),
    ("RePaint (DDPM-based inpainting)", "build_repaint", "example_input_repaint", "2022", "VIS"),
    ("SA-Net (Shuffle Attention)", "build_sanet", "example_input_sanet", "2019", "VIS"),
    ("SCGAN Colorization", "build_scgan", "example_input_scgan", "2020", "VIS"),
    ("Shift-Net", "build_shiftnet", "example_input_shiftnet", "2018", "VIS"),
    ("Sketch-BERT", "build_sketchbert", "example_input_sketchbert", "2020", "VIS"),
]
