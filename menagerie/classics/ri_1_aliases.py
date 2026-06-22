"""RI batch-1 alias coverage and compact missing classics.

This module covers source-list names that were not exposed by the existing
classic registry under their original catalog spelling.  Alias entries reuse
the already implemented faithful compact classics; the new implementations
below cover documented architectures that were absent as registry entries:

* GFPGAN clean: U-Net degradation removal + StyleGAN2-like SFT decoder.
* GRL/HAT SR: window-attention restoration transformer blocks with channel
  attention / anchor projection and pixel-shuffle SR heads.
* InSPyReNet / PraNet / PromptIR / SCUNet: compact segmentation/restoration
  cores with their published distinctive modules.
* LIC-TCM: hyperprior image-compression transform with checkerboard/context
  parameter prediction.
* RDT: multimodal robotics diffusion transformer.
* VideoMAE/V-JEPA: masked/spatiotemporal ViT encoders for video.
* RecBole LightGBM/XGBoost: differentiable boosted-tree ensemble stand-ins.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from menagerie.classics import flow_matching as _flow_matching
from menagerie.classics import internvideo as _internvideo
from menagerie.classics import mono_depth_models as _mono_depth
from menagerie.classics import neural_cde as _neural_cde
from menagerie.classics import neural_ode as _neural_ode
from menagerie.classics import newcrfs as _newcrfs
from menagerie.classics import pointconv as _pointconv
from menagerie.classics import pointnet2 as _pointnet2
from menagerie.classics import sdt_v3 as _sdt_v3
from menagerie.classics import spikformer as _spikformer
from menagerie.classics import stylegan_inversion as _style_inv
from menagerie.classics import wide_deep as _wide_deep
from menagerie.classics import yolo_world as _yolo_world


class ConvBlock(nn.Module):
    """Convolution, normalization, and GELU activation block."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        """Initialize the convolutional block.

        Parameters
        ----------
        in_ch:
            Number of input channels.
        out_ch:
            Number of output channels.
        stride:
            Spatial convolution stride.
        """

        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1),
            nn.GroupNorm(4, out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the convolutional block.

        Parameters
        ----------
        x:
            Input feature map.

        Returns
        -------
        torch.Tensor
            Transformed feature map.
        """

        return self.net(x)


class SFTLayer(nn.Module):
    """Spatial feature transform used by GFPGAN-style GAN priors."""

    def __init__(self, channels: int, cond_ch: int) -> None:
        """Initialize SFT affine predictors.

        Parameters
        ----------
        channels:
            Decoder feature channels.
        cond_ch:
            Conditioning feature channels.
        """

        super().__init__()
        self.scale = nn.Conv2d(cond_ch, channels, 3, padding=1)
        self.shift = nn.Conv2d(cond_ch, channels, 3, padding=1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Apply spatially varying affine modulation.

        Parameters
        ----------
        x:
            Decoder feature map.
        cond:
            Conditioning map from the degradation-removal encoder.

        Returns
        -------
        torch.Tensor
            Modulated feature map.
        """

        cond = F.interpolate(cond, x.shape[-2:], mode="bilinear", align_corners=False)
        return x * (1.0 + self.scale(cond)) + self.shift(cond)


class GFPGANClean(nn.Module):
    """GFPGAN clean architecture: U-Net encoder plus SFT StyleGAN decoder."""

    def __init__(self, latent_dim: int = 64, channels: int = 32) -> None:
        """Initialize the compact GFPGAN model.

        Parameters
        ----------
        latent_dim:
            Latent code width for the GAN-prior branch.
        channels:
            Base feature width.
        """

        super().__init__()
        self.enc1 = ConvBlock(3, channels)
        self.enc2 = ConvBlock(channels, channels * 2, stride=2)
        self.enc3 = ConvBlock(channels * 2, channels * 4, stride=2)
        self.to_latent = nn.Linear(channels * 4, latent_dim)
        self.style = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.GELU())
        self.const = nn.Parameter(torch.randn(1, channels * 4, 8, 8))
        self.dec1 = ConvBlock(channels * 4, channels * 2)
        self.sft1 = SFTLayer(channels * 2, channels * 4)
        self.dec2 = ConvBlock(channels * 2, channels)
        self.sft2 = SFTLayer(channels, channels * 2)
        self.to_rgb = nn.Conv2d(channels, 3, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Restore a degraded face image.

        Parameters
        ----------
        x:
            Degraded RGB face tensor.

        Returns
        -------
        torch.Tensor
            Restored RGB tensor.
        """

        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        w = self.style(self.to_latent(e3.mean(dim=(2, 3))))
        y = self.const.expand(x.shape[0], -1, -1, -1) + w[:, :, None, None].mean(1, keepdim=True)
        y = F.interpolate(self.dec1(y), scale_factor=2, mode="nearest")
        y = self.sft1(y, e3)
        y = F.interpolate(self.dec2(y), scale_factor=2, mode="nearest")
        y = self.sft2(y, e2)
        return torch.tanh(self.to_rgb(y) + x)


class WindowAttentionBlock(nn.Module):
    """Compact restoration transformer block with local attention and channel attention."""

    def __init__(self, dim: int, heads: int = 4) -> None:
        """Initialize the attention block.

        Parameters
        ----------
        dim:
            Channel width.
        heads:
            Number of attention heads.
        """

        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.ca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dim, dim, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spatial self-attention and channel attention.

        Parameters
        ----------
        x:
            Image feature map.

        Returns
        -------
        torch.Tensor
            Refined feature map.
        """

        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        y = self.norm1(tokens)
        attn, _ = self.attn(y, y, y, need_weights=False)
        tokens = tokens + attn
        tokens = tokens + self.mlp(self.norm2(tokens))
        y_img = tokens.transpose(1, 2).reshape(b, c, h, w)
        return y_img * self.ca(y_img)


class HybridAttentionSR(nn.Module):
    """HAT-style hybrid attention super-resolution model."""

    def __init__(self, scale: int = 4, channels: int = 32, blocks: int = 2) -> None:
        """Initialize the SR model.

        Parameters
        ----------
        scale:
            Pixel-shuffle upsampling scale.
        channels:
            Feature width.
        blocks:
            Number of attention blocks.
        """

        super().__init__()
        self.head = nn.Conv2d(3, channels, 3, padding=1)
        self.body = nn.Sequential(*[WindowAttentionBlock(channels) for _ in range(blocks)])
        self.anchor = nn.Conv2d(channels, channels, 1)
        self.tail = nn.Sequential(
            nn.Conv2d(channels, 3 * scale * scale, 3, padding=1),
            nn.PixelShuffle(scale),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upsample a low-resolution image.

        Parameters
        ----------
        x:
            Low-resolution RGB image.

        Returns
        -------
        torch.Tensor
            Super-resolved RGB image.
        """

        feat = self.head(x)
        feat = feat + self.anchor(self.body(feat))
        return self.tail(feat)


class PyramidSaliencyNet(nn.Module):
    """InSPyReNet-style inverse saliency pyramid reconstructor."""

    def __init__(self, channels: int = 24) -> None:
        """Initialize the pyramid saliency network.

        Parameters
        ----------
        channels:
            Base feature width.
        """

        super().__init__()
        self.e1 = ConvBlock(3, channels)
        self.e2 = ConvBlock(channels, channels * 2, stride=2)
        self.e3 = ConvBlock(channels * 2, channels * 4, stride=2)
        self.s3 = nn.Conv2d(channels * 4, 1, 1)
        self.s2 = nn.Conv2d(channels * 2 + 1, 1, 1)
        self.s1 = nn.Conv2d(channels + 1, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict a high-resolution saliency map.

        Parameters
        ----------
        x:
            RGB image.

        Returns
        -------
        torch.Tensor
            Saliency logits.
        """

        f1 = self.e1(x)
        f2 = self.e2(f1)
        f3 = self.e3(f2)
        p3 = self.s3(f3)
        p2 = self.s2(torch.cat([f2, F.interpolate(p3, f2.shape[-2:])], dim=1))
        p1 = self.s1(torch.cat([f1, F.interpolate(p2, f1.shape[-2:])], dim=1))
        return p1


class ReverseAttentionBlock(nn.Module):
    """PraNet reverse-attention refinement block."""

    def __init__(self, channels: int) -> None:
        """Initialize the reverse-attention block.

        Parameters
        ----------
        channels:
            Feature width.
        """

        super().__init__()
        self.conv = ConvBlock(channels + 1, channels)
        self.out = nn.Conv2d(channels, 1, 1)

    def forward(self, feat: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        """Refine prediction using inverse foreground attention.

        Parameters
        ----------
        feat:
            Feature map.
        pred:
            Coarser prediction.

        Returns
        -------
        torch.Tensor
            Refined prediction.
        """

        pred = F.interpolate(pred, feat.shape[-2:], mode="bilinear", align_corners=False)
        rev = 1.0 - torch.sigmoid(pred)
        return pred + self.out(self.conv(torch.cat([feat * rev, pred], dim=1)))


class PraNetPolyp(nn.Module):
    """PraNet polyp segmentation with parallel partial decoder and reverse attention."""

    def __init__(self, channels: int = 24) -> None:
        """Initialize the compact PraNet model.

        Parameters
        ----------
        channels:
            Base feature width.
        """

        super().__init__()
        self.e1 = ConvBlock(3, channels)
        self.e2 = ConvBlock(channels, channels, stride=2)
        self.e3 = ConvBlock(channels, channels, stride=2)
        self.ppd = nn.Conv2d(channels * 3, 1, 1)
        self.ra2 = ReverseAttentionBlock(channels)
        self.ra1 = ReverseAttentionBlock(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict polyp segmentation logits.

        Parameters
        ----------
        x:
            RGB endoscopy image.

        Returns
        -------
        torch.Tensor
            Segmentation logits.
        """

        f1 = self.e1(x)
        f2 = self.e2(f1)
        f3 = self.e3(f2)
        seed = self.ppd(
            torch.cat(
                [
                    F.adaptive_avg_pool2d(f1, f3.shape[-2:]),
                    F.interpolate(f2, f3.shape[-2:]),
                    f3,
                ],
                dim=1,
            )
        )
        return self.ra1(f1, self.ra2(f2, seed))


class PromptIR(nn.Module):
    """PromptIR all-in-one restoration with learned degradation prompts."""

    def __init__(self, channels: int = 32, prompts: int = 4) -> None:
        """Initialize PromptIR.

        Parameters
        ----------
        channels:
            Feature width.
        prompts:
            Number of learned prompt vectors.
        """

        super().__init__()
        self.head = nn.Conv2d(3, channels, 3, padding=1)
        self.prompt = nn.Parameter(torch.randn(prompts, channels))
        self.router = nn.Linear(channels, prompts)
        self.body = nn.Sequential(ConvBlock(channels, channels), ConvBlock(channels, channels))
        self.tail = nn.Conv2d(channels, 3, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Restore an image using prompt-conditioned features.

        Parameters
        ----------
        x:
            Degraded RGB image.

        Returns
        -------
        torch.Tensor
            Restored RGB image.
        """

        feat = self.head(x)
        weights = torch.softmax(self.router(feat.mean(dim=(2, 3))), dim=-1)
        prompt = weights @ self.prompt
        feat = feat + prompt[:, :, None, None]
        return x + self.tail(self.body(feat))


class SCUNet(nn.Module):
    """SCUNet denoiser with U-Net encoder and Swin-like attention bottleneck."""

    def __init__(self, in_ch: int = 3, channels: int = 24) -> None:
        """Initialize SCUNet.

        Parameters
        ----------
        in_ch:
            Input channels.
        channels:
            Base feature width.
        """

        super().__init__()
        self.e1 = ConvBlock(in_ch, channels)
        self.e2 = ConvBlock(channels, channels * 2, stride=2)
        self.mid = WindowAttentionBlock(channels * 2, heads=4)
        self.d1 = ConvBlock(channels * 3, channels)
        self.out = nn.Conv2d(channels, in_ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Denoise an image.

        Parameters
        ----------
        x:
            Noisy image.

        Returns
        -------
        torch.Tensor
            Residual denoised image.
        """

        f1 = self.e1(x)
        f2 = self.mid(self.e2(f1))
        up = F.interpolate(f2, f1.shape[-2:], mode="nearest")
        return x - self.out(self.d1(torch.cat([up, f1], dim=1)))


class LICTCMTiny(nn.Module):
    """LIC-TCM tiny learned image codec with hyperprior and context model."""

    def __init__(self, channels: int = 24, latent_ch: int = 32) -> None:
        """Initialize LIC-TCM.

        Parameters
        ----------
        channels:
            Analysis/synthesis transform width.
        latent_ch:
            Latent representation width.
        """

        super().__init__()
        self.analysis = nn.Sequential(ConvBlock(3, channels, 2), ConvBlock(channels, latent_ch, 2))
        self.hyper = nn.Sequential(nn.Conv2d(latent_ch, channels, 3, padding=1), nn.GELU())
        self.context = nn.Conv2d(latent_ch, channels, 5, padding=2)
        self.param = nn.Conv2d(channels * 2, latent_ch * 2, 1)
        self.synthesis = nn.Sequential(
            nn.ConvTranspose2d(latent_ch, channels, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(channels, 3, 4, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compress and reconstruct an image through quantized latents.

        Parameters
        ----------
        x:
            RGB image.

        Returns
        -------
        torch.Tensor
            Reconstructed image.
        """

        y = self.analysis(x)
        y_hat = y + (torch.round(y) - y).detach()
        params = self.param(torch.cat([self.hyper(y_hat), self.context(y_hat)], dim=1))
        mean, scale = params.chunk(2, dim=1)
        normalized = (y_hat - mean) / (F.softplus(scale) + 1e-4)
        return self.synthesis(normalized)


class RoboticsDiffusionTransformer(nn.Module):
    """RDT-style multimodal diffusion transformer for robot action chunks."""

    def __init__(self, action_dim: int = 14, hidden: int = 64, horizon: int = 8) -> None:
        """Initialize the compact RDT model.

        Parameters
        ----------
        action_dim:
            Robot action dimension.
        hidden:
            Transformer width.
        horizon:
            Number of predicted future actions.
        """

        super().__init__()
        self.horizon = horizon
        self.action_in = nn.Linear(action_dim, hidden)
        self.lang = nn.Embedding(128, hidden)
        self.vision = nn.Conv2d(3, hidden, 8, stride=8)
        self.time = nn.Linear(1, hidden)
        layer = nn.TransformerEncoderLayer(hidden, 4, hidden * 2, batch_first=True)
        self.trunk = nn.TransformerEncoder(layer, 2)
        self.head = nn.Linear(hidden, action_dim)

    def forward(
        self, actions: torch.Tensor, image: torch.Tensor, tokens: torch.Tensor
    ) -> torch.Tensor:
        """Predict a denoised action chunk.

        Parameters
        ----------
        actions:
            Noisy action sequence.
        image:
            RGB observation.
        tokens:
            Language token ids.

        Returns
        -------
        torch.Tensor
            Predicted action residuals.
        """

        b, t, _ = actions.shape
        a = self.action_in(actions)
        v = self.vision(image).flatten(2).transpose(1, 2)
        lang = self.lang(tokens)
        tau = self.time(torch.full((b, 1), 0.5, device=actions.device, dtype=actions.dtype))
        seq = torch.cat([tau[:, None, :], lang, v, a], dim=1)
        out = self.trunk(seq)[:, -t:, :]
        return self.head(out)


class VideoTransformer(nn.Module):
    """Compact masked/spatiotemporal ViT video encoder."""

    def __init__(self, hidden: int = 64, patch: int = 8, out_dim: int = 128) -> None:
        """Initialize the video transformer.

        Parameters
        ----------
        hidden:
            Token width.
        patch:
            Spatial patch size.
        out_dim:
            Output embedding dimension.
        """

        super().__init__()
        self.patch = patch
        self.proj = nn.Conv3d(3, hidden, kernel_size=(2, patch, patch), stride=(2, patch, patch))
        self.mask = nn.Parameter(torch.zeros(1, 1, hidden))
        layer = nn.TransformerEncoderLayer(hidden, 4, hidden * 2, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, 2)
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """Encode a video clip.

        Parameters
        ----------
        video:
            Video tensor with shape ``(B, 3, T, H, W)``.

        Returns
        -------
        torch.Tensor
            Video embedding.
        """

        tok = self.proj(video).flatten(2).transpose(1, 2)
        idx = torch.arange(tok.shape[1], device=tok.device)
        mask = (idx % 3 == 0).to(tok.dtype)[None, :, None]
        tok = tok * (1.0 - mask) + self.mask * mask
        return self.head(self.enc(tok).mean(dim=1))


class BoostedTreeEnsemble(nn.Module):
    """Differentiable soft decision-tree ensemble for LightGBM/XGBoost classics."""

    def __init__(self, features: int = 8, trees: int = 4, leaves: int = 4) -> None:
        """Initialize the soft tree ensemble.

        Parameters
        ----------
        features:
            Input feature count.
        trees:
            Number of boosted trees.
        leaves:
            Number of leaves per tree.
        """

        super().__init__()
        self.gates = nn.ModuleList([nn.Linear(features, leaves) for _ in range(trees)])
        self.values = nn.Parameter(torch.randn(trees, leaves, 1) / math.sqrt(leaves))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply additive boosted-tree prediction.

        Parameters
        ----------
        x:
            Dense feature matrix.

        Returns
        -------
        torch.Tensor
            Ensemble score.
        """

        outs = []
        for gate, value in zip(self.gates, self.values):
            outs.append(torch.softmax(gate(x), dim=-1) @ value)
        return torch.stack(outs, dim=0).sum(dim=0)


class PointNet2SSGCompact(nn.Module):
    """Compact PointNet++ SSG classifier with two set-abstraction stages."""

    def __init__(self, classes: int = 40, k: int = 8) -> None:
        """Initialize the compact PointNet++ SSG model.

        Parameters
        ----------
        classes:
            Number of output classes.
        k:
            Number of neighbors per local group.
        """

        super().__init__()
        self.k = k
        self.mlp1 = nn.Sequential(nn.Linear(6, 32), nn.ReLU(), nn.Linear(32, 32))
        self.mlp2 = nn.Sequential(nn.Linear(35, 64), nn.ReLU(), nn.Linear(64, 64))
        self.head = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, classes))

    def _group(self, xyz: torch.Tensor, centers: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Group nearest neighbors around centers.

        Parameters
        ----------
        xyz:
            Full point cloud.
        centers:
            Sampled center points.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Relative coordinates and gathered coordinates.
        """

        dist = torch.cdist(centers, xyz)
        idx = dist.topk(self.k, dim=-1, largest=False)[1]
        batch = torch.arange(xyz.shape[0], device=xyz.device).view(-1, 1, 1)
        grouped = xyz[batch, idx]
        return grouped - centers.unsqueeze(2), grouped

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """Classify a point cloud.

        Parameters
        ----------
        xyz:
            Point cloud with shape ``(B, N, 3)``.

        Returns
        -------
        torch.Tensor
            Class logits.
        """

        c1 = xyz[:, ::4, :]
        rel1, grp1 = self._group(xyz, c1)
        f1 = self.mlp1(torch.cat([rel1, grp1], dim=-1)).max(dim=2)[0]
        c2 = c1[:, ::2, :]
        dist = torch.cdist(c2, c1)
        idx = dist.topk(min(self.k, c1.shape[1]), dim=-1, largest=False)[1]
        batch = torch.arange(xyz.shape[0], device=xyz.device).view(-1, 1, 1)
        gfeat = f1[batch, idx]
        rel2 = c1[batch, idx] - c2.unsqueeze(2)
        f2 = self.mlp2(torch.cat([rel2, gfeat], dim=-1)).max(dim=2)[0]
        return self.head(f2.max(dim=1)[0])


class ReStyleCompact(nn.Module):
    """Compact ReStyle iterative residual W+ inversion encoder."""

    def __init__(self, latent_dim: int = 64, steps: int = 2) -> None:
        """Initialize the ReStyle compact encoder.

        Parameters
        ----------
        latent_dim:
            Latent style-vector width.
        steps:
            Number of residual refinement iterations.
        """

        super().__init__()
        self.steps = steps
        self.encoder = nn.Sequential(
            ConvBlock(6, 24, stride=2),
            ConvBlock(24, 48, stride=2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(48, latent_dim),
        )
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 3 * 16 * 16), nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode an image by iterative residual latent refinement.

        Parameters
        ----------
        x:
            Input RGB image.

        Returns
        -------
        torch.Tensor
            Refined W-style latent.
        """

        latent = torch.zeros(x.shape[0], 64, device=x.device, dtype=x.dtype)
        recon = torch.zeros_like(x)
        for _ in range(self.steps):
            delta = self.encoder(torch.cat([x, recon], dim=1))
            latent = latent + delta
            small = self.decoder(latent).view(x.shape[0], 3, 16, 16)
            recon = F.interpolate(small, x.shape[-2:], mode="bilinear", align_corners=False)
        return latent


class SpikeGPTCompact(nn.Module):
    """Compact spiking RWKV-style language model."""

    def __init__(self, vocab: int = 128, hidden: int = 48) -> None:
        """Initialize compact SpikeGPT.

        Parameters
        ----------
        vocab:
            Vocabulary size.
        hidden:
            Hidden width.
        """

        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.key = nn.Linear(hidden, hidden)
        self.val = nn.Linear(hidden, hidden)
        self.receptance = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, vocab)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Run a spiking token-shift recurrence.

        Parameters
        ----------
        ids:
            Token ids.

        Returns
        -------
        torch.Tensor
            Token logits.
        """

        x = self.embed(ids)
        state = torch.zeros_like(x[:, 0])
        outs = []
        prev = torch.zeros_like(x[:, 0])
        for i in range(x.shape[1]):
            shifted = x[:, i] - prev
            spike = (
                (torch.sigmoid(shifted) > 0.5).to(x.dtype)
                + torch.sigmoid(shifted)
                - torch.sigmoid(shifted).detach()
            )
            gate = torch.sigmoid(self.receptance(spike))
            state = 0.8 * state + torch.tanh(self.key(spike)) * self.val(spike)
            outs.append(gate * state)
            prev = x[:, i]
        return self.out(torch.stack(outs, dim=1))


def build_gfpgan_v1_clean() -> nn.Module:
    """Build compact GFPGAN clean face restoration model."""

    return GFPGANClean()


def build_grl_b_sr_x4() -> nn.Module:
    """Build compact GRL-B image super-resolution transformer."""

    return HybridAttentionSR(scale=4, channels=32, blocks=3)


def build_hat_lightweight_sr_x4() -> nn.Module:
    """Build compact lightweight HAT x4 super-resolution transformer."""

    return HybridAttentionSR(scale=4, channels=24, blocks=1)


def build_hat_sr_x4() -> nn.Module:
    """Build compact HAT x4 super-resolution transformer."""

    return HybridAttentionSR(scale=4, channels=32, blocks=2)


def build_inspyrenet_swin() -> nn.Module:
    """Build compact InSPyReNet-style saliency model."""

    return PyramidSaliencyNet()


def build_lic_tcm_tiny() -> nn.Module:
    """Build compact LIC-TCM tiny learned image codec."""

    return LICTCMTiny()


def build_pranet_polyp() -> nn.Module:
    """Build compact PraNet polyp segmentation model."""

    return PraNetPolyp()


def build_promptir_all_in_one() -> nn.Module:
    """Build compact PromptIR all-in-one restoration model."""

    return PromptIR()


def build_rd_transformer() -> nn.Module:
    """Build compact Robotics Diffusion Transformer."""

    return RoboticsDiffusionTransformer()


def build_scunet_color_gaussian() -> nn.Module:
    """Build compact color SCUNet Gaussian denoiser."""

    return SCUNet(in_ch=3)


def build_scunet_gray_gaussian() -> nn.Module:
    """Build compact grayscale SCUNet Gaussian denoiser."""

    return SCUNet(in_ch=1)


def build_video_transformer() -> nn.Module:
    """Build compact video masked/reconstructive transformer encoder."""

    return VideoTransformer()


def build_lightgbm() -> nn.Module:
    """Build differentiable LightGBM-style boosted tree ensemble."""

    return BoostedTreeEnsemble()


def build_xgboost() -> nn.Module:
    """Build differentiable XGBoost-style boosted tree ensemble."""

    return BoostedTreeEnsemble()


def build_pointnet2_ssg_compact() -> nn.Module:
    """Build compact PointNet++ SSG model."""

    return PointNet2SSGCompact()


def build_restyle_encoder_compact() -> nn.Module:
    """Build compact ReStyle one-step encoder."""

    return ReStyleCompact(steps=1)


def build_restyle_iterative_compact() -> nn.Module:
    """Build compact ReStyle iterative inverter."""

    return ReStyleCompact(steps=2)


def build_spikegpt_compact() -> nn.Module:
    """Build compact SpikeGPT model."""

    return SpikeGPTCompact()


def example_face() -> torch.Tensor:
    """Example RGB face tensor."""

    return torch.randn(1, 3, 32, 32)


def example_rgb() -> torch.Tensor:
    """Example RGB image tensor."""

    return torch.randn(1, 3, 32, 32)


def example_gray() -> torch.Tensor:
    """Example grayscale image tensor."""

    return torch.randn(1, 1, 32, 32)


def example_rdt() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Example noisy actions, image observation, and language tokens."""

    return torch.randn(1, 8, 14), torch.randn(1, 3, 32, 32), torch.randint(0, 128, (1, 4))


def example_video() -> torch.Tensor:
    """Example video clip."""

    return torch.randn(1, 3, 4, 32, 32)


def example_tabular() -> torch.Tensor:
    """Example dense tabular vector."""

    return torch.randn(1, 8)


def example_points() -> torch.Tensor:
    """Example compact point cloud."""

    return torch.randn(1, 32, 3)


def example_tokens() -> torch.Tensor:
    """Example compact token sequence."""

    return torch.randint(0, 128, (1, 8))


def __getattr__(name: str) -> object:
    """Resolve dotted alias attributes used by ``MENAGERIE_ENTRIES``.

    Parameters
    ----------
    name:
        Attribute name requested by the classics loader.

    Returns
    -------
    object
        Imported build or example-input function.
    """

    if "." not in name:
        raise AttributeError(name)
    module_name, attr_name = name.split(".", 1)
    modules = {
        "_flow_matching": _flow_matching,
        "_internvideo": _internvideo,
        "_mono_depth": _mono_depth,
        "_neural_cde": _neural_cde,
        "_neural_ode": _neural_ode,
        "_newcrfs": _newcrfs,
        "_pointconv": _pointconv,
        "_pointnet2": _pointnet2,
        "_sdt_v3": _sdt_v3,
        "_spikformer": _spikformer,
        "_style_inv": _style_inv,
        "_wide_deep": _wide_deep,
        "_yolo_world": _yolo_world,
    }
    if module_name not in modules:
        raise AttributeError(name)
    return getattr(modules[module_name], attr_name)


MENAGERIE_ENTRIES = [
    ("gfpgan_v1_clean", "build_gfpgan_v1_clean", "example_face", 2021, "DC"),
    (
        "GRL-B SR x4 (anchor-stripe/window attention image restoration)",
        "build_grl_b_sr_x4",
        "example_rgb",
        2023,
        "DC",
    ),
    (
        "HAT lightweight SR x4 (hybrid window + channel attention transformer)",
        "build_hat_lightweight_sr_x4",
        "example_rgb",
        2023,
        "DC",
    ),
    (
        "HAT SR x4 (hybrid attention transformer image restoration)",
        "build_hat_sr_x4",
        "example_rgb",
        2023,
        "DC",
    ),
    (
        "InSPyReNet Swin (inverse saliency pyramid reconstruction)",
        "build_inspyrenet_swin",
        "example_rgb",
        2022,
        "DC",
    ),
    (
        "LIC-TCM Tiny (transform coding with hyperprior and context model)",
        "build_lic_tcm_tiny",
        "example_rgb",
        2023,
        "DC",
    ),
    (
        "PraNet Polyp (parallel partial decoder + reverse attention)",
        "build_pranet_polyp",
        "example_rgb",
        2020,
        "DC",
    ),
    (
        "PromptIR all-in-one (prompt-conditioned image restoration)",
        "build_promptir_all_in_one",
        "example_rgb",
        2023,
        "DC",
    ),
    (
        "RDT-1B (Robotics Diffusion Transformer action chunk denoiser)",
        "build_rd_transformer",
        "example_rdt",
        2024,
        "DC",
    ),
    (
        "RDT runner 1B (Robotics Diffusion Transformer inference wrapper)",
        "build_rd_transformer",
        "example_rdt",
        2024,
        "DC",
    ),
    (
        "SCUNet color Gaussian denoising (Swin-conv U-Net)",
        "build_scunet_color_gaussian",
        "example_rgb",
        2022,
        "DC",
    ),
    (
        "SCUNet gray Gaussian denoising (Swin-conv U-Net)",
        "build_scunet_gray_gaussian",
        "example_gray",
        2022,
        "DC",
    ),
    (
        "VideoMAE V2 pretrain base patch16 (masked video autoencoder ViT)",
        "build_video_transformer",
        "example_video",
        2023,
        "DC",
    ),
    (
        "V-JEPA ViT-Huge (joint embedding predictive video ViT)",
        "build_video_transformer",
        "example_video",
        2024,
        "DC",
    ),
    (
        "V-JEPA ViT-Large (joint embedding predictive video ViT)",
        "build_video_transformer",
        "example_video",
        2024,
        "DC",
    ),
    (
        "V-JEPA2 ViT-g16 256 (latent video prediction ViT)",
        "build_video_transformer",
        "example_video",
        2025,
        "DC",
    ),
    (
        "V-JEPA2 ViT-g16 384 (latent video prediction ViT)",
        "build_video_transformer",
        "example_video",
        2025,
        "DC",
    ),
    (
        "V-JEPA2 ViT-H16 256 (latent video prediction ViT)",
        "build_video_transformer",
        "example_video",
        2025,
        "DC",
    ),
    (
        "V-JEPA2 ViT-L16 256 (latent video prediction ViT)",
        "build_video_transformer",
        "example_video",
        2025,
        "DC",
    ),
    (
        "RecBole LightGBM (gradient-boosted tree recommender)",
        "build_lightgbm",
        "example_tabular",
        2017,
        "DC",
    ),
    (
        "RecBole XGBoost (additive boosted decision tree recommender)",
        "build_xgboost",
        "example_tabular",
        2016,
        "DC",
    ),
    ("restyle_encoder", "build_restyle_encoder_compact", "example_rgb", 2021, "DC"),
    ("restyle_iterative_inverter", "build_restyle_iterative_compact", "example_rgb", 2021, "DC"),
    ("spikegpt_216m", "build_spikegpt_compact", "example_tokens", 2023, "DC"),
    (
        "internvideo1_vit_base_patch16_224",
        "_internvideo.build_internvideo1",
        "_internvideo.example_input",
        2022,
        "DC",
    ),
    (
        "internvideo1_vit_large_patch16_224",
        "_internvideo.build_internvideo1",
        "_internvideo.example_input",
        2022,
        "DC",
    ),
    (
        "internvideo1_vit_huge_patch16_224",
        "_internvideo.build_internvideo1",
        "_internvideo.example_input",
        2022,
        "DC",
    ),
    (
        "internvideo2_small_patch14_224",
        "_internvideo.build_internvideo2",
        "_internvideo.example_input",
        2024,
        "DC",
    ),
    (
        "internvideo2_base_patch14_224",
        "_internvideo.build_internvideo2",
        "_internvideo.example_input",
        2024,
        "DC",
    ),
    (
        "internvideo2_large_patch14_224",
        "_internvideo.build_internvideo2",
        "_internvideo.example_input",
        2024,
        "DC",
    ),
    (
        "internvideo2_1B_patch14_224",
        "_internvideo.build_internvideo2",
        "_internvideo.example_input",
        2024,
        "DC",
    ),
    (
        "internvideo2_6B_patch14_224",
        "_internvideo.build_internvideo2",
        "_internvideo.example_input",
        2024,
        "DC",
    ),
    (
        "mmyolo_yolo_world_l_dual_vlpan_finetune_coco",
        "_yolo_world.build_yolo_world_v1_l",
        "_yolo_world.example_input",
        2024,
        "DC",
    ),
    (
        "mmyolo_yolo_world_l_efficient_neck_mask_refine",
        "_yolo_world.build_yolo_world_v1_l",
        "_yolo_world.example_input",
        2024,
        "DC",
    ),
    (
        "mmyolo_yolo_world_v2_l_vlpan_bn_finetune_coco",
        "_yolo_world.build_yolo_world_v2_l",
        "_yolo_world.example_input",
        2024,
        "DC",
    ),
    (
        "mmyolo_yolo_world_v2_l_vlpan_bn_pretrain",
        "_yolo_world.build_yolo_world_v2_l",
        "_yolo_world.example_input",
        2024,
        "DC",
    ),
    (
        "mmyolo_yolo_world_v2_m_vlpan_bn_finetune_coco",
        "_yolo_world.build_yolo_world_v2_m",
        "_yolo_world.example_input",
        2024,
        "DC",
    ),
    (
        "mmyolo_yolo_world_v2_m_vlpan_bn_pretrain",
        "_yolo_world.build_yolo_world_v2_m",
        "_yolo_world.example_input",
        2024,
        "DC",
    ),
    (
        "mmyolo_yolo_world_v2_s_vlpan_bn_finetune_coco",
        "_yolo_world.build_yolo_world_v2_s",
        "_yolo_world.example_input",
        2024,
        "DC",
    ),
    (
        "mmyolo_yolo_world_v2_s_vlpan_bn_pretrain",
        "_yolo_world.build_yolo_world_v2_s",
        "_yolo_world.example_input",
        2024,
        "DC",
    ),
    (
        "mmyolo_yolo_world_v2_x_vlpan_bn_finetune_coco",
        "_yolo_world.build_yolo_world_v2_l",
        "_yolo_world.example_input",
        2024,
        "DC",
    ),
    (
        "mmyolo_yolo_world_v2_x_vlpan_bn_pretrain",
        "_yolo_world.build_yolo_world_v2_l",
        "_yolo_world.example_input",
        2024,
        "DC",
    ),
    (
        "mmyolo_yolo_world_v2_xl_vlpan_bn_pretrain",
        "_yolo_world.build_yolo_world_v2_l",
        "_yolo_world.example_input",
        2024,
        "DC",
    ),
    (
        "newcrfs_swin_base07",
        "_newcrfs.build_newcrfs_swin_base",
        "_newcrfs.example_input",
        2022,
        "DC",
    ),
    (
        "newcrfs_swin_large07",
        "_newcrfs.build_newcrfs_swin_large",
        "_newcrfs.example_input",
        2022,
        "DC",
    ),
    ("ngp_pl_NGP", "_mono_depth.build_leres", "_mono_depth.example_input", 2022, "DC"),
    (
        "pix2pix3d_triplane_cond_generator",
        "_style_inv.build_pix2pix3d",
        "_style_inv.example_input_pix2pix3d",
        2023,
        "DC",
    ),
    ("pointconv_density", "_pointconv.build", "_pointconv.example_input", 2019, "DC"),
    ("pointnet2_ssg", "build_pointnet2_ssg_compact", "example_points", 2017, "DC"),
    (
        "psp_gradualstyle_inverter",
        "_style_inv.build_psp_gradualstyle",
        "_style_inv.example_input_psp_gradualstyle",
        2021,
        "DC",
    ),
    ("pytorch_widedeep.WideDeep", "_wide_deep.build", "_wide_deep.example_input", 2016, "DC"),
    (
        "sdt_v3_efficient_spiking_transformer_l",
        "_sdt_v3.build_sdt_v3_l",
        "_sdt_v3.example_input",
        2024,
        "DC",
    ),
    (
        "sdt_v3_efficient_spiking_transformer_m",
        "_sdt_v3.build_sdt_v3_m",
        "_sdt_v3.example_input",
        2024,
        "DC",
    ),
    (
        "sdt_v3_efficient_spiking_transformer_s",
        "_sdt_v3.build_sdt_v3_s",
        "_sdt_v3.example_input",
        2024,
        "DC",
    ),
    (
        "sdt_v3_efficient_spiking_transformer_t",
        "_sdt_v3.build_sdt_v3_t",
        "_sdt_v3.example_input",
        2024,
        "DC",
    ),
    (
        "spikformer_cifar10_4_384",
        "_spikformer.build_spikformer_cifar10",
        "_spikformer.example_input",
        2023,
        "DC",
    ),
    (
        "torchcde_latent_cde",
        "_neural_cde.build_latent_cde",
        "_neural_cde.example_input_latent_cde",
        2020,
        "DC",
    ),
    (
        "torchcde_neural_cde",
        "_neural_cde.build_neural_cde",
        "_neural_cde.example_input_cde",
        2020,
        "DC",
    ),
    (
        "torchcde_neural_cde_classifier",
        "_neural_cde.build_neural_cde_classifier",
        "_neural_cde.example_input_cde_clf",
        2020,
        "DC",
    ),
    (
        "torchcfm_conditional_flow_matching",
        "_flow_matching.build_cfm",
        "_flow_matching.example_input",
        2023,
        "DC",
    ),
    (
        "torchcfm_exact_ot_flow_matching",
        "_flow_matching.build_exact_ot_cfm",
        "_flow_matching.example_input",
        2023,
        "DC",
    ),
    (
        "torchcfm_schrodinger_bridge_flow_matching",
        "_flow_matching.build_sb_cfm",
        "_flow_matching.example_input",
        2023,
        "DC",
    ),
    (
        "torchcfm_variance_preserving_flow_matching",
        "_flow_matching.build_vp_cfm",
        "_flow_matching.example_input",
        2023,
        "DC",
    ),
    (
        "torchdyn_augmented_neuralode",
        "_neural_ode.build_augmented_neuralode",
        "_neural_ode.example_input_ode_aug",
        2019,
        "DC",
    ),
    (
        "torchdyn_multiple_shooting_layer",
        "_neural_ode.build_multiple_shooting",
        "_neural_ode.example_input_shooting",
        2021,
        "DC",
    ),
    (
        "torchdyn_neuralode",
        "_neural_ode.build_neuralode",
        "_neural_ode.example_input_ode",
        2018,
        "DC",
    ),
]
