"""Compact renderable replacements for previously non-model menagerie rows.

These are small, random-init, pure-PyTorch cores for architecture atlas rendering.
They preserve the load-bearing computation pattern of each family while avoiding
external runtimes, pretrained weights, training loops, and heavyweight pipelines.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvAct(nn.Module):
    """Convolution followed by group normalization and GELU."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1) -> None:
        """Initialize the convolutional block.

        Parameters
        ----------
        in_ch:
            Input channel count.
        out_ch:
            Output channel count.
        kernel:
            Spatial kernel size.
        stride:
            Spatial stride.
        """

        super().__init__()
        padding = kernel // 2
        groups = max(1, min(8, out_ch // 4))
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding)
        self.norm = nn.GroupNorm(groups, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution, normalization, and activation.

        Parameters
        ----------
        x:
            Feature map tensor.

        Returns
        -------
        torch.Tensor
            Activated feature map.
        """

        return F.gelu(self.norm(self.conv(x)))


class ResBlock(nn.Module):
    """Two-convolution residual block."""

    def __init__(self, channels: int) -> None:
        """Initialize the residual block.

        Parameters
        ----------
        channels:
            Number of feature channels.
        """

        super().__init__()
        self.net = nn.Sequential(ConvAct(channels, channels), ConvAct(channels, channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the residual block.

        Parameters
        ----------
        x:
            Input feature map.

        Returns
        -------
        torch.Tensor
            Residual output.
        """

        return x + self.net(x)


class SimpleSelfAttention(nn.Module):
    """Small explicit multi-head self-attention over sequence tokens."""

    def __init__(self, dim: int, heads: int = 4) -> None:
        """Initialize projections.

        Parameters
        ----------
        dim:
            Token feature dimension.
        heads:
            Number of attention heads.
        """

        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        """Apply self-attention or cross-attention.

        Parameters
        ----------
        x:
            Query tokens of shape ``(B, T, C)``.
        context:
            Optional context tokens. When absent, ``x`` is used.

        Returns
        -------
        torch.Tensor
            Attention output with residual connection.
        """

        base = self.norm(x)
        kv_source = base if context is None else self.norm(context)
        q = self.qkv(base)[..., : self.heads * self.head_dim]
        kv = self.qkv(kv_source)
        k, v = torch.chunk(kv[..., self.heads * self.head_dim :], 2, dim=-1)
        bsz, q_len, _ = q.shape
        k_len = k.shape[1]
        q = q.view(bsz, q_len, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, k_len, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, k_len, self.heads, self.head_dim).transpose(1, 2)
        score = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(float(self.head_dim))
        out = torch.matmul(torch.softmax(score, dim=-1), v)
        out = out.transpose(1, 2).reshape(bsz, q_len, self.heads * self.head_dim)
        return x + self.out(out)


class TokenMixerBlock(nn.Module):
    """Transformer-style attention and MLP token mixer."""

    def __init__(self, dim: int, heads: int = 4) -> None:
        """Initialize the mixer.

        Parameters
        ----------
        dim:
            Token feature dimension.
        heads:
            Number of attention heads.
        """

        super().__init__()
        self.attn = SimpleSelfAttention(dim, heads)
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mix tokens with attention and feed-forward layers.

        Parameters
        ----------
        x:
            Token tensor.

        Returns
        -------
        torch.Tensor
            Mixed tokens.
        """

        x = self.attn(x)
        return x + self.mlp(self.norm(x))


class ReptileMetaLearner(nn.Module):
    """Reptile-style bi-level learner with manual inner-loop tensor updates."""

    def __init__(self, in_dim: int = 4, hidden: int = 12, out_dim: int = 3, steps: int = 2) -> None:
        """Initialize base MLP parameters and fixed support batch.

        Parameters
        ----------
        in_dim:
            Input feature dimension.
        hidden:
            Hidden width.
        out_dim:
            Output feature dimension.
        steps:
            Number of inner adaptation steps.
        """

        super().__init__()
        self.steps = steps
        self.inner_lr = 0.15
        self.w1 = nn.Parameter(torch.randn(in_dim, hidden) * 0.2)
        self.b1 = nn.Parameter(torch.zeros(hidden))
        self.w2 = nn.Parameter(torch.randn(hidden, out_dim) * 0.2)
        self.b2 = nn.Parameter(torch.zeros(out_dim))
        self.register_buffer("support_x", torch.randn(6, in_dim))
        self.register_buffer("support_y", torch.randn(6, out_dim))

    def _forward_params(
        self,
        x: torch.Tensor,
        w1: torch.Tensor,
        b1: torch.Tensor,
        w2: torch.Tensor,
        b2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the base MLP with explicit parameter tensors.

        Parameters
        ----------
        x:
            Input batch.
        w1:
            First-layer weight.
        b1:
            First-layer bias.
        w2:
            Second-layer weight.
        b2:
            Second-layer bias.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Hidden activation and output prediction.
        """

        h_pre = x.matmul(w1) + b1
        h = torch.relu(h_pre)
        return h, h.matmul(w2) + b2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Adapt on support examples, then evaluate query examples.

        Parameters
        ----------
        x:
            Query batch.

        Returns
        -------
        torch.Tensor
            Query predictions after inner-loop adaptation.
        """

        w1, b1, w2, b2 = self.w1, self.b1, self.w2, self.b2
        sx = self.support_x.to(device=x.device, dtype=x.dtype)
        sy = self.support_y.to(device=x.device, dtype=x.dtype)
        for _ in range(self.steps):
            h, pred = self._forward_params(sx, w1, b1, w2, b2)
            err = (pred - sy) * (2.0 / float(sx.shape[0]))
            grad_w2 = h.transpose(0, 1).matmul(err)
            grad_b2 = err.sum(dim=0)
            grad_h = err.matmul(w2.transpose(0, 1)) * (h > 0).to(h.dtype)
            grad_w1 = sx.transpose(0, 1).matmul(grad_h)
            grad_b1 = grad_h.sum(dim=0)
            w1 = w1 - self.inner_lr * grad_w1
            b1 = b1 - self.inner_lr * grad_b1
            w2 = w2 - self.inner_lr * grad_w2
            b2 = b2 - self.inner_lr * grad_b2
        _h, out = self._forward_params(x, w1, b1, w2, b2)
        return out


def build_reptile() -> nn.Module:
    """Build a compact Reptile meta-learning architecture."""

    return ReptileMetaLearner()


def example_input_reptile() -> torch.Tensor:
    """Return a small Reptile query batch."""

    return torch.randn(2, 4)


class DiffusionPolicyDenoiser(nn.Module):
    """Compact 1D U-Net denoiser conditioned on observation tokens."""

    def __init__(self, action_ch: int = 4, obs_ch: int = 4, width: int = 24) -> None:
        """Initialize the diffusion policy denoiser.

        Parameters
        ----------
        action_ch:
            Noisy action channel count.
        obs_ch:
            Observation channel count.
        width:
            Hidden width.
        """

        super().__init__()
        self.obs_proj = nn.Conv1d(obs_ch, width, 1)
        self.time = nn.Linear(1, width)
        self.enc1 = nn.Conv1d(action_ch, width, 3, padding=1)
        self.enc2 = nn.Conv1d(width, width * 2, 4, stride=2, padding=1)
        self.mid = nn.Sequential(
            nn.GroupNorm(4, width * 2),
            nn.GELU(),
            nn.Conv1d(width * 2, width * 2, 3, padding=1),
            nn.GELU(),
        )
        self.up = nn.ConvTranspose1d(width * 2, width, 4, stride=2, padding=1)
        self.out = nn.Conv1d(width * 2, action_ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Denoise an action trajectory conditioned on observations.

        Parameters
        ----------
        x:
            Concatenated action and observation tensor ``(B, 8, T)``.

        Returns
        -------
        torch.Tensor
            Predicted denoising residual.
        """

        action, obs = x[:, :4], x[:, 4:]
        cond = self.obs_proj(obs)
        time = self.time(torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)).unsqueeze(-1)
        e1 = F.gelu(self.enc1(action) + cond + time)
        e2 = F.gelu(self.enc2(e1))
        mid = self.mid(e2)
        up = F.gelu(self.up(mid))
        return self.out(torch.cat([up, e1], dim=1))


def build_diffusion_policy_ema_model() -> nn.Module:
    """Build the wrapped denoiser behind Diffusion Policy EMA."""

    return DiffusionPolicyDenoiser()


def example_input_diffusion_policy() -> torch.Tensor:
    """Return a compact action-plus-observation trajectory."""

    return torch.randn(1, 8, 32)


class DepthAnythingDPT(nn.Module):
    """DINOv2-style ViT encoder stub with a DPT dense depth head."""

    def __init__(self, width: int = 48, patch: int = 4) -> None:
        """Initialize the depth architecture.

        Parameters
        ----------
        width:
            Token width.
        patch:
            Patch size.
        """

        super().__init__()
        self.patch = patch
        self.patch_embed = nn.Conv2d(3, width, patch, stride=patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, width))
        self.pos = nn.Parameter(torch.randn(1, 65, width) * 0.02)
        self.blocks = nn.ModuleList([TokenMixerBlock(width, 4) for _ in range(3)])
        self.neck = nn.Sequential(ConvAct(width, 32), ResBlock(32))
        self.fuse = nn.Sequential(ConvAct(32, 32), ConvAct(32, 16))
        self.out = nn.Conv2d(16, 1, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict monocular relative depth.

        Parameters
        ----------
        x:
            RGB image tensor.

        Returns
        -------
        torch.Tensor
            Dense depth map.
        """

        patch = self.patch_embed(x)
        bsz, channels, height, width = patch.shape
        tokens = patch.flatten(2).transpose(1, 2)
        cls = self.cls.expand(bsz, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1) + self.pos[:, : tokens.shape[1] + 1]
        for block in self.blocks:
            tokens = block(tokens)
        spatial = tokens[:, 1:].transpose(1, 2).reshape(bsz, channels, height, width)
        feat = self.neck(spatial)
        feat = F.interpolate(feat, scale_factor=self.patch, mode="bilinear", align_corners=False)
        return F.softplus(self.out(self.fuse(feat)))


def build_depth_anything_v3_note() -> nn.Module:
    """Build a compact Depth-Anything V2-style ViT+DPT graph for the NOTE row."""

    return DepthAnythingDPT()


def example_input_depth_anything() -> torch.Tensor:
    """Return a compact RGB image for Depth-Anything."""

    return torch.randn(1, 3, 32, 32)


class LatentDiffusionCore(nn.Module):
    """Small latent diffusion U-Net with token conditioning."""

    def __init__(self, in_ch: int = 4, cond_dim: int = 32, width: int = 32) -> None:
        """Initialize the latent diffusion core.

        Parameters
        ----------
        in_ch:
            Latent channel count.
        cond_dim:
            Conditioning token dimension.
        width:
            Hidden width.
        """

        super().__init__()
        self.cond = nn.Linear(cond_dim, width)
        self.down1 = ConvAct(in_ch, width)
        self.down2 = ConvAct(width, width * 2, stride=2)
        self.mid = ResBlock(width * 2)
        self.up = nn.ConvTranspose2d(width * 2, width, 4, stride=2, padding=1)
        self.out = nn.Conv2d(width * 2, in_ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a conditioned denoising U-Net.

        Parameters
        ----------
        x:
            Latent tensor, optionally with conditioning channels appended.

        Returns
        -------
        torch.Tensor
            Denoised latent tensor.
        """

        latent = x[:, :4]
        cond_tokens = x[:, 4:].flatten(2).mean(dim=-1)
        if cond_tokens.shape[1] < 32:
            cond_tokens = F.pad(cond_tokens, (0, 32 - cond_tokens.shape[1]))
        cond = self.cond(cond_tokens[:, :32]).unsqueeze(-1).unsqueeze(-1)
        e1 = self.down1(latent) + cond
        e2 = self.down2(e1)
        mid = self.mid(e2)
        up = F.gelu(self.up(mid))
        return self.out(torch.cat([up, e1], dim=1))


class ControlNetCore(nn.Module):
    """ControlNet-style locked U-Net residual plus zero-conv control branch."""

    def __init__(self) -> None:
        """Initialize the control branch and base denoiser."""

        super().__init__()
        self.hint = ConvAct(1, 32)
        self.base = LatentDiffusionCore(4, 32, 32)
        self.zero = nn.Conv2d(32, 4, 1)
        nn.init.zeros_(self.zero.weight)
        nn.init.zeros_(self.zero.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply controlled latent denoising.

        Parameters
        ----------
        x:
            Tensor ``(B, 5, H, W)`` containing latent plus control hint.

        Returns
        -------
        torch.Tensor
            Controlled latent residual.
        """

        latent = x[:, :4]
        hint = self.hint(x[:, 4:5])
        control = self.zero(hint)
        cond = torch.cat([latent + control, hint[:, :4]], dim=1)
        return self.base(cond)


class AnimateDiffCore(nn.Module):
    """AnimateDiff-style spatial denoiser with temporal motion attention."""

    def __init__(self, width: int = 24) -> None:
        """Initialize the video core.

        Parameters
        ----------
        width:
            Hidden width.
        """

        super().__init__()
        self.in3d = nn.Conv3d(3, width, 3, padding=1)
        self.motion = SimpleSelfAttention(width, 4)
        self.spatial = nn.Conv3d(width, width, (1, 3, 3), padding=(0, 1, 1))
        self.out = nn.Conv3d(width, 3, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process video frames with temporal attention.

        Parameters
        ----------
        x:
            Video tensor ``(B, 3, T, H, W)``.

        Returns
        -------
        torch.Tensor
            Video residual tensor.
        """

        feat = F.gelu(self.in3d(x))
        pooled = feat.mean(dim=(-1, -2)).transpose(1, 2)
        mixed = self.motion(pooled).transpose(1, 2).unsqueeze(-1).unsqueeze(-1)
        feat = F.gelu(self.spatial(feat + mixed))
        return self.out(feat)


class DeblurGANv2Core(nn.Module):
    """DeblurGAN-v2-style FPN generator."""

    def __init__(self) -> None:
        """Initialize encoder, pyramid fusion, and image head."""

        super().__init__()
        self.e1 = ConvAct(3, 16)
        self.e2 = ConvAct(16, 32, stride=2)
        self.e3 = ConvAct(32, 64, stride=2)
        self.l2 = nn.Conv2d(32, 32, 1)
        self.l3 = nn.Conv2d(64, 32, 1)
        self.head = nn.Sequential(ResBlock(32), nn.Conv2d(32, 3, 3, padding=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Deblur an image through an FPN generator.

        Parameters
        ----------
        x:
            Blurred RGB image.

        Returns
        -------
        torch.Tensor
            Restored RGB image.
        """

        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        top = F.interpolate(self.l3(e3), scale_factor=2, mode="bilinear", align_corners=False)
        mid = self.l2(e2) + top
        full = F.interpolate(mid, scale_factor=2, mode="bilinear", align_corners=False)
        return x + self.head(full)


class DICCore(nn.Module):
    """DIC face super-resolution with feedback hourglass refinement."""

    def __init__(self) -> None:
        """Initialize feedback and upsampling blocks."""

        super().__init__()
        self.inp = ConvAct(3, 24)
        self.down = ConvAct(24, 48, stride=2)
        self.mid = ResBlock(48)
        self.up = nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1)
        self.refine = ResBlock(24)
        self.to_rgb = nn.Sequential(nn.Conv2d(24, 12, 3, padding=1), nn.PixelShuffle(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Super-resolve a face crop.

        Parameters
        ----------
        x:
            Low-resolution RGB face.

        Returns
        -------
        torch.Tensor
            Super-resolved RGB image.
        """

        feat = self.inp(x)
        hidden = self.mid(self.down(feat))
        feedback = F.gelu(self.up(hidden)) + feat
        return self.to_rgb(self.refine(feedback))


class DIMCore(nn.Module):
    """Deep Image Matting encoder-decoder with alpha prediction."""

    def __init__(self) -> None:
        """Initialize matting encoder and decoder."""

        super().__init__()
        self.enc = nn.Sequential(ConvAct(4, 16), ConvAct(16, 32, stride=2), ResBlock(32))
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 1, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict alpha matte from RGB plus trimap.

        Parameters
        ----------
        x:
            Four-channel matting input.

        Returns
        -------
        torch.Tensor
            Alpha matte.
        """

        return torch.sigmoid(self.dec(self.enc(x)))


class GLEANCore(nn.Module):
    """GLEAN-style encoder with latent-bank modulation and upsampling decoder."""

    def __init__(self) -> None:
        """Initialize latent bank, encoder, and decoder."""

        super().__init__()
        self.latent = nn.Parameter(torch.randn(1, 32, 8, 8) * 0.02)
        self.enc = nn.Sequential(ConvAct(3, 16), ConvAct(16, 32, stride=2), ResBlock(32))
        self.fuse = nn.Conv2d(64, 32, 1)
        self.up = nn.Sequential(nn.PixelShuffle(2), ConvAct(8, 16), nn.Conv2d(16, 3, 3, padding=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Super-resolve using encoder features and a generative latent bank.

        Parameters
        ----------
        x:
            Low-resolution RGB image.

        Returns
        -------
        torch.Tensor
            Restored RGB image.
        """

        feat = self.enc(x)
        latent = self.latent.expand(x.shape[0], -1, -1, -1)
        fused = self.fuse(torch.cat([feat, latent], dim=1))
        return self.up(fused)


class GlobalLocalCore(nn.Module):
    """Global-local inpainting generator with fused branches."""

    def __init__(self) -> None:
        """Initialize global and local branches."""

        super().__init__()
        self.local = nn.Sequential(ConvAct(4, 16), ResBlock(16), ConvAct(16, 16))
        self.global_down = nn.Sequential(ConvAct(4, 16, stride=2), ConvAct(16, 32, stride=2))
        self.global_fc = nn.Linear(32, 16)
        self.out = nn.Conv2d(32, 3, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Inpaint from local texture and global context features.

        Parameters
        ----------
        x:
            RGB image plus mask.

        Returns
        -------
        torch.Tensor
            Inpainted RGB image.
        """

        local = self.local(x)
        glob = self.global_down(x).mean(dim=(-1, -2))
        glob = self.global_fc(glob).unsqueeze(-1).unsqueeze(-1).expand_as(local)
        return self.out(torch.cat([local, glob], dim=1))


class NAFBlock(nn.Module):
    """NAFNet block with simple gate and channel attention."""

    def __init__(self, channels: int) -> None:
        """Initialize the NAF block.

        Parameters
        ----------
        channels:
            Number of channels.
        """

        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.expand = nn.Conv2d(channels, channels * 2, 1)
        self.depthwise = nn.Conv2d(channels * 2, channels * 2, 3, padding=1, groups=channels * 2)
        self.attn = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels, 1))
        self.project = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply NAF simple-gate restoration block.

        Parameters
        ----------
        x:
            Feature map.

        Returns
        -------
        torch.Tensor
            Restored feature map.
        """

        feat = self.depthwise(self.expand(self.norm(x)))
        a, b = feat.chunk(2, dim=1)
        gated = a * b
        return x + self.project(gated * self.attn(gated))


class NAFNetCore(nn.Module):
    """Compact NAFNet image-restoration U-Net."""

    def __init__(self) -> None:
        """Initialize NAFNet blocks."""

        super().__init__()
        self.inp = nn.Conv2d(3, 24, 3, padding=1)
        self.blocks = nn.Sequential(NAFBlock(24), NAFBlock(24), NAFBlock(24))
        self.out = nn.Conv2d(24, 3, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Restore an image.

        Parameters
        ----------
        x:
            Degraded RGB image.

        Returns
        -------
        torch.Tensor
            Restored RGB image.
        """

        return x + self.out(self.blocks(self.inp(x)))


class RestormerCore(nn.Module):
    """Restormer-style image restoration transformer."""

    def __init__(self, width: int = 32) -> None:
        """Initialize patch projection and transformer blocks.

        Parameters
        ----------
        width:
            Token width.
        """

        super().__init__()
        self.inp = nn.Conv2d(3, width, 3, padding=1)
        self.blocks = nn.ModuleList([TokenMixerBlock(width, 4) for _ in range(2)])
        self.out = nn.Conv2d(width, 3, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Restore an image with token mixing.

        Parameters
        ----------
        x:
            Degraded RGB image.

        Returns
        -------
        torch.Tensor
            Restored RGB image.
        """

        feat = self.inp(x)
        bsz, channels, height, width = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)
        for block in self.blocks:
            tokens = block(tokens)
        feat = tokens.transpose(1, 2).reshape(bsz, channels, height, width)
        return x + self.out(feat)


class TTSRCore(nn.Module):
    """Texture Transformer Super-Resolution with LR-reference attention."""

    def __init__(self) -> None:
        """Initialize query, key, value, and upsampling blocks."""

        super().__init__()
        self.lr = ConvAct(3, 24)
        self.ref = ConvAct(3, 24)
        self.q = nn.Conv2d(24, 24, 1)
        self.k = nn.Conv2d(24, 24, 1)
        self.v = nn.Conv2d(24, 24, 1)
        self.up = nn.Sequential(
            nn.Conv2d(24, 96, 3, padding=1),
            nn.PixelShuffle(2),
            ConvAct(24, 16),
            nn.Conv2d(16, 3, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transfer reference texture into a super-resolution stream.

        Parameters
        ----------
        x:
            Concatenated low-resolution and reference RGB images.

        Returns
        -------
        torch.Tensor
            Super-resolved RGB image.
        """

        lr = self.lr(x[:, :3])
        ref = self.ref(x[:, 3:])
        bsz, channels, height, width = lr.shape
        q = self.q(lr).flatten(2).transpose(1, 2)
        k = self.k(ref).flatten(2)
        v = self.v(ref).flatten(2).transpose(1, 2)
        attn = torch.softmax(torch.matmul(q, k) / math.sqrt(float(channels)), dim=-1)
        transferred = torch.matmul(attn, v).transpose(1, 2).reshape(bsz, channels, height, width)
        return self.up(lr + transferred)


def build_mmagic_animatediff() -> nn.Module:
    """Build compact AnimateDiff motion-module architecture."""

    return AnimateDiffCore()


def build_mmagic_controlnet() -> nn.Module:
    """Build compact ControlNet architecture."""

    return ControlNetCore()


def build_mmagic_deblurganv2() -> nn.Module:
    """Build compact DeblurGAN-v2 architecture."""

    return DeblurGANv2Core()


def build_mmagic_dic() -> nn.Module:
    """Build compact DIC face super-resolution architecture."""

    return DICCore()


def build_mmagic_dim() -> nn.Module:
    """Build compact Deep Image Matting architecture."""

    return DIMCore()


def build_mmagic_disco_diffusion() -> nn.Module:
    """Build compact guided diffusion architecture used for Disco Diffusion rows."""

    return LatentDiffusionCore()


def build_mmagic_dreambooth() -> nn.Module:
    """Build compact personalized Stable Diffusion U-Net core for DreamBooth."""

    return LatentDiffusionCore()


def build_mmagic_fastcomposer() -> nn.Module:
    """Build compact subject-conditioned diffusion core for FastComposer."""

    return LatentDiffusionCore()


def build_mmagic_glean() -> nn.Module:
    """Build compact GLEAN architecture."""

    return GLEANCore()


def build_mmagic_global_local() -> nn.Module:
    """Build compact global-local inpainting architecture."""

    return GlobalLocalCore()


def build_mmagic_nafnet() -> nn.Module:
    """Build compact NAFNet architecture."""

    return NAFNetCore()


def build_mmagic_restormer() -> nn.Module:
    """Build compact Restormer architecture."""

    return RestormerCore()


def build_mmagic_stable_diffusion() -> nn.Module:
    """Build compact Stable Diffusion latent U-Net architecture."""

    return LatentDiffusionCore()


def build_mmagic_ttsr() -> nn.Module:
    """Build compact TTSR architecture."""

    return TTSRCore()


def example_input_video() -> torch.Tensor:
    """Return a compact video tensor."""

    return torch.randn(1, 3, 4, 32, 32)


def example_input_latent_control() -> torch.Tensor:
    """Return latent plus control hint."""

    return torch.randn(1, 5, 32, 32)


def example_input_image() -> torch.Tensor:
    """Return a compact RGB image."""

    return torch.randn(1, 3, 32, 32)


def example_input_face_lr() -> torch.Tensor:
    """Return a compact low-resolution face image."""

    return torch.randn(1, 3, 16, 16)


def example_input_matte() -> torch.Tensor:
    """Return RGB plus trimap input."""

    return torch.randn(1, 4, 32, 32)


def example_input_latent() -> torch.Tensor:
    """Return latent plus compact conditioning channels."""

    return torch.randn(1, 8, 32, 32)


def example_input_inpaint() -> torch.Tensor:
    """Return RGB plus inpainting mask input."""

    return torch.randn(1, 4, 32, 32)


def example_input_ttsr() -> torch.Tensor:
    """Return low-resolution and reference image pair."""

    return torch.randn(1, 6, 24, 24)


MENAGERIE_ENTRIES = [
    ("Reptile", "build_reptile", "example_input_reptile", "2018", "DA"),
    (
        "DiffusionPolicy_EMAModel",
        "build_diffusion_policy_ema_model",
        "example_input_diffusion_policy",
        "2023",
        "DC",
    ),
    (
        "depth_anything_v3:NOTE",
        "build_depth_anything_v3_note",
        "example_input_depth_anything",
        "2024",
        "DC",
    ),
    ("mmagic_animatediff", "build_mmagic_animatediff", "example_input_video", "2023", "DC"),
    ("mmagic_controlnet", "build_mmagic_controlnet", "example_input_latent_control", "2023", "DC"),
    ("mmagic_deblurganv2", "build_mmagic_deblurganv2", "example_input_image", "2019", "DC"),
    ("mmagic_dic", "build_mmagic_dic", "example_input_face_lr", "2020", "DC"),
    ("mmagic_dim", "build_mmagic_dim", "example_input_matte", "2017", "DC"),
    (
        "mmagic_disco_diffusion",
        "build_mmagic_disco_diffusion",
        "example_input_latent",
        "2021",
        "DC",
    ),
    ("mmagic_dreambooth", "build_mmagic_dreambooth", "example_input_latent", "2022", "DC"),
    ("mmagic_fastcomposer", "build_mmagic_fastcomposer", "example_input_latent", "2023", "DC"),
    ("mmagic_glean", "build_mmagic_glean", "example_input_face_lr", "2021", "DC"),
    ("mmagic_global_local", "build_mmagic_global_local", "example_input_inpaint", "2017", "DC"),
    ("mmagic_nafnet", "build_mmagic_nafnet", "example_input_image", "2022", "DC"),
    ("mmagic_restormer", "build_mmagic_restormer", "example_input_image", "2022", "DC"),
    (
        "mmagic_stable_diffusion",
        "build_mmagic_stable_diffusion",
        "example_input_latent",
        "2022",
        "DC",
    ),
    ("mmagic_ttsr", "build_mmagic_ttsr", "example_input_ttsr", "2020", "DC"),
]
