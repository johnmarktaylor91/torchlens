"""GigaGAN: Scaling up GANs for Text-to-Image Synthesis.

Kang et al., CVPR 2023.  arXiv:2303.05511.
Source: https://github.com/mingukkang/GigaGAN (official); faithful pure-torch reimpl.

GigaGAN's key architectural contributions:
  1. **Sample-adaptive kernel selection** (filter-bank conv): a bank of N convolutional
     filters is instantiated; a style-conditioned affine layer predicts per-sample
     softmax weights over the bank, producing one aggregated kernel per sample.
     This is the GigaGAN signature op that replaces AdaIN-style modulation.
  2. **Text cross-attention** woven into the synthesis blocks at coarse resolutions:
     the style code w (from mapping network) modulates the per-layer norm; local text
     descriptors provide keys/values for cross-attention.
  3. **Multi-scale discriminator** (MS-I/O): image pyramid is input to separate conv
     stacks, each also outputting predictions at multiple downsampling levels; a text
     projection branch provides additional conditioning.
  4. **Asymmetric U-Net upsampler**: 3 encoder (downsampling residual) blocks + 6
     decoder (upsampling residual + attention) blocks + skip connections, turning a
     64px low-res image into 512px (here compacted to 16->32 for speed).

All models here use random init, CPU, small channel counts, tiny spatial sizes -- the
goal is a faithful-shape compact unrolled graph, not a trainable model.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# Shared primitives
# ============================================================


class ModulatedNorm(nn.Module):
    """Per-sample affine scale+shift (style injection, a la StyleGAN AdaIN-lite)."""

    def __init__(self, channels: int, style_dim: int) -> None:
        super().__init__()
        self.norm = nn.InstanceNorm2d(channels, affine=False)
        self.style_scale = nn.Linear(style_dim, channels)
        self.style_bias = nn.Linear(style_dim, channels)

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # x: (B,C,H,W)  w: (B,style_dim)
        x = self.norm(x)
        scale = self.style_scale(w).unsqueeze(-1).unsqueeze(-1)  # (B,C,1,1)
        bias = self.style_bias(w).unsqueeze(-1).unsqueeze(-1)
        return x * (1 + scale) + bias


class AdaptiveKernelConv(nn.Module):
    """GigaGAN sample-adaptive kernel selection (filter-bank conv).

    Maintains a bank of N convolution kernels {K_i in R^{Cin x Cout x k x k}}.
    Given a style vector w, an affine projection predicts per-sample softmax
    weights over the N filters, yielding a single aggregated kernel per sample.
    The kernel is then applied via F.conv2d (per-sample in a loop) or via a
    grouped construction -- here we use a lightweight grouped-conv approximation
    that keeps the graph tractable and symbolically faithful to the equation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        style_dim: int,
        n_kernels: int = 4,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_kernels = n_kernels
        self.kernel_size = kernel_size
        # Filter bank: n_kernels independent conv kernels
        self.filter_bank = nn.Parameter(
            torch.randn(n_kernels, out_channels, in_channels, kernel_size, kernel_size) * 0.02
        )
        # Affine: style -> per-sample weights over the filter bank
        self.weight_proj = nn.Linear(style_dim + 1, n_kernels)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # x: (B,Cin,H,W)   w: (B,style_dim)
        B, Cin, H, W = x.shape
        k = self.kernel_size
        # Predict per-sample filter weights  (B, n_kernels)
        ones = torch.ones(B, 1, device=x.device)
        alpha = torch.softmax(self.weight_proj(torch.cat([w, ones], dim=1)), dim=-1)
        # Aggregate filter bank: (B, n_kernels) x (n_kernels, Cout, Cin, k, k)
        # filter_bank: (n_kernels, Cout, Cin, k, k)  reshaped to (N, Cout*Cin*k*k)
        fb_flat = self.filter_bank.view(self.n_kernels, -1)  # (N, Cout*Cin*k^2)
        K_flat = torch.einsum("bn,nd->bd", alpha, fb_flat)  # (B, Cout*Cin*k^2)
        K = K_flat.view(B, self.out_channels, self.in_channels, k, k)
        # Apply per-sample: loop over batch dimension for graph clarity
        outs = []
        for i in range(B):
            ki = K[i]  # (Cout, Cin, k, k)
            yi = F.conv2d(x[i : i + 1], ki, self.bias, stride=1, padding=k // 2)
            outs.append(yi)
        return torch.cat(outs, dim=0)


class TextCrossAttention(nn.Module):
    """Cross-attention: spatial features (query) attend to text tokens (key/value)."""

    def __init__(self, feat_dim: int, text_dim: int, n_heads: int = 2) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.hd = feat_dim // n_heads
        self.q_proj = nn.Conv2d(feat_dim, feat_dim, 1)
        self.k_proj = nn.Linear(text_dim, feat_dim)
        self.v_proj = nn.Linear(text_dim, feat_dim)
        self.out_proj = nn.Conv2d(feat_dim, feat_dim, 1)

    def forward(self, x: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        # x: (B,C,H,W)  text: (B,T,text_dim)
        B, C, H, W = x.shape
        T = text.size(1)
        # q: (B, nh, HW, hd)
        q = self.q_proj(x).view(B, self.n_heads, self.hd, H * W).permute(0, 1, 3, 2)
        # k, v: (B, nh, T, hd)
        k = self.k_proj(text).view(B, T, self.n_heads, self.hd).permute(0, 2, 1, 3)
        v = self.v_proj(text).view(B, T, self.n_heads, self.hd).permute(0, 2, 1, 3)
        # scores: (B, nh, HW, T)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.hd**0.5)
        attn = torch.softmax(scores, dim=-1)
        # out: (B, nh, HW, hd) -> (B, C, H, W)
        out = torch.matmul(attn, v)  # (B, nh, HW, hd)
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)
        return self.out_proj(out)


class MappingNetwork(nn.Module):
    """StyleGAN-style 4-layer MLP: z -> w."""

    def __init__(self, z_dim: int = 128, w_dim: int = 64, n_layers: int = 4) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = z_dim
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, w_dim), nn.LeakyReLU(0.2)]
            in_dim = w_dim
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class TextEncoder(nn.Module):
    """Compact learned text encoder: embedding + small transformer."""

    def __init__(self, vocab: int = 256, seq_len: int = 8, d: int = 32, out_dim: int = 64) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.randn(1, seq_len, d) * 0.01)
        enc_layer = nn.TransformerEncoderLayer(d, nhead=2, dim_feedforward=64, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=1)
        # local descriptors (per token) and global descriptor (mean-pool)
        self.local_proj = nn.Linear(d, out_dim)
        self.global_proj = nn.Linear(d, out_dim)

    def forward(self, ids: torch.Tensor):
        # ids: (B,T)
        x = self.embed(ids) + self.pos[:, : ids.size(1)]
        x = self.encoder(x)  # (B,T,d)
        local_desc = self.local_proj(x)  # (B,T,out_dim)
        global_desc = self.global_proj(x.mean(dim=1))  # (B,out_dim)
        return local_desc, global_desc


# ============================================================
# Generator synthesis blocks
# ============================================================


class SynthesisBlock(nn.Module):
    """One upsampling resolution stage of GigaGAN's synthesis network.

    Upsamples -> adaptive-kernel conv (filter-bank modulated by w) ->
    modulated norm -> optional text cross-attention -> to-RGB shortcut.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        style_dim: int,
        n_kernels: int = 4,
        text_dim: int = 0,
        use_attn: bool = False,
    ) -> None:
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = AdaptiveKernelConv(in_ch, out_ch, style_dim, n_kernels)
        self.norm = ModulatedNorm(out_ch, style_dim)
        self.act = nn.LeakyReLU(0.2)
        self.to_rgb = nn.Conv2d(out_ch, 3, 1)
        self.use_attn = use_attn
        if use_attn and text_dim > 0:
            self.cross_attn = TextCrossAttention(out_ch, text_dim)
            self.attn_norm = nn.GroupNorm(min(4, out_ch), out_ch)

    def forward(
        self, x: torch.Tensor, w: torch.Tensor, text: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.upsample(x)
        x = self.conv(x, w)
        x = self.norm(x, w)
        x = self.act(x)
        if self.use_attn and text is not None and hasattr(self, "cross_attn"):
            x = x + self.cross_attn(self.attn_norm(x), text)
        rgb = self.to_rgb(x)
        return x, rgb


# ============================================================
# 1. gigagan_generator_text2img
# ============================================================


class GigaGANGeneratorText2Img(nn.Module):
    """GigaGAN text-to-image generator (compact).

    Text tokens -> TextEncoder -> global_desc feeds MappingNetwork -> w;
    local_desc used for cross-attention in synthesis blocks.
    Synthesis: constant 4x4 learned input -> 2 upsampling stages with
    adaptive-kernel convs + text cross-attention at each resolution.
    """

    def __init__(
        self,
        z_dim: int = 128,
        w_dim: int = 64,
        base_ch: int = 32,
        n_kernels: int = 4,
        vocab: int = 256,
        seq_len: int = 8,
        text_dim: int = 32,
    ) -> None:
        super().__init__()
        self.text_encoder = TextEncoder(vocab, seq_len, text_dim, text_dim)
        # mapping network: z + global_text -> w
        self.map_net = MappingNetwork(z_dim + text_dim, w_dim, n_layers=3)
        # constant learned 4x4 start
        self.const = nn.Parameter(torch.randn(1, base_ch * 4, 4, 4))
        # stage 0: 4->8 with cross-attn
        self.stage0 = SynthesisBlock(
            base_ch * 4, base_ch * 2, w_dim, n_kernels, text_dim, use_attn=True
        )
        # stage 1: 8->16 with cross-attn
        self.stage1 = SynthesisBlock(
            base_ch * 2, base_ch, w_dim, n_kernels, text_dim, use_attn=True
        )
        # aggregate to-rgb skip
        self.rgb_combine = nn.Conv2d(3, 3, 1)

    def forward(self, z: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        # z: (B,z_dim)  ids: (B,T)
        local_desc, global_desc = self.text_encoder(ids)
        w = self.map_net(torch.cat([z, global_desc], dim=1))
        x = self.const.expand(z.size(0), -1, -1, -1)
        x, rgb0 = self.stage0(x, w, local_desc)
        x, rgb1 = self.stage1(x, w, local_desc)
        # sum skip connections (upsampled)
        rgb0_up = F.interpolate(rgb0, scale_factor=2, mode="bilinear", align_corners=False)
        return torch.tanh(self.rgb_combine(rgb0_up + rgb1))


def build_gigagan_generator_text2img() -> nn.Module:
    return GigaGANGeneratorText2Img()


def example_input_gigagan_generator_text2img() -> tuple:
    z = torch.randn(1, 128)
    ids = torch.randint(0, 256, (1, 8))
    return (z, ids)


# ============================================================
# 2 & 3. gigagan_generator_uncond / gigagan_generator_unconditional
# ============================================================


class GigaGANGeneratorUncond(nn.Module):
    """GigaGAN unconditional (no text) style-based generator (compact).

    z -> MappingNetwork -> w -> synthesis blocks with adaptive-kernel convs.
    No text encoder or cross-attention.
    """

    def __init__(
        self,
        z_dim: int = 128,
        w_dim: int = 64,
        base_ch: int = 32,
        n_kernels: int = 4,
    ) -> None:
        super().__init__()
        self.map_net = MappingNetwork(z_dim, w_dim, n_layers=3)
        self.const = nn.Parameter(torch.randn(1, base_ch * 4, 4, 4))
        self.stage0 = SynthesisBlock(base_ch * 4, base_ch * 2, w_dim, n_kernels, use_attn=False)
        self.stage1 = SynthesisBlock(base_ch * 2, base_ch, w_dim, n_kernels, use_attn=False)
        self.rgb_combine = nn.Conv2d(3, 3, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        w = self.map_net(z)
        x = self.const.expand(z.size(0), -1, -1, -1)
        x, rgb0 = self.stage0(x, w)
        x, rgb1 = self.stage1(x, w)
        rgb0_up = F.interpolate(rgb0, scale_factor=2, mode="bilinear", align_corners=False)
        return torch.tanh(self.rgb_combine(rgb0_up + rgb1))


def build_gigagan_generator_uncond() -> nn.Module:
    return GigaGANGeneratorUncond()


def build_gigagan_generator_unconditional() -> nn.Module:
    """Alias entry -- same architecture, separate catalog entry."""
    return GigaGANGeneratorUncond()


def example_input_gigagan_generator_uncond() -> torch.Tensor:
    return torch.randn(1, 128)


def example_input_gigagan_generator_unconditional() -> torch.Tensor:
    return torch.randn(1, 128)


# ============================================================
# Discriminator building blocks
# ============================================================


class DiscDownBlock(nn.Module):
    """Downsampling conv block for the multi-scale discriminator image branch."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)
        self.norm = nn.GroupNorm(min(4, out_ch), out_ch)
        self.act = nn.LeakyReLU(0.2)
        # per-scale logit head
        self.logit = nn.Conv2d(out_ch, 1, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.act(self.norm(self.conv(x)))
        return x, self.logit(x)


class TextDiscBranch(nn.Module):
    """Text branch of the GigaGAN discriminator: embedding -> projection."""

    def __init__(self, vocab: int = 256, seq_len: int = 8, d: int = 32, proj_dim: int = 16) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.pool = nn.Linear(d, proj_dim)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        # ids: (B,T) -> (B,proj_dim)
        return self.pool(self.embed(ids).mean(dim=1))


# ============================================================
# 4. gigagan_discriminator (text-conditioned multi-scale)
# ============================================================


class GigaGANDiscriminator(nn.Module):
    """GigaGAN multi-scale MS-I/O discriminator with text conditioning.

    Image branch: builds a 3-level pyramid (H, H/2, H/4), passes each through
    separate conv stacks, emits per-scale logits (MS-I/O).
    Text branch: encodes token ids -> projection vector added to each logit map.
    """

    def __init__(
        self,
        base_ch: int = 16,
        img_ch: int = 3,
        vocab: int = 256,
        seq_len: int = 8,
        text_dim: int = 16,
    ) -> None:
        super().__init__()
        # stem per pyramid scale
        self.from_rgb_0 = nn.Conv2d(img_ch, base_ch, 1)
        self.from_rgb_1 = nn.Conv2d(img_ch, base_ch, 1)
        self.from_rgb_2 = nn.Conv2d(img_ch, base_ch, 1)
        # down stacks (shared depth at each scale)
        self.down0 = DiscDownBlock(base_ch, base_ch * 2)
        self.down1 = DiscDownBlock(base_ch * 2 + base_ch, base_ch * 4)
        self.down2 = DiscDownBlock(base_ch * 4 + base_ch, base_ch * 4)
        # text branch
        self.text_branch = TextDiscBranch(vocab, seq_len, text_dim, text_dim)
        self.text_proj = nn.Linear(text_dim, 1)
        # final logit aggregator
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.final_fc = nn.Linear(base_ch * 4, 1)

    def forward(self, img: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        # Build pyramid
        img_half = F.avg_pool2d(img, 2)
        img_quarter = F.avg_pool2d(img_half, 2)
        f0 = F.leaky_relu(self.from_rgb_0(img))
        f1 = F.leaky_relu(self.from_rgb_1(img_half))
        f2 = F.leaky_relu(self.from_rgb_2(img_quarter))
        # Scale 0 -> down
        x0, logit0 = self.down0(f0)  # -> half res
        # Merge with scale-1 features
        x01 = torch.cat([x0, f1], dim=1)
        x1, logit1 = self.down1(x01)  # -> quarter res
        # Merge with scale-2 features
        x12 = torch.cat([x1, f2], dim=1)
        x2, logit2 = self.down2(x12)
        # Text conditioning
        t = self.text_branch(ids)  # (B,text_dim)
        t_score = self.text_proj(t)  # (B,1)
        # Aggregate multi-scale logits + text
        agg = (
            self.global_pool(logit0).squeeze(-1).squeeze(-1)
            + self.global_pool(logit1).squeeze(-1).squeeze(-1)
            + self.global_pool(logit2).squeeze(-1).squeeze(-1)
        )  # (B,1)
        feat = self.global_pool(x2).view(x2.size(0), -1)
        return self.final_fc(feat) + agg + t_score


def build_gigagan_discriminator() -> nn.Module:
    return GigaGANDiscriminator()


def example_input_gigagan_discriminator() -> tuple:
    img = torch.randn(1, 3, 32, 32)
    ids = torch.randint(0, 256, (1, 8))
    return (img, ids)


# ============================================================
# 5. gigagan_discriminator_uncond (unconditional multi-scale)
# ============================================================


class GigaGANDiscriminatorUncond(nn.Module):
    """GigaGAN multi-scale MS-I/O discriminator, unconditional variant.

    Same multi-scale image pyramid architecture as the text-conditioned version,
    but without any text branch.
    """

    def __init__(self, base_ch: int = 16, img_ch: int = 3) -> None:
        super().__init__()
        self.from_rgb_0 = nn.Conv2d(img_ch, base_ch, 1)
        self.from_rgb_1 = nn.Conv2d(img_ch, base_ch, 1)
        self.from_rgb_2 = nn.Conv2d(img_ch, base_ch, 1)
        self.down0 = DiscDownBlock(base_ch, base_ch * 2)
        self.down1 = DiscDownBlock(base_ch * 2 + base_ch, base_ch * 4)
        self.down2 = DiscDownBlock(base_ch * 4 + base_ch, base_ch * 4)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.final_fc = nn.Linear(base_ch * 4, 1)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        img_half = F.avg_pool2d(img, 2)
        img_quarter = F.avg_pool2d(img_half, 2)
        f0 = F.leaky_relu(self.from_rgb_0(img))
        f1 = F.leaky_relu(self.from_rgb_1(img_half))
        f2 = F.leaky_relu(self.from_rgb_2(img_quarter))
        x0, logit0 = self.down0(f0)
        x01 = torch.cat([x0, f1], dim=1)
        x1, logit1 = self.down1(x01)
        x12 = torch.cat([x1, f2], dim=1)
        x2, logit2 = self.down2(x12)
        agg = (
            self.global_pool(logit0).squeeze(-1).squeeze(-1)
            + self.global_pool(logit1).squeeze(-1).squeeze(-1)
            + self.global_pool(logit2).squeeze(-1).squeeze(-1)
        )
        feat = self.global_pool(x2).view(x2.size(0), -1)
        return self.final_fc(feat) + agg


def build_gigagan_discriminator_uncond() -> nn.Module:
    return GigaGANDiscriminatorUncond()


def example_input_gigagan_discriminator_uncond() -> torch.Tensor:
    return torch.randn(1, 3, 32, 32)


# ============================================================
# 6. gigagan_unet_upsampler
# ============================================================


class UpsamplerResBlock(nn.Module):
    """Residual block for the GigaGAN U-Net upsampler encoder or decoder."""

    def __init__(
        self, in_ch: int, out_ch: int, upsample: bool = False, downsample: bool = False
    ) -> None:
        super().__init__()
        self.do_upsample = upsample
        self.do_downsample = downsample
        self.norm1 = nn.GroupNorm(min(4, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(4, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.leaky_relu(self.norm1(x))
        if self.do_upsample:
            h = F.interpolate(h, scale_factor=2, mode="nearest")
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        elif self.do_downsample:
            h = F.avg_pool2d(h, 2)
            x = F.avg_pool2d(x, 2)
        h = self.conv1(h)
        h = self.conv2(F.leaky_relu(self.norm2(h)))
        return h + self.skip(x)


class UpsamplerSelfAttention(nn.Module):
    """Spatial self-attention for the U-Net upsampler bottleneck."""

    def __init__(self, ch: int, n_heads: int = 2) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(min(4, ch), ch)
        self.attn = nn.MultiheadAttention(ch, n_heads, batch_first=True)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, H * W).permute(0, 2, 1)  # (B,HW,C)
        h, _ = self.attn(h, h, h)
        h = h.permute(0, 2, 1).view(B, C, H, W)
        return x + self.proj(h)


class GigaGANUNetUpsampler(nn.Module):
    """GigaGAN asymmetric U-Net super-resolution upsampler (compact).

    Encoder: 3 downsampling residual blocks (16->8->4->2 with doubles channels).
    Bottleneck: self-attention at lowest resolution.
    Decoder: 3 upsampling residual blocks with skip connections from encoder
             + adaptive-kernel convs + self-attention layers.
    Output: to-RGB projection at final resolution.

    Compact: 16->32 pixel (instead of paper's 64->512), base_ch=16.
    """

    def __init__(
        self,
        in_ch: int = 3,
        base_ch: int = 16,
        style_dim: int = 32,
        n_kernels: int = 4,
    ) -> None:
        super().__init__()
        # Input projection
        self.from_input = nn.Conv2d(in_ch, base_ch, 3, padding=1)
        # Encoder (3 downsample stages)
        self.enc0 = UpsamplerResBlock(base_ch, base_ch * 2, downsample=True)
        self.enc1 = UpsamplerResBlock(base_ch * 2, base_ch * 4, downsample=True)
        self.enc2 = UpsamplerResBlock(base_ch * 4, base_ch * 4, downsample=True)
        # Bottleneck
        self.bottleneck = UpsamplerResBlock(base_ch * 4, base_ch * 4)
        self.bottleneck_attn = UpsamplerSelfAttention(base_ch * 4)
        # Latent style code (from global avg pool of lowest-res features)
        self.style_proj = nn.Linear(base_ch * 4, style_dim)
        # Decoder (3 upsample stages with skip connections + adaptive-kernel conv)
        # Pattern: upsample bt first, then cat with skip at same resolution.
        self.dec2_up = UpsamplerResBlock(base_ch * 4, base_ch * 4, upsample=True)  # 2->4
        self.dec2 = UpsamplerResBlock(base_ch * 4 + base_ch * 4, base_ch * 4)  # cat e2
        self.dec2_akc = AdaptiveKernelConv(base_ch * 4, base_ch * 4, style_dim, n_kernels)
        self.dec2_attn = UpsamplerSelfAttention(base_ch * 4)
        self.dec1_up = UpsamplerResBlock(base_ch * 4, base_ch * 2, upsample=True)  # 4->8
        self.dec1 = UpsamplerResBlock(base_ch * 2 + base_ch * 2, base_ch * 2)  # cat e1
        self.dec1_akc = AdaptiveKernelConv(base_ch * 2, base_ch * 2, style_dim, n_kernels)
        self.dec1_attn = UpsamplerSelfAttention(base_ch * 2)
        self.dec0_up = UpsamplerResBlock(base_ch * 2, base_ch, upsample=True)  # 8->16
        self.dec0 = UpsamplerResBlock(base_ch + base_ch, base_ch)  # cat e0
        self.dec0_akc = AdaptiveKernelConv(base_ch, base_ch, style_dim, n_kernels)
        # Output to RGB
        self.to_rgb = nn.Sequential(
            nn.GroupNorm(min(4, base_ch), base_ch),
            nn.SiLU(),
            nn.Conv2d(base_ch, 3, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,3,H,W)  low-res input
        h = self.from_input(x)  # (B,base_ch,H,W)
        # Encoder
        e0 = h  # (B,base_ch,H,W)
        e1 = self.enc0(e0)  # (B,base_ch*2,H/2,W/2)
        e2 = self.enc1(e1)  # (B,base_ch*4,H/4,W/4)
        bt = self.enc2(e2)  # (B,base_ch*4,H/8,W/8)
        # Bottleneck + style
        bt = self.bottleneck_attn(self.bottleneck(bt))
        style = self.style_proj(bt.mean(dim=[-2, -1]))  # (B,style_dim)
        # Decoder: upsample first, then cat skip at same resolution
        d2 = self.dec2(torch.cat([self.dec2_up(bt), e2], dim=1))  # (B,4C,4,4)
        d2 = self.dec2_attn(self.dec2_akc(d2, style))
        d1 = self.dec1(torch.cat([self.dec1_up(d2), e1], dim=1))  # (B,2C,8,8)
        d1 = self.dec1_attn(self.dec1_akc(d1, style))
        d0 = self.dec0(torch.cat([self.dec0_up(d1), e0], dim=1))  # (B,C,16,16)
        d0 = self.dec0_akc(d0, style)
        return torch.tanh(self.to_rgb(d0))


def build_gigagan_unet_upsampler() -> nn.Module:
    return GigaGANUNetUpsampler()


def example_input_gigagan_unet_upsampler() -> torch.Tensor:
    """Low-res 16x16 image as upsampler input (compact; paper uses 64x64 -> 512x512)."""
    return torch.randn(1, 3, 16, 16)


# ============================================================
# MENAGERIE_ENTRIES (self-declaring classics registry)
# ============================================================

MENAGERIE_ENTRIES = [
    (
        "gigagan_generator_text2img",
        "build_gigagan_generator_text2img",
        "example_input_gigagan_generator_text2img",
        "2023",
        "DC",
    ),
    (
        "gigagan_generator_uncond",
        "build_gigagan_generator_uncond",
        "example_input_gigagan_generator_uncond",
        "2023",
        "DC",
    ),
    (
        "gigagan_generator_unconditional",
        "build_gigagan_generator_unconditional",
        "example_input_gigagan_generator_unconditional",
        "2023",
        "DC",
    ),
    (
        "gigagan_discriminator",
        "build_gigagan_discriminator",
        "example_input_gigagan_discriminator",
        "2023",
        "DC",
    ),
    (
        "gigagan_discriminator_uncond",
        "build_gigagan_discriminator_uncond",
        "example_input_gigagan_discriminator_uncond",
        "2023",
        "DC",
    ),
    (
        "gigagan_unet_upsampler",
        "build_gigagan_unet_upsampler",
        "example_input_gigagan_unet_upsampler",
        "2023",
        "DC",
    ),
]
