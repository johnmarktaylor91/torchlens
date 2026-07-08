# SOURCE: vendored from giuvecchio/matfuse @ main
#
# MatFuse (Vecchio, Prinzi, Carrara, Pietroni, Palomba, Cucchiara. 2024,
# CVPR, "MatFuse: Controllable Material Generation with Diffusion Models")
# -- a latent-diffusion pipeline for controllable PBR material (SVBRDF map)
# generation, with a 4-map (diffuse/normal/roughness/specular) multi-encoder
# VQ-VAE feeding a shared decoder and a UNet2DConditionModel diffusion prior.
# The traced model here is `MatFuseVQModel`, the real repo's
# `diffusers`-native (`ModelMixin`/`ConfigMixin`) VQ-VAE autoencoder --
# vendored VERBATIM (only cosmetic, no architectural change: the
# `super().__init__()` no-arg call already matches `diffusers`' expected
# `ModelMixin` contract, so nothing needed adjusting).
#
# Real source file (copied verbatim below):
#   https://raw.githubusercontent.com/giuvecchio/matfuse/main/src/diffusers_pipeline/vae_matfuse.py
#
# MENAGERIE_ZOO = "vendored-pytorch"

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

MENAGERIE_ZOO = "vendored-pytorch"


def Normalize(in_channels: int, num_groups: int = 32) -> nn.GroupNorm:
    """Group normalization."""
    return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


def nonlinearity(x: torch.Tensor) -> torch.Tensor:
    """Swish activation."""
    return x * torch.sigmoid(x)


class Upsample(nn.Module):
    """Upsampling layer with optional convolution."""

    def __init__(self, in_channels: int, with_conv: bool = True):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """Downsampling layer with optional convolution."""

    def __init__(self, in_channels: int, with_conv: bool = True):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = F.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    """Residual block with optional time embedding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        conv_shortcut: bool = False,
        dropout: float = 0.0,
        temb_channels: int = 0,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels, out_channels)

        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=1
                )
            else:
                self.nin_shortcut = nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, padding=0
                )

    def forward(self, x: torch.Tensor, temb: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None and hasattr(self, "temb_proj"):
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class AttnBlock(nn.Module):
    """Self-attention block."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # Compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b, hw, c
        k = k.reshape(b, c, h * w)  # b, c, hw
        w_ = torch.bmm(q, k)  # b, hw, hw
        w_ = w_ * (int(c) ** (-0.5))
        w_ = F.softmax(w_, dim=2)

        # Attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b, hw, hw
        h_ = torch.bmm(v, w_)  # b, c, hw
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_


class Encoder(nn.Module):
    """Encoder module for VQ-VAE."""

    def __init__(
        self,
        ch: int = 128,
        ch_mult: Tuple[int, ...] = (1, 1, 2, 4),
        num_res_blocks: int = 2,
        attn_resolutions: Tuple[int, ...] = (),
        dropout: float = 0.0,
        in_channels: int = 3,
        resolution: int = 256,
        z_channels: int = 256,
        double_z: bool = False,
        **ignore_kwargs,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # Downsampling
        self.conv_in = nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()

        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]

            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))

            down = nn.Module()
            down.block = block
            down.attn = attn

            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, with_conv=True)
                curr_res = curr_res // 2

            self.down.append(down)

        # Middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout
        )
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout
        )

        # End
        self.norm_out = Normalize(block_in)
        out_channels = 2 * z_channels if double_z else z_channels
        self.conv_out = nn.Conv2d(block_in, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Downsampling
        h = self.conv_in(x)

        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h, None)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
            if hasattr(self.down[i_level], "downsample"):
                h = self.down[i_level].downsample(h)

        # Middle
        h = self.mid.block_1(h, None)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, None)

        # End
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)

        return h


class Decoder(nn.Module):
    """Decoder module for VQ-VAE."""

    def __init__(
        self,
        ch: int = 128,
        out_ch: int = 12,
        ch_mult: Tuple[int, ...] = (1, 1, 2, 4),
        num_res_blocks: int = 2,
        attn_resolutions: Tuple[int, ...] = (),
        dropout: float = 0.0,
        in_channels: int = 3,
        resolution: int = 256,
        z_channels: int = 256,
        give_pre_end: bool = False,
        **ignore_kwargs,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end

        # Compute in_ch_mult and block_in
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // (2 ** (self.num_resolutions - 1))

        # z to block_in
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # Middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout
        )
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout
        )

        # Upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]

            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))

            up = nn.Module()
            up.block = block
            up.attn = attn

            if i_level != 0:
                up.upsample = Upsample(block_in, with_conv=True)
                curr_res = curr_res * 2

            self.up.insert(0, up)

        # End
        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z to block_in
        h = self.conv_in(z)

        # Middle
        h = self.mid.block_1(h, None)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, None)

        # Upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, None)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if hasattr(self.up[i_level], "upsample"):
                h = self.up[i_level].upsample(h)

        # End
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)

        return h


class VectorQuantizer(nn.Module):
    """Vector Quantizer module. Discretizes the input vectors using a
    learned codebook."""

    def __init__(self, n_embed: int, embed_dim: int, beta: float = 0.25):
        super().__init__()
        self.n_embed = n_embed
        self.embed_dim = embed_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_embed, self.embed_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_embed, 1.0 / self.n_embed)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Tuple]:
        # Reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.embed_dim)

        # Distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.einsum("bd,dn->bn", z_flattened, self.embedding.weight.t())
        )

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        # Compute loss for embedding
        loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + torch.mean((z_q - z.detach()) ** 2)

        # Preserve gradients
        z_q = z + (z_q - z).detach()

        # Reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q, loss, (None, None, min_encoding_indices)

    def get_codebook_entry(
        self, indices: torch.Tensor, shape: Optional[Tuple] = None
    ) -> torch.Tensor:
        # Get quantized latent vectors
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            # Reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q


class MatFuseVQModel(ModelMixin, ConfigMixin):
    """MatFuse VQ-VAE Model.

    This model has 4 separate encoders for each material map (diffuse,
    normal, roughness, specular) and 4 separate VQ quantizers, with a
    single shared decoder that outputs 12 channels."""

    @register_to_config
    def __init__(
        self,
        ch: int = 128,
        ch_mult: Tuple[int, ...] = (1, 1, 2, 4),
        num_res_blocks: int = 2,
        attn_resolutions: Tuple[int, ...] = (),
        dropout: float = 0.0,
        in_channels: int = 3,
        out_channels: int = 12,
        resolution: int = 256,
        z_channels: int = 256,
        n_embed: int = 4096,
        embed_dim: int = 3,
        scaling_factor: float = 1.0,
    ):
        super().__init__()

        self.scaling_factor = scaling_factor
        self.embed_dim = embed_dim

        ddconfig = dict(
            ch=ch,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            attn_resolutions=attn_resolutions,
            dropout=dropout,
            in_channels=in_channels,
            resolution=resolution,
            z_channels=z_channels,
            double_z=False,
        )

        # 4 separate encoders for each material map
        self.encoder_0 = Encoder(**ddconfig)
        self.encoder_1 = Encoder(**ddconfig)
        self.encoder_2 = Encoder(**ddconfig)
        self.encoder_3 = Encoder(**ddconfig)

        # Single decoder
        decoder_config = dict(
            ch=ch,
            out_ch=out_channels,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            attn_resolutions=attn_resolutions,
            dropout=dropout,
            in_channels=in_channels,
            resolution=resolution,
            z_channels=z_channels,
        )
        self.decoder = Decoder(**decoder_config)

        # 4 separate quantizers
        self.quantize_0 = VectorQuantizer(n_embed, embed_dim)
        self.quantize_1 = VectorQuantizer(n_embed, embed_dim)
        self.quantize_2 = VectorQuantizer(n_embed, embed_dim)
        self.quantize_3 = VectorQuantizer(n_embed, embed_dim)

        # Quant convolutions
        self.quant_conv_0 = nn.Conv2d(z_channels, embed_dim, 1)
        self.quant_conv_1 = nn.Conv2d(z_channels, embed_dim, 1)
        self.quant_conv_2 = nn.Conv2d(z_channels, embed_dim, 1)
        self.quant_conv_3 = nn.Conv2d(z_channels, embed_dim, 1)

        # Post quant convolution (takes 4 * embed_dim channels)
        self.post_quant_conv = nn.Conv2d(embed_dim * 4, z_channels, 1)

    def encode_to_prequant(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to pre-quantized latent space."""
        h_0 = self.encoder_0(x[:, :3])
        h_1 = self.encoder_1(x[:, 3:6])
        h_2 = self.encoder_2(x[:, 6:9])
        h_3 = self.encoder_3(x[:, 9:12])

        h_0 = self.quant_conv_0(h_0)
        h_1 = self.quant_conv_1(h_1)
        h_2 = self.quant_conv_2(h_2)
        h_3 = self.quant_conv_3(h_3)

        h = torch.cat((h_0, h_1, h_2, h_3), dim=1)
        return h

    def quantize_latent(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize the latent space."""
        quant_0, emb_loss_0, info_0 = self.quantize_0(h[:, : self.embed_dim])
        quant_1, emb_loss_1, info_1 = self.quantize_1(h[:, self.embed_dim : 2 * self.embed_dim])
        quant_2, emb_loss_2, info_2 = self.quantize_2(h[:, 2 * self.embed_dim : 3 * self.embed_dim])
        quant_3, emb_loss_3, info_3 = self.quantize_3(h[:, 3 * self.embed_dim :])

        quant = torch.cat((quant_0, quant_1, quant_2, quant_3), dim=1)
        emb_loss = emb_loss_0 + emb_loss_1 + emb_loss_2 + emb_loss_3
        info = torch.stack([info_0[-1], info_1[-1], info_2[-1], info_3[-1]], dim=0)

        return quant, emb_loss, info

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to quantized latent space."""
        h = self.encode_to_prequant(x)
        quant, _, _ = self.quantize_latent(h)
        return quant * self.scaling_factor

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space to image."""
        z = z / self.scaling_factor
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the VQ-VAE."""
        h = self.encode_to_prequant(x)
        quant, diff, _ = self.quantize_latent(h)
        dec = self.decode(quant * self.scaling_factor)
        return dec, diff


def build_matfuse_vq():
    """Builds the real MatFuseVQModel at tiny dims (real defaults are
    ch=128, ch_mult=(1,1,2,4), resolution=256, z_channels=256, n_embed=4096;
    here scaled down to a 2-stage ch_mult, resolution=16, n_embed=16 for a
    fast trace). `ch`/`z_channels` are kept at 32 (not shrunk further)
    because `Normalize` (real code) hardcodes `nn.GroupNorm(num_groups=32,
    ...)` with no override anywhere in the real model -- channel counts must
    stay multiples of 32."""
    return MatFuseVQModel(
        ch=32,
        ch_mult=(1, 2),
        num_res_blocks=1,
        attn_resolutions=(),
        dropout=0.0,
        in_channels=3,
        out_channels=12,
        resolution=16,
        z_channels=32,
        n_embed=16,
        embed_dim=2,
        scaling_factor=1.0,
    )


def example_input_matfuse_vq():
    """The real model consumes a 12-channel packed SVBRDF map (4 material
    maps x 3 channels each: diffuse/normal/roughness/specular), matching
    `encode_to_prequant`'s `x[:, :3]`/`x[:, 3:6]`/`x[:, 6:9]`/`x[:, 9:12]`
    channel slicing."""
    return torch.randn(1, 12, 16, 16)


MENAGERIE_ENTRIES = [
    ("MatFuse VQ-VAE", "build_matfuse_vq", "example_input_matfuse_vq", 2024, "vendored-pytorch"),
]
