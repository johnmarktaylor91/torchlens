# SOURCE: vendored from wzzheng/OccWorld @ main
# (model/VAE/vae_2d_resnet.py + model/VAE/quantizer.py + model/transformer/modules.py
#  + model/transformer/pose_encoder.py + model/transformer/pose_decoder.py
#  + model/transformer/PlanUtransformer.py + model/TransVQVAE.py)
"""OccWorld: GPT-style generative world model for 3D semantic occupancy
forecasting in autonomous driving (ECCV 2024).

Vendored real nn.Module code: a 2D-ResNet VAE (with vector-quantized latent
codebook) compresses per-frame voxelized occupancy into a token grid, and a
multi-scale U-Net "PlanUAutoRegTransformer" (spatio-temporal cross-attention
between learnable query tokens and past frame tokens, jointly with an
ego-pose token stream) autoregressively predicts future occupancy tokens
conditioned on past frames and ego motion.
"""

from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mmengine.model import BaseModule
from mmengine.registry import MODELS

MENAGERIE_ZOO = "vendored-pytorch"


# ============================================================================
# model/VAE/quantizer.py (real repo code, verbatim)
# ============================================================================


@MODELS.register_module()
class VectorQuantizer(BaseModule):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """

    def __init__(
        self,
        n_e,
        e_dim,
        beta,
        z_channels,
        remap=None,
        unknown_index="random",
        sane_index_shape=False,
        legacy=True,
        use_voxel=True,
    ):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed + 1
        else:
            self.re_embed = n_e

        self.sane_index_shape = sane_index_shape

        conv_class = torch.nn.Conv3d if use_voxel else torch.nn.Conv2d
        self.quant_conv = conv_class(z_channels, self.e_dim, 1)
        self.post_quant_conv = conv_class(self.e_dim, z_channels, 1)

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        match = (inds[:, :, None] == used[None, None, ...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2) < 1
        if self.unknown_index == "random":
            new[unknown] = torch.randint(0, self.re_embed, size=new[unknown].shape).to(
                device=new.device
            )
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]:
            inds[inds >= self.used.shape[0]] = 0
        back = torch.gather(used[None, :][inds.shape[0] * [0], :], 1, inds)
        return back.reshape(ishape)

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False, is_voxel=False):
        z = self.quant_conv(z)
        z_q, loss, (perplexity, min_encodings, min_encoding_indices) = self.forward_quantizer(
            z, temp, rescale_logits, return_logits, is_voxel
        )
        z_q = self.post_quant_conv(z_q)
        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

    def forward_quantizer(
        self, z, temp=None, rescale_logits=False, return_logits=False, is_voxel=False
    ):
        assert temp is None or temp == 1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits == False, "Only for interface compatible with Gumbel"
        assert return_logits == False, "Only for interface compatible with Gumbel"

        if not is_voxel:
            z = rearrange(z, "b c h w -> b h w c").contiguous()
        else:
            z = rearrange(z, "b c d h w -> b d h w c").contiguous()
        z_flattened = z.view(-1, self.e_dim)

        d = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2
            * torch.einsum("bd,dn->bn", z_flattened, rearrange(self.embedding.weight, "n d -> d n"))
        )

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        perplexity = None
        min_encodings = None

        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + torch.mean(
                (z_q - z.detach()) ** 2
            )
        else:
            loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean(
                (z_q - z.detach()) ** 2
            )

        z_q = z + (z_q - z).detach()

        if not is_voxel:
            z_q = rearrange(z_q, "b h w c -> b c h w").contiguous()
        else:
            z_q = rearrange(z_q, "b d h w c -> b c d h w").contiguous()

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0], -1)
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1, 1)

        if self.sane_index_shape:
            if not is_voxel:
                min_encoding_indices = min_encoding_indices.reshape(
                    z_q.shape[0], z_q.shape[2], z_q.shape[3]
                )
            else:
                min_encoding_indices = min_encoding_indices.reshape(
                    z_q.shape[0], z_q.shape[2], z_q.shape[3], z_q.shape[4]
                )

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape):
        if self.remap is not None:
            indices = indices.reshape(shape[0], -1)
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1)

        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q


# ============================================================================
# model/VAE/vae_2d_resnet.py (real repo code; `.cuda()` calls in the
# inference-time `forward()` are replaced with `.to(x.device)` so the
# vendored module runs on whatever device it was invoked on -- the repo
# hardcodes `.cuda()` because it is a single-GPU training/eval-only repo)
# ============================================================================


def nonlinearity(x):
    return x * torch.sigmoid(x)


def Normalize(in_channels):
    if in_channels <= 32:
        num_groups = in_channels // 4
    else:
        num_groups = 32
    return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, 3, 1, 1)

    def forward(self, x, shape):
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        diffY = shape[0] - x.size()[2]
        diffX = shape[1] - x.size()[3]

        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, 3, 2, 1)

    def forward(self, x):
        if self.with_conv:
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool3d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(
        self, *, in_channels, out_channels=None, conv_shortcut=False, dropout, temb_channels=512
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=1
                )
            else:
                self.nin_shortcut = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, padding=0
                )

    def forward(self, x, temb=None):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
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
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)
        k = k.reshape(b, c, h * w)
        w_ = torch.bmm(q, k)
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_


@MODELS.register_module()
class VAERes2D(BaseModule):
    def __init__(
        self, encoder_cfg, decoder_cfg, num_classes=18, expansion=8, vqvae_cfg=None, init_cfg=None
    ):
        super().__init__(init_cfg)

        self.expansion = expansion
        self.num_cls = num_classes

        self.encoder = MODELS.build(encoder_cfg)
        self.decoder = MODELS.build(decoder_cfg)
        self.class_embeds = nn.Embedding(num_classes, expansion)

        if vqvae_cfg:
            self.vqvae = MODELS.build(vqvae_cfg)
        self.use_vq = vqvae_cfg is not None

    def sample_z(self, z):
        dim = z.shape[1] // 2
        mu = z[:, :dim]
        sigma = torch.exp(z[:, dim:] / 2)
        eps = torch.randn_like(mu)
        return mu + sigma * eps, mu, sigma

    def forward_encoder(self, x):
        bs, F, H, W, D = x.shape
        x = self.class_embeds(x)
        x = x.reshape(bs * F, H, W, D * self.expansion).permute(0, 3, 1, 2)

        z, shapes = self.encoder(x)
        return z, shapes

    def forward_decoder(self, z, shapes, input_shape):
        logits = self.decoder(z, shapes)

        bs, F, H, W, D = input_shape
        logits = logits.permute(0, 2, 3, 1).reshape(-1, D, self.expansion)
        template = self.class_embeds.weight.T.unsqueeze(0)
        similarity = torch.matmul(logits, template)
        return similarity.reshape(bs, F, H, W, D, self.num_cls)

    def forward(self, x, **kwargs):
        output_dict = {}
        z, shapes = self.forward_encoder(x)
        if self.use_vq:
            z_sampled, loss, info = self.vqvae(z, is_voxel=False)
            output_dict.update({"embed_loss": loss})
        else:
            z_sampled, z_mu, z_sigma = self.sample_z(z)
            output_dict.update({"z_mu": z_mu, "z_sigma": z_sigma})

        logits = self.forward_decoder(z_sampled, shapes, x.shape)

        output_dict.update({"logits": logits})

        if not self.training:
            pred = logits.argmax(dim=-1).detach().to(x.device)
            output_dict["sem_pred"] = pred
            pred_iou = deepcopy(pred)

            pred_iou[pred_iou != 17] = 1
            pred_iou[pred_iou == 17] = 0
            output_dict["iou_pred"] = pred_iou

        return output_dict

    def generate(self, z, shapes, input_shape):
        logits = self.forward_decoder(z, shapes, input_shape)
        return {"logits": logits}


@MODELS.register_module()
class Encoder2D(BaseModule):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels,
        resolution,
        z_channels,
        double_z=True,
        **ignore_kwargs,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        self.conv_in = torch.nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

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
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout
        )
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout
        )

        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(
            block_in, 2 * z_channels if double_z else z_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        shapes = []
        temb = None

        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h, temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                shapes.append(h.shape[-2:])
                h = self.down[i_level].downsample(h)

        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h, shapes


@MODELS.register_module()
class Decoder2D(BaseModule):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels,
        resolution,
        z_channels,
        give_pre_end=False,
        **ignorekwargs,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end

        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res, curr_res)

        self.conv_in = torch.nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout
        )
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout
        )

        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
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
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)

        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z, shapes):
        self.last_z_shape = z.shape

        temb = None

        h = self.conv_in(z)

        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h, shapes.pop())

        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


# ============================================================================
# model/transformer/modules.py (real repo code, verbatim)
# ============================================================================


class FFN(nn.Module):
    def __init__(self, dims, hidden_dims, act_layer=nn.GELU, drop=0.0):
        super().__init__()

        self.fc1 = nn.Linear(dims, hidden_dims)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_dims, dims)
        self.drop = nn.Dropout(p=drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# ============================================================================
# model/transformer/pose_encoder.py + pose_decoder.py (real repo code, verbatim)
# ============================================================================


@MODELS.register_module()
class PoseEncoder(BaseModule):
    def __init__(
        self, in_channels, out_channels, num_layers=2, num_modes=3, num_fut_ts=1, init_cfg=None
    ):
        super().__init__(init_cfg)
        self.num_modes = num_modes
        self.num_fut_ts = num_fut_ts
        assert num_fut_ts == 1

        pose_encoder = []

        for _ in range(num_layers - 1):
            pose_encoder.extend([nn.Linear(in_channels, out_channels), nn.ReLU(True)])
            in_channels = out_channels
        pose_encoder.append(nn.Linear(out_channels, out_channels))
        self.pose_encoder = nn.Sequential(*pose_encoder)

    def forward(self, x):
        pose_feat = self.pose_encoder(x)
        return pose_feat


@MODELS.register_module()
class PoseDecoder(BaseModule):
    def __init__(self, in_channels, num_layers=2, num_modes=3, num_fut_ts=1, init_cfg=None):
        super().__init__(init_cfg)

        self.num_modes = num_modes
        self.num_fut_ts = num_fut_ts
        assert num_fut_ts == 1

        pose_decoder = []
        for _ in range(num_layers - 1):
            pose_decoder.extend([nn.Linear(in_channels, in_channels), nn.ReLU(True)])
        pose_decoder.append(nn.Linear(in_channels, num_modes * num_fut_ts * 2))
        self.pose_decoder = nn.Sequential(*pose_decoder)

    def forward(self, x):
        rel_pose = self.pose_decoder(x).reshape(*x.shape[:-1], self.num_modes, 2)
        return rel_pose


# ============================================================================
# model/transformer/PlanUtransformer.py (real repo code, verbatim)
# ============================================================================


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x


class IdentityUnetBlock(nn.Module):
    def __init__(self, shape, in_c, out_c, residual=True):
        super().__init__()
        self.ln = nn.LayerNorm(shape)
        self.conv1 = nn.Conv2d(in_c, out_c, 1, 1, 0)
        self.act = nn.ReLU()

    def forward(self, input):
        output = self.ln(input)
        output = self.conv1(output)
        output = self.act(output)
        return output


class UnetBlock(nn.Module):
    def __init__(self, shape, in_c, out_c, residual=True):
        super().__init__()
        self.ln = nn.LayerNorm(shape)
        self.conv1 = nn.Conv2d(in_c, out_c, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.act = nn.ReLU()
        self.residual = residual
        if residual:
            if in_c == out_c:
                self.shortcut = nn.Identity()
            if in_c != out_c:
                self.shortcut = nn.Conv2d(in_c, out_c, 1, 1, 0)

    def forward(self, input):
        output = self.ln(input)
        output = self.conv1(output)
        output = self.act(output)
        output = self.conv2(output)
        if self.residual:
            output = output + self.shortcut(input)
        output = self.act(output)
        return output


@MODELS.register_module()
class PlanUAutoRegTransformer(BaseModule):
    def __init__(
        self,
        num_tokens,
        num_frames,
        num_layers,
        img_shape,
        pose_shape,
        tpe_dim=10,
        output_channel=1024,
        channels=[1, 2, 3],
        ffn_dims=None,
        temporal_attn_layers=1,
        pose_attn_layers=1,
        num_heads=8,
        pose_output_channel=None,
        conditional=True,
        tokens_untouched=False,
        add_aggregate=False,
        learnable_queries=True,
        without_multiscale=False,
        without_spatial_attn=False,
        without_pose_spatial_attn=False,
        without_pose_temporal_attn=False,
        without_temporal_attn=False,
    ) -> None:
        super().__init__()
        if without_multiscale:
            assert len(channels) == 2
        self.num_tokens = num_tokens
        self.num_frames = num_frames
        self.num_layers = num_layers
        self.channels = channels
        self.learnable_queries = learnable_queries
        if self.learnable_queries:
            self.queries = nn.Embedding(img_shape[1] * img_shape[2], img_shape[0])
        self.offset = 1 if conditional else 0
        self.temporal_embeddings = nn.Embedding(num_frames + self.offset, tpe_dim)
        if self.learnable_queries:
            self.pose_queries = nn.Embedding(pose_shape[0], pose_shape[1])
        self.pose_temporal_embeddings = nn.Embedding(num_frames + self.offset, tpe_dim)
        self.pose_attn_layers = (
            pose_attn_layers if pose_attn_layers is not None else temporal_attn_layers
        )

        self.temporal_attentions_en = nn.ModuleList([])
        self.temporal_attentions_de = nn.ModuleList([])
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        self.pose_attn_en = nn.ModuleList([])
        self.pose_en = nn.ModuleList()
        self.pose_attn_de = nn.ModuleList([])
        self.pose_de = nn.ModuleList()
        self.pose_up = nn.ModuleList()

        self.downsamples = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        up_down_sample_params = dict(kernel_size=2, stride=2, padding=0)
        self.up_down_sample_params = up_down_sample_params
        self.unfold_params = dict(kernel_size=[1, 2], stride=[1, 2], padding=[0, 0])

        C, H, W = img_shape
        layers = len(channels)
        Hs = [H]
        Ws = [W]
        cH = H
        cW = W
        for _ in range(layers - 1):
            cH = (cH) // 2
            cW = (cW) // 2
            Hs.append(cH)
            Ws.append(cW)
        pre_c = C
        for channel, cH, cW in zip(channels[0:-1], Hs[0:-1], Ws[0:-1]):
            temporal_attn_layer = nn.ModuleList()
            for i in range(temporal_attn_layers):
                if without_temporal_attn:
                    temporal_attn_layer.append(
                        nn.ModuleList(
                            [Identity(), nn.LayerNorm(pre_c), Identity(), nn.LayerNorm(pre_c)]
                        )
                    )
                else:
                    temporal_attn_layer.append(
                        nn.ModuleList(
                            [
                                nn.MultiheadAttention(pre_c, num_heads, batch_first=True),
                                nn.LayerNorm(pre_c),
                                FFN(pre_c, pre_c * 4),
                                nn.LayerNorm(pre_c),
                            ]
                        )
                    )
            self.temporal_attentions_en.append(temporal_attn_layer)
            if without_spatial_attn:
                self.encoders.append(
                    nn.Sequential(
                        IdentityUnetBlock((cH, cW), pre_c, channel),
                        IdentityUnetBlock((cH, cW), channel, channel),
                    )
                )
            else:
                self.encoders.append(
                    nn.Sequential(
                        UnetBlock((cH, cW), pre_c, channel, True),
                        UnetBlock((cH, cW), channel, channel, True),
                    )
                )
            if without_multiscale:
                self.downsamples.append(Identity())
            else:
                self.downsamples.append(nn.Conv2d(channel, channel, **up_down_sample_params))
            self.pose_en.append(
                nn.Sequential(
                    nn.Linear(pre_c, channel), nn.ReLU(), nn.Linear(channel, channel), nn.ReLU()
                )
            )
            pose_attn_layer = nn.ModuleList()
            for i in range(pose_attn_layers):
                if without_pose_temporal_attn:
                    pose_attn_layer.append(
                        nn.ModuleList(
                            [
                                Identity(),
                                nn.LayerNorm(pre_c),
                                nn.MultiheadAttention(pre_c, num_heads, batch_first=True),
                                nn.LayerNorm(pre_c),
                                FFN(pre_c, pre_c * 4),
                                nn.LayerNorm(pre_c),
                            ]
                        )
                    )
                elif without_pose_spatial_attn:
                    pose_attn_layer.append(
                        nn.ModuleList(
                            [
                                nn.MultiheadAttention(pre_c, num_heads, batch_first=True),
                                nn.LayerNorm(pre_c),
                                Identity(),
                                nn.LayerNorm(pre_c),
                                FFN(pre_c, pre_c * 4),
                                nn.LayerNorm(pre_c),
                            ]
                        )
                    )
                else:
                    pose_attn_layer.append(
                        nn.ModuleList(
                            [
                                nn.MultiheadAttention(pre_c, num_heads, batch_first=True),
                                nn.LayerNorm(pre_c),
                                nn.MultiheadAttention(pre_c, num_heads, batch_first=True),
                                nn.LayerNorm(pre_c),
                                FFN(pre_c, pre_c * 4),
                                nn.LayerNorm(pre_c),
                            ]
                        )
                    )
            self.pose_attn_en.append(pose_attn_layer)
            pre_c = channel
        channel = channels[-1]
        if without_multiscale:
            if without_spatial_attn:
                self.mid = nn.Sequential(
                    IdentityUnetBlock((pre_c, Hs[0], Ws[0]), pre_c, channel, True),
                    IdentityUnetBlock((channel, Hs[0], Ws[0]), channel, channel, True),
                )
            else:
                self.mid = nn.Sequential(
                    UnetBlock((pre_c, Hs[0], Ws[0]), pre_c, channel, True),
                    UnetBlock((channel, Hs[0], Ws[0]), channel, channel, True),
                )
        else:
            self.mid = nn.Sequential(
                UnetBlock((pre_c, Hs[-1], Ws[-1]), pre_c, channel, True),
                UnetBlock((channel, Hs[-1], Ws[-1]), channel, channel, True),
            )
        self.pose_mid = nn.Sequential(
            nn.Linear(pre_c, channel), nn.ReLU(), nn.Linear(channel, channel), nn.ReLU()
        )
        pre_c = channel
        for channel, cH, cW in zip(channels[-2::-1], Hs[-2::-1], Ws[-2::-1]):
            channel_agg = channel if add_aggregate else channel * 2
            if without_multiscale:
                self.upsamples.append(Identity())
            else:
                self.upsamples.append(nn.ConvTranspose2d(pre_c, channel, **up_down_sample_params))
            temporal_attn_layer = nn.ModuleList()
            for i in range(temporal_attn_layers):
                if without_temporal_attn:
                    temporal_attn_layer.append(
                        nn.ModuleList(
                            [
                                Identity(),
                                nn.LayerNorm(channel_agg),
                                Identity(),
                                nn.LayerNorm(channel_agg),
                            ]
                        )
                    )
                else:
                    temporal_attn_layer.append(
                        nn.ModuleList(
                            [
                                nn.MultiheadAttention(channel_agg, num_heads, batch_first=True),
                                nn.LayerNorm(channel_agg),
                                FFN(channel_agg, channel_agg * 4),
                                nn.LayerNorm(channel_agg),
                            ]
                        )
                    )
            self.temporal_attentions_de.append(temporal_attn_layer)
            if without_spatial_attn:
                self.decoders.append(
                    nn.Sequential(
                        IdentityUnetBlock((channel_agg, cH, cW), channel_agg, channel, True),
                        IdentityUnetBlock((channel, cH, cW), channel, channel, True),
                    )
                )
            else:
                self.decoders.append(
                    nn.Sequential(
                        UnetBlock((channel_agg, cH, cW), channel_agg, channel, True),
                        UnetBlock((channel, cH, cW), channel, channel, True),
                    )
                )
            pose_attn_layer = nn.ModuleList()
            self.pose_up.append(nn.Linear(pre_c, channel))
            for i in range(pose_attn_layers):
                if without_pose_temporal_attn:
                    pose_attn_layer.append(
                        nn.ModuleList(
                            [
                                Identity(),
                                nn.LayerNorm(channel_agg),
                                nn.MultiheadAttention(channel_agg, num_heads, batch_first=True),
                                nn.LayerNorm(channel_agg),
                                FFN(channel_agg, channel_agg * 4),
                                nn.LayerNorm(channel_agg),
                            ]
                        )
                    )
                elif without_pose_spatial_attn:
                    pose_attn_layer.append(
                        nn.ModuleList(
                            [
                                nn.MultiheadAttention(channel_agg, num_heads, batch_first=True),
                                nn.LayerNorm(channel_agg),
                                Identity(),
                                nn.LayerNorm(channel_agg),
                                FFN(channel_agg, channel_agg * 4),
                                nn.LayerNorm(channel_agg),
                            ]
                        )
                    )
                else:
                    pose_attn_layer.append(
                        nn.ModuleList(
                            [
                                nn.MultiheadAttention(channel_agg, num_heads, batch_first=True),
                                nn.LayerNorm(channel_agg),
                                nn.MultiheadAttention(channel_agg, num_heads, batch_first=True),
                                nn.LayerNorm(channel_agg),
                                FFN(channel_agg, channel_agg * 4),
                                nn.LayerNorm(channel_agg),
                            ]
                        )
                    )
            self.pose_attn_de.append(pose_attn_layer)
            self.pose_de.append(
                nn.Sequential(
                    nn.Linear(channel_agg, channel),
                    nn.ReLU(),
                    nn.Linear(channel, channel),
                    nn.ReLU(),
                )
            )
            pre_c = channel

        self.conv_out = nn.Conv2d(pre_c, output_channel, 3, 1, 1)

        self.pose_out = nn.Linear(
            pre_c, pose_output_channel if pose_output_channel is not None else output_channel
        )

        self.tokens_untouched = tokens_untouched

        if tokens_untouched:
            assert all([ch == channels[0] for ch in channels])
            for scale in range(len(channels) - 1):
                num_tokens = self.unfold_params["kernel_size"][scale] ** 2
                attn_mask = torch.zeros(num_frames, num_frames * num_tokens, dtype=torch.bool)
                for i_frame in range(num_frames):
                    start = (
                        i_frame * num_tokens + num_tokens if conditional else i_frame * num_tokens
                    )
                    attn_mask[i_frame, start:] = True
                self.register_buffer(f"attn_mask_{scale}", attn_mask, False)
        else:
            attn_mask = torch.zeros(
                num_frames * num_tokens, num_frames * num_tokens, dtype=torch.bool
            )
            for i_frame in range(num_frames):
                start1 = i_frame * num_tokens
                start2 = start1 + num_tokens if conditional else start1
                attn_mask[start1 : (start1 + num_tokens), start2:] = True
            self.register_buffer("attn_mask", attn_mask, False)

    def forward(self, tokens, pose_tokens):
        # tokens: bs, f, c, h, w
        # pose_tokens, bs, f, c
        bs, F, C, H, W = tokens.shape
        assert F == self.num_frames
        tokens = rearrange(tokens, "b f c h w -> b f h w c")
        if self.learnable_queries:
            queries = self.queries.weight.reshape(1, 1, H, W, C).expand(bs, F, H, W, C)
        else:
            queries = tokens
        queries = queries + self.temporal_embeddings.weight[
            None, self.offset :, None, None, :
        ].expand(bs, -1, H, W, -1)
        tokens = tokens + self.temporal_embeddings.weight[
            None, : self.num_frames, None, None, :
        ].expand(bs, -1, H, W, -1)

        if self.learnable_queries:
            pose_queries = self.pose_queries.weight.reshape(1, 1, C).expand(bs, F, C)
        else:
            pose_queries = pose_tokens
        pose_queries = pose_queries + self.pose_temporal_embeddings.weight[
            None, self.offset :, :
        ].expand(bs, -1, -1)
        pose_tokens = pose_tokens + self.pose_temporal_embeddings.weight[
            None, : self.num_frames, :
        ].expand(bs, -1, -1)

        encoder_outs_tokens = []
        encoder_outs_queries = []
        encoder_outs_pose_tokens = []
        encoder_outs_pose_queries = []

        for temporal_attn, encoder, down, pose_attn_en, pose_en in zip(
            self.temporal_attentions_en,
            self.encoders,
            self.downsamples,
            self.pose_attn_en,
            self.pose_en,
        ):
            b, f, h, w, c = tokens.shape

            for (
                pose_temporal_attn,
                pose_temporal_norm,
                spatial_attn,
                spatial_norm,
                ffn,
                ffn_norm,
            ) in pose_attn_en:
                pose_queries = (
                    pose_queries
                    + pose_temporal_attn(
                        pose_queries,
                        pose_tokens,
                        pose_tokens,
                        need_weights=False,
                        attn_mask=self.attn_mask,
                    )[0]
                )
                pose_queries = pose_temporal_norm(pose_queries)
                pose_queries = rearrange(pose_queries, "b f c -> (b f) 1 c")
                queries = rearrange(queries, "b f h w c -> (b f) (h w) c")
                pose_queries = (
                    pose_queries
                    + spatial_attn(
                        pose_queries, queries, queries, need_weights=False, attn_mask=None
                    )[0]
                )
                pose_queries = spatial_norm(pose_queries)

                pose_queries = pose_queries + ffn(pose_queries)
                pose_queries = ffn_norm(pose_queries)
                pose_queries = rearrange(pose_queries, "(b f) 1 c -> b f c", b=b, f=f)
                queries = rearrange(queries, "(b f) (h w) c -> b f h w c", b=b, f=f, h=h, w=w)

            pose_queries = pose_en(pose_queries)
            pose_tokens = pose_en(pose_tokens)
            encoder_outs_pose_queries.append(pose_queries)
            encoder_outs_pose_tokens.append(pose_tokens)

            queries = rearrange(queries, "b f h w c -> (b h w) f c")
            tokens = rearrange(tokens, "b f h w c -> (b h w) f c")
            for cross_attn, cross_norm, ffn, ffn_norm in temporal_attn:
                queries = (
                    queries
                    + cross_attn(
                        queries, tokens, tokens, need_weights=False, attn_mask=self.attn_mask
                    )[0]
                )
                queries = cross_norm(queries)

                queries = queries + ffn(queries)
                queries = ffn_norm(queries)

            queries = rearrange(queries, "(b h w) f c -> (b f) c h w", b=b, h=h, w=w)
            tokens = rearrange(tokens, "(b h w) f c -> (b f) c h w", b=b, h=h, w=w)
            queries = encoder(queries)
            tokens = encoder(tokens)
            encoder_outs_tokens.append(tokens)
            encoder_outs_queries.append(queries)
            queries = down(queries)
            tokens = down(tokens)
            queries = rearrange(queries, "(b f) c h w -> b f h w c", b=b, f=f)
            tokens = rearrange(tokens, "(b f) c h w -> b f h w c", b=b, f=f)
        b, f, h, w, c = queries.shape
        queries = rearrange(queries, "b f h w c -> (b f) c h w")
        tokens = rearrange(tokens, "b f h w c -> (b f) c h w")
        queries = self.mid(queries)
        tokens = self.mid(tokens)

        pose_queries = self.pose_mid(pose_queries)
        pose_tokens = self.pose_mid(pose_tokens)

        for (
            temporal_attn,
            decoder,
            up,
            encoder_out_queries,
            encoder_out_tokens,
            pose_attn_de,
            pose_de_,
            encoder_out_pose_queries,
            encoder_out_pose_tokens,
            pose_up,
        ) in zip(
            self.temporal_attentions_de,
            self.decoders,
            self.upsamples,
            encoder_outs_queries[::-1],
            encoder_outs_tokens[::-1],
            self.pose_attn_de,
            self.pose_de,
            encoder_outs_pose_queries[::-1],
            encoder_outs_pose_tokens[::-1],
            self.pose_up,
        ):
            queries = up(queries)
            tokens = up(tokens)

            pad_x_queries = encoder_out_queries.shape[2] - queries.shape[2]
            pad_y_queries = encoder_out_queries.shape[3] - queries.shape[3]
            queries = nn.functional.pad(
                queries,
                (
                    pad_x_queries // 2,
                    pad_x_queries - pad_x_queries // 2,
                    pad_y_queries // 2,
                    pad_y_queries - pad_y_queries // 2,
                ),
            )
            pad_x_tokens = encoder_out_tokens.shape[2] - tokens.shape[2]
            pad_y_tokens = encoder_out_tokens.shape[3] - tokens.shape[3]
            tokens = nn.functional.pad(
                tokens,
                (
                    pad_x_tokens // 2,
                    pad_x_tokens - pad_x_tokens // 2,
                    pad_y_tokens // 2,
                    pad_y_tokens - pad_y_tokens // 2,
                ),
            )
            queries = torch.cat([queries, encoder_out_queries], dim=1)
            tokens = torch.cat([tokens, encoder_out_tokens], dim=1)
            c, h, w = queries.shape[-3:]
            queries = rearrange(queries, "(b f) c h w -> (b h w) f c", b=b, f=f)
            tokens = rearrange(tokens, "(b f) c h w -> (b h w) f c", b=b, f=f)
            for cross_attn, cross_norm, ffn, ffn_norm in temporal_attn:
                queries = (
                    queries
                    + cross_attn(
                        queries, tokens, tokens, need_weights=False, attn_mask=self.attn_mask
                    )[0]
                )
                queries = cross_norm(queries)

                queries = queries + ffn(queries)
                queries = ffn_norm(queries)
            queries = rearrange(queries, "(b h w) f c -> (b f) c h w", b=b, h=h, w=w)
            tokens = rearrange(tokens, "(b h w) f c -> (b f) c h w", b=b, h=h, w=w)

            pose_queries = pose_up(pose_queries)
            pose_tokens = pose_up(pose_tokens)
            pose_queries = torch.cat([pose_queries, encoder_out_pose_queries], dim=2)
            pose_tokens = torch.cat([pose_tokens, encoder_out_pose_tokens], dim=2)

            for (
                pose_temporal_attn,
                pose_temporal_norm,
                spatial_attn,
                spatial_norm,
                ffn,
                ffn_norm,
            ) in pose_attn_de:
                pose_queries = (
                    pose_queries
                    + pose_temporal_attn(
                        pose_queries,
                        pose_tokens,
                        pose_tokens,
                        need_weights=False,
                        attn_mask=self.attn_mask,
                    )[0]
                )
                pose_queries = pose_temporal_norm(pose_queries)
                pose_queries = rearrange(pose_queries, "b f c -> (b f) 1 c")
                queries = rearrange(queries, "(b f) c h w -> (b f) (h w) c", b=b, f=f, h=h, w=w)
                pose_queries = (
                    pose_queries
                    + spatial_attn(
                        pose_queries, queries, queries, need_weights=False, attn_mask=None
                    )[0]
                )
                pose_queries = spatial_norm(pose_queries)

                pose_queries = pose_queries + ffn(pose_queries)
                pose_queries = ffn_norm(pose_queries)
                queries = rearrange(queries, "(b f) (h w) c -> (b f) c h w", b=b, f=f, h=h, w=w)
                pose_queries = rearrange(pose_queries, "(b f) 1 c -> b f c", b=b, f=f)
            pose_queries = pose_de_(pose_queries)
            pose_tokens = pose_de_(pose_tokens)
            queries = decoder(queries)
            tokens = decoder(tokens)

        queries = self.conv_out(queries)
        pose_queries = self.pose_out(pose_queries)
        queries = rearrange(queries, "(b f) c h w -> b f c h w", b=b, f=f)

        return queries, pose_queries


# ============================================================================
# model/TransVQVAE.py (real repo code; the two inference-time `.cuda()`
# calls on the argmax'd semantic prediction are replaced with `.to(x.device)`
# for the same reason as VAERes2D above. `forward_train_with_plan` /
# `forward_inference_with_plan` are the pose-conditioned paths used by this
# staging module.)
# ============================================================================


@MODELS.register_module()
class TransVQVAE(BaseModule):
    def __init__(
        self,
        vae,
        transformer,
        num_frames=10,
        offset=1,
        pose_encoder=None,
        pose_decoder=None,
        pose_actor=None,
        give_hiddens=False,
        delta_input=False,
        without_all=False,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.offset = offset
        self.vae = MODELS.build(vae)
        self.transformer = MODELS.build(transformer)
        if pose_encoder is not None:
            self.pose_encoder = MODELS.build(pose_encoder)
        if pose_decoder is not None:
            self.pose_decoder = MODELS.build(pose_decoder)
        if pose_actor is not None:
            self.pose_actor = MODELS.build(pose_actor)
        self.give_hiddens = give_hiddens
        self.delta_input = delta_input
        self.planning_metric = None
        self.without_all = without_all

    def forward(self, x, metas=None):
        if hasattr(self, "pose_encoder"):
            if self.training:
                return self.forward_train_with_plan(x, metas)
            else:
                return self.forward_inference_with_plan(x, metas)
        if self.training:
            return self.forward_train(x)
        else:
            return self.forward_inference(x)

    def forward_train(self, x):
        assert hasattr(self.vae, "vqvae")
        bs, F, H, W, D = x.shape
        assert F == self.num_frames + self.offset
        output_dict = {}
        z, shape = self.vae.forward_encoder(x)
        z = self.vae.vqvae.quant_conv(z)
        z_q, loss, (perplexity, min_encodings, min_encoding_indices) = (
            self.vae.vqvae.forward_quantizer(z, is_voxel=False)
        )
        min_encoding_indices = rearrange(min_encoding_indices, "(b f) h w -> b f h w", b=bs)
        output_dict["ce_labels"] = min_encoding_indices[:, self.offset :].detach().flatten(0, 1)
        z_q = rearrange(z_q, "(b f) c h w -> b f c h w", b=bs)
        hidden = None
        if self.give_hiddens:
            hidden = z_q[:, : self.offset]
        z_q_predict = self.transformer(z_q[:, : self.num_frames], hidden=hidden)
        z_q_predict = z_q_predict.flatten(0, 1)
        output_dict["ce_inputs"] = z_q_predict
        return output_dict

    def forward_inference(self, x):
        bs, F, H, W, D = x.shape
        output_dict = {}
        output_dict["target_occs"] = x[:, self.offset :]
        z, shape = self.vae.forward_encoder(x)
        z = self.vae.vqvae.quant_conv(z)
        z_q, loss, (perplexity, min_encodings, min_encoding_indices) = (
            self.vae.vqvae.forward_quantizer(z, is_voxel=False)
        )
        min_encoding_indices = rearrange(min_encoding_indices, "(b f) h w -> b f h w", b=bs)
        output_dict["ce_labels"] = min_encoding_indices[:, self.offset :].detach().flatten(0, 1)
        z_q = rearrange(z_q, "(b f) c h w -> b f c h w", b=bs)
        hidden = None
        if self.give_hiddens:
            hidden = z_q[:, : self.offset]
        z_q_predict = self.transformer(z_q[:, : self.num_frames], hidden=hidden)
        z_q_predict = z_q_predict.flatten(0, 1)
        output_dict["ce_inputs"] = z_q_predict
        z_q_predict = z_q_predict.argmax(dim=1)
        z_q_predict = self.vae.vqvae.get_codebook_entry(z_q_predict, shape=None)
        z_q_predict = rearrange(z_q_predict, "bf h w c-> bf c h w")
        z_q_predict = self.vae.vqvae.post_quant_conv(z_q_predict)

        z_q_predict = self.vae.forward_decoder(z_q_predict, shape, output_dict["target_occs"].shape)
        output_dict["logits"] = z_q_predict
        pred = z_q_predict.argmax(dim=-1).detach().to(x.device)
        output_dict["sem_pred"] = pred
        pred_iou = deepcopy(pred)

        pred_iou[pred_iou != 17] = 1
        pred_iou[pred_iou == 17] = 0
        output_dict["iou_pred"] = pred_iou

        return output_dict

    def forward_train_with_plan(self, x, metas):
        assert hasattr(self.vae, "vqvae")
        assert hasattr(self, "pose_encoder")
        bs, F, H, W, D = x.shape
        assert F == self.num_frames + self.offset
        output_dict = {}
        z, shape = self.vae.forward_encoder(x)
        z = self.vae.vqvae.quant_conv(z)
        z_q, loss, (perplexity, min_encodings, min_encoding_indices) = (
            self.vae.vqvae.forward_quantizer(z, is_voxel=False)
        )
        min_encoding_indices = rearrange(min_encoding_indices, "(b f) h w -> b f h w", b=bs)
        output_dict["ce_labels"] = min_encoding_indices[:, self.offset :].detach().flatten(0, 1)
        z_q = rearrange(z_q, "(b f) c h w -> b f c h w", b=bs)
        hidden = None
        if self.give_hiddens:
            hidden = z_q[:, : self.offset]

        rel_poses, output_metas = self._get_pose_feature(metas, F - self.offset)

        z_q_predict, rel_poses = self.transformer(z_q[:, : self.num_frames], pose_tokens=rel_poses)

        pose_decoded = self.pose_decoder(rel_poses)
        output_dict["pose_decoded"] = pose_decoded
        output_dict["output_metas"] = output_metas

        z_q_predict = z_q_predict.flatten(0, 1)
        output_dict["ce_inputs"] = z_q_predict
        return output_dict

    def forward_inference_with_plan(self, x, metas):
        bs, F, H, W, D = x.shape
        output_dict = {}
        output_dict["target_occs"] = x[:, self.offset :]
        z, shape = self.vae.forward_encoder(x)
        z = self.vae.vqvae.quant_conv(z)
        z_q, loss, (perplexity, min_encodings, min_encoding_indices) = (
            self.vae.vqvae.forward_quantizer(z, is_voxel=False)
        )
        min_encoding_indices = rearrange(min_encoding_indices, "(b f) h w -> b f h w", b=bs)
        output_dict["ce_labels"] = min_encoding_indices[:, self.offset :].detach().flatten(0, 1)
        z_q = rearrange(z_q, "(b f) c h w -> b f c h w", b=bs)
        hidden = None
        if self.give_hiddens:
            hidden = z_q[:, : self.offset]

        rel_poses, output_metas = self._get_pose_feature(metas, F - self.offset)

        z_q_predict, rel_poses = self.transformer(z_q[:, : self.num_frames], pose_tokens=rel_poses)

        pose_decoded = self.pose_decoder(rel_poses)
        output_dict["pose_decoded"] = pose_decoded
        output_dict["output_metas"] = output_metas

        z_q_predict = z_q_predict.flatten(0, 1)
        output_dict["ce_inputs"] = z_q_predict
        z_q_predict = z_q_predict.argmax(dim=1)
        z_q_predict = self.vae.vqvae.get_codebook_entry(z_q_predict, shape=None)
        z_q_predict = rearrange(z_q_predict, "bf h w c-> bf c h w")
        z_q_predict = self.vae.vqvae.post_quant_conv(z_q_predict)

        z_q_predict = self.vae.forward_decoder(z_q_predict, shape, output_dict["target_occs"].shape)
        output_dict["logits"] = z_q_predict
        pred = z_q_predict.argmax(dim=-1).detach().to(x.device)
        output_dict["sem_pred"] = pred
        pred_iou = deepcopy(pred)

        pred_iou[pred_iou != 17] = 1
        pred_iou[pred_iou == 17] = 0
        output_dict["iou_pred"] = pred_iou

        return output_dict

    def _get_pose_feature(self, metas=None, F=None):
        rel_poses, output_metas = None, None
        if hasattr(self, "pose_encoder"):
            assert hasattr(self, "pose_decoder")
            assert metas is not None
            output_metas = []
            for meta in metas:
                output_meta = dict()
                output_meta["rel_poses"] = meta["rel_poses"][self.offset :]
                output_meta["gt_mode"] = meta["gt_mode"][self.offset :]
                output_metas.append(output_meta)

            rel_poses = np.array([meta["rel_poses"] for meta in metas])
            gt_mode = np.array([meta["gt_mode"] for meta in metas])

            gt_mode = torch.tensor(gt_mode)
            rel_poses = torch.tensor(rel_poses)
            if self.delta_input:
                rel_poses_pre = torch.cat(
                    [torch.zeros_like(rel_poses[:, :1]), rel_poses[:, :-1]], dim=1
                )
                rel_poses = rel_poses - rel_poses_pre
            if F > self.num_frames:
                assert F == self.num_frames + self.offset
            else:
                assert F == self.num_frames
                gt_mode = gt_mode[:, : -self.offset, :]
                rel_poses = rel_poses[:, : -self.offset, :]

            rel_poses = torch.cat([rel_poses, gt_mode], dim=-1)
            rel_poses = self.pose_encoder(rel_poses.float())
        return rel_poses, output_metas


# ============================================================================
# staging harness: tiny config + synthetic-but-shape-correct input
# ============================================================================
# Mirrors config/train_occworld.py's `model = dict(type='TransVQVAE', ...)`
# structure exactly, at drastically reduced scale: resolution 16 (vs 200),
# base_channel 4 (vs 64), 2-level encoder/decoder (vs 3), n_frames=3 (vs 15),
# tiny codebook (vs 512 entries). Every mechanism (VQ-VAE compression,
# multi-scale U-Net temporal cross-attention, pose token fusion) is exercised.


def _tiny_occworld_config():
    base_channel = 4
    _dim_ = 4
    expansion = 2
    n_e_ = 16
    num_frames = 2  # excludes the +offset conditioning frame
    offset = 1
    resolution = 16
    return dict(
        model=dict(
            type="TransVQVAE",
            num_frames=num_frames,
            delta_input=False,
            offset=offset,
            vae=dict(
                type="VAERes2D",
                encoder_cfg=dict(
                    type="Encoder2D",
                    ch=base_channel,
                    out_ch=base_channel,
                    ch_mult=(1, 2),
                    num_res_blocks=1,
                    attn_resolutions=(),
                    dropout=0.0,
                    resamp_with_conv=True,
                    in_channels=_dim_ * expansion,
                    resolution=resolution,
                    z_channels=base_channel * 2,
                    double_z=False,
                ),
                decoder_cfg=dict(
                    type="Decoder2D",
                    ch=base_channel,
                    out_ch=_dim_ * expansion,
                    ch_mult=(1, 2),
                    num_res_blocks=1,
                    attn_resolutions=(),
                    dropout=0.0,
                    resamp_with_conv=True,
                    in_channels=_dim_ * expansion,
                    resolution=resolution,
                    z_channels=base_channel * 2,
                    give_pre_end=False,
                ),
                num_classes=18,
                expansion=expansion,
                vqvae_cfg=dict(
                    type="VectorQuantizer",
                    sane_index_shape=True,
                    n_e=n_e_,
                    e_dim=base_channel * 2,
                    beta=1.0,
                    z_channels=base_channel * 2,
                    use_voxel=False,
                ),
            ),
            transformer=dict(
                type="PlanUAutoRegTransformer",
                num_tokens=1,
                num_frames=num_frames,
                num_layers=1,
                img_shape=(base_channel * 2, resolution // 2, resolution // 2),
                pose_shape=(1, base_channel * 2),
                pose_attn_layers=1,
                pose_output_channel=base_channel * 2,
                tpe_dim=base_channel * 2,
                channels=(base_channel * 2, base_channel * 4),
                temporal_attn_layers=1,
                output_channel=n_e_,
                learnable_queries=False,
            ),
            pose_encoder=dict(
                type="PoseEncoder",
                in_channels=5,
                out_channels=base_channel * 2,
                num_layers=2,
                num_modes=3,
                num_fut_ts=1,
            ),
            pose_decoder=dict(
                type="PoseDecoder",
                in_channels=base_channel * 2,
                num_layers=2,
                num_modes=3,
                num_fut_ts=1,
            ),
        ),
        base_channel=base_channel,
        _dim_=_dim_,
        expansion=expansion,
        num_frames=num_frames,
        offset=offset,
        resolution=resolution,
    )


def build_occworld():
    torch.manual_seed(0)
    cfg = _tiny_occworld_config()
    model = MODELS.build(cfg["model"])
    model._staged_cfg = cfg
    model.eval()  # forward_inference_with_plan path (exercises the argmax/codebook-lookup branch)
    return model


def example_input_occworld():
    torch.manual_seed(0)
    cfg = _tiny_occworld_config()
    dim = cfg["_dim_"]
    res = cfg["resolution"]
    num_frames = cfg["num_frames"]
    offset = cfg["offset"]
    total_frames = num_frames + offset
    bs = 1

    # x: bs, F, H, W, D voxel occupancy class-index grid (class_embeds lookup)
    x = torch.randint(0, 18, (bs, total_frames, res, res, dim), dtype=torch.long)

    # rel_poses: per-frame (x, y) ego displacement, shape (F, 2)
    # gt_mode: per-frame one-hot over 3 discrete motion modes, shape (F, 3)
    # concatenated on the last dim inside `_get_pose_feature` -> (F, 5),
    # matching PoseEncoder(in_channels=5, ...) above.
    metas = [
        {
            "rel_poses": np.random.randn(total_frames, 2).astype(np.float32),
            "gt_mode": np.eye(3, dtype=np.float32)[np.random.randint(0, 3, size=(total_frames,))],
        }
        for _ in range(bs)
    ]
    return (x, metas)


MENAGERIE_ENTRIES = [
    ("occworld", "build_occworld", "example_input_occworld", 2024, "vendored-pytorch"),
]
