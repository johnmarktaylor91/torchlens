# SOURCE: vendored from YtongXie/CoTr @ 521b4cc4cc6128c0f0e08057b0b15e5c1da7cad4
# (CoTr_package/CoTr/network_architecture/{ResTranUnet.py,CNNBackbone.py,
#  DeTrans/{DeformableTrans.py,position_encoding.py,ops/functions/ms_deform_attn_func.py,
#  ops/modules/ms_deform_attn.py}})
"""CoTr: efficiently bridging CNN and Transformer for 3D medical image segmentation
(Xie et al., MICCAI 2021). A 3D ResNet-style CNN backbone extracts multi-scale volumetric
features; the two deepest scales are flattened, position-encoded, and fed into a
multi-scale 3D deformable-attention Transformer encoder (an extension of Deformable DETR's
MSDeformAttn to 3D volumes); the transformer-refined features are then upsampled back
through a U-Net-style decoder with skip connections to a per-voxel segmentation head.

Vendored verbatim from the real repo's ``network_architecture`` package (the actual model
definition files). Two mechanical (non-architectural) fixes were required to run outside
the original nnU-Net-trainer harness:

1. ``ResTranUnet`` inherited from nnU-Net's ``SegmentationNetwork`` (in
   ``CoTr/network_architecture/neural_network.py``), whose module import chain requires
   ``batchgenerators``/``nnunet`` purely for unused sliding-window-inference helper methods
   that this vendored copy never calls. ``ResTranUnet`` here inherits directly from
   ``nn.Module`` instead; the constructor and ``forward`` logic are unchanged from the
   original.
2. ``posi_mask`` (in ``U_ResTran3D``) and ``PositionEmbeddingSine.forward`` hardcoded
   ``.cuda()`` on the boolean mask tensor. Both are changed to ``.to(x.device)`` /
   ``.to(fea.device)`` so the model runs on CPU; the mask's *values* (all-``False``, i.e. no
   padding) and the resulting computation are identical.

The multi-scale deformable attention module ships an optional custom CUDA kernel
(``MSDeformAttnFunction``) alongside a pure-PyTorch fallback,
``ms_deform_attn_core_pytorch_3D`` (``grid_sample``-based trilinear sampling); the repo's
own ``MSDeformAttn.forward`` calls the pure-PyTorch fallback directly (not the CUDA
extension), so no custom op / compiled extension is needed here.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from typing import Optional, List

from torch.nn.init import xavier_uniform_, constant_, normal_

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# from CoTr/network_architecture/DeTrans/ops/functions/ms_deform_attn_func.py
# ---------------------------------------------------------------------------
def ms_deform_attn_core_pytorch_3D(
    value, value_spatial_shapes, sampling_locations, attention_weights
):
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    value_list = value.split([T_ * H_ * W_ for T_, H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (T_, H_, W_) in enumerate(value_spatial_shapes):
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_ * M_, D_, T_, H_, W_)
        sampling_grid_l_ = (
            sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)[:, None, :, :, :]
        )
        sampling_value_l_ = F.grid_sample(
            value_l_,
            sampling_grid_l_.to(dtype=value_l_.dtype),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )[:, :, 0]
        sampling_value_list.append(sampling_value_l_)
    attention_weights = attention_weights.transpose(1, 2).reshape(N_ * M_, 1, Lq_, L_ * P_)
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .view(N_, M_ * D_, Lq_)
    )
    return output.transpose(1, 2).contiguous()


# ---------------------------------------------------------------------------
# from CoTr/network_architecture/DeTrans/ops/modules/ms_deform_attn.py
# ---------------------------------------------------------------------------
class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                "d_model must be divisible by n_heads, but got {} and {}".format(d_model, n_heads)
            )
        _d_per_head = d_model // n_heads

        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 3)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack(
            [thetas.cos(), thetas.sin() * thetas.cos(), thetas.sin() * thetas.sin()], -1
        )
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.n_heads, 1, 1, 3)
            .repeat(1, self.n_levels, self.n_points, 1)
        )
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.0)
        constant_(self.attention_weights.bias.data, 0.0)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.0)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.0)

    def forward(
        self,
        query,
        reference_points,
        input_flatten,
        input_spatial_shapes,
        input_level_start_index,
        input_padding_mask=None,
    ):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 3)
        :param input_flatten               (N, sum_l D_l*H_l*W_l, C)
        :param input_spatial_shapes        (n_levels, 3), [(D_0, H_0, W_0), ...]
        :param input_level_start_index     (n_levels, )
        :param input_padding_mask          (N, sum_l D_l*H_l*W_l), True for padding elements

        :return output                     (N, Length_{query}, C)
        """
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (
            input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1] * input_spatial_shapes[:, 2]
        ).sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(
            N, Len_q, self.n_heads, self.n_levels, self.n_points, 3
        )
        attention_weights = self.attention_weights(query).view(
            N, Len_q, self.n_heads, self.n_levels * self.n_points
        )
        attention_weights = F.softmax(attention_weights, -1).view(
            N, Len_q, self.n_heads, self.n_levels, self.n_points
        )

        if reference_points.shape[-1] == 3:
            offset_normalizer = torch.stack(
                [
                    input_spatial_shapes[..., 0],
                    input_spatial_shapes[..., 2],
                    input_spatial_shapes[..., 1],
                ],
                -1,
            )
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )

        output = ms_deform_attn_core_pytorch_3D(
            value, input_spatial_shapes, sampling_locations, attention_weights
        )

        output = self.output_proj(output)
        return output


# ---------------------------------------------------------------------------
# from CoTr/network_architecture/DeTrans/position_encoding.py
# ---------------------------------------------------------------------------
class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=[64, 64, 64], temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        bs, c, d, h, w = x.shape
        mask = torch.zeros(bs, d, h, w, dtype=torch.bool).to(x.device)  # menagerie: was .cuda()
        assert mask is not None
        not_mask = ~mask
        d_embed = not_mask.cumsum(1, dtype=torch.float32)
        y_embed = not_mask.cumsum(2, dtype=torch.float32)
        x_embed = not_mask.cumsum(3, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            d_embed = (d_embed - 0.5) / (d_embed[:, -1:, :, :] + eps) * self.scale
            y_embed = (y_embed - 0.5) / (y_embed[:, :, -1:, :] + eps) * self.scale
            x_embed = (x_embed - 0.5) / (x_embed[:, :, :, -1:] + eps) * self.scale

        dim_tx = torch.arange(self.num_pos_feats[0], dtype=torch.float32, device=x.device)
        dim_tx = self.temperature ** (3 * (dim_tx // 3) / self.num_pos_feats[0])

        dim_ty = torch.arange(self.num_pos_feats[1], dtype=torch.float32, device=x.device)
        dim_ty = self.temperature ** (3 * (dim_ty // 3) / self.num_pos_feats[1])

        dim_td = torch.arange(self.num_pos_feats[2], dtype=torch.float32, device=x.device)
        dim_td = self.temperature ** (3 * (dim_td // 3) / self.num_pos_feats[2])

        pos_x = x_embed[:, :, :, :, None] / dim_tx
        pos_y = y_embed[:, :, :, :, None] / dim_ty
        pos_d = d_embed[:, :, :, :, None] / dim_td

        pos_x = torch.stack(
            (pos_x[:, :, :, :, 0::2].sin(), pos_x[:, :, :, :, 1::2].cos()), dim=5
        ).flatten(4)
        pos_y = torch.stack(
            (pos_y[:, :, :, :, 0::2].sin(), pos_y[:, :, :, :, 1::2].cos()), dim=5
        ).flatten(4)
        pos_d = torch.stack(
            (pos_d[:, :, :, :, 0::2].sin(), pos_d[:, :, :, :, 1::2].cos()), dim=5
        ).flatten(4)

        pos = torch.cat((pos_d, pos_y, pos_x), dim=4).permute(0, 4, 1, 2, 3)
        return pos


def build_position_encoding(mode, hidden_dim):
    N_steps = hidden_dim // 3
    if (hidden_dim % 3) != 0:
        N_steps = [N_steps, N_steps, N_steps + hidden_dim % 3]
    else:
        N_steps = [N_steps, N_steps, N_steps]

    if mode in ("v2", "sine"):
        position_embedding = PositionEmbeddingSine(num_pos_feats=N_steps, normalize=True)
    else:
        raise ValueError(f"not supported {mode}")

    return position_embedding


# ---------------------------------------------------------------------------
# from CoTr/network_architecture/DeTrans/DeformableTrans.py
# ---------------------------------------------------------------------------
class DeformableTransformer(nn.Module):
    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        activation="relu",
        num_feature_levels=4,
        enc_n_points=4,
    ):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead

        encoder_layer = DeformableTransformerEncoderLayer(
            d_model, dim_feedforward, dropout, activation, num_feature_levels, nhead, enc_n_points
        )
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, D, H, W = mask.shape
        valid_D = torch.sum(~mask[:, :, 0, 0], 1)
        valid_H = torch.sum(~mask[:, 0, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, 0, :], 1)

        valid_ratio_d = valid_D.float() / D
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_d, valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds):
        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, d, h, w = src.shape
            spatial_shape = (d, h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=src_flatten.device
        )
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder
        memory = self.encoder(
            src_flatten,
            spatial_shapes,
            level_start_index,
            valid_ratios,
            lvl_pos_embed_flatten,
            mask_flatten,
        )

        return memory


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation="relu",
        n_levels=4,
        n_heads=8,
        n_points=4,
    ):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(
        self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None
    ):
        # self attention
        src2 = self.self_attn(
            self.with_pos_embed(src, pos),
            reference_points,
            src,
            spatial_shapes,
            level_start_index,
            padding_mask,
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (D_, H_, W_) in enumerate(spatial_shapes):
            ref_d, ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, D_ - 0.5, D_, dtype=torch.float32, device=device),
                torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device),
            )

            ref_d = ref_d.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * D_)
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 2] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * W_)

            ref = torch.stack((ref_d, ref_x, ref_y), -1)  # D W H
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(
        self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None
    ):
        output = src
        reference_points = self.get_reference_points(
            spatial_shapes, valid_ratios, device=src.device
        )
        for _, layer in enumerate(self.layers):
            output = layer(
                output, pos, reference_points, spatial_shapes, level_start_index, padding_mask
            )

        return output


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


# ---------------------------------------------------------------------------
# from CoTr/network_architecture/CNNBackbone.py
# ---------------------------------------------------------------------------
class Conv3d_wd(nn.Conv3d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=(1, 1, 1),
        padding=(0, 0, 0),
        dilation=(1, 1, 1),
        groups=1,
        bias=False,
    ):
        super(Conv3d_wd, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias
        )

    def forward(self, x):
        weight = self.weight
        weight_mean = (
            weight.mean(dim=1, keepdim=True)
            .mean(dim=2, keepdim=True)
            .mean(dim=3, keepdim=True)
            .mean(dim=4, keepdim=True)
        )
        weight = weight - weight_mean
        std = torch.sqrt(torch.var(weight.view(weight.size(0), -1), dim=1) + 1e-12).view(
            -1, 1, 1, 1, 1
        )
        weight = weight / std.expand_as(weight)
        return F.conv3d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def conv3x3x3(
    in_planes,
    out_planes,
    kernel_size,
    stride=(1, 1, 1),
    padding=(0, 0, 0),
    dilation=(1, 1, 1),
    groups=1,
    bias=False,
    weight_std=False,
):
    "3x3x3 convolution with padding"
    if weight_std:
        return Conv3d_wd(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
    else:
        return nn.Conv3d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )


def Norm_layer(norm_cfg, inplanes):
    if norm_cfg == "BN":
        out = nn.BatchNorm3d(inplanes)
    elif norm_cfg == "SyncBN":
        out = nn.SyncBatchNorm(inplanes)
    elif norm_cfg == "GN":
        out = nn.GroupNorm(16, inplanes)
    elif norm_cfg == "IN":
        out = nn.InstanceNorm3d(inplanes, affine=True)

    return out


def Activation_layer(activation_cfg, inplace=True):
    if activation_cfg == "ReLU":
        out = nn.ReLU(inplace=inplace)
    elif activation_cfg == "LeakyReLU":
        out = nn.LeakyReLU(negative_slope=1e-2, inplace=inplace)

    return out


class BackboneResBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        norm_cfg,
        activation_cfg,
        stride=(1, 1, 1),
        downsample=None,
        weight_std=False,
    ):
        super(BackboneResBlock, self).__init__()
        self.conv1 = conv3x3x3(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            weight_std=weight_std,
        )
        self.norm1 = Norm_layer(norm_cfg, planes)
        self.nonlin = Activation_layer(activation_cfg, inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.nonlin(out)

        return out


class Backbone(nn.Module):
    arch_settings = {9: (BackboneResBlock, (3, 3, 2))}

    def __init__(
        self, depth, in_channels=1, norm_cfg="BN", activation_cfg="ReLU", weight_std=False
    ):
        super(Backbone, self).__init__()

        if depth not in self.arch_settings:
            raise KeyError("invalid depth {} for resnet".format(depth))
        self.depth = depth
        block, layers = self.arch_settings[depth]
        self.inplanes = 64
        self.conv1 = conv3x3x3(
            in_channels,
            64,
            kernel_size=7,
            stride=(1, 2, 2),
            padding=3,
            bias=False,
            weight_std=weight_std,
        )
        self.norm1 = Norm_layer(norm_cfg, 64)
        self.nonlin = Activation_layer(activation_cfg, inplace=True)
        self.layer1 = self._make_layer(
            block,
            192,
            layers[0],
            stride=(2, 2, 2),
            norm_cfg=norm_cfg,
            activation_cfg=activation_cfg,
            weight_std=weight_std,
        )
        self.layer2 = self._make_layer(
            block,
            384,
            layers[1],
            stride=(2, 2, 2),
            norm_cfg=norm_cfg,
            activation_cfg=activation_cfg,
            weight_std=weight_std,
        )
        self.layer3 = self._make_layer(
            block,
            384,
            layers[2],
            stride=(2, 2, 2),
            norm_cfg=norm_cfg,
            activation_cfg=activation_cfg,
            weight_std=weight_std,
        )
        self.layers = []

        for m in self.modules():
            if isinstance(m, (nn.Conv3d, Conv3d_wd)):
                m.weight = nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm, nn.InstanceNorm3d, nn.SyncBatchNorm)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(
        self,
        block,
        planes,
        blocks,
        stride=(1, 1, 1),
        norm_cfg="BN",
        activation_cfg="ReLU",
        weight_std=False,
    ):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv3x3x3(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                    weight_std=weight_std,
                ),
                Norm_layer(norm_cfg, planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                norm_cfg,
                activation_cfg,
                stride=stride,
                downsample=downsample,
                weight_std=weight_std,
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, norm_cfg, activation_cfg, weight_std=weight_std)
            )

        return nn.Sequential(*layers)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, Conv3d_wd)):
                m.weight = nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm, nn.InstanceNorm3d, nn.SyncBatchNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = []
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.nonlin(x)
        out.append(x)

        x = self.layer1(x)
        out.append(x)
        x = self.layer2(x)
        out.append(x)
        x = self.layer3(x)
        out.append(x)

        return out


# ---------------------------------------------------------------------------
# from CoTr/network_architecture/ResTranUnet.py
# ---------------------------------------------------------------------------
def resblock_conv3x3x3(
    in_planes,
    out_planes,
    kernel_size,
    stride=(1, 1, 1),
    padding=(0, 0, 0),
    dilation=(1, 1, 1),
    groups=1,
    bias=False,
    weight_std=False,
):
    "3x3x3 convolution with padding"
    if weight_std:
        return Conv3d_wd(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
    else:
        return nn.Conv3d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )


class Conv3dBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        norm_cfg,
        activation_cfg,
        kernel_size,
        stride=(1, 1, 1),
        padding=(0, 0, 0),
        dilation=(1, 1, 1),
        bias=False,
        weight_std=False,
    ):
        super(Conv3dBlock, self).__init__()
        self.conv = resblock_conv3x3x3(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            weight_std=weight_std,
        )
        self.norm = Norm_layer(norm_cfg, out_channels)
        self.nonlin = Activation_layer(activation_cfg, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.nonlin(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, inplanes, planes, norm_cfg, activation_cfg, weight_std=False):
        super(ResBlock, self).__init__()
        self.resconv1 = Conv3dBlock(
            inplanes,
            planes,
            norm_cfg,
            activation_cfg,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            weight_std=weight_std,
        )
        self.resconv2 = Conv3dBlock(
            planes,
            planes,
            norm_cfg,
            activation_cfg,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            weight_std=weight_std,
        )

    def forward(self, x):
        residual = x

        out = self.resconv1(x)
        out = self.resconv2(out)
        out = out + residual

        return out


class U_ResTran3D(nn.Module):
    def __init__(
        self,
        norm_cfg="BN",
        activation_cfg="ReLU",
        img_size=None,
        num_classes=None,
        weight_std=False,
    ):
        super(U_ResTran3D, self).__init__()

        self.MODEL_NUM_CLASSES = num_classes

        self.upsamplex2 = nn.Upsample(scale_factor=(1, 2, 2), mode="trilinear")

        self.transposeconv_stage2 = nn.ConvTranspose3d(
            384, 384, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False
        )
        self.transposeconv_stage1 = nn.ConvTranspose3d(
            384, 192, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False
        )
        self.transposeconv_stage0 = nn.ConvTranspose3d(
            192, 64, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False
        )

        self.stage2_de = ResBlock(384, 384, norm_cfg, activation_cfg, weight_std=weight_std)
        self.stage1_de = ResBlock(192, 192, norm_cfg, activation_cfg, weight_std=weight_std)
        self.stage0_de = ResBlock(64, 64, norm_cfg, activation_cfg, weight_std=weight_std)

        self.ds2_cls_conv = nn.Conv3d(384, self.MODEL_NUM_CLASSES, kernel_size=1)
        self.ds1_cls_conv = nn.Conv3d(192, self.MODEL_NUM_CLASSES, kernel_size=1)
        self.ds0_cls_conv = nn.Conv3d(64, self.MODEL_NUM_CLASSES, kernel_size=1)

        self.cls_conv = nn.Conv3d(64, self.MODEL_NUM_CLASSES, kernel_size=1)

        for m in self.modules():
            if isinstance(m, (nn.Conv3d, Conv3d_wd, nn.ConvTranspose3d)):
                m.weight = nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, (nn.BatchNorm3d, nn.SyncBatchNorm, nn.InstanceNorm3d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.backbone = Backbone(
            depth=9, norm_cfg=norm_cfg, activation_cfg=activation_cfg, weight_std=weight_std
        )

        self.position_embed = build_position_encoding(mode="v2", hidden_dim=384)
        self.encoder_Detrans = DeformableTransformer(
            d_model=384,
            dim_feedforward=1536,
            dropout=0.1,
            activation="gelu",
            num_feature_levels=2,
            nhead=6,
            num_encoder_layers=6,
            enc_n_points=4,
        )

    def posi_mask(self, x):
        x_fea = []
        x_posemb = []
        masks = []
        for lvl, fea in enumerate(x):
            if lvl > 1:
                x_fea.append(fea)
                x_posemb.append(self.position_embed(fea))
                masks.append(
                    torch.zeros(
                        (fea.shape[0], fea.shape[2], fea.shape[3], fea.shape[4]), dtype=torch.bool
                    ).to(fea.device)
                )  # menagerie: was .cuda()

        return x_fea, masks, x_posemb

    def forward(self, inputs):
        # # %%%%%%%%%%%%% CoTr
        x_convs = self.backbone(inputs)
        x_fea, masks, x_posemb = self.posi_mask(x_convs)
        x_trans = self.encoder_Detrans(x_fea, masks, x_posemb)

        # # Multi-scale
        x = self.transposeconv_stage2(
            x_trans[:, x_fea[0].shape[-3] * x_fea[0].shape[-2] * x_fea[0].shape[-1] : :]
            .transpose(-1, -2)
            .view(x_convs[-1].shape)
        )  # x_trans length: 12*24*24+6*12*12=7776
        skip2 = (
            x_trans[:, 0 : x_fea[0].shape[-3] * x_fea[0].shape[-2] * x_fea[0].shape[-1]]
            .transpose(-1, -2)
            .view(x_convs[-2].shape)
        )

        x = x + skip2
        x = self.stage2_de(x)
        ds2 = self.ds2_cls_conv(x)

        x = self.transposeconv_stage1(x)
        skip1 = x_convs[-3]
        x = x + skip1
        x = self.stage1_de(x)
        ds1 = self.ds1_cls_conv(x)

        x = self.transposeconv_stage0(x)
        skip0 = x_convs[-4]
        x = x + skip0
        x = self.stage0_de(x)
        ds0 = self.ds0_cls_conv(x)

        result = self.upsamplex2(x)
        result = self.cls_conv(result)

        return [result, ds0, ds1, ds2]


class ResTranUnet(nn.Module):
    """
    ResTran-3D Unet

    menagerie note: the real repo's ResTranUnet inherits nnU-Net's SegmentationNetwork
    (adds unused sliding-window-inference helper methods only); vendored here as a plain
    nn.Module. forward()/__init__() logic is unchanged.
    """

    def __init__(
        self,
        norm_cfg="BN",
        activation_cfg="ReLU",
        img_size=None,
        num_classes=None,
        weight_std=False,
        deep_supervision=False,
    ):
        super().__init__()
        self.do_ds = False
        self.U_ResTran3D = U_ResTran3D(
            norm_cfg, activation_cfg, img_size, num_classes, weight_std
        )  # U_ResTran3D

        if weight_std == False:  # noqa: E712 (kept verbatim from real repo)
            self.conv_op = nn.Conv3d
        else:
            self.conv_op = Conv3d_wd
        if norm_cfg == "BN":
            self.norm_op = nn.BatchNorm3d
        if norm_cfg == "SyncBN":
            self.norm_op = nn.SyncBatchNorm
        if norm_cfg == "GN":
            self.norm_op = nn.GroupNorm
        if norm_cfg == "IN":
            self.norm_op = nn.InstanceNorm3d
        self.dropout_op = nn.Dropout3d
        self.num_classes = num_classes
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision

    def forward(self, x):
        seg_output = self.U_ResTran3D(x)
        if self._deep_supervision and self.do_ds:
            return seg_output
        else:
            return seg_output[0]


def build_cotr():
    # menagerie-sized: depth-9 backbone + 2-level deformable transformer, real num_classes~14
    return ResTranUnet(
        norm_cfg="BN",
        activation_cfg="ReLU",
        num_classes=3,
        weight_std=False,
        deep_supervision=False,
    )


def example_input_cotr():
    # backbone needs 3 strided stages (2,2,2 each) before the 2 transformer feature levels,
    # so D/H/W must be divisible by 8; keep this tiny for menagerie tracing.
    return torch.randn(1, 1, 16, 32, 32)


MENAGERIE_ENTRIES = [
    ("CoTr", "build_cotr", "example_input_cotr", 2021, "SOURCE_AVAILABLE"),
]
