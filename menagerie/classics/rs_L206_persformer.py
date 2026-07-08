# SOURCE: vendored from https://github.com/OpenDriveLab/PersFormer_3DLane @ main
#
# PersFormer: a perspective-to-BEV transformer for 3D lane detection (ECCV 2022 Oral,
# OpenDriveLab). Vendored files (all real repo code, architecture unmodified):
#   models/PersFormer.py            -> PersFormer, PerspectiveTransformer, BEVHead, SegmentHead
#   models/networks/feature_extractor.py -> deepFeatureExtractor_ResNext101 (default encoder only)
#   models/networks/Lane2D.py       -> FrontViewPathway, LaneATTHead (2D lane head)
#   models/networks/Lane3D.py       -> RefPntsNoGradGenerator, LanePredictionHead,
#                                       SingleTopViewPathway, EasyDown2TopViewPathway
#   models/networks/Layers.py       -> FFN, EncoderLayer (transformer encoder block)
#   models/networks/Deform_ATTN.py  -> _get_activation_fn, _get_clones
#   models/networks/PE.py           -> PositionEmbeddingLearned
#   models/networks/Unet_parts.py   -> Doublecnv, Down, Up (segmentation head U-Net)
#   models/networks/libs/layers.py  -> make_one_layer
#   models/networks/libs/lane.py, libs/focal_loss.py, libs/matching.py -> loss/postproc-only
#     (not on the construction/forward path exercised here; omitted)
#   models/ops/modules/ms_deform_attn.py    -> MSDeformAttn / IdentityMSDeformAttn /
#                                               DropoutMSDeformAttn (deformable attention)
#   models/ops/functions/ms_deform_attn_func.py -> ms_deform_attn_core_pytorch (the
#     repo's OWN pure-pytorch reference implementation of MSDeformAttnFunction, shipped
#     in the same file as the CUDA-extension version "for debug and test only, need to
#     use cuda version instead" -- used here in place of the compiled
#     `MultiScaleDeformableAttention` CUDA extension, which is not pip-installable in a
#     base env. Same op, same math, no architecture change.)
#   utils/utils.py (subset) -> homography_im2ipm_norm/_ipmnorm2g/_crop_resize helpers,
#     define_init_weights + weights_init_* (network weight init only)
#
# Minimal compatibility fixes applied (no architectural changes):
#   - `from nms import nms` (compiled C-extension, eval-time-only NMS postprocessing,
#     never reached during model construction/forward) removed; only the loss/postproc
#     methods that used it (LaneATTHead.nms, Lane2D loss functions) were not vendored.
#   - `import geffnet` (separate pip package, only used by the EfficientNet encoder
#     branch which we don't construct) removed; only `deepFeatureExtractor_ResNext101`
#     (the repo's DEFAULT encoder, args.encoder == 'ResNext101') is vendored.
#   - `models.resnext101_32x8d(pretrained=True)` -> `pretrained=False` (random init;
#     avoid network weight download during trace/validate, as with every menagerie entry).
#   - `np.float` / `np.int` (removed in numpy>=1.24) -> `float` / `int`.
#   - `PerspectiveTransformer.get_reference_points(..., device='cuda')` default ->
#     resolved to the actual input device at call time instead of a hardcoded 'cuda'
#     (CPU-tracing compatibility; the repo already threads `no_cuda` through the rest
#     of the model for exactly this purpose).
#   - `im_anchor_origins=None, im_anchor_angles=None` passed to `LaneATTHead` (the
#     repo's own documented no-dataset default-anchor code path; `args.use_default_anchor`
#     mirrors this) instead of building a `LaneDataset` to harvest per-dataset anchors.
#
# Original license: Apache License 2.0 (The PersFormer Authors).

from __future__ import annotations

import copy
import math
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from torch import Tensor
from torch.nn.init import constant_, xavier_uniform_

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# utils/utils.py (subset): homography helpers + weight init
# ---------------------------------------------------------------------------


def homography_crop_resize(org_img_size, crop_y, resize_img_size):
    ratio_x = resize_img_size[1] / org_img_size[1]
    ratio_y = resize_img_size[0] / (org_img_size[0] - crop_y)
    H_c = np.array([[ratio_x, 0, 0], [0, ratio_y, -ratio_y * crop_y], [0, 0, 1]])
    return H_c


def homography_ipmnorm2g(top_view_region):
    import cv2

    src = np.float32([[0, 0], [1, 0], [0, 1], [1, 1]])
    H_ipmnorm2g = cv2.getPerspectiveTransform(src, np.float32(top_view_region))
    return H_ipmnorm2g


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 or classname.find("ConvTranspose") != -1:
        try:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()
        except Exception:
            pass
    elif classname.find("Linear") != -1:
        try:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()
        except Exception:
            pass
    elif classname.find("BatchNorm2d") != -1:
        try:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)
        except Exception:
            pass


def define_init_weights(model, init_w="normal"):
    if init_w == "normal":
        model.apply(weights_init_normal)
    else:
        model.apply(weights_init_normal)


# ---------------------------------------------------------------------------
# models/networks/libs/layers.py
# ---------------------------------------------------------------------------


def make_one_layer(
    in_channels, out_channels, kernel_size=3, padding=1, stride=1, batch_norm=False, inplace=True
):
    conv2d = nn.Conv2d(
        in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride
    )
    if batch_norm:
        layers = [conv2d, nn.BatchNorm2d(out_channels), nn.ReLU(inplace)]
    else:
        layers = [conv2d, nn.ReLU(inplace)]
    return layers


# ---------------------------------------------------------------------------
# models/networks/Unet_parts.py
# ---------------------------------------------------------------------------


class Doublecnv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            Doublecnv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = Doublecnv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = Doublecnv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


# ---------------------------------------------------------------------------
# models/networks/feature_extractor.py (ResNext101 branch only -- default encoder)
# ---------------------------------------------------------------------------


class deepFeatureExtractor_ResNext101(nn.Module):
    def __init__(self, lv6=False):
        super().__init__()
        # SOURCE fix: pretrained=True -> pretrained=False (random init, no network I/O)
        self.encoder = tv_models.resnext101_32x8d(weights=None)
        self.fixList = ["layer1.0", "layer1.1", ".bn"]
        self.lv6 = lv6

        if lv6 is True:
            self.layerList = ["relu", "layer1", "layer2", "layer3", "layer4"]
            self.dimList = [64, 256, 512, 1024, 2048]
        else:
            del self.encoder.layer4
            del self.encoder.fc
            self.layerList = ["relu", "layer1", "layer2", "layer3"]
            self.dimList = [64, 256, 512, 1024]

        for name, parameters in self.encoder.named_parameters():
            if name == "conv1.weight":
                parameters.requires_grad = False
            if any(x in name for x in self.fixList):
                parameters.requires_grad = False

    def forward(self, x):
        out_featList = []
        feature = x
        for k, v in self.encoder._modules.items():
            if k == "avgpool":
                break
            feature = v(feature)
            if any(x in k for x in self.layerList):
                out_featList.append(feature)
        return out_featList


# ---------------------------------------------------------------------------
# models/networks/PE.py
# ---------------------------------------------------------------------------


class PositionEmbeddingLearned(nn.Module):
    """Absolute pos embedding, learned."""

    def __init__(self, h=50, w=50, num_pos_feats=256):
        super().__init__()
        self.col_embed = nn.Embedding(h, num_pos_feats)
        self.row_embed = nn.Embedding(w, num_pos_feats)
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)
        self.w = w
        self.h = h

    def forward(self, uv_feat):
        x = uv_feat
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.row_embed(i)
        y_emb = self.col_embed(j)
        pos = (
            torch.cat(
                [
                    x_emb.unsqueeze(0).repeat(h, 1, 1),
                    y_emb.unsqueeze(1).repeat(1, w, 1),
                ],
                dim=-1,
            )
            .permute(2, 0, 1)
            .unsqueeze(0)
            .repeat(x.shape[0], 1, 1, 1)
        )
        return pos


# ---------------------------------------------------------------------------
# models/ops/functions/ms_deform_attn_func.py -- pure-pytorch reference impl
# (this is the repo's OWN fallback function, not a torchlens-authored substitute)
# ---------------------------------------------------------------------------


def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_ * M_, D_, H_, W_)
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        sampling_value_l_ = F.grid_sample(
            value_l_, sampling_grid_l_, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        sampling_value_list.append(sampling_value_l_)
    attention_weights = attention_weights.transpose(1, 2).reshape(N_ * M_, 1, Lq_, L_ * P_)
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .view(N_, M_ * D_, Lq_)
    )
    return output.transpose(1, 2).contiguous()


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError(f"invalid input for _is_power_of_2: {n} (type: {type(n)})")
    return (n & (n - 1) == 0) and n != 0


# ---------------------------------------------------------------------------
# models/ops/modules/ms_deform_attn.py
# SOURCE fix: MSDeformAttnFunction.apply(...) (compiled CUDA extension
# `MultiScaleDeformableAttention`) -> ms_deform_attn_core_pytorch(...), the same
# repo's own pure-pytorch reference implementation of the identical operation.
# ---------------------------------------------------------------------------


class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model must be divisible by n_heads, but got {d_model} and {n_heads}"
            )
        self.im2col_step = 64
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.n_heads, 1, 1, 2)
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

    def _sample(
        self,
        query,
        reference_points,
        input_flatten,
        input_spatial_shapes,
        input_level_start_index,
        input_padding_mask,
    ):
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(
            N, Len_q, self.n_heads, self.n_levels, self.n_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            N, Len_q, self.n_heads, self.n_levels * self.n_points
        )
        attention_weights = F.softmax(attention_weights, -1).view(
            N, Len_q, self.n_heads, self.n_levels, self.n_points
        )
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1
            )
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
            )
        else:
            raise ValueError(
                f"Last dim of reference_points must be 2 or 4, but get {reference_points.shape[-1]} instead."
            )
        output = ms_deform_attn_core_pytorch(
            value, input_spatial_shapes, sampling_locations, attention_weights
        )
        return self.output_proj(output)

    def forward(
        self,
        query,
        reference_points,
        input_flatten,
        input_spatial_shapes,
        input_level_start_index,
        input_padding_mask=None,
    ):
        return self._sample(
            query,
            reference_points,
            input_flatten,
            input_spatial_shapes,
            input_level_start_index,
            input_padding_mask,
        )


class IdentityMSDeformAttn(MSDeformAttn):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=8, dropout=0.1):
        super().__init__(d_model=d_model, n_levels=n_levels, n_heads=n_heads, n_points=n_points)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query,
        reference_points,
        input_flatten,
        input_spatial_shapes,
        input_level_start_index,
        input_padding_mask=None,
        identity=None,
    ):
        if identity is None:
            identity = query
        output = self._sample(
            query,
            reference_points,
            input_flatten,
            input_spatial_shapes,
            input_level_start_index,
            input_padding_mask,
        )
        return self.dropout(output) + identity


class DropoutMSDeformAttn(MSDeformAttn):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4, dropout=0.1):
        super().__init__(d_model=d_model, n_levels=n_levels, n_heads=n_heads, n_points=n_points)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query,
        reference_points,
        input_flatten,
        input_spatial_shapes,
        input_level_start_index,
        input_padding_mask=None,
    ):
        output = self._sample(
            query,
            reference_points,
            input_flatten,
            input_spatial_shapes,
            input_level_start_index,
            input_padding_mask,
        )
        return self.dropout(output)


# ---------------------------------------------------------------------------
# models/networks/Deform_ATTN.py
# ---------------------------------------------------------------------------


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


# ---------------------------------------------------------------------------
# models/networks/Layers.py
# ---------------------------------------------------------------------------


class FFN(nn.Module):
    def __init__(
        self, d_model=256, dim_ff=1024, activation="relu", ffn_dropout=0.0, add_identity=True
    ):
        super().__init__()
        self.d_model = d_model
        self.feedforward_channels = dim_ff
        self.linear1 = nn.Linear(d_model, dim_ff)
        self.activation = _get_activation_fn(activation)
        self.dropout1 = nn.Dropout(ffn_dropout)
        self.linear2 = nn.Linear(dim_ff, d_model)
        self.dropout2 = nn.Dropout(ffn_dropout)
        self.add_identity = add_identity
        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.linear1.weight.data)
        constant_(self.linear1.bias.data, 0.0)
        xavier_uniform_(self.linear2.weight.data)
        constant_(self.linear2.bias.data, 0.0)

    def forward(self, x, identity=None):
        inter = self.linear2(self.dropout1(self.activation(self.linear1(x))))
        out = self.dropout2(inter)
        if not self.add_identity:
            return out
        if identity is None:
            identity = x
        return identity + out


class EncoderLayer(nn.Module):
    """self-attn -> norm -> cross-attn -> norm -> ffn -> norm"""

    def __init__(
        self,
        d_model=None,
        dim_ff=None,
        activation="relu",
        ffn_dropout=0.0,
        num_levels=4,
        num_points=8,
        num_heads=8,
    ):
        super().__init__()
        self.fp16_enabled = False
        self.self_attn = IdentityMSDeformAttn(d_model=d_model, n_levels=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = DropoutMSDeformAttn(
            d_model=d_model, n_levels=num_levels, n_points=num_points, n_heads=num_heads
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FFN(
            d_model=d_model, dim_ff=dim_ff, activation=activation, ffn_dropout=ffn_dropout
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        query=None,
        value=None,
        bev_pos=None,
        ref_2d=None,
        ref_3d=None,
        bev_h=None,
        bev_w=None,
        spatial_shapes=None,
        level_start_index=None,
    ):
        identity = query
        temp_value = query
        query = self.self_attn(
            query + bev_pos,
            reference_points=ref_2d,
            input_flatten=temp_value,
            input_spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
            input_level_start_index=torch.tensor([0], device=query.device),
            identity=identity,
        )
        identity = query
        query = self.norm1(query)
        query = self.cross_attn(
            query,
            reference_points=ref_3d,
            input_flatten=value,
            input_spatial_shapes=spatial_shapes,
            input_level_start_index=level_start_index,
        )
        query = query + identity
        query = self.norm2(query)
        query = self.ffn(query)
        query = self.norm3(query)
        return query


# ---------------------------------------------------------------------------
# models/networks/Lane2D.py (FrontViewPathway + LaneATTHead construction/forward only)
# ---------------------------------------------------------------------------


class FrontViewPathway(nn.Module):
    def __init__(self, input_channels, num_proj, init_weights=True):
        super().__init__()
        self.input_channels = input_channels
        self.num_proj = num_proj

        self.layers = nn.ModuleList()
        output_channels = input_channels
        for i in range(num_proj - 1):
            if i < num_proj - 2:
                output_channels *= 2
            layers = []
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            conv2d = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)
            layers += [conv2d, nn.BatchNorm2d(output_channels), nn.ReLU(inplace=True)]
            conv2d_add = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)
            layers += [conv2d_add, nn.BatchNorm2d(output_channels), nn.ReLU(inplace=True)]
            layers += [conv2d_add, nn.BatchNorm2d(output_channels), nn.ReLU(inplace=True)]
            self.layers.append(nn.Sequential(*layers))
            input_channels = output_channels

        if init_weights:
            self._initialize_weights()

    def forward(self, input):
        outs = [input]
        for layer in self.layers:
            input = layer(input)
            outs.append(input)
        return outs

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class LaneATTHead(nn.Module):
    def __init__(
        self,
        stride,
        input_channels,
        im_anchor_origins,
        im_anchor_angles,
        img_w=640,
        img_h=360,
        S=72,
        anchor_feat_channels=64,
        num_category=2,
    ):
        super().__init__()
        self.stride = stride
        self.img_w = img_w
        self.img_h = img_h
        self.hw_ratio = img_h / img_w
        self.fmap_h = img_h // stride
        fmap_w = img_w // stride
        self.fmap_w = fmap_w
        self.n_strips = S - 1
        self.n_offsets = S
        self.anchor_feat_channels = anchor_feat_channels
        self.anchor_ys = torch.linspace(1, 0, steps=self.n_offsets, dtype=torch.float32)
        self.anchor_cut_ys = torch.linspace(1, 0, steps=self.fmap_h, dtype=torch.float32)
        self.num_category = num_category

        # SOURCE: repo's documented no-dataset default-anchor path (im_anchor_origins/
        # im_anchor_angles are None unless a LaneDataset harvested them)
        self.use_default_anchor = True
        self.left_angles = [72.0, 60.0, 49.0, 39.0, 30.0, 22.0]
        self.right_angles = [108.0, 120.0, 131.0, 141.0, 150.0, 158.0]
        self.bottom_angles = [
            165.0,
            150.0,
            141.0,
            131.0,
            120.0,
            108.0,
            100.0,
            90.0,
            80.0,
            72.0,
            60.0,
            49.0,
            39.0,
            30.0,
            15.0,
        ]
        self.anchors, self.anchors_cut = self.generate_anchors(lateral_n=72, bottom_n=128)

        self.cut_zs, self.cut_ys, self.cut_xs, self.invalid_mask = self.compute_anchor_cut_indices(
            self.anchor_feat_channels, fmap_w, self.fmap_h
        )

        self.conv1 = nn.Conv2d(input_channels, self.anchor_feat_channels, kernel_size=1)
        self.cls_layer = nn.Linear(2 * self.anchor_feat_channels * self.fmap_h, self.num_category)
        self.reg_layer = nn.Linear(2 * self.anchor_feat_channels * self.fmap_h, 2 * self.n_offsets)
        self.attention_layer = nn.Linear(
            self.anchor_feat_channels * self.fmap_h, len(self.anchors) - 1
        )
        self.initialize_layer(self.attention_layer)
        self.initialize_layer(self.conv1)
        self.initialize_layer(self.cls_layer)
        self.initialize_layer(self.reg_layer)

    def forward(self, batch_features, conf_threshold=None, nms_thres=0, nms_topk=3000, eval=False):
        batch_features = self.conv1(batch_features)
        batch_anchor_features = self.cut_anchor_features(batch_features)

        # Join proposals from all images into a single proposals features batch
        batch_anchor_features = batch_anchor_features.view(
            -1, self.anchor_feat_channels * self.fmap_h
        )

        # Move relevant tensors to device
        self.anchors = self.anchors.to(device=batch_features.device)

        # Add attention features
        softmax = nn.Softmax(dim=1)
        scores = self.attention_layer(batch_anchor_features)
        attention = softmax(scores).reshape(batch_features.shape[0], len(self.anchors), -1)
        attention_matrix = torch.eye(attention.shape[1], device=batch_features.device).repeat(
            batch_features.shape[0], 1, 1
        )
        non_diag_inds = torch.nonzero(attention_matrix == 0.0, as_tuple=False)
        attention_matrix[:] = 0
        attention_matrix[non_diag_inds[:, 0], non_diag_inds[:, 1], non_diag_inds[:, 2]] = (
            attention.flatten()
        )
        batch_anchor_features = batch_anchor_features.reshape(
            batch_features.shape[0], len(self.anchors), -1
        )
        attention_features = torch.bmm(
            torch.transpose(batch_anchor_features, 1, 2), torch.transpose(attention_matrix, 1, 2)
        ).transpose(1, 2)
        attention_features = attention_features.reshape(-1, self.anchor_feat_channels * self.fmap_h)
        batch_anchor_features = batch_anchor_features.reshape(
            -1, self.anchor_feat_channels * self.fmap_h
        )
        batch_anchor_features = torch.cat((attention_features, batch_anchor_features), dim=1)

        # Predict
        cls_logits = self.cls_layer(batch_anchor_features)
        reg = self.reg_layer(batch_anchor_features)

        # Undo joining
        cls_logits = cls_logits.reshape(batch_features.shape[0], -1, cls_logits.shape[1])
        reg = reg.reshape(batch_features.shape[0], -1, reg.shape[1])
        sigmoid = nn.Sigmoid()
        reg[:, :, self.n_offsets :] = sigmoid(reg[:, :, self.n_offsets :])

        # Add offsets to anchors
        reg_proposals = torch.zeros(
            (*cls_logits.shape[:2], self.num_category + 2 + 2 * self.n_offsets),
            device=batch_features.device,
        )
        reg_proposals += self.anchors
        reg_proposals[:, :, : self.num_category] = cls_logits
        reg_proposals[:, :, self.num_category + 2 :] += reg

        proposals_list = []
        for proposals, att_matrix in zip(reg_proposals, attention_matrix):
            anchor_inds = torch.arange(reg_proposals.shape[1], device=proposals.device)
            proposals_list.append((proposals, self.anchors, att_matrix, anchor_inds))

        return proposals_list

    @staticmethod
    def initialize_layer(layer):
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            torch.nn.init.normal_(layer.weight, mean=0.0, std=0.001)
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, 0)

    def cut_anchor_features(self, features):
        batch_size = features.shape[0]
        n_proposals = len(self.anchors)
        n_fmaps = features.shape[1]
        batch_anchor_features = torch.zeros(
            (batch_size, n_proposals, n_fmaps, self.fmap_h, 1), device=features.device
        )
        for batch_idx, img_features in enumerate(features):
            rois = img_features[self.cut_zs, self.cut_ys, self.cut_xs].view(
                n_proposals, n_fmaps, self.fmap_h, 1
            )
            rois[self.invalid_mask] = 0
            batch_anchor_features[batch_idx] = rois
        return batch_anchor_features

    def compute_anchor_cut_indices(self, n_fmaps, fmaps_w, fmaps_h):
        n_proposals = len(self.anchors_cut)
        unclamped_xs = torch.flip(
            (
                self.anchors_cut[:, self.num_category + 2 : self.num_category + 2 + fmaps_h]
                / self.stride
            )
            .round()
            .long(),
            dims=(1,),
        )
        unclamped_xs = unclamped_xs.unsqueeze(2)
        unclamped_xs = torch.repeat_interleave(unclamped_xs, n_fmaps, dim=0).reshape(-1, 1)
        cut_xs = torch.clamp(unclamped_xs, 0, fmaps_w - 1)
        unclamped_xs = unclamped_xs.reshape(n_proposals, n_fmaps, fmaps_h, 1)
        invalid_mask = (unclamped_xs < 0) | (unclamped_xs > fmaps_w)
        cut_ys = torch.arange(0, fmaps_h)
        cut_ys = cut_ys.repeat(n_fmaps * n_proposals)[:, None].reshape(
            n_proposals, n_fmaps, fmaps_h
        )
        cut_ys = cut_ys.reshape(-1, 1)
        cut_zs = torch.arange(n_fmaps).repeat_interleave(fmaps_h).repeat(n_proposals)[:, None]
        return cut_zs, cut_ys, cut_xs, invalid_mask

    def generate_anchors(self, lateral_n, bottom_n):
        left_anchors, left_cut = self.generate_side_anchors(
            self.left_angles, x=0.0, nb_origins=lateral_n
        )
        right_anchors, right_cut = self.generate_side_anchors(
            self.right_angles, x=1.0, nb_origins=lateral_n
        )
        bottom_anchors, bottom_cut = self.generate_side_anchors(
            self.bottom_angles, y=1.0, nb_origins=bottom_n
        )
        return torch.cat([left_anchors, bottom_anchors, right_anchors]), torch.cat(
            [left_cut, bottom_cut, right_cut]
        )

    def generate_side_anchors(self, angles, nb_origins, x=None, y=None):
        if x is None and y is not None:
            starts = [(x, y) for x in np.linspace(1.0, 0.0, num=nb_origins)]
        elif x is not None and y is None:
            starts = [(x, y) for y in np.linspace(1.0, 0.0, num=nb_origins)]
        else:
            raise Exception("Please define exactly one of `x` or `y` (not neither nor both)")

        n_anchors = nb_origins * len(angles)
        anchors = torch.zeros((n_anchors, self.num_category + 2 + 2 * self.n_offsets))
        anchors_cut = torch.zeros((n_anchors, self.num_category + 2 + 2 * self.fmap_h))
        for i, start in enumerate(starts):
            for j, angle in enumerate(angles):
                k = i * len(angles) + j
                anchors[k] = self.generate_anchor(start, angle)
                anchors_cut[k] = self.generate_anchor(start, angle, cut=True)
        return anchors, anchors_cut

    def generate_anchor(self, start, angle, cut=False):
        if cut:
            anchor_ys = self.anchor_cut_ys
            anchor = torch.zeros(self.num_category + 2 + 2 * self.fmap_h)
        else:
            anchor_ys = self.anchor_ys
            anchor = torch.zeros(self.num_category + 2 + 2 * self.n_offsets)
        angle = angle * math.pi / 180.0
        start_x, start_y = start
        anchor[self.num_category] = 1 - start_y
        anchor[self.num_category + 1] = start_x
        if self.use_default_anchor:
            if cut:
                anchor[self.num_category + 2 : self.num_category + 2 + self.fmap_h] = (
                    start_x + (1 - anchor_ys - 1 + start_y) / math.tan(angle)
                ) * self.img_w
            else:
                anchor[self.num_category + 2 : self.num_category + 2 + self.n_offsets] = (
                    start_x + (1 - anchor_ys - 1 + start_y) / math.tan(angle)
                ) * self.img_w
        else:
            if cut:
                anchor[self.num_category + 2 : self.num_category + 2 + self.fmap_h] = (
                    start_x + (1 - anchor_ys - 1 + start_y) / math.tan(angle) * self.hw_ratio
                ) * self.img_w
            else:
                anchor[self.num_category + 2 : self.num_category + 2 + self.n_offsets] = (
                    start_x + (1 - anchor_ys - 1 + start_y) / math.tan(angle) * self.hw_ratio
                ) * self.img_w
        return anchor


# ---------------------------------------------------------------------------
# models/networks/Lane3D.py (subset on the forward path)
# ---------------------------------------------------------------------------


class RefPntsNoGradGenerator(nn.Module):
    def __init__(self, size_ipm, M, no_cuda):
        super().__init__()
        self.H, self.W = size_ipm
        linear_points_W = torch.linspace(0, 1 - 1 / self.W, self.W)
        linear_points_H = torch.linspace(0, 1 - 1 / self.H, self.H)
        self.base_grid = torch.zeros(self.H, self.W, 3)
        self.base_grid[:, :, 0] = torch.ger(torch.ones(self.H), linear_points_W)
        self.base_grid[:, :, 1] = torch.ger(linear_points_H, torch.ones(self.W))
        self.base_grid[:, :, 2] = 1
        if not no_cuda and torch.cuda.is_available():
            self.base_grid = self.base_grid.cuda()

    def forward(self, M):
        with torch.no_grad():
            grid = torch.matmul(
                self.base_grid.to(M.device).view(self.H * self.W, 3), M.transpose(1, 2)
            )
            grid = torch.div(grid[:, :, 0:2], grid[:, :, 2:])
        return grid


class LanePredictionHead(nn.Module):
    def __init__(
        self,
        input_channels,
        num_lane_type,
        num_y_steps,
        num_category,
        fmap_mapping_interp_index,
        fmap_mapping_interp_weight,
        no_3d=False,
        batch_norm=False,
        no_cuda=False,
    ):
        super().__init__()
        self.num_lane_type = num_lane_type
        self.num_y_steps = num_y_steps
        self.no_3d = no_3d
        if no_3d:
            self.anchor_dim = self.num_y_steps + num_category
        else:
            self.anchor_dim = 3 * self.num_y_steps + num_category
        self.num_category = num_category

        layers = []
        layers += make_one_layer(
            input_channels, 64, kernel_size=3, padding=(0, 1), batch_norm=batch_norm
        )
        layers += make_one_layer(64, 64, kernel_size=3, padding=(0, 1), batch_norm=batch_norm)
        layers += make_one_layer(64, 64, kernel_size=3, padding=(0, 1), batch_norm=batch_norm)
        layers += make_one_layer(64, 64, kernel_size=5, padding=(0, 2), batch_norm=batch_norm)
        layers += make_one_layer(64, 64, kernel_size=5, padding=(0, 2), batch_norm=batch_norm)
        layers += make_one_layer(64, 64, kernel_size=5, padding=(0, 2), batch_norm=batch_norm)
        layers += make_one_layer(64, 64, kernel_size=5, padding=(0, 2), batch_norm=batch_norm)
        self.features = nn.Sequential(*layers)

        dim_rt_layers = []
        dim_rt_layers += make_one_layer(
            256, 128, kernel_size=(5, 1), padding=(2, 0), batch_norm=batch_norm
        )
        dim_rt_layers += [
            nn.Conv2d(128, self.num_lane_type * self.anchor_dim, kernel_size=(5, 1), padding=(2, 0))
        ]
        self.dim_rt = nn.Sequential(*dim_rt_layers)

        self.use_default_anchor = True
        if fmap_mapping_interp_index is not None and fmap_mapping_interp_weight is not None:
            self.use_default_anchor = False
            self.fmap_mapping_interp_index = torch.tensor(fmap_mapping_interp_index)
            self.fmap_mapping_interp_weight = torch.tensor(fmap_mapping_interp_weight)

    def forward(self, x):
        if not self.use_default_anchor:
            batch_size, channel, fmap_h, fmap_w = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
            sheared_feature_map = torch.zeros((batch_size, channel, fmap_h, fmap_w * 6)).to(
                x.device
            )
            v_arange = torch.arange(fmap_h).unsqueeze(dim=1).repeat(1, fmap_w * 6).to(x.device)
            self.fmap_mapping_interp_index = self.fmap_mapping_interp_index.to(x.device)
            self.fmap_mapping_interp_weight = self.fmap_mapping_interp_weight.to(x.device)
            for batch_idx, x_feature_map in enumerate(x):
                sheared_feature_map[batch_idx] = (
                    x_feature_map[:, v_arange, self.fmap_mapping_interp_index[:, :, 0]]
                    * self.fmap_mapping_interp_weight[:, :, 0]
                    + x_feature_map[:, v_arange, self.fmap_mapping_interp_index[:, :, 1]]
                    * self.fmap_mapping_interp_weight[:, :, 1]
                )
            x = torch.cat((x, sheared_feature_map), dim=3)

        x = self.features(x)
        sizes = x.shape
        x = x.reshape(sizes[0], sizes[1] * sizes[2], sizes[3], 1)
        x = self.dim_rt(x)
        x = x.squeeze(-1).transpose(1, 2)
        if self.no_3d:
            return x

        for i in range(self.num_lane_type):
            x[
                :,
                :,
                i * self.anchor_dim + 2 * self.num_y_steps : i * self.anchor_dim
                + 3 * self.num_y_steps,
            ] = torch.sigmoid(
                x[
                    :,
                    :,
                    i * self.anchor_dim + 2 * self.num_y_steps : i * self.anchor_dim
                    + 3 * self.num_y_steps,
                ]
            )
        return x


class SingleTopViewPathway(nn.Module):
    def __init__(self, input_channels, batch_norm=False, init_weights=True):
        super().__init__()
        self.input_channels = input_channels
        layers = [nn.MaxPool2d(kernel_size=2, stride=2)]
        conv2d = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1)
        layers += [conv2d, nn.BatchNorm2d(input_channels), nn.ReLU(inplace=True)]
        conv2d_add = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1)
        layers += [conv2d_add, nn.BatchNorm2d(input_channels), nn.ReLU(inplace=True)]
        layers += [conv2d_add, nn.BatchNorm2d(input_channels), nn.ReLU(inplace=True)]
        self.feature = nn.Sequential(*layers)
        if init_weights:
            self._initialize_weights()

    def forward(self, input):
        return self.feature(input)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class EasyDown2TopViewPathway(nn.Module):
    def __init__(self, input_channels, batch_norm=False, init_weights=True):
        super().__init__()
        self.input_channels = input_channels
        output_channels = input_channels // 2
        layers = [nn.MaxPool2d(kernel_size=2, stride=2)]
        conv2d = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)
        layers += [conv2d, nn.BatchNorm2d(output_channels), nn.ReLU(inplace=True)]
        conv2d_add = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)
        layers += [conv2d_add, nn.BatchNorm2d(output_channels), nn.ReLU(inplace=True)]
        layers += [conv2d_add, nn.BatchNorm2d(output_channels), nn.ReLU(inplace=True)]
        self.feature = nn.Sequential(*layers)
        if init_weights:
            self._initialize_weights()

    def forward(self, input):
        return self.feature(input)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


# ---------------------------------------------------------------------------
# models/PersFormer.py
# ---------------------------------------------------------------------------


class PerspectiveTransformer(nn.Module):
    def __init__(
        self, args, channels, bev_h, bev_w, uv_h, uv_w, M_inv, num_att, num_proj, nhead, npoints
    ):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.uv_h = uv_h
        self.uv_w = uv_w
        self.M_inv = M_inv
        self.num_att = num_att
        self.num_proj = num_proj
        self.nhead = nhead
        self.npoints = npoints

        self.query_embeds = nn.ModuleList()
        self.pe = nn.ModuleList()
        self.el = nn.ModuleList()
        self.project_layers = nn.ModuleList()
        self.ref_2d = []
        self.input_spatial_shapes = []
        self.input_level_start_index = []

        uv_feat_c = channels
        for i in range(self.num_proj):
            if i > 0:
                bev_h = bev_h // 2
                bev_w = bev_w // 2
                uv_h = uv_h // 2
                uv_w = uv_w // 2
                if i != self.num_proj - 1:
                    uv_feat_c = uv_feat_c * 2

            bev_feat_len = bev_h * bev_w
            query_embed = nn.Embedding(bev_feat_len, uv_feat_c)
            self.query_embeds.append(query_embed)
            position_embed = PositionEmbeddingLearned(bev_h, bev_w, num_pos_feats=uv_feat_c // 2)
            self.pe.append(position_embed)

            # SOURCE fix: get_reference_points(..., device='cuda') hardcoded default ->
            # resolved to 'cpu' here (module built before any input is seen; forward()
            # already re-homes every per-projection tensor onto input.device).
            ref_point = self.get_reference_points(H=bev_h, W=bev_w, dim="2d", bs=1, device="cpu")
            self.ref_2d.append(ref_point)

            size_top = torch.Size([bev_h, bev_w])
            project_layer = RefPntsNoGradGenerator(size_top, self.M_inv, args.no_cuda)
            self.project_layers.append(project_layer)

            spatial_shape = torch.as_tensor([(uv_h, uv_w)], dtype=torch.long)
            self.input_spatial_shapes.append(spatial_shape)

            level_start_index = torch.as_tensor([0.0], dtype=torch.long)
            self.input_level_start_index.append(level_start_index)

            for _ in range(self.num_att):
                encoder_layers = EncoderLayer(
                    d_model=uv_feat_c,
                    dim_ff=uv_feat_c * 2,
                    num_levels=1,
                    num_points=self.npoints,
                    num_heads=self.nhead,
                )
                self.el.append(encoder_layers)

    def forward(self, input, frontview_features, _M_inv=None):
        projs = []
        for i in range(self.num_proj):
            if i == 0:
                bev_h = self.bev_h
                bev_w = self.bev_w
            else:
                bev_h = bev_h // 2
                bev_w = bev_w // 2
            bs, c, h, w = frontview_features[i].shape
            query_embed = self.query_embeds[i].weight.unsqueeze(0).repeat(bs, 1, 1)
            src = frontview_features[i].flatten(2).permute(0, 2, 1)
            bev_mask = torch.zeros((bs, bev_h, bev_w), device=query_embed.device).to(
                query_embed.dtype
            )
            bev_pos = self.pe[i](bev_mask).to(query_embed.dtype)
            bev_pos = bev_pos.flatten(2).permute(0, 2, 1)
            ref_2d = self.ref_2d[i].repeat(bs, 1, 1, 1).to(input.device)
            ref_pnts = self.project_layers[i](_M_inv).unsqueeze(-2)
            input_spatial_shapes = self.input_spatial_shapes[i].to(input.device)
            input_level_start_index = self.input_level_start_index[i].to(input.device)
            for j in range(self.num_att):
                query_embed = self.el[i * self.num_att + j](
                    query=query_embed,
                    value=src,
                    bev_pos=bev_pos,
                    ref_2d=ref_2d,
                    ref_3d=ref_pnts,
                    bev_h=bev_h,
                    bev_w=bev_w,
                    spatial_shapes=input_spatial_shapes,
                    level_start_index=input_level_start_index,
                )
            query_embed = query_embed.permute(0, 2, 1).view(bs, c, bev_h, bev_w).contiguous()
            projs.append(query_embed)
        return projs

    @staticmethod
    def get_reference_points(H, W, Z=8, D=4, dim="3d", bs=1, device="cpu", dtype=torch.long):
        if dim == "3d":
            raise Exception("get reference points 3d not supported")
        elif dim == "2d":
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device),
                indexing="ij",
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d


class BEVHead(nn.Module):
    def __init__(self, args, channels=128):
        super().__init__()
        self.size_reduce_layer_1 = SingleTopViewPathway(channels)
        self.size_reduce_layer_2 = SingleTopViewPathway(channels * 2)
        self.size_dim_reduce_layer_3 = EasyDown2TopViewPathway(channels * 4)

        self.dim_reduce_layers = nn.ModuleList()
        self.dim_reduce_layers.append(
            nn.Sequential(
                *make_one_layer(
                    channels * 2, channels, kernel_size=1, padding=0, batch_norm=args.batch_norm
                )
            )
        )
        self.dim_reduce_layers.append(
            nn.Sequential(
                *make_one_layer(
                    channels * 4, channels * 2, kernel_size=1, padding=0, batch_norm=args.batch_norm
                )
            )
        )
        self.dim_reduce_layers.append(
            nn.Sequential(
                *make_one_layer(
                    channels * 4, channels * 2, kernel_size=1, padding=0, batch_norm=args.batch_norm
                )
            )
        )

    def forward(self, projs):
        bev_feat_1 = self.size_reduce_layer_1(projs[0])
        rts_proj_feat_1 = self.dim_reduce_layers[0](projs[1])
        bev_feat_2 = self.size_reduce_layer_2(torch.cat((bev_feat_1, rts_proj_feat_1), 1))
        rts_proj_feat_2 = self.dim_reduce_layers[1](projs[2])
        bev_feat_3 = self.size_dim_reduce_layer_3(torch.cat((bev_feat_2, rts_proj_feat_2), 1))
        rts_proj_feat_3 = self.dim_reduce_layers[2](projs[3])
        bev_feat = torch.cat((bev_feat_3, rts_proj_feat_3), 1)
        return bev_feat


class SegmentHead(nn.Module):
    def __init__(self, channels=128):
        super().__init__()
        self.down1 = Down(channels, channels * 2)
        self.down2 = Down(channels * 2, channels * 4)
        self.down3 = Down(channels * 4, channels * 4)
        self.up1 = Up(channels * 8, channels * 2)
        self.up2 = Up(channels * 4, channels)
        self.up3 = Up(channels * 2, channels)
        self.segment_head = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, input):
        x1 = self.down1(input)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x_out = self.up1(x3, x2)
        x_out = self.up2(x_out, x1)
        x_out = self.up3(x_out, input)
        pred_seg_bev_map = self.segment_head(x_out)
        return pred_seg_bev_map


class PersFormer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.no_cuda = args.no_cuda
        self.batch_size = args.batch_size
        self.num_lane_type = 1
        self.num_y_steps = args.num_y_steps
        self.max_lanes = args.max_lanes
        self.num_category = args.num_category
        self.num_proj = args.num_proj
        self.num_att = args.num_att

        self.M_inv, self.cam_height, self.cam_pitch = self.get_transform_matrices(args)

        self.encoder = deepFeatureExtractor_ResNext101(lv6=False)
        self.neck = nn.Sequential(
            *make_one_layer(self.encoder.dimList[0], args.feature_channels, batch_norm=True),
            *make_one_layer(args.feature_channels, args.feature_channels, batch_norm=True),
        )
        self.shared_encoder = FrontViewPathway(args.feature_channels, args.num_proj)
        stride = 2
        self.laneatt_head = LaneATTHead(
            stride * pow(2, args.num_proj - 1),
            args.feature_channels * pow(2, args.num_proj - 2),
            None,
            None,
            img_w=args.resize_w,
            img_h=args.resize_h,
            S=args.S,
            anchor_feat_channels=args.anchor_feat_channels,
            num_category=args.num_category,
        )
        self.pers_tr = PerspectiveTransformer(
            args,
            channels=args.feature_channels,
            bev_h=args.ipm_h,
            bev_w=args.ipm_w,
            uv_h=args.resize_h // stride,
            uv_w=args.resize_w // stride,
            M_inv=self.M_inv,
            num_att=self.num_att,
            num_proj=self.num_proj,
            nhead=args.nhead,
            npoints=args.npoints,
        )
        self.bev_head = BEVHead(args, channels=args.feature_channels)
        self.lane_out = LanePredictionHead(
            args.feature_channels * pow(2, self.num_proj - 2),
            self.num_lane_type,
            self.num_y_steps,
            args.num_category,
            None,
            None,
            args.no_3d,
            args.batch_norm,
            args.no_cuda,
        )
        self.segment_head = SegmentHead(channels=args.feature_channels)
        self.uncertainty_loss = nn.Parameter(
            torch.tensor(
                [
                    args._3d_vis_loss_weight,
                    args._3d_prob_loss_weight,
                    args._3d_reg_loss_weight,
                    args._2d_vis_loss_weight,
                    args._2d_prob_loss_weight,
                    args._2d_reg_loss_weight,
                    args._seg_loss_weight,
                ]
            ),
            requires_grad=True,
        )
        self._initialize_weights(args)

    def forward(self, input, _M_inv=None):
        out_featList = self.encoder(input)
        neck_out = self.neck(out_featList[0])
        frontview_features = self.shared_encoder(neck_out)
        frontview_final_feat = frontview_features[-1]

        laneatt_proposals_list = self.laneatt_head(frontview_final_feat)

        projs = self.pers_tr(input, frontview_features, _M_inv)

        bev_feat = self.bev_head(projs)

        out = self.lane_out(bev_feat)

        cam_height = self.cam_height.to(input.device)
        cam_pitch = self.cam_pitch.to(input.device)

        pred_seg_bev_map = self.segment_head(projs[0])

        uncertainty_loss = torch.tensor(1.0).to(input.device) * self.uncertainty_loss.to(
            input.device
        )

        return (
            laneatt_proposals_list,
            out,
            cam_height,
            cam_pitch,
            pred_seg_bev_map,
            uncertainty_loss,
        )

    def _initialize_weights(self, args):
        define_init_weights(self.neck, args.weight_init)
        define_init_weights(self.shared_encoder, args.weight_init)
        define_init_weights(self.laneatt_head, args.weight_init)
        define_init_weights(self.pers_tr, args.weight_init)
        define_init_weights(self.bev_head, args.weight_init)
        define_init_weights(self.lane_out, args.weight_init)
        define_init_weights(self.segment_head, args.weight_init)

    def get_transform_matrices(self, args):
        org_img_size = np.array([args.org_h, args.org_w])
        resize_img_size = np.array([args.resize_h, args.resize_w])
        cam_pitch = np.pi / 180 * args.pitch

        S_im_inv = torch.from_numpy(
            np.array(
                [
                    [1 / float(args.resize_w), 0, 0],
                    [0, 1 / float(args.resize_h), 0],
                    [0, 0, 1],
                ],
                dtype=np.float32,
            )
        )
        S_im_inv_batch = (
            S_im_inv.unsqueeze_(0).expand([self.batch_size, 3, 3]).type(torch.FloatTensor)
        )

        H_c = homography_crop_resize(org_img_size, args.crop_y, resize_img_size)
        H_c = (
            torch.from_numpy(H_c)
            .unsqueeze_(0)
            .expand([self.batch_size, 3, 3])
            .type(torch.FloatTensor)
        )

        K = (
            torch.from_numpy(args.K)
            .unsqueeze_(0)
            .expand([self.batch_size, 3, 3])
            .type(torch.FloatTensor)
        )

        H_g2cam = np.array(
            [
                [1, 0, 0],
                [0, np.sin(-cam_pitch), args.cam_height],
                [0, np.cos(-cam_pitch), 0],
            ]
        )
        H_g2cam = (
            torch.from_numpy(H_g2cam)
            .unsqueeze_(0)
            .expand([self.batch_size, 3, 3])
            .type(torch.FloatTensor)
        )

        H_ipmnorm2g = homography_ipmnorm2g(args.top_view_region)
        H_ipmnorm2g = (
            torch.from_numpy(H_ipmnorm2g)
            .unsqueeze_(0)
            .expand([self.batch_size, 3, 3])
            .type(torch.FloatTensor)
        )

        M_ipm2im = torch.bmm(H_g2cam, H_ipmnorm2g)
        M_ipm2im = torch.bmm(K, M_ipm2im)
        M_ipm2im = torch.bmm(H_c, M_ipm2im)
        M_ipm2im = torch.bmm(S_im_inv_batch, M_ipm2im)
        M_ipm2im = torch.div(
            M_ipm2im,
            M_ipm2im[:, 2, 2].reshape([self.batch_size, 1, 1]).expand([self.batch_size, 3, 3]),
        )
        M_inv = M_ipm2im

        cam_height = (
            torch.tensor(args.cam_height)
            .unsqueeze_(0)
            .expand([self.batch_size, 1])
            .type(torch.FloatTensor)
        )
        cam_pitch = (
            torch.tensor(cam_pitch)
            .unsqueeze_(0)
            .expand([self.batch_size, 1])
            .type(torch.FloatTensor)
        )

        return M_inv, cam_height, cam_pitch


# ---------------------------------------------------------------------------
# Menagerie staging entry point
# ---------------------------------------------------------------------------


class _Args:
    """Minimal stand-in for the repo's argparse.Namespace, populated with the
    same defaults `define_args()` + `config.openlane_config()` set (see
    config/persformer_openlane.py), sized down for a tiny trace."""

    def __init__(self):
        # dataset / camera (openlane_config defaults)
        self.org_h = 128
        self.org_w = 192
        self.crop_y = 0
        self.no_centerline = True
        self.no_3d = False
        self.fix_cam = False
        self.pred_cam = False
        self.K = np.array([[1000.0, 0.0, 96.0], [0.0, 1000.0, 64.0], [0.0, 0.0, 1.0]])
        self.top_view_region = np.array([[-10, 103], [10, 103], [-10, 3], [10, 3]])
        self.anchor_y_steps = np.array([5, 10, 15, 20, 30, 40, 50, 60, 80, 100])
        self.num_y_steps = len(self.anchor_y_steps)
        self.max_lanes = 4
        self.num_category = 3
        self.prob_th = 0.5
        self.y_ref = 5
        self.pitch = 3
        self.cam_height = 1.55

        # PersFormer settings
        # NOTE: num_proj is architecturally fixed at 4 in the real repo (BEVHead's
        # dim_reduce_layers hardcode a channels*2/channels*4 schedule tied to a
        # 4-level FPN); shrinking it breaks the channel-doubling invariant, so we
        # keep it at the repo default and shrink feature_channels/S/anchor counts
        # instead for a tiny trace.
        self.mod = "PersFormer"
        self.pretrained = False
        self.batch_norm = True
        self.encoder = "ResNext101"
        self.feature_channels = 8
        self.num_proj = 4
        self.num_att = 1
        self.use_default_anchor = True
        self.resize_h = 64
        self.resize_w = 96
        # ipm_h must survive BEVHead's 3 halvings (num_proj-1) followed by
        # LanePredictionHead's fixed 7-conv chain (3x kernel=3/pad=(0,1) losing 2
        # rows each, 4x kernel=5/pad=(0,2) losing 4 rows each = 22 rows lost) AND
        # land on exactly height=4 afterward, because LanePredictionHead.dim_rt's
        # first conv hardcodes in_channels=256 in the real repo (assuming the
        # reshape `x.reshape(sizes[0], sizes[1]*sizes[2], sizes[3], 1)` with
        # sizes[1]=64 (fixed channel count of self.features' last layer) and
        # sizes[2](=height)=4 multiplies out to 256). 208 is the smallest value
        # hitting that fixed floor exactly.
        self.ipm_h = 208
        self.ipm_w = 32
        self.nhead = 2
        self.npoints = 4

        # LaneATT settings
        self.S = 8
        self.anchor_feat_channels = 4

        # general
        self.batch_size = 1
        self.no_cuda = True
        self.weight_init = "normal"

        # learnable per-term loss weights (config.py defaults)
        self._3d_vis_loss_weight = 0.0
        self._3d_prob_loss_weight = 0.0
        self._3d_reg_loss_weight = 0.0
        self._2d_vis_loss_weight = 0.0
        self._2d_prob_loss_weight = 0.0
        self._2d_reg_loss_weight = 0.0
        self._seg_loss_weight = 0.0


def build_persformer():
    args = _Args()
    return PersFormer(args)


def example_input_persformer():
    # PersFormer.forward(self, input, _M_inv=None) needs the per-batch inverse
    # homography matrix explicitly; the repo's training loop always passes
    # `self.M_inv` (recomputed per batch from ground-truth camera params) or an
    # online-predicted variant, never relies on the None default. We build a
    # throwaway model instance purely to reuse its own get_transform_matrices,
    # matching how the reference trainer calls `model(input, model.M_inv)`.
    args = _Args()
    model_input = torch.randn(args.batch_size, 3, args.resize_h, args.resize_w)
    m_inv = build_persformer().M_inv
    return (model_input, m_inv)


MENAGERIE_ENTRIES = [
    ("PersFormer", build_persformer, example_input_persformer, 2022, "vendored-pytorch"),
]
