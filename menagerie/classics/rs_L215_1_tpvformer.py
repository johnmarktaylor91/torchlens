# FAITHFUL PORT of wzzheng/TPVFormer @ main (original framework: PyTorch + mmcv/mmdet/mmseg
# registry ecosystem, with a custom-compiled CUDA multi-scale-deformable-attention extension)
#
# https://github.com/wzzheng/TPVFormer
# https://raw.githubusercontent.com/wzzheng/TPVFormer/main/tpvformer04/tpvformer.py
# https://raw.githubusercontent.com/wzzheng/TPVFormer/main/tpvformer04/tpv_head.py
# https://raw.githubusercontent.com/wzzheng/TPVFormer/main/tpvformer04/tpv_aggregator.py
# https://raw.githubusercontent.com/wzzheng/TPVFormer/main/tpvformer04/modules/encoder.py
# https://raw.githubusercontent.com/wzzheng/TPVFormer/main/tpvformer04/modules/tpvformer_layer.py
# https://raw.githubusercontent.com/wzzheng/TPVFormer/main/tpvformer04/modules/cross_view_hybrid_attention.py
# https://raw.githubusercontent.com/wzzheng/TPVFormer/main/tpvformer04/modules/image_cross_attention.py
# https://raw.githubusercontent.com/wzzheng/TPVFormer/main/config/tpv04_occupancy.py
#
# TPVFormer (Huang, Zheng, Zuo, Lu, Lu, Zhou. CVPR 2023, "Tri-Perspective View for
# Vision-Based 3D Semantic Occupancy Prediction"). Predicts 3D semantic occupancy from
# multi-camera images by encoding a tri-perspective-view (TPV, 3 orthogonal 2D planes
# standing in for a dense 3D voxel grid) representation via deformable attention,
# aggregating the three planes back into a dense voxel grid.
#
# `TPVFormerHead`/`TPVFormerEncoder`/`TPVFormerLayer`/`TPVCrossViewHybridAttention`/
# `TPVImageCrossAttention`/`TPVMSDeformableAttention3D`/`TPVAggregator`/`TPVFormer` are
# transcribed FAITHFULLY from the real `tpvformer04/` module files above (every
# reference-point computation, sampling-offset/attention-weight head, cross-view fusion,
# hybrid self-attention rebatching, and the LMA-style image-cross-attention rebatch/scatter
# loop are preserved verbatim from the source, only re-indented/renamed where mmcv's own
# `xavier_init`/`constant_init` helpers are inlined).
#
# Why a PORT and not a vendor: the real repo hard-depends on the OpenMMLab registry stack
# (`mmcv.runner.BaseModule`/`force_fp32`/`auto_fp16`, `mmcv.cnn.bricks.transformer.build_*`,
# `mmseg.models.HEADS`/`SEGMENTORS` registries, `mmdet.models.utils.LearnedPositionalEncoding`)
# AND a custom-compiled CUDA extension (`mmcv.ops.multi_scale_deform_attn`'s
# `ms_deform_attn_forward`/`ms_deform_attn_backward`, loaded via `mmcv.utils.ext_loader` from a
# source build against the local CUDA toolchain -- no prebuilt PyPI wheel). Neither mmcv nor
# mmdet/mmseg is installed in base env and installing the full OpenMMLab registry+CUDA-ext
# stack is out of scope (rung 2 vendor is not possible).
#
# What is transcribed 1:1, and from where:
#   - `TPVMSDeformableAttention3D`/`TPVCrossViewHybridAttention` CPU sampling path: the real
#     repo's own `image_cross_attention.py`/`cross_view_hybrid_attention.py` already branch to
#     `multi_scale_deformable_attn_pytorch` (a pure-PyTorch `grid_sample`-based fallback) when
#     not running on CUDA -- see the `if torch.cuda.is_available() and value.is_cuda: ... else:
#     output = multi_scale_deformable_attn_pytorch(...)` branch in both files. This is the
#     repo's OWN documented CPU-equivalent code path, not an approximation invented here; the
#     `multi_scale_deformable_attn_pytorch` function itself is transcribed verbatim from
#     `mmcv/ops/multi_scale_deform_attn.py` (open-mmlab/mmcv @ v1.7.1), the file the TPVFormer
#     modules import it from.
#   - `FFN`: transcribed verbatim from `mmcv/cnn/bricks/transformer.py` (open-mmlab/mmcv @
#     v1.7.1) -- the exact class `TPVFormerLayer`'s `ffn_cfgs=dict(type='FFN', ...)` builds.
#   - `LearnedPositionalEncoding`: transcribed verbatim from
#     `mmdet/models/utils/positional_encoding.py` (open-mmlab/mmdetection @ v2.28.2) -- the
#     exact class `tpv_head.py`'s `positional_encoding=dict(type='LearnedPositionalEncoding',
#     ...)` builds.
#   - `FPN` (img_neck): transcribed verbatim from `mmdet/models/necks/fpn.py`
#     (open-mmlab/mmdetection @ v2.28.2) -- the exact neck the reference config
#     (`config/tpv04_occupancy.py`) wires as `img_neck=dict(type='FPN', ...)`, with mmcv's
#     `ConvModule(conv, norm, act)` inlined as plain `nn.Conv2d`+optional norm/act (no
#     `conv_cfg`/`norm_cfg`/`act_cfg` are set in the reference config, so `ConvModule`
#     collapses to a bare `nn.Conv2d`).
#   - `img_backbone` (ResNet-101, caffe style, DCNv2 at stages 3/4): built on
#     `torchvision.models.resnet.Bottleneck`-equivalent blocks (transcribed inline, matching
#     mmcv's `ResNet` block-for-block for the caffe stem + stage layout the reference config
#     selects), with `torchvision.ops.DeformConv2d` substituted 1:1 for mmcv's
#     `ModulatedDeformConv2dPack` (mmcv's own DCNv2 docstring notes its
#     `ModulatedDeformConv2dPack_MLU` variant IS `torchvision`'s deform_conv2d op under the
#     hood; `torchvision.ops.DeformConv2d` natively supports the `mask` argument needed for
#     DCNv2 modulation, so the offset+mask conv_offset head is reproduced exactly as mmcv's
#     `ModulatedDeformConv2dPack.forward` does: `conv_offset(x) -> chunk(3) -> (offset, mask)`,
#     `mask = sigmoid(mask)`) at `stage_with_dcn=(False, False, True, True)` (stages 3 and 4),
#     matching the reference config's `dcn=dict(type='DCNv2', ...), stage_with_dcn=(False,
#     False, True, True)`.
#   - `point_sampling`'s camera-calibration projection (`lidar2img` 4x4 matrices projecting
#     3D reference points into each of 6 camera image planes) is the real transcribed
#     algorithm from `encoder.py::point_sampling`; this build's `example_input_tpvformer`
#     supplies synthetic (not real nuScenes) `lidar2img` calibration matrices and `img_shape`
#     metadata via the same `img_metas` dict contract `TPVFormerEncoder.forward` expects
#     (`img_metas[i]['lidar2img']`, `img_metas[i]['img_shape']`), since no real calibrated
#     nuScenes sample is bundled with a toy-scale trace.
#
# Build-time shrinkage (config values, not architecture changes): the reference
# `config/tpv04_occupancy.py` uses `tpv_h=tpv_w=100`, `tpv_z=8`, `embed_dims=256`,
# `num_layers=3`, ResNet-101 img_backbone, 6 cameras, 4 FPN feature levels,
# `num_points_in_pillar=[4, 32, 32]`, `num_points=[8, 64, 64]`. This build uses a tiny
# ResNet-18-equivalent-depth img_backbone (`layers=[1, 1, 1, 1]`), `tpv_h=tpv_w=6`, `tpv_z=4`,
# `embed_dims=16` (a power of 2, matching the repo's own `_is_power_of_2(dim_per_head)`
# efficiency assertion), `num_heads=2`, `num_layers=1`, `num_cams=2`, `num_feature_levels=4`
# (3 backbone levels + 1 extra FPN level, matching the reference config's own
# 3-backbone-level-in/4-level-out FPN shape), small pillar/point counts -- every mechanism
# (self cross-view hybrid attn, image cross attn with per-camera rebatching, TPV aggregation)
# still executes on the shrunk sizes.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d


# ---------------------------------------------------------------------------
# mmcv/cnn/bricks/transformer.py::FFN (open-mmlab/mmcv @ v1.7.1), verbatim
# ---------------------------------------------------------------------------


class FFN(nn.Module):
    """Implements feed-forward networks (FFNs) with identity connection."""

    def __init__(
        self, embed_dims=256, feedforward_channels=1024, num_fcs=2, ffn_drop=0.0, add_identity=True
    ):
        super().__init__()
        assert num_fcs >= 2
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.activate = nn.ReLU(inplace=True)

        layers = []
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(
                nn.Sequential(
                    nn.Linear(in_channels, feedforward_channels),
                    self.activate,
                    nn.Dropout(ffn_drop),
                )
            )
            in_channels = feedforward_channels
        layers.append(nn.Linear(feedforward_channels, embed_dims))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = nn.Sequential(*layers)
        self.add_identity = add_identity

    def forward(self, x, identity=None):
        out = self.layers(x)
        if not self.add_identity:
            return out
        if identity is None:
            identity = x
        return identity + out


# ---------------------------------------------------------------------------
# mmdet/models/utils/positional_encoding.py::LearnedPositionalEncoding
# (open-mmlab/mmdetection @ v2.28.2), verbatim
# ---------------------------------------------------------------------------


class LearnedPositionalEncoding(nn.Module):
    """Position embedding with learnable embedding weights."""

    def __init__(self, num_feats, row_num_embed=50, col_num_embed=50):
        super().__init__()
        self.row_embed = nn.Embedding(row_num_embed, num_feats)
        self.col_embed = nn.Embedding(col_num_embed, num_feats)
        self.num_feats = num_feats
        self.row_num_embed = row_num_embed
        self.col_num_embed = col_num_embed

    def forward(self, mask):
        h, w = mask.shape[-2:]
        x = torch.arange(w, device=mask.device)
        y = torch.arange(h, device=mask.device)
        x_embed = self.col_embed(x)
        y_embed = self.row_embed(y)
        pos = (
            torch.cat(
                (x_embed.unsqueeze(0).repeat(h, 1, 1), y_embed.unsqueeze(1).repeat(1, w, 1)), dim=-1
            )
            .permute(2, 0, 1)
            .unsqueeze(0)
            .repeat(mask.shape[0], 1, 1, 1)
        )
        return pos


# ---------------------------------------------------------------------------
# mmcv/ops/multi_scale_deform_attn.py::multi_scale_deformable_attn_pytorch
# (open-mmlab/mmcv @ v1.7.1), verbatim -- this is the real repo's own documented
# CPU-equivalent fallback (see module docstring); both TPVFormer attention modules
# call this exact function when not running on a CUDA device.
# ---------------------------------------------------------------------------


def multi_scale_deformable_attn_pytorch(
    value, value_spatial_shapes, sampling_locations, attention_weights
):
    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    value_list = value.split([int(H_) * int(W_) for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        H_, W_ = int(H_), int(W_)
        value_l_ = (
            value_list[level].flatten(2).transpose(1, 2).reshape(bs * num_heads, embed_dims, H_, W_)
        )
        sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        sampling_value_l_ = F.grid_sample(
            value_l_, sampling_grid_l_, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        sampling_value_list.append(sampling_value_l_)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points
    )
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .view(bs, num_heads * embed_dims, num_queries)
    )
    return output.transpose(1, 2).contiguous()


def _xavier_init(module, gain=1, bias=0, distribution="uniform"):
    if hasattr(module, "weight") and module.weight is not None:
        if distribution == "uniform":
            nn.init.xavier_uniform_(module.weight, gain=gain)
        else:
            nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def _constant_init(module, val, bias=0):
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


# ---------------------------------------------------------------------------
# tpvformer04/modules/cross_view_hybrid_attention.py::TPVCrossViewHybridAttention
# ---------------------------------------------------------------------------


class TPVCrossViewHybridAttention(nn.Module):
    """Cross view hybrid attention module used in TPVFormer. Based on deformable attention."""

    def __init__(
        self, embed_dims=256, num_heads=8, num_levels=4, num_points=4, dropout=0.1, num_tpv_queue=2
    ):
        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(
                f"embed_dims must be divisible by num_heads, but got {embed_dims} and {num_heads}"
            )
        self.dropout = nn.Dropout(dropout)
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_tpv_queue = num_tpv_queue
        self.sampling_offsets = nn.Linear(
            embed_dims * num_tpv_queue, num_tpv_queue * num_heads * num_levels * num_points * 2
        )
        self.attention_weights = nn.Linear(
            embed_dims * num_tpv_queue, num_tpv_queue * num_heads * num_levels * num_points
        )
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.init_weights()

    def init_weights(self):
        _constant_init(self.sampling_offsets, 0.0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.num_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.num_heads, 1, 1, 2)
            .repeat(1, self.num_levels * self.num_tpv_queue, self.num_points, 1)
        )
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        self.sampling_offsets.bias.data = grid_init.view(-1)
        _constant_init(self.attention_weights, val=0.0, bias=0.0)
        _xavier_init(self.value_proj, distribution="uniform", bias=0.0)
        _xavier_init(self.output_proj, distribution="uniform", bias=0.0)

    def forward(
        self,
        query,
        key=None,
        value=None,
        identity=None,
        query_pos=None,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        **kwargs,
    ):
        if value is None:
            value = torch.cat([query, query], 0)
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        # batch_first=True in this build's usage, so no permute here (upstream's
        # `not self.batch_first` branch is dead for the reference config, which sets
        # `attn_cfgs[index]['batch_first'] = self.batch_first` from `TPVFormerLayer(batch_first=True)`)
        bs, num_query, _ = query.shape
        _, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value
        assert self.num_tpv_queue == 2

        query = torch.cat([value[:bs], query], -1)
        value = self.value_proj(value)
        value = value.reshape(self.num_tpv_queue * bs, num_value, self.num_heads, -1)

        sampling_offsets = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.view(
            bs, num_query, self.num_heads, self.num_tpv_queue, self.num_levels, self.num_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_tpv_queue, self.num_levels * self.num_points
        )
        attention_weights = attention_weights.softmax(-1)
        attention_weights = attention_weights.view(
            bs, num_query, self.num_heads, self.num_tpv_queue, self.num_levels, self.num_points
        )

        attention_weights = (
            attention_weights.permute(3, 0, 1, 2, 4, 5)
            .reshape(
                bs * self.num_tpv_queue, num_query, self.num_heads, self.num_levels, self.num_points
            )
            .contiguous()
        )
        sampling_offsets = sampling_offsets.permute(3, 0, 1, 2, 4, 5, 6).reshape(
            bs * self.num_tpv_queue, num_query, self.num_heads, self.num_levels, self.num_points, 2
        )

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        else:
            raise ValueError(
                f"Last dim of reference_points must be 2, got {reference_points.shape[-1]}"
            )

        output = multi_scale_deformable_attn_pytorch(
            value, spatial_shapes, sampling_locations, attention_weights
        )
        # output shape (bs*num_tpv_queue, num_query, embed_dims)
        output = output.permute(1, 2, 0)
        output = (output[..., :bs] + output[..., bs:]) / self.num_tpv_queue
        output = output.permute(2, 0, 1)
        output = self.output_proj(output)
        return self.dropout(output) + identity


# ---------------------------------------------------------------------------
# tpvformer04/modules/image_cross_attention.py::TPVMSDeformableAttention3D,
# TPVImageCrossAttention
# ---------------------------------------------------------------------------


class TPVMSDeformableAttention3D(nn.Module):
    """An attention module used in TPVFormer based on Deformable-Detr."""

    def __init__(
        self,
        embed_dims=256,
        num_heads=8,
        num_levels=4,
        num_points=(8, 64, 64),
        num_z_anchors=(4, 32, 32),
        im2col_step=64,
        dropout=0.1,
        batch_first=True,
        floor_sampling_offset=False,
        tpv_h=None,
        tpv_w=None,
        tpv_z=None,
    ):
        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(
                f"embed_dims must be divisible by num_heads, but got {embed_dims} and {num_heads}"
            )
        self.batch_first = batch_first
        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = list(num_points)
        self.num_z_anchors = list(num_z_anchors)
        self.base_num_points = num_points[0]
        self.base_z_anchors = num_z_anchors[0]
        self.points_multiplier = [p // self.base_z_anchors for p in num_z_anchors]
        self.tpv_h, self.tpv_w, self.tpv_z = tpv_h, tpv_w, tpv_z
        self.sampling_offsets = nn.ModuleList(
            [nn.Linear(embed_dims, num_heads * num_levels * num_points[i] * 2) for i in range(3)]
        )
        self.floor_sampling_offset = floor_sampling_offset
        self.attention_weights = nn.ModuleList(
            [nn.Linear(embed_dims, num_heads * num_levels * num_points[i]) for i in range(3)]
        )
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.init_weights()

    def init_weights(self):
        for i in range(3):
            _constant_init(self.sampling_offsets[i], 0.0)
            thetas = torch.arange(self.num_heads, dtype=torch.float32) * (
                2.0 * math.pi / self.num_heads
            )
            grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
            grid_init = (
                (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
                .view(self.num_heads, 1, 1, 2)
                .repeat(1, self.num_levels, self.num_points[i], 1)
            )
            grid_init = grid_init.reshape(
                self.num_heads, self.num_levels, self.num_z_anchors[i], -1, 2
            )
            for j in range(self.num_points[i] // self.num_z_anchors[i]):
                grid_init[:, :, :, j, :] *= j + 1
            self.sampling_offsets[i].bias.data = grid_init.reshape(-1)
            _constant_init(self.attention_weights[i], val=0.0, bias=0.0)
        _xavier_init(self.value_proj, distribution="uniform", bias=0.0)
        _xavier_init(self.output_proj, distribution="uniform", bias=0.0)

    def get_sampling_offsets_and_attention(self, queries):
        offsets = []
        attns = []
        for i, (query, fc, attn) in enumerate(
            zip(queries, self.sampling_offsets, self.attention_weights)
        ):
            bs, seq_len, d = query.shape
            offset = fc(query).reshape(
                bs, seq_len, self.num_heads, self.num_levels, self.points_multiplier[i], -1, 2
            )
            offset = offset.permute(0, 1, 4, 2, 3, 5, 6).flatten(1, 2)
            offsets.append(offset)
            attention = attn(query).reshape(bs, seq_len, self.num_heads, -1)
            attention = attention.softmax(-1)
            attention = attention.view(
                bs, seq_len, self.num_heads, self.num_levels, self.points_multiplier[i], -1
            )
            attention = attention.permute(0, 1, 4, 2, 3, 5).flatten(1, 2)
            attns.append(attention)
        offsets = torch.cat(offsets, dim=1)
        attns = torch.cat(attns, dim=1)
        return offsets, attns

    def reshape_reference_points(self, reference_points):
        reference_point_list = []
        for i, reference_point in enumerate(reference_points):
            bs, seq_len, z_anchors, _ = reference_point.shape
            reference_point = reference_point.reshape(bs, seq_len, self.points_multiplier[i], -1, 2)
            reference_point = reference_point.flatten(1, 2)
            reference_point_list.append(reference_point)
        return torch.cat(reference_point_list, dim=1)

    def reshape_output(self, output, lens):
        bs, _, d = output.shape
        outputs = torch.split(
            output,
            [
                lens[0] * self.points_multiplier[0],
                lens[1] * self.points_multiplier[1],
                lens[2] * self.points_multiplier[2],
            ],
            dim=1,
        )
        outputs = [
            o.reshape(bs, -1, self.points_multiplier[i], d).sum(dim=2)
            for i, o in enumerate(outputs)
        ]
        return outputs

    def forward(
        self,
        query,
        key=None,
        value=None,
        identity=None,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        **kwargs,
    ):
        if value is None:
            value = query
        if identity is None:
            identity = query
        if not self.batch_first:
            query = [q.permute(1, 0, 2) for q in query]
            value = value.permute(1, 0, 2)

        query_lens = [q.shape[1] for q in query]
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        value = value.view(bs, num_value, self.num_heads, -1)

        sampling_offsets, attention_weights = self.get_sampling_offsets_and_attention(query)
        reference_points = self.reshape_reference_points(reference_points)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            bs, num_query, num_Z_anchors, xy = reference_points.shape
            reference_points = reference_points[:, :, None, None, :, None, :]
            sampling_offsets = sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            bs, num_query, num_heads, num_levels, num_all_points, xy = sampling_offsets.shape
            sampling_offsets = sampling_offsets.view(
                bs,
                num_query,
                num_heads,
                num_levels,
                num_Z_anchors,
                num_all_points // num_Z_anchors,
                xy,
            )
            sampling_locations = reference_points + sampling_offsets
            bs, num_query, num_heads, num_levels, num_points, num_Z_anchors, xy = (
                sampling_locations.shape
            )
            assert num_all_points == num_points * num_Z_anchors
            sampling_locations = sampling_locations.view(
                bs, num_query, num_heads, num_levels, num_all_points, xy
            )
            if self.floor_sampling_offset:
                sampling_locations = sampling_locations - torch.floor(sampling_locations)
        else:
            raise ValueError(
                f"Last dim of reference_points must be 2, got {reference_points.shape[-1]}"
            )

        output = multi_scale_deformable_attn_pytorch(
            value, spatial_shapes, sampling_locations, attention_weights
        )
        output = self.reshape_output(output, query_lens)
        if not self.batch_first:
            output = [o.permute(1, 0, 2) for o in output]
        return output


class TPVImageCrossAttention(nn.Module):
    """Image cross attention module used in TPVFormer."""

    def __init__(
        self,
        embed_dims=256,
        num_cams=6,
        dropout=0.1,
        batch_first=False,
        deformable_attention=None,
        tpv_h=None,
        tpv_w=None,
        tpv_z=None,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.deformable_attention = deformable_attention
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.batch_first = batch_first
        self.tpv_h, self.tpv_w, self.tpv_z = tpv_h, tpv_w, tpv_z
        self.init_weight()

    def init_weight(self):
        _xavier_init(self.output_proj, distribution="uniform", bias=0.0)

    def forward(
        self,
        query,
        key,
        value,
        residual=None,
        spatial_shapes=None,
        reference_points_cams=None,
        tpv_masks=None,
        level_start_index=None,
        **kwargs,
    ):
        if key is None:
            key = query
        if value is None:
            value = key
        if residual is None:
            inp_residual = query

        bs, num_query, _ = query.size()
        queries = torch.split(
            query,
            [self.tpv_h * self.tpv_w, self.tpv_z * self.tpv_h, self.tpv_w * self.tpv_z],
            dim=1,
        )
        if residual is None:
            slots = [torch.zeros_like(q) for q in queries]
        indexeses = []
        max_lens = []
        queries_rebatches = []
        reference_points_rebatches = []
        for tpv_idx, tpv_mask in enumerate(tpv_masks):
            indexes = []
            for _, mask_per_img in enumerate(tpv_mask):
                index_query_per_img = mask_per_img[0].sum(-1).nonzero().squeeze(-1)
                indexes.append(index_query_per_img)
            max_len = max(1, max(len(each) for each in indexes))
            max_lens.append(max_len)
            indexeses.append(indexes)

            reference_points_cam = reference_points_cams[tpv_idx]
            D = reference_points_cam.size(3)

            queries_rebatch = queries[tpv_idx].new_zeros(
                [bs * self.num_cams, max_len, self.embed_dims]
            )
            reference_points_rebatch = reference_points_cam.new_zeros(
                [bs * self.num_cams, max_len, D, 2]
            )

            for i, reference_points_per_img in enumerate(reference_points_cam):
                for j in range(bs):
                    index_query_per_img = indexes[i]
                    if len(index_query_per_img) > 0:
                        queries_rebatch[j * self.num_cams + i, : len(index_query_per_img)] = (
                            queries[tpv_idx][j, index_query_per_img]
                        )
                        reference_points_rebatch[
                            j * self.num_cams + i, : len(index_query_per_img)
                        ] = reference_points_per_img[j, index_query_per_img]

            queries_rebatches.append(queries_rebatch)
            reference_points_rebatches.append(reference_points_rebatch)

        num_cams, seq_len, bs, embed_dims = key.shape
        key = key.permute(0, 2, 1, 3).reshape(self.num_cams * bs, seq_len, self.embed_dims)
        value = value.permute(0, 2, 1, 3).reshape(self.num_cams * bs, seq_len, self.embed_dims)

        queries = self.deformable_attention(
            query=queries_rebatches,
            key=key,
            value=value,
            reference_points=reference_points_rebatches,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        )

        for tpv_idx, indexes in enumerate(indexeses):
            for i, index_query_per_img in enumerate(indexes):
                for j in range(bs):
                    if len(index_query_per_img) > 0:
                        slots[tpv_idx][j, index_query_per_img] += queries[tpv_idx][
                            j * self.num_cams + i, : len(index_query_per_img)
                        ]

            count = tpv_masks[tpv_idx].sum(-1) > 0
            count = count.permute(1, 2, 0).sum(-1)
            count = torch.clamp(count, min=1.0)
            slots[tpv_idx] = slots[tpv_idx] / count[..., None]
        slots = torch.cat(slots, dim=1)
        slots = self.output_proj(slots)
        return self.dropout(slots) + inp_residual


# ---------------------------------------------------------------------------
# tpvformer04/modules/tpvformer_layer.py::TPVFormerLayer
# ---------------------------------------------------------------------------


class TPVFormerLayer(nn.Module):
    """Base transformer layer for TPVFormer: self_attn(hybrid cross-view) -> norm ->
    cross_attn(image cross attn) -> norm -> ffn -> norm."""

    def __init__(
        self,
        hybrid_attn,
        image_attn,
        feedforward_channels=1024,
        ffn_dropout=0.1,
        embed_dims=256,
        operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
    ):
        super().__init__()
        self.operation_order = operation_order
        self.embed_dims = embed_dims
        self.attentions = nn.ModuleList([hybrid_attn, image_attn])
        self.ffns = nn.ModuleList(
            [
                FFN(
                    embed_dims=embed_dims,
                    feedforward_channels=feedforward_channels,
                    ffn_drop=ffn_dropout,
                )
            ]
        )
        num_norms = operation_order.count("norm")
        self.norms = nn.ModuleList([nn.LayerNorm(embed_dims) for _ in range(num_norms)])

    def forward(
        self,
        query,
        key=None,
        value=None,
        tpv_pos=None,
        ref_2d=None,
        tpv_h=None,
        tpv_w=None,
        tpv_z=None,
        reference_points_cams=None,
        tpv_masks=None,
        spatial_shapes=None,
        level_start_index=None,
        **kwargs,
    ):
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query

        for layer in self.operation_order:
            if layer == "self_attn":
                query_0 = self.attentions[attn_index](
                    query[0],
                    None,
                    None,
                    identity[0] if isinstance(identity, (list, tuple)) else None,
                    query_pos=tpv_pos[0],
                    reference_points=ref_2d,
                    spatial_shapes=torch.tensor([[tpv_h, tpv_w]], device=query[0].device),
                    level_start_index=torch.tensor([0], device=query[0].device),
                    **kwargs,
                )
                attn_index += 1
                query = torch.cat([query_0, query[1], query[2]], dim=1)
                identity = query
            elif layer == "norm":
                query = self.norms[norm_index](query)
                norm_index += 1
            elif layer == "cross_attn":
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    None,
                    reference_points_cams=reference_points_cams,
                    tpv_masks=tpv_masks,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    **kwargs,
                )
                attn_index += 1
                identity = query
            elif layer == "ffn":
                query = self.ffns[ffn_index](query, None)
                ffn_index += 1
        query = torch.split(query, [tpv_h * tpv_w, tpv_z * tpv_h, tpv_w * tpv_z], dim=1)
        return query


# ---------------------------------------------------------------------------
# tpvformer04/modules/encoder.py::TPVFormerEncoder
# ---------------------------------------------------------------------------


class TPVFormerEncoder(nn.Module):
    """Attention with both self (cross-view hybrid) and cross (image cross) attention."""

    def __init__(
        self,
        layers,
        tpv_h,
        tpv_w,
        tpv_z,
        pc_range,
        num_points_in_pillar=(4, 32, 32),
        return_intermediate=False,
    ):
        super().__init__()
        self.layers = layers
        self.return_intermediate = return_intermediate
        self.tpv_h, self.tpv_w, self.tpv_z = tpv_h, tpv_w, tpv_z
        self.num_points_in_pillar = num_points_in_pillar
        assert num_points_in_pillar[1] == num_points_in_pillar[2]
        assert num_points_in_pillar[1] % num_points_in_pillar[0] == 0
        self.pc_range = pc_range

        ref_3d_hw = self.get_reference_points(
            tpv_h, tpv_w, pc_range[5] - pc_range[2], num_points_in_pillar[0], "3d", device="cpu"
        )
        ref_3d_zh = self.get_reference_points(
            tpv_z, tpv_h, pc_range[3] - pc_range[0], num_points_in_pillar[1], "3d", device="cpu"
        )
        ref_3d_zh = ref_3d_zh.permute(3, 0, 1, 2)[[2, 0, 1]]
        ref_3d_zh = ref_3d_zh.permute(1, 2, 3, 0)
        ref_3d_wz = self.get_reference_points(
            tpv_w, tpv_z, pc_range[4] - pc_range[1], num_points_in_pillar[2], "3d", device="cpu"
        )
        ref_3d_wz = ref_3d_wz.permute(3, 0, 1, 2)[[1, 2, 0]]
        ref_3d_wz = ref_3d_wz.permute(1, 2, 3, 0)
        self.register_buffer("ref_3d_hw", ref_3d_hw)
        self.register_buffer("ref_3d_zh", ref_3d_zh)
        self.register_buffer("ref_3d_wz", ref_3d_wz)

        ref_2d_hw = self.get_reference_points(tpv_h, tpv_w, dim="2d", bs=1, device="cpu")
        ref_2d_zh = self.get_reference_points(tpv_z, tpv_h, dim="2d", bs=1, device="cpu")
        ref_2d_wz = self.get_reference_points(tpv_w, tpv_z, dim="2d", bs=1, device="cpu")
        self.register_buffer("ref_2d_hw", ref_2d_hw)
        self.register_buffer("ref_2d_zh", ref_2d_zh)
        self.register_buffer("ref_2d_wz", ref_2d_wz)

    @staticmethod
    def get_reference_points(
        H, W, Z=8, num_points_in_pillar=4, dim="3d", bs=1, device="cpu", dtype=torch.float
    ):
        if dim == "3d":
            zs = (
                torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype, device=device)
                .view(-1, 1, 1)
                .expand(num_points_in_pillar, H, W)
                / Z
            )
            xs = (
                torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device)
                .view(1, 1, -1)
                .expand(num_points_in_pillar, H, W)
                / W
            )
            ys = (
                torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device)
                .view(1, -1, 1)
                .expand(num_points_in_pillar, H, W)
                / H
            )
            ref_3d = torch.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
            return ref_3d
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

    def point_sampling(self, reference_points, pc_range, img_metas):
        lidar2img = [img_meta["lidar2img"] for img_meta in img_metas]
        lidar2img = reference_points.new_tensor(lidar2img)  # (B, N, 4, 4)
        reference_points = reference_points.clone()

        reference_points[..., 0:1] = (
            reference_points[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
        )
        reference_points[..., 1:2] = (
            reference_points[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
        )
        reference_points[..., 2:3] = (
            reference_points[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]
        )

        reference_points = torch.cat(
            (reference_points, torch.ones_like(reference_points[..., :1])), -1
        )
        reference_points = reference_points.permute(1, 0, 2, 3)
        D, B, num_query = reference_points.size()[:3]
        num_cam = lidar2img.size(1)

        reference_points = (
            reference_points.view(D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)
        )
        lidar2img = lidar2img.view(1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)

        reference_points_cam = torch.matmul(
            lidar2img.to(torch.float32), reference_points.to(torch.float32)
        ).squeeze(-1)
        eps = 1e-5
        tpv_mask = reference_points_cam[..., 2:3] > eps
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps
        )

        reference_points_cam[..., 0] /= img_metas[0]["img_shape"][0][1]
        reference_points_cam[..., 1] /= img_metas[0]["img_shape"][0][0]

        tpv_mask = (
            tpv_mask
            & (reference_points_cam[..., 1:2] > 0.0)
            & (reference_points_cam[..., 1:2] < 1.0)
            & (reference_points_cam[..., 0:1] < 1.0)
            & (reference_points_cam[..., 0:1] > 0.0)
        )
        tpv_mask = torch.nan_to_num(tpv_mask)

        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
        tpv_mask = tpv_mask.permute(2, 1, 3, 0, 4).squeeze(-1)
        return reference_points_cam, tpv_mask

    def forward(
        self,
        tpv_query,
        key,
        value,
        *args,
        tpv_h=None,
        tpv_w=None,
        tpv_z=None,
        tpv_pos=None,
        spatial_shapes=None,
        level_start_index=None,
        **kwargs,
    ):
        output = tpv_query
        intermediate = []
        bs = tpv_query[0].shape[0]

        reference_points_cams, tpv_masks = [], []
        ref_3ds = [self.ref_3d_hw, self.ref_3d_zh, self.ref_3d_wz]
        for ref_3d in ref_3ds:
            reference_points_cam, tpv_mask = self.point_sampling(
                ref_3d, self.pc_range, kwargs["img_metas"]
            )
            reference_points_cams.append(reference_points_cam)
            tpv_masks.append(tpv_mask)

        ref_2d_hw = self.ref_2d_hw.clone().expand(bs, -1, -1, -1)
        hybird_ref_2d = torch.cat([ref_2d_hw, ref_2d_hw], 0)

        for lid, layer in enumerate(self.layers):
            output = layer(
                tpv_query,
                key,
                value,
                *args,
                tpv_pos=tpv_pos,
                ref_2d=hybird_ref_2d,
                tpv_h=tpv_h,
                tpv_w=tpv_w,
                tpv_z=tpv_z,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points_cams=reference_points_cams,
                tpv_masks=tpv_masks,
                **kwargs,
            )
            tpv_query = output
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)
        return output


# ---------------------------------------------------------------------------
# tpvformer04/tpv_head.py::TPVFormerHead
# ---------------------------------------------------------------------------


class TPVFormerHead(nn.Module):
    def __init__(
        self,
        positional_encoding,
        tpv_h=30,
        tpv_w=30,
        tpv_z=30,
        pc_range=(-51.2, -51.2, -5, 51.2, 51.2, 3),
        num_feature_levels=4,
        num_cams=6,
        encoder=None,
        embed_dims=256,
    ):
        super().__init__()
        self.tpv_h = tpv_h
        self.tpv_w = tpv_w
        self.tpv_z = tpv_z
        self.pc_range = pc_range
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.real_z = self.pc_range[5] - self.pc_range[2]

        self.positional_encoding = positional_encoding
        tpv_mask_hw = torch.zeros(1, tpv_h, tpv_w)
        self.register_buffer("tpv_mask_hw", tpv_mask_hw)

        self.encoder = encoder
        self.level_embeds = nn.Parameter(torch.Tensor(self.num_feature_levels, self.embed_dims))
        self.cams_embeds = nn.Parameter(torch.Tensor(self.num_cams, self.embed_dims))
        self.tpv_embedding_hw = nn.Embedding(self.tpv_h * self.tpv_w, self.embed_dims)
        self.tpv_embedding_zh = nn.Embedding(self.tpv_z * self.tpv_h, self.embed_dims)
        self.tpv_embedding_wz = nn.Embedding(self.tpv_w * self.tpv_z, self.embed_dims)
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, (TPVMSDeformableAttention3D, TPVCrossViewHybridAttention)):
                m.init_weights()
        nn.init.normal_(self.level_embeds)
        nn.init.normal_(self.cams_embeds)

    def forward(self, mlvl_feats, img_metas):
        bs = mlvl_feats[0].shape[0]
        dtype = mlvl_feats[0].dtype
        device = mlvl_feats[0].device

        tpv_queries_hw = self.tpv_embedding_hw.weight.to(dtype)
        tpv_queries_zh = self.tpv_embedding_zh.weight.to(dtype)
        tpv_queries_wz = self.tpv_embedding_wz.weight.to(dtype)
        tpv_queries_hw = tpv_queries_hw.unsqueeze(0).repeat(bs, 1, 1)
        tpv_queries_zh = tpv_queries_zh.unsqueeze(0).repeat(bs, 1, 1)
        tpv_queries_wz = tpv_queries_wz.unsqueeze(0).repeat(bs, 1, 1)
        tpv_mask_hw = self.tpv_mask_hw.expand(bs, -1, -1)
        tpv_pos_hw = self.positional_encoding(tpv_mask_hw).to(dtype)
        tpv_pos_hw = tpv_pos_hw.flatten(2).transpose(1, 2)

        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2)  # num_cam, bs, hw, c
            feat = feat + self.cams_embeds[:, None, None, :].to(dtype)
            feat = feat + self.level_embeds[None, None, lvl : lvl + 1, :].to(dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 2)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=device)
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )
        feat_flatten = feat_flatten.permute(0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)

        tpv_embed = self.encoder(
            [tpv_queries_hw, tpv_queries_zh, tpv_queries_wz],
            feat_flatten,
            feat_flatten,
            tpv_h=self.tpv_h,
            tpv_w=self.tpv_w,
            tpv_z=self.tpv_z,
            tpv_pos=[tpv_pos_hw, None, None],
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            img_metas=img_metas,
        )
        return tpv_embed


# ---------------------------------------------------------------------------
# tpvformer04/tpv_aggregator.py::TPVAggregator
# ---------------------------------------------------------------------------


class TPVAggregator(nn.Module):
    def __init__(
        self,
        tpv_h,
        tpv_w,
        tpv_z,
        nbr_classes=20,
        in_dims=64,
        hidden_dims=128,
        out_dims=None,
        scale_h=2,
        scale_w=2,
        scale_z=2,
        use_checkpoint=False,
    ):
        super().__init__()
        self.tpv_h = tpv_h
        self.tpv_w = tpv_w
        self.tpv_z = tpv_z
        self.scale_h = scale_h
        self.scale_w = scale_w
        self.scale_z = scale_z
        out_dims = in_dims if out_dims is None else out_dims
        self.decoder = nn.Sequential(
            nn.Linear(in_dims, hidden_dims), nn.Softplus(), nn.Linear(hidden_dims, out_dims)
        )
        self.classifier = nn.Linear(out_dims, nbr_classes)
        self.classes = nbr_classes
        self.use_checkpoint = use_checkpoint

    def forward(self, tpv_list, points=None):
        tpv_hw, tpv_zh, tpv_wz = tpv_list[0], tpv_list[1], tpv_list[2]
        bs, _, c = tpv_hw.shape
        tpv_hw = tpv_hw.permute(0, 2, 1).reshape(bs, c, self.tpv_h, self.tpv_w)
        tpv_zh = tpv_zh.permute(0, 2, 1).reshape(bs, c, self.tpv_z, self.tpv_h)
        tpv_wz = tpv_wz.permute(0, 2, 1).reshape(bs, c, self.tpv_w, self.tpv_z)

        if self.scale_h != 1 or self.scale_w != 1:
            tpv_hw = F.interpolate(
                tpv_hw, size=(self.tpv_h * self.scale_h, self.tpv_w * self.scale_w), mode="bilinear"
            )
        if self.scale_z != 1 or self.scale_h != 1:
            tpv_zh = F.interpolate(
                tpv_zh, size=(self.tpv_z * self.scale_z, self.tpv_h * self.scale_h), mode="bilinear"
            )
        if self.scale_w != 1 or self.scale_z != 1:
            tpv_wz = F.interpolate(
                tpv_wz, size=(self.tpv_w * self.scale_w, self.tpv_z * self.scale_z), mode="bilinear"
            )

        tpv_hw = (
            tpv_hw.unsqueeze(-1)
            .permute(0, 1, 3, 2, 4)
            .expand(-1, -1, -1, -1, self.scale_z * self.tpv_z)
        )
        tpv_zh = (
            tpv_zh.unsqueeze(-1)
            .permute(0, 1, 4, 3, 2)
            .expand(-1, -1, self.scale_w * self.tpv_w, -1, -1)
        )
        tpv_wz = (
            tpv_wz.unsqueeze(-1)
            .permute(0, 1, 2, 4, 3)
            .expand(-1, -1, -1, self.scale_h * self.tpv_h, -1)
        )

        fused = tpv_hw + tpv_zh + tpv_wz
        fused = fused.permute(0, 2, 3, 4, 1)
        if self.use_checkpoint:
            fused = torch.utils.checkpoint.checkpoint(self.decoder, fused)
            logits = torch.utils.checkpoint.checkpoint(self.classifier, fused)
        else:
            fused = self.decoder(fused)
            logits = self.classifier(fused)
        logits = logits.permute(0, 4, 1, 2, 3)
        return logits


# ---------------------------------------------------------------------------
# Image backbone (ResNet, caffe-style stem, DCNv2 at stages 3/4) + FPN neck
# ---------------------------------------------------------------------------


class _ModulatedDeformConv2dPack(nn.Module):
    """mmcv.ops.modulated_deform_conv.ModulatedDeformConv2dPack (DCNv2), transcribed with
    torchvision.ops.DeformConv2d as the underlying op (mmcv's own MLU backend documents this
    substitution as exact); the conv_offset head layout (offset+mask from one 3x3 conv,
    chunked into (o1, o2, mask), mask passed through sigmoid) matches mmcv's forward exactly."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.deform_conv = DeformConv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False
        )
        self.conv_offset = nn.Conv2d(
            in_channels,
            3 * kernel_size * kernel_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        )
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()

    def forward(self, x):
        out = self.conv_offset(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return self.deform_conv(x, offset, mask)


class _Bottleneck(nn.Module):
    """mmcv ResNet Bottleneck block (caffe style: stride on the 3x3 conv), with the 3x3
    conv optionally replaced by DCNv2 (matching `dcn=dict(type='DCNv2', ...)`)."""

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_dcn=False):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        if use_dcn:
            self.conv2 = _ModulatedDeformConv2dPack(
                planes, planes, kernel_size=3, stride=stride, padding=1
            )
        else:
            self.conv2 = nn.Conv2d(
                planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
            )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        return self.relu(out)


class _ResNet(nn.Module):
    """mmcv-style ResNet (caffe stem, frozen_stages=1 semantics dropped for a trainable
    tiny-scale trace) with DCNv2 at `stage_with_dcn` stages, `out_indices` selected."""

    def __init__(
        self,
        layers=(1, 1, 1, 1),
        base_width=16,
        out_indices=(1, 2, 3),
        stage_with_dcn=(False, False, True, True),
    ):
        super().__init__()
        self.inplanes = base_width
        self.conv1 = nn.Conv2d(3, base_width, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(base_width)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        widths = [base_width, base_width * 2, base_width * 4, base_width * 8]
        self.layer1 = self._make_layer(widths[0], layers[0], stride=1, use_dcn=stage_with_dcn[0])
        self.layer2 = self._make_layer(widths[1], layers[1], stride=2, use_dcn=stage_with_dcn[1])
        self.layer3 = self._make_layer(widths[2], layers[2], stride=2, use_dcn=stage_with_dcn[2])
        self.layer4 = self._make_layer(widths[3], layers[3], stride=2, use_dcn=stage_with_dcn[3])
        self.out_indices = out_indices
        self.out_channels = [w * _Bottleneck.expansion for w in widths]

    def _make_layer(self, planes, blocks, stride, use_dcn):
        downsample = None
        if stride != 1 or self.inplanes != planes * _Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * _Bottleneck.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * _Bottleneck.expansion),
            )
        layers = [
            _Bottleneck(
                self.inplanes, planes, stride=stride, downsample=downsample, use_dcn=use_dcn
            )
        ]
        self.inplanes = planes * _Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(_Bottleneck(self.inplanes, planes, use_dcn=use_dcn))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        outs = []
        x = self.layer1(x)
        if 0 in self.out_indices:
            outs.append(x)
        x = self.layer2(x)
        if 1 in self.out_indices:
            outs.append(x)
        x = self.layer3(x)
        if 2 in self.out_indices:
            outs.append(x)
        x = self.layer4(x)
        if 3 in self.out_indices:
            outs.append(x)
        return tuple(outs)


class _FPN(nn.Module):
    """mmdet/models/necks/fpn.py::FPN (open-mmlab/mmdetection @ v2.28.2), transcribed with
    mmcv's `ConvModule` collapsed to plain `nn.Conv2d` (the reference config sets no
    `conv_cfg`/`norm_cfg`/`act_cfg`, so `ConvModule` is a bare conv). `add_extra_convs='on_output'`,
    `relu_before_extra_convs=True` match the reference config."""

    def __init__(
        self,
        in_channels,
        out_channels,
        num_outs,
        start_level=0,
        add_extra_convs="on_output",
        relu_before_extra_convs=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.start_level = start_level
        self.backbone_end_level = self.num_ins
        self.add_extra_convs = add_extra_convs
        self.relu_before_extra_convs = relu_before_extra_convs

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            self.lateral_convs.append(nn.Conv2d(in_channels[i], out_channels, 1))
            self.fpn_convs.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))

        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if extra_levels >= 1:
            for i in range(extra_levels):
                extra_in = (
                    in_channels[-1] if (i == 0 and add_extra_convs == "on_input") else out_channels
                )
                self.fpn_convs.append(nn.Conv2d(extra_in, out_channels, 3, stride=2, padding=1))

    def forward(self, inputs):
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=prev_shape, mode="nearest"
            )

        outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)]
        if self.num_outs > len(outs):
            if self.add_extra_convs == "on_output":
                extra_source = outs[-1]
            else:
                extra_source = laterals[-1]
            outs.append(self.fpn_convs[used_backbone_levels](extra_source))
            for i in range(used_backbone_levels + 1, self.num_outs):
                if self.relu_before_extra_convs:
                    outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                else:
                    outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)


# ---------------------------------------------------------------------------
# tpvformer04/tpvformer.py::TPVFormer (top-level model)
# ---------------------------------------------------------------------------


class TPVFormer(nn.Module):
    def __init__(self, img_backbone, img_neck, tpv_head, tpv_aggregator, num_cams):
        super().__init__()
        self.img_backbone = img_backbone
        self.img_neck = img_neck
        self.tpv_head = tpv_head
        self.tpv_aggregator = tpv_aggregator
        self.num_cams = num_cams

    def extract_img_feat(self, img):
        B, N, C, H, W = img.size()
        img = img.reshape(B * N, C, H, W)
        img_feats = self.img_backbone(img)
        img_feats = self.img_neck(img_feats)
        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, N, C, H, W))
        return img_feats_reshaped

    def forward(self, img, img_metas):
        img_feats = self.extract_img_feat(img)
        outs = self.tpv_head(img_feats, img_metas)
        outs = self.tpv_aggregator(outs)
        return outs


# ---------------------------------------------------------------------------
# Tiny-scale build
# ---------------------------------------------------------------------------


def _make_img_metas(num_cams, img_h, img_w):
    lidar2img = [torch.eye(4, dtype=torch.float32) for _ in range(num_cams)]
    for i, m in enumerate(lidar2img):
        # small offset per camera so tpv_masks are non-degenerate across cameras
        m[0, 3] = 0.1 * i
        m[2, 3] = 5.0
    return [
        {
            "lidar2img": [m.numpy().tolist() for m in lidar2img],
            "img_shape": [(img_h, img_w, 3)] * num_cams,
        }
    ]


def build_tpvformer():
    embed_dims = 16
    num_heads = 2
    num_levels = 4
    num_cams = 2
    tpv_h, tpv_w, tpv_z = 6, 6, 4
    num_points_in_pillar = [2, 4, 4]
    num_points = [2, 4, 4]
    nbr_class = 5
    pc_range = [-4.0, -4.0, -2.0, 4.0, 4.0, 2.0]

    pos_enc = LearnedPositionalEncoding(
        num_feats=embed_dims // 2, row_num_embed=tpv_h, col_num_embed=tpv_w
    )

    layers = nn.ModuleList()
    for _ in range(1):
        hybrid_attn = TPVCrossViewHybridAttention(
            embed_dims=embed_dims, num_heads=num_heads, num_levels=1, num_points=2
        )
        img_deform_attn = TPVMSDeformableAttention3D(
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_points=num_points,
            num_z_anchors=num_points_in_pillar,
            num_levels=num_levels,
            floor_sampling_offset=False,
            tpv_h=tpv_h,
            tpv_w=tpv_w,
            tpv_z=tpv_z,
        )
        image_attn = TPVImageCrossAttention(
            embed_dims=embed_dims,
            num_cams=num_cams,
            deformable_attention=img_deform_attn,
            tpv_h=tpv_h,
            tpv_w=tpv_w,
            tpv_z=tpv_z,
        )
        layers.append(
            TPVFormerLayer(
                hybrid_attn,
                image_attn,
                feedforward_channels=embed_dims * 2,
                ffn_dropout=0.0,
                embed_dims=embed_dims,
            )
        )

    encoder = TPVFormerEncoder(
        layers,
        tpv_h=tpv_h,
        tpv_w=tpv_w,
        tpv_z=tpv_z,
        pc_range=pc_range,
        num_points_in_pillar=num_points_in_pillar,
        return_intermediate=False,
    )

    tpv_head = TPVFormerHead(
        positional_encoding=pos_enc,
        tpv_h=tpv_h,
        tpv_w=tpv_w,
        tpv_z=tpv_z,
        pc_range=pc_range,
        num_feature_levels=num_levels,
        num_cams=num_cams,
        encoder=encoder,
        embed_dims=embed_dims,
    )

    tpv_aggregator = TPVAggregator(
        tpv_h=tpv_h,
        tpv_w=tpv_w,
        tpv_z=tpv_z,
        nbr_classes=nbr_class,
        in_dims=embed_dims,
        hidden_dims=embed_dims * 2,
        out_dims=embed_dims,
        scale_h=1,
        scale_w=1,
        scale_z=1,
        use_checkpoint=False,
    )

    img_backbone = _ResNet(
        layers=(1, 1, 1, 1),
        base_width=4,
        out_indices=(1, 2, 3),
        stage_with_dcn=(False, False, True, True),
    )
    img_neck = _FPN(
        in_channels=img_backbone.out_channels[1:],
        out_channels=embed_dims,
        num_outs=num_levels,
        start_level=0,
        add_extra_convs="on_output",
        relu_before_extra_convs=True,
    )

    return TPVFormer(
        img_backbone=img_backbone,
        img_neck=img_neck,
        tpv_head=tpv_head,
        tpv_aggregator=tpv_aggregator,
        num_cams=num_cams,
    )


def example_input_tpvformer():
    num_cams = 2
    img_h, img_w = 64, 64
    img = torch.randn(1, num_cams, 3, img_h, img_w)
    img_metas = _make_img_metas(num_cams, img_h, img_w)
    return (img, img_metas)


MENAGERIE_ZOO = "ported-pytorch"
MENAGERIE_ENTRIES = [
    ("TPVFormer", "build_tpvformer", "example_input_tpvformer", 2023, "ported"),
]
