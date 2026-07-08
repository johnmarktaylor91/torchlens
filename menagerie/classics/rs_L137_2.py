# FAITHFUL PORT of zehuichen123/AutoAlignV2 @ main (original framework: mmdet3d/mmcv registry
# 3D detector; the fusion mechanism below is transcribed torch code, not a paper gist)
# https://github.com/zehuichen123/AutoAlignV2
# Files transcribed (real math kept verbatim, only mmcv/mmdet3d registry scaffolding removed):
#   https://raw.githubusercontent.com/zehuichen123/AutoAlignV2/main/mmdet3d/models/fusion_layers/deform_point_fusion_v2.py
#   https://raw.githubusercontent.com/zehuichen123/AutoAlignV2/main/mmdet3d/models/fusion_layers/deform_layer.py
#   https://raw.githubusercontent.com/zehuichen123/AutoAlignV2/main/ops/modules/ms_deform_attn.py
#   https://raw.githubusercontent.com/zehuichen123/AutoAlignV2/main/ops/functions/ms_deform_attn_func.py
#     (`ms_deform_attn_core_pytorch`, the real CUDA-free reference path shipped in the same repo)
#
# Chen et al. 2022 (ECCV/IJCAI) "AutoAlignV2: Deformable Feature Aggregation for Dynamic
# Multi-Modal 3D Object Detection". AutoAlignV2's full system is a LiDAR-voxel + camera-image
# 3D detector (mmdet3d `MVXTwoStageDetector` subclass) whose LiDAR branch needs a sparse-conv
# voxel backbone (`spconv`, custom CUDA) -- not portable in a base torch/torchvision/timm env.
# The paper's actual architectural CONTRIBUTION, however, is the camera<-LiDAR fusion module
# itself (`DeformPointFusionV2`): for every LiDAR point, project it into each camera image
# level, treat the projected 2D location as a deformable-attention *reference point*, and let
# `DeformTransLayer` (built on `MSDeformAttn`, the Deformable-DETR multi-scale deformable
# attention op) look up and aggregate a *learned neighborhood* of image features around that
# projection -- rather than a single bilinearly-sampled pixel (AutoAlignV1) or a fixed k x k
# window. That fusion module's `forward(img_feats, pts, pts_feats, img_metas)` signature
# already takes pre-extracted per-point LiDAR features as a plain tensor (the voxel backbone
# lives upstream of this module, not inside it), so it is faithfully portable on its own with
# a plain per-point feature tensor standing in for the (CUDA-only) voxel-backbone output.
#
# Mechanical changes made ONLY to strip framework scaffolding / unavailable CUDA ops:
#   - `DeformPointFusionV2(BaseModule)` -> `nn.Module`; dropped `@FUSION_LAYERS.register_module()`
#     and the `init_cfg`-based Xavier init (kept default nn.Linear/Conv2d init instead).
#   - `mmcv.cnn.ConvModule` (conv+norm+act) lateral convs -> a plain `nn.Sequential(Conv2d,
#     BatchNorm2d, ReLU)` (identical op composition to the default `ConvModule` config used
#     here: conv_cfg=None, norm_cfg=None -> BN, act_cfg=None -> ReLU is what upstream configs
#     for this fusion layer pass).
#   - `get_reference_feats`/`get_proj_mat_by_coord_type`/`apply_3d_transformation`/
#     `points_cam2img` (mmdet3d dataset/coord-frame utilities that convert LiDAR points to
#     camera pixel coordinates via a stored per-sample calibration matrix) are replaced by a
#     single explicit pinhole projection + `F.grid_sample`-based `get_reference_feats`, doing
#     the exact same "project 3D point -> normalized image grid coords -> sample" job with one
#     fixed 3x4 projection matrix per call instead of mmdet3d's `img_meta`-driven multi-stage
#     scale/crop/flip bookkeeping (that bookkeeping is dataset augmentation plumbing, not part
#     of the fusion architecture itself).
#   - `DeformTransLayer.forward` / `MSDeformAttn.forward` bodies are copied verbatim from
#     `deform_layer.py` / `ms_deform_attn.py`. The one substitution: `MSDeformAttnFunction.apply`
#     (the compiled CUDA extension) is swapped for `ms_deform_attn_core_pytorch`, which is the
#     literal alternate code path already present in the same upstream file
#     (`ops/functions/ms_deform_attn_func.py`, labeled "for debug and test only, need to use
#     cuda version instead" -- i.e. the real CUDA-free reference implementation Deformable-DETR
#     ships for exactly this situation), not a rewritten approximation.
#   - `obtain_mlvl_feats`/`sample_single` are copied verbatim (module-level orchestration of
#     per-sample, per-level feature sampling + concatenation), only substituting the mmdet3d
#     `img_meta` projection lookup for the explicit fixed-matrix `get_reference_feats` above.

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------------------
# ops/functions/ms_deform_attn_func.py -- ms_deform_attn_core_pytorch (verbatim; this is
# the real CUDA-free fallback the upstream repo itself ships, not a rewrite)
# --------------------------------------------------------------------------------------
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


# --------------------------------------------------------------------------------------
# ops/modules/ms_deform_attn.py -- MSDeformAttn (verbatim, CUDA function swapped for the
# pure-pytorch core above)
# --------------------------------------------------------------------------------------
def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError(f"invalid input for _is_power_of_2: {n} (type: {type(n)})")
    return (n & (n - 1) == 0) and n != 0


class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4, version="v1"):
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

        plus_ratio = 1
        if version == "v2":
            plus_ratio = 2
        self.sampling_offsets = nn.Linear(d_model * plus_ratio, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model * plus_ratio, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        import math

        nn.init.constant_(self.sampling_offsets.weight.data, 0.0)
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
        nn.init.constant_(self.attention_weights.weight.data, 0.0)
        nn.init.constant_(self.attention_weights.bias.data, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight.data)
        nn.init.constant_(self.value_proj.bias.data, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight.data)
        nn.init.constant_(self.output_proj.bias.data, 0.0)

    def forward(
        self,
        query,
        reference_points,
        input_flatten,
        input_spatial_shapes,
        input_level_start_index,
        input_padding_mask=None,
    ):
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

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
        output = self.output_proj(output)
        return output


# --------------------------------------------------------------------------------------
# mmdet3d/models/fusion_layers/deform_layer.py -- DeformTransLayer (verbatim)
# --------------------------------------------------------------------------------------
def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


class DeformTransLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation="relu",
        n_levels=4,
        n_heads=4,
        n_points=8,
        light=True,
        norm=True,
        version="v1",
    ):
        super().__init__()
        self.light = light
        self.norm = norm
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points, version)
        self.dropout1 = nn.Dropout(dropout)
        if self.norm:
            self.norm1 = nn.LayerNorm(d_model)

        if self.light is False:
            self.linear1 = nn.Linear(d_model, d_ffn)
            self.activation = _get_activation_fn(activation)
            self.dropout2 = nn.Dropout(dropout)
            self.linear2 = nn.Linear(d_ffn, d_model)
            self.dropout3 = nn.Dropout(dropout)
            self.norm2 = nn.LayerNorm(d_model)

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(
        self,
        src_feat,
        query_feat,
        reference_points,
        key_feat,
        spatial_shapes,
        level_start_index,
        padding_mask=None,
    ):
        src2 = self.self_attn(
            query_feat, reference_points, key_feat, spatial_shapes, level_start_index, padding_mask
        )
        src_feat = src_feat + self.dropout1(src2)
        if self.norm:
            src_feat = self.norm1(src_feat)
        if self.light is False:
            src_feat = self.forward_ffn(src_feat)
        return src_feat


# --------------------------------------------------------------------------------------
# mmdet3d/models/fusion_layers/deform_point_fusion_v2.py -- point<-image reference
# sampling + DeformPointFusionV2 (verbatim math; mmdet3d img_meta coord-transform chain
# replaced by an explicit fixed pinhole projection matrix)
# --------------------------------------------------------------------------------------
def get_reference_feats(
    img_features,
    points,
    proj_mat,
    img_pad_shape,
    aligned=True,
    padding_mode="zeros",
    align_corners=True,
):
    """Real mechanism from upstream `get_reference_feats`: project LiDAR points into the
    image plane with a projection matrix, then `F.grid_sample` the image feature map at
    those (normalized) locations. Simplified to a single fixed 3x4 projection matrix
    instead of mmdet3d's per-sample `img_meta` scale/crop/flip chain (that bookkeeping is
    dataset augmentation, not part of the fusion architecture)."""
    ones = points.new_ones(points.shape[0], 1)
    points_h = torch.cat([points, ones], dim=-1)  # [N, 4]
    pts_2d = points_h @ proj_mat.t()  # [N, 3]
    pts_2d = pts_2d[:, :2] / pts_2d[:, 2:3].clamp(min=1e-6)

    coor_x, coor_y = torch.split(pts_2d, 1, dim=1)
    h, w = img_pad_shape
    coor_y = coor_y / h * 2 - 1
    coor_x = coor_x / w * 2 - 1
    grid = torch.cat([coor_x, coor_y], dim=1).unsqueeze(0).unsqueeze(0)  # [1, 1, N, 2]

    mode = "bilinear" if aligned else "nearest"
    point_features = F.grid_sample(
        img_features, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners
    )
    return point_features.squeeze(2).squeeze(0).t(), points.new_tensor((grid + 1) / 2)


class DeformPointFusionV2(nn.Module):
    """Fuse image features from multi-scale features with LiDAR point features via
    deformable cross-attention (the real AutoAlignV2 contribution)."""

    def __init__(
        self,
        img_channels,
        pts_channels,
        mid_channels,
        out_channels,
        img_levels=1,
        n_heads=4,
        n_points=4,
        activate_out=True,
        fuse_out=False,
        dropout_ratio=0.0,
        aligned=True,
        align_corners=True,
        padding_mode="zeros",
        lateral_conv=True,
    ):
        super(DeformPointFusionV2, self).__init__()
        if isinstance(img_levels, int):
            img_levels = [img_levels]
        if isinstance(img_channels, int):
            img_channels = [img_channels] * len(img_levels)

        self.img_levels = img_levels
        self.activate_out = activate_out
        self.fuse_out = fuse_out
        self.dropout_ratio = dropout_ratio
        self.img_channels = img_channels
        self.aligned = aligned
        self.align_corners = align_corners
        self.padding_mode = padding_mode
        self.mid_channels = mid_channels
        self.n_heads = n_heads
        self.n_points = n_points

        self.lateral_convs = None
        if lateral_conv:
            self.lateral_convs = nn.ModuleList()
            for i in range(len(img_channels)):
                l_conv = nn.Sequential(
                    nn.Conv2d(img_channels[i], mid_channels, 3, padding=1),
                    nn.BatchNorm2d(mid_channels),
                    nn.ReLU(inplace=False),
                )
                self.lateral_convs.append(l_conv)
            self.img_transform = nn.Sequential(
                nn.Linear(mid_channels * len(img_channels), out_channels),
                nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            )
        else:
            self.img_transform = nn.Sequential(
                nn.Linear(sum(img_channels), out_channels),
                nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            )
        self.img_key_proj = nn.Sequential(
            nn.Linear(mid_channels, mid_channels),
            nn.BatchNorm1d(mid_channels, eps=1e-3, momentum=0.01),
        )
        self.pts_transform = nn.Sequential(
            nn.Linear(pts_channels, out_channels),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        )
        self.deform_layers = nn.ModuleList()
        for _ in self.img_levels:
            deform_layer = DeformTransLayer(
                d_model=mid_channels, n_levels=1, n_heads=self.n_heads, n_points=self.n_points
            )
            self.deform_layers.append(deform_layer)

        if self.fuse_out:
            self.fuse_conv = nn.Sequential(
                nn.Linear(2 * out_channels, out_channels),
                nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=False),
            )

    def forward(self, img_feats, pts, pts_feats, proj_mat, img_pad_shape):
        img_pts = self.obtain_mlvl_feats(img_feats, pts, pts_feats, proj_mat, img_pad_shape)
        img_pre_fuse = self.img_transform(img_pts)
        if self.training and self.dropout_ratio > 0:
            img_pre_fuse = F.dropout(img_pre_fuse, self.dropout_ratio)
        pts_pre_fuse = self.pts_transform(pts_feats)

        fuse_out = torch.cat([img_pre_fuse, pts_pre_fuse], dim=-1)
        if self.activate_out:
            fuse_out = F.relu(fuse_out)
        if self.fuse_out:
            fuse_out = self.fuse_conv(fuse_out)
        return fuse_out

    def obtain_mlvl_feats(self, img_feats, pts, pts_feats, proj_mat, img_pad_shape):
        if self.lateral_convs is not None:
            img_ins = [
                lateral_conv(img_feats[i])
                for i, lateral_conv in zip(self.img_levels, self.lateral_convs)
            ]
        else:
            img_ins = img_feats
        mlvl_img_feats = []
        for level in range(len(self.img_levels)):
            mlvl_img_feats.append(
                self.sample_single(
                    img_ins[level], pts[:, :3], pts_feats, proj_mat, img_pad_shape, level_num=level
                )
            )
        img_pts = torch.cat(mlvl_img_feats, dim=-1)
        return img_pts

    def sample_single(self, img_feats, pts, pts_feats, proj_mat, img_pad_shape, level_num):
        ref_feats, ref_points = get_reference_feats(
            img_features=img_feats,
            points=pts,
            proj_mat=proj_mat,
            img_pad_shape=img_pad_shape,
            aligned=self.aligned,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        )
        bs, channel_num, h, w = img_feats.shape
        ref_feats = self.img_key_proj(ref_feats).unsqueeze(0)
        flatten_img_feat = img_feats.permute(0, 2, 3, 1).reshape(bs, h * w, channel_num)
        ref_points = ref_points.reshape(bs, -1, 1, 2)

        N, Len_in, _ = flatten_img_feat.shape
        level_spatial_shapes = pts_feats.new_tensor([(h, w)], dtype=torch.long)
        level_start_index = pts_feats.new_tensor([0], dtype=torch.long)
        # NOTE: upstream `sample_single` calls `self.deform_layers[level_num](ref_feats,
        # ref_points, flatten_img_feat, level_spatial_shapes, level_start_index)` -- only 5
        # positional args against `DeformTransLayer.forward(src_feat, query_feat,
        # reference_points, key_feat, spatial_shapes, level_start_index, padding_mask=None)`'s
        # 6 required params, and `DeformTransLayer.forward` in turn calls
        # `self.self_attn(query_feat, reference_points, key_feat, spatial_shapes,
        # level_start_index, padding_mask)` i.e. `MSDeformAttn.forward(query=query_feat,
        # reference_points=reference_points, input_flatten=key_feat,
        # input_spatial_shapes=spatial_shapes, input_level_start_index=level_start_index)`. The
        # upstream 5-positional call leaves `level_start_index` unfilled (a real bug -- raises
        # TypeError, exactly what this port hit when transcribed verbatim positionally). We
        # therefore call by keyword with each tensor bound to its semantically-intended
        # role -- src_feat/query_feat=ref_feats (per-point projected features, both the
        # attention query and the residual-add target, matching every other DeformTransLayer
        # call site in the repo), reference_points=ref_points (normalized image-plane
        # locations), key_feat=flatten_img_feat (the image feature map MSDeformAttn samples
        # from), spatial_shapes=level_spatial_shapes, level_start_index=level_start_index. This
        # is a wiring fix for a dead code path (restoring the intended data flow), not an
        # architecture change -- every module/op is exactly as upstream.
        img_pts = self.deform_layers[level_num](
            src_feat=ref_feats,
            query_feat=ref_feats,
            reference_points=ref_points,
            key_feat=flatten_img_feat,
            spatial_shapes=level_spatial_shapes,
            level_start_index=level_start_index,
        ).squeeze(0)
        invalid_idx = (
            (ref_points[:, :, :, 0] >= 0)
            & (ref_points[:, :, :, 1] >= 0)
            & (ref_points[:, :, :, 0] <= 1)
            & (ref_points[:, :, :, 1] <= 1)
        ).squeeze()
        img_pts = img_pts * invalid_idx.unsqueeze(-1).float()
        return img_pts


def build_autoalignv2():
    model = DeformPointFusionV2(
        img_channels=32,
        pts_channels=16,
        mid_channels=16,
        out_channels=16,
        img_levels=0,
        n_heads=2,
        n_points=4,
        fuse_out=True,
    )
    model.eval()
    return model


def example_input_autoalignv2():
    img_feats = [torch.randn(1, 32, 8, 8)]
    n_pts = 12
    pts = torch.rand(n_pts, 3) * 4.0
    pts_feats = torch.randn(n_pts, 16)
    proj_mat = torch.tensor([[6.0, 0.0, 4.0, 0.0], [0.0, 6.0, 4.0, 0.0], [0.0, 0.0, 1.0, 4.0]])
    img_pad_shape = (8, 8)
    return (img_feats, pts, pts_feats, proj_mat, img_pad_shape)


MENAGERIE_ZOO = "ported-pytorch"
MENAGERIE_ENTRIES = [
    ("AutoAlignV2", "build_autoalignv2", "example_input_autoalignv2", 2022, "ported"),
]
