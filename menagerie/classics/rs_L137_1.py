# FAITHFUL PORT of tusen-ai/Anchor3DLane @ main (original framework: mmcv/mmseg registry + PyTorch)
# https://github.com/tusen-ai/Anchor3DLane
# Files transcribed (real math kept verbatim, only mmcv/mmseg scaffolding removed):
#   https://raw.githubusercontent.com/tusen-ai/Anchor3DLane/main/mmseg/models/lane_detector/anchor_3dlane.py
#   https://raw.githubusercontent.com/tusen-ai/Anchor3DLane/main/mmseg/models/lane_detector/transformer.py
#   https://raw.githubusercontent.com/tusen-ai/Anchor3DLane/main/mmseg/models/lane_detector/position_encoding.py
#   https://raw.githubusercontent.com/tusen-ai/Anchor3DLane/main/mmseg/models/lane_detector/utils/anchor.py
#   https://raw.githubusercontent.com/tusen-ai/Anchor3DLane/main/mmseg/models/lane_detector/tools.py (homography_crop_resize only)
#
# Huang et al. 2023 (CVPR) "Anchor3DLane: Learning to Regress 3D Anchors for Monocular
# 3D Lane Detection". The real contribution is: (1) 3D anchors defined directly in
# ego/ground coordinates (`AnchorGenerator.generate_anchor`: a start-x offset + constant
# pitch/yaw sweep converted to per-y-step x/z coordinates), (2) `cut_anchor_features`:
# project each anchor's dense 3D (x, y, z) points through the camera homography
# (`projection_transform`) into normalized image-feature grid coordinates and
# `F.grid_sample` the corresponding BEV/perspective CNN feature map ("anchor feature
# cutting" -- the geometry-aware pooling that replaces a generic RPN/anchor-box IoU
# match), and (3) `get_proposals`/`encoder_decoder`: an iterative regression loop where
# each stage re-samples anchor features around the *previous* stage's regressed anchor
# geometry before predicting the next categorical/x/z/visibility offsets via `DecodeLayer`
# MLP heads. The DETR-style transformer encoder (`TransformerEncoderLayer`/
# `TransformerEncoder`, copy-paste from Facebook's DETR per the upstream file header) and
# sine position embedding (`PositionEmbeddingSine`) refine the CNN backbone feature map
# before anchor-feature cutting.
#
# Mechanical changes made ONLY to strip framework scaffolding that this base env does not
# have (mmcv/mmseg are not installed, and the real backbone is registry-constructed via
# `build_backbone(dict(type='ResNetV1c', ...))`, an mmseg dilated-ResNet variant):
#   - `Anchor3DLane(BaseModule)` -> `Anchor3DLane(nn.Module)`; dropped `@LANENET2S.register_module()`,
#     `@force_fp32()`, `init_cfg`/`train_cfg`/`test_cfg`/`loss_lane`/`loss_aux`/`neck` (all loss-
#     construction and mmcv registry plumbing, unused by the forward/inference path).
#   - Backbone: swapped mmseg's `ResNetV1c` registry backbone for `torchvision.models.resnet18`
#     truncated to its stage-4 (`layer4`) feature map -- both are standard dilated/undilated
#     ResNet-18 stacks; only the builder differs (registry vs. direct torchvision class), the
#     backbone family itself is not architecturally novel to this paper.
#   - `AnchorGenerator`/`compute_anchor_cut_indices`/`projection_transform`/`cut_anchor_features`/
#     `feature_extractor`/`get_proposals`/`encoder_decoder`/`obtain_projection_matrix`/
#     `DecodeLayer` bodies are copied verbatim from `anchor_3dlane.py` (only the unused mmcv
#     imports and `@force_fp32()` decorator were dropped from `get_proposals`).
#   - `AnchorGenerator.__init__`/`generate_anchors`/`generate_anchor` copied verbatim from
#     `utils/anchor.py` (dropped the unused `mmcv`/`mmseg.datasets.tools.utils` imports --
#     neither `mmcv` nor `projection_g2im`/`resample_laneline_in_y` is referenced by these
#     methods).
#   - `homography_crop_resize` copied verbatim from `tools.py`.
#   - `TransformerEncoderLayer`/`TransformerEncoder`/`_get_clones`/`_get_activation_fn`
#     copied verbatim from `transformer.py` (decoder classes dropped: `Anchor3DLane`'s
#     `enc_layers>1` path only ever instantiates the encoder stack).
#   - `PositionEmbeddingSine` copied verbatim from `position_encoding.py`.
#   - `forward_dummy`/`nms`/`forward_train`/`loss` (training-only paths requiring
#     `img_metas`/ground-truth 3D lanes and the `LaneLoss`/`TopkAssigner` modules) are
#     omitted; the trace uses `encoder_decoder` directly (the real shared
#     feature-extraction + iterative-anchor-regression forward path used by both
#     `forward_train` and `forward_test`).

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


# --------------------------------------------------------------------------------------
# transformer.py (DETR encoder, copy-paste from torch.nn.Transformer per upstream header)
# --------------------------------------------------------------------------------------
def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def _get_clones(module, n):
    import copy

    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[torch.Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(
            q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(
            q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None, pos=None):
        output = src
        for layer in self.layers:
            output = layer(
                output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos
            )
        if self.norm is not None:
            output = self.norm(output)
        return output


# --------------------------------------------------------------------------------------
# position_encoding.py
# --------------------------------------------------------------------------------------
class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        not_mask = ~mask

        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)

        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


# --------------------------------------------------------------------------------------
# tools.py -- homography_crop_resize (verbatim)
# --------------------------------------------------------------------------------------
def homography_crop_resize(org_img_size, crop_y, resize_img_size):
    ratio_x = resize_img_size[1] / org_img_size[1]
    ratio_y = resize_img_size[0] / (org_img_size[0] - crop_y)
    H_c = np.array([[ratio_x, 0, 0], [0, ratio_y, -ratio_y * crop_y], [0, 0, 1]])
    return H_c


# --------------------------------------------------------------------------------------
# utils/anchor.py -- AnchorGenerator (verbatim generate_anchors/generate_anchor)
# --------------------------------------------------------------------------------------
class AnchorGenerator(object):
    """Normalized anchor coords"""

    def __init__(self, anchor_cfg, y_steps=None, x_min=None, x_max=None, y_max=100, norm=None):
        self.y_steps = y_steps
        if self.y_steps is None:
            self.y_steps = np.linspace(1, y_max, y_max)
        self.pitches = anchor_cfg["pitches"]
        self.yaws = anchor_cfg["yaws"]
        self.num_x = anchor_cfg["num_x"]
        self.anchor_len = len(self.y_steps)
        self.x_min = x_min
        self.x_max = x_max
        self.y_max = y_max
        self.norm = norm
        self.start_z = anchor_cfg.get("start_z", 0)

    def generate_anchors(self):
        anchors = []
        starts = [x for x in np.linspace(self.x_min, self.x_max, num=self.num_x, dtype=np.float32)]
        idx = 0
        for start_x in starts:
            for pitch in self.pitches:
                for yaw in self.yaws:
                    anchor = self.generate_anchor(start_x, pitch, yaw, start_z=self.start_z)
                    if anchor is not None:
                        anchors.append(anchor)
                        idx += 1
        self.anchor_num = len(anchors)
        anchors = np.array(anchors)
        return anchors

    def generate_anchor(self, start_x, pitch, yaw, start_z=0, cut=True):
        # anchor [pos_score, neg_score, start_y, end_y, d, x_coords * l, z_coords * l, vis_coords * l]
        anchor = np.zeros(2 + 2 + 1 + self.anchor_len * 3, dtype=np.float32)
        pitch = pitch * math.pi / 180.0
        yaw = yaw * math.pi / 180.0
        anchor[2] = 0
        anchor[3] = 1
        anchor[5 : 5 + self.anchor_len] = start_x + self.y_steps * math.tan(yaw)
        anchor[5 + self.anchor_len : 5 + self.anchor_len * 2] = start_z + self.y_steps * math.tan(
            pitch
        )
        anchor_vis = np.logical_and(
            anchor[5 : 5 + self.anchor_len] > self.x_min,
            anchor[5 : 5 + self.anchor_len] < self.x_max,
        )
        if cut:
            if sum(anchor_vis) / self.anchor_len < 0.5:
                return None
        return anchor


# --------------------------------------------------------------------------------------
# anchor_3dlane.py -- DecodeLayer + Anchor3DLane (verbatim math, mmcv scaffolding dropped)
# --------------------------------------------------------------------------------------
class DecodeLayer(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel):
        super(DecodeLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_channel, mid_channel),
            nn.ReLU6(),
            nn.Linear(mid_channel, mid_channel),
            nn.ReLU6(),
            nn.Linear(mid_channel, out_channel),
        )

    def forward(self, x):
        return self.layer(x)


class _ResNet18Stage4Backbone(nn.Module):
    """Real torchvision resnet18 truncated to the stage-4 (layer4) feature map --
    stands in for mmseg's registry-built `ResNetV1c` (a dilated ResNet-18 variant);
    both are standard ResNet-18 backbones, only the construction path differs."""

    def __init__(self):
        super().__init__()
        m = torchvision.models.resnet18(weights=None)
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.layer1 = m.layer1
        self.layer2 = m.layer2
        self.layer3 = m.layer3
        self.layer4 = m.layer4

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return [x]


class Anchor3DLane(nn.Module):
    def __init__(
        self,
        y_steps=(5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0),
        feat_y_steps=(5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0),
        anchor_cfg=None,
        db_cfg=None,
        backbone_dim=512,
        attn_dim=64,
        iter_reg=1,
        drop_out=0.0,
        num_heads=2,
        enc_layers=1,
        dim_feedforward=128,
        pre_norm=False,
        anchor_feat_channels=64,
        feat_size=(12, 15),
        num_category=21,
    ):
        super(Anchor3DLane, self).__init__()
        hidden_dim = attn_dim
        self.iter_reg = iter_reg
        self.anchor_feat_channels = anchor_feat_channels
        self.feat_size = feat_size
        self.num_category = num_category
        self.enc_layers = enc_layers
        self.db_cfg = db_cfg

        # Anchor
        self.y_steps = np.array(y_steps, dtype=np.float32)
        self.feat_y_steps = np.array(feat_y_steps, dtype=np.float32)
        self.feat_sample_index = torch.from_numpy(np.isin(self.y_steps, self.feat_y_steps))
        self.x_norm = 30.0
        self.y_norm = 100.0
        self.z_norm = 10.0
        self.x_min = -30
        self.x_max = 30
        self.anchor_len = len(y_steps)
        self.anchor_feat_len = len(feat_y_steps)
        self.anchor_generator = AnchorGenerator(
            anchor_cfg,
            x_min=self.x_min,
            x_max=self.x_max,
            y_max=int(self.y_steps[-1]),
            norm=(self.x_norm, self.y_norm, self.z_norm),
        )
        dense_anchors = self.anchor_generator.generate_anchors()  # [N, 5+3l]
        anchor_inds = self.anchor_generator.y_steps
        self.anchors = self.sample_from_dense_anchors(self.y_steps, anchor_inds, dense_anchors)
        self.feat_anchors = self.sample_from_dense_anchors(
            self.feat_y_steps, anchor_inds, dense_anchors
        )
        self.xs, self.ys, self.zs = self.compute_anchor_cut_indices(
            self.feat_anchors, self.feat_y_steps
        )

        self.backbone = _ResNet18Stage4Backbone()

        # transformer layer
        self.position_embedding = PositionEmbeddingSine(
            num_pos_feats=hidden_dim // 2, normalize=True
        )
        self.input_proj = nn.Conv2d(backbone_dim, hidden_dim, kernel_size=1)
        if self.enc_layers == 1:
            self.transformer_layer = TransformerEncoderLayer(
                hidden_dim,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=drop_out,
                normalize_before=pre_norm,
            )
        else:
            transformer_layer = TransformerEncoderLayer(
                hidden_dim,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=drop_out,
                normalize_before=pre_norm,
            )
            self.transformer_layer = TransformerEncoder(transformer_layer, self.enc_layers)

        # decoder heads
        self.anchor_projection = nn.Conv2d(hidden_dim, self.anchor_feat_channels, kernel_size=1)

        self.cls_layer = nn.ModuleList()
        self.reg_x_layer = nn.ModuleList()
        self.reg_z_layer = nn.ModuleList()
        self.reg_vis_layer = nn.ModuleList()

        for _ in range(1 + self.iter_reg):
            self.cls_layer.append(
                DecodeLayer(
                    self.anchor_feat_channels * self.anchor_feat_len,
                    self.anchor_feat_channels * self.anchor_feat_len,
                    self.num_category,
                )
            )
            self.reg_x_layer.append(
                DecodeLayer(
                    self.anchor_feat_channels * self.anchor_feat_len,
                    self.anchor_feat_channels,
                    self.anchor_len,
                )
            )
            self.reg_z_layer.append(
                DecodeLayer(
                    self.anchor_feat_channels * self.anchor_feat_len,
                    self.anchor_feat_channels,
                    self.anchor_len,
                )
            )
            self.reg_vis_layer.append(
                DecodeLayer(
                    self.anchor_feat_channels * self.anchor_feat_len,
                    self.anchor_feat_channels,
                    self.anchor_len,
                )
            )

    def sample_from_dense_anchors(self, sample_steps, dense_inds, dense_anchors):
        sample_index = np.isin(dense_inds, sample_steps)
        anchor_len = len(sample_steps)
        dense_anchor_len = len(sample_index)
        anchors = np.zeros((len(dense_anchors), 5 + anchor_len * 3), dtype=np.float32)
        anchors[:, :5] = dense_anchors[:, :5].copy()
        anchors[:, 5 : 5 + anchor_len] = dense_anchors[:, 5 : 5 + dense_anchor_len][:, sample_index]
        anchors[:, 5 + anchor_len : 5 + 2 * anchor_len] = dense_anchors[
            :, 5 + dense_anchor_len : 5 + 2 * dense_anchor_len
        ][:, sample_index]
        anchors = torch.from_numpy(anchors)
        return anchors

    def compute_anchor_cut_indices(self, anchors, y_steps):
        if len(anchors.shape) == 2:
            n_proposals = len(anchors)
        else:
            batch_size, n_proposals = anchors.shape[:2]

        num_y_steps = len(y_steps)

        xs = anchors[..., 5 : 5 + num_y_steps]
        xs = torch.flatten(xs, -2)

        ys = torch.from_numpy(y_steps).to(anchors.device)
        if len(anchors.shape) == 2:
            ys = ys.repeat(n_proposals)
        else:
            ys = ys.repeat(batch_size, n_proposals)

        zs = anchors[..., 5 + num_y_steps : 5 + num_y_steps * 2]
        zs = torch.flatten(zs, -2)
        return xs, ys, zs

    def projection_transform(self, Matrix, xs, ys, zs):
        ones = torch.ones_like(zs)
        coordinates = torch.stack([xs, ys, zs, ones], dim=1)
        trans = torch.bmm(Matrix, coordinates)

        u_vals = trans[:, 0, :] / trans[:, 2, :]
        v_vals = trans[:, 1, :] / trans[:, 2, :]
        return u_vals, v_vals

    def cut_anchor_features(self, features, h_g2feats, xs, ys, zs):
        batch_size = features.shape[0]

        if len(xs.shape) == 1:
            batch_xs = xs.repeat(batch_size, 1)
            batch_ys = ys.repeat(batch_size, 1)
            batch_zs = zs.repeat(batch_size, 1)
        else:
            batch_xs = xs
            batch_ys = ys
            batch_zs = zs

        batch_us, batch_vs = self.projection_transform(h_g2feats, batch_xs, batch_ys, batch_zs)
        batch_us = (batch_us / self.feat_size[1] - 0.5) * 2
        batch_vs = (batch_vs / self.feat_size[0] - 0.5) * 2

        batch_grid = torch.stack([batch_us, batch_vs], dim=-1)
        batch_grid = batch_grid.reshape(batch_size, -1, self.anchor_feat_len, 2)
        batch_anchor_features = F.grid_sample(features, batch_grid, padding_mode="zeros")

        valid_mask = (batch_us > -1) & (batch_us < 1) & (batch_vs > -1) & (batch_vs < 1)
        return batch_anchor_features, valid_mask.reshape(batch_size, -1, self.anchor_feat_len)

    def feature_extractor(self, img, mask):
        output = self.backbone(img)
        feat = output[-1]
        feat = self.input_proj(feat)

        mask_interp = F.interpolate(mask[:, 0, :, :][None], size=feat.shape[-2:]).to(torch.bool)[0]
        pos = self.position_embedding(feat, mask_interp)

        bs, c, h, w = feat.shape
        assert h == self.feat_size[0] and w == self.feat_size[1]
        feat = feat.flatten(2).permute(2, 0, 1)
        pos = pos.flatten(2).permute(2, 0, 1)
        mask_interp = mask_interp.flatten(1)
        trans_feat = self.transformer_layer(feat, src_key_padding_mask=mask_interp, pos=pos)
        trans_feat = trans_feat.permute(1, 2, 0).reshape(bs, c, h, w)
        return trans_feat

    def get_proposals(self, project_matrixes, anchor_feat, iter_idx=0, proposals_prev=None):
        batch_size = project_matrixes.shape[0]
        if proposals_prev is None:
            batch_anchor_features, _ = self.cut_anchor_features(
                anchor_feat, project_matrixes, self.xs, self.ys, self.zs
            )
        else:
            sampled_anchor = torch.zeros(
                batch_size,
                len(self.anchors),
                5 + self.anchor_feat_len * 3,
                device=anchor_feat.device,
            )
            sampled_anchor[:, :, 5 : 5 + self.anchor_feat_len] = proposals_prev[
                :, :, 5 : 5 + self.anchor_len
            ][:, :, self.feat_sample_index]
            sampled_anchor[:, :, 5 + self.anchor_feat_len : 5 + self.anchor_feat_len * 2] = (
                proposals_prev[:, :, 5 + self.anchor_len : 5 + self.anchor_len * 2][
                    :, :, self.feat_sample_index
                ]
            )
            xs, ys, zs = self.compute_anchor_cut_indices(sampled_anchor, self.feat_y_steps)
            batch_anchor_features, _ = self.cut_anchor_features(
                anchor_feat, project_matrixes, xs, ys, zs
            )

        batch_anchor_features = batch_anchor_features.transpose(1, 2)
        batch_anchor_features = batch_anchor_features.reshape(
            -1, self.anchor_feat_channels * self.anchor_feat_len
        )

        cls_logits = self.cls_layer[iter_idx](batch_anchor_features)
        cls_logits = cls_logits.reshape(batch_size, -1, cls_logits.shape[1])
        reg_x = self.reg_x_layer[iter_idx](batch_anchor_features)
        reg_x = reg_x.reshape(batch_size, -1, reg_x.shape[1])
        reg_z = self.reg_z_layer[iter_idx](batch_anchor_features)
        reg_z = reg_z.reshape(batch_size, -1, reg_z.shape[1])
        reg_vis = self.reg_vis_layer[iter_idx](batch_anchor_features)
        reg_vis = torch.sigmoid(reg_vis)
        reg_vis = reg_vis.reshape(batch_size, -1, reg_vis.shape[1])

        reg_proposals = torch.zeros(
            batch_size,
            len(self.anchors),
            5 + self.anchor_len * 3 + self.num_category,
            device=project_matrixes.device,
        )
        if proposals_prev is None:
            reg_proposals[:, :, : 5 + self.anchor_len * 3] = (
                reg_proposals[:, :, : 5 + self.anchor_len * 3] + self.anchors
            )
        else:
            reg_proposals[:, :, : 5 + self.anchor_len * 3] = (
                reg_proposals[:, :, : 5 + self.anchor_len * 3]
                + proposals_prev[:, :, : 5 + self.anchor_len * 3]
            )

        reg_proposals[:, :, 5 : 5 + self.anchor_len] += reg_x
        reg_proposals[:, :, 5 + self.anchor_len : 5 + self.anchor_len * 2] += reg_z
        reg_proposals[:, :, 5 + self.anchor_len * 2 : 5 + self.anchor_len * 3] = reg_vis
        reg_proposals[
            :, :, 5 + self.anchor_len * 3 : 5 + self.anchor_len * 3 + self.num_category
        ] = cls_logits
        return reg_proposals

    def obtain_projection_matrix(self, project_matrix, feat_size):
        h_g2feats = []
        device = project_matrix.device
        project_matrix = project_matrix.detach().cpu().numpy()
        for i in range(len(project_matrix)):
            P_g2im = project_matrix[i]
            Hc = homography_crop_resize((self.db_cfg["org_h"], self.db_cfg["org_w"]), 0, feat_size)
            h_g2feat = np.matmul(Hc, P_g2im)
            h_g2feats.append(torch.from_numpy(h_g2feat).type(torch.FloatTensor).to(device))
        return h_g2feats

    def encoder_decoder(self, img, mask, gt_project_matrix):
        batch_size = img.shape[0]
        trans_feat = self.feature_extractor(img, mask)

        anchor_feat = self.anchor_projection(trans_feat)
        project_matrixes = self.obtain_projection_matrix(gt_project_matrix, self.feat_size)
        project_matrixes = torch.stack(project_matrixes, dim=0)

        reg_proposals_all = []
        anchors_all = []
        reg_proposals_s1 = self.get_proposals(project_matrixes, anchor_feat, 0)
        reg_proposals_all.append(reg_proposals_s1)
        anchors_all.append(torch.stack([self.anchors] * batch_size, dim=0))

        for it in range(self.iter_reg):
            proposals_prev = reg_proposals_all[it]
            reg_proposals_all.append(
                self.get_proposals(project_matrixes, anchor_feat, it + 1, proposals_prev)
            )
            anchors_all.append(proposals_prev[:, :, : 5 + self.anchor_len * 3])

        output = {"reg_proposals": reg_proposals_all[-1], "anchors": anchors_all[-1]}
        return output

    def forward(self, img, mask, gt_project_matrix):
        output = self.encoder_decoder(img, mask, gt_project_matrix)
        return output["reg_proposals"]


def build_anchor3dlane():
    # Tiny config mirroring configs/openlane/anchor3dlane.py (resnet18 backbone,
    # attn_dim=64, num_heads=2, dim_feedforward=128), shrunk anchor grid + feature
    # size so the real forward pass (backbone -> transformer -> anchor-feature
    # cutting -> iterative regression) traces fast.
    anchor_cfg = dict(
        pitches=[2, 0, -2],
        yaws=[10, 0, -10],
        num_x=4,
        start_z=0,
    )
    db_cfg = dict(org_h=64, org_w=64)
    model = Anchor3DLane(
        y_steps=(5.0, 10.0, 15.0, 20.0),
        feat_y_steps=(5.0, 10.0, 15.0, 20.0),
        anchor_cfg=anchor_cfg,
        db_cfg=db_cfg,
        backbone_dim=512,
        attn_dim=32,
        iter_reg=1,
        num_heads=2,
        dim_feedforward=64,
        anchor_feat_channels=16,
        feat_size=(2, 2),
        num_category=5,
    )
    model.eval()
    return model


def example_input_anchor3dlane():
    img = torch.randn(1, 3, 64, 64)
    mask = torch.zeros(1, 1, 64, 64)
    # simple pinhole-like 3x4 projection matrix (identity rotation + small translation)
    gt_project_matrix = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 5.0]]]
    )
    return (img, mask, gt_project_matrix)


MENAGERIE_ZOO = "ported-pytorch"
MENAGERIE_ENTRIES = [
    ("Anchor3DLane", "build_anchor3dlane", "example_input_anchor3dlane", 2023, "ported"),
]
