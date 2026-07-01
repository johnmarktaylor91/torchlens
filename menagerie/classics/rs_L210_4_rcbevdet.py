# SOURCE: vendored from VDIGPKU/RCBEVDet @ main (rcbevdet-master.zip, released
# 2024-06-01 per README "Update" section)
# (mmdet3d/models/backbones/radar_encoder.py)
"""RCBEVDet's RadarBEVNet: dual-stream radar-BEV feature-extraction backbone.

RCBEVDet (CVPR 2024, radar-camera fusion 3D detection in bird's-eye view)
gates its official code release behind a docx "application" note in the
README, but that note is struck through (``~~...~~``) as of the 2024-06-01
"Code for RCBEVDet is released in the zip file" update -- the code is
publicly downloadable from the repo's own
``rcbevdet-master.zip`` (fetched directly, no signing required).

``RadarBEVNet`` is the paper's headline architectural contribution -- "a
dual-stream radar backbone... a point-based encoder and a transformer-based
encoder... with an injection and extraction module to facilitate
communication between the two encoders" (repo README). It is copied
verbatim from ``mmdet3d/models/backbones/radar_encoder.py``: the pillar-style
point encoder (``RFNLayer``, matching PointPillars' PFN layer), the
point-embedding stream (``PointEmbed``), the cross-attention
injector/extractor pair that lets the point stream and the pillar stream
exchange information (``Injector``/``Extractor``/``CrossAttention``), the
self-attention transformer block with a distance-modulated attention bias
(``SelfAttentionBlock``/``DMSA``), and the top-level ``RadarBEVNet`` module
tying them together exactly as ``forward`` does (alternating point-block
processing with injector/extractor exchange and a transformer refinement
step per point_block layer).

Only the mmcv/mmdet3d *registration plumbing* is swapped for its literal
torch equivalent (no architecture change):
  - ``@BACKBONES.register_module()`` (mmdet3d registry decorator) is
    dropped -- registration only, not part of the module.
  - ``build_norm_layer(dict(type="BN1d", eps=1e-3, momentum=0.01), units)[1]``
    (``RFNLayer.norm``) is replaced by the literal
    ``nn.BatchNorm1d(units, eps=1e-3, momentum=0.01)`` it constructs under
    the hood (mmcv's ``BN1d`` registry entry maps directly to
    ``nn.BatchNorm1d``).
  - ``mmcv.cnn.bricks.transformer.MultiheadAttention`` (used by ``DMSA``) is
    a thin wrapper over ``torch.nn.MultiheadAttention`` with an identity/
    residual add (``identity + proj_drop(out)``, no ``query_pos``/
    ``key_pos``/``identity`` override in this call site) -- replaced here by
    ``torch.nn.MultiheadAttention`` plus the same explicit residual add.
  - ``FFN`` is imported from ``mmcv.cnn.bricks.transformer`` in the original
    but never called anywhere in the file (dead import) -- dropped.
  - ``timm.models.layers.{DropPath, Mlp, to_2tuple}`` are used as in the
    original, imported from the non-deprecated ``timm.layers`` path
    (``to_2tuple`` is unused in this file but kept for import-parity).

``forward`` is unchanged end-to-end.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, Mlp


def get_paddings_indicator(actual_num, max_num, axis=0):
    """Create boolean mask by actually number of a padded tensor."""
    actual_num = torch.unsqueeze(actual_num, axis + 1)
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis + 1] = -1
    max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
    paddings_indicator = actual_num.int() > max_num
    return paddings_indicator


class RFNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, norm_cfg=None, last_layer=False):
        """Pillar Feature Net Layer (PointPillars-style PFN layer)."""
        super().__init__()
        self.name = "RFNLayer"
        self.last_vfe = last_layer

        self.units = out_channels

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)
        self.norm_cfg = norm_cfg

        self.linear = nn.Linear(in_channels, self.units, bias=False)
        # build_norm_layer(norm_cfg, self.units)[1] -> nn.BatchNorm1d
        self.norm = nn.BatchNorm1d(self.units, eps=norm_cfg["eps"], momentum=norm_cfg["momentum"])

    def forward(self, inputs):
        x = self.linear(inputs)
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        x = F.relu(x)

        if self.last_vfe:
            x_max = torch.max(x, dim=1, keepdim=True)[0]
            return x_max
        else:
            return x


class PointEmbed(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_c, out_c, 1),
            nn.BatchNorm1d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_c, out_c, 1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_c * 2, out_c * 2, 1),
            nn.BatchNorm1d(out_c * 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_c * 2, out_c, 1),
        )

    def forward(self, points):
        bs, n, c = points.shape
        feature = self.conv1(points.transpose(2, 1))  # bs c n
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # bs c 1

        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)  # bs c*2 n
        feature = self.conv2(feature)

        return feature.transpose(2, 1)


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, c):
        B, N, C = x.shape
        kv = self.kv(c).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Extractor(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=6,
        n_points=4,
        n_levels=1,
        deform_ratio=1.0,
        with_cffn=True,
        cffn_ratio=0.25,
        drop=0.0,
        drop_path=0.0,
        norm_layer=lambda d: nn.LayerNorm(d, eps=1e-6),
        with_cp=False,
    ):
        super().__init__()
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = CrossAttention(dim, num_heads, qkv_bias=False, attn_drop=drop, proj_drop=drop)
        self.with_cffn = with_cffn
        self.with_cp = with_cp
        if with_cffn:
            self.ffn = Mlp(
                in_features=dim, hidden_features=int(dim * cffn_ratio), act_layer=nn.GELU, drop=drop
            )
            self.ffn_norm = norm_layer(dim)
            self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, query, feat):
        def _inner_forward(query, feat):
            attn = self.attn(self.query_norm(query), self.feat_norm(feat))
            query = query + attn

            query = self.drop_path(self.ffn(self.ffn_norm(query)))
            return query

        query = _inner_forward(query, feat)
        return query


class Injector(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=6,
        n_points=4,
        n_levels=1,
        deform_ratio=1.0,
        norm_layer=lambda d: nn.LayerNorm(d, eps=1e-6),
        init_values=0.0,
        with_cp=False,
        drop=0.0,
    ):
        super().__init__()
        self.with_cp = with_cp
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = CrossAttention(dim, num_heads, qkv_bias=False, attn_drop=drop, proj_drop=drop)
        self.gamma = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, query, feat):
        def _inner_forward(query, feat):
            attn = self.attn(self.query_norm(query), self.feat_norm(feat))
            return self.gamma * attn

        query = _inner_forward(query, feat)
        return query


class DMSA(nn.Module):
    """Distance-Modulated Self-Attention. mmcv's ``MultiheadAttention`` here
    is a thin wrapper over ``torch.nn.MultiheadAttention`` with an identity
    residual add and no query_pos/key_pos/identity override at this call
    site -- reproduced explicitly below."""

    def __init__(self, embed_dims=256, num_heads=8, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dims, num_heads, dropout=dropout, batch_first=True
        )
        self.attn_proj_drop = nn.Dropout(0.0)
        self.beta = nn.Linear(embed_dims, num_heads)

    @torch.no_grad()
    def init_weights(self):
        nn.init.zeros_(self.beta.weight)
        nn.init.uniform_(self.beta.bias, 0.0, 2.0)

    def inner_forward(self, query_bbox, query_feat, pre_attn_mask):
        dist = self.center_dists(query_bbox)
        beta = self.beta(query_feat)

        beta = beta.permute(0, 2, 1)
        attn_mask = dist[:, None, :, :] * beta[..., None]
        if pre_attn_mask is not None:
            attn_mask[:, :, pre_attn_mask] = float("-inf")
        attn_mask = attn_mask.flatten(0, 1)

        # mmcv MultiheadAttention.forward: key=value=query, identity=query,
        # dataflow left batch_first (batch_first=True passed through).
        out = self.attention(
            query=query_feat, key=query_feat, value=query_feat, attn_mask=attn_mask
        )[0]
        return query_feat + self.attn_proj_drop(out)

    def forward(self, query_bbox, query_feat, pre_attn_mask=None):
        return self.inner_forward(query_bbox, query_feat, pre_attn_mask)

    @torch.no_grad()
    def center_dists(self, points):
        centers = points[..., :2]
        dist = []
        for b in range(centers.shape[0]):
            dist_b = torch.norm(centers[b].reshape(-1, 1, 2) - centers[b].reshape(1, -1, 2), dim=-1)
            dist.append(dist_b[None, ...])

        dist = torch.cat(dist, dim=0)
        dist = -dist

        return dist


class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=6,
        n_points=4,
        n_levels=1,
        deform_ratio=1.0,
        with_cffn=True,
        cffn_ratio=0.25,
        drop=0.0,
        drop_path=0.0,
        norm_layer=lambda d: nn.LayerNorm(d, eps=1e-6),
        with_cp=False,
    ):
        super().__init__()
        self.query_norm = norm_layer(dim)
        self.attn = DMSA(dim, num_heads, dropout=drop)
        self.with_cffn = with_cffn
        self.with_cp = with_cp
        if with_cffn:
            self.ffn = Mlp(
                in_features=dim, hidden_features=int(dim * 2), act_layer=nn.GELU, drop=drop
            )
            self.ffn_norm = norm_layer(dim)
            self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, feat, points):
        def _inner_forward(feat, points):
            identity = feat
            feat = self.query_norm(feat)
            feat = self.attn(points, feat)
            feat = feat + identity

            feat = self.drop_path(self.ffn(self.ffn_norm(feat)))
            return feat

        query = _inner_forward(feat, points)
        return query


class RadarBEVNet(nn.Module):
    """Dual-stream radar BEV feature-extraction backbone (RCBEVDet's
    headline architectural contribution)."""

    def __init__(
        self,
        in_channels=4,
        feat_channels=(64,),
        with_distance=False,
        voxel_size=(0.2, 0.2, 4),
        point_cloud_range=(0, -40, -3, 70.4, 40, 1),
        norm_cfg=None,
        with_pos_embed=False,
        return_rcs=False,
        drop=0.0,
    ):
        super().__init__()
        self.return_rcs = return_rcs
        assert len(feat_channels) > 0

        self.in_channels = in_channels
        in_channels = in_channels + 2
        self._with_distance = with_distance

        # Create PillarFeatureNet layers
        feat_channels = [in_channels] + list(feat_channels)
        point_block = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            last_layer = False
            point_block.append(
                RFNLayer(in_filters, out_filters, norm_cfg=norm_cfg, last_layer=last_layer)
            )
        self.point_block = nn.ModuleList(point_block)

        num_heads = 2
        extractor = []
        for i in range(1, len(feat_channels)):
            extractor.append(
                Extractor(
                    feat_channels[i], num_heads=num_heads, cffn_ratio=1, drop=drop, drop_path=drop
                )
            )
        self.extractor = nn.ModuleList(extractor)

        injector = []
        for i in range(1, len(feat_channels)):
            injector.append(Injector(feat_channels[i], num_heads=num_heads, drop=drop))
        self.injector = nn.ModuleList(injector)

        transformer_block = []
        for i in range(1, len(feat_channels)):
            transformer_block.append(
                SelfAttentionBlock(
                    feat_channels[i], num_heads=num_heads, cffn_ratio=1, drop=drop, drop_path=drop
                )
            )
        self.transformer_block = nn.ModuleList(transformer_block)

        linear_module = []
        for i in range(1, len(feat_channels) - 1):
            linear_module.append(nn.Linear(feat_channels[i], feat_channels[i + 1]))
        self.linear_module = nn.ModuleList(linear_module)

        self.out_linear = nn.Linear(feat_channels[-1] * 2, feat_channels[-1])

        # Need pillar (voxel) size and x/y offset in order to calculate pillar offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.pc_range = point_cloud_range

        if with_pos_embed:
            embed_dims = feat_channels[1]
            self.pos_embed = nn.Sequential(
                nn.Linear(3, embed_dims),
                nn.LayerNorm(embed_dims),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dims, embed_dims),
                nn.LayerNorm(embed_dims),
                nn.ReLU(inplace=True),
            )
        self.with_pos_embed = with_pos_embed

        self.point_embed = PointEmbed(in_channels + 2, feat_channels[1])

    def compress(self, x):
        x = x.max(dim=1)[0]
        x = x.unsqueeze(dim=0)
        return x

    def forward(self, features, num_voxels, coors):
        dtype = features.dtype
        f_center = torch.zeros_like(features[:, :, :2])
        f_center[:, :, 0] = features[:, :, 0] - (
            coors[:, 1].to(dtype).unsqueeze(1) * self.vx + self.x_offset
        )
        f_center[:, :, 1] = features[:, :, 1] - (
            coors[:, 2].to(dtype).unsqueeze(1) * self.vy + self.y_offset
        )

        # normalize x,y,z to [0, 1]
        features = features.clone()
        features[:, :, 0:1] = (features[:, :, 0:1] - self.pc_range[0]) / (
            self.pc_range[3] - self.pc_range[0]
        )
        features[:, :, 1:2] = (features[:, :, 1:2] - self.pc_range[1]) / (
            self.pc_range[4] - self.pc_range[1]
        )
        features[:, :, 2:3] = (features[:, :, 2:3] - self.pc_range[2]) / (
            self.pc_range[5] - self.pc_range[2]
        )

        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_voxels, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)

        features_mean = torch.zeros_like(features[:, :, :2])

        features_mean[:, :, 0] = features[:, :, 0] - (
            (features[:, :, 0] * mask.squeeze()).sum(dim=1) / mask.squeeze().sum(dim=1)
        ).unsqueeze(1)
        features_mean[:, :, 1] = features[:, :, 1] - (
            (features[:, :, 1] * mask.squeeze()).sum(dim=1) / mask.squeeze().sum(dim=1)
        ).unsqueeze(1)

        rcs_features = features.clone()
        c = torch.cat([features, features_mean, f_center], dim=-1)
        x = torch.cat([features, f_center], dim=-1)

        # The feature decorations were calculated without regard to whether pillar was empty. Need to ensure that
        # empty pillars remain set to zeros.
        x = x * mask
        c = c * mask

        c = self.point_embed(c)
        if self.with_pos_embed:
            c = c + self.pos_embed(features[:, :, 0:3])
        points_coors = features[:, :, 0:3].detach()

        batch_size = coors[-1, 0] + 1
        if batch_size > 1:
            bs_list = [0]
            bs_info = coors[:, 0]
            pre = bs_info[0]
            for i in range(1, len(bs_info)):
                if pre != bs_info[i]:
                    bs_list.append(i)
                    pre = bs_info[i]
            bs_list.append(len(bs_info))
            bs_list = [bs_list[i + 1] - bs_list[i] for i in range(len(bs_list) - 1)]
        elif batch_size == 1:
            bs_list = [len(coors[:, 0])]
        else:
            assert False

        points_coors_split = torch.split(points_coors, bs_list)

        i = 0

        for rfn in self.point_block:
            x = rfn(x)
            x_split = torch.split(x, bs_list)
            c_split = torch.split(c, bs_list)

            x_out_list = []
            c_out_list = []
            for bs in range(len(x_split)):
                c_tmp = c_split[bs]
                x_tmp = x_split[bs]
                points_coors_tmp = points_coors_split[bs]
                c_tmp = c_tmp + self.injector[i](
                    self.compress(c_tmp), self.compress(x_tmp)
                ).transpose(1, 0).expand_as(c_tmp)
                x_tmp = x_tmp + self.extractor[i](
                    self.compress(x_tmp), self.compress(c_tmp)
                ).transpose(1, 0).expand_as(x_tmp)
                c_tmp = (
                    self.transformer_block[i](self.compress(c_tmp), self.compress(points_coors_tmp))
                    .transpose(1, 0)
                    .expand_as(c_tmp)
                )
                if i < len(self.point_block) - 1:
                    c_tmp = self.linear_module[i](c_tmp)

                c_out_list.append(c_tmp)
                x_out_list.append(x_tmp)

            x = torch.cat(x_out_list, dim=0)
            c = torch.cat(c_out_list, dim=0)
            i += 1
        c = self.out_linear(torch.cat([c, x], dim=-1))

        c = torch.max(c, dim=1, keepdim=True)[0]
        if not self.return_rcs:
            return c.squeeze()
        else:
            rcs = (rcs_features * mask).sum(dim=1) / mask.sum(dim=1)
            return c.squeeze(), rcs.squeeze()


def build_rcbevdet_radarbevnet() -> nn.Module:
    """Build a tiny RCBEVDet ``RadarBEVNet`` (dual-stream radar backbone).

    Uses a single pillar-feature layer and small channel width so the module
    traces quickly; the architecture (pillar point encoder + transformer
    point-embedding stream + cross-attention injector/extractor exchange +
    distance-modulated self-attention refinement) is unchanged from the real
    ``mmdet3d/models/backbones/radar_encoder.py``.

    Returns
    -------
    nn.Module
        Random-initialized ``RadarBEVNet`` (4 raw radar-point channels,
        64-wide pillar features, matching the repo's default constructor
        args).
    """
    return RadarBEVNet(
        in_channels=4,
        feat_channels=(64,),
        voxel_size=(0.2, 0.2, 4),
        point_cloud_range=(0, -40, -3, 70.4, 40, 1),
        with_pos_embed=False,
        return_rcs=False,
    )


def example_input_rcbevdet_radarbevnet() -> tuple:
    """Return small radar-pillar tensors matching ``build_rcbevdet_radarbevnet``.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ``features``: ``(num_pillars, max_points_per_pillar, in_channels)``
        raw radar-point features per pillar, shape ``(6, 8, 4)``.
        ``num_voxels``: ``(num_pillars,)`` true point count per pillar.
        ``coors``: ``(num_pillars, 4)`` int pillar coordinates
        ``[batch_idx, x, y, z]`` (single batch element, matching
        ``coors[-1, 0] + 1 == 1`` in ``forward``).
    """
    num_pillars, max_points, in_channels = 6, 8, 4
    features = torch.rand(num_pillars, max_points, in_channels) * 10.0
    num_voxels = torch.randint(1, max_points + 1, (num_pillars,))
    coors = torch.zeros(num_pillars, 4, dtype=torch.long)
    coors[:, 1] = torch.randint(0, 50, (num_pillars,))
    coors[:, 2] = torch.randint(0, 50, (num_pillars,))
    return (features, num_voxels, coors)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    (
        "RCBEVDet RadarBEVNet",
        "build_rcbevdet_radarbevnet",
        "example_input_rcbevdet_radarbevnet",
        2024,
        "radarbevnet",
    ),
]
