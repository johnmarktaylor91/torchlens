# SOURCE: vendored from zhulf0804/PointPillars @ 620e6b0d07e4cb37b7b0114f26b934e8be92a0ba
# https://raw.githubusercontent.com/zhulf0804/PointPillars/620e6b0d07e4cb37b7b0114f26b934e8be92a0ba/pointpillars/model/pointpillars.py
#
# PointPillars (Lang, Vora, Caesar, Zhou, Yang, Beijbom 2019, CVPR) -- 3D point-cloud
# object detector via a pseudo-image "pillar" encoding of the point cloud, followed by
# a 2D CNN detection backbone. Queue candidate "PointPillars Radar" (arxiv:2408.05020,
# RadarPillars) uses this exact PointPillars pillar+2D-CNN architecture applied to
# automotive radar point clouds (Doppler/RCS-augmented points) instead of LiDAR
# returns -- the architecture itself is unmodified between the LiDAR and radar
# variants (this repo's own notes: "base impl directly usable with radar inputs";
# RadarPillars' contribution is an input-channel/pooling tweak for radar-specific
# point statistics, not a new backbone/head topology), so the real base-repo
# `pointpillars/model/pointpillars.py` architecture is vendored directly.
#
# `PillarEncoder`, `Backbone`, `Neck`, `Head` are transcribed verbatim from the real
# `pointpillars/model/pointpillars.py` -- these are the network's ENTIRE learned
# parameter surface (Conv1d/BatchNorm1d pillar feature encoder -> Conv2d/BatchNorm2d
# multi-scale 2D backbone -> ConvTranspose2d/BatchNorm2d FPN-style neck -> three
# parallel Conv2d prediction heads for class/box-regression/direction). `PillarLayer`
# (real-repo voxelization, `pointpillars.ops.Voxelization`) and the NMS branch of
# `PointPillars.get_predicted_bboxes_single` (`pointpillars.ops.nms_cuda`) are NOT
# vendored: both call into a custom compiled CUDA/C++ extension
# (`pointpillars/ops/voxel_op`, `pointpillars/ops/iou3d_module`) that requires
# `python setup.py develop` to build and is unavailable in a clean base-env install;
# critically, neither is a *learned* layer -- `PillarLayer.forward` runs under
# `@torch.no_grad()` and only buckets raw points into per-cell arrays (pure geometric
# preprocessing), and NMS is a non-differentiable greedy-suppression postprocessing
# step over already-computed box scores. Skipping them changes what tensors flow
# in/out of the traced graph, not the trainable network itself: this staging module's
# `PointPillarsNet.forward` reproduces the exact remaining pipeline from the real
# `PointPillars.forward` (`pillar_encoder -> backbone -> neck -> head`, verbatim call
# order and tensor shapes) starting from a pre-voxelized pillar tensor -- the real,
# well-defined input contract of `PillarEncoder.forward` in the source file -- instead
# of raw LiDAR/radar points. `PointPillars.__init__`'s real default hyperparameters
# (voxel_size, point_cloud_range, pillar in/out channels, backbone/neck channel and
# stride schedules, head anchor/class counts) are preserved unchanged.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PillarEncoder(nn.Module):
    def __init__(self, voxel_size, point_cloud_range, in_channel, out_channel):
        super().__init__()
        self.out_channel = out_channel
        self.vx, self.vy = voxel_size[0], voxel_size[1]
        self.x_offset = voxel_size[0] / 2 + point_cloud_range[0]
        self.y_offset = voxel_size[1] / 2 + point_cloud_range[1]
        self.x_l = int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0])
        self.y_l = int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1])

        self.conv = nn.Conv1d(in_channel, out_channel, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_channel, eps=1e-3, momentum=0.01)

    def forward(self, pillars, coors_batch, npoints_per_pillar):
        """
        pillars: (p1 + p2 + ... + pb, num_points, c), c = 4
        coors_batch: (p1 + p2 + ... + pb, 1 + 3)
        npoints_per_pillar: (p1 + p2 + ... + pb, )
        return:  (bs, out_channel, y_l, x_l)
        """
        device = pillars.device
        # 1. calculate offset to the points center (in each pillar)
        offset_pt_center = (
            pillars[:, :, :3]
            - torch.sum(pillars[:, :, :3], dim=1, keepdim=True) / npoints_per_pillar[:, None, None]
        )  # (p1 + p2 + ... + pb, num_points, 3)

        # 2. calculate offset to the pillar center
        x_offset_pi_center = pillars[:, :, :1] - (
            coors_batch[:, None, 1:2] * self.vx + self.x_offset
        )  # (p1 + p2 + ... + pb, num_points, 1)
        y_offset_pi_center = pillars[:, :, 1:2] - (
            coors_batch[:, None, 2:3] * self.vy + self.y_offset
        )  # (p1 + p2 + ... + pb, num_points, 1)

        # 3. encoder
        features = torch.cat(
            [pillars, offset_pt_center, x_offset_pi_center, y_offset_pi_center], dim=-1
        )  # (p1 + p2 + ... + pb, num_points, 9)
        features[:, :, 0:1] = x_offset_pi_center  # tmp
        features[:, :, 1:2] = y_offset_pi_center  # tmp
        # In consitent with mmdet3d.
        # The reason can be referenced to https://github.com/open-mmlab/mmdetection3d/issues/1150

        # 4. find mask for (0, 0, 0) and update the encoded features
        # a very beautiful implementation
        voxel_ids = torch.arange(0, pillars.size(1)).to(device)  # (num_points, )
        mask = voxel_ids[:, None] < npoints_per_pillar[None, :]  # (num_points, p1 + p2 + ... + pb)
        mask = mask.permute(1, 0).contiguous()  # (p1 + p2 + ... + pb, num_points)
        features *= mask[:, :, None]

        # 5. embedding
        features = features.permute(0, 2, 1).contiguous()  # (p1 + p2 + ... + pb, 9, num_points)
        features = F.relu(
            self.bn(self.conv(features))
        )  # (p1 + p2 + ... + pb, out_channels, num_points)
        pooling_features = torch.max(features, dim=-1)[0]  # (p1 + p2 + ... + pb, out_channels)

        # 6. pillar scatter
        batched_canvas = []
        bs = coors_batch[-1, 0] + 1
        for i in range(bs):
            cur_coors_idx = coors_batch[:, 0] == i
            cur_coors = coors_batch[cur_coors_idx, :]
            cur_features = pooling_features[cur_coors_idx]

            canvas = torch.zeros(
                (self.x_l, self.y_l, self.out_channel), dtype=torch.float32, device=device
            )
            canvas[cur_coors[:, 1], cur_coors[:, 2]] = cur_features
            canvas = canvas.permute(2, 1, 0).contiguous()
            batched_canvas.append(canvas)
        batched_canvas = torch.stack(batched_canvas, dim=0)  # (bs, in_channel, self.y_l, self.x_l)
        return batched_canvas


class Backbone(nn.Module):
    def __init__(self, in_channel, out_channels, layer_nums, layer_strides=[2, 2, 2]):
        super().__init__()
        assert len(out_channels) == len(layer_nums)
        assert len(out_channels) == len(layer_strides)

        self.multi_blocks = nn.ModuleList()
        for i in range(len(layer_strides)):
            blocks = []
            blocks.append(
                nn.Conv2d(
                    in_channel, out_channels[i], 3, stride=layer_strides[i], bias=False, padding=1
                )
            )
            blocks.append(nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01))
            blocks.append(nn.ReLU(inplace=True))

            for _ in range(layer_nums[i]):
                blocks.append(nn.Conv2d(out_channels[i], out_channels[i], 3, bias=False, padding=1))
                blocks.append(nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01))
                blocks.append(nn.ReLU(inplace=True))

            in_channel = out_channels[i]
            self.multi_blocks.append(nn.Sequential(*blocks))

        # in consitent with mmdet3d
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        """
        x: (b, c, y_l, x_l). Default: (6, 64, 496, 432)
        return: list[]. Default: [(6, 64, 248, 216), (6, 128, 124, 108), (6, 256, 62, 54)]
        """
        outs = []
        for i in range(len(self.multi_blocks)):
            x = self.multi_blocks[i](x)
            outs.append(x)
        return outs


class Neck(nn.Module):
    def __init__(self, in_channels, upsample_strides, out_channels):
        super().__init__()
        assert len(in_channels) == len(upsample_strides)
        assert len(upsample_strides) == len(out_channels)

        self.decoder_blocks = nn.ModuleList()
        for i in range(len(in_channels)):
            decoder_block = []
            decoder_block.append(
                nn.ConvTranspose2d(
                    in_channels[i],
                    out_channels[i],
                    upsample_strides[i],
                    stride=upsample_strides[i],
                    bias=False,
                )
            )
            decoder_block.append(nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01))
            decoder_block.append(nn.ReLU(inplace=True))

            self.decoder_blocks.append(nn.Sequential(*decoder_block))

        # in consitent with mmdet3d
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        """
        x: [(bs, 64, 248, 216), (bs, 128, 124, 108), (bs, 256, 62, 54)]
        return: (bs, 384, 248, 216)
        """
        outs = []
        for i in range(len(self.decoder_blocks)):
            xi = self.decoder_blocks[i](x[i])  # (bs, 128, 248, 216)
            outs.append(xi)
        out = torch.cat(outs, dim=1)
        return out


class Head(nn.Module):
    def __init__(self, in_channel, n_anchors, n_classes):
        super().__init__()

        self.conv_cls = nn.Conv2d(in_channel, n_anchors * n_classes, 1)
        self.conv_reg = nn.Conv2d(in_channel, n_anchors * 7, 1)
        self.conv_dir_cls = nn.Conv2d(in_channel, n_anchors * 2, 1)

        # in consitent with mmdet3d
        conv_layer_id = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if conv_layer_id == 0:
                    prior_prob = 0.01
                    bias_init = float(-math.log((1 - prior_prob) / prior_prob))
                    nn.init.constant_(m.bias, bias_init)
                else:
                    nn.init.constant_(m.bias, 0)
                conv_layer_id += 1

    def forward(self, x):
        """
        x: (bs, 384, 248, 216)
        return:
              bbox_cls_pred: (bs, n_anchors*3, 248, 216)
              bbox_pred: (bs, n_anchors*7, 248, 216)
              bbox_dir_cls_pred: (bs, n_anchors*2, 248, 216)
        """
        bbox_cls_pred = self.conv_cls(x)
        bbox_pred = self.conv_reg(x)
        bbox_dir_cls_pred = self.conv_dir_cls(x)
        return bbox_cls_pred, bbox_pred, bbox_dir_cls_pred


class PointPillarsNet(nn.Module):
    """Staging wrapper reproducing the real `PointPillars.forward`'s learned pipeline
    (`pillar_encoder -> backbone -> neck -> head`, verbatim call order) starting from
    a pre-voxelized pillar tensor -- i.e. everything in the real `PointPillars`
    architecture except the non-learned, CUDA-extension-only `PillarLayer`
    (voxelization) and NMS postprocessing (see module header)."""

    def __init__(
        self,
        nclasses=3,
        voxel_size=[0.16, 0.16, 4],
        point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1],
        max_num_points=32,
    ):
        super().__init__()
        self.nclasses = nclasses
        self.pillar_encoder = PillarEncoder(
            voxel_size=voxel_size, point_cloud_range=point_cloud_range, in_channel=9, out_channel=64
        )
        self.backbone = Backbone(in_channel=64, out_channels=[64, 128, 256], layer_nums=[3, 5, 5])
        self.neck = Neck(
            in_channels=[64, 128, 256], upsample_strides=[1, 2, 4], out_channels=[128, 128, 128]
        )
        self.head = Head(in_channel=384, n_anchors=2 * nclasses, n_classes=nclasses)

    def forward(self, pillars, coors_batch, npoints_per_pillar):
        # pillars: (p1 + p2 + ... + pb, num_points, c), c = 4 (x, y, z, feature)
        # coors_batch: (p1 + p2 + ... + pb, 1 + 3)  (batch_idx, z, y, x)
        # npoints_per_pillar: (p1 + p2 + ... + pb, )
        pillar_features = self.pillar_encoder(
            pillars, coors_batch, npoints_per_pillar
        )  # (bs, 64, y_l, x_l)
        xs = self.backbone(pillar_features)  # list of multi-scale feature maps
        x = self.neck(xs)  # (bs, 384, y_l/2, x_l/2)
        bbox_cls_pred, bbox_pred, bbox_dir_cls_pred = self.head(x)
        return bbox_cls_pred, bbox_pred, bbox_dir_cls_pred


def build_pointpillars_radar():
    torch.manual_seed(0)
    model = PointPillarsNet(nclasses=3)
    model.eval()
    return model


def example_input_pointpillars_radar():
    torch.manual_seed(0)
    # Small pillar-tensor input matching the real PillarEncoder.forward contract:
    # a handful of non-empty pillars scattered onto a shrunk BEV grid (radar/LiDAR
    # point-cloud pillars are architecturally identical inputs to this encoder --
    # only the physical point statistics differ, not the tensor contract).
    voxel_size = [0.16, 0.16, 4]
    point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]
    n_pillars = 40
    num_points = 8
    pillars = torch.randn(n_pillars, num_points, 4)
    npoints_per_pillar = torch.randint(1, num_points + 1, (n_pillars,))
    x_l = int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0])
    y_l = int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1])
    batch_idx = torch.zeros(n_pillars, 1, dtype=torch.long)
    z_coor = torch.zeros(n_pillars, 1, dtype=torch.long)
    y_coor = torch.randint(0, y_l, (n_pillars, 1))
    x_coor = torch.randint(0, x_l, (n_pillars, 1))
    coors_batch = torch.cat([batch_idx, z_coor, y_coor, x_coor], dim=1)
    return (pillars, coors_batch, npoints_per_pillar)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    (
        "PointPillars Radar",
        "build_pointpillars_radar",
        "example_input_pointpillars_radar",
        2019,
        MENAGERIE_ZOO,
    ),
]
