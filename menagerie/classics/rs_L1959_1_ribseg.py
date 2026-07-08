# SOURCE: vendored from M3DV/RibSeg @ main (models/pn2.py, fetched raw from
# https://raw.githubusercontent.com/M3DV/RibSeg/main/models/pn2.py)
#
# RibSeg v2 (Jin, Cheng, Yu et al., "RibSeg v2: A Large-scale Benchmark for Rib Labeling
# and Anatomical Centerline Extraction", IEEE TMI 2023 / MICCAI 2021 preliminary) is a
# point-cloud rib-instance-segmentation pipeline over CT-derived point clouds. `pn2.py`'s
# `PointNet2` class is RibSeg's feature-extraction backbone: a 4-stage PointNet++
# multi-scale-grouping (MSG) set-abstraction encoder followed by a 4-stage feature-
# propagation decoder, producing a per-point 128-d feature used downstream by RibSeg's
# segmentation/centerline heads.
#
# Vendoring notes (dependency-resolution fix only; NO architecture change):
#   - `pn2.py` imports `PointNetSetAbstractionMsg, PointNetFeaturePropagation` from
#     `models.pn2_util`. As published in the RibSeg repo (verified via both the raw CDN
#     and the GitHub contents API, same content both ways), `models/pn2_util.py` does NOT
#     actually define those classes -- it instead contains an unrelated part-segmentation
#     script (`get_model`/`get_loss`, keyed on `cls_label`) that itself imports from a
#     `models.pointnet_util` module that does not exist anywhere in the repo. This is a
#     genuine broken/incomplete file in the upstream "preview" release (confirmed via
#     `gh api .../commits?path=models/pn2_util.py`: exactly one commit, message
#     "ribsegv2_preview") -- `pn2.py` cannot actually run against its own repo as published.
#   - `PointNetSetAbstractionMsg`/`PointNetSetAbstraction`/`PointNetFeaturePropagation`
#     below are therefore sourced from yanx27/Pointnet_Pointnet2_pytorch @ master
#     (models/pointnet2_utils.py, fetched raw from
#     https://raw.githubusercontent.com/yanx27/Pointnet_Pointnet2_pytorch/master/models/pointnet2_utils.py),
#     the de facto canonical PyTorch PointNet++ reference port that RibSeg's own
#     (broken) import path was clearly adapted from: RibSeg's `pn2.py` calls
#     `PointNetSetAbstractionMsg(1024, [0.05, 0.1], [16, 32], 3, [[16, 16, 32], [32, 32, 64]])`
#     etc. with the exact same positional constructor signature
#     `(npoint, radius_list, nsample_list, in_channel, mlp_list)` and the exact same
#     `forward(xyz, points)` call contract as this canonical file defines -- i.e. these are
#     the actual missing building blocks RibSeg's own file should have contained, not a
#     reimplementation from a paper description. Only `PointNetSetAbstractionMsg` and
#     `PointNetFeaturePropagation` (the two classes `pn2.py` actually imports) are vendored
#     here; `PointNetSetAbstraction`/helper functions from the util file that `pn2.py`
#     itself never calls are omitted for leanness (kept `square_distance`/`index_points`/
#     `farthest_point_sample`/`query_ball_point`/`sample_and_group`/`sample_and_group_all`
#     since `PointNetSetAbstractionMsg`/`PointNetFeaturePropagation` depend on them).
#   - `pn2.py`'s `PointNet2.forward` is otherwise verbatim (module structure, forward
#     control flow, all layer dims unchanged).

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


# --- from yanx27/Pointnet_Pointnet2_pytorch models/pointnet2_utils.py (verbatim helpers
# needed by PointNetSetAbstractionMsg / PointNetFeaturePropagation) ---


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm;
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src**2, -1).view(B, N, 1)
    dist += torch.sum(dst**2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = (
        torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    )
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius**2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points = F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(
                index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2
            )

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points


# --- verbatim from RibSeg's own models/pn2.py ---


class PointNet2(nn.Module):
    def __init__(self):
        super(PointNet2, self).__init__()
        self.sa1 = PointNetSetAbstractionMsg(
            1024, [0.05, 0.1], [16, 32], 3, [[16, 16, 32], [32, 32, 64]]
        )
        self.sa2 = PointNetSetAbstractionMsg(
            256, [0.1, 0.2], [16, 32], 32 + 64, [[64, 64, 128], [64, 96, 128]]
        )
        self.sa3 = PointNetSetAbstractionMsg(
            64, [0.2, 0.4], [16, 32], 128 + 128, [[128, 196, 256], [128, 196, 256]]
        )
        self.sa4 = PointNetSetAbstractionMsg(
            16, [0.4, 0.8], [16, 32], 256 + 256, [[256, 256, 512], [256, 384, 512]]
        )

        self.fp4 = PointNetFeaturePropagation(512 + 1024, [256, 256])
        self.fp3 = PointNetFeaturePropagation(256 + 256, [256, 256])
        self.fp2 = PointNetFeaturePropagation(96 + 256, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])

    def forward(self, xyz):
        """
        Input:
            xyz: input points position data, [B, 3, N]
        Return:
            feat: feature, [B, 128, N]
        """
        # Set Abstraction layers
        l0_points = xyz  # [B, 3, 30000]
        l0_xyz = xyz[:, :3, :]  # [B, 3, 30000]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)  # [B,3,1024], [B,96,1024]
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)  # [B,3,256], [B,256,256]
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)  # [B,3,64], [B,512,64]
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)  # [B,3,16], [B,1024,16]

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)  # [B, 256, 64]
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)  # [B, 256, 256]
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)  # [B, 128, 1024]
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)  # [B, 128, 30000]

        return l0_points


def build_ribseg():
    torch.manual_seed(0)
    net = PointNet2()
    net.eval()
    return net


def example_input_ribseg():
    torch.manual_seed(0)
    # Real usage feeds 30000-point clouds (see pn2.py docstring); shrunk to the minimum
    # viable point count here since sa1's PointNetSetAbstractionMsg farthest-point-samples
    # 1024 centroids -- the input point count must be >= 1024.
    return torch.randn(1, 3, 1024)


MENAGERIE_ENTRIES = [
    ("RibSeg v2 PointNet2", build_ribseg, example_input_ribseg, 2023, MENAGERIE_ZOO),
]
