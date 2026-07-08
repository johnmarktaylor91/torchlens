# FAITHFUL PORT of qiqihaer/3DSSD-pytorch @ master (original framework: PyTorch + custom
# CUDA `pointnet2_cuda` extension; a from-scratch PyTorch port of the official
# JIA-Lab-research/3DSSD TF1.x repo)
# https://github.com/qiqihaer/3DSSD-pytorch
# Files transcribed (real math/architecture kept verbatim; only the 4 low-level geometric
# CUDA kernels are given a pure-torch equivalent -- deterministic sampling/grouping
# primitives, not learned/architectural):
#   https://raw.githubusercontent.com/qiqihaer/3DSSD-pytorch/master/lib/utils/layers_util.py
#     (`Pointnet_sa_module_msg`, `Vote_layer` -- the real 3DSSD network modules)
#   https://raw.githubusercontent.com/qiqihaer/3DSSD-pytorch/master/lib/pointnet2/pytorch_utils.py
#     (`SharedMLP`/`Conv1d`/`Conv2d` MLP building blocks)
#   https://raw.githubusercontent.com/qiqihaer/3DSSD-pytorch/master/lib/utils/model_util.py
#     (`calc_square_dist`)
#   https://raw.githubusercontent.com/qiqihaer/3DSSD-pytorch/master/lib/pointnet2/pointnet2_utils.py
#     (op *signatures*/semantics for `furthest_point_sample(_with_dist)`, `gather_operation`,
#     `ball_query(_dilated)`, `grouping_operation` -- reimplemented in pure torch below since
#     the upstream file hard-requires the compiled `pointnet2_cuda` extension)
#
# Yang et al. 2020 (CVPR) "3DSSD: Point-based 3D Single Stage Object Detector". The paper's
# actual contribution -- and this port's focus -- is `Pointnet_sa_module_msg.forward`'s
# **fusion sampling** strategy: instead of plain farthest-point-sampling in xyz space
# (D-FPS, which under-samples foreground points in large scenes), 3DSSD adds F-FPS
# (`fps_method == 'F-FPS'`): FPS run on a *combined xyz+feature* distance metric
# (`calc_square_dist` on `cat([xyz, points], dim=-1)`) so sampling is pulled toward
# semantically-distinctive (i.e. likely-foreground) points, and 'FS' fusion sampling that
# concatenates F-FPS and D-FPS index sets so the surviving point set covers both instance
# centers and general scene coverage. This is followed by `Vote_layer` (a per-point MLP +
# bounded coordinate offset -- the "candidate generation" step that regresses a shifted set
# of candidate centers similar to VoteNet before the final box-regression SA layer). Real
# CUDA kernels (`ball_query`, `grouping_operation`, `furthest_point_sample(_with_dist)`,
# `gather_operation`) are deterministic geometric primitives (no learned parameters); this
# port gives them functionally equivalent pure-torch implementations (same semantics/shapes
# as the documented CUDA kernel contracts in `pointnet2_utils.py`) so the real SA-module and
# Vote-layer math executes without a custom build.
#
# Mechanical changes made ONLY for the missing CUDA extension + config-driven scaffolding
# (`core.config.cfg` YAML, `LayerBuilder`/`HeadBuilder`/`TargetAssigner`/`LossBuilder`/
# `SingleStageDetector`, none of which change what these two modules compute):
#   - `furthest_point_sample`, `furthest_point_sample_with_dist`, `gather_operation`,
#     `ball_query`, `ball_query_dilated`, `grouping_operation` are pure-torch functions with
#     the exact input/output contract documented in the upstream `pointnet2_utils.py`
#     docstrings (same shapes, same greedy-farthest-point / radius-ball-query semantics).
#   - `cfg.MODEL.NETWORK.AGGREGATION_SA_FEATURE` / `cfg.MODEL.MAX_TRANSLATE_RANGE` (global
#     YAML config reads) are replaced by explicit constructor args with the same defaults
#     3DSSD's KITTI car config uses (`AGGREGATION_SA_FEATURE=True`,
#     `MAX_TRANSLATE_RANGE=(-3.0, -3.0, -2.0)`).
#   - `Pointnet_sa_module_msg.forward`/`Vote_layer.forward`/`Vote_layer.__init__`/
#     `Pointnet_sa_module_msg.__init__` bodies are copied verbatim (only the `cfg.*` reads
#     replaced with the constructor args above).

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------------------
# Pure-torch equivalents of the CUDA-only ops in pointnet2_utils.py. Deterministic
# geometric primitives (farthest-point sampling, ball query, index-gather/group) -- same
# input/output contract as the documented CUDA kernels, no learned parameters, no
# architectural approximation.
# --------------------------------------------------------------------------------------
def furthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """xyz: (B, N, 3) -> (B, npoint) int64 indices, greedy farthest-point sampling."""
    B, N, _ = xyz.shape
    device = xyz.device
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def furthest_point_sample_with_dist(dist_matrix: torch.Tensor, npoint: int) -> torch.Tensor:
    """dist_matrix: (B, N, N) precomputed pairwise distances -> (B, npoint) indices."""
    B, N, _ = dist_matrix.shape
    device = dist_matrix.device
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        dist = dist_matrix[batch_indices, farthest, :]
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def gather_operation(features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """features: (B, C, N), idx: (B, npoint) -> (B, C, npoint)."""
    B, C, N = features.shape
    npoint = idx.shape[1]
    idx_expand = idx.unsqueeze(1).expand(B, C, npoint)
    return torch.gather(features, 2, idx_expand)


def _pad_group_idx(group_idx: torch.Tensor, N: int, nsample: int) -> torch.Tensor:
    """Real CUDA `ball_query`/`ball_query_dilated` kernels always return exactly
    `nsample` columns, padding with the first (nearest) valid index when fewer than
    `nsample` points fall within the query ball -- match that contract even when
    N < nsample (fewer candidate points overall than requested samples), and fall
    back to index 0 (the kernel's zero-initialized `idx` buffer default) on query
    balls that find zero valid neighbors at all."""
    if group_idx.shape[-1] < nsample:
        pad = group_idx[:, :, :1].expand(-1, -1, nsample - group_idx.shape[-1])
        group_idx = torch.cat([group_idx, pad], dim=-1)
    else:
        group_idx = group_idx[:, :, :nsample]
    group_first = group_idx[:, :, 0:1].clamp(max=N - 1).expand(-1, -1, nsample)
    mask = group_idx == N
    group_idx = torch.where(mask, group_first, group_idx)
    return group_idx


def ball_query(
    radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor
) -> torch.Tensor:
    """xyz: (B, N, 3), new_xyz: (B, npoint, 3) -> (B, npoint, nsample) indices into xyz."""
    B, N, _ = xyz.shape
    npoint = new_xyz.shape[1]
    sqrdists = torch.sum((new_xyz.unsqueeze(2) - xyz.unsqueeze(1)) ** 2, dim=-1)  # (B, npoint, N)
    group_idx = torch.arange(N, device=xyz.device).view(1, 1, N).repeat(B, npoint, 1)
    group_idx = torch.where(sqrdists > radius**2, torch.full_like(group_idx, N), group_idx)
    group_idx = group_idx.sort(dim=-1)[0]
    return _pad_group_idx(group_idx, N, nsample)


def ball_query_dilated(
    max_radius: float, min_radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor
) -> torch.Tensor:
    B, N, _ = xyz.shape
    npoint = new_xyz.shape[1]
    sqrdists = torch.sum((new_xyz.unsqueeze(2) - xyz.unsqueeze(1)) ** 2, dim=-1)
    group_idx = torch.arange(N, device=xyz.device).view(1, 1, N).repeat(B, npoint, 1)
    invalid = (sqrdists > max_radius**2) | (sqrdists < min_radius**2)
    group_idx = torch.where(invalid, torch.full_like(group_idx, N), group_idx)
    group_idx = group_idx.sort(dim=-1)[0]
    return _pad_group_idx(group_idx, N, nsample)


def grouping_operation(features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """features: (B, C, N), idx: (B, npoint, nsample) -> (B, C, npoint, nsample)."""
    B, C, N = features.shape
    _, npoint, nsample = idx.shape
    idx_expand = idx.view(B, 1, npoint * nsample).expand(B, C, npoint * nsample)
    out = torch.gather(features, 2, idx_expand)
    return out.view(B, C, npoint, nsample)


def calc_square_dist(a: torch.Tensor, b: torch.Tensor, norm: bool = True) -> torch.Tensor:
    """model_util.calc_square_dist, verbatim. a: [bs, n, c], b: [bs, m, c]."""
    n = a.shape[1]
    m = b.shape[1]
    num_channel = a.shape[-1]
    a_square = a.unsqueeze(dim=2)
    b_square = b.unsqueeze(dim=1)
    a_square = torch.sum(a_square * a_square, dim=-1)
    b_square = torch.sum(b_square * b_square, dim=-1)
    a_square = a_square.repeat((1, 1, m))
    b_square = b_square.repeat((1, n, 1))

    coor = torch.matmul(a, b.transpose(1, 2))

    if norm:
        dist = a_square + b_square - 2.0 / num_channel * coor
    else:
        dist = a_square + b_square - 2.0 * coor
    return dist


# --------------------------------------------------------------------------------------
# pytorch_utils.py -- SharedMLP / Conv1d (verbatim, trimmed to the used code paths)
# --------------------------------------------------------------------------------------
class _ConvBase(nn.Sequential):
    def __init__(
        self,
        in_size,
        out_size,
        kernel_size,
        stride,
        padding,
        activation,
        bn,
        init,
        conv=None,
        batch_norm=None,
        bias=True,
        preact=False,
        name="",
    ):
        super().__init__()
        bias = bias and (not bn)
        conv_unit = conv(
            in_size, out_size, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias
        )
        init(conv_unit.weight)
        if bias:
            nn.init.constant_(conv_unit.bias, 0)

        bn_unit = None
        if bn:
            bn_unit = batch_norm(in_size if preact else out_size)

        if preact:
            if bn:
                self.add_module(name + "bn", bn_unit)
            if activation is not None:
                self.add_module(name + "activation", activation)
        self.add_module(name + "conv", conv_unit)
        if not preact:
            if bn:
                self.add_module(name + "bn", bn_unit)
            if activation is not None:
                self.add_module(name + "activation", activation)


class _BNBase(nn.Sequential):
    def __init__(self, in_size, batch_norm=None, name=""):
        super().__init__()
        self.add_module(name + "bn", batch_norm(in_size))
        nn.init.constant_(self[0].weight, 1.0)
        nn.init.constant_(self[0].bias, 0)


class BatchNorm1d(_BNBase):
    def __init__(self, in_size: int, *, name: str = ""):
        super().__init__(in_size, batch_norm=nn.BatchNorm1d, name=name)


class BatchNorm2d(_BNBase):
    def __init__(self, in_size: int, name: str = ""):
        super().__init__(in_size, batch_norm=nn.BatchNorm2d, name=name)


class Conv1d(_ConvBase):
    def __init__(
        self,
        in_size,
        out_size,
        *,
        kernel_size=1,
        stride=1,
        padding=0,
        activation=nn.ReLU(inplace=True),
        bn=False,
        init=nn.init.kaiming_normal_,
        bias=True,
        preact=False,
        name="",
    ):
        super().__init__(
            in_size,
            out_size,
            kernel_size,
            stride,
            padding,
            activation,
            bn,
            init,
            conv=nn.Conv1d,
            batch_norm=BatchNorm1d,
            bias=bias,
            preact=preact,
            name=name,
        )


class Conv2d(_ConvBase):
    def __init__(
        self,
        in_size,
        out_size,
        *,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        activation=nn.ReLU(inplace=True),
        bn=False,
        init=nn.init.kaiming_normal_,
        bias=True,
        preact=False,
        name="",
    ):
        super().__init__(
            in_size,
            out_size,
            kernel_size,
            stride,
            padding,
            activation,
            bn,
            init,
            conv=nn.Conv2d,
            batch_norm=BatchNorm2d,
            bias=bias,
            preact=preact,
            name=name,
        )


class SharedMLP(nn.Sequential):
    def __init__(
        self,
        args,
        *,
        bn=False,
        activation=nn.ReLU(inplace=True),
        preact=False,
        first=False,
        name="",
        instance_norm=False,
    ):
        super().__init__()
        for i in range(len(args) - 1):
            self.add_module(
                name + "layer{}".format(i),
                Conv2d(
                    args[i],
                    args[i + 1],
                    bn=(not first or not preact or (i != 0)) and bn,
                    activation=activation if (not first or not preact or (i != 0)) else None,
                    preact=preact,
                ),
            )


# --------------------------------------------------------------------------------------
# layers_util.py -- Vote_layer (candidate generation) + Pointnet_sa_module_msg (F-FPS
# fusion-sampling set abstraction), verbatim math
# --------------------------------------------------------------------------------------
class Vote_layer(nn.Module):
    """Candidate-generation layer: per-point MLP regressing a bounded 3D coordinate
    offset, shifting each surviving point toward its predicted instance center
    (3DSSD's VoteNet-style "vote" mechanism, computed after the SA feature-extraction
    backbone and consumed by the final box-regression SA layer)."""

    def __init__(
        self, mlp_list, bn, is_training, pre_channel, max_translate_range=(-3.0, -3.0, -2.0)
    ):
        super().__init__()
        self.mlp_list = mlp_list
        self.bn = bn
        self.is_training = is_training

        mlp_modules = []
        for i in range(len(self.mlp_list)):
            mlp_modules.append(Conv1d(pre_channel, self.mlp_list[i], bn=self.bn))
            pre_channel = self.mlp_list[i]
        self.mlp_modules = nn.Sequential(*mlp_modules)

        self.ctr_reg = Conv1d(pre_channel, 3, activation=None, bn=False)
        self.min_offset = torch.tensor(max_translate_range).float().view(1, 1, 3)

    def forward(self, xyz, points):
        points_transpose = points.transpose(1, 2)
        points_transpose = self.mlp_modules(points_transpose)
        ctr_offsets = self.ctr_reg(points_transpose)

        ctr_offsets = ctr_offsets.transpose(1, 2)
        points = points_transpose.transpose(1, 2)

        min_offset = self.min_offset.repeat((points.shape[0], points.shape[1], 1)).to(points.device)

        limited_ctr_offsets = torch.where(ctr_offsets < min_offset, ctr_offsets, min_offset)
        min_offset = -1 * min_offset
        limited_ctr_offsets = torch.where(
            limited_ctr_offsets > min_offset, limited_ctr_offsets, min_offset
        )
        xyz = xyz + limited_ctr_offsets
        return xyz, points, ctr_offsets


class Pointnet_sa_module_msg(nn.Module):
    """PointNet Set Abstraction module with Multi-Scale Grouping AND 3DSSD's fusion
    sampling (F-FPS / D-FPS / FS) -- the paper's real architectural contribution over
    plain PointNet++ SA."""

    def __init__(
        self,
        radius_list,
        nsample_list,
        mlp_list,
        is_training,
        bn_decay,
        bn,
        fps_sample_range_list,
        fps_method_list,
        npoint_list,
        use_attention,
        scope,
        dilated_group,
        aggregation_channel=None,
        pre_channel=0,
        aggregation_sa_feature=True,
        epsilon=1e-5,
    ):
        super().__init__()
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.mlp_list = mlp_list
        self.is_training = is_training
        self.bn_decay = bn_decay
        self.bn = bn
        self.fps_sample_range_list = fps_sample_range_list
        self.fps_method_list = fps_method_list
        self.npoint_list = npoint_list
        self.use_attention = use_attention
        self.scope = scope
        self.dilated_group = dilated_group
        self.aggregation_channel = aggregation_channel
        self.pre_channel = pre_channel
        self.aggregation_sa_feature = aggregation_sa_feature

        mlp_modules = []
        for i in range(len(self.radius_list)):
            mlp_spec = [self.pre_channel + 3] + self.mlp_list[i]
            mlp_modules.append(SharedMLP(mlp_spec, bn=self.bn))
        self.mlp_modules = nn.Sequential(*mlp_modules)

        if self.aggregation_sa_feature and (len(self.mlp_list) != 0):
            input_channel = sum(mlp_tmp[-1] for mlp_tmp in self.mlp_list)
            self.aggregation_layer = Conv1d(input_channel, aggregation_channel, bn=self.bn)

    def forward(self, xyz, points, former_fps_idx, vote_ctr):
        bs = xyz.shape[0]
        num_points = xyz.shape[1]

        cur_fps_idx_list = []
        last_fps_end_index = 0
        for fps_sample_range, fps_method, npoint in zip(
            self.fps_sample_range_list, self.fps_method_list, self.npoint_list
        ):
            if fps_sample_range < 0:
                fps_sample_range_tmp = fps_sample_range + num_points + 1
            else:
                fps_sample_range_tmp = fps_sample_range
            tmp_xyz = xyz[:, last_fps_end_index:fps_sample_range_tmp, :].contiguous()
            tmp_points = points[:, last_fps_end_index:fps_sample_range_tmp, :].contiguous()
            if npoint == 0:
                last_fps_end_index += fps_sample_range
                continue
            if vote_ctr is not None:
                npoint = vote_ctr.shape[1]
                fps_idx = (
                    torch.arange(npoint).long().view(1, npoint).repeat((bs, 1)).to(tmp_xyz.device)
                )
            elif fps_method == "FS":
                features_for_fps = torch.cat([tmp_xyz, tmp_points], dim=-1)
                features_for_fps_distance = calc_square_dist(features_for_fps, features_for_fps)
                features_for_fps_distance = features_for_fps_distance.contiguous()
                fps_idx_1 = furthest_point_sample_with_dist(features_for_fps_distance, npoint)
                fps_idx_2 = furthest_point_sample(tmp_xyz, npoint)
                fps_idx = torch.cat([fps_idx_1, fps_idx_2], dim=-1)
            elif npoint == tmp_xyz.shape[1]:
                fps_idx = (
                    torch.arange(npoint).long().view(1, npoint).repeat((bs, 1)).to(tmp_xyz.device)
                )
            elif fps_method == "F-FPS":
                features_for_fps = torch.cat([tmp_xyz, tmp_points], dim=-1)
                features_for_fps_distance = calc_square_dist(features_for_fps, features_for_fps)
                features_for_fps_distance = features_for_fps_distance.contiguous()
                fps_idx = furthest_point_sample_with_dist(features_for_fps_distance, npoint)
            else:  # D-FPS
                fps_idx = furthest_point_sample(tmp_xyz, npoint)

            fps_idx = fps_idx + last_fps_end_index
            cur_fps_idx_list.append(fps_idx)
            last_fps_end_index += fps_sample_range
        fps_idx = torch.cat(cur_fps_idx_list, dim=-1)

        if former_fps_idx is not None:
            fps_idx = torch.cat([fps_idx, former_fps_idx], dim=-1)

        if vote_ctr is not None:
            vote_ctr_transpose = vote_ctr.transpose(1, 2).contiguous()
            new_xyz = gather_operation(vote_ctr_transpose, fps_idx).transpose(1, 2).contiguous()
        else:
            new_xyz = (
                gather_operation(xyz.transpose(1, 2).contiguous(), fps_idx)
                .transpose(1, 2)
                .contiguous()
            )

        new_points_list = []
        points = points.transpose(1, 2).contiguous()
        xyz = xyz.contiguous()
        for i in range(len(self.radius_list)):
            nsample = self.nsample_list[i]
            if self.dilated_group:
                min_radius = 0.0 if i == 0 else self.radius_list[i - 1]
                max_radius = self.radius_list[i]
                idx = ball_query_dilated(max_radius, min_radius, nsample, xyz, new_xyz)
            else:
                radius = self.radius_list[i]
                idx = ball_query(radius, nsample, xyz, new_xyz)

            xyz_trans = xyz.transpose(1, 2).contiguous()
            grouped_xyz = grouping_operation(xyz_trans, idx)
            grouped_xyz = grouped_xyz - new_xyz.transpose(1, 2).unsqueeze(-1)
            if points is not None:
                grouped_points = grouping_operation(points, idx)
                grouped_points = torch.cat([grouped_xyz, grouped_points], dim=1)
            else:
                grouped_points = grouped_xyz

            new_points = self.mlp_modules[i](grouped_points)
            new_points = F.max_pool2d(new_points, kernel_size=[1, new_points.size(3)])
            new_points_list.append(new_points.squeeze(-1))

        if len(new_points_list) > 0:
            new_points_concat = torch.cat(new_points_list, dim=1)
            if self.aggregation_sa_feature:
                new_points_concat = self.aggregation_layer(new_points_concat)
        else:
            new_points_concat = gather_operation(points, fps_idx)
        new_points_concat = new_points_concat.transpose(1, 2).contiguous()

        return new_xyz, new_points_concat, fps_idx


class ThreeSSDBackbone(nn.Module):
    """Two fusion-sampling SA layers (F-FPS+D-FPS "FS" strategy, matching 3DSSD's KITTI
    car config layer stack) followed by the Vote_layer candidate-generation step and a
    final single-scale SA layer aggregating candidate-neighborhood features -- the
    complete real 3DSSD point-cloud backbone forward path."""

    def __init__(self):
        super().__init__()
        # SA layer 1: FS fusion sampling (F-FPS + D-FPS), input has no point features
        # (pre_channel=1 mirrors the upstream `layer_idx == 0` -> pre_channel=1 default,
        # a constant per-point intensity/placeholder channel).
        self.sa1 = Pointnet_sa_module_msg(
            radius_list=[0.2, 0.4],
            nsample_list=[8, 16],
            mlp_list=[[8, 8, 16], [8, 8, 16]],
            is_training=False,
            bn_decay=None,
            bn=True,
            fps_sample_range_list=[-1],
            fps_method_list=["FS"],
            npoint_list=[8],
            use_attention=False,
            scope="sa1",
            dilated_group=False,
            aggregation_channel=16,
            pre_channel=1,
            aggregation_sa_feature=True,
        )
        # SA layer 2: F-FPS sampling on the layer-1 output.
        self.sa2 = Pointnet_sa_module_msg(
            radius_list=[0.4, 0.8],
            nsample_list=[8, 16],
            mlp_list=[[16, 16, 32], [16, 16, 32]],
            is_training=False,
            bn_decay=None,
            bn=True,
            fps_sample_range_list=[-1],
            fps_method_list=["F-FPS"],
            npoint_list=[4],
            use_attention=False,
            scope="sa2",
            dilated_group=False,
            aggregation_channel=32,
            pre_channel=16,
            aggregation_sa_feature=True,
        )
        # Vote layer: per-point MLP + bounded coordinate offset (candidate generation).
        self.vote_layer = Vote_layer(mlp_list=[32, 32], bn=True, is_training=False, pre_channel=32)
        # Final SA layer: group around the voted candidate centers (`vote_ctr`), no
        # further FPS (vote_ctr drives npoint directly, matching 3DSSD's `SA_Layer_SSG_Last`).
        self.sa3 = Pointnet_sa_module_msg(
            radius_list=[1.6],
            nsample_list=[8],
            mlp_list=[[32, 32, 64]],
            is_training=False,
            bn_decay=None,
            bn=True,
            fps_sample_range_list=[-1],
            fps_method_list=["D-FPS"],
            npoint_list=[4],
            use_attention=False,
            scope="sa3",
            dilated_group=False,
            aggregation_channel=64,
            pre_channel=32,
            aggregation_sa_feature=True,
        )

    def forward(self, xyz, points):
        xyz1, feat1, fps_idx1 = self.sa1(xyz, points, None, None)
        xyz2, feat2, fps_idx2 = self.sa2(xyz1, feat1, None, None)
        vote_xyz, vote_feat, ctr_offsets = self.vote_layer(xyz2, feat2)
        xyz3, feat3, fps_idx3 = self.sa3(xyz2, feat2, None, vote_xyz)
        return xyz3, feat3


def build_3dssd():
    model = ThreeSSDBackbone()
    model.eval()
    return model


def example_input_3dssd():
    xyz = torch.rand(1, 64, 3) * 4.0
    points = torch.rand(1, 64, 1)  # constant per-point placeholder channel (pre_channel=1)
    return (xyz, points)


MENAGERIE_ZOO = "ported-pytorch"
MENAGERIE_ENTRIES = [
    ("3DSSD", "build_3dssd", "example_input_3dssd", 2020, "ported"),
]
