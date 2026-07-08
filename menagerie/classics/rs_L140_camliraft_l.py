# SOURCE: vendored from MCG-NJU/CamLiFlow @ 3bf1974f77a8 (models/base.py, models/camliraft_l.py,
# models/camliraft_l_core.py, models/point_conv.py, models/mlp.py, models/utils.py, models/losses.py,
# models/ids.py)
"""CamLiRAFT-L: the LiDAR-only variant of CamLiRAFT (Learning Optical Flow and Scene Flow
with Bidirectional Camera-LiDAR Fusion, TPAMI 2023 extended version of CamLiFlow, CVPR 2022
Oral). CamLiRAFT-L is a RAFT-style recurrent point-cloud scene-flow network: multi-scale
point encoders (Encoder3D/PointConv), an all-pairs 3D correlation volume built via furthest
point sampling + k-NN (Correlation3D), a ConvGRU-style recurrent update operator over point
features (GRU3D/MotionEncoder3D), and an iterative flow-refinement head (FlowHead3D).

The upstream repo accelerates furthest-point-sampling / k-NN / 2D correlation with custom
CUDA kernels (models/csrc/), but every one of those ops ships a pure-PyTorch fallback in
models/csrc/wrapper.py (`cpp_impl=True` requires `xyz.is_cuda`, otherwise the `_*_py` path
runs). CamLiRAFT-L only calls `furthest_point_sampling` and `k_nearest_neighbor` (no 2D
correlation, since the LiDAR-only branch has no image stream), so on CPU it runs through the
real Python reference implementations verbatim -- this file inlines that fallback logic from
wrapper.py as the `csrc` shim below (same code, same semantics) instead of building the CUDA
extension. The full bidirectional camera-LiDAR CamLiRAFT (and CamLiFlow/PWC variants) import
`mmdet.models.backbones.ResNet` for the 2D branch, which is not a base-env dependency; only
the LiDAR-only CamLiRAFT-L avoids that coupling, so it is the variant vendored here.

The only non-architectural edit vs. upstream: `models/base.py`'s `dist_reduce_sum` used
`torch.distributed` (fine as-is, no-op unless `torch.distributed.is_initialized()`), and
`models/camliraft_l_core.py`'s `@timer.timer_func` decorator is preserved via a trivial
inline `Timer` (upstream `models/utils.py::Timer`, no-op unless `.enabled` is set).
"""

import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# models/csrc/wrapper.py (verbatim pure-Python fallback paths; CUDA extension not built)
# ---------------------------------------------------------------------------
def squared_distance(xyz1: torch.Tensor, xyz2: torch.Tensor):
    """
    Calculate the Euclidean squared distance between every two points.
    :param xyz1: the 1st set of points, [batch_size, n_points_1, 3]
    :param xyz2: the 2nd set of points, [batch_size, n_points_2, 3]
    :return: squared distance between every two points, [batch_size, n_points_1, n_points_2]
    """
    assert xyz1.shape[-1] == xyz2.shape[-1] and xyz1.shape[-1] <= 3  # assert channel_last
    batch_size, n_points1, n_points2 = xyz1.shape[0], xyz1.shape[1], xyz2.shape[1]
    dist = -2 * torch.matmul(xyz1, xyz2.permute(0, 2, 1))
    dist += torch.sum(xyz1**2, -1).view(batch_size, n_points1, 1)
    dist += torch.sum(xyz2**2, -1).view(batch_size, 1, n_points2)
    return dist


def furthest_point_sampling(xyz: torch.Tensor, n_samples: int, cpp_impl=True):
    """
    Perform furthest point sampling on a set of points.
    :param xyz: a set of points, [batch_size, n_points, 3]
    :param n_samples: number of samples, int
    :return: indices of sampled points, [batch_size, n_samples]
    """

    def _furthest_point_sampling_py(_xyz: torch.Tensor, _n_samples: int):
        batch_size, n_points, _ = _xyz.shape
        farthest_indices = torch.zeros(
            batch_size, _n_samples, dtype=torch.int64, device=_xyz.device
        )
        distances = torch.ones(batch_size, n_points, device=_xyz.device) * 1e10
        batch_indices = torch.arange(batch_size, dtype=torch.int64, device=_xyz.device)
        curr_farthest_idx = torch.zeros(batch_size, dtype=torch.int64, device=_xyz.device)
        for i in range(_n_samples):
            farthest_indices[:, i] = curr_farthest_idx
            curr_farthest = _xyz[batch_indices, curr_farthest_idx, :].view(batch_size, 1, 3)
            new_distances = torch.sum((_xyz - curr_farthest) ** 2, -1)
            mask = new_distances < distances
            distances[mask] = new_distances[mask]
            curr_farthest_idx = torch.max(distances, -1)[1]
        return farthest_indices

    assert xyz.shape[2] == 3 and xyz.shape[1] > n_samples
    # No CUDA extension built in this environment -> always take the reference Python path,
    # exactly as upstream falls back to when `xyz.is_cuda` is False.
    return _furthest_point_sampling_py(xyz, n_samples).to(torch.int64)


def k_nearest_neighbor(input_xyz: torch.Tensor, query_xyz: torch.Tensor, k: int, cpp_impl=True):
    """
    Calculate k-nearest neighbor for each query.
    :param input_xyz: a set of points, [batch_size, n_points, 3] or [batch_size, 3, n_points]
    :param query_xyz: a set of centroids, [batch_size, n_queries, 3] or [batch_size, 3, n_queries]
    :param k: int
    :return: indices of k-nearest neighbors, [batch_size, n_queries, k]
    """

    def _k_nearest_neighbor_py(_input_xyz: torch.Tensor, _query_xyz: torch.Tensor, _k: int):
        dists = squared_distance(_query_xyz, _input_xyz)
        return dists.topk(_k, dim=2, largest=False).indices.to(torch.long)

    if input_xyz.shape[1] <= 3:  # channel_first to channel_last
        assert query_xyz.shape[1] == input_xyz.shape[1]
        input_xyz = input_xyz.transpose(1, 2).contiguous()
        query_xyz = query_xyz.transpose(1, 2).contiguous()

    return _k_nearest_neighbor_py(input_xyz, query_xyz, k)


# ---------------------------------------------------------------------------
# models/utils.py (Timer + point-cloud utilities actually used by CamLiRAFT-L)
# ---------------------------------------------------------------------------
class Timer:
    def __init__(self):
        self.enabled = False
        self.timing_stat = {}

    def timer_func(self, func):
        def wrap_func(*args, **kwargs):
            if not self.enabled:
                return func(*args, **kwargs)
            return func(*args, **kwargs)

        return wrap_func


timer = Timer()


def batch_indexing(
    batched_data: torch.Tensor, batched_indices: torch.Tensor, layout="channel_first"
):
    def batch_indexing_channel_first(batched_data: torch.Tensor, batched_indices: torch.Tensor):
        """
        :param batched_data: [batch_size, C, N]
        :param batched_indices: [batch_size, I1, I2, ..., Im]
        :return: indexed data: [batch_size, C, I1, I2, ..., Im]
        """

        def product(arr):
            p = 1
            for i in arr:
                p *= i
            return p

        assert batched_data.shape[0] == batched_indices.shape[0]
        batch_size, n_channels = batched_data.shape[:2]
        indices_shape = list(batched_indices.shape[1:])
        batched_indices = batched_indices.reshape([batch_size, 1, -1])
        batched_indices = batched_indices.expand([batch_size, n_channels, product(indices_shape)])
        result = torch.gather(batched_data, dim=2, index=batched_indices.to(torch.int64))
        result = result.view([batch_size, n_channels] + indices_shape)
        return result

    def batch_indexing_channel_last(batched_data: torch.Tensor, batched_indices: torch.Tensor):
        """
        :param batched_data: [batch_size, N, C]
        :param batched_indices: [batch_size, I1, I2, ..., Im]
        :return: indexed data: [batch_size, I1, I2, ..., Im, C]
        """
        assert batched_data.shape[0] == batched_indices.shape[0]
        batch_size = batched_data.shape[0]
        view_shape = [batch_size] + [1] * (len(batched_indices.shape) - 1)
        expand_shape = [batch_size] + list(batched_indices.shape)[1:]
        indices_of_batch = torch.arange(batch_size, dtype=torch.long, device=batched_data.device)
        indices_of_batch = indices_of_batch.view(view_shape).expand(expand_shape)
        if len(batched_data.shape) == 2:
            return batched_data[indices_of_batch, batched_indices.to(torch.long)]
        else:
            return batched_data[indices_of_batch, batched_indices.to(torch.long), :]

    if layout == "channel_first":
        return batch_indexing_channel_first(batched_data, batched_indices)
    elif layout == "channel_last":
        return batch_indexing_channel_last(batched_data, batched_indices)
    else:
        raise ValueError


def build_pc_pyramid(pc1, pc2, n_samples_list):
    batch_size, _, n_points = pc1.shape

    pc_both = torch.cat([pc1, pc2], dim=0)
    sample_index_both = furthest_point_sampling(pc_both.transpose(1, 2), max(n_samples_list))
    sample_index1 = sample_index_both[:batch_size]
    sample_index2 = sample_index_both[batch_size:]

    lv0_index = torch.arange(n_points, device=pc1.device)
    lv0_index = lv0_index[None, :].expand(batch_size, n_points)
    xyzs1, xyzs2, sample_indices1, sample_indices2 = [pc1], [pc2], [lv0_index], [lv0_index]

    for n_samples in n_samples_list:
        sample_indices1.append(sample_index1[:, :n_samples])
        sample_indices2.append(sample_index2[:, :n_samples])
        xyzs1.append(batch_indexing(pc1, sample_index1[:, :n_samples]))
        xyzs2.append(batch_indexing(pc2, sample_index2[:, :n_samples]))

    return xyzs1, xyzs2, sample_indices1, sample_indices2


def knn_interpolation(input_xyz, input_features, query_xyz, k=3):
    """
    :param input_xyz: 3D locations of input points, [batch_size, 3, n_inputs]
    :param input_features: features of input points, [batch_size, n_features, n_inputs]
    :param query_xyz: 3D locations of query points, [batch_size, 3, n_queries]
    :param k: k-nearest neighbor, int
    :return interpolated features: [batch_size, n_features, n_queries]
    """
    knn_indices = k_nearest_neighbor(input_xyz, query_xyz, k)
    knn_xyz = batch_indexing(input_xyz, knn_indices)
    knn_dists = torch.linalg.norm(knn_xyz - query_xyz[..., None], dim=1).clamp(1e-8)
    knn_weights = 1.0 / knn_dists
    knn_weights = knn_weights / torch.sum(knn_weights, dim=-1, keepdim=True)
    knn_features = batch_indexing(input_features, knn_indices)
    interpolated = torch.sum(knn_features * knn_weights[:, None, :, :], dim=-1)

    return interpolated


def backwarp_3d(xyz1, xyz2, flow12, k=3):
    """
    :param xyz1: 3D locations of points1, [batch_size, 3, n_points]
    :param xyz2: 3D locations of points2, [batch_size, 3, n_points]
    :param flow12: scene flow, [batch_size, 3, n_points]
    """
    xyz1_warp = xyz1 + flow12
    flow21 = knn_interpolation(xyz1_warp, -flow12, query_xyz=xyz2, k=k)
    xyz2_warp = xyz2 + flow21
    return xyz2_warp


def resize_flow2d(flow, target_h, target_w):
    origin_h, origin_w = flow.shape[2:]
    if target_h == origin_h and target_w == origin_w:
        return flow
    flow = torch.nn.functional.interpolate(
        flow, size=(target_h, target_w), mode="bilinear", align_corners=True
    )
    flow[:, 0] *= target_w / origin_w
    flow[:, 1] *= target_h / origin_h
    return flow


def dist_reduce_sum(value):
    if torch.distributed.is_initialized():
        value_t = torch.Tensor([value]).cuda()
        torch.distributed.all_reduce(value_t)
        return value_t
    else:
        return value


# ---------------------------------------------------------------------------
# models/losses.py (calc_sequence_loss_3d only; used by CamLiRAFT_L.forward during training)
# ---------------------------------------------------------------------------
def calc_sequence_loss_3d(flow_preds, target, cfgs):
    """Sequence loss for Point-RAFT."""
    n_preds = len(flow_preds)
    total_loss = 0

    if target.shape[1] == 4:
        flow_mask = target[:, 3] > 0
    else:
        flow_mask = torch.ones_like(target)[:, 0] > 0

    for i in range(n_preds):
        diff = flow_preds[i] - target[:, :3]

        if cfgs.order == "l2-norm":
            loss = torch.linalg.norm(diff, dim=1)[flow_mask].mean()
        elif cfgs.order == "l1":
            loss = torch.sum(diff.abs(), dim=1)[flow_mask].mean()
        elif cfgs.order == "robust":
            loss = torch.pow(diff.abs().sum(dim=1) + 0.01, 0.4)[flow_mask].mean()
        else:
            raise ValueError

        weight = cfgs.gamma ** (n_preds - i - 1)
        total_loss += weight * loss

    return total_loss


# ---------------------------------------------------------------------------
# models/ids.py (Implicit Disparity Space projection, used when cfgs.ids.enabled)
# ---------------------------------------------------------------------------
def persp2paral(xyz, perspect_camera_info, parallel_camera_info):
    """Perspective projection -> Parallel projection."""
    src_x, src_y, src_z = xyz[:, 0, :], xyz[:, 1, :], xyz[:, 2, :]

    batch_size, n_points = src_x.shape
    f = perspect_camera_info["f"][:, None].expand([batch_size, n_points])
    cx = perspect_camera_info["cx"][:, None].expand([batch_size, n_points])
    cy = perspect_camera_info["cy"][:, None].expand([batch_size, n_points])

    dst_x = cx + (f / src_z) * src_x
    dst_y = cy + (f / src_z) * src_y
    dst_z = f * torch.log(src_z) + 1

    perspect_h, perspect_w = perspect_camera_info["sensor_h"], perspect_camera_info["sensor_w"]
    parallel_h, parallel_w = parallel_camera_info["sensor_h"], parallel_camera_info["sensor_w"]

    scale_ratio_w = (parallel_w - 1) / (perspect_w - 1)
    scale_ratio_h = (parallel_h - 1) / (perspect_h - 1)

    dst_xyz = torch.cat(
        [
            dst_x[:, None, :] * scale_ratio_w - (parallel_w - 1) / 2,
            dst_y[:, None, :] * scale_ratio_h - (parallel_h - 1) / 2,
            dst_z[:, None, :] * min(scale_ratio_w, scale_ratio_h),
        ],
        dim=1,
    )

    return dst_xyz


def paral2persp(xyz, perspect_camera_info, parallel_camera_info):
    """Parallel projection -> Perspective projection."""
    src_x, src_y, src_z = xyz[:, 0, :], xyz[:, 1, :], xyz[:, 2, :]

    perspect_h, perspect_w = perspect_camera_info["sensor_h"], perspect_camera_info["sensor_w"]
    parallel_h, parallel_w = parallel_camera_info["sensor_h"], parallel_camera_info["sensor_w"]

    scale_ratio_w = (parallel_w - 1) / (perspect_w - 1)
    scale_ratio_h = (parallel_h - 1) / (perspect_h - 1)

    src_x = (src_x + (parallel_w - 1) / 2) / scale_ratio_w
    src_y = (src_y + (parallel_h - 1) / 2) / scale_ratio_h
    src_z = src_z / min(scale_ratio_w, scale_ratio_h)

    batch_size, n_points = src_x.shape
    f = perspect_camera_info["f"][:, None].expand([batch_size, n_points])
    cx = perspect_camera_info["cx"][:, None].expand([batch_size, n_points])
    cy = perspect_camera_info["cy"][:, None].expand([batch_size, n_points])

    dst_z = torch.exp((src_z - 1) / f)
    dst_x = (src_x - cx) * dst_z / f
    dst_y = (src_y - cy) * dst_z / f

    return torch.cat(
        [
            dst_x[:, None, :],
            dst_y[:, None, :],
            dst_z[:, None, :],
        ],
        dim=1,
    )


# ---------------------------------------------------------------------------
# models/mlp.py (verbatim)
# ---------------------------------------------------------------------------
class LayerNormCF1d(nn.Module):
    """LayerNorm that supports the channel_first data format."""

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None] * x + self.bias[:, None]
        return x


class Conv1dNormRelu(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        norm=None,
        act="leaky_relu",
    ):
        super().__init__()

        self.conv_fn = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=norm is None,
        )

        if norm == "batch_norm":
            self.norm_fn = nn.BatchNorm1d(out_channels, affine=True)
        elif norm == "instance_norm":
            self.norm_fn = nn.InstanceNorm1d(out_channels)
        elif norm == "instance_norm_affine":
            self.norm_fn = nn.InstanceNorm1d(out_channels, affine=True)
        elif norm == "layer_norm":
            self.norm_fn = LayerNormCF1d(out_channels)
        elif norm is None:
            self.norm_fn = nn.Identity()
        else:
            raise NotImplementedError("Unknown normalization function: %s" % norm)

        if act == "relu":
            self.act_fn = nn.ReLU(inplace=True)
        elif act == "leaky_relu":
            self.act_fn = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif act == "sigmoid":
            self.act_fn = nn.Sigmoid()
        elif act is None:
            self.act_fn = nn.Identity()
        else:
            raise NotImplementedError("Unknown activation function: %s" % act)

    def forward(self, x):
        x = self.conv_fn(x)
        x = self.norm_fn(x)
        x = self.act_fn(x)
        return x


class Conv2dNormRelu(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        norm=None,
        act="leaky_relu",
    ):
        super().__init__()

        self.conv_fn = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=norm is None,
        )

        if norm == "batch_norm":
            self.norm_fn = nn.BatchNorm2d(out_channels)
        elif norm == "instance_norm":
            self.norm_fn = nn.InstanceNorm2d(out_channels)
        elif norm == "instance_norm_affine":
            self.norm_fn = nn.InstanceNorm2d(out_channels, affine=True)
        elif norm == "layer_norm":
            self.norm_fn = LayerNormCF2d(out_channels)
        elif norm is None:
            self.norm_fn = nn.Identity()
        else:
            raise NotImplementedError("Unknown normalization function: %s" % norm)

        if act == "relu":
            self.act_fn = nn.ReLU(inplace=True)
        elif act == "leaky_relu":
            self.act_fn = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif act == "sigmoid":
            self.act_fn = nn.Sigmoid()
        elif act is None:
            self.act_fn = nn.Identity()
        else:
            raise NotImplementedError("Unknown activation function: %s" % act)

    def forward(self, x):
        x = self.conv_fn(x)
        x = self.norm_fn(x)
        x = self.act_fn(x)
        return x


class LayerNormCF2d(nn.Module):
    """LayerNorm that supports the channel_first data format."""

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class MLP1d(nn.Module):
    def __init__(self, in_channels, mlp_channels, norm=None, act="leaky_relu"):
        super().__init__()
        assert isinstance(in_channels, int)
        assert isinstance(mlp_channels, list)
        n_channels = [in_channels] + mlp_channels

        self.convs = nn.ModuleList()
        for in_channels, out_channels in zip(n_channels[:-1], n_channels[1:]):
            self.convs.append(Conv1dNormRelu(in_channels, out_channels, norm=norm, act=act))

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        return x


class MLP2d(nn.Module):
    def __init__(self, in_channels, mlp_channels, norm=None, act="leaky_relu"):
        super().__init__()
        assert isinstance(in_channels, int)
        assert isinstance(mlp_channels, list)
        n_channels = [in_channels] + mlp_channels

        self.convs = nn.ModuleList()
        for in_channels, out_channels in zip(n_channels[:-1], n_channels[1:]):
            self.convs.append(Conv2dNormRelu(in_channels, out_channels, norm=norm, act=act))

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        return x


# ---------------------------------------------------------------------------
# models/point_conv.py (PointConv, PointConvDW; verbatim)
# ---------------------------------------------------------------------------
class PointConv(nn.Module):
    def __init__(self, in_channels, out_channels, norm=None, act="leaky_relu", k=16):
        super().__init__()
        self.k = k

        self.weight_net = MLP2d(3, [8, 16], act=act)
        self.linear = nn.Linear(16 * (in_channels + 3), out_channels)

        if norm == "batch_norm":
            self.norm_fn = nn.BatchNorm1d(out_channels, affine=True)
        elif norm == "instance_norm":
            self.norm_fn = nn.InstanceNorm1d(out_channels, affine=True)
        elif norm == "layer_norm":
            self.norm_fn = LayerNormCF1d(out_channels)
        elif norm is None:
            self.norm_fn = nn.Identity()
        else:
            raise NotImplementedError("Unknown normalization function: %s" % norm)

        if act == "relu":
            self.act_fn = nn.ReLU(inplace=True)
        elif act == "leaky_relu":
            self.act_fn = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif act is None:
            self.act_fn = nn.Identity()
        else:
            raise NotImplementedError("Unknown activation function: %s" % act)

    def forward(self, xyz, features, sampled_xyz=None, knn_indices=None):
        """
        :param xyz: 3D locations of points, [batch_size, 3, n_points]
        :param features: features of points, [batch_size, in_channels, n_points]
        :param sampled_xyz: 3D locations of sampled points, [batch_size, 3, n_samples]
        :return weighted_features: features of sampled points, [batch_size, out_channels, n_samples]
        """
        if sampled_xyz is None:
            sampled_xyz = xyz

        bs, n_samples = sampled_xyz.shape[0], sampled_xyz.shape[-1]
        features = torch.cat([xyz, features], dim=1)
        features_cl = features.transpose(1, 2)

        if knn_indices is None:
            knn_indices = k_nearest_neighbor(xyz, sampled_xyz, self.k)
        else:
            assert knn_indices.shape[:2] == torch.Size([bs, n_samples])
            assert knn_indices.shape[2] >= self.k
            knn_indices = knn_indices[:, :, : self.k]

        knn_xyz = batch_indexing(xyz, knn_indices)
        knn_xyz_norm = knn_xyz - sampled_xyz[:, :, :, None]
        weights = self.weight_net(knn_xyz_norm)

        weights = weights.transpose(1, 2)
        knn_features = batch_indexing(features_cl, knn_indices, layout="channel_last")
        out = torch.matmul(weights, knn_features)
        out = out.view(bs, n_samples, -1)
        out = self.linear(out)
        out = self.act_fn(self.norm_fn(out.transpose(1, 2)))

        return out


class PointConvDW(nn.Module):
    def __init__(self, in_channels, out_channels, norm=None, act="leaky_relu", k=16):
        super().__init__()
        self.k = k
        self.mlp = MLP1d(in_channels, [out_channels], norm, act)
        self.weight_net = MLP2d(3, [8, 32, out_channels], act="relu")

    def forward(self, xyz, features, sampled_xyz=None, knn_indices=None):
        if sampled_xyz is None:
            sampled_xyz = xyz

        if knn_indices is None:
            knn_indices = k_nearest_neighbor(xyz, sampled_xyz, self.k)
        else:
            bs, n_points = sampled_xyz.shape[0], sampled_xyz.shape[-1]
            assert knn_indices.shape[:2] == torch.Size([bs, n_points])
            assert knn_indices.shape[2] >= self.k
            knn_indices = knn_indices[:, :, : self.k]

        knn_xyz = batch_indexing(xyz, knn_indices)
        knn_offset = knn_xyz - sampled_xyz[:, :, :, None]

        features = self.mlp(features)
        features = batch_indexing(features, knn_indices)
        features = features * self.weight_net(knn_offset)
        features = torch.max(features, dim=-1)[0]

        return features


# ---------------------------------------------------------------------------
# models/base.py (BaseModel, FlowModel; verbatim)
# ---------------------------------------------------------------------------
class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = None
        self.metrics = {}

    def clear_metrics(self):
        self.metrics = {}

    @torch.no_grad()
    def update_metrics(self, name, var):
        if isinstance(var, torch.Tensor):
            var = var.reshape(-1)
            count = var.shape[0]
            var = var.float().sum().item()

        var = dist_reduce_sum(var)
        count = dist_reduce_sum(count)

        if count <= 0:
            return

        if name not in self.metrics.keys():
            self.metrics[name] = [0, 0]

        self.metrics[name][0] += var
        self.metrics[name][1] += count

    def get_metrics(self):
        results = {}
        for name, (var, count) in self.metrics.items():
            results[name] = var / count
        return results

    def get_loss(self):
        if self.loss is None:
            raise ValueError("Loss is empty.")
        return self.loss

    @staticmethod
    def is_better(curr_metrics, best_metrics):
        raise RuntimeError("Function `is_better` must be implemented.")


class FlowModel(BaseModel):
    def __init__(self):
        super(FlowModel, self).__init__()

    @torch.no_grad()
    def update_3d_metrics(self, pred, target, occ_mask=None):
        if target.shape[1] == 4:
            mask = target[:, 3, :] > 0
            target = target[:, :3, :]
        else:
            mask = torch.ones_like(target)[:, 0, :] > 0

        diff = pred - target
        epe3d_map = torch.linalg.norm(diff, dim=1)
        acc3d_map = epe3d_map < 0.05

        if occ_mask is not None:
            mask = torch.logical_and(occ_mask == 0, mask)
            self.update_metrics("epe3d_noc", epe3d_map[mask])
            self.update_metrics("acc3d_5cm_noc", acc3d_map[mask])
        else:
            self.update_metrics("epe3d", epe3d_map[mask])
            self.update_metrics("acc3d_5cm", acc3d_map[mask])


# ---------------------------------------------------------------------------
# models/camliraft_l_core.py (verbatim)
# ---------------------------------------------------------------------------
class Encoder3D(nn.Module):
    def __init__(self, n_channels, norm=None, k=16):
        super().__init__()

        self.level0_mlp = MLP1d(3, [n_channels[0], n_channels[0]])

        self.mlps = nn.ModuleList()
        self.convs = nn.ModuleList()

        for i in range(len(n_channels) - 1):
            self.mlps.append(MLP1d(n_channels[i], [n_channels[i], n_channels[i + 1]]))
            self.convs.append(PointConv(n_channels[i + 1], n_channels[i + 1], norm=norm, k=k))

    @timer.timer_func
    def forward(self, xyzs):
        """
        :param xyzs: pyramid of points
        :return feats: pyramid of features
        """
        assert len(xyzs) == len(self.mlps) + 1

        inputs = xyzs[0]
        feats = [self.level0_mlp(inputs)]

        for i in range(len(xyzs) - 1):
            feat = self.mlps[i](feats[-1])
            feat = self.convs[i](xyzs[i], feat, xyzs[i + 1])
            feats.append(feat)

        return feats


class Correlation3D(nn.Module):
    def __init__(self, out_channels, k=16):
        super().__init__()
        self.k = k

        self.cost_mlp = MLP2d(4, [out_channels // 4, out_channels // 4], act="relu")
        self.merge = Conv1dNormRelu(out_channels, out_channels)

        self.cost_volume_pyramid = None

    def build_cost_volume_pyramid(self, feat1, feat2, xyzs2, k=3):
        cost_volume = torch.bmm(feat1.float().transpose(1, 2), feat2.float())
        cost_volume = cost_volume / feat1.shape[1]
        self.cost_volume_pyramid = [cost_volume]

        for i in range(1, len(xyzs2)):
            knn_indices = k_nearest_neighbor(xyzs2[i - 1], xyzs2[i], k=k)
            knn_corr = batch_indexing(self.cost_volume_pyramid[i - 1], knn_indices)
            avg_corr = torch.mean(knn_corr, dim=-1)
            self.cost_volume_pyramid.append(avg_corr)

    def calc_matching_cost(self, xyz1, xyz2, cost_volume):
        bs, n_points1, n_points2 = cost_volume.shape

        knn_indices_cross = k_nearest_neighbor(input_xyz=xyz2, query_xyz=xyz1, k=self.k)
        knn_xyz2 = batch_indexing(xyz2, knn_indices_cross)
        knn_xyz2_norm = knn_xyz2 - xyz1.view(bs, 3, n_points1, 1)

        knn_corr = batch_indexing(
            cost_volume.reshape(bs * n_points1, n_points2),
            knn_indices_cross.reshape(bs * n_points1, self.k),
            layout="channel_last",
        ).reshape(bs, 1, n_points1, self.k)

        cost = self.cost_mlp(torch.cat([knn_xyz2_norm, knn_corr], dim=1))
        cost = torch.sum(cost, dim=-1)

        return cost

    @timer.timer_func
    def forward(self, xyz1, xyzs2):
        cost0 = self.calc_matching_cost(xyz1, xyzs2[0], self.cost_volume_pyramid[0])
        cost1 = self.calc_matching_cost(xyz1, xyzs2[1], self.cost_volume_pyramid[1])
        cost2 = self.calc_matching_cost(xyz1, xyzs2[2], self.cost_volume_pyramid[2])
        cost3 = self.calc_matching_cost(xyz1, xyzs2[3], self.cost_volume_pyramid[3])

        costs = torch.cat([cost0, cost1, cost2, cost3], dim=1)
        costs = self.merge(costs)

        return costs


class FlowHead3D(nn.Module):
    def __init__(self, input_dim=128):
        super().__init__()
        self.conv1 = PointConvDW(input_dim, 128, k=32)
        self.conv2 = PointConvDW(128, 64, k=32)
        self.fc = nn.Conv1d(64, 3, kernel_size=1)

    @timer.timer_func
    def forward(self, xyz, features, knn_indices=None):
        features = features.float()
        features = self.conv1(xyz, features, knn_indices=knn_indices)
        features = self.conv2(xyz, features, knn_indices=knn_indices)
        return self.fc(features)


class GRU3D(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.conv_z = PointConvDW(hidden_dim + input_dim, hidden_dim, act=None, k=4)
        self.conv_r = PointConvDW(hidden_dim + input_dim, hidden_dim, act=None, k=4)
        self.conv_q = PointConvDW(hidden_dim + input_dim, hidden_dim, act=None, k=4)

    @timer.timer_func
    def forward(self, xyz, h, x, knn_indices=None):
        h, x = h.float(), x.float()
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.conv_z(xyz, hx, knn_indices=knn_indices))
        r = torch.sigmoid(self.conv_r(xyz, hx, knn_indices=knn_indices))
        q = torch.tanh(self.conv_q(xyz, torch.cat([r * h, x], dim=1), knn_indices=knn_indices))
        h = (1 - z) * h + z * q
        return h


class MotionEncoder3D(nn.Module):
    def __init__(self, corr_dim=128):
        super(MotionEncoder3D, self).__init__()
        self.conv_c1 = PointConvDW(corr_dim, corr_dim)
        self.conv_f1 = PointConvDW(3, 32, k=32)
        self.conv_f2 = PointConvDW(32, 16, k=16)
        self.conv = PointConvDW(corr_dim + 16, 128 - 3, k=16)

    @timer.timer_func
    def forward(self, xyz, flow, corr, knn_indices):
        corr, flow = corr.float(), flow.float()
        corr_feat = self.conv_c1(xyz, corr, knn_indices=knn_indices)
        flow_feat = self.conv_f1(xyz, flow, knn_indices=knn_indices)
        flow_feat = self.conv_f2(xyz, flow_feat, knn_indices=knn_indices)

        corr_flow_feat = torch.cat([corr_feat, flow_feat], dim=1)
        out = self.conv(xyz, corr_flow_feat, knn_indices=knn_indices)

        return torch.cat([out, flow], dim=1)


class CamLiRAFT_L_Core(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.cfgs = cfgs

        self.fnet = Encoder3D(n_channels=[64, 96, 128], norm="batch_norm", k=16)
        self.cnet = Encoder3D(n_channels=[64, 96, 128], norm="batch_norm", k=16)
        self.cnet_aligner = nn.Conv1d(128, 256, kernel_size=1)
        self.correlation = Correlation3D(out_channels=128, k=16)
        self.motion_encoder = MotionEncoder3D(corr_dim=128)
        self.gru = GRU3D(input_dim=128 + 128, hidden_dim=128)
        self.flow_head = FlowHead3D(input_dim=128)

    def forward(self, pc1, pc2):
        flow_preds = []

        xyzs1, xyzs2, _, _ = build_pc_pyramid(pc1, pc2, [4096, 2048, 1024, 512, 256])

        feat1 = self.fnet(xyzs1[:3])[2]
        feat2 = self.fnet(xyzs2[:3])[2]
        featc = self.cnet(xyzs1[:3])[2]
        featc = self.cnet_aligner(featc)

        xyzs1, xyzs2 = xyzs1[2:], xyzs2[2:]
        xyz1, xyz2 = xyzs1[0], xyzs2[0]  # noqa: F841 (unused xyz2 is vendored as-is from upstream)

        self.correlation.build_cost_volume_pyramid(feat1, feat2, xyzs2)

        h, x = torch.split(featc, [128, 128], dim=1)
        h = torch.tanh(h)
        x = torch.relu(x)

        knn_indices = k_nearest_neighbor(xyz1, xyz1, k=32)

        if self.training:
            n_iters = self.cfgs.n_iters_train
        else:
            n_iters = self.cfgs.n_iters_eval

        for it in range(n_iters):
            if it > 0:
                flow_pred = flow_pred.detach()  # noqa: F821 (guarded by `it > 0`; vendored as-is from upstream)
                xyzs2_warp = [backwarp_3d(xyz1, xyz2_lvl, flow_pred) for xyz2_lvl in xyzs2]
            else:
                flow_pred = torch.zeros_like(xyz1)
                xyzs2_warp = xyzs2

            corr = self.correlation(xyz1, xyzs2_warp)

            motion_feat = self.motion_encoder(xyz1, flow_pred, corr, knn_indices=knn_indices)

            h = self.gru(xyz1, h=h, x=torch.cat([x, motion_feat], dim=1), knn_indices=knn_indices)

            flow_delta = self.flow_head(xyz1, h, knn_indices)
            flow_pred = flow_pred + flow_delta.float()

            flow_preds.append(flow_pred)

        for i in range(len(flow_preds)):
            flow_preds[i] = knn_interpolation(xyz1, flow_preds[i], pc1, k=3)

        return flow_preds


# ---------------------------------------------------------------------------
# models/camliraft_l.py (verbatim)
# ---------------------------------------------------------------------------
class CamLiRAFT_L(FlowModel):
    def __init__(self, cfgs):
        super(CamLiRAFT_L, self).__init__()
        self.cfgs = cfgs
        self.core = CamLiRAFT_L_Core(cfgs)

    def forward(self, inputs):
        pc1, pc2 = inputs["pcs"][:, :3], inputs["pcs"][:, 3:]
        intrinsics = inputs["intrinsics"]

        persp_cam_info = {
            "projection_mode": "perspective",
            "sensor_h": 540,
            "sensor_w": 960,
            "f": intrinsics[:, 0],
            "cx": intrinsics[:, 1],
            "cy": intrinsics[:, 2],
        }

        if self.cfgs.ids.enabled:
            paral_cam_info = {
                "projection_mode": "parallel",
                "sensor_h": round(540 / 32),
                "sensor_w": round(960 / 32),
                "cx": (round(960 / 32) - 1) / 2,
                "cy": (round(540 / 32) - 1) / 2,
            }
            pc1 = persp2paral(pc1, persp_cam_info, paral_cam_info)
            pc2 = persp2paral(pc2, persp_cam_info, paral_cam_info)
        else:
            paral_cam_info = None

        if "src_mean" in inputs and "dst_mean" in inputs:
            src_mean = inputs["src_mean"][..., None]
            dst_mean = inputs["dst_mean"][..., None]
            src_std = inputs["src_std"][..., None]
            dst_std = inputs["dst_std"][..., None]

            pc1 = ((pc1 - src_mean) / src_std) * dst_std + dst_mean
            pc2 = ((pc2 - src_mean) / src_std) * dst_std + dst_mean

        flow_preds = self.core.forward(pc1, pc2)

        if "src_mean" in inputs and "dst_mean" in inputs:
            for i in range(len(flow_preds)):
                pcw = pc1 + flow_preds[i]
                flow_preds[i] = (((pcw - dst_mean) / dst_std) * src_std + src_mean) - (
                    ((pc1 - dst_mean) / dst_std) * src_std + src_mean
                )

            pc1 = ((pc1 - dst_mean) / dst_std) * src_std + src_mean

        if self.cfgs.ids.enabled:
            for i in range(len(flow_preds)):
                flow_preds[i] = paral2persp(
                    pc1 + flow_preds[i], persp_cam_info, paral_cam_info
                ) - paral2persp(pc1, persp_cam_info, paral_cam_info)

        final_flow_3d = flow_preds[-1]

        if "flow_3d" not in inputs:
            return {"flow_3d": final_flow_3d}

        target_3d = inputs["flow_3d"][:, :3]
        self.loss = calc_sequence_loss_3d(flow_preds, target_3d, self.cfgs.loss)

        self.update_metrics("loss3d", self.loss)
        self.update_3d_metrics(final_flow_3d, target_3d)

        return {"flow_3d": final_flow_3d}

    @staticmethod
    def is_better(curr_summary, best_summary):
        if best_summary is None:
            return True
        return curr_summary["epe3d"] < best_summary["epe3d"]


# ---------------------------------------------------------------------------
# Menagerie build/example wiring
# ---------------------------------------------------------------------------
class _CamLiRAFTLConfig:
    """Minimal stand-in for the upstream OmegaConf `cfgs` object, attribute-accessed the
    same way (`cfgs.n_iters_train`, `cfgs.ids.enabled`, `cfgs.loss.order`/`.gamma`)."""

    class _IDS:
        enabled = False

    class _Loss:
        order = "l2-norm"
        gamma = 0.8

    def __init__(self, n_iters_train=4, n_iters_eval=4):
        self.n_iters_train = n_iters_train
        self.n_iters_eval = n_iters_eval
        self.ids = self._IDS()
        self.loss = self._Loss()


def build_camliraft_l():
    # n_iters=1 (upstream default is 8 GRU refinement iterations) keeps the forward trace
    # tractable: the architecture hardcodes a furthest-point-sampling pyramid down to 4096
    # points inside build_pc_pyramid (models/camliraft_l_core.py), which alone traces to
    # ~46k fine-grained tensor ops (the CUDA kernel is unavailable in base env, so the real
    # upstream pure-Python fallback path in models/csrc/wrapper.py runs, exactly as it would
    # on any CPU-only install of the real repo) -- reducing the GRU iteration count (a
    # runtime hyperparameter, not part of the fixed architecture) is what keeps this in the
    # menagerie's traceable range rather than skipping the model.
    cfgs = _CamLiRAFTLConfig(n_iters_train=1, n_iters_eval=1)
    model = CamLiRAFT_L(cfgs)
    model.eval()
    return model


def example_input_camliraft_l():
    # CamLiRAFT-L consumes paired point clouds via `inputs['pcs']` ([B, 6, N] = xyz1 ++ xyz2)
    # and camera intrinsics (unused on the pure-LiDAR path unless cfgs.ids.enabled). The
    # network's internal FPS pyramid samples down to 4096 points, so n_points must exceed that.
    n_points = 4200
    pcs = torch.randn(1, 6, n_points)
    intrinsics = torch.tensor([[1050.0, 480.0, 270.0]])
    return {"pcs": pcs, "intrinsics": intrinsics}


MENAGERIE_ENTRIES = [
    ("CamLiRAFT-L", build_camliraft_l, example_input_camliraft_l, 2023, MENAGERIE_ZOO),
]
