# FAITHFUL PORT of ricsonc/grnn @ master (original framework: TensorFlow 1.x)
#
# "Learning Spatial Common Sense with Geometry-Aware Recurrent Networks" (Tung, Cheng,
# Fragkiadaki; CVPR 2019) -- code: https://github.com/ricsonc/grnn (TF 1.13 + tf.contrib.slim
# + tf.contrib.rnn; cannot run in a base torch env, so it is hand-ported below).
# This GRNN architecture is the model applied in "3D View Prediction Models of the Dorsal
# Visual Stream" (Sarch, Fang, Harvey, Jain, Getty, Bhalla, Beeler, Fragkiadaki; CCN 2023,
# arXiv:2309.01782), the queue candidate ("3D View Prediction Dorsal Stream models") -- the
# CCN paper ships no repo of its own but directly reuses the GRNN architecture above.
#
# Ported mechanism (faithful to nets.py / utils/nets.py / utils/voxel.py / utils/convlstm.py):
#   1. unproject_image / unproject_voxel: each 2D view (mask, depth, RGB image) is tiled
#      along a depth axis, then resampled through the INVERSE camera-projection matrix
#      (utils/voxel.py:get_transform_matrix_tf_) via trilinear sampling
#      (utils/voxel.py:transformer/_interpolate) -- ported here with torch's F.grid_sample,
#      the standard equivalent of the hand-written gather/interpolate in the original.
#   2. translate_given_angles: each additional view's unprojected voxel grid is rigidly
#      rotated into view-0's canonical camera frame via two composed voxel rotations
#      (rotate_voxel, again via the camera matrix + trilinear resample).
#   3. gru_aggregator: the aligned per-view voxel grids are aggregated over "time" (view
#      index) via a 2-layer stack of 3D convolutional GRUs (utils/convlstm.py:ConvGRUCell,
#      conv-recurrent gating) -- the "recurrent" half of GRNN's name; the final hidden
#      states of both layers are summed, matching gru_aggregator's `sum(l.state for l in
#      layers)`.
#   4. voxel_net_3d: a 5-level 3D-conv encoder / 5-level 3D-deconv decoder with U-Net-style
#      skip connections predicts a persistent 3D occupancy memory (sigmoid head).
#   5. project_voxel + voxel2depth_aligned/voxel2mask_aligned: the predicted voxel memory is
#      projected back through the (identity) query camera and flattened into a rendered
#      (depth, mask) view-prediction pair -- the model's headline "view prediction" output.
#
# Sizes shrunk for tracing (S=16 voxel grid, d0=4 base channels, 2 input views); TF-only
# training/eval infra (placeholders, sessions, checkpoint savers, data loaders, summaries,
# visualization) is dropped -- only the real forward architecture is ported.

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "ported-pytorch"


# ---------------------------------------------------------------------------
# utils/voxel.py -- camera geometry + differentiable voxel resampling
# ---------------------------------------------------------------------------
def _camera_matrix(
    theta: torch.Tensor,
    phi: torch.Tensor,
    radius: float,
    focal_length: float,
    invert_rot: bool = False,
    invert_focal: bool = False,
) -> torch.Tensor:
    """Port of utils/voxel.py:get_transform_matrix_tf_ (batched). theta/phi in degrees, (BS,)."""
    device = theta.device
    sin_phi, cos_phi = torch.sin(phi * math.pi / 180.0), torch.cos(phi * math.pi / 180.0)
    sin_theta, cos_theta = torch.sin(theta * math.pi / 180.0), torch.cos(theta * math.pi / 180.0)
    bs = theta.shape[0]
    zeros, ones = torch.zeros_like(sin_theta), torch.ones_like(sin_theta)

    rot_azimuth = torch.stack(
        [cos_theta, zeros, -sin_theta, zeros, ones, zeros, sin_theta, zeros, cos_theta], dim=-1
    ).view(bs, 3, 3)
    rot_elevation = torch.stack(
        [cos_phi, sin_phi, zeros, -sin_phi, cos_phi, zeros, zeros, zeros, ones], dim=-1
    ).view(bs, 3, 3)
    rotation = torch.bmm(rot_azimuth, rot_elevation)
    if invert_rot:
        rotation = torch.linalg.inv(rotation)

    displacement_local = (
        torch.tensor([radius, 0.0, 0.0], device=device).view(1, 3, 1).expand(bs, 3, 1)
    )
    displacement = torch.bmm(rotation, displacement_local)

    extrinsic = torch.zeros(bs, 4, 4, device=device)
    extrinsic[:, :3, :3] = rotation
    extrinsic[:, :3, 3:4] = -displacement
    extrinsic[:, 3, 3] = 1.0

    if invert_focal:
        intrinsic_diag = torch.tensor([1.0, focal_length, focal_length, 1.0], device=device)
    else:
        intrinsic_diag = torch.tensor(
            [1.0, 1.0 / focal_length, 1.0 / focal_length, 1.0], device=device
        )
    intrinsic = torch.diag(intrinsic_diag).unsqueeze(0).expand(bs, 4, 4)

    return torch.bmm(extrinsic, intrinsic)


def _voxel_meshgrid(
    d: int,
    h: int,
    w: int,
    z_near: float,
    z_far: float,
    mode: str,
    focal_length: float,
    device: torch.device,
) -> torch.Tensor:
    """Port of utils/voxel.py:_meshgrid / _invproj_meshgrid / _noproj_meshgrid. Returns (4, D*H*W)."""
    zlin = torch.linspace(z_near, z_far, d, device=device).view(d, 1, 1).expand(d, h, w)
    x_t = torch.linspace(-1.0, 1.0, w, device=device).view(1, 1, w).expand(d, h, w).clone()
    y_t = torch.linspace(-1.0, 1.0, h, device=device).view(1, h, 1).expand(d, h, w).clone()
    d_t = zlin.clone()

    if mode == "project":
        x_t = x_t * zlin
        y_t = y_t * zlin
    elif mode == "invproj":
        x_t = x_t / zlin
        y_t = y_t / zlin
    elif mode == "noproj":
        x_t = x_t * focal_length
        y_t = y_t * focal_length
    else:
        raise ValueError(mode)

    ones = torch.ones_like(x_t)
    return torch.stack([d_t.reshape(-1), y_t.reshape(-1), x_t.reshape(-1), ones.reshape(-1)], dim=0)


def _voxel_transform(
    voxel: torch.Tensor,
    camera_matrix: torch.Tensor,
    out_size: tuple[int, int, int],
    z_near: float,
    z_far: float,
    mode: str,
    focal_length: float,
) -> torch.Tensor:
    """Port of utils/voxel.py:transformer/_transform/_interpolate.

    voxel: (BS, D, H, W, C) NDHWC layout (matches the original TF tensor layout). Trilinearly
    resamples `voxel` at the coordinates obtained by mapping the `out_size` grid through
    `camera_matrix` -- implemented with F.grid_sample (same trilinear-sampling semantics as
    the hand-written gather/interpolate in the original `_interpolate`).
    """
    bs = voxel.shape[0]
    d_out, h_out, w_out = out_size
    grid = _voxel_meshgrid(d_out, h_out, w_out, z_near, z_far, mode, focal_length, voxel.device)
    grid = grid.unsqueeze(0).expand(bs, 4, grid.shape[-1])
    t_g = torch.bmm(camera_matrix, grid)  # (BS, 4, N)
    z_s, y_s, x_s = t_g[:, 0, :], t_g[:, 1, :], t_g[:, 2, :]
    sample_grid = torch.stack([x_s, y_s, z_s], dim=-1).view(bs, d_out, h_out, w_out, 3)

    voxel_ncdhw = voxel.permute(0, 4, 1, 2, 3).contiguous()
    out = F.grid_sample(
        voxel_ncdhw, sample_grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )
    return out.permute(0, 2, 3, 4, 1).contiguous()


def rotate_voxel(
    voxel: torch.Tensor,
    camera_matrix: torch.Tensor,
    z_near: float,
    z_far: float,
    focal_length: float,
) -> torch.Tensor:
    """Port of utils/voxel.py:rotate_voxel (do_project=False -> 'noproj' meshgrid)."""
    s = voxel.shape[1]
    return _voxel_transform(voxel, camera_matrix, (s, s, s), z_near, z_far, "noproj", focal_length)


def project_voxel(
    voxel: torch.Tensor, z_near: float, z_far: float, focal_length: float
) -> torch.Tensor:
    """Port of utils/voxel.py:project_voxel (identity camera, do_project=True -> 'project' meshgrid)."""
    bs, s = voxel.shape[0], voxel.shape[1]
    zero = torch.zeros(bs, device=voxel.device)
    cam = _camera_matrix(zero, zero, radius=0.0, focal_length=focal_length)
    return _voxel_transform(voxel, cam, (s, s, s), z_near, z_far, "project", focal_length)


def unproject_image(
    img: torch.Tensor, size: int, z_near: float, z_far: float, focal_length: float
) -> torch.Tensor:
    """Port of utils/nets.py:unproject + utils/voxel.py:unproject_image/unproject_voxel.

    img: (BS, H, W, C) NHWC. Returns an (BS, S, S, S, C) voxel grid.
    """
    bs = img.shape[0]
    voxel = img.unsqueeze(1).expand(bs, size, *img.shape[1:]).contiguous()  # tile along depth axis
    cam = _camera_matrix(
        torch.zeros(bs, device=img.device),
        torch.zeros(bs, device=img.device),
        radius=0.0,
        focal_length=focal_length,
        invert_focal=True,
    )
    voxel = _voxel_transform(voxel, cam, (size, size, size), z_near, z_far, "invproj", focal_length)
    return torch.flip(voxel, dims=[2, 3])


def translate_given_angles(
    dtheta: torch.Tensor,
    phi1: torch.Tensor,
    phi2: torch.Tensor,
    voxel: torch.Tensor,
    z_near: float,
    z_far: float,
    focal_length: float,
) -> torch.Tensor:
    """Port of utils/voxel.py:translate_given_angles (two composed rigid voxel rotations)."""
    zero = torch.zeros_like(phi1)
    rot1 = _camera_matrix(zero, -phi1, radius=0.0, focal_length=focal_length)
    rot2 = _camera_matrix(dtheta, phi2, radius=0.0, focal_length=focal_length)
    voxel = rotate_voxel(voxel, rot1, z_near, z_far, focal_length)
    voxel = rotate_voxel(voxel, rot2, z_near, z_far, focal_length)
    return voxel


def voxel2mask_aligned(voxel: torch.Tensor) -> torch.Tensor:
    """Port of utils/voxel.py:voxel2mask_aligned (max-project along the depth axis)."""
    return torch.amax(voxel, dim=3)


def voxel2depth_aligned(voxel: torch.Tensor) -> torch.Tensor:
    """Port of utils/voxel.py:voxel2depth_aligned (first-hit argmin depth along the depth axis)."""
    bs, s = voxel.shape[0], voxel.shape[1]
    voxel = voxel.squeeze(4)
    costgrid = (
        torch.arange(s, device=voxel.device, dtype=torch.float32)
        .view(1, 1, 1, s)
        .expand(bs, s, s, s)
    )
    invalid = 1000.0 * (voxel < 0.5).float()
    invalid_mask = (
        torch.cat([torch.ones(s - 1, device=voxel.device), torch.zeros(1, device=voxel.device)])
        .view(1, 1, 1, s)
        .expand(bs, s, s, s)
    )
    costgrid = costgrid + invalid * invalid_mask
    return torch.argmin(costgrid, dim=3, keepdim=True).float()


# ---------------------------------------------------------------------------
# utils/convlstm.py -- convolutional GRU cell (conv-recurrent gating)
# ---------------------------------------------------------------------------
class ConvGRUCell3D(nn.Module):
    """Port of utils/convlstm.py:ConvGRUCell for 3D (D, H, W) feature grids."""

    def __init__(self, in_channels: int, filters: int, kernel: tuple[int, int, int]) -> None:
        super().__init__()
        self.filters = filters
        pad = tuple(k // 2 for k in kernel)
        self.gates = nn.Conv3d(in_channels + filters, 2 * filters, kernel, padding=pad)
        self.candidate = nn.Conv3d(in_channels + filters, filters, kernel, padding=pad)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # x, h: (BS, C, D, H, W)
        gates = self.gates(torch.cat([x, h], dim=1))
        reset_gate, update_gate = torch.chunk(gates, 2, dim=1)
        reset_gate = F.layer_norm(reset_gate, reset_gate.shape[1:])
        update_gate = F.layer_norm(update_gate, update_gate.shape[1:])
        reset_gate, update_gate = torch.sigmoid(reset_gate), torch.sigmoid(update_gate)

        candidate = self.candidate(torch.cat([x, reset_gate * h], dim=1))
        candidate = F.layer_norm(candidate, candidate.shape[1:])
        candidate = torch.tanh(candidate)

        return update_gate * h + (1.0 - update_gate) * candidate


class GRUAggregator3D(nn.Module):
    """Port of utils/nets.py:gru_aggregator (2-layer stacked ConvGRU over the view sequence)."""

    def __init__(
        self, in_channels: int, filters: int = 4, kernel: tuple[int, int, int] = (5, 5, 5)
    ) -> None:
        super().__init__()
        self.filters = filters
        self.cell0 = ConvGRUCell3D(in_channels, filters, kernel)
        self.cell1 = ConvGRUCell3D(filters, filters, kernel)

    def forward(self, views: list[torch.Tensor]) -> torch.Tensor:
        # views: list of (BS, C, D, H, W) tensors, one per input view (the "time" sequence)
        bs, _, d, h, w = views[0].shape
        state0 = torch.zeros(bs, self.filters, d, h, w, device=views[0].device)
        outputs0 = []
        for v in views:
            state0 = self.cell0(v, state0)
            outputs0.append(state0)

        state1 = torch.zeros(bs, self.filters, d, h, w, device=views[0].device)
        for o in outputs0:
            state1 = self.cell1(o, state1)

        return state0 + state1  # matches gru_aggregator's `sum(l.state for l in layers)`


# ---------------------------------------------------------------------------
# utils/nets.py:voxel_net_3d -- 5-level 3D conv encoder / decoder with U-Net-style skips
# ---------------------------------------------------------------------------
class VoxelNet3D(nn.Module):
    """Port of utils/nets.py:voxel_net_3d."""

    def __init__(self, in_channels: int, voxel_size: int, d0: int = 4) -> None:
        super().__init__()
        if voxel_size % 16 != 0:
            raise ValueError("voxel_size must be a multiple of 16 (4 stride-2 halvings)")
        last_k = voxel_size // 16

        dims = [d0, 2 * d0, 4 * d0, 8 * d0, 16 * d0]
        enc_channels = [in_channels, *dims]
        self.encoders = nn.ModuleList()
        self.encoder_bn = nn.ModuleList()
        for i in range(5):
            k, s, pad = (4, 2, 1) if i < 4 else (last_k, 1, 0)
            self.encoders.append(
                nn.Conv3d(enc_channels[i], enc_channels[i + 1], k, stride=s, padding=pad)
            )
            self.encoder_bn.append(nn.BatchNorm3d(enc_channels[i + 1]))

        chans = [8 * d0, 4 * d0, 2 * d0, d0, 1]
        dec_in = [16 * d0, chans[0] + 8 * d0, chans[1] + 4 * d0, chans[2] + 2 * d0, chans[3] + d0]
        self.decoders = nn.ModuleList()
        self.decoder_bn = nn.ModuleList()
        for i in range(5):
            k, s, pad = (last_k, 1, 0) if i == 0 else (4, 2, 1)
            self.decoders.append(nn.ConvTranspose3d(dec_in[i], chans[i], k, stride=s, padding=pad))
            self.decoder_bn.append(nn.BatchNorm3d(chans[i]))

        self.final = nn.Conv3d(chans[-1] + in_channels, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = [x]
        net = x
        for conv, bn in zip(self.encoders, self.encoder_bn):
            net = F.relu(bn(conv(net)))
            skips.append(net)
        skips.pop()  # drop the innermost (bottleneck) feature map, matching skipcons.pop()

        for i, (deconv, bn) in enumerate(zip(self.decoders, self.decoder_bn)):
            net = deconv(net)
            net = bn(net)
            if i < 4:
                net = F.relu(net)  # last decoder layer has activation_fn=None in the source
            net = torch.cat([net, skips.pop()], dim=1)

        logit = self.final(net)
        return torch.sigmoid(logit)


# ---------------------------------------------------------------------------
# nets.py:MultiViewReconstructionNet -- top-level GRNN forward pass
# ---------------------------------------------------------------------------
class GRNNMultiView(nn.Module):
    """Faithful port of the GRNN multi-view -> 3D-occupancy -> view-prediction pipeline
    (nets.py:MultiViewReconstructionNet.predict_/reproject, models.py:MultiViewReconstruction).
    """

    def __init__(
        self,
        voxel_size: int = 16,
        d0: int = 4,
        gru_filters: int = 4,
        focal_length: float = 1.0,
        radius: float = 4.0,
    ) -> None:
        super().__init__()
        self.voxel_size = voxel_size
        self.focal_length = focal_length
        self.radius = radius
        self.z_near, self.z_far = radius - 2.0, radius + 2.0
        self.per_view_channels = 5  # mask(1) + depth(1) + rgb(3) -- get_views_for_prediction
        self.aggregator = GRUAggregator3D(self.per_view_channels, filters=gru_filters)
        self.voxel_net = VoxelNet3D(gru_filters, voxel_size, d0=d0)

    def forward(
        self,
        images: torch.Tensor,
        depths: torch.Tensor,
        masks: torch.Tensor,
        thetas: torch.Tensor,
        phis: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        images: (BS, NUM_VIEWS, 3, S, S); depths/masks: (BS, NUM_VIEWS, 1, S, S).
        thetas/phis: (BS, NUM_VIEWS) camera azimuth/elevation in degrees.
        Returns (pred_voxel, pred_depth, pred_mask).
        """
        num_views = images.shape[1]
        s = self.voxel_size

        per_view_voxels = []
        for v in range(num_views):
            # get_views_for_prediction: concat mask, depth, image (order matters, per source)
            feat = torch.cat([masks[:, v], depths[:, v], images[:, v]], dim=1)  # (BS, 5, S, S)
            feat = F.interpolate(feat, size=(s, s), mode="bilinear", align_corners=True)
            feat_nhwc = feat.permute(0, 2, 3, 1)  # (BS, S, S, 5) -- NHWC for unproject
            per_view_voxels.append(
                unproject_image(feat_nhwc, s, self.z_near, self.z_far, self.focal_length)
            )

        # translate_views_multi: align views[1:] into view 0's canonical frame
        aligned = [per_view_voxels[0]]
        theta0, phi0 = thetas[:, 0], phis[:, 0]
        for v in range(1, num_views):
            dtheta = theta0 - thetas[:, v]
            aligned.append(
                translate_given_angles(
                    dtheta,
                    phis[:, v],
                    phi0,
                    per_view_voxels[v],
                    self.z_near,
                    self.z_far,
                    self.focal_length,
                )
            )

        aligned_ncdhw = [
            a.permute(0, 4, 1, 2, 3) for a in aligned
        ]  # NDHWC -> NCDHW for the ConvGRU
        aggregated = self.aggregator(aligned_ncdhw)  # (BS, gru_filters, S, S, S)

        pred_voxel = self.voxel_net(aggregated)  # (BS, 1, S, S, S)

        # reproject to the query (identity) viewpoint and flatten to a rendered view
        pred_voxel_ndhwc = pred_voxel.permute(0, 2, 3, 4, 1)
        projected = project_voxel(pred_voxel_ndhwc, self.z_near, self.z_far, self.focal_length)
        delta = 1e-4
        projected = projected * (1.0 - delta) + delta / 2.0
        pred_mask = voxel2mask_aligned(projected)
        pred_depth = voxel2depth_aligned(projected)

        return pred_voxel, pred_depth, pred_mask


def build_grnn() -> nn.Module:
    """Tiny GRNN multi-view reconstruction / view-prediction network (random init)."""
    return GRNNMultiView(voxel_size=16, d0=4, gru_filters=4).eval()


def example_input_grnn() -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """2 views of a 16x16 RGB-D+mask scene, seen from two camera azimuths/elevations."""
    bs, num_views, s = 1, 2, 16
    images = torch.rand(bs, num_views, 3, s, s)
    depths = torch.rand(bs, num_views, 1, s, s) * 2.0 + 3.0  # roughly at the camera radius
    masks = (torch.rand(bs, num_views, 1, s, s) > 0.3).float()
    thetas = torch.tensor([[0.0, 40.0]])
    phis = torch.tensor([[10.0, 15.0]])
    return images, depths, masks, thetas, phis


MENAGERIE_ENTRIES = [
    (
        "GRNN (Geometry-Aware Recurrent Neural Network, multi-view 3D view prediction)",
        "build_grnn",
        "example_input_grnn",
        2019,
        "RM1",
    ),
]
