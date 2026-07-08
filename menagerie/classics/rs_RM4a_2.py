# FAITHFUL PORT of ricsonc/grnn @ master (original framework: TensorFlow 1.x + tf.contrib.slim)
# https://github.com/ricsonc/grnn
# Paper: Tung, Cheng & Fragkiadaki, "Learning Spatial Common Sense with Geometry-Aware
# Recurrent Networks" (GRNN), CVPR 2019. The same GRNN architecture (self-supervised 3D
# feature memory trained to predict novel camera views) is reused, unmodified, as the "3D
# View Prediction Model of the Dorsal Visual Stream" in Sarch, Tung, Wang, Prince & Tarr,
# "3D View Prediction Models of the Dorsal Visual Stream" (CCN 2023 / arXiv:2309.01782),
# which is what this menagerie candidate names.
#
# The real repo is TensorFlow-1.x (tf.contrib.slim, tf.placeholder, no eager default) and its
# `tensorflow.contrib` dependency cannot be installed into a modern base env -- so per the
# source ladder this is a RUNG-3 FAITHFUL PORT, not a RUNG-2 vendor.
#
# What is transcribed faithfully from the real repo (utils/nets.py, utils/voxel.py,
# nets.py::GQN3D), mechanism-for-mechanism:
#   - encoder2D            (utils/nets.py)        -- multi-scale 2D CNN encoder
#   - get_transform_matrix_tf_ (utils/voxel.py)    -- spherical camera extrinsic@intrinsic 4x4
#   - transformer / _meshgrid / _invproj_meshgrid / _noproj_meshgrid (utils/voxel.py)
#                                                   -- the perspective-transformer voxel warp;
#                                                      re-expressed with torch.nn.functional
#                                                      .grid_sample (5D volumetric grid_sample
#                                                      IS the same trilinear-interpolation
#                                                      camera-matrix warp as their hand-rolled
#                                                      TF `_interpolate`, just not hand-rolled)
#   - unproject_image / unproject_voxel (utils/voxel.py) -- lift a 2D feature map to a 3D
#                                                      voxel memory via the inverse camera warp
#   - translate_given_angles (utils/voxel.py)      -- egomotion-stabilized alignment: rotate
#                                                      each view's voxel memory into a common
#                                                      (first-view / query-view) frame
#   - encoder_decoder3D    (utils/nets.py)         -- the 3D "conv3d down / conv3d_transpose
#                                                      up" memory processor, additively fusing
#                                                      the multi-scale unprojected memory at
#                                                      each matching resolution (GQN3D.predict's
#                                                      get_outputs3D)
#   - project_voxel        (utils/voxel.py)        -- canonical (theta=phi=0) forward camera
#                                                      warp, projecting the 3D memory back for
#                                                      the query view (GQN3D.get_inputs2Ddec)
#   - depth_channel_net_v2 (utils/nets.py)         -- collapse the projected voxel's depth axis
#                                                      (maxpool + 1x1 conv2d) into a 2D feature
#   - decoder2D            (utils/nets.py)         -- the plain feed-forward conv2d image head.
#     GQN3D's *default* config (`grnn_shapenet_train`) instead selects the stochastic
#     ConvLSTM/DRAW decoder (`GQN3D_CONVLSTM=True` -> utils/gqn_network.make_lstmConv, a
#     separate, reusable ~700-line GQN component that is not specific to GRNN's 3D-vs-2D
#     contribution). `decoder2D` is the alternative, non-recurrent head already present
#     verbatim in the same nets.py, gated by the same `GQN3D_CONVLSTM` flag; it is used here
#     to keep this port self-contained while still tracing the real 2D-encode -> unproject ->
#     egomotion-align -> aggregate -> 3D-UNet -> project -> depth-collapse -> decode pipeline
#     end to end.
#
# Per constants.py's actual 'shapenet'/GRNN config (OG(...,'gqn3d_shapenet_base',...)):
# USE_MESHGRID=False, USE_OUTLINE=False, AGGREGATION_METHOD='average' -- so the extra
# meshgrid-z / outline channels `unproject()` can optionally append are OFF here too (as in
# the real trained GRNN-shapenet model), and view aggregation is a plain mean.
#
# Sizes/channel counts below are a "tiny config" (image 16x16, 3 encoder scales instead of the
# repo's hardcoded H=W in {64,128} 4-scale ladder) for fast TorchLens tracing; the *ladder
# mechanism* -- start the 3D UNet bottleneck at the largest unprojected scale, downsample while
# additively fusing each next-smaller unprojected scale, then symmetric upsample while fusing
# the matching encoder skip -- is preserved exactly, just generalized from 4 rungs to 3 so the
# channel/spatial bookkeeping (real repo: encoder2D dims [32,64,128,256] line up with
# encoder_decoder3D's d0*[1,2,4,8] exactly when USE_MESHGRID/USE_OUTLINE are off) is kept
# self-consistent at a smaller scale ([4,8,16] here).
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def _camera_matrix(
    theta_deg: torch.Tensor,
    phi_deg: torch.Tensor,
    radius: float,
    focal_length: float,
    invert_rot: bool = False,
    invert_focal: bool = False,
) -> torch.Tensor:
    """Port of voxel.get_transform_matrix_tf_: batched spherical-camera extrinsic@intrinsic."""
    sin_phi = torch.sin(phi_deg * math.pi / 180.0)
    cos_phi = torch.cos(phi_deg * math.pi / 180.0)
    sin_theta = torch.sin(theta_deg * math.pi / 180.0)
    cos_theta = torch.cos(theta_deg * math.pi / 180.0)
    zeros = torch.zeros_like(sin_phi)
    ones = torch.ones_like(sin_phi)
    bs = theta_deg.shape[0]

    rot_azimuth = torch.stack(
        [cos_theta, zeros, -sin_theta, zeros, ones, zeros, sin_theta, zeros, cos_theta], dim=-1
    ).view(bs, 3, 3)
    rot_elevation = torch.stack(
        [cos_phi, sin_phi, zeros, -sin_phi, cos_phi, zeros, zeros, zeros, ones], dim=-1
    ).view(bs, 3, 3)

    rot = torch.bmm(rot_azimuth, rot_elevation)
    if invert_rot:
        rot = torch.linalg.inv(rot)

    displacement = torch.zeros(bs, 3, 1, dtype=rot.dtype, device=rot.device)
    displacement[:, 0, 0] = radius
    displacement = torch.bmm(rot, displacement)

    bottom_row = torch.zeros(bs, 1, 4, dtype=rot.dtype, device=rot.device)
    bottom_row[:, 0, 3] = 1.0
    extrinsic = torch.cat([torch.cat([rot, -displacement], dim=2), bottom_row], dim=1)

    if invert_focal:
        diag = torch.tensor(
            [1.0, focal_length, focal_length, 1.0], dtype=rot.dtype, device=rot.device
        )
    else:
        diag = torch.tensor(
            [1.0, 1.0 / focal_length, 1.0 / focal_length, 1.0], dtype=rot.dtype, device=rot.device
        )
    intrinsic = torch.diag(diag).unsqueeze(0).expand(bs, 4, 4)
    return torch.bmm(extrinsic, intrinsic)


def _meshgrid_coords(
    d: int,
    h: int,
    w: int,
    z_near: float,
    z_far: float,
    mode: str,
    focal_length: float,
    device,
    dtype,
) -> torch.Tensor:
    """Port of voxel._meshgrid / _invproj_meshgrid / _noproj_meshgrid (the 3 `do_project`
    variants), returning the homogeneous (4, D*H*W) ray grid [d_t, y_t, x_t, 1]."""
    x_t = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype).view(1, 1, w).expand(d, h, w)
    y_t = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype).view(1, h, 1).expand(d, h, w)
    z_lin = (
        torch.linspace(z_near, z_far, d, device=device, dtype=dtype).view(d, 1, 1).expand(d, h, w)
    )

    if mode == "project":  # voxel.py::_meshgrid -- x_t /= z_t, with z_t = 1/z_lin
        x_t = x_t * z_lin
        y_t = y_t * z_lin
    elif mode == "unproject":  # voxel.py::_invproj_meshgrid -- x_t *= z_t
        x_t = x_t / z_lin
        y_t = y_t / z_lin
    elif mode == "rotate":  # voxel.py::_noproj_meshgrid -- x_t *= const.focal_length
        x_t = x_t * focal_length
        y_t = y_t * focal_length
    else:
        raise ValueError(mode)

    ones = torch.ones_like(x_t)
    return torch.stack([z_lin, y_t, x_t, ones], dim=0).reshape(4, -1)


def _perspective_warp(
    voxel: torch.Tensor,
    cam_mat: torch.Tensor,
    z_near: float,
    z_far: float,
    mode: str,
    focal_length: float,
) -> torch.Tensor:
    """Port of voxel.transformer: apply a 4x4 camera matrix to a voxel grid via the same
    ray-grid construction as the original, using grid_sample for the trilinear resample
    (the differentiable-warp mechanism `_interpolate` hand-rolled with tf.gather)."""
    bs, _, d, h, w = voxel.shape
    grid4 = _meshgrid_coords(d, h, w, z_near, z_far, mode, focal_length, voxel.device, voxel.dtype)
    grid4 = grid4.unsqueeze(0).expand(bs, 4, -1)
    warped = torch.bmm(cam_mat, grid4)  # (BS, 4, D*H*W) -> rows [z_s, y_s, x_s, 1]
    z_s, y_s, x_s = warped[:, 0], warped[:, 1], warped[:, 2]
    grid = torch.stack([x_s, y_s, z_s], dim=-1).view(bs, d, h, w, 3)
    return F.grid_sample(voxel, grid, mode="bilinear", padding_mode="zeros", align_corners=True)


class GRNNGeometry:
    """Bundles the fixed camera constants (radius/focal_length/near/far), matching
    constants.py's defaults (fov=30, radius=4.0, SCENE_SIZE=1.0)."""

    def __init__(self, radius: float = 4.0, fov_deg: float = 30.0, scene_size: float = 1.0):
        self.radius = radius
        self.focal_length = 1.0 / math.tan(fov_deg * math.pi / 360.0)
        self.z_near = radius - scene_size
        self.z_far = radius + scene_size

    def unproject(self, feat2d: torch.Tensor) -> torch.Tensor:
        """voxel.unproject_image + unproject_voxel: broadcast a 2D feature map across a new
        depth axis, then warp with the canonical (theta=phi=0), focal-inverted camera."""
        bs, c, size = feat2d.shape[0], feat2d.shape[1], feat2d.shape[2]
        voxel = feat2d.unsqueeze(2).expand(bs, c, size, size, size).contiguous()
        theta0 = torch.zeros(bs, dtype=feat2d.dtype, device=feat2d.device)
        cam = _camera_matrix(theta0, theta0, self.radius, self.focal_length, invert_focal=True)
        return _perspective_warp(
            voxel, cam, self.z_near, self.z_far, "unproject", self.focal_length
        )

    def rotate(
        self,
        voxel: torch.Tensor,
        dtheta_deg: torch.Tensor,
        phi1_deg: torch.Tensor,
        phi2_deg: torch.Tensor,
    ) -> torch.Tensor:
        """voxel.translate_given_angles: kill the source elevation, then rotate to the target
        (theta, phi) -- the egomotion-stabilization step that aligns a view's 3D memory into a
        common frame before aggregation / after decoding."""
        zeros = torch.zeros_like(phi1_deg)
        cam1 = _camera_matrix(zeros, -phi1_deg, self.radius, self.focal_length)
        cam2 = _camera_matrix(dtheta_deg, phi2_deg, self.radius, self.focal_length)
        voxel = _perspective_warp(voxel, cam1, self.z_near, self.z_far, "rotate", self.focal_length)
        voxel = _perspective_warp(voxel, cam2, self.z_near, self.z_far, "rotate", self.focal_length)
        return voxel

    def project(self, voxel: torch.Tensor) -> torch.Tensor:
        """voxel.project_voxel: canonical (theta=phi=0) forward perspective warp, projecting
        the (already query-aligned) 3D memory back down for 2D decoding."""
        bs = voxel.shape[0]
        zeros = torch.zeros(bs, dtype=voxel.dtype, device=voxel.device)
        cam = _camera_matrix(zeros, zeros, self.radius, self.focal_length)
        return _perspective_warp(voxel, cam, self.z_near, self.z_far, "project", self.focal_length)


class Encoder2D(nn.Module):
    """Port of utils.nets.encoder2D: repeated (conv stride2 -> relu -> BN, conv stride1 ->
    relu -> BN), returning the feature map after every stage (largest spatial size first)."""

    def __init__(self, in_channels: int = 3, dims=(4, 8, 16)):
        super().__init__()
        self.stages = nn.ModuleList()
        c = in_channels
        for dim in dims:
            self.stages.append(
                nn.ModuleList(
                    [
                        nn.Conv2d(c, dim, 3, stride=2, padding=1),
                        nn.BatchNorm2d(dim),
                        nn.Conv2d(dim, dim, 3, stride=1, padding=1),
                        nn.BatchNorm2d(dim),
                    ]
                )
            )
            c = dim

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        outputs = []
        net = x
        for conv1, bn1, conv2, bn2 in self.stages:
            net = bn1(F.relu(conv1(net)))
            net = bn2(F.relu(conv2(net)))
            outputs.append(net)
        return outputs


class EncoderDecoder3D(nn.Module):
    """Port of utils.nets.encoder_decoder3D: a 3D conv down-ladder that additively fuses each
    successively-smaller unprojected memory scale, then a symmetric conv_transpose up-ladder
    that additively fuses the matching encoder skip connection. Generalized from the repo's
    hardcoded H=W in {64,128} 4-scale ladder to N scales (N=len(dims)) for tiny tracing."""

    def __init__(self, dims=(4, 8, 16)):
        super().__init__()
        self.n = len(dims)
        enc_dims = list(dims[1:]) + [dims[-1] * 2]  # e.g. (4,8,16) -> down out chans (8,16,32)
        self.down = nn.ModuleList()
        in_c = dims[0]
        for i, out_c in enumerate(enc_dims):
            last = i == len(enc_dims) - 1
            ksize, stride, pad = (2, 2, 0) if last else (4, 2, 1)
            self.down.append(
                nn.ModuleDict(
                    {
                        "conv": nn.Conv3d(in_c, out_c, ksize, stride=stride, padding=pad),
                        "bn": nn.BatchNorm3d(out_c),
                    }
                )
            )
            in_c = out_c

        dec_dims = list(reversed(dims))  # channel target of each upsample = matching skip's ch
        self.up = nn.ModuleList()
        in_c = enc_dims[-1]
        for i, out_c in enumerate(dec_dims):
            first = i == 0
            ksize, stride, pad = (2, 2, 0) if first else (4, 2, 1)
            self.up.append(
                nn.ModuleDict(
                    {
                        "conv": nn.ConvTranspose3d(in_c, out_c, ksize, stride=stride, padding=pad),
                        "bn": nn.BatchNorm3d(out_c),
                    }
                )
            )
            in_c = out_c

    def forward(self, scales: list[torch.Tensor]) -> list[torch.Tensor]:
        # `scales`: unprojected memory at each Encoder2D resolution, largest spatial first.
        remaining = list(scales[1:])  # consumed smallest-scale-last, matching voxel.pop() order
        net = scales[0]
        skipcons = [net]
        for stage in self.down:
            net = F.relu(stage["conv"](net))
            net = stage["bn"](net)
            if remaining:
                net = net + remaining.pop(0)
            skipcons.append(net)
        skipcons.pop()  # the innermost (bottleneck) layer is not reused as a skip

        outputs = []
        for stage in self.up:
            net = F.relu(stage["conv"](net))
            net = stage["bn"](net)
            if skipcons:
                net = net + skipcons.pop()
            outputs.append(net)
        return outputs


def _depth_channel_net(feature: torch.Tensor, conv: nn.Conv2d, pool_k: int) -> torch.Tensor:
    """Port of utils.nets.depth_channel_net_v2: maxpool the depth axis, flatten depth*channel,
    1x1 conv2d back down to a fixed 2D channel count."""
    bs, c, d, h, w = feature.shape
    feature = feature.permute(0, 1, 3, 4, 2)  # move depth to the end like TF's NDHWC layout
    feature = F.max_pool3d(feature, kernel_size=(1, 1, pool_k), stride=(1, 1, pool_k))
    _, c2, h2, w2, d2 = feature.shape
    feature = feature.permute(0, 1, 4, 2, 3).reshape(bs, c2 * d2, h2, w2)
    return conv(feature)


class GRNN(nn.Module):
    """GRNN (Tung, Cheng & Fragkiadaki, CVPR 2019) / the "3D View Prediction Model of the
    Dorsal Visual Stream" (Sarch et al. 2023): encode each context view with a 2D CNN,
    unproject every scale into a 3D voxel memory via the real camera-projective warp,
    egomotion-align and average the per-view memories, refine with a 3D conv U-Net, then
    rotate/project/decode the fused memory into the query view's predicted RGB image."""

    def __init__(self, dims=(4, 8, 16), img_size: int = 16):
        super().__init__()
        self.dims = dims
        self.img_size = img_size
        self.encoder2d = Encoder2D(in_channels=3, dims=dims)
        self.encoder_decoder3d = EncoderDecoder3D(dims=dims)
        self.geometry = GRNNGeometry()

        pool_k = 2
        self.pool_k = pool_k
        # depth_channel_net_v2 collapses (channels * depth/pool_k) -> channels, per scale.
        # decoded_scales (EncoderDecoder3D.forward's `outputs`) are produced smallest-spatial
        # first, i.e. in reversed(dims)/reversed(sizes) order -- match that order here so
        # zip(projected, self.depth_conv) in forward() pairs up correctly.
        sizes = list(reversed(self._scale_sizes()))
        self.depth_conv = nn.ModuleList(
            [nn.Conv2d(c * max(1, s // pool_k), c, 1) for c, s in zip(reversed(dims), sizes)]
        )

        # decoder2D: `net = 0; for dim in dims: net += features.pop(); net = conv2d(net, dim);
        # net = upsample2x(net)`. Each conv's OUTPUT channel count must match the channel count
        # of the *next* feature about to be added (that's what makes the running `net +=`
        # additions channel-consistent) -- i.e. conv_i projects to channels(depth_collapsed[i+1]),
        # with the final stage's target left free (no further addition follows it).
        rev_dims = list(reversed(dims))  # == channel count of depth_collapsed[i], in order
        conv_targets = rev_dims[1:] + [rev_dims[-1]]
        decoder_convs = []
        in_c = rev_dims[0]
        for target in conv_targets:
            decoder_convs.append(nn.Conv2d(in_c, target, 3, stride=1, padding=1))
            in_c = target
        self.decoder_convs = nn.ModuleList(decoder_convs)
        self.decoder_bns = nn.ModuleList([nn.BatchNorm2d(target) for target in conv_targets])
        self.decoder_out = nn.Conv2d(conv_targets[-1], 3, 3, stride=1, padding=1)

    def _scale_sizes(self) -> list[int]:
        sizes = []
        s = self.img_size
        for _ in self.dims:
            s = s // 2
            sizes.append(s)
        return sizes

    def _encode_and_unproject(self, img: torch.Tensor) -> list[torch.Tensor]:
        feats = self.encoder2d(img)
        return [self.geometry.unproject(f) for f in feats]

    def forward(
        self,
        context_images: torch.Tensor,
        context_thetas: torch.Tensor,
        context_phis: torch.Tensor,
        query_theta: torch.Tensor,
        query_phi: torch.Tensor,
    ) -> torch.Tensor:
        """context_images: (B, V, 3, H, W); context_thetas/phis: (B, V) degrees;
        query_theta/phi: (B,) degrees. Returns the predicted query-view RGB image."""
        num_views = context_images.shape[1]

        per_view_scales = []
        for v in range(num_views):
            scales = self._encode_and_unproject(context_images[:, v])
            dtheta = context_thetas[:, 0] - context_thetas[:, v]
            phi1 = context_phis[:, v]
            phi2 = context_phis[:, 0]
            aligned = [self.geometry.rotate(s, dtheta, phi1, phi2) for s in scales]
            per_view_scales.append(aligned)

        aggregated = [
            torch.stack([per_view_scales[v][i] for v in range(num_views)], dim=0).mean(dim=0)
            for i in range(len(self.dims))
        ]

        decoded_scales = self.encoder_decoder3d(aggregated)

        dtheta_q = query_theta - context_thetas[:, 0]
        phi1_q = context_phis[:, 0]
        phi2_q = query_phi
        query_aligned = [self.geometry.rotate(s, dtheta_q, phi1_q, phi2_q) for s in decoded_scales]
        projected = [self.geometry.project(s) for s in query_aligned]

        # `projected` (like decoded_scales) is smallest-spatial-first; decoder2D's own
        # `features[::-1]` (largest-first) immediately gets consumed back-to-front via
        # `.pop()`, i.e. smallest-first again -- so we iterate `projected` directly, unreversed.
        depth_collapsed = [
            _depth_channel_net(feat, conv, self.pool_k)
            for feat, conv in zip(projected, self.depth_conv)
        ]

        net = None
        for i, (feat, conv, bn) in enumerate(
            zip(depth_collapsed, self.decoder_convs, self.decoder_bns)
        ):
            net = feat if net is None else net + feat
            net = bn(F.relu(conv(net)))
            net = F.interpolate(net, scale_factor=2, mode="nearest")

        net = self.decoder_out(net)
        return torch.tanh(net) * 0.5 + 0.5  # tfutil.tanh01


def build_grnn() -> GRNN:
    """Build a tiny GRNN (3D view-prediction / dorsal-stream model) for TorchLens tracing."""
    model = GRNN(dims=(4, 8, 16), img_size=16)
    model.eval()
    return model


def example_input_grnn():
    """Two context views (RGB images + spherical camera angles) and one query camera pose,
    matching GQN3D.setup_data's (context.frames, context.cameras, query_camera) inputs."""
    bs, num_views, size = 1, 2, 16
    context_images = torch.randn(bs, num_views, 3, size, size)
    context_thetas = torch.tensor([[0.0, 90.0]])
    context_phis = torch.tensor([[20.0, 20.0]])
    query_theta = torch.tensor([45.0])
    query_phi = torch.tensor([20.0])
    return context_images, context_thetas, context_phis, query_theta, query_phi


MENAGERIE_ZOO = "ported-pytorch"
MENAGERIE_ENTRIES = [
    ("GRNN", "build_grnn", "example_input_grnn", 2019, "ported-pytorch"),
]
