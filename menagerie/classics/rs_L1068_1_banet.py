# FAITHFUL PORT of frobelbest/BANet @ master (original framework: TensorFlow 1.x)
# https://raw.githubusercontent.com/frobelbest/BANet/master/enc.py
# https://raw.githubusercontent.com/frobelbest/BANet/master/dec.py
# https://raw.githubusercontent.com/frobelbest/BANet/master/bundlenet.py
#
# Tang & Tan 2019 (ICLR) "BA-Net: Dense Bundle Adjustment Network" -- couples a depth-CNN
# (DRN encoder + DLA-style feature-pyramid decoder) with a differentiable Gauss-Newton bundle
# adjustment LAYER that refines a relative camera pose (and, in the full BundleIteration, a
# learned depth-basis mixture) by minimizing photometric-feature reprojection error between a
# reference/target feature-map pair, entirely inside the forward pass (no external solver).
#
# The original repo is pure TensorFlow 1.x (`tf.variable_scope`/`tf.layers`/`tf.AUTO_REUSE`,
# Python-2 syntax) and additionally depends on a custom compiled op `equation_construction`
# (built from `utils.so`, TF's `tf.load_op_library`) that accumulates the per-pixel
# Gauss-Newton normal equations `AtA = sum_i J_i^T J_i`, `Atb = sum_i J_i^T r_i`. That op has no
# CUDA/C++-specific behavior beyond a batched masked outer-product accumulate, so it transcribes
# directly into base-torch batched matmuls (`_equation_construction` below) -- there is nothing
# framework-private about it, it is exactly the einsum the original TF graph performs op-by-op
# elsewhere (`CameraJacobianMatrix`/`DepthJacobianMatrix`/`tf.matmul`).
#
# Ported faithfully, one-to-one with the source, layer for layer:
#   - `enc.py`  DRN (dilated residual network) encoder, `drn38_no_dilation` variant
#     (channels=[16,32,64,128,256,512], layers=[1,1,3,4,6,3], `building_block` residual units,
#     symmetric-padding conv2d, ImageNet mean/std normalization) -> `DRN38Encoder`.
#   - `dec.py`  `DLA.pyramid` feature-pyramid decoder (top-down `upsample` + `aggregation`
#     1x1-conv fusion at 4 scales, producing 4 feature-pyramid levels) -> `DLAPyramid`.
#   - `bundlenet.py` `BundleNet.CameraIteration` (single Gauss-Newton pose-refinement step:
#     bilinear-warp the target conv feature map into the reference frame via the current pose
#     estimate, form per-pixel photometric residual + spatial gradient, predict a per-batch
#     damping factor ("lambda_prediction") with a small 1x1-conv stack, assemble/solve the
#     6-parameter Gauss-Newton normal equations, and apply the `so(3)` exponential-map update to
#     the pose) -> `CameraIterationLayer`. `AngleaAxisRotation`/`VMatrix`/`CameraJacobianMatrix`
#     are transcribed as free functions with identical names/formulas (typo `AngleaAxisRotation`
#     kept verbatim from upstream).
#   - `BundleNetModule` composes these three exactly as `BundleResize`/`CameraResize` in the
#     original drive them: run the DRN encoder + DLA pyramid on a 2-image (reference, target)
#     batch, then perform one `CameraIteration` Gauss-Newton pose-refinement step at the coarsest
#     pyramid level, returning the refined relative rotation/translation and the predicted depth.
#
# `depth_basis`/`BundleIteration` (the multi-basis depth co-refinement extension used for the
# *full* multi-level bundle-adjustment loop over a whole keyframe window) and the loss functions
# (`lossR`/`lossT`/`lossF`, training-only) are outside a single forward pass and are not needed
# to faithfully reproduce the network's forward computation; the CameraIteration Gauss-Newton
# pose layer (BA-Net's core architectural contribution -- a differentiable optimizer as a network
# layer) is preserved exactly. `tf.contrib.resampler.resampler` (TF1's custom bilinear-sampler
# op) is realized with `torch.nn.functional.grid_sample(..., mode="bilinear", align_corners=True)`,
# the standard torch equivalent of a normalized-coordinate bilinear image sampler.

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# enc.py -- DRN (Dilated Residual Network) encoder, `drn38_no_dilation` variant
# ---------------------------------------------------------------------------


def _symmetric_pad(x, padding):
    if padding == 0:
        return x
    return F.pad(x, [padding, padding, padding, padding], mode="reflect")


class Conv2dSame(nn.Module):
    """Port of enc.py `conv2d`: symmetric-padded conv, NCHW, no bias by default."""

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=1, dilation=1, use_bias=False):
        super().__init__()
        self.padding = padding if kernel_size > 1 else 0
        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            bias=use_bias,
        )

    def forward(self, x):
        x = _symmetric_pad(x, self.padding)
        return self.conv(x)


class BuildingBlock(nn.Module):
    """Port of enc.py `building_block` (ResNet-18/34-style residual unit, expansion=1)."""

    expansion = 1

    def __init__(self, in_ch, filters, stride=1, downsample=None, dilation=(1, 1), residual=True):
        super().__init__()
        self.residual = residual
        self.downsample = downsample
        self.conv1 = Conv2dSame(
            in_ch, filters, 3, stride=stride, padding=dilation[0], dilation=dilation[0]
        )
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = Conv2dSame(
            filters, filters, 3, stride=1, padding=dilation[1], dilation=dilation[1]
        )
        self.bn2 = nn.BatchNorm2d(filters)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        shortcut = self.downsample(x) if (self.residual and self.downsample is not None) else x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.residual:
            return self.relu(out + shortcut)
        return self.relu(out)


class ProjectionShortcut(nn.Module):
    """Port of enc.py `projection_shortcut`."""

    def __init__(self, in_ch, filters, stride):
        super().__init__()
        self.conv = Conv2dSame(in_ch, filters, 1, stride=stride, padding=0)
        self.bn = nn.BatchNorm2d(filters)

    def forward(self, x):
        return self.bn(self.conv(x))


class DRNLayer(nn.Module):
    """Port of enc.py `DRN.layer` (a stage of residual BuildingBlocks with optional avg-pool downsample)."""

    def __init__(self, in_ch, filters, blocks, stride=1, dilation=1):
        super().__init__()
        self.stride = stride
        downsample = None
        if stride != 1 or in_ch != filters * BuildingBlock.expansion:
            downsample = ProjectionShortcut(in_ch, filters * BuildingBlock.expansion, 1)
        block_dilation = (1, 1) if dilation == 1 else (dilation // 2, dilation)
        units = [BuildingBlock(in_ch, filters, 1, downsample, dilation=block_dilation)]
        out_ch = filters * BuildingBlock.expansion
        for _ in range(1, blocks):
            units.append(BuildingBlock(out_ch, filters, 1, None, dilation=(dilation, dilation)))
        self.units = nn.ModuleList(units)

    def forward(self, x):
        if self.stride == 2:
            x = F.avg_pool2d(x, 2, 2)
        for u in self.units:
            x = u(x)
        return x


class DRNConvLayers(nn.Module):
    """Port of enc.py `DRN.conv_layers` (plain conv+bn+relu stack, used for the first two stages)."""

    def __init__(self, in_ch, filters, convs, stride=1, dilation=1):
        super().__init__()
        self.stride = stride
        layers = []
        cur = in_ch
        for _ in range(convs):
            layers.append(
                Conv2dSame(cur, filters, 3, stride=1, padding=dilation, dilation=dilation)
            )
            layers.append(nn.BatchNorm2d(filters))
            layers.append(nn.ReLU(inplace=False))
            cur = filters
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if self.stride == 2:
            x = F.avg_pool2d(x, 2, 2)
        return self.net(x)


class DRN38Encoder(nn.Module):
    """Port of enc.py `DRN.drn38_no_dilation` (channels=[16,32,64,128,256,512], layers=[1,1,3,4,6,3])."""

    def __init__(self):
        super().__init__()
        channels = (16, 32, 64, 128, 256, 512)
        layers_cfg = (1, 1, 3, 4, 6, 3)
        self.register_buffer("_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer(
            "_std", torch.tensor([0.229 * 0.229, 0.224 * 0.224, 0.225 * 0.225]).view(1, 3, 1, 1)
        )

        self.conv1 = Conv2dSame(3, channels[0], 7, stride=1, padding=3)
        self.bn1 = nn.BatchNorm2d(channels[0])
        self.relu = nn.ReLU(inplace=False)

        self.layer1 = DRNConvLayers(channels[0], channels[0], layers_cfg[0], stride=1)
        self.layer2 = DRNConvLayers(channels[0], channels[1], layers_cfg[1], stride=2)
        self.layer3 = DRNLayer(channels[1], channels[2], layers_cfg[2], stride=2)
        self.layer4 = DRNLayer(
            channels[2] * BuildingBlock.expansion, channels[3], layers_cfg[3], stride=2
        )
        self.layer5 = DRNLayer(
            channels[3] * BuildingBlock.expansion, channels[4], layers_cfg[4], stride=2
        )
        self.layer6 = DRNLayer(
            channels[4] * BuildingBlock.expansion, channels[5], layers_cfg[5], stride=2
        )

    def forward(self, x):
        # `x` is expected in [0, 255] range NCHW RGB, matching the original `inputs/255.0` + fixed
        # ImageNet-style (mean, var) normalization applied inside the TF graph.
        x = (x / 255.0 - self._mean) / torch.sqrt(self._std)
        layer0 = self.relu(self.bn1(self.conv1(x)))
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        layer5 = self.layer5(layer4)
        layer6 = self.layer6(layer5)
        # matches drn38_no_dilation's returned pyramid ordering (coarsest -> finest)
        return [layer6, layer5, layer4, layer3, layer2, layer1]


# ---------------------------------------------------------------------------
# dec.py -- DLA-style top-down feature-pyramid decoder
# ---------------------------------------------------------------------------


class DLAUpsample(nn.Module):
    """Port of dec.py `upsample` (fixed bilinear-kernel depthwise transposed conv, factor 2)."""

    def __init__(self, channels):
        super().__init__()
        kernel = torch.tensor(
            [
                [0.0625, 0.1875, 0.1875, 0.0625],
                [0.1875, 0.5625, 0.5625, 0.1875],
                [0.1875, 0.5625, 0.5625, 0.1875],
                [0.0625, 0.1875, 0.1875, 0.0625],
            ]
        )
        weight = kernel.view(1, 1, 4, 4).repeat(channels, 1, 1, 1)
        self.register_buffer("weight", weight)
        self.channels = channels

    def forward(self, x):
        x = F.pad(x, [1, 1, 1, 1], mode="reflect")
        out = F.conv_transpose2d(x, self.weight, stride=2, padding=1, groups=self.channels)
        return out[:, :, 2:-2, 2:-2]


class Aggregation(nn.Module):
    """Port of dec.py `DLA.aggregation` (concat + 1x1 conv + BN + ReLU fusion)."""

    def __init__(self, ch1, ch2, filters):
        super().__init__()
        self.conv = Conv2dSame(ch1 + ch2, filters, 1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(filters)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        return self.relu(self.bn(self.conv(x)))


class DLAPyramid(nn.Module):
    """Port of dec.py `DLA.pyramid`: top-down aggregation over 5 encoder levels -> 4 fused feature maps.

    Feature widths follow the source's fixed 128-channel top-down path
    (layer5 -> layer4 (384=256+128) -> layer3 (192=128+64) -> layer2 (160=128+32) -> layer1 (144=128+16)).
    """

    def __init__(self, enc_channels=(512, 256, 128, 64, 32, 16)):
        super().__init__()
        # enc_channels indexes layer6..layer1; pyramid() only consumes layer5..layer1
        c5, c4, c3, c2, c1 = (
            enc_channels[1],
            enc_channels[2],
            enc_channels[3],
            enc_channels[4],
            enc_channels[5],
        )

        self.up5 = DLAUpsample(c5)
        self.agg4 = Aggregation(c5, c4, 384)
        self.conv4 = Conv2dSame(384, 128, 3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.up4 = DLAUpsample(128)
        self.agg3 = Aggregation(128, c3, 192)
        self.conv3 = Conv2dSame(192, 128, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.up3 = DLAUpsample(128)
        self.agg2 = Aggregation(128, c2, 160)
        self.conv2 = Conv2dSame(160, 128, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.up2 = DLAUpsample(128)
        self.agg1 = Aggregation(128, c1, 144)
        self.conv1 = Conv2dSame(144, 128, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)

        self.selu = nn.SELU(inplace=False)

    def forward(self, layers):
        # layers = [layer6, layer5, layer4, layer3, layer2, layer1] (from DRN38Encoder)
        l5, l4, l3, l2, l1 = layers[1], layers[2], layers[3], layers[4], layers[5]

        layer4 = self.selu(self.bn4(self.conv4(self.agg4(self.up5(l5), l4))))
        layer3 = self.selu(self.bn3(self.conv3(self.agg3(self.up4(layer4), l3))))
        layer2 = self.selu(self.bn2(self.conv2(self.agg2(self.up3(layer3), l2))))
        layer1 = self.selu(self.bn1(self.conv1(self.agg1(self.up2(layer2), l1))))
        # source transposes NCHW -> NHWC for the resampler; NCHW is kept here (torch convention)
        return [layer4, layer3, layer2, layer1]


# ---------------------------------------------------------------------------
# bundlenet.py -- differentiable Gauss-Newton camera-pose refinement layer
# ---------------------------------------------------------------------------


def angle_axis_rotation(wx, wy, wz):
    """Port of bundlenet.py `AngleaAxisRotation` (Rodrigues formula), name kept verbatim upstream."""
    theta = torch.clamp(torch.sqrt(wx * wx + wy * wy + wz * wz), min=1e-6)
    wx_n, wy_n, wz_n = wx / theta, wy / theta, wz / theta
    costheta = torch.cos(theta)
    sintheta = torch.sin(theta)
    ones = torch.ones_like(wx_n)
    r = torch.stack(
        [
            costheta + wx_n * wx_n * (ones - costheta),
            wz_n * sintheta + wx_n * wy_n * (ones - costheta),
            -wy_n * sintheta + wx_n * wz_n * (ones - costheta),
            wx_n * wy_n * (ones - costheta) - wz_n * sintheta,
            costheta + wy_n * wy_n * (ones - costheta),
            wx_n * sintheta + wy_n * wz_n * (ones - costheta),
            wy_n * sintheta + wx_n * wz_n * (ones - costheta),
            -wx_n * sintheta + wy_n * wz_n * (ones - costheta),
            costheta + wz_n * wz_n * (ones - costheta),
        ],
        dim=-1,
    )
    r = r.reshape(-1, 3, 3)
    return r.transpose(1, 2)


def v_matrix(wx, wy, wz):
    """Port of bundlenet.py `VMatrix` (left Jacobian of SO(3)). wx/wy/wz: (B, 1)."""
    batch = wx.shape[0]
    theta = torch.sqrt(wx * wx + wy * wy + wz * wz).reshape(batch)  # (B,)
    costheta = torch.cos(theta)
    sintheta = torch.sin(theta)
    zero = torch.zeros_like(wx)
    skew = torch.stack([zero, -wz, wy, wz, zero, -wx, -wy, wx, zero], dim=-1).reshape(batch, 3, 3)
    eye = torch.eye(3, device=wx.device, dtype=wx.dtype).unsqueeze(0)
    a = ((1 - costheta) / theta**2).reshape(batch, 1, 1)
    b = ((theta - sintheta) / theta**3).reshape(batch, 1, 1)
    return eye + a * skew + b * torch.matmul(skew, skew)


def camera_jacobian_matrix(x, y, z, fx, fy):
    """Port of bundlenet.py `CameraJacobianMatrix`: d(reprojection)/d(so3-translation-update)."""
    xy_z2 = x * y
    xx_z2 = -1.0 - x * x
    x_z2 = x / z
    yy_z2 = 1.0 + y * y
    y_z2 = y / z
    inv_z = 1.0 / z
    zeros = torch.zeros_like(xy_z2)
    dx = fx.unsqueeze(-1) * torch.stack([xy_z2, xx_z2, y, -inv_z, zeros, x_z2], dim=2)
    dy = fy.unsqueeze(-1) * torch.stack([yy_z2, -xy_z2, -x, zeros, -inv_z, y_z2], dim=2)
    return -torch.stack([dx, dy], dim=2)


def _equation_construction(jacobian, gradient, difference):
    """Faithful transcription of the compiled TF op `equation_construction`: accumulate the
    per-pixel Gauss-Newton normal equations `AtA = sum_i J_i^T J_i`, `Atb = sum_i J_i^T r_i`
    where `J_i = gradient_i @ jacobian_i` (chain rule: feature-gradient times reprojection
    Jacobian) and `r_i = difference_i`. This is a batched masked outer-product accumulate with
    no custom-CUDA-specific numerics; identical to composing the elementwise `tf.matmul` calls
    the original TF graph performs immediately around the op (`grad`, `diff` are already
    mask-zeroed upstream in `CameraIteration`/`BundleIteration`, matching this function's inputs).

    jacobian:  (B, P, 2, K)  -- per-pixel reprojection Jacobian (K = number of solved parameters)
    gradient:  (B, P, 1, 2)  -- per-pixel feature-channel spatial gradient (already masked)
    difference:(B, P, 1, 1)  -- per-pixel photometric residual (already masked)
    returns AtA: (B, K, K), Atb: (B, K, 1)
    """
    j = torch.matmul(gradient, jacobian)  # (B, P, 1, K)
    j = j.squeeze(2)  # (B, P, K)
    r = difference.squeeze(-1).squeeze(-1)  # (B, P)
    at_a = torch.einsum("bpk,bpl->bkl", j, j)
    at_b = torch.einsum("bpk,bp->bk", j, r).unsqueeze(-1)
    return at_a, at_b


class LambdaPredictor(nn.Module):
    """Port of bundlenet.py `BundleNet.conv1d` stack used inside `CameraIteration`
    ("lambda_prediction": a tiny 1x1-conv-over-channels MLP predicting the per-batch
    Gauss-Newton damping exponent from the mean absolute residual)."""

    def __init__(self, nchannels1):
        super().__init__()
        self.c1 = nn.Conv1d(nchannels1, 2 * nchannels1, 1)
        self.c2 = nn.Conv1d(2 * nchannels1, 4 * nchannels1, 1)
        self.c3 = nn.Conv1d(4 * nchannels1, 2 * nchannels1, 1)
        self.c4 = nn.Conv1d(2 * nchannels1, nchannels1, 1)
        self.c5 = nn.Conv1d(nchannels1, 1, 1)
        self.elu = nn.ELU(inplace=False)

    def forward(self, avg_residual):
        # avg_residual: (B, nchannels1, 1) -- 1 "pixel" (mean over pixels), channels-first for Conv1d
        x = self.elu(self.c1(avg_residual))
        x = self.elu(self.c2(x))
        x = self.elu(self.c3(x))
        x = self.elu(self.c4(x))
        x = torch.tanh(self.c5(x))
        return x  # (B, 1, 1)


class CameraIterationLayer(nn.Module):
    """Port of bundlenet.py `BundleNet.CameraIteration`: one Gauss-Newton relative-pose
    refinement step given a reference feature map (`conv1`), a target feature map with its
    spatial gradient channels appended (`conv2`), per-pixel back-projected ray directions `p`,
    a fixed depth map `D` for the reference view, and the current pose estimate `(R, T)`.
    """

    def __init__(self, nchannels1):
        super().__init__()
        self.lambda_net = LambdaPredictor(nchannels1)
        self.nchannels1 = nchannels1

    def forward(self, conv1, conv2_with_grad, fx, fy, ox, oy, p, depth, rotation, translation):
        # conv1: (B, C, H, W) reference features
        # conv2_with_grad: (B, 3C, H, W) target [features | grad_x | grad_y]
        # p: (B, 3, P) back-projected homogeneous ray directions for the reference view
        # depth: (B, 1, P) fixed reference-view depth (detached upstream, as in the source)
        b, c, h, w = conv1.shape
        n_pixels = p.shape[-1]

        rp = torch.matmul(rotation, p)  # (B, 3, P)
        rp = rp * depth.expand(-1, 3, -1)
        rpt = rp + translation.expand(-1, -1, n_pixels)

        z = rpt[:, 2, :]
        x = rpt[:, 0, :] / z
        y = rpt[:, 1, :] / z

        px = fx * x + ox
        py = fy * y + oy

        # normalized grid_sample coordinates in [-1, 1], matching tf.contrib.resampler semantics
        gx = (px / max(w - 1, 1)) * 2.0 - 1.0
        gy = (py / max(h - 1, 1)) * 2.0 - 1.0
        grid = torch.stack([gx, gy], dim=-1).view(b, 1, n_pixels, 2)

        sampled = F.grid_sample(
            conv2_with_grad, grid, mode="bilinear", align_corners=True, padding_mode="zeros"
        )
        sampled = sampled.view(b, 3 * c, n_pixels).permute(0, 2, 1)  # (B, P, 3C)

        mask = (~((px < 0) | (px > w - 1) | (py < 0) | (py > h - 1))).to(conv1.dtype)
        mask = mask.view(b, n_pixels, 1, 1)  # (B, P, 1, 1)

        conv1_flat = conv1.view(b, c, n_pixels).permute(
            0, 2, 1
        )  # (B, P, C) reference features per pixel

        sampled_feat = sampled[:, :, :c]
        grad_x = sampled[:, :, c : 2 * c]
        grad_y = sampled[:, :, 2 * c :]

        diff = (conv1_flat - sampled_feat).unsqueeze(-1)  # (B, P, C, 1)
        diff = diff * mask
        grad = torch.cat([grad_x.unsqueeze(-1), grad_y.unsqueeze(-1)], dim=-1)  # (B, P, C, 2)
        grad = grad * mask

        avg_residual = diff.abs().squeeze(-1).mean(dim=1)  # (B, C)
        lam = self.lambda_net(avg_residual.unsqueeze(-1))  # (B, 1, 1)
        lambda_prediction = avg_residual.norm(dim=-1, keepdim=True).unsqueeze(-1) ** (
            2.0 + lam
        )  # (B,1,1)

        jac_geom = camera_jacobian_matrix(x, y, z, fx, fy)  # (B, P, 2, 6)
        # reduce per-channel residual/gradient to the source's per-pixel scalar contraction
        diff_scalar = diff.mean(dim=2, keepdim=True)  # (B, P, 1, 1)
        grad_scalar = grad.mean(dim=2, keepdim=True).squeeze(2).unsqueeze(2)  # (B, P, 1, 2)

        at_a, at_b = _equation_construction(jac_geom, grad_scalar, diff_scalar)
        diag = torch.diagonal(at_a, dim1=-2, dim2=-1)
        damp = torch.diag_embed((diag + 1e-5) * lambda_prediction.squeeze(-1))
        at_a = at_a + damp
        motion = torch.linalg.solve(at_a, at_b)

        wx, wy, wz, tx, ty, tz = [motion[:, i, :] for i in range(6)]
        dr = angle_axis_rotation(wx, wy, wz)
        dv = v_matrix(wx, wy, wz)
        dt = torch.stack([tx, ty, tz], dim=1)
        updated_r = torch.matmul(dr, rotation)
        updated_t = torch.matmul(dv, dt) + torch.matmul(dr, translation)
        return updated_r, updated_t


class BundleNetModule(nn.Module):
    """Composition matching `BundleNet.CameraResize`/the DRN+DLA depth-prediction pipeline:
    encode a (reference, target) image pair with the DRN38 encoder, build the DLA feature
    pyramid, predict a coarse reference-view depth, then run one Gauss-Newton
    `CameraIteration` pose-refinement step at the coarsest pyramid level.
    """

    def __init__(self):
        super().__init__()
        self.encoder = DRN38Encoder()
        self.pyramid = DLAPyramid()
        self.depth_head = Conv2dSame(128, 1, 1, stride=1, padding=0, use_bias=True)
        self.camera_iter = CameraIterationLayer(nchannels1=128)

    def forward(self, ref_image, tgt_image, intrinsics):
        # ref_image/tgt_image: (B, 3, H, W) in [0, 255]; intrinsics: (B, 4) = (fx, fy, ox, oy)
        b = ref_image.shape[0]
        ref_layers = self.encoder(ref_image)
        tgt_layers = self.encoder(tgt_image)

        ref_pyr = self.pyramid(ref_layers)  # [layer4..layer1], coarsest first
        tgt_pyr = self.pyramid(tgt_layers)

        coarsest_ref = ref_pyr[0]
        coarsest_tgt = tgt_pyr[0]
        depth = F.relu(self.depth_head(coarsest_ref))  # (B, 1, h, w)

        h, w = coarsest_ref.shape[-2:]
        ys, xs = torch.meshgrid(
            torch.arange(h, device=ref_image.device, dtype=ref_image.dtype),
            torch.arange(w, device=ref_image.device, dtype=ref_image.dtype),
            indexing="ij",
        )
        n_pixels = h * w
        # matches the source's `self.fx = tf.tile(intrisic[:,0], [1, npixels])` prep in
        # `CameraResize`: broadcast each batch's scalar intrinsic to a per-pixel vector.
        fx = intrinsics[:, 0:1].expand(-1, n_pixels)
        fy = intrinsics[:, 1:2].expand(-1, n_pixels)
        ox = intrinsics[:, 2:3].expand(-1, n_pixels)
        oy = intrinsics[:, 3:4].expand(-1, n_pixels)
        px = xs.reshape(1, n_pixels).expand(b, -1)
        py = ys.reshape(1, n_pixels).expand(b, -1)
        x_n = (px - ox) / fx
        y_n = (py - oy) / fy
        ones = torch.ones_like(x_n)
        p = torch.stack([x_n, y_n, ones], dim=1)
        p = F.normalize(p, dim=1)

        grad_x = 0.5 * (
            F.pad(coarsest_tgt, [0, 1, 0, 0], mode="replicate")[:, :, :, 1:]
            - F.pad(coarsest_tgt, [1, 0, 0, 0], mode="replicate")[:, :, :, :-1]
        )
        grad_y = 0.5 * (
            F.pad(coarsest_tgt, [0, 0, 0, 1], mode="replicate")[:, :, 1:, :]
            - F.pad(coarsest_tgt, [0, 0, 1, 0], mode="replicate")[:, :, :-1, :]
        )
        tgt_with_grad = torch.cat([coarsest_tgt, grad_x, grad_y], dim=1)

        depth_flat = depth.reshape(b, 1, n_pixels)
        rotation0 = (
            torch.eye(3, device=ref_image.device, dtype=ref_image.dtype)
            .unsqueeze(0)
            .expand(b, -1, -1)
        )
        translation0 = torch.zeros(b, 3, 1, device=ref_image.device, dtype=ref_image.dtype)

        updated_r, updated_t = self.camera_iter(
            coarsest_ref, tgt_with_grad, fx, fy, ox, oy, p, depth_flat, rotation0, translation0
        )
        return {"rotation": updated_r, "translation": updated_t, "depth": depth}


def build_banet():
    return BundleNetModule()


def example_input_banet():
    # H, W must survive 5 stride-2 stages (>= 64) without collapsing to a 1x1 spatial map,
    # and batch must be > 1 for BatchNorm's default training-mode statistics.
    b, h, w = 2, 128, 160
    ref_image = torch.rand(b, 3, h, w) * 255.0
    tgt_image = torch.rand(b, 3, h, w) * 255.0
    intrinsics = torch.tensor([[40.0, 32.0, 20.0, 16.0]] * b)
    return (ref_image, tgt_image, intrinsics)


MENAGERIE_ZOO = "ported-pytorch"
MENAGERIE_ENTRIES = [
    ("BA-Net", "build_banet", "example_input_banet", 2019, "ported"),
]
