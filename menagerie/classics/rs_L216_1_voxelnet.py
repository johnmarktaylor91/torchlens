# SOURCE: vendored from Hqss/VoxelNet_PyTorch @ master (fetched 2026-07-01)
# (https://github.com/Hqss/VoxelNet_PyTorch)
#
# VoxelNet (Zhou & Tuzel, CVPR 2018, "VoxelNet: End-to-End Learning for Point
# Cloud Based 3D Object Detection"). No official code release from the Apple
# authors; Hqss/VoxelNet_PyTorch is a widely-used, faithful community PyTorch
# reimplementation (the architecture itself -- VFE stacked voxel-feature
# encoding + a 3D-conv "middle layer" + a 2D-conv RPN with multi-scale
# deconv fusion -- is transcribed directly from the paper's Fig. 2/3/4 and is
# the de-facto reference PyTorch port cited by downstream repos).
#
# Files vendored (near-verbatim; only the shape/knob source is changed from
# a module-level `cfg` singleton to explicit constructor arguments so the
# module can be built standalone, and the KITTI-label loss/NMS/eval code in
# the original `model/model.py::RPN3D` is dropped since it is training/eval
# scaffolding, not architecture -- see below):
#   - model/group_pointcloud.py  (VFELayer, FeatureNet -- stacked Voxel
#     Feature Encoding, verbatim compute graph)
#   - model/rpn.py                (ConvMD, Deconv2D, MiddleAndRPN -- verbatim
#     compute graph: 3D middle conv layer -> 3 parallel 2D conv blocks with
#     deconv fusion -> sigmoid prob head + regression head)
#
# What was intentionally NOT vendored (RPN3D.forward in model/model.py):
# the real `RPN3D.forward` interleaves the network forward pass with
# `cal_rpn_target(label, ...)` (reads KITTI ground-truth label files),
# anchor-matching, and a hand-rolled BCE + smooth-L1 loss -- this is a
# training-loop / label-reading wrapper around the network, not part of the
# VoxelNet architecture itself (analogous to dropping a Lightning training
# step to keep the bare nn.Module). The two real submodules it wraps
# (`self.feature = FeatureNet()`, `self.rpn = MiddleAndRPN()`) ARE the
# architecture and are vendored here verbatim, composed the same way
# (`FeatureNet` -> dense-voxel-grid -> `MiddleAndRPN`).
#
# Config choice for tracing (shrunk purely in scale from the real KITTI "Car"
# config in config.py -- every shape relation/layer is identical to the real
# code, only the raw voxel-grid extent is smaller so capture stays small; the
# depth must land on final middle-layer depth==2 -- same as the real
# INPUT_DEPTH=10 config -- so the post-middle_layer `.view(batch, -1, H, W)`
# reshape (64 channels * depth) matches block1's hardcoded 128 in_channels,
# unchanged from the real code):
#   real: INPUT_DEPTH=10, INPUT_HEIGHT=400, INPUT_WIDTH=352, VOXEL_POINT_COUNT=35
#   here: depth=9,        height=32,        width=32,       voxel_point_count=8
# FeatureNet's per-voxel VFE stack (7 -> 32 -> 128 channels) and MiddleAndRPN's
# channel counts (128 -> 64 middle -> 128/256 RPN blocks) are untouched.
#
# Environment substitution (NOT an architecture change): the original
# `FeatureNet.forward` scatters per-voxel features into a dense grid via
# `torch.sparse.FloatTensor(...).to_dense()` (legacy pre-1.x sparse-tensor
# constructor, removed in modern PyTorch). We use the modern equivalent
# `torch.sparse_coo_tensor(...).to_dense()` -- same sparse-COO-to-dense
# semantics, just the current non-deprecated constructor name.
#
# MENAGERIE_ZOO = "vendored-pytorch"

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


# --------------------------------------------------------------------------- #
# model/group_pointcloud.py (vendored, cfg.VOXEL_POINT_COUNT -> ctor arg)
# --------------------------------------------------------------------------- #


class VFELayer(nn.Module):
    def __init__(self, in_channels, out_channels, voxel_point_count):
        super(VFELayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.units = int(out_channels / 2)
        self.voxel_point_count = voxel_point_count

        self.dense = nn.Sequential(nn.Linear(self.in_channels, self.units), nn.ReLU())
        self.batch_norm = nn.BatchNorm1d(self.units)

    def forward(self, inputs, mask):
        # [SigmaK, T, in_ch] -> [SigmaK, T, units] -> [SigmaK, units, T]
        tmp = self.dense(inputs).transpose(1, 2)
        # [SigmaK, units, T] -> [SigmaK, T, units]
        pointwise = self.batch_norm(tmp).transpose(1, 2)

        # [SigmaK, 1, units]
        aggregated, _ = torch.max(pointwise, dim=1, keepdim=True)

        # [SigmaK, T, units]
        repeated = aggregated.expand(-1, self.voxel_point_count, -1)

        # [SigmaK, T, 2 * units]
        concatenated = torch.cat([pointwise, repeated], dim=2)

        # [SigmaK, T, 1] -> [SigmaK, T, 2 * units]
        mask = mask.expand(-1, -1, 2 * self.units)

        concatenated = concatenated * mask.float()

        return concatenated


class FeatureNet(nn.Module):
    def __init__(self, voxel_point_count, input_depth, input_height, input_width):
        super(FeatureNet, self).__init__()

        self.voxel_point_count = voxel_point_count
        self.input_depth = input_depth
        self.input_height = input_height
        self.input_width = input_width

        self.vfe1 = VFELayer(7, 32, voxel_point_count)
        self.vfe2 = VFELayer(32, 128, voxel_point_count)

    def forward(self, feature, number, coordinate):
        batch_size = len(feature)

        feature = torch.cat(feature, dim=0)  # [SigmaK, voxel_point_count, 7]
        coordinate = torch.cat(coordinate, dim=0)  # [SigmaK, 4]; (batch, d, h, w)

        vmax, _ = torch.max(feature, dim=2, keepdim=True)
        mask = vmax != 0  # [SigmaK, T, 1]

        x = self.vfe1(feature, mask)
        x = self.vfe2(x, mask)

        # [SigmaK, 128]
        voxelwise, _ = torch.max(x, dim=1)

        # Car: [B, D, H, W, 128]
        outputs = torch.sparse_coo_tensor(
            coordinate.t(),
            voxelwise,
            torch.Size([batch_size, self.input_depth, self.input_height, self.input_width, 128]),
        )

        outputs = outputs.to_dense()

        return outputs


# --------------------------------------------------------------------------- #
# model/rpn.py (vendored, cfg.DETECT_OBJ / cfg.FEATURE_* -> ctor args)
# --------------------------------------------------------------------------- #


class ConvMD(nn.Module):
    def __init__(self, M, cin, cout, k, s, p, bn=True, activation=True):
        super(ConvMD, self).__init__()

        self.M = M  # Dimension of input
        self.cin = cin
        self.cout = cout
        self.k = k
        self.s = s
        self.p = p
        self.bn = bn
        self.activation = activation

        if self.M == 2:  # 2D input
            self.conv = nn.Conv2d(self.cin, self.cout, self.k, self.s, self.p)
            if self.bn:
                self.batch_norm = nn.BatchNorm2d(self.cout)
        elif self.M == 3:  # 3D input
            self.conv = nn.Conv3d(self.cin, self.cout, self.k, self.s, self.p)
            if self.bn:
                self.batch_norm = nn.BatchNorm3d(self.cout)
        else:
            raise Exception("No such mode!")

    def forward(self, inputs):
        out = self.conv(inputs)

        if self.bn:
            out = self.batch_norm(out)

        if self.activation:
            return F.relu(out)
        else:
            return out


class Deconv2D(nn.Module):
    def __init__(self, cin, cout, k, s, p, bn=True):
        super(Deconv2D, self).__init__()

        self.cin = cin
        self.cout = cout
        self.k = k
        self.s = s
        self.p = p
        self.bn = bn

        self.deconv = nn.ConvTranspose2d(self.cin, self.cout, self.k, self.s, self.p)

        if self.bn:
            self.batch_norm = nn.BatchNorm2d(self.cout)

    def forward(self, inputs):
        out = self.deconv(inputs)

        if self.bn:
            out = self.batch_norm(out)

        return F.relu(out)


class MiddleAndRPN(nn.Module):
    def __init__(self, feature_height, feature_width, detect_obj="Car"):
        super(MiddleAndRPN, self).__init__()

        self.middle_layer = nn.Sequential(
            ConvMD(3, 128, 64, 3, (2, 1, 1), (1, 1, 1)),
            ConvMD(3, 64, 64, 3, (1, 1, 1), (0, 1, 1)),
            ConvMD(3, 64, 64, 3, (2, 1, 1), (1, 1, 1)),
        )

        if detect_obj == "Car":
            self.block1 = nn.Sequential(
                ConvMD(2, 128, 128, 3, (2, 2), (1, 1)),
                ConvMD(2, 128, 128, 3, (1, 1), (1, 1)),
                ConvMD(2, 128, 128, 3, (1, 1), (1, 1)),
                ConvMD(2, 128, 128, 3, (1, 1), (1, 1)),
                ConvMD(2, 128, 128, 3, (1, 1), (1, 1)),
            )
        else:  # Pedestrian/Cyclist
            self.block1 = nn.Sequential(
                ConvMD(2, 128, 128, 3, (1, 1), (1, 1)),
                ConvMD(2, 128, 128, 3, (1, 1), (1, 1)),
                ConvMD(2, 128, 128, 3, (1, 1), (1, 1)),
                ConvMD(2, 128, 128, 3, (1, 1), (1, 1)),
                ConvMD(2, 128, 128, 3, (1, 1), (1, 1)),
            )

        self.deconv1 = Deconv2D(128, 256, 3, (1, 1), (1, 1))

        self.block2 = nn.Sequential(
            ConvMD(2, 128, 128, 3, (2, 2), (1, 1)),
            ConvMD(2, 128, 128, 3, (1, 1), (1, 1)),
            ConvMD(2, 128, 128, 3, (1, 1), (1, 1)),
            ConvMD(2, 128, 128, 3, (1, 1), (1, 1)),
            ConvMD(2, 128, 128, 3, (1, 1), (1, 1)),
            ConvMD(2, 128, 128, 3, (1, 1), (1, 1)),
        )

        self.deconv2 = Deconv2D(128, 256, 2, (2, 2), (0, 0))

        self.block3 = nn.Sequential(
            ConvMD(2, 128, 256, 3, (2, 2), (1, 1)),
            ConvMD(2, 256, 256, 3, (1, 1), (1, 1)),
            ConvMD(2, 256, 256, 3, (1, 1), (1, 1)),
            ConvMD(2, 256, 256, 3, (1, 1), (1, 1)),
            ConvMD(2, 256, 256, 3, (1, 1), (1, 1)),
            ConvMD(2, 256, 256, 3, (1, 1), (1, 1)),
        )

        self.deconv3 = Deconv2D(256, 256, 4, (4, 4), (0, 0))

        self.prob_conv = ConvMD(2, 768, 2, 1, (1, 1), (0, 0), bn=False, activation=False)

        self.reg_conv = ConvMD(2, 768, 14, 1, (1, 1), (0, 0), bn=False, activation=False)

        self.output_shape = [feature_height, feature_width]

    def forward(self, inputs):
        batch_size, DEPTH, HEIGHT, WIDTH, C = inputs.shape

        inputs = inputs.permute(0, 4, 1, 2, 3)  # (B, D, H, W, C) -> (B, C, D, H, W)

        temp_conv = self.middle_layer(inputs)
        temp_conv = temp_conv.view(batch_size, -1, HEIGHT, WIDTH)

        temp_conv = self.block1(temp_conv)
        temp_deconv1 = self.deconv1(temp_conv)

        temp_conv = self.block2(temp_conv)
        temp_deconv2 = self.deconv2(temp_conv)

        temp_conv = self.block3(temp_conv)
        temp_deconv3 = self.deconv3(temp_conv)

        temp_conv = torch.cat([temp_deconv3, temp_deconv2, temp_deconv1], dim=1)

        # Probability score map
        p_map = self.prob_conv(temp_conv)

        # Regression map
        r_map = self.reg_conv(temp_conv)

        return torch.sigmoid(p_map), r_map


# --------------------------------------------------------------------------- #
# Composition wrapper (mirrors model/model.py::RPN3D.__init__ submodule
# wiring: self.feature = FeatureNet(); self.rpn = MiddleAndRPN(); the
# forward pass below is the real `features = self.feature(...); prob_output,
# delta_output = self.rpn(features)` sequence -- everything after that in the
# real RPN3D.forward is the KITTI-label loss computation, not the network).
# --------------------------------------------------------------------------- #


class VoxelNet(nn.Module):
    def __init__(
        self,
        voxel_point_count=8,
        input_depth=4,
        input_height=32,
        input_width=32,
        detect_obj="Car",
    ):
        super(VoxelNet, self).__init__()

        self.voxel_point_count = voxel_point_count
        self.input_depth = input_depth
        self.input_height = input_height
        self.input_width = input_width

        self.feature = FeatureNet(voxel_point_count, input_depth, input_height, input_width)
        self.rpn = MiddleAndRPN(input_height // 2, input_width // 2, detect_obj=detect_obj)

    def forward(self, vox_feature, vox_number, vox_coordinate):
        features = self.feature(vox_feature, vox_number, vox_coordinate)
        prob_output, delta_output = self.rpn(features)
        return prob_output, delta_output


_VOXEL_POINT_COUNT = 8
_INPUT_DEPTH = 9
_INPUT_HEIGHT = 32
_INPUT_WIDTH = 32


def build_voxelnet():
    return VoxelNet(
        voxel_point_count=_VOXEL_POINT_COUNT,
        input_depth=_INPUT_DEPTH,
        input_height=_INPUT_HEIGHT,
        input_width=_INPUT_WIDTH,
        detect_obj="Car",
    )


def example_input_voxelnet():
    # Real pipeline: a KITTI point-cloud preprocessor (utils/preprocess.py)
    # groups raw LiDAR points into non-empty voxels and returns, per batch
    # element, a ragged list of (K_b, voxel_point_count, 7) point-feature
    # tensors [x, y, z, reflectance, x-mean, y-mean, z-mean offsets],
    # (K_b, 4) voxel-grid coordinates [batch_idx, d, h, w], and per-voxel
    # point counts. Synthesized here as valid random tensors of the exact
    # shapes/dtypes FeatureNet.forward expects (K_b non-empty voxels per
    # batch element, coordinates within the configured grid extent) --
    # exactly analogous to synthesizing input_ids for a language model
    # rather than running a real tokenizer / point-cloud voxelizer.
    batch_size = 1
    k_per_batch = 6

    vox_feature = []
    vox_number = []
    vox_coordinate = []
    for b in range(batch_size):
        vox_feature.append(torch.randn(k_per_batch, _VOXEL_POINT_COUNT, 7))
        vox_number.append(torch.randint(1, _VOXEL_POINT_COUNT + 1, (k_per_batch,)))

        batch_idx = torch.full((k_per_batch, 1), b, dtype=torch.long)
        d_idx = torch.randint(0, _INPUT_DEPTH, (k_per_batch, 1))
        h_idx = torch.randint(0, _INPUT_HEIGHT, (k_per_batch, 1))
        w_idx = torch.randint(0, _INPUT_WIDTH, (k_per_batch, 1))
        coords = torch.cat([batch_idx, d_idx, h_idx, w_idx], dim=1)
        # De-duplicate coordinates within a batch element (sparse scatter
        # requires unique indices), matching the real voxelizer's guarantee
        # that each grid cell is grouped into at most one voxel.
        coords = torch.unique(coords, dim=0)
        vox_coordinate.append(coords)
        vox_feature[-1] = vox_feature[-1][: coords.shape[0]]
        vox_number[-1] = vox_number[-1][: coords.shape[0]]

    return (vox_feature, vox_number, vox_coordinate)


MENAGERIE_ENTRIES = [
    ("VoxelNet", "build_voxelnet", "example_input_voxelnet", 2018, "vendored-pytorch"),
]
