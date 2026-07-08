# SOURCE: vendored from https://github.com/PeizhuoLi/ganimator @ main
#
# GANimator (SIGGRAPH 2022): "GANimator: Neural Motion Synthesis from a Single
# Sequence" -- SinGAN-style multi-stage skeleton-aware temporal-pyramid GAN
# trained on a single motion clip. Vendored verbatim from models/skeleton.py
# (SkeletonConv -- the paper's core skeleton-aware-convolution contribution,
# plus find_neighbor_joint) and models/gan1d.py (Conv1dModel, the generator/
# discriminator backbone built from SkeletonConv). Only unused GAN-training
# helpers (losses, image pools, GAN_model orchestration) are dropped since
# they are optimization scaffolding, not architecture; the layers are
# copied unmodified. The real code derives `neighbour_list` from a BVH
# skeleton via `find_neighbor_joint(parents, threshold)`; we synthesize a
# small parent-chain skeleton and call the SAME real function, exactly as
# models/architecture.py / train.py do for a loaded BVH skeleton.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


"""
===============================================================================
skeleton-aware layers (models/skeleton.py, vendored)
===============================================================================
"""


class SkeletonConv(nn.Module):
    def __init__(
        self,
        neighbour_list,
        in_channels,
        out_channels,
        kernel_size,
        joint_num,
        stride=1,
        padding=0,
        bias=True,
        padding_mode="zeros",
        add_offset=False,
        in_offset_channel=0,
    ):
        super(SkeletonConv, self).__init__()

        if in_channels % joint_num != 0 or out_channels % joint_num != 0:
            raise Exception("in/out channels should be divided by joint_num")
        self.in_channels_per_joint = in_channels // joint_num
        self.out_channels_per_joint = out_channels // joint_num

        if padding_mode == "zeros":
            padding_mode = "constant"

        self.expanded_neighbour_list = []
        self.expanded_neighbour_list_offset = []
        self.neighbour_list = neighbour_list
        self.add_offset = add_offset
        self.joint_num = joint_num

        self.stride = stride
        self.dilation = 1
        self.groups = 1
        self.padding = padding
        self.padding_mode = padding_mode
        self._padding_repeated_twice = (padding, padding)

        for neighbour in neighbour_list:
            expanded = []
            for k in neighbour:
                for i in range(self.in_channels_per_joint):
                    expanded.append(k * self.in_channels_per_joint + i)
            self.expanded_neighbour_list.append(expanded)

        if self.add_offset:
            self.offset_enc = SkeletonLinear(
                neighbour_list, in_offset_channel * len(neighbour_list), out_channels
            )

            for neighbour in neighbour_list:
                expanded = []
                for k in neighbour:
                    for i in range(add_offset):
                        expanded.append(k * in_offset_channel + i)
                self.expanded_neighbour_list_offset.append(expanded)

        self.weight = torch.zeros(out_channels, in_channels, kernel_size)
        if bias:
            self.bias = torch.zeros(out_channels)
        else:
            self.register_parameter("bias", None)

        self.mask = torch.zeros_like(self.weight)
        for i, neighbour in enumerate(self.expanded_neighbour_list):
            self.mask[
                self.out_channels_per_joint * i : self.out_channels_per_joint * (i + 1),
                neighbour,
                ...,
            ] = 1
        self.mask = nn.Parameter(self.mask, requires_grad=False)

        self.description = (
            "SkeletonConv(in_channels_per_armature={}, out_channels_per_armature={}, kernel_size={}, "
            "joint_num={}, stride={}, padding={}, bias={})".format(
                in_channels // joint_num,
                out_channels // joint_num,
                kernel_size,
                joint_num,
                stride,
                padding,
                bias,
            )
        )

        self.reset_parameters()

    def reset_parameters(self):
        for i, neighbour in enumerate(self.expanded_neighbour_list):
            """ Use temporary variable to avoid assign to copy of slice, which might lead to un expected result """
            tmp = torch.zeros_like(
                self.weight[
                    self.out_channels_per_joint * i : self.out_channels_per_joint * (i + 1),
                    neighbour,
                    ...,
                ]
            )
            nn.init.kaiming_uniform_(tmp, a=math.sqrt(5))
            self.weight[
                self.out_channels_per_joint * i : self.out_channels_per_joint * (i + 1),
                neighbour,
                ...,
            ] = tmp
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(
                    self.weight[
                        self.out_channels_per_joint * i : self.out_channels_per_joint * (i + 1),
                        neighbour,
                        ...,
                    ]
                )
                bound = 1 / math.sqrt(fan_in)
                tmp = torch.zeros_like(
                    self.bias[
                        self.out_channels_per_joint * i : self.out_channels_per_joint * (i + 1)
                    ]
                )
                nn.init.uniform_(tmp, -bound, bound)
                self.bias[
                    self.out_channels_per_joint * i : self.out_channels_per_joint * (i + 1)
                ] = tmp

        self.weight = nn.Parameter(self.weight)
        if self.bias is not None:
            self.bias = nn.Parameter(self.bias)

    def set_offset(self, offset):
        if not self.add_offset:
            raise Exception("Wrong Combination of Parameters")
        self.offset = offset.reshape(offset.shape[0], -1)

    def forward(self, input):
        weight_masked = self.weight * self.mask
        res = F.conv1d(
            F.pad(input, self._padding_repeated_twice, mode=self.padding_mode),
            weight_masked,
            self.bias,
            self.stride,
            0,
            self.dilation,
            self.groups,
        )

        if self.add_offset:
            offset_res = self.offset_enc(self.offset)
            offset_res = offset_res.reshape(offset_res.shape + (1,))
            res += offset_res / 100
        return res

    def __repr__(self):
        return self.description


class SkeletonLinear(nn.Module):
    def __init__(self, neighbour_list, in_channels, out_channels, extra_dim1=False):
        super(SkeletonLinear, self).__init__()
        self.neighbour_list = neighbour_list
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_channels_per_joint = in_channels // len(neighbour_list)
        self.out_channels_per_joint = out_channels // len(neighbour_list)
        self.extra_dim1 = extra_dim1
        self.expanded_neighbour_list = []

        for neighbour in neighbour_list:
            expanded = []
            for k in neighbour:
                for i in range(self.in_channels_per_joint):
                    expanded.append(k * self.in_channels_per_joint + i)
            self.expanded_neighbour_list.append(expanded)

        self.weight = torch.zeros(out_channels, in_channels)
        self.mask = torch.zeros(out_channels, in_channels)
        self.bias = nn.Parameter(torch.Tensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        for i, neighbour in enumerate(self.expanded_neighbour_list):
            tmp = torch.zeros_like(
                self.weight[
                    i * self.out_channels_per_joint : (i + 1) * self.out_channels_per_joint,
                    neighbour,
                ]
            )
            self.mask[
                i * self.out_channels_per_joint : (i + 1) * self.out_channels_per_joint, neighbour
            ] = 1
            nn.init.kaiming_uniform_(tmp, a=math.sqrt(5))
            self.weight[
                i * self.out_channels_per_joint : (i + 1) * self.out_channels_per_joint, neighbour
            ] = tmp

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

        self.weight = nn.Parameter(self.weight)
        self.mask = nn.Parameter(self.mask, requires_grad=False)

    def forward(self, input):
        input = input.reshape(input.shape[0], -1)
        weight_masked = self.weight * self.mask
        res = F.linear(input, weight_masked, self.bias)
        if self.extra_dim1:
            res = res.reshape(res.shape + (1,))
        return res


def find_neighbor_joint(parents, threshold):
    n_joint = len(parents)
    dist_mat = np.empty((n_joint, n_joint), dtype=np.int64)
    dist_mat[:, :] = 100000
    for i, p in enumerate(parents):
        dist_mat[i, i] = 0
        if i != 0:
            dist_mat[i, p] = dist_mat[p, i] = 1

    """
    Floyd's algorithm
    """
    for k in range(n_joint):
        for i in range(n_joint):
            for j in range(n_joint):
                dist_mat[i, j] = min(dist_mat[i, j], dist_mat[i, k] + dist_mat[k, j])

    neighbor_list = []
    for i in range(n_joint):
        neighbor = []
        for j in range(n_joint):
            if dist_mat[i, j] <= threshold:
                neighbor.append(j)
        neighbor_list.append(neighbor)

    return neighbor_list


"""
===============================================================================
1D conv generator/discriminator backbone (models/gan1d.py, vendored)
===============================================================================
"""


class Conv1dModel(nn.Module):
    def __init__(
        self,
        channels,
        kernel_size,
        padding=-1,
        last_active="None",
        padding_mode="zeros",
        batch_norm=False,
        skeleton_aware=False,
        neighbour_list=None,
        activation="LeakyReLU",
    ):
        super(Conv1dModel, self).__init__()
        import functools

        conv1d = (
            functools.partial(
                SkeletonConv, neighbour_list=neighbour_list, joint_num=len(neighbour_list)
            )
            if skeleton_aware
            else nn.Conv1d
        )
        if padding == -1:
            if kernel_size % 2 == 0:
                raise Exception("Only support odd kernel size for now")
            padding = (kernel_size - 1) // 2
        self.layers = nn.ModuleList()
        for i in range(len(channels) - 1):
            in_c = channels[i]
            out_c = channels[i + 1]
            if i == len(channels) - 2 and out_c == 1:
                conv1d = nn.Conv1d
            bias = not batch_norm
            if i == len(channels) - 2:
                bias = True
            seq = [
                conv1d(
                    in_channels=in_c,
                    out_channels=out_c,
                    kernel_size=kernel_size,
                    padding=padding,
                    padding_mode=padding_mode,
                    bias=bias,
                )
            ]
            if i < len(channels) - 2:
                if batch_norm:
                    seq.append(nn.BatchNorm1d(num_features=out_c))

                if activation == "LeakyReLU":
                    seq.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
                elif activation == "ReLU":
                    seq.append(nn.ReLU(inplace=True))
                else:
                    raise Exception("Unknown activation")
            else:
                if last_active is not None:
                    seq.append(last_active)
            self.layers.append(nn.Sequential(*seq))
        self.output = None

    def forward(self, input, prev_img=None, cond=None, cond_requires_mask=False):
        for layer in self.layers:
            input = layer(input)

        self.output = input
        return input


# --- staging harness: build + example input ---------------------------------


def build_ganimator_conv1d_generator():
    # synthetic 5-joint parent-chain skeleton (root=0), same shape of input
    # `find_neighbor_joint` real repo code expects when derived from a BVH
    # skeleton (models/architecture.py / bvh_parser.py get_neighbor()).
    parents = [-1, 0, 1, 2, 3]
    neighbour_list = find_neighbor_joint(parents, threshold=1)
    joint_num = len(neighbour_list)  # 5

    n_channels = joint_num * 3  # 3 channels per joint (matches repo's per-joint channel convention)
    channels_list = [n_channels, n_channels, n_channels]

    gen = Conv1dModel(
        channels_list,
        kernel_size=3,
        last_active=None,
        padding_mode="zeros",
        batch_norm=False,
        neighbour_list=neighbour_list,
        skeleton_aware=True,
    )
    return gen


def example_input_ganimator_conv1d_generator():
    joint_num = 5
    n_channels = joint_num * 3
    batch, length = 1, 32
    return torch.randn(batch, n_channels, length)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    (
        "GANimator-SkeletonConv1D",
        build_ganimator_conv1d_generator,
        example_input_ganimator_conv1d_generator,
        2022,
        MENAGERIE_ZOO,
    ),
]
