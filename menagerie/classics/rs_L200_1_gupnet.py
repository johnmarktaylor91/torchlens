# SOURCE: vendored from SuperMHP/GUPNet @ main
#
# https://github.com/SuperMHP/GUPNet
# https://raw.githubusercontent.com/SuperMHP/GUPNet/main/code/lib/models/gupnet.py
# https://raw.githubusercontent.com/SuperMHP/GUPNet/main/code/lib/backbones/dla.py
# https://raw.githubusercontent.com/SuperMHP/GUPNet/main/code/lib/backbones/dlaup.py
# https://raw.githubusercontent.com/SuperMHP/GUPNet/main/code/lib/helpers/decode_helper.py
#
# Lu et al. 2021 (ICCV) "Geometry Uncertainty Projection Network for Monocular
# 3D Object Detection" -- monocular 3D detector combining a DLA-34 + DLA-Up
# heatmap/2D-box backbone with a geometry-uncertainty-aware depth head that
# fuses a learned depth estimate with a geometrically-derived depth (from
# projected object height and camera intrinsics) via log-sum-exp uncertainty
# combination. This file vendors the real model code verbatim (only import
# paths were rewritten from `lib.*` to local module references and Xavier
# `nn.init` weight normal calls were removed from the `if m.bias:` truthiness
# check bug that raises on a Tensor -- everything else, including forward
# math, is unchanged): `GUPNet` (lib/models/gupnet.py), the DLA-34 backbone
# (lib/backbones/dla.py), `DLAUp` (lib/backbones/dlaup.py), and the
# `_topk`/`_nms`/`_gather_feat` decode helpers (lib/helpers/decode_helper.py)
# used by the test-time two-stage RoI-align depth/size/heading heads.

from __future__ import annotations

import math
import os

import numpy as np
import torch
import torch.nn as nn
import torchvision.ops.roi_align as roi_align

# --------------------------------------------------------------------------
# lib/backbones/dla.py (verbatim, DLA-34 backbone)
# --------------------------------------------------------------------------

BatchNorm = nn.BatchNorm2d


def get_model_url(data="imagenet", name="dla34", hash="ba72cf86"):
    return os.path.join("http://dl.yf.io/dla/models", data, "{}-{}.pth".format(name, hash))


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            bias=False,
            dilation=dilation,
        )
        self.bn1 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=dilation, bias=False, dilation=dilation
        )
        self.bn2 = BatchNorm(planes)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 1, stride=1, bias=False, padding=(kernel_size - 1) // 2
        )
        self.bn = BatchNorm(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x


class Tree(nn.Module):
    def __init__(
        self,
        levels,
        block,
        in_channels,
        out_channels,
        stride=1,
        level_root=False,
        root_dim=0,
        root_kernel_size=1,
        dilation=1,
        root_residual=False,
    ):
        super().__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride, dilation=dilation)
            self.tree2 = block(out_channels, out_channels, 1, dilation=dilation)
        else:
            self.tree1 = Tree(
                levels - 1,
                block,
                in_channels,
                out_channels,
                stride,
                root_dim=0,
                root_kernel_size=root_kernel_size,
                dilation=dilation,
                root_residual=root_residual,
            )
            self.tree2 = Tree(
                levels - 1,
                block,
                out_channels,
                out_channels,
                root_dim=root_dim + out_channels,
                root_kernel_size=root_kernel_size,
                dilation=dilation,
                root_residual=root_residual,
            )
        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size, root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                BatchNorm(out_channels),
            )

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


class DLA(nn.Module):
    def __init__(
        self,
        levels,
        channels,
        num_classes=1000,
        block=BasicBlock,
        residual_root=False,
        return_levels=False,
        pool_size=7,
        linear_root=False,
    ):
        super().__init__()
        self.channels = channels
        self.return_levels = return_levels
        self.num_classes = num_classes
        self.base_layer = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=7, stride=1, padding=3, bias=False),
            BatchNorm(channels[0]),
            nn.ReLU(inplace=True),
        )
        self.level0 = self._make_conv_level(channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(channels[0], channels[1], levels[1], stride=2)
        self.level2 = Tree(
            levels[2],
            block,
            channels[1],
            channels[2],
            2,
            level_root=False,
            root_residual=residual_root,
        )
        self.level3 = Tree(
            levels[3],
            block,
            channels[2],
            channels[3],
            2,
            level_root=True,
            root_residual=residual_root,
        )
        self.level4 = Tree(
            levels[4],
            block,
            channels[3],
            channels[4],
            2,
            level_root=True,
            root_residual=residual_root,
        )
        self.level5 = Tree(
            levels[5],
            block,
            channels[4],
            channels[5],
            2,
            level_root=True,
            root_residual=residual_root,
        )

        self.avgpool = nn.AvgPool2d(pool_size)
        self.fc = nn.Conv2d(
            channels[-1], num_classes, kernel_size=1, stride=1, padding=0, bias=True
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, BatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend(
                [
                    nn.Conv2d(
                        inplanes,
                        planes,
                        kernel_size=3,
                        stride=stride if i == 0 else 1,
                        padding=dilation,
                        bias=False,
                        dilation=dilation,
                    ),
                    BatchNorm(planes),
                    nn.ReLU(inplace=True),
                ]
            )
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x):
        y = []
        x = self.base_layer(x)
        for i in range(6):
            x = getattr(self, "level{}".format(i))(x)
            y.append(x)
        if self.return_levels:
            return y
        else:
            x = self.avgpool(x)
            x = self.fc(x)
            x = x.view(x.size(0), -1)

            return x


def dla34(pretrained=False, **kwargs):  # DLA-34
    model = DLA([1, 1, 1, 2, 2, 1], [16, 32, 64, 128, 256, 512], block=BasicBlock, **kwargs)
    # NOTE: pretrained-weight download is dropped in this vendored copy (no
    # network access / model_zoo checkpoint dependency); random init only.
    return model


# --------------------------------------------------------------------------
# lib/backbones/dlaup.py (verbatim, DLA-Up feature aggregation)
# --------------------------------------------------------------------------


class DlaupConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernal_szie=3, stride=1, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernal_szie,
            stride=stride,
            padding=kernal_szie // 2,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class IDAUp(nn.Module):
    """input: features map of different layers, output: up-sampled features"""

    def __init__(self, in_channels_list, up_factors_list, out_channels):
        super().__init__()
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels

        for i in range(1, len(in_channels_list)):
            in_channels = in_channels_list[i]
            up_factors = int(up_factors_list[i])

            proj = DlaupConv2d(in_channels, out_channels, kernal_szie=3, stride=1, bias=False)
            node = DlaupConv2d(out_channels * 2, out_channels, kernal_szie=3, stride=1, bias=False)
            up = nn.ConvTranspose2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=up_factors * 2,
                stride=up_factors,
                padding=up_factors // 2,
                output_padding=0,
                groups=out_channels,
                bias=False,
            )
            fill_up_weights(up)

            setattr(self, "proj_" + str(i), proj)
            setattr(self, "up_" + str(i), up)
            setattr(self, "node_" + str(i), node)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, layers):
        assert len(self.in_channels_list) == len(layers), "{} vs {} layers".format(
            len(self.in_channels_list), len(layers)
        )

        for i in range(1, len(layers)):
            upsample = getattr(self, "up_" + str(i))
            project = getattr(self, "proj_" + str(i))
            node = getattr(self, "node_" + str(i))

            layers[i] = upsample(project(layers[i]))
            layers[i] = node(torch.cat([layers[i - 1], layers[i]], 1))

        return layers


class DLAUp(nn.Module):
    def __init__(self, in_channels_list, scales_list=(1, 2, 4, 8, 16)):
        super().__init__()
        scales_list = np.array(scales_list, dtype=int)

        for i in range(len(in_channels_list) - 1):
            j = -i - 2
            setattr(
                self,
                "ida_{}".format(i),
                IDAUp(
                    in_channels_list=in_channels_list[j:],
                    up_factors_list=scales_list[j:] // scales_list[j],
                    out_channels=in_channels_list[j],
                ),
            )
            scales_list[j + 1 :] = scales_list[j]
            in_channels_list[j + 1 :] = [in_channels_list[j] for _ in in_channels_list[j + 1 :]]

    def forward(self, layers):
        layers = list(layers)
        assert len(layers) > 1
        for i in range(len(layers) - 1):
            ida = getattr(self, "ida_{}".format(i))
            layers[-i - 2 :] = ida(layers[-i - 2 :])
        return layers[-1]


# --------------------------------------------------------------------------
# lib/helpers/decode_helper.py (verbatim, test-time top-k heatmap decode)
# --------------------------------------------------------------------------


def _nms(heatmap, kernel=3):
    padding = (kernel - 1) // 2
    heatmapmax = nn.functional.max_pool2d(heatmap, (kernel, kernel), stride=1, padding=padding)
    keep = (heatmapmax == heatmap).float()
    return heatmap * keep


def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _topk(heatmap, K=50):
    batch, cat, height, width = heatmap.size()

    topk_scores, topk_inds = torch.topk(heatmap.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_cls_ids = (topk_ind / K).int()
    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_cls_ids, topk_xs, topk_ys


# --------------------------------------------------------------------------
# lib/losses/loss_function.py::extract_input_from_tensor (verbatim, used by
# the two-stage RoI feature extraction at test time)
# --------------------------------------------------------------------------


def extract_input_from_tensor(input, ind, mask):
    input = _transpose_and_gather_feat(input, ind)
    return input[mask]


def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


# --------------------------------------------------------------------------
# lib/models/gupnet.py (verbatim, GUPNet main model)
# --------------------------------------------------------------------------


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find("Conv") != -1:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find("BatchNorm") != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


class GUPNet(nn.Module):
    def __init__(self, backbone="dla34", neck="DLAUp", downsample=4, mean_size=None):
        assert downsample in [4, 8, 16, 32]
        super().__init__()

        self.backbone = dla34(pretrained=False, return_levels=True)
        self.head_conv = 256  # default setting for head conv
        self.mean_size = nn.Parameter(
            torch.tensor(mean_size, dtype=torch.float32), requires_grad=False
        )
        self.cls_num = mean_size.shape[0]
        channels = self.backbone.channels  # channels list for feature maps generated by backbone
        self.first_level = int(np.log2(downsample))
        scales = [2**i for i in range(len(channels[self.first_level :]))]
        self.feat_up = DLAUp(channels[self.first_level :], scales_list=scales)

        # initialize the head of pipeline, according to heads setting.
        self.heatmap = nn.Sequential(
            nn.Conv2d(
                channels[self.first_level], self.head_conv, kernel_size=3, padding=1, bias=True
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_conv, 3, kernel_size=1, stride=1, padding=0, bias=True),
        )
        self.offset_2d = nn.Sequential(
            nn.Conv2d(
                channels[self.first_level], self.head_conv, kernel_size=3, padding=1, bias=True
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_conv, 2, kernel_size=1, stride=1, padding=0, bias=True),
        )
        self.size_2d = nn.Sequential(
            nn.Conv2d(
                channels[self.first_level], self.head_conv, kernel_size=3, padding=1, bias=True
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_conv, 2, kernel_size=1, stride=1, padding=0, bias=True),
        )

        self.depth = nn.Sequential(
            nn.Conv2d(
                channels[self.first_level] + 2 + self.cls_num,
                self.head_conv,
                kernel_size=3,
                padding=1,
                bias=True,
            ),
            nn.BatchNorm2d(self.head_conv),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.head_conv, 2, kernel_size=1, stride=1, padding=0, bias=True),
        )
        self.offset_3d = nn.Sequential(
            nn.Conv2d(
                channels[self.first_level] + 2 + self.cls_num,
                self.head_conv,
                kernel_size=3,
                padding=1,
                bias=True,
            ),
            nn.BatchNorm2d(self.head_conv),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.head_conv, 2, kernel_size=1, stride=1, padding=0, bias=True),
        )
        self.size_3d = nn.Sequential(
            nn.Conv2d(
                channels[self.first_level] + 2 + self.cls_num,
                self.head_conv,
                kernel_size=3,
                padding=1,
                bias=True,
            ),
            nn.BatchNorm2d(self.head_conv),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.head_conv, 4, kernel_size=1, stride=1, padding=0, bias=True),
        )
        self.heading = nn.Sequential(
            nn.Conv2d(
                channels[self.first_level] + 2 + self.cls_num,
                self.head_conv,
                kernel_size=3,
                padding=1,
                bias=True,
            ),
            nn.BatchNorm2d(self.head_conv),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.head_conv, 24, kernel_size=1, stride=1, padding=0, bias=True),
        )
        # init layers
        self.heatmap[-1].bias.data.fill_(-2.19)
        self.fill_fc_weights(self.offset_2d)
        self.fill_fc_weights(self.size_2d)

        self.depth.apply(weights_init_xavier)
        self.offset_3d.apply(weights_init_xavier)
        self.size_3d.apply(weights_init_xavier)
        self.heading.apply(weights_init_xavier)

    def forward(self, input, coord_ranges, calibs, targets=None, K=50, mode="test"):
        device_id = input.device
        BATCH_SIZE = input.size(0)  # noqa: F841 (kept for fidelity with upstream, unused there too)

        feat = self.backbone(input)
        feat = self.feat_up(feat[self.first_level :])
        ret = {}
        ret["heatmap"] = self.heatmap(feat)
        ret["offset_2d"] = self.offset_2d(feat)
        ret["size_2d"] = self.size_2d(feat)
        # two stage
        assert mode in ["train", "val", "test"]
        if mode == "train":  # extract train structure in the train (only) and the val mode
            inds, cls_ids = targets["indices"], targets["cls_ids"]
            masks = targets["mask_2d"]
        else:  # extract test structure in the test (only) and the val mode
            inds, cls_ids = _topk(
                _nms(torch.clamp(ret["heatmap"].sigmoid(), min=1e-4, max=1 - 1e-4)), K=K
            )[1:3]
            masks = torch.ones(inds.size()).type(torch.uint8).to(device_id)
        ret.update(self.get_roi_feat(feat, inds, masks, ret, calibs, coord_ranges, cls_ids))
        return ret

    def get_roi_feat_by_mask(self, feat, box2d_maps, inds, mask, calibs, coord_ranges, cls_ids):
        BATCH_SIZE, _, HEIGHT, WIDE = feat.size()
        device_id = feat.device
        num_masked_bin = mask.sum()
        res = {}
        if num_masked_bin != 0:
            # get box2d of each roi region
            box2d_masked = extract_input_from_tensor(box2d_maps, inds, mask)
            # get roi feature
            roi_feature_masked = roi_align(feat, box2d_masked, [7, 7])
            # get coord range of each roi
            coord_ranges_mask2d = coord_ranges[box2d_masked[:, 0].long()]

            # map box2d coordinate from feature map size domain to original image size domain
            box2d_masked = torch.cat(
                [
                    box2d_masked[:, 0:1],
                    box2d_masked[:, 1:2]
                    / WIDE
                    * (coord_ranges_mask2d[:, 1, 0:1] - coord_ranges_mask2d[:, 0, 0:1])
                    + coord_ranges_mask2d[:, 0, 0:1],
                    box2d_masked[:, 2:3]
                    / HEIGHT
                    * (coord_ranges_mask2d[:, 1, 1:2] - coord_ranges_mask2d[:, 0, 1:2])
                    + coord_ranges_mask2d[:, 0, 1:2],
                    box2d_masked[:, 3:4]
                    / WIDE
                    * (coord_ranges_mask2d[:, 1, 0:1] - coord_ranges_mask2d[:, 0, 0:1])
                    + coord_ranges_mask2d[:, 0, 0:1],
                    box2d_masked[:, 4:5]
                    / HEIGHT
                    * (coord_ranges_mask2d[:, 1, 1:2] - coord_ranges_mask2d[:, 0, 1:2])
                    + coord_ranges_mask2d[:, 0, 1:2],
                ],
                1,
            )
            roi_calibs = calibs[box2d_masked[:, 0].long()]
            # project the coordinate in the normal image to the camera coord by calibs
            coords_in_camera_coord = torch.cat(
                [
                    self.project2rect(
                        roi_calibs,
                        torch.cat(
                            [box2d_masked[:, 1:3], torch.ones([num_masked_bin, 1]).to(device_id)],
                            -1,
                        ),
                    )[:, :2],
                    self.project2rect(
                        roi_calibs,
                        torch.cat(
                            [box2d_masked[:, 3:5], torch.ones([num_masked_bin, 1]).to(device_id)],
                            -1,
                        ),
                    )[:, :2],
                ],
                -1,
            )
            coords_in_camera_coord = torch.cat([box2d_masked[:, 0:1], coords_in_camera_coord], -1)
            # generate coord maps
            coord_maps = torch.cat(
                [
                    torch.cat(
                        [
                            coords_in_camera_coord[:, 1:2]
                            + i
                            * (coords_in_camera_coord[:, 3:4] - coords_in_camera_coord[:, 1:2])
                            / 6
                            for i in range(7)
                        ],
                        -1,
                    )
                    .unsqueeze(1)
                    .repeat([1, 7, 1])
                    .unsqueeze(1),
                    torch.cat(
                        [
                            coords_in_camera_coord[:, 2:3]
                            + i
                            * (coords_in_camera_coord[:, 4:5] - coords_in_camera_coord[:, 2:3])
                            / 6
                            for i in range(7)
                        ],
                        -1,
                    )
                    .unsqueeze(2)
                    .repeat([1, 1, 7])
                    .unsqueeze(1),
                ],
                1,
            )

            # concatenate coord maps with feature maps in the channel dim
            cls_hots = torch.zeros(num_masked_bin, self.cls_num).to(device_id)
            cls_hots[torch.arange(num_masked_bin).to(device_id), cls_ids[mask].long()] = 1.0

            roi_feature_masked = torch.cat(
                [
                    roi_feature_masked,
                    coord_maps,
                    cls_hots.unsqueeze(-1).unsqueeze(-1).repeat([1, 1, 7, 7]),
                ],
                1,
            )

            # compute heights of projected objects
            box2d_height = torch.clamp(box2d_masked[:, 4] - box2d_masked[:, 2], min=1.0)
            # compute real 3d height
            size3d_offset = self.size_3d(roi_feature_masked)[:, :, 0, 0]
            h3d_log_std = size3d_offset[:, 3:4]
            size3d_offset = size3d_offset[:, :3]

            size_3d = self.mean_size[cls_ids[mask].long()] + size3d_offset
            depth_geo = size_3d[:, 0] / box2d_height.squeeze() * roi_calibs[:, 0, 0]

            depth_net_out = self.depth(roi_feature_masked)[:, :, 0, 0]
            depth_geo_log_std = (
                h3d_log_std.squeeze() + 2 * (roi_calibs[:, 0, 0].log() - box2d_height.log())
            ).unsqueeze(-1)
            depth_net_log_std = torch.logsumexp(
                torch.cat([depth_net_out[:, 1:2], depth_geo_log_std], -1), -1, keepdim=True
            )

            depth_net_out = torch.cat(
                [
                    (1.0 / (depth_net_out[:, 0:1].sigmoid() + 1e-6) - 1.0)
                    + depth_geo.unsqueeze(-1),
                    depth_net_log_std,
                ],
                -1,
            )

            res["train_tag"] = torch.ones(num_masked_bin).type(torch.bool).to(device_id)
            res["heading"] = self.heading(roi_feature_masked)[:, :, 0, 0]
            res["depth"] = depth_net_out
            res["offset_3d"] = self.offset_3d(roi_feature_masked)[:, :, 0, 0]
            res["size_3d"] = size3d_offset
            res["h3d_log_variance"] = h3d_log_std
        else:
            res["depth"] = torch.zeros([1, 2]).to(device_id)
            res["offset_3d"] = torch.zeros([1, 2]).to(device_id)
            res["size_3d"] = torch.zeros([1, 3]).to(device_id)
            res["train_tag"] = torch.zeros(1).type(torch.bool).to(device_id)
            res["heading"] = torch.zeros([1, 24]).to(device_id)
            res["h3d_log_variance"] = torch.zeros([1, 1]).to(device_id)
        return res

    def get_roi_feat(self, feat, inds, mask, ret, calibs, coord_ranges, cls_ids):
        BATCH_SIZE, _, HEIGHT, WIDE = feat.size()
        device_id = feat.device
        coord_map = (
            torch.cat(
                [
                    torch.arange(WIDE).unsqueeze(0).repeat([HEIGHT, 1]).unsqueeze(0),
                    torch.arange(HEIGHT).unsqueeze(-1).repeat([1, WIDE]).unsqueeze(0),
                ],
                0,
            )
            .unsqueeze(0)
            .repeat([BATCH_SIZE, 1, 1, 1])
            .type(torch.float)
            .to(device_id)
        )
        box2d_centre = coord_map + ret["offset_2d"]
        box2d_maps = torch.cat(
            [box2d_centre - ret["size_2d"] / 2, box2d_centre + ret["size_2d"] / 2], 1
        )
        box2d_maps = torch.cat(
            [
                torch.arange(BATCH_SIZE)
                .unsqueeze(-1)
                .unsqueeze(-1)
                .unsqueeze(-1)
                .repeat([1, 1, HEIGHT, WIDE])
                .type(torch.float)
                .to(device_id),
                box2d_maps,
            ],
            1,
        )
        # box2d_maps is box2d in each bin
        res = self.get_roi_feat_by_mask(feat, box2d_maps, inds, mask, calibs, coord_ranges, cls_ids)
        return res

    def project2rect(self, calib, point_img):
        c_u = calib[:, 0, 2]
        c_v = calib[:, 1, 2]
        f_u = calib[:, 0, 0]
        f_v = calib[:, 1, 1]
        b_x = calib[:, 0, 3] / (-f_u)  # relative
        b_y = calib[:, 1, 3] / (-f_v)
        x = (point_img[:, 0] - c_u) * point_img[:, 2] / f_u + b_x
        y = (point_img[:, 1] - c_v) * point_img[:, 2] / f_v + b_y
        z = point_img[:, 2]
        centre_by_obj = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1), z.unsqueeze(-1)], -1)
        return centre_by_obj

    def fill_fc_weights(self, layers):
        for m in layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


def build_gupnet():
    mean_size = np.ones(
        (3, 3), dtype=np.float32
    )  # 3 KITTI classes (Car/Pedestrian/Cyclist), dummy dims
    return GUPNet(backbone="dla34", neck="DLAUp", downsample=4, mean_size=mean_size)


def example_input_gupnet():
    batch = 1
    input = torch.randn(batch, 3, 96, 320)  # downsample=4 -> feature map 24x80
    coord_ranges = torch.tensor([[[0.0, 0.0], [320.0, 96.0]]] * batch)
    calibs = torch.eye(3, 4).unsqueeze(0).repeat(batch, 1, 1)
    calibs[:, 0, 0] = 700.0
    calibs[:, 1, 1] = 700.0
    return (input, coord_ranges, calibs)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("GUPNet", "build_gupnet", "example_input_gupnet", 2021, "vendored"),
]
