# SOURCE: vendored from garrickbrazil/M3D-RPN @ master
#
# https://github.com/garrickbrazil/M3D-RPN
# https://raw.githubusercontent.com/garrickbrazil/M3D-RPN/master/models/densenet121_3d_dilate_depth_aware.py
# https://raw.githubusercontent.com/garrickbrazil/M3D-RPN/master/lib/rpn_util.py
#
# Brazil & Liu 2019 (ICCV Oral), "M3D-RPN: Monocular 3D Region Proposal
# Network for Object Detection". This vendors the actual `LocalConv2d` /
# `RPN` classes and `dilate_layer` function from
# `models/densenet121_3d_dilate_depth_aware.py` verbatim (the
# depth-aware/row-wise variant, "3d_dilate_depth_aware" in the original repo
# naming), backed by the REAL `torchvision.models.densenet121` trunk with the
# dilated denseblock4 exactly as in the original `build()` (16 of 16
# denseblock4 layers dilated to rate 2, `transition3.pool` deleted). The
# `flatten_tensor`/`locate_anchors`/`calc_output_size` helpers used by `RPN`
# are copied verbatim from `lib/rpn_util.py` (pure numpy/torch, no
# architectural content) rather than importing that module directly, since
# `lib/rpn_util.py`'s own top-of-file imports (`lib.util` -> `cv2`,
# `lib.core` -> `easydict`/`shapely`/`visdom`, `lib.nms.gpu_nms` -> a
# compiled CUDA NMS extension) drag in non-base, non-installed packages that
# are unrelated to the RPN head's forward computation itself.
#
# Deviations from the upstream script are environment-portability fixes only,
# not architecture changes: the original hardcodes `.type(torch.cuda.FloatTensor)`
# for `self.rois` / the learnable blend-scalar parameters and reads `conf.bins`
# / `conf.anchors` / `conf.feat_stride` / `conf.crop_size` / `conf.test_scale`
# / `conf.bbox_means` / `conf.bbox_stds` off an `EasyDict` training config;
# here those become plain CPU tensors/attrs on an equivalent lightweight
# config object built with small (rather than KITTI-scale) anchor/crop
# dimensions.

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models


# ---- verbatim pure-numpy/torch helpers from lib/rpn_util.py ----


def flatten_tensor(input):
    """
    Flattens and permutes a tensor from size
    [B x C x W x H] --> [B x (W x H) x C]
    """
    bsize = input.shape[0]
    csize = input.shape[1]

    return input.permute(0, 2, 3, 1).contiguous().view(bsize, -1, csize)


def calc_output_size(res, stride):
    """
    Approximate the output size of a network
    """
    return np.ceil(np.array(res) / stride).astype(int)


def locate_anchors(anchors, feat_size, stride, convert_tensor=False):
    """
    Spreads each anchor shape across a feature map of size feat_size spaced by a known stride.
    """
    # compute rois
    shift_x = np.array(range(0, feat_size[1], 1)) * float(stride)
    shift_y = np.array(range(0, feat_size[0], 1)) * float(stride)
    [shift_x, shift_y] = np.meshgrid(shift_x, shift_y)

    rois = np.expand_dims(anchors[:, 0:4], axis=1)
    shift_x = np.expand_dims(shift_x, axis=0)
    shift_y = np.expand_dims(shift_y, axis=0)

    shift_x1 = shift_x + np.expand_dims(rois[:, :, 0], axis=2)
    shift_y1 = shift_y + np.expand_dims(rois[:, :, 1], axis=2)
    shift_x2 = shift_x + np.expand_dims(rois[:, :, 2], axis=2)
    shift_y2 = shift_y + np.expand_dims(rois[:, :, 3], axis=2)

    # compute anchor tracker
    anchor_tracker = np.zeros(shift_x1.shape, dtype=float)
    for aind in range(0, rois.shape[0]):
        anchor_tracker[aind, :, :] = aind

    stack_size = feat_size[0] * anchors.shape[0]

    if convert_tensor:
        shift_x1 = torch.from_numpy(shift_x1).view(1, stack_size, feat_size[1])
        shift_y1 = torch.from_numpy(shift_y1).view(1, stack_size, feat_size[1])
        shift_x2 = torch.from_numpy(shift_x2).view(1, stack_size, feat_size[1])
        shift_y2 = torch.from_numpy(shift_y2).view(1, stack_size, feat_size[1])
        anchor_tracker = torch.from_numpy(anchor_tracker).view(1, stack_size, feat_size[1])

        shift_x1.requires_grad = False
        shift_y1.requires_grad = False
        shift_x2.requires_grad = False
        shift_y2.requires_grad = False
        anchor_tracker.requires_grad = False

        shift_x1 = shift_x1.permute(1, 2, 0).contiguous().view(-1, 1)
        shift_y1 = shift_y1.permute(1, 2, 0).contiguous().view(-1, 1)
        shift_x2 = shift_x2.permute(1, 2, 0).contiguous().view(-1, 1)
        shift_y2 = shift_y2.permute(1, 2, 0).contiguous().view(-1, 1)
        anchor_tracker = anchor_tracker.permute(1, 2, 0).contiguous().view(-1, 1)

        rois = torch.cat((shift_x1, shift_y1, shift_x2, shift_y2, anchor_tracker), 1)
    else:
        shift_x1 = shift_x1.reshape(1, stack_size, feat_size[1]).reshape(-1, 1)
        shift_y1 = shift_y1.reshape(1, stack_size, feat_size[1]).reshape(-1, 1)
        shift_x2 = shift_x2.reshape(1, stack_size, feat_size[1]).reshape(-1, 1)
        shift_y2 = shift_y2.reshape(1, stack_size, feat_size[1]).reshape(-1, 1)
        anchor_tracker = anchor_tracker.reshape(1, stack_size, feat_size[1]).reshape(-1, 1)

        rois = np.concatenate((shift_x1, shift_y1, shift_x2, shift_y2, anchor_tracker), 1)

    return rois


# ---- verbatim (module-level) from models/densenet121_3d_dilate_depth_aware.py ----


def dilate_layer(layer, val):
    layer.dilation = val
    layer.padding = val


class LocalConv2d(nn.Module):
    def __init__(self, num_rows, num_feats_in, num_feats_out, kernel=1, padding=0):
        super(LocalConv2d, self).__init__()

        self.num_rows = num_rows
        self.out_channels = num_feats_out
        self.kernel = kernel
        self.pad = padding

        self.group_conv = nn.Conv2d(
            num_feats_in * num_rows, num_feats_out * num_rows, kernel, stride=1, groups=num_rows
        )

    def forward(self, x):
        b, c, h, w = x.size()

        if self.pad:
            x = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode="constant", value=0)

        t = int(h / self.num_rows)

        # unfold by rows
        x = x.unfold(2, t + self.pad * 2, t)
        x = x.permute([0, 2, 1, 4, 3]).contiguous()
        x = x.view(b, c * self.num_rows, t + self.pad * 2, (w + self.pad * 2)).contiguous()

        # group convolution for efficient parallel processing
        y = self.group_conv(x)
        y = y.view(b, self.num_rows, self.out_channels, t, w).contiguous()
        y = y.permute([0, 2, 1, 3, 4]).contiguous()
        y = y.view(b, self.out_channels, h, w)

        return y


class RPN(nn.Module):
    def __init__(self, phase, base, conf):
        super(RPN, self).__init__()

        self.base = base

        del self.base.transition3.pool

        # dilate
        dilate_layer(self.base.denseblock4.denselayer1.conv2, 2)
        dilate_layer(self.base.denseblock4.denselayer2.conv2, 2)
        dilate_layer(self.base.denseblock4.denselayer3.conv2, 2)
        dilate_layer(self.base.denseblock4.denselayer4.conv2, 2)
        dilate_layer(self.base.denseblock4.denselayer5.conv2, 2)
        dilate_layer(self.base.denseblock4.denselayer6.conv2, 2)
        dilate_layer(self.base.denseblock4.denselayer7.conv2, 2)
        dilate_layer(self.base.denseblock4.denselayer8.conv2, 2)
        dilate_layer(self.base.denseblock4.denselayer9.conv2, 2)
        dilate_layer(self.base.denseblock4.denselayer10.conv2, 2)
        dilate_layer(self.base.denseblock4.denselayer11.conv2, 2)
        dilate_layer(self.base.denseblock4.denselayer12.conv2, 2)
        dilate_layer(self.base.denseblock4.denselayer13.conv2, 2)
        dilate_layer(self.base.denseblock4.denselayer14.conv2, 2)
        dilate_layer(self.base.denseblock4.denselayer15.conv2, 2)
        dilate_layer(self.base.denseblock4.denselayer16.conv2, 2)

        # settings
        self.phase = phase
        self.num_classes = len(conf["lbls"]) + 1
        self.num_anchors = conf["anchors"].shape[0]

        self.num_rows = int(
            min(conf["bins"], calc_output_size(conf["test_scale"], conf["feat_stride"]))
        )

        self.prop_feats = nn.Sequential(
            nn.Conv2d(self.base[-1].num_features, 512, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        # outputs
        self.cls = nn.Conv2d(
            self.prop_feats[0].out_channels, self.num_classes * self.num_anchors, 1
        )

        # bbox 2d
        self.bbox_x = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_y = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_w = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_h = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)

        # bbox 3d
        self.bbox_x3d = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_y3d = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_z3d = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_w3d = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_h3d = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_l3d = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_rY3d = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)

        self.prop_feats_loc = nn.Sequential(
            LocalConv2d(self.num_rows, self.base[-1].num_features, 512, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # outputs
        self.cls_loc = LocalConv2d(
            self.num_rows, self.prop_feats[0].out_channels, self.num_classes * self.num_anchors, 1
        )

        # bbox 2d
        self.bbox_x_loc = LocalConv2d(
            self.num_rows, self.prop_feats[0].out_channels, self.num_anchors, 1
        )
        self.bbox_y_loc = LocalConv2d(
            self.num_rows, self.prop_feats[0].out_channels, self.num_anchors, 1
        )
        self.bbox_w_loc = LocalConv2d(
            self.num_rows, self.prop_feats[0].out_channels, self.num_anchors, 1
        )
        self.bbox_h_loc = LocalConv2d(
            self.num_rows, self.prop_feats[0].out_channels, self.num_anchors, 1
        )

        # bbox 3d
        self.bbox_x3d_loc = LocalConv2d(
            self.num_rows, self.prop_feats[0].out_channels, self.num_anchors, 1
        )
        self.bbox_y3d_loc = LocalConv2d(
            self.num_rows, self.prop_feats[0].out_channels, self.num_anchors, 1
        )
        self.bbox_z3d_loc = LocalConv2d(
            self.num_rows, self.prop_feats[0].out_channels, self.num_anchors, 1
        )
        self.bbox_w3d_loc = LocalConv2d(
            self.num_rows, self.prop_feats[0].out_channels, self.num_anchors, 1
        )
        self.bbox_h3d_loc = LocalConv2d(
            self.num_rows, self.prop_feats[0].out_channels, self.num_anchors, 1
        )
        self.bbox_l3d_loc = LocalConv2d(
            self.num_rows, self.prop_feats[0].out_channels, self.num_anchors, 1
        )
        self.bbox_rY3d_loc = LocalConv2d(
            self.num_rows, self.prop_feats[0].out_channels, self.num_anchors, 1
        )

        self.cls_ble = nn.Parameter(torch.tensor(10e-5))

        self.bbox_x_ble = nn.Parameter(torch.tensor(10e-5))
        self.bbox_y_ble = nn.Parameter(torch.tensor(10e-5))
        self.bbox_w_ble = nn.Parameter(torch.tensor(10e-5))
        self.bbox_h_ble = nn.Parameter(torch.tensor(10e-5))

        self.bbox_x3d_ble = nn.Parameter(torch.tensor(10e-5))
        self.bbox_y3d_ble = nn.Parameter(torch.tensor(10e-5))
        self.bbox_z3d_ble = nn.Parameter(torch.tensor(10e-5))
        self.bbox_w3d_ble = nn.Parameter(torch.tensor(10e-5))
        self.bbox_h3d_ble = nn.Parameter(torch.tensor(10e-5))
        self.bbox_l3d_ble = nn.Parameter(torch.tensor(10e-5))
        self.bbox_rY3d_ble = nn.Parameter(torch.tensor(10e-5))

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

        self.feat_stride = conf["feat_stride"]
        self.feat_size = calc_output_size(np.array(conf["crop_size"]), self.feat_stride)
        self.rois = locate_anchors(
            conf["anchors"], self.feat_size, conf["feat_stride"], convert_tensor=True
        )
        self.rois = self.rois.type(torch.FloatTensor)
        self.anchors = conf["anchors"]
        self.bbox_means = conf["bbox_means"]
        self.bbox_stds = conf["bbox_stds"]

    def forward(self, x):
        batch_size = x.size(0)

        # densenet
        x = self.base(x)

        prop_feats = self.prop_feats(x)
        prop_feats_loc = self.prop_feats_loc(x)

        cls = self.cls(prop_feats)

        # bbox 2d
        bbox_x = self.bbox_x(prop_feats)
        bbox_y = self.bbox_y(prop_feats)
        bbox_w = self.bbox_w(prop_feats)
        bbox_h = self.bbox_h(prop_feats)

        # bbox 3d
        bbox_x3d = self.bbox_x3d(prop_feats)
        bbox_y3d = self.bbox_y3d(prop_feats)
        bbox_z3d = self.bbox_z3d(prop_feats)
        bbox_w3d = self.bbox_w3d(prop_feats)
        bbox_h3d = self.bbox_h3d(prop_feats)
        bbox_l3d = self.bbox_l3d(prop_feats)
        bbox_rY3d = self.bbox_rY3d(prop_feats)

        cls_loc = self.cls_loc(prop_feats_loc)

        # bbox 2d
        bbox_x_loc = self.bbox_x_loc(prop_feats_loc)
        bbox_y_loc = self.bbox_y_loc(prop_feats_loc)
        bbox_w_loc = self.bbox_w_loc(prop_feats_loc)
        bbox_h_loc = self.bbox_h_loc(prop_feats_loc)

        # bbox 3d
        bbox_x3d_loc = self.bbox_x3d_loc(prop_feats_loc)
        bbox_y3d_loc = self.bbox_y3d_loc(prop_feats_loc)
        bbox_z3d_loc = self.bbox_z3d_loc(prop_feats_loc)
        bbox_w3d_loc = self.bbox_w3d_loc(prop_feats_loc)
        bbox_h3d_loc = self.bbox_h3d_loc(prop_feats_loc)
        bbox_l3d_loc = self.bbox_l3d_loc(prop_feats_loc)
        bbox_rY3d_loc = self.bbox_rY3d_loc(prop_feats_loc)

        cls_ble = self.sigmoid(self.cls_ble)

        # bbox 2d
        bbox_x_ble = self.sigmoid(self.bbox_x_ble)
        bbox_y_ble = self.sigmoid(self.bbox_y_ble)
        bbox_w_ble = self.sigmoid(self.bbox_w_ble)
        bbox_h_ble = self.sigmoid(self.bbox_h_ble)

        # bbox 3d
        bbox_x3d_ble = self.sigmoid(self.bbox_x3d_ble)
        bbox_y3d_ble = self.sigmoid(self.bbox_y3d_ble)
        bbox_z3d_ble = self.sigmoid(self.bbox_z3d_ble)
        bbox_w3d_ble = self.sigmoid(self.bbox_w3d_ble)
        bbox_h3d_ble = self.sigmoid(self.bbox_h3d_ble)
        bbox_l3d_ble = self.sigmoid(self.bbox_l3d_ble)
        bbox_rY3d_ble = self.sigmoid(self.bbox_rY3d_ble)

        # blend
        cls = (cls * cls_ble) + (cls_loc * (1 - cls_ble))

        bbox_x = (bbox_x * bbox_x_ble) + (bbox_x_loc * (1 - bbox_x_ble))
        bbox_y = (bbox_y * bbox_y_ble) + (bbox_y_loc * (1 - bbox_y_ble))
        bbox_w = (bbox_w * bbox_w_ble) + (bbox_w_loc * (1 - bbox_w_ble))
        bbox_h = (bbox_h * bbox_h_ble) + (bbox_h_loc * (1 - bbox_h_ble))

        bbox_x3d = (bbox_x3d * bbox_x3d_ble) + (bbox_x3d_loc * (1 - bbox_x3d_ble))
        bbox_y3d = (bbox_y3d * bbox_y3d_ble) + (bbox_y3d_loc * (1 - bbox_y3d_ble))
        bbox_z3d = (bbox_z3d * bbox_z3d_ble) + (bbox_z3d_loc * (1 - bbox_z3d_ble))
        bbox_h3d = (bbox_h3d * bbox_h3d_ble) + (bbox_h3d_loc * (1 - bbox_h3d_ble))
        bbox_w3d = (bbox_w3d * bbox_w3d_ble) + (bbox_w3d_loc * (1 - bbox_w3d_ble))
        bbox_l3d = (bbox_l3d * bbox_l3d_ble) + (bbox_l3d_loc * (1 - bbox_l3d_ble))
        bbox_rY3d = (bbox_rY3d * bbox_rY3d_ble) + (bbox_rY3d_loc * (1 - bbox_rY3d_ble))

        feat_h = cls.size(2)
        feat_w = cls.size(3)

        # reshape for cross entropy
        cls = cls.view(batch_size, self.num_classes, feat_h * self.num_anchors, feat_w)

        # score probabilities
        prob = self.softmax(cls)

        # reshape for consistency
        bbox_x = flatten_tensor(bbox_x.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_y = flatten_tensor(bbox_y.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_w = flatten_tensor(bbox_w.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_h = flatten_tensor(bbox_h.view(batch_size, 1, feat_h * self.num_anchors, feat_w))

        bbox_x3d = flatten_tensor(bbox_x3d.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_y3d = flatten_tensor(bbox_y3d.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_z3d = flatten_tensor(bbox_z3d.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_w3d = flatten_tensor(bbox_w3d.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_h3d = flatten_tensor(bbox_h3d.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_l3d = flatten_tensor(bbox_l3d.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_rY3d = flatten_tensor(bbox_rY3d.view(batch_size, 1, feat_h * self.num_anchors, feat_w))

        # bundle
        bbox_2d = torch.cat((bbox_x, bbox_y, bbox_w, bbox_h), dim=2)
        bbox_3d = torch.cat(
            (bbox_x3d, bbox_y3d, bbox_z3d, bbox_w3d, bbox_h3d, bbox_l3d, bbox_rY3d), dim=2
        )

        feat_size = [feat_h, feat_w]

        cls = flatten_tensor(cls)
        prob = flatten_tensor(prob)

        if self.feat_size[0] != feat_h or self.feat_size[1] != feat_w:
            self.feat_size = [feat_h, feat_w]
            self.rois = locate_anchors(
                self.anchors, self.feat_size, self.feat_stride, convert_tensor=True
            )
            self.rois = self.rois.type(torch.FloatTensor)

        if self.training:
            return cls, prob, bbox_2d, bbox_3d, feat_size
        else:
            return cls, prob, bbox_2d, bbox_3d, feat_size, self.rois


def build_m3drpn():
    densenet121 = models.densenet121(weights=None)

    # Small stand-in training config (EasyDict in the original repo), sized
    # down from the real KITTI config (crop_size=[512,1760], 12 anchor
    # scale/ratio combinations) so the traced network stays tiny.
    conf = {
        "lbls": ["Car"],
        "anchors": np.zeros((3, 4), dtype=np.float32),
        "bins": 2,
        "test_scale": 32,
        "feat_stride": 16,
        "crop_size": [32, 64],
        "bbox_means": np.zeros((1, 11), dtype=np.float32),
        "bbox_stds": np.ones((1, 11), dtype=np.float32),
    }

    rpn_net = RPN("train", densenet121.features, conf)
    rpn_net.eval()
    return rpn_net


def example_input_m3drpn():
    return torch.randn(1, 3, 32, 64)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("M3D-RPN", "build_m3drpn", "example_input_m3drpn", 2019, "vendored"),
]
