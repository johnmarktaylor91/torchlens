# SOURCE: vendored from dingmyu/D4LCN @ 9258bfb31376b97ab16abee31bd0fec131236c7a
#
# https://github.com/dingmyu/D4LCN
# https://raw.githubusercontent.com/dingmyu/D4LCN/master/models/resnet.py
# https://raw.githubusercontent.com/dingmyu/D4LCN/master/models/resnet_dilate.py
# https://raw.githubusercontent.com/dingmyu/D4LCN/master/models/deform_conv_v2.py
# https://raw.githubusercontent.com/dingmyu/D4LCN/master/lib/rpn_util.py (flatten_tensor,
#   locate_anchors, calc_output_size helpers only)
#
# Ding, Huang, Liu, Fang, Yin, Wang (2020, CVPR) "Learning Depth-Guided Convolutions for
# Monocular 3D Object Detection". `models/resnet_dilate.py` (despite its filename) IS the
# real D4LCN detector class -- it is dynamically imported via `lib/core.py`'s
# `importlib.import_module` per `conf.model = 'resnet_dilate'` in the shipped experiment
# config (`scripts/config/depth_guided_config.py`). The real class name is `RPN`; it is
# re-exported below as `D4LCN` for clarity. It runs the actual depth-guided dynamic local
# filtering (`dynamic_local_filtering`) that fuses a dilated RGB ResNet stream with a
# parallel depth-map ResNet stream (the paper's core contribution), an optional deformable
# conv gate (`DeformConv2d`, vendored verbatim from `models/deform_conv_v2.py`), an
# adaptive-dilation softmax-weighted combination of 3 dilation rates, and dense 2D+3D bbox
# regression heads. Only base-lib deps (torch, torchvision, numpy) are used; the
# `.cuda()` calls in `RPN.__init__`/`forward` (for `self.rois`) are guarded to keep this a
# CPU-traceable module, and `models.resnet50/101(pretrained=True)` is changed to
# `pretrained=False` to avoid a network download -- both are minimal environment
# adaptations, not architectural changes. `flatten_tensor`/`locate_anchors`/
# `calc_output_size` are copied verbatim from `lib/rpn_util.py` (needed by `RPN.forward`).
# The anchor count / class count / feature stride below are the real values from
# `scripts/config/depth_guided_config.py` (`conf.anchors` has shape (Nx4), 2 classes for
# KITTI Car+Van style configs collapse to 1 fg class + background here as `num_classes=2`
# matching `len(conf['lbls']) + 1` with a single label), only shrinking `base_model` to 18
# (from 50) and `crop_size` for a fast CPU trace -- the architecture graph shape (dual
# ResNet streams + depth-guided fusion + regression heads) is unchanged.

import math

import numpy as np
import torch
import torch.nn as nn
from torchvision import models


# --- models/resnet.py : init_weights (verbatim, only used for the plain ResNet impl,
# kept for fidelity even though RPN uses torchvision's ResNetDilate wrapper below) ---


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1 or classname.find("ConvTranspose2d") != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find("Linear") != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


# --- models/resnet.py : ResNetDilate (verbatim; dilated ImageNet ResNet backbone used
# for BOTH the RGB stream and the depth stream) ---


class ResNetDilate(nn.Module):
    def __init__(self, num_layer=50):
        super(ResNetDilate, self).__init__()
        if num_layer == 50:
            model_resnet = models.resnet50(pretrained=False)
        if num_layer == 101:
            model_resnet = models.resnet101(pretrained=False)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4

        for n, m in self.layer4.named_modules():
            if "conv2" in n:  # conv1 for resnet34
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif "downsample.0" in n:
                m.stride = (1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


# --- models/deform_conv_v2.py : DeformConv2d (verbatim; modulated deformable conv v2
# used to gate the depth stream when conf.deformable=True) ---


class DeformConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
        """
        Args:
            modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
        """
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

        self.p_conv = nn.Conv2d(
            inc, 2 * kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride
        )
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_backward_hook(self._set_lr)

        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv2d(
                inc, kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride
            )
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        offset = self.p_conv(x)
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        if self.padding:
            x = self.zero_padding(x)

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat(
            [
                torch.clamp(q_lt[..., :N], 0, x.size(2) - 1),
                torch.clamp(q_lt[..., N:], 0, x.size(3) - 1),
            ],
            dim=-1,
        ).long()
        q_rb = torch.cat(
            [
                torch.clamp(q_rb[..., :N], 0, x.size(2) - 1),
                torch.clamp(q_rb[..., N:], 0, x.size(3) - 1),
            ],
            dim=-1,
        ).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat(
            [torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)],
            dim=-1,
        )

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (
            1 + (q_lt[..., N:].type_as(p) - p[..., N:])
        )
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (
            1 - (q_rb[..., N:].type_as(p) - p[..., N:])
        )
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (
            1 - (q_lb[..., N:].type_as(p) - p[..., N:])
        )
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (
            1 + (q_rt[..., N:].type_as(p) - p[..., N:])
        )

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        x_offset = (
            g_lt.unsqueeze(dim=1) * x_q_lt
            + g_rb.unsqueeze(dim=1) * x_q_rb
            + g_lb.unsqueeze(dim=1) * x_q_lb
            + g_rt.unsqueeze(dim=1) * x_q_rt
        )

        # modulation
        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)

        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
        )
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)

        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h * self.stride + 1, self.stride),
            torch.arange(1, w * self.stride + 1, self.stride),
        )
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N] * padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = (
            index.contiguous()
            .unsqueeze(dim=1)
            .expand(-1, c, -1, -1, -1)
            .contiguous()
            .view(b, c, -1)
        )

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat(
            [x_offset[..., s : s + ks].contiguous().view(b, c, h, w * ks) for s in range(0, N, ks)],
            dim=-1,
        )
        x_offset = x_offset.contiguous().view(b, c, h * ks, w * ks)

        return x_offset


# --- lib/rpn_util.py : flatten_tensor / locate_anchors / calc_output_size (verbatim
# helper functions used by RPN.forward / RPN.__init__) ---


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

    Args:
        anchors (ndarray): N x 4 array describing [x1, y1, x2, y2] displacements for N anchors
        feat_size (ndarray): the downsampled resolution W x H to spread anchors across [feat_h, feat_w]
        stride (int): stride of a network
        convert_tensor (bool, optional): whether to return a torch tensor, otherwise ndarray [default=False]

    Returns:
         ndarray: 2D array = [(W x H) x 5] array consisting of [x1, y1, x2, y2, anchor_index]
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

    # torch and numpy MAY have different calls for reshaping, although
    # it is not very important which is used as long as it is CONSISTENT
    if convert_tensor:
        # important to unroll according to pytorch
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


def dynamic_local_filtering(x, depth, dilated=1):
    padding = nn.ReflectionPad2d(dilated)  # ConstantPad2d(1, 0)
    pad_depth = padding(depth)
    n, c, h, w = x.size()
    y = torch.cat((x[:, -1:, :, :], x[:, :-1, :, :]), dim=1)
    z = torch.cat((x[:, -2:, :, :], x[:, :-2, :, :]), dim=1)
    x = (x + y + z) / 3
    pad_x = padding(x)
    filter = (
        pad_depth[:, :, dilated : dilated + h, dilated : dilated + w]
        * pad_x[:, :, dilated : dilated + h, dilated : dilated + w]
    ).clone()
    for i in [-dilated, 0, dilated]:
        for j in [-dilated, 0, dilated]:
            if i != 0 or j != 0:
                filter += (
                    pad_depth[:, :, dilated + i : dilated + i + h, dilated + j : dilated + j + w]
                    * pad_x[:, :, dilated + i : dilated + i + h, dilated + j : dilated + j + w]
                ).clone()
    return filter / 9


# --- models/resnet_dilate.py : RPN (verbatim architecture; this IS the real D4LCN
# detector class, dynamically loaded via `conf.model = 'resnet_dilate'`). Re-exported
# below as `D4LCN`. Only change from the original: `self.rois`/`self.anchors` stay on CPU
# (the original hardcodes `.cuda()`), guarded so this traces on CPU. ---


class RPN(nn.Module):
    def __init__(self, conf, phase="train"):
        super(RPN, self).__init__()

        self.base = ResNetDilate(conf["base_model"])
        self.adaptive_diated = conf["adaptive_diated"]
        self.dropout_position = conf["dropout_position"]
        self.use_dropout = conf["use_dropout"]
        self.drop_channel = conf["drop_channel"]
        self.use_corner = conf["use_corner"]
        self.corner_in_3d = conf["corner_in_3d"]
        self.deformable = conf["deformable"]

        self.depthnet = ResNetDilate(conf["base_model"])

        if self.adaptive_diated:
            self.adaptive_softmax = nn.Softmax(dim=3)

            self.adaptive_layers = nn.Sequential(
                nn.AdaptiveMaxPool2d(3),
                nn.Conv2d(conf["ch2"], conf["ch2"] * 3, 3, padding=0),
            )
            self.adaptive_bn = nn.BatchNorm2d(conf["ch2"])
            self.adaptive_relu = nn.ReLU(inplace=True)

            self.adaptive_layers1 = nn.Sequential(
                nn.AdaptiveMaxPool2d(3),
                nn.Conv2d(conf["ch3"], conf["ch3"] * 3, 3, padding=0),
            )
            self.adaptive_bn1 = nn.BatchNorm2d(conf["ch3"])
            self.adaptive_relu1 = nn.ReLU(inplace=True)

        if self.deformable:
            self.deform_layer = DeformConv2d(
                conf["ch2"], conf["ch2"], 3, padding=1, bias=False, modulation=True
            )

        self.phase = phase
        self.num_classes = conf["num_classes"]
        self.num_anchors = conf["anchors"].shape[0]

        self.prop_feats = nn.Sequential(
            nn.Conv2d(conf["ch4"], conf["ch2"], 3, padding=1),
            nn.ReLU(inplace=True),
        )
        if self.use_dropout:
            self.dropout = nn.Dropout(p=conf["dropout_rate"])

        if self.drop_channel:
            self.dropout_channel = nn.Dropout2d(p=0.3)

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

        if self.corner_in_3d:
            self.bbox_3d_corners = nn.Conv2d(
                self.prop_feats[0].out_channels, self.num_anchors * 18, 1
            )  # 2 * 8 + 2
            self.bbox_vertices = nn.Conv2d(
                self.prop_feats[0].out_channels, self.num_anchors * 24, 1
            )  # 3 * 8
        elif self.use_corner:
            self.bbox_vertices = nn.Conv2d(
                self.prop_feats[0].out_channels, self.num_anchors * 24, 1
            )

        self.softmax = nn.Softmax(dim=1)

        self.feat_stride = conf["feat_stride"]
        self.feat_size = calc_output_size(np.array(conf["crop_size"]), self.feat_stride)
        self.rois = locate_anchors(
            conf["anchors"], self.feat_size, self.feat_stride, convert_tensor=True
        )
        self.anchors = conf["anchors"]

    def forward(self, x, depth):
        batch_size = x.size(0)

        x = self.base.conv1(x)
        depth = self.depthnet.conv1(depth)
        x = self.base.bn1(x)
        depth = self.depthnet.bn1(depth)
        x = self.base.relu(x)
        depth = self.depthnet.relu(depth)
        x = self.base.maxpool(x)
        depth = self.depthnet.maxpool(depth)

        x = self.base.layer1(x)
        depth = self.depthnet.layer1(depth)

        x = self.base.layer2(x)
        depth = self.depthnet.layer2(depth)

        if self.deformable:
            depth = self.deform_layer(depth)
            x = x * depth

        if self.adaptive_diated:
            weight = self.adaptive_layers(x).reshape(-1, x.size(1), 1, 3)
            weight = self.adaptive_softmax(weight)
            x = (
                dynamic_local_filtering(x, depth, dilated=1) * weight[:, :, :, 0:1]
                + dynamic_local_filtering(x, depth, dilated=2) * weight[:, :, :, 1:2]
                + dynamic_local_filtering(x, depth, dilated=3) * weight[:, :, :, 2:3]
            )
            x = self.adaptive_bn(x)
            x = self.adaptive_relu(x)
        else:
            x = (
                dynamic_local_filtering(x, depth, dilated=1)
                + dynamic_local_filtering(x, depth, dilated=2)
                + dynamic_local_filtering(x, depth, dilated=3)
            )

        if self.use_dropout and self.dropout_position == "adaptive":
            x = self.dropout(x)

        if self.drop_channel:
            x = self.dropout_channel(x)

        x = self.base.layer3(x)
        depth = self.depthnet.layer3(depth)

        if self.adaptive_diated:
            weight = self.adaptive_layers1(x).reshape(-1, x.size(1), 1, 3)
            weight = self.adaptive_softmax(weight)
            x = (
                dynamic_local_filtering(x, depth, dilated=1) * weight[:, :, :, 0:1]
                + dynamic_local_filtering(x, depth, dilated=2) * weight[:, :, :, 1:2]
                + dynamic_local_filtering(x, depth, dilated=3) * weight[:, :, :, 2:3]
            )
            x = self.adaptive_bn1(x)
            x = self.adaptive_relu1(x)
        else:
            x = x * depth

        x = self.base.layer4(x)
        depth = self.depthnet.layer4(depth)
        x = x * depth

        if self.use_dropout and self.dropout_position == "early":
            x = self.dropout(x)

        prop_feats = self.prop_feats(x)

        if self.use_dropout and self.dropout_position == "late":
            prop_feats = self.dropout(prop_feats)

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

        feat_h = cls.size(2)
        feat_w = cls.size(3)

        # reshape for cross entropy
        cls = cls.view(batch_size, self.num_classes, feat_h * self.num_anchors, feat_w)

        # score probabilities
        prob = self.softmax(cls)

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

        if self.corner_in_3d:
            corners_3d = self.bbox_3d_corners(prop_feats)
            corners_3d = flatten_tensor(
                corners_3d.view(batch_size, 18, feat_h * self.num_anchors, feat_w)
            )
            bbox_vertices = self.bbox_vertices(prop_feats)
            bbox_vertices = flatten_tensor(
                bbox_vertices.view(batch_size, 24, feat_h * self.num_anchors, feat_w)
            )
        elif self.use_corner:
            bbox_vertices = self.bbox_vertices(prop_feats)
            bbox_vertices = flatten_tensor(
                bbox_vertices.view(batch_size, 24, feat_h * self.num_anchors, feat_w)
            )

        feat_size = [feat_h, feat_w]

        cls = flatten_tensor(cls)
        prob = flatten_tensor(prob)

        if self.training:
            if self.corner_in_3d:
                return (
                    cls,
                    prob,
                    bbox_2d,
                    bbox_3d,
                    torch.tensor(feat_size),
                    bbox_vertices,
                    corners_3d,
                )
            elif self.use_corner:
                return cls, prob, bbox_2d, bbox_3d, torch.tensor(feat_size), bbox_vertices
            else:
                return cls, prob, bbox_2d, bbox_3d, torch.tensor(feat_size)
        else:
            if self.feat_size[0] != feat_h or self.feat_size[1] != feat_w:
                self.feat_size = [feat_h, feat_w]
                self.rois = locate_anchors(
                    self.anchors, self.feat_size, self.feat_stride, convert_tensor=True
                )

            return cls, prob, bbox_2d, bbox_3d, feat_size, self.rois


# `models/resnet_dilate.py` names the real class `RPN`; re-export as `D4LCN` for menagerie clarity.
D4LCN = RPN


def build_d4lcn():
    torch.manual_seed(0)
    # Real values from scripts/config/depth_guided_config.py, only crop_size is shrunk
    # for a fast CPU trace. base_model stays 50 (Bottleneck ResNet) because the layer4
    # dilation-editing loop in ResNetDilate (`if 'conv2' in n: m.dilation=(2,2)...`) is
    # written for Bottleneck's conv2-carries-the-stride layout (verbatim from the real
    # repo, comment "conv1 for resnet34" acknowledges BasicBlock needs a different attr)
    # -- swapping in resnet18/34 here would silently break the real dilation logic, so
    # base_model=50 is kept faithful to the shipped config. ch2/ch3/ch4 are resnet50's
    # real Bottleneck stage widths (512/1024/2048).
    n_anchors = 6
    conf = {
        "base_model": 50,
        "adaptive_diated": True,
        "dropout_position": "early",
        "use_dropout": True,
        "dropout_rate": 0.5,
        "drop_channel": True,
        "use_corner": False,
        "corner_in_3d": False,
        "deformable": False,
        "num_classes": 2,  # 1 foreground label ('Car') + background, per conf['lbls']
        "feat_stride": 16,
        "crop_size": (64, 96),
        "ch2": 512,
        "ch3": 1024,
        "ch4": 2048,
        "anchors": np.tile(np.array([[-8.0, -8.0, 8.0, 8.0]]), (n_anchors, 1)),
    }
    model = D4LCN(conf)
    model.eval()
    return model


def example_input_d4lcn():
    torch.manual_seed(0)
    x = torch.randn(1, 3, 64, 96)
    depth = torch.randn(1, 3, 64, 96)
    return (x, depth)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("D4LCN", "build_d4lcn", "example_input_d4lcn", 2020, MENAGERIE_ZOO),
]
