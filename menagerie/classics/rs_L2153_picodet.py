# SOURCE: vendored from https://github.com/yolyyin/picodet_pytorch @ 3dcf295a69aaa5f9431ce948dfe7a848b2d4e9a5
# (picodet/esnet.py, picodet/csp_pan.py, picodet/pico_head.py, picodet/varifocal_loss.py,
# picodet/iou_loss.py, picodet/dfl_loss.py, picodet/simota_assigner.py, picodet/utils.py,
# picodet/picodet.py) -- PicoDet: A Better, Faster and Stronger Object Detector for
# Mobile Devices (arXiv:2111.00902). The official implementation lives in PaddlePaddle
# (PaddlePaddle/PaddleDetection); this is a real, independent PyTorch reimplementation of
# the full PicoDet stack (ESNet backbone + CSP-PAN neck + PicoHead), vendored verbatim.
# Only import paths were flattened from the original multi-file `picodet/` package into
# this single staging module (module bodies, class/method structure, and forward logic
# are unchanged); the training-only loss/assigner/NMS constructor args are still built
# exactly as in engine/model.py:create_model() even though the traced forward pass below
# does not exercise their code paths (PicoHead in eval mode with export_post_process=False
# never calls get_loss/post_process).
"""PicoDet mobile object detector, vendored from yolyyin/picodet_pytorch."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import AdaptiveAvgPool2d, BatchNorm2d, Conv2d, GroupNorm, MaxPool2d

MENAGERIE_ZOO = "vendored-pytorch"


# ============================== picodet/esnet.py ==============================


def make_divisible(v, divisor=16, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def channel_shuffle(x, groups):
    batch_size, num_channels, height, width = x.shape[0:4]
    assert num_channels % groups == 0, "num_channels should be divisible by groups"
    channels_per_group = num_channels // groups
    x = torch.reshape(input=x, shape=(batch_size, groups, channels_per_group, height, width))
    x = torch.permute(x, (0, 2, 1, 3, 4))
    x = torch.reshape(x, (batch_size, num_channels, height, width))
    return x


class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(output_size=1)
        self.conv1 = Conv2d(
            in_channels=channel,
            out_channels=channel // reduction,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.conv2 = Conv2d(
            in_channels=channel // reduction,
            out_channels=channel,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        torch.nn.init.normal_(self.conv1.weight)
        torch.nn.init.normal_(self.conv2.weight)
        torch.nn.init.constant_(self.conv1.bias, 0.0)
        torch.nn.init.constant_(self.conv2.bias, 0.0)

    def forward(self, inputs):
        outputs = self.avg_pool(inputs)
        outputs = self.conv1(outputs)
        outputs = F.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = F.hardsigmoid(outputs)
        return torch.multiply(inputs, outputs)


class ConvBNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1, act=None):
        super(ConvBNLayer, self).__init__()
        self._conv = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        nn.init.kaiming_normal_(self._conv.weight)

        self._batch_norm = BatchNorm2d(out_channels)
        if act == "hard_swish":
            act = "hardswish"
        self.act = act

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self._batch_norm(y)
        if self.act:
            y = getattr(F, self.act)(y)
        return y


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride, act="relu"):
        super(InvertedResidual, self).__init__()
        self._conv_pw = ConvBNLayer(
            in_channels=in_channels // 2,
            out_channels=mid_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=act,
        )
        self._conv_dw = ConvBNLayer(
            in_channels=mid_channels // 2,
            out_channels=mid_channels // 2,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=mid_channels // 2,
            act=None,
        )
        self._se = SEModule(mid_channels)

        self._conv_linear = ConvBNLayer(
            in_channels=mid_channels,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=act,
        )

    def forward(self, inputs):
        x1, x2 = torch.split(
            inputs, split_size_or_sections=[inputs.shape[1] // 2, inputs.shape[1] // 2], dim=1
        )
        x2 = self._conv_pw(x2)
        x3 = self._conv_dw(x2)
        x3 = torch.cat([x2, x3], dim=1)
        x3 = self._se(x3)
        x3 = self._conv_linear(x3)
        out = torch.cat([x1, x3], dim=1)
        return channel_shuffle(out, 2)


class InvertedResidualDS(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride, act="relu"):
        super(InvertedResidualDS, self).__init__()

        self._conv_dw_1 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            act=None,
        )
        self._conv_linear_1 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=act,
        )
        self._conv_pw_2 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=mid_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=act,
        )
        self._conv_dw_2 = ConvBNLayer(
            in_channels=mid_channels // 2,
            out_channels=mid_channels // 2,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=mid_channels // 2,
            act=None,
        )
        self._se = SEModule(mid_channels // 2)
        self._conv_linear_2 = ConvBNLayer(
            in_channels=mid_channels // 2,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=act,
        )
        self._conv_dw_mv1 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=out_channels,
            act="hard_swish",
        )
        self._conv_pw_mv1 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            act="hard_swish",
        )

    def forward(self, inputs):
        x1 = self._conv_dw_1(inputs)
        x1 = self._conv_linear_1(x1)
        x2 = self._conv_pw_2(inputs)
        x2 = self._conv_dw_2(x2)
        x2 = self._se(x2)
        x2 = self._conv_linear_2(x2)
        out = torch.cat([x1, x2], dim=1)
        out = self._conv_dw_mv1(out)
        out = self._conv_pw_mv1(out)

        return out


class ESNet(nn.Module):
    def __init__(
        self,
        scale=1.0,
        act="hard_swish",
        feature_maps=[4, 11, 14],
        channel_ratio=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ):
        super(ESNet, self).__init__()
        self.scale = scale
        self.feature_maps = feature_maps
        stage_repeats = [3, 7, 3]
        stage_out_channels = [
            -1,
            24,
            make_divisible(128 * scale),
            make_divisible(256 * scale),
            make_divisible(512 * scale),
            1024,
        ]
        self._out_channels = []
        self._feature_idx = 0
        self._conv1 = ConvBNLayer(
            in_channels=3,
            out_channels=stage_out_channels[1],
            kernel_size=3,
            stride=2,
            padding=1,
            act=act,
        )
        self._max_pool = MaxPool2d(kernel_size=3, stride=2, padding=1)
        self._feature_idx += 1

        self._block_list = []
        arch_idx = 0
        for stage_id, num_repeat in enumerate(stage_repeats):
            for i in range(num_repeat):
                channels_scales = channel_ratio[arch_idx]
                mid_c = make_divisible(
                    int(stage_out_channels[stage_id + 2] * channels_scales), divisor=8
                )
                if i == 0:
                    self.add_module(
                        name=str(stage_id + 2) + "_" + str(i + 1),
                        module=InvertedResidualDS(
                            in_channels=stage_out_channels[stage_id + 1],
                            mid_channels=mid_c,
                            out_channels=stage_out_channels[stage_id + 2],
                            stride=2,
                            act=act,
                        ),
                    )
                    block = self.get_submodule(str(stage_id + 2) + "_" + str(i + 1))
                else:
                    self.add_module(
                        name=str(stage_id + 2) + "_" + str(i + 1),
                        module=InvertedResidual(
                            in_channels=stage_out_channels[stage_id + 2],
                            mid_channels=mid_c,
                            out_channels=stage_out_channels[stage_id + 2],
                            stride=1,
                            act=act,
                        ),
                    )
                    block = self.get_submodule(str(stage_id + 2) + "_" + str(i + 1))
                self._block_list.append(block)
                arch_idx += 1
                self._feature_idx += 1
                self._update_out_channels(
                    stage_out_channels[stage_id + 2], self._feature_idx, self.feature_maps
                )

    def _update_out_channels(self, channel, feature_idx, feature_maps):
        if feature_idx in feature_maps:
            self._out_channels.append(channel)

    def forward(self, inputs):
        images, _ = inputs
        images = torch.stack(images, dim=0)
        y = self._conv1(images)
        y = self._max_pool(y)
        outs = []
        for i, inv in enumerate(self._block_list):
            y = inv(y)
            if i + 2 in self.feature_maps:
                outs.append(y)

        return outs


# ============================== picodet/csp_pan.py ==============================


class _CspConvBNLayer(nn.Module):
    def __init__(
        self, in_channel=96, out_channel=96, kernel_size=3, stride=1, groups=1, act="leaky_relu"
    ):
        super(_CspConvBNLayer, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=kernel_size,
            groups=groups,
            padding=(kernel_size - 1) // 2,
            stride=stride,
            bias=False,
        )
        torch.nn.init.kaiming_normal_(self.conv.weight)
        self.bn = nn.BatchNorm2d(out_channel)
        if act == "hard_swish":
            act = "hardswish"
        self.act = act

    def forward(self, x):
        x = self.bn(self.conv(x))
        if self.act:
            x = getattr(F, self.act)(x)
        return x


class DPModule(nn.Module):
    def __init__(
        self,
        in_channel=96,
        out_channel=96,
        kernel_size=3,
        stride=1,
        act="leaky_relu",
        use_act_in_out=True,
    ):
        super(DPModule, self).__init__()
        self.use_act_in_out = use_act_in_out
        self.dwconv = nn.Conv2d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=kernel_size,
            groups=out_channel,
            padding=(kernel_size - 1) // 2,
            stride=stride,
            bias=False,
        )
        torch.nn.init.kaiming_normal_(self.dwconv.weight)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.pwconv = nn.Conv2d(
            in_channels=out_channel,
            out_channels=out_channel,
            kernel_size=1,
            groups=1,
            padding=0,
            stride=1,
            bias=False,
        )
        torch.nn.init.kaiming_normal_(self.pwconv.weight)
        self.bn2 = nn.BatchNorm2d(out_channel)
        if act == "hard_swish":
            act = "hardswish"
        self.act = act

    def forward(self, x):
        x = self.bn1(self.dwconv(x))
        if self.act:
            x = getattr(F, self.act)(x)
        x = self.bn2(self.pwconv(x))
        if self.use_act_in_out and self.act:
            x = getattr(F, self.act)(x)
        return x


class DarknetBottleneck(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        expansion=0.5,
        add_identity=True,
        use_depthwise=False,
        act="leaky_relu",
    ):
        super(DarknetBottleneck, self).__init__()
        hidden_channels = int(out_channels * expansion)
        conv_func = DPModule if use_depthwise else _CspConvBNLayer
        self.conv1 = _CspConvBNLayer(
            in_channel=in_channels, out_channel=hidden_channels, kernel_size=1, act=act
        )
        self.conv2 = conv_func(
            in_channel=hidden_channels,
            out_channel=out_channels,
            kernel_size=kernel_size,
            stride=1,
            act=act,
        )
        self.add_identity = add_identity and in_channels == out_channels

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.add_identity:
            return out + identity
        else:
            return out


class CSPLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        expand_ratio=0.5,
        num_blocks=1,
        add_identity=True,
        use_depthwise=False,
        act="leaky_relu",
    ):
        super().__init__()
        mid_channels = int(out_channels * expand_ratio)
        self.main_conv = _CspConvBNLayer(in_channels, mid_channels, 1, act=act)
        self.short_conv = _CspConvBNLayer(in_channels, mid_channels, 1, act=act)
        self.final_conv = _CspConvBNLayer(2 * mid_channels, out_channels, 1, act=act)

        self.blocks = nn.Sequential(
            *[
                DarknetBottleneck(
                    mid_channels,
                    mid_channels,
                    kernel_size,
                    1.0,
                    add_identity,
                    use_depthwise,
                    act=act,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x):
        x_short = self.short_conv(x)

        x_main = self.main_conv(x)
        x_main = self.blocks(x_main)

        x_final = torch.cat((x_main, x_short), dim=1)
        return self.final_conv(x_final)


class Channel_T(nn.Module):
    def __init__(self, in_channels=[116, 232, 464], out_channels=96, act="leaky_relu"):
        super(Channel_T, self).__init__()
        self.convs = nn.ModuleList()
        for i in range(len(in_channels)):
            self.convs.append(_CspConvBNLayer(in_channels[i], out_channels, 1, act=act))

    def forward(self, x):
        outs = [self.convs[i](x[i]) for i in range(len(x))]
        return outs


class CSPPAN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=5,
        num_features=3,
        num_csp_blocks=1,
        use_depthwise=True,
        act="hard_swish",
        spatial_scales=[0.125, 0.0625, 0.03125],
    ):
        super(CSPPAN, self).__init__()
        self.conv_t = Channel_T(in_channels, out_channels, act=act)
        in_channels = [out_channels] * len(spatial_scales)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial_scales = spatial_scales
        self.num_features = num_features
        conv_func = DPModule if use_depthwise else _CspConvBNLayer

        if self.num_features == 4:
            self.first_top_conv = conv_func(
                in_channels[0], in_channels[0], kernel_size, stride=2, act=act
            )
            self.second_top_conv = conv_func(
                in_channels[0], in_channels[0], kernel_size, stride=2, act=act
            )
            self.spatial_scales.append(self.spatial_scales[-1] / 2)

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.top_down_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1, 0, -1):
            self.top_down_blocks.append(
                CSPLayer(
                    in_channels[idx - 1] * 2,
                    in_channels[idx - 1],
                    kernel_size=kernel_size,
                    num_blocks=num_csp_blocks,
                    add_identity=False,
                    use_depthwise=use_depthwise,
                    act=act,
                )
            )

        self.downsamples = nn.ModuleList()
        self.bottom_up_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1):
            self.downsamples.append(
                conv_func(
                    in_channels[idx], in_channels[idx], kernel_size=kernel_size, stride=2, act=act
                )
            )
            self.bottom_up_blocks.append(
                CSPLayer(
                    in_channels[idx] * 2,
                    in_channels[idx + 1],
                    kernel_size=kernel_size,
                    num_blocks=num_csp_blocks,
                    add_identity=False,
                    use_depthwise=use_depthwise,
                    act=act,
                )
            )

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        inputs = self.conv_t(inputs)

        inner_outs = [inputs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = inputs[idx - 1]

            upsample_feat = self.upsample(feat_heigh)

            inner_out = self.top_down_blocks[len(self.in_channels) - 1 - idx](
                torch.cat([upsample_feat, feat_low], 1)
            )
            inner_outs.insert(0, inner_out)

        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsamples[idx](feat_low)
            out = self.bottom_up_blocks[idx](torch.cat([downsample_feat, feat_height], 1))
            outs.append(out)

        top_features = None
        if self.num_features == 4:
            top_features = self.first_top_conv(inputs[-1])
            top_features = top_features + self.second_top_conv(outs[-1])
            outs.append(top_features)

        return tuple(outs)


# ============================== picodet/utils.py (bbox helpers only) ==============================


def batch_distance2bbox(points, distance, max_shapes=None):
    lt, rb = torch.split(distance, 2, -1)
    x1y1 = -lt + points
    x2y2 = rb + points
    out_bbox = torch.cat([x1y1, x2y2], -1)
    if max_shapes is not None:
        max_shapes = max_shapes.flip(-1).tile([1, 2])
        delta_dim = out_bbox.dim() - max_shapes.dim()
        for _ in range(delta_dim):
            max_shapes.unsqueeze(1)
        out_bbox = torch.where(out_bbox < max_shapes, out_bbox, max_shapes)
        out_bbox = torch.where(out_bbox > 0, out_bbox, torch.zeros_like(out_bbox))
    return out_bbox


def get_level_anchors(featmap_size, stride, device, cell_offset=0.5):
    h, w = featmap_size
    shift_x = (torch.arange(end=w, device=device) + cell_offset) * stride
    shift_y = (torch.arange(end=h, device=device) + cell_offset) * stride
    shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing="ij")
    return shift_y.reshape(-1), shift_x.reshape(-1)


def bbox2distance(points, bbox, max_dis=None, eps=0.1):
    left = points[:, 0] - bbox[:, 0]
    top = points[:, 1] - bbox[:, 1]
    right = bbox[:, 2] - points[:, 0]
    bottom = bbox[:, 3] - points[:, 1]
    if max_dis is not None:
        left = left.clamp(min=0, max=max_dis - eps)
        top = top.clamp(min=0, max=max_dis - eps)
        right = right.clamp(min=0, max=max_dis - eps)
        bottom = bottom.clamp(min=0, max=max_dis - eps)
    return torch.stack([left, top, right, bottom], -1)


class MultiClassNMS(object):
    def __init__(self, nms_top_k=1000, keep_top_k=100, score_threshold=0.05, nms_threshold=0.5):
        super().__init__()
        self.nms_top_k = nms_top_k
        self.keep_top_k = keep_top_k
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold


# ============================== picodet/varifocal_loss.py ==============================


class VarifocalLoss(nn.Module):
    def __init__(self, use_sigmoid=True, alpha=0.75, gamma=2.0, iou_weighted=True, loss_weight=1.0):
        super(VarifocalLoss, self).__init__()
        assert alpha >= 0.0
        self.use_sigmoid = use_sigmoid
        self.alpha = alpha
        self.gamma = gamma
        self.iou_weighted = iou_weighted
        self.loss_weight = loss_weight


# ============================== picodet/iou_loss.py ==============================


class GIoULoss(object):
    def __init__(self, loss_weight=1.0, eps=1e-10):
        self.loss_weight = loss_weight
        self.eps = eps


# ============================== picodet/dfl_loss.py ==============================


class DistributionFocalLoss(nn.Module):
    def __init__(self, loss_weight=0.25):
        super(DistributionFocalLoss, self).__init__()
        self.loss_weight = loss_weight


# ============================== picodet/simota_assigner.py ==============================


class SimOTAAssigner(object):
    def __init__(
        self,
        center_radius=2.5,
        candidate_topk=10,
        iou_weight=3.0,
        cls_weight=1.0,
        num_classes=80,
        use_vfl=True,
    ):
        self.center_radius = center_radius
        self.candidate_topk = candidate_topk
        self.iou_weight = iou_weight
        self.cls_weight = cls_weight
        self.num_classes = num_classes
        self.use_vfl = use_vfl


# ============================== picodet/pico_head.py ==============================


class Integral(nn.Module):
    def __init__(self, reg_max=16):
        super(Integral, self).__init__()
        self.reg_max = reg_max
        self.possible_lengths = torch.linspace(0, self.reg_max, self.reg_max + 1)

    def forward(self, x):
        x = F.softmax(x.reshape([-1, self.reg_max + 1]), dim=1)
        values = self.possible_lengths.to(x.device)
        x = F.linear(x, values)
        if self.training:
            x = x.reshape([-1, 4])
        return x


class ConvNormLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        groups=1,
        norm_type="bn",
        norm_groups=32,
    ):
        super(ConvNormLayer, self).__init__()
        self.conv = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            groups=groups,
            bias=False,
        )
        nn.init.normal_(self.conv.weight, mean=0.0, std=0.01)

        if norm_type in ["bn", "sync_bn"]:
            self.norm = BatchNorm2d(out_channels)
        elif norm_type == "gn":
            self.norm = GroupNorm(norm_groups, out_channels)
        else:
            self.norm = None

    def forward(self, inputs):
        y = self.conv(inputs)
        if self.norm is not None:
            y = self.norm(y)
        return y


class PicoFeat(nn.Module):
    def __init__(
        self,
        feat_in=256,
        feat_out=96,
        num_fpn_stride=3,
        num_convs=2,
        norm_type="bn",
        share_cls_reg=False,
        act="hard_swish",
        use_se=False,
    ):
        super(PicoFeat, self).__init__()
        if use_se:
            raise RuntimeError("SE is temporarily not supported in the codes! by yty")
        if not share_cls_reg:
            raise RuntimeError("Temporarily only support share_cls_reg in the codes! by yty")
        self.num_convs = num_convs
        self.norm_type = norm_type
        self.act = act
        self.cls_convs = []

        for stage_idx in range(num_fpn_stride):
            cls_subnet_convs = []
            for i in range(self.num_convs):
                in_c = feat_in if i == 0 else feat_out
                self.add_module(
                    "cls_conv_dw{}_{}".format(stage_idx, i),
                    ConvNormLayer(
                        in_channels=in_c,
                        out_channels=feat_out,
                        kernel_size=5,
                        stride=1,
                        groups=feat_out,
                        norm_type=norm_type,
                    ),
                )
                cls_conv_dw = self.get_submodule("cls_conv_dw{}_{}".format(stage_idx, i))
                cls_subnet_convs.append(cls_conv_dw)
                self.add_module(
                    "cls_conv_pw{}_{}".format(stage_idx, i),
                    ConvNormLayer(
                        in_channels=in_c,
                        out_channels=feat_out,
                        kernel_size=1,
                        stride=1,
                        groups=1,
                        norm_type=norm_type,
                    ),
                )
                cls_conv_pw = self.get_submodule("cls_conv_pw{}_{}".format(stage_idx, i))
                cls_subnet_convs.append(cls_conv_pw)

            self.cls_convs.append(cls_subnet_convs)

    def act_func(self, x):
        if self.act == "leaky_relu":
            x = F.leaky_relu(x)
        elif self.act == "hard_swish":
            x = F.hardswish(x)
        elif self.act == "relu6":
            x = F.relu6(x)
        return x

    def forward(self, fpn_feat, stage_idx):
        assert stage_idx < len(self.cls_convs)
        cls_feat = fpn_feat
        reg_feat = fpn_feat
        for i in range(len(self.cls_convs[stage_idx])):
            cls_feat = self.act_func(self.cls_convs[stage_idx][i](cls_feat))
            reg_feat = cls_feat

        return cls_feat, reg_feat


class PicoHead(nn.Module):
    def __init__(
        self,
        conv_feat,
        loss_vfl,
        loss_dfl,
        loss_iou,
        assigner,
        num_classes=80,
        fpn_stride=[8, 16, 32],
        prior_prob=0.01,
        reg_max=16,
        feat_in_chan=96,
        nms=None,
        nms_pre=1000,
        cell_offset=0,
    ):
        super(PicoHead, self).__init__()
        self.conv_feat = conv_feat
        self.loss_vfl = loss_vfl
        self.loss_dfl = loss_dfl
        self.loss_iou = loss_iou
        self.assigner = assigner
        self.num_classes = num_classes
        self.fpn_stride = fpn_stride
        self.prior_prob = prior_prob
        self.reg_max = reg_max
        self.feat_in_chan = feat_in_chan
        self.nms = nms
        self.nms_pre = nms_pre
        self.cell_offset = cell_offset
        self.distribution_project = Integral(self.reg_max)

        self.use_sigmoid = True
        if self.use_sigmoid:
            self.cls_out_channels = self.num_classes
        else:
            self.cls_out_channels = self.num_classes + 1
        bias_init_value = -math.log((1 - self.prior_prob) / self.prior_prob)

        self.head_cls_list = []
        for i in range(len(fpn_stride)):
            self.add_module(
                "head_cls" + str(i),
                Conv2d(
                    in_channels=self.feat_in_chan,
                    out_channels=self.cls_out_channels + 4 * (self.reg_max + 1),
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ),
            )
            head_cls = self.get_submodule("head_cls" + str(i))
            torch.nn.init.normal_(head_cls.weight, mean=0.0, std=0.01)
            torch.nn.init.constant_(head_cls.bias, bias_init_value)
            self.head_cls_list.append(head_cls)

    def forward(self, fpn_feats, export_post_process=True):
        assert len(fpn_feats) == len(self.fpn_stride), (
            "The size of fpn_feats is not equal to size of fpn_stride"
        )

        if self.training:
            return self.forward_train(fpn_feats)
        else:
            return self.forward_eval(fpn_feats, export_post_process=export_post_process)

    def forward_train(self, fpn_feats):
        cls_logits_list, bboxes_reg_list = [], []
        for i, fpn_feat in enumerate(fpn_feats):
            conv_cls_feat, _ = self.conv_feat(fpn_feat, i)
            cls_logits = self.head_cls_list[i](conv_cls_feat)
            cls_score, bbox_pred = torch.split(
                cls_logits,
                [self.cls_out_channels, 4 * (self.reg_max + 1)],
                dim=1,
            )

            cls_logits_list.append(cls_score)
            bboxes_reg_list.append(bbox_pred)

        return cls_logits_list, bboxes_reg_list

    def forward_eval(self, fpn_feats, export_post_process=True):
        anchor_points, stride_tensor = self._generate_anchors(fpn_feats)
        cls_logits_list, bboxes_reg_list = [], []
        for i, fpn_feat in enumerate(fpn_feats):
            conv_cls_feat, _ = self.conv_feat(fpn_feat, i)
            cls_logits = self.head_cls_list[i](conv_cls_feat)
            cls_score, bbox_pred = torch.split(
                cls_logits,
                [self.cls_out_channels, 4 * (self.reg_max + 1)],
                dim=1,
            )
            if not export_post_process:
                cls_score_out = (
                    torch.sigmoid(cls_score).reshape([1, self.cls_out_channels, -1]).transpose(1, 2)
                )
                bbox_pred = bbox_pred.reshape([1, (self.reg_max + 1) * 4, -1]).transpose(1, 2)
            else:
                _, _, h, w = fpn_feat.shape
                l = h * w  # noqa: E741 (kept as in upstream source)
                cls_score_out = torch.sigmoid(cls_score.reshape([-1, self.cls_out_channels, l]))
                bbox_pred = bbox_pred.permute(0, 2, 3, 1)
                bbox_pred = self.distribution_project(bbox_pred)
                bbox_pred = bbox_pred.reshape([-1, l, 4])

            cls_logits_list.append(cls_score_out)
            bboxes_reg_list.append(bbox_pred)

        if export_post_process:
            cls_logits_list = torch.cat(cls_logits_list, dim=-1)
            bboxes_reg_list = torch.cat(bboxes_reg_list, dim=1)

            bboxes_reg_list = batch_distance2bbox(anchor_points, bboxes_reg_list)
            bboxes_reg_list *= stride_tensor

        return cls_logits_list, bboxes_reg_list

    def _generate_anchors(self, feats=None):
        anchor_points = []
        stride_tensor = []
        for i, stride in enumerate(self.fpn_stride):
            _, _, h, w = feats[i].shape
            device = feats[i].device
            shift_x = torch.arange(end=w, device=device) + self.cell_offset
            shift_y = torch.arange(end=h, device=device) + self.cell_offset
            shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing="ij")
            anchor_point = torch.stack([shift_x, shift_y], dim=-1).to(torch.float32)
            anchor_points.append(anchor_point.reshape([-1, 2]))
            stride_tensor.append(torch.full([h * w, 1], stride, dtype=torch.float32, device=device))
        anchor_points = torch.cat(anchor_points)
        stride_tensor = torch.cat(stride_tensor)
        return anchor_points, stride_tensor


# ============================== picodet/picodet.py ==============================


class PicoDet(nn.Module):
    """
    Generalized Focal Loss network, see arxiv.org/abs/2006.04388

    Args:
        backbone(nn.Module): backbone cnn, namely Esnet
        neck(nn.Module): fnn cnn, namely csp-pan
        head(nn.Module): head cnn, namely picohead
    """

    def __init__(self, backbone, neck, head):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.export_nms = True
        self.export_post_process = False
        self.inputs = {}

    def _forward(self):
        backbone_feats = self.backbone(self.inputs)
        fpn_feats = self.neck(backbone_feats)
        cls_logits_list, bboxes_reg_list = self.head(fpn_feats, self.export_post_process)
        if self.training or not self.export_post_process:
            return cls_logits_list, bboxes_reg_list
        else:
            images, targets = self.inputs
            scale_factor = [t["scale_factor"] for t in targets]
            scale_factor = torch.stack(scale_factor, dim=0)
            bboxes, bbox_num = self.head.post_process(
                [cls_logits_list, bboxes_reg_list], scale_factor, export_nms=True
            )
            return bboxes, bbox_num

    def forward(self, inputs):
        self.inputs = inputs
        if self.training:
            loss = {}
            cls_logits_list, bboxes_reg_list = self._forward()
            loss_gfl = self.head.get_loss([cls_logits_list, bboxes_reg_list], self.inputs)
            loss.update(loss_gfl)
            total_loss = torch.stack(list(loss.values())).sum()
            loss.update({"loss": total_loss})
            return loss
        else:
            if not self.export_post_process:
                return {"picodet": self._forward()}
            else:
                bboxes, bbox_num = self._forward()
                output = {"bbox": bboxes, "bbox_num": bbox_num}
                return output


# ============================== staging build/entry points ==============================


class PicoDetTraceWrapper(nn.Module):
    """Thin single-tensor-input wrapper: real PicoDet.forward expects a
    ``(list_of_image_tensors, targets)`` tuple (PaddleDetection-style batched
    input contract). This wrapper reproduces that call exactly, with a
    single-image batch and export_post_process=False (deploy-eval feature-map
    output, no NMS/scale_factor bookkeeping needed), so the traced graph is
    the real PicoDet backbone+neck+head forward pass end to end.
    """

    def __init__(self, picodet: PicoDet):
        super().__init__()
        self.picodet = picodet
        self.picodet.eval()
        self.picodet.export_post_process = False

    def forward(self, x):
        out = self.picodet(([x[0]], [{}]))
        return out["picodet"]


def _create_picodet(num_classes=80):
    """Mirrors engine/model.py:create_model(), at a smaller ESNet scale for a fast trace."""
    backbone = ESNet(
        scale=0.5,
        channel_ratio=[0.875, 0.5, 0.5, 0.5, 0.625, 0.5, 0.625, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    )
    neck = CSPPAN(
        in_channels=backbone._out_channels,
        out_channels=32,
        use_depthwise=True,
        num_csp_blocks=1,
        num_features=4,
    )
    conv_feat = PicoFeat(
        feat_in=32,
        feat_out=32,
        num_convs=2,
        num_fpn_stride=4,
        norm_type="bn",
        share_cls_reg=True,
    )
    loss_class = VarifocalLoss(use_sigmoid=True, iou_weighted=True)
    loss_dfl = DistributionFocalLoss(loss_weight=0.25)
    loss_box = GIoULoss(loss_weight=2.0)
    assigner = SimOTAAssigner(candidate_topk=10, iou_weight=6, num_classes=num_classes)
    nms = MultiClassNMS(nms_top_k=1000, keep_top_k=100, score_threshold=0.0, nms_threshold=0.6)
    head = PicoHead(
        conv_feat=conv_feat,
        num_classes=num_classes,
        fpn_stride=[8, 16, 32, 64],
        prior_prob=0.01,
        loss_vfl=loss_class,
        loss_dfl=loss_dfl,
        loss_iou=loss_box,
        assigner=assigner,
        reg_max=7,
        feat_in_chan=32,
        nms=nms,
        cell_offset=0.5,
    )

    model = PicoDet(backbone, neck, head)
    return model


def build_picodet():
    return PicoDetTraceWrapper(_create_picodet(num_classes=4))


def example_input_picodet():
    return torch.randn(1, 3, 128, 128)


MENAGERIE_ENTRIES = [
    ("PicoDet", "build_picodet", "example_input_picodet", 2021, "SOURCE_AVAILABLE"),
]
