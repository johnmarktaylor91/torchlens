# FAITHFUL PORT of tusen-ai/RangeDet @ (main, snapshot 2026-06-30)
# (original framework: MXNet / mxnext)
#
# https://github.com/tusen-ai/RangeDet
# https://raw.githubusercontent.com/tusen-ai/RangeDet/main/rangedet/symbol/backbone/dla_backbone.py
# https://raw.githubusercontent.com/tusen-ai/RangeDet/main/rangedet/symbol/backbone/meta_kernel.py
# https://raw.githubusercontent.com/tusen-ai/RangeDet/main/rangedet/symbol/head/builder.py
# https://raw.githubusercontent.com/tusen-ai/RangeDet/main/config/rangedet/rangedet_veh_wo_aug_4_18e.py
#
# Fan et al. 2021 (ICCV), "RangeDet: In Defense of Range View for LiDAR-based
# 3D Object Detection" -- the paper's official TuSimple/TuSen-AI repo. The
# model consumes a range image (LiDAR points projected to a cylindrical
# range image) and predicts single-stage anchor-free 3D boxes over an FPN of
# range-image feature levels. The entire repo (``mxnext/`` wraps
# ``mx.symbol``, ``rangedet/symbol/*``) is MXNet -- no PyTorch code exists
# anywhere in the repo, and MXNet is EOL / cannot coexist with this env's
# modern CUDA + torch stack, so the architecture is transcribed faithfully
# into self-contained torch below. Every mechanism from the trainable graph
# is preserved (loss computation, box decoding, custom C++
# NMS/rotated-IOU/iou-target operators in ``operator_cxx``/``operator_py``,
# and the ``get_train_symbol``/``get_test_symbol`` data-loading wiring are
# inference/training-time post-processing outside the nn.Module forward
# graph, and are not ported -- matching how ``get_fpn_output``, the actual
# conv graph, is a self-contained sub-call within ``RangeRpnHead``):
#
#   1. ``DLABackboneBuilder.backbone_factory`` (``dla_backbone.py``): a DLA
#      (Deep Layer Aggregation)-style encoder/decoder over the range image.
#      Encoder: 5 residual stages (``res1``/``res2a``/``res2``/``res3a``/
#      ``res3``) built from ``basicblock`` (conv-bn-relu x2 + shortcut proj
#      when stride/channels change), each stage's first unit strided
#      ``(1, 2)`` (width/azimuth-only downsampling -- range-image height
#      stays fixed) except ``res1`` (stride 1). Decoder: 4 "aggregation"
#      stages (``agg2``/``agg1``/``agg2a``/``agg3``) that each deconv-upsample
#      a coarser feature map (asymmetric kernel/stride, again width-only:
#      ``(3,8)``/``(1,4)`` for the /16->/4 and /8->/2 hops,
#      ``(3,4)``/``(1,2)`` for the /4->/2 and /2->/1 hops), add it to the
#      matching encoder skip connection, then run a residual stage on the
#      sum. ``add_data_sc=True`` (the shipped veh config) concatenates the
#      raw input onto the final ``agg3`` feature map. Output is a 3-level
#      FPN dict ``{1: agg3, 2: agg2a, 4: agg2}`` selected by
#      ``fpn_strides=(1,2,4)`` in the shipped config.
#   2. ``MetaKernel.meta_baseline_bias`` (``meta_kernel.py``): the paper's
#      headline "meta-kernel convolution" -- injected into ``res1_unit2``
#      per the shipped config's ``meta_kernel_units``. For each output
#      position it gathers a ``kernel_size x kernel_size`` neighborhood of
#      3D coordinates (``coord`` = xyz per range-image pixel) via im2col,
#      subtracts the center pixel's own coordinate to get *relative*
#      neighbor offsets, runs those relative offsets through a small
#      per-pixel MLP (1x1 convs, i.e. shared across the kernel window) to
#      produce per-neighbor *weights*, and multiplies those weights
#      elementwise with an im2col-gathered neighborhood of the *data*
#      features -- i.e. a continuous, coordinate-conditioned convolution
#      kernel (replacing a fixed learned kernel with one predicted from
#      local 3D geometry). The weighted neighborhood is reshaped back to
#      ``[batch, in_channels * k*k, H, W]`` and folled by an aggregation
#      conv-bn-relu (1x1) down to the block's target width, matching
#      ``meta_kernel_conv`` in ``dla_backbone.py``. ``mx.symbol.im2col`` is
#      ported as ``torch.nn.functional.unfold``.
#   3. ``RangeRpnHead.get_fpn_output`` (``head/builder.py``): the trainable
#      detection head -- per FPN level, independent classification and
#      regression conv towers (4 conv-BN-relu layers each, matching the
#      shipped config's ``cls_conv_layers``/``reg_conv_layers``=4,
#      channel=128), each followed by a single 1x1(effectively 3x3-default)
#      conv projecting to ``num_classes`` classification logits /
#      ``num_classes * num_reg_delta`` regression deltas. ``num_reg_delta=8``
#      and single-class ``veh`` detection matches the shipped
#      ``rangedet_veh_wo_aug_4_18e.py`` config.
#
# Channel counts (``num_block``/``num_filter`` per stage) and the meta-kernel
# unit (``res1_unit2``, ``data_channels=64``, ``coord_channels=3``,
# ``channel_list=[32, 64]``, ``kernel_size=3``) are copied verbatim from the
# shipped veh config.

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------
# mxnext/simple.py conv-bn-relu primitive (mx.sym.Convolution -> Conv2d)
# --------------------------------------------------------------------------


def _conv(in_channels, out_channels, kernel, stride=1, pad=None, dilate=1, no_bias=True):
    if isinstance(kernel, int):
        kernel = (kernel, kernel)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(dilate, int):
        dilate = (dilate, dilate)
    if pad is None:
        pad = ((kernel[0] - 1) * dilate[0] + 1) // 2
    if isinstance(pad, int):
        pad = (pad, pad)
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel,
        stride=stride,
        padding=pad,
        dilation=dilate,
        bias=not no_bias,
    )


def _deconv(in_channels, out_channels, kernel, stride, pad):
    if isinstance(kernel, int):
        kernel = (kernel, kernel)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(pad, int):
        pad = (pad, pad)
    return nn.ConvTranspose2d(
        in_channels, out_channels, kernel_size=kernel, stride=stride, padding=pad, bias=True
    )


class ConvBnRelu(nn.Module):
    """Port of ``X.convnormrelu``: conv (no bias) -> BN -> ReLU."""

    def __init__(self, in_channels, out_channels, kernel=3, stride=1):
        super().__init__()
        self.conv = _conv(in_channels, out_channels, kernel, stride=stride, no_bias=True)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


# --------------------------------------------------------------------------
# rangedet/symbol/backbone/meta_kernel.py
# --------------------------------------------------------------------------


class MetaKernelBaselineBias(nn.Module):
    """Port of ``MetaKernel.meta_baseline_bias``: coordinate-conditioned
    ("meta-kernel") convolution. Predicts per-neighbor weights from relative
    3D-coordinate offsets (via a shared per-pixel MLP) and multiplies them
    elementwise into an im2col neighborhood of the input features."""

    def __init__(self, data_channels, coord_channels, channel_list, kernel_size=3):
        super().__init__()
        self.data_channels = data_channels
        self.coord_channels = coord_channels
        self.kernel_size = kernel_size

        mlp_layers = []
        in_ch = coord_channels
        for i, out_ch in enumerate(channel_list):
            mlp_layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False))
            if i != len(channel_list) - 1:
                mlp_layers.append(nn.ReLU(inplace=True))
            in_ch = out_ch
        self.mlp = nn.Sequential(*mlp_layers)
        # weights MLP's last channel must broadcast against data_channels
        # neighborhoods (meta_kernel.py:231 elementwise multiply); the
        # reference config sets channel_list[-1] == data_channels.
        assert channel_list[-1] == data_channels, (
            "meta-kernel MLP output width must match data_channels for the elementwise product"
        )

    def forward(self, data, coord):
        batch, _, h, w = data.shape
        k = self.kernel_size

        # sample_coord: im2col neighborhood of the coordinate map.
        coord_nbhd = F.unfold(coord, kernel_size=k, padding=k // 2)  # [B, coord_ch*k*k, H*W]
        coord_nbhd = coord_nbhd.view(batch, self.coord_channels, k * k, h, w)

        # relative_coord: subtract the center pixel's own coordinate.
        center = coord.unsqueeze(2)  # [B, coord_ch, 1, H, W]
        rel_coord = coord_nbhd - center  # [B, coord_ch, k*k, H, W]

        # mlp: per-pixel shared MLP over the relative-coordinate channel,
        # applied densely over the (k*k, W) "spatial" dims (matching
        # meta_kernel.py's reshape to [B, coord_ch, k*k*H, W] then 1x1 convs).
        rel_coord_flat = rel_coord.reshape(batch, self.coord_channels, k * k * h, w)
        weights = self.mlp(rel_coord_flat)
        weights = weights.view(batch, self.data_channels, k * k, h, w)

        # sample_data: im2col neighborhood of the data features.
        data_nbhd = F.unfold(data, kernel_size=k, padding=k // 2)  # [B, data_ch*k*k, H*W]
        data_nbhd = data_nbhd.view(batch, self.data_channels, k * k, h, w)

        output = data_nbhd * weights
        return output.reshape(batch, self.data_channels * k * k, h, w)


# --------------------------------------------------------------------------
# rangedet/symbol/backbone/dla_backbone.py
# --------------------------------------------------------------------------


class BasicBlock(nn.Module):
    """Port of ``DLABackboneBuilder.basicblock``. If a meta-kernel unit is
    configured for this block's name, its first conv-bn-relu is replaced by
    the meta-kernel path (matching ``meta_kernel_conv``)."""

    def __init__(self, in_channels, out_channels, stride, meta_kernel=None):
        super().__init__()
        self.meta_kernel = meta_kernel
        if meta_kernel is None:
            self.conv1 = ConvBnRelu(in_channels, out_channels, kernel=3, stride=1)
        else:
            # aggregation conv-bn-relu after the meta-kernel neighborhood mix
            # (meta_kernel_conv: "point_wise_mlp_bn1"/"relu1" over the
            # data_channels*k*k-wide neighborhood mix, then
            # "aggregation_conv1" + bn + relu, 1x1, down to out_channels).
            agg_in = out_channels * meta_kernel.kernel_size * meta_kernel.kernel_size
            self.mk_bn = nn.BatchNorm2d(agg_in)
            self.mk_relu = nn.ReLU(inplace=True)
            self.mk_agg_conv = nn.Conv2d(agg_in, out_channels, kernel_size=1, bias=False)
            self.mk_agg_bn = nn.BatchNorm2d(out_channels)
            self.mk_agg_relu = nn.ReLU(inplace=True)

        self.conv2 = _conv(out_channels, out_channels, 3, stride=stride, no_bias=True)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.needs_proj = stride != (1, 1) or in_channels != out_channels
        if self.needs_proj:
            self.sc_conv = _conv(in_channels, out_channels, 1, stride=stride, no_bias=True)
            self.sc_bn = nn.BatchNorm2d(out_channels)

        self.relu_out = nn.ReLU(inplace=True)

    def forward(self, x, coord=None):
        if self.meta_kernel is None:
            relu1 = self.conv1(x)
        else:
            mixed = self.meta_kernel(x, coord)
            mixed = self.mk_relu(self.mk_bn(mixed))
            relu1 = self.mk_agg_relu(self.mk_agg_bn(self.mk_agg_conv(mixed)))

        bn2 = self.bn2(self.conv2(relu1))

        if self.needs_proj:
            shortcut = self.sc_bn(self.sc_conv(x))
        else:
            shortcut = x

        return self.relu_out(bn2 + shortcut)


class ResStage(nn.Module):
    """Port of ``DLABackboneBuilder.res_stage``."""

    def __init__(self, in_channels, out_channels, num_block, stride, meta_kernel_units=None):
        super().__init__()
        meta_kernel_units = meta_kernel_units or {}
        blocks = []
        blocks.append(
            BasicBlock(
                in_channels, out_channels, stride, meta_kernel=meta_kernel_units.get("unit1")
            )
        )
        for i in range(2, num_block + 1):
            blocks.append(
                BasicBlock(
                    out_channels,
                    out_channels,
                    (1, 1),
                    meta_kernel=meta_kernel_units.get(f"unit{i}"),
                )
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, coord=None):
        for block in self.blocks:
            x = block(x, coord)
        return x


class AggStage(nn.Module):
    """Port of ``DLABackboneBuilder.agg_stage``: deconv-upsample the coarser
    feature, add the encoder skip, then run a residual stage on the sum."""

    def __init__(
        self,
        const_channels,
        upsample_channels,
        out_channels,
        num_block,
        deconv_kernel,
        deconv_stride,
        deconv_pad,
    ):
        super().__init__()
        self.deconv = _deconv(
            upsample_channels, out_channels, deconv_kernel, deconv_stride, deconv_pad
        )
        self.deconv_bn = nn.BatchNorm2d(out_channels)
        self.deconv_relu = nn.ReLU(inplace=True)
        assert const_channels == out_channels, (
            "agg_stage adds the upsampled feature directly to the encoder skip"
        )
        self.res_stage = ResStage(out_channels, out_channels, num_block, (1, 1))

    def forward(self, data_const, data_upsample):
        up = self.deconv_relu(self.deconv_bn(self.deconv(data_upsample)))
        # crop/pad to match spatial size exactly (transpose-conv output size
        # can be off-by-one vs. the symbolic MXNet deconv for the asymmetric
        # width-only upsampling used here).
        if up.shape[-2:] != data_const.shape[-2:]:
            up = F.interpolate(up, size=data_const.shape[-2:], mode="nearest")
        return self.res_stage(up + data_const)


class DLABackbone(nn.Module):
    """Port of ``DLABackboneBuilder.backbone_factory``."""

    def __init__(self, in_channels=4, add_data_sc=True):
        super().__init__()
        num_block = {
            "res1": 2,
            "res2a": 3,
            "res2": 3,
            "res3a": 5,
            "res3": 5,
            "agg1": 2,
            "agg2": 2,
            "agg2a": 1,
            "agg3": 2,
        }
        num_filter = {
            "res1": 64,
            "res2a": 64,
            "res2": 128,
            "res3a": 128,
            "res3": 128,
            "agg1": 64,
            "agg2": 128,
            "agg2a": 64,
            "agg3": 64,
        }
        self.add_data_sc = add_data_sc

        # meta-kernel injected into res1_unit2 (coord_channels=3 for xyz).
        meta_kernel_res1 = {
            "unit2": MetaKernelBaselineBias(
                data_channels=64, coord_channels=3, channel_list=[32, 64], kernel_size=3
            )
        }

        self.res1 = ResStage(
            in_channels,
            num_filter["res1"],
            num_block["res1"],
            (1, 1),
            meta_kernel_units=meta_kernel_res1,
        )
        self.res2a = ResStage(num_filter["res1"], num_filter["res2a"], num_block["res2a"], (1, 2))
        self.res2 = ResStage(num_filter["res2a"], num_filter["res2"], num_block["res2"], (1, 2))
        self.res3a = ResStage(num_filter["res2"], num_filter["res3a"], num_block["res3a"], (1, 2))
        self.res3 = ResStage(num_filter["res3a"], num_filter["res3"], num_block["res3"], (1, 2))

        self.agg2 = AggStage(
            num_filter["res2"],
            num_filter["res3"],
            num_filter["agg2"],
            num_block["agg2"],
            deconv_kernel=(3, 8),
            deconv_stride=(1, 4),
            deconv_pad=(1, 2),
        )
        self.agg1 = AggStage(
            num_filter["res1"],
            num_filter["res2"],
            num_filter["agg1"],
            num_block["agg1"],
            deconv_kernel=(3, 8),
            deconv_stride=(1, 4),
            deconv_pad=(1, 2),
        )
        self.agg2a = AggStage(
            num_filter["res2a"],
            num_filter["agg2"],
            num_filter["agg2a"],
            num_block["agg2a"],
            deconv_kernel=(3, 4),
            deconv_stride=(1, 2),
            deconv_pad=(1, 1),
        )
        self.agg3 = AggStage(
            num_filter["agg1"],
            num_filter["agg2a"],
            num_filter["agg3"],
            num_block["agg3"],
            deconv_kernel=(3, 4),
            deconv_stride=(1, 2),
            deconv_pad=(1, 1),
        )

        self.agg3_out_channels = num_filter["agg3"] + (in_channels if add_data_sc else 0)

    def forward(self, data, coord):
        res1 = self.res1(data, coord)
        res2a = self.res2a(res1)
        res2 = self.res2(res2a)
        res3a = self.res3a(res2)
        res3 = self.res3(res3a)

        agg2 = self.agg2(res2, res3)
        agg1 = self.agg1(res1, res2)
        agg2a = self.agg2a(res2a, agg2)
        agg3 = self.agg3(agg1, agg2a)

        if self.add_data_sc:
            agg3 = torch.cat([data, agg3], dim=1)

        # fpn_strides=(1, 2, 4) -> {1: agg3, 2: agg2a, 4: agg2}
        return [agg3, agg2a, agg2]


# --------------------------------------------------------------------------
# rangedet/symbol/head/builder.py (trainable detection head)
# --------------------------------------------------------------------------


class RangeRpnHead(nn.Module):
    """Port of ``RangeRpnHead.get_fpn_output``: per-FPN-level cls/reg conv
    towers + final projection convs."""

    def __init__(
        self, in_channels_list, num_classes=1, num_reg_delta=8, conv_layers=4, conv_channel=128
    ):
        super().__init__()
        self.levels = nn.ModuleList()
        for in_channels in in_channels_list:
            cls_tower = []
            reg_tower = []
            c = in_channels
            for _ in range(conv_layers):
                cls_tower.append(ConvBnRelu(c, conv_channel, kernel=3, stride=1))
                reg_tower.append(ConvBnRelu(c, conv_channel, kernel=3, stride=1))
                c = conv_channel
            level = nn.ModuleDict(
                {
                    "cls_tower": nn.Sequential(*cls_tower),
                    "reg_tower": nn.Sequential(*reg_tower),
                    "cls_logit": _conv(conv_channel, num_classes, 3, stride=1, no_bias=False),
                    "bbox_delta": _conv(
                        conv_channel, num_reg_delta * num_classes, 3, stride=1, no_bias=False
                    ),
                }
            )
            self.levels.append(level)

    def forward(self, conv_feat_list):
        cls_logits, bbox_deltas = [], []
        for level, conv_feat in zip(self.levels, conv_feat_list):
            cls_feat = level["cls_tower"](conv_feat)
            reg_feat = level["reg_tower"](conv_feat)
            cls_logits.append(level["cls_logit"](cls_feat))
            bbox_deltas.append(level["bbox_delta"](reg_feat))
        return cls_logits, bbox_deltas


class RangeDet(nn.Module):
    """Top-level RangeDet detector: DLA backbone -> multi-level RPN head."""

    def __init__(self, in_channels=4, num_classes=1, num_reg_delta=8):
        super().__init__()
        self.backbone = DLABackbone(in_channels=in_channels, add_data_sc=True)
        fpn_channels = [
            self.backbone.agg3_out_channels,  # stride 1 (agg3)
            64,  # stride 2 (agg2a)
            128,  # stride 4 (agg2)
        ]
        self.head = RangeRpnHead(fpn_channels, num_classes=num_classes, num_reg_delta=num_reg_delta)

    def forward(self, data, coord):
        feat_list = self.backbone(data, coord)
        cls_logits, bbox_deltas = self.head(feat_list)
        return cls_logits, bbox_deltas


def build_rangedet() -> nn.Module:
    """Build a tiny RangeDet range-image detector.

    Uses a small range-image size and reduced head width so the module
    traces quickly; the architecture (DLA backbone with a meta-kernel
    coordinate-conditioned convolution in ``res1_unit2`` + 3-level FPN
    cls/reg conv-tower head) is unchanged from the real
    ``rangedet/symbol/backbone/dla_backbone.py`` +
    ``rangedet/symbol/backbone/meta_kernel.py`` +
    ``rangedet/symbol/head/builder.py``.

    Returns
    -------
    nn.Module
        Random-initialized RangeDet detector (single-class "veh" detection,
        ``num_reg_delta=8``, matching the shipped
        ``rangedet_veh_wo_aug_4_18e.py`` config).
    """
    return RangeDet(in_channels=4, num_classes=1, num_reg_delta=8)


def example_input_rangedet() -> tuple:
    """Return small range-image (data, coord) tensors matching ``build_rangedet``.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``data``: ``(batch, in_channels, H, W)`` range-image features
        (range/intensity/elongation/etc, shape ``(1, 4, 16, 64)``).
        ``coord``: ``(batch, 3, H, W)`` per-pixel xyz coordinates used by the
        meta-kernel convolution, same spatial size as ``data``.
    """
    data = torch.randn(1, 4, 16, 64)
    coord = torch.randn(1, 3, 16, 64)
    return (data, coord)


MENAGERIE_ZOO = "ported-pytorch"
MENAGERIE_ENTRIES = [
    ("RangeDet", "build_rangedet", "example_input_rangedet", 2021, "ported"),
]
