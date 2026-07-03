# SOURCE: vendored from mit-han-lab/litepose @ main
# https://raw.githubusercontent.com/mit-han-lab/litepose/main/lib/models/pose_mobilenet.py
# https://raw.githubusercontent.com/mit-han-lab/litepose/main/lib/models/layers/layers.py
# https://raw.githubusercontent.com/mit-han-lab/litepose/main/arch_manager.py
# https://raw.githubusercontent.com/mit-han-lab/litepose/main/experiments/coco/mobilenet/mobile.yaml
#
# Wang, Zhang, Cai, Han 2022 (CVPR) "LitePose: Efficient Architecture Design for 2D Human
# Pose Estimation" -- single-branch MobileNetV2-style inverted-residual backbone (no
# multi-resolution HRNet branches) topped with a "Fusion Deconv Head": at each deconv stage
# a high-level ("refined") feature and a skip-connected low-level ("raw") feature are each
# upsampled via their own ConvTranspose2d and summed, and the final heatmap/tag predictions
# similarly fuse a "refined" and "raw" SepConv2d head. LitePose (the fixed post-arch-search
# deployment model, as opposed to SuperLitePose used during neural-architecture-search
# training) and its InvBottleneck/convbnrelu/SepConv2d building blocks are copied verbatim
# from lib/models/pose_mobilenet.py and lib/models/layers/layers.py. ArchManager.fixed_sample
# (copied verbatim from arch_manager.py) is used to generate the real `cfg_arch` backbone
# spec (search-space schema, not tuned weights) that LitePose's constructor consumes, exactly
# as gen_arch.py does; a plain attribute-namespace `_CN` stands in for the real repo's `yacs`
# CfgNode (only ever attribute-accessed by this model, so behaviorally identical) so the
# staging module doesn't add a non-base dependency, with values taken from the real
# experiments/coco/mobilenet/mobile.yaml training config (MODEL.EXTRA.NUM_DECONV_LAYERS=3,
# NUM_DECONV_FILTERS=[64,48,32], NUM_DECONV_KERNELS=[4,4,4], NUM_JOINTS=17,
# TAG_PER_JOINT=True, LOSS.WITH_HEATMAPS_LOSS=[True,True], LOSS.WITH_AE_LOSS=[True,False]).
"""LitePose: single-branch MobileNetV2 backbone + Fusion (refined+raw) Deconv Head."""

import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


# --- vendored from lib/models/layers/layers.py ---
class convbnrelu(nn.Sequential):
    def __init__(self, inp, oup, ker=3, stride=1, groups=1):
        super(convbnrelu, self).__init__(
            nn.Conv2d(inp, oup, ker, stride, ker // 2, groups=groups, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=True),
        )


class InvBottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, ker=3, exp=6):
        super(InvBottleneck, self).__init__()
        feature_dim = _make_divisible(round(inplanes * exp), 8)
        self.inv = nn.Sequential(
            nn.Conv2d(inplanes, feature_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU6(inplace=True),
        )
        self.depth_conv = nn.Sequential(
            nn.Conv2d(
                feature_dim, feature_dim, ker, stride, ker // 2, groups=feature_dim, bias=False
            ),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU6(inplace=True),
        )
        self.point_conv = nn.Sequential(
            nn.Conv2d(feature_dim, planes, 1, 1, 0, bias=False), nn.BatchNorm2d(planes)
        )
        self.stride = stride
        self.use_residual_connection = stride == 1 and inplanes == planes

    def forward(self, x):
        out = self.inv(x)
        out = self.depth_conv(out)
        out = self.point_conv(out)
        if self.use_residual_connection:
            out += x
        return out


class SepConv2d(nn.Module):
    def __init__(self, inp, oup, ker=3, stride=1):
        super(SepConv2d, self).__init__()
        conv = [
            nn.Conv2d(inp, inp, ker, stride, ker // 2, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU(inplace=True),
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        ]
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        output = self.conv(x)
        return output


# --- vendored from lib/models/pose_mobilenet.py ---
class LitePose(nn.Module):
    def __init__(self, cfg, width_mult=1.0, round_nearest=8, cfg_arch=None):
        super(LitePose, self).__init__()
        backbone_setting = cfg_arch["backbone_setting"]
        input_channel = cfg_arch["input_channel"]
        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.first = nn.Sequential(
            convbnrelu(3, 32, ker=3, stride=2),
            convbnrelu(32, 32, ker=3, stride=1, groups=32),
            nn.Conv2d(32, input_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(input_channel),
        )
        self.channel = [input_channel]
        # building inverted residual blocks
        self.stage = []
        for id_stage in range(len(backbone_setting)):
            n = backbone_setting[id_stage]["num_blocks"]
            s = backbone_setting[id_stage]["stride"]
            c = backbone_setting[id_stage]["channel"]
            c = _make_divisible(c * width_mult, round_nearest)
            block_setting = backbone_setting[id_stage]["block_setting"]
            layer = []
            for id_block in range(n):
                t, k = block_setting[id_block]
                stride = s if id_block == 0 else 1
                layer.append(InvBottleneck(input_channel, c, stride, ker=k, exp=t))
                input_channel = c
            layer = nn.Sequential(*layer)
            self.stage.append(layer)
            self.channel.append(c)
        self.stage = nn.ModuleList(self.stage)
        extra = cfg.MODEL.EXTRA
        self.filters = cfg_arch["deconv_setting"]
        self.inplanes = self.channel[-1]
        self.deconv_refined, self.deconv_raw, self.deconv_bnrelu = self._make_deconv_layers(
            extra.NUM_DECONV_LAYERS,
            self.filters,
            extra.NUM_DECONV_KERNELS,
        )
        self.final_refined, self.final_raw, self.final_channel = self._make_final_layers(
            cfg, self.filters
        )
        self.num_deconv_layers = extra.NUM_DECONV_LAYERS
        self.loss_config = cfg.LOSS

    def _get_deconv_cfg(self, deconv_kernel):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_final_layers(self, cfg, num_filters):
        dim_tag = cfg.MODEL.NUM_JOINTS if cfg.MODEL.TAG_PER_JOINT else 1
        extra = cfg.MODEL.EXTRA
        final_raw = []
        final_refined = []
        final_channel = []
        for i in range(1, extra.NUM_DECONV_LAYERS):
            oup_joint = cfg.MODEL.NUM_JOINTS if cfg.LOSS.WITH_HEATMAPS_LOSS[i - 1] else 0
            oup_tag = dim_tag if cfg.LOSS.WITH_AE_LOSS[i - 1] else 0
            final_refined.append(SepConv2d(num_filters[i], oup_joint + oup_tag, ker=5))
            final_raw.append(SepConv2d(self.channel[-i - 3], oup_joint + oup_tag, ker=5))
            final_channel.append(oup_joint + oup_tag)

        return nn.ModuleList(final_refined), nn.ModuleList(final_raw), final_channel

    def _make_deconv_layers(self, num_layers, num_filters, num_kernels):
        deconv_refined = []
        deconv_raw = []
        deconv_bnrelu = []
        for i in range(num_layers):
            kernel, padding, output_padding = self._get_deconv_cfg(num_kernels[i])
            planes = num_filters[i]
            layers = []
            deconv_refined.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False,
                )
            )
            deconv_raw.append(
                nn.ConvTranspose2d(
                    in_channels=self.channel[-i - 2],
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False,
                )
            )
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes
            deconv_bnrelu.append(nn.Sequential(*layers))

        return (
            nn.ModuleList(deconv_refined),
            nn.ModuleList(deconv_raw),
            nn.ModuleList(deconv_bnrelu),
        )

    def forward(self, x):
        x = self.first(x)
        x_list = [x]
        for i in range(len(self.stage)):
            tmp = self.stage[i](x_list[-1])
            x_list.append(tmp)
        final_outputs = []
        input_refined = x_list[-1]
        input_raw = x_list[-2]
        for i in range(self.num_deconv_layers):
            next_input_refined = self.deconv_refined[i](input_refined)
            next_input_raw = self.deconv_raw[i](input_raw)
            input_refined = self.deconv_bnrelu[i](next_input_refined + next_input_raw)
            input_raw = x_list[-i - 3]
            if i > 0:
                final_refined = self.final_refined[i - 1](input_refined)
                final_raw = self.final_raw[i - 1](input_raw)
                final_outputs.append(final_refined + final_raw)

        return final_outputs


# --- vendored from arch_manager.py (search-space schema generator; no tunable weights) ---
class ArchManager:
    def __init__(self, cfg):
        self.cfg = cfg
        self.expansion = [6]
        self.kernel_size = [7]
        self.input_channel = 24
        self.width_mult = [1.0, 0.75, 0.5, 0.25]
        self.deconv_setting = cfg.MODEL.EXTRA.NUM_DECONV_FILTERS
        self.is_search = False
        self.search_arch = None
        self.arch_setting = [
            # c, n, s
            [32, 4, 2],
            [64, 6, 2],
            [96, 8, 2],
            [160, 8, 1],
        ]

    def fixed_sample(self, reso=256, ratio=0.5):
        cfg_arch = {}
        cfg_arch["img_size"] = reso
        cfg_arch["input_channel"] = _make_divisible(self.input_channel * ratio, 8)
        cfg_arch["deconv_setting"] = []
        for i in range(len(self.deconv_setting)):
            cfg_arch["deconv_setting"].append(_make_divisible(self.deconv_setting[i] * ratio, 8))
        cfg_arch["backbone_setting"] = []
        for i in range(len(self.arch_setting)):
            stage = {}
            c, n, s = self.arch_setting[i]
            stage["num_blocks"] = n
            stage["stride"] = s
            stage["channel"] = _make_divisible(c * ratio, 8)
            stage["block_setting"] = []
            for j in range(stage["num_blocks"]):
                stage["block_setting"].append([6, 7])
            cfg_arch["backbone_setting"].append(stage)
        return cfg_arch


# Plain attribute-namespace standing in for the real repo's `yacs` CfgNode (only ever
# attribute-accessed by LitePose/ArchManager, so behaviorally identical for this purpose).
class _CN:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def _build_cfg():
    # Values taken verbatim from experiments/coco/mobilenet/mobile.yaml (the real LitePose
    # COCO training config).
    return _CN(
        MODEL=_CN(
            NUM_JOINTS=17,
            TAG_PER_JOINT=True,
            EXTRA=_CN(
                FINAL_CONV_KERNEL=1,
                NUM_DECONV_LAYERS=3,
                NUM_DECONV_FILTERS=[64, 48, 32],
                NUM_DECONV_KERNELS=[4, 4, 4],
            ),
        ),
        LOSS=_CN(
            WITH_HEATMAPS_LOSS=[True, True],
            WITH_AE_LOSS=[True, False],
        ),
    )


def build_litepose():
    cfg = _build_cfg()
    # Small ratio + reduced resolution (vs. the real reso=256, ratio=0.5) keeps the trace
    # fast; the arch-search schema and every layer/mechanism are unchanged.
    cfg_arch = ArchManager(cfg).fixed_sample(reso=128, ratio=0.25)
    return LitePose(cfg, cfg_arch=cfg_arch)


def example_input_litepose():
    torch.manual_seed(0)
    return (torch.randn(1, 3, 128, 128),)


MENAGERIE_ENTRIES = [
    ("LitePose", "build_litepose", "example_input_litepose", 2022, "vendored"),
]
