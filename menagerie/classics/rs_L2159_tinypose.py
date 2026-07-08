# FAITHFUL PORT of PaddlePaddle/PaddleDetection @ release/2.6 (original framework: PaddlePaddle)
#
# PP-TinyPose: a lightweight top-down keypoint-detection model shipped in
# PaddleDetection (https://github.com/PaddlePaddle/PaddleDetection). No PyTorch port
# exists anywhere (confirmed via GitHub code search). PaddlePaddle is a different deep
# learning framework entirely (not a missing pip package we can install into the torch
# base env) -- so the real code cannot be vendored/run as-is (RUNG 2 fails). This module
# faithfully transcribes the ACTUAL nn.Layer graph from the real source files into
# self-contained base-env torch:
#   - configs/keypoint/tiny_pose/tinypose_128x96.yml: architecture=TopDownHRNet,
#     backbone=LiteHRNet(network_type='wider_naive', freeze_at=-1, freeze_norm=false,
#     return_idx=[0]), width=40, num_joints=17.
#   - ppdet/modeling/architectures/keypoint_hrnet.py::TopDownHRNet: backbone(x) then a
#     single 1x1 Conv2d(width, num_joints) "final_conv" producing the heatmaps (the
#     flip-test-time-augmentation and post-processing/DARK-decoding paths are
#     inference-time non-differentiable bookkeeping and are intentionally omitted, same
#     as the other detector/pose ports in this pass).
#   - ppdet/modeling/backbones/lite_hrnet.py::LiteHRNet (itself a port of
#     HRNet/Lite-HRNet, "based on
#     https://github.com/HRNet/Lite-HRNet/blob/hrnet/models/backbones/litehrnet.py" per
#     that file's own docstring): Stem -> 3 HRNet-style multi-resolution stages (each
#     stage = a channel-count `transition` from the previous stage's branch list, then
#     `num_modules` x `LiteHRNetModule`) -> `IterativeHead` fusion -> return_idx=[0]
#     (the highest-resolution branch only, matching return_idx in the config).
#     `module_type='NAIVE'` for `network_type='wider_naive'` (per module_configs table)
#     means each `LiteHRNetModule` runs `ShuffleUnit` blocks per branch (NOT the LITE
#     variant's `ConditionalChannelWeightingBlock` -- that class exists in the real file
#     but is architecturally dead code for this particular network_type, so it is
#     omitted here exactly as the source's `if self.module_type == 'NAIVE':` branch
#     would skip it at runtime) followed by the standard HRNet multi-scale fuse_layers
#     (1x1-conv+BN+upsample for coarser->finer, strided depthwise+pointwise convs for
#     finer->coarser).
#   - "wider_naive" config: num_modules=[2,4,2], num_branches=[2,3,4], num_blocks=[2,2,2],
#     module_type=[NAIVE,NAIVE,NAIVE], num_channels=[[40,80],[40,80,160],[40,80,160,320]]
#     -- copied verbatim from LiteHRNet.module_configs["wider_naive"].
#   - `channel_shuffle` (ppdet/modeling/ops.py) is the standard ShuffleNet channel
#     shuffle (reshape -> transpose -> reshape), transcribed identically.
#
# Trained weights (tinypose_128x96.pdparams) are not used; this module constructs the
# architecture at random init for tracing.

import torch
import torch.nn as nn
import torch.nn.functional as F


def channel_shuffle(x, groups):
    """ppdet/modeling/ops.py channel_shuffle(), faithful transcription."""
    b, c, h, w = x.shape
    channels_per_group = c // groups
    x = x.reshape(b, groups, channels_per_group, h, w)
    x = x.transpose(1, 2).contiguous()
    x = x.reshape(b, c, h, w)
    return x


# ---------------------------------------------------------------------------
# ConvNormLayer (lite_hrnet.py ConvNormLayer): conv -> BN -> optional act.
# norm_type is always 'bn' for every call site used by network_type='wider_naive'.
# ---------------------------------------------------------------------------
class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, filter_size, stride=1, groups=1, act=None, use_norm=True):
        super().__init__()
        self.act = act
        self.conv = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            bias=not use_norm,
        )
        self.norm = nn.BatchNorm2d(ch_out) if use_norm else None

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act == "relu":
            x = F.relu(x)
        elif self.act == "sigmoid":
            x = torch.sigmoid(x)
        return x


class DepthWiseSeparableConvNormLayer(nn.Module):
    """lite_hrnet.py DepthWiseSeparableConvNormLayer."""

    def __init__(self, ch_in, ch_out, filter_size, dw_act=None, pw_act=None):
        super().__init__()
        self.depthwise_conv = ConvNormLayer(
            ch_in, ch_in, filter_size, stride=1, groups=ch_in, act=dw_act
        )
        self.pointwise_conv = ConvNormLayer(ch_in, ch_out, 1, stride=1, act=pw_act)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x


class ShuffleUnit(nn.Module):
    """lite_hrnet.py ShuffleUnit (used for module_type='NAIVE' branches)."""

    def __init__(self, in_channel, out_channel, stride):
        super().__init__()
        branch_channel = out_channel // 2
        self.stride = stride
        if stride > 1:
            self.branch1 = nn.Sequential(
                ConvNormLayer(in_channel, in_channel, 3, stride=stride, groups=in_channel),
                ConvNormLayer(in_channel, branch_channel, 1, stride=1, act="relu"),
            )
        else:
            self.branch1 = None
        self.branch2 = nn.Sequential(
            ConvNormLayer(
                branch_channel if stride == 1 else in_channel,
                branch_channel,
                1,
                stride=1,
                act="relu",
            ),
            ConvNormLayer(branch_channel, branch_channel, 3, stride=stride, groups=branch_channel),
            ConvNormLayer(branch_channel, branch_channel, 1, stride=1, act="relu"),
        )

    def forward(self, x):
        if self.stride > 1:
            x1 = self.branch1(x)
            x2 = self.branch2(x)
        else:
            x1, x2 = x.chunk(2, dim=1)
            x2 = self.branch2(x2)
        out = torch.cat([x1, x2], dim=1)
        out = channel_shuffle(out, groups=2)
        return out


class IterativeHead(nn.Module):
    """lite_hrnet.py IterativeHead: coarse-to-fine iterative feature fusion applied
    to the final multi-resolution branch list."""

    def __init__(self, in_channels):
        super().__init__()
        num_branches = len(in_channels)
        self.in_channels = in_channels[::-1]
        projects = []
        for i in range(num_branches):
            if i != num_branches - 1:
                projects.append(
                    DepthWiseSeparableConvNormLayer(
                        self.in_channels[i], self.in_channels[i + 1], 3, dw_act=None, pw_act="relu"
                    )
                )
            else:
                projects.append(
                    DepthWiseSeparableConvNormLayer(
                        self.in_channels[i], self.in_channels[i], 3, dw_act=None, pw_act="relu"
                    )
                )
        self.projects = nn.ModuleList(projects)

    def forward(self, x):
        x = x[::-1]
        y = []
        last_x = None
        for i, s in enumerate(x):
            if last_x is not None:
                last_x = F.interpolate(
                    last_x, size=s.shape[-2:], mode="bilinear", align_corners=True
                )
                s = s + last_x
            s = self.projects[i](s)
            y.append(s)
            last_x = s
        return y[::-1]


class Stem(nn.Module):
    """lite_hrnet.py Stem."""

    def __init__(self, in_channel, stem_channel, out_channel, expand_ratio):
        super().__init__()
        self.conv1 = ConvNormLayer(in_channel, stem_channel, 3, stride=2, act="relu")
        mid_channel = int(round(stem_channel * expand_ratio))
        branch_channel = stem_channel // 2
        if stem_channel == out_channel:
            inc_channel = out_channel - branch_channel
        else:
            inc_channel = out_channel - stem_channel
        self.branch1 = nn.Sequential(
            ConvNormLayer(branch_channel, branch_channel, 3, stride=2, groups=branch_channel),
            ConvNormLayer(branch_channel, inc_channel, 1, stride=1, act="relu"),
        )
        self.expand_conv = ConvNormLayer(branch_channel, mid_channel, 1, stride=1, act="relu")
        self.depthwise_conv = ConvNormLayer(
            mid_channel, mid_channel, 3, stride=2, groups=mid_channel
        )
        self.linear_conv = ConvNormLayer(
            mid_channel,
            branch_channel if stem_channel == out_channel else stem_channel,
            1,
            stride=1,
            act="relu",
        )

    def forward(self, x):
        x = self.conv1(x)
        x1, x2 = x.chunk(2, dim=1)
        x1 = self.branch1(x1)
        x2 = self.expand_conv(x2)
        x2 = self.depthwise_conv(x2)
        x2 = self.linear_conv(x2)
        out = torch.cat([x1, x2], dim=1)
        out = channel_shuffle(out, groups=2)
        return out


class LiteHRNetModule(nn.Module):
    """lite_hrnet.py LiteHRNetModule, module_type='NAIVE' path only (the only path
    exercised by network_type='wider_naive')."""

    def __init__(
        self, num_branches, num_blocks, in_channels, multiscale_output=False, with_fuse=True
    ):
        super().__init__()
        self.num_branches = num_branches
        self.in_channels = in_channels
        self.multiscale_output = multiscale_output
        self.with_fuse = with_fuse

        branches = []
        for branch_idx in range(num_branches):
            layers = []
            for _ in range(num_blocks):
                layers.append(
                    ShuffleUnit(in_channels[branch_idx], in_channels[branch_idx], stride=1)
                )
            branches.append(nn.Sequential(*layers))
        self.layers = nn.ModuleList(branches)

        if self.with_fuse:
            self.fuse_layers = self._make_fuse_layers()
            self.relu = nn.ReLU()

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None
        fuse_layers = []
        num_out_branches = self.num_branches if self.multiscale_output else 1
        for i in range(num_out_branches):
            fuse_layer = []
            for j in range(self.num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(
                                self.in_channels[j],
                                self.in_channels[i],
                                kernel_size=1,
                                stride=1,
                                bias=False,
                            ),
                            nn.BatchNorm2d(self.in_channels[i]),
                            nn.Upsample(scale_factor=2 ** (j - i), mode="nearest"),
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv_downsamples = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            conv_downsamples.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        self.in_channels[j],
                                        self.in_channels[j],
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        groups=self.in_channels[j],
                                        bias=False,
                                    ),
                                    nn.BatchNorm2d(self.in_channels[j]),
                                    nn.Conv2d(
                                        self.in_channels[j],
                                        self.in_channels[i],
                                        kernel_size=1,
                                        stride=1,
                                        bias=False,
                                    ),
                                    nn.BatchNorm2d(self.in_channels[i]),
                                )
                            )
                        else:
                            conv_downsamples.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        self.in_channels[j],
                                        self.in_channels[j],
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        groups=self.in_channels[j],
                                        bias=False,
                                    ),
                                    nn.BatchNorm2d(self.in_channels[j]),
                                    nn.Conv2d(
                                        self.in_channels[j],
                                        self.in_channels[j],
                                        kernel_size=1,
                                        stride=1,
                                        bias=False,
                                    ),
                                    nn.BatchNorm2d(self.in_channels[j]),
                                    nn.ReLU(),
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv_downsamples))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)

    def forward(self, x):
        if self.num_branches == 1:
            return [self.layers[0](x[0])]
        for i in range(self.num_branches):
            x[i] = self.layers[i](x[i])
        out = x
        if self.with_fuse:
            out_fuse = []
            for i in range(len(self.fuse_layers)):
                y = out[0] if i == 0 else self.fuse_layers[i][0](out[0])
                for j in range(self.num_branches):
                    if j == 0:
                        y = y + y
                    elif i == j:
                        y = y + out[j]
                    else:
                        y = y + self.fuse_layers[i][j](out[j])
                    if i == 0:
                        out[i] = y
                out_fuse.append(self.relu(y))
            out = out_fuse
        elif not self.multiscale_output:
            out = [out[0]]
        return out


# ---------------------------------------------------------------------------
# LiteHRNet backbone (network_type='wider_naive', return_idx=[0], per
# configs/keypoint/tiny_pose/tinypose_128x96.yml).
# ---------------------------------------------------------------------------
class LiteHRNet(nn.Module):
    def __init__(self, return_idx=(0,)):
        super().__init__()
        self.return_idx = list(return_idx)

        # module_configs["wider_naive"], copied verbatim.
        self.stages_config = {
            "num_modules": [2, 4, 2],
            "num_branches": [2, 3, 4],
            "num_blocks": [2, 2, 2],
            "num_channels": [[40, 80], [40, 80, 160], [40, 80, 160, 320]],
        }

        self.stem = Stem(3, 32, 32, 1)
        num_channels_pre_layer = [32]
        for stage_idx in range(3):
            num_channels = self.stages_config["num_channels"][stage_idx]
            setattr(
                self,
                f"transition{stage_idx}",
                self._make_transition_layer(num_channels_pre_layer, num_channels),
            )
            stage, num_channels_pre_layer = self._make_stage(stage_idx, num_channels, True)
            setattr(self, f"stage{stage_idx}", stage)
        self.head_layer = IterativeHead(num_channels_pre_layer)

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_pre = len(num_channels_pre_layer)
        num_branches_cur = len(num_channels_cur_layer)
        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_channels_pre_layer[i],
                                num_channels_pre_layer[i],
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                groups=num_channels_pre_layer[i],
                                bias=False,
                            ),
                            nn.BatchNorm2d(num_channels_pre_layer[i]),
                            nn.Conv2d(
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                kernel_size=1,
                                stride=1,
                                bias=False,
                            ),
                            nn.BatchNorm2d(num_channels_cur_layer[i]),
                            nn.ReLU(),
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                conv_downsamples = []
                for j in range(i + 1 - num_branches_pre):
                    conv_downsamples.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_channels_pre_layer[-1],
                                num_channels_pre_layer[-1],
                                groups=num_channels_pre_layer[-1],
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                bias=False,
                            ),
                            nn.BatchNorm2d(num_channels_pre_layer[-1]),
                            nn.Conv2d(
                                num_channels_pre_layer[-1],
                                num_channels_cur_layer[i]
                                if j == i - num_branches_pre
                                else num_channels_pre_layer[-1],
                                kernel_size=1,
                                stride=1,
                                bias=False,
                            ),
                            nn.BatchNorm2d(
                                num_channels_cur_layer[i]
                                if j == i - num_branches_pre
                                else num_channels_pre_layer[-1]
                            ),
                            nn.ReLU(),
                        )
                    )
                transition_layers.append(nn.Sequential(*conv_downsamples))
        return nn.ModuleList(transition_layers)

    def _make_stage(self, stage_idx, in_channels, multiscale_output):
        num_modules = self.stages_config["num_modules"][stage_idx]
        num_branches = self.stages_config["num_branches"][stage_idx]
        num_blocks = self.stages_config["num_blocks"][stage_idx]

        modules = []
        for i in range(num_modules):
            reset_multiscale_output = not (not multiscale_output and i == num_modules - 1)
            module = LiteHRNetModule(
                num_branches,
                num_blocks,
                in_channels,
                multiscale_output=reset_multiscale_output,
                with_fuse=True,
            )
            modules.append(module)
            in_channels = module.in_channels
        return nn.Sequential(*modules), in_channels

    def forward(self, x):
        x = self.stem(x)
        y_list = [x]
        for stage_idx in range(3):
            x_list = []
            transition = getattr(self, f"transition{stage_idx}")
            for j in range(self.stages_config["num_branches"][stage_idx]):
                if transition[j] is not None:
                    if j >= len(y_list):
                        x_list.append(transition[j](y_list[-1]))
                    else:
                        x_list.append(transition[j](y_list[j]))
                else:
                    x_list.append(y_list[j])
            y_list = getattr(self, f"stage{stage_idx}")(x_list)
        x = self.head_layer(y_list)
        res = [layer for i, layer in enumerate(x) if i in self.return_idx]
        return res


# ---------------------------------------------------------------------------
# TopDownHRNet (keypoint_hrnet.py TopDownHRNet, forward-only inference path):
# backbone(x) -> 1x1 conv to num_joints heatmap channels.
# ---------------------------------------------------------------------------
class TopDownHRNet(nn.Module):
    def __init__(self, width=40, num_joints=17):
        super().__init__()
        self.backbone = LiteHRNet(return_idx=(0,))
        self.final_conv = nn.Conv2d(
            width, num_joints, kernel_size=1, stride=1, padding=0, bias=True
        )

    def forward(self, x):
        feats = self.backbone(x)
        heatmaps = self.final_conv(feats[0])
        return heatmaps


def build_tinypose():
    model = TopDownHRNet(width=40, num_joints=17)
    model.eval()
    return model


def example_input_tinypose():
    torch.manual_seed(0)
    return torch.randn(1, 3, 128, 96)


MENAGERIE_ZOO = "ported-pytorch"

MENAGERIE_ENTRIES = [
    ("PP-TinyPose", "build_tinypose", "example_input_tinypose", 2021, "PORT"),
]
