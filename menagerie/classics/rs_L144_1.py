# FAITHFUL PORT of TRI-ML/dd3d @ main (original framework: detectron2)
#
# Source files transcribed (forward/prediction path only; losses/postprocessing
# that require detectron2 Instances/Boxes structures are intentionally omitted,
# since TorchLens traces the network's tensor computation, not detection
# postprocessing plumbing):
#   tridet/modeling/dd3d/fcos2d.py  (FCOS2DHead)
#   tridet/modeling/dd3d/fcos3d.py  (FCOS3DHead, predictions_to_boxes3d)
#   tridet/layers/normalization.py  (Scale, Offset, ModuleListDial)
#
# DD3D ("Is Pseudo-Lidar needed for Monocular 3D Object Detection?", ICCV 2021,
# Park et al., TRI-ML) predicts disentangled 2D+3D boxes directly from a single
# image via two parallel dense FCOS-style conv towers over an FPN pyramid: a
# 2D box/classification/centerness head (FCOS2DHead) and a 3D box head
# (FCOS3DHead) predicting allocentric quaternion, projected center, depth,
# size, and confidence per anchor location. This module combines both heads
# (DD3DHead) exactly as they are wired together inside tridet/modeling/dd3d/core.py's
# DD3D.forward, running on synthetic FPN-shaped feature maps rather than a
# full detectron2 ResNet+FPN backbone (which needs the detectron2 dependency
# we don't have installed).
#
# detectron2.layers.Conv2d/get_norm (norm+activation-fused Conv2d) are
# replaced 1:1 with the plain-torch equivalent (nn.Conv2d followed by a norm
# layer and an activation module) -- functionally identical, no architecture
# change.

import math

import torch
from torch import nn
import torch.nn.functional as F

EPS = 1e-7


# ---------------------------------------------------------------------------
# tridet/layers/normalization.py
# ---------------------------------------------------------------------------
class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


class Offset(nn.Module):
    def __init__(self, init_value=0.0):
        super().__init__()
        self.bias = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input + self.bias


class ModuleListDial(nn.ModuleList):
    """Each FPN level dials to its own submodule (its own BN statistics)."""

    def __init__(self, modules=None):
        super().__init__(modules)
        self.cur_position = 0

    def forward(self, x):
        result = self[self.cur_position](x)
        self.cur_position += 1
        if self.cur_position >= len(self):
            self.cur_position = 0
        return result


class ConvNormAct(nn.Module):
    """Plain-torch stand-in for detectron2.layers.Conv2d(..., norm=, activation=)."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=True,
        norm=None,
        activation=None,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.norm = norm
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


def get_norm(norm, out_channels, num_levels):
    if not norm:
        return None
    if norm == "BN":
        return ModuleListDial([nn.BatchNorm2d(out_channels) for _ in range(num_levels)])
    if norm == "GN":
        return nn.GroupNorm(32, out_channels)
    raise ValueError(f"Unsupported norm: {norm}")


# ---------------------------------------------------------------------------
# tridet/modeling/dd3d/fcos2d.py :: FCOS2DHead
# ---------------------------------------------------------------------------
class FCOS2DHead(nn.Module):
    def __init__(
        self,
        in_channels,
        num_levels,
        num_classes=10,
        num_cls_convs=4,
        num_box_convs=4,
        norm="BN",
        use_scale=True,
        box2d_scale_init_factor=1.0,
        in_strides=(8, 16, 32, 64, 128),
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_levels = num_levels
        self.use_scale = use_scale
        self.in_strides = in_strides

        head_configs = {"cls": num_cls_convs, "box2d": num_box_convs}
        for head_name, num_convs in head_configs.items():
            tower = []
            for _ in range(num_convs):
                norm_layer = get_norm(norm, in_channels, num_levels)
                tower.append(
                    ConvNormAct(
                        in_channels,
                        in_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=norm_layer is None,
                        norm=norm_layer,
                        activation=nn.ReLU(),
                    )
                )
            self.add_module(f"{head_name}_tower", nn.Sequential(*tower))

        self.cls_logits = nn.Conv2d(in_channels, num_classes, kernel_size=3, stride=1, padding=1)
        self.box2d_reg = nn.Conv2d(in_channels, 4, kernel_size=3, stride=1, padding=1)
        self.centerness = nn.Conv2d(in_channels, 1, kernel_size=3, stride=1, padding=1)

        if self.use_scale:
            self.scales_box2d_reg = nn.ModuleList(
                [Scale(init_value=stride * box2d_scale_init_factor) for stride in in_strides]
            )

        self.init_weights()

    def init_weights(self):
        for tower in [self.cls_tower, self.box2d_tower]:
            for layer_mod in tower.modules():
                if isinstance(layer_mod, nn.Conv2d):
                    nn.init.kaiming_normal_(layer_mod.weight, mode="fan_out", nonlinearity="relu")
                    if layer_mod.bias is not None:
                        nn.init.constant_(layer_mod.bias, 0)
        for modules in [self.cls_logits, self.box2d_reg, self.centerness]:
            for layer_mod in modules.modules():
                if isinstance(layer_mod, nn.Conv2d):
                    nn.init.kaiming_uniform_(layer_mod.weight, a=1)
                    if layer_mod.bias is not None:
                        nn.init.constant_(layer_mod.bias, 0)

    def forward(self, x):
        logits, box2d_reg, centerness = [], [], []
        cls_tower_out = []
        for lvl, feature in enumerate(x):
            cls_out = self.cls_tower(feature)
            bbox_out = self.box2d_tower(feature)

            logits.append(self.cls_logits(cls_out))
            centerness.append(self.centerness(bbox_out))
            box_reg = self.box2d_reg(bbox_out)
            if self.use_scale:
                box_reg = self.scales_box2d_reg[lvl](box_reg)
            box2d_reg.append(F.relu(box_reg))
            cls_tower_out.append(cls_out)
        return logits, box2d_reg, centerness, cls_tower_out


# ---------------------------------------------------------------------------
# tridet/modeling/dd3d/fcos3d.py :: FCOS3DHead + predictions_to_boxes3d
# ---------------------------------------------------------------------------
class FCOS3DHead(nn.Module):
    def __init__(
        self,
        in_channels,
        num_levels,
        num_classes=10,
        num_convs=4,
        norm="BN",
        use_scale=True,
        depth_scale_init_factor=0.3,
        proj_ctr_scale_init_factor=1.0,
        per_level_predictors=False,
        mean_depth_per_level=(28.0, 32.0, 41.0, 46.0, 51.0),
        std_depth_per_level=(15.0, 15.0, 16.0, 16.0, 16.0),
        in_strides=(8, 16, 32, 64, 128),
        class_agnostic_box3d=True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_levels = num_levels
        self.use_scale = use_scale
        self.use_per_level_predictors = per_level_predictors
        self.in_strides = in_strides

        self.register_buffer(
            "mean_depth_per_level", torch.tensor(mean_depth_per_level, dtype=torch.float32)
        )
        self.register_buffer(
            "std_depth_per_level", torch.tensor(std_depth_per_level, dtype=torch.float32)
        )

        tower = []
        for _ in range(num_convs):
            norm_layer = get_norm(norm, in_channels, num_levels)
            tower.append(
                ConvNormAct(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=norm_layer is None,
                    norm=norm_layer,
                    activation=nn.ReLU(),
                )
            )
        self.add_module("box3d_tower", nn.Sequential(*tower))

        num_classes_out = num_classes if not class_agnostic_box3d else 1
        num_levels_out = num_levels if per_level_predictors else 1

        self.box3d_quat = nn.ModuleList(
            [
                nn.Conv2d(in_channels, 4 * num_classes_out, 3, 1, 1, bias=True)
                for _ in range(num_levels_out)
            ]
        )
        self.box3d_ctr = nn.ModuleList(
            [
                nn.Conv2d(in_channels, 2 * num_classes_out, 3, 1, 1, bias=True)
                for _ in range(num_levels_out)
            ]
        )
        self.box3d_depth = nn.ModuleList(
            [
                nn.Conv2d(in_channels, 1 * num_classes_out, 3, 1, 1, bias=(not use_scale))
                for _ in range(num_levels_out)
            ]
        )
        self.box3d_size = nn.ModuleList(
            [
                nn.Conv2d(in_channels, 3 * num_classes_out, 3, 1, 1, bias=True)
                for _ in range(num_levels_out)
            ]
        )
        self.box3d_conf = nn.ModuleList(
            [
                nn.Conv2d(in_channels, 1 * num_classes_out, 3, 1, 1, bias=True)
                for _ in range(num_levels_out)
            ]
        )

        if self.use_scale:
            self.scales_proj_ctr = nn.ModuleList(
                [Scale(init_value=stride * proj_ctr_scale_init_factor) for stride in in_strides]
            )
            self.scales_size = nn.ModuleList([Scale(init_value=1.0) for _ in range(num_levels)])
            self.scales_conf = nn.ModuleList([Scale(init_value=1.0) for _ in range(num_levels)])
            self.scales_depth = nn.ModuleList(
                [
                    Scale(init_value=float(sigma) * depth_scale_init_factor)
                    for sigma in std_depth_per_level
                ]
            )
            self.offsets_depth = nn.ModuleList(
                [Offset(init_value=float(b)) for b in mean_depth_per_level]
            )

        self._init_weights()

    def _init_weights(self):
        for layer_mod in self.box3d_tower.modules():
            if isinstance(layer_mod, nn.Conv2d):
                nn.init.kaiming_normal_(layer_mod.weight, mode="fan_out", nonlinearity="relu")
                if layer_mod.bias is not None:
                    nn.init.constant_(layer_mod.bias, 0)
        for modules in [
            self.box3d_quat,
            self.box3d_ctr,
            self.box3d_depth,
            self.box3d_size,
            self.box3d_conf,
        ]:
            for layer_mod in modules.modules():
                if isinstance(layer_mod, nn.Conv2d):
                    nn.init.kaiming_uniform_(layer_mod.weight, a=1)
                    if layer_mod.bias is not None:
                        nn.init.constant_(layer_mod.bias, 0)

    def forward(self, x):
        box3d_quat, box3d_ctr, box3d_depth, box3d_size, box3d_conf = [], [], [], [], []
        for lvl, features in enumerate(x):
            tower_out = self.box3d_tower(features)
            _l = lvl if self.use_per_level_predictors else 0

            quat = self.box3d_quat[_l](tower_out)
            proj_ctr = self.box3d_ctr[_l](tower_out)
            depth = self.box3d_depth[_l](tower_out)
            size3d = self.box3d_size[_l](tower_out)
            conf3d = self.box3d_conf[_l](tower_out)

            if self.use_scale:
                proj_ctr = self.scales_proj_ctr[lvl](proj_ctr)
                size3d = self.scales_size[lvl](size3d)
                conf3d = self.scales_conf[lvl](conf3d)
                depth = self.offsets_depth[lvl](self.scales_depth[lvl](depth))

            box3d_quat.append(quat)
            box3d_ctr.append(proj_ctr)
            box3d_depth.append(depth)
            box3d_size.append(size3d)
            box3d_conf.append(conf3d)
        return box3d_quat, box3d_ctr, box3d_depth, box3d_size, box3d_conf


# ---------------------------------------------------------------------------
# Combined DD3D head, matching how DD3D.forward wires FCOS2DHead + FCOS3DHead
# together (tridet/modeling/dd3d/core.py), operating on a feature pyramid.
# ---------------------------------------------------------------------------
class DD3DHead(nn.Module):
    def __init__(
        self, in_channels=64, num_levels=5, num_classes=10, in_strides=(8, 16, 32, 64, 128)
    ):
        super().__init__()
        self.fcos2d_head = FCOS2DHead(
            in_channels, num_levels, num_classes=num_classes, in_strides=in_strides
        )
        self.fcos3d_head = FCOS3DHead(
            in_channels, num_levels, num_classes=num_classes, in_strides=in_strides
        )

    def forward(self, features):
        logits, box2d_reg, centerness, _ = self.fcos2d_head(features)
        box3d_quat, box3d_ctr, box3d_depth, box3d_size, box3d_conf = self.fcos3d_head(features)
        return {
            "logits": logits,
            "box2d_reg": box2d_reg,
            "centerness": centerness,
            "box3d_quat": box3d_quat,
            "box3d_ctr": box3d_ctr,
            "box3d_depth": box3d_depth,
            "box3d_size": box3d_size,
            "box3d_conf": box3d_conf,
        }


def build_dd3d_head():
    return DD3DHead(in_channels=32, num_levels=5, num_classes=10, in_strides=(8, 16, 32, 64, 128))


def example_input_dd3d_head():
    # Synthetic FPN pyramid: 5 levels, strides 8/16/32/64/128 for a 256x256 image.
    base = 32
    sizes = [(32, 32), (16, 16), (8, 8), (4, 4), (2, 2)]
    return ([torch.randn(2, base, h, w) for h, w in sizes],)


MENAGERIE_ZOO = "ported-pytorch"
MENAGERIE_ENTRIES = [
    ("DD3D-FCOS-Head", build_dd3d_head, example_input_dd3d_head, 2021, "vision"),
]
