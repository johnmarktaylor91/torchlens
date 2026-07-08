# FAITHFUL PORT of Xianpeng919/MonoCon @ main (original framework: PyTorch +
# mmcv/mmdet/mmdet3d OpenMMLab registry ecosystem)
#
# https://github.com/Xianpeng919/MonoCon
# https://raw.githubusercontent.com/Xianpeng919/MonoCon/main/monocon/mmdet3d/models/backbones/dla.py
# https://raw.githubusercontent.com/Xianpeng919/MonoCon/main/monocon/mmdet3d/models/necks/dlaup.py
# https://raw.githubusercontent.com/Xianpeng919/MonoCon/main/monocon/mmdet3d/models/dense_heads/monocon_head.py
# https://raw.githubusercontent.com/Xianpeng919/MonoCon/main/monocon/mmdet3d/ops/attentive_norm.py
# https://raw.githubusercontent.com/Xianpeng919/MonoCon/main/monocon/configs/_base_/models/monocon_dla34.py
#
# MonoCon (Liu, Xue, Wu, Zhang, Cao, Xu, Bao. AAAI 2022, "Learning Auxiliary Monocular
# Contexts Helps Monocular 3D Object Detection"). CenterNet-style monocular 3D detector:
# a DLA-34 backbone, a DLAUp feature-fusion neck, and MonoConHead -- a multi-branch
# CenterNet head (heatmap/wh/offset/keypoint-heatmap/keypoint-offset/dim/depth/alpha
# direction-classification+regression) that additionally uses Attentive Normalization
# (AttnBatchNorm2d, https://arxiv.org/abs/1908.01259) instead of plain BatchNorm in the
# head's conv towers when `use_AN=True` (the paper's own default / "auxiliary context"
# framing). Every conv/BN/tree-recursion/IDAUp-fusion/attentive-norm layer below is
# transcribed verbatim from the real repo files above.
#
# Why a PORT and not a vendor: the real repo classes (`DLA`, `DLAUp`, `MonoConHead`,
# `CenterNetMono3D`) are registered into and constructed through the OpenMMLab
# mmcv/mmdet/mmdet3d registry+config-build system (`@BACKBONES.register_module()`,
# `@NECKS.register_module()`, `@HEADS.register_module()`, `@DETECTORS.register_module()`,
# `SingleStageDetector` base class) and use `mmcv.cnn.build_norm_layer(norm_cfg, ...)` as
# a config-driven normalization-layer dispatcher. Neither mmcv nor mmdet/mmdet3d is
# installed in base env (not base libs; the vendored 2.11.0-era mmdetection tree the repo
# ships is itself a full framework, not something to `pip install`), so the classes cannot
# be literally imported -- hence PORT, not vendor.
#
# What is transcribed 1:1, and what is inlined:
#   - `DLA`/`Tree`/`BasicBlock`/`Bottleneck`/`Root` (DLA-34 backbone): every conv/bn/relu,
#     the full recursive Tree aggregation topology, residual-root handling -- unchanged
#     from `mmdet3d/models/backbones/dla.py`. The only glue removed is the
#     `@BACKBONES.register_module()` decorator (dead weight at inference) and
#     `mmcv.cnn.build_norm_layer(norm_cfg, planes, postfix=1)[1]` calls, which for this
#     model's only used `norm_cfg=dict(type='BN')` are a config-dispatcher for exactly
#     `nn.BatchNorm2d(planes)` -- inlined literally, not an architectural change.
#   - `DLAUp`/`IDAUp`/`Conv2d` (feature-fusion neck): unchanged from
#     `mmdet3d/models/necks/dlaup.py`, same `build_norm_layer(BN)` -> `nn.BatchNorm2d`
#     inlining, `@NECKS.register_module()` dropped.
#   - `AttnWeights`/`AttnBatchNorm2d`/`HSigmoidv2` (Attentive Normalization): unchanged
#     from `mmdet3d/ops/attentive_norm.py`, including the RSD (regional statistics
#     descriptor) attention-weight computation and the per-sample affine mixture. Only the
#     final `NORM_LAYERS.register_module(...)` mmcv-registry side-effect lines (irrelevant
#     without the registry) and the `from mmcv.cnn import NORM_LAYERS` import are dropped;
#     `kaiming_init`/`constant_init` (originally re-exported from mmcv.cnn but defined
#     locally in the source file) are copied verbatim, `trunc_normal_` comes from timm
#     (already a base lib, same call the source file makes).
#   - `MonoConHead._build_head`/`_build_dir_head`/`_get_norm_layer`/`forward_single`
#     (the head's actual forward computation): unchanged from
#     `mmdet3d/models/dense_heads/monocon_head.py`. Only the loss/`get_bboxes`/`get_targets`
#     training-and-postprocessing methods are dropped (irrelevant to a forward pass; they
#     depend on mmdet's `multi_apply`/`force_fp32`/`build_loss`/gaussian-target utilities,
#     none of which touch the architecture), and `@HEADS.register_module()` is dropped.
#     `forward()` (`multi_apply(self.forward_single, feats)`) is kept, using a local
#     `multi_apply` reimplementation identical to mmdet's own (`tuple(map(list, zip(*map(
#     func, *args))))`) since that utility is pure Python plumbing, not architecture.
#   - `CenterNetMono3D` (the detector wrapper) is reproduced as a plain nn.Module
#     (`backbone -> neck -> bbox_head`) matching `SingleStageDetector.extract_feat` +
#     `simple_test`'s `outs = self.bbox_head(x)` call, dropping the mmdet3d BaseDetector
#     scaffolding (dataset/`img_metas`/postprocessing-only code) that is orthogonal to the
#     network's forward computation.
#
# Config used (from `monocon/configs/_base_/models/monocon_dla34.py`): DLA depth=34,
# DLAUp in_channels_list=[64,128,256,512] scales_list=(1,2,4,8) start_level=2,
# MonoConHead in_channel=64 feat_channel=64 num_classes=3 num_alpha_bins=12.

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

MENAGERIE_ZOO = "ported-pytorch"


# ---------------------------------------------------------------------------
# mmcv.cnn.build_norm_layer(dict(type='BN'), planes) inlined literally as
# nn.BatchNorm2d(planes) -- the only norm_cfg this model ever uses.
# ---------------------------------------------------------------------------
def _build_bn(planes: int) -> nn.BatchNorm2d:
    return nn.BatchNorm2d(planes)


# ---------------------------------------------------------------------------
# From mmdet3d/models/backbones/dla.py (DLA-34 backbone)
# ---------------------------------------------------------------------------
class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1, norm_cfg=None):
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
        self.bn1 = _build_bn(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=dilation, bias=False, dilation=dilation
        )
        self.bn2 = _build_bn(planes)
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
    def __init__(self, in_channels, out_channels, kernel_size, residual, norm_cfg=None):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 1, stride=1, bias=False, padding=(kernel_size - 1) // 2
        )
        self.bn = _build_bn(out_channels)
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
        norm_cfg=None,
    ):
        super().__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(
                in_channels, out_channels, stride, dilation=dilation, norm_cfg=norm_cfg
            )
            self.tree2 = block(out_channels, out_channels, 1, dilation=dilation, norm_cfg=norm_cfg)
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
                norm_cfg=norm_cfg,
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
                norm_cfg=norm_cfg,
            )
        if levels == 1:
            self.root = Root(
                root_dim, out_channels, root_kernel_size, root_residual, norm_cfg=norm_cfg
            )
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels

        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            norm = _build_bn(out_channels)
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                norm,
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
    arch_settings = {
        34: (BasicBlock, (1, 1, 1, 2, 2, 1), (16, 32, 64, 128, 256, 512), False),
    }

    def __init__(self, depth=34, in_channels=3, norm_cfg=None, norm_eval=False):
        super().__init__()
        block, levels, channels, residual_root = self.arch_settings[depth]
        self.channels = channels
        self.norm_eval = norm_eval
        norm1 = _build_bn(channels[0])
        self.base_layer = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], kernel_size=7, stride=1, padding=3, bias=False),
            norm1,
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

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        norm = _build_bn(planes)
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
                    norm,
                    nn.ReLU(inplace=True),
                ]
            )
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x):
        y = []
        x = self.base_layer(x)
        for i in range(6):
            x = getattr(self, f"level{i}")(x)
            y.append(x)
        return tuple(y)


# ---------------------------------------------------------------------------
# From mmdet3d/models/necks/dlaup.py (DLAUp feature-fusion neck)
# ---------------------------------------------------------------------------
class NeckConv2d(nn.Module):
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
        self.norm1 = _build_bn(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm1(x)
        x = self.relu(x)
        return x


def _fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for cc in range(1, w.size(0)):
        w[cc, 0, :, :] = w[0, 0, :, :]


class IDAUp(nn.Module):
    def __init__(self, in_channels_list, up_factors_list, out_channels):
        super().__init__()
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels

        for i in range(1, len(in_channels_list)):
            in_channels = in_channels_list[i]
            up_factors = int(up_factors_list[i])

            proj = NeckConv2d(in_channels, out_channels, kernal_szie=3, stride=1, bias=False)
            node = NeckConv2d(out_channels * 2, out_channels, kernal_szie=3, stride=1, bias=False)
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
            _fill_up_weights(up)

            setattr(self, f"proj_{i}", proj)
            setattr(self, f"up_{i}", up)
            setattr(self, f"node_{i}", node)

    def forward(self, layers):
        assert len(self.in_channels_list) == len(layers)
        for i in range(1, len(layers)):
            upsample = getattr(self, f"up_{i}")
            project = getattr(self, f"proj_{i}")
            node = getattr(self, f"node_{i}")
            layers[i] = upsample(project(layers[i]))
            layers[i] = node(torch.cat([layers[i - 1], layers[i]], 1))
        return layers


class DLAUp(nn.Module):
    def __init__(
        self, in_channels_list=(64, 128, 256, 512), scales_list=(1, 2, 4, 8), start_level=2
    ):
        super().__init__()
        in_channels_list = list(in_channels_list)
        scales_list = list(scales_list)
        self.in_channels_list = in_channels_list
        self.start_level = start_level

        for i in range(len(in_channels_list) - 1):
            j = -i - 2
            setattr(
                self,
                f"ida_{i}",
                IDAUp(
                    in_channels_list=in_channels_list[j:],
                    up_factors_list=[s // scales_list[j] for s in scales_list[j:]],
                    out_channels=in_channels_list[j],
                ),
            )
            scales_list[j + 1 :] = [scales_list[j] for _ in scales_list[j + 1 :]]
            in_channels_list[j + 1 :] = [in_channels_list[j] for _ in in_channels_list[j + 1 :]]

    def forward(self, layers):
        layers = list(layers[self.start_level :])
        assert len(layers) > 1
        for i in range(len(layers) - 1):
            ida = getattr(self, f"ida_{i}")
            layers[-i - 2 :] = ida(layers[-i - 2 :])
        return [layers[-1]]


# ---------------------------------------------------------------------------
# From mmdet3d/ops/attentive_norm.py (Attentive Normalization)
# ---------------------------------------------------------------------------
class HSigmoidv2(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3.0, inplace=self.inplace) / 6.0


def _constant_init(module, val, bias=0):
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def _kaiming_init(module, a=0, mode="fan_out", nonlinearity="relu", bias=0, distribution="normal"):
    if hasattr(module, "weight") and module.weight is not None:
        if distribution == "uniform":
            nn.init.kaiming_uniform_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_normal_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class AttnWeights(nn.Module):
    def __init__(
        self,
        attn_mode,
        num_features,
        num_affine_trans,
        num_groups=1,
        use_rsd=True,
        use_maxpool=False,
        use_bn=True,
        eps=1e-3,
    ):
        super().__init__()
        if use_rsd:
            use_maxpool = False

        self.num_affine_trans = num_affine_trans
        self.use_rsd = use_rsd
        self.use_maxpool = use_maxpool
        self.eps = eps
        if not self.use_rsd:
            self.avgpool = nn.AdaptiveAvgPool2d(1)

        if attn_mode == 0:
            layers = [
                nn.Conv2d(num_features, num_affine_trans, 1, bias=not use_bn),
                nn.BatchNorm2d(num_affine_trans) if use_bn else nn.Identity(),
                HSigmoidv2(),
            ]
        elif attn_mode == 1:
            if num_groups > 0:
                layers = [
                    nn.Conv2d(num_features, num_affine_trans, 1, bias=False),
                    nn.GroupNorm(num_channels=num_affine_trans, num_groups=num_groups),
                    HSigmoidv2(),
                ]
            else:
                layers = [
                    nn.Conv2d(num_features, num_affine_trans, 1, bias=False),
                    nn.BatchNorm2d(num_affine_trans) if use_bn else nn.Identity(),
                    HSigmoidv2(),
                ]
        else:
            raise NotImplementedError("Unknown attention weight type")

        self.attention = nn.Sequential(*layers)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                _kaiming_init(m)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                _constant_init(m, 1)

    def forward(self, x):
        b, c, h, w = x.size()
        if self.use_rsd:
            var, mean = torch.var_mean(x, dim=(2, 3), keepdim=True)
            y = mean * (var + self.eps).rsqrt()
        else:
            y = self.avgpool(x)
            if self.use_maxpool:
                y = y + F.max_pool2d(x, (h, w), stride=(h, w)).view(b, c, 1, 1)
        return self.attention(y).view(b, self.num_affine_trans)


class AttnBatchNorm2d(nn.BatchNorm2d):
    def __init__(
        self,
        num_features,
        num_affine_trans,
        attn_mode=0,
        eps=1e-5,
        momentum=0.1,
        track_running_stats=True,
        use_rsd=True,
        use_maxpool=False,
        use_bn=True,
        eps_var=1e-3,
    ):
        super().__init__(
            num_features,
            affine=False,
            eps=eps,
            momentum=momentum,
            track_running_stats=track_running_stats,
        )

        self.num_affine_trans = num_affine_trans
        self.attn_mode = attn_mode
        self.use_rsd = use_rsd
        self.eps_var = eps_var

        self.weight_ = nn.Parameter(torch.Tensor(num_affine_trans, num_features))
        self.bias_ = nn.Parameter(torch.Tensor(num_affine_trans, num_features))

        self.attn_weights = AttnWeights(
            attn_mode,
            num_features,
            num_affine_trans,
            use_rsd=use_rsd,
            use_maxpool=use_maxpool,
            use_bn=use_bn,
            eps=eps_var,
        )
        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.weight_, 1.0, 0.1)
        nn.init.normal_(self.bias_, 0.0, 0.1)

    def forward(self, x):
        output = super().forward(x)
        size = output.size()
        y = self.attn_weights(x)

        weight = y @ self.weight_
        bias = y @ self.bias_
        weight = weight.unsqueeze(-1).unsqueeze(-1).expand(size)
        bias = bias.unsqueeze(-1).unsqueeze(-1).expand(size)

        return weight * output + bias


# ---------------------------------------------------------------------------
# From mmdet3d/models/dense_heads/monocon_head.py (MonoConHead)
# ---------------------------------------------------------------------------
EPS = 1e-12


def _multi_apply(func, *args, **kwargs):
    # mmdet.core.multi_apply, pure Python plumbing (no architecture): applies `func` to
    # each set of positional args and transposes the per-call output tuples into
    # per-output-slot lists.
    pfunc = func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


class MonoConHead(nn.Module):
    def __init__(
        self,
        in_channel,
        feat_channel,
        num_classes,
        bbox3d_code_size=7,
        num_kpt=9,
        num_alpha_bins=12,
        pred_bbox2d=True,
        use_AN=True,
        num_AN_affine=10,
    ):
        super().__init__()
        assert bbox3d_code_size >= 7
        self.num_classes = num_classes
        self.bbox_code_size = bbox3d_code_size
        self.pred_bbox2d = pred_bbox2d
        self.num_kpt = num_kpt
        self.num_alpha_bins = num_alpha_bins

        self.use_AN = use_AN
        self.num_AN_affine = num_AN_affine
        self.norm = AttnBatchNorm2d if use_AN else nn.BatchNorm2d

        self.heatmap_head = self._build_head(in_channel, feat_channel, num_classes)
        self.wh_head = self._build_head(in_channel, feat_channel, 2)
        self.offset_head = self._build_head(in_channel, feat_channel, 2)
        self.center2kpt_offset_head = self._build_head(in_channel, feat_channel, self.num_kpt * 2)
        self.kpt_heatmap_head = self._build_head(in_channel, feat_channel, self.num_kpt)
        self.kpt_heatmap_offset_head = self._build_head(in_channel, feat_channel, 2)
        self.dim_head = self._build_head(in_channel, feat_channel, 3)
        self.depth_head = self._build_head(in_channel, feat_channel, 2)
        self._build_dir_head(in_channel, feat_channel)

    def _build_head(self, in_channel, feat_channel, out_channel):
        return nn.Sequential(
            nn.Conv2d(in_channel, feat_channel, kernel_size=3, padding=1),
            self._get_norm_layer(feat_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channel, out_channel, kernel_size=1),
        )

    def _build_dir_head(self, in_channel, feat_channel):
        self.dir_feat = nn.Sequential(
            nn.Conv2d(in_channel, feat_channel, kernel_size=3, padding=1),
            self._get_norm_layer(feat_channel),
            nn.ReLU(inplace=True),
        )
        self.dir_cls = nn.Sequential(nn.Conv2d(feat_channel, self.num_alpha_bins, kernel_size=1))
        self.dir_reg = nn.Sequential(nn.Conv2d(feat_channel, self.num_alpha_bins, kernel_size=1))

    def _get_norm_layer(self, feat_channel):
        return (
            self.norm(feat_channel, momentum=0.03, eps=0.001)
            if not self.use_AN
            else self.norm(feat_channel, self.num_AN_affine, momentum=0.03, eps=0.001)
        )

    def forward(self, feats):
        return _multi_apply(self.forward_single, feats)

    def forward_single(self, feat):
        center_heatmap_pred = self.heatmap_head(feat).sigmoid()
        center_heatmap_pred = torch.clamp(center_heatmap_pred, min=1e-4, max=1 - 1e-4)
        kpt_heatmap_pred = self.kpt_heatmap_head(feat).sigmoid()
        kpt_heatmap_pred = torch.clamp(kpt_heatmap_pred, min=1e-4, max=1 - 1e-4)

        offset_pred = self.offset_head(feat)
        kpt_heatmap_offset_pred = self.kpt_heatmap_offset_head(feat)

        wh_pred = self.wh_head(feat)
        center2kpt_offset_pred = self.center2kpt_offset_head(feat)
        dim_pred = self.dim_head(feat)
        depth_pred = self.depth_head(feat)
        depth_pred = torch.cat(
            [
                1.0 / (depth_pred[:, 0:1, :, :].sigmoid() + EPS) - 1,
                depth_pred[:, 1:2, :, :],
            ],
            dim=1,
        )

        alpha_feat = self.dir_feat(feat)
        alpha_cls_pred = self.dir_cls(alpha_feat)
        alpha_offset_pred = self.dir_reg(alpha_feat)
        return (
            center_heatmap_pred,
            wh_pred,
            offset_pred,
            center2kpt_offset_pred,
            kpt_heatmap_pred,
            kpt_heatmap_offset_pred,
            dim_pred,
            alpha_cls_pred,
            alpha_offset_pred,
            depth_pred,
        )


# ---------------------------------------------------------------------------
# From mmdet3d/models/detectors/mono_centernet3d.py (CenterNetMono3D), reduced to the
# plain nn.Module forward path (backbone -> neck -> bbox_head), matching
# SingleStageDetector.extract_feat + simple_test's `outs = self.bbox_head(x)`.
# ---------------------------------------------------------------------------
class CenterNetMono3D(nn.Module):
    def __init__(self, backbone: nn.Module, neck: nn.Module, bbox_head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.bbox_head = bbox_head

    def extract_feat(self, img):
        x = self.backbone(img)
        x = self.neck(x)
        return x

    def forward(self, img):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs


def build_monocon_dla34(num_classes: int = 3) -> CenterNetMono3D:
    backbone = DLA(depth=34)
    neck = DLAUp(in_channels_list=(64, 128, 256, 512), scales_list=(1, 2, 4, 8), start_level=2)
    bbox_head = MonoConHead(
        in_channel=64, feat_channel=64, num_classes=num_classes, num_alpha_bins=12
    )
    # .eval(): AttnBatchNorm2d's internal BatchNorm2d (a real architectural component,
    # see AttnWeights above) requires >1 sample per channel in training mode; eval mode
    # is also how this detector is actually run at inference time (batch=1 image).
    return CenterNetMono3D(backbone, neck, bbox_head).eval()


def build_monocon():
    return build_monocon_dla34()


def example_input_monocon():
    return torch.randn(1, 3, 96, 320)


MENAGERIE_ENTRIES = [
    ("MonoCon", "build_monocon", "example_input_monocon", 2022, "ported-pytorch"),
]
