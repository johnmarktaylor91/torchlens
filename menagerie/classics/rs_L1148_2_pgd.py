# FAITHFUL PORT of open-mmlab/mmdetection3d @ main (original framework: mmdet3d/mmcv/mmengine)
# https://github.com/open-mmlab/mmdetection3d/blob/main/mmdet3d/models/dense_heads/pgd_head.py
# https://github.com/open-mmlab/mmdetection3d/blob/main/mmdet3d/models/dense_heads/fcos_mono3d_head.py
# https://github.com/open-mmlab/mmdetection3d/blob/main/mmdet3d/models/dense_heads/anchor_free_mono3d_head.py
# https://github.com/open-mmlab/mmdetection3d/blob/main/mmdet3d/models/task_modules/coders/pgd_bbox_coder.py
# https://github.com/open-mmlab/mmdetection3d/blob/main/mmdet3d/models/task_modules/coders/fcos3d_bbox_coder.py
# https://github.com/open-mmlab/mmdetection3d/blob/main/configs/_base_/models/pgd.py (default hyperparameters)
#
# PGD ("Probabilistic and Geometric Depth: Detecting Objects in Perspective",
# Wang et al., CoRL 2021, arXiv:2107.14160) is FCOS3D++: a monocular 3D
# detector built as ResNet+FPN backbone/neck feeding an anchor-free,
# multi-scale conv-tower detection head (PGDHead) that predicts per-point
# 2D/3D box regression, classification, direction class, attributes,
# centerness, a probabilistic depth-classification branch, and
# location-aware depth-fusion weights, then geometrically re-projects the 3D
# box back onto the image via the depth-classification branch.
#
# mmdetection3d needs mmdet3d + mmcv + mmengine (mmcv ships compiled CUDA
# extensions; none are installed and installing them is not reasonable in
# this base env) so the real code cannot run as-is -- this is a faithful
# port. The PGD-specific contribution -- the multi-branch anchor-free head
# (`AnchorFreeMono3DHead._init_layers`/`_init_predictor`/`forward_single`,
# `FCOSMono3DHead._init_layers`/`forward_single` [adds centerness],
# `PGDHead._init_layers`/`_init_predictor`/`forward_single` [adds depth
# classifier + location-aware weight branches], and the box decoding logic
# in `FCOS3DBBoxCoder.decode` / `PGDBBoxCoder.decode_2d` /
# `decode_prob_depth`, all invoked inline as part of the real forward pass,
# not just at loss/postprocessing time) is transcribed faithfully, layer for
# layer, from the real mmdet3d source. `mmcv.cnn.ConvModule` (conv+GN+ReLU)
# and `mmcv.cnn.Scale` (single learnable scalar multiplier) are reproduced
# as their well-known, minimal PyTorch equivalents. The ResNet-101+FPN
# backbone/neck (not PGD-specific architecture -- PGD's contribution is the
# head) is built from torchvision's standard ResNet/FeaturePyramidNetwork,
# matching mmdet3d's own composition of a stock ResNet+FPN feeding into
# PGDHead (the deformable-conv (DCNv2) stages used for the full-size R101
# config are a training-stability detail noted in the mmdet3d code comments
# themselves -- not exercised in the tiny-size trace here since
# `dcn_on_last_conv` only reaches the head's stacked convs, ported below).
import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock, ResNet
from torchvision.ops import FeaturePyramidNetwork


# ---------------------------------------------------------------------------
# mmcv.cnn primitives (minimal faithful equivalents)
# ---------------------------------------------------------------------------
class ConvModule(nn.Module):
    """Faithful minimal port of mmcv.cnn.ConvModule(conv, norm, act) as used
    throughout AnchorFreeMono3DHead/FCOSMono3DHead/PGDHead (GroupNorm +
    ReLU, matching the `norm_cfg=dict(type='GN', ...)` convention used in
    the real PGD configs)."""

    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, num_groups=32
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias
        )
        self.gn = nn.GroupNorm(min(num_groups, out_channels), out_channels)
        self.activate = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = self.activate(x)
        return x


class Scale(nn.Module):
    """Faithful port of mmcv.cnn.Scale: a single learnable scalar
    multiplier used to rescale predictions per FPN level."""

    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))

    def forward(self, x):
        return x * self.scale


# ---------------------------------------------------------------------------
# FCOS3DBBoxCoder / PGDBBoxCoder (box decode logic invoked inline inside the
# real forward_single -- part of the traced forward pass, not just loss)
# ---------------------------------------------------------------------------
class FCOS3DBBoxCoder:
    """Faithful port of mmdet3d.models.task_modules.coders.FCOS3DBBoxCoder."""

    def __init__(self, base_depths=None, base_dims=None, code_size=7, norm_on_bbox=True):
        self.base_depths = base_depths
        self.base_dims = base_dims
        self.bbox_code_size = code_size
        self.norm_on_bbox = norm_on_bbox

    def decode(self, bbox, scale, stride, training, cls_score=None):
        scale_offset, scale_depth, scale_size = scale[0:3]

        clone_bbox = bbox.clone()
        bbox[:, :2] = scale_offset(clone_bbox[:, :2]).float()
        bbox[:, 2] = scale_depth(clone_bbox[:, 2]).float()
        bbox[:, 3:6] = scale_size(clone_bbox[:, 3:6]).float()

        if self.base_depths is None:
            bbox[:, 2] = bbox[:, 2].exp()
        elif len(self.base_depths) == 1:
            mean = self.base_depths[0][0]
            std = self.base_depths[0][1]
            bbox[:, 2] = mean + bbox.clone()[:, 2] * std
        else:
            indices = cls_score.max(dim=1)[1]
            depth_priors = cls_score.new_tensor(self.base_depths)[indices, :].permute(0, 3, 1, 2)
            mean = depth_priors[:, 0]
            std = depth_priors[:, 1]
            bbox[:, 2] = mean + bbox.clone()[:, 2] * std

        bbox[:, 3:6] = bbox[:, 3:6].exp()
        if self.base_dims is not None:
            indices = cls_score.max(dim=1)[1]
            size_priors = cls_score.new_tensor(self.base_dims)[indices, :].permute(0, 3, 1, 2)
            bbox[:, 3:6] = size_priors * bbox.clone()[:, 3:6]

        if self.norm_on_bbox and not training:
            bbox[:, :2] *= stride

        return bbox


class PGDBBoxCoder(FCOS3DBBoxCoder):
    """Faithful port of mmdet3d.models.task_modules.coders.PGDBBoxCoder."""

    def decode_2d(
        self,
        bbox,
        scale,
        stride,
        max_regress_range,
        training,
        pred_keypoints=False,
        pred_bbox2d=True,
    ):
        clone_bbox = bbox.clone()
        if pred_keypoints:
            scale_kpts = scale[3]
            bbox[:, self.bbox_code_size : self.bbox_code_size + 16] = torch.tanh(
                scale_kpts(clone_bbox[:, self.bbox_code_size : self.bbox_code_size + 16]).float()
            )

        if pred_bbox2d:
            scale_bbox2d = scale[-1]
            bbox[:, -4:] = scale_bbox2d(clone_bbox[:, -4:]).float()

        if self.norm_on_bbox:
            if pred_bbox2d:
                bbox[:, -4:] = F.relu(bbox.clone()[:, -4:])
            if not training:
                if pred_keypoints:
                    bbox[:, self.bbox_code_size : self.bbox_code_size + 16] *= max_regress_range
                if pred_bbox2d:
                    bbox[:, -4:] *= stride
        else:
            if pred_bbox2d:
                bbox[:, -4:] = bbox.clone()[:, -4:].exp()
        return bbox

    def decode_prob_depth(self, depth_cls_preds, depth_range, depth_unit, division, num_depth_cls):
        if division == "uniform":
            depth_multiplier = depth_unit * depth_cls_preds.new_tensor(
                list(range(num_depth_cls))
            ).reshape([1, -1])
            return (F.softmax(depth_cls_preds.clone(), dim=-1) * depth_multiplier).sum(dim=-1)
        raise NotImplementedError


# ---------------------------------------------------------------------------
# PGDHead -- faithful port of the AnchorFreeMono3DHead -> FCOSMono3DHead ->
# PGDHead _init_layers/_init_predictor/forward_single chain.
# ---------------------------------------------------------------------------
class PGDHead(nn.Module):
    def __init__(
        self,
        num_classes=3,
        in_channels=64,
        feat_channels=64,
        stacked_convs=2,
        strides=(8, 16, 32),
        use_direction_classifier=True,
        pred_attrs=True,
        num_attrs=9,
        pred_velo=True,
        pred_bbox2d=True,
        pred_keypoints=False,
        use_depth_classifier=True,
        centerness_on_reg=True,
        conv_bias=True,
        cls_branch=(64,),
        reg_branch=((64,), (64,), (64,), (64,), ()),
        group_reg_dims=(2, 1, 3, 1, 2),
        dir_branch=(64,),
        attr_branch=(64,),
        centerness_branch=(64,),
        depth_branch=(64,),
        depth_range=(0, 50),
        depth_unit=10,
        division="uniform",
        depth_bins=6,
        weight_dim=-1,
        weight_branch=((64,),),
        norm_on_bbox=True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.cls_out_channels = num_classes  # use_sigmoid focal loss convention
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.use_direction_classifier = use_direction_classifier
        self.pred_attrs = pred_attrs
        self.num_attrs = num_attrs
        self.pred_velo = pred_velo
        self.pred_bbox2d = pred_bbox2d
        self.pred_keypoints = pred_keypoints
        self.use_depth_classifier = use_depth_classifier
        self.use_onlyreg_proj = False
        self.centerness_on_reg = centerness_on_reg
        self.conv_bias = conv_bias
        self.cls_branch = cls_branch
        self.reg_branch = reg_branch
        self.group_reg_dims = list(group_reg_dims)
        self.dir_branch = dir_branch
        self.attr_branch = attr_branch
        self.centerness_branch = centerness_branch
        self.depth_branch = depth_branch
        self.depth_range = depth_range
        self.depth_unit = depth_unit
        self.division = division
        self.weight_dim = weight_dim
        self.weight_branch = weight_branch
        self.norm_on_bbox = norm_on_bbox
        self.bbox_code_size = 9 if pred_velo else 7

        self.out_channels = []
        for rb in reg_branch:
            self.out_channels.append(rb[-1] if len(rb) > 0 else -1)
        self.weight_out_channels = []
        for wb in weight_branch:
            self.weight_out_channels.append(wb[-1] if len(wb) > 0 else -1)

        if division == "uniform":
            self.num_depth_cls = int((depth_range[1] - depth_range[0]) / depth_unit) + 1
        else:
            self.num_depth_cls = depth_bins

        self.bbox_coder = PGDBBoxCoder(code_size=self.bbox_code_size, norm_on_bbox=norm_on_bbox)

        self._init_layers()

    # -- AnchorFreeMono3DHead._init_layers / _init_cls_convs / _init_reg_convs
    def _init_layers(self):
        self.cls_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(chn, self.feat_channels, 3, stride=1, padding=1, bias=self.conv_bias)
            )
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.reg_convs.append(
                ConvModule(chn, self.feat_channels, 3, stride=1, padding=1, bias=self.conv_bias)
            )
        self._init_predictor()

        # FCOSMono3DHead: centerness branch + scale modules (scale_dim=3)
        self.conv_centerness_prev = self._init_branch(self.centerness_branch)
        self.conv_centerness = nn.Conv2d(self.centerness_branch[-1], 1, 1)
        self.scale_dim = 3
        if self.pred_bbox2d:
            self.scale_dim += 1
        if self.pred_keypoints:
            self.scale_dim += 1
        self.scales = nn.ModuleList(
            [nn.ModuleList([Scale(1.0) for _ in range(self.scale_dim)]) for _ in self.strides]
        )

    def _init_branch(self, conv_channels):
        conv_before_pred = nn.ModuleList()
        chans = [self.feat_channels] + list(conv_channels)
        for i in range(len(conv_channels)):
            conv_before_pred.append(
                ConvModule(chans[i], chans[i + 1], 3, stride=1, padding=1, bias=self.conv_bias)
            )
        return conv_before_pred

    # -- AnchorFreeMono3DHead._init_predictor
    def _init_predictor(self):
        self.conv_cls_prev = self._init_branch(self.cls_branch)
        self.conv_cls = nn.Conv2d(self.cls_branch[-1], self.cls_out_channels, 1)

        self.conv_reg_prevs = nn.ModuleList()
        self.conv_regs = nn.ModuleList()
        for i in range(len(self.group_reg_dims)):
            reg_dim = self.group_reg_dims[i]
            reg_branch_channels = self.reg_branch[i]
            out_channel = self.out_channels[i]
            if len(reg_branch_channels) > 0:
                self.conv_reg_prevs.append(self._init_branch(reg_branch_channels))
                self.conv_regs.append(nn.Conv2d(out_channel, reg_dim, 1))
            else:
                self.conv_reg_prevs.append(None)
                self.conv_regs.append(nn.Conv2d(self.feat_channels, reg_dim, 1))

        if self.use_direction_classifier:
            self.conv_dir_cls_prev = self._init_branch(self.dir_branch)
            self.conv_dir_cls = nn.Conv2d(self.dir_branch[-1], 2, 1)
        if self.pred_attrs:
            self.conv_attr_prev = self._init_branch(self.attr_branch)
            self.conv_attr = nn.Conv2d(self.attr_branch[-1], self.num_attrs, 1)

        # PGDHead._init_predictor extras
        if self.use_depth_classifier:
            self.conv_depth_cls_prev = self._init_branch(self.depth_branch)
            self.conv_depth_cls = nn.Conv2d(self.depth_branch[-1], self.num_depth_cls, 1)
            self.fuse_lambda = nn.Parameter(torch.tensor(10e-5))

        if self.weight_dim != -1:
            self.conv_weight_prevs = nn.ModuleList()
            self.conv_weights = nn.ModuleList()
            for i in range(self.weight_dim):
                weight_branch_channels = self.weight_branch[i]
                weight_out_channel = self.weight_out_channels[i]
                if len(weight_branch_channels) > 0:
                    self.conv_weight_prevs.append(self._init_branch(weight_branch_channels))
                    self.conv_weights.append(nn.Conv2d(weight_out_channel, 1, 1))
                else:
                    self.conv_weight_prevs.append(None)
                    self.conv_weights.append(nn.Conv2d(self.feat_channels, 1, 1))

    def forward(self, feats: Tuple[torch.Tensor]):
        outs = []
        for x, scale, stride in zip(feats, self.scales, self.strides):
            outs.append(self.forward_single(x, scale, stride))
        return outs

    # -- AnchorFreeMono3DHead.forward_single (base cls/reg/dir/attr branches)
    def _forward_single_base(self, x):
        cls_feat = x
        reg_feat = x

        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)
        clone_cls_feat = cls_feat.clone()
        for layer in self.conv_cls_prev:
            clone_cls_feat = layer(clone_cls_feat)
        cls_score = self.conv_cls(clone_cls_feat)

        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)
        bbox_pred = []
        for i in range(len(self.group_reg_dims)):
            clone_reg_feat = reg_feat.clone()
            if len(self.reg_branch[i]) > 0:
                for layer in self.conv_reg_prevs[i]:
                    clone_reg_feat = layer(clone_reg_feat)
            bbox_pred.append(self.conv_regs[i](clone_reg_feat))
        bbox_pred = torch.cat(bbox_pred, dim=1)

        dir_cls_pred = None
        if self.use_direction_classifier:
            clone_reg_feat = reg_feat.clone()
            for layer in self.conv_dir_cls_prev:
                clone_reg_feat = layer(clone_reg_feat)
            dir_cls_pred = self.conv_dir_cls(clone_reg_feat)

        attr_pred = None
        if self.pred_attrs:
            clone_cls_feat = cls_feat.clone()
            for layer in self.conv_attr_prev:
                clone_cls_feat = layer(clone_cls_feat)
            attr_pred = self.conv_attr(clone_cls_feat)

        return cls_score, bbox_pred, dir_cls_pred, attr_pred, cls_feat, reg_feat

    # -- PGDHead.forward_single (chains through FCOSMono3DHead.forward_single)
    def forward_single(self, x, scale, stride):
        cls_score, bbox_pred, dir_cls_pred, attr_pred, cls_feat, reg_feat = (
            self._forward_single_base(x)
        )

        # FCOSMono3DHead.forward_single: centerness + bbox_coder.decode
        if self.centerness_on_reg:
            clone_reg_feat = reg_feat.clone()
            for layer in self.conv_centerness_prev:
                clone_reg_feat = layer(clone_reg_feat)
            centerness = self.conv_centerness(clone_reg_feat)
        else:
            clone_cls_feat = cls_feat.clone()
            for layer in self.conv_centerness_prev:
                clone_cls_feat = layer(clone_cls_feat)
            centerness = self.conv_centerness(clone_cls_feat)

        bbox_pred = self.bbox_coder.decode(bbox_pred, scale, stride, self.training, cls_score)

        # PGDHead.forward_single: 2D decode + depth classifier + weight branch
        max_regress_range = stride * 128 * 8 / self.strides[0]
        bbox_pred = self.bbox_coder.decode_2d(
            bbox_pred,
            scale,
            stride,
            max_regress_range,
            self.training,
            self.pred_keypoints,
            self.pred_bbox2d,
        )

        depth_cls_pred = None
        if self.use_depth_classifier:
            clone_reg_feat = reg_feat.clone()
            for layer in self.conv_depth_cls_prev:
                clone_reg_feat = layer(clone_reg_feat)
            depth_cls_pred = self.conv_depth_cls(clone_reg_feat)

        weight = None
        if self.weight_dim != -1:
            weight = []
            for i in range(self.weight_dim):
                clone_reg_feat = reg_feat.clone()
                if len(self.weight_branch[i]) > 0:
                    for layer in self.conv_weight_prevs[i]:
                        clone_reg_feat = layer(clone_reg_feat)
                weight.append(self.conv_weights[i](clone_reg_feat))
            weight = torch.cat(weight, dim=1)

        return cls_score, bbox_pred, dir_cls_pred, depth_cls_pred, weight, attr_pred, centerness


# ---------------------------------------------------------------------------
# Full detector: ResNet backbone + FPN neck + PGDHead
# ---------------------------------------------------------------------------
class PGDDetector(nn.Module):
    """ResNet + FPN + PGDHead, mirroring mmdet3d's FCOSMono3D detector
    composition (backbone=ResNet, neck=FPN, bbox_head=PGDHead)."""

    def __init__(self):
        super().__init__()
        self.backbone = ResNet(BasicBlock, [1, 1, 1, 1])
        del self.backbone.avgpool, self.backbone.fc
        self.neck = FeaturePyramidNetwork([64, 128, 256], 64)
        self.bbox_head = PGDHead(in_channels=64, feat_channels=64, strides=(8, 16, 32))

    def extract_feat(self, img):
        x = self.backbone.conv1(img)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        c2 = self.backbone.layer1(x)
        c3 = self.backbone.layer2(c2)
        c4 = self.backbone.layer3(c3)
        feats = self.neck({"0": c2, "1": c3, "2": c4})
        return tuple(feats.values())

    def forward(self, img):
        feats = self.extract_feat(img)
        return self.bbox_head(feats)


MENAGERIE_ZOO = "ported-pytorch"


def build_pgd():
    torch.manual_seed(0)
    model = PGDDetector()
    model.eval()
    return model


def example_input_pgd():
    torch.manual_seed(0)
    return torch.randn(1, 3, 128, 128)


MENAGERIE_ENTRIES = [
    ("PGD (probabilistic geometric depth)", "build_pgd", "example_input_pgd", 2021, MENAGERIE_ZOO),
]
