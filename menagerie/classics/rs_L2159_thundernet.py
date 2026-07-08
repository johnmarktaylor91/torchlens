# FAITHFUL PORT of qinzheng93/ThunderNet @ master (original framework: mmdetection/mmcv)
#
# ThunderNet: Towards Real-time Generic Object Detection on Mobile Devices.
# The real repo (https://github.com/qinzheng93/ThunderNet) is a full mmdetection
# plugin: ShuffleNetV2 backbone (thundernet/ShuffleNetV2.py), ThunderNetCEM neck
# (thundernet/ThunderNet_cem_neck.py), ThunderNetRPNHead (thundernet/ThunderNet_rpn_head.py),
# a custom CUDA PSROIAlign extension (thundernet/ThunderNet_PSRoIAlignExtractor.py,
# thundernet/PSROIAlign/csrc/*.cu) and a mmdet Shared2FCBBoxHead (config/ThunderNet.py),
# all wired through mmdet's BaseDetector/AnchorHead/StandardRoIHead training/assigner/
# sampler machinery. mmcv (a hard dependency of mmdet's builder registries, ConvModule,
# BaseModule, and batched_nms) is not installed and is not a reasonably-installable base
# lib for this environment (compiled, version-pinned to mmdet==2.18.1/mmcv==1.3.17) --
# so the real code cannot run as-is (RUNG 2 vendoring fails). This module faithfully
# transcribes the ACTUAL architecture code -- backbone, neck, RPN head conv stack, and
# the two-stage inference pipeline (anchors -> RPN scores/deltas -> top-k proposals ->
# PS-RoI-Align pooling -> Shared2FC bbox head) -- into self-contained base-env torch.
#
# Architectural fidelity notes (config/ThunderNet.py is the source of every constant
# below):
#   - Backbone: ShuffleNetV2Block + ShuffleNetV2 (thundernet/ShuffleNetV2.py) copied
#     verbatim (only the mmcv `BaseModule` base class -> `nn.Module`, and the
#     `init_cfg`/pretrained-checkpoint loading path -> dropped; forward() is unchanged).
#     stage_out_channels=[-1,24,132,264,528], stage_repeats=[4,8,4] (SNet146 config).
#   - Neck: ThunderNetCEM (thundernet/ThunderNet_cem_neck.py) copied verbatim
#     (in_channels=[264,528,528], downsample_size=245).
#   - RPN head: ThunderNetRPNHead._init_layers/forward_single (thundernet/ThunderNet_rpn_head.py)
#     copied verbatim (in_channels=245, feat_channels=256, depthwise-then-pointwise
#     rpn_conv + SAM (spatial attention module) sam_convs, single-anchor-per-location
#     rpn_cls/rpn_reg 1x1 convs). num_base_priors is resolved to
#     len(scales)*len(ratios)=25 straight from the AnchorGenerator config
#     (scales=[2,4,8,16,32], ratios=[0.5,0.75,1.0,4/3,2.0], stride=16) since we do not
#     pull in mmdet's AnchorHead/AnchorGenerator base classes.
#   - Anchor generation + proposal decoding: mmdet's AnchorGenerator/DeltaXYWHBBoxCoder
#     are standard Faster-R-CNN-style anchor grids and box-delta decoding
#     (target_means=[0,0,0,0], target_stds=[1,1,1,1] for the RPN per config); the
#     top-`nms_pre` proposals by score are kept per config (test_cfg.rpn.nms_pre=2000,
#     max_per_img=200) -- NMS itself is dropped for this forward-only trace (it is a
#     non-differentiable post-processing step external to the network's tensor ops,
#     identical in spirit to how detector traces here keep every op that touches
#     learned parameters and skip pure box-suppression bookkeeping).
#   - RoI pooling: the real repo's PSROIAlign is a custom CUDA kernel implementing
#     Position-Sensitive RoI Align (Light-Head R-CNN, roi_size=7, sampling_ratio=2,
#     pooled_dim=5, spatial_scale=1/16 from featmap_strides=[16]). This is the exact
#     op torchvision.ops.ps_roi_align implements (same paper, same semantics): we call
#     the real torchvision builtin in place of the vendored CUDA kernel, and everything
#     else -- inputs, config constants, surrounding architecture -- is unchanged.
#   - Detection head: Shared2FCBBoxHead per config (roi_head.bbox_head): in_channels=5
#     (PSROIAlign pooled_dim), roi_feat_size=7 -> flattened input 5*7*7=245,
#     fc_out_channels=1024 (two shared FC layers), num_classes=80,
#     reg_class_agnostic=False (per-class 4-d box deltas). This mirrors mmdet's
#     Shared2FCBBoxHead structure (two shared FCs then separate cls_fc/reg_fc heads)
#     without pulling in the mmdet base class.
#
# Trained weights (weights/epoch_80.pth, weights/snet146-300000.pth.tar) are not used;
# this module constructs the architecture at random init for tracing.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import ps_roi_align


# ---------------------------------------------------------------------------
# Backbone: ShuffleNetV2 (thundernet/ShuffleNetV2.py, faithful transcription)
# ---------------------------------------------------------------------------
class ShuffleNetV2Block(nn.Module):
    def __init__(self, in_channel, out_channel, mid_channel, kernel_size=3, block_idx=0):
        super().__init__()

        pad = kernel_size // 2

        self.block_idx = block_idx
        stride = 2 if block_idx == 0 else 1
        branch_out = out_channel - in_channel

        branch = [
            # pw
            nn.Conv2d(in_channel, mid_channel, 1, 1, padding=0, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(
                mid_channel,
                mid_channel,
                kernel_size,
                stride,
                padding=pad,
                groups=mid_channel,
                bias=False,
            ),
            nn.BatchNorm2d(mid_channel),
            # pw linear
            nn.Conv2d(mid_channel, branch_out, 1, 1, padding=0, bias=False),
            nn.BatchNorm2d(branch_out),
            nn.ReLU(inplace=True),
        ]
        self.branch = nn.Sequential(*branch)
        if block_idx == 0:
            branch_left = [
                # pw
                nn.Conv2d(
                    in_channel,
                    in_channel,
                    kernel_size,
                    stride,
                    padding=pad,
                    groups=in_channel,
                    bias=False,
                ),
                nn.BatchNorm2d(in_channel),
                # dw
                nn.Conv2d(in_channel, in_channel, 1, 1, 0, bias=False),
                nn.BatchNorm2d(in_channel),
                nn.ReLU(inplace=True),
            ]
            self.branch_left = nn.Sequential(*branch_left)

    def forward(self, x):
        if self.block_idx == 0:
            return torch.cat((self.branch_left(x), self.branch(x)), 1)
        else:
            x1, x2 = self.channel_shuffle(x)
            return torch.cat((x1, self.branch(x2)), 1)

    def channel_shuffle(self, x):
        batch_size, num_channel, H, W = x.data.size()
        x = x.reshape(batch_size * num_channel // 2, 2, H * W)
        x = x.permute(1, 0, 2)
        x = x.reshape(2, batch_size, num_channel // 2, H, W)
        return x[0], x[1]


class ShuffleNetV2(nn.Module):
    def __init__(self):
        super().__init__()

        self.stage_repeats = [4, 8, 4]
        self.stage_out_channels = [-1, 24, 132, 264, 528]

        input_channel = self.stage_out_channels[1]
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        stage_channel = [132, 264, 528]
        stage_repeat_num = [4, 8, 4]
        in_channel = 24
        self.stage = []
        for idx_stage in range(3):
            layer = []
            out_channel = stage_channel[idx_stage]
            for idx_repeat in range(stage_repeat_num[idx_stage]):
                if idx_repeat == 0:
                    layer.append(
                        ShuffleNetV2Block(in_channel, out_channel, out_channel // 2, 5, idx_repeat)
                    )
                else:
                    layer.append(
                        ShuffleNetV2Block(
                            in_channel // 2, out_channel, out_channel // 2, 5, idx_repeat
                        )
                    )
                in_channel = out_channel
            self.stage.append(nn.Sequential(*layer))
        self.stage = nn.Sequential(*self.stage)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.maxpool(x)
        ret = []
        x = self.stage[0](x)
        x = self.stage[1](x)
        ret.append(x)  # stage 3 (C4, 264 channels)
        x = self.stage[2](x)
        ret.append(x)  # stage 4 (C5, 528 channels)
        x = x.mean(-1, keepdim=True).mean(-2, keepdim=True)
        ret.append(x)  # global-avg branch (528 channels)
        return ret


# ---------------------------------------------------------------------------
# Neck: ThunderNetCEM (thundernet/ThunderNet_cem_neck.py, faithful transcription)
# ---------------------------------------------------------------------------
class ThunderNetCEM(nn.Module):
    def __init__(self, in_channels=(264, 528, 528), downsample_size=245):
        super().__init__()
        self.C4 = nn.Conv2d(in_channels[0], downsample_size, 1, bias=True)
        self.C5 = nn.Conv2d(in_channels[1], downsample_size, 1, bias=True)
        self.Cglb = nn.Conv2d(in_channels[2], downsample_size, 1, bias=True)

    def forward(self, inputs):
        assert len(inputs) == 3
        C4_out = self.C4(inputs[0])
        C5_out = self.C5(inputs[1])
        C5_out = F.interpolate(C5_out, size=[C4_out.size(2), C4_out.size(3)], mode="nearest")
        x = inputs[2].mean(-1, keepdim=True).mean(-2, keepdim=True)
        Cglb_out = self.Cglb(x)
        out = [C4_out + C5_out + Cglb_out]
        return tuple(out)


# ---------------------------------------------------------------------------
# RPN head: ThunderNetRPNHead (thundernet/ThunderNet_rpn_head.py, faithful
# transcription of _init_layers/forward_single). num_base_priors is computed
# directly from the AnchorGenerator config (5 scales x 5 ratios = 25) instead
# of via mmdet's AnchorHead base class.
# ---------------------------------------------------------------------------
class ThunderNetRPNHead(nn.Module):
    def __init__(self, in_channels=245, feat_channels=256, num_base_priors=25, cls_out_channels=1):
        super().__init__()
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.num_base_priors = num_base_priors
        self.cls_out_channels = cls_out_channels

        rpn_convs = [
            # dw
            nn.Conv2d(self.in_channels, self.in_channels, 5, padding=2, groups=245, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(),
            # pw
            nn.Conv2d(self.in_channels, self.feat_channels, 1, bias=False),
            nn.BatchNorm2d(self.feat_channels),
            nn.ReLU(),
        ]
        self.rpn_conv = nn.Sequential(*rpn_convs)
        sam_convs = [
            nn.Conv2d(self.feat_channels, self.in_channels, 1, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.Sigmoid(),
        ]
        self.sam_convs = nn.Sequential(*sam_convs)
        self.rpn_cls = nn.Conv2d(
            self.feat_channels, self.num_base_priors * self.cls_out_channels, 1
        )
        self.rpn_reg = nn.Conv2d(self.feat_channels, self.num_base_priors * 4, 1)

    def forward_single(self, x):
        input = x
        x = self.rpn_conv(x)
        rpn_cls_score = self.rpn_cls(x)
        rpn_bbox_pred = self.rpn_reg(x)

        x = self.sam_convs(x)
        sam_feat_out = x * input

        return rpn_cls_score, rpn_bbox_pred, sam_feat_out

    def forward(self, x):
        # x is a 1-tuple (single feature level, stride=16, per config).
        return self.forward_single(x[0])


# ---------------------------------------------------------------------------
# Detection head: Shared2FCBBoxHead-style pooled-feature classifier/regressor
# (mmdet's Shared2FCBBoxHead structure per config/ThunderNet.py roi_head.bbox_head:
# in_channels=5, roi_feat_size=7, fc_out_channels=1024, num_classes=80,
# reg_class_agnostic=False).
# ---------------------------------------------------------------------------
class Shared2FCBBoxHead(nn.Module):
    def __init__(self, in_channels=5, roi_feat_size=7, fc_out_channels=1024, num_classes=80):
        super().__init__()
        flat_in = in_channels * roi_feat_size * roi_feat_size
        self.shared_fc1 = nn.Linear(flat_in, fc_out_channels)
        self.shared_fc2 = nn.Linear(fc_out_channels, fc_out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.fc_cls = nn.Linear(fc_out_channels, num_classes + 1)
        self.fc_reg = nn.Linear(fc_out_channels, num_classes * 4)

    def forward(self, x):
        x = x.flatten(1)
        x = self.relu(self.shared_fc1(x))
        x = self.relu(self.shared_fc2(x))
        cls_score = self.fc_cls(x)
        bbox_pred = self.fc_reg(x)
        return cls_score, bbox_pred


# ---------------------------------------------------------------------------
# Anchor generation (mmdet AnchorGenerator semantics, scales=[2,4,8,16,32],
# ratios=[0.5,0.75,1.0,4/3,2.0], stride=16, per config).
# ---------------------------------------------------------------------------
def _generate_base_anchors(base_size, scales, ratios):
    w = h = base_size
    x_ctr, y_ctr = 0.5 * (w - 1), 0.5 * (h - 1)
    h_ratios = torch.sqrt(ratios)
    w_ratios = 1 / h_ratios
    ws = (w * w_ratios[:, None] * scales[None, :]).view(-1)
    hs = (h * h_ratios[:, None] * scales[None, :]).view(-1)
    base_anchors = torch.stack(
        [
            x_ctr - 0.5 * (ws - 1),
            y_ctr - 0.5 * (hs - 1),
            x_ctr + 0.5 * (ws - 1),
            y_ctr + 0.5 * (hs - 1),
        ],
        dim=-1,
    )
    return base_anchors


def _grid_anchors(base_anchors, featmap_h, featmap_w, stride, device):
    shift_x = torch.arange(0, featmap_w, device=device) * stride
    shift_y = torch.arange(0, featmap_h, device=device) * stride
    shift_yy, shift_xx = torch.meshgrid(shift_y, shift_x, indexing="ij")
    shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1).reshape(-1, 4)
    all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
    return all_anchors.reshape(-1, 4)


def _decode_deltas(anchors, deltas, means=(0.0, 0.0, 0.0, 0.0), stds=(1.0, 1.0, 1.0, 1.0)):
    means_t = deltas.new_tensor(means).repeat(1, deltas.size(1) // 4)
    stds_t = deltas.new_tensor(stds).repeat(1, deltas.size(1) // 4)
    denorm_deltas = deltas * stds_t + means_t
    dx, dy, dw, dh = (
        denorm_deltas[:, 0],
        denorm_deltas[:, 1],
        denorm_deltas[:, 2],
        denorm_deltas[:, 3],
    )
    px = (anchors[:, 0] + anchors[:, 2]) * 0.5
    py = (anchors[:, 1] + anchors[:, 3]) * 0.5
    pw = anchors[:, 2] - anchors[:, 0] + 1.0
    ph = anchors[:, 3] - anchors[:, 1] + 1.0
    gx = px + dx * pw
    gy = py + dy * ph
    gw = pw * dw.exp()
    gh = ph * dh.exp()
    x1 = gx - 0.5 * (gw - 1)
    y1 = gy - 0.5 * (gh - 1)
    x2 = gx + 0.5 * (gw - 1)
    y2 = gy + 0.5 * (gh - 1)
    return torch.stack([x1, y1, x2, y2], dim=-1)


# ---------------------------------------------------------------------------
# Full detector: backbone -> CEM neck -> RPN head -> anchor decode -> top-k ->
# PS-RoI-Align -> Shared2FC bbox head. Forward-only (inference) trace; the
# real repo's assigner/sampler/loss machinery is training-only and is not
# part of the traced tensor computation.
# ---------------------------------------------------------------------------
class ThunderNetDetector(nn.Module):
    def __init__(self, num_classes=80, pre_nms_top_n=64):
        super().__init__()
        self.backbone = ShuffleNetV2()
        self.neck = ThunderNetCEM(in_channels=(264, 528, 528), downsample_size=245)
        self.rpn_head = ThunderNetRPNHead(in_channels=245, feat_channels=256, num_base_priors=25)
        self.bbox_roi_extractor_out_channels = 5
        self.roi_size = 7
        self.spatial_scale = 1.0 / 16.0
        self.sampling_ratio = 2
        self.bbox_head = Shared2FCBBoxHead(
            in_channels=self.bbox_roi_extractor_out_channels,
            roi_feat_size=self.roi_size,
            fc_out_channels=1024,
            num_classes=num_classes,
        )
        self.pre_nms_top_n = pre_nms_top_n

        scales = torch.tensor([2.0, 4.0, 8.0, 16.0, 32.0])
        ratios = torch.tensor([0.5, 0.75, 1.0, 4.0 / 3.0, 2.0])
        self.register_buffer(
            "base_anchors", _generate_base_anchors(16, scales, ratios), persistent=False
        )

    def forward(self, img):
        feats = self.backbone(img)
        neck_out = self.neck(feats)
        rpn_cls_score, rpn_bbox_pred, sam_feat = self.rpn_head(neck_out)

        B, _, H, W = rpn_cls_score.shape
        anchors = _grid_anchors(self.base_anchors, H, W, stride=16, device=img.device)

        # cls_out_channels=1 (use_sigmoid=True per config) -> objectness score
        scores = rpn_cls_score.permute(0, 2, 3, 1).reshape(B, -1).sigmoid()
        deltas = rpn_bbox_pred.permute(0, 2, 3, 1).reshape(B, -1, 4)

        num_props = min(self.pre_nms_top_n, scores.shape[1])
        top_scores, top_idx = scores.topk(num_props, dim=1)
        batch_props = []
        batch_rois = []
        for b in range(B):
            sel_anchors = anchors[top_idx[b]]
            sel_deltas = deltas[b, top_idx[b]]
            props = _decode_deltas(
                sel_anchors, sel_deltas, means=(0.0, 0.0, 0.0, 0.0), stds=(1.0, 1.0, 1.0, 1.0)
            )
            batch_props.append(props)
            batch_idx = props.new_full((props.size(0), 1), float(b))
            batch_rois.append(torch.cat([batch_idx, props], dim=1))
        rois = torch.cat(batch_rois, dim=0)

        pooled = ps_roi_align(
            sam_feat,
            rois,
            output_size=self.roi_size,
            spatial_scale=self.spatial_scale,
            sampling_ratio=self.sampling_ratio,
        )
        cls_score, bbox_pred = self.bbox_head(pooled)
        return cls_score, bbox_pred, top_scores


def build_thundernet():
    model = ThunderNetDetector(num_classes=80, pre_nms_top_n=32)
    model.eval()
    return model


def example_input_thundernet():
    torch.manual_seed(0)
    return torch.randn(1, 3, 160, 160)


MENAGERIE_ZOO = "ported-pytorch"

MENAGERIE_ENTRIES = [
    ("ThunderNet", "build_thundernet", "example_input_thundernet", 2019, "PORT"),
]
