# FAITHFUL REIMPLEMENTATION from Chen et al., "Multi-View 3D Object Detection
# Network for Autonomous Driving" (CVPR 2017, arXiv:1611.07759) -- no runnable
# public code exists. Every candidate "MV3D PyTorch" repository found on GitHub
# (e.g. collector-m/MV3D-Pytorch, despite the name) turned out to be TensorFlow
# 1.x code with a custom `roi_pooling_op` compiled via `tf.load_op_library`
# (verified by reading dummynet.py / net/blocks.py / net/roipooling_op/*.py at
# https://github.com/collector-m/MV3D-Pytorch); the two other public
# reimplementations found (sublucky/MV3D, AutoLidarPerception/MV3D,
# wayne0908/Multi-View-3D-Object-Detection-Network-for-Autonomous-Driving) are
# likewise plain TensorFlow 1.x (tf.variable_scope / TensorflowUtils.py) with no
# base-torch-installable path (custom TF ops, TF1 graph mode). Per the ladder
# this repo cannot be vendored/ported -- it is reimplemented here from the
# paper's explicit "3.4. Implementation / Network Architecture" description.
#
# Architecture, transcribed directly from the paper text (Section 3.2-3.4):
#   - Three input branches with an IDENTICAL trunk architecture: Bird's-Eye-View
#     (BV, (M+2)-channel height/intensity/density map), Front-View (FV,
#     3-channel height/distance/intensity cylindrical projection), and RGB
#     image. Trunk = 16-layer VGG ("VGG-16") with: (a) channels reduced to half
#     of the original VGG-16 width, (b) the 4th max-pool removed so the conv
#     stack proceeds to 8x downsampling instead of 16x (this repo therefore
#     uses VGG-16's first 4 conv stages -- 64/128/256/512 halved to
#     32/64/128/256 -- with only 3 pooling ops between them, dropping the 5th
#     512-channel stage/pool entirely since it is downstream of the removed
#     pool).
#   - 3D Proposal Network: operates on the BV trunk's last conv feature map,
#     first 2x bilinearly upsampled (so overall BV downsampling is 4x). Two
#     sibling 1x1-conv heads: an "objectness" classifier (2 logits per anchor)
#     and a 3D box regressor (6 numbers per anchor: dx,dy,dz,dl,dw,dh) for
#     N=4 prior boxes (2 (l,w) car-size clusters x {0 deg, 90 deg} rotations).
#   - Region-based Fusion Network: 3D proposals are projected into all three
#     views (T_3D->v) to get per-view ROIs; each view is upsampled before ROI
#     pooling ("4x/4x/2x upsampling layer... for the BV/FV/RGB branch") then
#     ROI-pooled to a fixed-length per-view feature vector -- here via the
#     base-torch `torchvision.ops.roi_align` (the modern drop-in successor of
#     the Fast R-CNN `ROIPooling` the paper cites at [10]). "Deep fusion":
#     for L fusion layers (paper uses fc6/fc7/(extra)fc8, i.e. L=3), each
#     layer applies a per-view FC transform then joins all three views with an
#     ELEMENT-WISE MEAN (paper Eq. 6 exactly: f_l = H_BV_l(f_{l-1}) (+)
#     H_FV_l(f_{l-1}) (+) H_RGB_l(f_{l-1}), join = mean). The fused output
#     feeds a multiclass softmax classifier and an oriented-3D-box regressor
#     (24-D: 8 corners x (dx,dy,dz), per paper Section "Oriented 3D Box
#     Regression").
#
# Geometric proposal generation (BV box decoding -> LIDAR 3D coordinates ->
# T_3D->v projection into BV/FV/RGB pixel ROIs, using KITTI camera calibration)
# is calibration/dataset machinery external to the network itself -- exactly
# analogous to a real Faster-RCNN's anchor-decoding+NMS pipeline. It is not
# reimplemented; instead this module supplies a fixed, valid, randomly placed
# set of per-view ROI boxes as the "proposals" fed into the fusion network, the
# same way object-detector traces in this menagerie feed random boxes to an
# ROI head.
#
# MENAGERIE_ZOO = "reimpl-pytorch"

import torch
import torch.nn as nn
import torchvision.ops as tvops

MENAGERIE_ZOO = "reimpl-pytorch"


def _vgg_block(in_ch, out_ch, n_convs):
    layers = []
    c = in_ch
    for _ in range(n_convs):
        layers += [
            nn.Conv2d(c, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        c = out_ch
    return nn.Sequential(*layers)


class HalfVGGTrunk(nn.Module):
    """VGG-16 trunk, channels halved, 4th pool removed (paper Section 3.4).

    Real VGG-16 conv stage widths: 64,128,256,512,512 with 5 max-pools (2 convs
    per stage for the first two, 3 for the last three) -> 16x downsampling.
    Paper modification: halve every channel width (32,64,128,256,256) and drop
    the 4th (and, since nothing feeds it, the 5th) pooling stage, leaving 3
    pools -> 8x downsampling before the proposal-network upsampling layers.
    """

    def __init__(self, in_channels):
        super().__init__()
        self.stage1 = _vgg_block(in_channels, 32, 2)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.stage2 = _vgg_block(32, 64, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.stage3 = _vgg_block(64, 128, 3)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.stage4 = _vgg_block(128, 256, 3)  # no pool after this stage (4th pool removed)

    def forward(self, x):
        x = self.stage1(x)
        x = self.pool1(x)
        x = self.stage2(x)
        x = self.pool2(x)
        x = self.stage3(x)
        x = self.pool3(x)
        x = self.stage4(x)
        return x  # 8x downsampled, 256 channels


class ProposalNetwork(nn.Module):
    """3D Proposal Network on the bird's-eye-view trunk (paper Section 3.2)."""

    def __init__(self, in_channels=256, n_anchors=4):
        super().__init__()
        # "we use 2x bilinear upsampling after the last conv layer" -> 4x overall
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.objectness = nn.Conv2d(in_channels, n_anchors * 2, kernel_size=1)
        self.box_reg = nn.Conv2d(in_channels, n_anchors * 6, kernel_size=1)  # dx,dy,dz,dl,dw,dh

    def forward(self, bv_feat):
        up = self.upsample(bv_feat)
        objectness = self.objectness(up)
        box_reg = self.box_reg(up)
        return up, objectness, box_reg


class DeepFusionHead(nn.Module):
    """Region-based deep-fusion network (paper Section 3.3, Eq. 6): L fully
    connected transform layers per view, joined by element-wise mean at every
    layer, feeding a multiclass classifier + oriented 3D box regressor."""

    def __init__(self, in_features, fc_dim=512, n_classes=2, n_fusion_layers=3):
        super().__init__()
        self.n_fusion_layers = n_fusion_layers
        views = ("bv", "fv", "rgb")
        self.view_fc = nn.ModuleDict()
        for v in views:
            layers = nn.ModuleList()
            d_in = in_features
            for _ in range(n_fusion_layers):
                layers.append(nn.Sequential(nn.Linear(d_in, fc_dim), nn.ReLU(inplace=True)))
                d_in = fc_dim
            self.view_fc[v] = layers

        self.cls_head = nn.Linear(fc_dim, n_classes)
        self.box_head = nn.Linear(fc_dim, 8 * 3)  # 8 corners x (dx, dy, dz)

    def forward(self, f_bv, f_fv, f_rgb):
        feats = {"bv": f_bv, "fv": f_fv, "rgb": f_rgb}
        for layer_idx in range(self.n_fusion_layers):
            transformed = {v: self.view_fc[v][layer_idx](feats[v]) for v in feats}
            fused = torch.stack(list(transformed.values()), dim=0).mean(dim=0)
            feats = {
                v: fused for v in feats
            }  # deep fusion: joined feature feeds every view's next layer

        fused_final = feats["bv"]
        cls_logits = self.cls_head(fused_final)
        box_reg = self.box_head(fused_final)
        return cls_logits, box_reg


class MV3D(nn.Module):
    def __init__(
        self,
        bv_channels=6,  # (M+2) with M=4 height slices, per paper Section 3.1
        fv_channels=3,  # height, distance, intensity
        rgb_channels=3,
        n_anchors=4,
        n_classes=2,  # car / background
        roi_output_size=7,
        fc_dim=512,
        n_fusion_layers=3,
    ):
        super().__init__()

        self.bv_trunk = HalfVGGTrunk(bv_channels)
        self.fv_trunk = HalfVGGTrunk(fv_channels)
        self.rgb_trunk = HalfVGGTrunk(rgb_channels)

        self.proposal_net = ProposalNetwork(in_channels=256, n_anchors=n_anchors)

        # "4x/4x/2x upsampling layer... for the BV/FV/RGB branch" before ROI pooling
        self.bv_roi_upsample = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        self.fv_roi_upsample = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        self.rgb_roi_upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.roi_output_size = roi_output_size
        roi_feat_dim = 256 * roi_output_size * roi_output_size

        self.fusion_head = DeepFusionHead(
            in_features=roi_feat_dim,
            fc_dim=fc_dim,
            n_classes=n_classes,
            n_fusion_layers=n_fusion_layers,
        )

    def forward(self, bv_input, fv_input, rgb_input, bv_rois, fv_rois, rgb_rois):
        bv_feat = self.bv_trunk(bv_input)
        fv_feat = self.fv_trunk(fv_input)
        rgb_feat = self.rgb_trunk(rgb_input)

        bv_up, objectness, box_reg = self.proposal_net(bv_feat)

        bv_roi_map = self.bv_roi_upsample(bv_feat)
        fv_roi_map = self.fv_roi_upsample(fv_feat)
        rgb_roi_map = self.rgb_roi_upsample(rgb_feat)

        f_bv = tvops.roi_align(
            bv_roi_map, bv_rois, output_size=self.roi_output_size, spatial_scale=1.0
        )
        f_fv = tvops.roi_align(
            fv_roi_map, fv_rois, output_size=self.roi_output_size, spatial_scale=1.0
        )
        f_rgb = tvops.roi_align(
            rgb_roi_map, rgb_rois, output_size=self.roi_output_size, spatial_scale=1.0
        )

        n_rois = f_bv.shape[0]
        f_bv = f_bv.reshape(n_rois, -1)
        f_fv = f_fv.reshape(n_rois, -1)
        f_rgb = f_rgb.reshape(n_rois, -1)

        cls_logits, corner_reg = self.fusion_head(f_bv, f_fv, f_rgb)

        return {
            "objectness": objectness,
            "proposal_box_reg": box_reg,
            "cls_logits": cls_logits,
            "corner_reg": corner_reg,
        }


# --------------------------------------------------------------------------- #
# menagerie staging entry points
# --------------------------------------------------------------------------- #

_BV_H, _BV_W = 96, 96  # discretized bird's-eye-view grid (real cfg ~704x800), shrunk
_FV_H, _FV_W = 32, 128  # cylindrical front-view projection (real cfg 64x512), shrunk
_RGB_H, _RGB_W = 96, 128
_N_ROIS = 3


def build_mv3d():
    return MV3D(bv_channels=6, fv_channels=3, rgb_channels=3, n_anchors=4, n_classes=2)


def example_input_mv3d():
    bv_input = torch.rand(1, 6, _BV_H, _BV_W)
    fv_input = torch.rand(1, 3, _FV_H, _FV_W)
    rgb_input = torch.rand(1, 3, _RGB_H, _RGB_W)

    def _random_rois(h, w, n):
        boxes = torch.zeros(n, 4)
        for i in range(n):
            x1 = torch.rand(1).item() * (w * 0.5)
            y1 = torch.rand(1).item() * (h * 0.5)
            x2 = x1 + torch.rand(1).item() * (w * 0.5) + 1.0
            y2 = y1 + torch.rand(1).item() * (h * 0.5) + 1.0
            boxes[i] = torch.tensor([x1, y1, x2, y2])
        return [boxes]  # torchvision roi_align list-of-Tensor-per-image format

    # ROI coordinates given in the *upsampled* (post roi_upsample) feature-map
    # pixel space of each view (BV/FV upsampled 4x, RGB upsampled 2x from the
    # 8x-downsampled trunk output -> BV/FV effective stride 2, RGB stride 4).
    bv_rois = _random_rois(_BV_H // 2, _BV_W // 2, _N_ROIS)
    fv_rois = _random_rois(_FV_H // 2, _FV_W // 2, _N_ROIS)
    rgb_rois = _random_rois(_RGB_H // 4, _RGB_W // 4, _N_ROIS)

    return (bv_input, fv_input, rgb_input, bv_rois, fv_rois, rgb_rois)


MENAGERIE_ENTRIES = [
    ("MV3D", "build_mv3d", "example_input_mv3d", 2017, "reimpl-pytorch"),
]
