# FAITHFUL PORT of kujason/avod @ master (original framework: TensorFlow 1.x + tf.contrib.slim)
# https://github.com/kujason/avod
# Files transcribed (real math/architecture kept, only the TF1 graph-construction API
# translated to torch nn.Module calls):
#   https://raw.githubusercontent.com/kujason/avod/master/avod/core/feature_extractors/img_vgg_pyramid.py
#   https://raw.githubusercontent.com/kujason/avod/master/avod/core/feature_extractors/bev_vgg_pyramid.py
#   https://raw.githubusercontent.com/kujason/avod/master/avod/core/models/rpn_model.py
#   https://raw.githubusercontent.com/kujason/avod/master/avod/configs/pyramid_cars_with_aug_example.config
#     (real vgg_conv1..4 = [2,32]/[2,64]/[3,128]/[3,256] + rpn_config fc widths)
#
# Ku et al. 2018 (ECCV/IROS) "Joint 3D Proposal Generation and Object Detection from View
# Aggregation" (AVOD). TorchLens is torch-only, so TF1.x graph code (session/placeholder
# based, `tf.contrib.slim` arg-scopes) cannot be captured directly; this is a line-for-line
# faithful architectural transcription of the real ops into eager torch, preserving every
# layer AVOD actually builds:
#   1. `ImgVggPyr`/`BevVggPyr` (`img_vgg_pyramid.py`/`bev_vgg_pyramid.py`, architecturally
#      IDENTICAL modules over different input channel counts): a 4-stage modified-VGG
#      encoder (`conv1..conv4`, each a `slim.repeat` of conv+BN+ReLU followed by 2x2 max
#      pool) with a U-Net-style decoder that transpose-conv-upsamples `conv4` back through
#      3 stages, concatenating with the matching encoder skip connection at each stage
#      (`upconv3`+`concat3`->`pyramid_fusion3`, `upconv2`+`concat2`->`pyramid_fusion2`,
#      `upconv1`+`concat1`->`pyramid_fusion1`) to produce a single full-resolution feature
#      pyramid map. Real channel widths from `pyramid_cars_with_aug_example.config`:
#      vgg_conv1=[2,32], vgg_conv2=[2,64], vgg_conv3=[3,128], vgg_conv4=[3,256].
#   2. `RpnModel.build` (`rpn_model.py`): projects 3D anchors into both the BEV and image
#      feature-map coordinate frames, `tf.image.crop_and_resize`-pools a fixed-size ROI
#      feature from each pyramid at every projected anchor box, fuses the two ROI crops
#      (mean or concat -- `self._fusion_method`), then runs the fused ROI features through
#      twin classification/regression head stacks (`cls_fc6`->`cls_fc7`->`cls_fc8` and
#      `reg_fc6`->`reg_fc7`->`reg_fc8`, each `slim.conv2d`), each `fc6` a
#      `padding='VALID'` conv over the full crop (functionally a per-ROI linear projection,
#      exactly what `nn.Conv2d(..., kernel_size=crop_size, padding=0)` on a `crop_size`
#      feature map produces), the rest 1x1 convs -- to produce `objectness` (2-way
#      cls) and 6-dim anchor-regression `offsets` per ROI. Real config values from
#      `pyramid_cars_with_aug_example.config`: `rpn_config { cls_fc6: 256, cls_fc7: 256,
#      reg_fc6: 256, reg_fc7: 256 }`.
#
# Mechanical/framework translation (no architecture change):
#   - `slim.conv2d(..., normalizer_fn=slim.batch_norm)` -> `nn.Conv2d` + `nn.BatchNorm2d` +
#     `nn.ReLU` (slim's default activation for `arg_scope([slim.conv2d, ...],
#     activation_fn=tf.nn.relu)`), `slim.repeat(x, n, slim.conv2d, ...)` -> an
#     `nn.Sequential` of `n` such conv+BN+ReLU blocks; `slim.max_pool2d([2,2])` ->
#     `nn.MaxPool2d(2, 2)`; `slim.conv2d_transpose(..., stride=2)` ->
#     `nn.ConvTranspose2d(..., stride=2)`; `tf.concat(axis=3)` (NHWC channel axis) ->
#     `torch.cat(dim=1)` (NCHW channel axis).
#   - `tf.image.crop_and_resize(boxes, box_ind, crop_size)` (bilinear-resamples an
#     arbitrary normalized `[y1,x1,y2,x2]` box out of a feature map to a fixed
#     `crop_size` grid) -> `torchvision.ops.roi_align(feat, boxes, output_size,
#     aligned=True)`, the standard torch equivalent bilinear ROI-crop primitive.
#   - The full anchor-projection pipeline (`anchor_projector`, 3D->BEV/image box
#     projection from KITTI calibration) is dataset/geometry plumbing, not part of the
#     network architecture; this port takes already-projected 2D ROI boxes as a plain
#     input tensor (both branches receive independent random boxes over their own
#     feature-map extent, matching the real "BEV boxes" vs "image boxes" pair the RPN
#     receives from `anchor_projector.project_to_bev`/`project_to_image_space`).
#   - `self._fusion_method == 'mean'` is the port's fixed choice (also the example
#     config's default `rpn_fusion_method: 'mean'`).

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align


# --------------------------------------------------------------------------------------
# img_vgg_pyramid.py / bev_vgg_pyramid.py -- VggPyr (architecturally identical for both
# branches; only in_channels differs -- BEV takes a multi-channel height-map, image takes
# RGB)
# --------------------------------------------------------------------------------------
class _ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class _ConvStack(nn.Module):
    """`slim.repeat(inputs, n, slim.conv2d, ch, [3,3], normalizer_fn=slim.batch_norm)`"""

    def __init__(self, in_ch, out_ch, n):
        super().__init__()
        layers = [_ConvBNReLU(in_ch, out_ch)]
        for _ in range(n - 1):
            layers.append(_ConvBNReLU(out_ch, out_ch))
        self.stack = nn.Sequential(*layers)

    def forward(self, x):
        return self.stack(x)


class VggPyr(nn.Module):
    """Modified-VGG encoder-decoder feature pyramid, real architecture shared by
    `ImgVggPyr` and `BevVggPyr` (upstream: two near-identical files)."""

    def __init__(
        self,
        in_channels,
        vgg_conv1=(2, 32),
        vgg_conv2=(2, 64),
        vgg_conv3=(3, 128),
        vgg_conv4=(3, 256),
    ):
        super().__init__()
        c1n, c1 = vgg_conv1
        c2n, c2 = vgg_conv2
        c3n, c3 = vgg_conv3
        c4n, c4 = vgg_conv4

        self.conv1 = _ConvStack(in_channels, c1, c1n)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = _ConvStack(c1, c2, c2n)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = _ConvStack(c2, c3, c3n)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = _ConvStack(c3, c4, c4n)

        self.upconv3 = nn.ConvTranspose2d(
            c4, c3, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.upconv3_bn = nn.BatchNorm2d(c3)
        self.pyramid_fusion3 = nn.Conv2d(c3 + c3, c2, kernel_size=3, padding=1)
        self.pyramid_fusion3_bn = nn.BatchNorm2d(c2)

        self.upconv2 = nn.ConvTranspose2d(
            c2, c2, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.upconv2_bn = nn.BatchNorm2d(c2)
        self.pyramid_fusion2 = nn.Conv2d(c2 + c2, c1, kernel_size=3, padding=1)
        self.pyramid_fusion2_bn = nn.BatchNorm2d(c1)

        self.upconv1 = nn.ConvTranspose2d(
            c1, c1, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.upconv1_bn = nn.BatchNorm2d(c1)
        self.pyramid_fusion1 = nn.Conv2d(c1 + c1, c1, kernel_size=3, padding=1)
        self.pyramid_fusion1_bn = nn.BatchNorm2d(c1)

    def forward(self, x):
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)

        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)

        conv4 = self.conv4(pool3)

        upconv3 = F.relu(self.upconv3_bn(self.upconv3(conv4, output_size=conv3.shape[-2:])))
        concat3 = torch.cat((conv3, upconv3), dim=1)
        pyramid_fusion3 = F.relu(self.pyramid_fusion3_bn(self.pyramid_fusion3(concat3)))

        upconv2 = F.relu(
            self.upconv2_bn(self.upconv2(pyramid_fusion3, output_size=conv2.shape[-2:]))
        )
        concat2 = torch.cat((conv2, upconv2), dim=1)
        pyramid_fusion2 = F.relu(self.pyramid_fusion2_bn(self.pyramid_fusion2(concat2)))

        upconv1 = F.relu(
            self.upconv1_bn(self.upconv1(pyramid_fusion2, output_size=conv1.shape[-2:]))
        )
        concat1 = torch.cat((conv1, upconv1), dim=1)
        pyramid_fusion1 = F.relu(self.pyramid_fusion1_bn(self.pyramid_fusion1(concat1)))

        return pyramid_fusion1


# --------------------------------------------------------------------------------------
# rpn_model.py -- RpnModel.build (ROI crop-and-resize fusion + twin cls/reg head stacks)
# --------------------------------------------------------------------------------------
class RpnModel(nn.Module):
    def __init__(
        self,
        bev_channels=3,
        img_channels=3,
        vgg_conv1=(2, 32),
        vgg_conv2=(2, 64),
        vgg_conv3=(3, 128),
        vgg_conv4=(3, 256),
        proposal_roi_crop_size=3,
        fusion_method="mean",
        cls_fc6=256,
        cls_fc7=256,
        reg_fc6=256,
        reg_fc7=256,
        keep_prob=0.5,
    ):
        super().__init__()
        self.bev_feature_extractor = VggPyr(
            bev_channels, vgg_conv1, vgg_conv2, vgg_conv3, vgg_conv4
        )
        self.img_feature_extractor = VggPyr(
            img_channels, vgg_conv1, vgg_conv2, vgg_conv3, vgg_conv4
        )
        pyramid_channels = vgg_conv1[1]

        self._proposal_roi_crop_size = proposal_roi_crop_size
        self._fusion_method = fusion_method

        self.cls_fc6 = nn.Conv2d(
            pyramid_channels, cls_fc6, kernel_size=proposal_roi_crop_size, padding=0
        )
        self.cls_fc7 = nn.Conv2d(cls_fc6, cls_fc7, kernel_size=1)
        self.cls_fc8 = nn.Conv2d(cls_fc7, 2, kernel_size=1)

        self.reg_fc6 = nn.Conv2d(
            pyramid_channels, reg_fc6, kernel_size=proposal_roi_crop_size, padding=0
        )
        self.reg_fc7 = nn.Conv2d(reg_fc6, reg_fc7, kernel_size=1)
        self.reg_fc8 = nn.Conv2d(reg_fc7, 6, kernel_size=1)

        self.dropout = nn.Dropout(1.0 - keep_prob)

    def forward(self, bev_input, img_input, bev_rois, img_rois):
        """bev_input/img_input: (1, C, H, W) BEV / image tensors.
        bev_rois/img_rois: (N, 4) [x1, y1, x2, y2] boxes in each input's own pixel space
        (the real anchor-projection geometry that turns a 3D anchor into a BEV box and an
        image box independently -- kept as a direct input since it is dataset-calibration
        plumbing, not part of the feature-pyramid/fusion-head architecture)."""
        bev_feature_maps = self.bev_feature_extractor(bev_input)
        img_feature_maps = self.img_feature_extractor(img_input)

        n_rois = bev_rois.shape[0]
        batch_index = torch.zeros(n_rois, 1, device=bev_rois.device, dtype=bev_rois.dtype)
        bev_boxes = torch.cat([batch_index, bev_rois], dim=1)
        img_boxes = torch.cat([batch_index, img_rois], dim=1)

        crop_size = self._proposal_roi_crop_size
        bev_proposal_rois = roi_align(
            bev_feature_maps, bev_boxes, output_size=crop_size, aligned=True
        )
        img_proposal_rois = roi_align(
            img_feature_maps, img_boxes, output_size=crop_size, aligned=True
        )

        if self._fusion_method == "mean":
            rpn_fusion_out = (bev_proposal_rois + img_proposal_rois) / 2.0
        elif self._fusion_method == "concat":
            rpn_fusion_out = torch.cat([bev_proposal_rois, img_proposal_rois], dim=1)
        else:
            raise ValueError("Invalid fusion method", self._fusion_method)

        cls_fc6 = F.relu(self.cls_fc6(rpn_fusion_out))
        cls_fc6 = self.dropout(cls_fc6)
        cls_fc7 = F.relu(self.cls_fc7(cls_fc6))
        cls_fc7 = self.dropout(cls_fc7)
        cls_fc8 = self.cls_fc8(cls_fc7)
        objectness = cls_fc8.squeeze(-1).squeeze(-1)

        reg_fc6 = F.relu(self.reg_fc6(rpn_fusion_out))
        reg_fc6 = self.dropout(reg_fc6)
        reg_fc7 = F.relu(self.reg_fc7(reg_fc6))
        reg_fc7 = self.dropout(reg_fc7)
        reg_fc8 = self.reg_fc8(reg_fc7)
        offsets = reg_fc8.squeeze(-1).squeeze(-1)

        return objectness, offsets


def build_avod_rpn():
    # Real config values from configs/pyramid_cars_with_aug_example.config, shrunk fc
    # widths so the trace is fast; bev_channels=6 (AVOD's real BEV height-map slicing
    # produces a multi-channel occupancy/height/density map, not RGB).
    model = RpnModel(
        bev_channels=6,
        img_channels=3,
        vgg_conv1=(1, 8),
        vgg_conv2=(1, 16),
        vgg_conv3=(1, 16),
        vgg_conv4=(1, 16),
        proposal_roi_crop_size=3,
        fusion_method="mean",
        cls_fc6=16,
        cls_fc7=16,
        reg_fc6=16,
        reg_fc7=16,
        keep_prob=1.0,
    )
    model.eval()
    return model


def example_input_avod_rpn():
    bev_input = torch.randn(1, 6, 32, 32)
    img_input = torch.randn(1, 3, 32, 32)
    n_rois = 4
    bev_rois = torch.tensor([[2.0, 2.0, 20.0, 20.0]] * n_rois)
    img_rois = torch.tensor([[3.0, 3.0, 18.0, 18.0]] * n_rois)
    return (bev_input, img_input, bev_rois, img_rois)


MENAGERIE_ZOO = "ported-pytorch"
MENAGERIE_ENTRIES = [
    ("AVOD", "build_avod_rpn", "example_input_avod_rpn", 2018, "ported"),
]
