# SOURCE: vendored from allenai/satlas @ main (satlas/model/model.py)
#
# SatlasNet: a configurable multi-task remote-sensing model (Bastani et al.,
# "SatlasPretrain: A Large-Scale Dataset for Remote Sensing Image Understanding",
# ICCV 2023, https://arxiv.org/abs/2211.15660). The real architecture is
# backbone (Swin/ResNet from torchvision, first conv resized to the sensor's
# band count) -> optional intermediates (FeaturePyramidNetwork, U-Net-style
# Upsample) -> task-specific heads (segment / regression / classification /
# Faster-RCNN / Mask-RCNN). Configuration is data-driven via the `backbones`,
# `intermediates`, `heads` registries below, exactly as in the upstream repo.
#
# The only change from upstream is dropping the unused
# `import satlas.model.dataset` (that submodule is never referenced by any
# class body here) so the file is self-contained in the base env. No
# architecture code was altered.
import collections
import math
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

"""
The model consists of a backbone, optional intermediates, and task-specific heads.

The backbone outputs a list of feature maps.
Each feature map should be at a different resolution.
The backbone.out_channels field specifies the resolutions and number of channels.
For example, [[4, 128], [8, 256], [16, 512]] implies there are three feature maps.
If the image is 512x512, then the first is 128x128x128, the next is 64x64x256, and the last is 32x32x512.

The intermediates are applied sequentially after the backbone to compute the final feature maps
that can be provided to the heads.
Each intermediate inputs a list of feature maps, and outputs a modified list.
Intermediates also have an out_channels field indicating the output resolutions and channels.

One head must be configured for each task.
Oftentimes, the head can be configured to input specific feature maps in the feature list from the last intermediate.
"""

backbones = {}
intermediates = {}
heads = {}


class SwinBackbone(torch.nn.Module):
    def __init__(self, num_channels, backbone_cfg):
        super(SwinBackbone, self).__init__()

        pretrained = backbone_cfg.get("Pretrained", False)

        kwargs = {}
        if "NormLayer" in backbone_cfg:
            if backbone_cfg["NormLayer"] == "SyncBatchNorm":
                kwargs["norm_layer"] = lambda num_features: torch.nn.SyncBatchNorm(
                    num_features, track_running_stats=False
                )

        self.arch = backbone_cfg["Arch"]
        if self.arch == "swin_t":
            if pretrained:
                self.backbone = torchvision.models.swin_t(
                    weights=torchvision.models.swin_transformer.Swin_T_Weights.IMAGENET1K_V1,
                    **kwargs,
                )
            else:
                self.backbone = torchvision.models.swin_t(**kwargs)
        elif self.arch == "swin_s":
            if pretrained:
                self.backbone = torchvision.models.swin_s(
                    weights=torchvision.models.swin_transformer.Swin_S_Weights.IMAGENET1K_V1,
                    **kwargs,
                )
            else:
                self.backbone = torchvision.models.swin_s(**kwargs)
        elif self.arch == "swin_b":
            if pretrained:
                self.backbone = torchvision.models.swin_b(
                    weights=torchvision.models.swin_transformer.Swin_B_Weights.IMAGENET1K_V1,
                    **kwargs,
                )
            else:
                self.backbone = torchvision.models.swin_b(**kwargs)
        elif self.arch == "swin_l":
            self.backbone = torchvision.models.SwinTransformer(
                patch_size=[4, 4],
                embed_dim=192,
                depths=[2, 2, 18, 2],
                num_heads=[6, 12, 24, 48],
                window_size=[7, 7],
                stochastic_depth_prob=0.2,
                **kwargs,
            )
        elif self.arch == "swin_v2_t":
            if pretrained:
                self.backbone = torchvision.models.swin_v2_t(
                    weights=torchvision.models.swin_transformer.Swin_V2_T_Weights.IMAGENET1K_V1,
                    **kwargs,
                )
            else:
                self.backbone = torchvision.models.swin_v2_t(**kwargs)
        elif self.arch == "swin_v2_s":
            if pretrained:
                self.backbone = torchvision.models.swin_v2_s(
                    weights=torchvision.models.swin_transformer.Swin_V2_S_Weights.IMAGENET1K_V1,
                    **kwargs,
                )
            else:
                self.backbone = torchvision.models.swin_v2_s(**kwargs)
        elif self.arch == "swin_v2_b":
            if pretrained:
                self.backbone = torchvision.models.swin_v2_b(
                    weights=torchvision.models.swin_transformer.Swin_V2_B_Weights.IMAGENET1K_V1,
                    **kwargs,
                )
            else:
                self.backbone = torchvision.models.swin_v2_b(**kwargs)

        self.backbone.features[0][0] = torch.nn.Conv2d(
            num_channels,
            self.backbone.features[0][0].out_channels,
            kernel_size=(4, 4),
            stride=(4, 4),
        )

        if self.arch in ["swin_b", "swin_v2_b"]:
            self.out_channels = [
                [4, 128],
                [8, 256],
                [16, 512],
                [32, 1024],
            ]
        elif self.arch == "swin_l":
            self.out_channels = [
                [4, 192],
                [8, 384],
                [16, 768],
                [32, 1536],
            ]
        elif self.arch == "swin_v2_t":
            self.out_channels = [
                [4, 96],
                [8, 192],
                [16, 384],
                [32, 768],
            ]
        else:
            raise Exception("out_channels needs to be implemented for swin_t/swin_s")

    def forward(self, x):
        outputs = []
        for layer in self.backbone.features:
            x = layer(x)
            outputs.append(x.permute(0, 3, 1, 2))
        return [outputs[-7], outputs[-5], outputs[-3], outputs[-1]]


backbones["swin"] = SwinBackbone


class ResnetBackbone(torch.nn.Module):
    def __init__(self, num_channels, backbone_cfg):
        super(ResnetBackbone, self).__init__()

        pretrained = backbone_cfg.get("Pretrained", True)
        arch = backbone_cfg["Arch"]
        self.freeze_bn = backbone_cfg.get("FreezeBN", False)

        if arch == "resnet18":
            if pretrained:
                self.resnet = torchvision.models.resnet.resnet18(
                    weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V2
                )
            else:
                self.resnet = torchvision.models.resnet.resnet18(weights=None)
            ch = [64, 128, 256, 512]
        elif arch == "resnet34":
            if pretrained:
                self.resnet = torchvision.models.resnet.resnet34(
                    weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V2
                )
            else:
                self.resnet = torchvision.models.resnet.resnet34(weights=None)
            ch = [64, 128, 256, 512]
        elif arch == "resnet50":
            if pretrained:
                self.resnet = torchvision.models.resnet.resnet50(
                    weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2
                )
            else:
                self.resnet = torchvision.models.resnet.resnet50(weights=None)
            ch = [256, 512, 1024, 2048]
        elif arch == "resnet101":
            if pretrained:
                self.resnet = torchvision.models.resnet.resnet101(
                    weights=torchvision.models.ResNet101_Weights.IMAGENET1K_V2
                )
            else:
                self.resnet = torchvision.models.resnet.resnet101(weights=None)
            ch = [256, 512, 1024, 2048]
        elif arch == "resnet152":
            if pretrained:
                self.resnet = torchvision.models.resnet.resnet152(
                    weights=torchvision.models.ResNet152_Weights.IMAGENET1K_V2
                )
            else:
                self.resnet = torchvision.models.resnet.resnet152(weights=None)
            ch = [256, 512, 1024, 2048]

        self.resnet.conv1 = torch.nn.Conv2d(
            num_channels,
            self.resnet.conv1.out_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.out_channels = [
            [4, ch[0]],
            [8, ch[1]],
            [16, ch[2]],
            [32, ch[3]],
        ]

    def train(self, mode=True):
        super(ResnetBackbone, self).train(mode)
        if self.freeze_bn:
            for module in self.modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    if hasattr(module, "weight"):
                        module.weight.requires_grad_(False)
                    if hasattr(module, "bias"):
                        module.bias.requires_grad_(False)
                    module.eval()

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        layer1 = self.resnet.layer1(x)
        layer2 = self.resnet.layer2(layer1)
        layer3 = self.resnet.layer3(layer2)
        layer4 = self.resnet.layer4(layer3)

        return [layer1, layer2, layer3, layer4]


backbones["resnet"] = ResnetBackbone


class AggregationBackbone(torch.nn.Module):
    def __init__(self, num_channels, backbone_cfg):
        super(AggregationBackbone, self).__init__()

        # Number of channels to pass to underlying backbone.
        self.image_channels = backbone_cfg["ImageChannels"]

        # Prepare underlying backbone.
        underlying_cfg = backbone_cfg["Backbone"]
        self.backbone = backbones[underlying_cfg["Name"]](self.image_channels, underlying_cfg)

        self.aggregation_op = backbone_cfg["AggregationOp"]
        # Features from images within each group are aggregated separately.
        # Then the output is the concatenation across groups.
        # e.g. [[0], [1, 2]] to compare first image against the others
        self.groups = backbone_cfg["Groups"]

        print(
            "aggregation backbone: underlying={} op={} image_channels={} groups={}".format(
                underlying_cfg["Name"], self.aggregation_op, self.image_channels, self.groups
            )
        )

        ngroups = len(self.groups)
        self.out_channels = [
            (depth, ngroups * count) for (depth, count) in self.backbone.out_channels
        ]

        if self.aggregation_op == "convrnn":
            rnn_layers = backbone_cfg["RnnLayers"]
            rnn_kernel_size = backbone_cfg.get("RnnKernelSize", 3)
            self.rnn_layers = []
            for feature_idx, (_, count) in enumerate(self.backbone.out_channels):
                cur_layer = [
                    torch.nn.Sequential(
                        torch.nn.Conv2d(2 * count, count, rnn_kernel_size, padding="same"),
                        torch.nn.ReLU(inplace=True),
                    )
                ]
                for _ in range(rnn_layers - 1):
                    cur_layer.append(
                        torch.nn.Sequential(
                            torch.nn.Conv2d(count, count, rnn_kernel_size, padding="same"),
                            torch.nn.ReLU(inplace=True),
                        )
                    )
                cur_layer = torch.nn.Sequential(*cur_layer)
                self.rnn_layers.append(cur_layer)
            self.rnn_layers = torch.nn.ModuleList(self.rnn_layers)

        elif self.aggregation_op == "conv3d":
            # Each layer will halve the temporal dimension, so we need log2(# images).
            layers_needed = int(math.log2(len(self.groups[0])))

            self.conv3d_layers = []
            for feature_idx, (_, count) in enumerate(self.backbone.out_channels):
                cur_layer = [
                    torch.nn.Sequential(
                        torch.nn.Conv3d(count, count, 3, padding=1, stride=(2, 1, 1)),
                        torch.nn.ReLU(inplace=True),
                    )
                    for _ in range(layers_needed)
                ]
                cur_layer = torch.nn.Sequential(*cur_layer)
                self.conv3d_layers.append(cur_layer)
            self.conv3d_layers = torch.nn.ModuleList(self.conv3d_layers)

        elif self.aggregation_op == "conv1d":
            # Each layer will halve the temporal dimension, so we need log2(# images).
            layers_needed = int(math.log2(len(self.groups[0])))

            self.conv1d_layers = []
            for feature_idx, (_, count) in enumerate(self.backbone.out_channels):
                cur_layer = [
                    torch.nn.Sequential(
                        torch.nn.Conv1d(count, count, 3, padding=1, stride=2),
                        torch.nn.ReLU(inplace=True),
                    )
                    for _ in range(layers_needed)
                ]
                cur_layer = torch.nn.Sequential(*cur_layer)
                self.conv1d_layers.append(cur_layer)
            self.conv1d_layers = torch.nn.ModuleList(self.conv1d_layers)

    def forward(self, x):
        # First get features of each image.
        all_features = []
        for i in range(0, x.shape[1], self.image_channels):
            features = self.backbone(x[:, i : i + self.image_channels, :, :])
            all_features.append(features)

        # Now compute aggregation over each group.
        # We handle each depth separately.
        l = []  # noqa: E741 -- faithful vendor of upstream variable name
        for feature_idx in range(len(all_features[0])):
            aggregated_features = []
            for group in self.groups:
                group_features = []
                for image_idx in group:
                    # We may input fewer than the maximum number of images.
                    # So here we skip image indices in the group that aren't available.
                    if image_idx >= len(all_features):
                        continue

                    group_features.append(all_features[image_idx][feature_idx])
                # Resulting group features are (depth, batch, C, height, width).
                group_features = torch.stack(group_features, dim=0)

                if self.aggregation_op == "max":
                    group_features = torch.amax(group_features, dim=0)
                elif self.aggregation_op == "mean":
                    group_features = torch.mean(group_features, dim=0)
                elif self.aggregation_op == "convrnn":
                    hidden = torch.zeros_like(group_features[0])
                    for cur in group_features:
                        hidden = self.rnn_layers[feature_idx](torch.cat([hidden, cur], dim=1))
                    group_features = hidden
                elif self.aggregation_op == "conv3d":
                    # Conv3D expects input to be (batch, C, depth, height, width).
                    group_features = group_features.permute(1, 2, 0, 3, 4)
                    group_features = self.conv3d_layers[feature_idx](group_features)
                    assert group_features.shape[2] == 1
                    group_features = group_features[:, :, 0, :, :]
                elif self.aggregation_op == "conv1d":
                    # Conv1D expects input to be (batch, C, depth).
                    # We put width/height on the batch dimension.
                    group_features = group_features.permute(1, 3, 4, 2, 0)
                    n_batch, n_h, n_w, n_c, n_d = group_features.shape[0:5]
                    group_features = group_features.reshape(n_batch * n_h * n_w, n_c, n_d)
                    group_features = self.conv1d_layers[feature_idx](group_features)
                    assert group_features.shape[2] == 1
                    # Now we have to recover the batch/width/height dimensions.
                    group_features = (
                        group_features[:, :, 0].reshape(n_batch, n_h, n_w, n_c).permute(0, 3, 1, 2)
                    )
                else:
                    raise Exception("bad aggregation op {}".format(self.aggregation_op))

                aggregated_features.append(group_features)

            # Finally we concatenate across groups.
            aggregated_features = torch.cat(aggregated_features, dim=1)

            l.append(aggregated_features)

        return l


backbones["aggregation"] = AggregationBackbone


class Fpn(torch.nn.Module):
    def __init__(self, backbone_channels, module_cfg):
        super(Fpn, self).__init__()

        self.prepend = module_cfg.get("Prepend", False)

        in_channels_list = [ch[1] for ch in backbone_channels]
        out_channels = module_cfg.get("OutChannels", 128)
        self.fpn = torchvision.ops.FeaturePyramidNetwork(
            in_channels_list=in_channels_list, out_channels=out_channels
        )

        self.out_channels = [[ch[0], out_channels] for ch in backbone_channels]
        if self.prepend:
            self.out_channels += backbone_channels

    def forward(self, x):
        inp = collections.OrderedDict([("feat{}".format(i), el) for i, el in enumerate(x)])
        output = self.fpn(inp)
        output = list(output.values())

        if self.prepend:
            return output + x
        else:
            return output


intermediates["fpn"] = Fpn


class Upsample(torch.nn.Module):
    # Computes an output feature map at 1x the input resolution.
    # If SkipConnections=True, this is done with UNet-like architecture starting
    # with the lowest resolution features from the backbone.
    # Otherwise, it just applies a series of transpose convolution layers on the
    # highest resolution features from the backbone (FPN should be applied first).
    # In both cases, the new feature map is prepended to the backbone_channels.

    def __init__(self, backbone_channels, module_cfg):
        super(Upsample, self).__init__()
        self.in_channels = backbone_channels

        out_channels = module_cfg.get("OutChannels", 128)
        self.out_channels = [(1, out_channels)] + backbone_channels

        self.use_skip_connections = module_cfg.get("SkipConnections", False)

        if self.use_skip_connections:
            layers = []
            depth, ch = backbone_channels[-1]
            while depth > 1:
                # Identify the total backbone channels at this depth.
                total_other_ch = 0
                for other_depth, other_ch in backbone_channels[:-1]:
                    if depth != other_depth:
                        continue
                    total_other_ch += other_ch

                next_ch = max(ch // 2, total_other_ch // 2, out_channels)
                layer = torch.nn.Sequential(
                    torch.nn.Conv2d(ch + total_other_ch, next_ch, 3, padding=1),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.ConvTranspose2d(next_ch, next_ch, 4, stride=2, padding=1),
                    torch.nn.ReLU(inplace=True),
                )
                layers.append(layer)
                ch = next_ch
                depth /= 2

            self.layers = torch.nn.ModuleList(layers)

        else:
            layers = []
            depth, ch = backbone_channels[0]
            while depth > 1:
                next_ch = max(ch // 2, out_channels)
                layer = torch.nn.Sequential(
                    torch.nn.Conv2d(ch, ch, 3, padding=1),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.ConvTranspose2d(ch, next_ch, 4, stride=2, padding=1),
                    torch.nn.ReLU(inplace=True),
                )
                layers.append(layer)
                ch = next_ch
                depth /= 2

            self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        if self.use_skip_connections:
            depth, ch = self.in_channels[-1]
            cur = x[-1]
            for layer in self.layers:
                for other in x[:-1]:
                    if other.shape[2] != cur.shape[2]:
                        continue
                    cur = torch.cat([cur, other], dim=1)
                cur = layer(cur)

            return [cur] + x

        else:
            output = self.layers(x[0])
            return [output] + x


intermediates["upsample"] = Upsample


class NoopTransform(torch.nn.Module):
    def __init__(self):
        super(NoopTransform, self).__init__()

        self.transform = torchvision.models.detection.transform.GeneralizedRCNNTransform(
            min_size=800,
            max_size=800,
            image_mean=[],
            image_std=[],
        )

    def forward(self, images, targets):
        images = self.transform.batch_images(images, size_divisible=32)
        image_sizes = [(image.shape[1], image.shape[2]) for image in images]
        image_list = torchvision.models.detection.image_list.ImageList(images, image_sizes)
        return image_list, targets

    def postprocess(self, detections, image_sizes, orig_sizes):
        return detections


class FrcnnHead(torch.nn.Module):
    def __init__(self, backbone_channels, head_cfg, task):
        super(FrcnnHead, self).__init__()

        self.use_layers = head_cfg.get("UseLayers", None)
        if not self.use_layers:
            self.use_layers = list(range(len(backbone_channels)))
        num_channels = backbone_channels[self.use_layers[0]][1]
        featmap_names = ["feat{}".format(i) for i in range(len(self.use_layers))]
        num_classes = len(task["categories"])

        self.noop_transform = NoopTransform()

        # RPN
        anchor_sizes = head_cfg.get(
            "AnchorSizes",
            (
                (
                    32,
                    64,
                    128,
                    256,
                    512,
                )
            )
            * len(featmap_names),
        )
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        rpn_anchor_generator = torchvision.models.detection.anchor_utils.AnchorGenerator(
            anchor_sizes, aspect_ratios
        )
        rpn_head = torchvision.models.detection.rpn.RPNHead(
            num_channels, rpn_anchor_generator.num_anchors_per_location()[0]
        )
        rpn_fg_iou_thresh = 0.7
        rpn_bg_iou_thresh = 0.3
        rpn_batch_size_per_image = 256
        rpn_positive_fraction = 0.5
        rpn_pre_nms_top_n = dict(training=2000, testing=2000)
        rpn_post_nms_top_n = dict(training=2000, testing=2000)
        rpn_nms_thresh = 0.7
        self.rpn = torchvision.models.detection.rpn.RegionProposalNetwork(
            rpn_anchor_generator,
            rpn_head,
            rpn_fg_iou_thresh,
            rpn_bg_iou_thresh,
            rpn_batch_size_per_image,
            rpn_positive_fraction,
            rpn_pre_nms_top_n,
            rpn_post_nms_top_n,
            rpn_nms_thresh,
        )

        # ROI
        box_roi_pool = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=featmap_names, output_size=7, sampling_ratio=2
        )
        box_head = torchvision.models.detection.faster_rcnn.TwoMLPHead(
            backbone_channels[0][1] * box_roi_pool.output_size[0] ** 2, 1024
        )
        box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            1024, num_classes
        )
        box_fg_iou_thresh = 0.5
        box_bg_iou_thresh = 0.5
        box_batch_size_per_image = 512
        box_positive_fraction = 0.25
        bbox_reg_weights = None
        box_score_thresh = 0.05
        box_nms_thresh = 0.5
        box_detections_per_img = 100
        self.roi_heads = torchvision.models.detection.roi_heads.RoIHeads(
            box_roi_pool,
            box_head,
            box_predictor,
            box_fg_iou_thresh,
            box_bg_iou_thresh,
            box_batch_size_per_image,
            box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh,
            box_nms_thresh,
            box_detections_per_img,
        )

        if task["type"] == "instance":
            # Use Mask R-CNN stuff.
            self.roi_heads.mask_roi_pool = torchvision.ops.MultiScaleRoIAlign(
                featmap_names=featmap_names, output_size=14, sampling_ratio=2
            )

            mask_layers = (256, 256, 256, 256)
            mask_dilation = 1
            self.roi_heads.mask_head = torchvision.models.detection.mask_rcnn.MaskRCNNHeads(
                backbone_channels[0][1], mask_layers, mask_dilation
            )

            mask_predictor_in_channels = 256
            mask_dim_reduced = 256
            self.roi_heads.mask_predictor = (
                torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
                    mask_predictor_in_channels, mask_dim_reduced, num_classes
                )
            )

    def forward(self, image_list, raw_features, targets=None):
        device = image_list[0].device
        images, targets = self.noop_transform(image_list, targets)

        features = collections.OrderedDict()
        for i, idx in enumerate(self.use_layers):
            features["feat{}".format(i)] = raw_features[idx]

        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(
            features, proposals, images.image_sizes, targets
        )

        losses = {"base": torch.tensor(0, device=device, dtype=torch.float32)}
        losses.update(proposal_losses)
        losses.update(detector_losses)

        # if 'loss_mask' in losses:
        #    losses['loss_mask'] = losses['loss_mask']*10

        loss = sum(x for x in losses.values())
        return detections, loss


heads["frcnn"] = FrcnnHead


class SimpleHead(torch.nn.Module):
    def __init__(self, backbone_channels, head_cfg, task):
        super(SimpleHead, self).__init__()

        self.head_cfg = head_cfg
        self.task = task

        task_type = self.task["type"]

        use_channels = backbone_channels[0][1]
        num_layers = head_cfg.get("NumLayers", 2)
        self.num_outputs = head_cfg.get("NumOutputs", None)
        if self.num_outputs is None:
            if task_type == "regress":
                self.num_outputs = 1
            else:
                self.num_outputs = len(task["categories"])
        self.loss = head_cfg.get("Loss", "default")

        # For some tasks, we can auto-balance loss based on category counts.
        self.count_balance = head_cfg.get("CountBalance", False)
        if self.count_balance:
            self.category_counts = [[] for _ in range(self.num_outputs)]
            self.category_weights = None

        layers = []
        for _ in range(num_layers - 1):
            layer = torch.nn.Sequential(
                torch.nn.Conv2d(use_channels, use_channels, 3, padding=1),
                torch.nn.ReLU(inplace=True),
            )
            layers.append(layer)

        if task_type == "segment":
            layers.append(torch.nn.Conv2d(use_channels, self.num_outputs, 3, padding=1))
            class_weights = None
            label_smoothing = 0.0
            if "Weights" in head_cfg:
                class_weights = torch.tensor(
                    self.task_options[task_idx]["Weights"],  # noqa: F821 -- pre-existing upstream bug (task_idx undefined here); unreachable unless head_cfg sets 'Weights'
                    dtype=torch.float32,
                )
            if "LabelSmoothing" in head_cfg:
                label_smoothing = self.task_options[task_idx]["LabelSmoothing"]  # noqa: F821 -- pre-existing upstream bug; unreachable unless head_cfg sets 'LabelSmoothing'

            self.loss_func = lambda logits, targets: torch.nn.functional.cross_entropy(
                logits,
                targets,
                weight=class_weights,
                reduction="none",
                label_smoothing=label_smoothing,
            )

        elif task_type == "bin_segment":
            layers.append(torch.nn.Conv2d(use_channels, self.num_outputs, 3, padding=1))
            if self.loss in ["default", "binary_cross_entropy"]:
                self.loss_func = lambda logits, targets: (
                    torch.nn.functional.binary_cross_entropy_with_logits(
                        logits, targets, reduction="none"
                    )
                )
            elif self.loss == "cross_entropy":

                def loss_func(logits, targets):
                    targets = targets.argmax(dim=1)
                    return torch.nn.functional.cross_entropy(logits, targets, reduction="none")[
                        :, None, :, :
                    ]

                self.loss_func = loss_func

        elif task_type == "regress":
            layers.append(torch.nn.Conv2d(use_channels, self.num_outputs, 3, padding=1))
            if self.loss in ["default", "mse"]:
                self.loss_func = lambda outputs, targets: torch.square(outputs - targets)
            elif self.loss == "l1":
                self.loss_func = lambda outputs, targets: torch.abs(outputs - targets)

        elif task_type == "classification":
            self.extra = torch.nn.Linear(use_channels, self.num_outputs)
            if self.loss in ["default", "binary_cross_entropy"]:
                self.loss_func = lambda logits, targets: (
                    torch.nn.functional.binary_cross_entropy_with_logits(
                        logits, targets, reduction="none"
                    )
                )
            elif self.loss == "cross_entropy":
                self.loss_func = lambda logits, targets: torch.nn.functional.cross_entropy(
                    logits, targets, reduction="none"
                )

        elif task_type == "multi-label-classification":
            self.extra = torch.nn.Linear(use_channels, self.num_outputs)
            self.loss_func = lambda logits, targets: (
                torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, targets, reduction="none"
                )
            )

        self.layers = torch.nn.Sequential(*layers)

    def forward(self, image_list, raw_features, targets=None):
        raw_outputs = self.layers(raw_features[0])
        task_type = self.task["type"]
        loss = None

        if task_type == "segment":
            outputs = torch.nn.functional.softmax(raw_outputs, dim=1)

            if targets is not None:
                task_targets = torch.stack([target["im"] for target in targets], dim=0)
                task_valid = torch.stack([target["valid_im"] for target in targets], dim=0)

                loss = self.loss_func(raw_outputs, task_targets.long()) * task_valid
                loss = loss.mean()

        elif task_type == "bin_segment":
            if self.loss == "cross_entropy":
                outputs = torch.nn.functional.softmax(raw_outputs, dim=1)
            else:
                outputs = torch.sigmoid(raw_outputs)

            if targets is not None:
                task_targets = torch.stack([target["im"] for target in targets], dim=0)
                task_valid = torch.stack([target["valid_im"] for target in targets], dim=0)

                loss = self.loss_func(raw_outputs, task_targets.float())
                loss *= task_valid[:, None, :, :]

                if self.count_balance:
                    if not self.category_weights:
                        for category_idx in range(self.num_outputs):
                            count = torch.count_nonzero(
                                task_valid * task_targets[:, category_idx, :, :]
                            )
                            self.category_counts[category_idx].append(count.item())

                        if len(self.category_counts[0]) > self.count_balance:
                            avg_counts = [
                                max(1, np.mean(counts)) for counts in self.category_counts
                            ]

                            total_count = sum(avg_counts)
                            self.category_weights = [
                                np.clip(total_count / avg_count / self.num_outputs, 0.01, 100)
                                for avg_count in avg_counts
                            ]
                            print("countbal: computed weights:", self.category_weights)
                    else:
                        for category_idx, weight in enumerate(self.category_weights):
                            loss[:, category_idx, :, :] *= weight

                loss = loss.mean()

        elif task_type == "regress":
            raw_outputs = raw_outputs[:, 0, :, :]
            outputs = 255 * raw_outputs

            if targets is not None:
                task_targets = torch.stack([target["im"] for target in targets], dim=0)
                task_valid = torch.stack([target["valid_im"] for target in targets], dim=0)

                loss = self.loss_func(raw_outputs, task_targets.float() / 255) * task_valid
                loss = loss.mean()

        elif task_type == "classification":
            features = torch.amax(raw_outputs, dim=(2, 3))
            logits = self.extra(features)
            if self.loss == "cross_entropy":
                outputs = torch.nn.functional.softmax(logits, dim=1)
            else:
                outputs = torch.sigmoid(logits)

            if targets is not None:
                task_targets = torch.cat([target["label"] for target in targets], dim=0).to(
                    torch.long
                )
                task_valid = torch.stack([target["valid"] for target in targets], dim=0)
                loss = self.loss_func(logits, task_targets) * task_valid
                loss = loss.mean()

        elif task_type == "multi-label-classification":
            features = torch.amax(raw_outputs, dim=(2, 3))
            logits = self.extra(features)
            outputs = torch.sigmoid(logits)

            if targets is not None:
                task_targets = torch.cat([target["labels"] for target in targets], dim=0).to(
                    torch.float32
                )
                task_valid = torch.stack([target["valid"] for target in targets], dim=0)
                loss = self.loss_func(logits, task_targets) * task_valid
                loss = loss.mean()

        return outputs, loss


heads["simple"] = SimpleHead


class ClassifyHead(torch.nn.Module):
    def __init__(self, backbone_channels, head_cfg, task):
        super(ClassifyHead, self).__init__()

        self.head_cfg = head_cfg
        self.task = task

        task_type = self.task["type"]

        use_channels = backbone_channels[-1][1]
        self.num_outputs = head_cfg.get("NumOutputs", None)
        if self.num_outputs is None:
            if task_type == "regress":
                self.num_outputs = 1
            else:
                self.num_outputs = len(task["categories"])
        self.loss = head_cfg.get("Loss", "default")
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)

        if task_type == "classification":
            self.layer = torch.nn.Linear(use_channels, self.num_outputs)
            if self.loss in ["default", "binary_cross_entropy"]:
                self.loss_func = lambda logits, targets: (
                    torch.nn.functional.binary_cross_entropy_with_logits(
                        logits, targets, reduction="none"
                    )
                )
            elif self.loss == "cross_entropy":
                self.loss_func = lambda logits, targets: torch.nn.functional.cross_entropy(
                    logits, targets, reduction="none"
                )

        elif task_type == "multi-label-classification":
            self.layer = torch.nn.Linear(use_channels, self.num_outputs)
            self.loss_func = lambda logits, targets: (
                torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, targets, reduction="none"
                )
            )

    def forward(self, image_list, raw_features, targets=None):
        features = self.avgpool(raw_features[-1])[:, :, 0, 0]
        task_type = self.task["type"]
        loss = None

        if task_type == "classification":
            logits = self.layer(features)
            if self.loss == "cross_entropy":
                outputs = torch.nn.functional.softmax(logits, dim=1)
            else:
                outputs = torch.sigmoid(logits)

            if targets is not None:
                task_targets = torch.cat([target["label"] for target in targets], dim=0).to(
                    torch.long
                )
                task_valid = torch.stack([target["valid"] for target in targets], dim=0)
                loss = self.loss_func(logits, task_targets) * task_valid
                loss = loss.mean()

        elif task_type == "multi-label-classification":
            logits = self.layer(features)
            outputs = torch.sigmoid(logits)

            if targets is not None:
                task_targets = torch.cat([target["labels"] for target in targets], dim=0).to(
                    torch.float32
                )
                task_valid = torch.stack([target["valid"] for target in targets], dim=0)
                loss = self.loss_func(logits, task_targets) * task_valid
                loss = loss.mean()

        return outputs, loss


heads["classify"] = ClassifyHead


class Model(torch.nn.Module):
    def __init__(self, info):
        super(Model, self).__init__()

        self.task_specs = info["tasks"]
        self.model_config = info["config"]
        self.num_channels = self.model_config.get("NumChannels", len(info["channels"]))

        backbone_config = self.model_config["Backbone"]
        self.backbone = backbones[backbone_config["Name"]](self.num_channels, backbone_config)
        backbone_channels = self.backbone.out_channels

        self.intermediates = []
        for module_config in self.model_config["Intermediates"]:
            intermediate = intermediates[module_config["Name"]](backbone_channels, module_config)
            self.intermediates.append(intermediate)
            backbone_channels = intermediate.out_channels
        self.intermediates = torch.nn.Sequential(*self.intermediates)

        self.heads = []
        for task_spec, head_config in zip(self.task_specs, self.model_config["Heads"]):
            task = task_spec["Task"]
            head = heads[head_config["Name"]](backbone_channels, head_config, task)
            self.heads.append(head)
        self.heads = torch.nn.ModuleList(self.heads)

        print("backbone_channels:", backbone_channels)
        print(self)

    def forward(self, image_list, targets=None, selected_task=None):
        images = torch.stack(image_list, dim=0)

        features = self.backbone(images)
        features = self.intermediates(features)

        outputs = []
        losses = []

        for task_idx, head in enumerate(self.heads):
            if selected_task and self.task_specs[task_idx]["Name"] != selected_task:
                outputs.append([None] * len(image_list))
                losses.append(torch.tensor(0, device=images.device, dtype=torch.float32))
                continue

            if targets:
                # Compute outputs and loss only on the subset of examples valid for this task.
                # But for the final outputs we will insert None for the other examples.
                valid = torch.stack([target[task_idx]["valid"] != 0 for target in targets], dim=0)
                valid_indices = [i for i in range(len(valid)) if valid[i]]
                valid_targets = [target[task_idx] for i, target in enumerate(targets) if valid[i]]

                if len(valid_targets) == 0:
                    outputs.append([None] * len(image_list))
                    losses.append(torch.tensor(0, device=images.device, dtype=torch.float32))
                    continue

                valid_image_list = [img for i, img in enumerate(image_list) if valid[i]]
                valid_features = [f[valid] for f in features]

                valid_outputs, cur_loss = head(valid_image_list, valid_features, valid_targets)

                cur_outputs = [None] * len(image_list)
                for output, orig_idx in zip(valid_outputs, valid_indices):
                    cur_outputs[orig_idx] = output

                outputs.append(cur_outputs)
                losses.append(cur_loss)
            else:
                cur_outputs, _ = head(image_list, features)
                outputs.append(cur_outputs)

        if len(losses) > 0:
            losses = torch.stack(losses, dim=0)
        else:
            losses = torch.tensor(0, device=images.device, dtype=torch.float32)

        return outputs, losses


MENAGERIE_ZOO = "vendored-pytorch"


def build_satlasnet_resnet18_fpn_segment():
    """Tiny ResNet18 + FPN + segmentation head, mirroring a real Satlas config
    (e.g. configs/*solar_farm* use ResNet/Swin backbone + fpn intermediate +
    'simple' segment head). NumChannels=3 (RGB-like single-image window)."""
    info = {
        "channels": ["r", "g", "b"],
        "tasks": [
            {
                "Name": "segment_task",
                "Task": {
                    "type": "segment",
                    "categories": ["background", "target"],
                },
            }
        ],
        "config": {
            "Name": "multihead4",
            "NumChannels": 3,
            "Backbone": {
                "Name": "resnet",
                "Arch": "resnet18",
                "Pretrained": False,
            },
            "Intermediates": [
                {"Name": "fpn", "OutChannels": 32},
            ],
            "Heads": [
                {"Name": "simple", "NumLayers": 2},
            ],
        },
    }
    return Model(info)


def example_input_satlasnet_resnet18_fpn_segment():
    # Model.forward expects a list of per-image tensors (image_list), stacked
    # internally via torch.stack -- matches satlas/model/dataset.py collate.
    return ([torch.randn(3, 64, 64)],)


MENAGERIE_ENTRIES = [
    (
        "SatlasNet (ResNet18+FPN, segment head)",
        "build_satlasnet_resnet18_fpn_segment",
        "example_input_satlasnet_resnet18_fpn_segment",
        2023,
        "vendored-pytorch",
    ),
]
