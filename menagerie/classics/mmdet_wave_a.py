"""Torch-only OpenMMLab MMDetection classics for wave A.

Paper/source coverage: Ren et al. 2015 Faster R-CNN; Lin et al. 2017 FPN,
RetinaNet, and Mask R-CNN; Cai and Vasconcelos 2018 Cascade R-CNN; Chen et al.
2019 Hybrid Task Cascade; Tian et al. 2019 FCOS; Zhou et al. 2019 CenterNet;
Carion et al. 2020 DETR; Zhu et al. 2021 Deformable DETR; Cheng et al. 2021
MaskFormer; Cheng et al. 2022 Mask2Former; Zhang et al. 2022 DINO; and the
MMDetection config families named in ``/tmp/wave_A.txt``.

These modules replace OpenMMLab constructors whose mmcv/mmengine dependencies are
not installable in the base TorchLens environment. They intentionally preserve
the load-bearing inference structure of each detector family with random
initialization and small tensor sizes: convolutional/residual backbones,
multi-scale FPN-like necks, RPN/ROI/cascade/mask heads, dense anchor or
anchor-free heads, DETR-style query decoders, mask-query segmentation, and MOT
ReID/track heads.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class VariantSpec:
    """Configuration for a compact MMDetection classic.

    Parameters
    ----------
    kind:
        Detector family to instantiate.
    backbone:
        Backbone style marker used to select faithful structural features.
    neck:
        Neck/fusion style marker.
    roi:
        ROI or head specialization.
    stages:
        Number of cascade/refinement stages.
    queries:
        Number of object or mask queries.
    classes:
        Number of class logits.
    """

    kind: str
    backbone: str = "resnet"
    neck: str = "fpn"
    roi: str = "box"
    stages: int = 1
    queries: int = 12
    classes: int = 5


def conv_norm_act(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    stride: int = 1,
    groups: int = 1,
) -> nn.Sequential:
    """Create a convolution, normalization, and activation block.

    Parameters
    ----------
    in_channels:
        Number of input channels.
    out_channels:
        Number of output channels.
    kernel_size:
        Convolution kernel size.
    stride:
        Convolution stride.
    groups:
        Convolution group count.

    Returns
    -------
    nn.Sequential
        Convolutional block.
    """

    padding = kernel_size // 2
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels),
        nn.SiLU(inplace=False),
    )


class ResidualBlock(nn.Module):
    """Compact residual block with optional grouped or split convolution."""

    def __init__(self, channels: int, style: str = "resnet") -> None:
        """Initialize residual block.

        Parameters
        ----------
        channels:
            Feature channel count.
        style:
            Structural style marker such as ``res2net``, ``regnet``, or ``dcn``.
        """

        super().__init__()
        groups = 4 if style in {"regnet", "resnext"} else 1
        self.style = style
        self.conv1 = conv_norm_act(channels, channels, 1)
        self.conv2 = conv_norm_act(channels, channels, 3, groups=groups)
        self.conv3 = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False), nn.BatchNorm2d(channels)
        )
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid(),
        )
        self.offset = nn.Conv2d(channels, 2, 3, padding=1) if style == "dcn" else None
        self.scale_gate = nn.Conv2d(channels // 2, channels // 2, 1) if style == "res2net" else None

    def forward(self, x: Tensor) -> Tensor:
        """Apply residual feature refinement.

        Parameters
        ----------
        x:
            Input feature map.

        Returns
        -------
        Tensor
            Refined feature map.
        """

        y = self.conv1(x)
        if self.scale_gate is not None:
            left, right = y.chunk(2, dim=1)
            y = torch.cat((left, right + torch.tanh(self.scale_gate(left)) * right), dim=1)
        if self.offset is not None:
            y = y + 0.05 * torch.tanh(self.offset(y)).mean(dim=1, keepdim=True)
        y = self.conv3(self.conv2(y))
        if self.style in {"gcnet", "resnest"}:
            y = y * self.global_context(y)
        return F.silu(x + y)


class ConvNeXtBlock(nn.Module):
    """ConvNeXt-style depthwise MLP block."""

    def __init__(self, channels: int) -> None:
        """Initialize ConvNeXt block.

        Parameters
        ----------
        channels:
            Feature channel count.
        """

        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, 7, padding=3, groups=channels)
        self.norm = nn.GroupNorm(1, channels)
        self.pw1 = nn.Conv2d(channels, channels * 2, 1)
        self.pw2 = nn.Conv2d(channels * 2, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Apply ConvNeXt residual update.

        Parameters
        ----------
        x:
            Input feature map.

        Returns
        -------
        Tensor
            Updated feature map.
        """

        y = self.depthwise(x)
        y = self.pw2(F.gelu(self.pw1(self.norm(y))))
        return x + y


class TinyBackbone(nn.Module):
    """Backbone factory covering MMDetection CNN variants."""

    def __init__(self, style: str = "resnet", width: int = 24) -> None:
        """Initialize multi-scale backbone.

        Parameters
        ----------
        style:
            Backbone style marker.
        width:
            Base channel count.
        """

        super().__init__()
        self.style = style
        self.stem = conv_norm_act(3, width, 7, stride=2)
        block: nn.Module
        if style == "convnext":
            block = ConvNeXtBlock(width * 2)
        else:
            block = ResidualBlock(width * 2, style=style)
        self.stage2 = nn.Sequential(conv_norm_act(width, width * 2, 3, stride=2), block)
        self.stage3 = nn.Sequential(
            conv_norm_act(width * 2, width * 4, 3, stride=2),
            ConvNeXtBlock(width * 4)
            if style == "convnext"
            else ResidualBlock(width * 4, style=style),
        )
        self.stage4 = nn.Sequential(
            conv_norm_act(width * 4, width * 8, 3, stride=2),
            ConvNeXtBlock(width * 8)
            if style == "convnext"
            else ResidualBlock(width * 8, style=style),
        )
        self.hr_fuse = nn.Conv2d(width * 2 + width * 4 + width * 8, width * 8, 1)
        self.hr_proj3 = nn.Conv2d(width * 8, width * 2, 1)
        self.hr_proj4 = nn.Conv2d(width * 8, width * 4, 1)

    def forward(self, x: Tensor) -> list[Tensor]:
        """Return three backbone feature scales.

        Parameters
        ----------
        x:
            Input RGB image.

        Returns
        -------
        list[Tensor]
            Feature maps ordered high to low resolution.
        """

        x = self.stem(x)
        c3 = self.stage2(x)
        c4 = self.stage3(c3)
        c5 = self.stage4(c4)
        if self.style == "hrnet":
            c4_up = F.interpolate(c4, size=c3.shape[-2:], mode="nearest")
            c5_up = F.interpolate(c5, size=c3.shape[-2:], mode="nearest")
            c5 = self.hr_fuse(torch.cat((c3, c4_up, c5_up), dim=1))
            c4 = self.hr_proj4(F.avg_pool2d(c5, 2))
            c3 = self.hr_proj3(c5)
        return [c3, c4, c5]


class PyramidNeck(nn.Module):
    """FPN, NAS-FPN, RFP, BFP, or YOLO-style neck."""

    def __init__(self, channels: int = 24, style: str = "fpn") -> None:
        """Initialize neck layers.

        Parameters
        ----------
        channels:
            Base backbone channel count.
        style:
            Neck style marker.
        """

        super().__init__()
        self.style = style
        out = channels * 2
        self.lateral = nn.ModuleList(
            [
                nn.Conv2d(channels * 2, out, 1),
                nn.Conv2d(channels * 4, out, 1),
                nn.Conv2d(channels * 8, out, 1),
            ]
        )
        self.smooth = nn.ModuleList([conv_norm_act(out, out) for _ in range(3)])
        self.bfp = conv_norm_act(out, out)
        self.nas_mix = nn.ModuleList([nn.Conv2d(out, out, 3, padding=1) for _ in range(4)])
        self.rfp_conv = conv_norm_act(out, out)

    def forward(self, feats: list[Tensor]) -> list[Tensor]:
        """Fuse backbone features into a pyramid.

        Parameters
        ----------
        feats:
            Backbone feature maps.

        Returns
        -------
        list[Tensor]
            Fused pyramid feature maps.
        """

        c3, c4, c5 = feats
        p5 = self.lateral[2](c5)
        p4 = self.lateral[1](c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.lateral[0](c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        levels = [self.smooth[0](p3), self.smooth[1](p4), self.smooth[2](p5)]
        if self.style == "bfp":
            target = levels[1].shape[-2:]
            pooled = sum(F.adaptive_avg_pool2d(level, target) for level in levels) / len(levels)
            refined = self.bfp(pooled)
            return [
                level + F.interpolate(refined, size=level.shape[-2:], mode="nearest")
                for level in levels
            ]
        if self.style == "nasfpn":
            p3, p4, p5 = levels
            p4 = p4 + self.nas_mix[0](F.max_pool2d(p3, 2))
            p3 = p3 + F.interpolate(self.nas_mix[1](p4), size=p3.shape[-2:], mode="nearest")
            p5 = p5 + self.nas_mix[2](F.max_pool2d(p4, 2))
            p4 = p4 + F.interpolate(self.nas_mix[3](p5), size=p4.shape[-2:], mode="nearest")
            return [p3, p4, p5]
        if self.style == "rfp":
            return [level + self.rfp_conv(level) for level in levels]
        if self.style == "yolox":
            p3, p4, p5 = levels
            p4 = p4 + F.interpolate(p5, size=p4.shape[-2:], mode="nearest")
            p3 = p3 + F.interpolate(p4, size=p3.shape[-2:], mode="nearest")
            return [p3, p4 + F.max_pool2d(p3, 2), p5]
        return levels


class RPNHead(nn.Module):
    """Region proposal head with objectness and box deltas."""

    def __init__(self, channels: int, stages: int = 1) -> None:
        """Initialize RPN layers.

        Parameters
        ----------
        channels:
            Feature channel count.
        stages:
            Number of adaptive RPN refinement stages.
        """

        super().__init__()
        self.refine = nn.ModuleList([conv_norm_act(channels, channels) for _ in range(stages)])
        self.objectness = nn.Conv2d(channels, 3, 1)
        self.box = nn.Conv2d(channels, 12, 1)

    def forward(self, feature: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Predict proposals from one pyramid level.

        Parameters
        ----------
        feature:
            Pyramid feature map.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Refined feature, objectness scores, and box deltas.
        """

        for stage in self.refine:
            feature = feature + stage(feature)
        return feature, self.objectness(feature), F.softplus(self.box(feature))


class ROIExtractor(nn.Module):
    """Traceable top-score ROI feature extractor."""

    def __init__(self, proposals: int = 8) -> None:
        """Initialize extractor.

        Parameters
        ----------
        proposals:
            Number of ROI features to gather.
        """

        super().__init__()
        self.proposals = proposals

    def forward(self, feature: Tensor, objectness: Tensor) -> Tensor:
        """Gather top-scoring spatial features as ROI descriptors.

        Parameters
        ----------
        feature:
            Source feature map.
        objectness:
            RPN objectness logits.

        Returns
        -------
        Tensor
            ROI feature tensor.
        """

        scores = objectness.flatten(2).max(dim=1).values
        topk = torch.topk(scores, self.proposals, dim=1).indices
        flat = feature.flatten(2).transpose(1, 2)
        gather = topk.unsqueeze(-1).expand(-1, -1, flat.shape[-1])
        return torch.gather(flat, 1, gather)


class ROIStage(nn.Module):
    """Second-stage box, mask, and specialization heads."""

    def __init__(self, channels: int, classes: int = 5, roi: str = "box") -> None:
        """Initialize ROI heads.

        Parameters
        ----------
        channels:
            ROI feature width.
        classes:
            Number of classes.
        roi:
            ROI specialization marker.
        """

        super().__init__()
        self.roi = roi
        hidden = channels * 2 if roi == "double" else channels
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden), nn.ReLU(), nn.Linear(hidden, channels), nn.ReLU()
        )
        self.cls = nn.Linear(channels, classes)
        self.box = nn.Linear(channels, 4)
        self.mask = nn.Linear(channels, 16)
        self.quality = nn.Linear(channels, 1)
        self.boundary = nn.Linear(channels, 4)
        self.grid = nn.Linear(channels, 9)
        self.point = nn.Linear(channels, 4)

    def forward(self, roi_features: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Predict ROI outputs.

        Parameters
        ----------
        roi_features:
            Proposal feature tensor.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Class logits, box deltas, and mask or auxiliary logits.
        """

        feat = self.fc(roi_features)
        box = torch.sigmoid(self.box(feat))
        aux = self.mask(feat)
        if self.roi == "mask_iou":
            aux = aux * torch.sigmoid(self.quality(feat))
        elif self.roi == "sabl":
            box = box + 0.1 * torch.tanh(self.boundary(feat))
        elif self.roi == "grid":
            aux = self.grid(feat)
        elif self.roi == "point_rend":
            aux = aux + F.pad(self.point(feat), (0, aux.shape[-1] - 4))
        elif self.roi in {"pisa", "dynamic", "crowd"}:
            feat = feat * torch.sigmoid(self.quality(feat))
        return self.cls(feat), box, aux


class TwoStageDetector(nn.Module):
    """Faster/Mask/Cascade/HTC/SCNet style MMDetection model."""

    def __init__(self, spec: VariantSpec) -> None:
        """Initialize two-stage detector.

        Parameters
        ----------
        spec:
            Variant configuration.
        """

        super().__init__()
        self.spec = spec
        self.backbone = TinyBackbone(spec.backbone)
        self.neck = PyramidNeck(style=spec.neck)
        channels = 48
        rpn_stages = 2 if spec.kind == "cascade_rpn" else 1
        self.rpn = RPNHead(channels, stages=rpn_stages)
        self.extractor = ROIExtractor(proposals=spec.queries)
        self.stages = nn.ModuleList(
            [ROIStage(channels, spec.classes, spec.roi) for _ in range(spec.stages)]
        )
        self.semantic = nn.Conv2d(channels, spec.classes, 1)
        self.track = nn.Linear(channels, 16)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Run two-stage detection and optional mask/semantic heads.

        Parameters
        ----------
        x:
            Input RGB image.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor, Tensor]
            RPN boxes, class logits, ROI boxes, and auxiliary mask/semantic data.
        """

        pyramid = self.neck(self.backbone(x))
        rpn_feature, objectness, rpn_boxes = self.rpn(pyramid[0])
        roi = self.extractor(rpn_feature, objectness)
        cls_out = roi.new_zeros(roi.shape[0], roi.shape[1], self.spec.classes)
        box_out = roi.new_zeros(roi.shape[0], roi.shape[1], 4)
        aux_out = roi.new_zeros(roi.shape[0], roi.shape[1], 16)
        for stage in self.stages:
            cls_out, box_out, aux_out = stage(roi + 0.1 * box_out.mean(dim=-1, keepdim=True))
        if self.spec.kind in {"htc", "scnet", "panoptic"}:
            semantic = F.adaptive_avg_pool2d(self.semantic(pyramid[0]), (4, 4)).flatten(1)
            aux_out = aux_out + semantic[:, None, : aux_out.shape[-1]]
        if self.spec.kind in {"deepsort", "strongsort"}:
            aux_out = aux_out + self.track(roi)
        return rpn_boxes.flatten(2).transpose(1, 2), cls_out, box_out, aux_out


class DenseHead(nn.Module):
    """RetinaNet, FCOS, CenterNet, SSD, and instance-grid dense heads."""

    def __init__(self, channels: int, classes: int = 5, mode: str = "retina") -> None:
        """Initialize dense head.

        Parameters
        ----------
        channels:
            Feature channel count.
        classes:
            Number of classes.
        mode:
            Dense detector variant.
        """

        super().__init__()
        self.mode = mode
        self.cls_tower = nn.Sequential(
            conv_norm_act(channels, channels), conv_norm_act(channels, channels)
        )
        self.box_tower = nn.Sequential(
            conv_norm_act(channels, channels), conv_norm_act(channels, channels)
        )
        self.cls = nn.Conv2d(channels, classes, 3, padding=1)
        self.box = nn.Conv2d(channels, 4, 3, padding=1)
        self.center = nn.Conv2d(channels, 1, 3, padding=1)
        self.corner = nn.Conv2d(channels, 4, 3, padding=1)
        self.mask_kernel = nn.Conv2d(channels, 8, 3, padding=1)
        self.scale = nn.Parameter(torch.ones(3))

    def forward(self, features: list[Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        """Predict dense outputs across pyramid levels.

        Parameters
        ----------
        features:
            Pyramid feature maps.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Class, box, and auxiliary dense predictions.
        """

        cls_items: list[Tensor] = []
        box_items: list[Tensor] = []
        aux_items: list[Tensor] = []
        for level, feature in enumerate(features):
            cls_feature = self.cls_tower(feature)
            box_feature = self.box_tower(feature)
            cls = self.cls(cls_feature)
            box = self.box(box_feature)
            aux = self.center(box_feature)
            if self.mode in {"fcos", "nas_fcos", "ddod"}:
                box = torch.exp(self.scale[level] * box)
            elif self.mode in {"centernet", "cornernet", "centripetal"}:
                aux = torch.cat((self.center(box_feature), self.corner(box_feature)), dim=1)
            elif self.mode in {"condinst", "boxinst", "solo"}:
                aux = self.mask_kernel(box_feature)
            elif self.mode in {"sabl", "ghm"}:
                box = F.softplus(box)
            cls_items.append(cls.flatten(2).transpose(1, 2))
            box_items.append(box.flatten(2).transpose(1, 2))
            aux_items.append(aux.flatten(2).transpose(1, 2))
        return torch.cat(cls_items, dim=1), torch.cat(box_items, dim=1), torch.cat(aux_items, dim=1)


class DenseDetector(nn.Module):
    """One-stage MMDetection model with FPN and dense heads."""

    def __init__(self, spec: VariantSpec) -> None:
        """Initialize dense detector.

        Parameters
        ----------
        spec:
            Variant configuration.
        """

        super().__init__()
        self.backbone = TinyBackbone(spec.backbone)
        self.neck = PyramidNeck(style=spec.neck)
        self.head = DenseHead(48, spec.classes, mode=spec.kind)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Run dense detection.

        Parameters
        ----------
        x:
            Input RGB image.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Class, box, and auxiliary predictions.
        """

        return self.head(self.neck(self.backbone(x)))


class QueryBackbone(nn.Module):
    """Multi-scale tokenizing backbone for DETR-like models."""

    def __init__(self, dim: int = 48, style: str = "resnet") -> None:
        """Initialize token backbone.

        Parameters
        ----------
        dim:
            Token width.
        style:
            Backbone style marker.
        """

        super().__init__()
        self.backbone = TinyBackbone(style, width=24)
        self.project = nn.ModuleList(
            [nn.Conv2d(48, dim, 1), nn.Conv2d(48, dim, 1), nn.Conv2d(48, dim, 1)]
        )
        self.level = nn.Parameter(torch.zeros(3, dim))

    def forward(self, x: Tensor) -> Tensor:
        """Return concatenated image tokens.

        Parameters
        ----------
        x:
            Input RGB image.

        Returns
        -------
        Tensor
            Multi-scale token tensor.
        """

        tokens = []
        for index, (feature, project) in enumerate(
            zip(PyramidNeck()(self.backbone(x)), self.project)
        ):
            token = project(feature).flatten(2).transpose(1, 2)
            tokens.append(token + self.level[index].view(1, 1, -1))
        return torch.cat(tokens, dim=1)


class TransformerBlock(nn.Module):
    """Transformer block with optional cross attention."""

    def __init__(self, dim: int = 48, heads: int = 4, cross: bool = False) -> None:
        """Initialize attention and feed-forward layers.

        Parameters
        ----------
        dim:
            Token width.
        heads:
            Number of attention heads.
        cross:
            Whether to include cross attention.
        """

        super().__init__()
        self.cross = cross
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.ReLU(), nn.Linear(dim * 4, dim))
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x: Tensor, memory: Tensor | None = None) -> Tensor:
        """Apply transformer update.

        Parameters
        ----------
        x:
            Query or memory tokens.
        memory:
            Optional cross-attention memory.

        Returns
        -------
        Tensor
            Updated tokens.
        """

        x = self.norm1(x + self.self_attn(x, x, x)[0])
        if self.cross and memory is not None:
            x = self.norm2(x + self.cross_attn(x, memory, memory)[0])
        return self.norm3(x + self.ffn(x))


class DETRDetector(nn.Module):
    """DETR, Conditional/DAB/Deformable DETR, DDQ, and DINO detector."""

    def __init__(self, spec: VariantSpec) -> None:
        """Initialize query detector.

        Parameters
        ----------
        spec:
            Variant configuration.
        """

        super().__init__()
        dim = 48
        self.spec = spec
        self.tokens = QueryBackbone(dim=dim, style=spec.backbone)
        self.encoder = nn.ModuleList([TransformerBlock(dim) for _ in range(2)])
        self.query = nn.Embedding(spec.queries, dim)
        self.anchor = nn.Embedding(spec.queries, 4)
        self.ref_proj = nn.Linear(dim, 2)
        self.decoder = nn.ModuleList([TransformerBlock(dim, cross=True) for _ in range(2)])
        self.denoise = nn.Linear(dim, dim)
        self.cls = nn.Linear(dim, spec.classes)
        self.box = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 4))
        self.dense_query = nn.Linear(dim, 1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Run query-based object detection.

        Parameters
        ----------
        x:
            Input RGB image.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Query class logits, boxes, and auxiliary query scores.
        """

        memory = self.tokens(x)
        for layer in self.encoder:
            memory = layer(memory)
        batch = x.shape[0]
        queries = self.query.weight.unsqueeze(0).expand(batch, -1, -1)
        if self.spec.kind in {"dab_detr", "dino", "ddq"}:
            queries = queries + self.anchor.weight[:, : queries.shape[-1] % 4].mean() * 0.0
            queries = queries + self.denoise(queries)
        if self.spec.kind in {"deformable_detr", "dino", "ddq"}:
            scores = self.dense_query(memory).squeeze(-1)
            topk = torch.topk(scores, self.spec.queries, dim=1).indices
            selected = torch.gather(memory, 1, topk.unsqueeze(-1).expand(-1, -1, memory.shape[-1]))
            queries = queries + selected + self.ref_proj(selected).mean(dim=-1, keepdim=True)
        for layer in self.decoder:
            queries = layer(queries, memory)
        aux = self.dense_query(queries)
        if self.spec.kind == "dino":
            aux = aux + torch.cosine_similarity(
                queries, self.denoise(queries), dim=-1, eps=1e-6
            ).unsqueeze(-1)
        return self.cls(queries), torch.sigmoid(self.box(queries)), aux


class MaskQuerySegmenter(nn.Module):
    """MaskFormer and Mask2Former mask-classification segmenter."""

    def __init__(self, spec: VariantSpec) -> None:
        """Initialize mask query segmenter.

        Parameters
        ----------
        spec:
            Variant configuration.
        """

        super().__init__()
        dim = 48
        self.spec = spec
        self.backbone = TinyBackbone(spec.backbone)
        self.neck = PyramidNeck()
        self.pixel_decoder = nn.Sequential(conv_norm_act(dim, dim), nn.Conv2d(dim, dim, 1))
        self.query = nn.Embedding(spec.queries, dim)
        self.decoder = nn.ModuleList([TransformerBlock(dim, cross=True) for _ in range(2)])
        self.mask_embed = nn.Linear(dim, dim)
        self.class_embed = nn.Linear(dim, spec.classes + 1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Predict mask-query classes and masks.

        Parameters
        ----------
        x:
            Input RGB image.

        Returns
        -------
        tuple[Tensor, Tensor]
            Query classes and mask logits.
        """

        pixel = self.pixel_decoder(self.neck(self.backbone(x))[0])
        memory = pixel.flatten(2).transpose(1, 2)
        query = self.query.weight.unsqueeze(0).expand(x.shape[0], -1, -1)
        for layer in self.decoder:
            if self.spec.kind == "mask2former":
                mask_hint = torch.sigmoid(
                    torch.matmul(layer.norm1(query), memory.transpose(1, 2))
                ).mean(dim=-1, keepdim=True)
                query = query + mask_hint
            query = layer(query, memory)
        mask_embed = self.mask_embed(query)
        masks = torch.matmul(mask_embed, memory.transpose(1, 2)).view(
            x.shape[0], self.spec.queries, *pixel.shape[-2:]
        )
        return self.class_embed(query), masks


class ReIDTracker(nn.Module):
    """DeepSORT/StrongSORT/ByteTrack-style detection plus appearance model."""

    def __init__(self, spec: VariantSpec) -> None:
        """Initialize tracker.

        Parameters
        ----------
        spec:
            Variant configuration.
        """

        super().__init__()
        self.detector = DenseDetector(VariantSpec(kind="yolox", neck="yolox", classes=spec.classes))
        self.embedding = nn.Sequential(
            conv_norm_act(3, 24, 7, stride=2),
            conv_norm_act(24, 48, 3, stride=2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(48, 32),
        )
        self.motion_gate = nn.Linear(32, 4)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Run detection and appearance/motion heads.

        Parameters
        ----------
        x:
            Input RGB image.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor, Tensor]
            Dense classes, boxes, auxiliary objectness, and track embedding.
        """

        cls, box, aux = self.detector(x)
        embedding = F.normalize(self.embedding(x), dim=-1)
        motion = self.motion_gate(embedding)
        return cls, box + motion[:, None, :], aux, embedding


class ViTReID(nn.Module):
    """Vision-transformer style ReID/headless SAM placeholder from mmpretrain configs."""

    def __init__(self, classes: int = 5, dim: int = 48, patches: int = 256) -> None:
        """Initialize compact ViT ReID model.

        Parameters
        ----------
        classes:
            Number of identity logits.
        dim:
            Token width.
        patches:
            Number of spatial patches per side after projection.
        """

        super().__init__()
        self.patch = nn.Conv2d(3, dim, 4, stride=4)
        self.pos = nn.Parameter(torch.randn(patches, dim) * 0.02)
        self.blocks = nn.ModuleList([TransformerBlock(dim) for _ in range(2)])
        self.neck = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, classes)

    def forward(self, x: Tensor) -> Tensor:
        """Classify identity tokens.

        Parameters
        ----------
        x:
            Input RGB image.

        Returns
        -------
        Tensor
            Identity logits.
        """

        tokens = self.patch(x).flatten(2).transpose(1, 2)
        tokens = tokens + self.pos[: tokens.shape[1]].view(1, tokens.shape[1], -1)
        for block in self.blocks:
            tokens = block(tokens)
        return self.head(self.neck(tokens.mean(dim=1)))


def _spec_for(name: str) -> VariantSpec:
    """Return the compact architecture spec for a wave-A stable name.

    Parameters
    ----------
    name:
        Catalog model name or family hint.

    Returns
    -------
    VariantSpec
        Architecture configuration.
    """

    lower = name.lower()
    backbone = "resnet"
    if "convnext" in lower:
        backbone = "convnext"
    elif "hrnet" in lower:
        backbone = "hrnet"
    elif "regnet" in lower:
        backbone = "regnet"
    elif "res2net" in lower:
        backbone = "res2net"
    elif "resnest" in lower or "_s101" in lower:
        backbone = "resnest"
    elif "dcn" in lower or "dconv" in lower:
        backbone = "dcn"
    elif "gcnet" in lower or "gcb" in lower:
        backbone = "gcnet"
    neck = "fpn"
    if "nas_fpn" in lower:
        neck = "nasfpn"
    elif "libra" in lower:
        neck = "bfp"
    elif "rfp" in lower:
        neck = "rfp"
    elif "bytetrack" in lower or "yolox" in lower:
        neck = "yolox"
    if "mask2former" in lower:
        return VariantSpec("mask2former", backbone=backbone, queries=10)
    if "maskformer" in lower:
        return VariantSpec("maskformer", backbone=backbone, queries=10)
    if "deformable_detr" in lower:
        return VariantSpec("deformable_detr", backbone=backbone, queries=10)
    if "conditional_detr" in lower:
        return VariantSpec("conditional_detr", backbone=backbone, queries=10)
    if "dab_detr" in lower:
        return VariantSpec("dab_detr", backbone=backbone, queries=10)
    if "dino" in lower:
        return VariantSpec("dino", backbone=backbone, queries=10)
    if "ddq" in lower:
        return VariantSpec("ddq", backbone=backbone, queries=10)
    if "reid" in lower or "sam_headless" in lower:
        return VariantSpec("vit_reid", backbone=backbone)
    if "deepsort" in lower or "strongsort" in lower or "bytetrack" in lower:
        return VariantSpec("tracker", backbone=backbone, neck=neck)
    if "fcos" in lower:
        return VariantSpec("nas_fcos" if "nas" in lower else "fcos", backbone=backbone, neck=neck)
    if "centernet" in lower:
        return VariantSpec("centernet", backbone=backbone, neck=neck)
    if "cornernet" in lower:
        return VariantSpec("cornernet", backbone=backbone, neck=neck)
    if "centripetal" in lower:
        return VariantSpec("centripetal", backbone=backbone, neck=neck)
    if "condinst" in lower:
        return VariantSpec("condinst", backbone=backbone, neck=neck)
    if "boxinst" in lower:
        return VariantSpec("boxinst", backbone=backbone, neck=neck)
    if "solo" in lower:
        return VariantSpec("solo", backbone=backbone, neck=neck)
    if "retinanet" in lower or "timm_example" in lower or "nas_fpn" in lower:
        return VariantSpec("retina", backbone=backbone, neck=neck)
    if "ssd" in lower or "legacy_1_x" in lower:
        return VariantSpec("ssd", backbone=backbone, neck=neck)
    if "ghm" in lower:
        return VariantSpec("ghm", backbone=backbone, neck=neck)
    if "sabl" in lower:
        return VariantSpec("sabl", backbone=backbone, neck=neck)
    if "ddod" in lower:
        return VariantSpec("ddod", backbone=backbone, neck=neck)
    if "cascade_rpn" in lower:
        return VariantSpec("cascade_rpn", backbone=backbone, neck=neck, stages=1)
    if "cascade" in lower:
        return VariantSpec("cascade_rcnn", backbone=backbone, neck=neck, roi="mask", stages=3)
    if "htc" in lower:
        return VariantSpec("htc", backbone=backbone, neck=neck, roi="mask", stages=3)
    if "scnet" in lower:
        return VariantSpec("scnet", backbone=backbone, neck=neck, roi="mask", stages=2)
    if "double" in lower or "_dh_" in lower:
        return VariantSpec("faster_rcnn", backbone=backbone, neck=neck, roi="double")
    if "grid" in lower:
        return VariantSpec("faster_rcnn", backbone=backbone, neck=neck, roi="grid")
    if "ms_rcnn" in lower or "mask-scoring" in lower:
        return VariantSpec("mask_rcnn", backbone=backbone, neck=neck, roi="mask_iou")
    if "point_rend" in lower:
        return VariantSpec("mask_rcnn", backbone=backbone, neck=neck, roi="point_rend")
    if "pisa" in lower:
        return VariantSpec("faster_rcnn", backbone=backbone, neck=neck, roi="pisa")
    if "dynamic" in lower:
        return VariantSpec("faster_rcnn", backbone=backbone, neck=neck, roi="dynamic")
    if "crowddet" in lower:
        return VariantSpec("faster_rcnn", backbone=backbone, neck=neck, roi="crowd")
    if "trident" in lower:
        return VariantSpec("faster_rcnn", backbone=backbone, neck=neck, roi="box", stages=3)
    if "panoptic" in lower:
        return VariantSpec("panoptic", backbone=backbone, neck=neck, roi="mask")
    if "mask_rcnn" in lower or "mask-rcnn" in lower or "seesaw" in lower:
        return VariantSpec("mask_rcnn", backbone=backbone, neck=neck, roi="mask")
    return VariantSpec("faster_rcnn", backbone=backbone, neck=neck, roi="box")


def build_mmdet_wave_a(name: str = "cascade_rcnn") -> nn.Module:
    """Build a compact faithful MMDetection classic by catalog name.

    Parameters
    ----------
    name:
        Stable catalog name or family hint.

    Returns
    -------
    nn.Module
        Random-initialized Torch-only detector.
    """

    spec = _spec_for(name)
    if spec.kind in {"deformable_detr", "conditional_detr", "dab_detr", "dino", "ddq"}:
        return DETRDetector(spec).eval()
    if spec.kind in {"maskformer", "mask2former"}:
        return MaskQuerySegmenter(spec).eval()
    if spec.kind == "tracker":
        return ReIDTracker(spec).eval()
    if spec.kind == "vit_reid":
        return ViTReID(classes=spec.classes).eval()
    if spec.kind in {
        "retina",
        "fcos",
        "nas_fcos",
        "centernet",
        "cornernet",
        "centripetal",
        "condinst",
        "boxinst",
        "solo",
        "ssd",
        "ghm",
        "sabl",
        "ddod",
    }:
        return DenseDetector(spec).eval()
    return TwoStageDetector(spec).eval()


def example_input() -> Tensor:
    """Return a real RGB image input for all wave-A detector recipes.

    Returns
    -------
    Tensor
        Tensor with shape ``(1, 3, 64, 64)``.
    """

    return torch.randn(1, 3, 64, 64)
