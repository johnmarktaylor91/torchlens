"""Additional compact PaddleDetection task-family reconstructions.

Papers/Systems: PaddleDetection toolkit; PP-YOLOE (Xu et al., 2022),
RT-DETR/Mask-RT-DETR (Lv et al., 2023+), QueryInst (Fang et al., 2021),
HRNet pose models (Sun et al., 2019), ByteTrack/DeepSORT-style MOT, PP-YOLOE-R,
DenseTeacher/semi-det, and SNIPER.

These random-init modules keep task-specific primitives rather than a generic
detector: anchor-free PP-YOLOE heads with task alignment, HRNet heatmaps,
rotated angle regression, transformer query instance masks, temporal track
association embeddings, teacher/student semi-det outputs, and chip-level SNIPER
aggregation.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class PaddleBackbone(nn.Module):
    """Small multi-scale CSP-style backbone shared by PaddleDetection heads."""

    def __init__(self, width: int = 32) -> None:
        """Initialize convolutional stages."""

        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, 3, stride=2, padding=1), nn.BatchNorm2d(width), nn.SiLU()
        )
        self.c3 = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1), nn.SiLU(), nn.Conv2d(width, width, 1), nn.SiLU()
        )
        self.c4 = nn.Sequential(nn.Conv2d(width, width * 2, 3, stride=2, padding=1), nn.SiLU())
        self.c5 = nn.Sequential(nn.Conv2d(width * 2, width * 2, 3, stride=2, padding=1), nn.SiLU())

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return C3/C4/C5 feature maps."""

        c3 = self.c3(self.stem(image))
        c4 = self.c4(c3)
        c5 = self.c5(c4)
        return c3, c4, c5


class PPYOLOEHead(nn.Module):
    """Anchor-free PP-YOLOE task-aligned detection head."""

    def __init__(self, in_channels: int = 64, classes: int = 10, rotated: bool = False) -> None:
        """Initialize class, box, objectness, and optional angle heads."""

        super().__init__()
        self.cls = nn.Conv2d(in_channels, classes, 1)
        self.box = nn.Conv2d(in_channels, 4, 1)
        self.obj = nn.Conv2d(in_channels, 1, 1)
        self.angle = nn.Conv2d(in_channels, 1, 1) if rotated else None

    def forward(self, feat: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Predict task-aligned class scores and boxes."""

        cls = self.cls(feat)
        box = torch.sigmoid(self.box(feat))
        obj = torch.sigmoid(self.obj(feat))
        if self.angle is not None:
            box = torch.cat([box, torch.tanh(self.angle(feat))], dim=1)
        return cls, box, obj * cls.sigmoid().amax(dim=1, keepdim=True)


class PaddlePPYOLOE(nn.Module):
    """Compact PP-YOLOE/seg/face/layout/rotated detector."""

    def __init__(self, classes: int = 10, seg: bool = False, rotated: bool = False) -> None:
        """Initialize backbone, PAN fusion, and heads."""

        super().__init__()
        self.backbone = PaddleBackbone()
        self.pan = nn.Conv2d(160, 64, 3, padding=1)
        self.head = PPYOLOEHead(64, classes, rotated=rotated)
        self.seg = nn.Conv2d(64, 8, 1) if seg else None

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Run compact PP-YOLOE-style inference."""

        c3, c4, c5 = self.backbone(image)
        c4_up = F.interpolate(c4, size=c3.shape[-2:], mode="nearest")
        c5_up = F.interpolate(c5, size=c3.shape[-2:], mode="nearest")
        feat = self.pan(torch.cat([c3, c4_up, c5_up], dim=1))
        cls, box, align = self.head(feat)
        masks = self.seg(feat) if self.seg is not None else feat[:, :1]
        return cls, box, align, masks


class PaddleKeypoint(nn.Module):
    """Compact HRNet-style keypoint detector."""

    def __init__(self, keypoints: int = 17) -> None:
        """Initialize parallel high/low-resolution branches."""

        super().__init__()
        self.high = nn.Sequential(
            nn.Conv2d(3, 24, 3, padding=1), nn.ReLU(), nn.Conv2d(24, 24, 3, padding=1)
        )
        self.low = nn.Sequential(
            nn.Conv2d(3, 24, 3, stride=2, padding=1), nn.ReLU(), nn.Conv2d(24, 24, 3, padding=1)
        )
        self.fuse = nn.Conv2d(48, 32, 1)
        self.heatmap = nn.Conv2d(32, keypoints, 1)

    def forward(self, image: Tensor) -> Tensor:
        """Predict per-keypoint heatmaps from high-resolution features."""

        high = self.high(image)
        low = F.interpolate(self.low(image), size=high.shape[-2:], mode="nearest")
        return self.heatmap(F.relu(self.fuse(torch.cat([high, low], dim=1))))


class PaddleMOT(nn.Module):
    """Compact detector plus appearance embeddings for MOT association."""

    def __init__(self) -> None:
        """Initialize detector and re-identification head."""

        super().__init__()
        self.detector = PaddlePPYOLOE(classes=4)
        self.reid = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 16))

    def forward(self, frames: Tensor) -> tuple[Tensor, Tensor]:
        """Return frame detections and ByteTrack/DeepSORT-style affinities."""

        batch, time, channels, height, width = frames.shape
        flat = frames.view(batch * time, channels, height, width)
        cls, box, align, _ = self.detector(flat)
        _, _, feat = self.detector.backbone(flat)
        embed = F.normalize(self.reid(feat).view(batch, time, -1), dim=-1)
        affinity = torch.matmul(embed, embed.transpose(1, 2))
        return align.view(batch, time, *align.shape[1:]).mean(dim=(3, 4)), affinity


class PaddleQueryInst(nn.Module):
    """Compact QueryInst instance segmentation transformer."""

    def __init__(self, queries: int = 12, dim: int = 48) -> None:
        """Initialize query decoder and dynamic mask heads."""

        super().__init__()
        self.backbone = PaddleBackbone(width=24)
        self.proj = nn.Conv2d(48, dim, 1)
        self.query = nn.Embedding(queries, dim)
        self.decoder = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.cls = nn.Linear(dim, 8)
        self.dynamic = nn.Linear(dim, dim)

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor]:
        """Predict query classes and query-conditioned masks."""

        _, _, c5 = self.backbone(image)
        feat = self.proj(c5)
        memory = feat.flatten(2).transpose(1, 2)
        query = self.query.weight.unsqueeze(0).expand(image.shape[0], -1, -1)
        query = self.decoder(query, memory, memory, need_weights=False)[0]
        kernels = self.dynamic(query)
        masks = torch.einsum("bqd,bdhw->bqhw", kernels, feat)
        return self.cls(query), masks


class PaddleSemiDet(nn.Module):
    """Compact DenseTeacher-style semi-supervised detector."""

    def __init__(self) -> None:
        """Initialize student and EMA-teacher heads."""

        super().__init__()
        self.student = PaddlePPYOLOE(classes=6)
        self.teacher = PaddlePPYOLOE(classes=6)

    def forward(self, weak: Tensor, strong: Tensor) -> tuple[Tensor, Tensor]:
        """Return teacher pseudo labels and student predictions."""

        teacher_scores = self.teacher(weak)[2].detach()
        student_scores = self.student(strong)[2]
        return teacher_scores, student_scores


class PaddleSNIPER(nn.Module):
    """Compact SNIPER multi-chip detector."""

    def __init__(self) -> None:
        """Initialize chip detector and aggregation projection."""

        super().__init__()
        self.detector = PaddlePPYOLOE(classes=5)
        self.aggregate = nn.Linear(5, 5)

    def forward(self, chips: Tensor) -> Tensor:
        """Aggregate detections from image chips."""

        batch, chips_n, channels, height, width = chips.shape
        cls, _, _, _ = self.detector(chips.view(batch * chips_n, channels, height, width))
        pooled = cls.mean(dim=(2, 3)).view(batch, chips_n, -1)
        return self.aggregate(pooled).amax(dim=1)


class BlazeBlock(nn.Module):
    """BlazeFace double 5x5 depthwise block."""

    def __init__(self, channels: int) -> None:
        """Initialize compact BlazeBlock."""

        super().__init__()
        self.dw1 = nn.Conv2d(channels, channels, 5, padding=2, groups=channels)
        self.pw1 = nn.Conv2d(channels, channels, 1)
        self.dw2 = nn.Conv2d(channels, channels, 5, padding=2, groups=channels)
        self.pw2 = nn.Conv2d(channels, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Apply residual BlazeFace feature extraction."""

        y = F.relu(self.pw1(self.dw1(x)))
        y = self.pw2(self.dw2(y))
        return F.relu(x + y)


class PaddleBlazeFace(nn.Module):
    """BlazeFace-style anchor face detector with six landmarks."""

    def __init__(self, anchors: int = 6) -> None:
        """Initialize face detector."""

        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 24, 3, stride=2, padding=1), nn.ReLU(), BlazeBlock(24), BlazeBlock(24)
        )
        self.anchors = anchors
        self.cls = nn.Conv2d(24, anchors, 1)
        self.box = nn.Conv2d(24, anchors * 4, 1)
        self.landmarks = nn.Conv2d(24, anchors * 12, 1)
        self.register_buffer(
            "anchor_grid", torch.linspace(-1.0, 1.0, anchors).view(1, anchors, 1, 1)
        )

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Predict face scores, anchor boxes, and six landmarks."""

        feat = self.stem(image)
        scores = torch.sigmoid(self.cls(feat))
        boxes = torch.sigmoid(
            self.box(feat).view(image.shape[0], self.anchors, 4, *feat.shape[-2:])
        )
        boxes = boxes + self.anchor_grid[:, :, None]
        landmarks = self.landmarks(feat).view(image.shape[0], self.anchors, 12, *feat.shape[-2:])
        return scores, boxes, landmarks


class PaddleLayoutDetector(nn.Module):
    """Document-layout detector with PicoDet-like head and layout coordinates."""

    def __init__(self, classes: int = 3) -> None:
        """Initialize compact layout detector."""

        super().__init__()
        self.backbone = PaddleBackbone(width=24)
        self.coord = nn.Conv2d(50, 16, 1)
        self.head = PPYOLOEHead(88, classes)
        self.reading_order = nn.Conv2d(88, 1, 1)

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Predict document regions and reading-order scores."""

        c3, c4, c5 = self.backbone(image)
        y = (
            torch.linspace(-1.0, 1.0, c5.shape[-2], device=image.device)
            .view(1, 1, -1, 1)
            .expand(c5.shape[0], 1, -1, c5.shape[-1])
        )
        x = (
            torch.linspace(-1.0, 1.0, c5.shape[-1], device=image.device)
            .view(1, 1, 1, -1)
            .expand(c5.shape[0], 1, c5.shape[-2], -1)
        )
        doc_feat = torch.cat([c5, x, y], dim=1)
        feat = torch.cat(
            [
                F.interpolate(c3, c5.shape[-2:]),
                F.interpolate(c4, c5.shape[-2:]),
                self.coord(doc_feat),
            ],
            dim=1,
        )
        cls, box, align = self.head(feat)
        return cls, box, align, torch.sigmoid(self.reading_order(feat))


class PaddlePPYOLOv3(nn.Module):
    """PP-YOLO as YOLOv3 plus CoordConv, SPP, IoU-aware objectness."""

    def __init__(self, classes: int = 10, anchors: int = 3) -> None:
        """Initialize compact PP-YOLO detector."""

        super().__init__()
        self.backbone = PaddleBackbone(width=32)
        self.spp_proj = nn.Conv2d(64 * 4 + 2, 64, 1)
        self.cls = nn.Conv2d(64, anchors * classes, 1)
        self.box = nn.Conv2d(64, anchors * 4, 1)
        self.obj = nn.Conv2d(64, anchors, 1)
        self.iou_aware = nn.Conv2d(64, anchors, 1)

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Run PP-YOLOv3-style detection heads."""

        _, _, c5 = self.backbone(image)
        y = (
            torch.linspace(-1.0, 1.0, c5.shape[-2], device=image.device)
            .view(1, 1, -1, 1)
            .expand(c5.shape[0], 1, -1, c5.shape[-1])
        )
        x = (
            torch.linspace(-1.0, 1.0, c5.shape[-1], device=image.device)
            .view(1, 1, 1, -1)
            .expand(c5.shape[0], 1, c5.shape[-2], -1)
        )
        spp = torch.cat(
            [
                c5,
                F.max_pool2d(c5, 5, stride=1, padding=2),
                F.max_pool2d(c5, 9, stride=1, padding=4),
                F.max_pool2d(c5, 13, stride=1, padding=6),
                x,
                y,
            ],
            dim=1,
        )
        feat = F.silu(self.spp_proj(spp))
        obj = torch.sigmoid(self.obj(feat))
        iou = torch.sigmoid(self.iou_aware(feat))
        return self.cls(feat), torch.sigmoid(self.box(feat)), obj * iou, iou


class PaddleMaskRTDETR(nn.Module):
    """RT-DETR hybrid encoder/query detector with dynamic mask extension."""

    def __init__(self, queries: int = 16, dim: int = 48) -> None:
        """Initialize compact Mask-RT-DETR."""

        super().__init__()
        self.backbone = PaddleBackbone(width=24)
        self.proj = nn.Conv2d(48, dim, 1)
        layer = nn.TransformerEncoderLayer(dim, 4, dim * 2, batch_first=True, norm_first=True)
        self.hybrid_encoder = nn.TransformerEncoder(layer, 1)
        self.query = nn.Embedding(queries, dim)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(dim, 4, dim * 2, batch_first=True, norm_first=True),
            1,
        )
        self.cls = nn.Linear(dim, 8)
        self.box = nn.Linear(dim, 4)
        self.mask_kernel = nn.Linear(dim, dim)

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Predict RT-DETR query boxes and query-conditioned masks."""

        _, _, c5 = self.backbone(image)
        feat = self.proj(c5)
        memory = self.hybrid_encoder(feat.flatten(2).transpose(1, 2))
        query = self.query.weight.unsqueeze(0).expand(image.shape[0], -1, -1)
        decoded = self.decoder(query, memory)
        masks = torch.einsum("bqd,bdhw->bqhw", self.mask_kernel(decoded), feat)
        return self.cls(decoded), torch.sigmoid(self.box(decoded)), masks


def build_face() -> nn.Module:
    """Build compact PaddleDetection face detector."""

    return PaddleBlazeFace().eval()


def build_layout() -> nn.Module:
    """Build compact PicoDet-style layout detector."""

    return PaddleLayoutDetector().eval()


def build_ppyolo() -> nn.Module:
    """Build compact PP-YOLO detector."""

    return PaddlePPYOLOv3(classes=10).eval()


def build_ppyoloe_seg() -> nn.Module:
    """Build compact PP-YOLOE segmentation model."""

    return PaddlePPYOLOE(classes=10, seg=True).eval()


def build_rotate() -> nn.Module:
    """Build compact PP-YOLOE-R rotated detector."""

    return PaddlePPYOLOE(classes=10, rotated=True).eval()


def build_keypoint() -> nn.Module:
    """Build compact HRNet keypoint detector."""

    return PaddleKeypoint().eval()


def build_pose3d() -> nn.Module:
    """Build compact 3D pose detector with x/y/z heatmaps."""

    return PaddleKeypoint(keypoints=51).eval()


def build_mot() -> nn.Module:
    """Build compact MOT model."""

    return PaddleMOT().eval()


def build_queryinst() -> nn.Module:
    """Build compact QueryInst model."""

    return PaddleQueryInst().eval()


def build_mask_rtdetr() -> nn.Module:
    """Build compact Mask-RT-DETR-like query mask model."""

    return PaddleMaskRTDETR(queries=16).eval()


def build_semi_det() -> nn.Module:
    """Build compact semi-supervised detector."""

    return PaddleSemiDet().eval()


def build_sniper() -> nn.Module:
    """Build compact SNIPER chip detector."""

    return PaddleSNIPER().eval()


def example_image() -> Tensor:
    """Return one small RGB image."""

    return torch.randn(1, 3, 64, 64)


def example_frames() -> Tensor:
    """Return a short RGB frame sequence."""

    return torch.randn(1, 3, 3, 64, 64)


def example_pair() -> tuple[Tensor, Tensor]:
    """Return weak and strong augmentations."""

    return torch.randn(1, 3, 64, 64), torch.randn(1, 3, 64, 64)


def example_chips() -> Tensor:
    """Return image chips for SNIPER."""

    return torch.randn(1, 3, 3, 48, 48)


MENAGERIE_ENTRIES = [
    ("paddledet_face_detection", "build_face", "example_image", "2021", "DET"),
    ("paddledet_keypoint", "build_keypoint", "example_image", "2019", "POSE"),
    ("paddledet_layout_analysis", "build_layout", "example_image", "2021", "DET"),
    ("paddledet_mot", "build_mot", "example_frames", "2021", "DET"),
    ("paddledet_pose3d", "build_pose3d", "example_image", "2021", "POSE"),
    ("paddledet_ppyolo", "build_ppyolo", "example_image", "2020", "DET"),
    ("paddledet_ppyoloe_seg", "build_ppyoloe_seg", "example_image", "2022", "DET"),
    ("paddledet_queryinst", "build_queryinst", "example_image", "2021", "DET"),
    ("paddledet_rotate", "build_rotate", "example_image", "2022", "DET"),
    ("paddledet_mask_rtdetr", "build_mask_rtdetr", "example_image", "2024", "DET"),
    ("paddledet_semi_det", "build_semi_det", "example_pair", "2022", "DET"),
    ("paddledet_sniper", "build_sniper", "example_chips", "2018", "DET"),
]
