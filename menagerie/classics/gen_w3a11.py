"""Compact faithful reconstructions of five 3D-object-detection architectures.

Sources checked (repo READMEs / arXiv abstracts + linked source files, no clone/install):
  - CLOCs: https://github.com/pangsu0613/CLOCs (arXiv:2009.00784, IROS 2020)
  - CMT (Cross-Modal Transformer, 3D detection): https://github.com/junjie18/CMT
    (arXiv:2301.01283, ICCV 2023)
  - Complex-YOLO: https://github.com/maudzung/Complex-YOLOv4-Pytorch
    (arXiv:1803.06199)
  - CT3D: https://github.com/hlsheng1/CT3D (arXiv:2108.10723, ICCV 2021)
  - Cube R-CNN: https://github.com/facebookresearch/omni3d
    (cubercnn/modeling/roi_heads/cube_head.py; arXiv:2207.10660, CVPR 2023)

Every model below reimplements the paper's distinctive mechanism from scratch in
base-env torch with small random-init dimensions; none of these load pretrained
weights or third-party code.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# CLOCs: Camera-LiDAR Object Candidates fusion.
#
# Distinctive mechanism: instead of fusing raw features, CLOCs builds a sparse
# geometric+semantic consistency tensor over every (2D box, 3D box) candidate
# pair -- IoU, normalized distance, and both detector confidences -- laid out
# on a [n_2d, n_3d] grid, then runs a small 2D conv "fusion" network over that
# tensor to output one fused confidence logit per pair (pre-NMS candidates in,
# fused per-pair score out).
# ---------------------------------------------------------------------------


class CLOCsFusion(nn.Module):
    """CLOCs 2D conv fusion network over a 2D-3D candidate consistency tensor."""

    def __init__(self, channels: int = 4, hidden: int = 18) -> None:
        """Initialize the 1x1-conv fusion stack over the consistency tensor."""

        super().__init__()
        self.conv1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.conv2 = nn.Conv2d(hidden, hidden, kernel_size=1)
        self.conv3 = nn.Conv2d(hidden, 1, kernel_size=1)

    def forward(self, boxes_2d: Tensor, boxes_3d_proj: Tensor) -> Tensor:
        """Score every (2D box, projected-3D box) candidate pair.

        Parameters
        ----------
        boxes_2d : Tensor
            ``(n_2d, 5)`` 2D candidate boxes as ``(x1, y1, x2, y2, score)``.
        boxes_3d_proj : Tensor
            ``(n_3d, 5)`` 3D boxes projected to image plane, same layout.

        Returns
        -------
        Tensor
            ``(n_2d, n_3d)`` fused confidence logits, one per candidate pair.
        """

        iou = self._pairwise_iou(boxes_2d[:, :4], boxes_3d_proj[:, :4])
        c2 = boxes_2d[:, 0:2] + boxes_2d[:, 2:4]
        c3 = boxes_3d_proj[:, 0:2] + boxes_3d_proj[:, 2:4]
        dist = torch.cdist(c2[None], c3[None])[0] * 0.5
        dist = 1.0 / (1.0 + dist)
        score_2d = boxes_2d[:, 4:5].expand(-1, boxes_3d_proj.shape[0])
        score_3d = boxes_3d_proj[:, 4:5].transpose(0, 1).expand(boxes_2d.shape[0], -1)
        tensor = torch.stack([iou, dist, score_2d, score_3d], dim=0).unsqueeze(0)
        hidden = F.relu(self.conv1(tensor))
        hidden = F.relu(self.conv2(hidden))
        return self.conv3(hidden)[0, 0]

    @staticmethod
    def _pairwise_iou(a: Tensor, b: Tensor) -> Tensor:
        """Compute pairwise IoU between two axis-aligned box sets."""

        area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
        area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
        lt = torch.max(a[:, None, :2], b[None, :, :2])
        rb = torch.min(a[:, None, 2:], b[None, :, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[..., 0] * wh[..., 1]
        union = area_a[:, None] + area_b[None, :] - inter
        return inter / union.clamp(min=1e-6)


def build_clocs() -> nn.Module:
    """Build a compact random-init CLOCs fusion network."""

    return CLOCsFusion().eval()


def example_input_clocs() -> tuple[Tensor, Tensor]:
    """Return small 2D and projected-3D candidate box sets."""

    boxes_2d = torch.rand(6, 5)
    boxes_2d[:, 2:4] += boxes_2d[:, 0:2] + 0.1
    boxes_3d = torch.rand(5, 5)
    boxes_3d[:, 2:4] += boxes_3d[:, 0:2] + 0.1
    return boxes_2d, boxes_3d


# ---------------------------------------------------------------------------
# CMT: Cross-Modal Transformer for 3D detection (Yan et al., ICCV 2023).
#
# Distinctive mechanism: a DETR-like decoder where 3D object queries are
# generated from 3D anchor points encoded with the SAME coordinate positional
# encoding used to tag the multi-modal (camera + LiDAR/BEV) tokens, so the
# queries can directly cross-attend to image and point tokens without any
# explicit image-to-BEV view transform.
# ---------------------------------------------------------------------------


class CMTCoordinatePE(nn.Module):
    """Shared 3D-coordinate positional encoding for tokens and queries."""

    def __init__(self, dim: int) -> None:
        """Initialize the coordinate MLP shared across modalities."""

        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(3, dim), nn.ReLU(), nn.Linear(dim, dim))

    def forward(self, xyz: Tensor) -> Tensor:
        """Encode 3D coordinates into positional embeddings."""

        return self.mlp(xyz)


class CMTDetector(nn.Module):
    """Compact CMT: coordinate-encoded multi-modal tokens, DETR-style query decode."""

    def __init__(self, dim: int = 32, n_queries: int = 8, n_layers: int = 2) -> None:
        """Initialize modality embeddings, shared coordinate PE, and query decoder."""

        super().__init__()
        self.img_embed = nn.Linear(16, dim)
        self.pts_embed = nn.Linear(4, dim)
        self.coord_pe = CMTCoordinatePE(dim)
        self.query_anchors = nn.Parameter(torch.rand(n_queries, 3) * 2 - 1)
        self.query_content = nn.Parameter(torch.zeros(n_queries, dim))
        layer = nn.TransformerDecoderLayer(
            d_model=dim, nhead=4, dim_feedforward=dim * 2, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=n_layers)
        self.box_head = nn.Linear(dim, 7)
        self.cls_head = nn.Linear(dim, 3)

    def forward(
        self, img_feats: Tensor, img_xyz: Tensor, pts_feats: Tensor, pts_xyz: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Cross-attend 3D queries to coordinate-tagged image and point tokens.

        Parameters
        ----------
        img_feats : Tensor
            ``(B, N_img, 16)`` per-pixel image tokens.
        img_xyz : Tensor
            ``(B, N_img, 3)`` 3D positions (frustum points) tagging image tokens.
        pts_feats : Tensor
            ``(B, N_pts, 4)`` per-voxel LiDAR features.
        pts_xyz : Tensor
            ``(B, N_pts, 3)`` voxel centers tagging point tokens.

        Returns
        -------
        tuple[Tensor, Tensor]
            Per-query 3D box regression ``(B, Q, 7)`` and class logits ``(B, Q, 3)``.
        """

        img_tokens = self.img_embed(img_feats) + self.coord_pe(img_xyz)
        pts_tokens = self.pts_embed(pts_feats) + self.coord_pe(pts_xyz)
        memory = torch.cat([img_tokens, pts_tokens], dim=1)
        batch = img_feats.shape[0]
        queries = self.query_content[None].expand(batch, -1, -1) + self.coord_pe(
            self.query_anchors[None].expand(batch, -1, -1)
        )
        decoded = self.decoder(queries, memory)
        return self.box_head(decoded), self.cls_head(decoded)


def build_cmt3d() -> nn.Module:
    """Build a compact random-init CMT 3D detector."""

    return CMTDetector().eval()


def example_input_cmt3d() -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return image tokens/positions and LiDAR voxel tokens/positions."""

    return (
        torch.randn(1, 12, 16),
        torch.rand(1, 12, 3) * 2 - 1,
        torch.randn(1, 20, 4),
        torch.rand(1, 20, 3) * 2 - 1,
    )


# ---------------------------------------------------------------------------
# Complex-YOLO: real-time 3D detection on a bird's-eye-view point-cloud raster.
#
# Distinctive mechanism: a YOLO-style dense grid detector where object heading
# is regressed as a COMPLEX number (Im, Re) rather than a raw angle, giving a
# continuous, discontinuity-free E-RPN (Euler-Region-Proposal-Network) head
# alongside the usual per-cell box/objectness/class outputs.
# ---------------------------------------------------------------------------


class ComplexYOLOHead(nn.Module):
    """Euler-RPN detection head: per-cell box + complex-angle heading."""

    def __init__(self, in_channels: int, n_classes: int = 3, n_anchors: int = 2) -> None:
        """Initialize the per-cell prediction conv (box + objectness + im/re + class)."""

        super().__init__()
        self.n_anchors = n_anchors
        self.n_classes = n_classes
        out_per_anchor = 4 + 1 + 2 + n_classes  # x,y,w,l + obj + (Im,Re) + classes
        self.pred = nn.Conv2d(in_channels, n_anchors * out_per_anchor, kernel_size=1)

    def forward(self, feats: Tensor) -> Tensor:
        """Predict per-cell, per-anchor BEV boxes with complex heading encoding."""

        batch, _, h, w = feats.shape
        out = self.pred(feats)
        return out.view(batch, self.n_anchors, -1, h, w)


class ComplexYOLONet(nn.Module):
    """Compact Complex-YOLO: BEV raster backbone + Euler-RPN head."""

    def __init__(self, in_channels: int = 3) -> None:
        """Initialize the BEV conv backbone and Euler-RPN detection head."""

        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
        )
        self.head = ComplexYOLOHead(32)

    def forward(self, bev: Tensor) -> Tensor:
        """Detect oriented BEV boxes from a rasterized point-cloud height map."""

        feats = self.backbone(bev)
        raw = self.head(feats)
        im, re = raw[:, :, 5:6], raw[:, :, 6:7]
        norm = torch.sqrt(im**2 + re**2).clamp(min=1e-6)
        raw = torch.cat([raw[:, :, :5], im / norm, re / norm, raw[:, :, 7:]], dim=2)
        return raw


def build_complex_yolo() -> nn.Module:
    """Build a compact random-init Complex-YOLO BEV detector."""

    return ComplexYOLONet().eval()


def example_input_complex_yolo() -> Tensor:
    """Return a small rasterized BEV point-cloud height/density/intensity map."""

    return torch.randn(1, 3, 32, 32)


# ---------------------------------------------------------------------------
# CT3D: Channel-wise Transformer for 3D proposal refinement (Sheng et al.,
# ICCV 2021).
#
# Distinctive mechanism: a two-stage refinement head with (1) a standard
# proposal-aware spatial encoder attending over per-proposal keypoints, then
# (2) a CHANNEL-WISE decoding module that re-weights and aggregates context
# across feature CHANNELS (not tokens) via channel-key/channel-query
# attention, before predicting the final box residual and IoU confidence.
# ---------------------------------------------------------------------------


class ChannelwiseDecoder(nn.Module):
    """Channel-wise re-weighting attention: attention over the channel axis."""

    def __init__(self, dim: int, n_points: int) -> None:
        """Initialize channel-query/channel-key projections over pooled tokens."""

        super().__init__()
        self.channel_q = nn.Linear(n_points, n_points, bias=False)
        self.channel_k = nn.Linear(n_points, n_points, bias=False)
        self.channel_v = nn.Linear(n_points, n_points, bias=False)
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, tokens: Tensor) -> Tensor:
        """Re-weight per-channel context via channel-wise self-attention.

        Parameters
        ----------
        tokens : Tensor
            ``(B, N, C)`` proposal-aware point embeddings.
        """

        chan = tokens.transpose(1, 2)  # (B, C, N): channels as the attended axis
        q, k, v = self.channel_q(chan), self.channel_k(chan), self.channel_v(chan)
        attn = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / (chan.shape[-1] ** 0.5), dim=-1)
        merged = torch.matmul(attn, v).transpose(1, 2)  # back to (B, N, C)
        return self.out_norm(tokens + merged)


class CT3DRefiner(nn.Module):
    """Compact CT3D: spatial keypoint encoder + channel-wise decoding refinement."""

    def __init__(self, dim: int = 24, n_points: int = 16) -> None:
        """Initialize keypoint embedding, spatial encoder, and channel decoder."""

        super().__init__()
        self.embed = nn.Linear(3 + 4, dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=4, dim_feedforward=dim * 2, batch_first=True
        )
        self.spatial_encoder = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.channel_decoder = ChannelwiseDecoder(dim, n_points)
        self.box_residual = nn.Linear(dim, 7)
        self.iou_conf = nn.Linear(dim, 1)

    def forward(self, keypoint_xyz: Tensor, keypoint_feats: Tensor) -> tuple[Tensor, Tensor]:
        """Refine a 3D proposal from its sampled interior keypoints.

        Parameters
        ----------
        keypoint_xyz : Tensor
            ``(B, N, 3)`` proposal-relative keypoint coordinates.
        keypoint_feats : Tensor
            ``(B, N, 4)`` per-keypoint raw point features (e.g. intensity + local desc).

        Returns
        -------
        tuple[Tensor, Tensor]
            Box residual ``(B, 7)`` and IoU-confidence logit ``(B, 1)``.
        """

        tokens = self.embed(torch.cat([keypoint_xyz, keypoint_feats], dim=-1))
        tokens = self.spatial_encoder(tokens)
        tokens = self.channel_decoder(tokens)
        pooled = tokens.mean(dim=1)
        return self.box_residual(pooled), self.iou_conf(pooled)


def build_ct3d() -> nn.Module:
    """Build a compact random-init CT3D proposal refiner."""

    return CT3DRefiner().eval()


def example_input_ct3d() -> tuple[Tensor, Tensor]:
    """Return proposal-relative keypoint coordinates and per-keypoint features."""

    return torch.randn(2, 16, 3), torch.randn(2, 16, 4)


# ---------------------------------------------------------------------------
# Cube R-CNN: monocular 3D cuboid detector (Brazil et al., CVPR 2023, Omni3D).
#
# Distinctive mechanism: a Faster-R-CNN-style ROI head with FIVE parallel
# prediction branches per RoI -- 2D-plane center offset (XY), physical
# dimensions, 6D continuous rotation, and a "virtual depth" Z branch that
# predicts depth through discrete CLUSTER BINS (camera-scale-invariant depth
# clustering) plus an optional cuboid confidence -- fused into one 3D cuboid
# parameterization per detected object.
# ---------------------------------------------------------------------------


class CubeHead(nn.Module):
    """Compact Cube R-CNN ROI cube head: 5 parallel branches -> one 3D cuboid."""

    def __init__(self, in_dim: int = 64, n_classes: int = 5, cluster_bins: int = 3) -> None:
        """Initialize per-branch feature generators and cuboid parameter heads."""

        super().__init__()
        self.n_classes = n_classes
        self.cluster_bins = cluster_bins
        fc_dim = 32
        self.feat_xy = nn.Sequential(nn.Linear(in_dim, fc_dim), nn.ReLU())
        self.feat_dims = nn.Sequential(nn.Linear(in_dim, fc_dim), nn.ReLU())
        self.feat_pose = nn.Sequential(nn.Linear(in_dim, fc_dim), nn.ReLU())
        self.feat_z = nn.Sequential(nn.Linear(in_dim, fc_dim), nn.ReLU())
        self.feat_conf = nn.Sequential(nn.Linear(in_dim, fc_dim), nn.ReLU())

        self.bbox_3d_center_xy = nn.Linear(fc_dim, n_classes * 2)
        self.bbox_3d_dims = nn.Linear(fc_dim, n_classes * 3)
        self.bbox_3d_pose_6d = nn.Linear(fc_dim, n_classes * 6)
        self.bbox_3d_center_depth = nn.Linear(fc_dim, n_classes * cluster_bins)
        self.bbox_3d_uncertainty = nn.Linear(fc_dim, n_classes)

    def forward(self, roi_feats: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Predict a per-class 3D cuboid parameterization from pooled RoI features.

        Parameters
        ----------
        roi_feats : Tensor
            ``(N, in_dim)`` pooled RoI-Align features, one row per proposal.

        Returns
        -------
        tuple[Tensor, ...]
            ``(xy_delta, dims, pose_6d, virtual_depth_bins, confidence)``, each
            shaped ``(N, n_classes, ...)``.
        """

        n = roi_feats.shape[0]
        xy = self.bbox_3d_center_xy(self.feat_xy(roi_feats)).view(n, self.n_classes, 2)
        dims = self.bbox_3d_dims(self.feat_dims(roi_feats)).view(n, self.n_classes, 3)
        pose = self.bbox_3d_pose_6d(self.feat_pose(roi_feats)).view(n, self.n_classes, 6)
        depth_bins = self.bbox_3d_center_depth(self.feat_z(roi_feats)).view(
            n, self.n_classes, self.cluster_bins
        )
        conf = torch.sigmoid(self.bbox_3d_uncertainty(self.feat_conf(roi_feats)))
        return xy, dims, pose, depth_bins, conf


def build_cube_rcnn() -> nn.Module:
    """Build a compact random-init Cube R-CNN cube head."""

    return CubeHead().eval()


def example_input_cube_rcnn() -> Tensor:
    """Return pooled RoI-Align features for a small batch of proposals."""

    return torch.randn(5, 64)


MENAGERIE_ENTRIES = [
    ("CLOCs", "build_clocs", "example_input_clocs", "2020", "DET"),
    (
        "CMT (Cross-Modal Transformer for 3D detection)",
        "build_cmt3d",
        "example_input_cmt3d",
        "2023",
        "DET",
    ),
    ("Complex-YOLO", "build_complex_yolo", "example_input_complex_yolo", "2018", "DET"),
    ("CT3D", "build_ct3d", "example_input_ct3d", "2021", "DET"),
    ("Cube R-CNN", "build_cube_rcnn", "example_input_cube_rcnn", "2023", "DET"),
]
