"""Menagerie batch w3a19: LiDAR/camera fusion and point-cloud perception architectures.

Sources checked (reference only; no cloning, no pip installs):
  - PointAugmenting: Wang et al., CVPR 2021. Paper
    https://openaccess.thecvf.com/content/CVPR2021/html/Wang_PointAugmenting_Cross-Modal_Augmentation_for_3D_Object_Detection_CVPR_2021_paper.html,
    official source https://github.com/VISION-SJTU/PointAugmenting. Cross-modal 3D
    detection that "paints" each LiDAR point with a *deep camera feature vector*
    (rather than a shallow semantic-segmentation score, as in PointPainting) sampled
    from a 2D image backbone at the point's projected pixel location; the concatenated
    (xyz + deep-image-feature) point is then consumed by a LiDAR point/voxel backbone.
    The distinctive mechanism is this deep-feature-level (not label-level) cross-modal
    decoration, applied consistently through a GT-Paste augmentation pipeline that
    keeps camera and LiDAR data mutually consistent.
  - PointBeV: Chambon et al., CVPR 2024. Paper https://arxiv.org/abs/2312.00703,
    official source https://github.com/valeoai/PointBeV. Sparse bird's-eye-view (BEV)
    segmentation that predicts on a *sparse set of query points* in BEV space instead
    of a dense BEV grid. The defining mechanism is Sparse Feature Pulling: each BEV
    query point's 3D location is projected into every camera view via known
    intrinsics/extrinsics, image features are bilinearly sampled (`grid_sample`) at
    the projected pixel for each camera, and the per-camera samples are pooled
    (max) into one feature per query point -- so compute and memory scale with the
    number of sparse points, not the full BEV grid resolution, and a decoder predicts
    per-point occupancy/semantics directly from the pulled features.
  - PointContrast: Xie et al., ECCV 2020. Paper https://arxiv.org/abs/2007.10985,
    official source https://github.com/facebookresearch/PointContrast. Unsupervised
    pre-training for a 3D point-cloud backbone: two partially-overlapping views of the
    same scene are each encoded point-wise, and a PointInfoNCE contrastive loss is
    applied *directly at the level of individual corresponding points* (not
    segments/objects/cells) -- for each matched point pair (found via known
    view-to-view registration) the two point embeddings are pulled together while all
    other points in the batch act as negatives, via a temperature-scaled softmax over
    per-point cosine similarities. The point-level (finest possible) correspondence
    granularity, applied uniformly across every point in the overlap region, is the
    defining idea (contrast with cell-level BEVContrast or segment-level TARL).
  - PointPainting: Vora et al., CVPR 2020. Paper https://arxiv.org/abs/1911.10150,
    open-source reference https://github.com/Song-Jingyu/PointPainting. Sequential
    (not end-to-end) fusion of 2D semantic segmentation into 3D LiDAR detection: an
    image semantic-segmentation network first produces a per-pixel class-score
    distribution over the camera image; every LiDAR point is projected into the image
    plane via the calibration matrix and "painted" by concatenating its projected
    pixel's per-class score vector onto its (x, y, z, intensity) features; a standard
    LiDAR-only 3D detector then consumes this class-score-augmented point cloud. The
    defining mechanism is this shallow *semantic-score* (as opposed to PointAugmenting's
    deep-feature) decoration of raw points, keeping the two modalities' networks fully
    decoupled and swappable.
  - PolarNet: Zhang & Zhou et al., CVPR 2020. Paper https://arxiv.org/abs/2003.14032,
    official source https://github.com/edwardzhou130/PolarSeg. Online LiDAR
    point-cloud semantic segmentation via a polar (not Cartesian) BEV grid: each point's
    (x, y) is converted to polar coordinates (radius, angle) and quantized into a polar
    grid, which naturally balances point density across range (near-sensor points are
    dense in a Cartesian grid, but the polar grid's angular bins widen with radius,
    equalizing per-cell point counts). A per-cell PointNet-style per-point MLP
    aggregates (max-pools) the points falling into each polar ring/sector into one grid
    cell feature, producing a dense 2D polar-BEV feature map that a ring convolution
    ("ring CNN", implemented as a standard 2D CNN with circular/wrap-around padding
    along the angular axis so features are continuous across the 0/2pi boundary)
    processes into a per-point segmentation label.
  - PolyLaneNet: Tabelini et al., ICPR 2020. Paper https://arxiv.org/abs/2004.10924,
    official source https://github.com/lucastabelini/PolyLaneNet. Lane detection cast
    as direct polynomial-coefficient regression: a CNN backbone (EfficientNet in the
    official repo) extracts a global image feature which is fed through a small
    regression head that outputs, *for each of a fixed maximum number of lane slots*,
    a confidence score, the upper/lower vertical extent of the lane, and the
    coefficients of a low-degree polynomial y = f(x) describing the lane's horizontal
    position as a function of image row -- lanes are recovered analytically by
    evaluating each slot's polynomial over its predicted vertical range, with no
    anchor boxes, no segmentation mask, and no post-hoc curve fitting.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

# ---------------------------------------------------------------------------
# PointAugmenting: cross-modal deep-feature painting of LiDAR points
# ---------------------------------------------------------------------------


class ImageFeatureBackbone(nn.Module):
    """Small CNN stand-in for the DLA34 image backbone producing a dense feature map."""

    def __init__(self, in_ch: int = 3, feat_dim: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, feat_dim, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim, feat_dim, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.net(image)


class PointBackbone3D(nn.Module):
    """Compact per-point MLP standing in for a 3D voxel/point backbone."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.ReLU(inplace=True), nn.Linear(out_dim, out_dim)
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return self.mlp(points)


class PointAugmenting(nn.Module):
    """PointAugmenting: paints each LiDAR point with a deep camera feature vector.

    Each point's known (precomputed) pixel projection `(u, v)` in [-1, 1] normalized
    image coordinates is used to bilinearly sample the dense image feature map
    (`grid_sample`), giving every point a deep image-feature vector; this is
    concatenated onto the point's raw (x, y, z, intensity) features before the
    LiDAR point backbone and a detection head regress per-point objectness/box deltas.
    """

    def __init__(
        self, img_feat_dim: int = 16, point_feat_dim: int = 32, num_classes: int = 3
    ) -> None:
        super().__init__()
        self.img_backbone = ImageFeatureBackbone(in_ch=3, feat_dim=img_feat_dim)
        self.point_backbone = PointBackbone3D(in_dim=4 + img_feat_dim, out_dim=point_feat_dim)
        self.cls_head = nn.Linear(point_feat_dim, num_classes)
        self.box_head = nn.Linear(point_feat_dim, 7)

    def forward(
        self, points: torch.Tensor, image: torch.Tensor, pixel_uv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        img_feat_map = self.img_backbone(image)  # (B, C, H', W')
        grid = pixel_uv.unsqueeze(2)  # (B, N, 1, 2) for grid_sample
        sampled = F.grid_sample(img_feat_map, grid, align_corners=False)  # (B, C, N, 1)
        sampled = sampled.squeeze(-1).transpose(1, 2)  # (B, N, C)
        painted = torch.cat([points, sampled], dim=-1)
        point_feat = self.point_backbone(painted)
        return self.cls_head(point_feat), self.box_head(point_feat)


def build_pointaugmenting() -> nn.Module:
    """Build a small PointAugmenting cross-modal detection module."""
    return PointAugmenting(img_feat_dim=16, point_feat_dim=32, num_classes=3).eval()


def example_input_pointaugmenting() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(points (1,64,4), image (1,3,64,64), normalized pixel_uv (1,64,2))."""
    points = torch.randn(1, 64, 4)
    image = torch.randn(1, 3, 64, 64)
    pixel_uv = torch.rand(1, 64, 2) * 2.0 - 1.0
    return points, image, pixel_uv


# ---------------------------------------------------------------------------
# PointBeV: sparse point-query BEV segmentation via camera feature pulling
# ---------------------------------------------------------------------------


class MultiCamBackbone(nn.Module):
    """Small shared CNN over a stack of camera images, producing per-camera feature maps."""

    def __init__(self, in_ch: int = 3, feat_dim: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, feat_dim, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim, feat_dim, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        b, n_cam, c, h, w = images.shape
        feats = self.net(images.reshape(b * n_cam, c, h, w))
        return feats.reshape(b, n_cam, *feats.shape[1:])


class SparseFeaturePulling(nn.Module):
    """Bilinearly samples each sparse BEV query point from every camera, then max-pools.

    Each query's per-camera normalized pixel coordinate (precomputed from the query's
    3D BEV location + known camera calibration) is used to `grid_sample` that camera's
    feature map; the per-camera samples are max-pooled across cameras -- the sparse
    analogue of a dense Lift-Splat view transform, evaluated only at query locations.
    """

    def forward(self, cam_feats: torch.Tensor, query_uv: torch.Tensor) -> torch.Tensor:
        b, n_cam, c, h, w = cam_feats.shape
        n_pts = query_uv.shape[2]
        flat_feats = cam_feats.reshape(b * n_cam, c, h, w)
        grid = query_uv.reshape(b * n_cam, n_pts, 1, 2)
        sampled = F.grid_sample(flat_feats, grid, align_corners=False)  # (B*Ncam, C, N, 1)
        sampled = sampled.squeeze(-1).reshape(b, n_cam, c, n_pts)
        return sampled.max(dim=1).values.transpose(1, 2)  # (B, N, C)


class PointBeV(nn.Module):
    """PointBeV: sparse point-query BEV occupancy segmentation.

    A fixed sparse set of BEV query points is pulled from multi-camera image features
    (`SparseFeaturePulling`), refined with a lightweight point-wise self-attention
    (the "sparse attention" module), and decoded to a per-point occupancy/class score
    -- avoiding the memory cost of a dense BEV grid entirely.
    """

    def __init__(self, feat_dim: int = 16, n_classes: int = 2) -> None:
        super().__init__()
        self.cam_backbone = MultiCamBackbone(in_ch=3, feat_dim=feat_dim)
        self.pulling = SparseFeaturePulling()
        self.attn = nn.MultiheadAttention(feat_dim, num_heads=2, batch_first=True)
        self.head = nn.Linear(feat_dim, n_classes)

    def forward(self, images: torch.Tensor, query_uv: torch.Tensor) -> torch.Tensor:
        cam_feats = self.cam_backbone(images)
        pulled = self.pulling(cam_feats, query_uv)
        attended, _ = self.attn(pulled, pulled, pulled)
        return self.head(pulled + attended)


def build_pointbev() -> nn.Module:
    """Build a small PointBeV sparse BEV segmentation module."""
    return PointBeV(feat_dim=16, n_classes=2).eval()


def example_input_pointbev() -> tuple[torch.Tensor, torch.Tensor]:
    """(images (1, 3cams, 3, 32, 32), query_uv (1, 3cams, 40pts, 2) normalized)."""
    images = torch.randn(1, 3, 3, 32, 32)
    query_uv = torch.rand(1, 3, 40, 2) * 2.0 - 1.0
    return images, query_uv


# ---------------------------------------------------------------------------
# PointContrast: point-level contrastive pre-training via PointInfoNCE
# ---------------------------------------------------------------------------


class PointEncoder(nn.Module):
    """Compact per-point feature extractor standing in for a sparse-conv 3D backbone."""

    def __init__(self, in_dim: int = 3, feat_dim: int = 32) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, feat_dim), nn.ReLU(inplace=True), nn.Linear(feat_dim, feat_dim)
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.mlp(points), dim=-1)


class PointContrast(nn.Module):
    """PointContrast: point-level PointInfoNCE contrastive pre-training.

    Two overlapping views of the same 3D scene are each encoded point-wise; matched
    point pairs (assumed pre-aligned/ordered here for a traceable forward -- in the
    original pipeline correspondence comes from known view registration) are scored
    with a temperature-scaled cosine-similarity matrix, giving the finest-granularity
    (per-point, not per-cell/segment) contrastive supervision signal.
    """

    def __init__(self, feat_dim: int = 32, temperature: float = 0.07) -> None:
        super().__init__()
        self.encoder = PointEncoder(in_dim=3, feat_dim=feat_dim)
        self.temperature = temperature

    def forward(self, points_a: torch.Tensor, points_b: torch.Tensor) -> torch.Tensor:
        feat_a = self.encoder(points_a)  # (B, N, C)
        feat_b = self.encoder(points_b)  # (B, N, C)
        logits = torch.bmm(feat_a, feat_b.transpose(1, 2)) / self.temperature
        return logits  # (B, N, N) PointInfoNCE similarity matrix


def build_pointcontrast() -> nn.Module:
    """Build a small PointContrast point-level contrastive pretraining module."""
    return PointContrast(feat_dim=32, temperature=0.07).eval()


def example_input_pointcontrast() -> tuple[torch.Tensor, torch.Tensor]:
    """Two paired/overlapping point sets, each (1, N=48, 3) xyz coordinates."""
    points_a = torch.randn(1, 48, 3)
    points_b = points_a + 0.05 * torch.randn(1, 48, 3)
    return points_a, points_b


# ---------------------------------------------------------------------------
# PointPainting: sequential semantic-score decoration of LiDAR points
# ---------------------------------------------------------------------------


class SemanticSegHead(nn.Module):
    """Small CNN standing in for the 2D semantic segmentation network (e.g. DeepLabV3)."""

    def __init__(self, in_ch: int = 3, n_classes: int = 5) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, n_classes, 1),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(image), dim=1)  # (B, n_classes, H, W)


class LidarDetectorHead(nn.Module):
    """Compact PointNet-style detector consuming class-score-decorated points."""

    def __init__(self, in_dim: int, feat_dim: int = 32) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, feat_dim), nn.ReLU(inplace=True), nn.Linear(feat_dim, feat_dim)
        )
        self.cls_head = nn.Linear(feat_dim, 3)
        self.box_head = nn.Linear(feat_dim, 7)

    def forward(self, painted_points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.mlp(painted_points)
        return self.cls_head(feat), self.box_head(feat)


class PointPainting(nn.Module):
    """PointPainting: decorates raw LiDAR points with 2D per-pixel semantic-class scores.

    Sequential (non-end-to-end) fusion: an image semantic-segmentation network scores
    every pixel over `n_seg_classes`; each LiDAR point's known projected pixel is used
    to `grid_sample` that per-pixel score vector, which is concatenated onto the raw
    (x, y, z, intensity) point before a standard LiDAR-only detector head -- shallower
    (semantic-score-level) decoration than PointAugmenting's deep-feature painting.
    """

    def __init__(self, n_seg_classes: int = 5) -> None:
        super().__init__()
        self.seg_net = SemanticSegHead(in_ch=3, n_classes=n_seg_classes)
        self.detector = LidarDetectorHead(in_dim=4 + n_seg_classes, feat_dim=32)

    def forward(
        self, points: torch.Tensor, image: torch.Tensor, pixel_uv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seg_scores = self.seg_net(image)  # (B, n_seg_classes, H, W)
        grid = pixel_uv.unsqueeze(2)  # (B, N, 1, 2)
        sampled = F.grid_sample(seg_scores, grid, align_corners=False)  # (B, n_seg_classes, N, 1)
        sampled = sampled.squeeze(-1).transpose(1, 2)  # (B, N, n_seg_classes)
        painted = torch.cat([points, sampled], dim=-1)
        return self.detector(painted)


def build_pointpainting() -> nn.Module:
    """Build a small PointPainting sequential-fusion detection module."""
    return PointPainting(n_seg_classes=5).eval()


def example_input_pointpainting() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(points (1,64,4), image (1,3,48,48), normalized pixel_uv (1,64,2))."""
    points = torch.randn(1, 64, 4)
    image = torch.randn(1, 3, 48, 48)
    pixel_uv = torch.rand(1, 64, 2) * 2.0 - 1.0
    return points, image, pixel_uv


# ---------------------------------------------------------------------------
# PolarNet: polar-BEV grid representation + ring (wrap-padded) CNN segmentation
# ---------------------------------------------------------------------------


class PolarGridPooling(nn.Module):
    """Quantizes points into a polar (radius, angle) grid and max-pools per-cell features.

    Each point's Cartesian (x, y) is converted to polar (r, theta), discretized into a
    ring/sector grid, and a per-point MLP's features are scatter-max-pooled into their
    cell -- the polar quantization (vs. a Cartesian grid) keeps per-cell point density
    roughly uniform across range, since angular bin width grows with radius.
    """

    def __init__(self, feat_dim: int, n_rings: int, n_sectors: int, max_radius: float) -> None:
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(3, feat_dim), nn.ReLU(inplace=True), nn.Linear(feat_dim, feat_dim)
        )
        self.n_rings = n_rings
        self.n_sectors = n_sectors
        self.max_radius = max_radius

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        b, n, _ = points.shape
        x, y, z = points[..., 0], points[..., 1], points[..., 2]
        r = torch.sqrt(x**2 + y**2).clamp(max=self.max_radius)
        theta = torch.atan2(y, x)  # (-pi, pi]

        ring_idx = (
            (r / self.max_radius * (self.n_rings - 1)).round().long().clamp(0, self.n_rings - 1)
        )
        sector_idx = (
            ((theta + torch.pi) / (2 * torch.pi)) * self.n_sectors
        ).long() % self.n_sectors
        flat_idx = ring_idx * self.n_sectors + sector_idx  # (B, N)

        feats = self.point_mlp(torch.stack([r, theta, z], dim=-1))  # (B, N, C)
        c = feats.shape[-1]
        n_cells = self.n_rings * self.n_sectors
        grid = feats.new_full((b, n_cells, c), float("-inf"))
        grid.scatter_reduce_(1, flat_idx.unsqueeze(-1).expand(-1, -1, c), feats, reduce="amax")
        grid = torch.where(torch.isinf(grid), torch.zeros_like(grid), grid)
        return grid.transpose(1, 2).view(b, c, self.n_rings, self.n_sectors)


class RingConv2d(nn.Module):
    """2D conv with circular padding along the angular (sector) axis -- a "ring" CNN layer.

    Wraps the sector dimension so features near theta=+pi and theta=-pi see each
    other as neighbors (no artificial seam at the polar angle's wrap-around point),
    while the radial (ring) axis uses ordinary zero padding.
    """

    def __init__(self, in_ch: int, out_ch: int, k: int = 3) -> None:
        super().__init__()
        self.pad = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, k, padding=(self.pad, 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.pad, self.pad, 0, 0), mode="circular")
        return self.conv(x)


class PolarNet(nn.Module):
    """PolarNet: polar-BEV grid pooling + ring-CNN LiDAR semantic segmentation head."""

    def __init__(
        self,
        feat_dim: int = 16,
        n_rings: int = 12,
        n_sectors: int = 24,
        max_radius: float = 20.0,
        n_classes: int = 6,
    ) -> None:
        super().__init__()
        self.pooling = PolarGridPooling(feat_dim, n_rings, n_sectors, max_radius)
        self.ring_conv1 = RingConv2d(feat_dim, feat_dim, k=3)
        self.ring_conv2 = RingConv2d(feat_dim, n_classes, k=3)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        grid = self.pooling(points)
        h = F.relu(self.ring_conv1(grid))
        return self.ring_conv2(h)  # (B, n_classes, n_rings, n_sectors)


def build_polarnet() -> nn.Module:
    """Build a small PolarNet polar-BEV LiDAR segmentation module."""
    return PolarNet(feat_dim=16, n_rings=12, n_sectors=24, max_radius=20.0, n_classes=6).eval()


def example_input_polarnet() -> torch.Tensor:
    """A single LiDAR scan: (1, N=200, 3) xyz points."""
    return torch.randn(1, 200, 3) * 5.0


# ---------------------------------------------------------------------------
# PolyLaneNet: direct polynomial-coefficient lane regression
# ---------------------------------------------------------------------------


class PolyLaneNet(nn.Module):
    """PolyLaneNet: CNN backbone -> direct per-lane-slot polynomial regression.

    A CNN encodes the input image to a single global feature vector; a linear head
    regresses, for each of `max_lanes` fixed output slots, one confidence score, the
    lane's upper/lower vertical extent, and `poly_degree + 1` polynomial coefficients
    describing horizontal lane position as a function of image row -- lanes are
    recovered analytically from the polynomial with no anchors, masks, or NMS.
    """

    def __init__(self, max_lanes: int = 4, poly_degree: int = 3, feat_dim: int = 32) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, feat_dim, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim, feat_dim, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.max_lanes = max_lanes
        self.poly_degree = poly_degree
        # Per lane slot: 1 confidence + 2 (upper, lower) + (poly_degree + 1) coefficients.
        out_per_lane = 1 + 2 + (poly_degree + 1)
        self.head = nn.Linear(feat_dim, max_lanes * out_per_lane)
        self._out_per_lane = out_per_lane

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(image).flatten(1)
        out = self.head(feat)
        return out.view(-1, self.max_lanes, self._out_per_lane)


def build_polylanenet() -> nn.Module:
    """Build a small PolyLaneNet direct polynomial lane-regression module."""
    return PolyLaneNet(max_lanes=4, poly_degree=3, feat_dim=32).eval()


def example_input_polylanenet() -> torch.Tensor:
    """A single road-scene image, (1, 3, 64, 64)."""
    return torch.randn(1, 3, 64, 64)


MENAGERIE_ENTRIES = [
    ("PointAugmenting", "build_pointaugmenting", "example_input_pointaugmenting", "2021", "VIS"),
    ("PointBEV", "build_pointbev", "example_input_pointbev", "2024", "VIS"),
    ("PointContrast", "build_pointcontrast", "example_input_pointcontrast", "2020", "VIS"),
    ("PointPainting", "build_pointpainting", "example_input_pointpainting", "2020", "VIS"),
    ("PolarNet", "build_polarnet", "example_input_polarnet", "2020", "VIS"),
    ("PolyLaneNet", "build_polylanenet", "example_input_polylanenet", "2020", "VIS"),
]
