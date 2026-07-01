"""Menagerie batch w3a20: LiDAR/radar range-view and BEV 3D-perception architectures.

Sources checked (reference only; no cloning, no pip installs):
  - Pyramid R-CNN: Mao et al., ICCV 2021. Paper https://arxiv.org/abs/2109.02499,
    official source https://github.com/PointsCoder/Pyramid-RCNN. Two-stage 3D point-cloud
    detector whose distinctive second stage is a "pyramid RoI head": for each proposal
    RoI, a **RoI-grid pyramid** collects points of interest at *multiple concentric
    search radii* around each grid point (rather than one fixed radius), a **RoI-grid
    attention** operator aggregates the per-radius point sets with attention-weighted
    pooling, and a **Density-Aware Radius Prediction (DARP)** module predicts, from the
    local point density around each grid point, which radius level to trust most --
    adaptively widening the search when points are sparse (far away) and narrowing it
    when points are dense (nearby). This directly fixes the sparsity/non-uniformity
    problem that plain fixed-radius ball-query RoI pooling (e.g. PV-RCNN's RoI-grid
    pooling) suffers on far-away objects.
  - RADDet: Zhang, Nowruzi & Laganiere, IV 2021. Paper https://arxiv.org/abs/2105.00363
    (candidate row cites 2105.05116, but the RADDet paper's actual id is 2105.00363),
    official source https://github.com/ZhangAoCanada/RADDet. One-stage anchor-based
    detector operating directly on the **Range-Azimuth-Doppler (RAD) tensor** -- a dense
    3D radar cube (range x azimuth x Doppler) taken straight from the radar signal chain
    after the azimuth-angle FFT, *before* any BEV collapse. A 3D-conv backbone extracts
    features jointly across all three physical axes (so Doppler/velocity information is
    never discarded, unlike range-azimuth-only BEV radar detectors), then a detection
    head with anchors placed on the range-azimuth plane regresses per-cell
    objectness/class plus 3D box parameters (range, azimuth, Doppler extent) directly
    from the RAD volume.
  - RangeDet: Fan, Xiong & Wang et al., ICCV 2021. Paper https://arxiv.org/abs/2103.10039,
    official source https://github.com/tusen-ai/RangeDet. Single-stage, anchor-free LiDAR
    3D detector operating purely in the **range-image** view (the raw (range, azimuth,
    elevation) projection of the point cloud, one point per pixel; no voxelization/BEV).
    Two mechanisms address range-view-specific issues: (1) a **Meta-Kernel convolution**
    -- a per-pixel input-conditioned convolution whose kernel weights are generated (via a
    small MLP) from the *relative 3D xyz offsets* of each neighboring pixel's point to the
    center pixel's point, so the convolution is aware of real 3D geometry rather than only
    2D range-image adjacency (fixing the mismatch between 2D image-grid neighborhoods and
    3D Euclidean neighborhoods); and (2) a **Range Conditioned Pyramid (RCP)** -- multiple
    parallel prediction heads, one per range interval, so the network can learn
    scale-appropriate box regression separately for near (large-apparent) and far
    (small-apparent) objects instead of a single shared-scale head.
  - RangeNet++: Milioto, Vizzo, Behley & Stachniss, IROS 2019. Paper
    https://arxiv.org/abs/1904.01831, official source
    https://github.com/PRBonn/lidar-bonnetal. Real-time LiDAR point-cloud semantic
    segmentation via spherical range-image projection: every 3D point is projected by its
    (yaw, pitch) spherical angles onto a fixed-size 2D range image storing per-pixel
    (range, x, y, z, remission); a 2D CNN encoder-decoder (Darknet-inspired backbone,
    here compacted) performs dense per-pixel semantic segmentation on this range image
    exactly like ordinary image segmentation; each labeled pixel's class is then
    transferred back to its source 3D point, and a **GPU k-nearest-neighbor (kNN)
    post-processing** step reassigns each point's final label by a majority vote over its
    k nearest *3D* neighbors (using true xyz distance, not 2D range-image adjacency) to
    repair "bleeding" artifacts at depth-discontinuity boundaries where the 2D projection
    puts unrelated 3D points next to each other.
  - RCBEVDet (candidate name "RCBEV"): Lin et al., CVPR 2024. Paper
    https://arxiv.org/abs/2403.16440, official source
    https://github.com/VDIGPKU/RCBEVDet. Radar-camera fusion 3D detector in bird's-eye
    view. Two distinctive components: **RadarBEVNet** -- a dual-stream radar point
    encoder combining a per-point PointNet-style branch with a transformer branch (with an
    injection/extraction module letting the two streams exchange information), scattered
    into a BEV radar feature map that is further reweighted by predicted per-cell
    **Radar-Cross-Section (RCS)**-aware attention (RCS is a physical radar return-strength
    cue correlated with object size, used to help place point energy at the right BEV
    extent despite radar's poor angular resolution); and **CAMF (Cross-Attention
    Multi-layer Fusion)** -- a deformable cross-attention module where each BEV query
    location adaptively samples and fuses camera-BEV features around a *learned offset*
    from the radar-BEV feature at that location, correcting for radar's azimuth-angle
    localization error rather than assuming pixel-exact radar/camera BEV alignment.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# Pyramid R-CNN: multi-radius RoI-grid pyramid + density-aware radius weighting
# ---------------------------------------------------------------------------


class PointVoxelBackbone(nn.Module):
    """Compact per-point MLP standing in for the first-stage 3D backbone."""

    def __init__(self, in_dim: int = 3, feat_dim: int = 32) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, feat_dim), nn.ReLU(inplace=True), nn.Linear(feat_dim, feat_dim)
        )

    def forward(self, points: Tensor) -> Tensor:
        return self.mlp(points)


class PyramidRoIHead(nn.Module):
    """RoI-grid pyramid pooling with density-aware radius-level fusion.

    For each of `n_grid` grid points per RoI, point features within *multiple*
    concentric radii (`radii`) are max-pooled separately (the "pyramid" of search
    scales); a small DARP (density-aware radius prediction) head looks at how many
    raw points fell within the smallest radius (a density proxy) and predicts a
    softmax weighting over the radius levels, so RoIs backed by sparse/far point
    sets lean on the wider radii while dense/near RoIs lean on the tighter radius.
    """

    def __init__(
        self, feat_dim: int, n_grid: int = 4, radii: tuple[float, ...] = (0.5, 1.0, 2.0)
    ) -> None:
        super().__init__()
        self.n_grid = n_grid
        self.radii = radii
        self.n_levels = len(radii)
        self.darp = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU(inplace=True), nn.Linear(16, self.n_levels)
        )
        self.grid_offsets = nn.Parameter(torch.randn(n_grid, 3) * 0.3, requires_grad=False)

    def forward(self, points: Tensor, point_feat: Tensor, roi_centers: Tensor) -> Tensor:
        b, n_roi, _ = roi_centers.shape
        n_pts = points.shape[1]
        grid_pts = roi_centers.unsqueeze(2) + self.grid_offsets.view(1, 1, self.n_grid, 3)
        # (B, n_roi, n_grid, N) pairwise distances from each grid point to every raw point.
        diff = grid_pts.unsqueeze(3) - points.unsqueeze(1).unsqueeze(1)
        dist = diff.norm(dim=-1)

        level_feats = []
        densities = []
        for radius in self.radii:
            mask = (dist <= radius).float()  # (B, n_roi, n_grid, N)
            counts = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
            densities.append(counts)
            weighted = mask.unsqueeze(-1) * point_feat.view(b, 1, 1, n_pts, -1)
            pooled = weighted.sum(dim=3) / counts
            level_feats.append(pooled)  # (B, n_roi, n_grid, C)

        stacked = torch.stack(level_feats, dim=-2)  # (B, n_roi, n_grid, n_levels, C)
        density_feat = densities[0]  # smallest-radius count is the density proxy
        level_weights = F.softmax(self.darp(density_feat), dim=-1)  # (B, n_roi, n_grid, n_levels)
        fused = (stacked * level_weights.unsqueeze(-1)).sum(dim=-2)  # (B, n_roi, n_grid, C)
        return fused.flatten(2)  # (B, n_roi, n_grid * C)


class PyramidRCNN(nn.Module):
    """Pyramid R-CNN: point backbone + multi-radius pyramid RoI-grid refinement head."""

    def __init__(self, feat_dim: int = 32, n_grid: int = 4, n_classes: int = 3) -> None:
        super().__init__()
        self.backbone = PointVoxelBackbone(in_dim=3, feat_dim=feat_dim)
        self.roi_head = PyramidRoIHead(feat_dim, n_grid=n_grid, radii=(0.5, 1.0, 2.0))
        refine_in = n_grid * feat_dim
        self.cls_head = nn.Linear(refine_in, n_classes)
        self.box_head = nn.Linear(refine_in, 7)

    def forward(self, points: Tensor, roi_centers: Tensor) -> tuple[Tensor, Tensor]:
        point_feat = self.backbone(points)
        roi_feat = self.roi_head(points, point_feat, roi_centers)
        return self.cls_head(roi_feat), self.box_head(roi_feat)


def build_pyramidrcnn() -> nn.Module:
    """Build a small Pyramid R-CNN density-aware pyramid RoI-head detector."""
    return PyramidRCNN(feat_dim=32, n_grid=4, n_classes=3).eval()


def example_input_pyramidrcnn() -> tuple[Tensor, Tensor]:
    """(points (1, N=96, 3) raw LiDAR xyz, roi_centers (1, n_roi=6, 3) proposal centers)."""
    points = torch.randn(1, 96, 3) * 3.0
    roi_centers = torch.randn(1, 6, 3) * 2.0
    return points, roi_centers


# ---------------------------------------------------------------------------
# RADDet: 3D-conv detection directly on the Range-Azimuth-Doppler radar tensor
# ---------------------------------------------------------------------------


class RADBackbone3D(nn.Module):
    """3D-conv backbone over the (range, azimuth, Doppler) radar cube.

    Convolving jointly across all three physical axes keeps the Doppler/velocity
    dimension in the feature extraction instead of collapsing it away first (as a
    BEV-only range-azimuth radar detector would), which is RADDet's defining choice.
    """

    def __init__(self, in_ch: int = 1, feat_dim: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, feat_dim, 3, padding=1),
            nn.BatchNorm3d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Conv3d(feat_dim, feat_dim * 2, 3, stride=2, padding=1),
            nn.BatchNorm3d(feat_dim * 2),
            nn.ReLU(inplace=True),
        )

    def forward(self, rad: Tensor) -> Tensor:
        return self.net(rad)


class RADDet(nn.Module):
    """RADDet: one-stage anchor-based detector on the dense RAD radar tensor.

    A 3D-conv backbone extracts range-azimuth-Doppler features; anchors are placed
    on the range-azimuth plane (Doppler is pooled into the per-anchor feature) and a
    detection head regresses per-cell class scores plus 3D (range, azimuth, Doppler
    extent) box parameters straight from the RAD volume.
    """

    def __init__(self, feat_dim: int = 16, n_anchors: int = 3, n_classes: int = 4) -> None:
        super().__init__()
        self.backbone = RADBackbone3D(in_ch=1, feat_dim=feat_dim)
        out_ch = feat_dim * 2
        self.cls_head = nn.Conv2d(out_ch, n_anchors * n_classes, 1)
        self.box_head = nn.Conv2d(out_ch, n_anchors * 6, 1)
        self.n_anchors = n_anchors
        self.n_classes = n_classes

    def forward(self, rad_tensor: Tensor) -> tuple[Tensor, Tensor]:
        feat = self.backbone(rad_tensor)  # (B, C, R', A', D')
        bev = feat.mean(dim=4)  # collapse Doppler into per-cell context for the head
        cls = self.cls_head(bev)
        box = self.box_head(bev)
        return cls, box


def build_raddet() -> nn.Module:
    """Build a small RADDet range-azimuth-Doppler 3D-conv radar detector."""
    return RADDet(feat_dim=16, n_anchors=3, n_classes=4).eval()


def example_input_raddet() -> Tensor:
    """A single RAD radar cube: (1, 1, range=32, azimuth=32, doppler=16)."""
    return torch.randn(1, 1, 32, 32, 16)


# ---------------------------------------------------------------------------
# RangeDet: range-view LiDAR detection via Meta-Kernel conv + Range Conditioned Pyramid
# ---------------------------------------------------------------------------


class MetaKernelConv(nn.Module):
    """Per-pixel input-conditioned convolution driven by real 3D xyz offsets.

    For every output pixel, the kxk neighborhood's *3D xyz offsets relative to the
    center pixel's point* are fed through a small shared MLP to produce per-neighbor
    weight vectors, which then reweight that neighbor's range-image feature before
    summation -- so the "convolution" adapts to true 3D geometry instead of treating
    all 2D-adjacent range-image pixels as equally relevant neighbors.
    """

    def __init__(self, in_ch: int, out_ch: int, k: int = 3) -> None:
        super().__init__()
        self.k = k
        self.pad = k // 2
        self.unfold = nn.Unfold(kernel_size=k, padding=self.pad)
        self.offset_mlp = nn.Sequential(
            nn.Linear(3, 16), nn.ReLU(inplace=True), nn.Linear(16, in_ch)
        )
        self.proj = nn.Conv1d(in_ch, out_ch, 1)

    def forward(self, feat: Tensor, xyz: Tensor) -> Tensor:
        b, c, h, w = feat.shape
        neighborhood = self.unfold(feat).view(b, c, self.k * self.k, h * w)  # (B, C, k^2, HW)
        xyz_unf = self.unfold(xyz).view(b, 3, self.k * self.k, h * w)  # (B, 3, k^2, HW)
        center_xyz = xyz.view(b, 3, 1, h * w)
        rel_offset = (xyz_unf - center_xyz).permute(0, 3, 2, 1)  # (B, HW, k^2, 3)
        weights = torch.sigmoid(self.offset_mlp(rel_offset)).permute(0, 3, 2, 1)  # (B, C, k^2, HW)
        weighted = (neighborhood * weights).sum(dim=2)  # (B, C, HW)
        return self.proj(weighted).view(b, -1, h, w)


class RangeConditionedPyramid(nn.Module):
    """Multiple range-interval-specific prediction heads sharing a common feature.

    Splits pixels into near/mid/far range buckets (by their scalar range channel)
    and lets a separate small head regress each bucket's boxes, so near
    (large-apparent) and far (small-apparent) objects get scale-appropriate
    regression instead of one shared-scale head.
    """

    def __init__(
        self, feat_dim: int, n_classes: int, n_bands: int = 3, max_range: float = 40.0
    ) -> None:
        super().__init__()
        self.n_bands = n_bands
        self.max_range = max_range
        self.band_heads = nn.ModuleList(
            [nn.Conv2d(feat_dim, n_classes + 7, 1) for _ in range(n_bands)]
        )

    def forward(self, feat: Tensor, range_channel: Tensor) -> Tensor:
        band_idx = (range_channel / self.max_range * self.n_bands).long().clamp(0, self.n_bands - 1)
        outputs = torch.stack(
            [head(feat) for head in self.band_heads], dim=1
        )  # (B, n_bands, C_out, H, W)
        band_idx_exp = band_idx.unsqueeze(2).expand(
            -1, -1, outputs.shape[2], -1, -1
        )  # (B, 1, C_out, H, W)
        selected = torch.gather(outputs, 1, band_idx_exp).squeeze(1)
        return selected  # (B, C_out, H, W)


class RangeDet(nn.Module):
    """RangeDet: range-image LiDAR detector with Meta-Kernel conv + Range Conditioned Pyramid."""

    def __init__(self, feat_dim: int = 24, n_classes: int = 3) -> None:
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(5, feat_dim, 3, padding=1), nn.ReLU(inplace=True))
        self.meta_kernel = MetaKernelConv(feat_dim, feat_dim, k=3)
        self.rcp_head = RangeConditionedPyramid(feat_dim, n_classes, n_bands=3, max_range=40.0)

    def forward(self, range_image: Tensor, xyz_image: Tensor) -> Tensor:
        feat = self.stem(range_image)
        feat = F.relu(self.meta_kernel(feat, xyz_image))
        range_channel = range_image[:, 0:1]
        return self.rcp_head(feat, range_channel)


def build_rangedet() -> nn.Module:
    """Build a small RangeDet Meta-Kernel + Range Conditioned Pyramid detector."""
    return RangeDet(feat_dim=24, n_classes=3).eval()


def example_input_rangedet() -> tuple[Tensor, Tensor]:
    """(range_image (1,5,32,64) [range,x,y,z,intensity], xyz_image (1,3,32,64))."""
    range_image = torch.rand(1, 5, 32, 64) * 20.0
    xyz_image = torch.randn(1, 3, 32, 64) * 10.0
    return range_image, xyz_image


# ---------------------------------------------------------------------------
# RangeNet++: spherical range-image segmentation + 3D-kNN post-processing
# ---------------------------------------------------------------------------


class RangeImageEncoderDecoder(nn.Module):
    """Compact Darknet-inspired encoder-decoder for per-pixel range-image segmentation."""

    def __init__(self, in_ch: int = 5, feat_dim: int = 24, n_classes: int = 6) -> None:
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(in_ch, feat_dim, 3, padding=1), nn.ReLU(inplace=True))
        self.enc2 = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim * 2, 3, stride=2, padding=1), nn.ReLU(inplace=True)
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(feat_dim * 2, feat_dim, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(feat_dim, n_classes, 1)

    def forward(self, range_image: Tensor) -> Tensor:
        e1 = self.enc1(range_image)
        e2 = self.enc2(e1)
        d1 = self.dec1(e2) + e1
        return self.head(d1)  # (B, n_classes, H, W)


class KnnLabelSmoothing(nn.Module):
    """GPU k-nearest-neighbor post-processing over true 3D xyz distance.

    Repairs "bleeding" at depth-discontinuity boundaries: each point's final class
    logit is replaced by a distance-weighted average of its `k` nearest neighbors'
    logits *in 3D Euclidean space* (not 2D range-image adjacency), so points that
    happen to be image-adjacent but are actually far apart in 3D (e.g. a foreground
    object edge against background) no longer contaminate each other's label.
    """

    def __init__(self, k: int = 5) -> None:
        super().__init__()
        self.k = k

    def forward(self, point_xyz: Tensor, point_logits: Tensor) -> Tensor:
        dist = torch.cdist(point_xyz, point_xyz)  # (B, N, N)
        k = min(self.k, dist.shape[-1])
        neg_dist, idx = torch.topk(-dist, k=k, dim=-1)  # nearest k, including self
        weights = F.softmax(neg_dist, dim=-1)  # closer neighbors weighted more heavily
        gathered = torch.gather(
            point_logits.unsqueeze(1).expand(-1, point_logits.shape[1], -1, -1),
            2,
            idx.unsqueeze(-1).expand(-1, -1, -1, point_logits.shape[-1]),
        )  # (B, N, k, n_classes)
        return (gathered * weights.unsqueeze(-1)).sum(dim=2)


class RangeNetPP(nn.Module):
    """RangeNet++: range-image CNN segmentation + point-transfer + 3D-kNN smoothing."""

    def __init__(self, feat_dim: int = 24, n_classes: int = 6, knn_k: int = 5) -> None:
        super().__init__()
        self.net = RangeImageEncoderDecoder(in_ch=5, feat_dim=feat_dim, n_classes=n_classes)
        self.knn = KnnLabelSmoothing(k=knn_k)

    def forward(self, range_image: Tensor, pixel_uv: Tensor, point_xyz: Tensor) -> Tensor:
        logits_2d = self.net(range_image)  # (B, n_classes, H, W)
        grid = pixel_uv.unsqueeze(2)  # (B, N, 1, 2) normalized coords for each source point
        transferred = F.grid_sample(logits_2d, grid, align_corners=False)  # (B, n_classes, N, 1)
        point_logits = transferred.squeeze(-1).transpose(1, 2)  # (B, N, n_classes)
        return self.knn(point_xyz, point_logits)


def build_rangenetpp() -> nn.Module:
    """Build a small RangeNet++ range-image segmentation + kNN post-processing module."""
    return RangeNetPP(feat_dim=24, n_classes=6, knn_k=5).eval()


def example_input_rangenetpp() -> tuple[Tensor, Tensor, Tensor]:
    """(range_image (1,5,32,64), pixel_uv (1,N=80,2) normalized, point_xyz (1,80,3))."""
    range_image = torch.rand(1, 5, 32, 64) * 20.0
    pixel_uv = torch.rand(1, 80, 2) * 2.0 - 1.0
    point_xyz = torch.randn(1, 80, 3) * 5.0
    return range_image, pixel_uv, point_xyz


# ---------------------------------------------------------------------------
# RCBEVDet (RCBEV): RadarBEVNet dual-stream encoder + CAMF deformable radar-camera fusion
# ---------------------------------------------------------------------------


class RadarBEVNet(nn.Module):
    """Dual-stream radar point encoder scattered into an RCS-reweighted BEV feature map.

    A per-point MLP stream and a lightweight self-attention (transformer) stream both
    encode the radar points; their outputs are summed (the injection/extraction
    exchange, simplified to an additive fusion) before points are scatter-mean-pooled
    into BEV grid cells. Each cell's pooled feature is then rescaled by a sigmoid gate
    predicted from that cell's mean Radar-Cross-Section (RCS) value -- a physical
    return-strength cue correlated with object size, letting the network trust
    high-RCS cells more despite radar's poor angular localization.
    """

    def __init__(self, feat_dim: int = 24, grid_size: int = 16, max_range: float = 20.0) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.max_range = max_range
        self.point_stream = nn.Sequential(
            nn.Linear(4, feat_dim), nn.ReLU(inplace=True), nn.Linear(feat_dim, feat_dim)
        )
        self.attn_stream = nn.MultiheadAttention(feat_dim, num_heads=2, batch_first=True)
        self.attn_in = nn.Linear(4, feat_dim)
        self.rcs_gate = nn.Sequential(nn.Linear(1, feat_dim), nn.Sigmoid())

    def forward(self, radar_points: Tensor) -> Tensor:
        b, n, _ = radar_points.shape
        xy = radar_points[..., :2]
        rcs = radar_points[..., 3:4]

        point_feat = self.point_stream(radar_points)
        attn_in = self.attn_in(radar_points)
        attn_feat, _ = self.attn_stream(attn_in, attn_in, attn_in)
        fused = point_feat + attn_feat  # injection/extraction stream exchange, simplified

        cell = ((xy / self.max_range * 0.5 + 0.5) * (self.grid_size - 1)).round().long()
        cell = cell.clamp(0, self.grid_size - 1)
        flat_idx = cell[..., 0] * self.grid_size + cell[..., 1]  # (B, N)

        c = fused.shape[-1]
        n_cells = self.grid_size * self.grid_size
        bev = fused.new_zeros(b, n_cells, c)
        counts = fused.new_zeros(b, n_cells, 1)
        bev.scatter_add_(1, flat_idx.unsqueeze(-1).expand(-1, -1, c), fused)
        counts.scatter_add_(1, flat_idx.unsqueeze(-1), fused.new_ones(b, n, 1))
        bev = bev / counts.clamp(min=1.0)

        rcs_gate = fused.new_zeros(b, n_cells, 1)
        rcs_gate.scatter_add_(1, flat_idx.unsqueeze(-1), rcs)
        rcs_gate = rcs_gate / counts.clamp(min=1.0)
        bev = bev * self.rcs_gate(rcs_gate)

        return bev.transpose(1, 2).view(b, c, self.grid_size, self.grid_size)


class CamfDeformableFusion(nn.Module):
    """Cross-Attention Multi-layer Fusion: learned-offset deformable radar-camera fusion.

    For every radar-BEV query location, a small offset head predicts a spatial
    offset (in normalized BEV coordinates) at which to sample the camera-BEV
    feature map, then that sampled camera feature is fused (concatenate + project)
    back with the radar feature -- correcting for radar's azimuth-angle
    localization error rather than assuming the two modalities are pixel-aligned.
    """

    def __init__(self, feat_dim: int) -> None:
        super().__init__()
        self.offset_head = nn.Conv2d(feat_dim, 2, 3, padding=1)
        self.fuse = nn.Conv2d(feat_dim * 2, feat_dim, 1)

    def forward(self, radar_bev: Tensor, camera_bev: Tensor) -> Tensor:
        b, c, h, w = radar_bev.shape
        offsets = torch.tanh(self.offset_head(radar_bev)) * 0.5  # bounded sample-offset field

        ys, xs = torch.meshgrid(
            torch.linspace(-1, 1, h, device=radar_bev.device),
            torch.linspace(-1, 1, w, device=radar_bev.device),
            indexing="ij",
        )
        base_grid = torch.stack([xs, ys], dim=0).unsqueeze(0).expand(b, -1, -1, -1)
        sample_grid = (base_grid + offsets).permute(0, 2, 3, 1)  # (B, H, W, 2)

        sampled_camera = F.grid_sample(camera_bev, sample_grid, align_corners=False)
        fused = torch.cat([radar_bev, sampled_camera], dim=1)
        return self.fuse(fused)


class RCBEVDet(nn.Module):
    """RCBEVDet: RadarBEVNet radar encoder + camera BEV backbone + CAMF deformable fusion."""

    def __init__(self, feat_dim: int = 24, grid_size: int = 16, n_classes: int = 5) -> None:
        super().__init__()
        self.radar_net = RadarBEVNet(feat_dim=feat_dim, grid_size=grid_size, max_range=20.0)
        self.camera_backbone = nn.Sequential(
            nn.Conv2d(3, feat_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim, feat_dim, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.camf = CamfDeformableFusion(feat_dim)
        self.cls_head = nn.Conv2d(feat_dim, n_classes, 1)
        self.box_head = nn.Conv2d(feat_dim, 7, 1)

    def forward(self, radar_points: Tensor, camera_bev_image: Tensor) -> tuple[Tensor, Tensor]:
        radar_bev = self.radar_net(radar_points)
        camera_bev = self.camera_backbone(camera_bev_image)
        camera_bev = F.interpolate(
            camera_bev, size=radar_bev.shape[-2:], mode="bilinear", align_corners=False
        )
        fused = self.camf(radar_bev, camera_bev)
        return self.cls_head(fused), self.box_head(fused)


def build_rcbevdet() -> nn.Module:
    """Build a small RCBEVDet RadarBEVNet + CAMF radar-camera BEV fusion detector."""
    return RCBEVDet(feat_dim=24, grid_size=16, n_classes=5).eval()


def example_input_rcbevdet() -> tuple[Tensor, Tensor]:
    """(radar_points (1, N=64, 4) [x,y,z,rcs], camera_bev_image (1,3,32,32))."""
    radar_points = torch.randn(1, 64, 4) * 5.0
    radar_points[..., 3] = torch.rand(1, 64).abs()  # RCS is non-negative
    camera_bev_image = torch.randn(1, 3, 32, 32)
    return radar_points, camera_bev_image


MENAGERIE_ENTRIES = [
    ("Pyramid R-CNN for 3D", "build_pyramidrcnn", "example_input_pyramidrcnn", "2021", "VIS"),
    ("RADDet", "build_raddet", "example_input_raddet", "2021", "VIS"),
    ("RangeDet", "build_rangedet", "example_input_rangedet", "2021", "VIS"),
    ("RangeNet++", "build_rangenetpp", "example_input_rangenetpp", "2019", "VIS"),
    ("RCBEV", "build_rcbevdet", "example_input_rcbevdet", "2024", "VIS"),
]
