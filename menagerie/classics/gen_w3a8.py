"""Menagerie batch w3a8: camera/LiDAR bird's-eye-view (BEV) 3D perception architectures.

Sources checked (reference only; no cloning, no pip installs):
  - AutoAlignV2: Chen et al., ECCV/IJCAI 2022. Paper https://arxiv.org/abs/2207.10316,
    official source https://github.com/zehuichen123/AutoAlignV2. Multi-modal (camera +
    LiDAR) 3D detection built on AutoAlign. The distinctive mechanism is the Cross-Domain
    DeformCAFA module: for each LiDAR voxel/point query, a small set of *learnable
    sampling offsets* (predicted from the query feature, in the style of Deformable DETR)
    are added to a nominal camera-projected reference point, and image features are
    bilinearly sampled (`grid_sample`) at those sparse offset locations and aggregated
    via query-conditioned attention weights -- replacing AutoAlign's expensive global
    cross-attention with sparse deformable sampling, which both speeds up fusion and
    tolerates camera-LiDAR calibration error.
  - AVOD (Aggregate View Object Detection): Ku et al., IROS 2018. Paper
    https://arxiv.org/abs/1712.02294, official source https://github.com/kujason/avod.
    Two-stream (BEV LiDAR-height-map + front-view camera image) feature-pyramid encoders
    feed a shared Region Proposal Network: 3D anchors are projected into *both* the BEV
    and image feature maps, per-anchor RoI-cropped features from each view are fused
    (element-wise mean, "the proposed RPN performs multimodal feature fusion on high
    resolution feature maps"), and the fused feature drives objectness + 3D box-proposal
    regression. A second-stage detector head re-crops fused features at the proposal
    boxes for final oriented 3D box regression + classification -- the "aggregate view"
    (BEV + image, fused at two stages) is the defining idea.
  - BEV-LaneDet: Wang et al., CVPR 2023. Paper https://arxiv.org/abs/2210.06006, official
    source https://github.com/gigo-team/bev_lane_det. Monocular 3D lane detection with
    three contributions folded into one compact pipeline: (1) a Virtual Camera -- a fixed
    canonical extrinsic/intrinsic homography that every input image is implicitly
    re-projected through before feature extraction, decoupling the network from a
    particular vehicle's camera mount; (2) a Spatial Transformation Pyramid -- multiple
    front-view feature-pyramid levels are each projected to the BEV plane via a per-level
    learned MLP acting on the flattened spatial dimension (view-relation-module-style
    dense projection, applied at multiple scales then summed) rather than a heavy 3D
    lifting step; (3) Key-Points Representation -- the BEV head predicts, per BEV grid
    cell, a lane-existence heatmap plus a local (dx, dz) offset and embedding vector
    (for clustering points into lane instances), instead of parametric curve coefficients.
  - BEVContrast: Sautier et al., 3DV 2024. Paper https://arxiv.org/abs/2310.17281,
    official source https://github.com/valeoai/BEVContrast. Self-supervised
    pretraining of a LiDAR point-cloud backbone via a contrastive loss defined at the
    granularity of 2D BEV grid cells (a middle ground between point-level PointContrast
    and segment/object-level TARL). Two overlapping LiDAR scans of the same scene are
    each voxelized to a BEV feature grid by scatter-mean-pooling per-point backbone
    features into their (x, y) BEV cell; corresponding cells between the two views
    (matched via known ego-motion) are pulled together and non-corresponding cells
    pushed apart with an InfoNCE-style contrastive loss over the pooled cell
    embeddings.
  - BEVDet: Huang et al., 2021. Paper https://arxiv.org/abs/2112.11790, official source
    https://github.com/HuangJunJie2017/BEVDet. Pure-camera multi-view 3D detection via
    the Lift-Splat-Shoot (LSS) view transformer: a shared image backbone extracts
    per-camera features, a small head predicts a per-pixel categorical depth
    distribution over discrete depth bins, each pixel feature is "lifted" into a 3D
    frustum by outer-producting the feature with its depth distribution, the frustum
    points are projected with known camera extrinsics into ego/BEV coordinates and
    "splat"-pooled (scatter-sum into BEV grid cells) into a unified multi-camera BEV
    feature map, and a BEV-space CNN backbone + detection head regresses per-cell
    objectness and 3D box parameters -- the "lift-splat" view transform is the
    architecture's defining mechanism.
  - BEVDet4D: Huang & Huang, 2022. Paper https://arxiv.org/abs/2203.17054, code
    integrated into the same https://github.com/HuangJunJie2017/BEVDet repo. Extends
    BEVDet from spatial-only 3D to spatial-temporal 4D by keeping the previous frame's
    BEV feature map, warping/aligning it into the current frame's ego frame via the
    known ego-motion transform (a BEV-plane affine `grid_sample`), and concatenating
    the aligned previous-frame BEV feature with the current-frame BEV feature before
    the BEV backbone -- giving the detection+velocity head access to temporal cues
    with negligible extra compute versus single-frame BEVDet.

All models below are compact, faithfully-reimplemented-from-scratch nn.Modules with
random init and small dims for TorchLens architecture-catalog tracing (not a
trained-weights zoo).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# Shared helpers
# ============================================================


def _cbr(in_ch: int, out_ch: int, k: int = 3, stride: int = 1, padding: int = 1) -> nn.Sequential:
    """Conv-BatchNorm-ReLU block."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


# ============================================================
# AutoAlignV2 -- Cross-Domain DeformCAFA sparse deformable camera-LiDAR fusion
# ============================================================


class CrossDomainDeformCAFA(nn.Module):
    """Deformable cross-attention fusion of camera features onto LiDAR queries.

    For each LiDAR BEV query cell, predicts a small set of learnable 2D sampling
    offsets (relative to a nominal projected reference point) plus per-offset
    attention weights from the query feature -- exactly the Deformable-DETR-style
    sparse sampling AutoAlignV2 substitutes for AutoAlign's costly global attention.
    Samples the camera feature map at those offsets via `grid_sample` and aggregates
    with the predicted weights.
    """

    def __init__(self, lidar_ch: int, cam_ch: int, n_points: int = 4) -> None:
        super().__init__()
        self.n_points = n_points
        self.offset_head = nn.Conv2d(lidar_ch, n_points * 2, 1)
        self.weight_head = nn.Conv2d(lidar_ch, n_points, 1)
        self.value_proj = nn.Conv2d(cam_ch, lidar_ch, 1)
        self.out_proj = nn.Conv2d(lidar_ch, lidar_ch, 1)

    def forward(
        self, lidar_query: torch.Tensor, cam_feat: torch.Tensor, ref_point: torch.Tensor
    ) -> torch.Tensor:
        b, c, h, w = lidar_query.shape
        offsets = self.offset_head(lidar_query).view(b, self.n_points, 2, h, w)
        weights = torch.softmax(self.weight_head(lidar_query), dim=1)  # (B, n_points, H, W)
        value = self.value_proj(cam_feat)

        ref = ref_point.unsqueeze(1)  # (B, 1, 2, H, W)
        sample_grid = (ref + 0.2 * offsets).permute(0, 1, 3, 4, 2)  # (B, n_points, H, W, 2)

        sampled = torch.stack(
            [
                F.grid_sample(value, sample_grid[:, p], mode="bilinear", align_corners=False)
                for p in range(self.n_points)
            ],
            dim=1,
        )  # (B, n_points, C, H, W)
        fused = (sampled * weights.unsqueeze(2)).sum(dim=1)  # (B, C, H, W)
        return self.out_proj(fused) + lidar_query


class AutoAlignV2(nn.Module):
    """AutoAlignV2: sparse deformable camera-to-LiDAR feature aggregation.

    A LiDAR-BEV backbone produces query features per BEV cell; a camera backbone
    produces image features. Cross-Domain DeformCAFA fuses the two via sparse
    learnable-offset sampling (not global attention), and a detection head
    regresses per-BEV-cell objectness + 3D box parameters from the fused feature.
    """

    def __init__(self, ch: int = 16, bev_h: int = 24, bev_w: int = 24) -> None:
        super().__init__()
        self.lidar_backbone = nn.Sequential(_cbr(ch, ch), _cbr(ch, ch))
        self.cam_backbone = nn.Sequential(_cbr(3, ch, stride=2), _cbr(ch, ch, stride=2))
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, bev_h), torch.linspace(-1, 1, bev_w), indexing="ij"
        )
        self.register_buffer(
            "ref_point", torch.stack([xx, yy], dim=0).unsqueeze(0)
        )  # (1, 2, bev_h, bev_w)
        self.fusion = CrossDomainDeformCAFA(ch, ch, n_points=4)
        self.head = nn.Conv2d(ch, 7, 1)

    def forward(self, lidar_bev: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        lidar_feat = self.lidar_backbone(lidar_bev)
        cam_feat = self.cam_backbone(image)
        ref = self.ref_point.expand(lidar_feat.shape[0], -1, -1, -1)
        fused = self.fusion(lidar_feat, cam_feat, ref)
        return self.head(fused)


def build_autoalignv2() -> nn.Module:
    """Build a small AutoAlignV2 camera+LiDAR deformable fusion detector."""
    return AutoAlignV2(ch=16, bev_h=24, bev_w=24).eval()


def example_input_autoalignv2() -> tuple[torch.Tensor, torch.Tensor]:
    """(LiDAR BEV pseudo-feature grid (1,16,24,24), camera image (1,3,96,96))."""
    return torch.randn(1, 16, 24, 24), torch.randn(1, 3, 96, 96)


# ============================================================
# AVOD -- Aggregate View Object Detection (BEV + image fused RPN)
# ============================================================


class FeaturePyramidEncoder(nn.Module):
    """Compact 2-stage encoder-decoder feature extractor (stands in for AVOD's FPN)."""

    def __init__(self, in_ch: int, base: int) -> None:
        super().__init__()
        self.down1 = _cbr(in_ch, base, stride=2)
        self.down2 = _cbr(base, base * 2, stride=2)
        self.up = nn.Sequential(
            nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.down1(x)
        d2 = self.down2(d1)
        u = self.up(d2)
        return u + d1


class AVOD(nn.Module):
    """AVOD: BEV LiDAR + front-view camera aggregate-view fused RPN + detector.

    Independent feature-pyramid encoders process a BEV LiDAR height-map and a
    front-view camera image. A shared set of 3D anchors is projected into both
    the BEV and image feature maps; per-anchor RoI-cropped features from each
    view are element-wise fused (mean), driving a shared RPN objectness + 3D
    box-proposal head. A second-stage head re-crops fused features at proposal
    locations for final box regression + classification.
    """

    def __init__(self, base: int = 12, n_anchors: int = 6, roi: int = 4) -> None:
        super().__init__()
        self.roi = roi
        self.bev_encoder = FeaturePyramidEncoder(4, base)
        self.img_encoder = FeaturePyramidEncoder(3, base)
        self.n_anchors = n_anchors
        centers = torch.rand(n_anchors, 2) * 2 - 1
        self.register_buffer("anchor_centers", centers)
        self.rpn_head = nn.Sequential(
            nn.Linear(base * roi * roi, base), nn.ReLU(inplace=True), nn.Linear(base, 7)
        )
        self.det_head = nn.Sequential(
            nn.Linear(base * roi * roi, base), nn.ReLU(inplace=True), nn.Linear(base, 3 + 7)
        )

    def _crop(self, feat: torch.Tensor, box_size: float) -> torch.Tensor:
        b = feat.shape[0]
        offs = torch.linspace(-box_size, box_size, self.roi, device=feat.device)
        gy, gx = torch.meshgrid(offs, offs, indexing="ij")
        local = torch.stack([gx, gy], dim=-1).view(1, 1, self.roi, self.roi, 2)
        centers = self.anchor_centers.view(1, self.n_anchors, 1, 1, 2)
        grid = (
            (centers + local).view(1, self.n_anchors * self.roi, self.roi, 2).expand(b, -1, -1, -1)
        )
        sampled = F.grid_sample(feat, grid, mode="bilinear", align_corners=False)
        c = feat.shape[1]
        return sampled.view(b, c, self.n_anchors, self.roi, self.roi).permute(0, 2, 1, 3, 4)

    def forward(
        self, bev_map: torch.Tensor, image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bev_feat = self.bev_encoder(bev_map)
        img_feat = self.img_encoder(image)

        bev_roi = self._crop(bev_feat, box_size=0.2)  # (B, n_anchors, C, roi, roi)
        img_roi = self._crop(img_feat, box_size=0.2)
        fused = 0.5 * (bev_roi + img_roi)
        b, na = fused.shape[0], fused.shape[1]
        flat = fused.reshape(b, na, -1)

        proposals = self.rpn_head(flat)  # (B, n_anchors, 7): objectness + 6 box params

        det_bev_roi = self._crop(bev_feat, box_size=0.1)
        det_img_roi = self._crop(img_feat, box_size=0.1)
        det_fused = (0.5 * (det_bev_roi + det_img_roi)).reshape(b, na, -1)
        detections = self.det_head(det_fused)  # (B, n_anchors, 3 classes + 7 box)
        return proposals, detections


def build_avod() -> nn.Module:
    """Build a small AVOD aggregate-view (BEV LiDAR + image) detector."""
    return AVOD(base=12, n_anchors=6, roi=4).eval()


def example_input_avod() -> tuple[torch.Tensor, torch.Tensor]:
    """(BEV LiDAR height-map (1,4,64,64), front-view camera image (1,3,64,64))."""
    return torch.randn(1, 4, 64, 64), torch.randn(1, 3, 64, 64)


# ============================================================
# BEV-LaneDet -- Virtual Camera + Spatial Transformation Pyramid + Key-Points head
# ============================================================


class VirtualCameraWarp(nn.Module):
    """Warps a raw camera image into a fixed canonical (virtual-camera) frame.

    Standing in for BEV-LaneDet's per-vehicle intrinsic/extrinsic normalization: a
    fixed, non-learned resampling grid re-projects the input image as if captured
    by a single canonical camera pose, decoupling downstream layers from the
    mounting camera's specific calibration.
    """

    def __init__(self, out_h: int, out_w: int) -> None:
        super().__init__()
        yy, xx = torch.meshgrid(
            torch.linspace(-0.8, 0.8, out_h), torch.linspace(-0.9, 0.9, out_w), indexing="ij"
        )
        self.register_buffer("grid", torch.stack([xx, yy], dim=-1).unsqueeze(0))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        grid = self.grid.expand(image.shape[0], -1, -1, -1)
        return F.grid_sample(image, grid, mode="bilinear", align_corners=False)


class SpatialTransformationPyramidLevel(nn.Module):
    """One pyramid level: dense-MLP projection of a front-view feature map to BEV."""

    def __init__(self, ch: int, fv_hw: int, bev_hw: int) -> None:
        super().__init__()
        self.fv_hw, self.bev_hw = fv_hw, bev_hw
        self.project = nn.Linear(fv_hw * fv_hw, bev_hw * bev_hw)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        b, c, h, w = feat.shape
        flat = feat.reshape(b, c, h * w)
        bev_flat = self.project(flat)
        return bev_flat.view(b, c, self.bev_hw, self.bev_hw)


class BEVLaneDet(nn.Module):
    """BEV-LaneDet: virtual-camera warp + multi-scale FV->BEV projection pyramid.

    A fixed virtual-camera warp normalizes the input image, a shared CNN backbone
    extracts two front-view feature scales, a Spatial Transformation Pyramid
    projects each scale to the BEV plane with its own dense-MLP projection and
    sums them, and a Key-Points Representation head predicts per-BEV-cell lane
    existence, a local (dx, dz) offset, and an embedding vector for instance
    clustering.
    """

    def __init__(self, base: int = 12, bev_hw: int = 16, embed_dim: int = 4) -> None:
        super().__init__()
        self.virtual_cam = VirtualCameraWarp(64, 64)
        self.stem = _cbr(3, base, stride=1)
        self.stage1 = _cbr(base, base, stride=2)  # 32x32
        self.stage2 = _cbr(base, base, stride=2)  # 16x16
        self.pyramid1 = SpatialTransformationPyramidLevel(base, 32, bev_hw)
        self.pyramid2 = SpatialTransformationPyramidLevel(base, 16, bev_hw)
        self.bev_refine = _cbr(base, base)
        self.head = nn.Conv2d(base, 1 + 2 + embed_dim, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        warped = self.virtual_cam(image)
        s0 = self.stem(warped)
        s1 = self.stage1(s0)
        s2 = self.stage2(s1)
        bev = self.pyramid1(s1) + self.pyramid2(s2)
        bev = self.bev_refine(bev)
        out = self.head(bev)
        existence = torch.sigmoid(out[:, :1])
        offset = out[:, 1:3]
        embedding = out[:, 3:]
        return torch.cat([existence, offset, embedding], dim=1)


def build_bev_lanedet() -> nn.Module:
    """Build a small BEV-LaneDet monocular 3D lane detector."""
    return BEVLaneDet(base=12, bev_hw=16, embed_dim=4).eval()


def example_input_bev_lanedet() -> torch.Tensor:
    """Front camera RGB image (1, 3, 64, 64)."""
    return torch.randn(1, 3, 64, 64)


# ============================================================
# BEVContrast -- BEV-cell-level contrastive self-supervision of a LiDAR backbone
# ============================================================


class PointBackbone(nn.Module):
    """Compact per-point feature extractor (stands in for a sparse-conv 3D backbone)."""

    def __init__(self, in_dim: int, feat_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, feat_dim), nn.ReLU(inplace=True), nn.Linear(feat_dim, feat_dim)
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return self.mlp(points)


class BEVCellPooling(nn.Module):
    """Scatter-mean-pools per-point features into a 2D BEV grid of cells.

    Uses `scatter_mean`-style pooling built from `index_add_` (dense small grid,
    kept traceable): each point's (x, y) coordinate is discretized to a BEV cell
    index, per-point features are summed per cell, and divided by the per-cell
    point count -- exactly BEVContrast's cell-level (not point- or segment-level)
    pooled representation.
    """

    def __init__(self, grid_size: int, extent: float) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.extent = extent

    def forward(self, points_xy: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        b, n, c = feats.shape
        g = self.grid_size
        idx_xy = ((points_xy / self.extent).clamp(-1, 1) * 0.5 + 0.5) * (g - 1)
        idx_xy = idx_xy.round().long().clamp(0, g - 1)
        flat_idx = idx_xy[..., 1] * g + idx_xy[..., 0]  # (B, N)

        cell_sum = feats.new_zeros(b, g * g, c)
        cell_count = feats.new_zeros(b, g * g, 1)
        ones = feats.new_ones(b, n, 1)
        cell_sum.scatter_add_(1, flat_idx.unsqueeze(-1).expand(-1, -1, c), feats)
        cell_count.scatter_add_(1, flat_idx.unsqueeze(-1), ones)
        cell_mean = cell_sum / cell_count.clamp(min=1.0)
        return cell_mean.transpose(1, 2).view(b, c, g, g)


class BEVContrast(nn.Module):
    """BEVContrast: per-point backbone + BEV-cell pooling for contrastive pretraining.

    Two (paired) LiDAR point clouds are each encoded per-point, scatter-mean-pooled
    into a BEV grid of cell embeddings, and L2-normalized -- the two resulting BEV
    feature grids are what an external InfoNCE loss contrasts cell-to-cell between
    corresponding scans (ego-motion cell correspondence supplied externally).
    """

    def __init__(self, feat_dim: int = 16, grid_size: int = 12, extent: float = 20.0) -> None:
        super().__init__()
        self.backbone = PointBackbone(in_dim=4, feat_dim=feat_dim)
        self.pooling = BEVCellPooling(grid_size=grid_size, extent=extent)

    def _encode(self, points: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(points)
        bev = self.pooling(points[..., :2], feats)
        return F.normalize(bev, dim=1)

    def forward(
        self, points_a: torch.Tensor, points_b: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._encode(points_a), self._encode(points_b)


def build_bevcontrast() -> nn.Module:
    """Build a small BEVContrast point-cloud contrastive-pretraining module."""
    return BEVContrast(feat_dim=16, grid_size=12, extent=20.0).eval()


def example_input_bevcontrast() -> tuple[torch.Tensor, torch.Tensor]:
    """Two paired LiDAR point clouds, each (1, N=200, 4) as (x, y, z, intensity)."""
    pts_a = torch.randn(1, 200, 4) * 5.0
    pts_b = torch.randn(1, 200, 4) * 5.0
    return pts_a, pts_b


# ============================================================
# BEVDet -- Lift-Splat-Shoot (LSS) camera-only BEV 3D detection
# ============================================================


class LiftSplatShoot(nn.Module):
    """Lift-Splat-Shoot view transformer: image features -> unified BEV feature map.

    For each camera-feature pixel, predicts a categorical distribution over
    ``n_depth`` discrete depth bins and outer-products it with the pixel's channel
    feature ("lift" into a 3D frustum of (depth x channel) features per pixel).
    Each frustum point is projected with a (fixed, per-example) camera-to-BEV
    homography into BEV-plane coordinates and scatter-summed ("splat") into the
    unified BEV grid -- the defining LSS mechanism.
    """

    def __init__(self, in_ch: int, out_ch: int, n_depth: int, bev_size: int) -> None:
        super().__init__()
        self.n_depth = n_depth
        self.bev_size = bev_size
        self.depth_head = nn.Conv2d(in_ch, n_depth, 1)
        self.feat_proj = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, img_feat: torch.Tensor, bev_grid: torch.Tensor) -> torch.Tensor:
        b, c_in, h, w = img_feat.shape
        depth_prob = torch.softmax(self.depth_head(img_feat), dim=1)  # (B, D, H, W)
        feat = self.feat_proj(img_feat)  # (B, C, H, W)
        c = feat.shape[1]

        # lift: outer product of per-pixel feature and depth distribution
        lifted = depth_prob.unsqueeze(2) * feat.unsqueeze(1)  # (B, D, C, H, W)
        lifted = lifted.permute(0, 1, 3, 4, 2).reshape(b, self.n_depth * h * w, c)

        # splat: scatter-sum each frustum point into its BEV cell via the supplied
        # (fixed, precomputed) per-pixel-per-depth-bin BEV cell index
        g = self.bev_size
        flat_idx = bev_grid.view(b, -1).clamp(0, g * g - 1)
        bev_sum = feat.new_zeros(b, g * g, c)
        bev_sum.scatter_add_(1, flat_idx.unsqueeze(-1).expand(-1, -1, c), lifted)
        return bev_sum.transpose(1, 2).view(b, c, g, g)


class BEVDet(nn.Module):
    """BEVDet: shared image backbone + LSS view transformer + BEV detection head.

    A shared 2D CNN backbone extracts per-camera image features (multi-view
    cameras stacked into the batch dimension); LSS lifts and splats them into a
    single unified BEV feature map, a BEV-space CNN backbone refines it, and a
    detection head regresses per-BEV-cell objectness + 3D box parameters.
    """

    def __init__(self, ch: int = 12, n_depth: int = 8, bev_size: int = 16) -> None:
        super().__init__()
        self.img_backbone = nn.Sequential(_cbr(3, ch, stride=2), _cbr(ch, ch, stride=2))
        self.view_transform = LiftSplatShoot(ch, ch, n_depth=n_depth, bev_size=bev_size)
        self.bev_backbone = nn.Sequential(_cbr(ch, ch), _cbr(ch, ch))
        self.head = nn.Conv2d(ch, 7, 1)
        self.bev_size = bev_size
        self.n_depth = n_depth

    def forward(self, multi_view_images: torch.Tensor, bev_cell_idx: torch.Tensor) -> torch.Tensor:
        n_cams, _, h_img, w_img = multi_view_images.shape
        img_feat = self.img_backbone(multi_view_images)  # (n_cams, ch, h, w)
        bev_per_cam = self.view_transform(img_feat, bev_cell_idx)  # (n_cams, ch, bev, bev)
        bev = bev_per_cam.sum(dim=0, keepdim=True)  # merge multi-camera BEV into one ego-frame grid
        bev = self.bev_backbone(bev)
        return self.head(bev)


def build_bevdet() -> nn.Module:
    """Build a small BEVDet camera-only LSS BEV 3D detector."""
    return BEVDet(ch=12, n_depth=8, bev_size=16).eval()


def example_input_bevdet() -> tuple[torch.Tensor, torch.Tensor]:
    """(multi-view camera images (n_cams=3, 3, 32, 32), BEV cell index per frustum point)."""
    n_cams, ch, n_depth, h, w, bev = 3, 12, 8, 8, 8, 16
    images = torch.randn(n_cams, 3, 32, 32)
    bev_cell_idx = torch.randint(0, bev * bev, (n_cams, n_depth, h, w))
    return images, bev_cell_idx


# ============================================================
# BEVDet4D -- BEVDet + temporal BEV feature alignment and fusion
# ============================================================


class TemporalBEVAlign(nn.Module):
    """Warps the previous frame's BEV feature into the current frame's ego frame.

    A fixed (per-example) rigid ego-motion transform expressed as a sampling grid
    is applied via `grid_sample` -- BEVDet4D's "align previous-frame BEV feature"
    step -- before concatenating with the current-frame BEV feature.
    """

    def forward(self, prev_bev: torch.Tensor, ego_motion_grid: torch.Tensor) -> torch.Tensor:
        return F.grid_sample(prev_bev, ego_motion_grid, mode="bilinear", align_corners=False)


class BEVDet4D(nn.Module):
    """BEVDet4D: BEVDet lifted from spatial-only 3D to spatial-temporal 4D.

    Reuses BEVDet's image backbone + LSS view transformer to compute the
    current-frame BEV feature; the previous frame's BEV feature is warped into
    the current ego frame via `TemporalBEVAlign` and concatenated channel-wise
    with the current BEV feature before the BEV backbone + detection head, giving
    the head access to temporal cues (used downstream for velocity prediction).
    """

    def __init__(self, ch: int = 12, n_depth: int = 8, bev_size: int = 16) -> None:
        super().__init__()
        self.img_backbone = nn.Sequential(_cbr(3, ch, stride=2), _cbr(ch, ch, stride=2))
        self.view_transform = LiftSplatShoot(ch, ch, n_depth=n_depth, bev_size=bev_size)
        self.temporal_align = TemporalBEVAlign()
        self.fuse = nn.Conv2d(ch * 2, ch, 1)
        self.bev_backbone = nn.Sequential(_cbr(ch, ch), _cbr(ch, ch))
        self.head = nn.Conv2d(ch, 9, 1)  # objectness + 6 box params + (vx, vy) velocity

    def forward(
        self,
        multi_view_images: torch.Tensor,
        bev_cell_idx: torch.Tensor,
        prev_bev: torch.Tensor,
        ego_motion_grid: torch.Tensor,
    ) -> torch.Tensor:
        img_feat = self.img_backbone(multi_view_images)
        bev_per_cam = self.view_transform(img_feat, bev_cell_idx)
        cur_bev = bev_per_cam.sum(dim=0, keepdim=True)

        aligned_prev = self.temporal_align(prev_bev, ego_motion_grid)
        fused = self.fuse(torch.cat([cur_bev, aligned_prev], dim=1))
        fused = self.bev_backbone(fused)
        return self.head(fused)


def build_bevdet4d() -> nn.Module:
    """Build a small BEVDet4D temporal camera-only BEV 3D detector."""
    return BEVDet4D(ch=12, n_depth=8, bev_size=16).eval()


def example_input_bevdet4d() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(multi-view images, BEV cell index, previous-frame BEV feature, ego-motion grid)."""
    n_cams, ch, n_depth, h, w, bev = 3, 12, 8, 8, 8, 16
    images = torch.randn(n_cams, 3, 32, 32)
    bev_cell_idx = torch.randint(0, bev * bev, (n_cams, n_depth, h, w))
    prev_bev = torch.randn(1, ch, bev, bev)
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, bev), torch.linspace(-1, 1, bev), indexing="ij")
    ego_motion_grid = torch.stack([xx, yy], dim=-1).unsqueeze(0) * 0.9
    return images, bev_cell_idx, prev_bev, ego_motion_grid


# ============================================================
# Registry
# ============================================================

MENAGERIE_ENTRIES = [
    ("AutoAlignV2", "build_autoalignv2", "example_input_autoalignv2", "2022", "VIS"),
    ("AVOD", "build_avod", "example_input_avod", "2018", "VIS"),
    ("BEV-LaneDet", "build_bev_lanedet", "example_input_bev_lanedet", "2023", "VIS"),
    ("BEVContrast", "build_bevcontrast", "example_input_bevcontrast", "2024", "VIS"),
    ("BEVDet", "build_bevdet", "example_input_bevdet", "2021", "VIS"),
    ("BEVDet4D", "build_bevdet4d", "example_input_bevdet4d", "2022", "VIS"),
]
