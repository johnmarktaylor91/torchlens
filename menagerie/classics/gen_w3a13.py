"""Menagerie batch w3a13: LiDAR/camera 3D perception architectures for autonomous driving.

Sources checked (reference only; no cloning, no pip installs):
  - FB-OCC: Li et al., NVIDIA, 2023. Paper https://arxiv.org/abs/2307.01492, official
    source https://github.com/NVlabs/FB-BEV (repo hosts both FB-BEV and FB-OCC; CVPR
    2023 nuScenes Occupancy Challenge winner). The distinctive mechanism is the
    Forward-Backward view transformation: a *forward* projection step lifts each
    per-camera image feature into a 3D voxel grid via a predicted per-pixel depth
    distribution (LSS-style outer product + scatter-splat, like BEVDet/BEVFormer's
    forward view transformers), producing a coarse but complete voxel volume; then a
    *backward* projection step re-samples that voxel volume back into the image plane
    with known camera geometry (`grid_sample`) to pull in image-aligned features via a
    small deformable/BEVFormer-style query-attention refinement, correcting the holes
    and geometric distortion the forward pass alone leaves behind. The two
    representations (voxel + re-sampled-image) are fused before a lightweight 3D conv
    head predicts per-voxel semantic occupancy.
  - Frustum PointNets: Qi et al., CVPR 2018. Paper https://arxiv.org/abs/1711.08488,
    official source https://github.com/charlesq34/frustum-pointnets (TensorFlow; PyTorch
    port at https://github.com/simon3dv/frustum_pointnets_pytorch). The defining idea is
    a three-stage frustum pipeline operating directly on raw point clouds: (1) a 2D
    detector's image bounding box is *extruded* into a 3D viewing frustum that crops the
    LiDAR point cloud down to only the points inside that frustum (search-space
    reduction using mature 2D detectors); (2) a PointNet instance-segmentation module
    scores each point in the frustum as foreground/background and the foreground points
    are re-centered (translated to their centroid) -- "T-Net" light-weight regression
    predicting a residual center shift for translation invariance; (3) a second PointNet
    regresses the amodal 3D bounding box (center, size-cluster residual, heading-bin +
    residual) directly from the centered, segmented point subset. All three stages are
    plain PointNet-style per-point MLP + symmetric max-pool blocks.
  - FUTR3D: Chen et al., Tsinghua MARS Lab, 2022. Paper https://arxiv.org/abs/2203.10642,
    official source https://github.com/Tsinghua-MARS-Lab/futr3d (CVPRW 2023). The
    defining mechanism is the Modality-Agnostic Feature Sampler (MAFS): a shared set of
    learned 3D object queries, each carrying a predicted 3D reference point, is used to
    sample features from *any* combination of per-modality feature maps (camera images,
    LiDAR/radar BEV features) by projecting the reference point into each modality's own
    coordinate frame (camera: perspective projection + bilinear `grid_sample` on the
    image feature map; LiDAR/radar: direct bilinear sample on the BEV feature grid), then
    concatenating (or summing) the per-modality sampled features into one query feature
    -- unifying arbitrary sensor sets behind one query-based sampling API instead of
    per-modality fusion heuristics. A transformer decoder iteratively refines the query
    reference points and regresses 3D boxes via set-to-set (DETR-style) prediction.
  - FUTR3D-Radar: same architecture and repo as FUTR3D (cand_00278); this candidate
    exercises the same MAFS with the 4D-imaging-radar BEV branch included alongside
    camera + LiDAR in the per-modality feature-map list, so it is built here by
    instantiating `FUTR3D` with `use_radar=True` (an extra BEV-style branch reusing the
    identical sampler) rather than a separate class -- the modality-agnostic sampler is
    literally the same code path regardless of which BEV modality (LiDAR or radar) it is
    fed.
  - GaussianFormer: Huang et al., ECCV 2024. Paper https://arxiv.org/abs/2405.17429,
    official source https://github.com/huang-yh/GaussianFormer. The defining idea
    replaces dense voxel occupancy with an object-centric set of sparse 3D semantic
    Gaussians: a fixed number of learnable Gaussian queries (each parameterized by a 3D
    mean/position, a diagonal covariance/scale, and a semantic-class logit vector) is
    iteratively refined by cross-attending each Gaussian's projected-image reference
    point into the multi-camera image features (deformable-attention-style sampling, as
    in BEVFormer/DETR3D) and self-attending among Gaussians, updating position,
    covariance, and semantics each layer. A final Gaussian-to-voxel splat step
    rasterizes only the *neighboring* Gaussians (Gaussian kernel weight by distance) into
    each queried voxel cell to produce the dense semantic occupancy grid, avoiding ever
    materializing dense 3D features during the network's forward pass.
  - GUPNet: Lu et al., ICCV 2021. Paper https://arxiv.org/abs/2107.13774, official source
    https://github.com/SuperMHP/GUPNet. Monocular 3D object detection where the
    distinctive mechanism is the Uncertainty Geometry Projection: depth is not
    regressed directly but *derived geometrically* from a predicted 2D bounding-box
    height `h_2d` and a predicted 3D object height `h_3d` via the pinhole projection
    identity `depth = f * h_3d / h_2d` (camera focal length `f`); because `h_3d` and
    `h_2d` are each noisy network outputs, GUPNet additionally predicts a per-instance
    log-variance for `h_3d`, analytically propagates it through the projection formula
    to obtain a *geometry-derived* depth uncertainty, and reports both depth and its
    uncertainty (used at inference to weight/calibrate confident vs. unreliable depth
    predictions) -- turning geometric projection itself into an uncertainty-aware
    differentiable layer instead of a black-box depth regressor.

All models use tiny random-init dims (architecture catalog, not a trained-weights zoo).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def _cbr(in_ch: int, out_ch: int, k: int = 3, stride: int = 1, padding: int = 1) -> nn.Sequential:
    """Conv-BatchNorm-ReLU block, the recurring image/voxel-backbone primitive."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, stride=stride, padding=padding),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


# ============================================================
# FB-OCC -- forward-backward view transformation for 3D occupancy
# ============================================================


class ForwardViewTransform(nn.Module):
    """LSS-style forward lift: per-pixel depth distribution outer-producted with

    the image feature, then scatter-summed into a 3D voxel grid via a supplied
    flattened voxel index per (camera, depth-bin, pixel) location.
    """

    def __init__(self, in_ch: int, out_ch: int, n_depth: int) -> None:
        super().__init__()
        self.depth_head = nn.Conv2d(in_ch, n_depth, 1)
        self.feat_head = nn.Conv2d(in_ch, out_ch, 1)
        self.n_depth = n_depth
        self.out_ch = out_ch

    def forward(
        self, img_feat: torch.Tensor, voxel_idx: torch.Tensor, n_voxels: int
    ) -> torch.Tensor:
        n_cams, _, h, w = img_feat.shape
        depth_dist = self.depth_head(img_feat).softmax(dim=1)  # (n_cams, n_depth, h, w)
        feat = self.feat_head(img_feat)  # (n_cams, out_ch, h, w)
        # outer product: (n_cams, n_depth, out_ch, h, w)
        frustum = depth_dist.unsqueeze(2) * feat.unsqueeze(1)
        frustum = frustum.permute(0, 1, 3, 4, 2).reshape(
            -1, self.out_ch
        )  # (n_cams*n_depth*h*w, out_ch)
        flat_idx = voxel_idx.reshape(-1)
        voxel = torch.zeros(n_voxels, self.out_ch, device=frustum.device, dtype=frustum.dtype)
        voxel.index_add_(0, flat_idx, frustum)
        return voxel


class BackwardViewTransform(nn.Module):
    """Re-samples the forward voxel volume back into each camera's image plane

    via a supplied sampling grid (`grid_sample`), then a small conv refines the
    voxel-BEV feature using the re-projected image-aligned cues -- correcting
    holes/distortion left by the forward-only projection.
    """

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.refine = nn.Conv2d(ch * 2, ch, 3, padding=1)

    def forward(
        self, voxel_bev: torch.Tensor, img_feat_bev: torch.Tensor, grid: torch.Tensor
    ) -> torch.Tensor:
        resampled = F.grid_sample(img_feat_bev, grid, mode="bilinear", align_corners=False)
        fused = torch.cat([voxel_bev, resampled], dim=1)
        return self.refine(fused)


class FBOcc(nn.Module):
    """FB-OCC: forward-projection voxel lift + backward-projection image resampling,

    fused for dense per-voxel semantic occupancy prediction.
    """

    def __init__(
        self, ch: int = 8, n_depth: int = 6, bev_size: int = 8, n_classes: int = 5
    ) -> None:
        super().__init__()
        self.img_backbone = nn.Sequential(_cbr(3, ch, stride=2), _cbr(ch, ch, stride=2))
        self.forward_transform = ForwardViewTransform(ch, ch, n_depth)
        self.bev_size = bev_size
        self.ch = ch
        self.img_to_bev = nn.Conv2d(ch, ch, 1)
        self.backward_transform = BackwardViewTransform(ch)
        self.occ_head = nn.Conv2d(ch, n_classes, 1)

    def forward(
        self,
        multi_view_images: torch.Tensor,
        voxel_bev_idx: torch.Tensor,
        resample_grid: torch.Tensor,
    ) -> torch.Tensor:
        n_cams = multi_view_images.shape[0]
        img_feat = self.img_backbone(multi_view_images)
        n_voxels = self.bev_size * self.bev_size
        voxel_flat = self.forward_transform(img_feat, voxel_bev_idx, n_voxels)
        voxel_bev = voxel_flat.t().view(1, self.ch, self.bev_size, self.bev_size)

        img_feat_bev = self.img_to_bev(img_feat).mean(dim=0, keepdim=True)
        img_feat_bev = F.interpolate(
            img_feat_bev, size=(self.bev_size, self.bev_size), mode="bilinear"
        )

        fused = self.backward_transform(voxel_bev, img_feat_bev, resample_grid)
        return self.occ_head(fused)


def build_fb_occ() -> nn.Module:
    """Build a small FB-OCC forward-backward voxel/BEV occupancy network."""
    return FBOcc(ch=8, n_depth=6, bev_size=8, n_classes=5).eval()


def example_input_fb_occ() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(multi-view camera images, per-(cam,depth,pixel) voxel index, backward resample grid)."""
    n_cams, n_depth, h, w, bev = 3, 6, 8, 8, 8
    images = torch.randn(n_cams, 3, 32, 32)
    voxel_bev_idx = torch.randint(0, bev * bev, (n_cams, n_depth, h, w))
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, bev), torch.linspace(-1, 1, bev), indexing="ij")
    resample_grid = torch.stack([xx, yy], dim=-1).unsqueeze(0) * 0.9
    return images, voxel_bev_idx, resample_grid


# ============================================================
# Frustum PointNets -- 2D-proposal-guided frustum point-cloud detection
# ============================================================


class _PointNetBlock(nn.Module):
    """Per-point shared MLP followed by a symmetric max-pool -- the recurring

    PointNet primitive used by all three Frustum PointNets sub-stages.
    """

    def __init__(self, in_ch: int, hidden: int, out_ch: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(in_ch, hidden, 1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, out_ch, 1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """points: (B, C, N) -> (per_point_feat (B, out_ch, N), global_feat (B, out_ch))."""
        per_point = self.mlp(points)
        global_feat = per_point.max(dim=2).values
        return per_point, global_feat


class FrustumPointNets(nn.Module):
    """Three-stage frustum pipeline: instance segmentation on the frustum-cropped

    point cloud, a light T-Net center-shift regression that re-centers the
    segmented foreground points, then amodal 3D box regression (center, size
    residual, heading bin + residual) on the centered points.
    """

    def __init__(self, n_size_clusters: int = 3, n_heading_bins: int = 4) -> None:
        super().__init__()
        self.seg_net = _PointNetBlock(3, 32, 32)
        self.seg_head = nn.Conv1d(32, 2, 1)  # per-point foreground/background logits

        self.tnet = _PointNetBlock(3, 32, 32)
        self.tnet_head = nn.Linear(32, 3)  # predicted center-shift residual

        self.box_net = _PointNetBlock(3, 32, 64)
        self.box_head = nn.Linear(64, 3 + n_size_clusters * 4 + n_heading_bins * 2)
        self.n_size_clusters = n_size_clusters
        self.n_heading_bins = n_heading_bins

    def forward(
        self, frustum_points: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """frustum_points: (B, 3, N) points already cropped to the 2D-box frustum."""
        per_point_feat, _ = self.seg_net(frustum_points)
        seg_logits = self.seg_head(per_point_feat)  # (B, 2, N)

        fg_weight = seg_logits.softmax(dim=1)[:, 1:2, :]  # soft foreground mask, differentiable
        weighted_points = frustum_points * fg_weight

        _, tnet_feat = self.tnet(weighted_points)
        center_shift = self.tnet_head(tnet_feat)  # (B, 3)
        centered_points = weighted_points - center_shift.unsqueeze(-1)

        _, box_feat = self.box_net(centered_points)
        box_params = self.box_head(box_feat)
        return seg_logits, center_shift, box_params


def build_frustum_pointnets() -> nn.Module:
    """Build a small Frustum PointNets 3-stage detector."""
    return FrustumPointNets(n_size_clusters=3, n_heading_bins=4).eval()


def example_input_frustum_pointnets() -> torch.Tensor:
    """(batch of frustum-cropped point clouds, shape (B, 3, N))."""
    return torch.randn(2, 3, 256)


# ============================================================
# FUTR3D / FUTR3D-Radar -- modality-agnostic query-based sensor fusion
# ============================================================


class ModalityAgnosticFeatureSampler(nn.Module):
    """Samples each modality's feature map at a shared query's projected reference

    point (camera: perspective-style `grid_sample` on the image feature; BEV
    modalities such as LiDAR/radar: direct `grid_sample` on the BEV feature grid)
    and concatenates the per-modality sampled features into one query feature.
    """

    def forward(
        self,
        cam_feats: torch.Tensor,
        cam_ref_grid: torch.Tensor,
        bev_feats_list: list[torch.Tensor],
        bev_ref_grid: torch.Tensor,
    ) -> torch.Tensor:
        n_cams = cam_feats.shape[0]
        cam_ref_grid = cam_ref_grid.expand(
            n_cams, -1, -1, -1
        )  # broadcast the shared query grid per camera
        cam_sampled = F.grid_sample(cam_feats, cam_ref_grid, mode="bilinear", align_corners=False)
        cam_sampled = cam_sampled.mean(dim=0, keepdim=True)  # (1, C, 1, n_query) fused across cams
        sampled = [cam_sampled]
        for bev_feat in bev_feats_list:
            sampled.append(
                F.grid_sample(bev_feat, bev_ref_grid, mode="bilinear", align_corners=False)
            )
        fused = torch.cat(sampled, dim=1)  # (1, sum_C, 1, n_query)
        return fused.squeeze(0).squeeze(1).t()  # (n_query, sum_C)


class FUTR3D(nn.Module):
    """FUTR3D: shared 3D object queries sampled across an arbitrary modality set

    via the Modality-Agnostic Feature Sampler (MAFS), refined by a transformer
    decoder and regressed to 3D boxes (DETR-style set prediction). `use_radar`
    adds a second BEV-style branch (radar) alongside LiDAR, reusing the identical
    sampler code path -- this is the mechanism FUTR3D-Radar exercises.
    """

    def __init__(self, ch: int = 8, n_query: int = 12, use_radar: bool = False) -> None:
        super().__init__()
        self.img_backbone = nn.Sequential(_cbr(3, ch, stride=2), _cbr(ch, ch, stride=2))
        self.lidar_backbone = nn.Sequential(_cbr(ch, ch), _cbr(ch, ch))
        self.use_radar = use_radar
        if use_radar:
            self.radar_backbone = nn.Sequential(_cbr(ch, ch), _cbr(ch, ch))

        self.query_embed = nn.Parameter(torch.randn(n_query, ch))
        self.ref_point_head = nn.Linear(ch, 3)  # predicts (x, y, z) reference point per query

        self.sampler = ModalityAgnosticFeatureSampler()
        n_bev_modalities = 2 if use_radar else 1
        fused_ch = ch + n_bev_modalities * ch
        self.query_proj = nn.Linear(fused_ch, ch)
        self.decoder_layer = nn.TransformerEncoderLayer(
            d_model=ch, nhead=2, dim_feedforward=ch * 2, batch_first=True
        )
        self.box_head = nn.Linear(ch, 3 + 3 + 1)  # center + size + heading
        self.n_query = n_query

    def forward(
        self,
        multi_view_images: torch.Tensor,
        lidar_bev: torch.Tensor,
        radar_bev: torch.Tensor | None = None,
    ) -> torch.Tensor:
        img_feat = self.img_backbone(multi_view_images)
        lidar_feat = self.lidar_backbone(lidar_bev)

        ref_points = self.ref_point_head(self.query_embed).tanh()  # (n_query, 3) in [-1, 1]
        cam_grid = ref_points[:, :2].unsqueeze(0).unsqueeze(0)  # (1, 1, n_query, 2)
        bev_grid = ref_points[:, :2].unsqueeze(0).unsqueeze(0)

        bev_feats = [lidar_feat]
        if self.use_radar:
            assert radar_bev is not None
            bev_feats.append(self.radar_backbone(radar_bev))

        sampled = self.sampler(img_feat, cam_grid, bev_feats, bev_grid)  # (n_query, fused_ch)
        query_feat = self.query_proj(sampled).unsqueeze(0)  # (1, n_query, ch)
        decoded = self.decoder_layer(query_feat)
        return self.box_head(decoded.squeeze(0))


def build_futr3d() -> nn.Module:
    """Build a small camera+LiDAR FUTR3D detector."""
    return FUTR3D(ch=8, n_query=12, use_radar=False).eval()


def example_input_futr3d() -> tuple[torch.Tensor, torch.Tensor]:
    """(multi-view camera images, LiDAR BEV feature map)."""
    return torch.randn(3, 3, 32, 32), torch.randn(1, 8, 16, 16)


def build_futr3d_radar() -> nn.Module:
    """Build a small camera+LiDAR+radar FUTR3D detector (radar branch enabled)."""
    return FUTR3D(ch=8, n_query=12, use_radar=True).eval()


def example_input_futr3d_radar() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(multi-view camera images, LiDAR BEV feature map, radar BEV feature map)."""
    return torch.randn(3, 3, 32, 32), torch.randn(1, 8, 16, 16), torch.randn(1, 8, 16, 16)


# ============================================================
# GaussianFormer -- sparse 3D semantic Gaussians for occupancy prediction
# ============================================================


class GaussianCrossAttention(nn.Module):
    """Each Gaussian query's projected reference point samples multi-camera image

    features (deformable-attention-style `grid_sample`), and the sampled feature
    updates the Gaussian's position, covariance (log-scale), and semantic logits.
    """

    def __init__(self, ch: int, n_classes: int) -> None:
        super().__init__()
        self.ref_point_head = nn.Linear(ch, 3)
        self.update = nn.Sequential(nn.Linear(ch + ch, ch), nn.ReLU(inplace=True))
        self.pos_head = nn.Linear(ch, 3)
        self.scale_head = nn.Linear(ch, 3)
        self.sem_head = nn.Linear(ch, n_classes)

    def forward(
        self, gaussian_feat: torch.Tensor, cam_feats: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        n_cams = cam_feats.shape[0]
        ref_points = self.ref_point_head(gaussian_feat).tanh()  # (n_gauss, 3)
        grid = ref_points[:, :2].unsqueeze(0).unsqueeze(0)  # (1, 1, n_gauss, 2)
        grid = grid.expand(n_cams, -1, -1, -1)  # broadcast the shared query grid per camera
        sampled = F.grid_sample(
            cam_feats, grid, mode="bilinear", align_corners=False
        )  # (n_cams, ch, 1, n_gauss)
        sampled = sampled.mean(dim=0).squeeze(1).t()  # (n_gauss, ch), fused across cameras

        gaussian_feat = self.update(torch.cat([gaussian_feat, sampled], dim=-1))
        position = self.pos_head(gaussian_feat)
        scale = self.scale_head(gaussian_feat)
        semantics = self.sem_head(gaussian_feat)
        return gaussian_feat, position, scale, semantics


class GaussianFormer(nn.Module):
    """GaussianFormer: a fixed set of learnable sparse 3D-Gaussian queries is

    iteratively refined by cross-attending multi-camera image features and
    self-attending among Gaussians, then a Gaussian-to-voxel splat (each queried
    voxel weighted by its distance to every Gaussian's mean, via a Gaussian
    kernel over the predicted scale) rasterizes the sparse Gaussians into dense
    per-voxel semantic occupancy -- without ever forming a dense 3D feature grid.
    """

    def __init__(
        self, ch: int = 8, n_gaussians: int = 16, n_classes: int = 5, voxel_res: int = 6
    ) -> None:
        super().__init__()
        self.img_backbone = nn.Sequential(_cbr(3, ch, stride=2), _cbr(ch, ch, stride=2))
        self.gaussian_embed = nn.Parameter(torch.randn(n_gaussians, ch))
        self.cross_attn = GaussianCrossAttention(ch, n_classes)
        self.self_attn = nn.TransformerEncoderLayer(
            d_model=ch, nhead=2, dim_feedforward=ch * 2, batch_first=True
        )
        self.n_classes = n_classes
        self.voxel_res = voxel_res

    def forward(self, multi_view_images: torch.Tensor, voxel_centers: torch.Tensor) -> torch.Tensor:
        """voxel_centers: (n_voxels, 3) query coordinates in the same [-1, 1] frame."""
        cam_feats = self.img_backbone(multi_view_images)
        gaussian_feat = self.gaussian_embed

        gaussian_feat, position, log_scale, semantics = self.cross_attn(gaussian_feat, cam_feats)
        gaussian_feat = self.self_attn(gaussian_feat.unsqueeze(0)).squeeze(0)

        # Gaussian-to-voxel splat: distance-weighted (Gaussian kernel) aggregation of
        # semantic logits from every Gaussian into each voxel center.
        scale = (
            log_scale.exp().mean(dim=-1, keepdim=True) + 1e-3
        )  # (n_gauss, 1), isotropic for compactness
        diff = voxel_centers.unsqueeze(1) - position.unsqueeze(0)  # (n_voxels, n_gauss, 3)
        sq_dist = (diff**2).sum(dim=-1)  # (n_voxels, n_gauss)
        weight = torch.exp(
            -sq_dist / (2 * scale.squeeze(-1).unsqueeze(0) ** 2)
        )  # (n_voxels, n_gauss)
        weight = weight / (weight.sum(dim=-1, keepdim=True) + 1e-6)
        voxel_semantics = weight @ semantics  # (n_voxels, n_classes)
        return voxel_semantics


def build_gaussianformer() -> nn.Module:
    """Build a small GaussianFormer sparse-Gaussian occupancy network."""
    return GaussianFormer(ch=8, n_gaussians=16, n_classes=5, voxel_res=6).eval()


def example_input_gaussianformer() -> tuple[torch.Tensor, torch.Tensor]:
    """(multi-view camera images, flattened voxel-center query coordinates)."""
    images = torch.randn(3, 3, 32, 32)
    res = 6
    lin = torch.linspace(-1, 1, res)
    zz, yy, xx = torch.meshgrid(lin, lin, lin, indexing="ij")
    voxel_centers = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
    return images, voxel_centers


# ============================================================
# GUPNet -- uncertainty-aware geometry projection for monocular 3D detection
# ============================================================


class UncertaintyGeometryProjection(nn.Module):
    """Derives depth geometrically from predicted 2D box height and 3D object

    height via the pinhole identity `depth = f * h_3d / h_2d`, and analytically
    propagates the predicted 3D-height log-variance through that (nonlinear)
    projection to obtain a geometry-derived depth uncertainty -- rather than
    regressing depth directly as a black-box output.
    """

    def forward(
        self,
        h_2d: torch.Tensor,
        h_3d: torch.Tensor,
        h_3d_log_var: torch.Tensor,
        focal_length: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_2d = h_2d.clamp(min=1e-3)
        depth = focal_length * h_3d / h_2d
        # first-order error propagation: Var[depth] ~= (d depth / d h_3d)^2 * Var[h_3d]
        d_depth_d_h3d = focal_length / h_2d
        depth_var = (d_depth_d_h3d**2) * h_3d_log_var.exp()
        depth_log_var = depth_var.clamp(min=1e-6).log()
        return depth, depth_log_var


class GUPNet(nn.Module):
    """GUPNet: a CNN backbone predicts 2D box height, 3D object height (+ its

    log-variance), heading, and offsets per RoI; depth and its uncertainty are
    then *derived* by the UncertaintyGeometryProjection layer instead of being
    regressed directly, so the projection identity itself is part of the
    differentiable forward pass.
    """

    def __init__(self, ch: int = 8, n_roi: int = 4) -> None:
        super().__init__()
        self.backbone = nn.Sequential(_cbr(3, ch, stride=2), _cbr(ch, ch, stride=2))
        self.roi_pool = nn.AdaptiveAvgPool2d(4)
        self.n_roi = n_roi

        feat_dim = ch * 4 * 4
        self.h2d_head = nn.Linear(feat_dim, 1)
        self.h3d_head = nn.Linear(feat_dim, 1)
        self.h3d_logvar_head = nn.Linear(feat_dim, 1)
        self.offset_head = nn.Linear(feat_dim, 2)  # 2D center offset for 3D center projection
        self.heading_head = nn.Linear(feat_dim, 2)  # (sin, cos)
        self.size_head = nn.Linear(feat_dim, 3)  # (w, h, l) residual

        self.geo_projection = UncertaintyGeometryProjection()

    def forward(self, image: torch.Tensor, focal_length: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.backbone(image)
        roi_feat = self.roi_pool(feat).flatten(1).expand(self.n_roi, -1)

        h_2d = F.softplus(self.h2d_head(roi_feat)).squeeze(-1)
        h_3d = F.softplus(self.h3d_head(roi_feat)).squeeze(-1)
        h_3d_log_var = self.h3d_logvar_head(roi_feat).squeeze(-1)
        offset = self.offset_head(roi_feat)
        heading = self.heading_head(roi_feat)
        size = self.size_head(roi_feat)

        depth, depth_log_var = self.geo_projection(h_2d, h_3d, h_3d_log_var, focal_length)
        return {
            "depth": depth,
            "depth_log_var": depth_log_var,
            "offset": offset,
            "heading": heading,
            "size": size,
        }


def build_gupnet() -> nn.Module:
    """Build a small GUPNet monocular 3D detector."""
    return GUPNet(ch=8, n_roi=4).eval()


def example_input_gupnet() -> tuple[torch.Tensor, torch.Tensor]:
    """(monocular RGB image, scalar camera focal length)."""
    return torch.randn(1, 3, 32, 32), torch.tensor(721.5)


# ============================================================
# Registry
# ============================================================

MENAGERIE_ENTRIES = [
    ("FB-OCC", "build_fb_occ", "example_input_fb_occ", "2023", "VIS"),
    (
        "Frustum PointNets",
        "build_frustum_pointnets",
        "example_input_frustum_pointnets",
        "2018",
        "VIS",
    ),
    ("FUTR3D", "build_futr3d", "example_input_futr3d", "2022", "VIS"),
    ("FUTR3D-Radar", "build_futr3d_radar", "example_input_futr3d_radar", "2022", "VIS"),
    ("GaussianFormer", "build_gaussianformer", "example_input_gaussianformer", "2024", "VIS"),
    ("GUPNet", "build_gupnet", "example_input_gupnet", "2021", "VIS"),
]
