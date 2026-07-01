"""Menagerie batch w3a21: radar-camera BEV fusion, NeRF-style occupancy, and
monocular/point-cloud 3D detection architectures.

Sources checked (reference only; no cloning, no pip installs):
  - RCBEVDet: Lin et al., CVPR 2024. Paper https://arxiv.org/abs/2403.16440,
    official source https://github.com/VDIGPKU/RCBEVDet. Radar-camera fusion 3D
    object detection in bird's-eye view (BEV). The distinctive mechanism is
    RadarBEVNet: a *dual-stream* radar backbone combining a point-based encoder
    (per-point MLP over raw radar points) and a transformer-based encoder
    (self-attention over the same radar points), with an injection/extraction
    module that lets the two streams exchange information at each stage; radar
    points are then scattered into a BEV grid using their Radar Cross-Section
    (RCS) value as an object-size prior (RCS-aware BEV encoder), producing a
    radar BEV feature map. This radar BEV map is fused with a camera BEV map
    (from a standard image-to-BEV lift) via a Cross-Attention Multi-layer Fusion
    (CAMF) module: deformable cross-attention aligns the two modalities'
    spatially-offset features, followed by channel- and spatial-gated fusion
    layers, before a detection head regresses BEV boxes.
  - RenderOcc: Pan et al., ICRA 2024. Paper https://arxiv.org/abs/2309.09502,
    official source https://github.com/pmj110119/RenderOcc. Vision-centric 3D
    occupancy prediction supervised *only with 2D labels* (no 3D occupancy
    ground truth). The distinctive mechanism is NeRF-style volume rendering used
    as the supervision bridge: multi-view images are lifted into a 3D feature
    volume (per-voxel density + semantic-embedding features, exactly like a NeRF
    density/radiance field), and for training, camera rays are cast through this
    volume and per-ray density-weighted (alpha-compositing) accumulation renders
    2D depth maps and 2D semantic maps, which are compared against ordinary 2D
    depth/semantic labels; at inference the 3D density/semantic volume itself
    is read out directly as the occupancy prediction, without ever needing 3D
    occupancy annotations at train time.
  - RTM3D: Li et al., ECCV 2020. Paper https://arxiv.org/abs/2001.03343, official
    source https://github.com/Banconxuan/RTM3D. Real-time monocular 3D object
    detection from RGB alone (no depth sensor, no CAD models, no stereo). The
    distinctive mechanism is keypoint-based 3D box parameterization: a CenterNet-
    style backbone predicts (a) a per-class main-center heatmap, (b) 8 heatmaps
    (or offset vectors from the main center) for the projected image-plane
    locations of a 3D bounding box's 8 corner vertices, plus (c) auxiliary
    regression heads for object dimensions, observation angle (encoded via
    trigonometric bins), and depth; the 9 keypoints (main center + 8 vertices)
    over-determine the projective geometry of a cuboid, so a differentiable
    geometric-reasoning step (least-squares / SVD on the projection equations)
    reconstructs a globally consistent 3D box from the noisy keypoint estimates.
  - SE-SSD: Zheng et al., CVPR 2021. Paper https://arxiv.org/abs/2104.09804,
    official source https://github.com/Vegeta2020/SE-SSD. Self-ensembling
    single-stage LiDAR 3D object detector. The distinctive mechanism is a
    teacher-student pair of *identical* single-stage (SSD/anchor-based) point-
    cloud detectors: the teacher consumes a mildly-augmented point cloud and
    produces soft box/confidence pseudo-targets; the student consumes a more
    aggressively "shape-aware" augmented version of the same (globally
    transformed) point cloud and is supervised by a consistency loss against the
    teacher's soft targets (in addition to ordinary hard-label loss); the
    teacher's weights are never trained directly but instead updated as an
    exponential moving average (EMA) of the student's weights after every step,
    so the ensembling happens purely at the weight level and costs nothing extra
    at inference (only the student network is deployed).
  - SelfOcc: Huang et al., CVPR 2024. Paper https://arxiv.org/abs/2311.12754,
    official source https://github.com/huang-yh/SelfOcc. Self-supervised vision-
    based 3D occupancy prediction trained only from posed video (no 3D or even
    2D occupancy/segmentation labels required in the base setting). The
    distinctive mechanism is treating a lifted tri-perspective-view (TPV) 3D
    representation as a *signed distance field* (SDF): three orthogonal 2D
    feature planes are combined per 3D query point (via projection + summation)
    into a signed-distance value (and an appearance/semantic embedding), and
    volume-rendering weights are derived from the SDF (via a sigmoid/CDF-style
    conversion, in the spirit of NeuS/VolSDF) rather than from a raw NeRF
    density head; rays cast from the (posed) camera through this SDF-volume are
    rendered into color/depth for photometric self-supervision against
    neighbouring video frames, so 3D occupancy emerges purely from the implicit
    zero-level-set of the learned signed distance field.

Skipped:
  - SA-SSD (arxiv:2004.03752, CVPR 2020, github.com/skyhehe123/SA-SSD) is
    already in the catalog as ``mmdet3d:sassd`` /
    ``menagerie/classics/reimpl5_openmmlab.py::build_sassd`` (PointPillarsSECOND
    with ``mode="sassd"``), which implements the paper's distinctive auxiliary
    point-level foreground/structure-supervision branch. Not rebuilt here.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

# ---------------------------------------------------------------------------
# RCBEVDet
# ---------------------------------------------------------------------------


class RadarPointStream(nn.Module):
    """Per-point MLP encoder for raw radar points (point-based radar stream)."""

    def __init__(self, in_dim: int, width: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, width),
            nn.ReLU(inplace=True),
            nn.Linear(width, width),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """Encode ``points`` of shape (B, N, in_dim) -> (B, N, width)."""
        return self.mlp(points)


class RadarTransformerStream(nn.Module):
    """Self-attention encoder over the same radar points (transformer stream)."""

    def __init__(self, in_dim: int, width: int, heads: int = 2) -> None:
        super().__init__()
        self.proj_in = nn.Linear(in_dim, width)
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm = nn.LayerNorm(width)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """Encode ``points`` of shape (B, N, in_dim) -> (B, N, width)."""
        feat = self.proj_in(points)
        attended, _ = self.attn(feat, feat, feat)
        return self.norm(feat + attended)


class RadarBEVNet(nn.Module):
    """Dual-stream radar backbone with injection/extraction + RCS-aware BEV scatter.

    Combines a point-based encoder and a transformer-based encoder over the
    same raw radar points, exchanges information between them at one stage
    (the injection/extraction module), and finally scatters the fused
    per-point features onto a BEV grid using each point's Radar Cross-Section
    (RCS) value as a soft, learned object-size prior.
    """

    def __init__(self, in_dim: int = 6, width: int = 24, bev_size: int = 12) -> None:
        super().__init__()
        self.point_stream = RadarPointStream(in_dim, width)
        self.transformer_stream = RadarTransformerStream(in_dim, width)
        # Injection/extraction: each stream reads the other's features.
        self.inject_to_point = nn.Linear(width, width)
        self.inject_to_transformer = nn.Linear(width, width)
        self.fuse = nn.Linear(2 * width, width)
        # RCS-aware size prior: maps a scalar RCS value to a per-point kernel
        # weight controlling how "wide" the point's scatter footprint is.
        self.rcs_prior = nn.Sequential(nn.Linear(1, 8), nn.ReLU(inplace=True), nn.Linear(8, 1))
        self.bev_size = bev_size
        self.width = width

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """``points``: (B, N, in_dim) with xy in [:, :, :2] and RCS in [:, :, -1].

        Returns a radar BEV feature map of shape (B, width, bev_size, bev_size).
        """
        pt_feat = self.point_stream(points)
        tr_feat = self.transformer_stream(points)
        # Injection/extraction exchange between the two streams.
        pt_feat = pt_feat + self.inject_to_point(tr_feat)
        tr_feat = tr_feat + self.inject_to_transformer(pt_feat)
        fused = self.fuse(torch.cat([pt_feat, tr_feat], dim=-1))

        rcs = points[..., -1:]
        size_weight = torch.sigmoid(self.rcs_prior(rcs))  # (B, N, 1), RCS-aware prior
        weighted = fused * size_weight

        xy = points[..., :2].clamp(-1.0, 1.0)
        grid = ((xy + 1.0) * 0.5 * (self.bev_size - 1)).long().clamp(0, self.bev_size - 1)
        batch, n_pts, _ = points.shape
        bev = points.new_zeros(batch, self.width, self.bev_size, self.bev_size)
        for b in range(batch):
            idx = grid[b, :, 1] * self.bev_size + grid[b, :, 0]
            flat = bev[b].reshape(self.width, -1)
            flat.index_add_(1, idx, weighted[b].transpose(0, 1))
            bev[b] = flat.reshape(self.width, self.bev_size, self.bev_size)
        return bev


class CrossAttentionMultiLayerFusion(nn.Module):
    """CAMF: deformable-style cross-attention alignment + channel/spatial fusion.

    Aligns the radar and camera BEV feature maps (which may be spatially
    offset due to different sensing geometries) with an attention-based
    alignment step, then fuses the aligned features through a channel-gate
    followed by a spatial-gate, in the spirit of the paper's CAMF module.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.align_q = nn.Conv2d(channels, channels, 1)
        self.align_k = nn.Conv2d(channels, channels, 1)
        self.align_v = nn.Conv2d(channels, channels, 1)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, channels, 1),
            nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(nn.Conv2d(channels * 2, 1, 3, padding=1), nn.Sigmoid())
        self.out_proj = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, camera_bev: torch.Tensor, radar_bev: torch.Tensor) -> torch.Tensor:
        """``camera_bev``/``radar_bev``: (B, C, H, W) -> fused (B, C, H, W)."""
        b, c, h, w = camera_bev.shape
        q = self.align_q(camera_bev).flatten(2).transpose(1, 2)  # (B, HW, C)
        k = self.align_k(radar_bev).flatten(2)  # (B, C, HW)
        v = self.align_v(radar_bev).flatten(2).transpose(1, 2)  # (B, HW, C)
        attn = torch.softmax(q @ k / (c**0.5), dim=-1)
        aligned_radar = (attn @ v).transpose(1, 2).reshape(b, c, h, w)

        concat = torch.cat([camera_bev, aligned_radar], dim=1)
        gated = camera_bev * self.channel_gate(concat) + aligned_radar * (
            1 - self.channel_gate(concat)
        )
        spatial = self.spatial_gate(torch.cat([gated, aligned_radar], dim=1))
        fused = gated * spatial + aligned_radar * (1 - spatial)
        return self.out_proj(fused)


class RCBEVDet(nn.Module):
    """RCBEVDet: RadarBEVNet + camera BEV lift, fused via CAMF, box head."""

    def __init__(self, bev_size: int = 12, width: int = 24) -> None:
        super().__init__()
        self.radar_backbone = RadarBEVNet(in_dim=6, width=width, bev_size=bev_size)
        self.image_backbone = nn.Sequential(
            nn.Conv2d(3, width, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(bev_size),
        )
        self.camf = CrossAttentionMultiLayerFusion(width)
        self.cls_head = nn.Conv2d(width, 3, 1)
        self.box_head = nn.Conv2d(width, 7, 1)

    def forward(
        self, images: torch.Tensor, radar_points: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``images``: (B, n_cam, 3, H, W); ``radar_points``: (B, N, 6)."""
        b, n_cam, c, h, w = images.shape
        cam_feats = self.image_backbone(images.reshape(b * n_cam, c, h, w))
        cam_bev = cam_feats.reshape(b, n_cam, *cam_feats.shape[1:]).mean(dim=1)

        radar_bev = self.radar_backbone(radar_points)
        fused = self.camf(cam_bev, radar_bev)
        return self.cls_head(fused), self.box_head(fused)


def build_rcbevdet() -> nn.Module:
    """Build a compact RCBEVDet radar-camera BEV fusion detector."""
    return RCBEVDet(bev_size=12, width=24).eval()


def example_input_rcbevdet() -> tuple[torch.Tensor, torch.Tensor]:
    """2 camera views (1, 2, 3, 32, 32) + 40 radar points (1, 40, 6: x,y,z,vx,vy,rcs)."""
    images = torch.randn(1, 2, 3, 32, 32)
    radar_points = torch.randn(1, 40, 6)
    radar_points[..., :2] = radar_points[..., :2].clamp(-1.0, 1.0)
    return images, radar_points


# ---------------------------------------------------------------------------
# RenderOcc
# ---------------------------------------------------------------------------


class RenderOcc(nn.Module):
    """RenderOcc: NeRF-style 3D volume from multi-view images + volume rendering.

    Multi-view images are lifted into a per-voxel (density, semantic-feature)
    3D volume. During training, camera rays are cast through the volume and
    density-weighted alpha-compositing accumulates per-ray depth and semantic
    predictions, so the model can be supervised purely with ordinary 2D depth
    and semantic labels while learning a genuinely 3D occupancy volume.
    """

    def __init__(
        self,
        vox: int = 8,
        n_classes: int = 5,
        feat_dim: int = 16,
        n_samples: int = 6,
    ) -> None:
        super().__init__()
        self.vox = vox
        self.n_samples = n_samples
        self.image_backbone = nn.Sequential(
            nn.Conv2d(3, feat_dim, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim, feat_dim, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        # Lift 2D features to a dense 3D (density + semantic) voxel volume.
        self.lift = nn.Linear(feat_dim, vox * (1 + n_classes))
        self.n_classes = n_classes

    def forward(
        self, images: torch.Tensor, ray_dirs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``images``: (B, n_cam, 3, H, W); ``ray_dirs``: (B, n_rays, 3).

        Returns rendered per-ray (depth, semantic_logits) via alpha compositing
        through the lifted density/semantic volume, plus (implicitly) exposes
        the raw volume as ``self._volume`` for direct occupancy read-out.
        """
        b, n_cam, c, h, w = images.shape
        feat = self.image_backbone(images.reshape(b * n_cam, c, h, w))
        pooled = feat.mean(dim=(2, 3)).reshape(b, n_cam, -1).mean(dim=1)  # (B, feat_dim)

        volume = self.lift(pooled).reshape(b, self.vox, 1 + self.n_classes)
        density = F.softplus(volume[..., 0])  # (B, vox)
        semantics = volume[..., 1:]  # (B, vox, n_classes)
        self._volume = volume

        n_rays = ray_dirs.shape[1]
        # Sample `n_samples` depths per ray along the shared voxel axis.
        depths = torch.linspace(0.0, 1.0, self.n_samples, device=images.device)
        sample_idx = (depths * (self.vox - 1)).long()  # (n_samples,)
        sample_density = density[:, sample_idx]  # (B, n_samples)
        sample_semantics = semantics[:, sample_idx]  # (B, n_samples, n_classes)

        delta = 1.0 / self.n_samples
        alpha = 1.0 - torch.exp(-sample_density * delta)  # (B, n_samples)
        transmittance = torch.cumprod(
            torch.cat([torch.ones_like(alpha[:, :1]), 1.0 - alpha + 1e-10], dim=1), dim=1
        )[:, :-1]
        weights = (
            (alpha * transmittance).unsqueeze(1).expand(-1, n_rays, -1)
        )  # (B, n_rays, n_samples)

        rendered_depth = (weights * depths.view(1, 1, -1)).sum(dim=-1)  # (B, n_rays)
        rendered_sem = torch.einsum(
            "brs,bsk->brk", weights, sample_semantics
        )  # (B, n_rays, n_classes)
        return rendered_depth, rendered_sem


def build_renderocc() -> nn.Module:
    """Build a compact RenderOcc NeRF-style occupancy-via-rendering module."""
    return RenderOcc(vox=8, n_classes=5, feat_dim=16, n_samples=6).eval()


def example_input_renderocc() -> tuple[torch.Tensor, torch.Tensor]:
    """3 camera views (1, 3, 3, 32, 32) + 20 sampled ray directions (1, 20, 3)."""
    images = torch.randn(1, 3, 3, 32, 32)
    ray_dirs = F.normalize(torch.randn(1, 20, 3), dim=-1)
    return images, ray_dirs


# ---------------------------------------------------------------------------
# RTM3D
# ---------------------------------------------------------------------------


class RTM3D(nn.Module):
    """RTM3D: CenterNet-style keypoint heatmaps + geometric box reconstruction.

    A shared backbone predicts a main-center heatmap, 8 corner-vertex
    heatmaps/offsets, and per-object dimension/orientation/depth regression.
    The 9 keypoints (center + 8 projected 3D-box corners) are then combined
    via a small differentiable geometric-reasoning head that reconstructs a
    single consistent (dimension, orientation, depth) box estimate, mirroring
    the paper's SVD-based geometric constraint solver.
    """

    def __init__(self, n_classes: int = 3, width: int = 24) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, width, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.center_heatmap = nn.Conv2d(width, n_classes, 3, padding=1)
        self.vertex_heatmap = nn.Conv2d(width, 8, 3, padding=1)  # 8 projected 3D-box corners
        self.vertex_offset = nn.Conv2d(width, 8 * 2, 3, padding=1)  # sub-pixel (dx, dy) per vertex
        self.dim_head = nn.Conv2d(width, 3, 3, padding=1)  # (h, w, l)
        self.orientation_head = nn.Conv2d(width, 8, 3, padding=1)  # multi-bin sin/cos encoding
        self.depth_head = nn.Conv2d(width, 1, 3, padding=1)
        # Geometric-reasoning module: consumes the 9 keypoint locations
        # (flattened) to solve a single consistent 3D box (least-squares
        # analogue of the paper's SVD-based constraint solver).
        self.geometric_reasoning = nn.Sequential(
            nn.Linear(9 * 2, 32), nn.ReLU(inplace=True), nn.Linear(32, 7)
        )

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """``image``: (B, 3, H, W) -> dict of heatmaps/regressions + geometric box."""
        feat = self.backbone(image)
        center = self.center_heatmap(feat)
        vertices = self.vertex_heatmap(feat)
        vertex_offsets = self.vertex_offset(feat)
        dims = self.dim_head(feat)
        orientation = self.orientation_head(feat)
        depth = self.depth_head(feat)

        # Collapse spatial keypoint maps to per-image (x, y) via a soft-argmax
        # so the geometric-reasoning head can consume a fixed-size keypoint set.
        center_xy = _soft_argmax(center.mean(dim=1, keepdim=True))
        vertex_xy = torch.cat(
            [_soft_argmax(vertices[:, i : i + 1]) for i in range(vertices.shape[1])], dim=1
        )
        keypoints = torch.cat([center_xy, vertex_xy], dim=1).flatten(1)  # (B, 9*2)
        box_3d = self.geometric_reasoning(
            keypoints
        )  # (B, 7): dims(3) + orient(2) + depth(1) + score(1)

        return {
            "center_heatmap": center,
            "vertex_heatmap": vertices,
            "vertex_offset": vertex_offsets,
            "dims": dims,
            "orientation": orientation,
            "depth": depth,
            "box_3d": box_3d,
        }


def _soft_argmax(heatmap: torch.Tensor) -> torch.Tensor:
    """Differentiable (x, y) location of a single-channel heatmap's peak."""
    b, _, h, w = heatmap.shape
    flat = heatmap.reshape(b, -1)
    weights = torch.softmax(flat, dim=-1).reshape(b, h, w)
    ys = torch.linspace(0, 1, h, device=heatmap.device).view(1, h, 1)
    xs = torch.linspace(0, 1, w, device=heatmap.device).view(1, 1, w)
    x = (weights * xs).sum(dim=(1, 2), keepdim=True).squeeze(-1)
    y = (weights * ys).sum(dim=(1, 2), keepdim=True).squeeze(-1)
    return torch.cat([x, y], dim=1)


def build_rtm3d() -> nn.Module:
    """Build a compact RTM3D monocular keypoint-based 3D detector."""
    return RTM3D(n_classes=3, width=24).eval()


def example_input_rtm3d() -> torch.Tensor:
    """A single monocular RGB frame, (1, 3, 64, 64)."""
    return torch.randn(1, 3, 64, 64)


# ---------------------------------------------------------------------------
# SE-SSD
# ---------------------------------------------------------------------------


class _SimplePointPillarSSD(nn.Module):
    """Minimal single-stage point-cloud detector (pillar-scatter + BEV CNN + head)."""

    def __init__(self, bev_size: int = 12, width: int = 16) -> None:
        super().__init__()
        self.bev_size = bev_size
        self.pillar_encoder = nn.Sequential(nn.Linear(4, width), nn.ReLU(inplace=True))
        self.bev_cnn = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.cls_head = nn.Conv2d(width, 1, 1)
        self.box_head = nn.Conv2d(width, 7, 1)
        self.width = width

    def forward(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``points``: (B, N, 4: x, y, z, intensity) -> (cls_map, box_map)."""
        feat = self.pillar_encoder(points)  # (B, N, width)
        xy = points[..., :2].clamp(-1.0, 1.0)
        grid = ((xy + 1.0) * 0.5 * (self.bev_size - 1)).long().clamp(0, self.bev_size - 1)
        b, n_pts, _ = points.shape
        bev = points.new_zeros(b, self.width, self.bev_size, self.bev_size)
        for i in range(b):
            idx = grid[i, :, 1] * self.bev_size + grid[i, :, 0]
            flat = bev[i].reshape(self.width, -1)
            flat.index_add_(1, idx, feat[i].transpose(0, 1))
            bev[i] = flat.reshape(self.width, self.bev_size, self.bev_size)
        bev = self.bev_cnn(bev)
        return self.cls_head(bev), self.box_head(bev)


class SESSD(nn.Module):
    """SE-SSD: teacher-student self-ensembling single-stage 3D detector.

    Wraps two structurally-identical single-stage point-cloud detectors. The
    teacher runs on a mildly-augmented point cloud and produces soft
    pseudo-targets; the student runs on a shape-aware-augmented (here: jittered
    + randomly-scaled) version of the same point cloud. Only the student's
    weights receive gradients; the teacher's weights are conceptually an EMA
    shadow of the student (represented here via ``update_teacher_ema``, kept
    out of ``forward`` so tracing sees a pure inference graph).
    """

    def __init__(self, bev_size: int = 12, width: int = 16, ema_decay: float = 0.999) -> None:
        super().__init__()
        self.student = _SimplePointPillarSSD(bev_size=bev_size, width=width)
        self.teacher = _SimplePointPillarSSD(bev_size=bev_size, width=width)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.ema_decay = ema_decay

    @torch.no_grad()
    def update_teacher_ema(self) -> None:
        """Update teacher weights as an exponential moving average of the student."""
        for t_param, s_param in zip(self.teacher.parameters(), self.student.parameters()):
            t_param.mul_(self.ema_decay).add_(s_param, alpha=1.0 - self.ema_decay)

    def forward(
        self, points: torch.Tensor, shape_aware_points: torch.Tensor
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        """``points``: teacher input; ``shape_aware_points``: student's augmented input.

        Returns ((teacher_cls, teacher_box), (student_cls, student_box)); a
        consistency loss compares the two at train time.
        """
        teacher_out = self.teacher(points)
        student_out = self.student(shape_aware_points)
        return teacher_out, student_out


def build_se_ssd() -> nn.Module:
    """Build a compact SE-SSD teacher-student self-ensembling 3D detector."""
    return SESSD(bev_size=12, width=16).eval()


def example_input_se_ssd() -> tuple[torch.Tensor, torch.Tensor]:
    """Teacher points (1, 50, 4) and a shape-aware-augmented student copy."""
    points = torch.randn(1, 50, 4)
    points[..., :2] = points[..., :2].clamp(-1.0, 1.0)
    shape_aware_points = points + 0.05 * torch.randn_like(points)
    shape_aware_points[..., :2] = shape_aware_points[..., :2].clamp(-1.0, 1.0)
    return points, shape_aware_points


# ---------------------------------------------------------------------------
# SelfOcc
# ---------------------------------------------------------------------------


class TriPerspectiveViewSDF(nn.Module):
    """Tri-perspective-view (TPV) feature planes read out as a signed distance field.

    Three orthogonal 2D feature planes (XY, YZ, XZ) are each produced from the
    pooled image feature, and for a batch of query points, each point's
    projection onto the three planes is bilinearly sampled and summed into a
    per-point feature, which a small MLP head converts into a signed-distance
    value plus an appearance embedding -- the paper's SDF-based 3D
    representation in place of a raw NeRF density field.
    """

    def __init__(
        self, feat_dim: int, plane_res: int, plane_channels: int, appearance_dim: int
    ) -> None:
        super().__init__()
        self.plane_res = plane_res
        self.plane_channels = plane_channels
        self.to_planes = nn.Linear(feat_dim, 3 * plane_channels * plane_res * plane_res)
        self.sdf_head = nn.Sequential(
            nn.Linear(plane_channels, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1 + appearance_dim),
        )

    def forward(self, pooled_feat: torch.Tensor, query_xyz: torch.Tensor) -> torch.Tensor:
        """``pooled_feat``: (B, feat_dim); ``query_xyz``: (B, n_pts, 3) in [-1, 1].

        Returns (B, n_pts, 1 + appearance_dim): signed distance + appearance.
        """
        b = pooled_feat.shape[0]
        planes = self.to_planes(pooled_feat).reshape(
            b, 3, self.plane_channels, self.plane_res, self.plane_res
        )
        xy = query_xyz[..., [0, 1]]
        yz = query_xyz[..., [1, 2]]
        xz = query_xyz[..., [0, 2]]

        def sample(plane: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
            grid = coords.unsqueeze(1)  # (B, 1, n_pts, 2)
            sampled = F.grid_sample(plane, grid, align_corners=True)  # (B, C, 1, n_pts)
            return sampled.squeeze(2).transpose(1, 2)  # (B, n_pts, C)

        feat = sample(planes[:, 0], xy) + sample(planes[:, 1], yz) + sample(planes[:, 2], xz)
        return self.sdf_head(feat)


class SelfOcc(nn.Module):
    """SelfOcc: image -> TPV-SDF representation -> SDF-derived volume rendering.

    Converts an SDF value along a ray to alpha-compositing weights via a
    logistic (sigmoid-CDF) transform in the spirit of NeuS/VolSDF, then
    renders per-ray depth and appearance for photometric self-supervision.
    """

    def __init__(
        self,
        feat_dim: int = 16,
        plane_res: int = 8,
        plane_channels: int = 8,
        appearance_dim: int = 3,
        n_samples: int = 6,
        sdf_scale: float = 10.0,
    ) -> None:
        super().__init__()
        self.image_backbone = nn.Sequential(
            nn.Conv2d(3, feat_dim, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim, feat_dim, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.tpv_sdf = TriPerspectiveViewSDF(feat_dim, plane_res, plane_channels, appearance_dim)
        self.n_samples = n_samples
        self.sdf_scale = sdf_scale

    def forward(
        self, image: torch.Tensor, ray_origin: torch.Tensor, ray_dir: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``image``: (B, 3, H, W); ``ray_origin``/``ray_dir``: (B, 3).

        Samples `n_samples` points per ray, evaluates the TPV signed distance
        field at each, converts SDF -> rendering weights via a logistic CDF,
        and alpha-composites depth + appearance.
        """
        b = image.shape[0]
        pooled = self.image_backbone(image).flatten(1)

        depths = torch.linspace(0.1, 1.0, self.n_samples, device=image.device)
        pts = ray_origin.unsqueeze(1) + ray_dir.unsqueeze(1) * depths.view(1, -1, 1)  # (B, S, 3)
        pts = pts.clamp(-1.0, 1.0)

        out = self.tpv_sdf(pooled, pts)  # (B, S, 1 + appearance_dim)
        sdf = out[..., 0]
        appearance = out[..., 1:]

        # NeuS/VolSDF-style SDF -> opacity: logistic CDF of the (negated,
        # scaled) signed distance, so the implicit zero-level-set becomes the
        # surface where opacity rises sharply.
        opacity = torch.sigmoid(-sdf * self.sdf_scale)
        weights = opacity * torch.cumprod(
            torch.cat(
                [torch.ones(b, 1, device=image.device), 1.0 - opacity[:, :-1] + 1e-10], dim=1
            ),
            dim=1,
        )

        rendered_depth = (weights * depths.view(1, -1)).sum(dim=1)
        rendered_appearance = torch.einsum("bs,bsk->bk", weights, appearance)
        return rendered_depth, rendered_appearance


def build_selfocc() -> nn.Module:
    """Build a compact SelfOcc TPV signed-distance-field occupancy module."""
    return SelfOcc(feat_dim=16, plane_res=8, plane_channels=8, appearance_dim=3, n_samples=6).eval()


def example_input_selfocc() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One image (1, 3, 32, 32) + one ray (origin, direction), each (1, 3)."""
    image = torch.randn(1, 3, 32, 32)
    ray_origin = torch.zeros(1, 3)
    ray_dir = F.normalize(torch.randn(1, 3), dim=-1)
    return image, ray_origin, ray_dir


MENAGERIE_ENTRIES = [
    ("RCBEVDet", "build_rcbevdet", "example_input_rcbevdet", "2024", "VIS"),
    ("RenderOcc", "build_renderocc", "example_input_renderocc", "2024", "VIS"),
    ("RTM3D", "build_rtm3d", "example_input_rtm3d", "2020", "VIS"),
    ("SE-SSD", "build_se_ssd", "example_input_se_ssd", "2021", "VIS"),
    ("SelfOcc", "build_selfocc", "example_input_selfocc", "2024", "VIS"),
]
