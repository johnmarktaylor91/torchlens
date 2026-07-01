"""Menagerie batch w3a9: BEV/3D-perception architectures for autonomous driving.

Sources checked (reference only; no cloning, no pip installs):
  - BEVerse: Zhang et al., 2022. Paper https://arxiv.org/abs/2205.09743, official
    source https://github.com/zhangyp15/BEVerse. Unified multi-camera perception +
    prediction: a per-camera image backbone extracts features, a Lift-Splat-Shoot
    (LSS) view transform (per-pixel categorical depth distribution outer-producted
    with image context features, then splatted/pooled into a BEV grid) lifts each
    camera into a shared BEV plane, multiple BEV frames across time are warped to
    the current ego frame and temporally fused (4D), and three task-specific BEV
    decoder heads (3D detection, map segmentation, motion prediction) share the
    fused BEV feature. DISTINCTIVE: multi-camera LSS lift + explicit MULTI-FRAME
    temporal BEV fusion + multi-task decoder heads sharing one BEV feature map.
  - BEVGuide: Man et al., 2022 (BEV-Guided Multi-Modality Fusion for Driving
    Perception). Paper https://arxiv.org/abs/2212.09349, project page
    https://yunzeman.github.io/BEVGuide/ (official code not released at capture
    time; architecture reimplemented directly from the paper description).
    DISTINCTIVE: a learned BEV query grid (one embedding per BEV cell) acts as the
    QUERY in a cross-attention block against per-sensor feature tokens (here two
    heterogeneous sensor streams -- camera and LiDAR/radar-style token sets) --
    BEV-guided multi-sensor cross-attention with NO explicit geometric projection,
    contrasted with LSS-style methods (BEVerse/BEVHeight/BEVStereo below) that
    project pixels into BEV via depth/height bins.
  - BEVHeight: Yang et al., CVPR 2023. Paper https://arxiv.org/abs/2303.08498,
    official source https://github.com/ADLab-AutoDrive/BEVHeight. Roadside
    (infrastructure-mounted, near-flat/oblique-view camera) 3D detector. Verified
    from `layers/backbones/lss_fpn.py`: a `HeightNet` mirrors BEVDepth's DepthNet
    exactly in structure (camera-parameter MLP -> two SE-gated branches) but the
    per-pixel distribution branch predicts a HEIGHT-above-ground bin distribution
    instead of a depth bin distribution -- roadside cameras make height a far
    better-conditioned per-pixel geometric signal than depth. The height
    distribution outer-products with a context feature branch and is
    splatted/pooled into the BEV grid exactly as in LSS. DISTINCTIVE: camera-aware
    (intrinsics/extrinsics MLP -> SE gate) HEIGHT-distribution view transform
    (not depth).
  - BEVStereo: Li et al., 2022. Paper https://arxiv.org/abs/2209.10248, official
    source https://github.com/Megvii-BaseDetection/BEVStereo. Verified from
    `layers/backbones/lss_fpn.py`: extends a BEVDepth-style camera-aware DepthNet
    with a TEMPORAL STEREO cost volume -- multi-sweep image features are
    homography-warped onto the reference (key) frame across a set of candidate
    depth hypotheses and correlated (dot-product) against the reference feature to
    build a per-pixel depth-hypothesis cost volume, which is aggregated
    (3D-conv-style) into a stereo depth confidence that refines the mono depth
    distribution before the LSS splat. DISTINCTIVE: monocular camera-aware depth
    net FUSED with a multi-frame homography-warped stereo cost volume that
    directly refines the depth-bin distribution (not just a temporal BEV feature
    warp, contrasted with BEVerse's post-hoc BEV-level temporal fusion).
  - BtcDet (Behind the Curtain): Xu et al., AAAI 2022. Paper
    https://arxiv.org/abs/2112.02205, official source
    https://github.com/Xharlie/BtcDet (built on OpenPCDet). Verified from
    `btcdet/models/backbones_3d/vfe/occ_vfe.py` and
    `btcdet/models/occ_pnt/occ_dense_heads/`: a two-branch voxel feature encoder
    separately pools RAW (observed LiDAR) points and OCCLUDED-region SHAPE-
    completion points per voxel (an auxiliary occupancy/shape head predicts dense
    voxel occupancy behind observed surfaces, i.e. "behind the curtain"), then
    concatenates raw + occupancy-conditioned features before a shared 3D sparse
    backbone and detection head. DISTINCTIVE: an auxiliary DENSE OCCUPANCY
    (shape-completion) branch predicts occluded voxel occupancy from sparse input,
    and that occupancy signal is explicitly fused with raw point features before
    3D detection -- learning occluded shapes to disambiguate self-occluded
    objects.
  - Cam4DOcc / OCFNet: Ma et al., CVPR 2024. Paper
    https://arxiv.org/abs/2311.17663, official source
    https://github.com/haomo-ai/Cam4DOcc (benchmark + OCFNet baseline, see
    `OCFNet.png` in the repo). Camera-only 4D (space + time) occupancy
    FORECASTING: a multi-camera LSS-style encoder lifts each timestep's images
    into a BEV/voxel occupancy feature, a temporal module fuses a short history of
    BEV features (warped to the present ego frame), and the network then predicts
    FUTURE occupancy + instance-flow grids for several future timesteps from a
    single shared spatiotemporal feature -- i.e. it forecasts, not just perceives,
    dense occupancy. DISTINCTIVE: single BEV encoder + temporal fusion followed by
    a MULTI-STEP FUTURE occupancy/flow decoder (one head per forecast horizon
    sharing the same fused history feature), contrasted with BEVerse (which
    predicts one future flow field, not stacked per-timestep occupancy) and plain
    single-frame occupancy networks.

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


class _ImageBackbone(nn.Module):
    """Small strided-conv image feature extractor (stand-in for a ResNet+FPN neck)."""

    def __init__(self, in_ch: int = 3, out_ch: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            _cbr(in_ch, 16, stride=2),
            _cbr(16, out_ch, stride=2),
            _cbr(out_ch, out_ch, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (B, out_ch, H/8, W/8)


class _SELayer(nn.Module):
    """Squeeze-excite gate conditioned on an external context vector (camera-aware gate)."""

    def __init__(self, channels: int, ctx_dim: int) -> None:
        super().__init__()
        self.ctx_proj = nn.Linear(ctx_dim, channels)
        self.gate = nn.Sequential(nn.Conv2d(channels, channels, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        se = self.ctx_proj(ctx)[..., None, None]
        return x * self.gate(x + se)


# ============================================================
# BEVerse -- multi-camera LSS lift + multi-frame BEV temporal fusion + multi-task heads
# ============================================================


class _LiftSplat(nn.Module):
    """Per-camera LSS view transform: depth-distribution outer-product context -> BEV splat."""

    def __init__(self, feat_ch: int = 32, n_depth: int = 8, bev_ch: int = 24) -> None:
        super().__init__()
        self.n_depth = n_depth
        self.depth_context_head = nn.Conv2d(feat_ch, n_depth + bev_ch, 1)
        self.bev_ch = bev_ch

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        # feat: (B*n_cam, feat_ch, h, w)
        out = self.depth_context_head(feat)
        depth_logits, context = out[:, : self.n_depth], out[:, self.n_depth :]
        depth_prob = depth_logits.softmax(dim=1)  # (B*n_cam, n_depth, h, w)
        # Outer product depth x context -> pseudo point-cloud feature volume,
        # collapsed along the depth axis via a weighted sum as a compact stand-in
        # for the LSS voxel-pooling / cumsum-trick splat onto the BEV grid.
        lifted = torch.einsum("bdhw,bchw->bcdhw", depth_prob, context)
        bev = lifted.sum(dim=2)  # (B*n_cam, bev_ch, h, w) -- collapse depth axis
        return bev


class BEVerse(nn.Module):
    """BEVerse: multi-camera LSS lift, multi-frame BEV temporal fusion, multi-task heads."""

    def __init__(
        self,
        n_cam: int = 3,
        n_frames: int = 2,
        feat_ch: int = 32,
        n_depth: int = 8,
        bev_ch: int = 24,
    ) -> None:
        super().__init__()
        self.n_cam = n_cam
        self.n_frames = n_frames
        self.backbone = _ImageBackbone(3, feat_ch)
        self.lift_splat = _LiftSplat(feat_ch, n_depth, bev_ch)
        # Multi-frame temporal BEV fusion: concat warped-to-present frames, fuse with conv.
        self.temporal_fuse = _cbr(bev_ch * n_frames, bev_ch)
        # Multi-task decoder heads sharing the fused BEV feature.
        self.det_head = nn.Conv2d(bev_ch, 8, 1)  # 3D box regression + score
        self.map_head = nn.Conv2d(bev_ch, 4, 1)  # map segmentation classes
        self.motion_head = nn.Conv2d(bev_ch, 2, 1)  # motion flow (dx, dy)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        # images: (B, n_frames, n_cam, 3, H, W)
        b, t, n, c, h, w = images.shape
        flat = images.reshape(b * t * n, c, h, w)
        feat = self.backbone(flat)
        bev_per_cam = self.lift_splat(feat)  # (B*t*n, bev_ch, h', w')
        bev_per_cam = bev_per_cam.reshape(b, t, n, *bev_per_cam.shape[1:])
        bev_per_frame = bev_per_cam.sum(dim=2)  # fuse cameras -> (B, t, bev_ch, h', w')
        bev_cat = bev_per_frame.reshape(b, t * bev_per_frame.shape[2], *bev_per_frame.shape[3:])
        fused = self.temporal_fuse(bev_cat)  # (B, bev_ch, h', w')
        return {
            "detection": self.det_head(fused),
            "map": self.map_head(fused),
            "motion": self.motion_head(fused),
        }


def build_beverse() -> nn.Module:
    """Build a compact BEVerse (multi-camera, multi-frame, multi-task)."""
    return BEVerse(n_cam=3, n_frames=2, feat_ch=32, n_depth=8, bev_ch=24).eval()


def example_input_beverse() -> torch.Tensor:
    """(1, 2 frames, 3 cameras, 3, 64, 64) multi-camera multi-frame image stack."""
    return torch.randn(1, 2, 3, 3, 64, 64)


# ============================================================
# BEVGuide -- BEV query grid cross-attends multi-sensor tokens (no geometric lift)
# ============================================================


class _BEVGuidedAttention(nn.Module):
    """A learned BEV embedding grid cross-attends per-sensor feature tokens."""

    def __init__(self, bev_dim: int, n_heads: int = 4) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(bev_dim, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(bev_dim)
        self.ffn = nn.Sequential(
            nn.Linear(bev_dim, bev_dim * 2), nn.ReLU(inplace=True), nn.Linear(bev_dim * 2, bev_dim)
        )
        self.norm2 = nn.LayerNorm(bev_dim)

    def forward(self, bev_query: torch.Tensor, sensor_tokens: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(bev_query, sensor_tokens, sensor_tokens)
        bev_query = self.norm(bev_query + attn_out)
        bev_query = self.norm2(bev_query + self.ffn(bev_query))
        return bev_query


class BEVGuide(nn.Module):
    """BEVGuide: BEV-guided multi-sensor (camera + lidar/radar-style) cross-attention."""

    def __init__(
        self,
        bev_h: int = 8,
        bev_w: int = 8,
        bev_dim: int = 32,
        n_layers: int = 2,
        n_cam_tokens: int = 16,
        n_lidar_tokens: int = 16,
    ) -> None:
        super().__init__()
        self.bev_h, self.bev_w, self.bev_dim = bev_h, bev_w, bev_dim
        self.bev_query = nn.Parameter(torch.randn(1, bev_h * bev_w, bev_dim) * 0.02)
        self.cam_encoder = nn.Linear(48, bev_dim)  # flat camera-patch tokens -> bev_dim
        self.lidar_encoder = nn.Linear(4, bev_dim)  # (x, y, z, intensity) point tokens -> bev_dim
        self.layers = nn.ModuleList([_BEVGuidedAttention(bev_dim) for _ in range(n_layers)])
        self.seg_head = nn.Linear(bev_dim, 5)

    def forward(self, cam_tokens: torch.Tensor, lidar_tokens: torch.Tensor) -> torch.Tensor:
        # cam_tokens: (B, n_cam_tokens, 48); lidar_tokens: (B, n_lidar_tokens, 4)
        b = cam_tokens.shape[0]
        cam_feat = self.cam_encoder(cam_tokens)
        lidar_feat = self.lidar_encoder(lidar_tokens)
        sensor_tokens = torch.cat([cam_feat, lidar_feat], dim=1)
        bev_query = self.bev_query.expand(b, -1, -1)
        for layer in self.layers:
            bev_query = layer(bev_query, sensor_tokens)
        logits = self.seg_head(bev_query)  # (B, bev_h*bev_w, 5)
        return logits.transpose(1, 2).reshape(b, 5, self.bev_h, self.bev_w)


def build_bevguide() -> nn.Module:
    """Build a compact BEVGuide (BEV-guided multi-sensor cross-attention)."""
    return BEVGuide(
        bev_h=8, bev_w=8, bev_dim=32, n_layers=2, n_cam_tokens=16, n_lidar_tokens=16
    ).eval()


def example_input_bevguide() -> tuple[torch.Tensor, torch.Tensor]:
    """(camera_tokens (1, 16, 48), lidar_tokens (1, 16, 4))."""
    return torch.randn(1, 16, 48), torch.randn(1, 16, 4)


# ============================================================
# BEVHeight -- camera-aware HEIGHT-distribution view transform (roadside 3D detection)
# ============================================================


class _HeightNet(nn.Module):
    """Camera-aware HeightNet: intrinsics/extrinsics MLP -> SE gates -> height + context."""

    def __init__(
        self, feat_ch: int, mid_ch: int, ctx_ch: int, n_height: int, cam_param_dim: int = 12
    ) -> None:
        super().__init__()
        self.reduce = _cbr(feat_ch, mid_ch, k=1, padding=0)
        self.cam_mlp_height = nn.Sequential(
            nn.Linear(cam_param_dim, mid_ch), nn.ReLU(inplace=True), nn.Linear(mid_ch, mid_ch)
        )
        self.cam_mlp_ctx = nn.Sequential(
            nn.Linear(cam_param_dim, mid_ch), nn.ReLU(inplace=True), nn.Linear(mid_ch, mid_ch)
        )
        self.height_se = _SELayer(mid_ch, mid_ch)
        self.context_se = _SELayer(mid_ch, mid_ch)
        self.height_head = nn.Conv2d(mid_ch, n_height, 1)
        self.context_head = nn.Conv2d(mid_ch, ctx_ch, 1)

    def forward(
        self, feat: torch.Tensor, cam_params: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.reduce(feat)
        height_ctx = self.cam_mlp_height(cam_params)
        ctx_ctx = self.cam_mlp_ctx(cam_params)
        height_feat = self.height_se(x, height_ctx)
        context_feat = self.context_se(x, ctx_ctx)
        height = self.height_head(height_feat)  # (B, n_height, h, w) logits
        context = self.context_head(context_feat)  # (B, ctx_ch, h, w)
        return height, context


class BEVHeight(nn.Module):
    """BEVHeight: roadside 3D detector via height-distribution (not depth) view transform."""

    def __init__(
        self, feat_ch: int = 32, mid_ch: int = 32, ctx_ch: int = 16, n_height: int = 10
    ) -> None:
        super().__init__()
        self.backbone = _ImageBackbone(3, feat_ch)
        self.height_net = _HeightNet(feat_ch, mid_ch, ctx_ch, n_height)
        self.n_height = n_height
        self.det_head = nn.Conv2d(ctx_ch, 8, 1)

    def forward(self, image: torch.Tensor, cam_params: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(image)  # (B, feat_ch, h, w)
        height_logits, context = self.height_net(feat, cam_params)
        height_prob = height_logits.softmax(dim=1)  # (B, n_height, h, w)
        # LSS-style outer product of height-bin distribution with context, then
        # collapse the height axis (compact stand-in for voxel-pool splat onto BEV).
        lifted = torch.einsum(
            "bnhw,bchw->bchw",
            height_prob.mean(dim=1, keepdim=True).expand(-1, context.shape[1], -1, -1),
            context,
        )
        bev = lifted + torch.einsum("bnhw,bchw->bcnhw", height_prob, context).sum(dim=2)
        return self.det_head(bev)


def build_bevheight() -> nn.Module:
    """Build a compact BEVHeight (camera-aware height-distribution view transform)."""
    return BEVHeight(feat_ch=32, mid_ch=32, ctx_ch=16, n_height=10).eval()


def example_input_bevheight() -> tuple[torch.Tensor, torch.Tensor]:
    """(image (1, 3, 64, 64), camera params (1, 12): flattened intrinsics+extrinsics)."""
    return torch.randn(1, 3, 64, 64), torch.randn(1, 12)


# ============================================================
# BEVStereo -- mono depth net + multi-frame homography-warped stereo cost volume
# ============================================================


class _DepthNet(nn.Module):
    """Camera-aware mono depth/context net (as in BEVDepth/BEVStereo)."""

    def __init__(
        self, feat_ch: int, mid_ch: int, ctx_ch: int, n_depth: int, cam_param_dim: int = 12
    ) -> None:
        super().__init__()
        self.reduce = _cbr(feat_ch, mid_ch, k=1, padding=0)
        self.cam_mlp = nn.Sequential(
            nn.Linear(cam_param_dim, mid_ch), nn.ReLU(inplace=True), nn.Linear(mid_ch, mid_ch)
        )
        self.depth_se = _SELayer(mid_ch, mid_ch)
        self.depth_head = nn.Conv2d(mid_ch, n_depth, 1)
        self.context_head = nn.Conv2d(mid_ch, ctx_ch, 1)

    def forward(
        self, feat: torch.Tensor, cam_params: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.reduce(feat)
        ctx = self.cam_mlp(cam_params)
        depth_feat = self.depth_se(x, ctx)
        depth_logits = self.depth_head(depth_feat)
        context = self.context_head(x)
        return depth_logits, context


class BEVStereo(nn.Module):
    """BEVStereo: mono depth net refined by a multi-frame homography-warped stereo cost volume."""

    def __init__(
        self,
        feat_ch: int = 32,
        mid_ch: int = 32,
        ctx_ch: int = 16,
        n_depth: int = 8,
        n_frames: int = 2,
    ) -> None:
        super().__init__()
        self.n_frames = n_frames
        self.n_depth = n_depth
        self.backbone = _ImageBackbone(3, feat_ch)
        self.depth_net = _DepthNet(feat_ch, mid_ch, ctx_ch, n_depth)
        # Stereo-feature projector for the cost-volume correlation branch.
        self.stereo_proj = nn.Conv2d(feat_ch, mid_ch, 1)
        # 3D-conv-style cost-volume aggregation (kept 2D-per-depth-slice compact).
        self.cost_agg = nn.Sequential(
            nn.Conv2d(n_depth, n_depth, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_depth, n_depth, 3, padding=1),
        )
        self.det_head = nn.Conv2d(ctx_ch, 8, 1)

    def forward(self, images: torch.Tensor, cam_params: torch.Tensor) -> torch.Tensor:
        # images: (B, n_frames, 3, H, W) -- frame 0 is the reference (key) frame.
        b, t, c, h, w = images.shape
        flat = images.reshape(b * t, c, h, w)
        feat = self.backbone(flat)
        feat = feat.reshape(b, t, *feat.shape[1:])
        ref_feat = feat[:, 0]
        depth_logits, context = self.depth_net(ref_feat, cam_params)

        # Build a per-depth-hypothesis correlation cost volume between the reference
        # frame's stereo features and each other (sweep) frame's stereo features,
        # using a learned per-depth-index affine warp offset as a compact stand-in
        # for the true homography warp (which requires camera geometry at runtime).
        ref_stereo = self.stereo_proj(ref_feat)  # (B, mid_ch, h', w')
        cost_slices = []
        for d in range(self.n_depth):
            shift = d - self.n_depth // 2
            slice_costs = []
            for sweep in range(1, t):
                sweep_stereo = self.stereo_proj(feat[:, sweep])
                warped = torch.roll(sweep_stereo, shifts=(0, shift), dims=(2, 3))
                slice_costs.append((ref_stereo * warped).sum(dim=1, keepdim=True))
            cost_slices.append(
                torch.stack(slice_costs, dim=0).mean(dim=0)
                if slice_costs
                else torch.zeros_like(ref_stereo[:, :1])
            )
        cost_volume = torch.cat(cost_slices, dim=1)  # (B, n_depth, h', w')
        stereo_refine = self.cost_agg(cost_volume)

        refined_depth_logits = depth_logits + stereo_refine
        depth_prob = refined_depth_logits.softmax(dim=1)
        lifted = torch.einsum(
            "bdhw,bchw->bchw",
            depth_prob.mean(dim=1, keepdim=True).expand(-1, context.shape[1], -1, -1),
            context,
        )
        bev = lifted + torch.einsum("bdhw,bchw->bcdhw", depth_prob, context).sum(dim=2)
        return self.det_head(bev)


def build_bevstereo() -> nn.Module:
    """Build a compact BEVStereo (mono depth net + temporal stereo cost volume)."""
    return BEVStereo(feat_ch=32, mid_ch=32, ctx_ch=16, n_depth=8, n_frames=2).eval()


def example_input_bevstereo() -> tuple[torch.Tensor, torch.Tensor]:
    """(images (1, 2 frames, 3, 64, 64), camera params (1, 12))."""
    return torch.randn(1, 2, 3, 64, 64), torch.randn(1, 12)


# ============================================================
# BtcDet -- raw + occlusion-completion (shape) voxel branches fused before 3D detection
# ============================================================


class BtcDet(nn.Module):
    """BtcDet: auxiliary dense-occupancy (shape-completion) branch fused with raw voxels."""

    def __init__(self, n_raw_feat: int = 4, voxel_ch: int = 16, grid: int = 8) -> None:
        super().__init__()
        self.grid = grid
        # Raw-voxel encoder (pools observed LiDAR points per voxel).
        self.raw_vfe = nn.Sequential(
            nn.Linear(n_raw_feat, voxel_ch), nn.ReLU(inplace=True), nn.Linear(voxel_ch, voxel_ch)
        )
        # Occlusion-completion "behind the curtain" shape head: predicts DENSE voxel
        # occupancy behind observed surfaces from the sparse raw-voxel grid.
        self.shape_head = nn.Sequential(
            nn.Conv3d(voxel_ch, voxel_ch, 3, padding=1),
            nn.BatchNorm3d(voxel_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(voxel_ch, 1, 3, padding=1),
        )
        # Occupancy-conditioned shape-feature branch (fused with raw features).
        self.occ_feat = nn.Sequential(nn.Conv3d(1, voxel_ch, 3, padding=1), nn.ReLU(inplace=True))
        self.fuse = nn.Sequential(
            nn.Conv3d(voxel_ch * 2, voxel_ch, 3, padding=1), nn.ReLU(inplace=True)
        )
        # Shared 3D sparse-style backbone (kept dense/compact) + detection head.
        self.backbone3d = nn.Sequential(
            nn.Conv3d(voxel_ch, voxel_ch, 3, padding=1, stride=2),
            nn.BatchNorm3d(voxel_ch),
            nn.ReLU(inplace=True),
        )
        self.det_head = nn.Conv3d(voxel_ch, 8, 1)

    def forward(self, voxel_points: torch.Tensor, voxel_coords: torch.Tensor) -> torch.Tensor:
        # voxel_points: (N_voxels, max_pts, n_raw_feat); voxel_coords: (N_voxels, 3) int in [0, grid).
        raw_voxel_feat = self.raw_vfe(voxel_points).mean(dim=1)  # (N_voxels, voxel_ch)
        b = 1
        grid_feat = raw_voxel_feat.new_zeros(
            b, raw_voxel_feat.shape[-1], self.grid, self.grid, self.grid
        )
        idx = voxel_coords.long()
        grid_feat[0, :, idx[:, 0], idx[:, 1], idx[:, 2]] = raw_voxel_feat.transpose(0, 1)

        occ_logits = self.shape_head(
            grid_feat
        )  # (1, 1, G, G, G) dense occupancy behind observed surfaces
        occ_prob = torch.sigmoid(occ_logits)
        occ_feat = self.occ_feat(occ_prob)

        fused = self.fuse(torch.cat([grid_feat, occ_feat], dim=1))
        backbone_feat = self.backbone3d(fused)
        return self.det_head(backbone_feat)


def build_btcdet() -> nn.Module:
    """Build a compact BtcDet (occlusion-completion voxel branch fused with raw voxels)."""
    return BtcDet(n_raw_feat=4, voxel_ch=16, grid=8).eval()


def example_input_btcdet() -> tuple[torch.Tensor, torch.Tensor]:
    """(voxel_points (24 voxels, 5 pts, 4 feats), voxel_coords (24, 3) grid indices in [0, 8))."""
    torch.manual_seed(0)
    voxel_points = torch.randn(24, 5, 4)
    voxel_coords = torch.randint(0, 8, (24, 3))
    return voxel_points, voxel_coords


# ============================================================
# Cam4DOcc / OCFNet -- multi-camera BEV encoder + temporal fusion + multi-step
# FUTURE occupancy/flow forecasting decoder
# ============================================================


class OCFNet(nn.Module):
    """OCFNet (Cam4DOcc baseline): BEV encoder + temporal fusion -> future occupancy+flow."""

    def __init__(
        self,
        n_cam: int = 3,
        n_hist: int = 2,
        n_future: int = 3,
        feat_ch: int = 32,
        n_depth: int = 8,
        bev_ch: int = 24,
    ) -> None:
        super().__init__()
        self.n_cam, self.n_hist, self.n_future, self.bev_ch = n_cam, n_hist, n_future, bev_ch
        self.backbone = _ImageBackbone(3, feat_ch)
        self.lift_splat = _LiftSplat(feat_ch, n_depth, bev_ch)
        # Temporal history fusion (GRU-style single-step recurrent update over the
        # short BEV history, standing in for the warp-and-fuse temporal module).
        self.history_gru = nn.GRUCell(bev_ch, bev_ch)
        # One decoder head per future forecast horizon, sharing the fused history
        # feature -- the multi-step forecasting distinctive of Cam4DOcc/OCFNet.
        self.future_heads = nn.ModuleList(
            [nn.Conv2d(bev_ch, 3, 1) for _ in range(n_future)]
        )  # (occ_logit, flow_x, flow_y)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: (B, n_hist, n_cam, 3, H, W)
        b, t, n, c, h, w = images.shape
        flat = images.reshape(b * t * n, c, h, w)
        feat = self.backbone(flat)
        bev_per_cam = self.lift_splat(feat)
        bev_per_cam = bev_per_cam.reshape(b, t, n, self.bev_ch, *bev_per_cam.shape[2:])
        bev_per_frame = bev_per_cam.sum(dim=2)  # fuse cameras -> (B, t, bev_ch, h', w')

        gh, gw = bev_per_frame.shape[-2:]
        hidden = bev_per_frame.new_zeros(b * gh * gw, self.bev_ch)
        for step in range(self.n_hist):
            cur = bev_per_frame[:, step].permute(0, 2, 3, 1).reshape(b * gh * gw, self.bev_ch)
            hidden = self.history_gru(cur, hidden)
        fused_history = hidden.reshape(b, gh, gw, self.bev_ch).permute(
            0, 3, 1, 2
        )  # (B, bev_ch, h', w')

        future_preds = [head(fused_history) for head in self.future_heads]
        return torch.stack(future_preds, dim=1)  # (B, n_future, 3, h', w')


def build_cam4docc() -> nn.Module:
    """Build a compact Cam4DOcc / OCFNet (camera-only 4D occupancy forecasting)."""
    return OCFNet(n_cam=3, n_hist=2, n_future=3, feat_ch=32, n_depth=8, bev_ch=24).eval()


def example_input_cam4docc() -> torch.Tensor:
    """(1, 2 history frames, 3 cameras, 3, 64, 64) multi-camera history stack."""
    return torch.randn(1, 2, 3, 3, 64, 64)


# ============================================================
# Registry
# ============================================================

MENAGERIE_ENTRIES = [
    ("BEVerse", "build_beverse", "example_input_beverse", "2022", "VIS"),
    ("BEVGuide", "build_bevguide", "example_input_bevguide", "2022", "VIS"),
    ("BEVHeight", "build_bevheight", "example_input_bevheight", "2023", "VIS"),
    ("BEVStereo", "build_bevstereo", "example_input_bevstereo", "2022", "VIS"),
    ("BtcDet", "build_btcdet", "example_input_btcdet", "2022", "VIS"),
    ("Cam4DOcc", "build_cam4docc", "example_input_cam4docc", "2024", "VIS"),
]
