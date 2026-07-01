"""Menagerie batch w3a22: sparse-query and temporal-fusion architectures for
camera/LiDAR 3D perception and end-to-end autonomous driving.

Sources checked (reference only; no cloning, no pip installs):
  - SOLOFusion ("Time Will Tell: New Outlooks and A Baseline for Temporal
    Multi-View 3D Object Detection"). Park et al., ICCV 2023. Paper
    https://arxiv.org/abs/2210.15538 (aka arXiv:2210.02443), official source
    https://github.com/Divadi/SOLOFusion. Reformulates multi-view camera 3D
    detection as TEMPORAL STEREO matching that fuses SHORT-TERM, high-
    resolution (fine-grained, small depth-bin stride, few recent frames) and
    LONG-TERM, low-resolution (coarse depth-bin stride, many historical
    frames) feature matching, exploiting the time/resolution trade-off: long
    history at coarse granularity is cheap and long-term-informative, short
    history at fine granularity gives precise nearby matching, and the two
    are highly complementary. DISTINCTIVE: TWO temporal-stereo branches
    (short high-res, long low-res) with per-branch cost volumes over
    warped-history features, concatenated before the detection head --
    contrasted with single-resolution / single-window temporal fusion.
  - Sparse4D. Lin et al., 2022 (v1); Sparse4D v2/v3 extend it (2023-2024).
    Paper https://arxiv.org/abs/2211.10581, official source
    https://github.com/HorizonRobotics/Sparse4D. Sparse instance-centric
    queries: each 3D anchor box is assigned multiple learned 4D keypoints
    (spatial offsets around the anchor center plus a temporal index), which
    are projected into multi-view/multi-scale/multi-timestamp image feature
    maps to SAMPLE corresponding features directly (no dense BEV/view
    transform, no global cross-attention). Sampled features are hierarchically
    fused across camera views, timestamps, and keypoints (weighted sum with
    learned attention weights) into one instance feature per query, which
    refines the anchor box iteratively across decoder layers.
    DISTINCTIVE: 4D (space+time) KEYPOINT SAMPLING directly from raw image
    feature maps per query -- no dense/global intermediate representation.
  - SparseDrive: End-to-End Autonomous Driving via Sparse Scene
    Representation. Sun et al., 2024. Paper
    https://arxiv.org/abs/2405.19620, official source
    https://github.com/swc-17/SparseDrive. Three stages: (1) an image
    encoder; (2) a SYMMETRIC sparse-perception module where TWO parallel
    query groups -- agent instances (decoupled into an instance feature
    vector + a geometric anchor box, refined by deformable/sparse
    cross-attention against image features, as in Sparse4D) and map-element
    instances (same decoupled representation, refined against the same image
    features) -- learn a fully sparse scene representation with NO shared
    BEV grid; (3) a parallel motion planner where an ego instance
    cross-attends the agent + map instances and predicts multi-modal
    trajectories for both surrounding agents (motion) and the ego vehicle
    (planning) with the SAME decoder, then a hierarchical/collision-aware
    rescore module selects the final trajectory. DISTINCTIVE: symmetric
    twin sparse-query perception (agents + map, same decoupled
    feature+anchor design) directly followed by a SHARED motion/planning
    decoder over one more ego query -- unifying detection, mapping, motion
    prediction and planning as one query-refinement pipeline.
  - SparseFusion: Fusing Multi-Modal Sparse Representations for Multi-Sensor
    3D Object Detection. Xie et al., ICCV 2023. Paper
    https://arxiv.org/abs/2304.14340, official source
    https://github.com/yichen928/SparseFusion. Parallel, MODALITY-SPECIFIC
    sparse detectors (one camera-based, one LiDAR-based) each produce a small
    set of sparse object CANDIDATES (query embedding + 3D box) independently;
    camera candidates are lifted into LiDAR/3D coordinates via a learned
    geometric transform, and semantic/geometric CROSS-MODALITY TRANSFER
    modules exchange information between the two candidate sets (each
    modality's candidates attend to the other modality's raw sparse
    features) before a final lightweight self-attention fusion module merges
    the two candidate sets in one 3D space for the final box prediction.
    DISTINCTIVE: two independent single-modality sparse candidate generators
    + explicit cross-modal candidate-level transfer (not early feature-level
    concat, not dense BEV fusion) + self-attention candidate merge.
  - ST-P3: A Spatial-Temporal feature learning framework for Perception,
    Prediction, and Planning. Hu et al., ECCV 2022. Paper
    https://arxiv.org/abs/2207.07601, official source
    https://github.com/OpenDriveLab/ST-P3. A shared multi-camera BEV feature
    (LSS-style depth-distribution lift) is built per past timestep, then an
    EGOCENTRIC-ALIGNED ACCUMULATION module warps and accumulates each past
    BEV feature into the CURRENT ego frame using known ego-motion (rather
    than fusing post-hoc at a coarse BEV resolution, alignment happens
    progressively per timestep to preserve 3D geometry before/while pooling
    into BEV), producing one spatio-temporal BEV feature consumed by three
    heads: perception (segmentation), a recurrent future-prediction head that
    rolls the BEV feature forward with predicted ego-motion, and a planning
    head that decodes a trajectory from the predicted future BEV features
    refined by a temporal refinement unit. DISTINCTIVE: per-timestep
    EGO-MOTION-WARPED accumulation of BEV features (not a single post-hoc
    temporal fusion block) feeding one shared feature for perception +
    recurrent future prediction + trajectory planning.

All models below are compact, faithfully-reimplemented-from-scratch nn.Modules with
random init and small dims for TorchLens architecture-catalog tracing (not a
trained-weights zoo).
"""

from __future__ import annotations

import torch
import torch.nn as nn

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


# ============================================================
# SOLOFusion -- short-term high-res + long-term low-res temporal-stereo fusion
# ============================================================


class _TemporalStereoBranch(nn.Module):
    """One temporal-stereo branch: correlate reference features against a warped
    history stack at a given depth-hypothesis stride, producing a per-pixel cost
    volume that stands in for the branch's depth-bin matching resolution."""

    def __init__(self, feat_ch: int, n_bins: int, mid_ch: int = 16) -> None:
        super().__init__()
        self.n_bins = n_bins
        self.proj = nn.Conv2d(feat_ch, mid_ch, 1)
        self.cost_agg = nn.Sequential(
            nn.Conv2d(n_bins, n_bins, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_bins, n_bins, 3, padding=1),
        )

    def forward(self, ref_feat: torch.Tensor, history_feats: torch.Tensor) -> torch.Tensor:
        # ref_feat: (B, feat_ch, h, w); history_feats: (B, T_hist, feat_ch, h, w)
        ref_proj = self.proj(ref_feat)
        t_hist = history_feats.shape[1]
        cost_slices = []
        for d in range(self.n_bins):
            shift = d - self.n_bins // 2
            per_frame = []
            for t in range(t_hist):
                warped = torch.roll(self.proj(history_feats[:, t]), shifts=shift, dims=-1)
                per_frame.append((ref_proj * warped).sum(dim=1, keepdim=True))
            cost_slices.append(torch.stack(per_frame, dim=0).mean(dim=0))
        cost_volume = torch.cat(cost_slices, dim=1)  # (B, n_bins, h, w)
        return self.cost_agg(cost_volume)


class SOLOFusion(nn.Module):
    """SOLOFusion: short-term high-res + long-term low-res temporal-stereo fusion."""

    def __init__(
        self,
        feat_ch: int = 32,
        short_bins: int = 12,
        long_bins: int = 4,
        n_short_hist: int = 2,
        n_long_hist: int = 4,
    ) -> None:
        super().__init__()
        self.n_short_hist = n_short_hist
        self.n_long_hist = n_long_hist
        self.backbone = _ImageBackbone(3, feat_ch)
        # Short-term branch: few recent frames, fine (many) depth bins.
        self.short_branch = _TemporalStereoBranch(feat_ch, short_bins)
        # Long-term branch: many historical frames, coarse (few) depth bins.
        self.long_branch = _TemporalStereoBranch(feat_ch, long_bins)
        self.det_head = nn.Conv2d(short_bins + long_bins, 8, 1)

    def forward(
        self, ref_image: torch.Tensor, short_hist: torch.Tensor, long_hist: torch.Tensor
    ) -> torch.Tensor:
        # ref_image: (B, 3, H, W); short_hist: (B, n_short_hist, 3, H, W);
        # long_hist: (B, n_long_hist, 3, H, W)
        b = ref_image.shape[0]
        ref_feat = self.backbone(ref_image)

        short_flat = short_hist.reshape(b * self.n_short_hist, *short_hist.shape[2:])
        short_feats = self.backbone(short_flat).reshape(b, self.n_short_hist, *ref_feat.shape[1:])
        short_cost = self.short_branch(ref_feat, short_feats)

        long_flat = long_hist.reshape(b * self.n_long_hist, *long_hist.shape[2:])
        long_feats = self.backbone(long_flat).reshape(b, self.n_long_hist, *ref_feat.shape[1:])
        long_cost = self.long_branch(ref_feat, long_feats)

        fused = torch.cat([short_cost, long_cost], dim=1)
        return self.det_head(fused)


def build_solofusion() -> nn.Module:
    """Build a compact SOLOFusion (short-term + long-term temporal-stereo fusion)."""
    return SOLOFusion(feat_ch=32, short_bins=12, long_bins=4, n_short_hist=2, n_long_hist=4).eval()


def example_input_solofusion() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(ref_image (1, 3, 64, 64), short_hist (1, 2, 3, 64, 64), long_hist (1, 4, 3, 64, 64))."""
    return torch.randn(1, 3, 64, 64), torch.randn(1, 2, 3, 64, 64), torch.randn(1, 4, 3, 64, 64)


# ============================================================
# Sparse4D -- 4D keypoint sampling of raw image features per instance query
# ============================================================


class _DeformableAggregation(nn.Module):
    """Sample multi-view/multi-timestamp image features at each query's learned 4D
    keypoints (bilinear sample at a projected 2D location per view/frame), then
    fuse across (views x frames x keypoints) with learned attention weights."""

    def __init__(
        self, embed_dim: int, feat_ch: int, n_keypoints: int, n_views: int, n_frames: int
    ) -> None:
        super().__init__()
        self.n_keypoints = n_keypoints
        self.n_views = n_views
        self.n_frames = n_frames
        # Learned 2D sample offsets (in [-1, 1] grid_sample coords) per keypoint,
        # view, and frame, conditioned on the query embedding.
        self.offset_head = nn.Linear(embed_dim, n_keypoints * n_views * n_frames * 2)
        self.weight_head = nn.Linear(embed_dim, n_keypoints * n_views * n_frames)
        self.feat_proj = nn.Linear(feat_ch, embed_dim)

    def forward(self, query: torch.Tensor, mv_mf_feats: torch.Tensor) -> torch.Tensor:
        # query: (B, N_q, embed_dim)
        # mv_mf_feats: (B, n_views, n_frames, feat_ch, h, w)
        b, n_q, _ = query.shape
        offsets = self.offset_head(query).reshape(
            b, n_q, self.n_keypoints, self.n_views, self.n_frames, 2
        )
        weights = self.weight_head(query).reshape(
            b, n_q, self.n_keypoints * self.n_views * self.n_frames
        )
        weights = weights.softmax(dim=-1).reshape(
            b, n_q, self.n_keypoints, self.n_views, self.n_frames
        )

        sampled = query.new_zeros(b, n_q, query.shape[-1])
        for v in range(self.n_views):
            for f in range(self.n_frames):
                feat = mv_mf_feats[:, v, f]  # (B, feat_ch, h, w)
                grid = (
                    offsets[:, :, :, v, f, :].reshape(b, n_q * self.n_keypoints, 1, 2).clamp(-1, 1)
                )
                sampled_feat = torch.nn.functional.grid_sample(feat, grid, align_corners=False)
                # (B, feat_ch, n_q*n_keypoints, 1) -> (B, n_q, n_keypoints, feat_ch)
                sampled_feat = (
                    sampled_feat.squeeze(-1).permute(0, 2, 1).reshape(b, n_q, self.n_keypoints, -1)
                )
                sampled_feat = self.feat_proj(sampled_feat)
                w = weights[:, :, :, v, f].unsqueeze(-1)
                sampled = sampled + (sampled_feat * w).sum(dim=2)
        return sampled


class Sparse4D(nn.Module):
    """Sparse4D: sparse instance-centric queries with 4D (space+time) keypoint sampling."""

    def __init__(
        self,
        embed_dim: int = 64,
        n_queries: int = 12,
        n_keypoints: int = 4,
        n_views: int = 2,
        n_frames: int = 2,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        self.n_views = n_views
        self.n_frames = n_frames
        self.backbone = _ImageBackbone(3, embed_dim)
        self.query_embed = nn.Parameter(torch.randn(1, n_queries, embed_dim) * 0.02)
        self.anchor_head = nn.Linear(embed_dim, 7)  # (x, y, z, w, l, h, yaw)
        self.layers = nn.ModuleList(
            [
                _DeformableAggregation(embed_dim, embed_dim, n_keypoints, n_views, n_frames)
                for _ in range(n_layers)
            ]
        )
        self.refine_heads = nn.ModuleList([nn.Linear(embed_dim, 7) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: (B, n_views, n_frames, 3, H, W)
        b, v, f, c, h, w = images.shape
        flat = images.reshape(b * v * f, c, h, w)
        feats = self.backbone(flat).reshape(b, v, f, -1, h // 8, w // 8)

        query = self.query_embed.expand(b, -1, -1)
        anchors = self.anchor_head(query)
        for layer, refine_head in zip(self.layers, self.refine_heads):
            sampled = layer(query, feats)
            query = self.norm(query + sampled)
            anchors = anchors + refine_head(query)
        return anchors  # (B, n_queries, 7) refined 3D boxes


def build_sparse4d() -> nn.Module:
    """Build a compact Sparse4D (4D keypoint sampling of raw image features)."""
    return Sparse4D(
        embed_dim=64, n_queries=12, n_keypoints=4, n_views=2, n_frames=2, n_layers=2
    ).eval()


def example_input_sparse4d() -> torch.Tensor:
    """(1, 2 views, 2 frames, 3, 64, 64) multi-camera multi-frame image stack."""
    return torch.randn(1, 2, 2, 3, 64, 64)


# ============================================================
# SparseDrive -- symmetric sparse perception (agent + map queries) + shared
# motion/planning decoder over one more ego query
# ============================================================


class _SparseInstanceRefiner(nn.Module):
    """Refine a group of (feature, anchor) instance pairs via cross-attention
    against image feature tokens, then update the geometric anchor from the
    refined feature -- the decoupled feature+anchor representation shared by
    the agent and map query groups."""

    def __init__(self, embed_dim: int, anchor_dim: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, 4, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.anchor_proj = nn.Linear(anchor_dim, embed_dim)
        self.anchor_refine = nn.Linear(embed_dim, anchor_dim)

    def forward(
        self, feat: torch.Tensor, anchor: torch.Tensor, img_tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = feat + self.anchor_proj(anchor)
        attn_out, _ = self.attn(q, img_tokens, img_tokens)
        feat = self.norm(feat + attn_out)
        anchor = anchor + self.anchor_refine(feat)
        return feat, anchor


class SparseDrive(nn.Module):
    """SparseDrive: symmetric sparse agent+map perception feeding a shared ego
    motion/planning decoder -- unified detection, mapping, prediction, planning."""

    def __init__(
        self,
        embed_dim: int = 32,
        n_agents: int = 6,
        n_map: int = 6,
        n_modes: int = 3,
        n_future: int = 4,
    ) -> None:
        super().__init__()
        self.backbone = _ImageBackbone(3, embed_dim)
        self.agent_feat = nn.Parameter(torch.randn(1, n_agents, embed_dim) * 0.02)
        self.agent_anchor = nn.Parameter(torch.randn(1, n_agents, 7) * 0.02)
        self.map_feat = nn.Parameter(torch.randn(1, n_map, embed_dim) * 0.02)
        self.map_anchor = nn.Parameter(torch.randn(1, n_map, 4) * 0.02)
        self.agent_refiner = _SparseInstanceRefiner(embed_dim, 7)
        self.map_refiner = _SparseInstanceRefiner(embed_dim, 4)

        # Ego instance initialization + parallel motion/planning decoder: the ego
        # query cross-attends both agent and map instances, then a shared head
        # predicts multi-modal trajectories for agents (motion) and ego (planning).
        self.ego_feat = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.ego_attn = nn.MultiheadAttention(embed_dim, 4, batch_first=True)
        self.n_modes = n_modes
        self.n_future = n_future
        self.motion_head = nn.Linear(embed_dim, n_modes * n_future * 2)
        self.plan_head = nn.Linear(embed_dim, n_modes * n_future * 2)
        self.rescore_head = nn.Linear(embed_dim, n_modes)  # collision-aware mode rescoring

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        b = image.shape[0]
        feat = self.backbone(image)
        img_tokens = feat.flatten(2).transpose(1, 2)  # (B, h*w, embed_dim)

        agent_feat, agent_anchor = self.agent_refiner(
            self.agent_feat.expand(b, -1, -1), self.agent_anchor.expand(b, -1, -1), img_tokens
        )
        map_feat, map_anchor = self.map_refiner(
            self.map_feat.expand(b, -1, -1), self.map_anchor.expand(b, -1, -1), img_tokens
        )

        scene_tokens = torch.cat([agent_feat, map_feat], dim=1)
        ego_query = self.ego_feat.expand(b, -1, -1)
        ego_out, _ = self.ego_attn(ego_query, scene_tokens, scene_tokens)

        motion = self.motion_head(agent_feat).reshape(
            b, agent_feat.shape[1], self.n_modes, self.n_future, 2
        )
        plan = self.plan_head(ego_out).reshape(b, self.n_modes, self.n_future, 2)
        rescore = self.rescore_head(ego_out).reshape(b, self.n_modes)
        return {
            "agent_boxes": agent_anchor,
            "map_elements": map_anchor,
            "motion": motion,
            "plan_trajectories": plan,
            "plan_rescore": rescore,
        }


def build_sparsedrive() -> nn.Module:
    """Build a compact SparseDrive (symmetric sparse perception + shared motion/planning)."""
    return SparseDrive(embed_dim=32, n_agents=6, n_map=6, n_modes=3, n_future=4).eval()


def example_input_sparsedrive() -> torch.Tensor:
    """(1, 3, 64, 64) single-camera front-view image."""
    return torch.randn(1, 3, 64, 64)


# ============================================================
# SparseFusion -- parallel modality-specific sparse candidates + cross-modal
# semantic/geometric transfer + self-attention candidate merge
# ============================================================


class _ModalityCandidateHead(nn.Module):
    """Produce a small set of sparse object candidates (embedding + 3D box) from
    one modality's raw sparse feature tokens."""

    def __init__(self, in_dim: int, embed_dim: int, n_candidates: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, n_candidates, embed_dim) * 0.02)
        self.token_proj = nn.Linear(in_dim, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, 4, batch_first=True)
        self.box_head = nn.Linear(embed_dim, 7)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b = tokens.shape[0]
        proj = self.token_proj(tokens)
        q = self.query.expand(b, -1, -1)
        cand_feat, _ = self.attn(q, proj, proj)
        cand_box = self.box_head(cand_feat)
        return cand_feat, cand_box


class _CrossModalTransfer(nn.Module):
    """One modality's candidates cross-attend the OTHER modality's raw tokens
    (semantic + geometric transfer) before final fusion."""

    def __init__(self, embed_dim: int, other_dim: int) -> None:
        super().__init__()
        self.other_proj = nn.Linear(other_dim, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, 4, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, cand_feat: torch.Tensor, other_tokens: torch.Tensor) -> torch.Tensor:
        other_proj = self.other_proj(other_tokens)
        attn_out, _ = self.attn(cand_feat, other_proj, other_proj)
        return self.norm(cand_feat + attn_out)


class SparseFusion(nn.Module):
    """SparseFusion: parallel modality-specific sparse candidates + cross-modal
    transfer + self-attention candidate merge (camera + LiDAR 3D detection)."""

    def __init__(self, embed_dim: int = 32, n_cam_cand: int = 6, n_lidar_cand: int = 6) -> None:
        super().__init__()
        self.cam_backbone = _ImageBackbone(3, embed_dim)
        self.cam_head = _ModalityCandidateHead(embed_dim, embed_dim, n_cam_cand)
        self.lidar_point_enc = nn.Sequential(
            nn.Linear(4, embed_dim), nn.ReLU(inplace=True), nn.Linear(embed_dim, embed_dim)
        )
        self.lidar_head = _ModalityCandidateHead(embed_dim, embed_dim, n_lidar_cand)

        self.cam_to_lidar_transfer = _CrossModalTransfer(embed_dim, embed_dim)
        self.lidar_to_cam_transfer = _CrossModalTransfer(embed_dim, embed_dim)

        self.merge_attn = nn.MultiheadAttention(embed_dim, 4, batch_first=True)
        self.merge_norm = nn.LayerNorm(embed_dim)
        self.final_box_head = nn.Linear(embed_dim, 7)

    def forward(self, image: torch.Tensor, lidar_points: torch.Tensor) -> torch.Tensor:
        cam_feat = self.cam_backbone(image)
        cam_tokens = cam_feat.flatten(2).transpose(1, 2)
        cam_cand_feat, _ = self.cam_head(cam_tokens)

        lidar_tokens = self.lidar_point_enc(lidar_points)
        lidar_cand_feat, _ = self.lidar_head(lidar_tokens)

        # Semantic/geometric cross-modality transfer before merge.
        cam_cand_feat = self.cam_to_lidar_transfer(cam_cand_feat, lidar_tokens)
        lidar_cand_feat = self.lidar_to_cam_transfer(lidar_cand_feat, cam_tokens)

        # Lightweight self-attention merge of the two candidate sets in one space.
        merged = torch.cat([cam_cand_feat, lidar_cand_feat], dim=1)
        merged_attn, _ = self.merge_attn(merged, merged, merged)
        merged = self.merge_norm(merged + merged_attn)
        return self.final_box_head(merged)  # (B, n_cam_cand + n_lidar_cand, 7)


def build_sparsefusion() -> nn.Module:
    """Build a compact SparseFusion (parallel sparse candidates + cross-modal transfer)."""
    return SparseFusion(embed_dim=32, n_cam_cand=6, n_lidar_cand=6).eval()


def example_input_sparsefusion() -> tuple[torch.Tensor, torch.Tensor]:
    """(image (1, 3, 64, 64), lidar_points (1, 32, 4): flattened (x, y, z, intensity) tokens)."""
    return torch.randn(1, 3, 64, 64), torch.randn(1, 32, 4)


# ============================================================
# ST-P3 -- egocentric-aligned per-timestep BEV accumulation feeding shared
# perception + recurrent future-prediction + planning heads
# ============================================================


class _LiftSplat(nn.Module):
    """Per-camera LSS view transform: depth-distribution outer-product context -> BEV splat."""

    def __init__(self, feat_ch: int = 32, n_depth: int = 8, bev_ch: int = 24) -> None:
        super().__init__()
        self.n_depth = n_depth
        self.depth_context_head = nn.Conv2d(feat_ch, n_depth + bev_ch, 1)
        self.bev_ch = bev_ch

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        out = self.depth_context_head(feat)
        depth_logits, context = out[:, : self.n_depth], out[:, self.n_depth :]
        depth_prob = depth_logits.softmax(dim=1)
        lifted = torch.einsum("bdhw,bchw->bcdhw", depth_prob, context)
        return lifted.sum(dim=2)  # (B, bev_ch, h, w)


class _EgoAlignedAccumulator(nn.Module):
    """Warp the running accumulated BEV feature into the current ego frame (via a
    predicted per-step affine flow field) before fusing in the newest per-timestep
    BEV feature -- progressive ego-motion-aligned accumulation, not a single
    post-hoc temporal-fusion block."""

    def __init__(self, bev_ch: int, motion_dim: int = 3) -> None:
        super().__init__()
        self.flow_head = nn.Linear(motion_dim, 2)  # predicted (dx, dy) grid shift from ego-motion
        self.fuse = _cbr(bev_ch * 2, bev_ch)

    def forward(
        self, accumulated: torch.Tensor, new_bev: torch.Tensor, ego_motion: torch.Tensor
    ) -> torch.Tensor:
        b, c, h, w = accumulated.shape
        flow = self.flow_head(ego_motion)  # (B, 2)
        base_grid = torch.stack(
            torch.meshgrid(
                torch.linspace(-1, 1, h, device=accumulated.device),
                torch.linspace(-1, 1, w, device=accumulated.device),
                indexing="ij",
            ),
            dim=-1,
        ).unsqueeze(0)
        shift = flow.reshape(b, 1, 1, 2) / max(h, w)
        warp_grid = (base_grid + shift).clamp(-1, 1)
        warped = torch.nn.functional.grid_sample(accumulated, warp_grid, align_corners=False)
        return self.fuse(torch.cat([warped, new_bev], dim=1))


class STP3(nn.Module):
    """ST-P3: egocentric-aligned per-timestep BEV accumulation -> shared perception
    + recurrent future prediction + interpretable planning."""

    def __init__(
        self,
        feat_ch: int = 32,
        n_depth: int = 6,
        bev_ch: int = 24,
        n_hist: int = 3,
        n_future: int = 2,
    ) -> None:
        super().__init__()
        self.n_hist = n_hist
        self.n_future = n_future
        self.backbone = _ImageBackbone(3, feat_ch)
        self.lift_splat = _LiftSplat(feat_ch, n_depth, bev_ch)
        self.accumulator = _EgoAlignedAccumulator(bev_ch)

        self.seg_head = nn.Conv2d(bev_ch, 4, 1)  # perception: BEV segmentation
        # Recurrent future-prediction: roll the accumulated BEV feature forward
        # with the same ego-aligned accumulator, conditioned on a predicted
        # future ego-motion at each step.
        self.future_motion_head = nn.Linear(bev_ch, 3)
        self.future_bev_refine = _cbr(bev_ch, bev_ch)

        # Temporal refinement unit + interpretable trajectory planning head.
        self.temporal_refine = nn.GRUCell(bev_ch, bev_ch)
        self.plan_head = nn.Linear(bev_ch, n_future * 2)

    def forward(self, images: torch.Tensor, ego_motions: torch.Tensor) -> dict[str, torch.Tensor]:
        # images: (B, n_hist, 3, H, W); ego_motions: (B, n_hist, 3) per-step ego-motion
        b, t, c, h, w = images.shape
        flat = images.reshape(b * t, c, h, w)
        feats = self.backbone(flat)
        bev_frames = self.lift_splat(feats).reshape(b, t, -1, feats.shape[-2], feats.shape[-1])

        accumulated = bev_frames[:, 0]
        for step in range(1, t):
            accumulated = self.accumulator(accumulated, bev_frames[:, step], ego_motions[:, step])

        seg = self.seg_head(accumulated)

        # Recurrent future rollout, reusing the ego-aligned accumulator with a
        # self-predicted future ego-motion at each step.
        future_bevs = []
        cur = accumulated
        for _ in range(self.n_future):
            pooled = cur.mean(dim=(-2, -1))
            future_motion = self.future_motion_head(pooled)
            refined = self.future_bev_refine(cur)
            cur = self.accumulator(cur, refined, future_motion)
            future_bevs.append(cur)
        future_stack = torch.stack(future_bevs, dim=1)  # (B, n_future, bev_ch, h, w)

        # Temporal refinement unit + trajectory planning from the pooled rollout.
        gh, gw = accumulated.shape[-2:]
        hidden = accumulated.mean(dim=(-2, -1))
        for step in range(self.n_future):
            step_pooled = future_stack[:, step].mean(dim=(-2, -1))
            hidden = self.temporal_refine(step_pooled, hidden)
        trajectory = self.plan_head(hidden).reshape(b, self.n_future, 2)

        return {"segmentation": seg, "future_bev": future_stack, "trajectory": trajectory}


def build_stp3() -> nn.Module:
    """Build a compact ST-P3 (egocentric-aligned accumulation + perception/prediction/planning)."""
    return STP3(feat_ch=32, n_depth=6, bev_ch=24, n_hist=3, n_future=2).eval()


def example_input_stp3() -> tuple[torch.Tensor, torch.Tensor]:
    """(images (1, 3 hist frames, 3, 64, 64), ego_motions (1, 3, 3): per-step (dx, dy, dtheta))."""
    return torch.randn(1, 3, 3, 64, 64), torch.randn(1, 3, 3)


# ============================================================
# Registry
# ============================================================

MENAGERIE_ENTRIES = [
    ("SOLOFusion", "build_solofusion", "example_input_solofusion", "2023", "VIS"),
    ("Sparse4D", "build_sparse4d", "example_input_sparse4d", "2022", "VIS"),
    ("SparseDrive", "build_sparsedrive", "example_input_sparsedrive", "2024", "VIS"),
    ("SparseFusion", "build_sparsefusion", "example_input_sparsefusion", "2023", "VIS"),
    ("ST-P3", "build_stp3", "example_input_stp3", "2022", "VIS"),
]
