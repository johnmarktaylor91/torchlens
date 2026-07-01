"""Menagerie batch w3a23: streaming-temporal and topology-reasoning architectures
for camera-based autonomous-driving perception.

Sources checked (reference only; no cloning, no pip installs):
  - StreamMapNet: Yuan, Zhang, Li et al. (StreamMapNet: Streaming Mapping Network
    for Vectorized Online HD Map Construction), WACV 2024. Paper
    https://arxiv.org/abs/2308.12570, official source
    https://github.com/yuantianyuan01/StreamMapNet. Distinctive mechanism: a
    multi-camera BEV feature encoder feeds a set of learnable polyline instance
    queries that cross-attend into the BEV grid and self-attend among themselves
    (a DETR-style polyline decoder producing per-instance ordered point sets).
    Temporal query PROPAGATION carries the top-k highest-confidence queries from
    the previous frame forward: their positions are warped into the current
    ego frame (via a relative pose transform) and the warped queries are
    concatenated with a fresh set of learnable queries before the next
    decoder step, letting persistent map elements (which are static in the
    world) accumulate evidence across frames while still allowing newly
    visible elements to be discovered.
  - StreamPETR: Wang, Lian, Wei et al. (Exploring Object-Centric Temporal
    Modeling for Efficient Multi-View 3D Object Detection), ICCV 2023. Paper
    https://arxiv.org/abs/2303.11926, official source
    https://github.com/exiawsh/StreamPETR. Distinctive mechanism: object
    queries (rather than dense BEV features) are propagated frame-to-frame
    through a small memory queue. A "motion-aware layer norm" (MLN) injects
    ego-motion/timestamp deltas as a FiLM-style scale+shift applied to the
    propagated queries' normalized features, implicitly compensating for ego
    motion before a hybrid self-attention (propagated queries attend to both
    themselves and a fresh set of current-frame queries) and a cross-attention
    into the current multi-view image features.
  - SurroundOcc: Wei, Zhao, Zheng et al. (SurroundOcc: Multi-Camera 3D
    Occupancy Prediction for Autonomous Driving), ICCV 2023. Paper
    https://arxiv.org/abs/2303.09551, official source
    https://github.com/weiyithu/SurroundOcc. Distinctive mechanism: multi-scale
    2D image features are lifted into a 3D voxel volume via spatial cross
    attention (each voxel projects into the multi-camera images and samples
    deformable-attention-style features, approximated here with dense
    cross-attention against the flattened multi-camera tokens), then a stack of
    3D-convolutional upsampling blocks progressively increases voxel
    resolution while a per-level occupancy head is attached at EVERY scale
    (multi-level supervision), producing dense volumetric semantic occupancy.
  - TopoNet: Li, Chen, Wang et al. (Graph-based Topology Reasoning for Driving
    Scenes), arXiv 2304.05277 (Science China Information Sciences 2026).
    Official source https://github.com/OpenDriveLab/TopoNet. Distinctive
    mechanism: centerline queries and traffic-element queries are embedded
    into a shared feature space, then a Scene Graph Neural Network (SGNN)
    passes messages along THREE separate relation types -- centerline-to-
    centerline (lane connectivity), centerline-to-traffic-element (lane-sign
    association), and traffic-element-to-traffic-element -- each with its own
    learned relation-specific message function, followed by a per-relation
    edge-existence head that predicts topology adjacency matrices from
    dot-product query-pair affinities.
  - Ultra-Fast-Lane-Detection-v2 (UFLDv2): Qin, Wang, Wang et al. (Ultra Fast
    Deep Lane Detection With Hybrid Anchor Driven Ordinal Classification),
    TPAMI 2022. Paper https://arxiv.org/abs/2206.07389, official source
    https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2. Distinctive
    mechanism (the extension over UFLD-v1, already in menagerie/classics/
    lanedet.py): a HYBRID anchor system uses ROW anchors (horizontal scan
    lines, good for near-horizontal ego lanes) AND COLUMN anchors (vertical
    scan lines, good for near-vertical side lanes) simultaneously from one
    shared backbone feature map, each with its own ordinal-classification head
    plus an existence-classification head, and the two anchor systems' outputs
    are fused per-lane rather than choosing one representation globally.

All five models below are built from-scratch in base-env torch with small
random-init dimensions; no checkpoints are downloaded.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

# ============================================================
# StreamMapNet -- streaming vectorized HD map with query propagation
# ============================================================


class _CrossAttn(nn.Module):
    """Single-head cross attention: queries attend into a token sequence."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.scale = dim**-0.5

    def forward(self, queries: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        q, k, v = self.q(queries), self.k(tokens), self.v(tokens)
        attn = torch.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1)
        return queries + self.out(attn @ v)


class _SelfAttn(nn.Module):
    """Single-head self attention among a query set."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.attn = _CrossAttn(dim)

    def forward(self, queries: torch.Tensor) -> torch.Tensor:
        return self.attn(queries, queries)


class StreamMapNet(nn.Module):
    """Streaming polyline map decoder with temporal query propagation.

    A BEV feature encoder (patch-embedded multi-camera images averaged into a
    single BEV token grid) feeds a set of learnable polyline instance queries
    through cross-attention (into BEV) and self-attention (among instances),
    predicting per-instance ordered point sets. The top-k highest-confidence
    queries from the PREVIOUS frame are warped by a relative ego-pose offset
    and concatenated with a fresh set of learnable queries before the next
    frame's decode -- persistent map elements accumulate evidence over time.
    """

    def __init__(
        self,
        dim: int = 24,
        n_queries: int = 12,
        n_points: int = 5,
        n_propagate: int = 4,
        bev_size: int = 8,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.n_queries = n_queries
        self.n_points = n_points
        self.n_propagate = n_propagate
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=8, stride=8)
        self.fresh_queries = nn.Parameter(torch.randn(1, n_queries, dim) * 0.02)
        self.pos_embed = nn.Linear(2, dim)  # warped reference-point positional embed
        self.cross_attn = _CrossAttn(dim)
        self.self_attn = _SelfAttn(dim)
        self.point_head = nn.Linear(dim, n_points * 2)  # (x, y) per polyline point
        self.score_head = nn.Linear(dim, 1)
        self.bev_size = bev_size

    def forward(
        self, images: torch.Tensor, prev_queries: torch.Tensor, ego_delta: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """images: (n_cams,3,H,W); prev_queries: (1,n_propagate,dim) from last
        frame; ego_delta: (1,2) relative ego translation this frame."""
        feat = self.patch_embed(images)
        bev_tokens = feat.mean(dim=0, keepdim=True).flatten(2).transpose(1, 2)

        # temporal query propagation: warp previous top-k queries by ego motion
        warped_pos = ego_delta.unsqueeze(1).expand(-1, self.n_propagate, -1)
        propagated = prev_queries + self.pos_embed(warped_pos)

        queries = torch.cat([propagated, self.fresh_queries.expand(1, -1, -1)], dim=1)
        queries = self.cross_attn(queries, bev_tokens)
        queries = self.self_attn(queries)

        points = self.point_head(queries).view(1, -1, self.n_points, 2)
        scores = self.score_head(queries).squeeze(-1)

        # select top-(n_propagate) queries by score to propagate to next frame
        topk_idx = scores.squeeze(0).topk(self.n_propagate).indices
        next_prev_queries = queries[:, topk_idx, :].detach()
        return points, scores, next_prev_queries


def build_streammapnet() -> nn.Module:
    """Small StreamMapNet streaming polyline decoder."""
    return StreamMapNet(dim=24, n_queries=12, n_points=5, n_propagate=4, bev_size=8).eval()


def example_input_streammapnet() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(multi-camera images (4,3,32,32), propagated queries (1,4,24), ego delta (1,2))."""
    return torch.randn(4, 3, 32, 32), torch.randn(1, 4, 24), torch.randn(1, 2)


# ============================================================
# StreamPETR -- object-centric temporal propagation with motion-aware LN
# ============================================================


class _MotionAwareLN(nn.Module):
    """FiLM-style scale+shift on normalized features, conditioned on a motion
    embedding (ego pose delta + timestamp delta), implicitly compensating for
    ego motion between the propagated query's frame and the current frame."""

    def __init__(self, dim: int, motion_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.to_scale_shift = nn.Linear(motion_dim, 2 * dim)

    def forward(self, x: torch.Tensor, motion: torch.Tensor) -> torch.Tensor:
        scale, shift = self.to_scale_shift(motion).chunk(2, dim=-1)
        return self.norm(x) * (1 + scale) + shift


class StreamPETR(nn.Module):
    """Object-centric temporal 3D detector with propagated queries.

    A small memory queue holds object queries from the previous frame. Each
    propagated query passes through motion-aware layer normalization (a
    FiLM-style transform conditioned on ego-motion + time-delta) before hybrid
    self-attention (propagated queries attend to themselves AND a fresh set of
    current-frame queries) and cross-attention into the current multi-view
    image features, producing per-query 3D box + class predictions.
    """

    def __init__(
        self, dim: int = 24, n_queries: int = 10, n_memory: int = 6, motion_dim: int = 4
    ) -> None:
        super().__init__()
        self.dim = dim
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=8, stride=8)
        self.fresh_queries = nn.Parameter(torch.randn(1, n_queries, dim) * 0.02)
        self.mln = _MotionAwareLN(dim, motion_dim)
        self.self_attn = _SelfAttn(dim)
        self.cross_attn = _CrossAttn(dim)
        self.box_head = nn.Linear(dim, 8)  # cx,cy,cz,w,l,h,sin,cos
        self.cls_head = nn.Linear(dim, 3)

    def forward(
        self, images: torch.Tensor, memory_queries: torch.Tensor, motion: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """images: (n_cams,3,H,W); memory_queries: (1,n_memory,dim) propagated
        from the previous frame; motion: (1,motion_dim) ego pose+time delta."""
        feat = self.patch_embed(images)
        tokens = feat.flatten(2).transpose(1, 2).reshape(1, -1, self.dim)

        propagated = self.mln(
            memory_queries, motion.unsqueeze(1).expand(-1, memory_queries.shape[1], -1)
        )
        queries = torch.cat([propagated, self.fresh_queries.expand(1, -1, -1)], dim=1)

        queries = self.self_attn(queries)
        queries = self.cross_attn(queries, tokens)

        boxes = self.box_head(queries)
        logits = self.cls_head(queries)
        return boxes, logits


def build_streampetr() -> nn.Module:
    """Small StreamPETR object-centric temporal detector."""
    return StreamPETR(dim=24, n_queries=10, n_memory=6, motion_dim=4).eval()


def example_input_streampetr() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(multi-camera images (4,3,32,32), memory queries (1,6,24), motion delta (1,4))."""
    return torch.randn(4, 3, 32, 32), torch.randn(1, 6, 24), torch.randn(1, 4)


# ============================================================
# SurroundOcc -- multi-camera spatial cross attention + multi-level 3D occupancy
# ============================================================


class _SpatialCrossAttn3D(nn.Module):
    """Each voxel in a flattened 3D volume cross-attends into flattened
    multi-camera image tokens (approximating deformable-attention sampling)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.attn = _CrossAttn(dim)

    def forward(self, voxels: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        return self.attn(voxels, tokens)


class SurroundOcc(nn.Module):
    """Multi-camera 3D occupancy via spatial cross attention + progressive
    3D-conv upsampling with occupancy heads at EVERY resolution level
    (multi-level supervision, the paper's key training signal).
    """

    def __init__(self, dim: int = 16, vol: int = 4, n_classes: int = 3) -> None:
        super().__init__()
        self.dim, self.vol = dim, vol
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=8, stride=8)
        self.voxel_query = nn.Parameter(torch.randn(1, vol**3, dim) * 0.02)
        self.spatial_xattn = _SpatialCrossAttn3D(dim)

        # progressive 3D upsampling: vol -> 2*vol -> 4*vol, one occ head per level
        self.up1 = nn.ConvTranspose3d(dim, dim, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose3d(dim, dim, kernel_size=2, stride=2)
        self.occ_head_l1 = nn.Conv3d(dim, n_classes, kernel_size=1)
        self.occ_head_l2 = nn.Conv3d(dim, n_classes, kernel_size=1)
        self.occ_head_l3 = nn.Conv3d(dim, n_classes, kernel_size=1)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.patch_embed(images)
        tokens = feat.flatten(2).transpose(1, 2).reshape(1, -1, self.dim)

        voxels = self.spatial_xattn(self.voxel_query.expand(1, -1, -1), tokens)
        vol_feat = voxels.transpose(1, 2).reshape(1, self.dim, self.vol, self.vol, self.vol)

        occ_l1 = self.occ_head_l1(vol_feat)
        vol_feat = F.relu(self.up1(vol_feat))
        occ_l2 = self.occ_head_l2(vol_feat)
        vol_feat = F.relu(self.up2(vol_feat))
        occ_l3 = self.occ_head_l3(vol_feat)
        return occ_l1, occ_l2, occ_l3


def build_surroundocc() -> nn.Module:
    """Small SurroundOcc multi-level 3D occupancy predictor."""
    return SurroundOcc(dim=16, vol=4, n_classes=3).eval()


def example_input_surroundocc() -> torch.Tensor:
    """Multi-camera surround-view images (n_cams=4, 3, 32, 32)."""
    return torch.randn(4, 3, 32, 32)


# ============================================================
# TopoNet -- scene graph neural network over centerline + traffic-element queries
# ============================================================


class _RelationMessage(nn.Module):
    """One relation-specific message-passing step: query-pair affinities gate
    a learned message that is aggregated back into each query."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.msg = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, dim)

    def forward(self, src: torch.Tensor, dst: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """src: (1,Ns,dim) source queries, dst: (1,Nd,dim) destination queries.
        Returns (updated dst, adjacency logits (1,Nd,Ns))."""
        affinity = dst @ self.gate(src).transpose(-2, -1)  # (1,Nd,Ns)
        weights = torch.softmax(affinity, dim=-1)
        aggregated = weights @ self.msg(src)
        return dst + aggregated, affinity


class TopoNet(nn.Module):
    """Graph-based topology reasoning: centerline queries and traffic-element
    queries are embedded into a shared space, then a Scene Graph Neural
    Network passes relation-specific messages along THREE edge types
    (centerline<->centerline, centerline<->traffic-element,
    traffic-element<->traffic-element), each predicting its own topology
    adjacency matrix from query-pair affinities.
    """

    def __init__(self, dim: int = 20, n_centerline: int = 8, n_traffic: int = 6) -> None:
        super().__init__()
        self.dim = dim
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=8, stride=8)
        self.centerline_query = nn.Parameter(torch.randn(1, n_centerline, dim) * 0.02)
        self.traffic_query = nn.Parameter(torch.randn(1, n_traffic, dim) * 0.02)
        self.cl_embed_xattn = _CrossAttn(dim)
        self.te_embed_xattn = _CrossAttn(dim)

        self.cl_cl_relation = _RelationMessage(dim)
        self.cl_te_relation = _RelationMessage(dim)
        self.te_te_relation = _RelationMessage(dim)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.patch_embed(images)
        tokens = feat.flatten(2).transpose(1, 2).reshape(1, -1, self.dim)

        cl = self.cl_embed_xattn(self.centerline_query.expand(1, -1, -1), tokens)
        te = self.te_embed_xattn(self.traffic_query.expand(1, -1, -1), tokens)

        cl_updated, cl_cl_adj = self.cl_cl_relation(cl, cl)
        te_from_cl, cl_te_adj = self.cl_te_relation(cl_updated, te)
        te_updated, te_te_adj = self.te_te_relation(te_from_cl, te_from_cl)

        return cl_cl_adj, cl_te_adj, te_te_adj


def build_toponet() -> nn.Module:
    """Small TopoNet scene-graph topology reasoner."""
    return TopoNet(dim=20, n_centerline=8, n_traffic=6).eval()


def example_input_toponet() -> torch.Tensor:
    """Multi-camera surround-view images (n_cams=4, 3, 32, 32)."""
    return torch.randn(4, 3, 32, 32)


# ============================================================
# Ultra-Fast-Lane-Detection-v2 -- hybrid row+column anchor ordinal classification
# ============================================================


class _OrdinalAnchorHead(nn.Module):
    """One anchor-axis head: ordinal-classification location logits over a
    fixed grid PLUS a binary existence logit, per lane, per anchor line."""

    def __init__(self, in_dim: int, griding_num: int, n_anchors: int, n_lanes: int) -> None:
        super().__init__()
        self.griding_num, self.n_anchors, self.n_lanes = griding_num, n_anchors, n_lanes
        total_loc = (griding_num + 1) * n_anchors * n_lanes
        total_exist = 2 * n_anchors * n_lanes
        self.loc = nn.Linear(in_dim, total_loc)
        self.exist = nn.Linear(in_dim, total_exist)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        loc = self.loc(x).view(-1, self.griding_num + 1, self.n_anchors, self.n_lanes)
        exist = self.exist(x).view(-1, 2, self.n_anchors, self.n_lanes)
        return loc, exist


class UFLDv2(nn.Module):
    """Hybrid row+column anchor lane detector.

    A shared backbone feature map feeds TWO parallel ordinal-classification
    heads: a ROW-anchor head (horizontal scan lines, best for near-horizontal
    ego lanes) and a COLUMN-anchor head (vertical scan lines, best for
    near-vertical side lanes). Both heads run on the same pooled global
    feature and their location + existence outputs are returned together
    (fused per-lane downstream), rather than the v1 architecture's single
    row-anchor-only representation.
    """

    def __init__(
        self,
        griding_num: int = 50,
        n_row_anchors: int = 18,
        n_col_anchors: int = 18,
        n_lanes: int = 4,
    ) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 8)),
        )
        pooled_dim = 32 * 4 * 8
        hidden = 256
        self.fc = nn.Sequential(nn.Linear(pooled_dim, hidden), nn.ReLU(inplace=True))
        self.row_head = _OrdinalAnchorHead(hidden, griding_num, n_row_anchors, n_lanes)
        self.col_head = _OrdinalAnchorHead(hidden, griding_num, n_col_anchors, n_lanes)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.backbone(x)
        flat = self.fc(feat.flatten(1))
        row_loc, row_exist = self.row_head(flat)
        col_loc, col_exist = self.col_head(flat)
        return row_loc, row_exist, col_loc, col_exist


def build_ufldv2() -> nn.Module:
    """Small Ultra-Fast-Lane-Detection-v2 hybrid anchor model."""
    return UFLDv2(griding_num=50, n_row_anchors=18, n_col_anchors=18, n_lanes=4).eval()


def example_input_ufldv2() -> torch.Tensor:
    """RGB image (1, 3, 128, 256)."""
    return torch.randn(1, 3, 128, 256)


MENAGERIE_ENTRIES = [
    (
        "StreamMapNet",
        "build_streammapnet",
        "example_input_streammapnet",
        "2024",
        "VIS",
    ),
    (
        "StreamPETR",
        "build_streampetr",
        "example_input_streampetr",
        "2023",
        "VIS",
    ),
    (
        "SurroundOcc",
        "build_surroundocc",
        "example_input_surroundocc",
        "2023",
        "VIS",
    ),
    (
        "TopoNet",
        "build_toponet",
        "example_input_toponet",
        "2023",
        "VIS",
    ),
    (
        "Ultra-Fast v2",
        "build_ufldv2",
        "example_input_ufldv2",
        "2022",
        "VIS",
    ),
]
