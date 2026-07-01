"""Autonomous-driving perception architectures: HD-map graphs, lane affinity fields,
and video lane temporal-context aggregation.

Sources checked (reimplemented compactly from scratch in base-env torch; no clone/pip):

  - **InstaGraM** (Shin & Park, arXiv:2301.04470): "Instance-level Graph Modeling for
    Vectorized HD Map Learning". Official repo github.com/juyebshin/InstaGraM. The
    distinctive mechanism (`model/graphmap.py`) is a SuperGlue-style graph head applied
    to a BEV feature map: (1) a vertex heatmap + distance-transform embedding predict a
    sparse set of map-graph vertices, (2) each vertex's coordinate + DT-patch embedding is
    encoded via 1D-conv MLPs into a joint feature, (3) an `AttentionalGNN` (multi-head
    self-attention message passing over the vertex set, SuperGlue-style) refines vertex
    features, and (4) a `log_optimal_transport` Sinkhorn iteration turns a pairwise vertex
    affinity matrix into a soft vertex-adjacency (edge) assignment -- i.e. the map's graph
    topology is *predicted*, not decoded via heuristics. Reimplemented here as: BEV encoder
    -> vertex-score/embedding heads -> top-k vertex selection -> GraphEncoder (MLP) ->
    AttentionalGNN -> Sinkhorn log-optimal-transport edge matrix. Fixed top-k keeps shapes
    static (torchlens-traceable); the HDMapNet multi-camera IPM front end is out of scope
    (InstaGraM's contribution is the graph head, not the BEV lifting).
  - **LaneAF** (Abualsaud et al., IEEE RA-L 2021, arXiv:2010.02414): "Learning Lightweight
    Lane Detection CNNs by Self Attention Distillation" -- no, actually "LaneAF:
    Robust Multi-Lane Detection with Affinity Fields". Official repo github.com/sel118/LaneAF.
    The distinctive mechanism (`train_culane.py`, `models/dla/pose_dla_dcn.py`) is a
    DLA-34-style backbone with iterative-deep-aggregation (IDA) upsampling feeding THREE
    per-pixel heads: a binary lane/background heatmap (`hm`), a 2-channel Vertical Affinity
    Field (`vaf`, pointing toward the pixel's next-row lane neighbor), and a 1-channel
    Horizontal Affinity Field (`haf`, disambiguating merges/splits). At inference the
    affinity fields cluster heatmap pixels into lane instances without any learned instance
    embedding or anchor. Reimplemented here as a compact multi-level residual-tree backbone
    (faithful to DLA's iterative aggregation idea, reduced depth/width) + IDA-style upsample
    fusion + the three-head (`hm`/`vaf`/`haf`) output.
  - **LaneTCA** (paper "LaneTCA: Enhancing Video Lane Detection with Temporal Context
    Aggregation", arXiv:2408.13852). Official repo github.com/Alex-1337/LaneTCA
    (`Modeling/OpenLane-V/code/models/model.py`, `models/lstn/transformer.py`). The
    distinctive mechanism is an AOT/LSTN-style `LongShortTermTransformerBlock`: current-frame
    BEV/image features run self-attention, then attend to an *accumulative* long-term memory
    (all past frames, compressed) and an *adjacent* short-term memory (a local window of
    recent frames) via two additional multi-head attention passes, whose outputs are summed
    back into the token stream before a feed-forward block -- i.e. two extra memory-attention
    branches layered onto a standard transformer block, plus a deformable-conv regression head
    that predicts coefficients onto a learned low-rank lane-curve basis. Reimplemented here as
    a single-timestep LSAB (self-attn + long-term cross-attn + short-term cross-attn, summed)
    operating on flattened feature tokens, with explicit long-/short-term memory tensors as
    extra inputs (the recurrent cross-frame state loop itself is not traced; capturing one
    LSAB step under real memory tensors is what is architecturally distinctive) followed by a
    basis-coefficient regression head.

Random init, CPU, forward-only, small dims for fast tracing.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# InstaGraM: BEV vertex extraction + AttentionalGNN + Sinkhorn OT graph head
# ============================================================


def _mlp1d(channels: list[int]) -> nn.Sequential:
    """1D-conv MLP over a token axis (Conv1d==Linear per-vertex, SuperGlue-style)."""
    layers: list[nn.Module] = []
    n = len(channels)
    for i in range(1, n):
        layers.append(nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < n - 1:
            layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class GraphEncoder(nn.Module):
    """Encode per-vertex (coord, DT-embedding) features into the joint graph feature."""

    def __init__(self, in_ch: int, feature_dim: int) -> None:
        super().__init__()
        self.encoder = _mlp1d([in_ch, feature_dim, feature_dim])

    def forward(self, vertex_feat: torch.Tensor) -> torch.Tensor:
        """``vertex_feat``: (B, N, in_ch) -> (B, feature_dim, N)."""
        return self.encoder(vertex_feat.transpose(1, 2))


class MultiHeadedAttention(nn.Module):
    """SuperGlue-style multi-head attention over the vertex-token axis."""

    def __init__(self, num_heads: int, d_model: int) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.q_proj = nn.Conv1d(d_model, d_model, 1)
        self.k_proj = nn.Conv1d(d_model, d_model, 1)
        self.v_proj = nn.Conv1d(d_model, d_model, 1)
        self.merge = nn.Conv1d(d_model, d_model, 1)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        b = query.size(0)
        q = self.q_proj(query).view(b, self.dim, self.num_heads, -1)
        k = self.k_proj(key).view(b, self.dim, self.num_heads, -1)
        v = self.v_proj(value).view(b, self.dim, self.num_heads, -1)
        scores = torch.einsum("bdhn,bdhm->bhnm", q, k) / self.dim**0.5
        prob = F.softmax(scores, dim=-1)
        out = torch.einsum("bhnm,bdhm->bdhn", prob, v)
        return self.merge(out.contiguous().view(b, self.dim * self.num_heads, -1))


class AttentionalPropagation(nn.Module):
    """One self-attention message-passing round over the vertex graph."""

    def __init__(self, feature_dim: int, num_heads: int) -> None:
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = _mlp1d([feature_dim * 2, feature_dim * 2, feature_dim])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        message = self.attn(x, x, x)
        return self.mlp(torch.cat([x, message], dim=1))


class AttentionalGNN(nn.Module):
    """Stack of self-attention propagation rounds refining vertex features."""

    def __init__(self, feature_dim: int, num_layers: int, num_heads: int = 4) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [AttentionalPropagation(feature_dim, num_heads) for _ in range(num_layers)]
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            embedding = embedding + layer(embedding)
        return embedding


def log_sinkhorn_iterations(
    z: torch.Tensor, log_mu: torch.Tensor, log_nu: torch.Tensor, iters: int
) -> torch.Tensor:
    """Sinkhorn normalization in log-space for a differentiable soft assignment."""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(z + u.unsqueeze(2), dim=1)
    return z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores: torch.Tensor, alpha: torch.Tensor, iters: int) -> torch.Tensor:
    """Differentiable optimal-transport graph-edge assignment (SuperGlue/InstaGraM)."""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m * one).to(scores), (n * one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha_full = alpha.expand(b, 1, 1)

    couplings = torch.cat(
        [
            torch.cat([scores, bins0], -1),
            torch.cat([bins1, alpha_full], -1),
        ],
        1,
    )

    norm = -(ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu = log_mu[None].expand(b, -1)
    log_nu = log_nu[None].expand(b, -1)

    z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    return z - norm


class InstaGraM(nn.Module):
    """Compact InstaGraM graph head: BEV feature -> vertices -> GNN -> Sinkhorn edges."""

    def __init__(
        self,
        bev_ch: int = 32,
        feature_dim: int = 32,
        n_vertices: int = 24,
        gnn_layers: int = 3,
        sinkhorn_iters: int = 20,
    ) -> None:
        super().__init__()
        self.n_vertices = n_vertices
        self.bev_encoder = nn.Sequential(
            nn.Conv2d(3, bev_ch, 3, padding=1),
            nn.BatchNorm2d(bev_ch),
            nn.ReLU(),
            nn.Conv2d(bev_ch, bev_ch, 3, padding=1),
            nn.BatchNorm2d(bev_ch),
            nn.ReLU(),
        )
        self.vertex_score_head = nn.Conv2d(bev_ch, 1, 1)
        self.dt_embed_head = nn.Conv2d(bev_ch, 8, 1)  # distance-transform-style local embedding

        self.graph_encoder = GraphEncoder(in_ch=3 + 8, feature_dim=feature_dim)
        self.gnn = AttentionalGNN(feature_dim, gnn_layers)
        self.final_proj = nn.Conv1d(feature_dim, feature_dim, 1)
        self.bin_score = nn.Parameter(torch.tensor(1.0))
        self.sinkhorn_iters = sinkhorn_iters

    def forward(self, bev_image: torch.Tensor) -> torch.Tensor:
        """``bev_image``: (B, 3, H, W) top-down raster. Returns (B, N+1, N+1) log edge matrix."""
        feat = self.bev_encoder(bev_image)  # (B, C, H, W)
        b, c, h, w = feat.shape

        scores = torch.sigmoid(self.vertex_score_head(feat)).view(b, h * w)
        dt_embed = self.dt_embed_head(feat).view(b, 8, h * w).transpose(1, 2)  # (B, HW, 8)

        top_scores, top_idx = torch.topk(scores, self.n_vertices, dim=1)  # (B, N)
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, h, device=bev_image.device),
            torch.linspace(-1, 1, w, device=bev_image.device),
            indexing="ij",
        )
        coord_grid = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)  # (HW, 2)
        coord_grid = coord_grid.unsqueeze(0).expand(b, -1, -1)
        gathered_coord = torch.gather(
            coord_grid, 1, top_idx.unsqueeze(-1).expand(-1, -1, 2)
        )  # (B, N, 2)
        gathered_dt = torch.gather(dt_embed, 1, top_idx.unsqueeze(-1).expand(-1, -1, 8))

        vertex_feat = torch.cat(
            [gathered_coord, top_scores.unsqueeze(-1), gathered_dt], dim=-1
        )  # (B, N, 3+8)

        graph_feat = self.graph_encoder(vertex_feat)  # (B, feature_dim, N)
        graph_feat = self.gnn(graph_feat)
        graph_feat = self.final_proj(graph_feat)  # (B, feature_dim, N)

        sim = torch.einsum("bdn,bdm->bnm", graph_feat, graph_feat) / graph_feat.shape[1] ** 0.5
        edge_matrix = log_optimal_transport(sim, self.bin_score, self.sinkhorn_iters)
        return edge_matrix


def build_instagram() -> nn.Module:
    """Build the compact InstaGraM vertex-graph + Sinkhorn-matching head."""
    return InstaGraM(bev_ch=32, feature_dim=32, n_vertices=24, gnn_layers=3).eval()


def example_input_instagram() -> torch.Tensor:
    """Top-down BEV raster image (1, 3, 64, 64)."""
    return torch.randn(1, 3, 64, 64)


# ============================================================
# LaneAF: DLA-style iterative-aggregation backbone + heatmap/VAF/HAF heads
# ============================================================


class ResBlock(nn.Module):
    """Basic residual block (DLA `BasicBlock`)."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=False)
        self.project = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False), nn.BatchNorm2d(out_ch)
            )
            if (in_ch != out_ch or stride != 1)
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.project(x) if self.project is not None else x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


class DLARoot(nn.Module):
    """Aggregation root -- fuses a tree's two children (DLA `Root`)."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        out = self.conv(torch.cat([x1, x2], dim=1))
        return self.relu(self.bn(out))


class DLATreeLevel(nn.Module):
    """One shallow DLA tree level: two residual blocks + a root aggregation (depth-1 tree)."""

    def __init__(self, in_ch: int, out_ch: int, stride: int) -> None:
        super().__init__()
        self.block1 = ResBlock(in_ch, out_ch, stride=stride)
        self.block2 = ResBlock(out_ch, out_ch, stride=1)
        self.root = DLARoot(out_ch * 2, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.block1(x)
        x2 = self.block2(x1)
        return self.root(x1, x2)


class IDAUpStep(nn.Module):
    """Iterative Deep Aggregation upsample-and-fuse step (DLA `IDAUp`, plain conv variant)."""

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1)
        self.proj = nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch))
        self.node = nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch))
        self.relu = nn.ReLU(inplace=False)

    def forward(self, deep: torch.Tensor, shallow: torch.Tensor) -> torch.Tensor:
        up = self.relu(self.proj(self.up(deep)))
        return self.relu(self.node(up + shallow))


class LaneAFNet(nn.Module):
    """Compact DLA-style backbone + IDA upsampling + heatmap/VAF/HAF affinity-field heads."""

    def __init__(self, stem_ch: int = 16, ch: int = 32) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_ch, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(stem_ch),
            nn.ReLU(inplace=False),
        )
        self.level1 = DLATreeLevel(stem_ch, ch, stride=2)  # /4
        self.level2 = DLATreeLevel(ch, ch * 2, stride=2)  # /8
        self.level3 = DLATreeLevel(ch * 2, ch * 4, stride=2)  # /16

        self.proj2_to_4 = nn.Sequential(
            nn.Conv2d(ch * 2, ch * 4, 1, bias=False), nn.BatchNorm2d(ch * 4)
        )
        self.proj1_to_4 = nn.Sequential(
            nn.Conv2d(ch, ch * 4, 1, bias=False), nn.BatchNorm2d(ch * 4)
        )

        self.ida_up_a = IDAUpStep(ch * 4)  # /16 -> /8
        self.ida_up_b = IDAUpStep(ch * 4)  # /8 -> /4

        head_ch = ch * 4
        self.hm_head = nn.Sequential(
            nn.Conv2d(head_ch, head_ch, 3, padding=1), nn.ReLU(), nn.Conv2d(head_ch, 1, 1)
        )
        self.vaf_head = nn.Sequential(
            nn.Conv2d(head_ch, head_ch, 3, padding=1), nn.ReLU(), nn.Conv2d(head_ch, 2, 1)
        )
        self.haf_head = nn.Sequential(
            nn.Conv2d(head_ch, head_ch, 3, padding=1), nn.ReLU(), nn.Conv2d(head_ch, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x0 = self.stem(x)  # /2
        x1 = self.level1(x0)  # /4, ch
        x2 = self.level2(x1)  # /8, ch*2
        x3 = self.level3(x2)  # /16, ch*4

        x2_proj = self.proj2_to_4(x2)  # /8, ch*4
        x1_proj = self.proj1_to_4(x1)  # /4, ch*4

        fused_8 = self.ida_up_a(x3, x2_proj)  # /8
        fused_4 = self.ida_up_b(fused_8, x1_proj)  # /4

        return {
            "hm": torch.sigmoid(self.hm_head(fused_4)),
            "vaf": torch.tanh(self.vaf_head(fused_4)),
            "haf": torch.tanh(self.haf_head(fused_4)),
        }


def build_laneaf() -> nn.Module:
    """Build the compact LaneAF (heatmap + vertical/horizontal affinity field) detector."""
    return LaneAFNet(stem_ch=16, ch=32).eval()


def example_input_laneaf() -> torch.Tensor:
    """RGB road image (1, 3, 128, 256)."""
    return torch.randn(1, 3, 128, 256)


# ============================================================
# LaneTCA: single-step Long-Short-Term (memory) transformer block + basis-curve head
# ============================================================


class LongShortTermAttentionBlock(nn.Module):
    """One LSAB step (LaneTCA / AOT-style): self-attn + long-term + short-term cross-attn."""

    def __init__(self, d_model: int = 32, n_heads: int = 4, dim_ff: int = 64) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        self.norm2 = nn.LayerNorm(d_model)
        self.long_term_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.short_term_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        self.norm3 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_ff)
        self.linear2 = nn.Linear(dim_ff, d_model)
        self.act = nn.GELU()

    def forward(
        self,
        tokens: torch.Tensor,
        long_term_memory: torch.Tensor,
        short_term_memory: torch.Tensor,
    ) -> torch.Tensor:
        """``tokens``: (B, T, C) current-frame tokens; memories: (B, M, C)."""
        t = self.norm1(tokens)
        self_out, _ = self.self_attn(t, t, t)
        tokens = tokens + self_out

        t2 = self.norm2(tokens)
        long_out, _ = self.long_term_attn(t2, long_term_memory, long_term_memory)
        short_out, _ = self.short_term_attn(t2, short_term_memory, short_term_memory)
        tokens = tokens + long_out + short_out

        t3 = self.norm3(tokens)
        tokens = tokens + self.linear2(self.act(self.linear1(t3)))
        return tokens


class LaneTCA(nn.Module):
    """Compact LaneTCA: CNN feature -> LSAB memory attention -> lane-basis coefficient head."""

    def __init__(
        self,
        feat_ch: int = 32,
        n_basis: int = 8,
        n_points: int = 16,
        mem_len: int = 6,
    ) -> None:
        super().__init__()
        self.feat_ch = feat_ch
        self.encoder = nn.Sequential(
            nn.Conv2d(3, feat_ch, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(feat_ch, feat_ch, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.lsab = LongShortTermAttentionBlock(d_model=feat_ch)
        self.pos_embed = nn.Parameter(torch.zeros(1, 256, feat_ch))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # basis-coefficient regression head: predicts coefficients on a learned
        # low-rank lane-curve basis (LaneTCA's `deform_conv2d`-based coefficient map,
        # approximated here with a linear per-token basis-coefficient projection).
        self.basis = nn.Parameter(torch.randn(n_basis, n_points) * 0.1)
        self.coeff_head = nn.Sequential(nn.LayerNorm(feat_ch), nn.Linear(feat_ch, n_basis))
        self.exist_head = nn.Sequential(nn.LayerNorm(feat_ch), nn.Linear(feat_ch, 1))
        self.mem_len = mem_len

    def forward(
        self,
        image: torch.Tensor,
        long_term_memory: torch.Tensor,
        short_term_memory: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """``image``: (B, 3, H, W); memories: (B, mem_len, feat_ch) accumulated/adjacent frames."""
        feat = self.encoder(image)  # (B, C, h, w)
        b, c, h, w = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)  # (B, HW, C)
        tokens = tokens + self.pos_embed[:, : tokens.shape[1]]

        fused = self.lsab(tokens, long_term_memory, short_term_memory)  # (B, HW, C)

        pooled = fused.mean(dim=1)  # (B, C) -- per-lane-query pooled summary
        coeffs = self.coeff_head(pooled)  # (B, n_basis)
        curve = coeffs @ self.basis  # (B, n_points) lane x-offsets along fixed y-anchors
        exist = torch.sigmoid(self.exist_head(pooled))  # (B, 1)
        return {"curve": curve, "exist": exist, "tokens": fused}


def build_lanetca() -> nn.Module:
    """Build the compact LaneTCA temporal-context-aggregation lane detector."""
    return LaneTCA(feat_ch=32, n_basis=8, n_points=16, mem_len=6).eval()


def example_input_lanetca() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(image (1,3,64,64), long_term_memory (1,10,32), short_term_memory (1,6,32))."""
    image = torch.randn(1, 3, 64, 64)
    long_term_memory = torch.randn(1, 10, 32)
    short_term_memory = torch.randn(1, 6, 32)
    return image, long_term_memory, short_term_memory


# ============================================================
# Registry
# ============================================================

MENAGERIE_ENTRIES = [
    ("InstaGraM", "build_instagram", "example_input_instagram", "2023", "VIS"),
    ("LaneAF", "build_laneaf", "example_input_laneaf", "2021", "VIS"),
    ("LaneTCA", "build_lanetca", "example_input_lanetca", "2024", "VIS"),
]
