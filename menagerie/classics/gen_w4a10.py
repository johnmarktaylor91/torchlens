"""Trajectory-prediction / autonomous-driving classics (batch w4a10).

Sources checked (paper + official repo README/architecture description; no clone,
no pip install -- reimplemented from scratch in base-env torch):

- GroupNet: Xu, Li, Ni, Zhang & Chen, CVPR 2022, arXiv:2204.08770.
  https://github.com/MediaBrain-SJTU/GroupNet
  Multiscale hypergraph neural network for trajectory prediction: a CVAE
  encoder/decoder wraps a stack of *multiscale hypergraph message-passing*
  layers that alternate node->edge and edge->node updates over pairwise
  (2-agent) and higher-order group hyperedges in parallel, so relational
  reasoning happens at multiple group sizes simultaneously before the
  aggregated interaction features condition a GRU trajectory decoder.

- HiVT: Zhou, Ye, Wang, Wu & Lu, CVPR 2022, arXiv:2206.10047.
  https://github.com/ZikangZhou/HiVT
  Hierarchical Vector Transformer: a *local* stage extracts translation- and
  rotation-invariant per-agent context via per-timestep agent-agent attention
  followed by a temporal transformer over the history window, then a
  *global* stage runs self-attention across all agents' local embeddings to
  model scene-level interaction before a multi-modal Laplacian trajectory
  decoder. The local/global split (not one flat transformer) is HiVT's
  namesake mechanism.

- HPNet: Tang, Kan, Shan, Ji, Bai & Chen, CVPR 2024, arXiv:2404.06351.
  https://github.com/XiaolongTang23/HPNet
  Dynamic trajectory forecasting with a *Triple Factorized Attention* block:
  Mode Attention (across the multiple predicted trajectory modes), Agent
  Attention (across agents at a timestep), and a novel Historical Prediction
  Attention (each new prediction attends back over the model's own *past
  predictions* at earlier timesteps, not just past observations) are
  factorized into three separate attention passes applied in sequence at
  every decoding step, giving temporally stable multi-modal forecasts.

- HPTR: Zhang, Liniger, Sakaridis, Yu & Van Gool, NeurIPS 2023, arXiv:2310.12970.
  https://github.com/zhejz/HPTR (also KIT-MRT/hptr)
  Heterogeneous Polyline Transformer with Relative Pose Encoding: KNARPE
  (K-nearest-neighbor attention with relative pose encoding) lets a query
  token attend to only its k nearest heterogeneous polyline tokens (map
  lanes, traffic lights, agent history) using a *pairwise-relative* pose
  encoding injected into the attention logits instead of absolute
  positional embeddings, stacked hierarchically for scene tokens -> agent
  tokens -> per-mode trajectory queries.

- InterFuser: Shao, Wang, Chen, Li & Liu, CoRL 2022, arXiv:2207.14024.
  https://github.com/opendilab/InterFuser
  Interpretable sensor-fusion transformer for end-to-end driving: separate
  small CNN backbones tokenize multi-view camera images and a LiDAR BEV
  grid; all image/LiDAR tokens plus learned view-position embeddings feed a
  shared Transformer encoder-decoder, where a fixed bank of *query tokens*
  (waypoint queries + a spatial grid of object-density/traffic-rule queries)
  cross-attends to the fused tokens and decodes both the safety-relevant
  interpretable density map and the future ego waypoints jointly.

- LAV: Chen & Krahenbuhl, CVPR 2022, arXiv:2203.11934.
  https://github.com/dotchen/LAV
  Learning from All Vehicles: a dynamic PointNet aggregates raw LiDAR points
  into BEV pillars by scatter-max pooling points that fall in the same
  pillar cell (no fixed voxel binning of point clouds -- pillars are formed
  by parallel point aggregation), a small BEV CNN encodes the resulting grid,
  and the same shared trajectory decoder head -- crop features around each
  detected car, one command-conditioned GRU rollout branch selected by a
  discrete navigation command -- is applied identically to the ego vehicle
  and every other perceived vehicle, which is LAV's namesake "learn from
  all vehicles" reuse of a single planner across agents.

Random init, tiny dims, CPU-only -- architecture catalog entries, not trained
weights.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

# ---------------------------------------------------------------------------
# GroupNet -- multiscale hypergraph neural network (CVPR 2022)
# ---------------------------------------------------------------------------


class HypergraphMPLayer(nn.Module):
    """One node<->edge message-passing pass over a fixed group-size hyperedge set.

    Parameters
    ----------
    dim:
        Node/edge feature width.
    group_size:
        Number of agents joined by each hyperedge (2 = pairwise, >2 = group).
    """

    def __init__(self, dim: int, group_size: int) -> None:
        super().__init__()
        self.group_size = group_size
        self.node2edge = nn.Linear(dim * group_size, dim)
        self.edge_gate = nn.Sequential(nn.Linear(dim, 1), nn.Sigmoid())
        self.edge2node = nn.Linear(dim, dim)
        self.update = nn.GRUCell(dim, dim)

    def _hyperedges(self, n: int) -> Tensor:
        """Enumerate all ``group_size``-subsets of ``n`` agents as index rows."""
        idx = torch.combinations(torch.arange(n), r=self.group_size)
        if idx.numel() == 0:
            idx = torch.arange(n).unsqueeze(1).repeat(1, self.group_size)
        return idx

    def forward(self, nodes: Tensor) -> Tensor:
        """Run one hypergraph message-passing pass.

        Parameters
        ----------
        nodes:
            ``(batch, n_agents, dim)`` node features.

        Returns
        -------
        Tensor
            Updated ``(batch, n_agents, dim)`` node features.
        """
        b, n, d = nodes.shape
        edge_idx = self._hyperedges(n).to(nodes.device)
        gathered = nodes[:, edge_idx, :].reshape(b, edge_idx.shape[0], -1)
        edge_feat = torch.tanh(self.node2edge(gathered))
        weight = self.edge_gate(edge_feat)
        edge_feat = edge_feat * weight

        incoming = torch.zeros(b, n, edge_feat.shape[-1], device=nodes.device, dtype=nodes.dtype)
        counts = torch.zeros(b, n, 1, device=nodes.device, dtype=nodes.dtype)
        for k in range(self.group_size):
            member = edge_idx[:, k]
            incoming.index_add_(1, member, edge_feat)
            counts.index_add_(
                1,
                member,
                torch.ones(b, edge_idx.shape[0], 1, device=nodes.device, dtype=nodes.dtype),
            )
        incoming = incoming / counts.clamp_min(1.0)
        msg = self.edge2node(incoming)
        out = self.update(msg.reshape(b * n, d), nodes.reshape(b * n, d))
        return out.reshape(b, n, d)


class GroupNetPredictor(nn.Module):
    """Multiscale hypergraph trajectory predictor (GroupNet, simplified CVAE decoder).

    Parameters
    ----------
    past_len:
        Number of observed timesteps.
    future_len:
        Number of timesteps to forecast.
    dim:
        Hidden feature width.
    """

    def __init__(self, past_len: int = 8, future_len: int = 6, dim: int = 32) -> None:
        super().__init__()
        self.past_len = past_len
        self.future_len = future_len
        self.dim = dim
        self.traj_encoder = nn.GRU(2, dim, batch_first=True)
        self.scales = nn.ModuleList(
            [HypergraphMPLayer(dim, group_size=2), HypergraphMPLayer(dim, group_size=3)]
        )
        self.fuse = nn.Linear(dim * (1 + len(self.scales)), dim)
        self.decoder_cell = nn.GRUCell(2, dim)
        self.out_head = nn.Linear(dim, 2)

    def forward(self, past_traj: Tensor) -> Tensor:
        """Predict future trajectories from observed history.

        Parameters
        ----------
        past_traj:
            ``(batch, n_agents, past_len, 2)`` observed (x, y) positions.

        Returns
        -------
        Tensor
            ``(batch, n_agents, future_len, 2)`` predicted future positions.
        """
        b, n, t, _ = past_traj.shape
        flat = past_traj.reshape(b * n, t, 2)
        _, h = self.traj_encoder(flat)
        node = h.squeeze(0).reshape(b, n, self.dim)

        multiscale = [node]
        for layer in self.scales:
            multiscale.append(layer(node))
        fused = torch.tanh(self.fuse(torch.cat(multiscale, dim=-1)))

        hidden = fused.reshape(b * n, self.dim)
        step_in = past_traj[:, :, -1, :].reshape(b * n, 2)
        outputs = []
        for _ in range(self.future_len):
            hidden = self.decoder_cell(step_in, hidden)
            step_out = self.out_head(hidden)
            outputs.append(step_out)
            step_in = step_out
        return torch.stack(outputs, dim=1).reshape(b, n, self.future_len, 2)


def build_groupnet() -> nn.Module:
    """Build a compact GroupNet multiscale-hypergraph trajectory predictor.

    Returns
    -------
    nn.Module
        Random-initialized ``GroupNetPredictor`` in eval mode.
    """
    return GroupNetPredictor(past_len=8, future_len=6, dim=32).eval()


def example_input_groupnet() -> Tensor:
    """Example input for :func:`build_groupnet`.

    Returns
    -------
    Tensor
        ``(2, 5, 8, 2)`` batch of 5-agent, 8-step observed trajectories.
    """
    return torch.randn(2, 5, 8, 2)


# ---------------------------------------------------------------------------
# HiVT -- Hierarchical Vector Transformer (CVPR 2022)
# ---------------------------------------------------------------------------


class LocalAgentEncoder(nn.Module):
    """Local stage: per-timestep agent-agent attention + temporal transformer.

    Parameters
    ----------
    dim:
        Embedding width.
    n_heads:
        Attention head count.
    """

    def __init__(self, dim: int = 32, n_heads: int = 4) -> None:
        super().__init__()
        self.embed = nn.Linear(2, dim)
        self.spatial_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        temporal_layer = nn.TransformerEncoderLayer(
            dim, n_heads, dim_feedforward=dim * 2, batch_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(temporal_layer, num_layers=2)

    def forward(self, past_traj: Tensor) -> Tensor:
        """Extract per-agent local context.

        Parameters
        ----------
        past_traj:
            ``(batch, n_agents, t, 2)`` displacement-encoded history.

        Returns
        -------
        Tensor
            ``(batch, n_agents, dim)`` local per-agent embeddings.
        """
        b, n, t, _ = past_traj.shape
        tok = self.embed(past_traj)
        per_t = []
        for step in range(t):
            frame = tok[:, :, step, :]
            attended, _ = self.spatial_attn(frame, frame, frame)
            per_t.append(attended)
        seq = torch.stack(per_t, dim=2).reshape(b * n, t, -1)
        out = self.temporal_encoder(seq)
        return out[:, -1, :].reshape(b, n, -1)


class GlobalInteractor(nn.Module):
    """Global stage: scene-level self-attention across all agents' local embeddings."""

    def __init__(self, dim: int = 32, n_heads: int = 4) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(dim, n_heads, dim_feedforward=dim * 2, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)

    def forward(self, local_embed: Tensor) -> Tensor:
        """Refine local embeddings with global agent-agent context.

        Parameters
        ----------
        local_embed:
            ``(batch, n_agents, dim)``.

        Returns
        -------
        Tensor
            ``(batch, n_agents, dim)`` globally-contextualized embeddings.
        """
        return self.encoder(local_embed)


class HiVT(nn.Module):
    """Hierarchical Vector Transformer: local context then global interaction.

    Parameters
    ----------
    future_len:
        Number of future timesteps to decode.
    dim:
        Embedding width.
    """

    def __init__(self, future_len: int = 6, dim: int = 32) -> None:
        super().__init__()
        self.local_encoder = LocalAgentEncoder(dim)
        self.global_interactor = GlobalInteractor(dim)
        self.future_len = future_len
        self.decoder = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, future_len * 2))

    def forward(self, past_traj: Tensor) -> Tensor:
        """Predict future trajectories.

        Parameters
        ----------
        past_traj:
            ``(batch, n_agents, t, 2)`` observed positions.

        Returns
        -------
        Tensor
            ``(batch, n_agents, future_len, 2)`` predicted future positions.
        """
        b, n, _, _ = past_traj.shape
        local = self.local_encoder(past_traj)
        global_ctx = self.global_interactor(local)
        out = self.decoder(global_ctx)
        return out.reshape(b, n, self.future_len, 2)


def build_hivt() -> nn.Module:
    """Build a compact HiVT hierarchical vector transformer.

    Returns
    -------
    nn.Module
        Random-initialized ``HiVT`` in eval mode.
    """
    return HiVT(future_len=6, dim=32).eval()


def example_input_hivt() -> Tensor:
    """Example input for :func:`build_hivt`.

    Returns
    -------
    Tensor
        ``(2, 6, 8, 2)`` batch of 6-agent, 8-step observed trajectories.
    """
    return torch.randn(2, 6, 8, 2)


# ---------------------------------------------------------------------------
# HPNet -- Triple Factorized Attention (CVPR 2024)
# ---------------------------------------------------------------------------


class TripleFactorizedAttention(nn.Module):
    """Mode attention, agent attention, and historical-prediction attention in sequence.

    Parameters
    ----------
    dim:
        Feature width.
    n_heads:
        Attention head count.
    """

    def __init__(self, dim: int = 32, n_heads: int = 4) -> None:
        super().__init__()
        self.mode_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.agent_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.hist_pred_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, modes: Tensor, hist_preds: Tensor) -> Tensor:
        """Apply the three factorized attention passes.

        Parameters
        ----------
        modes:
            ``(batch * n_agents, n_modes, dim)`` per-mode query features.
        hist_preds:
            ``(batch * n_agents, n_hist, dim)`` the model's own past-step
            predictions, attended to by each mode query (Historical
            Prediction Attention).

        Returns
        -------
        Tensor
            ``(batch * n_agents, n_modes, dim)`` updated mode features.
        """
        ba, n_modes, dim = modes.shape
        modes, _ = self.mode_attn(modes, modes, modes)

        agent_view = modes.transpose(0, 1)
        agent_view, _ = self.agent_attn(agent_view, agent_view, agent_view)
        modes = agent_view.transpose(0, 1)

        hist_out, _ = self.hist_pred_attn(modes, hist_preds, hist_preds)
        return self.norm(modes + hist_out)


class HPNet(nn.Module):
    """Dynamic trajectory forecaster driven by Triple Factorized Attention.

    Parameters
    ----------
    future_len:
        Number of future timesteps.
    n_modes:
        Number of multi-modal trajectory hypotheses.
    dim:
        Feature width.
    """

    def __init__(self, future_len: int = 6, n_modes: int = 3, dim: int = 32) -> None:
        super().__init__()
        self.future_len = future_len
        self.n_modes = n_modes
        self.dim = dim
        self.hist_encoder = nn.GRU(2, dim, batch_first=True)
        self.mode_embed = nn.Parameter(torch.randn(n_modes, dim) * 0.02)
        self.tfa_layers = nn.ModuleList([TripleFactorizedAttention(dim) for _ in range(2)])
        self.step_head = nn.Linear(dim, 2)

    def forward(self, past_traj: Tensor) -> Tensor:
        """Forecast multi-modal future trajectories with historical-prediction attention.

        Parameters
        ----------
        past_traj:
            ``(batch, n_agents, t, 2)`` observed positions.

        Returns
        -------
        Tensor
            ``(batch, n_agents, n_modes, future_len, 2)`` predicted trajectories.
        """
        b, n, t, _ = past_traj.shape
        flat = past_traj.reshape(b * n, t, 2)
        _, h = self.hist_encoder(flat)
        agent_ctx = h.squeeze(0)  # (b*n, dim)

        modes = self.mode_embed.unsqueeze(0).expand(b * n, -1, -1) + agent_ctx.unsqueeze(1)
        hist_preds = agent_ctx.unsqueeze(1).expand(-1, self.n_modes, -1)

        preds = []
        for _ in range(self.future_len):
            for layer in self.tfa_layers:
                modes = layer(modes, hist_preds)
            step = self.step_head(modes)  # (b*n, n_modes, 2)
            preds.append(step)
            hist_preds = torch.cat([hist_preds, modes], dim=1)[:, -self.n_modes :, :]
        out = torch.stack(preds, dim=2)  # (b*n, n_modes, future_len, 2)
        return out.reshape(b, n, self.n_modes, self.future_len, 2)


def build_hpnet() -> nn.Module:
    """Build a compact HPNet triple-factorized-attention trajectory forecaster.

    Returns
    -------
    nn.Module
        Random-initialized ``HPNet`` in eval mode.
    """
    return HPNet(future_len=4, n_modes=3, dim=32).eval()


def example_input_hpnet() -> Tensor:
    """Example input for :func:`build_hpnet`.

    Returns
    -------
    Tensor
        ``(2, 4, 6, 2)`` batch of 4-agent, 6-step observed trajectories.
    """
    return torch.randn(2, 4, 6, 2)


# ---------------------------------------------------------------------------
# HPTR -- Heterogeneous Polyline Transformer with Relative Pose Encoding
# ---------------------------------------------------------------------------


class KNARPEBlock(nn.Module):
    """K-nearest-neighbor attention with relative pose encoding (KNARPE).

    A query token attends only to its ``k`` nearest heterogeneous polyline
    tokens (by Euclidean position), with each key modulated by an MLP encoding
    of the *relative pose* ``(query_pos - key_pos)`` injected additively into
    the attended value -- HPTR's namesake relative, not absolute, positional
    mechanism.

    Parameters
    ----------
    dim:
        Feature width.
    k:
        Number of nearest neighbors attended to per query.
    """

    def __init__(self, dim: int = 32, k: int = 4) -> None:
        super().__init__()
        self.k = k
        self.dim = dim
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.rel_pose_mlp = nn.Sequential(nn.Linear(2, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, query: Tensor, query_pos: Tensor, ctx: Tensor, ctx_pos: Tensor) -> Tensor:
        """Attend each query to its k nearest context polyline tokens.

        Parameters
        ----------
        query:
            ``(batch, n_q, dim)`` query token features.
        query_pos:
            ``(batch, n_q, 2)`` query token 2D positions.
        ctx:
            ``(batch, n_ctx, dim)`` context (polyline/agent) token features.
        ctx_pos:
            ``(batch, n_ctx, 2)`` context token 2D positions.

        Returns
        -------
        Tensor
            ``(batch, n_q, dim)`` updated query features.
        """
        b, n_q, _ = query.shape
        n_ctx = ctx.shape[1]
        k = min(self.k, n_ctx)

        dist = torch.cdist(query_pos, ctx_pos)  # (b, n_q, n_ctx)
        knn_idx = dist.topk(k, dim=-1, largest=False).indices  # (b, n_q, k)

        batch_idx = torch.arange(b, device=query.device).view(b, 1, 1).expand(-1, n_q, k)
        ctx_knn = ctx[batch_idx, knn_idx]  # (b, n_q, k, dim)
        ctx_pos_knn = ctx_pos[batch_idx, knn_idx]  # (b, n_q, k, 2)

        rel_pose = query_pos.unsqueeze(2) - ctx_pos_knn
        pose_embed = self.rel_pose_mlp(rel_pose)

        q = self.q_proj(query).unsqueeze(2)  # (b, n_q, 1, dim)
        key = self.k_proj(ctx_knn) + pose_embed
        val = self.v_proj(ctx_knn) + pose_embed

        logits = (q * key).sum(-1) / math.sqrt(self.dim)  # (b, n_q, k)
        weights = F.softmax(logits, dim=-1).unsqueeze(-1)
        attended = (weights * val).sum(2)
        return self.out_proj(attended) + query


class HPTR(nn.Module):
    """Hierarchical heterogeneous polyline transformer with relative pose encoding.

    Parameters
    ----------
    future_len:
        Number of future timesteps.
    dim:
        Feature width.
    """

    def __init__(self, future_len: int = 6, dim: int = 32) -> None:
        super().__init__()
        self.future_len = future_len
        self.dim = dim
        self.map_embed = nn.Linear(2, dim)
        self.agent_embed = nn.GRU(2, dim, batch_first=True)
        self.scene_layer = KNARPEBlock(dim, k=4)
        self.agent_layer = KNARPEBlock(dim, k=3)
        self.query_embed = nn.Parameter(torch.randn(1, dim) * 0.02)
        self.mode_layer = KNARPEBlock(dim, k=3)
        self.head = nn.Linear(dim, future_len * 2)

    def forward(self, agent_traj: Tensor, map_points: Tensor) -> Tensor:
        """Predict ego-agent future trajectory conditioned on map + agent polylines.

        Parameters
        ----------
        agent_traj:
            ``(batch, n_agents, t, 2)`` per-agent observed positions (index 0
            is the ego agent).
        map_points:
            ``(batch, n_map, 2)`` heterogeneous map polyline point positions.

        Returns
        -------
        Tensor
            ``(batch, future_len, 2)`` predicted ego future trajectory.
        """
        b, n_agents, t, _ = agent_traj.shape
        map_pos = map_points
        map_feat = self.map_embed(map_points)

        flat = agent_traj.reshape(b * n_agents, t, 2)
        _, h = self.agent_embed(flat)
        agent_feat = h.squeeze(0).reshape(b, n_agents, self.dim)
        agent_pos = agent_traj[:, :, -1, :]

        agent_feat = self.scene_layer(agent_feat, agent_pos, map_feat, map_pos)
        agent_feat = self.agent_layer(agent_feat, agent_pos, agent_feat, agent_pos)

        ego_query = self.query_embed.unsqueeze(0).expand(b, -1, -1)
        ego_pos = agent_pos[:, :1, :]
        ego_out = self.mode_layer(ego_query, ego_pos, agent_feat, agent_pos)
        return self.head(ego_out.squeeze(1)).reshape(b, self.future_len, 2)


def build_hptr() -> nn.Module:
    """Build a compact HPTR heterogeneous-polyline transformer.

    Returns
    -------
    nn.Module
        Random-initialized ``HPTR`` in eval mode.
    """
    return HPTR(future_len=6, dim=32).eval()


def example_input_hptr() -> tuple[Tensor, Tensor]:
    """Example input for :func:`build_hptr`.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``(agent_traj, map_points)`` of shapes ``(2, 5, 6, 2)`` and
        ``(2, 20, 2)``.
    """
    return torch.randn(2, 5, 6, 2), torch.randn(2, 20, 2)


# ---------------------------------------------------------------------------
# InterFuser -- interpretable multi-view + LiDAR sensor-fusion transformer
# ---------------------------------------------------------------------------


class _TinyCNNBackbone(nn.Module):
    """Small strided-conv tokenizer standing in for InterFuser's ResNet backbones."""

    def __init__(self, in_ch: int = 3, dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, dim, 3, stride=2, padding=1),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Tokenize an image/BEV grid into a flat sequence of patch tokens.

        Parameters
        ----------
        x:
            ``(batch, in_ch, h, w)``.

        Returns
        -------
        Tensor
            ``(batch, h/4 * w/4, dim)`` patch tokens.
        """
        feat = self.net(x)
        return feat.flatten(2).transpose(1, 2)


class InterFuser(nn.Module):
    """Interpretable sensor-fusion transformer: multi-view + LiDAR tokens, query decoding.

    Parameters
    ----------
    dim:
        Token embedding width.
    n_waypoints:
        Number of ego waypoints decoded.
    density_grid:
        Side length of the square interpretable object-density query grid.
    """

    def __init__(self, dim: int = 32, n_waypoints: int = 4, density_grid: int = 3) -> None:
        super().__init__()
        self.dim = dim
        self.n_views = 3  # left, center, right camera
        self.rgb_backbone = _TinyCNNBackbone(3, dim)
        self.lidar_backbone = _TinyCNNBackbone(2, dim)
        self.view_embed = nn.Parameter(torch.randn(self.n_views + 1, 1, dim) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            dim, nhead=4, dim_feedforward=dim * 2, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)

        self.n_waypoints = n_waypoints
        self.density_grid = density_grid
        n_density = density_grid * density_grid
        self.query_embed = nn.Parameter(torch.randn(n_waypoints + n_density, dim) * 0.02)
        dec_layer = nn.TransformerDecoderLayer(
            dim, nhead=4, dim_feedforward=dim * 2, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=2)

        self.waypoint_head = nn.Linear(dim, 2)
        self.density_head = nn.Linear(dim, 1)

    def forward(self, views: Tensor, lidar_bev: Tensor) -> tuple[Tensor, Tensor]:
        """Fuse multi-view camera + LiDAR BEV tokens and decode waypoints + density map.

        Parameters
        ----------
        views:
            ``(batch, n_views, 3, h, w)`` multi-view RGB camera images.
        lidar_bev:
            ``(batch, 2, h, w)`` LiDAR bird's-eye-view grid (height, density
            channels).

        Returns
        -------
        tuple[Tensor, Tensor]
            ``waypoints`` of shape ``(batch, n_waypoints, 2)`` and
            ``density_map`` of shape ``(batch, density_grid, density_grid)``.
        """
        b, n_views, c, h, w = views.shape
        tokens = []
        for v in range(n_views):
            tok = self.rgb_backbone(views[:, v]) + self.view_embed[v]
            tokens.append(tok)
        lidar_tok = self.lidar_backbone(lidar_bev) + self.view_embed[n_views]
        tokens.append(lidar_tok)
        fused_in = torch.cat(tokens, dim=1)

        memory = self.encoder(fused_in)

        query = self.query_embed.unsqueeze(0).expand(b, -1, -1)
        decoded = self.decoder(query, memory)

        wp = self.waypoint_head(decoded[:, : self.n_waypoints]).cumsum(dim=1)
        density = self.density_head(decoded[:, self.n_waypoints :]).squeeze(-1)
        density = density.reshape(b, self.density_grid, self.density_grid)
        return wp, density


def build_interfuser() -> nn.Module:
    """Build a compact InterFuser interpretable sensor-fusion transformer.

    Returns
    -------
    nn.Module
        Random-initialized ``InterFuser`` in eval mode.
    """
    return InterFuser(dim=32, n_waypoints=4, density_grid=3).eval()


def example_input_interfuser() -> tuple[Tensor, Tensor]:
    """Example input for :func:`build_interfuser`.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``(views, lidar_bev)`` of shapes ``(1, 3, 3, 32, 32)`` and
        ``(1, 2, 32, 32)``.
    """
    return torch.randn(1, 3, 3, 32, 32), torch.randn(1, 2, 32, 32)


# ---------------------------------------------------------------------------
# LAV -- Learning from All Vehicles (CVPR 2022)
# ---------------------------------------------------------------------------


class DynamicPointPillar(nn.Module):
    """Scatter-max point-pillar BEV encoder (dependency-free stand-in for torch_scatter).

    Points are bucketed into a pillar grid by coordinate; per-pillar features
    are aggregated by a parallel point-wise MLP followed by a scatter-max
    reduction into each occupied pillar cell -- LAV's *dynamic* (no fixed
    per-pillar point cap) point-pillar formation.

    Parameters
    ----------
    grid:
        Side length of the square BEV pillar grid.
    feat_dim:
        Output per-pillar feature width.
    """

    def __init__(self, grid: int = 16, feat_dim: int = 32) -> None:
        super().__init__()
        self.grid = grid
        self.feat_dim = feat_dim
        self.point_mlp = nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, feat_dim))

    def forward(self, points: Tensor) -> Tensor:
        """Scatter points into a BEV pillar feature grid.

        Parameters
        ----------
        points:
            ``(batch, n_points, 3)`` raw ``(x, y, z)`` LiDAR points, with
            ``x, y`` in ``[-1, 1]`` (normalized pillar extent).

        Returns
        -------
        Tensor
            ``(batch, feat_dim, grid, grid)`` BEV pillar feature map.
        """
        b, n_pts, _ = points.shape
        feat = self.point_mlp(points)  # (b, n_pts, feat_dim)

        gx = ((points[..., 0] * 0.5 + 0.5) * (self.grid - 1)).round().long().clamp(0, self.grid - 1)
        gy = ((points[..., 1] * 0.5 + 0.5) * (self.grid - 1)).round().long().clamp(0, self.grid - 1)
        cell_idx = gy * self.grid + gx  # (b, n_pts)

        n_cells = self.grid * self.grid
        pillar = torch.full(
            (b, n_cells, self.feat_dim), float("-inf"), device=points.device, dtype=feat.dtype
        )
        idx_expand = cell_idx.unsqueeze(-1).expand(-1, -1, self.feat_dim)
        pillar.scatter_reduce_(1, idx_expand, feat, reduce="amax", include_self=True)
        pillar = torch.where(torch.isinf(pillar), torch.zeros_like(pillar), pillar)
        return pillar.transpose(1, 2).reshape(b, self.feat_dim, self.grid, self.grid)


class LAVUniPlanner(nn.Module):
    """Command-conditioned GRU planner shared across the ego vehicle and all other vehicles.

    Parameters
    ----------
    feat_dim:
        BEV feature width fed into the planner.
    n_cmds:
        Number of discrete navigation commands (branch count).
    n_plan:
        Number of future waypoints decoded per rollout.
    """

    def __init__(self, feat_dim: int = 32, n_cmds: int = 4, n_plan: int = 5) -> None:
        super().__init__()
        self.n_plan = n_plan
        self.bev_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.bev_proj = nn.Linear(feat_dim, feat_dim)
        self.cast_grus = nn.ModuleList([nn.GRUCell(2, feat_dim) for _ in range(n_cmds)])
        self.cast_heads = nn.ModuleList([nn.Linear(feat_dim, 2) for _ in range(n_cmds)])

    def forward(self, bev_feat: Tensor, cmd: Tensor) -> Tensor:
        """Roll out a command-conditioned trajectory for each agent (ego or other).

        Parameters
        ----------
        bev_feat:
            ``(batch, n_agents, feat_dim, h, w)`` per-agent cropped BEV
            feature patch.
        cmd:
            ``(batch, n_agents)`` integer command id in ``[0, n_cmds)``
            selecting the rollout branch -- the *same* planner head is
            applied per-agent, LAV's shared-planner mechanism.

        Returns
        -------
        Tensor
            ``(batch, n_agents, n_plan, 2)`` predicted future waypoints.
        """
        b, n_agents, feat_dim, h, w = bev_feat.shape
        flat_feat = bev_feat.reshape(b * n_agents, feat_dim, h, w)
        pooled = self.bev_pool(flat_feat).flatten(1)
        hidden = torch.tanh(self.bev_proj(pooled))
        cmd_flat = cmd.reshape(b * n_agents)

        step_in = torch.zeros(b * n_agents, 2, device=bev_feat.device, dtype=bev_feat.dtype)
        outputs = []
        for _ in range(self.n_plan):
            step_out = torch.zeros(b * n_agents, 2, device=bev_feat.device, dtype=bev_feat.dtype)
            new_hidden = hidden.clone()
            for branch, (gru, head) in enumerate(zip(self.cast_grus, self.cast_heads)):
                mask = cmd_flat == branch
                if not mask.any():
                    continue
                branch_hidden = gru(step_in[mask], hidden[mask])
                new_hidden[mask] = branch_hidden
                step_out[mask] = head(branch_hidden)
            hidden = new_hidden
            outputs.append(step_out)
            step_in = step_out
        return torch.stack(outputs, dim=1).reshape(b, n_agents, self.n_plan, 2)


class LAV(nn.Module):
    """LAV: point-pillar LiDAR encoder + planner shared across ego and all other vehicles.

    Parameters
    ----------
    grid:
        BEV pillar grid side length.
    feat_dim:
        Pillar / planner feature width.
    n_cmds:
        Discrete navigation command count.
    n_plan:
        Future waypoints per rollout.
    """

    def __init__(
        self, grid: int = 16, feat_dim: int = 32, n_cmds: int = 4, n_plan: int = 5
    ) -> None:
        super().__init__()
        self.grid = grid
        self.point_pillar = DynamicPointPillar(grid, feat_dim)
        self.bev_cnn = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feat_dim, feat_dim, 3, padding=1),
        )
        self.crop = nn.AdaptiveAvgPool2d((4, 4))
        self.planner = LAVUniPlanner(feat_dim, n_cmds, n_plan)

    def forward(self, lidar_points: Tensor, agent_cmds: Tensor) -> Tensor:
        """Encode LiDAR and roll out shared-planner trajectories for all vehicles.

        Parameters
        ----------
        lidar_points:
            ``(batch, n_points, 3)`` raw LiDAR point cloud.
        agent_cmds:
            ``(batch, n_agents)`` per-agent (ego first, then others) discrete
            navigation command id.

        Returns
        -------
        Tensor
            ``(batch, n_agents, n_plan, 2)`` predicted future waypoints for
            every agent, produced by the single shared planner head.
        """
        b, n_agents = agent_cmds.shape
        pillar = self.point_pillar(lidar_points)
        bev = self.bev_cnn(pillar)
        cropped = self.crop(bev)  # (b, feat_dim, 4, 4), shared BEV patch reused per agent
        per_agent_feat = cropped.unsqueeze(1).expand(-1, n_agents, -1, -1, -1)
        return self.planner(per_agent_feat, agent_cmds)


def build_lav() -> nn.Module:
    """Build a compact LAV shared-planner self-driving stack.

    Returns
    -------
    nn.Module
        Random-initialized ``LAV`` in eval mode.
    """
    return LAV(grid=16, feat_dim=32, n_cmds=4, n_plan=5).eval()


def example_input_lav() -> tuple[Tensor, Tensor]:
    """Example input for :func:`build_lav`.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``(lidar_points, agent_cmds)`` of shapes ``(1, 200, 3)`` and
        ``(1, 3)`` (integer command ids).
    """
    points = torch.rand(1, 200, 3) * 2 - 1
    cmds = torch.randint(0, 4, (1, 3))
    return points, cmds


MENAGERIE_ENTRIES = [
    ("GroupNet", "build_groupnet", "example_input_groupnet", "2022", "SEQ"),
    ("HiVT", "build_hivt", "example_input_hivt", "2022", "SEQ"),
    ("HPNet", "build_hpnet", "example_input_hpnet", "2024", "SEQ"),
    ("HPTR", "build_hptr", "example_input_hptr", "2023", "SEQ"),
    ("InterFuser", "build_interfuser", "example_input_interfuser", "2022", "VIS"),
    ("LAV", "build_lav", "example_input_lav", "2022", "VIS"),
]
