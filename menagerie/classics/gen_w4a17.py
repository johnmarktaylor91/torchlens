"""Autonomous-driving trajectory-prediction / world-model + protein-hallucination classics (batch w4a17).

Sources checked (paper + official repo README/architecture description; no clone,
no pip install -- reimplemented from scratch in base-env torch):

- VectorNet: Gao et al., CVPR 2020, arXiv:2005.04259.
  https://github.com/Liang-ZX/VectorNet (community PyTorch port; no official
  Waymo open-source release)
  https://waymo.com/blog/2020/05/vectornet/
  A *hierarchical* graph neural network over a vectorized (polyline)
  representation of HD maps and agent trajectories. Level 1: every polyline
  (a lane segment, a crosswalk boundary, one agent's trajectory, ...) is
  first cut into short vectors; a stack of **polyline subgraph** layers
  (shared MLP encoder -> node-wise max-pool aggregation -> concat with each
  node, repeated for a few layers) produces one aggregated feature per
  polyline via a final max-pool. Level 2: all per-polyline features are
  treated as nodes of a fully-connected **global interaction graph** and
  refined with self-attention (a single GNN/attention layer), letting map
  polylines and agent trajectories exchange information. The target agent's
  refined node feature is decoded (small MLP head) into a future trajectory.

- WIMP (What-If Motion Predictor): Khandelwal et al., arXiv:2008.10587.
  https://github.com/wqi/WIMP (official)
  Three components: (i) a graph-based *scene encoder* -- per-agent LSTM
  history encoders whose final states are refined by a social-interaction
  graph-attention layer (actor-actor edges) and a **polyline attention**
  layer that lets each actor attend over lane-segment (map polyline)
  embeddings, selecting the relevant subset of the road network to condition
  on; (ii) an LSTM **decoder** that autoregressively rolls out K candidate
  future trajectories, at each step re-attending to both the social context
  and the lane polylines (closing the loop so lane relevance can change over
  the rollout); (iii) because the attention weights over actors/lanes are
  explicit and differentiable, injecting or removing an actor/lane node
  changes the attended context and thus the decoded trajectory -- giving the
  model its "what-if" counterfactual capability.

- Y-Net: Mangalam et al., ICCV 2021, arXiv:2012.01526.
  https://github.com/HarshayuGirase/Human-Path-Prediction (official, coauthor repo)
  https://karttikeya.github.io/publication/ynet/
  A U-Net-based *goal + waypoint heatmap* trajectory forecaster with three
  stages sharing one encoder-decoder skeleton (hence "Y" -- one encoder trunk
  feeding two decoder heads): (1) **U_e** encodes the past-trajectory
  heatmap stacked with a semantic-segmentation feature map of the scene
  through a convolutional encoder, producing multi-scale features; (2)
  **U_g**, the *goal + waypoint decoder*, is a U-Net-style decoder (with skip
  connections from U_e) that outputs a stack of heatmaps -- one per predicted
  waypoint/goal time index -- from which endpoints are sampled by taking the
  arg-max/soft-argmax location; (3) **U_t**, the *trajectory decoder*, is a
  second U-Net-style decoder that is additionally conditioned (via extra
  input channels) on the sampled goal/waypoint heatmaps and produces
  intermediate trajectory heatmaps, refining the full path consistent with
  the sampled goal. This factorizes epistemic (goal) vs aleatoric (path)
  uncertainty into separate heatmap stages.

- Vista: Gao et al., NeurIPS 2024, arXiv:2405.17398.
  https://github.com/OpenDriveLab/Vista (official, weights on HuggingFace)
  A latent video-diffusion **driving world model**: frames are encoded to a
  latent grid by a (frozen, here randomly-initialized small) VAE-style
  encoder; a **spatio-temporal denoising U-Net** (2D conv/attention blocks
  per frame interleaved with temporal-attention blocks across the frame
  axis, à la Stable Video Diffusion) predicts the noise added to a sequence
  of latent frames conditioned on (a) the diffusion timestep, (b) a context
  frame, and (c) a versatile **action-conditioning** vector (steering angle
  / speed / goal-point / high-level command are all projected into one
  conditioning embedding and injected via adaptive layer-norm / cross
  attention). Two auxiliary losses in the paper (dynamics + structure
  preservation) act only on the training objective and do not change the
  forward architecture captured here.

- WoVoGen: Lu et al., ECCV 2024, arXiv:2312.02934.
  https://github.com/fudan-zvg/WoVoGen (official)
  A two-phase **world-volume-aware** diffusion model for multi-camera
  driving-scene video generation. Phase 1: a lightweight diffusion module
  predicts the *future 4D world volume* (a compact HD-map + occupancy grid
  tensor) conditioned on the current world volume and a control-sequence
  embedding (ego action). Phase 2: a multi-camera **latent diffusion U-Net**
  generates each camera's street-view latent, conditioned on (a) the
  predicted world volume projected into each camera's frustum (an explicit
  volume-to-2D projection/cross-attention step so every camera sees a
  geometrically consistent scene), and (b) an **inter-view attention** layer
  that lets adjacent camera latents attend to each other so neighboring
  views agree at their shared field-of-view boundary. This world-volume
  intermediary is WoVoGen's distinguishing mechanism versus generic
  multi-camera diffusion (e.g. it prevents cameras from independently
  hallucinating incompatible geometry).

- AfDesign (ColabDesign / AF-hallucination): Ovchinnikov lab, ColabDesign repo.
  https://github.com/sokrypton/ColabDesign (official, `af/` submodule); paper
  companion: Wicky/Norn/... "Hallucinating structure-conditioned protein
  design", biorxiv 10.1101/2021.11.04.467194. JAX-based; reimplemented here
  structurally in torch (the model architecture, not the JAX autodiff
  machinery, is what is captured).
  A compact single-recycle **Evoformer-lite + structure-module-lite**
  stand-in for AlphaFold's differentiable-design loop: a learnable one-hot
  sequence logits tensor (the thing being "hallucinated"/optimized by
  gradient ascent on model confidence, not by direct supervision) is
  embedded and passed through a small stack of **pair-representation
  update** blocks (each block: row-wise self-attention over the sequence
  axis biased by a pairwise feature map, then an outer-product-mean update
  that writes sequence features back into the pairwise map -> mirroring
  Evoformer's MSA/pair communication, minus the MSA axis since hallucination
  typically runs single-sequence). A shallow **structure module** stand-in
  (a stack of pairwise-biased attention + small IPA-like coordinate update)
  turns the final pair representation into a per-residue confidence
  (pLDDT-like) scalar and predicted inter-residue distance logits, which is
  what the "hallucination" loop backpropagates through to update the design
  sequence logits (that backward optimization loop itself lives outside the
  nn.Module; only the frozen-random-weight forward network is captured).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# VectorNet: hierarchical polyline-subgraph + global interaction graph
# ---------------------------------------------------------------------------


class PolylineSubgraphLayer(nn.Module):
    """One VectorNet polyline-subgraph layer: shared MLP + node-wise max-pool.

    Parameters
    ----------
    dim : int
        Input/output feature dimension per node.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(dim, dim // 2), nn.ReLU(), nn.LayerNorm(dim // 2))

    def forward(self, nodes: Tensor) -> Tensor:
        """Encode then max-pool-aggregate-and-concat nodes within a polyline.

        Parameters
        ----------
        nodes : Tensor
            Shape ``(n_polylines, n_nodes, dim)``.

        Returns
        -------
        Tensor
            Shape ``(n_polylines, n_nodes, dim)`` (concat doubles then halves
            back to keep ``dim`` fixed across layers).
        """
        encoded = self.encoder(nodes)  # (P, N, dim//2)
        agg, _ = encoded.max(dim=1, keepdim=True)  # (P, 1, dim//2)
        agg = agg.expand(-1, encoded.shape[1], -1)
        return torch.cat([encoded, agg], dim=-1)  # (P, N, dim)


class VectorNet(nn.Module):
    """VectorNet: polyline subgraphs feeding a global interaction graph.

    Parameters
    ----------
    vec_dim : int
        Raw per-vector feature dimension (e.g. start/end xy + attributes).
    hidden : int
        Hidden width used throughout the subgraph and global stages.
    n_subgraph_layers : int
        Number of stacked polyline-subgraph layers (paper default: 3).
    """

    def __init__(self, vec_dim: int = 8, hidden: int = 64, n_subgraph_layers: int = 3) -> None:
        super().__init__()
        self.input_proj = nn.Linear(vec_dim, hidden)
        self.subgraph_layers = nn.ModuleList(
            [PolylineSubgraphLayer(hidden) for _ in range(n_subgraph_layers)]
        )
        self.global_attn = nn.MultiheadAttention(hidden, num_heads=4, batch_first=True)
        self.traj_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 60))

    def forward(self, polylines: Tensor) -> Tensor:
        """Encode polylines and decode the target agent's future trajectory.

        Parameters
        ----------
        polylines : Tensor
            Shape ``(n_polylines, n_nodes_per_polyline, vec_dim)``; polyline 0
            is treated as the target agent's own trajectory polyline.

        Returns
        -------
        Tensor
            Shape ``(30, 2)`` flattened future (x, y) waypoints.
        """
        nodes = self.input_proj(polylines)
        for layer in self.subgraph_layers:
            nodes = layer(nodes)
        poly_feats, _ = nodes.max(dim=1)  # (P, hidden)
        poly_feats = poly_feats.unsqueeze(0)  # (1, P, hidden)
        refined, _ = self.global_attn(poly_feats, poly_feats, poly_feats)
        target_feat = refined[0, 0]  # target agent is polyline 0
        return self.traj_head(target_feat).view(30, 2)


def build_vectornet() -> nn.Module:
    """Build a compact VectorNet.

    Returns
    -------
    nn.Module
        ``VectorNet`` instance in eval mode.
    """
    return VectorNet(vec_dim=8, hidden=64, n_subgraph_layers=3).eval()


def example_input_vectornet() -> Tensor:
    """Example scene: 12 polylines (1 agent + 11 map/other-agent polylines) of 9 vectors each.

    Returns
    -------
    Tensor
        Shape ``(12, 9, 8)``.
    """
    return torch.randn(12, 9, 8)


# ---------------------------------------------------------------------------
# WIMP: graph-attentional social + lane-polyline encoder, LSTM decoder
# ---------------------------------------------------------------------------


class WIMP(nn.Module):
    """WIMP: LSTM social/lane encoder + counterfactual-attention LSTM decoder.

    Parameters
    ----------
    n_actors : int
        Number of interacting actors in the scene (including the target).
    n_lanes : int
        Number of candidate lane-segment polylines.
    hidden : int
        LSTM/attention hidden size.
    horizon : int
        Number of future decoding steps.
    """

    def __init__(
        self, n_actors: int = 6, n_lanes: int = 10, hidden: int = 48, horizon: int = 20
    ) -> None:
        super().__init__()
        self.hidden = hidden
        self.horizon = horizon
        self.actor_encoder = nn.LSTM(2, hidden, batch_first=True)
        self.lane_encoder = nn.Linear(4, hidden)
        self.social_attn = nn.MultiheadAttention(hidden, num_heads=4, batch_first=True)
        self.lane_attn = nn.MultiheadAttention(hidden, num_heads=4, batch_first=True)
        self.decoder_cell = nn.LSTMCell(2, hidden)
        self.out_proj = nn.Linear(hidden, 2)

    def forward(self, actor_hist: Tensor, lane_polylines: Tensor) -> Tensor:
        """Encode social + lane context then autoregressively decode the target's future path.

        Parameters
        ----------
        actor_hist : Tensor
            Shape ``(n_actors, t_hist, 2)`` past (x, y) per actor; actor 0 is the target.
        lane_polylines : Tensor
            Shape ``(n_lanes, 4)`` lane-segment features (e.g. start/end xy).

        Returns
        -------
        Tensor
            Shape ``(horizon, 2)`` predicted future (x, y) offsets for the target actor.
        """
        _, (h_n, _) = self.actor_encoder(actor_hist)  # (1, n_actors, hidden)
        actor_feats = h_n.transpose(0, 1)  # (n_actors, 1, hidden) -> squeeze below
        actor_feats = actor_feats.squeeze(1).unsqueeze(0)  # (1, n_actors, hidden)
        social_ctx, _ = self.social_attn(actor_feats, actor_feats, actor_feats)

        lane_feats = self.lane_encoder(lane_polylines).unsqueeze(0)  # (1, n_lanes, hidden)
        target_q = social_ctx[:, :1]  # (1, 1, hidden)
        lane_ctx, _ = self.lane_attn(target_q, lane_feats, lane_feats)

        h = (social_ctx[0, 0] + lane_ctx[0, 0]).unsqueeze(0)  # (1, hidden)
        c = torch.zeros_like(h)
        step_in = actor_hist[0, -1].unsqueeze(0)  # (1, 2) last observed target position
        outputs = []
        for _ in range(self.horizon):
            h, c = self.decoder_cell(step_in, (h, c))
            delta = self.out_proj(h)
            outputs.append(delta)
            step_in = delta
        return torch.cat(outputs, dim=0)


def build_wimp() -> nn.Module:
    """Build a compact WIMP.

    Returns
    -------
    nn.Module
        ``WIMP`` instance in eval mode.
    """
    return WIMP(n_actors=6, n_lanes=10, hidden=48, horizon=20).eval()


def example_input_wimp() -> tuple[Tensor, Tensor]:
    """Example scene with 6 actors (20 history steps) and 10 lane segments.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``(actor_hist, lane_polylines)`` of shapes ``(6, 20, 2)`` and ``(10, 4)``.
    """
    return torch.randn(6, 20, 2), torch.randn(10, 4)


# ---------------------------------------------------------------------------
# Y-Net: shared encoder + goal/waypoint U-Net head + trajectory U-Net head
# ---------------------------------------------------------------------------


class ConvBlock(nn.Module):
    """Two 3x3 convolutions with ReLU, used throughout the Y-Net U-Nets."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply the two convolutions.

        Parameters
        ----------
        x : Tensor
            Shape ``(B, in_ch, H, W)``.

        Returns
        -------
        Tensor
            Shape ``(B, out_ch, H, W)``.
        """
        return self.net(x)


class YNet(nn.Module):
    """Y-Net: shared U-Net encoder trunk with two U-Net decoder heads (goal, trajectory).

    Parameters
    ----------
    in_ch : int
        Input channels (past-trajectory heatmap + semantic map channels).
    base : int
        Base channel width.
    n_waypoints : int
        Number of goal/waypoint heatmap channels the goal head predicts.
    """

    def __init__(self, in_ch: int = 4, base: int = 8, n_waypoints: int = 3) -> None:
        super().__init__()
        self.n_waypoints = n_waypoints
        # Shared encoder U_e (2-level).
        self.enc1 = ConvBlock(in_ch, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.pool = nn.MaxPool2d(2)

        # Goal/waypoint decoder U_g.
        self.goal_up = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.goal_dec = ConvBlock(base * 2, base)
        self.goal_head = nn.Conv2d(base, n_waypoints, 1)

        # Trajectory decoder U_t, conditioned on the goal/waypoint heatmaps
        # via extra input channels concatenated at the bottleneck.
        self.traj_bottleneck = nn.Conv2d(base * 2 + n_waypoints, base * 2, 1)
        self.traj_up = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.traj_dec = ConvBlock(base * 2, base)
        self.traj_head = nn.Conv2d(base, n_waypoints, 1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Predict goal/waypoint heatmaps and goal-conditioned trajectory heatmaps.

        Parameters
        ----------
        x : Tensor
            Shape ``(B, in_ch, H, W)`` stacked past-trajectory + semantic-map heatmap.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(goal_heatmaps, traj_heatmaps)``, each ``(B, n_waypoints, H, W)``.
        """
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))

        g_up = self.goal_up(e2)
        g_dec = self.goal_dec(torch.cat([g_up, e1], dim=1))
        goal_hm = self.goal_head(g_dec)

        bottleneck = self.traj_bottleneck(
            torch.cat([e2, F.adaptive_avg_pool2d(goal_hm, e2.shape[-2:])], dim=1)
        )
        t_up = self.traj_up(bottleneck)
        t_dec = self.traj_dec(torch.cat([t_up, e1], dim=1))
        traj_hm = self.traj_head(t_dec)
        return goal_hm, traj_hm


def build_ynet() -> nn.Module:
    """Build a compact Y-Net.

    Returns
    -------
    nn.Module
        ``YNet`` instance in eval mode.
    """
    return YNet(in_ch=4, base=8, n_waypoints=3).eval()


def example_input_ynet() -> Tensor:
    """Example stacked past-trajectory + semantic-map heatmap.

    Returns
    -------
    Tensor
        Shape ``(1, 4, 32, 32)``.
    """
    return torch.randn(1, 4, 32, 32)


# ---------------------------------------------------------------------------
# Vista: action-conditioned spatio-temporal latent-diffusion world model
# ---------------------------------------------------------------------------


class TemporalAttentionBlock(nn.Module):
    """Self-attention across the frame axis (shared across all spatial positions)."""

    def __init__(self, dim: int, n_heads: int = 4) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)

    def forward(self, x: Tensor) -> Tensor:
        """Apply temporal self-attention.

        Parameters
        ----------
        x : Tensor
            Shape ``(B, T, HW, C)``.

        Returns
        -------
        Tensor
            Shape ``(B, T, HW, C)``.
        """
        b, t, hw, c = x.shape
        flat = x.permute(0, 2, 1, 3).reshape(b * hw, t, c)  # (B*HW, T, C)
        normed = self.norm(flat)
        attended, _ = self.attn(normed, normed, normed)
        out = flat + attended
        return out.reshape(b, hw, t, c).permute(0, 2, 1, 3)


class SpatialConvBlock(nn.Module):
    """Per-frame spatial conv block applied independently to each frame."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm = nn.GroupNorm(4, channels)

    def forward(self, x: Tensor) -> Tensor:
        """Apply a per-frame residual conv.

        Parameters
        ----------
        x : Tensor
            Shape ``(B, T, C, H, W)``.

        Returns
        -------
        Tensor
            Shape ``(B, T, C, H, W)``.
        """
        b, t, c, h, w = x.shape
        flat = x.reshape(b * t, c, h, w)
        out = flat + self.conv(F.silu(self.norm(flat)))
        return out.reshape(b, t, c, h, w)


def sinusoidal_embedding(timesteps: Tensor, dim: int) -> Tensor:
    """Standard sinusoidal timestep embedding (DDPM-style).

    Parameters
    ----------
    timesteps : Tensor
        Shape ``(B,)`` integer/float diffusion timesteps.
    dim : int
        Embedding dimension.

    Returns
    -------
    Tensor
        Shape ``(B, dim)``.
    """
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32) / half)
    args = timesteps[:, None].float() * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class Vista(nn.Module):
    """Vista: action-conditioned spatio-temporal denoising U-Net over latent video frames.

    Parameters
    ----------
    latent_ch : int
        Latent channel width (VAE-latent stand-in).
    hidden : int
        Internal conv/attention width.
    action_dim : int
        Dimension of the versatile action-conditioning vector.
    """

    def __init__(self, latent_ch: int = 4, hidden: int = 16, action_dim: int = 6) -> None:
        super().__init__()
        self.hidden = hidden
        self.in_proj = nn.Conv2d(latent_ch, hidden, 3, padding=1)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.action_mlp = nn.Sequential(
            nn.Linear(action_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.spatial1 = SpatialConvBlock(hidden)
        self.temporal1 = TemporalAttentionBlock(hidden)
        self.spatial2 = SpatialConvBlock(hidden)
        self.temporal2 = TemporalAttentionBlock(hidden)
        self.out_proj = nn.Conv2d(hidden, latent_ch, 3, padding=1)

    def forward(self, latents: Tensor, timestep: Tensor, action: Tensor) -> Tensor:
        """Predict the denoising residual for a sequence of latent frames.

        Parameters
        ----------
        latents : Tensor
            Shape ``(B, T, latent_ch, H, W)`` noisy latent video.
        timestep : Tensor
            Shape ``(B,)`` diffusion timestep.
        action : Tensor
            Shape ``(B, action_dim)`` action-conditioning vector.

        Returns
        -------
        Tensor
            Shape ``(B, T, latent_ch, H, W)`` predicted noise.
        """
        b, t, c, h, w = latents.shape
        x = self.in_proj(latents.reshape(b * t, c, h, w)).reshape(b, t, self.hidden, h, w)

        cond = self.time_mlp(sinusoidal_embedding(timestep, self.hidden)) + self.action_mlp(action)
        x = x + cond[:, None, :, None, None]

        x = self.spatial1(x)
        x = self.temporal1(x.permute(0, 1, 3, 4, 2).reshape(b, t, h * w, self.hidden))
        x = x.reshape(b, t, h, w, self.hidden).permute(0, 1, 4, 2, 3)
        x = self.spatial2(x)
        x = self.temporal2(x.permute(0, 1, 3, 4, 2).reshape(b, t, h * w, self.hidden))
        x = x.reshape(b, t, h, w, self.hidden).permute(0, 1, 4, 2, 3)

        return self.out_proj(x.reshape(b * t, self.hidden, h, w)).reshape(b, t, c, h, w)


def build_vista() -> nn.Module:
    """Build a compact Vista driving world model.

    Returns
    -------
    nn.Module
        ``Vista`` instance in eval mode.
    """
    return Vista(latent_ch=4, hidden=16, action_dim=6).eval()


def example_input_vista() -> tuple[Tensor, Tensor, Tensor]:
    """Example noisy 4-frame latent clip, diffusion timestep, and action vector.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        ``(latents, timestep, action)`` of shapes ``(1, 4, 4, 8, 8)``, ``(1,)``, ``(1, 6)``.
    """
    return torch.randn(1, 4, 4, 8, 8), torch.randint(0, 1000, (1,)), torch.randn(1, 6)


# ---------------------------------------------------------------------------
# WoVoGen: world-volume prediction + volume-conditioned multi-camera diffusion
# ---------------------------------------------------------------------------


class WorldVolumePredictor(nn.Module):
    """Phase 1: predict the future 4D world volume from the current volume + ego action."""

    def __init__(self, vol_ch: int = 6, hidden: int = 16, action_dim: int = 4) -> None:
        super().__init__()
        self.action_proj = nn.Linear(action_dim, vol_ch)
        self.net = nn.Sequential(
            nn.Conv3d(vol_ch, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, vol_ch, 3, padding=1),
        )

    def forward(self, world_volume: Tensor, action: Tensor) -> Tensor:
        """Predict the next world volume.

        Parameters
        ----------
        world_volume : Tensor
            Shape ``(B, vol_ch, Z, H, W)`` current HD-map + occupancy volume.
        action : Tensor
            Shape ``(B, action_dim)`` ego control sequence embedding.

        Returns
        -------
        Tensor
            Shape ``(B, vol_ch, Z, H, W)`` predicted future world volume.
        """
        a = self.action_proj(action)[:, :, None, None, None]
        return world_volume + self.net(world_volume + a)


class InterViewAttention(nn.Module):
    """Lets neighboring camera-view latents attend to each other (ring topology)."""

    def __init__(self, dim: int, n_heads: int = 4) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)

    def forward(self, views: Tensor) -> Tensor:
        """Cross-attend each camera view's tokens against all other views.

        Parameters
        ----------
        views : Tensor
            Shape ``(n_views, HW, C)``.

        Returns
        -------
        Tensor
            Shape ``(n_views, HW, C)``.
        """
        n_views, hw, c = views.shape
        flat = views.reshape(1, n_views * hw, c)
        normed = self.norm(flat)
        attended, _ = self.attn(normed, normed, normed)
        return (flat + attended).reshape(n_views, hw, c)


class WoVoGen(nn.Module):
    """WoVoGen: world-volume prediction + volume-projected multi-camera diffusion U-Net.

    Parameters
    ----------
    vol_ch : int
        World-volume channel width.
    img_ch : int
        Per-camera latent channel width.
    hidden : int
        Internal conv width.
    n_views : int
        Number of cameras.
    """

    def __init__(
        self, vol_ch: int = 6, img_ch: int = 4, hidden: int = 12, n_views: int = 4
    ) -> None:
        super().__init__()
        self.n_views = n_views
        self.volume_predictor = WorldVolumePredictor(vol_ch=vol_ch, hidden=hidden, action_dim=4)
        # Volume -> per-camera projection: collapse the depth (Z) axis via a
        # learned linear combination standing in for frustum projection.
        self.volume_to_camera = nn.Conv2d(vol_ch, hidden, 3, padding=1)
        self.img_in = nn.Conv2d(img_ch, hidden, 3, padding=1)
        self.fuse = nn.Conv2d(hidden * 2, hidden, 3, padding=1)
        self.inter_view_attn = InterViewAttention(hidden)
        self.out_proj = nn.Conv2d(hidden, img_ch, 3, padding=1)

    def forward(
        self, world_volume: Tensor, camera_latents: Tensor, action: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Predict the future world volume and denoise multi-camera latents.

        Parameters
        ----------
        world_volume : Tensor
            Shape ``(1, vol_ch, Z, H, W)`` current world volume.
        camera_latents : Tensor
            Shape ``(n_views, img_ch, H, W)`` per-camera noisy latents.
        action : Tensor
            Shape ``(1, 4)`` ego control-sequence embedding.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(future_world_volume, camera_noise_pred)`` of shapes
            ``(1, vol_ch, Z, H, W)`` and ``(n_views, img_ch, H, W)``.
        """
        future_volume = self.volume_predictor(world_volume, action)
        volume_bev = future_volume.mean(
            dim=2
        )  # (1, vol_ch, H, W) -- collapse Z as a frustum-projection stand-in
        volume_feat = self.volume_to_camera(volume_bev)  # (1, hidden, H, W)
        volume_feat = volume_feat.expand(self.n_views, -1, -1, -1)

        img_feat = self.img_in(camera_latents)  # (n_views, hidden, H, W)
        fused = self.fuse(torch.cat([img_feat, volume_feat], dim=1))

        n, c, h, w = fused.shape
        tokens = fused.reshape(n, c, h * w).permute(0, 2, 1)  # (n_views, HW, C)
        attended = self.inter_view_attn(tokens)
        attended = attended.permute(0, 2, 1).reshape(n, c, h, w)

        noise_pred = self.out_proj(fused + attended)
        return future_volume, noise_pred


def build_wovogen() -> nn.Module:
    """Build a compact WoVoGen.

    Returns
    -------
    nn.Module
        ``WoVoGen`` instance in eval mode.
    """
    return WoVoGen(vol_ch=6, img_ch=4, hidden=12, n_views=4).eval()


def example_input_wovogen() -> tuple[Tensor, Tensor, Tensor]:
    """Example current world volume, 4-camera noisy latents, and ego action.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        ``(world_volume, camera_latents, action)`` of shapes ``(1, 6, 4, 8, 8)``,
        ``(4, 4, 8, 8)``, ``(1, 4)``.
    """
    return torch.randn(1, 6, 4, 8, 8), torch.randn(4, 4, 8, 8), torch.randn(1, 4)


# ---------------------------------------------------------------------------
# AfDesign / ColabDesign hallucination: Evoformer-lite pair-update + structure module
# ---------------------------------------------------------------------------


class PairUpdateBlock(nn.Module):
    """Evoformer-lite block: sequence self-attention biased by pair features + outer-product-mean write-back."""

    def __init__(self, seq_dim: int, pair_dim: int, n_heads: int = 4) -> None:
        super().__init__()
        self.seq_norm = nn.LayerNorm(seq_dim)
        self.pair_bias_proj = nn.Linear(pair_dim, n_heads)
        self.n_heads = n_heads
        self.head_dim = seq_dim // n_heads
        self.qkv = nn.Linear(seq_dim, seq_dim * 3)
        self.attn_out = nn.Linear(seq_dim, seq_dim)

        self.pair_norm = nn.LayerNorm(pair_dim)
        self.outer_proj = nn.Linear(seq_dim, pair_dim)
        self.pair_update = nn.Linear(pair_dim * 2, pair_dim)

    def forward(self, seq: Tensor, pair: Tensor) -> tuple[Tensor, Tensor]:
        """Update sequence features (pair-biased attention) and pair features (outer-product-mean).

        Parameters
        ----------
        seq : Tensor
            Shape ``(L, seq_dim)`` per-residue sequence features.
        pair : Tensor
            Shape ``(L, L, pair_dim)`` pairwise features.

        Returns
        -------
        tuple[Tensor, Tensor]
            Updated ``(seq, pair)`` with the same shapes.
        """
        length = seq.shape[0]
        normed = self.seq_norm(seq)
        qkv = self.qkv(normed).view(length, 3, self.n_heads, self.head_dim)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # each (L, n_heads, head_dim)

        bias = self.pair_bias_proj(pair)  # (L, L, n_heads)
        scores = torch.einsum("ihd,jhd->hij", q, k) / math.sqrt(self.head_dim)
        scores = scores + bias.permute(2, 0, 1)
        attn = scores.softmax(dim=-1)
        out = torch.einsum("hij,jhd->ihd", attn, v).reshape(length, -1)
        seq = seq + self.attn_out(out)

        outer = self.outer_proj(seq)  # (L, pair_dim)
        outer_pair = outer.unsqueeze(1) * outer.unsqueeze(0)  # (L, L, pair_dim)
        pair = pair + self.pair_update(torch.cat([self.pair_norm(pair), outer_pair], dim=-1))
        return seq, pair


class AfDesignHallucination(nn.Module):
    """AfDesign-style hallucination network: Evoformer-lite pair updates + structure-module-lite head.

    The module the search/gradient loop optimizes is a learnable one-hot-ish
    sequence-logits tensor; this ``nn.Module`` captures the forward network
    whose confidence output that outer optimization loop backpropagates
    through (the optimization loop itself is not part of the captured graph).

    Parameters
    ----------
    vocab : int
        Amino-acid alphabet size (20 standard residues).
    seq_dim : int
        Per-residue sequence-embedding width.
    pair_dim : int
        Pairwise-embedding width.
    n_blocks : int
        Number of stacked pair-update (Evoformer-lite) blocks.
    n_bins : int
        Number of inter-residue distance bins predicted by the structure head.
    """

    def __init__(
        self,
        vocab: int = 20,
        seq_dim: int = 32,
        pair_dim: int = 16,
        n_blocks: int = 2,
        n_bins: int = 16,
    ) -> None:
        super().__init__()
        self.seq_embed = nn.Linear(vocab, seq_dim)
        self.pair_init = nn.Linear(2, pair_dim)
        self.blocks = nn.ModuleList([PairUpdateBlock(seq_dim, pair_dim) for _ in range(n_blocks)])
        self.plddt_head = nn.Sequential(
            nn.Linear(seq_dim, seq_dim // 2), nn.ReLU(), nn.Linear(seq_dim // 2, 1)
        )
        self.distance_head = nn.Linear(pair_dim, n_bins)

    def forward(self, seq_logits: Tensor) -> tuple[Tensor, Tensor]:
        """Fold a soft (hallucinated) sequence and predict per-residue confidence + inter-residue distances.

        Parameters
        ----------
        seq_logits : Tensor
            Shape ``(L, vocab)`` softmax'd (relaxed one-hot) sequence logits
            -- the tensor a hallucination loop would optimize via gradient
            ascent on the confidence output.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(plddt, distance_logits)`` of shapes ``(L,)`` and ``(L, L, n_bins)``.
        """
        length = seq_logits.shape[0]
        probs = seq_logits.softmax(dim=-1)
        seq = self.seq_embed(probs)

        idx = torch.arange(length, dtype=torch.float32)
        rel_pos = (idx[:, None] - idx[None, :]).unsqueeze(-1)
        pair_seed = torch.cat([rel_pos, rel_pos.abs()], dim=-1)
        pair = self.pair_init(pair_seed)

        for block in self.blocks:
            seq, pair = block(seq, pair)

        plddt = self.plddt_head(seq).squeeze(-1).sigmoid()
        distance_logits = self.distance_head(pair)
        return plddt, distance_logits


def build_afdesign_hallucination() -> nn.Module:
    """Build a compact AfDesign-style hallucination network.

    Returns
    -------
    nn.Module
        ``AfDesignHallucination`` instance in eval mode.
    """
    return AfDesignHallucination(vocab=20, seq_dim=32, pair_dim=16, n_blocks=2, n_bins=16).eval()


def example_input_afdesign_hallucination() -> Tensor:
    """Example relaxed one-hot sequence logits for a 24-residue design.

    Returns
    -------
    Tensor
        Shape ``(24, 20)``.
    """
    return torch.randn(24, 20)


MENAGERIE_ENTRIES = [
    ("VectorNet", "build_vectornet", "example_input_vectornet", "2020", "SEQ"),
    ("WIMP", "build_wimp", "example_input_wimp", "2020", "SEQ"),
    ("Y-Net", "build_ynet", "example_input_ynet", "2021", "VIS"),
    ("Vista", "build_vista", "example_input_vista", "2024", "GEN"),
    ("WoVoGen", "build_wovogen", "example_input_wovogen", "2024", "GEN"),
    (
        "AfDesign hallucination",
        "build_afdesign_hallucination",
        "example_input_afdesign_hallucination",
        "2021",
        "BIO",
    ),
]
