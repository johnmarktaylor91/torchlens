"""Autonomous-driving perception / trajectory-prediction / traffic-simulation
classics (batch w4a15).

Sources checked (paper + official repo README/model-file architecture; no
clone, no pip install -- reimplemented from scratch in base-env torch):

- SparseBEV: Liu, Teng, Lu, Wang & Wang, ICCV 2023, arXiv:2308.09244.
  https://github.com/MCG-NJU/SparseBEV
  Fully sparse multi-camera 3D object detection: a fixed set of learned 3D
  box queries (center/size/rotation) are refined over decoder layers with no
  dense BEV grid at all. Each layer (1) predicts per-query, per-group,
  per-frame 3D offsets around the query's current box (``make_sample_points``
  in the official code), (2) projects those points into every camera view
  with a lidar2img matrix and keeps only the single valid (in-frustum) view
  per point ("adaptive spatio-temporal sampling"), (3) bilinearly samples
  multi-scale image features at the projected pixel locations and mixes them
  with *learned, query-conditioned scale weights* (multi-scale multi-view,
  or "MSMV", sampling), and (4) refines the query embedding with
  self-attention across queries plus a box-regression FFN before predicting
  the next layer's offsets. Reimplemented here with a single synthetic
  "camera feature map" standing in for the multi-camera backbone output and
  a batched ``grid_sample`` standing in for the custom ``msmv_sampling`` CUDA
  op, since the distinguishing mechanism is the learned-offset adaptive
  sampling + query-conditioned scale mixing, not the specific camera rig.

- STGAT: Huang, Bi, Li, Mao & Wang, ICCV 2019, arXiv:1907.01931.
  https://github.com/huang-xx/STGAT (official, paper author)
  Spatial-Temporal Graph Attention for pedestrian trajectory prediction: a
  per-agent LSTM encodes each pedestrian's observed trajectory step by step;
  at *every* observed timestep, the current per-agent hidden states of all
  pedestrians present in the scene are treated as nodes of a fully-connected
  graph and passed through a multi-head Graph Attention (GAT) layer, so
  spatial interaction is re-computed at every timestep (not once at the
  end); the resulting per-timestep graph-attention embeddings are fed
  through a second, temporal LSTM to summarize the interaction history, and
  the final trajectory/social embedding (plus injected Gaussian noise, GAN-
  style) seeds an autoregressive GRU/LSTM decoder that rolls out future
  positions. The alternating per-timestep GAT + temporal LSTM is STGAT's
  namesake spatial-temporal mechanism (official code: ``BatchMultiHeadGraphAttention``
  + ``GATEncoder`` interleaved with ``traj_lstm_model``/``graph_lstm_model``).

- T4P (Test-Time Training for Trajectory Prediction): Park, Jeong, Yoon,
  Jeong & Yoon, CVPR 2024, arXiv:2403.10052.
  https://github.com/daeheepark/T4P
  Built on a ForecastMAE-style masked-autoencoder trajectory backbone, T4P
  adds *actor-specific token memory* for online test-time training: a small
  external memory bank of learned tokens is queried (via attention) by each
  actor's current encoded trajectory to retrieve a personalized correction
  token, which is fused back into the actor's embedding before decoding.
  During self-supervised test-time adaptation, a random subset of the
  observed-trajectory timesteps is masked (dropped) and the encoder/decoder
  must reconstruct them, updating the memory bank online per test sequence
  without ground-truth futures. Reimplemented here as a Transformer MAE
  trajectory encoder-decoder plus a learned memory-token cross-attention
  retrieval module, with the masked-reconstruction path exposed as the
  default forward (the mechanism the architecture is defined by).

- TCP (Trajectory-guided Control Prediction): Wu, Jia, Mao, Weng, Li, Tian,
  Chen & Li, NeurIPS 2022, arXiv:2206.08129.
  https://github.com/OpenDriveLab/TCP (official, CARLA leaderboard 1st)
  Single front-camera end-to-end driving with *two coupled output branches*
  sharing one CNN perception backbone: a trajectory branch autoregressively
  decodes future waypoints with a GRUCell (``decoder_traj``), and a control
  branch autoregressively decodes low-level action distributions (steer/
  throttle/brake, as Beta-distribution parameters) with a second GRUCell
  (``decoder_ctrl``); crucially the control branch's every step attends back
  over the *trajectory branch's per-step hidden states* via a learned
  softmax attention (``wp_att``) that re-weights a spatial feature map, so
  the trajectory prediction directly guides (distills into) the control
  prediction rather than the two branches being independent heads. That
  trajectory->control cross-branch attention is TCP's distinguishing
  mechanism, reimplemented compactly here with a synthetic CNN backbone.

- ThinkTwice: Jia, Wu, Mao, Tian, Li & Yan, CVPR 2023, arXiv:2305.06242.
  https://github.com/OpenDriveLab/ThinkTwice
  A "scalable decoder" for end-to-end BEV driving built as a coarse-then-
  repeated-refine cascade: a coarse head regresses an initial waypoint +
  control proposal from a fused BEV feature, then several *refinement
  stages* run in sequence, each of which (1) projects the current proposal's
  waypoints as reference points into the (synthetic) camera/BEV feature map
  and samples "look" features there with a proposal-conditioned attention
  (``LookModule``/``SpatialCrossAttention`` + ``grid_sample`` in the official
  code), (2) feeds the looked-up features through a small spatial GRU
  (``SpatialGRU``) that updates a per-cell BEV feature state conditioned on
  the current waypoint + control, and (3) predicts a *residual* correction
  to the proposal from the updated state. Each refinement stage "looks
  twice" -- once to gather proposal-conditioned evidence, once to update and
  correct -- which is the paper's namesake scalable-decoder mechanism,
  reimplemented compactly here with a small synthetic BEV grid.

- TrafficBots: Zhang, Liniger, Dai, Yu & Van Gool, ICRA 2023, arXiv:2303.04116.
  https://github.com/zhejz/TrafficBots
  Multi-agent traffic simulation as a *world model*: every agent is given a
  learned per-agent, per-episode "personality" latent sampled from a
  Conditional-VAE posterior (encoded from that agent's full ground-truth
  future during training, prior-only at simulation time), which is
  concatenated into every agent's Transformer scene-encoding token so the
  same encoder/decoder produces agent-specific behavior. A Transformer scene
  encoder performs agent-agent and agent-map self-attention per simulation
  step, and a GRU-based closed-loop action decoder rolls the whole
  multi-agent scene forward step by step (each step's output feeds back in
  as next step's input), so all agents are unrolled jointly and consistently
  rather than independently per-agent. Reimplemented here with the CVAE
  personality-latent injection + closed-loop multi-agent GRU rollout as the
  defining mechanism, using a compact synthetic map/agent context encoder.

Random init, tiny dims, CPU-only -- architecture catalog entries, not trained
weights.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# SparseBEV
# ---------------------------------------------------------------------------


class SparseBEVLayer(nn.Module):
    """One SparseBEV decoder layer: adaptive sampling + MSMV mixing + refine."""

    def __init__(
        self, dim: int = 32, num_points: int = 4, num_groups: int = 2, num_levels: int = 2
    ) -> None:
        """Build the offset/scale-weight heads, self-attention and box FFN.

        Parameters
        ----------
        dim:
            Query embedding width.
        num_points:
            Sample points per group per query.
        num_groups:
            Number of independent sampling groups (AdaMixer-style).
        num_levels:
            Number of synthetic feature-pyramid levels to mix over.
        """
        super().__init__()
        self.num_points = num_points
        self.num_groups = num_groups
        self.num_levels = num_levels
        self.offset_head = nn.Linear(dim, num_groups * num_points * 2)
        self.scale_weight_head = nn.Linear(dim, num_groups * num_points * num_levels)
        self.sample_proj = nn.Linear(num_groups * num_points * dim, dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.box_ffn = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.ReLU(inplace=True), nn.Linear(dim * 2, 4)
        )
        self.query_ffn = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.ReLU(inplace=True), nn.Linear(dim * 2, dim)
        )

    def forward(
        self, query: Tensor, query_bbox: Tensor, mlvl_feats: list[Tensor]
    ) -> tuple[Tensor, Tensor]:
        """Refine ``query`` and ``query_bbox`` by one adaptive-sampling layer.

        Parameters
        ----------
        query:
            Query embeddings, shape ``(batch, num_queries, dim)``.
        query_bbox:
            Current normalized 2D box center per query, ``(batch, num_queries, 4)``
            (cx, cy, unused, unused -- kept 4-wide to mirror the 3D box head).
        mlvl_feats:
            List of ``num_levels`` synthetic camera feature maps, each
            ``(batch, dim, H, W)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Updated ``(query, query_bbox)``.
        """
        batch, num_q, dim = query.shape
        centers = query_bbox[..., :2]

        offsets = self.offset_head(query).view(batch, num_q, self.num_groups, self.num_points, 2)
        sample_xy = centers[:, :, None, None, :] + 0.2 * torch.tanh(offsets)
        sample_xy = sample_xy.clamp(-1.0, 1.0)

        scale_weights = self.scale_weight_head(query)
        scale_weights = scale_weights.view(
            batch, num_q, self.num_groups, self.num_points, self.num_levels
        )
        scale_weights = F.softmax(scale_weights, dim=-1)

        grid = sample_xy.reshape(batch, num_q * self.num_groups * self.num_points, 1, 2)
        mixed = 0.0
        for lvl, feat in enumerate(mlvl_feats):
            sampled = F.grid_sample(feat, grid, align_corners=False, mode="bilinear")
            sampled = sampled.squeeze(-1).permute(0, 2, 1)
            sampled = sampled.view(batch, num_q, self.num_groups, self.num_points, dim)
            weight = scale_weights[..., lvl].unsqueeze(-1)
            mixed = mixed + sampled * weight

        sampled_feat = mixed.reshape(batch, num_q, self.num_groups * self.num_points * dim)
        sampled_feat = self.sample_proj(sampled_feat)
        query = self.norm1(query + sampled_feat)

        attn_out, _ = self.self_attn(query, query, query)
        query = self.norm2(query + attn_out)
        query = query + self.query_ffn(query)

        box_delta = self.box_ffn(query)
        query_bbox = query_bbox + 0.1 * box_delta
        return query, query_bbox


class SparseBEV(nn.Module):
    """SparseBEV: sparse learned 3D box queries refined via adaptive sampling."""

    def __init__(
        self,
        dim: int = 32,
        num_queries: int = 12,
        num_layers: int = 3,
        num_levels: int = 2,
        num_frames: int = 2,
    ) -> None:
        """Build the query embeddings and stacked decoder layers.

        Parameters
        ----------
        dim:
            Query / feature embedding width.
        num_queries:
            Number of learned 3D box queries.
        num_layers:
            Number of iterative refinement layers.
        num_levels:
            Number of synthetic feature-pyramid levels per frame.
        num_frames:
            Number of temporal frames of camera features to fuse.
        """
        super().__init__()
        self.num_levels = num_levels
        self.num_frames = num_frames
        self.query_embed = nn.Parameter(torch.randn(1, num_queries, dim) * 0.02)
        self.bbox_embed = nn.Parameter(torch.rand(1, num_queries, 4) * 2 - 1)
        self.layers = nn.ModuleList(
            [SparseBEVLayer(dim, num_levels=num_levels) for _ in range(num_layers)]
        )
        self.cls_head = nn.Linear(dim, 1)

    def forward(self, mlvl_feats: list[Tensor]) -> tuple[Tensor, Tensor]:
        """Run the sparse decoder over multi-level, multi-frame camera features.

        Parameters
        ----------
        mlvl_feats:
            List of ``num_frames * num_levels`` feature maps, each
            ``(batch, dim, H, W)``, temporal frames concatenated with levels
            and mixed jointly (temporal fusion folded into level mixing for
            compactness).

        Returns
        -------
        tuple[Tensor, Tensor]
            Final query embeddings ``(batch, num_queries, dim)`` and
            classification logits ``(batch, num_queries, 1)``.
        """
        batch = mlvl_feats[0].shape[0]
        query = self.query_embed.expand(batch, -1, -1)
        query_bbox = self.bbox_embed.expand(batch, -1, -1)
        for layer in self.layers:
            query, query_bbox = layer(query, query_bbox, mlvl_feats)
        logits = self.cls_head(query)
        return query_bbox, logits


def build_sparsebev() -> nn.Module:
    """Build a compact SparseBEV model.

    Returns
    -------
    nn.Module
        Random-initialized ``SparseBEV`` in eval mode.
    """
    return SparseBEV().eval()


def example_input_sparsebev() -> list[Tensor]:
    """Create example multi-level camera feature maps.

    Returns
    -------
    list[Tensor]
        Two synthetic feature-pyramid levels, each ``(1, 32, 12, 16)``.
    """
    return [torch.randn(1, 32, 12, 16), torch.randn(1, 32, 6, 8)]


# ---------------------------------------------------------------------------
# STGAT
# ---------------------------------------------------------------------------


class GraphAttentionLayer(nn.Module):
    """Single-head graph attention over all agents present at one timestep."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        """Build the linear projection and additive attention scoring.

        Parameters
        ----------
        in_dim:
            Input per-agent feature width.
        out_dim:
            Output per-agent feature width.
        """
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.attn_src = nn.Linear(out_dim, 1)
        self.attn_dst = nn.Linear(out_dim, 1)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, h: Tensor) -> Tensor:
        """Apply one fully-connected graph-attention pass over agents.

        Parameters
        ----------
        h:
            Per-agent hidden states, shape ``(batch, num_agents, in_dim)``.

        Returns
        -------
        Tensor
            Attention-mixed per-agent features, ``(batch, num_agents, out_dim)``.
        """
        h_prime = self.proj(h)
        src = self.attn_src(h_prime)
        dst = self.attn_dst(h_prime)
        scores = self.leaky_relu(src + dst.transpose(1, 2))
        attn = F.softmax(scores, dim=-1)
        return torch.matmul(attn, h_prime)


class STGAT(nn.Module):
    """STGAT: per-timestep graph attention interleaved with temporal LSTMs."""

    def __init__(
        self, hidden: int = 24, graph_hidden: int = 16, noise_dim: int = 8, pred_len: int = 6
    ) -> None:
        """Build the trajectory LSTM, per-step GAT, graph LSTM and decoder.

        Parameters
        ----------
        hidden:
            Trajectory LSTM hidden width.
        graph_hidden:
            Graph-attention output / graph-LSTM hidden width.
        noise_dim:
            Width of the injected GAN-style noise vector.
        pred_len:
            Number of future steps to autoregressively decode.
        """
        super().__init__()
        self.hidden = hidden
        self.graph_hidden = graph_hidden
        self.pred_len = pred_len
        self.traj_lstm = nn.LSTMCell(2, hidden)
        self.gat = GraphAttentionLayer(hidden, graph_hidden)
        self.graph_lstm = nn.LSTMCell(graph_hidden, graph_hidden)
        self.noise_dim = noise_dim
        self.decoder = nn.LSTMCell(2, hidden + graph_hidden + noise_dim)
        self.hidden2pos = nn.Linear(hidden + graph_hidden + noise_dim, 2)

    def forward(self, obs_traj: Tensor) -> Tensor:
        """Encode observed multi-agent trajectories and roll out the future.

        Parameters
        ----------
        obs_traj:
            Observed relative displacements, shape
            ``(obs_len, num_agents, 2)`` for a single scene (all agents form
            one fully-connected interaction graph, following the official
            ``seq_start_end`` convention collapsed to one scene).

        Returns
        -------
        Tensor
            Predicted future relative displacements,
            ``(pred_len, num_agents, 2)``.
        """
        obs_len, num_agents, _ = obs_traj.shape
        h_t = torch.zeros(num_agents, self.hidden)
        c_t = torch.zeros(num_agents, self.hidden)
        g_h = torch.zeros(num_agents, self.graph_hidden)
        g_c = torch.zeros(num_agents, self.graph_hidden)

        for t in range(obs_len):
            h_t, c_t = self.traj_lstm(obs_traj[t], (h_t, c_t))
            graph_feat = self.gat(h_t.unsqueeze(0)).squeeze(0)
            g_h, g_c = self.graph_lstm(graph_feat, (g_h, g_c))

        noise = torch.randn(num_agents, self.noise_dim)
        dec_h = torch.cat([h_t, g_h, noise], dim=-1)
        dec_c = torch.zeros_like(dec_h)

        x = torch.zeros(num_agents, 2)
        outputs = []
        for _ in range(self.pred_len):
            dec_h, dec_c = self.decoder(x, (dec_h, dec_c))
            x = self.hidden2pos(dec_h)
            outputs.append(x)
        return torch.stack(outputs, dim=0)


def build_stgat() -> nn.Module:
    """Build a compact STGAT model.

    Returns
    -------
    nn.Module
        Random-initialized ``STGAT`` in eval mode.
    """
    return STGAT().eval()


def example_input_stgat() -> Tensor:
    """Create an example observed multi-agent trajectory.

    Returns
    -------
    Tensor
        Observed relative displacements, shape ``(8, 5, 2)`` (8 timesteps,
        5 agents in the scene).
    """
    return torch.randn(8, 5, 2)


# ---------------------------------------------------------------------------
# T4P
# ---------------------------------------------------------------------------


class ActorTokenMemory(nn.Module):
    """Learned memory bank queried per actor for a personalized correction token."""

    def __init__(self, dim: int, num_slots: int = 16) -> None:
        """Build the memory bank and attention-based retrieval.

        Parameters
        ----------
        dim:
            Token embedding width.
        num_slots:
            Number of learned memory slots.
        """
        super().__init__()
        self.memory = nn.Parameter(torch.randn(num_slots, dim) * 0.02)
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)

    def forward(self, actor_embed: Tensor) -> Tensor:
        """Retrieve a per-actor correction token from the memory bank.

        Parameters
        ----------
        actor_embed:
            Per-actor query embedding, shape ``(batch, num_actors, dim)``.

        Returns
        -------
        Tensor
            Retrieved correction token, same shape as ``actor_embed``.
        """
        q = self.query_proj(actor_embed)
        k = self.key_proj(self.memory)
        v = self.value_proj(self.memory)
        scores = torch.matmul(q, k.transpose(0, 1)) / (q.shape[-1] ** 0.5)
        attn = F.softmax(scores, dim=-1)
        return torch.matmul(attn, v)


class T4P(nn.Module):
    """T4P: masked-autoencoder trajectory backbone + actor-specific token memory."""

    def __init__(
        self, dim: int = 32, obs_len: int = 10, pred_len: int = 6, mask_ratio: float = 0.3
    ) -> None:
        """Build the MAE encoder/decoder and the actor token-memory module.

        Parameters
        ----------
        dim:
            Transformer embedding width.
        obs_len:
            Number of observed timesteps per actor.
        pred_len:
            Number of future timesteps to reconstruct/predict.
        mask_ratio:
            Fraction of observed tokens replaced with the mask token during
            the self-supervised masked-reconstruction forward pass.
        """
        super().__init__()
        self.dim = dim
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.mask_ratio = mask_ratio
        self.input_proj = nn.Linear(2, dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.randn(1, obs_len, dim) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            dim, nhead=4, dim_feedforward=dim * 2, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.token_memory = ActorTokenMemory(dim)
        self.fuse = nn.Linear(dim * 2, dim)
        dec_layer = nn.TransformerEncoderLayer(
            dim, nhead=4, dim_feedforward=dim * 2, batch_first=True
        )
        self.decoder = nn.TransformerEncoder(dec_layer, num_layers=2)
        self.recon_head = nn.Linear(dim, 2)
        self.future_query = nn.Parameter(torch.randn(1, pred_len, dim) * 0.02)
        self.future_head = nn.Linear(dim, 2)

    def forward(self, obs_traj: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """Masked-reconstruct the observed trajectory and predict the future.

        Parameters
        ----------
        obs_traj:
            Observed per-actor trajectory, shape ``(batch, num_actors, obs_len, 2)``.
        mask:
            Boolean mask, ``(batch, num_actors, obs_len)``, True where the
            timestep is masked out (test-time-training self-supervision
            signal).

        Returns
        -------
        tuple[Tensor, Tensor]
            Reconstructed observed positions ``(batch, num_actors, obs_len, 2)``
            and predicted future positions ``(batch, num_actors, pred_len, 2)``.
        """
        batch, num_actors, obs_len, _ = obs_traj.shape
        tokens = self.input_proj(obs_traj) + self.pos_embed
        mask_tok = self.mask_token.expand(batch, num_actors, obs_len, -1)
        tokens = torch.where(mask.unsqueeze(-1), mask_tok, tokens)

        flat = tokens.view(batch * num_actors, obs_len, self.dim)
        encoded = self.encoder(flat)
        actor_embed = encoded.mean(dim=1).view(batch, num_actors, self.dim)

        memory_token = self.token_memory(actor_embed)
        fused = self.fuse(torch.cat([actor_embed, memory_token], dim=-1))
        fused = fused.view(batch * num_actors, 1, self.dim).expand(-1, obs_len, -1)

        decoded = self.decoder(encoded + fused)
        recon = self.recon_head(decoded).view(batch, num_actors, obs_len, 2)

        future_q = self.future_query.expand(batch * num_actors, -1, -1)
        future_ctx = torch.cat([encoded, future_q], dim=1)
        future_decoded = self.decoder(future_ctx)[:, obs_len:, :]
        future = self.future_head(future_decoded).view(batch, num_actors, self.pred_len, 2)
        return recon, future


def build_t4p() -> nn.Module:
    """Build a compact T4P model.

    Returns
    -------
    nn.Module
        Random-initialized ``T4P`` in eval mode.
    """
    return T4P().eval()


def example_input_t4p() -> tuple[Tensor, Tensor]:
    """Create example observed trajectories with a masked-reconstruction mask.

    Returns
    -------
    tuple[Tensor, Tensor]
        Observed trajectory ``(1, 3, 10, 2)`` and boolean mask ``(1, 3, 10)``.
    """
    obs_traj = torch.randn(1, 3, 10, 2)
    mask = torch.zeros(1, 3, 10, dtype=torch.bool)
    mask[:, :, 3:6] = True
    return obs_traj, mask


# ---------------------------------------------------------------------------
# TCP (Trajectory-guided Control Prediction)
# ---------------------------------------------------------------------------


class TCP(nn.Module):
    """TCP: coupled trajectory + control decoders with cross-branch attention."""

    def __init__(
        self, feat_dim: int = 64, hidden: int = 32, pred_len: int = 4, spatial: int = 6
    ) -> None:
        """Build the perception trunk and coupled trajectory/control decoders.

        Parameters
        ----------
        feat_dim:
            Width of the pooled perception embedding.
        hidden:
            Recurrent decoder hidden width.
        pred_len:
            Number of future waypoints / control steps to decode.
        spatial:
            Side length of the synthetic spatial CNN feature map used by the
            control branch's waypoint-attention read-out.
        """
        super().__init__()
        self.pred_len = pred_len
        self.spatial = spatial
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, feat_dim, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.measurements = nn.Sequential(nn.Linear(3, 32), nn.ReLU(inplace=True))
        self.join_traj = nn.Linear(feat_dim + 32, hidden)
        self.decoder_traj = nn.GRUCell(input_size=2 + 2, hidden_size=hidden)
        self.output_traj = nn.Linear(hidden, 2)

        self.join_ctrl = nn.Linear(feat_dim + 32, hidden)
        self.decoder_ctrl = nn.GRUCell(input_size=hidden + 2, hidden_size=hidden)
        self.wp_attn = nn.Sequential(
            nn.Linear(hidden + hidden, spatial * spatial), nn.Softmax(dim=-1)
        )
        self.merge = nn.Linear(feat_dim + hidden, hidden)
        self.dist_mu = nn.Sequential(nn.Linear(hidden, 2), nn.Softplus())
        self.dist_sigma = nn.Sequential(nn.Linear(hidden, 2), nn.Softplus())

    def forward(
        self, img: Tensor, state: Tensor, target_point: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Jointly decode future waypoints and control-action distributions.

        Parameters
        ----------
        img:
            Front-camera image, shape ``(batch, 3, H, W)``.
        state:
            Ego measurement vector (speed + last steer/throttle etc.),
            shape ``(batch, 3)``.
        target_point:
            Route target point in ego frame, shape ``(batch, 2)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Predicted waypoints ``(batch, pred_len, 2)``, control-action
            means and stds each ``(batch, pred_len, 2)`` (stacked over steps).
        """
        batch = img.shape[0]
        spatial_feat = self.backbone(img)
        spatial_feat = F.adaptive_avg_pool2d(spatial_feat, (self.spatial, self.spatial))
        pooled = spatial_feat.mean(dim=(2, 3))
        meas = self.measurements(state)

        z_traj = self.join_traj(torch.cat([pooled, meas], dim=-1))
        x = torch.zeros(batch, 2)
        traj_hidden = []
        waypoints = []
        for _ in range(self.pred_len):
            x_in = torch.cat([x, target_point], dim=-1)
            z_traj = self.decoder_traj(x_in, z_traj)
            traj_hidden.append(z_traj)
            dx = self.output_traj(z_traj)
            x = x + dx
            waypoints.append(x)
        pred_wp = torch.stack(waypoints, dim=1)
        traj_hidden = torch.stack(traj_hidden, dim=1)

        z_ctrl = self.join_ctrl(torch.cat([pooled, meas], dim=-1))
        flat_spatial = spatial_feat.flatten(2)
        mu_list = []
        sigma_list = []
        for t in range(self.pred_len):
            attn = self.wp_attn(torch.cat([z_ctrl, traj_hidden[:, t]], dim=-1))
            looked = torch.bmm(flat_spatial, attn.unsqueeze(-1)).squeeze(-1)
            fused = self.merge(torch.cat([looked, z_ctrl], dim=-1))
            z_ctrl = self.decoder_ctrl(torch.cat([fused, x - x], dim=-1), z_ctrl)
            mu_list.append(self.dist_mu(z_ctrl))
            sigma_list.append(self.dist_sigma(z_ctrl))
        mu = torch.stack(mu_list, dim=1)
        sigma = torch.stack(sigma_list, dim=1)
        return pred_wp, mu, sigma


def build_tcp() -> nn.Module:
    """Build a compact TCP model.

    Returns
    -------
    nn.Module
        Random-initialized ``TCP`` in eval mode.
    """
    return TCP().eval()


def example_input_tcp() -> tuple[Tensor, Tensor, Tensor]:
    """Create example front-camera image, ego state and target point.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        Image ``(1, 3, 64, 64)``, state ``(1, 3)``, target point ``(1, 2)``.
    """
    img = torch.randn(1, 3, 64, 64)
    state = torch.randn(1, 3)
    target_point = torch.randn(1, 2)
    return img, state, target_point


# ---------------------------------------------------------------------------
# ThinkTwice
# ---------------------------------------------------------------------------


class LookModule(nn.Module):
    """Proposal-conditioned attention: project waypoints, sample BEV features."""

    def __init__(self, bev_dim: int) -> None:
        """Build the reference-point projection MLP and feature fusion.

        Parameters
        ----------
        bev_dim:
            Channel width of the synthetic BEV feature grid.
        """
        super().__init__()
        self.ref_proj = nn.Linear(2 + 2, 2)
        self.fuse = nn.Linear(bev_dim, bev_dim)

    def forward(self, waypoint: Tensor, control: Tensor, bev_feat: Tensor) -> Tensor:
        """Sample BEV features at the current proposal's projected location.

        Parameters
        ----------
        waypoint:
            Current proposal waypoint, shape ``(batch, 2)``.
        control:
            Current proposal control action, shape ``(batch, 2)``.
        bev_feat:
            Synthetic BEV feature grid, shape ``(batch, bev_dim, H, W)``.

        Returns
        -------
        Tensor
            Looked-up, fused feature at the reference point, ``(batch, bev_dim)``.
        """
        ref_xy = torch.tanh(self.ref_proj(torch.cat([waypoint, control], dim=-1)))
        grid = ref_xy.view(-1, 1, 1, 2)
        sampled = F.grid_sample(bev_feat, grid, align_corners=False, mode="bilinear")
        sampled = sampled.view(bev_feat.shape[0], bev_feat.shape[1])
        return self.fuse(sampled)


class SpatialGRUCellCompact(nn.Module):
    """Compact per-cell spatial GRU state update over the BEV grid."""

    def __init__(self, bev_dim: int) -> None:
        """Build gated update/reset convolutions over the BEV grid.

        Parameters
        ----------
        bev_dim:
            Channel width of the BEV feature grid.
        """
        super().__init__()
        self.update_gate = nn.Conv2d(bev_dim * 2, bev_dim, 3, padding=1)
        self.candidate = nn.Conv2d(bev_dim * 2, bev_dim, 3, padding=1)

    def forward(self, looked_feat: Tensor, bev_state: Tensor) -> Tensor:
        """Update the BEV state grid with the broadcasted looked-up feature.

        Parameters
        ----------
        looked_feat:
            Looked-up per-batch feature, shape ``(batch, bev_dim)``.
        bev_state:
            Current BEV feature state, shape ``(batch, bev_dim, H, W)``.

        Returns
        -------
        Tensor
            Updated BEV feature state, same shape as ``bev_state``.
        """
        h, w = bev_state.shape[-2:]
        broadcast = looked_feat.view(*looked_feat.shape, 1, 1).expand(-1, -1, h, w)
        combined = torch.cat([broadcast, bev_state], dim=1)
        z = torch.sigmoid(self.update_gate(combined))
        candidate = torch.tanh(self.candidate(combined))
        return (1 - z) * bev_state + z * candidate


class ThinkTwice(nn.Module):
    """ThinkTwice: coarse proposal + repeated look-refine cascade decoder."""

    def __init__(self, bev_dim: int = 16, num_refine: int = 3) -> None:
        """Build the coarse head and the stacked look-refine stages.

        Parameters
        ----------
        bev_dim:
            Channel width of the synthetic fused BEV feature grid.
        num_refine:
            Number of "think twice" (look + spatial-GRU-refine) stages.
        """
        super().__init__()
        self.bev_pool = nn.AdaptiveAvgPool2d(1)
        self.coarse_head = nn.Sequential(
            nn.Linear(bev_dim, 32), nn.ReLU(inplace=True), nn.Linear(32, 4)
        )
        self.look_modules = nn.ModuleList([LookModule(bev_dim) for _ in range(num_refine)])
        self.spatial_grus = nn.ModuleList(
            [SpatialGRUCellCompact(bev_dim) for _ in range(num_refine)]
        )
        self.residual_heads = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(bev_dim, 32), nn.ReLU(inplace=True), nn.Linear(32, 4))
                for _ in range(num_refine)
            ]
        )

    def forward(self, bev_feat: Tensor) -> Tensor:
        """Decode a coarse waypoint+control proposal then refine it repeatedly.

        Parameters
        ----------
        bev_feat:
            Synthetic fused BEV feature grid, shape ``(batch, bev_dim, H, W)``.

        Returns
        -------
        Tensor
            Final refined proposal (waypoint xy + control ab), ``(batch, 4)``.
        """
        pooled = self.bev_pool(bev_feat).flatten(1)
        proposal = self.coarse_head(pooled)
        bev_state = bev_feat
        for look, gru, residual_head in zip(
            self.look_modules, self.spatial_grus, self.residual_heads, strict=True
        ):
            waypoint, control = proposal[:, :2], proposal[:, 2:]
            looked = look(waypoint, control, bev_state)
            bev_state = gru(looked, bev_state)
            state_pooled = self.bev_pool(bev_state).flatten(1)
            proposal = proposal + residual_head(state_pooled)
        return proposal


def build_thinktwice() -> nn.Module:
    """Build a compact ThinkTwice model.

    Returns
    -------
    nn.Module
        Random-initialized ``ThinkTwice`` in eval mode.
    """
    return ThinkTwice().eval()


def example_input_thinktwice() -> Tensor:
    """Create an example synthetic fused BEV feature grid.

    Returns
    -------
    Tensor
        BEV feature grid, shape ``(1, 16, 8, 8)``.
    """
    return torch.randn(1, 16, 8, 8)


# ---------------------------------------------------------------------------
# TrafficBots
# ---------------------------------------------------------------------------


class PersonalityEncoder(nn.Module):
    """CVAE posterior over a per-agent personality latent."""

    def __init__(self, dim: int, latent_dim: int) -> None:
        """Build the posterior and prior heads for the personality latent.

        Parameters
        ----------
        dim:
            Per-agent context embedding width.
        latent_dim:
            Personality-latent width.
        """
        super().__init__()
        self.posterior = nn.Linear(dim * 2, latent_dim * 2)
        self.prior = nn.Linear(dim, latent_dim * 2)

    def forward(self, context: Tensor, future_summary: Tensor | None) -> Tensor:
        """Sample a per-agent personality latent from the posterior or prior.

        Parameters
        ----------
        context:
            Per-agent scene-encoder context, shape ``(batch, num_agents, dim)``.
        future_summary:
            Ground-truth-future summary for the posterior branch (training),
            same shape as ``context``, or ``None`` to sample from the prior
            (closed-loop simulation).

        Returns
        -------
        Tensor
            Sampled personality latent, ``(batch, num_agents, latent_dim)``.
        """
        if future_summary is not None:
            stats = self.posterior(torch.cat([context, future_summary], dim=-1))
        else:
            stats = self.prior(context)
        mu, logvar = stats.chunk(2, dim=-1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class TrafficBots(nn.Module):
    """TrafficBots: personality-conditioned Transformer scene encoder + closed-loop GRU rollout."""

    def __init__(self, dim: int = 32, latent_dim: int = 8, num_steps: int = 5) -> None:
        """Build the scene encoder, personality CVAE, and closed-loop decoder.

        Parameters
        ----------
        dim:
            Transformer / decoder embedding width.
        latent_dim:
            Personality-latent width.
        num_steps:
            Number of closed-loop simulation steps to unroll.
        """
        super().__init__()
        self.dim = dim
        self.latent_dim = latent_dim
        self.num_steps = num_steps
        self.state_proj = nn.Linear(4, dim)
        self.map_proj = nn.Linear(4, dim)
        enc_layer = nn.TransformerEncoderLayer(
            dim, nhead=4, dim_feedforward=dim * 2, batch_first=True
        )
        self.scene_encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.personality = PersonalityEncoder(dim, latent_dim)
        self.latent_proj = nn.Linear(latent_dim, dim)
        self.decoder_cell = nn.GRUCell(dim + dim, dim)
        self.action_head = nn.Linear(dim, 4)

    def forward(self, agent_state: Tensor, map_polylines: Tensor) -> Tensor:
        """Closed-loop unroll all agents jointly for ``num_steps`` steps.

        Parameters
        ----------
        agent_state:
            Initial per-agent state (x, y, heading, speed), shape
            ``(batch, num_agents, 4)``.
        map_polylines:
            Static map context tokens, shape ``(batch, num_map_tokens, 4)``.

        Returns
        -------
        Tensor
            Rolled-out per-agent actions over all steps,
            ``(batch, num_steps, num_agents, 4)``.
        """
        batch, num_agents, _ = agent_state.shape
        map_tokens = self.map_proj(map_polylines)

        state_tok = self.state_proj(agent_state)
        scene_in = torch.cat([state_tok, map_tokens], dim=1)
        scene_out = self.scene_encoder(scene_in)
        agent_context = scene_out[:, :num_agents, :]

        latent = self.personality(agent_context, future_summary=None)
        latent_tok = self.latent_proj(latent)

        hidden = torch.zeros(batch * num_agents, self.dim)
        current_state = agent_state
        actions = []
        for _ in range(self.num_steps):
            state_tok = self.state_proj(current_state)
            scene_in = torch.cat([state_tok, map_tokens], dim=1)
            scene_out = self.scene_encoder(scene_in)
            agent_context = scene_out[:, :num_agents, :]
            dec_in = torch.cat([agent_context, latent_tok], dim=-1).view(batch * num_agents, -1)
            hidden = self.decoder_cell(dec_in, hidden)
            action = self.action_head(hidden).view(batch, num_agents, 4)
            actions.append(action)
            current_state = current_state + action
        return torch.stack(actions, dim=1)


def build_trafficbots() -> nn.Module:
    """Build a compact TrafficBots model.

    Returns
    -------
    nn.Module
        Random-initialized ``TrafficBots`` in eval mode.
    """
    return TrafficBots().eval()


def example_input_trafficbots() -> tuple[Tensor, Tensor]:
    """Create example initial multi-agent state and static map polylines.

    Returns
    -------
    tuple[Tensor, Tensor]
        Agent state ``(1, 6, 4)`` and map polyline tokens ``(1, 10, 4)``.
    """
    agent_state = torch.randn(1, 6, 4)
    map_polylines = torch.randn(1, 10, 4)
    return agent_state, map_polylines


MENAGERIE_ENTRIES = [
    ("SparseBEV", "build_sparsebev", "example_input_sparsebev", "2023", "VIS"),
    ("STGAT", "build_stgat", "example_input_stgat", "2019", "SEQ"),
    ("T4P", "build_t4p", "example_input_t4p", "2024", "SEQ"),
    ("TCP", "build_tcp", "example_input_tcp", "2022", "SEQ"),
    ("ThinkTwice", "build_thinktwice", "example_input_thinktwice", "2023", "SEQ"),
    ("TrafficBots", "build_trafficbots", "example_input_trafficbots", "2023", "SEQ"),
]
