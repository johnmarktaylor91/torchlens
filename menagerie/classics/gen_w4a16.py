"""Autonomous-driving trajectory-prediction / end-to-end-driving classics
(build queue rows 97-102).

Sources checked (repo_url / desc_source from build_queue.tsv, architecture study only --
no clone, no pip install, faithful compact reimplementation from scratch in base-env
torch):

- Trajectron++: Salzmann, Ivanovic, Chakravarty & Pavone, ECCV 2020, arXiv:2001.03093.
  https://github.com/StanfordASL/Trajectron-plus-plus
  Graph-structured recurrent CVAE. Each agent has a per-node history LSTM encoder; agents
  of the same semantic type within a radius are aggregated with an element-wise sum and
  fed through a shared "edge LSTM" (the dynamic-edge encoder, so the interaction graph is
  built and re-encoded per scene rather than fixed) whose output is summed across edge
  types into one interaction-context vector alongside the node's own history embedding
  and a (here: linear, since map images are external) map encoding. A discrete-latent
  CVAE (encoder -> categorical posterior at train time, uniform prior at inference) then
  conditions a GRU decoder that autoregressively rolls out a distribution over future
  states. Reimplemented here with the dynamic same-radius edge aggregation + shared edge
  LSTM + discrete-latent CVAE + GRU decoder as the defining mechanism.

- TransFuser: Prakash, Chitta & Geiger, CVPR 2021 / PAMI 2023, arXiv:2205.15997.
  https://github.com/autonomousvision/transfuser
  Multi-modal fusion TRANSFORMER for end-to-end driving: separate RGB-image and
  LiDAR-BEV CNN backbones produce feature maps at several resolutions; at EACH resolution
  stage the flattened image-branch and LiDAR-branch tokens are concatenated with a
  learned positional embedding and passed through a small Transformer encoder, and the
  attended tokens are pooled, projected and added back into each branch's feature map
  before the next conv stage -- i.e. attention-based fusion is interleaved with the CNN
  backbones at multiple scales, not a single late-fusion step. The final fused feature
  feeds a GRU that autoregressively predicts BEV waypoints. Reimplemented compactly with
  two small conv stacks and two interleaved cross-modal Transformer fusion stages.

- TraPHic: Chandra, Bhattacharya, Bera & Manocha, CVPR 2019, arXiv:1812.04767 (build_queue
  cites arXiv:1906.00547, an extension abstract of the same TraPHic algorithm; official
  impl is in rohanchandra30/TrackNPred). Heterogeneous LSTM-CNN hybrid for dense,
  mixed-agent traffic: every neighboring road-agent's encoded LSTM state is placed into a
  spatial grid cell (by relative position) and weighted by a HORIZON-BASED "weighted
  interaction" mask that up-weights nearby, high-priority heterogeneous agents (e.g. a
  bus vs. a bicycle) within the ego agent's motion horizon; the weighted grid is then
  processed by a small CNN ("interaction CNN") whose pooled output is concatenated with
  the ego LSTM's own hidden state to drive an LSTM decoder. Reimplemented here with the
  per-agent-type horizon weighting + spatial-grid CNN pooling as the distinguishing
  mechanism.

- TUTR (Trajectory Unified Transformer): Shi, Jiang, Lu, Gong, Niu, Chen & Chang,
  ICCV 2023, arXiv:2307.03125. https://github.com/lssiair/TUTR
  Encoder-decoder that unifies multimodal mode prediction and social interaction without
  any post-hoc clustering/NMS. A small set of LEARNED MOTION-MODE QUERY TOKENS (fixed,
  trainable embeddings representing candidate intention modes) is concatenated with the
  ego agent's trajectory token and processed by a "global-interaction" Transformer
  encoder (self-attention across modes + ego token, i.e. mode-level relationships are
  modeled explicitly). A social-level Transformer DECODER then has each mode query
  cross-attend (encoder-decoder attention only, self-attention deliberately dropped, per
  the paper) over the neighboring agents' encoded tokens to inject social context. A
  "dual prediction" head reads the resulting per-mode tokens straight into (trajectory,
  probability) pairs in one forward pass. Reimplemented here with learned mode queries +
  mode-level self-attention encoder + social cross-attention-only decoder + dual head.

- UniDrive-WM: (authors per project site), Jan 2026, arXiv:2601.04453.
  https://github.com/UniDrive-WM/UniDrive-WM , https://unidrive-wm.github.io/UniDrive-WM/
  Unified VLM-style world model that performs scene UNDERSTANDING, trajectory PLANNING
  and trajectory-conditioned future-frame GENERATION in one architecture and closes the
  loop between them: a shared vision-language backbone encodes the current frame +
  scene tokens; an autoregressive planning head decodes a future ego trajectory from
  those tokens; the predicted trajectory is then injected (via cross-attention, "AR+
  diffusion" per the queue notes) as conditioning into a lightweight diffusion-style
  denoiser that generates the next BEV/image latent, and that generated latent is fed
  back in as additional context for the next planning step -- i.e. planning and
  generation are coupled in a single closed AR-then-diffusion loop rather than being
  independent heads. Reimplemented compactly here with a shared token encoder,
  autoregressive GRU trajectory planner, and a small trajectory-conditioned iterative
  denoiser (few fixed steps) feeding back into the encoder.

- Urban Driver: Scheel, Bergamini, Wolczyk, Osinski & Ondruska, CoRL 2021 (cited in the
  build queue as "NeurIPS 2021"), arXiv:2109.13333 (build_queue cites arXiv:2109.01687,
  the companion "SimNet" data-driven-simulator paper from the same Woven Planet Level-5
  group; Urban Driver is the closed-loop policy trained on top of it).
  https://github.com/woven-planet/l5kit
  Vectorized ("mid-level") scene representation: every polyline element of the scene
  (ego path, other agents' tracks, individual lane/crosswalk segments) is encoded
  independently by a shared small PointNet-style per-point MLP + max-pool ("local
  subgraph" encoder producing one feature vector per polyline), and the resulting set of
  per-element vectors is then aggregated with a SINGLE scaled dot-product self-attention
  layer over the whole variable-size element set (global interaction), with no
  rasterized image input anywhere. The pooled ego-conditioned output drives an MLP policy
  head that regresses a sequence of future ego waypoints, trained via (here, standalone)
  closed-loop policy-gradient rollout against the differentiable simulator. Reimplemented
  here with the per-polyline PointNet-subgraph encoder + single global self-attention
  layer + waypoint policy head as the defining mechanism.

All six are kept intentionally tiny (few agents/polylines, short horizons, small hidden
dims, few fusion/refinement stages) since this is an architecture catalog, not a
trained-weights zoo.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


# ---------------------------------------------------------------------------
# Trajectron++
# ---------------------------------------------------------------------------
class Trajectronpp(nn.Module):
    """Graph-structured recurrent CVAE trajectory forecaster (Trajectron++).

    Per-agent history LSTM encoding, dynamic same-radius-neighbor edge
    aggregation through a shared edge LSTM, a discrete-latent CVAE, and a
    GRU decoder that autoregressively rolls out future states.
    """

    def __init__(
        self,
        state_dim: int = 6,
        hidden_dim: int = 16,
        latent_dim: int = 4,
        future_steps: int = 8,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.future_steps = future_steps

        self.history_encoder = nn.LSTM(state_dim, hidden_dim, batch_first=True)
        self.edge_lstm = nn.LSTM(state_dim, hidden_dim, batch_first=True)
        self.map_encoder = nn.Linear(state_dim, hidden_dim)

        ctx_dim = hidden_dim * 3
        self.latent_encoder = nn.Linear(ctx_dim, latent_dim)
        self.decoder_init = nn.Linear(ctx_dim + latent_dim, hidden_dim)
        self.decoder_cell = nn.GRUCell(state_dim, hidden_dim)
        self.state_head = nn.Linear(hidden_dim, state_dim)

    def forward(self, agent_history: Tensor, map_context: Tensor) -> Tensor:
        """Forecast future states for every agent in the scene.

        Parameters
        ----------
        agent_history : Tensor
            Past agent states, shape ``(batch, num_agents, obs_steps, state_dim)``.
        map_context : Tensor
            Static per-agent map feature, shape ``(batch, num_agents, state_dim)``.

        Returns
        -------
        Tensor
            Predicted future states, shape
            ``(batch, num_agents, future_steps, state_dim)``.
        """
        batch, num_agents, obs_steps, state_dim = agent_history.shape
        flat_hist = agent_history.reshape(batch * num_agents, obs_steps, state_dim)
        _, (h_n, _) = self.history_encoder(flat_hist)
        node_embed = h_n[-1].view(batch, num_agents, self.hidden_dim)

        # Dynamic edges: aggregate ALL other agents' current states (same-type,
        # same-radius stand-in) with an element-wise sum, then run the shared
        # edge LSTM over that one aggregated step.
        agg_neighbors = agent_history[:, :, -1, :].sum(dim=1, keepdim=True)
        agg_neighbors = agg_neighbors.expand(-1, num_agents, -1)
        agg_neighbors = agg_neighbors - agent_history[:, :, -1, :]
        edge_in = agg_neighbors.reshape(batch * num_agents, 1, state_dim)
        _, (edge_h, _) = self.edge_lstm(edge_in)
        edge_embed = edge_h[-1].view(batch, num_agents, self.hidden_dim)

        map_embed = self.map_encoder(map_context)
        context = torch.cat([node_embed, edge_embed, map_embed], dim=-1)

        latent = self.latent_encoder(context)
        dec_ctx = torch.cat([context, latent], dim=-1)
        hidden = self.decoder_init(dec_ctx).reshape(batch * num_agents, self.hidden_dim)

        current_state = agent_history[:, :, -1, :].reshape(batch * num_agents, state_dim)
        outputs = []
        for _ in range(self.future_steps):
            hidden = self.decoder_cell(current_state, hidden)
            delta = self.state_head(hidden)
            current_state = current_state + delta
            outputs.append(current_state)
        out = torch.stack(outputs, dim=1).view(batch, num_agents, self.future_steps, state_dim)
        return out


def build_trajectronpp() -> nn.Module:
    """Build a compact Trajectron++ model.

    Returns
    -------
    nn.Module
        Random-initialized ``Trajectronpp`` in eval mode.
    """
    return Trajectronpp().eval()


def example_input_trajectronpp() -> tuple[Tensor, Tensor]:
    """Create example agent-history and map-context tensors.

    Returns
    -------
    tuple[Tensor, Tensor]
        Agent history ``(1, 4, 5, 6)`` and map context ``(1, 4, 6)``.
    """
    agent_history = torch.randn(1, 4, 5, 6)
    map_context = torch.randn(1, 4, 6)
    return agent_history, map_context


# ---------------------------------------------------------------------------
# TransFuser
# ---------------------------------------------------------------------------
class _FusionStage(nn.Module):
    """One multi-scale image/LiDAR cross-modal Transformer fusion stage."""

    def __init__(self, channels: int, num_tokens: int) -> None:
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, 2 * num_tokens, channels) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels, nhead=2, dim_feedforward=channels * 2, batch_first=True
        )
        self.fusion = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.image_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.lidar_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, image_feat: Tensor, lidar_feat: Tensor) -> tuple[Tensor, Tensor]:
        """Fuse image and LiDAR feature maps at this resolution.

        Parameters
        ----------
        image_feat : Tensor
            Image-branch feature map, shape ``(batch, channels, height, width)``.
        lidar_feat : Tensor
            LiDAR-BEV-branch feature map, shape ``(batch, channels, height, width)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Updated ``(image_feat, lidar_feat)`` after attention-fused residual.
        """
        b, c, h, w = image_feat.shape
        img_tok = image_feat.flatten(2).transpose(1, 2)
        lidar_tok = lidar_feat.flatten(2).transpose(1, 2)
        tokens = torch.cat([img_tok, lidar_tok], dim=1) + self.pos_embed
        fused = self.fusion(tokens)
        fused_img, fused_lidar = fused[:, : h * w], fused[:, h * w :]
        fused_img = fused_img.transpose(1, 2).reshape(b, c, h, w)
        fused_lidar = fused_lidar.transpose(1, 2).reshape(b, c, h, w)
        image_feat = self.image_conv(image_feat + fused_img)
        lidar_feat = self.lidar_conv(lidar_feat + fused_lidar)
        return image_feat, lidar_feat


class TransFuser(nn.Module):
    """Multi-modal fusion Transformer for end-to-end driving (TransFuser).

    Interleaves a small image-branch and LiDAR-BEV-branch conv backbone with
    cross-modal Transformer fusion stages at two resolutions, then decodes
    BEV waypoints autoregressively with a GRU.
    """

    def __init__(self, channels: int = 8, hidden_dim: int = 16, waypoints: int = 4) -> None:
        super().__init__()
        self.channels = channels
        self.waypoints = waypoints

        self.image_stem = nn.Conv2d(3, channels, kernel_size=3, padding=1)
        self.lidar_stem = nn.Conv2d(2, channels, kernel_size=3, padding=1)
        self.fuse1 = _FusionStage(channels, num_tokens=8 * 8)
        self.downsample_img = nn.Conv2d(channels, channels, kernel_size=2, stride=2)
        self.downsample_lidar = nn.Conv2d(channels, channels, kernel_size=2, stride=2)
        self.fuse2 = _FusionStage(channels, num_tokens=4 * 4)

        self.pool_proj = nn.Linear(channels * 2, hidden_dim)
        self.decoder_cell = nn.GRUCell(2, hidden_dim)
        self.waypoint_head = nn.Linear(hidden_dim, 2)

    def forward(self, image: Tensor, lidar_bev: Tensor) -> Tensor:
        """Predict a sequence of BEV waypoints from image + LiDAR-BEV input.

        Parameters
        ----------
        image : Tensor
            RGB image, shape ``(batch, 3, 8, 8)``.
        lidar_bev : Tensor
            LiDAR bird's-eye-view raster, shape ``(batch, 2, 8, 8)``.

        Returns
        -------
        Tensor
            Predicted waypoints, shape ``(batch, waypoints, 2)``.
        """
        img_feat = self.image_stem(image)
        lidar_feat = self.lidar_stem(lidar_bev)
        img_feat, lidar_feat = self.fuse1(img_feat, lidar_feat)

        img_feat = self.downsample_img(img_feat)
        lidar_feat = self.downsample_lidar(lidar_feat)
        img_feat, lidar_feat = self.fuse2(img_feat, lidar_feat)

        pooled = torch.cat([img_feat.mean(dim=(2, 3)), lidar_feat.mean(dim=(2, 3))], dim=-1)
        hidden = self.pool_proj(pooled)

        batch = image.shape[0]
        current_wp = torch.zeros(batch, 2, device=image.device, dtype=image.dtype)
        outputs = []
        for _ in range(self.waypoints):
            hidden = self.decoder_cell(current_wp, hidden)
            current_wp = current_wp + self.waypoint_head(hidden)
            outputs.append(current_wp)
        return torch.stack(outputs, dim=1)


def build_transfuser() -> nn.Module:
    """Build a compact TransFuser model.

    Returns
    -------
    nn.Module
        Random-initialized ``TransFuser`` in eval mode.
    """
    return TransFuser().eval()


def example_input_transfuser() -> tuple[Tensor, Tensor]:
    """Create example RGB image and LiDAR-BEV raster tensors.

    Returns
    -------
    tuple[Tensor, Tensor]
        Image ``(1, 3, 8, 8)`` and LiDAR BEV ``(1, 2, 8, 8)``.
    """
    image = torch.randn(1, 3, 8, 8)
    lidar_bev = torch.randn(1, 2, 8, 8)
    return image, lidar_bev


# ---------------------------------------------------------------------------
# TraPHic
# ---------------------------------------------------------------------------
class TraPHic(nn.Module):
    """Heterogeneous LSTM-CNN trajectory predictor for dense mixed traffic (TraPHic).

    Neighbor LSTM states are scattered into a spatial grid, weighted by a
    horizon-based heterogeneous-interaction mask, pooled by an interaction
    CNN, and combined with the ego agent's own LSTM state to drive an LSTM
    decoder.
    """

    def __init__(
        self,
        state_dim: int = 4,
        hidden_dim: int = 16,
        grid_size: int = 5,
        future_steps: int = 6,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.grid_size = grid_size
        self.future_steps = future_steps

        self.ego_encoder = nn.LSTM(state_dim, hidden_dim, batch_first=True)
        self.neighbor_encoder = nn.LSTM(state_dim, hidden_dim, batch_first=True)
        self.interaction_cnn = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.decoder_cell = nn.LSTMCell(state_dim, hidden_dim * 2)
        self.decoder_init = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.state_head = nn.Linear(hidden_dim * 2, state_dim)

    def forward(
        self,
        ego_history: Tensor,
        neighbor_history: Tensor,
        neighbor_grid_pos: Tensor,
        neighbor_priority: Tensor,
    ) -> Tensor:
        """Forecast the ego agent's future trajectory.

        Parameters
        ----------
        ego_history : Tensor
            Ego agent past states, shape ``(batch, obs_steps, state_dim)``.
        neighbor_history : Tensor
            Neighboring agents' past states,
            shape ``(batch, num_neighbors, obs_steps, state_dim)``.
        neighbor_grid_pos : Tensor
            Integer grid cell index (row, col) per neighbor, shape
            ``(batch, num_neighbors, 2)``.
        neighbor_priority : Tensor
            Horizon-based heterogeneous-interaction weight per neighbor
            (e.g. higher for large/near agents), shape ``(batch, num_neighbors)``.

        Returns
        -------
        Tensor
            Predicted ego states, shape ``(batch, future_steps, state_dim)``.
        """
        batch, obs_steps, state_dim = ego_history.shape
        _, (ego_h, _) = self.ego_encoder(ego_history)
        ego_embed = ego_h[-1]

        num_neighbors = neighbor_history.shape[1]
        flat_neighbors = neighbor_history.reshape(batch * num_neighbors, obs_steps, state_dim)
        _, (nbr_h, _) = self.neighbor_encoder(flat_neighbors)
        nbr_embed = nbr_h[-1].view(batch, num_neighbors, self.hidden_dim)

        weighted = nbr_embed * neighbor_priority.unsqueeze(-1)

        grid = torch.zeros(
            batch,
            self.hidden_dim,
            self.grid_size,
            self.grid_size,
            device=ego_history.device,
            dtype=ego_history.dtype,
        )
        for b in range(batch):
            for n in range(num_neighbors):
                r = int(neighbor_grid_pos[b, n, 0].clamp(0, self.grid_size - 1).item())
                c = int(neighbor_grid_pos[b, n, 1].clamp(0, self.grid_size - 1).item())
                grid[b, :, r, c] = grid[b, :, r, c] + weighted[b, n]

        interaction_ctx = self.interaction_cnn(grid).flatten(1)
        joint = torch.cat([ego_embed, interaction_ctx], dim=-1)
        hidden = self.decoder_init(joint)
        cell = torch.zeros_like(hidden)

        current_state = ego_history[:, -1, :]
        outputs = []
        for _ in range(self.future_steps):
            hidden, cell = self.decoder_cell(current_state, (hidden, cell))
            current_state = current_state + self.state_head(hidden)
            outputs.append(current_state)
        return torch.stack(outputs, dim=1)


def build_traphic() -> nn.Module:
    """Build a compact TraPHic model.

    Returns
    -------
    nn.Module
        Random-initialized ``TraPHic`` in eval mode.
    """
    return TraPHic().eval()


def example_input_traphic() -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Create example ego/neighbor histories, grid positions, and priorities.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor, Tensor]
        Ego history ``(1, 5, 4)``, neighbor history ``(1, 6, 5, 4)``,
        neighbor grid positions ``(1, 6, 2)`` (integer-valued floats in
        ``[0, 4]``), and neighbor priority weights ``(1, 6)``.
    """
    ego_history = torch.randn(1, 5, 4)
    neighbor_history = torch.randn(1, 6, 5, 4)
    neighbor_grid_pos = torch.randint(0, 5, (1, 6, 2)).float()
    neighbor_priority = torch.rand(1, 6)
    return ego_history, neighbor_history, neighbor_grid_pos, neighbor_priority


# ---------------------------------------------------------------------------
# TUTR
# ---------------------------------------------------------------------------
class TUTR(nn.Module):
    """Trajectory Unified Transformer for pedestrian trajectory prediction.

    Learned motion-mode query tokens attend to the ego trajectory token in a
    mode-level self-attention encoder, then a social-level decoder has each
    mode cross-attend (no self-attention) over neighbor tokens before a dual
    (trajectory, probability) prediction head.
    """

    def __init__(
        self,
        state_dim: int = 2,
        hidden_dim: int = 16,
        num_modes: int = 5,
        future_steps: int = 6,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_modes = num_modes
        self.future_steps = future_steps

        self.traj_proj = nn.Linear(state_dim, hidden_dim)
        self.neighbor_proj = nn.Linear(state_dim, hidden_dim)
        self.mode_queries = nn.Parameter(torch.randn(1, num_modes, hidden_dim) * 0.02)

        mode_encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=2, dim_feedforward=hidden_dim * 2, batch_first=True
        )
        self.mode_encoder = nn.TransformerEncoder(mode_encoder_layer, num_layers=1)

        self.social_cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=2, batch_first=True)

        self.traj_head = nn.Linear(hidden_dim, future_steps * state_dim)
        self.prob_head = nn.Linear(hidden_dim, 1)
        self.state_dim = state_dim

    def forward(self, ego_traj: Tensor, neighbor_traj: Tensor) -> tuple[Tensor, Tensor]:
        """Predict multimodal future trajectories with per-mode probabilities.

        Parameters
        ----------
        ego_traj : Tensor
            Ego agent observed trajectory, shape ``(batch, obs_steps, state_dim)``.
        neighbor_traj : Tensor
            Neighboring agents' observed trajectories,
            shape ``(batch, num_neighbors, obs_steps, state_dim)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Predicted trajectories ``(batch, num_modes, future_steps, state_dim)``
            and per-mode log-probabilities ``(batch, num_modes)``.
        """
        batch = ego_traj.shape[0]
        ego_token = self.traj_proj(ego_traj).mean(dim=1, keepdim=True)

        modes = self.mode_queries.expand(batch, -1, -1)
        mode_and_ego = torch.cat([modes, ego_token], dim=1)
        mode_and_ego = self.mode_encoder(mode_and_ego)
        mode_tokens = mode_and_ego[:, : self.num_modes]

        num_neighbors = neighbor_traj.shape[1]
        neighbor_tokens = self.neighbor_proj(neighbor_traj).mean(dim=2)
        neighbor_tokens = neighbor_tokens.view(batch, num_neighbors, self.hidden_dim)

        social_tokens, _ = self.social_cross_attn(mode_tokens, neighbor_tokens, neighbor_tokens)

        trajs = self.traj_head(social_tokens).view(
            batch, self.num_modes, self.future_steps, self.state_dim
        )
        log_probs = self.prob_head(social_tokens).squeeze(-1)
        return trajs, log_probs


def build_tutr() -> nn.Module:
    """Build a compact TUTR model.

    Returns
    -------
    nn.Module
        Random-initialized ``TUTR`` in eval mode.
    """
    return TUTR().eval()


def example_input_tutr() -> tuple[Tensor, Tensor]:
    """Create example ego and neighbor trajectory tensors.

    Returns
    -------
    tuple[Tensor, Tensor]
        Ego trajectory ``(1, 8, 2)`` and neighbor trajectories ``(1, 4, 8, 2)``.
    """
    ego_traj = torch.randn(1, 8, 2)
    neighbor_traj = torch.randn(1, 4, 8, 2)
    return ego_traj, neighbor_traj


# ---------------------------------------------------------------------------
# UniDrive-WM
# ---------------------------------------------------------------------------
class UniDriveWM(nn.Module):
    """Unified understanding / planning / generation world model (UniDrive-WM).

    A shared token encoder feeds an autoregressive GRU trajectory planner;
    the planned trajectory conditions a small iterative denoiser that
    generates the next scene latent, which is fed back as context for the
    next planning step -- coupling planning and generation in one loop.
    """

    def __init__(
        self,
        token_dim: int = 12,
        hidden_dim: int = 16,
        num_scene_tokens: int = 6,
        plan_steps: int = 3,
        denoise_steps: int = 2,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.plan_steps = plan_steps
        self.denoise_steps = denoise_steps

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=2, dim_feedforward=hidden_dim * 2, batch_first=True
        )
        self.token_proj = nn.Linear(token_dim, hidden_dim)
        self.shared_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.plan_cell = nn.GRUCell(2, hidden_dim)
        self.waypoint_head = nn.Linear(hidden_dim, 2)

        self.traj_cond_proj = nn.Linear(2, hidden_dim)
        self.denoise_step = nn.Linear(hidden_dim * 2, hidden_dim)
        self.latent_to_token = nn.Linear(hidden_dim, token_dim)

    def forward(self, scene_tokens: Tensor) -> tuple[Tensor, Tensor]:
        """Jointly plan a future ego trajectory and generate future scene tokens.

        Parameters
        ----------
        scene_tokens : Tensor
            Current-frame scene/VLM tokens, shape
            ``(batch, num_scene_tokens, token_dim)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Planned waypoints ``(batch, plan_steps, 2)`` and the final
            generated scene tokens ``(batch, num_scene_tokens, token_dim)``.
        """
        batch, num_tokens, token_dim = scene_tokens.shape
        current_tokens = scene_tokens
        hidden = torch.zeros(
            batch, self.hidden_dim, device=scene_tokens.device, dtype=scene_tokens.dtype
        )
        current_wp = torch.zeros(batch, 2, device=scene_tokens.device, dtype=scene_tokens.dtype)

        waypoints = []
        for _ in range(self.plan_steps):
            encoded = self.shared_encoder(self.token_proj(current_tokens))
            pooled = encoded.mean(dim=1)
            hidden = self.plan_cell(current_wp, hidden + pooled)
            current_wp = current_wp + self.waypoint_head(hidden)
            waypoints.append(current_wp)

            # Trajectory-conditioned iterative denoising generates the next
            # scene latent, fed back as context for the next planning step.
            traj_cond = self.traj_cond_proj(current_wp).unsqueeze(1).expand(-1, num_tokens, -1)
            latent = encoded
            for _ in range(self.denoise_steps):
                latent = self.denoise_step(torch.cat([latent, traj_cond], dim=-1))
            current_tokens = self.latent_to_token(latent)

        return torch.stack(waypoints, dim=1), current_tokens


def build_unidrive_wm() -> nn.Module:
    """Build a compact UniDrive-WM model.

    Returns
    -------
    nn.Module
        Random-initialized ``UniDriveWM`` in eval mode.
    """
    return UniDriveWM().eval()


def example_input_unidrive_wm() -> Tensor:
    """Create an example current-frame scene-token tensor.

    Returns
    -------
    Tensor
        Scene tokens, shape ``(1, 6, 12)``.
    """
    return torch.randn(1, 6, 12)


# ---------------------------------------------------------------------------
# Urban Driver
# ---------------------------------------------------------------------------
class UrbanDriver(nn.Module):
    """Vectorized closed-loop imitation planner (Urban Driver).

    Every scene polyline (ego path, agent tracks, lane segments) is encoded
    independently by a shared PointNet-style per-point MLP + max-pool
    subgraph encoder; the resulting per-element vectors are aggregated by a
    single global self-attention layer, and the ego-conditioned pooled
    output drives an MLP policy head over future waypoints.
    """

    def __init__(self, point_dim: int = 3, hidden_dim: int = 16, future_steps: int = 6) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.future_steps = future_steps

        self.point_mlp = nn.Sequential(
            nn.Linear(point_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.global_attn = nn.MultiheadAttention(hidden_dim, num_heads=2, batch_first=True)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, future_steps * 2),
        )

    def forward(self, polylines: Tensor) -> Tensor:
        """Predict future ego waypoints from a set of vectorized scene polylines.

        Parameters
        ----------
        polylines : Tensor
            Scene polylines (ego path first), shape
            ``(batch, num_elements, points_per_polyline, point_dim)``.

        Returns
        -------
        Tensor
            Predicted ego waypoints, shape ``(batch, future_steps, 2)``.
        """
        batch, num_elements, _, _ = polylines.shape
        point_feat = self.point_mlp(polylines)
        element_vecs = point_feat.max(dim=2).values

        attended, _ = self.global_attn(element_vecs, element_vecs, element_vecs)
        ego_context = attended[:, 0, :]

        out = self.policy_head(ego_context)
        return out.view(batch, self.future_steps, 2)


def build_urbandriver() -> nn.Module:
    """Build a compact Urban Driver model.

    Returns
    -------
    nn.Module
        Random-initialized ``UrbanDriver`` in eval mode.
    """
    return UrbanDriver().eval()


def example_input_urbandriver() -> Tensor:
    """Create an example vectorized scene-polyline tensor.

    Returns
    -------
    Tensor
        Polylines, shape ``(1, 7, 5, 3)`` (7 elements: 1 ego + 6 agents/lanes,
        5 points each, ``(x, y, type)`` per point).
    """
    return torch.randn(1, 7, 5, 3)


MENAGERIE_ENTRIES = [
    ("Trajectron++", "build_trajectronpp", "example_input_trajectronpp", "2020", "SEQ"),
    ("TransFuser", "build_transfuser", "example_input_transfuser", "2021", "VIS"),
    ("TraPHic", "build_traphic", "example_input_traphic", "2019", "SEQ"),
    ("TUTR", "build_tutr", "example_input_tutr", "2023", "SEQ"),
    ("UniDrive-WM", "build_unidrive_wm", "example_input_unidrive_wm", "2026", "GEN"),
    ("UrbanDriver", "build_urbandriver", "example_input_urbandriver", "2021", "SEQ"),
]
