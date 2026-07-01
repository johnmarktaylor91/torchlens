"""Compact faithful reimplementations for autonomous-driving perception/forecasting rows.

Sources checked (GitHub API contents + raw file reads, no clone/pip-install):

* FIERY -- https://github.com/wayveai/fiery (arxiv:2104.10490). Read
  ``fiery/models/fiery.py`` for the lift-splat-shoot camera encoder ->
  temporal warp -> present/future Gaussian distribution -> GRU-based future
  prediction -> BEV instance-segmentation decoder pipeline.
* Forecast-MAE -- https://github.com/jchengai/forecast-mae (arxiv:2308.09882).
  Read ``src/model/model_mae.py`` for the masked-autoencoder pretraining
  scheme: separate history/future/lane token embedders, random per-agent and
  per-lane masking, a shared transformer encoder over the visible tokens, and
  a lightweight decoder with learned mask tokens that reconstructs history,
  future, and lane geometry.
* GameFormer -- https://github.com/MCZhi/GameFormer (arxiv:2303.05046). Read
  ``model/modules.py`` for the hierarchical game-theoretic decoder: an
  ``InitialDecoder`` producing level-0 multi-modal GMM predictions per agent,
  followed by stacked ``InteractionDecoder`` levels that re-encode every
  agent's current-level trajectory distribution and cross-attend each agent's
  query against the *other* agents' encoded futures (Level-K iterative best
  response).
* GATraj -- https://github.com/mengmengliu1998/GATraj (arxiv:2209.07857).
  Read ``basemodel.py`` for the temporal encoder (causal 1D conv + Transformer
  encoder + LSTM) and the ``Global_interaction`` graph module: a relative
  position + motion-gate + attention message-passing block run for
  ``pass_time`` rounds, feeding a Laplacian mixture-density trajectory
  decoder.
* GenAD -- https://github.com/wzzheng/GenAD (arxiv:2402.11502). Read
  ``projects/mmdet3d_plugin/GenAD/generator/distributions.py`` and
  ``.../state_prediction.py`` and ``GenAD_head.py`` for the generative
  end-to-end design: DETR-style agent/map queries decoded from a scene
  feature map, a present-vs-future diagonal-Gaussian ``DistributionModule``
  pair (VAE latent, trained with a KL term), and a GRU ``PredictModel`` that
  rolls the sampled latent forward into a future trajectory.
* GRIP -- https://github.com/xincoder/GRIP (arxiv:1907.07792). Read
  ``model.py``, ``layers/graph_conv_block.py``, ``layers/graph_operation_layer.py``,
  and ``layers/seq2seq.py`` for the spatio-temporal graph conv (``einsum``
  graph convolution with a learned edge-importance weighting, wrapped in a
  temporal-conv residual block) feeding a GRU encoder-decoder that predicts
  trajectories as residual offsets from the last observed location.

All models below use small random-init dimensions; they are architecture
catalog entries, not trained-weight replicas.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

# ---------------------------------------------------------------------------
# FIERY -- lift-splat BEV encoder + temporal GRU + probabilistic future
# ---------------------------------------------------------------------------


class FieryCamEncoder(nn.Module):
    """Per-camera image feature extractor with a depth-probability head.

    Approximates the lift-splat-shoot ``Encoder``: a small CNN backbone
    produces per-pixel features and a depth distribution, which are combined
    by an outer product to "lift" 2D features into a discrete-depth frustum.
    """

    def __init__(self, in_channels: int = 3, feat_dim: int = 16, depth_bins: int = 8) -> None:
        """Build the backbone, depth head, and feature head.

        Parameters
        ----------
        in_channels:
            Number of input image channels.
        feat_dim:
            Channel width of the lifted per-pixel feature.
        depth_bins:
            Number of discrete depth bins in the frustum.
        """

        super().__init__()
        self.depth_bins = depth_bins
        self.feat_dim = feat_dim
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 24, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 24, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.depth_head = nn.Conv2d(24, depth_bins, 1)
        self.feat_head = nn.Conv2d(24, feat_dim, 1)

    def forward(self, image: Tensor) -> Tensor:
        """Lift a batch of camera images into a depth-feature frustum.

        Parameters
        ----------
        image:
            Camera images with shape ``(B, 3, H, W)``.

        Returns
        -------
        Tensor
            Lifted frustum features with shape ``(B, feat_dim, depth_bins, H', W')``.
        """

        x = self.backbone(image)
        depth = torch.softmax(self.depth_head(x), dim=1)
        feat = self.feat_head(x)
        return depth.unsqueeze(1) * feat.unsqueeze(2)


class FieryDistribution(nn.Module):
    """Diagonal Gaussian distribution head over a pooled BEV state."""

    def __init__(self, in_channels: int, latent_dim: int) -> None:
        """Build the pooling + linear parametrization of mu and log-sigma.

        Parameters
        ----------
        in_channels:
            Channel width of the input BEV state map.
        latent_dim:
            Dimensionality of the latent Gaussian.
        """

        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(in_channels, 2 * latent_dim)
        self.latent_dim = latent_dim

    def forward(self, state: Tensor) -> tuple[Tensor, Tensor]:
        """Parametrize a diagonal Gaussian from a pooled BEV state.

        Parameters
        ----------
        state:
            BEV feature map with shape ``(B, C, H, W)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Mean and log-sigma, each shape ``(B, latent_dim)``.
        """

        pooled = self.pool(state).flatten(1)
        mu_log_sigma = self.proj(pooled)
        mu, log_sigma = mu_log_sigma.chunk(2, dim=-1)
        return mu, log_sigma.clamp(-5.0, 5.0)


class Fiery(nn.Module):
    """Compact FIERY: lift-splat encoder, temporal GRU, probabilistic future."""

    def __init__(
        self,
        receptive_field: int = 3,
        n_future: int = 2,
        bev_size: int = 12,
        feat_dim: int = 16,
        latent_dim: int = 8,
    ) -> None:
        """Build the camera encoder, BEV projector, temporal GRU, and decoder.

        Parameters
        ----------
        receptive_field:
            Number of past frames (including present) consumed per camera.
        n_future:
            Number of future frames to roll out.
        bev_size:
            Spatial size of the bird's-eye-view grid.
        feat_dim:
            Channel width of BEV features.
        latent_dim:
            Dimensionality of the probabilistic future latent.
        """

        super().__init__()
        self.receptive_field = receptive_field
        self.n_future = n_future
        self.bev_size = bev_size
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim

        self.encoder = FieryCamEncoder(feat_dim=feat_dim)
        self.bev_project = nn.Conv3d(feat_dim, feat_dim, kernel_size=1)
        self.temporal_gru = nn.GRUCell(
            feat_dim * bev_size * bev_size, feat_dim * bev_size * bev_size
        )

        self.present_distribution = FieryDistribution(feat_dim, latent_dim)
        self.future_distribution = FieryDistribution(feat_dim, latent_dim)
        self.future_gru = nn.GRUCell(latent_dim, feat_dim * bev_size * bev_size)
        self.decoder = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim, 2, 1),
        )

    def forward(self, images: Tensor) -> Tensor:
        """Predict a sequence of future BEV instance-segmentation logits.

        Parameters
        ----------
        images:
            Camera sequence with shape ``(B, T, 3, H, W)`` where
            ``T == receptive_field``.

        Returns
        -------
        Tensor
            Future BEV logits with shape ``(B, n_future, 2, bev_size, bev_size)``.
        """

        b, t, c, h, w = images.shape
        state = images.new_zeros(b, self.feat_dim * self.bev_size * self.bev_size)
        for step in range(t):
            frustum = self.encoder(images[:, step])
            bev = frustum.mean(dim=2)
            bev = F.adaptive_avg_pool2d(bev, (self.bev_size, self.bev_size))
            bev = self.bev_project(bev.unsqueeze(2)).squeeze(2)
            state = self.temporal_gru(bev.flatten(1), state)

        present_state = state.view(b, self.feat_dim, self.bev_size, self.bev_size)
        mu, log_sigma = self.present_distribution(present_state)
        sample = mu + torch.exp(log_sigma) * torch.randn_like(mu)

        future_logits = []
        rolling = state
        for _ in range(self.n_future):
            rolling = self.future_gru(sample, rolling)
            bev_state = rolling.view(b, self.feat_dim, self.bev_size, self.bev_size)
            future_logits.append(self.decoder(bev_state))
        return torch.stack(future_logits, dim=1)


def build_fiery() -> nn.Module:
    """Build a compact random-init FIERY model."""

    return Fiery().eval()


def example_input_fiery() -> Tensor:
    """Return a short camera sequence for FIERY."""

    return torch.randn(1, 3, 3, 32, 32)


# ---------------------------------------------------------------------------
# Forecast-MAE -- masked autoencoder pretraining over agents + lanes
# ---------------------------------------------------------------------------


class MaeTokenEmbed(nn.Module):
    """1D-conv token embedder for a per-agent history/future sequence."""

    def __init__(self, in_channels: int, embed_dim: int) -> None:
        """Build the conv stack that embeds a temporal sequence into one token.

        Parameters
        ----------
        in_channels:
            Number of per-timestep input features.
        embed_dim:
            Output embedding dimension.
        """

        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, embed_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, seq: Tensor) -> Tensor:
        """Embed a batch of temporal sequences into single tokens.

        Parameters
        ----------
        seq:
            Sequence tensor with shape ``(N, in_channels, T)``.

        Returns
        -------
        Tensor
            Token embeddings with shape ``(N, embed_dim)``.
        """

        return self.conv(seq).squeeze(-1)


class MaeTransformerBlock(nn.Module):
    """Pre-norm self-attention + MLP transformer block."""

    def __init__(self, dim: int, num_heads: int) -> None:
        """Build attention, MLP, and pre-norm layers.

        Parameters
        ----------
        dim:
            Token embedding dimension.
        num_heads:
            Number of self-attention heads.
        """

        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x: Tensor) -> Tensor:
        """Apply one pre-norm self-attention + MLP block.

        Parameters
        ----------
        x:
            Token sequence with shape ``(B, N, dim)``.

        Returns
        -------
        Tensor
            Updated token sequence, same shape as ``x``.
        """

        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class ForecastMae(nn.Module):
    """Compact Forecast-MAE: masked history/future/lane token autoencoding."""

    def __init__(
        self,
        embed_dim: int = 32,
        encoder_depth: int = 2,
        decoder_depth: int = 2,
        num_heads: int = 4,
        hist_steps: int = 5,
        fut_steps: int = 6,
        lane_pts: int = 4,
    ) -> None:
        """Build the token embedders, encoder/decoder stacks, and mask tokens.

        Parameters
        ----------
        embed_dim:
            Shared token embedding width.
        encoder_depth:
            Number of visible-token encoder blocks.
        decoder_depth:
            Number of full-sequence decoder blocks.
        num_heads:
            Attention head count for both stacks.
        hist_steps:
            Number of observed history timesteps per agent.
        fut_steps:
            Number of future timesteps to reconstruct per agent.
        lane_pts:
            Number of points per lane polyline.
        """

        super().__init__()
        self.hist_steps = hist_steps
        self.fut_steps = fut_steps
        self.lane_pts = lane_pts

        self.hist_embed = MaeTokenEmbed(2, embed_dim)
        self.future_embed = MaeTokenEmbed(2, embed_dim)
        self.lane_embed = MaeTokenEmbed(2, embed_dim)

        self.encoder = nn.ModuleList(
            MaeTransformerBlock(embed_dim, num_heads) for _ in range(encoder_depth)
        )
        self.enc_norm = nn.LayerNorm(embed_dim)

        self.decoder_embed = nn.Linear(embed_dim, embed_dim)
        self.decoder = nn.ModuleList(
            MaeTransformerBlock(embed_dim, num_heads) for _ in range(decoder_depth)
        )
        self.dec_norm = nn.LayerNorm(embed_dim)

        self.hist_mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.future_mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.lane_mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.hist_pred = nn.Linear(embed_dim, hist_steps * 2)
        self.future_pred = nn.Linear(embed_dim, fut_steps * 2)
        self.lane_pred = nn.Linear(embed_dim, lane_pts * 2)

    def forward(
        self, history: Tensor, future: Tensor, lanes: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Randomly mask agent/lane tokens, encode the rest, and reconstruct all.

        Parameters
        ----------
        history:
            Observed agent trajectories with shape ``(B, A, hist_steps, 2)``.
        future:
            Ground-truth future agent trajectories with shape
            ``(B, A, fut_steps, 2)``.
        lanes:
            Lane polylines with shape ``(B, L, lane_pts, 2)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Reconstructed history, future, and lane geometry.
        """

        b, a, _, _ = history.shape
        _, num_lanes, _, _ = lanes.shape

        hist_tok = self.hist_embed(history.reshape(b * a, self.hist_steps, 2).transpose(1, 2)).view(
            b, a, -1
        )
        fut_tok = self.future_embed(future.reshape(b * a, self.fut_steps, 2).transpose(1, 2)).view(
            b, a, -1
        )
        lane_tok = self.lane_embed(
            lanes.reshape(b * num_lanes, self.lane_pts, 2).transpose(1, 2)
        ).view(b, num_lanes, -1)

        # Keep every other agent token and every other lane token visible
        # (deterministic 50% masking so the smoke trace has fixed shapes).
        keep_hist = hist_tok[:, ::2]
        keep_fut = fut_tok[:, 1::2] if a > 1 else fut_tok[:, :0]
        keep_lane = lane_tok[:, ::2]

        visible = torch.cat([keep_hist, keep_fut, keep_lane], dim=1)
        for blk in self.encoder:
            visible = blk(visible)
        visible = self.enc_norm(visible)
        visible = self.decoder_embed(visible)

        n_hist, n_fut, n_lane = keep_hist.shape[1], keep_fut.shape[1], keep_lane.shape[1]
        enc_hist, enc_fut, enc_lane = visible.split([n_hist, n_fut, n_lane], dim=1)

        full_hist = self.hist_mask_token.expand(b, a, -1).clone()
        full_hist[:, ::2] = enc_hist
        full_fut = self.future_mask_token.expand(b, a, -1).clone()
        if n_fut > 0:
            full_fut[:, 1::2] = enc_fut
        full_lane = self.lane_mask_token.expand(b, num_lanes, -1).clone()
        full_lane[:, ::2] = enc_lane

        full = torch.cat([full_hist, full_fut, full_lane], dim=1)
        for blk in self.decoder:
            full = blk(full)
        full = self.dec_norm(full)

        dec_hist, dec_fut, dec_lane = full.split([a, a, num_lanes], dim=1)
        hist_hat = self.hist_pred(dec_hist).view(b, a, self.hist_steps, 2)
        fut_hat = self.future_pred(dec_fut).view(b, a, self.fut_steps, 2)
        lane_hat = self.lane_pred(dec_lane).view(b, num_lanes, self.lane_pts, 2)
        return hist_hat, fut_hat, lane_hat


def build_forecast_mae() -> nn.Module:
    """Build a compact random-init Forecast-MAE model."""

    return ForecastMae().eval()


def example_input_forecast_mae() -> tuple[Tensor, Tensor, Tensor]:
    """Return history, future, and lane tensors for Forecast-MAE."""

    return (torch.randn(1, 6, 5, 2), torch.randn(1, 6, 6, 2), torch.randn(1, 4, 4, 2))


# ---------------------------------------------------------------------------
# GameFormer -- hierarchical game-theoretic Level-K transformer decoder
# ---------------------------------------------------------------------------


class GmmPredictor(nn.Module):
    """Multi-modal Gaussian-mixture trajectory head with mode scores."""

    def __init__(self, dim: int, future_len: int, modes: int) -> None:
        """Build the per-mode trajectory and score MLP heads.

        Parameters
        ----------
        dim:
            Query embedding dimension.
        future_len:
            Number of future timesteps predicted per mode.
        modes:
            Number of Gaussian mixture modes.
        """

        super().__init__()
        self.future_len = future_len
        self.modes = modes
        self.traj = nn.Sequential(nn.Linear(dim, dim), nn.ELU(), nn.Linear(dim, future_len * 4))
        self.score = nn.Sequential(nn.Linear(dim, dim // 2), nn.ELU(), nn.Linear(dim // 2, 1))

    def forward(self, query: Tensor) -> tuple[Tensor, Tensor]:
        """Decode per-mode Gaussian trajectory parameters and mode scores.

        Parameters
        ----------
        query:
            Per-agent, per-mode query content with shape ``(B, modes, dim)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Trajectory parameters ``(B, modes, future_len, 4)`` and mode
            scores ``(B, modes)``.
        """

        b, m, _ = query.shape
        traj = self.traj(query).view(b, m, self.future_len, 4)
        score = self.score(query).squeeze(-1)
        return traj, score


class GameFormerCrossAttn(nn.Module):
    """Cross-attention + FFN block used for query/context interaction."""

    def __init__(self, dim: int, heads: int) -> None:
        """Build cross-attention and feed-forward sublayers.

        Parameters
        ----------
        dim:
            Embedding dimension.
        heads:
            Number of attention heads.
        """

        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, query: Tensor, context: Tensor) -> Tensor:
        """Cross-attend a query sequence to a context sequence.

        Parameters
        ----------
        query:
            Query tokens with shape ``(B, M, dim)``.
        context:
            Context tokens with shape ``(B, N, dim)``.

        Returns
        -------
        Tensor
            Updated query tokens with shape ``(B, M, dim)``.
        """

        attn_out, _ = self.attn(query, context, context)
        query = self.norm1(attn_out)
        query = self.norm2(self.ffn(query) + query)
        return query


class GameFormer(nn.Module):
    """Compact GameFormer: Level-K hierarchical joint trajectory prediction."""

    def __init__(
        self,
        dim: int = 32,
        num_agents: int = 4,
        num_levels: int = 2,
        modes: int = 3,
        future_len: int = 6,
        heads: int = 4,
    ) -> None:
        """Build the encoder, per-level GMM decoders, and interaction attention.

        Parameters
        ----------
        dim:
            Shared embedding dimension.
        num_agents:
            Number of jointly-predicted agents.
        num_levels:
            Number of Level-K game-theoretic reasoning rounds.
        modes:
            Number of Gaussian-mixture modes per agent per level.
        future_len:
            Number of predicted future timesteps.
        heads:
            Attention head count.
        """

        super().__init__()
        self.num_agents = num_agents
        self.num_levels = num_levels
        self.modes = modes
        self.future_len = future_len

        self.agent_encoder = nn.LSTM(4, dim, batch_first=True)
        self.modal_query = nn.Parameter(torch.randn(modes, dim) * 0.02)
        self.agent_query = nn.Parameter(torch.randn(num_agents, dim) * 0.02)

        self.level0_attn = GameFormerCrossAttn(dim, heads)
        self.level0_head = GmmPredictor(dim, future_len, modes)

        self.future_encoder = nn.Sequential(
            nn.Linear(2, dim), nn.ReLU(inplace=True), nn.Linear(dim, dim)
        )
        self.interaction_attn = nn.ModuleList(
            GameFormerCrossAttn(dim, heads) for _ in range(num_levels)
        )
        self.level_heads = nn.ModuleList(
            GmmPredictor(dim, future_len, modes) for _ in range(num_levels)
        )

    def forward(self, agent_history: Tensor) -> tuple[Tensor, Tensor]:
        """Run Level-0 initial prediction followed by iterative Level-K refinement.

        Parameters
        ----------
        agent_history:
            Per-agent observed states with shape ``(B, num_agents, T, 4)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Final-level trajectories ``(B, num_agents, modes, future_len, 4)``
            and mode scores ``(B, num_agents, modes)``.
        """

        b, n, t, _ = agent_history.shape
        _, (h_n, _) = self.agent_encoder(agent_history.reshape(b * n, t, 4))
        encoding = h_n.squeeze(0).view(b, n, -1)

        trajs = []
        scores = []
        contents = []
        for agent_idx in range(n):
            query = (
                encoding[:, agent_idx : agent_idx + 1]
                + self.modal_query.unsqueeze(0)
                + self.agent_query[agent_idx].view(1, 1, -1)
            )
            content = self.level0_attn(query, encoding)
            traj, score = self.level0_head(content)
            traj = traj + encoding.new_zeros(1)
            trajs.append(traj)
            scores.append(score)
            contents.append(content)

        level_trajs = torch.stack(trajs, dim=1)
        level_scores = torch.stack(scores, dim=1)
        level_contents = torch.stack(contents, dim=1)

        for level in range(self.num_levels):
            weights = torch.softmax(level_scores, dim=-1).unsqueeze(-1).unsqueeze(-1)
            weighted_xy = (level_trajs[..., :2] * weights).sum(dim=2)
            future_feat = self.future_encoder(weighted_xy).mean(dim=2)

            new_trajs = []
            new_scores = []
            new_contents = []
            for agent_idx in range(n):
                context = torch.cat([future_feat, encoding], dim=1)
                query = level_contents[:, agent_idx] + future_feat[:, agent_idx : agent_idx + 1]
                content = self.interaction_attn[level](query, context)
                traj, score = self.level_heads[level](content)
                new_trajs.append(traj)
                new_scores.append(score)
                new_contents.append(content)
            level_trajs = torch.stack(new_trajs, dim=1)
            level_scores = torch.stack(new_scores, dim=1)
            level_contents = torch.stack(new_contents, dim=1)

        return level_trajs, level_scores


def build_gameformer() -> nn.Module:
    """Build a compact random-init GameFormer model."""

    return GameFormer().eval()


def example_input_gameformer() -> Tensor:
    """Return a batch of per-agent observed histories for GameFormer."""

    return torch.randn(1, 4, 5, 4)


# ---------------------------------------------------------------------------
# GATraj -- graph attention temporal encoder + Laplacian mixture decoder
# ---------------------------------------------------------------------------


class GaTrajTemporalEncoder(nn.Module):
    """Causal 1D conv + Transformer encoder + LSTM temporal encoder."""

    def __init__(self, hidden_size: int = 32, heads: int = 4) -> None:
        """Build the conv, transformer, and LSTM temporal stack.

        Parameters
        ----------
        hidden_size:
            Shared feature width.
        heads:
            Number of Transformer self-attention heads.
        """

        super().__init__()
        self.conv1d = nn.Conv1d(2, hidden_size, kernel_size=3, stride=1, padding=1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=heads, dim_feedforward=hidden_size, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Encode per-agent observed trajectories into a state and cell vector.

        Parameters
        ----------
        x:
            Trajectories with shape ``(N, 2, H)`` (channels-first, ``H``
            observed timesteps).

        Returns
        -------
        tuple[Tensor, Tensor]
            Hidden state and cell state, each shape ``(N, hidden_size)``.
        """

        dense = self.conv1d(x).permute(0, 2, 1)
        dense = self.transformer_encoder(dense) + dense
        _, (hn, cn) = self.lstm(dense)
        return hn.squeeze(0), cn.squeeze(0)


class GaTrajGlobalInteraction(nn.Module):
    """Relative-position motion-gated attention message passing over agents."""

    def __init__(self, hidden_size: int = 32) -> None:
        """Build the motion gate, relative-position embedding, and attention MLPs.

        Parameters
        ----------
        hidden_size:
            Per-agent hidden state width.
        """

        super().__init__()
        self.hidden_size = hidden_size
        self.relative_layer = nn.Linear(2, hidden_size)
        self.gate = nn.Sequential(nn.Linear(hidden_size * 3, hidden_size), nn.Sigmoid())
        self.attn = nn.Linear(hidden_size * 3, 1)
        self.weight = nn.Linear(hidden_size, hidden_size)

    def forward(self, positions: Tensor, hidden: Tensor, cell: Tensor) -> tuple[Tensor, Tensor]:
        """Refine agent states with one round of gated attention message passing.

        Parameters
        ----------
        positions:
            Current agent 2D positions with shape ``(N, 2)``.
        hidden:
            Current per-agent hidden state with shape ``(N, hidden_size)``.
        cell:
            Current per-agent cell state with shape ``(N, hidden_size)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Refined hidden state and cell state, both shape ``(N, hidden_size)``.
        """

        n = positions.shape[0]
        rel = positions.unsqueeze(1) - positions.unsqueeze(0)
        rel_embed = self.relative_layer(rel)

        hi = hidden.unsqueeze(0).expand(n, n, -1)
        hj = hidden.unsqueeze(1).expand(n, n, -1)
        combined = torch.cat([rel_embed, hi, hj], dim=-1)

        gate = self.gate(combined)
        score = self.attn(combined).squeeze(-1)
        eye_mask = torch.eye(n, device=positions.device, dtype=torch.bool)
        score = score.masked_fill(eye_mask, -1e4)
        weights = torch.softmax(score, dim=-1).unsqueeze(-1)

        messages = (hj * gate) * weights
        aggregated = self.weight(messages.sum(dim=1))
        new_cell = aggregated + cell
        new_hidden = hidden + torch.tanh(new_cell)
        return new_hidden, new_cell


class LaplacianDecoder(nn.Module):
    """Laplacian mixture-density trajectory decoder."""

    def __init__(self, hidden_size: int, future_len: int, modes: int) -> None:
        """Build the GRU rollout and per-mode Laplace parameter heads.

        Parameters
        ----------
        hidden_size:
            Encoder hidden state width.
        future_len:
            Number of future timesteps to predict.
        modes:
            Number of Laplace mixture modes.
        """

        super().__init__()
        self.future_len = future_len
        self.modes = modes
        self.gru = nn.GRUCell(hidden_size, hidden_size)
        self.loc_scale = nn.Linear(hidden_size, modes * 4)
        self.pi = nn.Linear(hidden_size, modes)

    def forward(self, hidden: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Roll a GRU forward to predict a Laplacian trajectory mixture.

        Parameters
        ----------
        hidden:
            Encoder hidden state with shape ``(N, hidden_size)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Location ``(modes, N, future_len, 2)``, scale (same shape), and
            mixture logits ``(N, modes)``.
        """

        n = hidden.shape[0]
        state = hidden
        locs, scales = [], []
        for _ in range(self.future_len):
            state = self.gru(hidden, state)
            out = self.loc_scale(state).view(n, self.modes, 4)
            locs.append(out[..., :2])
            scales.append(F.softplus(out[..., 2:]) + 1e-3)
        loc = torch.stack(locs, dim=2).permute(1, 0, 2, 3)
        scale = torch.stack(scales, dim=2).permute(1, 0, 2, 3)
        pi = self.pi(hidden)
        return loc, scale, pi


class GaTraj(nn.Module):
    """Compact GATraj: graph-attention social encoder + Laplacian decoder."""

    def __init__(
        self, hidden_size: int = 32, pass_time: int = 2, future_len: int = 6, modes: int = 3
    ) -> None:
        """Build the temporal encoder, interaction rounds, and mixture decoder.

        Parameters
        ----------
        hidden_size:
            Shared per-agent hidden width.
        pass_time:
            Number of graph message-passing rounds.
        future_len:
            Number of future timesteps predicted.
        modes:
            Number of Laplace mixture modes.
        """

        super().__init__()
        self.temporal_encoder = GaTrajTemporalEncoder(hidden_size)
        self.interaction_rounds = nn.ModuleList(
            GaTrajGlobalInteraction(hidden_size) for _ in range(pass_time)
        )
        self.decoder = LaplacianDecoder(hidden_size, future_len, modes)

    def forward(self, trajectories: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Predict a Laplacian trajectory mixture for a scene of agents.

        Parameters
        ----------
        trajectories:
            Observed per-agent trajectories with shape ``(N, H, 2)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Location, scale, and mixture logits from :class:`LaplacianDecoder`.
        """

        x = trajectories.permute(0, 2, 1)
        hidden, cell = self.temporal_encoder(x)
        last_pos = trajectories[:, -1]
        for block in self.interaction_rounds:
            hidden, cell = block(last_pos, hidden, cell)
        return self.decoder(hidden)


def build_gatraj() -> nn.Module:
    """Build a compact random-init GATraj model."""

    return GaTraj().eval()


def example_input_gatraj() -> Tensor:
    """Return a scene of observed agent trajectories for GATraj."""

    return torch.randn(5, 8, 2)


# ---------------------------------------------------------------------------
# GenAD -- generative end-to-end AD with VAE-latent trajectory rollout
# ---------------------------------------------------------------------------


class GenAdDistribution(nn.Module):
    """Diagonal Gaussian distribution head over pooled scene queries."""

    def __init__(self, in_dim: int, latent_dim: int) -> None:
        """Build the compression and Gaussian-parameter MLP.

        Parameters
        ----------
        in_dim:
            Input scene-query embedding dimension.
        latent_dim:
            Output latent dimensionality.
        """

        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Linear(in_dim // 2, 2 * latent_dim)
        self.latent_dim = latent_dim

    def forward(self, scene: Tensor) -> tuple[Tensor, Tensor]:
        """Parametrize a diagonal Gaussian from pooled scene features.

        Parameters
        ----------
        scene:
            Scene query tokens with shape ``(B, N, in_dim)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Mean and log-sigma, each shape ``(B, latent_dim)``.
        """

        pooled = scene.mean(dim=1)
        mu_log_sigma = self.head(self.encoder(pooled))
        mu, log_sigma = mu_log_sigma.chunk(2, dim=-1)
        return mu, log_sigma.clamp(-5.0, 5.0)


class GenAdPredictModel(nn.Module):
    """GRU rollout that predicts a future trajectory from a sampled latent."""

    def __init__(self, latent_dim: int, hidden_dim: int, out_dim: int, future_len: int) -> None:
        """Build the GRU cell and MLP output head.

        Parameters
        ----------
        latent_dim:
            Dimensionality of the sampled VAE latent.
        hidden_dim:
            GRU hidden width.
        out_dim:
            Per-timestep output dimensionality (e.g. 2 for ``(x, y)``).
        future_len:
            Number of future steps to roll out.
        """

        super().__init__()
        self.future_len = future_len
        self.gru = nn.GRUCell(latent_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, out_dim),
        )

    def forward(self, latent: Tensor, init_hidden: Tensor) -> Tensor:
        """Roll the GRU forward with a fixed latent input at every step.

        Parameters
        ----------
        latent:
            Sampled latent with shape ``(B, latent_dim)``.
        init_hidden:
            Initial GRU hidden state with shape ``(B, hidden_dim)``.

        Returns
        -------
        Tensor
            Predicted future trajectory with shape ``(B, future_len, out_dim)``.
        """

        state = init_hidden
        outputs = []
        for _ in range(self.future_len):
            state = self.gru(latent, state)
            outputs.append(self.head(state))
        return torch.stack(outputs, dim=1)


class GenAd(nn.Module):
    """Compact GenAD: DETR-style agent queries + VAE-latent trajectory generator."""

    def __init__(
        self,
        scene_tokens: int = 16,
        feat_dim: int = 32,
        num_agents: int = 4,
        latent_dim: int = 8,
        future_len: int = 6,
        heads: int = 4,
    ) -> None:
        """Build the scene encoder, agent-query decoder, and VAE trajectory head.

        Parameters
        ----------
        scene_tokens:
            Number of flattened BEV scene feature tokens.
        feat_dim:
            Shared embedding dimension.
        num_agents:
            Number of agent detection/trajectory queries.
        latent_dim:
            Dimensionality of the present/future VAE latent.
        future_len:
            Number of future trajectory timesteps predicted per agent.
        heads:
            Number of attention heads in the query decoder.
        """

        super().__init__()
        self.num_agents = num_agents
        self.future_len = future_len

        self.scene_proj = nn.Linear(feat_dim, feat_dim)
        self.agent_queries = nn.Parameter(torch.randn(num_agents, feat_dim) * 0.02)
        self.query_decoder = GameFormerCrossAttn(feat_dim, heads)

        self.present_distribution = GenAdDistribution(feat_dim, latent_dim)
        self.future_distribution = GenAdDistribution(feat_dim, latent_dim)
        self.predict_model = GenAdPredictModel(latent_dim, feat_dim, 2, future_len)

        self.cls_head = nn.Linear(feat_dim, 1)

    def forward(
        self, scene_feat: Tensor, future_scene_feat: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Decode agent queries and generate VAE-latent-conditioned futures.

        Parameters
        ----------
        scene_feat:
            Flattened present BEV scene tokens with shape ``(B, N, feat_dim)``.
        future_scene_feat:
            Flattened future BEV scene tokens (training-time future context)
            with shape ``(B, N, feat_dim)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
            Agent classification logits, per-agent future trajectories, and
            the present/future latent means/log-sigmas
            (``present_mu``, ``future_mu``).
        """

        b = scene_feat.shape[0]
        scene = self.scene_proj(scene_feat)
        query = self.agent_queries.unsqueeze(0).expand(b, -1, -1)
        content = self.query_decoder(query, scene)

        cls_logits = self.cls_head(content).squeeze(-1)

        present_mu, present_log_sigma = self.present_distribution(scene)
        future_mu, _ = self.future_distribution(future_scene_feat)
        sample = present_mu + torch.exp(present_log_sigma) * torch.randn_like(present_mu)

        trajs = []
        for agent_idx in range(self.num_agents):
            traj = self.predict_model(sample, content[:, agent_idx])
            trajs.append(traj)
        trajectories = torch.stack(trajs, dim=1)

        return cls_logits, trajectories, present_mu, future_mu, present_log_sigma


def build_genad() -> nn.Module:
    """Build a compact random-init GenAD model."""

    return GenAd().eval()


def example_input_genad() -> tuple[Tensor, Tensor]:
    """Return present and future flattened BEV scene tokens for GenAD."""

    return (torch.randn(1, 16, 32), torch.randn(1, 16, 32))


# ---------------------------------------------------------------------------
# GRIP -- spatio-temporal graph conv + GRU seq2seq trajectory prediction
# ---------------------------------------------------------------------------


class ConvTemporalGraphical(nn.Module):
    """Graph convolution over a fixed multi-hop adjacency tensor."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        """Build the 1x1 temporal conv that expands channels for graph mixing.

        Parameters
        ----------
        in_channels:
            Input channel width.
        out_channels:
            Output channel width per adjacency hop.
        kernel_size:
            Number of adjacency hops (spatial kernel size).
        """

        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels, out_channels * kernel_size, kernel_size=(1, 1))

    def forward(self, x: Tensor, adjacency: Tensor) -> Tensor:
        """Apply multi-hop graph convolution.

        Parameters
        ----------
        x:
            Node feature map with shape ``(N, C, T, V)``.
        adjacency:
            Adjacency tensor with shape ``(K, V, V)``.

        Returns
        -------
        Tensor
            Graph-convolved features with shape ``(N, out_channels, T, V)``.
        """

        x = self.conv(x)
        n, kc, t, v = x.shape
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        return torch.einsum("nkctv,kvw->nctw", x, adjacency)


class GraphConvBlock(nn.Module):
    """Graph conv + temporal conv residual block (ST-GCN unit)."""

    def __init__(
        self, in_channels: int, out_channels: int, spatial_kernel: int, temporal_kernel: int = 5
    ) -> None:
        """Build the graph conv, temporal conv, and residual projection.

        Parameters
        ----------
        in_channels:
            Input channel width.
        out_channels:
            Output channel width.
        spatial_kernel:
            Number of adjacency hops.
        temporal_kernel:
            Kernel size of the temporal convolution (must be odd).
        """

        super().__init__()
        padding = (temporal_kernel - 1) // 2
        self.gcn = ConvTemporalGraphical(in_channels, out_channels, spatial_kernel)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, (temporal_kernel, 1), (1, 1), (padding, 0)),
            nn.BatchNorm2d(out_channels),
        )
        if in_channels == out_channels:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1), nn.BatchNorm2d(out_channels)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor, adjacency: Tensor) -> Tensor:
        """Apply one ST-GCN block with a residual connection.

        Parameters
        ----------
        x:
            Node feature map with shape ``(N, C, T, V)``.
        adjacency:
            Adjacency tensor with shape ``(K, V, V)``.

        Returns
        -------
        Tensor
            Updated node feature map with shape ``(N, out_channels, T, V)``.
        """

        res = self.residual(x)
        out = self.gcn(x, adjacency)
        out = self.tcn(out) + res
        return self.relu(out)


class Seq2SeqTraj(nn.Module):
    """GRU encoder-decoder that predicts residual trajectory offsets."""

    def __init__(self, input_size: int, hidden_size: int, pred_length: int) -> None:
        """Build the GRU encoder and decoder with a linear output head.

        Parameters
        ----------
        input_size:
            Per-timestep encoder input feature width.
        hidden_size:
            GRU hidden width.
        pred_length:
            Number of future steps to decode.
        """

        super().__init__()
        self.pred_length = pred_length
        self.encoder = nn.GRU(input_size, hidden_size, batch_first=True)
        self.decoder = nn.GRU(2, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, 2)

    def forward(self, node_features: Tensor, last_location: Tensor) -> Tensor:
        """Encode a node feature sequence, then decode residual future offsets.

        Parameters
        ----------
        node_features:
            Encoder input sequence with shape ``(N, T, input_size)``.
        last_location:
            Last observed 2D location with shape ``(N, 1, 2)``.

        Returns
        -------
        Tensor
            Predicted future locations with shape ``(N, pred_length, 2)``.
        """

        _, hidden = self.encoder(node_features)
        decoder_input = last_location
        outputs = []
        for _ in range(self.pred_length):
            step_out, hidden = self.decoder(decoder_input, hidden)
            step_out = self.out(step_out) + decoder_input
            outputs.append(step_out)
            decoder_input = step_out
        return torch.cat(outputs, dim=1)


class Grip(nn.Module):
    """Compact GRIP: ST-GCN over a fixed graph + GRU seq2seq trajectory decoder."""

    def __init__(
        self,
        in_channels: int = 3,
        num_nodes: int = 6,
        hops: int = 2,
        hidden: int = 32,
        pred_length: int = 6,
    ) -> None:
        """Build the ST-GCN stack, fixed multi-hop adjacency, and seq2seq decoder.

        Parameters
        ----------
        in_channels:
            Input per-node channel width (e.g. ``x, y, mask``).
        num_nodes:
            Number of graph nodes (agents) in the scene.
        hops:
            Number of adjacency hops (spatial kernel size minus one).
        hidden:
            Hidden channel width used throughout the ST-GCN stack.
        pred_length:
            Number of future timesteps to predict.
        """

        super().__init__()
        spatial_kernel = hops + 1
        self.register_buffer(
            "adjacency", torch.ones(spatial_kernel, num_nodes, num_nodes) / num_nodes
        )
        self.edge_importance = nn.ParameterList(
            [nn.Parameter(torch.ones(spatial_kernel, num_nodes, num_nodes)) for _ in range(3)]
        )
        self.bn = nn.BatchNorm2d(in_channels)
        self.st_gcn = nn.ModuleList(
            [
                GraphConvBlock(in_channels, hidden, spatial_kernel),
                GraphConvBlock(hidden, hidden, spatial_kernel),
                GraphConvBlock(hidden, hidden, spatial_kernel),
            ]
        )
        self.seq2seq = Seq2SeqTraj(hidden, hidden, pred_length)

    def forward(self, x: Tensor) -> Tensor:
        """Predict future trajectories for every graph node.

        Parameters
        ----------
        x:
            Node feature map with shape ``(N, in_channels, T, V)`` where the
            first two input channels are ``(x, y)`` position.

        Returns
        -------
        Tensor
            Predicted future locations with shape ``(N, V, pred_length, 2)``.
        """

        feat = self.bn(x)
        for block, importance in zip(self.st_gcn, self.edge_importance):
            feat = block(feat, self.adjacency + importance)

        n, c, t, v = feat.shape
        node_seq = feat.permute(0, 3, 2, 1).reshape(n * v, t, c)
        last_loc = x[:, :2].permute(0, 3, 2, 1).reshape(n * v, t, 2)[:, -1:, :]

        pred = self.seq2seq(node_seq, last_loc)
        return pred.view(n, v, self.seq2seq.pred_length, 2)


def build_grip() -> nn.Module:
    """Build a compact random-init GRIP model."""

    return Grip().eval()


def example_input_grip() -> Tensor:
    """Return a batch of graph node feature maps for GRIP."""

    return torch.randn(1, 3, 8, 6)


MENAGERIE_ENTRIES = [
    ("FIERY", "build_fiery", "example_input_fiery", "2021", "VIS"),
    ("Forecast-MAE", "build_forecast_mae", "example_input_forecast_mae", "2023", "SEQ"),
    ("GameFormer", "build_gameformer", "example_input_gameformer", "2023", "SEQ"),
    ("GATraj", "build_gatraj", "example_input_gatraj", "2023", "SEQ"),
    ("GenAD", "build_genad", "example_input_genad", "2024", "GEN"),
    ("GRIP", "build_grip", "example_input_grip", "2019", "GRAPH"),
]
