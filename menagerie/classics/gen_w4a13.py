"""Autonomous-driving trajectory-prediction / VLM-planning classics (batch w4a13).

Sources checked (paper + official repo source; no clone, no pip install --
reimplemented from scratch in base-env torch):

- PlanTF: Cheng et al., ICRA 2024, arXiv:2309.10443.
  https://github.com/jchengai/planTF (official)
  Pure imitation-learning transformer planner on nuPlan. An agent encoder and
  a map/polygon encoder each produce per-entity tokens; a shared MLP
  positional embedding (from concatenated x/y position + cos/sin heading) is
  added to every token; the fused agent+polygon token sequence is
  self-attended through a stack of pre-norm Transformer encoder layers with a
  key-padding mask for invalid entities. The ego token (index 0) is read out
  through a multimodal trajectory decoder: a linear "multimodal projection"
  expands the single ego feature into ``num_modes`` per-mode embeddings, each
  independently decoded into a trajectory (loc head) and a mode-probability
  logit (pi head). A second, auxiliary MLP head regresses every other agent's
  future trajectory directly from its encoder token (dense forecasting
  auxiliary loss) -- reimplemented here as ``agent_predictor``.

- PRECOG (ESP -- Estimating Social-forecast Probabilistic futures): Rhinehart
  et al., ICCV 2019, arXiv:1905.01296.
  https://github.com/nrhinehart/precog (official, TensorFlow)
  Multi-agent joint trajectory forecasting as an autoregressive *normalizing
  flow* (the "ESP" bijection). At each future timestep t the model does not
  predict positions directly; instead a per-agent RNN context (conditioned on
  scene/social features) generates a *verlet step* mean-offset ``m_t`` and an
  affine-transform matrix ``sigma_t`` (via matrix-exponential of a generated
  generator matrix, to guarantee invertibility). The predictive mean combines
  a constant-velocity extrapolation of the last two positions with the
  learned offset: ``mu_t = m_t + 2*S_{t-1} - S_{t-2}``. A latent Gaussian
  sample ``Z_t`` is then affinely warped: ``S_t = mu_t + sigma_t @ Z_t``,
  giving an exactly-invertible (flow) map between latent noise and the joint
  multi-agent future trajectory (so exact likelihoods are computable). We
  reimplement the bijection's forward (sampling) direction with a GRU-driven
  per-step generator network producing ``m_t`` and a lower-triangular
  ``sigma_t``, in pure torch.

- QCNet (Query-Centric Trajectory Prediction): Zhou et al., CVPR 2023,
  arXiv:2306.10508.
  https://github.com/ZikangZhou/QCNet (official)
  Two distinguishing mechanisms: (1) a per-dimension learnable Fourier
  embedding (``FourierEmbedding``) that projects each continuous scalar
  feature through ``num_freq_bands`` learned frequencies before an
  independent per-dimension MLP, then sums across dimensions -- used
  throughout for encoding relative positions/headings/times in a
  *query-centric*, rotation-and-translation-invariant local reference frame
  per agent/polygon (so no global map normalization is needed); (2) a gated
  relative-position-biased self-attention block (`AttentionLayer`) with an
  explicit *gate* that mixes the attention output with a scalar projection of
  the un-attended input (``to_s``/``to_g``), rather than a plain residual
  add. We reimplement both compactly: a ``QCFourierEmbedding`` module and a
  ``GatedRelAttention`` block, stacked into a scene encoder followed by an
  anchor-free multimodal decoder (mode queries cross-attending to the scene).

- Roach: Zhang et al., ICCV 2021, arXiv:2108.08265.
  https://github.com/zhejz/carla-roach (official)
  A privileged reinforcement-learning "coach" policy for CARLA: a bird's-eye
  -view (BEV) semantic raster is passed through a CNN trunk (``XtMaCNN``,
  progressively-downsampling 2D convs) while a low-dimensional ego-state
  vector (speed, steer, etc.) is passed through a small MLP; the two feature
  streams are concatenated and fused through a final linear+ReLU stack into a
  single latent. Separate policy and value MLP heads read out from that
  latent: the policy head parameterizes a Beta-distribution action
  (throttle/steer, reimplemented as an alpha/beta pair via softplus) trained
  with PPO, and the value head regresses the scalar state value. Roach's
  privileged policy is later used to distill route-following knowledge into
  camera-only "student" agents (as in LBC-style setups).

- Senna: Jiang et al., CVPR-track 2024, arXiv:2410.22313.
  https://github.com/hustvl/Senna (official)
  A vision-language-model (LLaVA-architecture: frozen/fine-tuned CLIP vision
  tower -> learned MLP projector -> causal-LM decoder) fine-tuned to emit
  high-level driving decisions ("KEEP/ACCELERATE speed, LEFT_TURN/STRAIGHT
  path") as text tokens conditioned on multi-view camera images plus a text
  navigation instruction, paired with a lightweight downstream trajectory
  *Planner* module (a small Transformer/MLP) that fuses the VLM's discrete
  decision embedding with the ego vehicle's recent-history/state vector to
  regress a short future trajectory. We reimplement the two-stage
  architecture compactly: a tiny ViT-style vision encoder + text-token
  causal-decoder stand in for the LLaVA VLM (image patches and instruction
  tokens jointly self-attended, causal-masked, producing next-token driving-
  decision logits over a small decision vocabulary), and a separate
  ``SennaPlanner`` module consumes the VLM's pooled decision embedding plus
  ego history to regress the trajectory, matching the paper's VLM-then-
  planner staged design.

- SGNet (Stepwise Goal-Driven Network): Wang et al., RA-L/ICRA 2022,
  arXiv:2103.14107.
  https://github.com/ChuhuaW/SGNet.pytorch (official)
  The distinguishing mechanism is the *Stepwise Goal Estimator* (SGE): at
  every encoder timestep, a dedicated GRU cell is unrolled for
  ``dec_steps`` sub-steps starting from that timestep's hidden state,
  generating a *sequence of futures-so-far* goal-hidden-states (a nested
  "goal" GRU loop inside the outer trajectory-encoding GRU loop). Those
  per-substep goal hidden states are then attention-pooled twice: once
  (``enc_goal_attn``) to produce a single feedback vector fed back into the
  *next* outer encoder step (closing a stepwise-goal feedback loop into the
  trajectory encoder), and once per decoder step (``dec_goal_attn``, causal
  -- only goals from the current decode step onward are visible) to bias a
  separate decoder GRU that regresses the final trajectory. We reimplement
  this nested double-GRU-with-attention-pooling structure compactly as
  ``SGNetCore``.

Random init, tiny dims, CPU-only -- architecture catalog entries, not trained
weights.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# PlanTF
# ---------------------------------------------------------------------------


class PlanTFEncoderLayer(nn.Module):
    """Pre-norm Transformer encoder layer with a key-padding mask."""

    def __init__(self, dim: int = 32, num_heads: int = 4) -> None:
        """Initialize the self-attention block and feed-forward block.

        Parameters
        ----------
        dim:
            Shared token feature width.
        num_heads:
            Number of self-attention heads.
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

    def forward(self, tokens: Tensor, key_padding_mask: Tensor) -> Tensor:
        """Apply pre-norm self-attention (with padding mask) then an MLP block.

        Parameters
        ----------
        tokens:
            Fused agent+polygon tokens, shape ``(batch, entities, dim)``.
        key_padding_mask:
            Boolean mask, ``True`` at padded/invalid entity positions,
            shape ``(batch, entities)``.

        Returns
        -------
        Tensor
            Updated tokens, same shape as ``tokens``.
        """
        normed = self.norm1(tokens)
        attended, _ = self.attn(normed, normed, normed, key_padding_mask=key_padding_mask)
        tokens = tokens + attended
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens


class PlanTFTrajectoryDecoder(nn.Module):
    """Anchor-free multimodal trajectory decoder reading out the ego token."""

    def __init__(self, dim: int = 32, num_modes: int = 6, future_steps: int = 16) -> None:
        """Initialize the multimodal projection and per-mode loc/pi heads.

        Parameters
        ----------
        dim:
            Shared token feature width.
        num_modes:
            Number of predicted trajectory modes.
        future_steps:
            Number of future waypoints regressed per mode.
        """
        super().__init__()
        self.num_modes = num_modes
        self.future_steps = future_steps
        self.multimodal_proj = nn.Linear(dim, num_modes * dim)
        self.loc = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(dim * 2, future_steps * 4),
        )
        self.pi = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(dim * 2, 1),
        )

    def forward(self, ego_token: Tensor) -> tuple[Tensor, Tensor]:
        """Decode the ego token into per-mode trajectories and mode logits.

        Parameters
        ----------
        ego_token:
            Encoded ego-vehicle feature, shape ``(batch, dim)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Trajectories ``(batch, num_modes, future_steps, 4)`` (x, y, cos,
            sin) and mode-probability logits ``(batch, num_modes)``.
        """
        batch = ego_token.shape[0]
        modes = self.multimodal_proj(ego_token).view(batch, self.num_modes, -1)
        loc = self.loc(modes).view(batch, self.num_modes, self.future_steps, 4)
        pi = self.pi(modes).squeeze(-1)
        return loc, pi


class PlanTF(nn.Module):
    """PlanTF: agent+map query tokens self-attended, decoded into multimodal ego trajectories."""

    def __init__(
        self, dim: int = 32, depth: int = 2, num_modes: int = 6, future_steps: int = 16
    ) -> None:
        """Initialize the agent/polygon encoders, positional MLP, encoder stack, and decoders.

        Parameters
        ----------
        dim:
            Shared token feature width.
        depth:
            Number of Transformer encoder layers.
        num_modes:
            Number of predicted ego-trajectory modes.
        future_steps:
            Number of future waypoints regressed per mode.
        """
        super().__init__()
        self.dim = dim
        self.agent_encoder = nn.Sequential(
            nn.Linear(6, dim), nn.ReLU(inplace=True), nn.Linear(dim, dim)
        )
        self.map_encoder = nn.Sequential(
            nn.Linear(6, dim), nn.ReLU(inplace=True), nn.Linear(dim, dim)
        )
        self.pos_emb = nn.Sequential(nn.Linear(4, dim), nn.ReLU(inplace=True), nn.Linear(dim, dim))
        self.encoder_blocks = nn.ModuleList([PlanTFEncoderLayer(dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.trajectory_decoder = PlanTFTrajectoryDecoder(dim, num_modes, future_steps)
        self.agent_predictor = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.LayerNorm(dim * 2), nn.Linear(dim * 2, future_steps * 2)
        )
        self.future_steps = future_steps

    def forward(
        self,
        agent_features: Tensor,
        agent_pos: Tensor,
        agent_mask: Tensor,
        polygon_features: Tensor,
        polygon_pos: Tensor,
        polygon_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Fuse agent + polygon tokens, self-attend, and decode ego + agent predictions.

        Parameters
        ----------
        agent_features:
            Per-agent history-summary features, shape ``(batch, num_agents, 6)``.
        agent_pos:
            Per-agent ``(x, y, cos_heading, sin_heading)``, shape
            ``(batch, num_agents, 4)``.
        agent_mask:
            Boolean validity mask, ``True`` where an agent is present,
            shape ``(batch, num_agents)``.
        polygon_features:
            Per-map-polygon summary features, shape ``(batch, num_polygons, 6)``.
        polygon_pos:
            Per-polygon ``(x, y, cos_heading, sin_heading)``, shape
            ``(batch, num_polygons, 4)``.
        polygon_mask:
            Boolean validity mask for polygons, shape ``(batch, num_polygons)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Ego trajectories ``(batch, num_modes, future_steps, 4)``, mode
            logits ``(batch, num_modes)``, and dense per-agent prediction
            ``(batch, num_agents - 1, future_steps, 2)``.
        """
        batch, num_agents = agent_features.shape[:2]
        pos = torch.cat([agent_pos, polygon_pos], dim=1)
        pos_embed = self.pos_emb(pos)

        x_agent = self.agent_encoder(agent_features)
        x_polygon = self.map_encoder(polygon_features)
        tokens = torch.cat([x_agent, x_polygon], dim=1) + pos_embed

        key_padding_mask = ~torch.cat([agent_mask, polygon_mask], dim=1)
        for block in self.encoder_blocks:
            tokens = block(tokens, key_padding_mask)
        tokens = self.norm(tokens)

        trajectory, probability = self.trajectory_decoder(tokens[:, 0])
        prediction = self.agent_predictor(tokens[:, 1:num_agents]).view(
            batch, -1, self.future_steps, 2
        )
        return trajectory, probability, prediction


def build_plantf() -> nn.Module:
    """Build a compact PlanTF model.

    Returns
    -------
    nn.Module
        Random-initialized ``PlanTF`` in eval mode.
    """
    return PlanTF().eval()


def example_input_plantf() -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Create example agent and map-polygon tokens with validity masks.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]
        Agent features ``(1, 5, 6)``, agent positions ``(1, 5, 4)``, agent
        mask ``(1, 5)``, polygon features ``(1, 8, 6)``, polygon positions
        ``(1, 8, 4)``, and polygon mask ``(1, 8)``.
    """
    agent_features = torch.randn(1, 5, 6)
    agent_pos = torch.randn(1, 5, 4)
    agent_mask = torch.ones(1, 5, dtype=torch.bool)
    polygon_features = torch.randn(1, 8, 6)
    polygon_pos = torch.randn(1, 8, 4)
    polygon_mask = torch.ones(1, 8, dtype=torch.bool)
    return agent_features, agent_pos, agent_mask, polygon_features, polygon_pos, polygon_mask


# ---------------------------------------------------------------------------
# PRECOG (ESP)
# ---------------------------------------------------------------------------


class ESPStepGenerator(nn.Module):
    """Per-step generator producing a verlet-step offset and an affine flow matrix."""

    def __init__(self, hidden: int = 32) -> None:
        """Initialize the social/context GRU cell and the offset/generator heads.

        Parameters
        ----------
        hidden:
            Shared feature width.
        """
        super().__init__()
        self.cell = nn.GRUCell(4 + hidden, hidden)
        self.offset_head = nn.Linear(hidden, 2)
        self.generator_head = nn.Linear(hidden, 4)

    def forward(
        self, rnn_state: Tensor, s_last: Tensor, s_prev: Tensor, context: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Advance the per-agent GRU state and emit an offset + lower-triangular sigma.

        Parameters
        ----------
        rnn_state:
            Previous per-agent GRU hidden state, shape ``(batch*agents, hidden)``.
        s_last:
            Last known position ``S_{t-1}``, shape ``(batch*agents, 2)``.
        s_prev:
            Second-to-last position ``S_{t-2}``, shape ``(batch*agents, 2)``.
        context:
            Static per-agent social/scene context, shape ``(batch*agents, hidden)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Updated GRU state, verlet-corrected mean ``mu_t``
            ``(batch*agents, 2)``, and flow matrix ``sigma_t``
            ``(batch*agents, 2, 2)``.
        """
        step_input = torch.cat([s_last, s_prev, context], dim=-1)
        rnn_state = self.cell(step_input, rnn_state)
        offset = self.offset_head(rnn_state)
        mu_t = offset + 2.0 * s_last - s_prev
        generator = self.generator_head(rnn_state).view(-1, 2, 2)
        sigma_t = torch.linalg.matrix_exp(generator - generator.transpose(-1, -2))
        return rnn_state, mu_t, sigma_t


class PRECOG(nn.Module):
    """ESP: autoregressive normalizing-flow bijection over joint multi-agent futures."""

    def __init__(self, hidden: int = 32, horizon: int = 8) -> None:
        """Initialize the social context encoder and the ESP step generator.

        Parameters
        ----------
        hidden:
            Shared feature width.
        horizon:
            Number of future timesteps generated.
        """
        super().__init__()
        self.hidden = hidden
        self.horizon = horizon
        self.context_encoder = nn.Sequential(
            nn.Linear(8, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, hidden)
        )
        self.step_generator = ESPStepGenerator(hidden)

    def forward(
        self, past_positions: Tensor, overhead_features: Tensor, latent_noise: Tensor
    ) -> Tensor:
        """Autoregressively warp latent noise into a joint multi-agent future via the ESP bijection.

        Parameters
        ----------
        past_positions:
            Last two observed positions per agent, shape
            ``(batch, num_agents, 2, 2)`` (index -1 = most recent).
        overhead_features:
            Static per-agent scene/social context features, shape
            ``(batch, num_agents, 8)``.
        latent_noise:
            Latent Gaussian samples to warp, shape
            ``(batch, num_agents, horizon, 2)``.

        Returns
        -------
        Tensor
            Forecast joint trajectory ``S``, shape
            ``(batch, num_agents, horizon, 2)``.
        """
        batch, num_agents = past_positions.shape[:2]
        flat_context = self.context_encoder(overhead_features.reshape(batch * num_agents, -1))
        rnn_state = flat_context.new_zeros(batch * num_agents, self.hidden)
        s_prev = past_positions[:, :, 0].reshape(batch * num_agents, 2)
        s_last = past_positions[:, :, 1].reshape(batch * num_agents, 2)

        rollout = []
        for t in range(self.horizon):
            z_t = latent_noise[:, :, t].reshape(batch * num_agents, 2)
            rnn_state, mu_t, sigma_t = self.step_generator(rnn_state, s_last, s_prev, flat_context)
            s_t = mu_t + torch.einsum("nij,nj->ni", sigma_t, z_t)
            rollout.append(s_t.view(batch, num_agents, 2))
            s_prev, s_last = s_last, s_t

        return torch.stack(rollout, dim=2)


def build_precog() -> nn.Module:
    """Build a compact PRECOG (ESP) model.

    Returns
    -------
    nn.Module
        Random-initialized ``PRECOG`` in eval mode.
    """
    return PRECOG().eval()


def example_input_precog() -> tuple[Tensor, Tensor, Tensor]:
    """Create example past positions, overhead context, and latent noise.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        Past positions ``(1, 4, 2, 2)``, overhead features ``(1, 4, 8)``, and
        latent noise ``(1, 4, 8, 2)``.
    """
    past_positions = torch.randn(1, 4, 2, 2)
    overhead_features = torch.randn(1, 4, 8)
    latent_noise = torch.randn(1, 4, 8, 2)
    return past_positions, overhead_features, latent_noise


# ---------------------------------------------------------------------------
# QCNet
# ---------------------------------------------------------------------------


class QCFourierEmbedding(nn.Module):
    """Per-dimension learnable Fourier embedding (query-centric relative encoding)."""

    def __init__(self, input_dim: int = 3, hidden_dim: int = 32, num_freq_bands: int = 8) -> None:
        """Initialize per-dimension learnable frequencies and per-dimension MLPs.

        Parameters
        ----------
        input_dim:
            Number of continuous scalar features to embed independently.
        hidden_dim:
            Output embedding width.
        num_freq_bands:
            Number of learnable Fourier frequency bands per dimension.
        """
        super().__init__()
        self.input_dim = input_dim
        self.freqs = nn.Parameter(torch.randn(input_dim, num_freq_bands))
        self.mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(num_freq_bands * 2 + 1, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(input_dim)
            ]
        )
        self.to_out = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, continuous_inputs: Tensor) -> Tensor:
        """Embed each continuous scalar dimension through its own Fourier-MLP, then sum.

        Parameters
        ----------
        continuous_inputs:
            Relative position/heading/time features, shape ``(..., input_dim)``.

        Returns
        -------
        Tensor
            Fused embedding, shape ``(..., hidden_dim)``.
        """
        angles = continuous_inputs.unsqueeze(-1) * self.freqs * 2 * torch.pi
        features = torch.cat([angles.cos(), angles.sin(), continuous_inputs.unsqueeze(-1)], dim=-1)
        per_dim = [self.mlps[i](features[..., i, :]) for i in range(self.input_dim)]
        return self.to_out(torch.stack(per_dim, dim=0).sum(dim=0))


class GatedRelAttention(nn.Module):
    """Self-attention with a relative positional bias and an explicit output gate."""

    def __init__(self, hidden: int = 32, num_heads: int = 4) -> None:
        """Initialize the query/key/value/relative-bias projections and the output gate.

        Parameters
        ----------
        hidden:
            Shared token feature width.
        num_heads:
            Number of attention heads.
        """
        super().__init__()
        self.hidden = hidden
        self.attn = nn.MultiheadAttention(hidden, num_heads, batch_first=True)
        self.rel_proj = nn.Linear(hidden, hidden)
        self.gate_score = nn.Linear(hidden * 2, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, tokens: Tensor, rel_embed: Tensor) -> Tensor:
        """Apply relative-position-biased attention, gated (not simply residual-added) into tokens.

        Parameters
        ----------
        tokens:
            Scene tokens, shape ``(batch, num_tokens, hidden)``.
        rel_embed:
            Pooled relative-position Fourier embedding to bias keys/values,
            shape ``(batch, num_tokens, hidden)``.

        Returns
        -------
        Tensor
            Gated, attention-updated tokens, same shape as ``tokens``.
        """
        biased_kv = tokens + self.rel_proj(rel_embed)
        attended, _ = self.attn(tokens, biased_kv, biased_kv)
        gate = torch.sigmoid(self.gate_score(torch.cat([attended, tokens], dim=-1)))
        return self.norm(tokens + gate * attended)


class QCNetModeDecoder(nn.Module):
    """Anchor-free multimodal decoder: learned mode queries cross-attend to the scene."""

    def __init__(self, hidden: int = 32, num_modes: int = 6, horizon: int = 12) -> None:
        """Initialize the learned mode queries, cross-attention, and trajectory/prob heads.

        Parameters
        ----------
        hidden:
            Shared feature width.
        num_modes:
            Number of predicted trajectory modes.
        horizon:
            Number of future timesteps regressed per mode.
        """
        super().__init__()
        self.mode_queries = nn.Parameter(torch.randn(1, num_modes, hidden) * 0.02)
        self.cross_attn = nn.MultiheadAttention(hidden, 4, batch_first=True)
        self.traj_head = nn.Linear(hidden, horizon * 2)
        self.prob_head = nn.Linear(hidden, 1)
        self.horizon = horizon
        self.num_modes = num_modes

    def forward(self, scene_tokens: Tensor) -> tuple[Tensor, Tensor]:
        """Decode multimodal trajectories by cross-attending mode queries to the scene.

        Parameters
        ----------
        scene_tokens:
            Encoded agent+map scene tokens, shape ``(batch, num_tokens, hidden)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Trajectories ``(batch, num_modes, horizon, 2)`` and mode logits
            ``(batch, num_modes)``.
        """
        batch = scene_tokens.shape[0]
        queries = self.mode_queries.expand(batch, -1, -1)
        attended, _ = self.cross_attn(queries, scene_tokens, scene_tokens)
        trajectories = self.traj_head(attended).view(batch, self.num_modes, self.horizon, 2)
        probs = self.prob_head(attended).squeeze(-1)
        return trajectories, probs


class QCNet(nn.Module):
    """Query-centric trajectory prediction: Fourier relative embedding + gated attention encoder."""

    def __init__(
        self, hidden: int = 32, depth: int = 2, num_modes: int = 6, horizon: int = 12
    ) -> None:
        """Initialize the Fourier embedding, gated attention encoder, and mode decoder.

        Parameters
        ----------
        hidden:
            Shared feature width.
        depth:
            Number of gated relative-attention encoder layers.
        num_modes:
            Number of predicted trajectory modes.
        horizon:
            Number of future timesteps regressed per mode.
        """
        super().__init__()
        self.token_proj = nn.Linear(6, hidden)
        self.fourier = QCFourierEmbedding(input_dim=3, hidden_dim=hidden)
        self.layers = nn.ModuleList([GatedRelAttention(hidden) for _ in range(depth)])
        self.decoder = QCNetModeDecoder(hidden, num_modes, horizon)

    def forward(self, entity_features: Tensor, rel_pose: Tensor) -> tuple[Tensor, Tensor]:
        """Encode query-centric relative-pose scene tokens and decode multimodal trajectories.

        Parameters
        ----------
        entity_features:
            Per-entity (agent+map) raw features, shape
            ``(batch, num_entities, 6)``.
        rel_pose:
            Per-entity relative ``(dx, dy, dtheta)`` w.r.t. the query-centric
            local frame, shape ``(batch, num_entities, 3)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Trajectories ``(batch, num_modes, horizon, 2)`` and mode logits
            ``(batch, num_modes)``.
        """
        tokens = self.token_proj(entity_features)
        rel_embed = self.fourier(rel_pose)
        tokens = tokens + rel_embed
        for layer in self.layers:
            tokens = layer(tokens, rel_embed)
        return self.decoder(tokens)


def build_qcnet() -> nn.Module:
    """Build a compact QCNet model.

    Returns
    -------
    nn.Module
        Random-initialized ``QCNet`` in eval mode.
    """
    return QCNet().eval()


def example_input_qcnet() -> tuple[Tensor, Tensor]:
    """Create example entity features and query-centric relative poses.

    Returns
    -------
    tuple[Tensor, Tensor]
        Entity features ``(1, 10, 6)`` and relative poses ``(1, 10, 3)``.
    """
    entity_features = torch.randn(1, 10, 6)
    rel_pose = torch.randn(1, 10, 3)
    return entity_features, rel_pose


# ---------------------------------------------------------------------------
# Roach
# ---------------------------------------------------------------------------


class XtMaCNN(nn.Module):
    """BEV-raster CNN trunk + ego-state MLP fused into a single latent feature."""

    def __init__(self, bev_channels: int = 7, state_dim: int = 6, features_dim: int = 64) -> None:
        """Initialize the BEV conv trunk, state MLP, and fusion layer.

        Parameters
        ----------
        bev_channels:
            Number of semantic channels in the BEV raster.
        state_dim:
            Ego-state feature dimension (speed, steer, ...).
        features_dim:
            Fused output latent width.
        """
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(bev_channels, 8, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(8, 16, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.state_linear = nn.Sequential(nn.Linear(state_dim, 32), nn.ReLU())
        self.fuse = nn.Linear(512 + 32, features_dim)
        self.act = nn.ReLU()

    def forward(self, birdview: Tensor, state: Tensor) -> Tensor:
        """Fuse BEV-raster and ego-state features into a single latent.

        Parameters
        ----------
        birdview:
            BEV semantic raster, shape ``(batch, bev_channels, height, width)``.
        state:
            Ego-state vector, shape ``(batch, state_dim)``.

        Returns
        -------
        Tensor
            Fused latent feature, shape ``(batch, features_dim)``.
        """
        image_feat = self.cnn(birdview)
        state_feat = self.state_linear(state)
        return self.act(self.fuse(torch.cat([image_feat, state_feat], dim=-1)))


class Roach(nn.Module):
    """Roach: privileged PPO teacher over fused BEV+state features (Beta policy + value head)."""

    def __init__(self, features_dim: int = 64) -> None:
        """Initialize the fused features extractor and the policy/value heads.

        Parameters
        ----------
        features_dim:
            Fused BEV+state latent width.
        """
        super().__init__()
        self.features_extractor = XtMaCNN(features_dim=features_dim)
        self.policy_head = nn.Sequential(nn.Linear(features_dim, 64), nn.ReLU())
        self.action_alpha = nn.Linear(64, 2)
        self.action_beta = nn.Linear(64, 2)
        self.value_head = nn.Sequential(nn.Linear(features_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, birdview: Tensor, state: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Predict a Beta-distribution driving action and a state value from BEV + state.

        Parameters
        ----------
        birdview:
            BEV semantic raster, shape ``(batch, 7, height, width)``.
        state:
            Ego-state vector, shape ``(batch, 6)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Beta-distribution ``alpha`` and ``beta`` parameters (each
            ``(batch, 2)``, for throttle/steer) and the scalar state value
            ``(batch, 1)``.
        """
        features = self.features_extractor(birdview, state)
        latent_pi = self.policy_head(features)
        alpha = F.softplus(self.action_alpha(latent_pi)) + 1.0
        beta = F.softplus(self.action_beta(latent_pi)) + 1.0
        value = self.value_head(features)
        return alpha, beta, value


def build_roach() -> nn.Module:
    """Build a compact Roach model.

    Returns
    -------
    nn.Module
        Random-initialized ``Roach`` in eval mode.
    """
    return Roach().eval()


def example_input_roach() -> tuple[Tensor, Tensor]:
    """Create an example BEV raster and ego-state vector.

    Returns
    -------
    tuple[Tensor, Tensor]
        BEV raster ``(1, 7, 32, 32)`` and ego-state vector ``(1, 6)``.
    """
    birdview = torch.randn(1, 7, 32, 32)
    state = torch.randn(1, 6)
    return birdview, state


# ---------------------------------------------------------------------------
# Senna
# ---------------------------------------------------------------------------


class SennaVisionLanguagePlanner(nn.Module):
    """Tiny LLaVA-style VLM: patch+text tokens self-attended (causal) into decision logits."""

    def __init__(
        self, vocab_size: int = 64, hidden: int = 32, patch_size: int = 8, num_layers: int = 2
    ) -> None:
        """Initialize the vision patch embedding, text embedding, and causal decoder stack.

        Parameters
        ----------
        vocab_size:
            Size of the text/decision token vocabulary.
        hidden:
            Shared token feature width.
        patch_size:
            Side length of each square image patch.
        num_layers:
            Number of causal Transformer decoder layers.
        """
        super().__init__()
        self.patch_size = patch_size
        self.patch_embed = nn.Conv2d(3, hidden, kernel_size=patch_size, stride=patch_size)
        self.projector = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        self.token_embed = nn.Embedding(vocab_size, hidden)
        layer = nn.TransformerEncoderLayer(
            hidden, nhead=4, dim_feedforward=hidden * 2, batch_first=True
        )
        self.decoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.decision_head = nn.Linear(hidden, vocab_size)

    def forward(self, image: Tensor, instruction_tokens: Tensor) -> tuple[Tensor, Tensor]:
        """Jointly self-attend image patches + instruction tokens (causal) to predict decision logits.

        Parameters
        ----------
        image:
            Multi-view (flattened to single) camera image, shape
            ``(batch, 3, height, width)``.
        instruction_tokens:
            Text navigation-instruction token ids, shape
            ``(batch, num_text_tokens)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Next-token decision logits ``(batch, num_text_tokens, vocab_size)``
            and the pooled decision embedding ``(batch, hidden)`` (mean over
            text-token positions) used to condition the downstream planner.
        """
        patches = self.patch_embed(image).flatten(2).transpose(1, 2)
        vision_tokens = self.projector(patches)
        text_tokens = self.token_embed(instruction_tokens)
        tokens = torch.cat([vision_tokens, text_tokens], dim=1)

        num_vision = vision_tokens.shape[1]
        total = tokens.shape[1]
        causal_mask = torch.triu(
            torch.full((total, total), float("-inf"), device=tokens.device), diagonal=1
        )
        causal_mask[:, :num_vision] = 0.0

        encoded = self.decoder(tokens, mask=causal_mask)
        text_encoded = encoded[:, num_vision:]
        logits = self.decision_head(text_encoded)
        pooled = text_encoded.mean(dim=1)
        return logits, pooled


class SennaPlanner(nn.Module):
    """Lightweight downstream planner fusing the VLM decision embedding + ego history."""

    def __init__(self, hidden: int = 32, horizon: int = 6) -> None:
        """Initialize the ego-history encoder and the fused trajectory-regression head.

        Parameters
        ----------
        hidden:
            Shared feature width.
        horizon:
            Number of future waypoints regressed.
        """
        super().__init__()
        self.history_encoder = nn.GRU(2, hidden, batch_first=True)
        self.fuse = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(inplace=True))
        self.traj_head = nn.Linear(hidden, horizon * 2)
        self.horizon = horizon

    def forward(self, decision_embedding: Tensor, ego_history: Tensor) -> Tensor:
        """Regress a future trajectory from the VLM decision embedding and ego history.

        Parameters
        ----------
        decision_embedding:
            Pooled decision embedding from the VLM, shape ``(batch, hidden)``.
        ego_history:
            Recent ego-position history, shape ``(batch, time, 2)``.

        Returns
        -------
        Tensor
            Predicted future trajectory, shape ``(batch, horizon, 2)``.
        """
        _, h_n = self.history_encoder(ego_history)
        history_feat = h_n.squeeze(0)
        fused = self.fuse(torch.cat([decision_embedding, history_feat], dim=-1))
        return self.traj_head(fused).view(-1, self.horizon, 2)


class Senna(nn.Module):
    """Senna: LLaVA-style VLM emits driving decisions, a downstream planner regresses trajectory."""

    def __init__(self) -> None:
        """Initialize the VLM decision head and the downstream trajectory planner."""
        super().__init__()
        self.vlm = SennaVisionLanguagePlanner()
        self.planner = SennaPlanner()

    def forward(
        self, image: Tensor, instruction_tokens: Tensor, ego_history: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Run the VLM to emit decision logits, then plan a trajectory conditioned on the decision.

        Parameters
        ----------
        image:
            Camera image, shape ``(batch, 3, height, width)``.
        instruction_tokens:
            Text navigation-instruction token ids, shape
            ``(batch, num_text_tokens)``.
        ego_history:
            Recent ego-position history, shape ``(batch, time, 2)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Decision logits ``(batch, num_text_tokens, vocab_size)`` and the
            planned future trajectory ``(batch, horizon, 2)``.
        """
        logits, pooled = self.vlm(image, instruction_tokens)
        trajectory = self.planner(pooled, ego_history)
        return logits, trajectory


def build_senna() -> nn.Module:
    """Build a compact Senna model.

    Returns
    -------
    nn.Module
        Random-initialized ``Senna`` in eval mode.
    """
    return Senna().eval()


def example_input_senna() -> tuple[Tensor, Tensor, Tensor]:
    """Create an example camera image, instruction tokens, and ego history.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        Image ``(1, 3, 32, 32)``, instruction tokens ``(1, 6)``, and ego
        history ``(1, 4, 2)``.
    """
    image = torch.randn(1, 3, 32, 32)
    instruction_tokens = torch.randint(0, 64, (1, 6))
    ego_history = torch.randn(1, 4, 2)
    return image, instruction_tokens, ego_history


# ---------------------------------------------------------------------------
# SGNet (Stepwise Goal-Driven Network)
# ---------------------------------------------------------------------------


class SGNetCore(nn.Module):
    """Nested double-GRU: an inner stepwise-goal loop feeds back into the outer trajectory encoder."""

    def __init__(
        self, hidden: int = 32, enc_steps: int = 6, dec_steps: int = 6, pred_dim: int = 2
    ) -> None:
        """Initialize the trajectory-encoder, goal, and decoder GRU cells plus attention pools.

        Parameters
        ----------
        hidden:
            Trajectory-encoder / decoder hidden width.
        enc_steps:
            Number of outer (observed) encoding timesteps.
        dec_steps:
            Number of inner goal sub-steps and decoded future timesteps.
        pred_dim:
            Per-timestep output dimensionality (e.g. x, y).
        """
        super().__init__()
        self.hidden = hidden
        self.goal_dim = hidden // 4
        self.enc_steps = enc_steps
        self.dec_steps = dec_steps
        self.pred_dim = pred_dim

        self.traj_enc_cell = nn.GRUCell(hidden + self.goal_dim, hidden)
        self.goal_cell = nn.GRUCell(self.goal_dim, self.goal_dim)
        self.dec_cell = nn.GRUCell(hidden + self.goal_dim, hidden)

        self.enc_to_goal_hidden = nn.Sequential(
            nn.Linear(hidden, self.goal_dim), nn.ReLU(inplace=True)
        )
        self.enc_to_dec_hidden = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True))
        self.goal_hidden_to_input = nn.Sequential(
            nn.Linear(self.goal_dim, self.goal_dim), nn.ReLU(inplace=True)
        )
        self.goal_hidden_to_traj = nn.Sequential(
            nn.Linear(self.goal_dim, hidden), nn.ReLU(inplace=True)
        )
        self.goal_to_enc = nn.Sequential(
            nn.Linear(self.goal_dim, self.goal_dim), nn.ReLU(inplace=True)
        )
        self.goal_to_dec = nn.Sequential(
            nn.Linear(self.goal_dim, self.goal_dim), nn.ReLU(inplace=True)
        )
        self.dec_hidden_to_input = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True))

        self.enc_goal_attn = nn.Sequential(nn.Linear(self.goal_dim, 1), nn.ReLU(inplace=True))
        self.dec_goal_attn = nn.Sequential(nn.Linear(self.goal_dim, 1), nn.ReLU(inplace=True))
        self.regressor = nn.Linear(hidden, pred_dim)

    def _stepwise_goal_estimator(self, goal_hidden: Tensor) -> tuple[list[Tensor], Tensor]:
        """Unroll the inner goal GRU for ``dec_steps`` sub-steps and attention-pool the result.

        Parameters
        ----------
        goal_hidden:
            Initial per-agent goal hidden state, shape ``(batch, goal_dim)``.

        Returns
        -------
        tuple[list[Tensor], Tensor]
            List of per-substep decoder-conditioning goal features (length
            ``dec_steps``, each ``(batch, goal_dim)``) and the attention-
            pooled feedback vector for the outer encoder, shape
            ``(batch, goal_dim)``.
        """
        goal_input = goal_hidden.new_zeros(goal_hidden.shape[0], self.goal_dim)
        goal_list = []
        for _ in range(self.dec_steps):
            goal_hidden = self.goal_cell(goal_input, goal_hidden)
            goal_input = self.goal_hidden_to_input(goal_hidden)
            goal_list.append(goal_hidden)

        goal_for_dec = [self.goal_to_dec(goal) for goal in goal_list]
        goal_for_enc = torch.stack([self.goal_to_enc(goal) for goal in goal_list], dim=1)
        attn_logits = self.enc_goal_attn(torch.tanh(goal_for_enc)).squeeze(-1)
        attn_weights = F.softmax(attn_logits, dim=1).unsqueeze(1)
        pooled_goal = torch.bmm(attn_weights, goal_for_enc).squeeze(1)
        return goal_for_dec, pooled_goal

    def _decode(self, dec_hidden: Tensor, goal_for_dec: list[Tensor]) -> Tensor:
        """Regress a future trajectory, causally attending remaining stepwise-goal features.

        Parameters
        ----------
        dec_hidden:
            Initial decoder GRU hidden state, shape ``(batch, hidden)``.
        goal_for_dec:
            Per-substep goal features from the stepwise goal estimator.

        Returns
        -------
        Tensor
            Decoded trajectory, shape ``(batch, dec_steps, pred_dim)``.
        """
        batch = dec_hidden.shape[0]
        outputs = []
        for dec_step in range(self.dec_steps):
            remaining = torch.stack(goal_for_dec[dec_step:], dim=1)
            attn_logits = self.dec_goal_attn(torch.tanh(remaining)).squeeze(-1)
            attn_weights = F.softmax(attn_logits, dim=1).unsqueeze(1)
            goal_dec_input = torch.bmm(attn_weights, remaining).squeeze(1)

            dec_input_feat = self.dec_hidden_to_input(dec_hidden)
            dec_input = torch.cat([goal_dec_input, dec_input_feat], dim=-1)
            dec_hidden = self.dec_cell(dec_input, dec_hidden)
            outputs.append(self.regressor(dec_hidden))
        return torch.stack(outputs, dim=1).view(batch, self.dec_steps, self.pred_dim)

    def forward(self, traj_input: Tensor) -> Tensor:
        """Encode observed trajectory steps, closing the stepwise-goal feedback loop each step.

        Parameters
        ----------
        traj_input:
            Observed per-timestep trajectory features, shape
            ``(batch, enc_steps, hidden)``.

        Returns
        -------
        Tensor
            Decoded future trajectory from the final encoder step, shape
            ``(batch, dec_steps, pred_dim)``.
        """
        batch = traj_input.shape[0]
        goal_for_enc = traj_input.new_zeros(batch, self.goal_dim)
        traj_enc_hidden = traj_input.new_zeros(batch, self.hidden)

        dec_traj = traj_input.new_zeros(batch, self.dec_steps, self.pred_dim)
        for enc_step in range(self.enc_steps):
            enc_input = torch.cat([traj_input[:, enc_step], goal_for_enc], dim=-1)
            traj_enc_hidden = self.traj_enc_cell(enc_input, traj_enc_hidden)

            goal_hidden = self.enc_to_goal_hidden(traj_enc_hidden)
            dec_hidden = self.enc_to_dec_hidden(traj_enc_hidden)
            goal_for_dec, goal_for_enc = self._stepwise_goal_estimator(goal_hidden)
            dec_traj = self._decode(dec_hidden, goal_for_dec)
        return dec_traj


def build_sgnet() -> nn.Module:
    """Build a compact SGNet (Stepwise Goal-Driven Network) model.

    Returns
    -------
    nn.Module
        Random-initialized ``SGNetCore`` in eval mode.
    """
    return SGNetCore().eval()


def example_input_sgnet() -> Tensor:
    """Create an example observed-trajectory feature sequence.

    Returns
    -------
    Tensor
        Observed trajectory features, shape ``(1, 6, 32)``.
    """
    return torch.randn(1, 6, 32)


MENAGERIE_ENTRIES = [
    ("PlanTF", "build_plantf", "example_input_plantf", "2024", "SEQ"),
    ("PRECOG", "build_precog", "example_input_precog", "2019", "SEQ"),
    ("QCNet", "build_qcnet", "example_input_qcnet", "2023", "SEQ"),
    ("Roach", "build_roach", "example_input_roach", "2021", "RL"),
    ("Senna", "build_senna", "example_input_senna", "2024", "SEQ"),
    ("SgNet", "build_sgnet", "example_input_sgnet", "2022", "SEQ"),
]
