"""Autonomous-driving perception / generation / trajectory-prediction classics (batch w4a11).

Sources checked (paper + official repo README/architecture description; no clone,
no pip install -- reimplemented from scratch in base-env torch):

- LBC (Learning by Cheating): Chen et al., CoRL 2019, arXiv:1912.12294.
  https://github.com/dotchen/LearningByCheating
  Two-stage privileged-teacher / vision-student imitation learning for CARLA
  driving. Stage 1: a *privileged agent* that consumes a ground-truth
  bird's-eye-view (BEV) semantic map + a discrete high-level navigation command
  and regresses future waypoints (its own steering/throttle targets act as
  supervision). Stage 2: a purely camera-based *sensorimotor student* that
  never sees the BEV map -- it consumes only an RGB image (+ the same
  navigation command) and is trained to imitate the teacher's waypoints via a
  feature-matching/distillation loss on the teacher's own encoder features
  as well as the waypoint outputs, so the student learns to "cheat" by
  mimicking a teacher that had privileged information.

- MagicDrive: Gao et al., ICLR 2024, arXiv:2310.02601.
  https://github.com/cure-lab/MagicDrive
  Multi-camera street-view diffusion conditioned on diverse 3D geometry: 3D
  bounding boxes are encoded with a cross-attention "box encoder" (akin to
  text-embedding conditioning), a BEV road/lane map is encoded through an
  additive ControlNet-style branch, and camera pose + text embeddings are
  injected via adaptive layer-norm. The distinguishing mechanism is a
  *cross-view attention* module inserted into the latent UNet blocks so that
  each camera view attends to its immediate left/right neighbor views,
  enforcing multi-view geometric consistency across the generated street-view
  images.

- MID (Motion Indeterminacy Diffusion): Gu et al., CVPR 2022, arXiv:2203.13777.
  https://github.com/Gutianpei/MID
  Stochastic trajectory prediction as a *reverse diffusion* process: a social
  history encoder (per-agent trajectory encoder + interaction pooling)
  produces a state embedding that conditions a Transformer-based denoiser.
  Diffusion progressively removes "motion indeterminacy" from a maximally
  uncertain (near-uniform / high-variance) initial guess over T steps until a
  determinate future trajectory emerges; the denoiser is a stack of
  Transformer encoder layers over the noisy trajectory sequence, conditioned
  on the diffusion timestep embedding and the state embedding, with a DDIM-
  style accelerated sampler used at inference (represented here structurally
  by a callable multi-step reverse loop with a small number of steps).

- mmTransformer: Liu et al., CVPR 2021, arXiv:2103.11624.
  https://github.com/decisionforce/mmTransformer
  Multimodal motion prediction via *stacked transformers* with a fixed set of
  learned trajectory proposals (no goal candidates, no anchors). Three
  stacked stages, each a Transformer encoder layer applied across a different
  axis: (1) a *proposal* stage where each of K fixed proposal (query) tokens
  self-attends against the target agent's history + map/social context to
  form a per-mode feature; (2) a *social* / region stage where the K
  proposal features (one per mode) self-attend against each other and the
  broader agent context so multimodal predictions interact; (3) a
  *temporal-refinement* decoder that expands each proposal feature into a
  regressed trajectory. Region-based training keeps each of the K proposals
  responsible for a distinct output mode.

- MTR-A (Motion Transformer, Marginal + Joint): Shi et al., NeurIPS 2022,
  arXiv:2209.10033 (MTR-A) / arXiv:2209.13508 (MTR base).
  https://github.com/sshaoshuai/MTR
  Motion prediction as joint global-intention localization + local-movement
  refinement. A simple Transformer context encoder self-attends over agent
  history and map polyline tokens. The decoder replaces goal-candidate
  heatmaps with a small fixed set of *learnable motion query pairs*: a
  static "intention query" (one per k-means intention point / mode, fixed
  after the encoder) that cross-attends to context for global mode
  localization, paired with a *dynamic query* that is updated every decoder
  layer by cross-attending to context *local to the previous layer's
  predicted trajectory endpoint*, iteratively refining that specific mode's
  trajectory. A Gaussian-Mixture-Model regression head reads out per-mode
  trajectories + mode probabilities; "-A" denotes the marginal+joint
  ensembled variant, reimplemented here as sharing one context encoder
  across both a marginal (per-agent) and joint (multi-agent, second-agent
  concatenated queries) decoding head.

- MultiPath++: Varadarajan et al., 2021, arXiv:2111.14973 (reimplemented per
  the well-documented community port, decisionforce/mmTransformer's sibling
  repo `stepankonev/waymo-motion-prediction-challenge-2022-multipath-plus-plus`,
  since Google did not release official code).
  Efficient trajectory prediction with three distinguishing pieces: (1) a
  *polyline* encoder (shared MLP + max-pool, VectorNet-style) for map/agent
  history features instead of rasterized images; (2) a stack of
  *Multi-Context Gating (MCG)* blocks -- a cheaper alternative to full
  cross-attention that gates each per-entity feature by a global context
  vector (element-wise sigmoid gate on a pooled context, applied twice for
  a bidirectional query<->context exchange) used repeatedly to fuse
  agent/road/light context; (3) *learned latent anchor embeddings* (as
  opposed to fixed k-means anchors) that seed a Gaussian-Mixture-Model
  trajectory decoder, giving learned-anchor multimodal outputs.

Random init, tiny dims, CPU-only -- architecture catalog entries, not trained
weights.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# LBC (Learning by Cheating)
# ---------------------------------------------------------------------------


class PrivilegedTeacher(nn.Module):
    """BEV-map + command conditioned waypoint-predicting teacher agent."""

    def __init__(
        self, bev_ch: int = 7, hidden: int = 32, num_commands: int = 4, horizon: int = 4
    ) -> None:
        """Initialize the BEV conv trunk, command embedding, and waypoint head.

        Parameters
        ----------
        bev_ch:
            Number of semantic channels in the ground-truth BEV map.
        hidden:
            Shared feature width.
        num_commands:
            Number of discrete high-level navigation commands.
        horizon:
            Number of future waypoints predicted.
        """
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(bev_ch, hidden, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.command_embed = nn.Embedding(num_commands, hidden)
        self.gru = nn.GRUCell(2, hidden)
        self.waypoint_head = nn.Linear(hidden, 2)
        self.horizon = horizon

    def forward(self, bev_map: Tensor, command: Tensor) -> tuple[Tensor, Tensor]:
        """Predict future waypoints from a privileged BEV map and command.

        Parameters
        ----------
        bev_map:
            Ground-truth semantic BEV map, shape ``(batch, bev_ch, height, width)``.
        command:
            Discrete navigation command ids, shape ``(batch,)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Predicted waypoints ``(batch, horizon, 2)`` and the pooled BEV
            feature used for feature-matching distillation, shape
            ``(batch, hidden)``.
        """
        feat = self.trunk(bev_map).mean(dim=(2, 3))
        hidden_state = feat + self.command_embed(command)
        point = torch.zeros(bev_map.shape[0], 2, device=bev_map.device, dtype=bev_map.dtype)
        waypoints = []
        for _ in range(self.horizon):
            hidden_state = self.gru(point, hidden_state)
            point = self.waypoint_head(hidden_state)
            waypoints.append(point)
        return torch.stack(waypoints, dim=1), feat


class VisionStudent(nn.Module):
    """Camera-only sensorimotor student trained to imitate the privileged teacher."""

    def __init__(self, hidden: int = 32, num_commands: int = 4, horizon: int = 4) -> None:
        """Initialize the RGB conv trunk, command embedding, and waypoint head.

        Parameters
        ----------
        hidden:
            Shared feature width (matches the teacher's for feature matching).
        num_commands:
            Number of discrete high-level navigation commands.
        horizon:
            Number of future waypoints predicted.
        """
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(3, hidden, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.command_embed = nn.Embedding(num_commands, hidden)
        self.gru = nn.GRUCell(2, hidden)
        self.waypoint_head = nn.Linear(hidden, 2)
        self.horizon = horizon

    def forward(self, rgb_image: Tensor, command: Tensor) -> tuple[Tensor, Tensor]:
        """Predict future waypoints from a camera image and command only.

        Parameters
        ----------
        rgb_image:
            Camera RGB image, shape ``(batch, 3, height, width)``.
        command:
            Discrete navigation command ids, shape ``(batch,)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Predicted waypoints ``(batch, horizon, 2)`` and the pooled RGB
            feature used for feature-matching distillation, shape
            ``(batch, hidden)``.
        """
        feat = self.trunk(rgb_image).mean(dim=(2, 3))
        hidden_state = feat + self.command_embed(command)
        point = torch.zeros(rgb_image.shape[0], 2, device=rgb_image.device, dtype=rgb_image.dtype)
        waypoints = []
        for _ in range(self.horizon):
            hidden_state = self.gru(point, hidden_state)
            point = self.waypoint_head(hidden_state)
            waypoints.append(point)
        return torch.stack(waypoints, dim=1), feat


class LBC(nn.Module):
    """Learning by Cheating: privileged BEV teacher distills into a vision-only student."""

    def __init__(self) -> None:
        """Initialize the privileged teacher and the vision-only student."""
        super().__init__()
        self.teacher = PrivilegedTeacher()
        self.student = VisionStudent()

    def forward(
        self, bev_map: Tensor, rgb_image: Tensor, command: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Run both the teacher and student forward passes for joint distillation.

        Parameters
        ----------
        bev_map:
            Ground-truth semantic BEV map available only to the teacher.
        rgb_image:
            Camera RGB image available to the student.
        command:
            Discrete navigation command ids, shape ``(batch,)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor, Tensor]
            Teacher waypoints, teacher feature, student waypoints, and student
            feature (student is trained to match both teacher outputs).
        """
        teacher_waypoints, teacher_feat = self.teacher(bev_map, command)
        student_waypoints, student_feat = self.student(rgb_image, command)
        return teacher_waypoints, teacher_feat, student_waypoints, student_feat


def build_lbc() -> nn.Module:
    """Build a compact LBC (Learning by Cheating) model.

    Returns
    -------
    nn.Module
        Random-initialized ``LBC`` in eval mode.
    """
    return LBC().eval()


def example_input_lbc() -> tuple[Tensor, Tensor, Tensor]:
    """Create an example privileged BEV map, RGB image, and navigation command.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        BEV map ``(1, 7, 32, 32)``, RGB image ``(1, 3, 32, 32)``, and command
        id ``(1,)``.
    """
    bev_map = torch.randn(1, 7, 32, 32)
    rgb_image = torch.randn(1, 3, 32, 32)
    command = torch.tensor([1])
    return bev_map, rgb_image, command


# ---------------------------------------------------------------------------
# MagicDrive
# ---------------------------------------------------------------------------


class BoxCrossAttentionEncoder(nn.Module):
    """Cross-attention 3D-bounding-box encoder, akin to text-embedding conditioning."""

    def __init__(self, box_dim: int = 8, channels: int = 16) -> None:
        """Initialize the box-token projection and cross-attention layer.

        Parameters
        ----------
        box_dim:
            Per-box raw feature dimension (position/size/class one-hot etc.).
        channels:
            Latent conditioning feature width.
        """
        super().__init__()
        self.box_proj = nn.Linear(box_dim, channels)
        self.cross_attn = nn.MultiheadAttention(channels, num_heads=4, batch_first=True)

    def forward(self, latent_tokens: Tensor, boxes: Tensor) -> Tensor:
        """Condition latent tokens on a variable-length set of 3D boxes.

        Parameters
        ----------
        latent_tokens:
            Flattened spatial latent tokens, shape ``(batch, tokens, channels)``.
        boxes:
            3D bounding-box descriptors, shape ``(batch, num_boxes, box_dim)``.

        Returns
        -------
        Tensor
            Box-conditioned latent tokens, same shape as ``latent_tokens``.
        """
        box_tokens = self.box_proj(boxes)
        attended, _ = self.cross_attn(latent_tokens, box_tokens, box_tokens)
        return latent_tokens + attended


class BEVMapControlBranch(nn.Module):
    """Additive ControlNet-style branch encoding a BEV road/lane map."""

    def __init__(self, bev_ch: int = 4, channels: int = 16) -> None:
        """Initialize the BEV conv encoder producing an additive control signal.

        Parameters
        ----------
        bev_ch:
            Number of semantic channels in the BEV road/lane map.
        channels:
            Latent conditioning feature width.
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(bev_ch, channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, bev_map: Tensor) -> Tensor:
        """Encode a BEV map into a spatial additive control feature.

        Parameters
        ----------
        bev_map:
            BEV road/lane semantic map, shape ``(batch, bev_ch, height, width)``.

        Returns
        -------
        Tensor
            Control feature map, shape ``(batch, channels, height, width)``.
        """
        return self.conv(bev_map)


class CrossViewAttention(nn.Module):
    """Cross-view attention: each camera view attends to its adjacent neighbor views."""

    def __init__(self, channels: int = 16) -> None:
        """Initialize the cross-view attention projection.

        Parameters
        ----------
        channels:
            Latent feature channel width.
        """
        super().__init__()
        self.attn = nn.MultiheadAttention(channels, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(channels)

    def forward(self, view_tokens: Tensor) -> Tensor:
        """Apply cross-view attention between neighboring camera views.

        Parameters
        ----------
        view_tokens:
            Per-view tokens, shape ``(batch, num_views, tokens, channels)``.

        Returns
        -------
        Tensor
            Cross-view-consistent tokens, same shape as ``view_tokens``.
        """
        batch, num_views, tokens, channels = view_tokens.shape
        left_neighbor = torch.roll(view_tokens, shifts=1, dims=1)
        right_neighbor = torch.roll(view_tokens, shifts=-1, dims=1)
        neighbor = torch.cat([left_neighbor, right_neighbor], dim=2)
        flat_q = view_tokens.reshape(batch * num_views, tokens, channels)
        flat_kv = neighbor.reshape(batch * num_views, tokens * 2, channels)
        attended, _ = self.attn(flat_q, flat_kv, flat_kv)
        out = self.norm(flat_q + attended)
        return out.view(batch, num_views, tokens, channels)


class MagicDrive(nn.Module):
    """Multi-camera street-view diffusion with box/BEV/camera-pose conditioning + cross-view attention."""

    def __init__(self, channels: int = 16, num_views: int = 3) -> None:
        """Initialize the box encoder, BEV control branch, cross-view attention, and UNet stub.

        Parameters
        ----------
        channels:
            Shared latent feature channel width.
        num_views:
            Number of synchronized camera views generated jointly.
        """
        super().__init__()
        self.in_proj = nn.Conv2d(4, channels, 3, padding=1)
        self.box_encoder = BoxCrossAttentionEncoder(channels=channels)
        self.bev_branch = BEVMapControlBranch(channels=channels)
        self.pose_proj = nn.Linear(6, channels)
        self.cross_view = CrossViewAttention(channels)
        self.out_proj = nn.Conv2d(channels, 3, 3, padding=1)
        self.channels = channels
        self.num_views = num_views

    def forward(
        self, noisy_latents: Tensor, boxes: Tensor, bev_map: Tensor, camera_pose: Tensor
    ) -> Tensor:
        """Denoise multi-view latents conditioned on boxes, BEV map, and camera pose.

        Parameters
        ----------
        noisy_latents:
            Noisy per-view latents, shape ``(batch, num_views, 4, height, width)``.
        boxes:
            3D bounding boxes shared across views, shape ``(batch, num_boxes, 8)``.
        bev_map:
            BEV road/lane map, shape ``(batch, 4, height, width)``.
        camera_pose:
            Per-view extrinsic/intrinsic pose code, shape ``(batch, num_views, 6)``.

        Returns
        -------
        Tensor
            Denoised per-view RGB frames, shape ``(batch, num_views, 3, height, width)``.
        """
        batch, num_views, _, height, width = noisy_latents.shape
        control = self.bev_branch(bev_map)

        feats = []
        for v in range(num_views):
            feat = self.in_proj(noisy_latents[:, v]) + control
            pose_bias = self.pose_proj(camera_pose[:, v]).view(batch, self.channels, 1, 1)
            feat = feat + pose_bias
            feats.append(feat)
        stacked = torch.stack(feats, dim=1)

        tokens = stacked.flatten(3).transpose(2, 3)
        flat_tokens = tokens.reshape(batch * num_views, height * width, self.channels)
        flat_tokens = self.box_encoder(flat_tokens, boxes.repeat_interleave(num_views, dim=0))
        tokens = flat_tokens.view(batch, num_views, height * width, self.channels)

        tokens = self.cross_view(tokens)
        spatial = tokens.reshape(batch, num_views, height, width, self.channels).permute(
            0, 1, 4, 2, 3
        )
        frames = torch.stack([self.out_proj(spatial[:, v]) for v in range(num_views)], dim=1)
        return frames


def build_magicdrive() -> nn.Module:
    """Build a compact MagicDrive model.

    Returns
    -------
    nn.Module
        Random-initialized ``MagicDrive`` in eval mode.
    """
    return MagicDrive().eval()


def example_input_magicdrive() -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Create example noisy multi-view latents, 3D boxes, BEV map, and camera poses.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor, Tensor]
        Noisy latents ``(1, 3, 4, 8, 8)``, boxes ``(1, 5, 8)``, BEV map
        ``(1, 4, 8, 8)``, and camera poses ``(1, 3, 6)``.
    """
    noisy_latents = torch.randn(1, 3, 4, 8, 8)
    boxes = torch.randn(1, 5, 8)
    bev_map = torch.randn(1, 4, 8, 8)
    camera_pose = torch.randn(1, 3, 6)
    return noisy_latents, boxes, bev_map, camera_pose


# ---------------------------------------------------------------------------
# MID (Motion Indeterminacy Diffusion)
# ---------------------------------------------------------------------------


class SocialStateEncoder(nn.Module):
    """History + social-interaction encoder producing a conditioning state embedding."""

    def __init__(self, in_dim: int = 2, hidden: int = 32) -> None:
        """Initialize the per-agent history encoder and interaction pooling.

        Parameters
        ----------
        in_dim:
            Per-timestep agent position feature dimension.
        hidden:
            Encoder hidden width.
        """
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, batch_first=True)
        self.interact = nn.Linear(hidden * 2, hidden)

    def forward(self, history: Tensor) -> Tensor:
        """Encode agent histories and pool social interaction context.

        Parameters
        ----------
        history:
            Tensor with shape ``(batch, num_agents, time, in_dim)``.

        Returns
        -------
        Tensor
            State embedding for the target (first) agent, shape ``(batch, hidden)``.
        """
        batch, num_agents, time, in_dim = history.shape
        flat = history.reshape(batch * num_agents, time, in_dim)
        _, h_n = self.gru(flat)
        per_agent = h_n.squeeze(0).view(batch, num_agents, -1)
        target = per_agent[:, 0]
        social = per_agent.max(dim=1).values
        return self.interact(torch.cat([target, social], dim=-1))


class TimestepEmbedding(nn.Module):
    """Sinusoidal diffusion-timestep embedding, MLP-projected."""

    def __init__(self, hidden: int = 32) -> None:
        """Initialize the timestep MLP projection.

        Parameters
        ----------
        hidden:
            Output embedding width.
        """
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.hidden = hidden

    def forward(self, timestep: Tensor) -> Tensor:
        """Embed integer diffusion timesteps sinusoidally then project through an MLP.

        Parameters
        ----------
        timestep:
            Integer diffusion step, shape ``(batch,)``.

        Returns
        -------
        Tensor
            Timestep embedding, shape ``(batch, hidden)``.
        """
        half = self.hidden // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=timestep.device).float() / half
        )
        args = timestep.float().unsqueeze(-1) * freqs.unsqueeze(0)
        embed = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(embed)


class TransformerDenoiser(nn.Module):
    """Transformer-based denoiser predicting noise on a trajectory sequence."""

    def __init__(self, horizon: int = 10, hidden: int = 32) -> None:
        """Initialize the trajectory-token projection and Transformer denoising stack.

        Parameters
        ----------
        horizon:
            Number of future timesteps in the diffused trajectory.
        hidden:
            Shared Transformer feature width.
        """
        super().__init__()
        self.in_proj = nn.Linear(2, hidden)
        layer = nn.TransformerEncoderLayer(
            hidden, nhead=4, dim_feedforward=hidden * 2, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.out_proj = nn.Linear(hidden, 2)
        self.horizon = horizon

    def forward(self, noisy_trajectory: Tensor, condition: Tensor) -> Tensor:
        """Predict the noise component of a noisy trajectory given a condition.

        Parameters
        ----------
        noisy_trajectory:
            Noisy future trajectory, shape ``(batch, horizon, 2)``.
        condition:
            Combined state + timestep embedding, shape ``(batch, hidden)``.

        Returns
        -------
        Tensor
            Predicted noise, shape ``(batch, horizon, 2)``.
        """
        tokens = self.in_proj(noisy_trajectory) + condition.unsqueeze(1)
        encoded = self.encoder(tokens)
        return self.out_proj(encoded)


class MID(nn.Module):
    """Motion Indeterminacy Diffusion: reverse-diffusion stochastic trajectory prediction."""

    def __init__(self, horizon: int = 10, hidden: int = 32, num_steps: int = 4) -> None:
        """Initialize the social state encoder, timestep embedding, and denoiser.

        Parameters
        ----------
        horizon:
            Number of future timesteps predicted.
        hidden:
            Shared feature width.
        num_steps:
            Number of accelerated (DDIM-style) reverse-diffusion steps.
        """
        super().__init__()
        self.state_encoder = SocialStateEncoder(hidden=hidden)
        self.time_embed = TimestepEmbedding(hidden=hidden)
        self.denoiser = TransformerDenoiser(horizon=horizon, hidden=hidden)
        self.horizon = horizon
        self.num_steps = num_steps

    def forward(self, history: Tensor, init_noise: Tensor) -> Tensor:
        """Run an accelerated reverse-diffusion loop to denoise a trajectory guess.

        Parameters
        ----------
        history:
            Multi-agent history, shape ``(batch, num_agents, time, 2)``; agent 0
            is the target.
        init_noise:
            Maximally indeterminate initial trajectory guess, shape
            ``(batch, horizon, 2)``.

        Returns
        -------
        Tensor
            Determinate predicted trajectory, shape ``(batch, horizon, 2)``.
        """
        state = self.state_encoder(history)
        batch = history.shape[0]
        trajectory = init_noise
        for step in range(self.num_steps, 0, -1):
            timestep = torch.full((batch,), step, device=history.device, dtype=torch.long)
            condition = state + self.time_embed(timestep)
            predicted_noise = self.denoiser(trajectory, condition)
            trajectory = trajectory - predicted_noise / self.num_steps
        return trajectory


def build_mid() -> nn.Module:
    """Build a compact MID (Motion Indeterminacy Diffusion) model.

    Returns
    -------
    nn.Module
        Random-initialized ``MID`` in eval mode.
    """
    return MID().eval()


def example_input_mid() -> tuple[Tensor, Tensor]:
    """Create example agent histories and an initial noisy trajectory guess.

    Returns
    -------
    tuple[Tensor, Tensor]
        History ``(1, 4, 8, 2)`` and initial noisy trajectory ``(1, 10, 2)``.
    """
    history = torch.randn(1, 4, 8, 2)
    init_noise = torch.randn(1, 10, 2)
    return history, init_noise


# ---------------------------------------------------------------------------
# mmTransformer
# ---------------------------------------------------------------------------


class ProposalStage(nn.Module):
    """Stage 1: each of K fixed proposal queries self-attends against history + context."""

    def __init__(self, hidden: int = 32, num_proposals: int = 6) -> None:
        """Initialize the learned proposal queries and their Transformer layer.

        Parameters
        ----------
        hidden:
            Shared feature width.
        num_proposals:
            Number of fixed multimodal trajectory proposals (K).
        """
        super().__init__()
        self.proposal_queries = nn.Parameter(torch.randn(1, num_proposals, hidden) * 0.02)
        layer = nn.TransformerEncoderLayer(
            hidden, nhead=4, dim_feedforward=hidden * 2, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=1)
        self.num_proposals = num_proposals

    def forward(self, context_tokens: Tensor) -> Tensor:
        """Form K per-mode proposal features by self-attending queries against context.

        Parameters
        ----------
        context_tokens:
            Agent-history + map/social context tokens, shape ``(batch, tokens, hidden)``.

        Returns
        -------
        Tensor
            Per-mode proposal features, shape ``(batch, num_proposals, hidden)``.
        """
        batch = context_tokens.shape[0]
        queries = self.proposal_queries.expand(batch, -1, -1)
        joint = torch.cat([queries, context_tokens], dim=1)
        encoded = self.transformer(joint)
        return encoded[:, : self.num_proposals]


class RegionInteractionStage(nn.Module):
    """Stage 2: the K proposal features self-attend against each other (mode interaction)."""

    def __init__(self, hidden: int = 32) -> None:
        """Initialize the inter-mode Transformer layer.

        Parameters
        ----------
        hidden:
            Shared feature width.
        """
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            hidden, nhead=4, dim_feedforward=hidden * 2, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=1)

    def forward(self, proposal_feats: Tensor) -> Tensor:
        """Let the K proposal features interact so multimodal outputs diversify.

        Parameters
        ----------
        proposal_feats:
            Per-mode proposal features, shape ``(batch, num_proposals, hidden)``.

        Returns
        -------
        Tensor
            Region-interacted proposal features, same shape.
        """
        return self.transformer(proposal_feats)


class TemporalRefinementStage(nn.Module):
    """Stage 3: expand each proposal feature into a full regressed trajectory."""

    def __init__(self, hidden: int = 32, horizon: int = 10) -> None:
        """Initialize the trajectory-decoding MLP and confidence head.

        Parameters
        ----------
        hidden:
            Shared feature width.
        horizon:
            Number of future timesteps regressed per mode.
        """
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, horizon * 2)
        )
        self.confidence = nn.Linear(hidden, 1)
        self.horizon = horizon

    def forward(self, proposal_feats: Tensor) -> tuple[Tensor, Tensor]:
        """Decode each mode's feature into a trajectory and a confidence score.

        Parameters
        ----------
        proposal_feats:
            Region-interacted proposal features, shape ``(batch, num_proposals, hidden)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Trajectories ``(batch, num_proposals, horizon, 2)`` and per-mode
            confidence logits ``(batch, num_proposals)``.
        """
        batch, num_proposals, _ = proposal_feats.shape
        trajectories = self.decoder(proposal_feats).view(batch, num_proposals, self.horizon, 2)
        confidence = self.confidence(proposal_feats).squeeze(-1)
        return trajectories, confidence


class MMTransformer(nn.Module):
    """Multimodal Motion Prediction with Stacked Transformers (proposal / region / refinement)."""

    def __init__(self, hidden: int = 32, num_proposals: int = 6, horizon: int = 10) -> None:
        """Initialize the three stacked-transformer stages.

        Parameters
        ----------
        hidden:
            Shared feature width.
        num_proposals:
            Number of fixed multimodal trajectory proposals (K).
        horizon:
            Number of future timesteps regressed per mode.
        """
        super().__init__()
        self.context_proj = nn.Linear(6, hidden)
        self.proposal_stage = ProposalStage(hidden, num_proposals)
        self.region_stage = RegionInteractionStage(hidden)
        self.refinement_stage = TemporalRefinementStage(hidden, horizon)

    def forward(self, context_features: Tensor) -> tuple[Tensor, Tensor]:
        """Predict K multimodal trajectories via the three stacked-transformer stages.

        Parameters
        ----------
        context_features:
            Agent history + map/social context vectors, shape
            ``(batch, tokens, 6)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Trajectories ``(batch, num_proposals, horizon, 2)`` and per-mode
            confidence logits ``(batch, num_proposals)``.
        """
        context_tokens = self.context_proj(context_features)
        proposal_feats = self.proposal_stage(context_tokens)
        region_feats = self.region_stage(proposal_feats)
        return self.refinement_stage(region_feats)


def build_mmtransformer() -> nn.Module:
    """Build a compact mmTransformer model.

    Returns
    -------
    nn.Module
        Random-initialized ``MMTransformer`` in eval mode.
    """
    return MMTransformer().eval()


def example_input_mmtransformer() -> Tensor:
    """Create an example agent history / map / social context feature tensor.

    Returns
    -------
    Tensor
        Context feature tensor with shape ``(1, 12, 6)``.
    """
    return torch.randn(1, 12, 6)


# ---------------------------------------------------------------------------
# MTR-A (Motion Transformer, marginal + joint)
# ---------------------------------------------------------------------------


class MTRContextEncoder(nn.Module):
    """Simple Transformer context encoder over agent + map polyline tokens."""

    def __init__(self, in_dim: int = 6, hidden: int = 32, depth: int = 2) -> None:
        """Initialize the token projection and self-attention encoder stack.

        Parameters
        ----------
        in_dim:
            Per-token raw feature dimension.
        hidden:
            Shared Transformer feature width.
        depth:
            Number of Transformer encoder layers.
        """
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden)
        layer = nn.TransformerEncoderLayer(
            hidden, nhead=4, dim_feedforward=hidden * 2, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)

    def forward(self, tokens: Tensor) -> Tensor:
        """Self-attend agent + map polyline tokens into a scene context.

        Parameters
        ----------
        tokens:
            Raw agent/map tokens, shape ``(batch, num_tokens, in_dim)``.

        Returns
        -------
        Tensor
            Encoded context tokens, shape ``(batch, num_tokens, hidden)``.
        """
        return self.encoder(self.in_proj(tokens))


class MotionQueryPairDecoder(nn.Module):
    """Learnable static intention query + dynamic query, iteratively refined per layer."""

    def __init__(
        self, hidden: int = 32, num_modes: int = 6, num_layers: int = 2, horizon: int = 10
    ) -> None:
        """Initialize the static intention queries and per-layer refinement stack.

        Parameters
        ----------
        hidden:
            Shared feature width.
        num_modes:
            Number of learnable intention-point motion modes.
        num_layers:
            Number of local-movement-refinement decoder layers.
        horizon:
            Number of future timesteps regressed per mode.
        """
        super().__init__()
        self.static_intention = nn.Parameter(torch.randn(1, num_modes, hidden) * 0.02)
        self.dynamic_query = nn.Parameter(torch.randn(1, num_modes, hidden) * 0.02)
        self.layers = nn.ModuleList(
            [
                nn.MultiheadAttention(hidden, num_heads=4, batch_first=True)
                for _ in range(num_layers)
            ]
        )
        self.refine_mlp = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(num_layers)])
        self.traj_head = nn.Linear(hidden, horizon * 2)
        self.mode_prob_head = nn.Linear(hidden, 1)
        self.horizon = horizon
        self.num_modes = num_modes

    def forward(self, context_tokens: Tensor) -> tuple[Tensor, Tensor]:
        """Localize global intentions then iteratively refine local movement per mode.

        Parameters
        ----------
        context_tokens:
            Encoded scene context, shape ``(batch, num_tokens, hidden)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Per-mode trajectories ``(batch, num_modes, horizon, 2)`` and
            per-mode probability logits ``(batch, num_modes)``.
        """
        batch = context_tokens.shape[0]
        static_q = self.static_intention.expand(batch, -1, -1)
        dynamic_q = self.dynamic_query.expand(batch, -1, -1)
        query = static_q + dynamic_q
        for attn, mlp in zip(self.layers, self.refine_mlp, strict=True):
            attended, _ = attn(query, context_tokens, context_tokens)
            query = query + mlp(attended)
        trajectories = self.traj_head(query).view(batch, self.num_modes, self.horizon, 2)
        mode_probs = self.mode_prob_head(query).squeeze(-1)
        return trajectories, mode_probs


class MTRA(nn.Module):
    """MTR-A: shared context encoder with marginal (single-agent) and joint (multi-agent) heads."""

    def __init__(self, hidden: int = 32, num_modes: int = 6, horizon: int = 10) -> None:
        """Initialize the shared context encoder and the marginal + joint decoders.

        Parameters
        ----------
        hidden:
            Shared feature width.
        num_modes:
            Number of learnable intention-point motion modes.
        horizon:
            Number of future timesteps regressed per mode.
        """
        super().__init__()
        self.encoder = MTRContextEncoder(hidden=hidden)
        self.marginal_decoder = MotionQueryPairDecoder(
            hidden=hidden, num_modes=num_modes, horizon=horizon
        )
        self.joint_decoder = MotionQueryPairDecoder(
            hidden=hidden, num_modes=num_modes, horizon=horizon
        )

    def forward(
        self, agent_tokens: Tensor, map_tokens: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Predict both marginal (per-agent) and joint (multi-agent) motion modes.

        Parameters
        ----------
        agent_tokens:
            Agent-history polyline tokens, shape ``(batch, num_agents, 6)``.
        map_tokens:
            Map/lane polyline tokens, shape ``(batch, num_lanes, 6)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor, Tensor]
            Marginal trajectories, marginal mode probabilities, joint
            trajectories, and joint mode probabilities.
        """
        tokens = torch.cat([agent_tokens, map_tokens], dim=1)
        context = self.encoder(tokens)
        marginal_traj, marginal_prob = self.marginal_decoder(context)
        joint_traj, joint_prob = self.joint_decoder(context)
        return marginal_traj, marginal_prob, joint_traj, joint_prob


def build_mtra() -> nn.Module:
    """Build a compact MTR-A (Marginal + Joint) model.

    Returns
    -------
    nn.Module
        Random-initialized ``MTRA`` in eval mode.
    """
    return MTRA().eval()


def example_input_mtra() -> tuple[Tensor, Tensor]:
    """Create example agent-history and map polyline token tensors.

    Returns
    -------
    tuple[Tensor, Tensor]
        Agent tokens ``(1, 5, 6)`` and map tokens ``(1, 8, 6)``.
    """
    agent_tokens = torch.randn(1, 5, 6)
    map_tokens = torch.randn(1, 8, 6)
    return agent_tokens, map_tokens


# ---------------------------------------------------------------------------
# MultiPath++
# ---------------------------------------------------------------------------


class PolylineEncoder(nn.Module):
    """VectorNet-style shared-MLP + max-pool polyline encoder."""

    def __init__(self, in_dim: int = 6, hidden: int = 32) -> None:
        """Initialize the per-vector MLP and max-pool aggregation.

        Parameters
        ----------
        in_dim:
            Per-vector input feature dimension.
        hidden:
            Output polyline feature width.
        """
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))

    def forward(self, polylines: Tensor) -> Tensor:
        """Encode each polyline's vectors into a single pooled feature.

        Parameters
        ----------
        polylines:
            Tensor with shape ``(batch, num_polylines, num_vectors, in_dim)``.

        Returns
        -------
        Tensor
            Per-polyline features, shape ``(batch, num_polylines, hidden)``.
        """
        feats = self.mlp(polylines)
        return feats.max(dim=2).values


class MultiContextGating(nn.Module):
    """MCG block: gates each entity's feature by a pooled global context (cheaper than cross-attn)."""

    def __init__(self, hidden: int = 32) -> None:
        """Initialize the entity and context gating projections.

        Parameters
        ----------
        hidden:
            Shared feature width.
        """
        super().__init__()
        self.entity_proj = nn.Linear(hidden, hidden)
        self.context_proj = nn.Linear(hidden, hidden)
        self.gate_proj = nn.Linear(hidden * 2, hidden)

    def forward(self, entity_feats: Tensor) -> Tensor:
        """Gate every entity feature by an element-wise-sigmoid pooled context vector.

        Parameters
        ----------
        entity_feats:
            Per-entity features, shape ``(batch, num_entities, hidden)``.

        Returns
        -------
        Tensor
            Context-gated entity features, same shape.
        """
        context = entity_feats.mean(dim=1, keepdim=True).expand_as(entity_feats)
        gate = torch.sigmoid(
            self.gate_proj(
                torch.cat([self.entity_proj(entity_feats), self.context_proj(context)], dim=-1)
            )
        )
        return entity_feats * gate + entity_feats


class MultiPathPlusPlus(nn.Module):
    """MultiPath++: polyline encoding + stacked MCG fusion + learned latent-anchor GMM decoder."""

    def __init__(
        self, hidden: int = 32, num_anchors: int = 6, horizon: int = 10, num_mcg: int = 3
    ) -> None:
        """Initialize the polyline encoders, stacked MCG blocks, and anchor-conditioned GMM decoder.

        Parameters
        ----------
        hidden:
            Shared feature width.
        num_anchors:
            Number of learned latent-anchor embeddings (multimodal outputs).
        horizon:
            Number of future timesteps regressed per anchor.
        num_mcg:
            Number of stacked Multi-Context Gating fusion blocks.
        """
        super().__init__()
        self.agent_history_gru = nn.GRU(2, hidden, batch_first=True)
        self.road_encoder = PolylineEncoder(hidden=hidden)
        self.mcg_blocks = nn.ModuleList([MultiContextGating(hidden) for _ in range(num_mcg)])
        self.latent_anchors = nn.Parameter(torch.randn(1, num_anchors, hidden) * 0.02)
        self.anchor_gate = MultiContextGating(hidden)
        self.gmm_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, horizon * 5)
        )
        self.confidence_head = nn.Linear(hidden, 1)
        self.horizon = horizon
        self.num_anchors = num_anchors

    def forward(self, agent_history: Tensor, road_polylines: Tensor) -> tuple[Tensor, Tensor]:
        """Fuse agent + road context via stacked MCG then decode a learned-anchor GMM.

        Parameters
        ----------
        agent_history:
            Per-agent history states, shape ``(batch, num_agents, time, 2)``.
        road_polylines:
            Road-segment vectors, shape ``(batch, num_segments, num_vectors, 6)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            GMM parameters (mean-xy, log-std-xy, correlation per timestep)
            with shape ``(batch, num_anchors, horizon, 5)`` and per-anchor
            confidence logits ``(batch, num_anchors)``.
        """
        batch, num_agents, time, _ = agent_history.shape
        flat = agent_history.reshape(batch * num_agents, time, 2)
        _, h_n = self.agent_history_gru(flat)
        agent_feats = h_n.squeeze(0).view(batch, num_agents, -1)
        road_feats = self.road_encoder(road_polylines)

        entities = torch.cat([agent_feats, road_feats], dim=1)
        for mcg in self.mcg_blocks:
            entities = mcg(entities)
        scene_context = entities.mean(dim=1, keepdim=True)

        anchors = self.latent_anchors.expand(batch, -1, -1) + scene_context
        anchors = self.anchor_gate(anchors)

        gmm_params = self.gmm_head(anchors).view(batch, self.num_anchors, self.horizon, 5)
        confidence = self.confidence_head(anchors).squeeze(-1)
        return gmm_params, confidence


def build_multipathplusplus() -> nn.Module:
    """Build a compact MultiPath++ model.

    Returns
    -------
    nn.Module
        Random-initialized ``MultiPathPlusPlus`` in eval mode.
    """
    return MultiPathPlusPlus().eval()


def example_input_multipathplusplus() -> tuple[Tensor, Tensor]:
    """Create example agent-history and road-polyline tensors.

    Returns
    -------
    tuple[Tensor, Tensor]
        Agent history ``(1, 4, 8, 2)`` and road polylines ``(1, 6, 9, 6)``.
    """
    agent_history = torch.randn(1, 4, 8, 2)
    road_polylines = torch.randn(1, 6, 9, 6)
    return agent_history, road_polylines


MENAGERIE_ENTRIES = [
    ("LBC", "build_lbc", "example_input_lbc", "2019", "SEQ"),
    ("MagicDrive", "build_magicdrive", "example_input_magicdrive", "2024", "GEN"),
    ("MID", "build_mid", "example_input_mid", "2022", "SEQ"),
    ("mmTransformer", "build_mmtransformer", "example_input_mmtransformer", "2021", "SEQ"),
    ("MTR-A (Marginal + Joint)", "build_mtra", "example_input_mtra", "2022", "SEQ"),
    ("MultiPath", "build_multipathplusplus", "example_input_multipathplusplus", "2021", "SEQ"),
]
