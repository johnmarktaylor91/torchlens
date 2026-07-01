"""Autonomous-driving prediction / planning / world-model classics (batch w4a7).

Sources checked (paper + official repo README/architecture description; no clone,
no pip install -- reimplemented from scratch in base-env torch):

- DenseTNT: Gu et al., ICCV 2021, arXiv:2108.09640.
  https://github.com/Tsinghua-MARS-Lab/DenseTNT
  Anchor-free trajectory prediction: VectorNet-style polyline subgraph encoder,
  a global self-attention context encoder, then a *dense goal probability head*
  (one-layer self-attention + max-pool heatmap encoder, two-layer MLP decoder)
  that scores a dense set of candidate goal points directly, followed by an
  offset + trajectory-completion decoder -- no sparse anchors, no NMS.

- DIPP: Huang et al., IEEE TNNLS 2023, arXiv:2207.10422.
  https://github.com/MCZhi/DIPP
  Differentiable Integrated Prediction and Planning: a shared encoder + a
  multi-agent trajectory prediction decoder produce short-horizon trajectories
  and a *learnable per-term cost-function weight vector*; a differentiable
  nonlinear-optimizer *unrolled* over a handful of gradient-descent steps then
  refines an initial ego-trajectory plan against that learned cost, so the
  planner is trained end-to-end through the optimizer.

- Drive-JEPA: Wang et al., 2026, arXiv:2601.22032.
  https://github.com/linhanwang/Drive-JEPA
  Adapts Video-JEPA to driving: a ViT-style patch encoder over stacked video
  frames produces a *context* representation, a lightweight *predictor* network
  predicts masked/future patch embeddings in latent space (joint-embedding
  prediction, no pixel decoder), and a proposal-centric trajectory-distillation
  head decodes the context embedding into a set of candidate ego trajectories.

- Drive-WM: Wang et al., CVPR 2024, arXiv:2311.17918.
  https://github.com/BraveGroup/Drive-WM
  First multi-view driving world model: a latent video-diffusion UNet with
  *view-factorized cross-view attention* (intermediate camera views are
  predicted conditioned on already-generated adjacent views, not jointly from
  scratch) generates temporally- and view-consistent multi-camera video, and an
  image-based reward head scores imagined rollouts for tree-structured planning.

- DriveDreamer-2: Zhao et al., AAAI 2025, arXiv:2403.06845.
  https://github.com/f1yfisher/DriveDreamer2
  Three-stage pipeline: (1) an LLM trajectory interface maps a natural-language
  scenario into per-agent trajectories (reimplemented as a small transformer
  text-token encoder -> per-agent trajectory head, standing in for the
  fine-tuned GPT-3.5 primitive-calling interface); (2) a ControlNet-style
  latent-diffusion BEV HDMap generator conditions on those trajectories; (3) a
  Unified Multi-View Video Model (UniMVM) with shared cross-view + temporal
  attention denoises the final multi-camera driving video.

- DriveGAN: Kim et al., CVPR 2021, arXiv:2104.15060.
  https://github.com/nv-tlabs/DriveGAN_code
  Controllable neural driving simulator: an encoder disentangles each frame
  into an unsupervised *theme* latent (weather/background) and *content*
  latent (spatial layout); an RNN *Dynamics Engine* steps the content latent
  forward conditioned on an action code, further splitting it into
  action-dependent and action-independent parts; a StyleGAN-style
  modulated generator decodes the stepped latents back into a frame.

Random init, tiny dims, CPU-only -- architecture catalog entries, not trained
weights.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# DenseTNT
# ---------------------------------------------------------------------------


class PolylineSubgraph(nn.Module):
    """VectorNet-style polyline subgraph encoder (per-polyline node aggregation)."""

    def __init__(self, in_dim: int = 8, hidden: int = 32) -> None:
        """Initialize the subgraph MLP + max-pool aggregation layers.

        Parameters
        ----------
        in_dim:
            Per-vector input feature dimension.
        hidden:
            Hidden / output feature dimension.
        """
        super().__init__()
        self.node_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
        )
        self.merge = nn.Linear(hidden * 2, hidden)

    def forward(self, vectors: Tensor) -> Tensor:
        """Encode a polyline's vectors into a single polyline feature.

        Parameters
        ----------
        vectors:
            Tensor with shape ``(batch, num_vectors, in_dim)``.

        Returns
        -------
        Tensor
            Polyline feature with shape ``(batch, hidden)``.
        """
        node_feats = self.node_mlp(vectors)
        global_feat = node_feats.max(dim=1, keepdim=True).values.expand_as(node_feats)
        merged = self.merge(torch.cat([node_feats, global_feat], dim=-1))
        return merged.max(dim=1).values


class DenseGoalHead(nn.Module):
    """Dense goal-probability heatmap head: self-attention + max-pool encoder, MLP decoder."""

    def __init__(self, hidden: int = 32) -> None:
        """Initialize the heatmap self-attention encoder and MLP decoder.

        Parameters
        ----------
        hidden:
            Shared feature dimension for context and goal candidates.
        """
        super().__init__()
        self.query = nn.Linear(hidden, hidden)
        self.key = nn.Linear(hidden, hidden)
        self.value = nn.Linear(hidden, hidden)
        self.score_decoder = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )
        self.offset_decoder = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 2)
        )

    def forward(self, goal_feats: Tensor, context: Tensor) -> tuple[Tensor, Tensor]:
        """Score dense goal candidates and predict a refining offset for each.

        Parameters
        ----------
        goal_feats:
            Candidate goal features with shape ``(batch, num_goals, hidden)``.
        context:
            Scene context feature with shape ``(batch, hidden)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Per-goal probability logits ``(batch, num_goals)`` and per-goal xy
            offsets ``(batch, num_goals, 2)``.
        """
        fused = goal_feats + context.unsqueeze(1)
        q = self.query(fused)
        k = self.key(fused)
        v = self.value(fused)
        attn = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(k.shape[-1]), dim=-1)
        pooled = (attn @ v).max(dim=1, keepdim=True).values.expand_as(fused)
        heatmap_feat = fused + pooled
        logits = self.score_decoder(heatmap_feat).squeeze(-1)
        offsets = self.offset_decoder(heatmap_feat)
        return logits, offsets


class DenseTNT(nn.Module):
    """Anchor-free trajectory prediction from a dense goal-probability heatmap."""

    def __init__(self, hidden: int = 32, horizon: int = 12) -> None:
        """Initialize the polyline encoder, global context, and dense goal head.

        Parameters
        ----------
        hidden:
            Shared feature width.
        horizon:
            Number of future trajectory timesteps to complete.
        """
        super().__init__()
        self.agent_subgraph = PolylineSubgraph(in_dim=6, hidden=hidden)
        self.map_subgraph = PolylineSubgraph(in_dim=8, hidden=hidden)
        self.goal_proj = nn.Linear(2, hidden)
        self.context_attn = nn.MultiheadAttention(hidden, num_heads=4, batch_first=True)
        self.goal_head = DenseGoalHead(hidden)
        self.completion = nn.Sequential(
            nn.Linear(hidden + 2, hidden), nn.ReLU(), nn.Linear(hidden, horizon * 2)
        )
        self.horizon = horizon

    def forward(
        self, agent_polylines: Tensor, map_polylines: Tensor, goal_candidates: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Predict a dense goal heatmap and completed trajectories.

        Parameters
        ----------
        agent_polylines:
            Agent-history vectors with shape ``(batch, num_agents, num_vectors, 6)``.
        map_polylines:
            Lane-segment vectors with shape ``(batch, num_lanes, num_vectors, 8)``.
        goal_candidates:
            Dense candidate goal xy points with shape ``(batch, num_goals, 2)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Goal probability logits ``(batch, num_goals)`` and completed
            trajectories ``(batch, num_goals, horizon, 2)``.
        """
        batch, num_agents, _, _ = agent_polylines.shape
        num_lanes = map_polylines.shape[1]
        agent_feats = torch.stack(
            [self.agent_subgraph(agent_polylines[:, i]) for i in range(num_agents)], dim=1
        )
        lane_feats = torch.stack(
            [self.map_subgraph(map_polylines[:, i]) for i in range(num_lanes)], dim=1
        )
        polyline_feats = torch.cat([agent_feats, lane_feats], dim=1)
        attended, _ = self.context_attn(polyline_feats, polyline_feats, polyline_feats)
        context = attended.max(dim=1).values

        goal_feats = self.goal_proj(goal_candidates)
        logits, offsets = self.goal_head(goal_feats, context)
        refined_goals = goal_candidates + offsets

        completion_in = torch.cat([goal_feats + context.unsqueeze(1), refined_goals], dim=-1)
        trajectories = self.completion(completion_in).view(batch, -1, self.horizon, 2)
        return logits, trajectories


def build_densetnt() -> nn.Module:
    """Build a compact DenseTNT model.

    Returns
    -------
    nn.Module
        Random-initialized ``DenseTNT`` in eval mode.
    """
    return DenseTNT().eval()


def example_input_densetnt() -> tuple[Tensor, Tensor, Tensor]:
    """Create example agent/map polylines and dense goal candidates.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        Agent polylines ``(1, 3, 5, 6)``, map polylines ``(1, 4, 9, 8)``, and
        goal candidates ``(1, 16, 2)``.
    """
    agent_polylines = torch.randn(1, 3, 5, 6)
    map_polylines = torch.randn(1, 4, 9, 8)
    goal_candidates = torch.randn(1, 16, 2)
    return agent_polylines, map_polylines, goal_candidates


# ---------------------------------------------------------------------------
# DIPP
# ---------------------------------------------------------------------------


class DIPPEncoder(nn.Module):
    """Shared agent/map context encoder feeding both prediction and planning."""

    def __init__(self, in_dim: int = 6, hidden: int = 32) -> None:
        """Initialize the recurrent scene-context encoder.

        Parameters
        ----------
        in_dim:
            Per-timestep per-agent feature dimension.
        hidden:
            Encoder hidden width.
        """
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, batch_first=True)

    def forward(self, agent_history: Tensor) -> Tensor:
        """Encode each agent's history track into a context vector.

        Parameters
        ----------
        agent_history:
            Tensor with shape ``(batch, num_agents, time, in_dim)``.

        Returns
        -------
        Tensor
            Per-agent context with shape ``(batch, num_agents, hidden)``.
        """
        batch, num_agents, time, in_dim = agent_history.shape
        flat = agent_history.reshape(batch * num_agents, time, in_dim)
        _, h_n = self.gru(flat)
        return h_n.squeeze(0).view(batch, num_agents, -1)


class DIPPPredictionDecoder(nn.Module):
    """Multi-agent short-horizon trajectory decoder."""

    def __init__(self, hidden: int = 32, horizon: int = 8) -> None:
        """Initialize the trajectory-decoding MLP.

        Parameters
        ----------
        hidden:
            Encoder feature width.
        horizon:
            Number of predicted future timesteps.
        """
        super().__init__()
        self.horizon = horizon
        self.decoder = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, horizon * 2)
        )

    def forward(self, context: Tensor) -> Tensor:
        """Decode per-agent context into future trajectories.

        Parameters
        ----------
        context:
            Per-agent context with shape ``(batch, num_agents, hidden)``.

        Returns
        -------
        Tensor
            Predicted trajectories with shape ``(batch, num_agents, horizon, 2)``.
        """
        batch, num_agents, _ = context.shape
        return self.decoder(context).view(batch, num_agents, self.horizon, 2)


class DifferentiablePlanner(nn.Module):
    """Learnable-cost differentiable planner: gradient descent unrolled a fixed number of steps."""

    def __init__(
        self, horizon: int = 8, cost_terms: int = 4, num_steps: int = 3, step_size: float = 0.05
    ) -> None:
        """Initialize the learnable cost weights and optimizer unroll.

        Parameters
        ----------
        horizon:
            Number of ego-plan timesteps to refine.
        cost_terms:
            Number of scalar cost sub-terms combined into the total cost.
        num_steps:
            Number of unrolled gradient-descent refinement steps.
        step_size:
            Fixed step size for each refinement step.
        """
        super().__init__()
        self.horizon = horizon
        self.num_steps = num_steps
        self.step_size = step_size
        self.cost_weights = nn.Parameter(torch.ones(cost_terms))
        self.goal_proj = nn.Linear(2, cost_terms)

    def _cost(self, ego_plan: Tensor, predicted_others: Tensor, goal: Tensor) -> Tensor:
        """Compute a scalar weighted cost for the current ego plan.

        Parameters
        ----------
        ego_plan:
            Candidate ego trajectory with shape ``(batch, horizon, 2)``.
        predicted_others:
            Predicted trajectories of other agents ``(batch, num_agents, horizon, 2)``.
        goal:
            Ego goal point with shape ``(batch, 2)``.

        Returns
        -------
        Tensor
            Scalar batched cost with shape ``(batch,)``.
        """
        smoothness = (ego_plan[:, 1:] - ego_plan[:, :-1]).pow(2).sum(dim=(-1, -2))
        goal_dist = (ego_plan[:, -1] - goal).pow(2).sum(dim=-1)
        collision = (
            -((ego_plan.unsqueeze(1) - predicted_others).pow(2).sum(dim=-1) + 1.0).reciprocal()
        ).sum(dim=(1, 2))
        goal_terms = self.goal_proj(goal).abs().mean(dim=-1)
        terms = torch.stack([smoothness, goal_dist, -collision, goal_terms], dim=-1)
        weights = F.softplus(self.cost_weights)
        return (terms * weights).sum(dim=-1)

    def forward(self, init_plan: Tensor, predicted_others: Tensor, goal: Tensor) -> Tensor:
        """Refine an initial ego plan by unrolled differentiable gradient descent.

        Parameters
        ----------
        init_plan:
            Initial ego trajectory guess with shape ``(batch, horizon, 2)``.
        predicted_others:
            Predicted trajectories of surrounding agents.
        goal:
            Ego goal point with shape ``(batch, 2)``.

        Returns
        -------
        Tensor
            Refined ego trajectory with shape ``(batch, horizon, 2)``.
        """
        plan = init_plan
        for _ in range(self.num_steps):
            plan = plan.detach().requires_grad_(True)
            cost = self._cost(plan, predicted_others, goal).sum()
            (grad,) = torch.autograd.grad(cost, plan, create_graph=True)
            plan = plan - self.step_size * grad
        return plan


class DIPP(nn.Module):
    """Differentiable Integrated Prediction and Planning."""

    def __init__(self) -> None:
        """Initialize the shared encoder, prediction decoder, and differentiable planner."""
        super().__init__()
        self.encoder = DIPPEncoder()
        self.predictor = DIPPPredictionDecoder()
        self.planner = DifferentiablePlanner()

    def forward(
        self, agent_history: Tensor, init_ego_plan: Tensor, ego_goal: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Predict surrounding-agent trajectories and refine the ego plan.

        Parameters
        ----------
        agent_history:
            Tensor with shape ``(batch, num_agents, time, 6)``; agent 0 is ego.
        init_ego_plan:
            Initial ego trajectory guess with shape ``(batch, horizon, 2)``.
        ego_goal:
            Ego goal point with shape ``(batch, 2)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Predicted trajectories for all agents and the refined ego plan.
        """
        context = self.encoder(agent_history)
        predicted = self.predictor(context)
        with torch.enable_grad():
            refined_plan = self.planner(init_ego_plan, predicted[:, 1:], ego_goal)
        return predicted, refined_plan


def build_dipp() -> nn.Module:
    """Build a compact DIPP model.

    Returns
    -------
    nn.Module
        Random-initialized ``DIPP`` in eval mode.
    """
    return DIPP().eval()


def example_input_dipp() -> tuple[Tensor, Tensor, Tensor]:
    """Create example agent histories, initial ego plan, and ego goal.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        Agent history ``(1, 4, 6, 6)``, initial ego plan ``(1, 8, 2)``, and ego
        goal ``(1, 2)``.
    """
    agent_history = torch.randn(1, 4, 6, 6)
    init_ego_plan = torch.randn(1, 8, 2)
    ego_goal = torch.randn(1, 2)
    return agent_history, init_ego_plan, ego_goal


# ---------------------------------------------------------------------------
# Drive-JEPA
# ---------------------------------------------------------------------------


class PatchEncoder(nn.Module):
    """ViT-style patch encoder over a stack of video frames."""

    def __init__(
        self, patch_size: int = 4, in_ch: int = 3, embed_dim: int = 32, depth: int = 2
    ) -> None:
        """Initialize patch embedding and a small transformer encoder stack.

        Parameters
        ----------
        patch_size:
            Spatial patch size.
        in_ch:
            Input channel count.
        embed_dim:
            Token embedding dimension.
        depth:
            Number of transformer encoder layers.
        """
        super().__init__()
        self.patch_embed = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)
        layer = nn.TransformerEncoderLayer(
            embed_dim, nhead=4, dim_feedforward=embed_dim * 2, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)

    def forward(self, frames: Tensor) -> Tensor:
        """Encode stacked video frames into patch-token embeddings.

        Parameters
        ----------
        frames:
            Tensor with shape ``(batch, time, channels, height, width)``.

        Returns
        -------
        Tensor
            Token embeddings with shape ``(batch, time * num_patches, embed_dim)``.
        """
        batch, time, channels, height, width = frames.shape
        flat = frames.reshape(batch * time, channels, height, width)
        tokens = self.patch_embed(flat).flatten(2).transpose(1, 2)
        tokens = tokens.reshape(batch, time * tokens.shape[1], -1)
        return self.encoder(tokens)


class JEPAPredictor(nn.Module):
    """Lightweight predictor: context tokens + mask tokens -> predicted target embeddings."""

    def __init__(self, embed_dim: int = 32, depth: int = 1) -> None:
        """Initialize the predictor transformer and learned mask token.

        Parameters
        ----------
        embed_dim:
            Shared token embedding dimension.
        depth:
            Number of predictor transformer layers.
        """
        super().__init__()
        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            embed_dim, nhead=4, dim_feedforward=embed_dim * 2, batch_first=True
        )
        self.predictor = nn.TransformerEncoder(layer, num_layers=depth)

    def forward(self, context_tokens: Tensor, num_targets: int) -> Tensor:
        """Predict target-patch embeddings from visible context tokens.

        Parameters
        ----------
        context_tokens:
            Visible context token embeddings ``(batch, num_context, embed_dim)``.
        num_targets:
            Number of masked target tokens to predict.

        Returns
        -------
        Tensor
            Predicted target embeddings with shape ``(batch, num_targets, embed_dim)``.
        """
        batch = context_tokens.shape[0]
        mask_tokens = self.mask_token.expand(batch, num_targets, -1)
        joint = torch.cat([context_tokens, mask_tokens], dim=1)
        out = self.predictor(joint)
        return out[:, -num_targets:]


class DriveJEPA(nn.Module):
    """Video-JEPA driving encoder with proposal-centric trajectory distillation."""

    def __init__(self, embed_dim: int = 32, horizon: int = 6, num_proposals: int = 4) -> None:
        """Initialize the patch encoder, JEPA predictor, and trajectory head.

        Parameters
        ----------
        embed_dim:
            Shared token embedding dimension.
        horizon:
            Number of predicted future ego-trajectory timesteps.
        num_proposals:
            Number of candidate trajectories distilled from the context embedding.
        """
        super().__init__()
        self.encoder = PatchEncoder(embed_dim=embed_dim)
        self.predictor = JEPAPredictor(embed_dim=embed_dim)
        self.trajectory_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_proposals * horizon * 2),
        )
        self.horizon = horizon
        self.num_proposals = num_proposals

    def forward(self, context_frames: Tensor, num_masked_targets: int = 4) -> tuple[Tensor, Tensor]:
        """Encode context frames, predict masked embeddings, and distill trajectories.

        Parameters
        ----------
        context_frames:
            Visible video frames with shape ``(batch, time, channels, height, width)``.
        num_masked_targets:
            Number of masked future-patch embeddings the predictor must produce.

        Returns
        -------
        tuple[Tensor, Tensor]
            Predicted masked-target embeddings and candidate ego trajectories
            with shape ``(batch, num_proposals, horizon, 2)``.
        """
        context_tokens = self.encoder(context_frames)
        predicted_targets = self.predictor(context_tokens, num_masked_targets)
        pooled = context_tokens.mean(dim=1)
        trajectories = self.trajectory_head(pooled).view(-1, self.num_proposals, self.horizon, 2)
        return predicted_targets, trajectories


def build_drivejepa() -> nn.Module:
    """Build a compact Drive-JEPA model.

    Returns
    -------
    nn.Module
        Random-initialized ``DriveJEPA`` in eval mode.
    """
    return DriveJEPA().eval()


def example_input_drivejepa() -> Tensor:
    """Create an example stack of small driving video frames.

    Returns
    -------
    Tensor
        Frame tensor with shape ``(1, 3, 3, 16, 16)``.
    """
    return torch.randn(1, 3, 3, 16, 16)


# ---------------------------------------------------------------------------
# Drive-WM
# ---------------------------------------------------------------------------


class CrossViewAttentionBlock(nn.Module):
    """View-factorized cross-view attention: a view attends to already-generated adjacent views."""

    def __init__(self, channels: int = 16) -> None:
        """Initialize the cross-view attention projection.

        Parameters
        ----------
        channels:
            Feature channel width.
        """
        super().__init__()
        self.attn = nn.MultiheadAttention(channels, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(channels)

    def forward(self, view_feats: Tensor, adjacent_feats: Tensor) -> Tensor:
        """Condition a view's features on already-generated adjacent views.

        Parameters
        ----------
        view_feats:
            Current view tokens with shape ``(batch, tokens, channels)``.
        adjacent_feats:
            Adjacent (already generated) view tokens, same shape.

        Returns
        -------
        Tensor
            Cross-view conditioned tokens, same shape as ``view_feats``.
        """
        attended, _ = self.attn(view_feats, adjacent_feats, adjacent_feats)
        return self.norm(view_feats + attended)


class DriveWMUNetBlock(nn.Module):
    """Small conv block used inside the latent video-diffusion UNet."""

    def __init__(self, channels: int = 16) -> None:
        """Initialize a residual convolutional block.

        Parameters
        ----------
        channels:
            Feature channel width.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        """Apply a residual convolutional transform.

        Parameters
        ----------
        x:
            Feature map with shape ``(batch, channels, height, width)``.

        Returns
        -------
        Tensor
            Residual-refined feature map, same shape.
        """
        return x + self.conv2(self.act(self.conv1(x)))


class DriveWM(nn.Module):
    """Multi-view driving world model: view-factorized cross-view latent diffusion + reward head."""

    def __init__(self, channels: int = 16, num_views: int = 3) -> None:
        """Initialize the per-view latent UNet block, cross-view attention, and reward head.

        Parameters
        ----------
        channels:
            Latent feature channel width.
        num_views:
            Number of synchronized camera views.
        """
        super().__init__()
        self.in_proj = nn.Conv2d(3, channels, 3, padding=1)
        self.unet_block = DriveWMUNetBlock(channels)
        self.cross_view = CrossViewAttentionBlock(channels)
        self.action_proj = nn.Linear(2, channels)
        self.out_proj = nn.Conv2d(channels, 3, 3, padding=1)
        self.reward_head = nn.Sequential(
            nn.Linear(channels, channels), nn.ReLU(), nn.Linear(channels, 1)
        )
        self.num_views = num_views
        self.channels = channels

    def forward(self, multi_view_latents: Tensor, action: Tensor) -> tuple[Tensor, Tensor]:
        """Denoise multi-view latents with cross-view conditioning and score the rollout.

        Parameters
        ----------
        multi_view_latents:
            Noisy multi-view latent frames with shape ``(batch, num_views, 3, height, width)``.
        action:
            Ego action code with shape ``(batch, 2)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Denoised multi-view frames ``(batch, num_views, 3, height, width)`` and
            an image-based reward per view ``(batch, num_views)``.
        """
        batch, num_views, channels_in, height, width = multi_view_latents.shape
        action_bias = self.action_proj(action).view(batch, 1, self.channels, 1, 1)

        view_maps = []
        for view_idx in range(num_views):
            feat = self.in_proj(multi_view_latents[:, view_idx]) + action_bias[:, 0]
            feat = self.unet_block(feat)
            view_maps.append(feat)
        stacked = torch.stack(view_maps, dim=1)

        adjacent = torch.roll(stacked, shifts=1, dims=1)
        tokens = (
            stacked.flatten(3)
            .transpose(2, 3)
            .reshape(batch * num_views, height * width, self.channels)
        )
        adjacent_tokens = (
            adjacent.flatten(3)
            .transpose(2, 3)
            .reshape(batch * num_views, height * width, self.channels)
        )
        conditioned = self.cross_view(tokens, adjacent_tokens)
        conditioned = conditioned.reshape(batch, num_views, height, width, self.channels).permute(
            0, 1, 4, 2, 3
        )

        frames = torch.stack([self.out_proj(conditioned[:, v]) for v in range(num_views)], dim=1)
        pooled = conditioned.mean(dim=(3, 4))
        reward = self.reward_head(pooled).squeeze(-1)
        return frames, reward


def build_drivewm() -> nn.Module:
    """Build a compact Drive-WM model.

    Returns
    -------
    nn.Module
        Random-initialized ``DriveWM`` in eval mode.
    """
    return DriveWM().eval()


def example_input_drivewm() -> tuple[Tensor, Tensor]:
    """Create an example noisy multi-view latent and ego action code.

    Returns
    -------
    tuple[Tensor, Tensor]
        Multi-view latents ``(1, 3, 3, 16, 16)`` and action code ``(1, 2)``.
    """
    multi_view_latents = torch.randn(1, 3, 3, 16, 16)
    action = torch.randn(1, 2)
    return multi_view_latents, action


# ---------------------------------------------------------------------------
# DriveDreamer-2
# ---------------------------------------------------------------------------


class TrajectoryLLMInterface(nn.Module):
    """Small transformer text-token encoder -> per-agent trajectory head.

    Stands in for DriveDreamer-2's fine-tuned GPT-3.5 primitive-calling
    interface that maps a natural-language scenario query into agent
    trajectories.
    """

    def __init__(
        self, vocab: int = 64, embed_dim: int = 24, horizon: int = 6, num_agents: int = 3
    ) -> None:
        """Initialize the token embedding, transformer encoder, and trajectory head.

        Parameters
        ----------
        vocab:
            Token vocabulary size.
        embed_dim:
            Token embedding dimension.
        horizon:
            Number of predicted future timesteps per agent.
        num_agents:
            Number of agents whose trajectories are produced.
        """
        super().__init__()
        self.token_embed = nn.Embedding(vocab, embed_dim)
        layer = nn.TransformerEncoderLayer(
            embed_dim, nhead=4, dim_feedforward=embed_dim * 2, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.trajectory_head = nn.Linear(embed_dim, num_agents * horizon * 2)
        self.horizon = horizon
        self.num_agents = num_agents

    def forward(self, prompt_tokens: Tensor) -> Tensor:
        """Map a tokenized natural-language prompt into agent trajectories.

        Parameters
        ----------
        prompt_tokens:
            Integer token ids with shape ``(batch, seq_len)``.

        Returns
        -------
        Tensor
            Agent trajectories with shape ``(batch, num_agents, horizon, 2)``.
        """
        embedded = self.token_embed(prompt_tokens)
        encoded = self.encoder(embedded).mean(dim=1)
        return self.trajectory_head(encoded).view(-1, self.num_agents, self.horizon, 2)


class HDMapControlNet(nn.Module):
    """ControlNet-style latent-diffusion BEV HDMap generator conditioned on trajectories."""

    def __init__(self, channels: int = 16) -> None:
        """Initialize the trajectory-conditioning projection and BEV conv stack.

        Parameters
        ----------
        channels:
            Latent BEV feature channel width.
        """
        super().__init__()
        self.traj_proj = nn.Linear(2, channels)
        self.in_proj = nn.Conv2d(3, channels, 3, padding=1)
        self.control_conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.out_proj = nn.Conv2d(channels, 3, 3, padding=1)

    def forward(self, bev_latent: Tensor, trajectories: Tensor) -> Tensor:
        """Generate a controlled BEV HDMap conditioned on agent trajectories.

        Parameters
        ----------
        bev_latent:
            Noisy BEV latent map with shape ``(batch, 3, height, width)``.
        trajectories:
            Agent trajectories with shape ``(batch, num_agents, horizon, 2)``.

        Returns
        -------
        Tensor
            Denoised BEV HDMap latent with shape ``(batch, 3, height, width)``.
        """
        control_signal = self.traj_proj(trajectories.mean(dim=(1, 2)))
        feat = self.in_proj(bev_latent)
        feat = feat + control_signal.view(*control_signal.shape, 1, 1)
        feat = F.silu(self.control_conv(feat))
        return self.out_proj(feat)


class UniMVMBlock(nn.Module):
    """Unified Multi-View Video Model block: shared cross-view + temporal attention."""

    def __init__(self, channels: int = 16) -> None:
        """Initialize the shared spatial-temporal-view attention layer.

        Parameters
        ----------
        channels:
            Token feature channel width.
        """
        super().__init__()
        self.attn = nn.MultiheadAttention(channels, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(channels)

    def forward(self, tokens: Tensor) -> Tensor:
        """Apply joint cross-view + temporal self-attention across all frame/view tokens.

        Parameters
        ----------
        tokens:
            Tensor with shape ``(batch, views * time, channels)``.

        Returns
        -------
        Tensor
            Attention-refined tokens, same shape.
        """
        attended, _ = self.attn(tokens, tokens, tokens)
        return self.norm(tokens + attended)


class DriveDreamer2(nn.Module):
    """LLM trajectory interface -> BEV HDMap ControlNet -> unified multi-view video model."""

    def __init__(self, channels: int = 16, num_views: int = 2) -> None:
        """Initialize the trajectory interface, HDMap generator, and UniMVM stage.

        Parameters
        ----------
        channels:
            Latent feature channel width used throughout the pipeline.
        num_views:
            Number of synchronized camera views generated.
        """
        super().__init__()
        self.llm_interface = TrajectoryLLMInterface(embed_dim=channels + 8, num_agents=3)
        self.hdmap_gen = HDMapControlNet(channels)
        self.view_in_proj = nn.Conv2d(4, channels, 3, padding=1)
        self.unimvm = UniMVMBlock(channels)
        self.view_out_proj = nn.Conv2d(channels, 3, 3, padding=1)
        self.num_views = num_views
        self.channels = channels

    def forward(
        self, prompt_tokens: Tensor, bev_latent: Tensor, multi_view_latents: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Run the full LLM-interface / HDMap / multi-view video generation pipeline.

        Parameters
        ----------
        prompt_tokens:
            Tokenized natural-language scenario prompt ``(batch, seq_len)``.
        bev_latent:
            Noisy BEV latent map with shape ``(batch, 3, height, width)``.
        multi_view_latents:
            Noisy multi-camera video latents with shape
            ``(batch, num_views, 3, height, width)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Agent trajectories, denoised BEV HDMap, and denoised multi-view frames.
        """
        trajectories = self.llm_interface(prompt_tokens)
        hdmap = self.hdmap_gen(bev_latent, trajectories)

        batch, num_views, _, height, width = multi_view_latents.shape
        hdmap_cond = hdmap.unsqueeze(1).expand(-1, num_views, -1, -1, -1)
        conditioned_in = torch.cat([multi_view_latents, hdmap_cond[:, :, :1]], dim=2)
        feat = torch.stack(
            [self.view_in_proj(conditioned_in[:, v]) for v in range(num_views)], dim=1
        )

        tokens = (
            feat.flatten(3)
            .transpose(2, 3)
            .reshape(batch, num_views * height * width, self.channels)
        )
        refined = (
            self.unimvm(tokens)
            .reshape(batch, num_views, height, width, self.channels)
            .permute(0, 1, 4, 2, 3)
        )
        frames = torch.stack([self.view_out_proj(refined[:, v]) for v in range(num_views)], dim=1)
        return trajectories, hdmap, frames


def build_drivedreamer2() -> nn.Module:
    """Build a compact DriveDreamer-2 model.

    Returns
    -------
    nn.Module
        Random-initialized ``DriveDreamer2`` in eval mode.
    """
    return DriveDreamer2().eval()


def example_input_drivedreamer2() -> tuple[Tensor, Tensor, Tensor]:
    """Create example prompt tokens, BEV latent, and multi-view latents.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        Prompt tokens ``(1, 6)``, BEV latent ``(1, 3, 16, 16)``, and multi-view
        latents ``(1, 2, 3, 16, 16)``.
    """
    prompt_tokens = torch.randint(0, 64, (1, 6))
    bev_latent = torch.randn(1, 3, 16, 16)
    multi_view_latents = torch.randn(1, 2, 3, 16, 16)
    return prompt_tokens, bev_latent, multi_view_latents


# ---------------------------------------------------------------------------
# DriveGAN
# ---------------------------------------------------------------------------


class DisentanglingEncoder(nn.Module):
    """Encoder splitting a frame into unsupervised theme and content latents."""

    def __init__(self, in_ch: int = 3, theme_dim: int = 8, content_dim: int = 16) -> None:
        """Initialize the shared conv trunk and theme/content projection heads.

        Parameters
        ----------
        in_ch:
            Input frame channel count.
        theme_dim:
            Theme (weather/background) latent dimension.
        content_dim:
            Content (spatial layout) latent dimension.
        """
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.theme_head = nn.Linear(16, theme_dim)
        self.content_head = nn.Linear(16, content_dim)

    def forward(self, frame: Tensor) -> tuple[Tensor, Tensor]:
        """Encode a frame into disentangled theme and content latents.

        Parameters
        ----------
        frame:
            Frame tensor with shape ``(batch, in_ch, height, width)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Theme latent ``(batch, theme_dim)`` and content latent
            ``(batch, content_dim)``.
        """
        pooled = self.trunk(frame).mean(dim=(2, 3))
        return self.theme_head(pooled), self.content_head(pooled)


class DynamicsEngine(nn.Module):
    """RNN dynamics engine: steps the content latent forward given an action code."""

    def __init__(self, content_dim: int = 16, action_dim: int = 2, hidden: int = 24) -> None:
        """Initialize the action-conditioned recurrent cell and split heads.

        Parameters
        ----------
        content_dim:
            Content latent dimension.
        action_dim:
            Action-code dimension.
        hidden:
            Recurrent hidden width.
        """
        super().__init__()
        self.cell = nn.GRUCell(content_dim + action_dim, hidden)
        self.action_dependent_head = nn.Linear(hidden, content_dim // 2)
        self.action_independent_head = nn.Linear(hidden, content_dim - content_dim // 2)

    def forward(
        self, content: Tensor, action: Tensor, hidden_state: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Step the content latent forward conditioned on an action code.

        Parameters
        ----------
        content:
            Current content latent with shape ``(batch, content_dim)``.
        action:
            Action code with shape ``(batch, action_dim)``.
        hidden_state:
            Previous recurrent hidden state with shape ``(batch, hidden)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Next-step content latent and next recurrent hidden state.
        """
        next_hidden = self.cell(torch.cat([content, action], dim=-1), hidden_state)
        action_dependent = self.action_dependent_head(next_hidden)
        action_independent = self.action_independent_head(next_hidden)
        next_content = torch.cat([action_dependent, action_independent], dim=-1)
        return next_content, next_hidden


class ModulatedGenerator(nn.Module):
    """StyleGAN-style modulated generator: latent -> per-channel modulation -> image."""

    def __init__(self, latent_dim: int = 24, channels: int = 16, out_ch: int = 3) -> None:
        """Initialize the constant input, modulation projections, and output conv.

        Parameters
        ----------
        latent_dim:
            Combined theme+content latent dimension.
        channels:
            Generator feature channel width.
        out_ch:
            Output image channel count.
        """
        super().__init__()
        self.const = nn.Parameter(torch.randn(1, channels, 8, 8))
        self.mod = nn.Linear(latent_dim, channels)
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.to_rgb = nn.Conv2d(channels, out_ch, 1)

    def forward(self, latent: Tensor) -> Tensor:
        """Decode a style latent into an image via channel-wise modulation.

        Parameters
        ----------
        latent:
            Style latent with shape ``(batch, latent_dim)``.

        Returns
        -------
        Tensor
            Generated frame with shape ``(batch, out_ch, 8, 8)``.
        """
        batch = latent.shape[0]
        style = self.mod(latent).unsqueeze(-1).unsqueeze(-1)
        feat = self.const.expand(batch, -1, -1, -1) * (1.0 + style)
        feat = F.leaky_relu(self.conv(feat), 0.2)
        return torch.tanh(self.to_rgb(feat))


class DriveGAN(nn.Module):
    """Controllable neural driving simulator: disentangled encoder + dynamics engine + generator."""

    def __init__(self) -> None:
        """Initialize the disentangling encoder, dynamics engine, and modulated generator."""
        super().__init__()
        self.encoder = DisentanglingEncoder()
        self.dynamics = DynamicsEngine()
        self.generator = ModulatedGenerator(latent_dim=8 + 16)

    def forward(self, frame: Tensor, action: Tensor, hidden_state: Tensor) -> tuple[Tensor, Tensor]:
        """Encode a frame, step its content latent by one action, and re-render.

        Parameters
        ----------
        frame:
            Current observed frame with shape ``(batch, 3, height, width)``.
        action:
            Action code applied for this step, shape ``(batch, 2)``.
        hidden_state:
            Dynamics-engine recurrent hidden state, shape ``(batch, 24)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Next simulated frame and the next dynamics-engine hidden state.
        """
        theme, content = self.encoder(frame)
        next_content, next_hidden = self.dynamics(content, action, hidden_state)
        style_latent = torch.cat([theme, next_content], dim=-1)
        next_frame = self.generator(style_latent)
        return next_frame, next_hidden


def build_drivegan() -> nn.Module:
    """Build a compact DriveGAN model.

    Returns
    -------
    nn.Module
        Random-initialized ``DriveGAN`` in eval mode.
    """
    return DriveGAN().eval()


def example_input_drivegan() -> tuple[Tensor, Tensor, Tensor]:
    """Create an example frame, action code, and dynamics hidden state.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        Frame ``(1, 3, 32, 32)``, action code ``(1, 2)``, and hidden state ``(1, 24)``.
    """
    frame = torch.randn(1, 3, 32, 32)
    action = torch.randn(1, 2)
    hidden_state = torch.zeros(1, 24)
    return frame, action, hidden_state


MENAGERIE_ENTRIES = [
    ("DenseTNT", "build_densetnt", "example_input_densetnt", "2021", "SEQ"),
    ("DIPP", "build_dipp", "example_input_dipp", "2023", "SEQ"),
    ("Drive-JEPA", "build_drivejepa", "example_input_drivejepa", "2026", "VIS"),
    ("Drive-WM", "build_drivewm", "example_input_drivewm", "2024", "GEN"),
    ("DriveDreamer-2", "build_drivedreamer2", "example_input_drivedreamer2", "2025", "GEN"),
    ("DriveGAN", "build_drivegan", "example_input_drivegan", "2021", "GEN"),
]
