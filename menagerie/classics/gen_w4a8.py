"""Autonomous-driving perception / generation / forecasting classics (batch w4a8).

Sources checked (paper + official repo README/architecture description; no clone,
no pip install -- reimplemented from scratch in base-env torch):

- DriveLM: Sima et al., ECCV 2024 Oral, arXiv:2312.14150.
  https://github.com/opendrivelab/drivelm
  Graph Visual Question Answering for driving: a frozen-style vision backbone
  produces a scene feature map, then a *graph-structured QA* head chains three
  stages -- Perception nodes (object queries attend to the scene feature map
  to localize key objects), Prediction nodes (each perception node's feature
  is refined by attending to *all other* perception nodes, modelling pairwise
  object interactions), and Planning nodes (a small set of ego-plan queries
  attend to the prediction-stage node features) -- with each stage's node
  states fed into a shared text-token decoder head, mirroring the P1/P2/P3
  graph-of-QA-pairs reasoning chain (localization -> interaction -> action)
  described in the paper, reimplemented as three chained multi-head-attention
  stages over a CNN scene grid instead of the full VLM.

- DriveSceneGen: Sun et al., RA-L 2024, arXiv:2309.14685.
  https://github.com/SS47816/DriveSceneGen
  Two-stage generate-then-simulate traffic scenario synthesis: stage 1 is a
  denoising-diffusion U-Net that generates a rasterized bird's-eye-view (BEV)
  feature map (static lane map channels + dynamic agent-initial-state
  channels) from Gaussian noise conditioned on the diffusion timestep; stage 2
  is a graph-based vectorization + simulation network that pools the denoised
  BEV map at sampled agent locations and predicts each agent's multi-modal
  future-trajectory distribution (mixture of Gaussians) from the pooled scene
  feature, decoupling static-map generation from dynamic-agent rollout.

- DriveTransformer: Jia et al., ICLR 2025, arXiv:2503.07656.
  https://github.com/Thinklab-SJTU/DriveTransformer
  Unified end-to-end driving transformer replacing the sequential
  perception->prediction->planning pipeline with task-parallel blocks. Each
  block applies three unified operations in sequence: *task self-attention*
  (agent, map, and planning queries jointly self-attend so every task can
  condition on every other task at the same depth, not just the one before
  it), *sensor cross-attention* (task queries cross-attend directly to raw
  multi-camera sensor feature tokens -- sparse, no dense BEV bottleneck), and
  *temporal cross-attention* (task queries cross-attend to a stored bank of
  the previous timestep's queries for streaming history), stacked for a few
  blocks before agent/map/planning heads decode the final queries.

- DrivingDiffusion: Li et al., ECCV 2024, arXiv:2310.07771.
  https://github.com/shalfun/DrivingDiffusion
  Layout-guided multi-view driving-video latent diffusion model, cascaded in
  three stages: (1) a multi-view single-frame U-Net denoises N camera views
  jointly from a 3D layout condition, with a *cross-view attention* module
  exchanging information between adjacent camera embeddings so neighboring
  views agree at overlapping fields; (2) a single-view temporal U-Net (shared
  weights across views) denoises subsequent frames conditioned on the stage-1
  keyframe via *cross-frame attention* back to the keyframe's tokens; (3) a
  *local-prompt* cross-attention branch additionally conditions small
  instance regions (boxes) on per-instance text/layout prompts to sharpen
  foreground-object generation quality, reimplemented as an extra masked
  cross-attention term added at instance box locations.

- EigenTrajectory: Bae et al., ICCV 2023, arXiv:2307.09306.
  https://github.com/InhwanBae/EigenTrajectory
  Low-rank multi-modal trajectory forecasting: a *Trajectory Descriptor* first
  performs SVD over a bank of observed trajectory segments (offline, here
  reimplemented as a fixed learnable orthonormal basis of size k) to obtain a
  compact "ET space" -- any trajectory is represented as k eigen-coefficients
  instead of raw (x, y) waypoints. A *Social Interaction Encoder* (small
  self-attention block over agent history projected into ET space) aggregates
  per-agent social context, and a *Trajectory Predictor* head maps the
  social-context embedding to M sets of k eigen-coefficients (multi-modal
  anchors); coefficients are projected back into waypoint space by the fixed
  ET basis (anchor-based refinement in low-rank space rather than Euclidean
  waypoint space).

- ELM (Embodied Language Model for Driving): Zhou et al., ECCV 2024,
  arXiv:2403.04593. https://github.com/OpenDriveLab/ELM
  Space- and time-aware driving-scene language model: a frame-wise vision
  encoder first produces per-timestep patch tokens; a *space-aware*
  cross-attention module lifts these into a 3D-anchored token set using
  camera-pose embeddings (so tokens carry ego-relative spatial position
  across a large spatial span); a *time-aware token selection* module then
  scores every space-aware token across the observed video horizon with a
  lightweight relevance-gating MLP and *top-k selects* the most query-relevant
  tokens over the long time span (avoiding a dense space-time transformer over
  every frame); the selected tokens are finally cross-attended by a language
  decoder that autoregressively produces the answer token sequence.

Random init, tiny dims, CPU-only -- architecture catalog entries, not trained
weights.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# DriveLM: Graph Visual Question Answering
# ---------------------------------------------------------------------------


class DriveLMSceneEncoder(nn.Module):
    """Small CNN backbone producing a flattened scene feature-token grid."""

    def __init__(self, in_channels: int = 3, dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, image: Tensor) -> Tensor:
        """Encode an image into a token grid.

        Parameters
        ----------
        image : Tensor
            Input image, shape ``(batch, in_channels, height, width)``.

        Returns
        -------
        Tensor
            Scene tokens, shape ``(batch, num_tokens, dim)``.
        """
        feat = self.net(image)
        batch, dim, height, width = feat.shape
        return feat.flatten(2).transpose(1, 2).reshape(batch, height * width, dim)


class GraphQAStage(nn.Module):
    """One graph-QA stage: queries attend to a context, then self-refine."""

    def __init__(self, dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, queries: Tensor, context: Tensor) -> Tensor:
        """Attend queries to context, then let queries interact pairwise.

        Parameters
        ----------
        queries : Tensor
            Node queries, shape ``(batch, num_queries, dim)``.
        context : Tensor
            Context tokens to attend to, shape ``(batch, num_context, dim)``.

        Returns
        -------
        Tensor
            Updated node features, shape ``(batch, num_queries, dim)``.
        """
        attended, _ = self.cross_attn(queries, context, context)
        queries = self.norm1(queries + attended)
        refined, _ = self.self_attn(queries, queries, queries)
        return self.norm2(queries + refined)


class DriveLM(nn.Module):
    """Graph-VQA driving model: perception -> prediction -> planning nodes."""

    def __init__(
        self,
        dim: int = 32,
        num_perception: int = 4,
        num_prediction: int = 4,
        num_planning: int = 2,
        vocab_size: int = 64,
    ) -> None:
        super().__init__()
        self.encoder = DriveLMSceneEncoder(dim=dim)
        self.perception_queries = nn.Parameter(torch.randn(1, num_perception, dim) * 0.02)
        self.planning_queries = nn.Parameter(torch.randn(1, num_planning, dim) * 0.02)
        self.perception_stage = GraphQAStage(dim)
        self.prediction_stage = GraphQAStage(dim)
        self.planning_stage = GraphQAStage(dim)
        self.text_decoder = nn.Linear(dim, vocab_size)

    def forward(self, image: Tensor) -> Tensor:
        """Run the perception -> prediction -> planning graph-QA chain.

        Parameters
        ----------
        image : Tensor
            Input scene image, shape ``(batch, 3, height, width)``.

        Returns
        -------
        Tensor
            Per-planning-node vocabulary logits, shape
            ``(batch, num_planning, vocab_size)``.
        """
        batch = image.shape[0]
        scene_tokens = self.encoder(image)
        perception = self.perception_stage(
            self.perception_queries.expand(batch, -1, -1), scene_tokens
        )
        prediction = self.prediction_stage(perception, perception)
        planning = self.planning_stage(self.planning_queries.expand(batch, -1, -1), prediction)
        return self.text_decoder(planning)


def build_drivelm() -> nn.Module:
    """Build a compact DriveLM model.

    Returns
    -------
    nn.Module
        Random-initialized ``DriveLM`` in eval mode.
    """
    return DriveLM().eval()


def example_input_drivelm() -> Tensor:
    """Create an example driving-scene image.

    Returns
    -------
    Tensor
        Image, shape ``(1, 3, 32, 32)``.
    """
    return torch.randn(1, 3, 32, 32)


# ---------------------------------------------------------------------------
# DriveSceneGen: diffusion-based traffic scene generation
# ---------------------------------------------------------------------------


class DiffusionTimeEmbedding(nn.Module):
    """Sinusoidal timestep embedding followed by an MLP."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, timestep: Tensor) -> Tensor:
        """Embed integer diffusion timesteps.

        Parameters
        ----------
        timestep : Tensor
            Timestep indices, shape ``(batch,)``.

        Returns
        -------
        Tensor
            Timestep embedding, shape ``(batch, dim)``.
        """
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=timestep.device) / half)
        args = timestep.float()[:, None] * freqs[None, :]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(embedding)


class BEVDenoiseUNet(nn.Module):
    """Tiny conditional U-Net denoising a rasterized BEV scene map."""

    def __init__(self, channels: int = 4, base: int = 16, time_dim: int = 32) -> None:
        super().__init__()
        self.time_embed = DiffusionTimeEmbedding(time_dim)
        self.time_proj = nn.Linear(time_dim, base * 2)
        self.down = nn.Conv2d(channels, base, kernel_size=3, padding=1)
        self.bottleneck = nn.Conv2d(base, base * 2, kernel_size=3, stride=2, padding=1)
        self.up = nn.ConvTranspose2d(base * 2, base, kernel_size=4, stride=2, padding=1)
        self.out = nn.Conv2d(base, channels, kernel_size=3, padding=1)
        self.act = nn.SiLU()

    def forward(self, bev: Tensor, timestep: Tensor) -> Tensor:
        """Predict the denoising residual for a noisy BEV map.

        Parameters
        ----------
        bev : Tensor
            Noisy rasterized BEV map, shape ``(batch, channels, height, width)``.
        timestep : Tensor
            Diffusion timestep indices, shape ``(batch,)``.

        Returns
        -------
        Tensor
            Predicted noise residual, same shape as ``bev``.
        """
        time_feat = self.time_proj(self.time_embed(timestep))
        feat = self.act(self.down(bev))
        bottleneck = self.act(self.bottleneck(feat))
        bottleneck = bottleneck + time_feat[:, :, None, None]
        up = self.act(self.up(bottleneck))
        return self.out(up)


class AgentTrajectorySimNet(nn.Module):
    """Pool the denoised BEV map at agent seed points, predict a GMM future."""

    def __init__(self, channels: int, dim: int = 32, horizon: int = 8, num_modes: int = 3) -> None:
        super().__init__()
        self.project = nn.Linear(channels, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(inplace=True))
        self.head = nn.Linear(dim, num_modes * horizon * 2)
        self.horizon = horizon
        self.num_modes = num_modes

    def forward(self, bev_map: Tensor, agent_xy: Tensor) -> Tensor:
        """Predict multi-modal future trajectories for seeded agents.

        Parameters
        ----------
        bev_map : Tensor
            Denoised BEV feature map, shape ``(batch, channels, height, width)``.
        agent_xy : Tensor
            Normalized agent seed coordinates in ``[-1, 1]``, shape
            ``(batch, num_agents, 2)``.

        Returns
        -------
        Tensor
            Multi-modal trajectories, shape
            ``(batch, num_agents, num_modes, horizon, 2)``.
        """
        grid = agent_xy.unsqueeze(2)
        pooled = nn.functional.grid_sample(bev_map, grid, align_corners=False, mode="bilinear")
        pooled = pooled.squeeze(-1).transpose(1, 2)
        feat = self.mlp(self.project(pooled))
        out = self.head(feat)
        batch, num_agents, _ = out.shape
        return out.view(batch, num_agents, self.num_modes, self.horizon, 2)


class DriveSceneGen(nn.Module):
    """Two-stage diffusion generate-then-simulate traffic scenario model."""

    def __init__(self, channels: int = 4, base: int = 16) -> None:
        super().__init__()
        self.denoiser = BEVDenoiseUNet(channels=channels, base=base)
        self.sim_net = AgentTrajectorySimNet(channels=channels)

    def forward(
        self, noisy_bev: Tensor, timestep: Tensor, agent_xy: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Denoise one diffusion step and roll out agent futures.

        Parameters
        ----------
        noisy_bev : Tensor
            Noisy BEV map, shape ``(batch, channels, height, width)``.
        timestep : Tensor
            Diffusion timestep indices, shape ``(batch,)``.
        agent_xy : Tensor
            Normalized agent seed coordinates, shape ``(batch, num_agents, 2)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Denoised BEV map and multi-modal agent trajectories.
        """
        noise_pred = self.denoiser(noisy_bev, timestep)
        denoised_bev = noisy_bev - noise_pred
        trajectories = self.sim_net(denoised_bev, agent_xy)
        return denoised_bev, trajectories


def build_drivescenegen() -> nn.Module:
    """Build a compact DriveSceneGen model.

    Returns
    -------
    nn.Module
        Random-initialized ``DriveSceneGen`` in eval mode.
    """
    return DriveSceneGen().eval()


def example_input_drivescenegen() -> tuple[Tensor, Tensor, Tensor]:
    """Create an example noisy BEV map, timestep, and agent seed points.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        Noisy BEV map ``(1, 4, 32, 32)``, timestep ``(1,)``, and agent
        coordinates ``(1, 5, 2)``.
    """
    noisy_bev = torch.randn(1, 4, 32, 32)
    timestep = torch.randint(0, 1000, (1,))
    agent_xy = torch.rand(1, 5, 2) * 2 - 1
    return noisy_bev, timestep, agent_xy


# ---------------------------------------------------------------------------
# DriveTransformer: unified task-parallel transformer
# ---------------------------------------------------------------------------


class DriveTransformerBlock(nn.Module):
    """One block: task self-attention, sensor cross-attention, temporal cross-attention."""

    def __init__(self, dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.task_self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.sensor_cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.temporal_cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.ReLU(inplace=True), nn.Linear(dim * 2, dim)
        )
        self.norm4 = nn.LayerNorm(dim)

    def forward(self, tasks: Tensor, sensor_tokens: Tensor, history: Tensor) -> Tensor:
        """Update task queries via the three unified operations.

        Parameters
        ----------
        tasks : Tensor
            Concatenated agent/map/planning queries, shape
            ``(batch, num_tasks, dim)``.
        sensor_tokens : Tensor
            Raw multi-camera sensor feature tokens, shape
            ``(batch, num_sensor_tokens, dim)``.
        history : Tensor
            Previous-timestep task queries, shape ``(batch, num_tasks, dim)``.

        Returns
        -------
        Tensor
            Updated task queries, shape ``(batch, num_tasks, dim)``.
        """
        self_out, _ = self.task_self_attn(tasks, tasks, tasks)
        tasks = self.norm1(tasks + self_out)
        sensor_out, _ = self.sensor_cross_attn(tasks, sensor_tokens, sensor_tokens)
        tasks = self.norm2(tasks + sensor_out)
        temporal_out, _ = self.temporal_cross_attn(tasks, history, history)
        tasks = self.norm3(tasks + temporal_out)
        return self.norm4(tasks + self.ffn(tasks))


class DriveTransformer(nn.Module):
    """Task-parallel unified transformer for detection + prediction + planning."""

    def __init__(
        self,
        dim: int = 32,
        num_agent: int = 4,
        num_map: int = 4,
        num_planning: int = 2,
        num_blocks: int = 2,
    ) -> None:
        super().__init__()
        self.num_agent = num_agent
        self.num_map = num_map
        self.num_planning = num_planning
        self.sensor_proj = nn.Conv2d(3, dim, kernel_size=4, stride=4)
        self.agent_queries = nn.Parameter(torch.randn(1, num_agent, dim) * 0.02)
        self.map_queries = nn.Parameter(torch.randn(1, num_map, dim) * 0.02)
        self.planning_queries = nn.Parameter(torch.randn(1, num_planning, dim) * 0.02)
        self.blocks = nn.ModuleList([DriveTransformerBlock(dim) for _ in range(num_blocks)])
        self.agent_head = nn.Linear(dim, 4)
        self.map_head = nn.Linear(dim, 4)
        self.planning_head = nn.Linear(dim, 2)

    def forward(self, sensor_image: Tensor, history: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Run task-parallel blocks over sensor tokens and streaming history.

        Parameters
        ----------
        sensor_image : Tensor
            Raw camera image, shape ``(batch, 3, height, width)``.
        history : Tensor
            Previous-timestep task queries, shape
            ``(batch, num_agent + num_map + num_planning, dim)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Agent boxes, map point predictions, and planning waypoints.
        """
        batch = sensor_image.shape[0]
        sensor_feat = self.sensor_proj(sensor_image)
        sensor_tokens = sensor_feat.flatten(2).transpose(1, 2)
        tasks = torch.cat(
            [
                self.agent_queries.expand(batch, -1, -1),
                self.map_queries.expand(batch, -1, -1),
                self.planning_queries.expand(batch, -1, -1),
            ],
            dim=1,
        )
        for block in self.blocks:
            tasks = block(tasks, sensor_tokens, history)
        agent_tasks, map_tasks, planning_tasks = torch.split(
            tasks, [self.num_agent, self.num_map, self.num_planning], dim=1
        )
        return (
            self.agent_head(agent_tasks),
            self.map_head(map_tasks),
            self.planning_head(planning_tasks),
        )


def build_drivetransformer() -> nn.Module:
    """Build a compact DriveTransformer model.

    Returns
    -------
    nn.Module
        Random-initialized ``DriveTransformer`` in eval mode.
    """
    return DriveTransformer().eval()


def example_input_drivetransformer() -> tuple[Tensor, Tensor]:
    """Create an example sensor image and query history bank.

    Returns
    -------
    tuple[Tensor, Tensor]
        Sensor image ``(1, 3, 32, 32)`` and history queries ``(1, 10, 32)``.
    """
    sensor_image = torch.randn(1, 3, 32, 32)
    history = torch.randn(1, 10, 32)
    return sensor_image, history


# ---------------------------------------------------------------------------
# DrivingDiffusion: layout-guided multi-view video latent diffusion
# ---------------------------------------------------------------------------


class CrossViewAttention(nn.Module):
    """Exchange information between adjacent camera-view tokens."""

    def __init__(self, dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, view_tokens: Tensor) -> Tensor:
        """Attend each view's tokens to its cyclic-adjacent neighbor view.

        Parameters
        ----------
        view_tokens : Tensor
            Per-view tokens, shape ``(batch, num_views, tokens_per_view, dim)``.

        Returns
        -------
        Tensor
            Cross-view-refined tokens, same shape as ``view_tokens``.
        """
        batch, num_views, num_tokens, dim = view_tokens.shape
        flat = view_tokens.reshape(batch * num_views, num_tokens, dim)
        neighbor = torch.roll(view_tokens, shifts=1, dims=1).reshape(
            batch * num_views, num_tokens, dim
        )
        out, _ = self.attn(flat, neighbor, neighbor)
        out = self.norm(flat + out)
        return out.view(batch, num_views, num_tokens, dim)


class CrossFrameAttention(nn.Module):
    """Attend future-frame tokens back to the fixed stage-1 keyframe tokens."""

    def __init__(self, dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, frame_tokens: Tensor, keyframe_tokens: Tensor) -> Tensor:
        """Condition frame tokens on the keyframe tokens.

        Parameters
        ----------
        frame_tokens : Tensor
            Tokens of the frame being generated, shape
            ``(batch, num_tokens, dim)``.
        keyframe_tokens : Tensor
            Tokens of the fixed stage-1 keyframe, shape
            ``(batch, num_tokens, dim)``.

        Returns
        -------
        Tensor
            Temporally-conditioned frame tokens, shape
            ``(batch, num_tokens, dim)``.
        """
        out, _ = self.attn(frame_tokens, keyframe_tokens, keyframe_tokens)
        return self.norm(frame_tokens + out)


class LocalPromptAttention(nn.Module):
    """Extra masked cross-attention conditioning small instance regions."""

    def __init__(self, dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens: Tensor, instance_prompts: Tensor, instance_mask: Tensor) -> Tensor:
        """Sharpen instance-region tokens using per-instance prompt embeddings.

        Parameters
        ----------
        tokens : Tensor
            Frame tokens, shape ``(batch, num_tokens, dim)``.
        instance_prompts : Tensor
            Per-instance prompt embeddings, shape
            ``(batch, num_instances, dim)``.
        instance_mask : Tensor
            Boolean mask selecting instance-region tokens, shape
            ``(batch, num_tokens)``.

        Returns
        -------
        Tensor
            Tokens with instance regions refined, shape
            ``(batch, num_tokens, dim)``.
        """
        refined, _ = self.attn(tokens, instance_prompts, instance_prompts)
        gate = instance_mask.unsqueeze(-1).to(tokens.dtype)
        return self.norm(tokens + refined * gate)


class DrivingDiffusion(nn.Module):
    """Cascaded multi-view driving-video latent diffusion model."""

    def __init__(self, dim: int = 32, num_views: int = 3, num_heads: int = 4) -> None:
        super().__init__()
        self.dim = dim
        self.num_views = num_views
        self.patch = nn.Conv2d(3, dim, kernel_size=4, stride=4)
        self.layout_proj = nn.Linear(dim, dim)
        self.cross_view = CrossViewAttention(dim, num_heads)
        self.cross_frame = CrossFrameAttention(dim, num_heads)
        self.local_prompt = LocalPromptAttention(dim, num_heads)
        self.out_proj = nn.Linear(dim, dim)

    def forward(
        self,
        multi_view_frames: Tensor,
        layout: Tensor,
        instance_prompts: Tensor,
        instance_mask: Tensor,
    ) -> Tensor:
        """Denoise multi-view frames with cross-view/frame/local-prompt conditioning.

        Parameters
        ----------
        multi_view_frames : Tensor
            Noisy frames for each camera view, shape
            ``(batch, num_views, 3, height, width)``.
        layout : Tensor
            3D layout condition tokens, shape ``(batch, tokens, dim)``.
        instance_prompts : Tensor
            Per-instance local-prompt embeddings, shape
            ``(batch, num_instances, dim)``.
        instance_mask : Tensor
            Boolean mask over the flattened first-view tokens, shape
            ``(batch, tokens)``.

        Returns
        -------
        Tensor
            Denoised token features for the first camera view, shape
            ``(batch, tokens, dim)``.
        """
        batch, num_views = multi_view_frames.shape[:2]
        patches = self.patch(multi_view_frames.flatten(0, 1))
        _, dim, height, width = patches.shape
        tokens = patches.flatten(2).transpose(1, 2).view(batch, num_views, height * width, dim)
        tokens = self.cross_view(tokens)
        keyframe_tokens = tokens[:, 0] + self.layout_proj(layout.mean(dim=1, keepdim=True))
        first_view = self.cross_frame(tokens[:, 0], keyframe_tokens)
        first_view = self.local_prompt(first_view, instance_prompts, instance_mask)
        return self.out_proj(first_view)


def build_drivingdiffusion() -> nn.Module:
    """Build a compact DrivingDiffusion model.

    Returns
    -------
    nn.Module
        Random-initialized ``DrivingDiffusion`` in eval mode.
    """
    return DrivingDiffusion().eval()


def example_input_drivingdiffusion() -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Create example multi-view frames, layout tokens, and instance prompts.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor, Tensor]
        Multi-view frames ``(1, 3, 3, 32, 32)``, layout tokens ``(1, 6, 32)``,
        instance prompts ``(1, 2, 32)``, and instance mask ``(1, 64)``.
    """
    multi_view_frames = torch.randn(1, 3, 3, 32, 32)
    layout = torch.randn(1, 6, 32)
    instance_prompts = torch.randn(1, 2, 32)
    instance_mask = torch.zeros(1, 64, dtype=torch.bool)
    instance_mask[:, :8] = True
    return multi_view_frames, layout, instance_prompts, instance_mask


# ---------------------------------------------------------------------------
# EigenTrajectory: SVD low-rank trajectory descriptors
# ---------------------------------------------------------------------------


class TrajectoryDescriptor(nn.Module):
    """Fixed learnable orthonormal ET-space basis (stand-in for offline SVD)."""

    def __init__(self, num_waypoints: int = 8, rank: int = 6) -> None:
        super().__init__()
        raw = torch.randn(num_waypoints * 2, rank)
        basis, _ = torch.linalg.qr(raw)
        self.register_buffer("basis", basis)
        self.num_waypoints = num_waypoints
        self.rank = rank

    def encode(self, trajectory: Tensor) -> Tensor:
        """Project a waypoint trajectory into ET-space eigen-coefficients.

        Parameters
        ----------
        trajectory : Tensor
            Waypoint trajectory, shape ``(batch, num_waypoints, 2)``.

        Returns
        -------
        Tensor
            Eigen-coefficients, shape ``(batch, rank)``.
        """
        flat = trajectory.reshape(trajectory.shape[0], -1)
        return flat @ self.basis

    def decode(self, coefficients: Tensor) -> Tensor:
        """Project ET-space eigen-coefficients back to waypoints.

        Parameters
        ----------
        coefficients : Tensor
            Eigen-coefficients, shape ``(..., rank)``.

        Returns
        -------
        Tensor
            Waypoint trajectory, shape ``(..., num_waypoints, 2)``.
        """
        flat = coefficients @ self.basis.T
        return flat.view(*coefficients.shape[:-1], self.num_waypoints, 2)


class SocialInteractionEncoder(nn.Module):
    """Self-attention block aggregating per-agent social context in ET space."""

    def __init__(self, rank: int, dim: int = 32, num_heads: int = 4) -> None:
        super().__init__()
        self.in_proj = nn.Linear(rank, dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, history_coefficients: Tensor) -> Tensor:
        """Aggregate social context across agents from their ET-space history.

        Parameters
        ----------
        history_coefficients : Tensor
            Observed-history eigen-coefficients per agent, shape
            ``(batch, num_agents, rank)``.

        Returns
        -------
        Tensor
            Social-context embedding per agent, shape
            ``(batch, num_agents, dim)``.
        """
        feat = self.in_proj(history_coefficients)
        attended, _ = self.attn(feat, feat, feat)
        return self.norm(feat + attended)


class EigenTrajectory(nn.Module):
    """Low-rank multi-modal trajectory forecaster with anchor-based refinement."""

    def __init__(
        self,
        num_waypoints: int = 8,
        rank: int = 6,
        dim: int = 32,
        num_modes: int = 3,
    ) -> None:
        super().__init__()
        self.descriptor = TrajectoryDescriptor(num_waypoints=num_waypoints, rank=rank)
        self.social_encoder = SocialInteractionEncoder(rank, dim=dim)
        self.predictor = nn.Linear(dim, num_modes * rank)
        self.anchor_offsets = nn.Parameter(torch.randn(num_modes, rank) * 0.1)
        self.num_modes = num_modes
        self.rank = rank

    def forward(self, history_trajectory: Tensor) -> Tensor:
        """Predict multi-modal future trajectories from observed history.

        Parameters
        ----------
        history_trajectory : Tensor
            Observed per-agent trajectory, shape
            ``(batch, num_agents, num_waypoints, 2)``.

        Returns
        -------
        Tensor
            Multi-modal predicted future trajectories, shape
            ``(batch, num_agents, num_modes, num_waypoints, 2)``.
        """
        batch, num_agents = history_trajectory.shape[:2]
        flat_history = history_trajectory.reshape(batch * num_agents, *history_trajectory.shape[2:])
        history_coefficients = self.descriptor.encode(flat_history).view(
            batch, num_agents, self.rank
        )
        social_feat = self.social_encoder(history_coefficients)
        pred_coefficients = self.predictor(social_feat).view(
            batch, num_agents, self.num_modes, self.rank
        )
        pred_coefficients = pred_coefficients + self.anchor_offsets
        return self.descriptor.decode(pred_coefficients)


def build_eigentrajectory() -> nn.Module:
    """Build a compact EigenTrajectory model.

    Returns
    -------
    nn.Module
        Random-initialized ``EigenTrajectory`` in eval mode.
    """
    return EigenTrajectory().eval()


def example_input_eigentrajectory() -> Tensor:
    """Create an example observed multi-agent history trajectory.

    Returns
    -------
    Tensor
        History trajectory, shape ``(1, 5, 8, 2)``.
    """
    return torch.randn(1, 5, 8, 2)


# ---------------------------------------------------------------------------
# ELM: space-aware + time-aware embodied driving understanding
# ---------------------------------------------------------------------------


class SpaceAwareEncoder(nn.Module):
    """Lift per-frame patch tokens into ego-relative 3D-anchored tokens."""

    def __init__(self, dim: int = 32, num_heads: int = 4) -> None:
        super().__init__()
        self.patch = nn.Conv2d(3, dim, kernel_size=4, stride=4)
        self.pose_proj = nn.Linear(6, dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, frames: Tensor, camera_pose: Tensor) -> Tensor:
        """Encode frames into space-aware tokens conditioned on camera pose.

        Parameters
        ----------
        frames : Tensor
            Video frames, shape ``(batch, time, 3, height, width)``.
        camera_pose : Tensor
            Per-frame 6-DoF camera pose, shape ``(batch, time, 6)``.

        Returns
        -------
        Tensor
            Space-aware tokens, shape ``(batch, time, tokens_per_frame, dim)``.
        """
        batch, time = frames.shape[:2]
        patches = self.patch(frames.flatten(0, 1))
        _, dim, height, width = patches.shape
        tokens = patches.flatten(2).transpose(1, 2)
        pose_embedding = self.pose_proj(camera_pose.flatten(0, 1)).unsqueeze(1)
        attended, _ = self.attn(tokens, pose_embedding, pose_embedding)
        tokens = self.norm(tokens + attended)
        return tokens.view(batch, time, height * width, dim)


class TimeAwareTokenSelector(nn.Module):
    """Score every space-aware token across the horizon and top-k select."""

    def __init__(self, dim: int, top_k: int = 8) -> None:
        super().__init__()
        self.score_mlp = nn.Sequential(
            nn.Linear(dim, dim // 2), nn.ReLU(inplace=True), nn.Linear(dim // 2, 1)
        )
        self.top_k = top_k

    def forward(self, space_tokens: Tensor, query: Tensor) -> Tensor:
        """Select the top-k most query-relevant tokens across time and space.

        Parameters
        ----------
        space_tokens : Tensor
            Space-aware tokens, shape
            ``(batch, time, tokens_per_frame, dim)``.
        query : Tensor
            Query embedding steering relevance, shape ``(batch, dim)``.

        Returns
        -------
        Tensor
            Selected tokens, shape ``(batch, top_k, dim)``.
        """
        batch, time, tokens_per_frame, dim = space_tokens.shape
        flat = space_tokens.reshape(batch, time * tokens_per_frame, dim)
        gated = flat + query.unsqueeze(1)
        scores = self.score_mlp(gated).squeeze(-1)
        top_k = min(self.top_k, scores.shape[1])
        _, indices = torch.topk(scores, top_k, dim=1)
        gather_index = indices.unsqueeze(-1).expand(-1, -1, dim)
        return torch.gather(flat, 1, gather_index)


class ELM(nn.Module):
    """Space-aware + time-aware embodied driving-scene language model."""

    def __init__(
        self, dim: int = 32, top_k: int = 8, vocab_size: int = 64, answer_len: int = 4
    ) -> None:
        super().__init__()
        self.space_encoder = SpaceAwareEncoder(dim=dim)
        self.token_selector = TimeAwareTokenSelector(dim=dim, top_k=top_k)
        self.query_embed = nn.Embedding(vocab_size, dim)
        self.answer_queries = nn.Parameter(torch.randn(1, answer_len, dim) * 0.02)
        self.decoder_attn = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.decoder_norm = nn.LayerNorm(dim)
        self.vocab_head = nn.Linear(dim, vocab_size)

    def forward(self, frames: Tensor, camera_pose: Tensor, question_ids: Tensor) -> Tensor:
        """Answer a driving-scene question over a long spatio-temporal horizon.

        Parameters
        ----------
        frames : Tensor
            Video frames, shape ``(batch, time, 3, height, width)``.
        camera_pose : Tensor
            Per-frame 6-DoF camera pose, shape ``(batch, time, 6)``.
        question_ids : Tensor
            Question token ids, shape ``(batch, question_len)``.

        Returns
        -------
        Tensor
            Per-answer-token vocabulary logits, shape
            ``(batch, answer_len, vocab_size)``.
        """
        batch = frames.shape[0]
        space_tokens = self.space_encoder(frames, camera_pose)
        question_feat = self.query_embed(question_ids).mean(dim=1)
        selected = self.token_selector(space_tokens, question_feat)
        answer_queries = self.answer_queries.expand(batch, -1, -1)
        decoded, _ = self.decoder_attn(answer_queries, selected, selected)
        decoded = self.decoder_norm(answer_queries + decoded)
        return self.vocab_head(decoded)


def build_elm() -> nn.Module:
    """Build a compact ELM model.

    Returns
    -------
    nn.Module
        Random-initialized ``ELM`` in eval mode.
    """
    return ELM().eval()


def example_input_elm() -> tuple[Tensor, Tensor, Tensor]:
    """Create example video frames, camera poses, and question token ids.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        Frames ``(1, 4, 3, 32, 32)``, camera pose ``(1, 4, 6)``, and
        question ids ``(1, 6)``.
    """
    frames = torch.randn(1, 4, 3, 32, 32)
    camera_pose = torch.randn(1, 4, 6)
    question_ids = torch.randint(0, 64, (1, 6))
    return frames, camera_pose, question_ids


MENAGERIE_ENTRIES = [
    ("DriveLM", "build_drivelm", "example_input_drivelm", "2024", "VIS"),
    ("DriveSceneGen", "build_drivescenegen", "example_input_drivescenegen", "2024", "GEN"),
    ("DriveTransformer", "build_drivetransformer", "example_input_drivetransformer", "2025", "VIS"),
    ("DrivingDiffusion", "build_drivingdiffusion", "example_input_drivingdiffusion", "2024", "GEN"),
    ("EigenTrajectory", "build_eigentrajectory", "example_input_eigentrajectory", "2023", "SEQ"),
    ("ELM (Embodied Language Model for Driving)", "build_elm", "example_input_elm", "2024", "VIS"),
]
