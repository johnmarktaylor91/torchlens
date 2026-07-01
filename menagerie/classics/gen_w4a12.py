"""Autonomous-driving perception / generation / trajectory-prediction classics (batch w4a12).

Sources checked (paper + official repo README/architecture description; no clone,
no pip install -- reimplemented from scratch in base-env torch):

- NPSN (Non-Probability Sampling Network): Bae, Park & Jeon, CVPR 2022,
  arXiv:2203.13471. https://github.com/InhwanBae/NPSN
  A tiny *plug-in* sampler that replaces random Monte-Carlo sampling in any
  off-the-shelf stochastic (CVAE-style) trajectory predictor. Per-agent
  history features are refined with a graph-attention layer (multi-head
  attention over agents in the same scene, masked to the current scene),
  then a small MLP head regresses ``n`` purposive quasi-sample locations in
  ``[0, 1]^s`` (via a sigmoid + clamp). Those locations are pushed through a
  Box-Muller transform to standard-normal variates, which are then mapped
  through the predictor's own per-agent Gaussian (mean, Cholesky factor of
  covariance) to obtain the final purposive sample set -- i.e. NPSN learns
  *where* to sample the underlying Gaussian instead of sampling randomly.

- OccWorld: Zheng, Chen, Huang, Zhang, Duan & Lu, ECCV 2024, arXiv:2311.16038.
  https://github.com/wzzheng/OccWorld
  A GPT-style 3D-occupancy world model. Stage 1 is a small VQ-VAE scene
  tokenizer: a 3D-occupancy voxel grid is encoded to a discrete token grid
  via a learned codebook (vector quantization) and can be decoded back to
  occupancy logits. Stage 2 is a spatial-temporal generative transformer:
  per-frame scene tokens (flattened over space) are concatenated with a
  learned ego-motion token, spatial self-attention mixes tokens within a
  frame, and causal temporal self-attention across frames autoregressively
  predicts the next frame's scene tokens and ego token jointly (a GPT-like
  "world model" over discretized 3D occupancy rather than boxes/segmentation).

- Panacea: Wen, Zhao, Liu, Jia, Wang, Luo, Zhang, Wang, Sun & Zhang,
  CVPR 2024, arXiv:2311.16813. https://github.com/wenyuqing/panacea
  Panoramic, multi-view, multi-frame controllable video diffusion for
  driving scenes. The distinguishing mechanism is *4D attention* inside each
  denoising UNet block: standard spatial self-attention is followed by (a) a
  temporal-attention layer that lets each spatial location attend across the
  frame axis and (b) a cross-view-attention layer that lets each camera view
  attend to its neighboring camera views, enforcing both temporal and
  cross-view (panoramic) consistency. A BEV layout (drivable area, boxes,
  map elements, rendered to an image-like tensor) is injected through an
  additive ControlNet-style side branch summed into the UNet features.

- PECNet (Predicted Endpoint Conditioned Network): Mangalam, Girase,
  Agarwal, Lee, Adeli, Gaidon & Malik, ECCV 2020 (oral), arXiv:2004.02025.
  https://github.com/HarshayuGirase/PECNet
  A CVAE that predicts a distant trajectory *endpoint* first and conditions
  the rest of the path on it. A past-trajectory MLP encoder and a
  destination MLP encoder feed a latent encoder producing (mu, logvar); a
  decoder MLP reconstructs the endpoint from (past-features, latent code).
  The endpoint features, past features and each agent's initial position are
  then refined by a stack of *non-local social pooling* blocks -- a
  non-local-block-style (theta/phi/g, softmax-attention) operation over all
  agents in the scene, masked to same-scene neighbors -- before a predictor
  MLP regresses the interior waypoints between start and endpoint. Reimplemented
  here in inference mode: sample a latent from N(0, sigma) instead of the
  training-time CVAE posterior.

- PilotNet: Bojarski et al. (NVIDIA), 2016, arXiv:1604.07316.
  https://github.com/TerrisGO/PilotNet (community PyTorch port of the
  original NVIDIA "End to End Learning for Self-Driving Cars" architecture).
  The seminal end-to-end steering CNN: a single front-facing-camera RGB/YUV
  image is mapped directly to a steering-angle scalar with no explicit lane
  or path-planning stages. Architecture (as specified in the paper): a
  normalization layer, 3 strided 5x5 convolutions (24/36/48 channels,
  stride 2) followed by 2 non-strided 3x3 convolutions (64 channels each),
  flattened into a fully-connected stack of 1164 -> 100 -> 50 -> 10 -> 1
  units, with ELU/ReLU nonlinearities throughout, ending in a single
  steering-angle output.

- PlanT: Renz, Chitta, Mercea, Koepke, Akata & Geiger, CoRL 2022,
  arXiv:2210.14222. https://github.com/autonomousvision/plant
  An *object-level* (not pixel-level) planning transformer for self-driving.
  Each traffic participant and each route point is represented as a compact
  attribute vector (x, y, yaw, speed, extent-x, extent-y) and embedded with a
  per-object-type linear layer (vehicles vs. route points use distinct
  embeddings). A learned CLS token is prepended to the object-token sequence
  and the whole sequence is passed through a small GPT-2-style transformer
  encoder (via ``transformers.AutoModel``). The CLS token's output embedding
  is linearly projected and then autoregressively unrolled by a GRUCell into
  a sequence of future waypoints -- object-level, attention-explainable
  planning instead of dense BEV-grid convolutional planning.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import AutoConfig, AutoModel


def _box_muller(u: Tensor) -> Tensor:
    """Transform uniform samples in ``(0, 1)^2`` (last dim) to standard normal.

    Parameters
    ----------
    u:
        Uniform samples, shape ``(..., 2)``.

    Returns
    -------
    Tensor
        Standard-normal samples of the same shape, produced via the
        Box-Muller transform used by the reference NPSN implementation.
    """
    u1 = u[..., 0].clamp(min=1e-4, max=1 - 1e-4)
    u2 = u[..., 1]
    r = torch.sqrt(-2.0 * torch.log(u1))
    theta = 2.0 * math.pi * u2
    return torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=-1)


# ---------------------------------------------------------------------------
# NPSN
# ---------------------------------------------------------------------------


class _SceneGAT(nn.Module):
    """Single-layer multi-head graph attention over agents in a scene."""

    def __init__(self, in_feat: int, out_feat: int, n_head: int = 4) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.randn(n_head, in_feat, out_feat) * 0.1)
        self.a_src = nn.Parameter(torch.randn(n_head, out_feat, 1) * 0.1)
        self.a_dst = nn.Parameter(torch.randn(n_head, out_feat, 1) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_feat))
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, h: Tensor) -> Tensor:
        """Attend agent node features over the (fully-connected) scene graph.

        Parameters
        ----------
        h:
            Agent node features, shape ``(batch, n_agents, in_feat)``.

        Returns
        -------
        Tensor
            Updated node features, shape ``(batch, n_agents, out_feat)``.
        """
        h_prime = h.unsqueeze(1) @ self.w  # (batch, head, n_agents, out_feat)
        attn_src = h_prime @ self.a_src  # (batch, head, n_agents, 1)
        attn_dst = h_prime @ self.a_dst
        attn = self.leaky_relu(attn_src @ attn_dst.transpose(-1, -2))
        attn = F.softmax(attn, dim=-1)
        out = (attn @ h_prime).sum(dim=1) + self.bias
        return out + h_prime.sum(dim=1)


class NPSN(nn.Module):
    """Non-Probability Sampling Network: a learned purposive-sampling head.

    Wraps a toy per-agent Gaussian predictor (mean + Cholesky factor of a
    diagonal covariance) with a small GAT + MLP that regresses ``n``
    purposive sample locations, replacing random Monte-Carlo sampling.
    """

    def __init__(self, obs_len: int = 8, n_samples: int = 20, hidden: int = 32) -> None:
        super().__init__()
        self.obs_len = obs_len
        self.n_samples = n_samples
        in_feat = obs_len * 2
        self.gat = _SceneGAT(in_feat, hidden)
        self.head = nn.Sequential(
            nn.ReLU(), nn.Linear(hidden, 16), nn.ReLU(), nn.Linear(16, n_samples * 2)
        )
        self.mu_head = nn.Linear(hidden, 2)
        self.logstd_head = nn.Linear(hidden, 2)

    def forward(self, obs_traj: Tensor) -> Tensor:
        """Produce purposive endpoint samples for each agent in the scene.

        Parameters
        ----------
        obs_traj:
            Observed per-agent trajectories, shape ``(batch, n_agents, obs_len, 2)``.

        Returns
        -------
        Tensor
            Purposive endpoint samples, shape
            ``(batch, n_agents, n_samples, 2)``.
        """
        batch, n_agents, _, _ = obs_traj.shape
        node = obs_traj.reshape(batch, n_agents, -1)
        node = self.gat(node)

        loc = self.head(node).view(batch, n_agents, self.n_samples, 2)
        loc = loc.sigmoid().clamp(min=0.01, max=0.99)
        z = _box_muller(loc)  # purposive standard-normal draws

        mu = self.mu_head(node).unsqueeze(2)
        std = self.logstd_head(node).exp().unsqueeze(2)
        samples = mu + std * z
        return samples


def build_npsn() -> nn.Module:
    """Build a compact NPSN purposive-sampling module.

    Returns
    -------
    nn.Module
        Random-initialized ``NPSN`` in eval mode.
    """
    return NPSN().eval()


def example_input_npsn() -> Tensor:
    """Create an example batch of observed trajectories.

    Returns
    -------
    Tensor
        Observed trajectories, shape ``(2, 4, 8, 2)``.
    """
    return torch.randn(2, 4, 8, 2)


# ---------------------------------------------------------------------------
# OccWorld
# ---------------------------------------------------------------------------


class _OccVQVAE(nn.Module):
    """Compact 3D-occupancy vector-quantized autoencoder (scene tokenizer)."""

    def __init__(self, channels: int = 8, embed_dim: int = 16, n_codes: int = 32) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(channels, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, embed_dim, 3, padding=1),
        )
        self.codebook = nn.Parameter(torch.randn(n_codes, embed_dim) * 0.1)
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(embed_dim, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, channels, 3, padding=1),
        )

    def encode(self, occ: Tensor) -> tuple[Tensor, Tensor]:
        """Encode occupancy voxels to quantized tokens and their embeddings.

        Parameters
        ----------
        occ:
            Occupancy voxel grid, shape ``(batch, channels, D, H, W)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Token indices ``(batch, D', H', W')`` and quantized embeddings
            ``(batch, embed_dim, D', H', W')``.
        """
        z = self.encoder(occ)
        b, c, d, h, w = z.shape
        flat = z.permute(0, 2, 3, 4, 1).reshape(-1, c)
        dist = (flat.unsqueeze(1) - self.codebook.unsqueeze(0)).pow(2).sum(-1)
        idx = dist.argmin(dim=1)
        quantized = self.codebook[idx].view(b, d, h, w, c).permute(0, 4, 1, 2, 3)
        quantized = z + (quantized - z).detach()
        return idx.view(b, d, h, w), quantized

    def decode(self, quantized: Tensor) -> Tensor:
        """Decode quantized embeddings back to occupancy logits."""
        return self.decoder(quantized)


class OccWorld(nn.Module):
    """GPT-style spatiotemporal world model over discretized 3D occupancy."""

    def __init__(
        self,
        channels: int = 8,
        embed_dim: int = 16,
        n_codes: int = 32,
        n_frames: int = 3,
        n_layer: int = 2,
    ) -> None:
        super().__init__()
        self.tokenizer = _OccVQVAE(channels, embed_dim, n_codes)
        self.ego_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.temporal_pos = nn.Parameter(torch.randn(1, n_frames, 1, embed_dim) * 0.02)
        self.spatial_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(embed_dim, 2, embed_dim * 2, batch_first=True)
                for _ in range(n_layer)
            ]
        )
        self.temporal_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(embed_dim, 2, embed_dim * 2, batch_first=True)
                for _ in range(n_layer)
            ]
        )
        self.scene_head = nn.Linear(embed_dim, embed_dim)
        self.ego_head = nn.Linear(embed_dim, 3)

    def forward(self, occ_seq: Tensor) -> tuple[Tensor, Tensor]:
        """Autoregressively evolve tokenized occupancy + ego motion.

        Parameters
        ----------
        occ_seq:
            Occupancy voxel sequence, shape ``(batch, T, channels, D, H, W)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Predicted next-frame scene-token embeddings
            ``(batch, D'*H'*W', embed_dim)`` and predicted ego motion
            ``(batch, 3)``.
        """
        batch, t, c, d, h, w = occ_seq.shape
        frame_tokens = []
        for step in range(t):
            _, quantized = self.tokenizer.encode(occ_seq[:, step])
            bq, cq, dq, hq, wq = quantized.shape
            tokens = quantized.permute(0, 2, 3, 4, 1).reshape(bq, dq * hq * wq, cq)
            ego = self.ego_token.expand(bq, -1, -1)
            tokens = torch.cat([ego, tokens], dim=1)
            for layer in self.spatial_layers:
                tokens = layer(tokens)
            frame_tokens.append(tokens)

        stacked = torch.stack(frame_tokens, dim=1)  # (batch, T, 1+N, embed_dim)
        stacked = stacked + self.temporal_pos
        bq, tq, nq, cq = stacked.shape
        temporal_in = stacked.permute(0, 2, 1, 3).reshape(bq * nq, tq, cq)
        causal_mask = torch.triu(torch.full((tq, tq), float("-inf")), diagonal=1)
        for layer in self.temporal_layers:
            temporal_in = layer(temporal_in, src_mask=causal_mask)
        temporal_out = temporal_in.view(bq, nq, tq, cq).permute(0, 2, 1, 3)

        last = temporal_out[:, -1]
        ego_pred = self.ego_head(last[:, 0])
        scene_pred = self.scene_head(last[:, 1:])
        return scene_pred, ego_pred


def build_occworld() -> nn.Module:
    """Build a compact OccWorld tokenizer + spatiotemporal transformer.

    Returns
    -------
    nn.Module
        Random-initialized ``OccWorld`` in eval mode.
    """
    return OccWorld().eval()


def example_input_occworld() -> Tensor:
    """Create an example short sequence of occupancy voxel grids.

    Returns
    -------
    Tensor
        Occupancy sequence, shape ``(1, 3, 8, 8, 8, 8)``.
    """
    return torch.randn(1, 3, 8, 8, 8, 8)


# ---------------------------------------------------------------------------
# Panacea
# ---------------------------------------------------------------------------


class _FourDAttentionBlock(nn.Module):
    """Spatial self-attn + temporal attn + cross-view attn UNet block."""

    def __init__(self, dim: int, heads: int = 2) -> None:
        super().__init__()
        self.spatial_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.temporal_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.view_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        """Apply spatial, temporal, then cross-view attention.

        Parameters
        ----------
        x:
            Latent features, shape ``(batch, n_views, n_frames, hw, dim)``.

        Returns
        -------
        Tensor
            Updated latent features, same shape as ``x``.
        """
        b, v, t, hw, c = x.shape

        spatial_in = x.reshape(b * v * t, hw, c)
        attn_out, _ = self.spatial_attn(spatial_in, spatial_in, spatial_in)
        x = x + self.norm1(attn_out).view(b, v, t, hw, c)

        temporal_in = x.permute(0, 1, 3, 2, 4).reshape(b * v * hw, t, c)
        attn_out, _ = self.temporal_attn(temporal_in, temporal_in, temporal_in)
        temporal_out = self.norm2(attn_out).view(b, v, hw, t, c).permute(0, 1, 3, 2, 4)
        x = x + temporal_out

        view_in = x.permute(0, 2, 3, 1, 4).reshape(b * t * hw, v, c)
        attn_out, _ = self.view_attn(view_in, view_in, view_in)
        view_out = self.norm3(attn_out).view(b, t, hw, v, c).permute(0, 3, 1, 2, 4)
        x = x + view_out
        return x


class Panacea(nn.Module):
    """Panoramic multi-view video-diffusion denoiser with BEV control.

    A single denoising step of a small 4D-attention UNet block, with an
    additive ControlNet-style branch that injects a rendered BEV layout.
    """

    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.patch_embed = nn.Conv2d(3, dim, 4, stride=4)
        self.bev_branch = nn.Conv2d(3, dim, 4, stride=4)
        self.time_embed = nn.Embedding(1000, dim)
        self.block = _FourDAttentionBlock(dim)
        self.out_proj = nn.Conv2d(dim, 3, 3, padding=1)

    def forward(self, noisy_views: Tensor, bev_layout: Tensor, timestep: Tensor) -> Tensor:
        """Denoise multi-view video latents conditioned on BEV layout.

        Parameters
        ----------
        noisy_views:
            Noisy multi-view video, shape ``(batch, n_views, n_frames, 3, H, W)``.
        bev_layout:
            Rendered BEV control layout, shape ``(batch, 3, H, W)``.
        timestep:
            Diffusion timestep indices, shape ``(batch,)``.

        Returns
        -------
        Tensor
            Denoised patch-latent residual, shape
            ``(batch, n_views, n_frames, 3, H // 4, W // 4)`` (patchified
            latent space, as is standard for a diffusion UNet).
        """
        b, v, t, c, h, w = noisy_views.shape
        flat = noisy_views.reshape(b * v * t, c, h, w)
        feat = self.patch_embed(flat)
        _, dim, ph, pw = feat.shape
        feat = feat.view(b, v, t, dim, ph * pw).permute(0, 1, 2, 4, 3)

        bev_feat = self.bev_branch(bev_layout).view(b, dim, ph * pw).permute(0, 2, 1)
        feat = feat + bev_feat.unsqueeze(1).unsqueeze(1)

        time_feat = self.time_embed(timestep).view(b, 1, 1, 1, dim)
        feat = feat + time_feat

        feat = self.block(feat)

        feat = feat.permute(0, 1, 2, 4, 3).reshape(b * v * t, dim, ph, pw)
        out = self.out_proj(feat)
        return out.view(b, v, t, c, ph, pw)


def build_panacea() -> nn.Module:
    """Build a compact Panacea multi-view video-diffusion denoiser.

    Returns
    -------
    nn.Module
        Random-initialized ``Panacea`` in eval mode.
    """
    return Panacea().eval()


def example_input_panacea() -> tuple[Tensor, Tensor, Tensor]:
    """Create example noisy multi-view video, BEV layout and timestep.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        Noisy views ``(1, 2, 2, 3, 16, 16)``, BEV layout ``(1, 3, 16, 16)``,
        and timestep indices ``(1,)``.
    """
    noisy_views = torch.randn(1, 2, 2, 3, 16, 16)
    bev_layout = torch.randn(1, 3, 16, 16)
    timestep = torch.randint(0, 1000, (1,))
    return noisy_views, bev_layout, timestep


# ---------------------------------------------------------------------------
# PECNet
# ---------------------------------------------------------------------------


def _mlp(dims: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 2):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*layers)


class PECNet(nn.Module):
    """Endpoint-conditioned CVAE trajectory predictor with non-local social pooling."""

    def __init__(
        self,
        past_len: int = 8,
        future_len: int = 12,
        fdim: int = 16,
        zdim: int = 8,
        nonlocal_pools: int = 2,
        sigma: float = 1.3,
    ) -> None:
        super().__init__()
        self.zdim = zdim
        self.sigma = sigma
        self.nonlocal_pools = nonlocal_pools
        self.future_len = future_len

        self.encoder_past = _mlp([past_len * 2, 32, fdim])
        self.encoder_dest = _mlp([2, 16, fdim])
        self.decoder = _mlp([fdim + zdim, 32, 2])

        pred_in = 2 * fdim + 2
        self.non_local_theta = _mlp([pred_in, 16, 8])
        self.non_local_phi = _mlp([pred_in, 16, 8])
        self.non_local_g = _mlp([pred_in, 16, pred_in])
        self.predictor = _mlp([pred_in, 32, 2 * (future_len - 1)])

    def _non_local_social_pooling(self, feat: Tensor, mask: Tensor) -> Tensor:
        theta_x = self.non_local_theta(feat)
        phi_x = self.non_local_phi(feat).transpose(-1, -2)
        f = theta_x @ phi_x
        f_weights = F.softmax(f, dim=-1) * mask
        f_weights = F.normalize(f_weights, p=1, dim=-1)
        pooled = f_weights @ self.non_local_g(feat)
        return pooled + feat

    def forward(self, obs_traj: Tensor, initial_pos: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """Predict a trajectory endpoint and interior waypoints (inference mode).

        Parameters
        ----------
        obs_traj:
            Observed past trajectory, shape ``(n_agents, past_len, 2)``.
        initial_pos:
            Current per-agent position, shape ``(n_agents, 2)``.
        mask:
            Same-scene neighbor mask, shape ``(n_agents, n_agents)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Predicted destination ``(n_agents, 2)`` and interior waypoints
            ``(n_agents, future_len - 1, 2)``.
        """
        n_agents = obs_traj.size(0)
        ftraj = self.encoder_past(obs_traj.reshape(n_agents, -1))

        z = torch.randn(n_agents, self.zdim, device=obs_traj.device) * self.sigma
        generated_dest = self.decoder(torch.cat([ftraj, z], dim=1))

        dest_feat = self.encoder_dest(generated_dest)
        pred_feat = torch.cat([ftraj, dest_feat, initial_pos], dim=1)
        for _ in range(self.nonlocal_pools):
            pred_feat = self._non_local_social_pooling(pred_feat, mask)

        interior = self.predictor(pred_feat).view(n_agents, self.future_len - 1, 2)
        return generated_dest, interior


def build_pecnet() -> nn.Module:
    """Build a compact PECNet endpoint-conditioned trajectory predictor.

    Returns
    -------
    nn.Module
        Random-initialized ``PECNet`` in eval mode.
    """
    return PECNet().eval()


def example_input_pecnet() -> tuple[Tensor, Tensor, Tensor]:
    """Create example past trajectories, initial positions and a scene mask.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        Past trajectory ``(5, 8, 2)``, initial position ``(5, 2)``, and a
        fully-connected same-scene mask ``(5, 5)``.
    """
    obs_traj = torch.randn(5, 8, 2)
    initial_pos = torch.randn(5, 2)
    mask = torch.ones(5, 5)
    return obs_traj, initial_pos, mask


# ---------------------------------------------------------------------------
# PilotNet
# ---------------------------------------------------------------------------


class PilotNet(nn.Module):
    """NVIDIA end-to-end steering CNN (5 conv layers + 4 FC layers)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(24, 36, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(36, 48, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(48, 64, 3),
            nn.ELU(),
            nn.Conv2d(64, 64, 3),
            nn.ELU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 1 * 18, 1164),
            nn.ELU(),
            nn.Linear(1164, 100),
            nn.ELU(),
            nn.Linear(100, 50),
            nn.ELU(),
            nn.Linear(50, 10),
            nn.ELU(),
            nn.Linear(10, 1),
        )

    def forward(self, image: Tensor) -> Tensor:
        """Map a front-facing-camera image directly to a steering angle.

        Parameters
        ----------
        image:
            Normalized YUV/RGB image, shape ``(batch, 3, 66, 200)``.

        Returns
        -------
        Tensor
            Predicted steering angle, shape ``(batch, 1)``.
        """
        feat = self.conv(image)
        feat = feat.flatten(1)
        return self.fc(feat)


def build_pilotnet() -> nn.Module:
    """Build the PilotNet end-to-end steering CNN.

    Returns
    -------
    nn.Module
        Random-initialized ``PilotNet`` in eval mode.
    """
    return PilotNet().eval()


def example_input_pilotnet() -> Tensor:
    """Create an example front-facing-camera image batch.

    Returns
    -------
    Tensor
        Image batch, shape ``(1, 3, 66, 200)``.
    """
    return torch.randn(1, 3, 66, 200)


# ---------------------------------------------------------------------------
# PlanT
# ---------------------------------------------------------------------------


class PlanT(nn.Module):
    """Object-level planning transformer with GRU waypoint decoding."""

    def __init__(self, n_attr: int = 6, n_embd: int = 32, n_waypoints: int = 4) -> None:
        super().__init__()
        self.n_waypoints = n_waypoints
        config = AutoConfig.for_model(
            "gpt2", n_embd=n_embd, n_layer=2, n_head=2, n_positions=32, vocab_size=10
        )
        self.transformer = AutoModel.from_config(config)

        self.cls_emb = nn.Parameter(torch.randn(1, 1, n_attr))
        # Distinct linear embeddings per object type: vehicle, route point.
        self.obj_emb = nn.ModuleList([nn.Linear(n_attr, n_embd) for _ in range(2)])
        self.cls_proj = nn.Linear(n_attr, n_embd)

        self.wp_head = nn.Linear(n_embd, n_embd)
        self.wp_decoder = nn.GRUCell(input_size=2, hidden_size=n_embd)
        self.wp_output = nn.Linear(n_embd, 2)

    def forward(self, vehicles: Tensor, route: Tensor) -> Tensor:
        """Encode object tokens and autoregressively decode future waypoints.

        Parameters
        ----------
        vehicles:
            Per-vehicle attribute tokens (x, y, yaw, speed, extent-x,
            extent-y), shape ``(batch, n_vehicles, 6)``.
        route:
            Per-route-point attribute tokens (same 6 attributes, with
            speed/yaw unused slots kept for a uniform embedding), shape
            ``(batch, n_route, 6)``.

        Returns
        -------
        Tensor
            Predicted future waypoints, shape ``(batch, n_waypoints, 2)``.
        """
        batch = vehicles.size(0)
        cls = self.cls_proj(self.cls_emb.expand(batch, -1, -1))
        veh_tok = self.obj_emb[0](vehicles)
        route_tok = self.obj_emb[1](route)
        tokens = torch.cat([cls, veh_tok, route_tok], dim=1)

        hidden = self.transformer(inputs_embeds=tokens).last_hidden_state
        cls_out = self.wp_head(hidden[:, 0])

        wp_input = torch.zeros(batch, 2, device=vehicles.device)
        h = cls_out
        waypoints = []
        for _ in range(self.n_waypoints):
            h = self.wp_decoder(wp_input, h)
            wp_input = self.wp_output(h)
            waypoints.append(wp_input)
        return torch.stack(waypoints, dim=1)


def build_plant() -> nn.Module:
    """Build a compact PlanT object-level planning transformer.

    Returns
    -------
    nn.Module
        Random-initialized ``PlanT`` in eval mode.
    """
    return PlanT().eval()


def example_input_plant() -> tuple[Tensor, Tensor]:
    """Create example vehicle and route object tokens.

    Returns
    -------
    tuple[Tensor, Tensor]
        Vehicle tokens ``(1, 4, 6)`` and route tokens ``(1, 3, 6)``.
    """
    vehicles = torch.randn(1, 4, 6)
    route = torch.randn(1, 3, 6)
    return vehicles, route


MENAGERIE_ENTRIES = [
    ("NPSN", "build_npsn", "example_input_npsn", "2022", "SEQ"),
    ("OccWorld", "build_occworld", "example_input_occworld", "2024", "VIS"),
    ("Panacea", "build_panacea", "example_input_panacea", "2024", "GEN"),
    ("PECNet", "build_pecnet", "example_input_pecnet", "2020", "SEQ"),
    ("PilotNet", "build_pilotnet", "example_input_pilotnet", "2016", "VIS"),
    ("PlanT", "build_plant", "example_input_plant", "2022", "SEQ"),
]
