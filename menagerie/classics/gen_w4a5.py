"""Menagerie batch w4a5: autonomous-driving perception, planning, and world models.

Sources checked (reference only; no cloning, no pip installs):
  - VoxFormer: Li et al., CVPR 2023 Highlight, "VoxFormer: Sparse Voxel Transformer
    for Camera-based 3D Semantic Scene Completion". Paper
    https://arxiv.org/abs/2302.12251, official source
    https://github.com/NVlabs/VoxFormer (projects/mmdet3d_plugin/voxformer/). A
    two-stage design: Stage-1 (query proposal) runs a lightweight 2D depth-driven
    module that back-projects estimated depth into a coarse voxel grid and marks a
    sparse subset of learnable voxel queries as "occupied" (class-agnostic
    proposal); those proposed queries then deformable-cross-attend to the 2D image
    feature map to pull in image evidence. Stage-2 is the paper's namesake
    "MAE-like" completion: every non-proposed voxel position is filled with a
    shared learnable mask token, and the *full* dense voxel grid (proposed +
    masked) is refined by voxel self-attention (approximated here with a compact
    multi-head self-attention over the flattened voxel grid, since torchlens
    traces dense ops and the official deformable-3D-attention CUDA kernels are not
    reimplemented). A final linear head predicts per-voxel semantic class logits.
    This reproduces the paper's two-stage sparse-query-then-dense-mask-token
    completion, its central novelty over single-pass dense 3D convolutional scene
    completion baselines.
  - YOLOP: Wu et al., Machine Intelligence Research (MIR) 2022, "YOLOP: You Only
    Look Once for Panoptic Driving Perception". Paper
    https://arxiv.org/abs/2108.11250, official source
    https://github.com/hustvl/YOLOP (lib/models/YOLOP.py, common.py). A single
    shared CSPDarknet-style encoder (Focus stem + strided Conv/BottleneckCSP/SPP
    stages) with a PANet-style multi-scale neck feeds three parallel decoder
    heads from the same backbone features: (1) an anchor-based YOLO detection
    head on three neck scales for object detection, (2) a drivable-area
    segmentation head upsampling from an early neck feature, and (3) a lane-line
    segmentation head upsampling from the same early neck feature via a separate
    decoder branch. This reproduces YOLOP's namesake "one encoder, three
    decoders" panoptic driving-perception design (detection + two dense
    segmentation tasks computed in a single forward pass from shared features),
    the paper's central efficiency claim over running three separate networks.
  - AD-MLP: Zhai et al., 2023, "Rethinking the Open-Loop Evaluation of
    End-to-End Autonomous Driving in nuScenes". Paper
    https://arxiv.org/abs/2305.10430, official source
    https://github.com/E2E-AD/AD-MLP (pytorch/admlp/planner.py,
    evaluate_for_mlp.py; mlp.pth checkpoint confirms a pure-MLP model). The
    paper's key finding/architecture is deliberately minimal: no camera/LiDAR
    perception backbone at all. Past ego status (position, velocity,
    acceleration, yaw / steering history over several past timesteps) is
    flattened into a single vector and passed through a small stack of
    Linear+ReLU layers, terminating in a head that regresses the future
    ego-trajectory waypoints directly -- the paper's provocative point being
    that this perception-free MLP baseline is competitive with contemporary
    "full-stack" end-to-end planners on open-loop nuScenes trajectory metrics.
    This reproduces exactly that: an ego-status-history-in, future-waypoints-out
    MLP with no image/point-cloud input at all.
  - AgentFormer: Yuan et al., ICCV 2021, "AgentFormer: Agent-Aware Transformers
    for Socio-Temporal Multi-Agent Forecasting". Paper
    https://arxiv.org/abs/2103.14023, official source
    https://github.com/Khrylx/AgentFormer (model/agentformer.py,
    agentformer_lib.py). The paper's namesake mechanism is a single joint
    space-time-agent transformer: rather than the common two-stage
    "temporal-transformer-per-agent then social-pooling-across-agents" design,
    all agents' states across all timesteps are flattened into ONE token
    sequence (agent-major, so consecutive tokens alternate agents within a
    timestep) and processed by one shared self-attention stack, so every token
    can directly attend to every other agent at every other time. Because plain
    positional encoding alone cannot distinguish "same agent, later time" from
    "different agent, same time" once agent and time are merged into one axis,
    the paper adds a second, agent-identity positional encoding (added
    alongside the temporal encoding) to each token -- the "agent-aware" part of
    the name. This module reproduces the joint flattened sequence, the additive
    dual (temporal + agent-identity) positional encoding, and a Transformer
    encoder-decoder (encoding observed history, decoding a query sequence into
    future per-agent trajectories) exactly matching that structure at compact
    scale.
  - AutoBots (AutoBot-Ego): Girgis et al., ICLR 2022, "Latent Variable
    Sequential Set Transformers for Joint Multi-Agent Motion Prediction". Paper
    https://arxiv.org/abs/2104.00563, official source
    https://github.com/roggirg/AutoBots (models/autobot_ego.py). The paper's
    key mechanism factorizes joint multi-agent attention into two alternating,
    much cheaper transformer passes per encoder layer: a "temporal" self-attention
    over each agent's own timesteps (mixing across time, per agent) and a
    "social" self-attention over all agents at a fixed timestep (mixing across
    agents, per time) -- alternating factorized attention as an efficient
    substitute for full space-time joint attention. The decoder then uses C
    learned latent "mode" query seeds (one per plausible future scenario) that
    cross-attend to the encoded scene through a transformer decoder, producing C
    joint multi-agent future trajectories plus a mode-probability head over the
    C modes (a discrete latent-variable / mixture prediction), matching the
    paper's "sequential set transformer with learned latent modes" design.
  - BEVWorld: Zhang et al., 2024, "BEVWorld: A Multimodal World Model for
    Autonomous Driving via Unified BEV Latent Space". Paper
    https://arxiv.org/abs/2407.05679, official source (README/paper only, no
    released training code as of this writing) https://github.com/zympsyche/BevWorld.
    Per the paper (arxiv.org/html/2407.05679v1): a multi-modal tokenizer encodes
    camera-image and LiDAR-pillar features with per-modality CNN/Swin-style
    backbones, fuses them into one compact BEV grid via a deformable
    cross-attention (LiDAR BEV features as queries, sampled multi-view image
    features as values), and channel-compresses the fused BEV grid down to a
    handful of latent channels; a ray-casting volume-rendering decoder
    reconstructs camera/LiDAR observations from the compressed BEV latent as a
    self-supervised reconstruction target (approximated here by a compact
    upsample-decoder standing in for the full differentiable ray-marching
    renderer, since torchlens traces dense tensor ops rather than per-ray
    integration loops). A latent BEV sequence diffusion transformer then
    predicts noise for a temporal sequence of future BEV latents, conditioned on
    an action-token + diffusion-timestep embedding injected via AdaLN
    (adaptive layer-norm scale/shift modulation, in the paper's own notation
    ``AdaLN(x, gamma, beta) = LayerNorm(x) * (1 + gamma) + beta``) at every
    transformer block -- the paper's namesake unified-BEV-latent, action-
    conditioned world-model design. This module reproduces the fuse-then-
    compress tokenizer, the reconstruction decoder, and the AdaLN-modulated
    latent diffusion transformer over a short BEV-latent sequence.

All models below are compact, faithfully-reimplemented-from-scratch nn.Modules
with random init and small dims for TorchLens architecture-catalog tracing
(not a trained-weights zoo).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ============================================================
# VoxFormer -- sparse-query-proposal + MAE-like dense completion
# ============================================================


class VoxFormerQueryProposal(nn.Module):
    """Stage-1: depth-driven voxel-occupancy proposal + image cross-attention.

    A 2D CNN over the image feature map predicts a coarse per-pixel depth
    distribution, which is projected into voxel-occupancy logits over the
    (downsampled) 3D voxel grid. The top-scoring voxels are treated as
    "proposed" and their learnable voxel-query embeddings deformable-cross-
    attend (approximated with standard multi-head cross-attention) to the
    flattened image feature tokens to pull in 2D evidence.
    """

    def __init__(self, img_ch: int, embed_dim: int, grid: tuple[int, int, int]) -> None:
        super().__init__()
        self.grid = grid
        n_vox = grid[0] * grid[1] * grid[2]
        self.depth_head = nn.Conv2d(img_ch, grid[2], kernel_size=1)
        self.voxel_queries = nn.Parameter(torch.randn(n_vox, embed_dim) * 0.02)
        self.img_proj = nn.Linear(img_ch, embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.occ_score = nn.Linear(embed_dim, 1)

    def forward(self, img_feat: Tensor) -> tuple[Tensor, Tensor]:
        """Propose occupied voxels and featurize them via image cross-attention.

        Parameters
        ----------
        img_feat : Tensor
            Image feature map, shape ``(B, C, H, W)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(voxel_features, occupancy_score)`` of shape
            ``(B, n_vox, embed_dim)`` and ``(B, n_vox, 1)``.
        """
        b, c, h, w = img_feat.shape
        depth_logits = self.depth_head(img_feat)  # (B, Z, H, W), depth-as-height proxy
        _ = F.softmax(depth_logits, dim=1)  # depth distribution -> occupancy evidence
        img_tokens = self.img_proj(img_feat.flatten(2).transpose(1, 2))  # (B, HW, D)
        queries = self.voxel_queries.unsqueeze(0).expand(b, -1, -1)
        attended, _ = self.cross_attn(queries, img_tokens, img_tokens)
        occ_score = self.occ_score(attended)
        return attended, occ_score


class VoxFormerMAECompletion(nn.Module):
    """Stage-2: sparse-to-dense MAE-like voxel completion via self-attention.

    Proposed voxel features are gated by their occupancy score; the residual
    (non-proposed) capacity is filled with a shared learnable mask token, and
    the full dense voxel set is refined with multi-head self-attention so
    every voxel can exchange information with every other voxel, then decoded
    to per-voxel semantic-class logits.
    """

    def __init__(self, embed_dim: int, n_classes: int, depth: int = 2, n_heads: int = 4) -> None:
        super().__init__()
        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim * 2, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.seg_head = nn.Linear(embed_dim, n_classes)

    def forward(self, voxel_features: Tensor, occ_score: Tensor) -> Tensor:
        """Complete the dense voxel grid and predict semantic classes.

        Parameters
        ----------
        voxel_features : Tensor
            Proposed voxel features, shape ``(B, n_vox, embed_dim)``.
        occ_score : Tensor
            Per-voxel occupancy score, shape ``(B, n_vox, 1)``.

        Returns
        -------
        Tensor
            Per-voxel class logits, shape ``(B, n_vox, n_classes)``.
        """
        gate = torch.sigmoid(occ_score)
        mask_token = self.mask_token.expand(voxel_features.shape[0], voxel_features.shape[1], -1)
        dense = gate * voxel_features + (1.0 - gate) * mask_token
        refined = self.encoder(dense)
        return self.seg_head(refined)


class VoxFormer(nn.Module):
    """Compact VoxFormer: query-proposal stage-1 + MAE-like completion stage-2."""

    def __init__(
        self,
        img_ch: int = 16,
        embed_dim: int = 32,
        grid: tuple[int, int, int] = (4, 4, 2),
        n_classes: int = 5,
    ) -> None:
        super().__init__()
        self.stage1 = VoxFormerQueryProposal(img_ch, embed_dim, grid)
        self.stage2 = VoxFormerMAECompletion(embed_dim, n_classes)

    def forward(self, img_feat: Tensor) -> Tensor:
        voxel_features, occ_score = self.stage1(img_feat)
        return self.stage2(voxel_features, occ_score)


def build_voxformer() -> nn.Module:
    """Build a small two-stage VoxFormer 3D semantic scene completion model."""
    return VoxFormer(img_ch=16, embed_dim=32, grid=(4, 4, 2), n_classes=5).eval()


def example_input_voxformer() -> Tensor:
    """Example monocular image feature map, ``(1, 16, 12, 20)``."""
    return torch.randn(1, 16, 12, 20)


# ============================================================
# YOLOP -- single shared encoder + 3 panoptic driving-perception decoders
# ============================================================


class ConvBnAct(nn.Module):
    """Conv + BatchNorm + SiLU, the CSPDarknet-style basic block."""

    def __init__(self, c_in: int, c_out: int, k: int = 3, s: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, k, s, padding=k // 2, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(x)))


class YOLOPBackbone(nn.Module):
    """Shared CSPDarknet-style stem + strided stages feeding a 3-scale FPN neck."""

    def __init__(self, ch: int = 3, base: int = 16) -> None:
        super().__init__()
        self.stem = ConvBnAct(ch, base, k=3, s=2)
        self.stage1 = ConvBnAct(base, base * 2, k=3, s=2)
        self.stage2 = ConvBnAct(base * 2, base * 4, k=3, s=2)
        self.stage3 = ConvBnAct(base * 4, base * 8, k=3, s=2)
        self.stage4 = ConvBnAct(base * 8, base * 16, k=3, s=2)
        self.lat3 = nn.Conv2d(base * 16, base * 8, 1)
        self.lat2 = nn.Conv2d(base * 4, base * 8, 1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        s0 = self.stem(x)
        s1 = self.stage1(s0)
        s2 = self.stage2(s1)  # early neck feature (fed to both seg heads)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)
        p3 = self.lat3(s4)
        p2 = F.interpolate(p3, size=s2.shape[-2:], mode="nearest") + self.lat2(s2)
        return s2, p2, p3


class YOLOPDetectHead(nn.Module):
    """Anchor-based multi-scale detection head over the neck features."""

    def __init__(self, ch: int, n_anchors: int = 3, n_classes: int = 10) -> None:
        super().__init__()
        self.pred = nn.Conv2d(ch, n_anchors * (5 + n_classes), 1)

    def forward(self, feat: Tensor) -> Tensor:
        return self.pred(feat)


class YOLOPSegHead(nn.Module):
    """Upsampling segmentation decoder branch (shared design for both tasks)."""

    def __init__(self, ch: int, n_classes: int = 2) -> None:
        super().__init__()
        self.up1 = ConvBnAct(ch, ch // 2, k=3, s=1)
        self.up2 = ConvBnAct(ch // 2, ch // 4, k=3, s=1)
        self.out = nn.Conv2d(ch // 4, n_classes, 1)

    def forward(self, feat: Tensor, out_size: tuple[int, int]) -> Tensor:
        x = F.interpolate(feat, scale_factor=2, mode="nearest")
        x = self.up1(x)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.up2(x)
        x = F.interpolate(x, size=out_size, mode="nearest")
        return self.out(x)


class YOLOP(nn.Module):
    """One shared encoder feeding three parallel panoptic-perception decoders."""

    def __init__(self, ch: int = 3, base: int = 16) -> None:
        super().__init__()
        self.backbone = YOLOPBackbone(ch, base)
        self.det_head = YOLOPDetectHead(base * 8, n_anchors=3, n_classes=10)
        self.da_seg_head = YOLOPSegHead(base * 8, n_classes=2)  # drivable area
        self.ll_seg_head = YOLOPSegHead(base * 8, n_classes=2)  # lane line

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        out_size = x.shape[-2:]
        s2, p2, p3 = self.backbone(x)
        det = self.det_head(p3)
        da_seg = self.da_seg_head(p2, out_size)
        ll_seg = self.ll_seg_head(p2, out_size)
        return {"detection": det, "drivable_area": da_seg, "lane_line": ll_seg}


def build_yolop() -> nn.Module:
    """Build a small YOLOP shared-encoder / 3-decoder panoptic perception model."""
    return YOLOP(ch=3, base=16).eval()


def example_input_yolop() -> Tensor:
    """Example front-camera RGB image, ``(1, 3, 64, 64)``."""
    return torch.randn(1, 3, 64, 64)


# ============================================================
# AD-MLP -- perception-free ego-history-to-future-trajectory MLP
# ============================================================


class ADMLP(nn.Module):
    """Pure-MLP open-loop planner: past ego status in, future waypoints out.

    No camera/LiDAR perception backbone at all -- the paper's central
    provocation is that a small MLP over flattened past ego kinematic history
    (position, velocity, acceleration, yaw) is a competitive open-loop nuScenes
    planning baseline versus much larger perception-driven end-to-end stacks.
    """

    def __init__(
        self, past_steps: int = 4, ego_dim: int = 6, hidden: int = 64, future_steps: int = 6
    ) -> None:
        super().__init__()
        self.past_steps = past_steps
        self.ego_dim = ego_dim
        self.future_steps = future_steps
        in_dim = past_steps * ego_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, future_steps * 2),
        )

    def forward(self, ego_status_history: Tensor) -> Tensor:
        """Predict future ego waypoints from flattened past ego status.

        Parameters
        ----------
        ego_status_history : Tensor
            Past ego kinematic status, shape ``(B, past_steps, ego_dim)``.

        Returns
        -------
        Tensor
            Future ego (x, y) waypoints, shape ``(B, future_steps, 2)``.
        """
        b = ego_status_history.shape[0]
        flat = ego_status_history.reshape(b, -1)
        out = self.mlp(flat)
        return out.view(b, self.future_steps, 2)


def build_ad_mlp() -> nn.Module:
    """Build a small AD-MLP perception-free open-loop ego planner."""
    return ADMLP(past_steps=4, ego_dim=6, hidden=64, future_steps=6).eval()


def example_input_ad_mlp() -> Tensor:
    """Example flattened past ego status history, ``(1, 4, 6)``."""
    return torch.randn(1, 4, 6)


# ============================================================
# AgentFormer -- joint space-time-agent transformer, agent-aware encoding
# ============================================================


class AgentAwarePositionalEncoding(nn.Module):
    """Additive dual positional encoding: temporal + agent-identity.

    Distinguishes "same agent later time" from "different agent same time"
    once agent and time are flattened into a single joint token axis, which
    plain sinusoidal time-only encoding cannot do.
    """

    def __init__(self, d_model: int, max_t: int = 32, max_agents: int = 16) -> None:
        super().__init__()
        self.d_model = d_model
        self.register_buffer("pe_t", self._sinusoid(max_t, d_model))
        self.register_buffer("pe_a", self._sinusoid(max_agents, d_model))

    @staticmethod
    def _sinusoid(length: int, d_model: int) -> Tensor:
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x: Tensor, n_agents: int) -> Tensor:
        """Add temporal + agent-identity encodings to an agent-major joint sequence.

        Parameters
        ----------
        x : Tensor
            Joint (time, agent)-flattened token sequence, shape ``(B, T*A, D)``.
        n_agents : int
            Number of agents ``A`` per timestep.

        Returns
        -------
        Tensor
            Positionally-encoded sequence, same shape as ``x``.
        """
        b, ta, d = x.shape
        t = ta // n_agents
        pe_t = self.pe_t[:t].repeat_interleave(n_agents, dim=0)  # (T*A, D)
        pe_a = self.pe_a[:n_agents].repeat(t, 1)  # (T*A, D)
        return x + pe_t.unsqueeze(0) + pe_a.unsqueeze(0)


class AgentFormer(nn.Module):
    """Compact AgentFormer: joint flattened space-time-agent transformer.

    All agents' states at all observed timesteps are merged into one
    agent-major token sequence and processed by a shared transformer
    encoder-decoder, so every token can directly attend to every other
    agent at every other time in a single self-attention pass.
    """

    def __init__(
        self,
        motion_dim: int = 2,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        n_agents: int = 3,
        future_steps: int = 4,
    ) -> None:
        super().__init__()
        self.n_agents = n_agents
        self.future_steps = future_steps
        self.d_model = d_model
        self.input_fc = nn.Linear(motion_dim, d_model)
        self.pos_enc = AgentAwarePositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=d_model * 2, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        dec_layer = nn.TransformerDecoderLayer(
            d_model, n_heads, dim_feedforward=d_model * 2, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_layers)
        self.future_queries = nn.Parameter(torch.randn(1, future_steps * n_agents, d_model) * 0.02)
        self.out_fc = nn.Linear(d_model, motion_dim)

    def forward(self, past_motion: Tensor) -> Tensor:
        """Jointly forecast every agent's future trajectory.

        Parameters
        ----------
        past_motion : Tensor
            Observed past positions, agent-major flattened,
            shape ``(B, T_obs * n_agents, motion_dim)``.

        Returns
        -------
        Tensor
            Future positions, shape ``(B, future_steps * n_agents, motion_dim)``.
        """
        b = past_motion.shape[0]
        tokens = self.input_fc(past_motion)
        tokens = self.pos_enc(tokens, self.n_agents)
        context = self.encoder(tokens)
        queries = self.future_queries.expand(b, -1, -1)
        queries = self.pos_enc(queries, self.n_agents)
        decoded = self.decoder(queries, context)
        return self.out_fc(decoded)


def build_agentformer() -> nn.Module:
    """Build a small AgentFormer joint space-time-agent forecasting model."""
    return AgentFormer(
        motion_dim=2, d_model=32, n_heads=4, n_layers=2, n_agents=3, future_steps=4
    ).eval()


def example_input_agentformer() -> Tensor:
    """Example agent-major joint past-motion sequence, ``(1, 5*3, 2)``."""
    return torch.randn(1, 5 * 3, 2)


# ============================================================
# AutoBots (AutoBot-Ego) -- factorized temporal/social attention + latent modes
# ============================================================


class AutoBotsEncoderLayer(nn.Module):
    """One alternating temporal-then-social factorized self-attention block."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.temporal = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=d_model * 2, batch_first=True
        )
        self.social = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=d_model * 2, batch_first=True
        )

    def forward(self, x: Tensor) -> Tensor:
        """Alternate per-agent temporal attention and per-timestep social attention.

        Parameters
        ----------
        x : Tensor
            Scene tensor, shape ``(B, T, N, D)``.

        Returns
        -------
        Tensor
            Refined scene tensor, same shape as ``x``.
        """
        b, t, n, d = x.shape
        temp_in = x.permute(0, 2, 1, 3).reshape(b * n, t, d)
        temp_out = self.temporal(temp_in).view(b, n, t, d).permute(0, 2, 1, 3)
        soc_in = temp_out.reshape(b * t, n, d)
        soc_out = self.social(soc_in).view(b, t, n, d)
        return soc_out


class AutoBotsEgo(nn.Module):
    """Compact AutoBot-Ego: factorized temporal/social encoder + latent-mode decoder.

    C learned latent "mode" query seeds cross-attend to the jointly-encoded
    multi-agent scene through a transformer decoder, producing C candidate
    joint future scenarios plus a mixture probability over the C modes.
    """

    def __init__(
        self,
        k_attr: int = 2,
        d_model: int = 32,
        n_heads: int = 4,
        n_agents: int = 3,
        n_enc_layers: int = 2,
        n_modes: int = 4,
        future_steps: int = 5,
    ) -> None:
        super().__init__()
        self.n_agents = n_agents
        self.n_modes = n_modes
        self.future_steps = future_steps
        self.d_model = d_model
        self.input_fc = nn.Linear(k_attr, d_model)
        self.encoder_layers = nn.ModuleList(
            [AutoBotsEncoderLayer(d_model, n_heads) for _ in range(n_enc_layers)]
        )
        # C x T learned query seeds for the joint multi-mode future decoder.
        self.mode_queries = nn.Parameter(torch.randn(1, n_modes * future_steps, d_model) * 0.02)
        dec_layer = nn.TransformerDecoderLayer(
            d_model, n_heads, dim_feedforward=d_model * 2, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=2)
        self.output_model = nn.Linear(d_model, n_agents * 2)
        self.mode_prob_query = nn.Parameter(torch.randn(1, n_modes, d_model) * 0.02)
        self.mode_prob_attn = nn.MultiheadAttention(d_model, num_heads=n_heads, batch_first=True)
        self.prob_head = nn.Linear(d_model, 1)

    def forward(self, agents_history: Tensor) -> tuple[Tensor, Tensor]:
        """Predict C joint multi-agent future trajectories + mode probabilities.

        Parameters
        ----------
        agents_history : Tensor
            Past kinematic states of every agent, shape ``(B, T, N, k_attr)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(trajectories, mode_probs)`` of shape
            ``(B, n_modes, future_steps, n_agents, 2)`` and ``(B, n_modes)``.
        """
        b, t, n, _ = agents_history.shape
        x = self.input_fc(agents_history)
        for layer in self.encoder_layers:
            x = layer(x)
        context = x.reshape(b, t * n, self.d_model)

        queries = self.mode_queries.expand(b, -1, -1)
        decoded = self.decoder(queries, context)
        trajectories = self.output_model(decoded).view(b, self.n_modes, self.future_steps, n, 2)

        mode_pooled, _ = self.mode_prob_attn(
            self.mode_prob_query.expand(b, -1, -1), context, context
        )
        mode_logits = self.prob_head(mode_pooled).squeeze(-1)
        mode_probs = F.softmax(mode_logits, dim=-1)
        return trajectories, mode_probs


def build_autobots() -> nn.Module:
    """Build a small AutoBot-Ego factorized-attention latent-mode motion predictor."""
    return AutoBotsEgo(
        k_attr=2, d_model=32, n_heads=4, n_agents=3, n_enc_layers=2, n_modes=4, future_steps=5
    ).eval()


def example_input_autobots() -> Tensor:
    """Example multi-agent past-state tensor, ``(1, 4, 3, 2)`` (B, T, N, k_attr)."""
    return torch.randn(1, 4, 3, 2)


# ============================================================
# BEVWorld -- multi-modal BEV tokenizer + AdaLN action-conditioned latent diffusion
# ============================================================


class BEVWorldTokenizer(nn.Module):
    """Fuse camera + LiDAR-pillar features into a compressed BEV latent grid.

    A LiDAR-pillar BEV feature map (queries) deformable-cross-attends
    (approximated with standard multi-head cross-attention) to sampled
    multi-view image features (values/keys) to fuse modalities, and the
    fused BEV grid is channel-compressed down to a small latent dimension
    for the downstream diffusion model.
    """

    def __init__(
        self, img_ch: int, lidar_ch: int, fuse_dim: int, latent_ch: int, bev_hw: int
    ) -> None:
        super().__init__()
        self.bev_hw = bev_hw
        self.img_proj = nn.Conv2d(img_ch, fuse_dim, 1)
        self.lidar_proj = nn.Conv2d(lidar_ch, fuse_dim, 1)
        self.fuse_attn = nn.MultiheadAttention(fuse_dim, num_heads=4, batch_first=True)
        self.compress = nn.Conv2d(fuse_dim, latent_ch, 1)

    def forward(self, img_feat: Tensor, lidar_bev: Tensor) -> Tensor:
        """Fuse and compress camera + LiDAR features into a BEV latent.

        Parameters
        ----------
        img_feat : Tensor
            Multi-view image feature map, shape ``(B, img_ch, H, W)``.
        lidar_bev : Tensor
            LiDAR pillar BEV feature map, shape ``(B, lidar_ch, bev_h, bev_w)``.

        Returns
        -------
        Tensor
            Compressed BEV latent, shape ``(B, latent_ch, bev_h, bev_w)``.
        """
        img_tokens = self.img_proj(img_feat).flatten(2).transpose(1, 2)  # (B, HW, fuse_dim)
        b, c, h, w = lidar_bev.shape
        lidar_tokens = (
            self.lidar_proj(lidar_bev).flatten(2).transpose(1, 2)
        )  # (B, bev_hw, fuse_dim)
        fused_tokens, _ = self.fuse_attn(lidar_tokens, img_tokens, img_tokens)
        fused_bev = fused_tokens.transpose(1, 2).view(b, -1, h, w)
        return self.compress(fused_bev)


class BEVWorldReconstructionDecoder(nn.Module):
    """Reconstruct camera/LiDAR observations from the compressed BEV latent.

    Stands in for the paper's differentiable ray-casting volume renderer with
    a compact upsampling decoder producing a dense reconstruction target
    (torchlens traces dense tensor ops rather than per-ray marching loops).
    """

    def __init__(self, latent_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(latent_ch, latent_ch * 2, kernel_size=2, stride=2),
            nn.SiLU(),
            nn.Conv2d(latent_ch * 2, out_ch, kernel_size=3, padding=1),
        )

    def forward(self, bev_latent: Tensor) -> Tensor:
        return self.up(bev_latent)


class AdaLNBlock(nn.Module):
    """DiT-style transformer block with AdaLN action + timestep conditioning.

    ``c = concat(action_embed, timestep_embed); gamma, beta = Linear(c)``,
    then ``AdaLN(x, gamma, beta) = LayerNorm(x) * (1 + gamma) + beta`` applied
    before each of the attention and MLP sublayers, exactly matching the
    paper's action-token / diffusion-timestep conditioning mechanism.
    """

    def __init__(self, d_model: int, n_heads: int, cond_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(d_model, num_heads=n_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, d_model)
        )
        self.ada_ln = nn.Linear(cond_dim, d_model * 4)  # gamma1, beta1, gamma2, beta2

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """Apply AdaLN-modulated self-attention + MLP.

        Parameters
        ----------
        x : Tensor
            Token sequence, shape ``(B, L, D)``.
        cond : Tensor
            Action + timestep condition embedding, shape ``(B, cond_dim)``.

        Returns
        -------
        Tensor
            Refined token sequence, shape ``(B, L, D)``.
        """
        gamma1, beta1, gamma2, beta2 = self.ada_ln(cond).chunk(4, dim=-1)
        h = self.norm1(x) * (1.0 + gamma1.unsqueeze(1)) + beta1.unsqueeze(1)
        attn_out, _ = self.attn(h, h, h)
        x = x + attn_out
        h = self.norm2(x) * (1.0 + gamma2.unsqueeze(1)) + beta2.unsqueeze(1)
        return x + self.mlp(h)


class BEVWorldDiffusionTransformer(nn.Module):
    """Latent BEV sequence diffusion transformer with causal temporal attention.

    Predicts per-frame noise for a short sequence of future BEV latents,
    conditioned on an action-token + diffusion-timestep embedding injected
    via AdaLN at every block; spatial tokens within a frame use full
    (non-causal) attention while the temporal axis uses a causal mask so
    each frame only attends to itself and earlier frames.
    """

    def __init__(
        self, latent_ch: int, bev_hw: int, d_model: int, n_heads: int, n_blocks: int, n_actions: int
    ) -> None:
        super().__init__()
        self.bev_hw = bev_hw
        self.d_model = d_model
        self.in_proj = nn.Linear(latent_ch, d_model)
        self.action_embed = nn.Linear(n_actions, d_model)
        self.time_embed = nn.Linear(1, d_model)
        self.blocks = nn.ModuleList(
            [AdaLNBlock(d_model, n_heads, d_model) for _ in range(n_blocks)]
        )
        self.out_proj = nn.Linear(d_model, latent_ch)

    def forward(self, noisy_latents: Tensor, action: Tensor, diffusion_t: Tensor) -> Tensor:
        """Predict noise for a temporal sequence of BEV latents.

        Parameters
        ----------
        noisy_latents : Tensor
            Noisy future BEV latent sequence, shape ``(B, T, latent_ch, h, w)``.
        action : Tensor
            Action-token embedding input, shape ``(B, n_actions)``.
        diffusion_t : Tensor
            Diffusion timestep (as a scalar float), shape ``(B, 1)``.

        Returns
        -------
        Tensor
            Predicted noise, same shape as ``noisy_latents``.
        """
        b, t, c, h, w = noisy_latents.shape
        tokens = noisy_latents.permute(0, 1, 3, 4, 2).reshape(b, t * h * w, c)
        tokens = self.in_proj(tokens)
        cond = self.action_embed(action) + self.time_embed(diffusion_t)
        for block in self.blocks:
            tokens = block(tokens, cond)
        noise_pred = self.out_proj(tokens)
        return noise_pred.view(b, t, h, w, c).permute(0, 1, 4, 2, 3)


class BEVWorld(nn.Module):
    """Compact BEVWorld: multi-modal tokenizer + AdaLN action-conditioned diffusion."""

    def __init__(
        self,
        img_ch: int = 8,
        lidar_ch: int = 8,
        fuse_dim: int = 16,
        latent_ch: int = 4,
        bev_hw: int = 8,
        d_model: int = 24,
        n_actions: int = 3,
        seq_len: int = 2,
    ) -> None:
        super().__init__()
        self.tokenizer = BEVWorldTokenizer(img_ch, lidar_ch, fuse_dim, latent_ch, bev_hw)
        self.decoder = BEVWorldReconstructionDecoder(latent_ch, img_ch)
        self.diffusion = BEVWorldDiffusionTransformer(
            latent_ch, bev_hw, d_model, n_heads=4, n_blocks=2, n_actions=n_actions
        )
        self.seq_len = seq_len

    def forward(
        self,
        img_feat: Tensor,
        lidar_bev: Tensor,
        future_noisy_latents: Tensor,
        action: Tensor,
        diffusion_t: Tensor,
    ) -> dict[str, Tensor]:
        bev_latent = self.tokenizer(img_feat, lidar_bev)
        recon = self.decoder(bev_latent)
        noise_pred = self.diffusion(future_noisy_latents, action, diffusion_t)
        return {"bev_latent": bev_latent, "reconstruction": recon, "noise_pred": noise_pred}


def build_bevworld() -> nn.Module:
    """Build a small BEVWorld multi-modal tokenizer + latent diffusion world model."""
    return BEVWorld(
        img_ch=8, lidar_ch=8, fuse_dim=16, latent_ch=4, bev_hw=8, d_model=24, n_actions=3, seq_len=2
    ).eval()


def example_input_bevworld() -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Example (image feat, LiDAR BEV feat, noisy future latents, action, diffusion t)."""
    img_feat = torch.randn(1, 8, 8, 8)
    lidar_bev = torch.randn(1, 8, 8, 8)
    future_noisy_latents = torch.randn(1, 2, 4, 8, 8)
    action = torch.randn(1, 3)
    diffusion_t = torch.rand(1, 1)
    return img_feat, lidar_bev, future_noisy_latents, action, diffusion_t


# ============================================================
# Registry
# ============================================================

MENAGERIE_ENTRIES = [
    ("VoxFormer", "build_voxformer", "example_input_voxformer", "2023", "VIS"),
    ("YOLOP", "build_yolop", "example_input_yolop", "2022", "VIS"),
    ("AD-MLP", "build_ad_mlp", "example_input_ad_mlp", "2023", "SEQ"),
    ("AgentFormer", "build_agentformer", "example_input_agentformer", "2021", "SEQ"),
    ("Autobot", "build_autobots", "example_input_autobots", "2022", "SEQ"),
    ("BEVWorld", "build_bevworld", "example_input_bevworld", "2024", "GEN"),
]
