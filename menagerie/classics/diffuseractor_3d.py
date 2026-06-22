"""3D Diffuser Actor compact random-init reconstruction.

Paper: 3D Diffuser Actor: Policy Diffusion with 3D Scene Representations
(Ke, Gkanatsios, Fragkiadaki, CoRL/PMLR 2025).

The faithful primitive is a diffusion denoiser over future end-effector
trajectories conditioned on a 3D scene feature field, language, proprioception,
and timestep embeddings.  This compact version keeps 3D relative-position
attention between noisy trajectory queries and scene points and predicts
translation, rotation, and gripper residuals.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class RelativeSceneAttention(nn.Module):
    """Scene-to-trajectory attention with learned relative 3D position bias."""

    def __init__(self, dim: int = 48) -> None:
        """Initialize projections and relative-coordinate bias network."""

        super().__init__()
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.rel = nn.Sequential(nn.Linear(3, dim), nn.SiLU(), nn.Linear(dim, 1))
        self.out = nn.Linear(dim, dim)

    def forward(self, query: Tensor, query_xyz: Tensor, scene: Tensor, scene_xyz: Tensor) -> Tensor:
        """Attend trajectory queries to 3D scene features.

        Parameters
        ----------
        query:
            Trajectory token features with shape ``(B, T, D)``.
        query_xyz:
            Current noisy trajectory positions with shape ``(B, T, 3)``.
        scene:
            Scene point features with shape ``(B, N, D)``.
        scene_xyz:
            Scene coordinates with shape ``(B, N, 3)``.

        Returns
        -------
        Tensor
            Updated trajectory token features.
        """

        scale = query.shape[-1] ** -0.5
        rel_xyz = query_xyz[:, :, None, :] - scene_xyz[:, None, :, :]
        score = torch.matmul(self.q(query), self.k(scene).transpose(-1, -2)) * scale
        score = score + self.rel(rel_xyz).squeeze(-1)
        weights = torch.softmax(score, dim=-1)
        return self.out(torch.matmul(weights, self.v(scene)))


class DiffuserActor3D(nn.Module):
    """Compact 3D Diffuser Actor denoising policy."""

    def __init__(self, dim: int = 48, horizon: int = 4) -> None:
        """Initialize scene encoders, denoising blocks, and residual heads."""

        super().__init__()
        self.horizon = horizon
        self.scene_proj = nn.Linear(9, dim)
        self.action_proj = nn.Linear(7, dim)
        self.lang_proj = nn.Linear(16, dim)
        self.prop_proj = nn.Linear(8, dim)
        self.time_proj = nn.Sequential(nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.attn1 = RelativeSceneAttention(dim)
        self.attn2 = RelativeSceneAttention(dim)
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim * 2), nn.SiLU(), nn.Linear(dim * 2, dim)
        )
        self.delta_pos = nn.Linear(dim, 3)
        self.delta_rot = nn.Linear(dim, 3)
        self.gripper = nn.Linear(dim, 1)

    def forward(
        self, inputs: tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Predict diffusion residuals for a noisy 3D action trajectory.

        Parameters
        ----------
        inputs:
            ``(scene_xyz, scene_feat, noisy_action, language, timestep)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Translation residuals, rotation residuals, and gripper logits.
        """

        scene_xyz, scene_feat, noisy_action, language, timestep = inputs
        scene = self.scene_proj(torch.cat([scene_xyz, scene_feat], dim=-1))
        query_xyz = noisy_action[..., :3]
        query = self.action_proj(noisy_action)
        query = query + self.lang_proj(language).unsqueeze(1)
        query = query + self.time_proj(timestep[:, None].float()).unsqueeze(1)
        query = query + self.attn1(query, query_xyz, scene, scene_xyz)
        query = query + self.attn2(
            query, query_xyz + torch.tanh(self.delta_pos(query)), scene, scene_xyz
        )
        query = query + self.mlp(query)
        prop = self.prop_proj(torch.cat([noisy_action.mean(dim=1), language[:, :1]], dim=-1))
        query = query + prop.unsqueeze(1)
        return self.delta_pos(query), self.delta_rot(query), self.gripper(query)


def build() -> nn.Module:
    """Build a compact random-init 3D Diffuser Actor."""

    return DiffuserActor3D().eval()


def example_input() -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Return scene points, features, noisy trajectory, language vector, and timestep."""

    return (
        torch.randn(1, 12, 3),
        torch.randn(1, 12, 6),
        torch.randn(1, 4, 7),
        torch.randn(1, 16),
        torch.tensor([7]),
    )


MENAGERIE_ENTRIES = [
    ("DiffuserActor_3D", "build", "example_input", "2025", "DC"),
]
