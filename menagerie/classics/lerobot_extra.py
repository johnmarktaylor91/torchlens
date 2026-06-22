"""Additional compact LeRobot policy reconstructions.

Papers/Models: TD-MPC (Hansen et al., 2022), GR00T N1 (NVIDIA, 2025),
MolmoAct (Allen Institute for AI, 2025/2026), and Multitask DiT Policy
(LeRobot/Hugging Face, 2025).

This module fills dependency-gated LeRobot entries with faithful compact
primitives: Gaussian continuous-control actors, TD-MPC latent planning,
dual-system VLA plus diffusion transformer actions, spatial action-reasoning
traces, and multitask DiT denoising/flow action generation.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .lerobot_vla_policies import build_diffusion
from .lerobot_vla_policies import build_pi0
from .lerobot_vla_policies import build_pi05
from .lerobot_vla_policies import build_pi0fast
from .lerobot_vla_policies import build_smolvla
from .lerobot_vla_policies import build_tdmpc
from .lerobot_vla_policies import build_vqbet
from .lerobot_vla_policies import build_xvla
from .lerobot_vla_policies import example_diffusion
from .lerobot_vla_policies import example_tdmpc
from .lerobot_vla_policies import example_vla
from .lerobot_vla_policies import example_vqbet


class GaussianActor(nn.Module):
    """Continuous Gaussian actor for compact LeRobot RL policies."""

    def __init__(self, state_dim: int = 8, action_dim: int = 7) -> None:
        """Initialize tanh-squashed Gaussian actor heads."""

        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(state_dim, 48), nn.ELU(), nn.Linear(48, 48), nn.ELU())
        self.mean = nn.Linear(48, action_dim)
        self.logstd = nn.Linear(48, action_dim)

    def forward(self, state: Tensor) -> tuple[Tensor, Tensor]:
        """Return action mean and standard deviation."""

        feat = self.trunk(state)
        mean = torch.tanh(self.mean(feat))
        std = F.softplus(self.logstd(feat)) + 1e-3
        return mean, std


class MultiTaskDiTPolicy(nn.Module):
    """Diffusion Transformer policy conditioned on vision, language, and state."""

    def __init__(self, action_dim: int = 7, dim: int = 48, horizon: int = 8) -> None:
        """Initialize action tokens, condition encoders, and DiT blocks."""

        super().__init__()
        self.horizon = horizon
        self.image = nn.Sequential(nn.Conv2d(3, dim, 4, stride=4), nn.GELU())
        self.text = nn.Embedding(128, dim)
        self.state = nn.Linear(8, dim)
        self.action = nn.Linear(action_dim, dim)
        self.time = nn.Linear(1, dim)
        layer = nn.TransformerEncoderLayer(dim, 4, dim * 2, batch_first=True, norm_first=True)
        self.dit = nn.TransformerEncoder(layer, num_layers=2)
        self.out = nn.Linear(dim, action_dim)

    def forward(
        self, image: Tensor, state: Tensor, text: Tensor, noisy_actions: Tensor, sigma: Tensor
    ) -> Tensor:
        """Denoise or flow-match a future action trajectory."""

        visual = self.image(image).flatten(2).transpose(1, 2)[:, :4]
        context = torch.cat([visual, self.text(text), self.state(state).unsqueeze(1)], dim=1)
        actions = self.action(noisy_actions) + self.time(sigma[:, None, None])
        tokens = torch.cat([context, actions], dim=1)
        decoded = self.dit(tokens)
        return self.out(decoded[:, -self.horizon :])


class MolmoActPolicy(nn.Module):
    """MolmoAct-style depth, visual trace, and action token generator."""

    def __init__(self, dim: int = 48, action_dim: int = 7) -> None:
        """Initialize VLM tokens, trace projector, and action decoder."""

        super().__init__()
        self.vision = nn.Conv2d(3, dim, 4, stride=4)
        self.text = nn.Embedding(128, dim)
        self.trace = nn.Linear(3, dim)
        self.depth_code = nn.Linear(dim, 16)
        self.reason = nn.GRU(dim, dim, batch_first=True)
        self.trace_token = nn.Linear(dim, 2)
        self.action_token = nn.Linear(dim, action_dim)

    def forward(
        self, image: Tensor, text: Tensor, trace_points: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Generate depth tokens, visual traces, then action tokens."""

        visual = self.vision(image).flatten(2).transpose(1, 2)[:, :4]
        depth_tokens = self.depth_code(visual)
        tokens = torch.cat(
            [
                visual + depth_tokens.mean(dim=-1, keepdim=True),
                self.text(text),
                self.trace(trace_points),
            ],
            dim=1,
        )
        reasoned, _ = self.reason(tokens)
        trace_hidden = reasoned[:, -trace_points.shape[1] :]
        return depth_tokens, self.trace_token(trace_hidden), self.action_token(trace_hidden)


class GROOTN1Policy(nn.Module):
    """GR00T N1 dual-system VLA with fast diffusion action system."""

    def __init__(self, dim: int = 48, action_dim: int = 7, horizon: int = 8) -> None:
        """Initialize System-2 VLM and System-1 diffusion transformer."""

        super().__init__()
        self.horizon = horizon
        self.vlm = MultiTaskDiTPolicy(action_dim=action_dim, dim=dim, horizon=horizon)
        self.system2_plan = nn.Linear(action_dim, dim)
        self.system1 = nn.GRU(dim, dim, batch_first=True)
        self.head = nn.Linear(dim, action_dim)

    def forward(
        self, image: Tensor, state: Tensor, text: Tensor, noisy_actions: Tensor, sigma: Tensor
    ) -> Tensor:
        """Generate fast motor actions from a deliberative VLA plan."""

        plan = self.vlm(image, state, text, noisy_actions, sigma)
        tokens = self.system2_plan(plan)
        fast, _ = self.system1(tokens)
        return self.head(fast)


class EO1UnifiedPolicy(nn.Module):
    """EO-1 unified decoder-only VLA with language and flow action heads."""

    def __init__(self, dim: int = 48, action_dim: int = 7, horizon: int = 6) -> None:
        """Initialize EO-1 compact policy."""

        super().__init__()
        self.horizon = horizon
        self.image = nn.Conv2d(3, dim, 4, stride=4)
        self.text = nn.Embedding(128, dim)
        self.state = nn.Linear(8, dim)
        self.action = nn.Linear(action_dim, dim)
        self.time = nn.Linear(1, dim)
        layer = nn.TransformerEncoderLayer(dim, 4, dim * 2, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerEncoder(layer, num_layers=2)
        self.language_head = nn.Linear(dim, 128)
        self.flow_head = nn.Linear(dim, action_dim)

    def forward(
        self, images: Tensor, state: Tensor, text: Tensor, noisy_actions: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Decode language logits and continuous flow actions in one backbone."""

        visual = self.image(images[:, 0]).flatten(2).transpose(1, 2)[:, :4]
        text_tokens = self.text(text)
        state_token = self.state(state).unsqueeze(1)
        sigma = noisy_actions.new_full((noisy_actions.shape[0], 1, 1), 0.5)
        action_tokens = self.action(noisy_actions) + self.time(sigma)
        causal_tokens = torch.cat([visual, text_tokens, state_token, action_tokens], dim=1)
        decoded = self.decoder(causal_tokens)
        language_logits = self.language_head(
            decoded[:, visual.shape[1] : visual.shape[1] + text.shape[1]]
        )
        flow = self.flow_head(decoded[:, -self.horizon :])
        return language_logits, flow


class WallXPolicy(nn.Module):
    """WALL-X/WALL-OSS VLMoE policy with flow and FAST action branches."""

    def __init__(self, dim: int = 48, action_dim: int = 7, horizon: int = 6) -> None:
        """Initialize compact WALL-X policy."""

        super().__init__()
        self.horizon = horizon
        self.vision = nn.Conv2d(3, dim, 4, stride=4)
        self.language = nn.Embedding(128, dim)
        self.state = nn.Linear(8, dim)
        self.embodiment_gate = nn.Linear(8, 2)
        self.qwen_vl_block = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim, 4, dim * 2, batch_first=True, norm_first=True),
            num_layers=1,
        )
        self.flow_action = MultiTaskDiTPolicy(action_dim=action_dim, dim=dim, horizon=horizon)
        self.fast_branch = nn.GRU(dim, dim, batch_first=True)
        self.fast_head = nn.Linear(dim, action_dim)

    def forward(self, images: Tensor, state: Tensor, text: Tensor, noisy_actions: Tensor) -> Tensor:
        """Predict cross-embodiment actions from WALL-X action branches."""

        visual = self.vision(images[:, 0]).flatten(2).transpose(1, 2)[:, :4]
        context = torch.cat([visual, self.language(text), self.state(state).unsqueeze(1)], dim=1)
        context = self.qwen_vl_block(context)
        branch_gate = torch.softmax(self.embodiment_gate(state), dim=-1)
        sigma = noisy_actions.new_full((noisy_actions.shape[0],), 0.5)
        flow = self.flow_action(images[:, 0], state, text, noisy_actions, sigma)
        fast, _ = self.fast_branch(context[:, : self.horizon])
        fast = self.fast_head(fast)
        return branch_gate[:, :1, None] * flow + branch_gate[:, 1:, None] * fast


def build_gaussian_actor() -> nn.Module:
    """Build compact Gaussian actor."""

    return GaussianActor().eval()


def build_multitask_dit() -> nn.Module:
    """Build compact multitask DiT policy."""

    return MultiTaskDiTPolicy().eval()


def build_molmoact() -> nn.Module:
    """Build compact MolmoAct policy."""

    return MolmoActPolicy().eval()


def build_groot() -> nn.Module:
    """Build compact GR00T N1/GROOT VLA policy."""

    return GROOTN1Policy().eval()


def build_eo1() -> nn.Module:
    """Build compact EO-1 unified VLA policy."""

    return EO1UnifiedPolicy().eval()


def build_wall_x() -> nn.Module:
    """Build compact WALL-X/WALL-OSS policy."""

    return WallXPolicy().eval()


def example_actor() -> Tensor:
    """Return robot state for Gaussian actor."""

    return torch.randn(1, 8)


def example_dit() -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Return image, state, instruction, noisy actions, and noise level."""

    return (
        torch.randn(1, 3, 32, 32),
        torch.randn(1, 8),
        torch.randint(0, 128, (1, 5)),
        torch.randn(1, 8, 7),
        torch.tensor([0.5]),
    )


def example_molmoact() -> tuple[Tensor, Tensor, Tensor]:
    """Return image, instruction, and 3D trace points."""

    return torch.randn(1, 3, 32, 32), torch.randint(0, 128, (1, 5)), torch.randn(1, 6, 3)


MENAGERIE_ENTRIES = [
    ("eo1_lerobot", "build_eo1", "example_vla", "2025", "RL"),
    ("lerobot_gaussian_actor", "build_gaussian_actor", "example_actor", "2024", "RL"),
    ("groot_n1_lerobot", "build_groot", "example_dit", "2025", "RL"),
    ("lerobot_groot", "build_groot", "example_dit", "2025", "RL"),
    ("molmoact2_lerobot", "build_molmoact", "example_molmoact", "2026", "RL"),
    ("multi_task_dit_lerobot", "build_multitask_dit", "example_dit", "2025", "RL"),
    ("lerobot_tdmpc_simxarm", "build_tdmpc", "example_tdmpc", "2022", "RL"),
    ("wall_x_lerobot", "build_wall_x", "example_vla", "2025", "RL"),
    ("lerobot_wall_x", "build_wall_x", "example_vla", "2025", "RL"),
    ("lerobot_diffusion", "build_diffusion", "example_diffusion", "2023", "RL"),
    ("lerobot_vqbet", "build_vqbet", "example_vqbet", "2024", "RL"),
    ("lerobot_tdmpc", "build_tdmpc", "example_tdmpc", "2022", "RL"),
    ("lerobot_pi0", "build_pi0", "example_vla", "2024", "RL"),
    ("lerobot_pi0_fast", "build_pi0fast", "example_vla", "2024", "RL"),
    ("lerobot_pi05", "build_pi05", "example_vla", "2025", "RL"),
    ("lerobot_smolvla", "build_smolvla", "example_vla", "2025", "RL"),
]
