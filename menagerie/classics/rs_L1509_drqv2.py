# SOURCE: vendored from facebookresearch/drqv2 @ main
# https://github.com/facebookresearch/drqv2
# File vendored near-verbatim: drqv2.py (RandomShiftsAug, Encoder, Actor, Critic) plus the
# `weight_init` / `TruncatedNormal` / `schedule` helpers from utils.py that the model classes
# depend on directly. `hydra`/`omegaconf` (used only by the training-script `DrQV2Agent`
# class's config plumbing, not by the network modules themselves) are dropped -- the vendored
# code below is the real DrQ-v2 architecture: image augmentation + CNN encoder + actor/critic
# heads for continuous-control DDPG-style RL from pixels.
import math
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# utils.py (relevant subset, verbatim)
# ---------------------------------------------------------------------------


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


class TruncatedNormal(pyd.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)


def schedule(schdl, step):
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r"linear\((.+),(.+),(.+)\)", schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
        match = re.match(r"step_linear\((.+),(.+),(.+),(.+),(.+)\)", schdl)
        if match:
            init, final1, duration1, final2, duration2 = [float(g) for g in match.groups()]
            if step <= duration1:
                mix = np.clip(step / duration1, 0.0, 1.0)
                return (1.0 - mix) * init + mix * final1
            else:
                mix = np.clip((step - duration1) / duration2, 0.0, 1.0)
                return (1.0 - mix) * final1 + mix * final2
    raise NotImplementedError(schdl)


# ---------------------------------------------------------------------------
# drqv2.py (verbatim)
# ---------------------------------------------------------------------------


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(
            -1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype
        )[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(
            0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype
        )
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


class Encoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
        )

        self.apply(weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h


class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(repr_dim, feature_dim), nn.LayerNorm(feature_dim), nn.Tanh()
        )

        self.policy = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_shape[0]),
        )

        self.apply(weight_init)

    def forward(self, obs, std):
        h = self.trunk(obs)

        mu = self.policy(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = TruncatedNormal(mu, std)
        return dist


class Critic(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(repr_dim, feature_dim), nn.LayerNorm(feature_dim), nn.Tanh()
        )

        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        self.apply(weight_init)

    def forward(self, obs, action):
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2


# ---------------------------------------------------------------------------
# menagerie staging entry points
# ---------------------------------------------------------------------------

_OBS_SHAPE = (9, 84, 84)  # 3 stacked RGB frames, DeepMind Control default
_ACTION_SHAPE = (6,)
_FEATURE_DIM = 32
_HIDDEN_DIM = 32
_STDDEV = 0.2


class DrQV2AgentWrapper(nn.Module):
    """Wraps the real DrQ-v2 pipeline (RandomShiftsAug -> Encoder -> Actor + Critic, mirroring
    DrQV2Agent.act / update_critic data flow) behind a single-tensor forward."""

    def __init__(self):
        super().__init__()
        self.aug = RandomShiftsAug(pad=4)
        self.encoder = Encoder(_OBS_SHAPE)
        self.actor = Actor(self.encoder.repr_dim, _ACTION_SHAPE, _FEATURE_DIM, _HIDDEN_DIM)
        self.critic = Critic(self.encoder.repr_dim, _ACTION_SHAPE, _FEATURE_DIM, _HIDDEN_DIM)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs = self.aug(obs)
        h = self.encoder(obs)
        dist = self.actor(h, _STDDEV)
        action = dist.mean
        q1, q2 = self.critic(h, action)
        return torch.min(q1, q2)


def build_drqv2_agent() -> nn.Module:
    return DrQV2AgentWrapper()


def example_input_drqv2_agent() -> torch.Tensor:
    return torch.rand(2, *_OBS_SHAPE) * 255.0


MENAGERIE_ENTRIES = [
    (
        "DrQv2_pixel_ddpg_agent",
        build_drqv2_agent,
        example_input_drqv2_agent,
        2021,
        "vendored-pytorch",
    ),
]
