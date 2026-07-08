# SOURCE: vendored from https://github.com/xuehy/pytorch-maddpg @ master
# (model.py: Critic, Actor, lines 1-38)
#
# MADDPG (Lowe et al. 2017, "Multi-Agent Actor-Critic for Mixed
# Cooperative-Competitive Environments", arXiv:1706.02275). The paper's own
# reference code (openai/maddpg) is TensorFlow 1.x; xuehy/pytorch-maddpg is the
# well-known community PyTorch port used against a modified MADRL Waterworld
# pursuit environment, and is the canonical PyTorch reference cited across
# MADDPG reimplementations. The `Critic`/`Actor` classes have no dependency
# beyond torch, so they are vendored verbatim. Architecture: N decentralized
# `Actor` MLPs (per-agent policy, tanh-bounded continuous action) paired with N
# `Critic` MLPs that each consume the JOINT observation and JOINT action across
# all agents (centralized-critic / decentralized-actor pattern that is
# MADDPG's defining architectural move) -- captured here as a small multi-agent
# ensemble (`MADDPGNet`) wrapping 1 Actor + 1 Critic per agent so a single
# forward pass exercises the full joint-obs/joint-action critic construction.

import torch as th
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


# ---- model.py (vendored verbatim) ----
class Critic(nn.Module):
    def __init__(self, n_agent, dim_observation, dim_action):
        super(Critic, self).__init__()
        self.n_agent = n_agent
        self.dim_observation = dim_observation
        self.dim_action = dim_action
        obs_dim = dim_observation * n_agent
        act_dim = self.dim_action * n_agent

        self.FC1 = nn.Linear(obs_dim, 1024)
        self.FC2 = nn.Linear(1024 + act_dim, 512)
        self.FC3 = nn.Linear(512, 300)
        self.FC4 = nn.Linear(300, 1)

    # obs: batch_size * obs_dim
    def forward(self, obs, acts):
        result = F.relu(self.FC1(obs))
        combined = th.cat([result, acts], 1)
        result = F.relu(self.FC2(combined))
        return self.FC4(F.relu(self.FC3(result)))


class Actor(nn.Module):
    def __init__(self, dim_observation, dim_action):
        super(Actor, self).__init__()
        self.FC1 = nn.Linear(dim_observation, 500)
        self.FC2 = nn.Linear(500, 128)
        self.FC3 = nn.Linear(128, dim_action)

    # action output between -2 and 2
    def forward(self, obs):
        result = F.relu(self.FC1(obs))
        result = F.relu(self.FC2(result))
        result = th.tanh(self.FC3(result))
        return result


# ---- end vendored model.py ----


class MADDPGNet(nn.Module):
    """Staging wrapper exercising MADDPG.py's real per-agent construction
    (`self.actors = [Actor(...) for i in range(n_agents)]`,
    `self.critics = [Critic(n_agents, ...) for i in range(n_agents)]`) as a
    single traceable module: one Actor + one (joint-obs/joint-action) Critic
    per agent, matching the reference `main.py` 3-pursuer Waterworld setup at
    reduced width."""

    def __init__(self, n_agents=3, dim_obs=8, dim_act=2):
        super().__init__()
        self.n_agents = n_agents
        self.dim_obs = dim_obs
        self.dim_act = dim_act
        self.actors = nn.ModuleList([Actor(dim_obs, dim_act) for _ in range(n_agents)])
        self.critics = nn.ModuleList([Critic(n_agents, dim_obs, dim_act) for _ in range(n_agents)])

    def forward(self, obs_n):
        # obs_n: (batch, n_agents, dim_obs) -- per-agent local observations
        batch_size = obs_n.size(0)
        actions = [self.actors[i](obs_n[:, i, :]) for i in range(self.n_agents)]
        joint_obs = obs_n.view(batch_size, -1)
        joint_act = th.cat(actions, dim=1)
        values = [self.critics[i](joint_obs, joint_act) for i in range(self.n_agents)]
        return th.cat(actions, dim=1), th.cat(values, dim=1)


def build_maddpg():
    return MADDPGNet(n_agents=3, dim_obs=8, dim_act=2)


def example_input_maddpg():
    return (th.randn(2, 3, 8),)


MENAGERIE_ENTRIES = [
    ("MADDPG", "build_maddpg", "example_input_maddpg", 2017, "vendored-pytorch"),
]
