# SOURCE: vendored from https://github.com/TianhongDai/hindsight-experience-replay @ master
# (rl_modules/models.py, unmodified architecture.)
"""Hindsight Experience Replay (HER, Andrychowicz et al. 2017,
"Hindsight Experience Replay") actor/critic networks, paired with DDPG as in
the reference implementation. HER itself is a replay-buffer relabeling
technique (goal substitution), not an architectural change: the network is a
plain goal-conditioned DDPG actor (3-layer MLP + tanh-bounded action head) and
critic (3-layer MLP + scalar Q head over concatenated state/goal/action).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
the input x in both networks should be [o, g], where o is the observation and g is the goal.

"""


# define the actor network
class actor(nn.Module):
    def __init__(self, env_params):
        super(actor, self).__init__()
        self.max_action = env_params["action_max"]
        self.fc1 = nn.Linear(env_params["obs"] + env_params["goal"], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.action_out = nn.Linear(256, env_params["action"])

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.max_action * torch.tanh(self.action_out(x))

        return actions


class critic(nn.Module):
    def __init__(self, env_params):
        super(critic, self).__init__()
        self.max_action = env_params["action_max"]
        self.fc1 = nn.Linear(env_params["obs"] + env_params["goal"] + env_params["action"], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.q_out = nn.Linear(256, 1)

    def forward(self, x, actions):
        x = torch.cat([x, actions / self.max_action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)

        return q_value


# ---- staging scaffolding (env_params shim; no architecture changes) ----
_ENV_PARAMS = {"obs": 10, "goal": 3, "action": 4, "action_max": 1.0}


def build_her_actor():
    return actor(_ENV_PARAMS)


def example_input_her_actor():
    return torch.randn(4, _ENV_PARAMS["obs"] + _ENV_PARAMS["goal"])


def build_her_critic():
    return critic(_ENV_PARAMS)


def example_input_her_critic():
    x = torch.randn(4, _ENV_PARAMS["obs"] + _ENV_PARAMS["goal"])
    actions = torch.randn(4, _ENV_PARAMS["action"])
    return (x, actions)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    (
        "HER/DDPG Actor (goal-conditioned MLP)",
        "build_her_actor",
        "example_input_her_actor",
        2017,
        MENAGERIE_ZOO,
    ),
    (
        "HER/DDPG Critic (goal-conditioned MLP)",
        "build_her_critic",
        "example_input_her_critic",
        2017,
        MENAGERIE_ZOO,
    ),
]
