# SOURCE: vendored from https://github.com/sfujim/TD3_BC @ main
# (TD3_BC.py: Actor, Critic, lines 1-60)
#
# TD3+BC (Fujimoto & Gu, NeurIPS 2021, "A Minimalist Approach to Offline
# Reinforcement Learning", arXiv:2106.06860). This is the author's own
# official PyTorch repo (Scott Fujimoto) -- the reference implementation, not
# a community port. TD3+BC's architectural contribution over vanilla TD3 is
# NONE (it is the same twin-Q Actor/Critic network; the paper's contribution
# is purely an added behavior-cloning regularization TERM in the actor loss
# and a state-normalization trick, both training-loop-only). The `Actor`
# (3-layer MLP, tanh-bounded continuous action) and `Critic` (twin 3-layer-MLP
# Q-networks, the "Q1 architecture" / "Q2 architecture" duplicate heads that
# implement TD3's clipped double-Q trick) have no dependency beyond
# torch, so they are vendored verbatim. `copy`/`device`-handling
# training-loop code (the `TD3_BC` outer class's `train`/`save`/`load`) is
# omitted -- it is training-loop logic, not part of the traced architecture.

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


# ---- TD3_BC.py (vendored verbatim, forward-pass architecture only) ----
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


# ---- end vendored TD3_BC.py ----


class TD3BCNet(nn.Module):
    """Staging wrapper exercising the real Actor + twin-Q Critic construction
    as a single traceable module: one forward pass through the actor produces
    an action, which is fed (together with state) into the twin-Q critic --
    matching the exact tensors TD3_BC.train() consumes each step."""

    def __init__(self, state_dim=17, action_dim=6, max_action=1.0):
        super().__init__()
        self.actor = Actor(state_dim, action_dim, max_action)
        self.critic = Critic(state_dim, action_dim)

    def forward(self, state):
        action = self.actor(state)
        q1, q2 = self.critic(state, action)
        return action, q1, q2


def build_td3_bc():
    return TD3BCNet(state_dim=17, action_dim=6, max_action=1.0)


def example_input_td3_bc():
    return (torch.randn(4, 17),)


MENAGERIE_ENTRIES = [
    ("TD3+BC", build_td3_bc, example_input_td3_bc, 2021, "vendored-pytorch"),
]
