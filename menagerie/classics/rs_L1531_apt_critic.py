# SOURCE: vendored from seolhokim/APT @ master
# https://github.com/seolhokim/APT
# Files: drq.py (CriticEncoder, Critic), apt.py (RandomShiftsAug, APTAgent.projection head)
#
# APT (Active Pretraining with a Particle-Based Entropy Estimator, Liu & Abbeel
# 2021) reward-free exploration agent. The paper's contribution is the
# intrinsic-reward objective (a KNN particle-based entropy estimator over a
# learned representation), built on top of the DrQ (Data-regularized Q)
# image-based actor-critic backbone: a 4-layer strided-conv encoder -> a
# LayerNorm'd feature head -> a twin-Q (double Q-learning) critic. This module
# vendors the real DrQ-style Critic (CriticEncoder + Critic, the network APT's
# `compute_reward`/`update_representation` operate on to produce the particle-
# entropy intrinsic reward) plus APT's own contrastive projection head
# (`nn.Sequential(Linear, ReLU, Linear, LayerNorm)` from `APTAgent.__init__`),
# composed into a single forward-passable module. No architecture was altered;
# only this staging glue (build_/example_input_/MENAGERIE_ENTRIES) was added.

from __future__ import annotations

import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


def _weight_init(m):
    """Custom weight init for Conv2D and Linear layers (utils.weight_init)."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


def _tie_weights(src, trg):
    """utils.tie_weights"""
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


def _mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    """utils.mlp"""
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for _ in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk


class CriticEncoder(nn.Module):
    """Convolutional encoder for image-based observations. (drq.py CriticEncoder)"""

    def __init__(self, obs_shape, feature_dim):
        super().__init__()

        assert len(obs_shape) == 3
        self.num_layers = 4
        self.num_filters = 32
        self.output_dim = 35
        self.output_logits = False
        self.feature_dim = feature_dim

        self.convs = nn.ModuleList(
            [
                nn.Conv2d(obs_shape[0], self.num_filters, 3, stride=2),
                nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1),
                nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1),
                nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1),
            ]
        )

        self.head = nn.Sequential(
            nn.Linear(self.num_filters * 35 * 35, self.feature_dim), nn.LayerNorm(self.feature_dim)
        )

        self.outputs = dict()

    def forward_conv(self, obs):
        obs = obs / 255.0
        self.outputs["obs"] = obs

        conv = torch.relu(self.convs[0](obs))
        self.outputs["conv1"] = conv

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            self.outputs["conv%s" % (i + 1)] = conv

        h = conv.view(conv.size(0), -1)
        return h

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)

        if detach:
            h = h.detach()

        out = self.head(h)
        if not self.output_logits:
            out = torch.tanh(out)

        self.outputs["out"] = out

        return out

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        for i in range(self.num_layers):
            _tie_weights(src=source.convs[i], trg=self.convs[i])


class Critic(nn.Module):
    """Critic network, employs double Q-learning. (drq.py Critic)"""

    def __init__(self, obs_shape, action_shape, feature_dim, hidden_dim, hidden_depth):
        super().__init__()

        self.encoder = CriticEncoder(obs_shape, feature_dim)

        self.Q1 = _mlp(self.encoder.feature_dim + action_shape[0], hidden_dim, 1, hidden_depth)
        self.Q2 = _mlp(self.encoder.feature_dim + action_shape[0], hidden_dim, 1, hidden_depth)

        self.outputs = dict()
        self.apply(_weight_init)

    def forward(self, obs, action, detach_encoder=False):
        assert obs.size(0) == action.size(0)
        obs = self.encoder(obs, detach=detach_encoder)

        obs_action = torch.cat([obs, action], dim=-1)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)

        self.outputs["q1"] = q1
        self.outputs["q2"] = q2

        return q1, q2

    def get_representation(self, obs):
        return self.encoder(obs)


class APTProjectionCritic(nn.Module):
    """Composition of the real DrQ Critic representation encoder with APT's own
    contrastive projection head (apt.py APTAgent.__init__: `self.projection`),
    matching how `APTAgent.update_representation` chains
    `self.critic.get_representation(anchor)` -> `self.projection(...)`."""

    def __init__(self, obs_shape, action_shape, feature_dim, hidden_dim, hidden_depth):
        super().__init__()
        self.critic = Critic(obs_shape, action_shape, feature_dim, hidden_dim, hidden_depth)
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
        )

    def forward(self, obs):
        representation = self.critic.get_representation(obs)
        return self.projection(torch.relu(representation))


# ---------------------------------------------------------------------------
# Menagerie staging glue (not part of the original repo).
# ---------------------------------------------------------------------------


def build_apt_critic():
    torch.manual_seed(0)
    model = APTProjectionCritic(
        obs_shape=(3, 84, 84),
        action_shape=(6,),
        feature_dim=50,
        hidden_dim=32,
        hidden_depth=2,
    )
    model.eval()
    return model


def example_input_apt_critic():
    torch.manual_seed(0)
    # A single stacked-frame image observation, as passed through
    # CriticEncoder.forward -> APTAgent's contrastive projection head.
    return torch.rand(1, 3, 84, 84) * 255.0


MENAGERIE_ENTRIES = [
    (
        "APT (Active Pretraining, particle-entropy critic + contrastive projection)",
        "build_apt_critic",
        "example_input_apt_critic",
        2021,
        MENAGERIE_ZOO,
    ),
]
