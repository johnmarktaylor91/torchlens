# SOURCE: vendored from Khrylx/PyTorch-RL @ master
# https://github.com/Khrylx/PyTorch-RL
# Files: models/mlp_policy.py (Policy), models/mlp_discriminator.py (Discriminator)
#
# GAIL (Generative Adversarial Imitation Learning, Ho & Ermon 2016) recovers
# an expert policy by training a discriminator to distinguish expert
# state-action pairs from policy-generated ones, and training the policy to
# fool the discriminator (gail/gail_gym.py). This module vendors the real
# `Policy` (a tanh-MLP Gaussian policy over continuous actions,
# models/mlp_policy.py) and `Discriminator` (a tanh-MLP binary classifier
# over concatenated state-action pairs, models/mlp_discriminator.py) classes
# verbatim, and composes them exactly as gail_gym.py's `expert_reward`
# closure does: sample an action from the policy, concatenate it with the
# state, and score the pair with the discriminator
# (`discrim_net(torch.cat([state, action], dim=-1))`). No architecture was
# altered; only this staging glue (build_/example_input_/MENAGERIE_ENTRIES)
# was added.

from __future__ import annotations

import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


def normal_log_density(x, mean, log_std, std):
    """utils.math.normal_log_density (needed by Policy.get_log_prob, kept for
    fidelity even though this module's forward pass does not call it)."""
    var = std.pow(2)
    log_density = (
        -(x - mean).pow(2) / (2 * var)
        - 0.5 * torch.log(2 * torch.tensor(3.141592653589793))
        - log_std
    )
    return log_density.sum(1, keepdim=True)


class Policy(nn.Module):
    """Gaussian MLP policy. (models/mlp_policy.py Policy)"""

    def __init__(self, state_dim, action_dim, hidden_size=(128, 128), activation="tanh", log_std=0):
        super().__init__()
        self.is_disc_action = False
        if activation == "tanh":
            self.activation = torch.tanh
        elif activation == "relu":
            self.activation = torch.relu
        elif activation == "sigmoid":
            self.activation = torch.sigmoid

        self.affine_layers = nn.ModuleList()
        last_dim = state_dim
        for nh in hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

        self.action_mean = nn.Linear(last_dim, action_dim)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)

        self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * log_std)

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))

        action_mean = self.action_mean(x)
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)

        return action_mean, action_log_std, action_std

    def select_action(self, x):
        action_mean, _, action_std = self.forward(x)
        action = torch.normal(action_mean, action_std)
        return action

    def get_kl(self, x):
        mean1, log_std1, std1 = self.forward(x)

        mean0 = mean1.detach()
        log_std0 = log_std1.detach()
        std0 = std1.detach()
        kl = (
            log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        )
        return kl.sum(1, keepdim=True)

    def get_log_prob(self, x, actions):
        action_mean, action_log_std, action_std = self.forward(x)
        return normal_log_density(actions, action_mean, action_log_std, action_std)

    def get_fim(self, x):
        mean, _, _ = self.forward(x)
        cov_inv = self.action_log_std.exp().pow(-2).squeeze(0).repeat(x.size(0))
        param_count = 0
        std_index = 0
        id = 0
        for name, param in self.named_parameters():
            if name == "action_log_std":
                std_id = id
                std_index = param_count
            param_count += param.view(-1).shape[0]
            id += 1
        return cov_inv.detach(), mean, {"std_id": std_id, "std_index": std_index}


class Discriminator(nn.Module):
    """GAIL discriminator over state-action pairs. (models/mlp_discriminator.py Discriminator)"""

    def __init__(self, num_inputs, hidden_size=(128, 128), activation="tanh"):
        super().__init__()
        if activation == "tanh":
            self.activation = torch.tanh
        elif activation == "relu":
            self.activation = torch.relu
        elif activation == "sigmoid":
            self.activation = torch.sigmoid

        self.affine_layers = nn.ModuleList()
        last_dim = num_inputs
        for nh in hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

        self.logic = nn.Linear(last_dim, 1)
        self.logic.weight.data.mul_(0.1)
        self.logic.bias.data.mul_(0.0)

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))

        prob = torch.sigmoid(self.logic(x))
        return prob


class GAILPolicyDiscriminator(nn.Module):
    """Composition of the real Policy with the real Discriminator, matching
    gail_gym.py's `expert_reward` closure: sample an action from the policy
    given a state, concatenate state and action, and score the pair with the
    discriminator (`discrim_net(torch.cat([state, action], dim=-1))`)."""

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.policy = Policy(state_dim, action_dim)
        self.discriminator = Discriminator(state_dim + action_dim)

    def forward(self, state):
        action_mean, _, _ = self.policy(state)
        state_action = torch.cat([state, action_mean], dim=-1)
        return self.discriminator(state_action)


# ---------------------------------------------------------------------------
# Menagerie staging glue (not part of the original repo).
# ---------------------------------------------------------------------------


def build_gail():
    torch.manual_seed(0)
    model = GAILPolicyDiscriminator(state_dim=11, action_dim=3)
    model.eval()
    return model


def example_input_gail():
    torch.manual_seed(0)
    return torch.randn(1, 11)


MENAGERIE_ENTRIES = [
    (
        "GAIL (Generative Adversarial Imitation Learning, MLP policy + discriminator)",
        "build_gail",
        "example_input_gail",
        2016,
        MENAGERIE_ZOO,
    ),
]
