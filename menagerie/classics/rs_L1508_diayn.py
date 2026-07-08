# SOURCE: vendored from https://github.com/akazemipour/DIAYN-PyTorch @ main
# (Brain/model.py, verbatim -- only import path adjusted since we drop the package's
# relative-import wrapper; no architecture changes)
"""DIAYN (Diversity Is All You Need; Eysenbach et al. 2018) learns a set of diverse skills
without a reward signal by maximizing mutual information between skills and visited states,
using a skill-conditioned SAC agent. This vendors the real network classes from the
DIAYN-PyTorch reference implementation: `Discriminator` (predicts skill z from state s,
the DIAYN intrinsic-reward critic) and `PolicyNetwork` (the skill-conditioned SAC actor).
`ValueNetwork`/`QvalueNetwork` are the standard SAC value/Q critics also defined in the
same file; included here for completeness of the vendored module."""

from abc import ABC
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal


def init_weight(layer, initializer="he normal"):
    if initializer == "xavier uniform":
        nn.init.xavier_uniform_(layer.weight)
    elif initializer == "he normal":
        nn.init.kaiming_normal_(layer.weight)


class Discriminator(nn.Module, ABC):
    def __init__(self, n_states, n_skills, n_hidden_filters=256):
        super(Discriminator, self).__init__()
        self.n_states = n_states
        self.n_skills = n_skills
        self.n_hidden_filters = n_hidden_filters

        self.hidden1 = nn.Linear(in_features=self.n_states, out_features=self.n_hidden_filters)
        init_weight(self.hidden1)
        self.hidden1.bias.data.zero_()
        self.hidden2 = nn.Linear(
            in_features=self.n_hidden_filters, out_features=self.n_hidden_filters
        )
        init_weight(self.hidden2)
        self.hidden2.bias.data.zero_()
        self.q = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_skills)
        init_weight(self.q, initializer="xavier uniform")
        self.q.bias.data.zero_()

    def forward(self, states):
        x = F.relu(self.hidden1(states))
        x = F.relu(self.hidden2(x))
        logits = self.q(x)
        return logits


class ValueNetwork(nn.Module, ABC):
    def __init__(self, n_states, n_hidden_filters=256):
        super(ValueNetwork, self).__init__()
        self.n_states = n_states
        self.n_hidden_filters = n_hidden_filters

        self.hidden1 = nn.Linear(in_features=self.n_states, out_features=self.n_hidden_filters)
        init_weight(self.hidden1)
        self.hidden1.bias.data.zero_()
        self.hidden2 = nn.Linear(
            in_features=self.n_hidden_filters, out_features=self.n_hidden_filters
        )
        init_weight(self.hidden2)
        self.hidden2.bias.data.zero_()
        self.value = nn.Linear(in_features=self.n_hidden_filters, out_features=1)
        init_weight(self.value, initializer="xavier uniform")
        self.value.bias.data.zero_()

    def forward(self, states):
        x = F.relu(self.hidden1(states))
        x = F.relu(self.hidden2(x))
        return self.value(x)


class QvalueNetwork(nn.Module, ABC):
    def __init__(self, n_states, n_actions, n_hidden_filters=256):
        super(QvalueNetwork, self).__init__()
        self.n_states = n_states
        self.n_hidden_filters = n_hidden_filters
        self.n_actions = n_actions

        self.hidden1 = nn.Linear(
            in_features=self.n_states + self.n_actions, out_features=self.n_hidden_filters
        )
        init_weight(self.hidden1)
        self.hidden1.bias.data.zero_()
        self.hidden2 = nn.Linear(
            in_features=self.n_hidden_filters, out_features=self.n_hidden_filters
        )
        init_weight(self.hidden2)
        self.hidden2.bias.data.zero_()
        self.q_value = nn.Linear(in_features=self.n_hidden_filters, out_features=1)
        init_weight(self.q_value, initializer="xavier uniform")
        self.q_value.bias.data.zero_()

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=1)
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        return self.q_value(x)


class PolicyNetwork(nn.Module, ABC):
    def __init__(self, n_states, n_actions, action_bounds, n_hidden_filters=256):
        super(PolicyNetwork, self).__init__()
        self.n_states = n_states
        self.n_hidden_filters = n_hidden_filters
        self.n_actions = n_actions
        self.action_bounds = action_bounds

        self.hidden1 = nn.Linear(in_features=self.n_states, out_features=self.n_hidden_filters)
        init_weight(self.hidden1)
        self.hidden1.bias.data.zero_()
        self.hidden2 = nn.Linear(
            in_features=self.n_hidden_filters, out_features=self.n_hidden_filters
        )
        init_weight(self.hidden2)
        self.hidden2.bias.data.zero_()

        self.mu = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_actions)
        init_weight(self.mu, initializer="xavier uniform")
        self.mu.bias.data.zero_()

        self.log_std = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_actions)
        init_weight(self.log_std, initializer="xavier uniform")
        self.log_std.bias.data.zero_()

    def forward(self, states):
        x = F.relu(self.hidden1(states))
        x = F.relu(self.hidden2(x))

        mu = self.mu(x)
        log_std = self.log_std(x)
        std = log_std.clamp(min=-20, max=2).exp()
        dist = Normal(mu, std)
        return dist

    def sample_or_likelihood(self, states):
        dist = self(states)
        u = dist.rsample()
        action = torch.tanh(u)
        log_prob = dist.log_prob(value=u)
        log_prob -= torch.log(1 - action**2 + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        return (action * self.action_bounds[1]).clamp_(
            self.action_bounds[0], self.action_bounds[1]
        ), log_prob


def build_diayn_discriminator():
    # DIAYN's core novelty: the skill-discriminator q(z|s) providing the pseudo-reward
    # log q(z|s) - log p(z). n_states matches a small continuous-control obs (e.g. MountainCar).
    return Discriminator(n_states=8, n_skills=20, n_hidden_filters=64)


def example_input_diayn_discriminator():
    return torch.randn(4, 8)


def build_diayn_policy():
    # The skill-conditioned SAC actor: input is [state, one-hot skill] concatenated
    # (n_states + n_skills), matching how SACAgent constructs PolicyNetwork in agent.py.
    return PolicyNetwork(
        n_states=8 + 20, n_actions=2, action_bounds=[-1.0, 1.0], n_hidden_filters=64
    )


def example_input_diayn_policy():
    return torch.randn(4, 28)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    (
        "DIAYN Discriminator (skill classifier q(z|s))",
        build_diayn_discriminator,
        example_input_diayn_discriminator,
        2018,
        MENAGERIE_ZOO,
    ),
    (
        "DIAYN PolicyNetwork (skill-conditioned SAC actor)",
        build_diayn_policy,
        example_input_diayn_policy,
        2018,
        MENAGERIE_ZOO,
    ),
]
