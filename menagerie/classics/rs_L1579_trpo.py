# SOURCE: vendored from https://github.com/ikostrikov/pytorch-trpo @ master
# (models.py: Policy, Value, lines 1-40)
#
# TRPO (Schulman et al., ICML 2015, "Trust Region Policy Optimization",
# arXiv:1502.05477). ikostrikov/pytorch-trpo is the canonical, widely-cited
# clean PyTorch implementation of TRPO (conjugate-gradient + line-search KL
# constraint optimizer wrapped around a plain Gaussian-policy actor-critic
# pair) -- referenced across numerous TRPO reimplementations and course
# materials. The `Policy` (Gaussian continuous-action actor: tanh-MLP trunk +
# state-independent log-std parameter) and `Value` (tanh-MLP critic) network
# classes have no dependency beyond torch, so they are vendored verbatim.
# The conjugate-gradient/line-search TRPO step (`trpo.py`) is the
# training-loop optimizer logic, not part of the traced network architecture,
# and is omitted here.

import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


# ---- models.py (vendored verbatim) ----
class Policy(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(num_inputs, 64)
        self.affine2 = nn.Linear(64, 64)

        self.action_mean = nn.Linear(64, num_outputs)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)

        self.action_log_std = nn.Parameter(torch.zeros(1, num_outputs))

        self.saved_actions = []
        self.rewards = []
        self.final_value = 0

    def forward(self, x):
        x = torch.tanh(self.affine1(x))
        x = torch.tanh(self.affine2(x))

        action_mean = self.action_mean(x)
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)

        return action_mean, action_log_std, action_std


class Value(nn.Module):
    def __init__(self, num_inputs):
        super(Value, self).__init__()
        self.affine1 = nn.Linear(num_inputs, 64)
        self.affine2 = nn.Linear(64, 64)
        self.value_head = nn.Linear(64, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

    def forward(self, x):
        x = torch.tanh(self.affine1(x))
        x = torch.tanh(self.affine2(x))

        state_values = self.value_head(x)
        return state_values


# ---- end vendored models.py ----


class TRPONet(nn.Module):
    """Staging wrapper exercising the real Policy (Gaussian actor) + Value
    (critic) construction as a single traceable module: one forward pass
    produces the action distribution parameters and the state-value estimate,
    matching the exact tensors the TRPO training loop consumes each step."""

    def __init__(self, num_inputs=11, num_outputs=3):
        super().__init__()
        self.policy = Policy(num_inputs, num_outputs)
        self.value = Value(num_inputs)

    def forward(self, x):
        action_mean, action_log_std, action_std = self.policy(x)
        state_value = self.value(x)
        return action_mean, action_log_std, action_std, state_value


def build_trpo():
    return TRPONet(num_inputs=11, num_outputs=3)


def example_input_trpo():
    return (torch.randn(4, 11),)


MENAGERIE_ENTRIES = [
    ("TRPO", build_trpo, example_input_trpo, 2015, "vendored-pytorch"),
]
