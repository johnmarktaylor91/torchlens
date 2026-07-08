# SOURCE: vendored from ikostrikov/pytorch-a2c-ppo-acktr-gail @ master
# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
# Files: a2c_ppo_acktr/model.py, a2c_ppo_acktr/distributions.py, a2c_ppo_acktr/utils.py
# (helper functions `get_render_func`/`get_vec_normalize` that pull in the gym-only
# `envs.py` module were dropped; they are training-loop utilities, not part of the
# ACKTR actor-critic architecture.) License: MIT.
"""ACKTR (Actor-Critic using Kronecker-Factored Trust Region) actor-critic network.

This is the canonical reference `Policy`/`CNNBase`/`MLPBase` network used by
Wu et al. (2017) "Scalable trust-region method for deep reinforcement learning
using Kronecker-factored approximation" as packaged in ikostrikov's widely used
PyTorch reference implementation shared across A2C/PPO/ACKTR/GAIL. ACKTR itself
is an optimizer-level contribution (K-FAC natural-gradient trust region); the
network architecture it optimizes is this CNN/MLP actor-critic torso, which is
what TorchLens traces here.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def init(module: nn.Module, weight_init, bias_init, gain: float = 1) -> nn.Module:
    """Apply an init scheme to a module's weight/bias (verbatim from utils.py)."""

    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class AddBias(nn.Module):
    """Learnable bias-only module used for the diagonal-Gaussian log-std (verbatim)."""

    def __init__(self, bias: torch.Tensor) -> None:
        super().__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)
        return x + bias


class FixedCategorical(torch.distributions.Categorical):
    """Categorical distribution with a torchlens/actor-critic-friendly interface."""

    def sample(self) -> torch.Tensor:
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions: torch.Tensor) -> torch.Tensor:
        return super().log_prob(actions.squeeze(-1)).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

    def mode(self) -> torch.Tensor:
        return self.probs.argmax(dim=-1, keepdim=True)


class FixedNormal(torch.distributions.Normal):
    """Diagonal Gaussian distribution with the same convenience interface."""

    def log_probs(self, actions: torch.Tensor) -> torch.Tensor:
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entropy(self) -> torch.Tensor:
        return super().entropy().sum(-1)

    def mode(self) -> torch.Tensor:
        return self.mean


class Categorical(nn.Module):
    """Discrete-action head producing a `FixedCategorical` distribution."""

    def __init__(self, num_inputs: int, num_outputs: int) -> None:
        super().__init__()

        def init_(m: nn.Module) -> nn.Module:
            return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=0.01)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x: torch.Tensor) -> FixedCategorical:
        x = self.linear(x)
        return FixedCategorical(logits=x)


class DiagGaussian(nn.Module):
    """Continuous-action head producing a `FixedNormal` distribution."""

    def __init__(self, num_inputs: int, num_outputs: int) -> None:
        super().__init__()

        def init_(m: nn.Module) -> nn.Module:
            return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x: torch.Tensor) -> FixedNormal:
        action_mean = self.fc_mean(x)
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()
        action_logstd = self.logstd(zeros)
        return FixedNormal(action_mean, action_logstd.exp())


class NNBase(nn.Module):
    """Shared recurrent-or-feedforward base used by both CNN and MLP torsos."""

    def __init__(self, recurrent: bool, recurrent_input_size: int, hidden_size: int) -> None:
        super().__init__()
        self._hidden_size = hidden_size
        self._recurrent = recurrent
        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if "bias" in name:
                    nn.init.constant_(param, 0)
                elif "weight" in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self) -> bool:
        return self._recurrent

    @property
    def output_size(self) -> int:
        return self._hidden_size


class CNNBase(NNBase):
    """Nature-DQN style CNN torso with actor and critic heads (verbatim)."""

    def __init__(self, num_inputs: int, recurrent: bool = False, hidden_size: int = 512) -> None:
        super().__init__(recurrent, hidden_size, hidden_size)

        def init_(m: nn.Module) -> nn.Module:
            return init(
                m,
                nn.init.orthogonal_,
                lambda x: nn.init.constant_(x, 0),
                nn.init.calculate_gain("relu"),
            )

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)),
            nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            init_(nn.Linear(32 * 7 * 7, hidden_size)),
            nn.ReLU(),
        )

        def init2_(m: nn.Module) -> nn.Module:
            return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))

        self.critic_linear = init2_(nn.Linear(hidden_size, 1))
        self.train()

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.main(inputs / 255.0)
        return self.critic_linear(x), x


class MLPBase(NNBase):
    """Two-layer tanh MLP torso with separate actor/critic branches (verbatim)."""

    def __init__(self, num_inputs: int, recurrent: bool = False, hidden_size: int = 64) -> None:
        super().__init__(recurrent, num_inputs, hidden_size)

        def init_(m: nn.Module) -> nn.Module:
            return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
        )
        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
        )
        self.critic_linear = init_(nn.Linear(hidden_size, 1))
        self.train()

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_critic = self.critic(inputs)
        hidden_actor = self.actor(inputs)
        return self.critic_linear(hidden_critic), hidden_actor


class ACKTRCNNPolicy(nn.Module):
    """CNN actor-critic policy trained with ACKTR's K-FAC trust region.

    Traces the CNNBase torso plus a categorical action head, mirroring the
    `Policy` wrapper's `act()` composition (`base` -> `dist`) as a plain
    forward pass so TorchLens can capture it without a gym `action_space`.
    """

    def __init__(self, num_inputs: int = 4, num_actions: int = 6) -> None:
        super().__init__()
        self.base = CNNBase(num_inputs)
        self.dist = Categorical(self.base.output_size, num_actions)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        value, actor_features = self.base(inputs)
        dist = self.dist(actor_features)
        return value, dist.logits


def build_acktr_cnn() -> nn.Module:
    """Build the ACKTR CNN actor-critic (Atari-style frame stack input)."""

    return ACKTRCNNPolicy(num_inputs=4, num_actions=6)


def example_input_acktr_cnn() -> torch.Tensor:
    """Return an example Atari-style frame stack input."""

    return torch.randint(0, 256, (1, 4, 84, 84), dtype=torch.float32)


MENAGERIE_ENTRIES = [
    (
        "ACKTR CNN actor-critic (Wu et al. 2017)",
        "build_acktr_cnn",
        "example_input_acktr_cnn",
        "2017",
        "DC",
    ),
]
