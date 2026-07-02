# SOURCE: vendored from younggyoseo/pytorch-acer @ master
# https://github.com/younggyoseo/pytorch-acer
# Files: model.py, distributions.py, utils.py
# (helper functions `get_render_func`/`get_vec_normalize` that pull in the gym-only
# `envs.py` module were dropped; they are training-loop utilities, not part of the
# ACER actor-critic architecture.)
"""ACER (Actor-Critic with Experience Replay, Wang et al. 2017) actor-critic network.

Standalone PyTorch reference implementation of the ACER network head: a shared
CNN/MLP torso (same Nature-DQN convolutional torso family used across the
ikostrikov A2C/PPO/ACKTR lineage) feeding a per-action Q-value head and a
categorical/Gaussian policy head. ACER's value is `(q_value * dist.probs).sum(1)`,
matching the paper's dueling-style combination of a Q-network and a stochastic
policy for off-policy actor-critic with Retrace and trust-region correction.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def init(module: nn.Module, weight_init, bias_init, gain: float = 1) -> nn.Module:
    """Apply an init scheme to a module's weight/bias (verbatim from utils.py)."""

    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def init_normc_(weight: torch.Tensor, gain: float = 1) -> None:
    """Column-normalized init used for the MLP torso (verbatim from utils.py).

    https://github.com/openai/baselines/blob/master/baselines/common/tf_util.py#L87
    """

    weight.normal_(0, 1)
    weight *= gain / torch.sqrt(weight.pow(2).sum(1, keepdim=True))


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


class Categorical(nn.Module):
    """Discrete-action head producing a `torch.distributions.Categorical`."""

    def __init__(self, num_inputs: int, num_outputs: int) -> None:
        super().__init__()

        def init_(m: nn.Module) -> nn.Module:
            return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=0.01)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x: torch.Tensor) -> torch.distributions.Categorical:
        x = self.linear(x)
        return torch.distributions.Categorical(logits=x)


class DiagGaussian(nn.Module):
    """Continuous-action head producing a `torch.distributions.Normal`."""

    def __init__(self, num_inputs: int, num_outputs: int) -> None:
        super().__init__()

        def init_(m: nn.Module) -> nn.Module:
            return init(m, init_normc_, lambda x: nn.init.constant_(x, 0))

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x: torch.Tensor) -> torch.distributions.Normal:
        action_mean = self.fc_mean(x)
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()
        action_logstd = self.logstd(zeros)
        return torch.distributions.Normal(action_mean, action_logstd.exp())


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
    """Nature-DQN style CNN torso with a per-action Q-value head (verbatim)."""

    def __init__(
        self, num_inputs: int, num_outputs: int, recurrent: bool = False, hidden_size: int = 512
    ) -> None:
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

        self.critic_linear = init2_(nn.Linear(hidden_size, num_outputs))
        self.train()

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.main(inputs / 255.0)
        return self.critic_linear(x), x


class ACERCNNPolicy(nn.Module):
    """CNN actor-critic policy trained with ACER (Wang et al. 2017).

    Traces the CNNBase torso plus the categorical policy head and combines the
    per-action Q-values with policy probabilities into the scalar value
    estimate, mirroring `Policy.act()`'s `(q_value * dist.probs).sum(1)`.
    """

    def __init__(self, num_inputs: int = 4, num_actions: int = 6) -> None:
        super().__init__()
        self.base = CNNBase(num_inputs, num_actions)
        self.dist = Categorical(self.base.output_size, num_actions)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_value, actor_features = self.base(inputs)
        dist = self.dist(actor_features)
        value = (q_value * dist.probs).sum(1, keepdim=True)
        return dist.probs, value, q_value


def build_acer_cnn() -> nn.Module:
    """Build the ACER CNN actor-critic (Atari-style frame stack input)."""

    return ACERCNNPolicy(num_inputs=4, num_actions=6)


def example_input_acer_cnn() -> torch.Tensor:
    """Return an example Atari-style frame stack input."""

    return torch.randint(0, 256, (1, 4, 84, 84), dtype=torch.float32)


MENAGERIE_ENTRIES = [
    (
        "ACER CNN actor-critic (Wang et al. 2017)",
        "build_acer_cnn",
        "example_input_acer_cnn",
        "2017",
        "DC",
    ),
]
