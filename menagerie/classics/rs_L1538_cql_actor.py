# SOURCE: vendored from https://github.com/yihaosun1124/OfflineRL-Kit @ main (951302eed019)
# (queue-listed https://github.com/WeiChengTseng/CQL-pytorch contains no source files
#  (README + .gitignore only) -- OfflineRL-Kit ships a maintained, documented CQLPolicy
#  built from these real modules and is used instead)
# (offlinerlkit/nets/mlp.py + offlinerlkit/modules/actor_module.py +
#  offlinerlkit/modules/dist_module.py, minimal changes: merged into one file,
#  imports fixed)
"""CQL (Conservative Q-Learning, Kumar et al. 2020) policy network: a SAC-style
stochastic actor -- MLP backbone -> tanh-squashed diagonal Gaussian head. CQL's
distinguishing contribution is the conservative Q-regularizer applied during
*training* (a loss-term change, not an architecture change), so at the module
level CQL's real network is this ActorProb (identical to the SAC actor used as
CQLPolicy's base class in the reference repo)."""

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Dict, List, Union, Tuple, Optional


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Union[List[int], Tuple[int]],
        output_dim: Optional[int] = None,
        activation: nn.Module = nn.ReLU,
        dropout_rate: Optional[float] = None,
    ) -> None:
        super().__init__()
        hidden_dims = [input_dim] + list(hidden_dims)
        model = []
        for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
            model += [nn.Linear(in_dim, out_dim), activation()]
            if dropout_rate is not None:
                model += [nn.Dropout(p=dropout_rate)]

        self.output_dim = hidden_dims[-1]
        if output_dim is not None:
            model += [nn.Linear(hidden_dims[-1], output_dim)]
            self.output_dim = output_dim
        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class NormalWrapper(torch.distributions.Normal):
    def log_prob(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return self.mean


class TanhNormalWrapper(torch.distributions.Normal):
    def __init__(self, loc, scale, max_action):
        super().__init__(loc, scale)
        self._max_action = max_action

    def log_prob(self, action, raw_action=None):
        squashed_action = action / self._max_action
        if raw_action is None:
            raw_action = self.arctanh(squashed_action)
        log_prob = super().log_prob(raw_action).sum(-1, keepdim=True)
        eps = 1e-6
        log_prob = log_prob - torch.log(self._max_action * (1 - squashed_action.pow(2)) + eps).sum(
            -1, keepdim=True
        )
        return log_prob

    def mode(self):
        raw_action = self.mean
        action = self._max_action * torch.tanh(self.mean)
        return action, raw_action

    def arctanh(self, x):
        one_plus_x = (1 + x).clamp(min=1e-6)
        one_minus_x = (1 - x).clamp(min=1e-6)
        return 0.5 * torch.log(one_plus_x / one_minus_x)

    def rsample(self):
        raw_action = super().rsample()
        action = self._max_action * torch.tanh(raw_action)
        return action, raw_action


class TanhDiagGaussian(nn.Module):
    def __init__(
        self,
        latent_dim,
        output_dim,
        unbounded=False,
        conditioned_sigma=False,
        max_mu=1.0,
        sigma_min=-5.0,
        sigma_max=2.0,
    ):
        super().__init__()
        self.mu = nn.Linear(latent_dim, output_dim)
        self._c_sigma = conditioned_sigma
        if conditioned_sigma:
            self.sigma = nn.Linear(latent_dim, output_dim)
        else:
            self.sigma_param = nn.Parameter(torch.zeros(output_dim, 1))
        self._unbounded = unbounded
        self._max = max_mu
        self._sigma_min = sigma_min
        self._sigma_max = sigma_max

    def forward(self, logits):
        mu = self.mu(logits)
        if not self._unbounded:
            mu = self._max * torch.tanh(mu)
        if self._c_sigma:
            sigma = torch.clamp(self.sigma(logits), min=self._sigma_min, max=self._sigma_max).exp()
        else:
            shape = [1] * len(mu.shape)
            shape[1] = -1
            sigma = (self.sigma_param.view(shape) + torch.zeros_like(mu)).exp()
        return TanhNormalWrapper(mu, sigma, self._max)


class ActorProb(nn.Module):
    def __init__(self, backbone: nn.Module, dist_net: nn.Module, device: str = "cpu") -> None:
        super().__init__()

        self.device = torch.device(device)
        self.backbone = backbone.to(device)
        self.dist_net = dist_net.to(device)

    def forward(self, obs: Union[np.ndarray, torch.Tensor]):
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        logits = self.backbone(obs)
        dist = self.dist_net(logits)
        return dist.mode()[0]


def build_cql_actor():
    # Tiny MuJoCo-Hopper-style obs/action dims.
    obs_dim = 11
    action_dim = 3
    backbone = MLP(input_dim=obs_dim, hidden_dims=[32, 32])
    dist = TanhDiagGaussian(
        latent_dim=backbone.output_dim,
        output_dim=action_dim,
        unbounded=True,
        conditioned_sigma=True,
        max_mu=1.0,
    )
    return ActorProb(backbone, dist)


def example_input_cql_actor():
    return torch.randn(4, 11)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    (
        "CQL (Conservative Q-Learning)",
        build_cql_actor,
        example_input_cql_actor,
        2020,
        MENAGERIE_ZOO,
    ),
]
