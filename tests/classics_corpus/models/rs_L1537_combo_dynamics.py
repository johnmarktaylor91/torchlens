# SOURCE: vendored from https://github.com/yihaosun1124/OfflineRL-Kit @ main (951302eed019)
# (offlinerlkit/nets/ensemble_linear.py + offlinerlkit/modules/dynamics_module.py,
#  minimal changes: merged into one file, imports fixed)
"""COMBO (Conservative Offline Model-Based Optimization, Yu et al. 2021) -- the
distinguishing architecture vs. plain model-free offline RL (CQL/SAC) is COMBO's
learned world model: an ensemble of MLP dynamics heads (EnsembleLinear layers,
vectorized over the ensemble dimension via torch.einsum) predicting next-state
mean/logvar from (obs, action). COMBO's actor/critic are the same ActorProb/Critic
MLP stack shared with CQL/SAC in this repo; the EnsembleDynamicsModel is what makes
COMBO model-based and is captured here."""

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Dict, List, Union, Tuple, Optional


class EnsembleLinear(nn.Module):
    def __init__(
        self, input_dim: int, output_dim: int, num_ensemble: int, weight_decay: float = 0.0
    ) -> None:
        super().__init__()

        self.num_ensemble = num_ensemble

        self.register_parameter(
            "weight", nn.Parameter(torch.zeros(num_ensemble, input_dim, output_dim))
        )
        self.register_parameter("bias", nn.Parameter(torch.zeros(num_ensemble, 1, output_dim)))

        nn.init.trunc_normal_(self.weight, std=1 / (2 * input_dim**0.5))

        self.register_parameter("saved_weight", nn.Parameter(self.weight.detach().clone()))
        self.register_parameter("saved_bias", nn.Parameter(self.bias.detach().clone()))

        self.weight_decay = weight_decay

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        bias = self.bias

        if len(x.shape) == 2:
            x = torch.einsum("ij,bjk->bik", x, weight)
        else:
            x = torch.einsum("bij,bjk->bik", x, weight)

        x = x + bias

        return x

    def load_save(self) -> None:
        self.weight.data.copy_(self.saved_weight.data)
        self.bias.data.copy_(self.saved_bias.data)

    def update_save(self, indexes: List[int]) -> None:
        self.saved_weight.data[indexes] = self.weight.data[indexes]
        self.saved_bias.data[indexes] = self.bias.data[indexes]

    def get_decay_loss(self) -> torch.Tensor:
        decay_loss = self.weight_decay * (0.5 * ((self.weight**2).sum()))
        return decay_loss


class Swish(nn.Module):
    def __init__(self) -> None:
        super(Swish, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * torch.sigmoid(x)
        return x


def soft_clamp(
    x: torch.Tensor, _min: Optional[torch.Tensor] = None, _max: Optional[torch.Tensor] = None
) -> torch.Tensor:
    # clamp tensor values while mataining the gradient
    if _max is not None:
        x = _max - F.softplus(_max - x)
    if _min is not None:
        x = _min + F.softplus(x - _min)
    return x


class EnsembleDynamicsModel(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Union[List[int], Tuple[int]],
        num_ensemble: int = 7,
        num_elites: int = 5,
        activation: nn.Module = Swish,
        weight_decays: Optional[Union[List[float], Tuple[float]]] = None,
        with_reward: bool = True,
        device: str = "cpu",
    ) -> None:
        super().__init__()

        self.num_ensemble = num_ensemble
        self.num_elites = num_elites
        self._with_reward = with_reward
        self.device = torch.device(device)

        self.activation = activation()

        assert len(weight_decays) == (len(hidden_dims) + 1)

        module_list = []
        hidden_dims = [obs_dim + action_dim] + list(hidden_dims)
        if weight_decays is None:
            weight_decays = [0.0] * (len(hidden_dims) + 1)
        for in_dim, out_dim, weight_decay in zip(
            hidden_dims[:-1], hidden_dims[1:], weight_decays[:-1]
        ):
            module_list.append(EnsembleLinear(in_dim, out_dim, num_ensemble, weight_decay))
        self.backbones = nn.ModuleList(module_list)

        self.output_layer = EnsembleLinear(
            hidden_dims[-1], 2 * (obs_dim + self._with_reward), num_ensemble, weight_decays[-1]
        )

        self.register_parameter(
            "max_logvar",
            nn.Parameter(torch.ones(obs_dim + self._with_reward) * 0.5, requires_grad=True),
        )
        self.register_parameter(
            "min_logvar",
            nn.Parameter(torch.ones(obs_dim + self._with_reward) * -10, requires_grad=True),
        )

        self.register_parameter(
            "elites",
            nn.Parameter(torch.tensor(list(range(0, self.num_elites))), requires_grad=False),
        )

        self.to(self.device)

    def forward(self, obs_action: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        obs_action = torch.as_tensor(obs_action, dtype=torch.float32).to(self.device)
        output = obs_action
        for layer in self.backbones:
            output = self.activation(layer(output))
        mean, logvar = torch.chunk(self.output_layer(output), 2, dim=-1)
        logvar = soft_clamp(logvar, self.min_logvar, self.max_logvar)
        return mean, logvar

    def load_save(self) -> None:
        for layer in self.backbones:
            layer.load_save()
        self.output_layer.load_save()

    def update_save(self, indexes: List[int]) -> None:
        for layer in self.backbones:
            layer.update_save(indexes)
        self.output_layer.update_save(indexes)

    def get_decay_loss(self) -> torch.Tensor:
        decay_loss = 0
        for layer in self.backbones:
            decay_loss += layer.get_decay_loss()
        decay_loss += self.output_layer.get_decay_loss()
        return decay_loss

    def set_elites(self, indexes: List[int]) -> None:
        assert len(indexes) <= self.num_ensemble and max(indexes) < self.num_ensemble
        self.register_parameter("elites", nn.Parameter(torch.tensor(indexes), requires_grad=False))

    def random_elite_idxs(self, batch_size: int) -> np.ndarray:
        idxs = np.random.choice(self.elites.data.cpu().numpy(), size=batch_size)
        return idxs


def build_combo_dynamics():
    # Tiny MuJoCo-Hopper-style obs/action dims, small ensemble.
    return EnsembleDynamicsModel(
        obs_dim=11,
        action_dim=3,
        hidden_dims=[32, 32],
        num_ensemble=3,
        num_elites=2,
        weight_decays=[2.5e-5, 5e-5, 1e-4],
    )


def example_input_combo_dynamics():
    # forward() takes a single (obs, action) concatenated tensor.
    return torch.randn(4, 11 + 3)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    (
        "COMBO (Conservative Model-Based Offline RL)",
        build_combo_dynamics,
        example_input_combo_dynamics,
        2021,
        MENAGERIE_ZOO,
    ),
]
