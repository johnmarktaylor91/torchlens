# SOURCE: vendored from https://github.com/rail-berkeley/rlkit @ ac45a9db24b8
#   (rlkit/torch/networks/mlp.py: Mlp; rlkit/torch/sac/policies/gaussian_policy.py: GaussianPolicy;
#    rlkit/torch/distributions.py: MultivariateDiagonalNormal; rlkit/torch/pytorch_util.py: fanin_init)
#
# Implicit Q-Learning (Kostrikov, Nair, Levine, ICLR 2022, "Offline Reinforcement Learning with
# Implicit Q-Learning"). The official release (ikostrikov/implicit_q_learning) is JAX; rlkit is
# the canonical, actively-maintained PyTorch reimplementation used by the same offline-RL
# community (rail-berkeley = the authors' own lab, Sergey Levine's group at UC Berkeley) --
# rlkit's `IQLTrainer` + `examples/iql/mujoco_finetune.py` config wires up exactly this
# `GaussianPolicy(hidden_sizes=[256, 256], std_architecture="values")` as the actor network,
# alongside `ConcatMlp`-based twin Q critics and an `Mlp`-based value network (same `Mlp` base).
# Vendored verbatim except: (a) sibling policy variants (TanhGaussianPolicy, GaussianMixture*,
# CNN variants) not used by the IQL config are omitted for brevity -- GaussianPolicy itself is
# untouched; (b) `PyTorchModule`/`TorchStochasticPolicy` framework base classes (pull in
# rlkit's launcher/exploration-policy machinery, irrelevant to the network architecture) are
# collapsed to plain `torch.nn.Module`; (c) `ptu.device`/`ptu.zeros` (rlkit's global device
# singleton) replaced with direct `torch`-equivalent calls on the parameter's own device.

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.distributions import Independent, Normal

MENAGERIE_ZOO = "vendored-pytorch"

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


def identity(x):
    return x


def fanin_init(tensor):
    # rlkit/torch/pytorch_util.py:fanin_init, verbatim
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = 1.0 / np.sqrt(fan_in)
    return tensor.data.uniform_(-bound, bound)


class MultivariateDiagonalNormal(Independent):
    # rlkit/torch/distributions.py:MultivariateDiagonalNormal, verbatim modulo the
    # TorchDistributionWrapper indirection (collapsed directly onto torch.distributions.Independent).
    def __init__(self, loc, scale_diag, reinterpreted_batch_ndims=1):
        super().__init__(
            Normal(loc, scale_diag), reinterpreted_batch_ndims=reinterpreted_batch_ndims
        )


class Mlp(nn.Module):
    # rlkit/torch/networks/mlp.py:Mlp, verbatim (PyTorchModule -> nn.Module).
    def __init__(
        self,
        hidden_sizes,
        output_size,
        input_size,
        init_w=3e-3,
        hidden_activation=torch.relu,
        output_activation=identity,
        hidden_init=fanin_init,
        b_init_value=0.0,
        layer_norm=False,
        layer_norm_kwargs=None,
    ):
        super().__init__()

        if layer_norm_kwargs is None:
            layer_norm_kwargs = dict()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layer_norm = layer_norm
        self.fcs = []
        self.layer_norms = []
        in_size = input_size

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

            if self.layer_norm:
                ln = nn.LayerNorm(next_size, **layer_norm_kwargs)
                self.__setattr__("layer_norm{}".format(i), ln)
                self.layer_norms.append(ln)

        self.last_fc = nn.Linear(in_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.fill_(0)

    def forward(self, input, return_preactivations=False):
        h = input
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            if self.layer_norm and i < len(self.fcs) - 1:
                h = self.layer_norms[i](h)
            h = self.hidden_activation(h)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)
        if return_preactivations:
            return output, preactivation
        else:
            return output


class ConcatMlp(Mlp):
    # rlkit/torch/networks/mlp.py:ConcatMlp, verbatim. Used for the IQL twin Q critics
    # (qf1/qf2 in rlkit.torch.sac.iql_trainer.IQLTrainer), which take (obs, action).
    def __init__(self, *args, dim=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=self.dim)
        return super().forward(flat_inputs, **kwargs)


class GaussianPolicy(Mlp):
    # rlkit/torch/sac/policies/gaussian_policy.py:GaussianPolicy, verbatim
    # (Mlp, TorchStochasticPolicy -> Mlp only; std_architecture="values" matches the real
    # rlkit examples/iql/mujoco_finetune.py policy_kwargs).
    def __init__(
        self,
        hidden_sizes,
        obs_dim,
        action_dim,
        std=None,
        init_w=1e-3,
        min_log_std=None,
        max_log_std=None,
        std_architecture="shared",
        **kwargs,
    ):
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            init_w=init_w,
            output_activation=torch.tanh,
            **kwargs,
        )
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        self.log_std = None
        self.std = std
        self.std_architecture = std_architecture
        if std is None:
            if self.std_architecture == "shared":
                last_hidden_size = obs_dim
                if len(hidden_sizes) > 0:
                    last_hidden_size = hidden_sizes[-1]
                self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
                self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
                self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
            elif self.std_architecture == "values":
                self.log_std_logits = nn.Parameter(torch.zeros(action_dim, requires_grad=True))
            else:
                raise ValueError(self.std_architecture)
        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX

    def forward(self, obs):
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        preactivation = self.last_fc(h)
        mean = self.output_activation(preactivation)
        if self.std is None:
            if self.std_architecture == "shared":
                log_std = torch.sigmoid(self.last_fc_log_std(h))
            elif self.std_architecture == "values":
                log_std = torch.sigmoid(self.log_std_logits)
            else:
                raise ValueError(self.std_architecture)
            log_std = self.min_log_std + log_std * (self.max_log_std - self.min_log_std)
            std = torch.exp(log_std)
        else:
            std = (
                torch.from_numpy(
                    np.array(
                        [
                            self.std,
                        ]
                    )
                )
                .float()
                .to(mean.device)
            )

        return MultivariateDiagonalNormal(mean, std)


class IQLPolicyDeterministic(nn.Module):
    """Deployable wrapper mirroring rlkit's real `MakeDeterministic` (see
    rlkit/torch/sac/policies/base.py), which is exactly how rlkit's own IQL eval loop invokes
    the actor: `dist.mean` for a MultivariateDiagonalNormal *is* `mle_estimate()`. This wrapper
    returns a tensor (the deterministic action) so the network can be traced end-to-end."""

    def __init__(self, hidden_sizes, obs_dim, action_dim, **kwargs):
        super().__init__()
        self.policy = GaussianPolicy(hidden_sizes, obs_dim, action_dim, **kwargs)

    def forward(self, obs):
        dist = self.policy(obs)
        return dist.mean


def build_iql():
    # Matches rlkit/examples/iql/mujoco_finetune.py policy_kwargs (real IQL mujoco config),
    # shrunk to a tiny obs/action dim for a fast trace.
    return IQLPolicyDeterministic(
        hidden_sizes=[64, 64],
        obs_dim=17,
        action_dim=6,
        max_log_std=0,
        min_log_std=-6,
        std_architecture="values",
    )


def example_input_iql():
    return torch.randn(4, 17)


MENAGERIE_ENTRIES = [
    ("IQL (Implicit Q-Learning, rlkit actor)", build_iql, example_input_iql, 2022, "REAL"),
]
