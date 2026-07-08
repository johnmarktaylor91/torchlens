# SOURCE: vendored from xbpeng/MimicKit @ main
# https://raw.githubusercontent.com/xbpeng/MimicKit/main/mimickit/learning/ppo_model.py
# https://raw.githubusercontent.com/xbpeng/MimicKit/main/mimickit/learning/nets/fc_2layers_1024units.py
# https://raw.githubusercontent.com/xbpeng/MimicKit/main/mimickit/learning/distribution_gaussian_diag.py
# https://raw.githubusercontent.com/xbpeng/MimicKit/main/mimickit/util/torch_util.py (calc_layers_out_size)
#
# DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based Character
# Skills (Peng et al., SIGGRAPH 2018). The original xbpeng/DeepMimic repo's policy
# networks are TensorFlow 1.x (`learning/tf_agent.py`, `tf.contrib`/`tf.layers`,
# no PyTorch implementation in that repo). `xbpeng/MimicKit` is the same author's
# current, actively-maintained, official PyTorch reimplementation of DeepMimic's
# family of motion-imitation controllers (superseding the TF1.x DeepMimic repo);
# its `PPOModel` is the direct architectural descendant of DeepMimic's PPO actor-
# critic net (identical topology to `learning/nets/fc_2layers_1024units.py` in
# the original TF repo: a 2-layer [1024, 512] FC torso). `PPOModel` normally
# builds its actor/critic torsos via `net_builder.build_net()` and reads the
# action space from a `gymnasium` RL environment (`env.get_obs_space()` /
# `env.get_action_space()`); those two calls are pure environment-introspection
# glue, not architecture, so they are inlined below (tiny fixed obs/action sizes)
# to avoid pulling in a physics-sim environment. `fc_2layers_1024units.build_net`,
# `DistributionGaussianDiagBuilder`/`DistributionGaussianDiag`, and
# `calc_layers_out_size` are vendored verbatim; `PPOModel` itself is vendored
# verbatim, restructured only to take a plain `(obs_size, action_size)` config
# in place of the `(config, env)` pair (no dependency on `gymnasium` or a live
# environment instance).

import enum

import numpy as np
import torch

MENAGERIE_ZOO = "vendored-pytorch"


# ---- mimickit/learning/nets/fc_2layers_1024units.py (build_net, vendored verbatim) ----
def build_net(input_dict, activation):
    layer_sizes = [1024, 512]

    input_dim = np.sum([np.prod(curr_input.shape) for curr_input in input_dict.values()])

    in_size = input_dim
    layers = []
    for out_size in layer_sizes:
        curr_layer = torch.nn.Linear(in_size, out_size)
        torch.nn.init.zeros_(curr_layer.bias)

        layers.append(curr_layer)
        layers.append(activation())
        in_size = out_size

    net = torch.nn.Sequential(*layers)
    info = dict()

    return net, info


# ---- mimickit/util/torch_util.py (calc_layers_out_size, vendored verbatim) ----
def calc_layers_out_size(layers):
    modules = list(layers.modules())
    for m in reversed(modules):
        if hasattr(m, "out_features"):
            out_size = m.out_features
            break
    return out_size


# ---- mimickit/learning/distribution_gaussian_diag.py (vendored verbatim) ----
class StdType(enum.Enum):
    FIXED = 0
    CONSTANT = 1
    VARIABLE = 2


class DistributionGaussianDiagBuilder(torch.nn.Module):
    def __init__(self, in_size, out_size, std_type, init_std, init_output_scale=0.01):
        super().__init__()
        self._std_type = std_type

        self._build_params(in_size, out_size, init_std, init_output_scale)
        return

    def _build_params(self, in_size, out_size, init_std, init_output_scale):
        self._mean_net = torch.nn.Linear(in_size, out_size)
        torch.nn.init.uniform_(self._mean_net.weight, -init_output_scale, init_output_scale)
        torch.nn.init.zeros_(self._mean_net.bias)

        logstd = np.log(init_std)
        if self._std_type == StdType.FIXED:
            self._logstd_net = torch.nn.Parameter(
                torch.zeros(out_size, requires_grad=False, dtype=torch.float32), requires_grad=False
            )
            torch.nn.init.constant_(self._logstd_net, logstd)

        elif self._std_type == StdType.CONSTANT:
            self._logstd_net = torch.nn.Parameter(
                torch.zeros(out_size, requires_grad=True, dtype=torch.float32), requires_grad=True
            )
            torch.nn.init.constant_(self._logstd_net, logstd)

        elif self._std_type == StdType.VARIABLE:
            self._logstd_net = torch.nn.Linear(in_size, out_size)
            torch.nn.init.uniform_(self._logstd_net.weight, -init_output_scale, init_output_scale)
            torch.nn.init.constant_(self._logstd_net.bias, logstd)

        else:
            assert False, "Unsupported StdType: {}".format(self._std_type)
        return

    def forward(self, input):
        mean = self._mean_net(input)

        if self._std_type == StdType.FIXED or self._std_type == StdType.CONSTANT:
            logstd = torch.broadcast_to(self._logstd_net, mean.shape)
        elif self._std_type == StdType.VARIABLE:
            logstd = self._logstd_net(input)
        else:
            assert False, "Unsupported StdType: {}".format(self._std_type)

        dist = DistributionGaussianDiag(mean=mean, logstd=logstd)
        return dist


class DistributionGaussianDiag:
    def __init__(self, mean, logstd):
        self._mean = mean
        self._logstd = logstd
        self._std = torch.exp(self._logstd)
        self._dim = self._mean.shape[-1]
        return

    @property
    def stddev(self):
        return self._std

    @property
    def logstd(self):
        return self._logstd

    @property
    def mean(self):
        return self._mean

    @property
    def mode(self):
        return self._mean

    def sample(self):
        noise = torch.normal(torch.zeros_like(self._mean), torch.ones_like(self._std))
        x = self._mean + self._std * noise
        return x


# ---- mimickit/learning/ppo_model.py (PPOModel, vendored; (config, env) -> (obs_size, action_size)) ----
class PPOModel(torch.nn.Module):
    def __init__(self, obs_size, action_size, actor_init_output_scale=0.01, action_std=1.0):
        super().__init__()
        self._activation = torch.nn.ReLU

        self._build_actor(obs_size, action_size, actor_init_output_scale, action_std)
        self._build_critic(obs_size)
        return

    def eval_actor(self, obs):
        h = self._actor_layers(obs)
        a_dist = self._action_dist(h)
        return a_dist

    def eval_critic(self, obs):
        h = self._critic_layers(obs)
        val = self._critic_out(h)
        return val

    def forward(self, obs):
        a_dist = self.eval_actor(obs)
        val = self.eval_critic(obs)
        return a_dist.mean, a_dist.stddev, val

    def _build_actor(self, obs_size, action_size, actor_init_output_scale, action_std):
        input_dict = {"obs": torch.empty(obs_size)}
        self._actor_layers, _layers_info = build_net(input_dict, activation=self._activation)

        in_size = calc_layers_out_size(self._actor_layers)
        self._action_dist = DistributionGaussianDiagBuilder(
            in_size,
            action_size,
            std_type=StdType.CONSTANT,
            init_std=action_std,
            init_output_scale=actor_init_output_scale,
        )
        return

    def _build_critic(self, obs_size):
        input_dict = {"obs": torch.empty(obs_size)}
        self._critic_layers, _layers_info = build_net(input_dict, activation=self._activation)

        layers_out_size = calc_layers_out_size(self._critic_layers)
        self._critic_out = torch.nn.Linear(layers_out_size, 1)
        torch.nn.init.zeros_(self._critic_out.bias)
        return


# ---- staging wrapper ----
# Real DeepMimic humanoid observation is ~197-dim (joint positions/velocities);
# shrunk to a tiny obs/action size for a fast trace, same [1024, 512] FC torso
# architecture as the original.
_OBS_SIZE = (16,)
_ACTION_SIZE = 8


def build_deepmimic_ppo():
    model = PPOModel(obs_size=_OBS_SIZE, action_size=_ACTION_SIZE)
    model.eval()
    return model


def example_input_deepmimic_ppo():
    torch.manual_seed(0)
    batch = 4
    return torch.randn(batch, *_OBS_SIZE)


MENAGERIE_ENTRIES = [
    (
        "DeepMimic_PPO",
        "build_deepmimic_ppo",
        "example_input_deepmimic_ppo",
        2018,
        "vendored-pytorch",
    ),
]
