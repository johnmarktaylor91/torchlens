# FAITHFUL PORT of https://github.com/PCCproject/PCC-RL @ master (original framework:
# TensorFlow 1.x via `stable_baselines` PPO1)
#
# PCC-RL (the "Aurora" congestion-control RL agent, Jay, Rotman, Godfrey, Schapira &
# Tammar, "A Deep Reinforcement Learning Perspective on Internet Congestion Control",
# ICML 2019) itself only ships the Gym environment (`src/gym/network_sim.py`) -- the
# actual neural network is the third-party `stable_baselines` (TF1.x) actor-critic MLP
# policy that `src/gym/stable_solve.py` instantiates and trains:
#
#     from stable_baselines.common.policies import FeedForwardPolicy
#     class MyMlpPolicy(FeedForwardPolicy):
#         def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch,
#                      reuse=False, **_kwargs):
#             super().__init__(..., net_arch=[{"pi": arch, "vf": arch}],
#                               feature_extraction="mlp", **_kwargs)
#     model = PPO1(MyMlpPolicy, env, ...)   # default --arch "32,16"
#
# This IS a concrete, fully-specified architecture (just expressed via a library
# default rather than a bespoke nn.Module in the PCC-RL repo itself), so it is ported
# faithfully rather than reimplemented from a paper description. Two pieces are
# transcribed verbatim from stable_baselines (TF1.x, https://github.com/hill-a/
# stable-baselines, tag v2.10.2):
#   1. `mlp_extractor` (stable_baselines/common/policies.py) -- builds separate
#      (non-shared, since net_arch=[{"pi":[...], "vf":[...]}] has zero leading shared
#      ints) tanh-activated pi/vf towers.
#   2. `DiagGaussianProbabilityDistributionType.proba_distribution_from_latent`
#      (stable_baselines/common/distributions.py) -- PCC-RL's action space is a
#      continuous 1-D `gym.spaces.Box` (send-rate delta, USE_CWND=False in
#      network_sim.py), so PPO1 uses the diagonal-Gaussian policy head: a linear
#      `pi` layer producing the Gaussian mean (paired with a state-independent
#      learned log-std parameter, not consumed by a forward pass) plus a separate
#      linear `vf` value-function head off the value tower.
from __future__ import annotations

import torch
import torch.nn as nn

MENAGERIE_ZOO = "ported-pytorch"


class MlpExtractor(nn.Module):
    """Port of stable_baselines.common.policies.mlp_extractor with net_arch=[{"pi":
    arch, "vf": arch}] (PCC-RL's stable_solve.py default --arch "32,16" -> no shared
    layers, two parallel tanh MLP towers)."""

    def __init__(self, obs_dim, arch=(32, 16)):
        super().__init__()
        pi_layers = []
        vf_layers = []
        in_dim = obs_dim
        for layer_size in arch:
            pi_layers += [nn.Linear(in_dim, layer_size), nn.Tanh()]
            vf_layers += [nn.Linear(in_dim, layer_size), nn.Tanh()]
            in_dim = layer_size
        self.pi_tower = nn.Sequential(*pi_layers)
        self.vf_tower = nn.Sequential(*vf_layers)

    def forward(self, flat_observations):
        return self.pi_tower(flat_observations), self.vf_tower(flat_observations)


class AuroraMlpPolicy(nn.Module):
    """Port of `MyMlpPolicy(FeedForwardPolicy)` (stable_solve.py) built on
    stable_baselines' FeedForwardPolicy(feature_extraction="mlp") + the
    DiagGaussianProbabilityDistributionType policy head (continuous 1-D Box action
    space matching PCC-RL's send-rate action, USE_CWND=False)."""

    def __init__(self, obs_dim, action_dim=1, arch=(32, 16)):
        super().__init__()
        self.mlp_extractor = MlpExtractor(obs_dim, arch=arch)
        vf_hidden = arch[-1] if arch else obs_dim
        pi_hidden = arch[-1] if arch else obs_dim

        # FeedForwardPolicy: self._value_fn = linear(vf_latent, 'vf', 1)
        self.value_fn = nn.Linear(vf_hidden, 1)

        # DiagGaussianProbabilityDistributionType.proba_distribution_from_latent:
        # mean = linear(pi_latent, 'pi', size); logstd is a free (state-independent)
        # parameter, not part of the forward computation graph from pi_latent.
        self.pi_mean = nn.Linear(pi_hidden, action_dim)
        self.pi_logstd = nn.Parameter(torch.zeros(1, action_dim))
        # q_values = linear(vf_latent, 'q', size) -- PPO1's unused Q-head slot,
        # present in every FeedForwardPolicy regardless of algorithm.
        self.q_value = nn.Linear(vf_hidden, action_dim)

    def forward(self, obs):
        pi_latent, vf_latent = self.mlp_extractor(obs)
        value = self.value_fn(vf_latent)
        action_mean = self.pi_mean(pi_latent)
        action_logstd = self.pi_logstd.expand_as(action_mean)
        q_value = self.q_value(vf_latent)
        return action_mean, action_logstd, value, q_value


# --- staging harness ---
# obs_dim = history_len * n_features; network_sim.py defaults history_len=10 with a
# small fixed per-step feature vector (send rate, throughput, latency, loss, etc.).
def build_aurora_pcc_mlp_policy():
    return AuroraMlpPolicy(obs_dim=30, action_dim=1, arch=(32, 16))


def example_input_aurora_pcc_mlp_policy():
    torch.manual_seed(0)
    batch_size = 4
    obs = torch.randn(batch_size, 30)
    return (obs,)


MENAGERIE_ENTRIES = [
    (
        "Aurora_PCC_RL_MlpPolicy",
        "build_aurora_pcc_mlp_policy",
        "example_input_aurora_pcc_mlp_policy",
        "2019",
        "ported-pytorch",
    ),
]
