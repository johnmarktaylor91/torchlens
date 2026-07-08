# SOURCE: vendored from https://github.com/cyanrain7/TRPO-in-MARL @ master
# (algorithms/actor_critic.py + algorithms/utils/{util,mlp,act,distributions}.py,
# unmodified architecture; only added lightweight args/space shims so the real
# Actor/Critic modules can be constructed and called outside the full training
# runner. PKU-MARL/TRPO-PPO-in-MARL is a fork of this repo with the same code.)
"""HATRPO / HAPPO (Heterogeneous-Agent Trust Region / Proximal Policy
Optimisation, Kuba et al. 2021, "Trust Region Policy Optimisation in
Multi-Agent Reinforcement Learning") actor-critic network as shipped in the
reference implementation: an MLP feature base (LayerNorm + Tanh/ReLU stack)
feeding a diagonal-Gaussian continuous-action head (Actor) and a scalar value
head (Critic). Both networks share the same MLPBase trunk architecture; the
distinguishing HATRPO/HAPPO machinery (sequential trust-region updates across
agents) lives in the trainer, not the network, so the network module is
identical between the two algorithms.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import torch
import torch.nn as nn


# ---- algorithms/utils/util.py ----
def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def get_clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def check(x):
    return x


# ---- algorithms/utils/mlp.py ----
class MLPLayer(nn.Module):
    def __init__(self, input_dim, hidden_size, layer_N, use_orthogonal, use_ReLU):
        super(MLPLayer, self).__init__()
        self._layer_N = layer_N

        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(["tanh", "relu"][use_ReLU])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.fc1 = nn.Sequential(
            init_(nn.Linear(input_dim, hidden_size)), active_func, nn.LayerNorm(hidden_size)
        )
        self.fc2 = nn.ModuleList(
            [
                nn.Sequential(
                    init_(nn.Linear(hidden_size, hidden_size)),
                    active_func,
                    nn.LayerNorm(hidden_size),
                )
                for i in range(self._layer_N)
            ]
        )

    def forward(self, x):
        x = self.fc1(x)
        for i in range(self._layer_N):
            x = self.fc2[i](x)
        return x


class MLPBase(nn.Module):
    def __init__(self, args, obs_shape, cat_self=True, attn_internal=False):
        super(MLPBase, self).__init__()

        self._use_feature_normalization = args.use_feature_normalization
        self._use_orthogonal = args.use_orthogonal
        self._use_ReLU = args.use_ReLU
        self._stacked_frames = args.stacked_frames
        self._layer_N = args.layer_N
        self.hidden_size = args.hidden_size

        obs_dim = obs_shape[0]

        if self._use_feature_normalization:
            self.feature_norm = nn.LayerNorm(obs_dim)

        self.mlp = MLPLayer(
            obs_dim, self.hidden_size, self._layer_N, self._use_orthogonal, self._use_ReLU
        )

    def forward(self, x):
        if self._use_feature_normalization:
            x = self.feature_norm(x)

        x = self.mlp(x)

        return x


# ---- algorithms/utils/distributions.py ----
class FixedNormal(torch.distributions.Normal):
    def log_probs(self, actions):
        return super().log_prob(actions)

    def mode(self):
        return self.mean


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01, args=None):
        super(DiagGaussian, self).__init__()

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        if args is not None:
            self.std_x_coef = args.std_x_coef
            self.std_y_coef = args.std_y_coef
        else:
            self.std_x_coef = 1.0
            self.std_y_coef = 0.5
        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        log_std = torch.ones(num_outputs) * self.std_x_coef
        self.log_std = torch.nn.Parameter(log_std)

    def forward(self, x, available_actions=None):
        action_mean = self.fc_mean(x)
        action_std = torch.sigmoid(self.log_std / self.std_x_coef) * self.std_y_coef
        return FixedNormal(action_mean, action_std)


# ---- algorithms/utils/act.py (Box action space only, matches example input) ----
class ACTLayer(nn.Module):
    def __init__(self, action_space, inputs_dim, use_orthogonal, gain, args=None):
        super(ACTLayer, self).__init__()
        self.mixed_action = False
        self.multi_discrete = False
        self.action_type = action_space.__class__.__name__
        action_dim = action_space.shape[0]
        self.action_out = DiagGaussian(inputs_dim, action_dim, use_orthogonal, gain, args)

    def forward(self, x, available_actions=None, deterministic=False):
        action_logits = self.action_out(x, available_actions)
        actions = action_logits.mode() if deterministic else action_logits.sample()
        action_log_probs = action_logits.log_probs(actions)
        return actions, action_log_probs


# ---- algorithms/actor_critic.py ----
class Actor(nn.Module):
    """HATRPO/HAPPO actor: MLP trunk + diagonal-Gaussian continuous action head."""

    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(Actor, self).__init__()
        self.hidden_size = args.hidden_size
        self.args = args
        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = False
        self._use_recurrent_policy = False
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = obs_space.shape
        self.base = MLPBase(args, obs_shape)

        self.act = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain, args)

        self.to(device)

    def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        actor_features = self.base(obs)

        actions, action_log_probs = self.act(actor_features, available_actions, deterministic)

        return actions, action_log_probs, rnn_states


class Critic(nn.Module):
    """HATRPO/HAPPO critic: MLP trunk + scalar value head."""

    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        super(Critic, self).__init__()
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self._use_naive_recurrent_policy = False
        self._use_recurrent_policy = False
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        cent_obs_shape = cent_obs_space.shape
        self.base = MLPBase(args, cent_obs_shape)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        self.v_out = init_(nn.Linear(self.hidden_size, 1))

        self.to(device)

    def forward(self, cent_obs, rnn_states, masks):
        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        critic_features = self.base(cent_obs)
        values = self.v_out(critic_features)

        return values, rnn_states


# ---- staging scaffolding (args/space shims; no architecture changes) ----
_OBS_DIM = 18
_CENT_OBS_DIM = 24
_ACTION_DIM = 5
_HIDDEN = 32
_N_AGENTS = 3


class _BoxSpace:
    """Minimal duck-typed stand-in for gym.spaces.Box: only `.shape` is read
    by Actor/Critic/ACTLayer."""

    def __init__(self, dim):
        self.shape = (dim,)


def _make_args():
    return SimpleNamespace(
        hidden_size=_HIDDEN,
        gain=0.01,
        use_orthogonal=True,
        use_policy_active_masks=False,
        recurrent_N=1,
        use_feature_normalization=True,
        use_ReLU=True,
        stacked_frames=1,
        layer_N=1,
        std_x_coef=1.0,
        std_y_coef=0.5,
    )


def build_hatrpo_actor():
    args = _make_args()
    obs_space = _BoxSpace(_OBS_DIM)
    action_space = _BoxSpace(_ACTION_DIM)
    return Actor(args, obs_space, action_space)


def example_input_hatrpo_actor():
    obs = torch.randn(4, _OBS_DIM)
    rnn_states = torch.zeros(4, 1, _HIDDEN)
    masks = torch.ones(4, 1)
    return (obs, rnn_states, masks)


def build_hatrpo_critic():
    args = _make_args()
    cent_obs_space = _BoxSpace(_CENT_OBS_DIM)
    return Critic(args, cent_obs_space)


def example_input_hatrpo_critic():
    cent_obs = torch.randn(4, _CENT_OBS_DIM)
    rnn_states = torch.zeros(4, 1, _HIDDEN)
    masks = torch.ones(4, 1)
    return (cent_obs, rnn_states, masks)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    (
        "HATRPO/HAPPO Actor (MLP + diagonal Gaussian)",
        "build_hatrpo_actor",
        "example_input_hatrpo_actor",
        2021,
        MENAGERIE_ZOO,
    ),
    (
        "HATRPO/HAPPO Critic (MLP + value head)",
        "build_hatrpo_critic",
        "example_input_hatrpo_critic",
        2021,
        MENAGERIE_ZOO,
    ),
]
