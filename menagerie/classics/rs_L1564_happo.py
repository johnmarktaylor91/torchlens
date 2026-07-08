# SOURCE: vendored from PKU-MARL/TRPO-PPO-in-MARL @ master (a fork of the official
# cyanrain7/TRPO-in-MARL repo, the companion code release for HAPPO/HATRPO:
# "Trust Region Policy Optimisation in Multi-Agent Reinforcement Learning",
# Kuba, Chen, Wen, Wang, Zhang, Mguni, Wang, Yang; ICLR 2022, arXiv:2109.11251)
# https://github.com/PKU-MARL/TRPO-PPO-in-MARL (algorithms/actor_critic.py +
# algorithms/utils/{mlp,rnn,act,distributions,util}.py + utils/util.py)
#
# HAPPO's sequential-update multi-agent training scheme (the paper's actual contribution)
# lives in algorithms/happo_trainer.py and is orthogonal to the per-agent network
# architecture; the Actor/Critic networks themselves are the standard MAPPO-style
# feedforward (or optionally recurrent) policy/value MLPs, vendored verbatim below
# (imports/paths trimmed to a single file; only base libs -- torch, numpy -- required).
# The CNN observation-encoder branch (algorithms/utils/cnn.py, used only when
# obs_space is image-shaped) and the continuous/MultiDiscrete/mixed action-space
# branches of ACTLayer are omitted here since this staging module targets the common
# vector-observation, Discrete-action-space configuration (e.g. SMAC); the vendored
# Discrete-only ACTLayer path is unmodified real code, just with the other
# action-space branches' dead code trimmed for a single-purpose staging file.
import copy
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


# ---- algorithms/utils/util.py ----
def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def check(input):
    output = torch.from_numpy(input) if type(input) == np.ndarray else input
    return output


# ---- utils/util.py (top-level get_shape_from_obs_space, distinct from the
# algorithms/utils/util.py module above) ----
def get_shape_from_obs_space(obs_space):
    if obs_space.__class__.__name__ == "Box":
        obs_shape = obs_space.shape
    elif obs_space.__class__.__name__ == "list":
        obs_shape = obs_space
    else:
        raise NotImplementedError
    return obs_shape


# ---- algorithms/utils/distributions.py ----
class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return super().log_prob(actions.squeeze(-1)).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
        super(Categorical, self).__init__()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x, available_actions=None):
        x = self.linear(x)
        if available_actions is not None:
            x[available_actions == 0] = -1e10
        return FixedCategorical(logits=x)


# ---- algorithms/utils/act.py (Discrete-action-space path) ----
class ACTLayer(nn.Module):
    """
    MLP Module to compute actions.
    :param action_space: (gym.Space) action space.
    :param inputs_dim: (int) dimension of network input.
    :param use_orthogonal: (bool) whether to use orthogonal initialization.
    :param gain: (float) gain of the output layer of the network.
    """

    def __init__(self, action_space, inputs_dim, use_orthogonal, gain, args=None):
        super(ACTLayer, self).__init__()
        self.mixed_action = False
        self.multi_discrete = False
        self.action_type = action_space.__class__.__name__
        if action_space.__class__.__name__ == "Discrete":
            action_dim = action_space.n
            self.action_out = Categorical(inputs_dim, action_dim, use_orthogonal, gain)
        else:
            raise NotImplementedError(
                "This staging module vendors only the Discrete action-space path "
                "of the real ACTLayer; see the repo for Box/MultiBinary/MultiDiscrete/mixed."
            )

    def forward(self, x, available_actions=None, deterministic=False):
        action_logits = self.action_out(x, available_actions)
        actions = action_logits.mode() if deterministic else action_logits.sample()
        action_log_probs = action_logits.log_probs(actions)
        return actions, action_log_probs

    def evaluate_actions(self, x, action, available_actions=None, active_masks=None):
        action_logits = self.action_out(x, available_actions)
        action_log_probs = action_logits.log_probs(action)
        if active_masks is not None:
            if self.action_type == "Discrete":
                dist_entropy = (
                    action_logits.entropy() * active_masks.squeeze(-1)
                ).sum() / active_masks.sum()
            else:
                dist_entropy = (action_logits.entropy() * active_masks).sum() / active_masks.sum()
        else:
            dist_entropy = action_logits.entropy().mean()
        return action_log_probs, dist_entropy


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


# ---- algorithms/utils/rnn.py ----
class RNNLayer(nn.Module):
    def __init__(self, inputs_dim, outputs_dim, recurrent_N, use_orthogonal):
        super(RNNLayer, self).__init__()
        self._recurrent_N = recurrent_N
        self._use_orthogonal = use_orthogonal

        self.rnn = nn.GRU(inputs_dim, outputs_dim, num_layers=self._recurrent_N)
        for name, param in self.rnn.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                if self._use_orthogonal:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
        self.norm = nn.LayerNorm(outputs_dim)

    def forward(self, x, hxs, masks):
        # Real repo also handles the T*N-flattened multi-step-with-resets branch;
        # trimmed here to the single-step branch (x.size(0) == hxs.size(0)) actually
        # exercised on a live forward pass, matching the vendored-module contract of
        # tracing the real code rather than adding untraced dead branches.
        x, hxs = self.rnn(
            x.unsqueeze(0),
            (hxs * masks.repeat(1, self._recurrent_N).unsqueeze(-1)).transpose(0, 1).contiguous(),
        )
        x = x.squeeze(0)
        hxs = hxs.transpose(0, 1)
        x = self.norm(x)
        return x, hxs


# ---- algorithms/actor_critic.py ----
class Actor(nn.Module):
    """
    Actor network class for HAPPO. Outputs actions given observations.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param obs_space: (gym.Space) observation space.
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(Actor, self).__init__()
        self.hidden_size = args.hidden_size
        self.args = args
        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = get_shape_from_obs_space(obs_space)
        base = MLPBase
        self.base = base(args, obs_shape)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(
                self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal
            )

        self.act = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain, args)

        self.to(device)

    def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
        """
        Compute actions from the given inputs.
        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        actions, action_log_probs = self.act(actor_features, available_actions, deterministic)

        return actions, action_log_probs, rnn_states


class Critic(nn.Module):
    """
    Critic network class for HAPPO. Outputs value function predictions given
    centralized input (HAPPO) or local observations (IPPO).
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param cent_obs_space: (gym.Space) (centralized) observation space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        super(Critic, self).__init__()
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        base = MLPBase
        self.base = base(args, cent_obs_shape)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(
                self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal
            )

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        self.v_out = init_(nn.Linear(self.hidden_size, 1))

        self.to(device)

    def forward(self, cent_obs, rnn_states, masks):
        """
        Compute value function predictions from the given inputs.
        :return values: (torch.Tensor) value function predictions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        critic_features = self.base(cent_obs)
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)
        values = self.v_out(critic_features)

        return values, rnn_states


# ---- staging wrappers (tl.trace needs a single nn.Module whose forward returns
# tensors; the real Actor/Critic.forward already do, they just also thread through
# rnn_states as a 3rd return value alongside the RNN-disabled config used here) ----
class _HappoActorWrapper(nn.Module):
    def __init__(self, actor: Actor):
        super().__init__()
        self.actor = actor

    def forward(self, obs, rnn_states, masks):
        actions, action_log_probs, _ = self.actor(obs, rnn_states, masks)
        return actions, action_log_probs


class _HappoCriticWrapper(nn.Module):
    def __init__(self, critic: Critic):
        super().__init__()
        self.critic = critic

    def forward(self, cent_obs, rnn_states, masks):
        values, _ = self.critic(cent_obs, rnn_states, masks)
        return values


class _DiscreteSpace:
    def __init__(self, n: int):
        self.n = n
        self.__class__ = type("Discrete", (object,), {})


class _BoxSpace:
    def __init__(self, shape):
        self.shape = shape
        self.__class__ = type("Box", (object,), {})


def _happo_args() -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=16,
        gain=0.01,
        use_orthogonal=True,
        use_policy_active_masks=False,
        use_naive_recurrent_policy=False,
        use_recurrent_policy=False,
        recurrent_N=1,
        use_feature_normalization=True,
        use_ReLU=True,
        stacked_frames=1,
        layer_N=1,
        algorithm_name="happo",
    )


def build_happo_actor() -> nn.Module:
    args = _happo_args()
    obs_space = _BoxSpace((8,))
    act_space = _DiscreteSpace(4)
    return _HappoActorWrapper(Actor(args, obs_space, act_space))


def example_input_happo_actor():
    obs = torch.randn(2, 8)
    rnn_states = torch.zeros(2, 1, 16)
    masks = torch.ones(2, 1)
    return (obs, rnn_states, masks)


def build_happo_critic() -> nn.Module:
    args = _happo_args()
    cent_obs_space = _BoxSpace((8,))
    return _HappoCriticWrapper(Critic(args, cent_obs_space))


def example_input_happo_critic():
    cent_obs = torch.randn(2, 8)
    rnn_states = torch.zeros(2, 1, 16)
    masks = torch.ones(2, 1)
    return (cent_obs, rnn_states, masks)


MENAGERIE_ENTRIES = [
    (
        "HAPPO Actor",
        build_happo_actor,
        example_input_happo_actor,
        2022,
        MENAGERIE_ZOO,
    ),
    (
        "HAPPO Critic",
        build_happo_critic,
        example_input_happo_critic,
        2022,
        MENAGERIE_ZOO,
    ),
]
