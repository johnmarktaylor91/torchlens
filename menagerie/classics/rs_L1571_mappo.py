# SOURCE: vendored from https://github.com/marlbenchmark/on-policy @ main
# MAPPO (Multi-Agent PPO): shared recurrent actor-critic policy network used by
# the official "The Surprising Effectiveness of PPO in Cooperative Multi-Agent
# Games" (Yu et al., NeurIPS 2022 D&B) reference implementation.
#
# Vendored real repo code, combined from:
#   onpolicy/algorithms/r_mappo/algorithm/r_actor_critic.py (R_Actor, R_Critic)
#   onpolicy/algorithms/utils/mlp.py    (MLPLayer, MLPBase)
#   onpolicy/algorithms/utils/rnn.py    (RNNLayer)
#   onpolicy/algorithms/utils/act.py    (ACTLayer, discrete branch only)
#   onpolicy/algorithms/utils/distributions.py (Categorical, FixedCategorical)
#   onpolicy/algorithms/utils/util.py   (init, check)
# Only non-architectural portability fixes applied:
#   - the original `args` parameter is an argparse.Namespace built from CLI
#     flags; replaced with a plain `SimpleNamespace` carrying the same fields
#     so the module has no argparse/CLI dependency (values only, no logic
#     changed).
#   - the original `action_space`/`obs_space` are gym.Space objects; replaced
#     with tiny local duck-typed stand-ins exposing only the attributes the
#     real code reads (`obs_space.shape`, `action_space.__class__.__name__`,
#     `action_space.n`) so the module has no gym dependency. No branch in the
#     real R_Actor/ACTLayer code was changed -- the Discrete-action branch is
#     exercised exactly as the real code executes it.
#   - CNNBase (image-observation path) and PopArt-critic path are omitted
#     since this recipe exercises the (default) MLP-observation, non-PopArt
#     branch; nothing in the copied MLP/RNN/ACT/distributions code was
#     altered.
# No layer, head, or dataflow was changed from the real implementation.

import copy
from types import SimpleNamespace

import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


# ---- onpolicy/algorithms/utils/util.py ----
def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    if module.bias is not None:
        bias_init(module.bias.data)
    return module


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def check(input):
    import numpy as np

    output = torch.from_numpy(input) if type(input) == np.ndarray else input
    return output


# ---- onpolicy/algorithms/utils/distributions.py ----
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


# ---- onpolicy/algorithms/utils/mlp.py ----
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


# ---- onpolicy/algorithms/utils/rnn.py ----
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
        if x.size(0) == hxs.size(0):
            x, hxs = self.rnn(
                x.unsqueeze(0),
                (hxs * masks.repeat(1, self._recurrent_N).unsqueeze(-1))
                .transpose(0, 1)
                .contiguous(),
            )
            x = x.squeeze(0)
            hxs = hxs.transpose(0, 1)
        else:
            N = hxs.size(0)
            T = int(x.size(0) / N)

            x = x.view(T, N, x.size(1))
            masks = masks.view(T, N)

            has_zeros = (masks[1:] == 0.0).any(dim=-1).nonzero().squeeze().cpu()

            if has_zeros.dim() == 0:
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.transpose(0, 1)

            outputs = []
            for i in range(len(has_zeros) - 1):
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]
                temp = (
                    hxs * masks[start_idx].view(1, -1, 1).repeat(self._recurrent_N, 1, 1)
                ).contiguous()
                rnn_scores, hxs = self.rnn(x[start_idx:end_idx], temp)
                outputs.append(rnn_scores)

            x = torch.cat(outputs, dim=0)

            x = x.reshape(T * N, -1)
            hxs = hxs.transpose(0, 1)

        x = self.norm(x)
        return x, hxs


# ---- onpolicy/algorithms/utils/act.py (Discrete-action branch only) ----
class ACTLayer(nn.Module):
    def __init__(self, action_space, inputs_dim, use_orthogonal, gain, args=None):
        super(ACTLayer, self).__init__()
        self.mixed_action = False
        self.multi_discrete = False
        self.mujoco_box = False
        self.action_type = action_space.__class__.__name__

        # Real code's Discrete branch (the only branch this recipe exercises).
        action_dim = action_space.n
        self.action_out = Categorical(inputs_dim, action_dim, use_orthogonal, gain)

    def forward(self, x, available_actions=None, deterministic=False):
        action_logits = self.action_out(x, available_actions)
        actions = action_logits.mode() if deterministic else action_logits.sample()
        action_log_probs = action_logits.log_probs(actions)

        return actions, action_log_probs


# ---- onpolicy/algorithms/r_mappo/algorithm/r_actor_critic.py ----
class R_Actor(nn.Module):
    """Actor network class for MAPPO. Outputs actions given observations."""

    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(R_Actor, self).__init__()
        self.hidden_size = args.hidden_size

        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = obs_space.shape
        self.base = MLPBase(args, obs_shape)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(
                self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal
            )

        self.act = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain, args)

        self.to(device)
        self.algo = args.algorithm_name

    def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
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


class R_Critic(nn.Module):
    """Critic network class for MAPPO. Outputs value predictions given centralized obs."""

    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        super(R_Critic, self).__init__()
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self._use_popart = args.use_popart
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        cent_obs_shape = cent_obs_space.shape
        self.base = MLPBase(args, cent_obs_shape)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(
                self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal
            )

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        self.v_out = init_(nn.Linear(self.hidden_size, 1))

        self.to(device)

    def forward(self, cent_obs, rnn_states, masks):
        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        critic_features = self.base(cent_obs)
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)
        values = self.v_out(critic_features)

        return values, rnn_states


class _ObsSpace:
    """Minimal gym.spaces.Box duck-type: real code reads only `.shape`."""

    def __init__(self, shape):
        self.shape = shape


class _DiscreteActionSpace:
    """Minimal gym.spaces.Discrete duck-type: real code reads `.__class__.__name__` and `.n`."""

    __name__ = "Discrete"

    def __init__(self, n):
        self.n = n

    @property
    def __class__(self):
        return type("Discrete", (), {})


class MAPPOActorCritic(nn.Module):
    """Thin wrapper combining the real R_Actor + R_Critic so TorchLens traces
    both networks of the shared MAPPO policy in a single forward pass."""

    def __init__(self, args, obs_space, cent_obs_space, action_space):
        super().__init__()
        self.actor = R_Actor(args, obs_space, action_space)
        self.critic = R_Critic(args, cent_obs_space)

    def forward(self, obs, cent_obs, rnn_states_actor, rnn_states_critic, masks):
        actions, action_log_probs, rnn_states_actor = self.actor(obs, rnn_states_actor, masks)
        values, rnn_states_critic = self.critic(cent_obs, rnn_states_critic, masks)
        return actions, action_log_probs, values


def _mappo_args():
    # Real repo's default CLI-flag values relevant to the MLP + Discrete +
    # recurrent-policy branch (see onpolicy/config.py for the real defaults);
    # shrunk hidden_size/recurrent_N for a tiny trace.
    return SimpleNamespace(
        hidden_size=16,
        gain=0.01,
        use_orthogonal=True,
        use_policy_active_masks=True,
        use_naive_recurrent_policy=False,
        use_recurrent_policy=True,
        recurrent_N=1,
        use_feature_normalization=True,
        use_ReLU=False,
        stacked_frames=1,
        layer_N=1,
        use_popart=False,
        algorithm_name="rmappo",
    )


def build_mappo():
    args = _mappo_args()
    obs_space = _ObsSpace((8,))
    cent_obs_space = _ObsSpace((16,))
    action_space = _DiscreteActionSpace(5)
    return MAPPOActorCritic(args, obs_space, cent_obs_space, action_space)


def example_input_mappo():
    n_rollout_threads = 2
    obs = torch.randn(n_rollout_threads, 8)
    cent_obs = torch.randn(n_rollout_threads, 16)
    rnn_states_actor = torch.zeros(n_rollout_threads, 1, 16)
    rnn_states_critic = torch.zeros(n_rollout_threads, 1, 16)
    masks = torch.ones(n_rollout_threads, 1)
    return (obs, cent_obs, rnn_states_actor, rnn_states_critic, masks)


MENAGERIE_ENTRIES = [
    (
        "MAPPO (Multi-Agent PPO)",
        build_mappo,
        example_input_mappo,
        2021,
        MENAGERIE_ZOO,
    ),
]
