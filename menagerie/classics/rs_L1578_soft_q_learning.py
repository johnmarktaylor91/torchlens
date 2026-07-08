# FAITHFUL PORT of https://github.com/haarnoja/softqlearning @ master (original framework: TF1.x)
#
# Soft Q-Learning (Haarnoja, Tang, Abbeel, Levine; "Reinforcement Learning with Deep
# Energy-Based Policies", ICML 2017). Official repo is TensorFlow 1.x (`tf.placeholder`,
# `tf.variable_scope`, `tf.get_variable`) built on the (long-deprecated, pre-Gymnasium) `garage`
# / rllab RL framework -- neither TF1.x graph-mode nor `garage` are installable/runnable in this
# base torch env (TF1.x is EOL and not one of the declared base libs; `garage`'s pinned
# TF1-era deps are themselves unresolvable today). Per the ladder, this repo is a RUNG 3
# faithful port: the real network architecture is transcribed layer-for-layer from the actual
# TF1.x source (not guessed from the paper), just re-expressed in torch's imperative nn.Module
# style instead of TF1's placeholder/variable_scope graph-building style.
#
# Transcribed from softqlearning/misc/nn.py (`feedforward_net`, `MLPFunction`),
# softqlearning/value_functions/value_function.py (`NNQFunction`, `NNVFunction`), and
# softqlearning/policies/stochastic_policy.py (`StochasticNNPolicy`):
#   - `feedforward_net(inputs, layer_sizes, activation_fn=relu, output_nonlinearity=None)`:
#     concatenates all `inputs` tensors (via tensordot-then-sum, i.e. a per-input linear
#     projection into the first hidden layer, summed) then MLP layers hidden_layer_sizes + [1]
#     for MLPFunction (Q/V), each with relu except the last (linear); the real TF code sums a
#     separate `tf.tensordot` per input tensor at layer 0 rather than concatenating first --
#     mathematically equivalent to concat-then-single-linear for tensors of matching feature
#     dim (as used here: it is literally `sum_j W_j @ input_j`, identical to
#     `[input_0 | input_1] @ [W_0; W_1]` stacked), so we implement it via concatenation, which
#     is numerically identical and is the standard torch idiom for this pattern.
#   - `NNQFunction(env_spec, hidden_layer_sizes=(M, M))`: MLPFunction over (observations,
#     actions) -> scalar Q-value.
#   - `NNVFunction(env_spec, hidden_layer_sizes=(M, M))`: MLPFunction over (observations,)
#     -> scalar V-value.
#   - `StochasticNNPolicy(env_spec, hidden_layer_sizes=(M, M), squash=True)`: samples latent
#     noise ~ N(0, I) of the action dimension, feeds (observations, latents) through the same
#     feedforward_net pattern (layer_sizes = hidden_layer_sizes + [action_dim]), then
#     `tanh`-squashes the raw output into bounded actions (the amortized SVGD sampler network
#     used to approximate the soft-Q energy-based policy).
# M=128, hidden_layer_sizes=(M, M) matches the repo's own example configs (e.g.
# examples/mujoco_all_sql.py: `'layer_size': 128`, `qf = NNQFunction(..., hidden_layer_sizes=(M, M))`,
# `policy = StochasticNNPolicy(..., hidden_layer_sizes=(M, M))`).
# Not ported: the SVGD (Stein Variational Gradient Descent) training/sampling loop, the replay
# buffer, and the `garage`/rllab environment-spec plumbing -- those are the RL training harness,
# not network architecture. This module ports the real feedforward network structures only.

import torch
import torch.nn as nn

MENAGERIE_ZOO = "ported-pytorch"


class FeedforwardNet(nn.Module):
    """Faithful port of softqlearning/misc/nn.py:feedforward_net -- an MLP that takes one or
    more same-batch input tensors, projects+sums them into the first hidden layer (here:
    concatenate then linear, numerically identical to the original's per-input tensordot-sum),
    then applies `len(layer_sizes)` linear layers with relu activations between all but the
    last (the last layer is linear / has no output_nonlinearity, matching the original's
    `output_nonlinearity=None` default used by both MLPFunction and StochasticNNPolicy)."""

    def __init__(self, input_sizes, layer_sizes):
        super().__init__()
        self.input_sizes = list(input_sizes)
        self.layer_sizes = list(layer_sizes)
        in_features = sum(self.input_sizes)
        layers = []
        prev = in_features
        for i, size in enumerate(self.layer_sizes):
            layers.append(nn.Linear(prev, size))
            prev = size
        self.linears = nn.ModuleList(layers)
        self.activation = nn.ReLU()

    def forward(self, *inputs):
        out = torch.cat(inputs, dim=-1)
        for i, linear in enumerate(self.linears):
            out = linear(out)
            if i < len(self.linears) - 1:
                out = self.activation(out)
        return out


class MLPFunction(nn.Module):
    """Faithful port of softqlearning/misc/nn.py:MLPFunction -- feedforward_net with
    layer_sizes = list(hidden_layer_sizes) + [1], output squeezed to a scalar per batch
    element (`out[..., 0]` in the original `_output_for`)."""

    def __init__(self, input_sizes, hidden_layer_sizes):
        super().__init__()
        self.net = FeedforwardNet(input_sizes, list(hidden_layer_sizes) + [1])

    def forward(self, *inputs):
        return self.net(*inputs)[..., 0]


class NNQFunction(nn.Module):
    """Faithful port of softqlearning/value_functions/value_function.py:NNQFunction."""

    def __init__(self, observation_dim, action_dim, hidden_layer_sizes=(128, 128)):
        super().__init__()
        self.qf = MLPFunction((observation_dim, action_dim), hidden_layer_sizes)

    def forward(self, observations, actions):
        return self.qf(observations, actions)


class NNVFunction(nn.Module):
    """Faithful port of softqlearning/value_functions/value_function.py:NNVFunction."""

    def __init__(self, observation_dim, hidden_layer_sizes=(128, 128)):
        super().__init__()
        self.vf = MLPFunction((observation_dim,), hidden_layer_sizes)

    def forward(self, observations):
        return self.vf(observations)


class StochasticNNPolicy(nn.Module):
    """Faithful port of softqlearning/policies/stochastic_policy.py:StochasticNNPolicy
    (`actions_for`, n_action_samples=1 branch): draws latent noise ~ N(0, I) of the action
    dimension, feeds (observations, latents) through feedforward_net with
    layer_sizes = hidden_layer_sizes + [action_dim], then squashes with tanh (the original's
    `squash=True` default, giving bounded actions)."""

    def __init__(self, observation_dim, action_dim, hidden_layer_sizes=(128, 128), squash=True):
        super().__init__()
        self.action_dim = action_dim
        self.squash = squash
        self.net = FeedforwardNet(
            (observation_dim, action_dim), list(hidden_layer_sizes) + [action_dim]
        )

    def forward(self, observations):
        latents = torch.randn(*observations.shape[:-1], self.action_dim, device=observations.device)
        raw_actions = self.net(observations, latents)
        return torch.tanh(raw_actions) if self.squash else raw_actions


class SoftQLearningActorCritic(nn.Module):
    """Staging wrapper bundling the real Soft Q-Learning network trio (policy + Q-function +
    V-function) into a single traceable module, matching how they are jointly constructed and
    used together during SQL training/acting in the real repo (e.g. examples/mujoco_all_sql.py)."""

    def __init__(self, observation_dim=8, action_dim=4, hidden_layer_sizes=(128, 128)):
        super().__init__()
        self.policy = StochasticNNPolicy(observation_dim, action_dim, hidden_layer_sizes)
        self.qf = NNQFunction(observation_dim, action_dim, hidden_layer_sizes)
        self.vf = NNVFunction(observation_dim, hidden_layer_sizes)

    def forward(self, observations):
        actions = self.policy(observations)
        q_values = self.qf(observations, actions)
        v_values = self.vf(observations)
        return actions, q_values, v_values


def build_soft_q_learning():
    torch.manual_seed(0)
    return SoftQLearningActorCritic(observation_dim=8, action_dim=4, hidden_layer_sizes=(128, 128))


def example_input_soft_q_learning():
    torch.manual_seed(0)
    batch = 2
    observations = torch.randn(batch, 8)
    return (observations,)


MENAGERIE_ENTRIES = [
    (
        "SoftQLearning_ActorCritic",
        "build_soft_q_learning",
        "example_input_soft_q_learning",
        2017,
        "ported-pytorch",
    ),
]
