# SOURCE: vendored from DHDev0/Muzero-unplugged @ main (neural_network_mlp_model.py)
# https://github.com/DHDev0/Muzero-unplugged -- community PyTorch implementation of
# "Online and Offline Reinforcement Learning by Planning with a Learned Model"
# (Schrittwieser et al. 2021, arXiv:2104.06294), the "MuZero Unplugged" extension of
# MuZero that adds demonstration/reanalyze replay buffers for offline RL. The buffer
# machinery is a training-time data-pipeline change with no effect on the network
# architecture, so the network itself -- `Representation_function`,
# `Dynamics_function`, `Prediction_function` (the MLP variant for 1D
# observation/discrete action environments, per arxiv.org/pdf/1911.08265.pdf pages
# 3-4) -- is transcribed verbatim from neural_network_mlp_model.py. `scale_to_bound_
# action` (the hidden-state rescaling from the paper appendix, page 15) is included
# unmodified. A thin `MuZeroUnpluggedMLP` wrapper below composes the three functions
# into one nn.Module (representation -> dynamics -> prediction, matching one
# `Muzero.reanalyze_pipeline`/`Muzero.think` unroll step) purely for tracing; the
# wrapper is new plumbing, not new architecture.
import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


# ---- verbatim from neural_network_mlp_model.py ----
# https://arxiv.org/pdf/1911.08265.pdf [page: 15]
# To improve the learning process and bound the activations,
# we also scale the hidden state to the same range as
# the action input
def scale_to_bound_action(x):
    min_next_encoded_state = x.min(1, keepdim=True)[0]
    max_next_encoded_state = x.max(1, keepdim=True)[0]
    scale_next_encoded_state = max_next_encoded_state - min_next_encoded_state
    scale_next_encoded_state[scale_next_encoded_state < 1e-5] += 1e-5
    next_encoded_state_normalized = (x - min_next_encoded_state) / scale_next_encoded_state
    return next_encoded_state_normalized


# https://arxiv.org/pdf/1911.08265.pdf [page: 3 and 4] for the structure
# Multilayer perceptron (MLP) for muzero with 1D observation and discrete action
class Representation_function(nn.Module):
    def __init__(
        self,
        observation_space_dimensions,
        state_dimension,
        action_dimension,
        hidden_layer_dimensions,
        number_of_hidden_layer,
    ):
        super().__init__()
        self.action_space = action_dimension
        linear_in = nn.Linear(observation_space_dimensions, hidden_layer_dimensions)
        linear_mid = nn.Linear(hidden_layer_dimensions, hidden_layer_dimensions)

        self.scale = nn.Tanh()
        activation = nn.ELU()

        first_layer_sequence = [linear_in, activation]

        recursive_layer_sequence = [linear_mid, activation]

        sequence = first_layer_sequence + (recursive_layer_sequence * number_of_hidden_layer)

        self.state_norm = nn.Sequential(
            *tuple(sequence + [nn.Linear(hidden_layer_dimensions, state_dimension)])
        )

    def forward(self, state):
        return scale_to_bound_action(self.state_norm(state))


# https://arxiv.org/pdf/1911.08265.pdf [page: 3 and 4] for the structure
# Multilayer perceptron (MLP) for muzero with 1D observation and discrete action
class Dynamics_function(nn.Module):
    def __init__(
        self,
        state_dimension,
        action_dimension,
        observation_space_dimensions,
        hidden_layer_dimensions,
        number_of_hidden_layer,
    ):
        super().__init__()
        self.action_space = action_dimension
        linear_in = nn.Linear(state_dimension + action_dimension, hidden_layer_dimensions)
        linear_mid = nn.Linear(hidden_layer_dimensions, hidden_layer_dimensions)
        linear_out_reward = nn.Linear(hidden_layer_dimensions, state_dimension)
        linear_out_state = nn.Linear(hidden_layer_dimensions, state_dimension)

        activation = nn.ELU()

        first_layer_sequence = [linear_in, activation]

        recursive_layer_sequence = [linear_mid, activation]

        sequence = first_layer_sequence + (recursive_layer_sequence * number_of_hidden_layer)

        self.reward = nn.Sequential(*tuple(sequence + [linear_out_reward]))
        self.next_state_normalized = nn.Sequential(*tuple(sequence + [linear_out_state]))

    def forward(self, state_normalized, action):
        x = torch.cat([state_normalized.T, action.T]).T
        return self.reward(x), scale_to_bound_action(self.next_state_normalized(x))


# https://arxiv.org/pdf/1911.08265.pdf [page: 3 and 4] for the structure
# Multilayer perceptron (MLP) for muzero with 1D observation and discrete action
class Prediction_function(nn.Module):
    def __init__(
        self,
        state_dimension,
        action_dimension,
        observation_space_dimensions,
        hidden_layer_dimensions,
        number_of_hidden_layer,
    ):
        super().__init__()

        linear_in = nn.Linear(state_dimension, hidden_layer_dimensions)
        linear_mid = nn.Linear(hidden_layer_dimensions, hidden_layer_dimensions)
        linear_out_policy = nn.Linear(hidden_layer_dimensions, action_dimension)
        linear_out_value = nn.Linear(hidden_layer_dimensions, state_dimension)

        activation = nn.ELU()

        first_layer_sequence = [linear_in, activation]

        recursive_layer_sequence = [linear_mid, activation]

        sequence = first_layer_sequence + (recursive_layer_sequence * number_of_hidden_layer)

        self.policy = nn.Sequential(*tuple(sequence + [linear_out_policy]))
        self.value = nn.Sequential(*tuple(sequence + [linear_out_value]))

    def forward(self, state_normalized):
        return self.policy(state_normalized), self.value(state_normalized)


# ---- staging-only wrapper: composes the three real functions into a single traced
# unroll step (representation(obs) -> dynamics(state, action) -> prediction(state)).
# No new architecture; purely for a single tl.trace() forward call. ----
class MuZeroUnpluggedMLP(nn.Module):
    def __init__(
        self,
        observation_space_dimensions,
        state_dimension,
        action_dimension,
        hidden_layer_dimensions,
        number_of_hidden_layer,
    ):
        super().__init__()
        self.representation_function = Representation_function(
            observation_space_dimensions,
            state_dimension,
            action_dimension,
            hidden_layer_dimensions,
            number_of_hidden_layer,
        )
        self.dynamics_function = Dynamics_function(
            state_dimension,
            action_dimension,
            observation_space_dimensions,
            hidden_layer_dimensions,
            number_of_hidden_layer,
        )
        self.prediction_function = Prediction_function(
            state_dimension,
            action_dimension,
            observation_space_dimensions,
            hidden_layer_dimensions,
            number_of_hidden_layer,
        )

    def forward(self, observation, action):
        state_normalized = self.representation_function(observation)
        reward, next_state_normalized = self.dynamics_function(state_normalized, action)
        policy, value = self.prediction_function(next_state_normalized)
        return policy, value, reward


def build_muzero_unplugged_mlp():
    torch.manual_seed(0)
    model = MuZeroUnpluggedMLP(
        observation_space_dimensions=8,
        state_dimension=6,
        action_dimension=4,
        hidden_layer_dimensions=16,
        number_of_hidden_layer=1,
    )
    model.eval()
    return model


def example_input_muzero_unplugged_mlp():
    torch.manual_seed(0)
    batch_size = 2
    observation = torch.randn(batch_size, 8)
    action = torch.randn(batch_size, 4)
    return (observation, action)


MENAGERIE_ENTRIES = [
    (
        "MuZeroUnplugged_MLP",
        build_muzero_unplugged_mlp,
        example_input_muzero_unplugged_mlp,
        2022,
        "vendored-pytorch",
    ),
]
