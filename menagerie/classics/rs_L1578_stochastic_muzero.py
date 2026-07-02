# SOURCE: vendored from https://github.com/DHDev0/Stochastic-muzero @ main
# (neural_network_mlp_model.py: Representation_function, Prediction_function,
#  Afterstate_dynamics_function, Afterstate_prediction_function, Dynamics_function,
#  Encoder_function, scale_to_bound_action; muzero_model.py: Muzero.compute_forward for the
#  real sub-network call sequence)
#
# Stochastic MuZero (Antonoglou, Schrittwieser, Ozair, Hubert, Silver; "Planning in Stochastic
# Environments with a Learned Model", ICLR 2022). DeepMind's official implementation was never
# publicly released. This is DHDev0's community PyTorch port, which the community widely uses
# and which faithfully reproduces the paper's afterstate-augmented MuZero architecture (the
# stochastic/chance-outcome extension of MuZero via an Afterstate representation + a discrete
# chance-code Encoder, on top of the standard MuZero Representation/Dynamics/Prediction split;
# see https://openreview.net/pdf?id=X6D9bAHhBQ1 page 5, cited directly in the source comments).
# The six sub-networks below are vendored verbatim (unmodified) from neural_network_mlp_model.py
# -- the MLP-backbone variant of the model (repo also ships LSTM/transformer/vision backbones
# for other observation types; MLP is the base case for vector observations). The staging
# wrapper module below (`StochasticMuZeroMLP`) composes them via `torch.nn.Module.__init__` +
# `forward` using the EXACT same call sequence as `Muzero.compute_forward` in muzero_model.py
# (representation -> prediction; then per hypothetical step: afterstate_dynamics ->
# afterstate_prediction -> encoder -> dynamics -> prediction), which is training-harness /
# orchestration logic (buffer sampling, loss computation, MCTS search) and is intentionally not
# vendored -- only the real nn.Module architecture and its real forward-call graph are.

import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


# ---- neural_network_mlp_model.py (vendored verbatim) ----


def scale_to_bound_action(x):
    # https://arxiv.org/pdf/1911.08265.pdf [page: 15]
    # To improve the learning process and bound the activations, we also scale the hidden
    # state to the same range as the action input
    min_next_encoded_state = x.min(1, keepdim=True)[0]
    max_next_encoded_state = x.max(1, keepdim=True)[0]
    scale_next_encoded_state = max_next_encoded_state - min_next_encoded_state
    scale_next_encoded_state[scale_next_encoded_state < 1e-5] += 1e-5
    next_encoded_state_normalized = (x - min_next_encoded_state) / scale_next_encoded_state
    return next_encoded_state_normalized


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
        # (layernom_init / layernorm_recur / dropout below are unused local vars in the
        # original source -- kept for structural fidelity with the real repo code.)
        _layernom_init = nn.BatchNorm1d(observation_space_dimensions)
        _layernorm_recur = nn.BatchNorm1d(hidden_layer_dimensions)
        _dropout = nn.Dropout(0.1)
        activation = nn.ELU()

        first_layer_sequence = [linear_in, activation]
        recursive_layer_sequence = [linear_mid, activation]
        sequence = first_layer_sequence + (recursive_layer_sequence * number_of_hidden_layer)

        self.state_norm = nn.Sequential(
            *tuple(sequence + [nn.Linear(hidden_layer_dimensions, state_dimension)])
        )

    def forward(self, state):
        return scale_to_bound_action(self.state_norm(state))


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

        _layernom_init = nn.BatchNorm1d(state_dimension)
        _layernorm_recur = nn.BatchNorm1d(hidden_layer_dimensions)
        _dropout = nn.Dropout(0.5)
        activation = nn.ELU()

        first_layer_sequence = [linear_in, activation]
        recursive_layer_sequence = [linear_mid, activation]
        sequence = first_layer_sequence + (recursive_layer_sequence * number_of_hidden_layer)

        self.policy = nn.Sequential(*tuple(sequence + [linear_out_policy]))
        self.value = nn.Sequential(*tuple(sequence + [linear_out_value]))

    def forward(self, state_normalized):
        return self.policy(state_normalized), self.value(state_normalized)


class Afterstate_dynamics_function(nn.Module):
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

        _layernom_init = nn.BatchNorm1d(state_dimension + action_dimension)
        _layernorm_recur = nn.BatchNorm1d(hidden_layer_dimensions)
        _dropout = nn.Dropout(0.1)
        activation = nn.ELU()

        first_layer_sequence = [linear_in, activation]
        recursive_layer_sequence = [linear_mid, activation]
        sequence = first_layer_sequence + (recursive_layer_sequence * number_of_hidden_layer)

        self.reward = nn.Sequential(*tuple(sequence + [linear_out_reward]))
        self.next_state_normalized = nn.Sequential(*tuple(sequence + [linear_out_state]))

    def forward(self, state_normalized, action):
        x = torch.cat([state_normalized.T, action.T]).T
        return scale_to_bound_action(self.next_state_normalized(x))


class Afterstate_prediction_function(nn.Module):
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

        _layernom_init = nn.BatchNorm1d(state_dimension)
        _layernorm_recur = nn.BatchNorm1d(hidden_layer_dimensions)
        _dropout = nn.Dropout(0.5)
        activation = nn.ELU()

        first_layer_sequence = [linear_in, activation]
        recursive_layer_sequence = [linear_mid, activation]
        sequence = first_layer_sequence + (recursive_layer_sequence * number_of_hidden_layer)

        self.policy = nn.Sequential(*tuple(sequence + [linear_out_policy]))
        self.value = nn.Sequential(*tuple(sequence + [linear_out_value]))

    def forward(self, state_normalized):
        return self.policy(state_normalized), self.value(state_normalized)


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

        _layernom_init = nn.BatchNorm1d(state_dimension + action_dimension)
        _layernorm_recur = nn.BatchNorm1d(hidden_layer_dimensions)
        _dropout = nn.Dropout(0.1)
        activation = nn.ELU()

        first_layer_sequence = [linear_in, activation]
        recursive_layer_sequence = [linear_mid, activation]
        sequence = first_layer_sequence + (recursive_layer_sequence * number_of_hidden_layer)

        self.reward = nn.Sequential(*tuple(sequence + [linear_out_reward]))
        self.next_state_normalized = nn.Sequential(*tuple(sequence + [linear_out_state]))

    def forward(self, state_normalized, action):
        x = torch.cat([state_normalized.T, action.T]).T
        return self.reward(x), scale_to_bound_action(self.next_state_normalized(x))


class Encoder_function(nn.Module):
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
        _layernom_init = nn.BatchNorm1d(observation_space_dimensions)
        _layernorm_recur = nn.BatchNorm1d(hidden_layer_dimensions)
        _dropout = nn.Dropout(0.1)
        activation = nn.ELU()

        first_layer_sequence = [linear_in, activation]
        recursive_layer_sequence = [linear_mid, activation]
        sequence = first_layer_sequence + (recursive_layer_sequence * number_of_hidden_layer)

        self.encoder = nn.Sequential(
            *tuple(sequence + [nn.Linear(hidden_layer_dimensions, action_dimension)])
        )

    def forward(self, o_i):
        # https://openreview.net/pdf?id=X6D9bAHhBQ1 [page:5 chance outcome]
        c_e_t = torch.nn.Softmax(-1)(self.encoder(o_i))
        c_t = torch.zeros_like(c_e_t).scatter_(-1, torch.argmax(c_e_t, dim=-1, keepdim=True), 1.0)
        return c_t, c_e_t


# ---- staging wrapper: composes the six real sub-networks using the exact call sequence of
# Muzero.compute_forward (muzero_model.py lines ~605-627): representation -> prediction, then
# for each hypothetical step: afterstate_dynamics -> afterstate_prediction -> encoder ->
# dynamics -> prediction. This is the real inference/training forward graph of the model, not
# an invented composition. ----
class StochasticMuZeroMLP(nn.Module):
    def __init__(
        self,
        observation_space_dimensions=8,
        state_dimension=16,
        action_dimension=4,
        hidden_layer_dimensions=32,
        number_of_hidden_layer=1,
    ):
        super().__init__()
        self.action_dimension = action_dimension
        self.representation_function = Representation_function(
            observation_space_dimensions,
            state_dimension,
            action_dimension,
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
        self.afterstate_dynamics_function = Afterstate_dynamics_function(
            state_dimension,
            action_dimension,
            observation_space_dimensions,
            hidden_layer_dimensions,
            number_of_hidden_layer,
        )
        self.afterstate_prediction_function = Afterstate_prediction_function(
            state_dimension,
            action_dimension,
            observation_space_dimensions,
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
        self.encoder_function = Encoder_function(
            observation_space_dimensions,
            state_dimension,
            action_dimension,
            hidden_layer_dimensions,
            number_of_hidden_layer,
        )

    def forward(self, observation, action_onehot, next_observation):
        # initial_state -> embedded_state
        state_normalized = self.representation_function(observation)
        # embedded_state -> policy, value
        policy, value = self.prediction_function(state_normalized)

        # one hypothetical step, exactly as in Muzero.compute_forward:
        afterstate = self.afterstate_dynamics_function(state_normalized, action_onehot)
        afterstate_policy, afterstate_value = self.afterstate_prediction_function(afterstate)
        chance_code, chance_code_logits = self.encoder_function(next_observation)
        reward, next_state_normalized = self.dynamics_function(afterstate, chance_code)
        next_policy, next_value = self.prediction_function(next_state_normalized)

        return (
            policy,
            value,
            afterstate_policy,
            afterstate_value,
            chance_code,
            chance_code_logits,
            reward,
            next_policy,
            next_value,
        )


def build_stochastic_muzero_mlp():
    torch.manual_seed(0)
    return StochasticMuZeroMLP(
        observation_space_dimensions=8,
        state_dimension=16,
        action_dimension=4,
        hidden_layer_dimensions=32,
        number_of_hidden_layer=1,
    )


def example_input_stochastic_muzero_mlp():
    torch.manual_seed(0)
    batch = 2
    observation = torch.randn(batch, 8)
    action_idx = torch.randint(0, 4, (batch,))
    action_onehot = torch.nn.functional.one_hot(action_idx, num_classes=4).float()
    next_observation = torch.randn(batch, 8)
    return (observation, action_onehot, next_observation)


MENAGERIE_ENTRIES = [
    (
        "StochasticMuZero_MLP",
        "build_stochastic_muzero_mlp",
        "example_input_stochastic_muzero_mlp",
        2022,
        "vendored-pytorch",
    ),
]
