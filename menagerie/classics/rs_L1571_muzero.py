# SOURCE: vendored from https://github.com/werner-duvaud/muzero-general @ master
# MuZero: "Mastering Atari, Go, Chess and Shogi by Planning with a Learned
# Model" (Schrittwieser et al., Nature 2020). werner-duvaud/muzero-general is
# a widely-used community PyTorch reference implementation whose config
# explicitly exposes a `reanalyze_fraction`/self-play reanalyze training mode
# (see its `MuZeroConfig` / `replay_buffer.py`) built on top of this exact
# network. The Reanalyze mechanism itself is a training-time replay-buffer
# procedure -- it re-runs the *same* representation/dynamics/prediction
# networks defined here to refresh stored MCTS statistics, so the traced
# architecture is the actual reanalyze-capable MuZero network, unmodified.
#
# Vendored real repo code from models.py (fully-connected variant):
#   MuZeroNetwork (dispatcher), AbstractNetwork, MuZeroFullyConnectedNetwork,
#   mlp() helper.
# Only non-architectural portability fix applied:
#   - dropped the `torch.nn.DataParallel` wrapping around each subnetwork
#     (multi-GPU training convenience in the original code, not part of the
#     architecture) so the module traces as plain nn.Module calls; the
#     wrapped layers/weights/forward math are identical either way.
# No layer, head, or dataflow was changed from the real implementation.

from abc import ABC, abstractmethod

import torch

MENAGERIE_ZOO = "vendored-pytorch"


# ---- models.py: dict_to_cpu / AbstractNetwork (verbatim) ----
def dict_to_cpu(dictionary):
    cpu_dict = {}
    for key, value in dictionary.items():
        if isinstance(value, torch.Tensor):
            cpu_dict[key] = value.cpu()
        elif isinstance(value, dict):
            cpu_dict[key] = dict_to_cpu(value)
        else:
            cpu_dict[key] = value
    return cpu_dict


class AbstractNetwork(ABC, torch.nn.Module):
    def __init__(self):
        super().__init__()
        pass

    @abstractmethod
    def initial_inference(self, observation):
        pass

    @abstractmethod
    def recurrent_inference(self, encoded_state, action):
        pass

    def get_weights(self):
        return dict_to_cpu(self.state_dict())

    def set_weights(self, weights):
        self.load_state_dict(weights)


# ---- models.py: mlp() helper (verbatim) ----
def mlp(
    input_size,
    layer_sizes,
    output_size,
    output_activation=torch.nn.Identity,
    activation=torch.nn.ELU,
):
    sizes = [input_size] + layer_sizes + [output_size]
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [torch.nn.Linear(sizes[i], sizes[i + 1]), act()]
    return torch.nn.Sequential(*layers)


# ---- models.py: MuZeroFullyConnectedNetwork ----
# NOTE (portability fix): the real repo wraps each subnetwork in
# `torch.nn.DataParallel(...)` for multi-GPU training; DataParallel is
# dropped here (single-process CPU trace) but every layer/weight/forward
# computation inside each subnetwork is copied verbatim.
class MuZeroFullyConnectedNetwork(AbstractNetwork):
    def __init__(
        self,
        observation_shape,
        stacked_observations,
        action_space_size,
        encoding_size,
        fc_reward_layers,
        fc_value_layers,
        fc_policy_layers,
        fc_representation_layers,
        fc_dynamics_layers,
        support_size,
    ):
        super().__init__()
        self.action_space_size = action_space_size
        self.full_support_size = 2 * support_size + 1

        self.representation_network = mlp(
            observation_shape[0]
            * observation_shape[1]
            * observation_shape[2]
            * (stacked_observations + 1)
            + stacked_observations * observation_shape[1] * observation_shape[2],
            fc_representation_layers,
            encoding_size,
        )

        self.dynamics_encoded_state_network = mlp(
            encoding_size + self.action_space_size,
            fc_dynamics_layers,
            encoding_size,
        )
        self.dynamics_reward_network = mlp(encoding_size, fc_reward_layers, self.full_support_size)

        self.prediction_policy_network = mlp(
            encoding_size, fc_policy_layers, self.action_space_size
        )
        self.prediction_value_network = mlp(encoding_size, fc_value_layers, self.full_support_size)

    def prediction(self, encoded_state):
        policy_logits = self.prediction_policy_network(encoded_state)
        value = self.prediction_value_network(encoded_state)
        return policy_logits, value

    def representation(self, observation):
        encoded_state = self.representation_network(observation.view(observation.shape[0], -1))
        # Scale encoded state between [0, 1] (See appendix paper Training)
        min_encoded_state = encoded_state.min(1, keepdim=True)[0]
        max_encoded_state = encoded_state.max(1, keepdim=True)[0]
        scale_encoded_state = max_encoded_state - min_encoded_state
        scale_encoded_state[scale_encoded_state < 1e-5] += 1e-5
        encoded_state_normalized = (encoded_state - min_encoded_state) / scale_encoded_state
        return encoded_state_normalized

    def dynamics(self, encoded_state, action):
        # Stack encoded_state with a game specific one hot encoded action (See paper appendix Network Architecture)
        action_one_hot = (
            torch.zeros((action.shape[0], self.action_space_size)).to(action.device).float()
        )
        action_one_hot.scatter_(1, action.long(), 1.0)
        x = torch.cat((encoded_state, action_one_hot), dim=1)

        next_encoded_state = self.dynamics_encoded_state_network(x)

        reward = self.dynamics_reward_network(next_encoded_state)

        # Scale encoded state between [0, 1] (See paper appendix Training)
        min_next_encoded_state = next_encoded_state.min(1, keepdim=True)[0]
        max_next_encoded_state = next_encoded_state.max(1, keepdim=True)[0]
        scale_next_encoded_state = max_next_encoded_state - min_next_encoded_state
        scale_next_encoded_state[scale_next_encoded_state < 1e-5] += 1e-5
        next_encoded_state_normalized = (
            next_encoded_state - min_next_encoded_state
        ) / scale_next_encoded_state

        return next_encoded_state_normalized, reward

    def initial_inference(self, observation):
        encoded_state = self.representation(observation)
        policy_logits, value = self.prediction(encoded_state)
        # reward equal to 0 for consistency
        reward = torch.log(
            torch.zeros(1, self.full_support_size)
            .scatter(1, torch.tensor([[self.full_support_size // 2]]).long(), 1.0)
            .repeat(len(observation), 1)
            .to(observation.device)
        )

        return (
            value,
            reward,
            policy_logits,
            encoded_state,
        )

    def recurrent_inference(self, encoded_state, action):
        next_encoded_state, reward = self.dynamics(encoded_state, action)
        policy_logits, value = self.prediction(next_encoded_state)
        return value, reward, policy_logits, next_encoded_state


class MuZeroForward(torch.nn.Module):
    """Thin wrapper chaining initial_inference -> recurrent_inference so
    TorchLens traces the full reanalyze-capable MuZero network (representation
    + dynamics + prediction) in a single forward call."""

    def __init__(self, net: MuZeroFullyConnectedNetwork):
        super().__init__()
        self.net = net

    def forward(self, observation, action):
        value, reward, policy_logits, encoded_state = self.net.initial_inference(observation)
        next_value, next_reward, next_policy_logits, next_encoded_state = (
            self.net.recurrent_inference(encoded_state, action)
        )
        return value, reward, policy_logits, next_value, next_reward, next_policy_logits


def build_muzero_reanalyze():
    # Tiny config modeled on the real repo's CartPole preset (games/cartpole.py):
    # small fully-connected observation, small action space, small support size.
    net = MuZeroFullyConnectedNetwork(
        observation_shape=(1, 1, 4),
        stacked_observations=0,
        action_space_size=2,
        encoding_size=8,
        fc_reward_layers=[8],
        fc_value_layers=[8],
        fc_policy_layers=[8],
        fc_representation_layers=[8],
        fc_dynamics_layers=[8],
        support_size=5,
    )
    return MuZeroForward(net)


def example_input_muzero_reanalyze():
    batch = 2
    observation = torch.randn(batch, 1, 1, 4)
    action = torch.zeros(batch, 1, dtype=torch.long)
    return (observation, action)


MENAGERIE_ENTRIES = [
    (
        "MuZero Reanalyze",
        build_muzero_reanalyze,
        example_input_muzero_reanalyze,
        2021,
        MENAGERIE_ZOO,
    ),
]
