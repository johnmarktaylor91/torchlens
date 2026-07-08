# SOURCE: vendored from https://github.com/werner-duvaud/muzero-general @ continuous
# (models.py: MuZeroNetwork, AbstractNetwork, MuZeroFullyConnectedNetwork, mlp,
#  lines 1-232 approx of the `continuous` branch, vendored near-verbatim)
#
# Sampled MuZero (Hubert, Schrittwieser, Antonoglou, Barekatain, Schmitt, Silver 2021,
# "Learning and Planning in Complex Action Spaces", arXiv:2104.06303) extends MuZero to
# continuous/large action spaces by sampling candidate actions from a learned policy
# distribution rather than enumerating a discrete action set. werner-duvaud/muzero-general
# is the most actively maintained community PyTorch implementation of MuZero; its
# `continuous` branch is the repo's own implementation of this continuous/sampled-action
# variant: the prediction head outputs a diagonal-Gaussian policy (mu, log_std) over
# actions instead of discrete `policy_logits`, and `dynamics()` conditions on the raw
# continuous (sampled) action vector instead of a one-hot discrete action -- exactly the
# Sampled MuZero mechanism from the paper. Vendored verbatim (only the ResNet variant,
# `torch.multiprocessing`-based checkpoint I/O, and non-architectural helpers are
# omitted; the fully-connected representation/dynamics/prediction networks and the
# `initial_inference`/`recurrent_inference` MuZero inference API are kept intact).

from abc import ABC, abstractmethod

import torch

MENAGERIE_ZOO = "vendored-pytorch"


# ---- models.py (vendored, `continuous` branch) ----
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


class MuZeroFullyConnectedNetwork(AbstractNetwork):
    def __init__(
        self,
        observation_shape,
        stacked_observations,
        action_space_size,
        log_std_clamp,
        encoding_size,
        fc_reward_layers,
        fc_value_layers,
        fc_mu_policy_layers,
        fc_log_std_policy_layers,
        fc_representation_layers,
        fc_dynamics_layers,
        support_size,
    ):
        super().__init__()
        self.action_space_size = action_space_size
        self.log_std_clamp = log_std_clamp
        self.full_support_size = 2 * support_size + 1

        self.representation_network = torch.nn.DataParallel(
            mlp(
                observation_shape[0]
                * observation_shape[1]
                * observation_shape[2]
                * (stacked_observations + 1)
                + stacked_observations * observation_shape[1] * observation_shape[2],
                fc_representation_layers,
                encoding_size,
            )
        )

        self.dynamics_encoded_state_network = torch.nn.DataParallel(
            mlp(
                encoding_size + self.action_space_size,
                fc_dynamics_layers,
                encoding_size,
            )
        )
        self.dynamics_reward_network = torch.nn.DataParallel(
            mlp(encoding_size, fc_reward_layers, self.full_support_size)
        )

        self.prediction_policy_mu_network = torch.nn.DataParallel(
            mlp(
                encoding_size,
                fc_mu_policy_layers,
                self.action_space_size,
                # output_activation=torch.nn.Tanh,
            )
        )
        self.prediction_policy_logstd_network = torch.nn.DataParallel(
            mlp(encoding_size, fc_log_std_policy_layers, self.action_space_size)
        )
        self.prediction_value_network = torch.nn.DataParallel(
            mlp(encoding_size, fc_value_layers, self.full_support_size)
        )

    def prediction(self, encoded_state):
        mu = self.prediction_policy_mu_network(encoded_state)
        log_std = self.prediction_policy_logstd_network(encoded_state)
        log_std = torch.clamp(log_std, *self.log_std_clamp)
        value = self.prediction_value_network(encoded_state)
        return mu, log_std, value

    def representation(self, observation):
        encoded_state = self.representation_network(observation.view(observation.shape[0], -1))
        # Scale encoded state between [-1, 1]
        min_encoded_state = encoded_state.min(1, keepdim=True)[0]
        max_encoded_state = encoded_state.max(1, keepdim=True)[0]
        mean_encoded_state = (max_encoded_state + min_encoded_state) / 2
        scale_encoded_state = (max_encoded_state - min_encoded_state) / 2
        scale_encoded_state[scale_encoded_state < 1e-5] += 1e-5
        encoded_state_normalized = (encoded_state - mean_encoded_state) / scale_encoded_state
        return encoded_state_normalized

    def dynamics(self, encoded_state, action):
        # Stack encoded_state with the (continuous, sampled) action vector
        x = torch.cat((encoded_state, action), dim=1)

        next_encoded_state = self.dynamics_encoded_state_network(x)

        reward = self.dynamics_reward_network(next_encoded_state)

        # Scale encoded state between [0, 1] (See paper appendix Training)
        min_next_encoded_state = next_encoded_state.min(1, keepdim=True)[0]
        max_next_encoded_state = next_encoded_state.max(1, keepdim=True)[0]
        mean_next_encoded_state = (max_next_encoded_state + min_next_encoded_state) / 2
        scale_next_encoded_state = (max_next_encoded_state - min_next_encoded_state) / 2
        scale_next_encoded_state[scale_next_encoded_state < 1e-5] += 1e-5
        next_encoded_state_normalized = (
            next_encoded_state - mean_next_encoded_state
        ) / scale_next_encoded_state

        return next_encoded_state_normalized, reward

    def initial_inference(self, observation):
        encoded_state = self.representation(observation)
        mu, log_std, value = self.prediction(encoded_state)
        # reward equal to 0 for consistency
        reward = torch.log(
            (
                torch.zeros(1, self.full_support_size)
                .scatter(1, torch.tensor([[self.full_support_size // 2]]).long(), 1.0)
                .repeat(len(observation), 1)
                .to(observation.device)
            )
        )

        return (
            value,
            reward,
            mu,
            log_std,
            encoded_state,
        )

    def recurrent_inference(self, encoded_state, action):
        next_encoded_state, reward = self.dynamics(encoded_state, action)
        mu, log_std, value = self.prediction(next_encoded_state)
        return value, reward, mu, log_std, next_encoded_state


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


# ---- staging wrapper ----
class SampledMuZeroInferenceChain(torch.nn.Module):
    """Runs the real MuZero inference chain (initial_inference then one step of
    recurrent_inference, using the mean of the sampled continuous action, i.e. `mu`, as
    the action fed to the dynamics function) end to end in a single forward() so the
    whole representation/dynamics/prediction stack traces as one graph. This wrapper
    adds no new architecture; it only sequences the two real inference entry points
    exactly as `muzero-general`'s MCTS planner does at inference time.
    """

    def __init__(self, net: MuZeroFullyConnectedNetwork):
        super().__init__()
        self.net = net

    def forward(self, observation):
        value, reward, mu, log_std, encoded_state = self.net.initial_inference(observation)
        # Sampled MuZero samples a candidate action from N(mu, exp(log_std)); we feed
        # the deterministic mean action into recurrent_inference to keep the traced
        # forward pass deterministic.
        value2, reward2, mu2, log_std2, next_encoded_state = self.net.recurrent_inference(
            encoded_state, mu
        )
        return value, reward, mu, log_std, value2, reward2, mu2, log_std2, next_encoded_state


def build_sampled_muzero():
    net = MuZeroFullyConnectedNetwork(
        observation_shape=(3, 8, 8),
        stacked_observations=0,
        action_space_size=4,
        log_std_clamp=(-5.0, 2.0),
        encoding_size=16,
        fc_reward_layers=[16],
        fc_value_layers=[16],
        fc_mu_policy_layers=[16],
        fc_log_std_policy_layers=[16],
        fc_representation_layers=[16],
        fc_dynamics_layers=[16],
        support_size=5,
    )
    return SampledMuZeroInferenceChain(net)


def example_input_sampled_muzero():
    torch.manual_seed(0)
    observation = torch.rand(2, 3, 8, 8)
    return (observation,)


MENAGERIE_ENTRIES = [
    (
        "SampledMuZero_FullyConnected",
        "build_sampled_muzero",
        "example_input_sampled_muzero",
        2021,
        "vendored-pytorch",
    ),
]
