# SOURCE: vendored from https://github.com/MarcoMeter/recurrent-ppo-truncated-bptt @ master
# (model.py: ActorCriticModel, lines 1-153)
#
# Recurrent PPO with truncated backpropagation-through-time + burn-in (Marco "MarcoMeter" Pleines).
# This is the widely-cited canonical clean baseline implementation of adding a recurrent core
# (GRU or LSTM) to PPO with sequence-chunked truncated BPTT and an optional burn-in phase for the
# recurrent hidden state -- distinct from vanilla PPO (stateless MLP/CNN policy) via the explicit
# recurrent core sitting between the observation encoder and the (decoupled) policy/value heads,
# and via a multi-discrete/multi-branch categorical policy head (`policy_branches`, one linear
# head per action-space dimension). The `ActorCriticModel` class is vendored verbatim; only the
# gym `observation_space`/`action_space` duck-typed inputs it expects are supplied via a minimal
# local stand-in (below) so the real class can be constructed without an installed `gym`/
# `gymnasium` dependency -- no architecture/layer inside `ActorCriticModel` was changed.

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F

MENAGERIE_ZOO = "vendored-pytorch"


# ---- model.py (vendored verbatim) ----
class ActorCriticModel(nn.Module):
    def __init__(self, config, observation_space, action_space_shape):
        """Model setup

        Arguments:
            config {dict} -- Configuration and hyperparameters of the environment, trainer and model.
            observation_space {box} -- Properties of the agent's observation space
            action_space_shape {tuple} -- Dimensions of the action space
        """
        super().__init__()
        self.hidden_size = config["hidden_layer_size"]
        self.recurrence = config["recurrence"]
        self.observation_space_shape = observation_space.shape

        # Observation encoder
        if len(self.observation_space_shape) > 1:
            # Case: visual observation is available
            # Visual encoder made of 3 convolutional layers
            self.conv1 = nn.Conv2d(
                observation_space.shape[0],
                32,
                8,
                4,
            )
            self.conv2 = nn.Conv2d(32, 64, 4, 2, 0)
            self.conv3 = nn.Conv2d(64, 64, 3, 1, 0)
            nn.init.orthogonal_(self.conv1.weight, np.sqrt(2))
            nn.init.orthogonal_(self.conv2.weight, np.sqrt(2))
            nn.init.orthogonal_(self.conv3.weight, np.sqrt(2))
            # Compute output size of convolutional layers
            self.conv_out_size = self.get_conv_output(observation_space.shape)
            in_features_next_layer = self.conv_out_size
        else:
            # Case: vector observation is available
            in_features_next_layer = observation_space.shape[0]

        # Recurrent layer (GRU or LSTM)
        if self.recurrence["layer_type"] == "gru":
            self.recurrent_layer = nn.GRU(
                in_features_next_layer, self.recurrence["hidden_state_size"], batch_first=True
            )
        elif self.recurrence["layer_type"] == "lstm":
            self.recurrent_layer = nn.LSTM(
                in_features_next_layer, self.recurrence["hidden_state_size"], batch_first=True
            )
        # Init recurrent layer
        for name, param in self.recurrent_layer.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, np.sqrt(2))

        # Hidden layer
        self.lin_hidden = nn.Linear(self.recurrence["hidden_state_size"], self.hidden_size)
        nn.init.orthogonal_(self.lin_hidden.weight, np.sqrt(2))

        # Decouple policy from value
        # Hidden layer of the policy
        self.lin_policy = nn.Linear(self.hidden_size, self.hidden_size)
        nn.init.orthogonal_(self.lin_policy.weight, np.sqrt(2))

        # Hidden layer of the value function
        self.lin_value = nn.Linear(self.hidden_size, self.hidden_size)
        nn.init.orthogonal_(self.lin_value.weight, np.sqrt(2))

        # Outputs / Model heads
        # Policy (Multi-discrete categorical distribution)
        self.policy_branches = nn.ModuleList()
        for num_actions in action_space_shape:
            actor_branch = nn.Linear(in_features=self.hidden_size, out_features=num_actions)
            nn.init.orthogonal_(actor_branch.weight, np.sqrt(0.01))
            self.policy_branches.append(actor_branch)

        # Value function
        self.value = nn.Linear(self.hidden_size, 1)
        nn.init.orthogonal_(self.value.weight, 1)

    def forward(
        self,
        obs: torch.tensor,
        recurrent_cell: torch.tensor,
        device: torch.device,
        sequence_length: int = 1,
    ):
        """Forward pass of the model

        Arguments:
            obs {torch.tensor} -- Batch of observations
            recurrent_cell {torch.tensor} -- Memory cell of the recurrent layer
            device {torch.device} -- Current device
            sequence_length {int} -- Length of the fed sequences. Defaults to 1.

        Returns:
            {Categorical} -- Policy: Categorical distribution
            {torch.tensor} -- Value Function: Value
            {tuple} -- Recurrent cell
        """
        # Set observation as input to the model
        h = obs
        # Forward observation encoder
        if len(self.observation_space_shape) > 1:
            batch_size = h.size()[0]
            # Propagate input through the visual encoder
            h = F.relu(self.conv1(h))
            h = F.relu(self.conv2(h))
            h = F.relu(self.conv3(h))
            # Flatten the output of the convolutional layers
            h = h.reshape((batch_size, -1))

        # Forward reccurent layer (GRU or LSTM)
        if sequence_length == 1:
            # Case: sampling training data or model optimization using sequence length == 1
            h, recurrent_cell = self.recurrent_layer(h.unsqueeze(1), recurrent_cell)
            h = h.squeeze(1)  # Remove sequence length dimension
        else:
            # Case: Model optimization given a sequence length > 1
            # Reshape the to be fed data to batch_size, sequence_length, data
            h_shape = tuple(h.size())
            h = h.reshape((h_shape[0] // sequence_length), sequence_length, h_shape[1])

            # Forward recurrent layer
            h, recurrent_cell = self.recurrent_layer(h, recurrent_cell)

            # Reshape to the original tensor size
            h_shape = tuple(h.size())
            h = h.reshape(h_shape[0] * h_shape[1], h_shape[2])

        # The output of the recurrent layer is not activated as it already utilizes its own activations.

        # Feed hidden layer
        h = F.relu(self.lin_hidden(h))

        # Decouple policy from value
        # Feed hidden layer (policy)
        h_policy = F.relu(self.lin_policy(h))
        # Feed hidden layer (value function)
        h_value = F.relu(self.lin_value(h))
        # Head: Value function
        value = self.value(h_value).reshape(-1)
        # Head: Policy
        pi = [Categorical(logits=branch(h_policy)) for branch in self.policy_branches]

        return pi, value, recurrent_cell

    def get_conv_output(self, shape: tuple) -> int:
        """Computes the output size of the convolutional layers by feeding a dummy tensor.

        Arguments:
            shape {tuple} -- Input shape of the data feeding the first convolutional layer

        Returns:
            {int} -- Number of output features returned by the utilized convolutional layers
        """
        o = self.conv1(torch.zeros(1, *shape))
        o = self.conv2(o)
        o = self.conv3(o)
        return int(np.prod(o.size()))

    def init_recurrent_cell_states(self, num_sequences: int, device: torch.device) -> tuple:
        """Initializes the recurrent cell states (hxs, cxs) as zeros.

        Arguments:
            num_sequences {int} -- The number of sequences determines the number of the to be generated initial recurrent cell states.
            device {torch.device} -- Target device.

        Returns:
            {tuple} -- Depending on the used recurrent layer type, just hidden states (gru) or both hidden states and
                     cell states are returned using initial values.
        """
        hxs = torch.zeros(
            (num_sequences),
            self.recurrence["hidden_state_size"],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        cxs = None
        if self.recurrence["layer_type"] == "lstm":
            cxs = torch.zeros(
                (num_sequences),
                self.recurrence["hidden_state_size"],
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)
        return hxs, cxs


# ---- end vendored model.py ----


class _BoxSpaceStub:
    """Minimal duck-typed stand-in for a gym/gymnasium `Box` observation space --
    `ActorCriticModel.__init__` only reads `observation_space.shape`, so no real
    gym/gymnasium dependency is needed to construct the real class."""

    def __init__(self, shape):
        self.shape = shape


def build_rppo_vector():
    """Vector-observation configuration (matches the repo's `CartPole`/`PoC Memory Task`
    default configs: GRU recurrent core, sequence_length-driven truncated BPTT)."""
    config = {
        "hidden_layer_size": 64,
        "recurrence": {
            "layer_type": "gru",
            "hidden_state_size": 64,
        },
    }
    observation_space = _BoxSpaceStub(shape=(8,))
    action_space_shape = (4,)
    return ActorCriticModel(config, observation_space, action_space_shape)


def example_input_rppo_vector():
    batch_size, sequence_length = 3, 5
    obs = torch.randn(batch_size * sequence_length, 8)
    recurrent_cell = torch.zeros(1, batch_size, 64)
    device = torch.device("cpu")
    return obs, recurrent_cell, device, sequence_length


MENAGERIE_ENTRIES = [
    (
        "Recurrent PPO (truncated BPTT)",
        "build_rppo_vector",
        "example_input_rppo_vector",
        2021,
        "vendored-pytorch",
    ),
]
