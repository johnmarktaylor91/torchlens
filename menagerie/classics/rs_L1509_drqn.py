# FAITHFUL PORT of noamsgl/DRQN @ main (original framework: TensorFlow/Keras)
# https://github.com/noamsgl/DRQN
# Source file: RL211_HW5_V3.py, class Q_Learn.init_model(), branch `network == "DRQN"`:
#
#   self.model = tf.keras.models.Sequential([
#       layers.Conv2D(filters=32, kernel_size=(8, 8), strides=4, activation="relu",
#                     input_shape=(84, 84, 1)),
#       layers.Conv2D(filters=64, kernel_size=(4, 4), strides=2, activation="relu"),
#       layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1, activation="relu"),
#       layers.Reshape((1, 3136)),
#       layers.LSTM(units=256),
#       layers.Dense(units=self.nA)
#   ])
#
# This is the classic "Deep Recurrent Q-Network" (Hausknecht & Stone 2015) architecture: the
# Nature-DQN convolutional stack (8x8/s4 -> 4x4/s2 -> 3x3/s1, all ReLU) feeding a single-step
# LSTM memory (replacing DQN's frame-stack with genuine recurrence) and a linear Q-value head.
# The repo is TF1/Keras (`tensorflow.keras.layers`), not a base-env library here, so the network
# is transcribed faithfully into torch layer-for-layer (same conv channel/kernel/stride sequence,
# same flatten-to-LSTM reshape, same single Dense Q-head) rather than vendored verbatim.
import torch
import torch.nn as nn

MENAGERIE_ZOO = "ported-pytorch"


class DRQN(nn.Module):
    """Faithful torch port of the Q_Learn "DRQN" Keras Sequential model."""

    def __init__(self, in_channels: int = 1, num_actions: int = 4, lstm_units: int = 256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.relu = nn.ReLU()
        # Keras Reshape((1, 3136)) treats the flattened conv output as a length-1 sequence fed
        # to LSTM(units=256); nn.LSTM with batch_first=True over a single timestep reproduces
        # this exactly (LSTM returns only the final hidden state, matching Keras's default
        # return_sequences=False).
        self.lstm = nn.LSTM(input_size=3136, hidden_size=lstm_units, batch_first=True)
        self.q_head = nn.Linear(lstm_units, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.reshape(x.shape[0], 1, -1)
        _, (h_n, _) = self.lstm(x)
        q_values = self.q_head(h_n.squeeze(0))
        return q_values


def build_drqn() -> nn.Module:
    return DRQN(in_channels=1, num_actions=4, lstm_units=256)


def example_input_drqn() -> torch.Tensor:
    return torch.randn(2, 1, 84, 84)


MENAGERIE_ENTRIES = [
    (
        "DRQN_deep_recurrent_q_network",
        build_drqn,
        example_input_drqn,
        2015,
        "ported-pytorch",
    ),
]
