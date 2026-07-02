# FAITHFUL PORT of https://github.com/ys7yoo/MathDQN @ master (original framework: TensorFlow 1.x)
#   Ported file: DQN.py :: DQN.create_Q_network (identical MLP architecture is also present,
#   byte-for-byte, in DQN_v2.py :: DQN.create_Q_network, confirming this is the stable/official
#   Q-network shape used across the AAAI 2018 "MathDQN" repo revisions).
#
#   MathDQN (Wang et al., AAAI 2018) frames solving arithmetic word problems as a sequential
#   decision process: an RL agent selects which pair of numbers to combine and which operator
#   (+, -, reverse--) to apply at each step, until a single final value remains. The Q-network
#   itself -- the piece with actual "architecture" -- is a plain feedforward MLP over a
#   hand-engineered state-feature vector (`state_dim` -> 50 -> 50 -> `action_op_dim`), built
#   with raw `tf.compat.v1.placeholder` / `tf.Variable` graph-mode TF1 ops (`init_weight_variable`,
#   `init_bias_variable`, `tf.nn.relu`, `tf.matmul`). TF1 graph-mode code with
#   `tf.compat.v1.InteractiveSession` cannot run in this base env without a full legacy
#   TensorFlow 1.x install, so the MLP is transcribed faithfully into torch below: same 2
#   hidden layers of width 50, same ReLU activations, same truncated-normal weight init /
#   constant(0.01) bias init as `init_weight_variable`/`init_bias_variable`, same final
#   linear projection to `action_op_dim` (3 discrete operator actions).
#
#   Original TF1 reference (verbatim, for comparison):
#       def create_Q_network(self):
#           self.state_input = tf.compat.v1.placeholder("float",[None,self.state_dim])
#           W1 = self.init_weight_variable([self.state_dim, 50])
#           b1 = self.init_bias_variable([50])
#           h_layer_1 = tf.nn.relu(tf.matmul(self.state_input,W1) + b1)
#           W2 = self.init_weight_variable([50, 50])
#           b2 = self.init_bias_variable([50])
#           h_layer_2 = tf.nn.relu(tf.matmul(h_layer_1, W2) + b2)
#           W_action_op = self.init_weight_variable([50, self.action_op_dim])
#           b_action_op = self.init_bias_variable([self.action_op_dim])
#           self.Q_op_value = tf.matmul(h_layer_2, W_action_op) + b_action_op

import torch
import torch.nn as nn

MENAGERIE_ZOO = "ported-pytorch"


class MathDQNQNetwork(nn.Module):
    """Faithful torch port of MathDQN's `DQN.create_Q_network` (state_dim -> 50 -> 50 -> action_op_dim)."""

    def __init__(self, state_dim: int = 32, hidden_dim: int = 50, action_op_dim: int = 3):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_action_op = nn.Linear(hidden_dim, action_op_dim)
        # match init_weight_variable (truncated normal) / init_bias_variable (constant 0.01)
        for layer in (self.fc1, self.fc2, self.fc_action_op):
            nn.init.trunc_normal_(layer.weight)
            nn.init.constant_(layer.bias, 0.01)

    def forward(self, state_input: torch.Tensor) -> torch.Tensor:
        h_layer_1 = torch.relu(self.fc1(state_input))
        h_layer_2 = torch.relu(self.fc2(h_layer_1))
        q_op_value = self.fc_action_op(h_layer_2)
        return q_op_value


_STATE_DIM = 32
_ACTION_OP_DIM = 3


def build_mathdqn_qnetwork():
    model = MathDQNQNetwork(state_dim=_STATE_DIM, action_op_dim=_ACTION_OP_DIM)
    model.eval()
    return model


def example_input_mathdqn_qnetwork():
    return (torch.randn(1, _STATE_DIM),)


MENAGERIE_ENTRIES = [
    ("MathDQN", build_mathdqn_qnetwork, example_input_mathdqn_qnetwork, 2018, "MENAGERIE_ZOO"),
]
