# FAITHFUL PORT of RLAgent/h-DQN @ master (original framework: Keras)
# https://github.com/RLAgent/h-DQN (agent/hDQN.py)
#
# h-DQN (Kulkarni, Narasimhan, Saeedi & Tenenbaum 2016, arXiv:1604.06057) is a two-level
# hierarchical DQN: a meta-controller that selects subgoals and a controller/actor that
# executes primitive actions conditioned on the current subgoal. The official reference
# implementation (RLAgent/h-DQN, a maintained fork of asmith26/h-DQN) is Keras/TF1, not
# PyTorch (despite the queue notes), and is not installable/runnable in this base env
# (uses `keras.optimizers.RMSprop`/legacy `init=` kwarg from TF1-era Keras). Both networks
# are plain feedforward stacks of Dense+ReLU layers, so they are transcribed faithfully
# into torch layer-for-layer.
#
# Original (agent/hDQN.py, hDQN.get_meta_controller / hDQN.get_actor):
#   self.n_states = 6
#   self.meta_nodes = [self.n_states, 30, 30, 30, self.n_states]
#   meta = keras.models.Sequential()
#   for node in self.meta_nodes:
#       meta.add(keras.layers.Dense(node, init=self.meta_init, input_shape=(node,)))
#       meta.add(keras.layers.Activation(self.meta_activation))  # 'relu'
#
#   self.nodes = [self.n_states * 2, 30, 30, 30, self.n_states]
#   actor = keras.models.Sequential()
#   for node in self.nodes:
#       actor.add(keras.layers.Dense(node, init=self.init, input_shape=(node,)))
#       actor.add(keras.layers.Activation(self.activation))  # 'relu'
#
# Keras Sequential Dense layers infer `in_features` from the previous layer's output at
# build time (the per-layer `input_shape=(node,)` kwarg is a Keras quirk that only binds
# the very first layer's input dimensionality, not a per-layer constraint), so the
# faithful torch shape chain is meta_nodes[i-1] -> meta_nodes[i] for i > 0 (and
# n_states -> n_states for the first meta-controller layer; n_states*2 -> n_states*2 for
# the first controller layer, matching select_move's
# np.concatenate([state, goal], axis=1) input).
import torch
import torch.nn as nn

MENAGERIE_ZOO = "ported-pytorch"


class MetaController(nn.Module):
    """Meta-controller Q-network: state -> Q-values over goals (subgoals)."""

    def __init__(self, n_states: int = 6):
        super().__init__()
        nodes = [n_states, 30, 30, 30, n_states]
        layers = []
        in_dim = n_states
        for node in nodes:
            layers.append(nn.Linear(in_dim, node))
            layers.append(nn.ReLU())
            in_dim = node
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Controller(nn.Module):
    """Controller/actor Q-network: concat([state, goal]) -> Q-values over actions."""

    def __init__(self, n_states: int = 6):
        super().__init__()
        nodes = [n_states * 2, 30, 30, 30, n_states]
        layers = []
        in_dim = n_states * 2
        for node in nodes:
            layers.append(nn.Linear(in_dim, node))
            layers.append(nn.ReLU())
            in_dim = node
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_hdqn_meta_controller() -> nn.Module:
    return MetaController(n_states=6)


def example_input_hdqn_meta_controller() -> torch.Tensor:
    return torch.randn(2, 6)


def build_hdqn_controller() -> nn.Module:
    return Controller(n_states=6)


def example_input_hdqn_controller() -> torch.Tensor:
    return torch.randn(2, 12)


MENAGERIE_ENTRIES = [
    (
        "h-DQN Meta-Controller",
        build_hdqn_meta_controller,
        example_input_hdqn_meta_controller,
        2016,
        MENAGERIE_ZOO,
    ),
    (
        "h-DQN Controller",
        build_hdqn_controller,
        example_input_hdqn_controller,
        2016,
        MENAGERIE_ZOO,
    ),
]
