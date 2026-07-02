# FAITHFUL PORT of google-research/pddm @ master (original framework: TensorFlow 1.x)
#
# https://raw.githubusercontent.com/google-research/pddm/master/pddm/regressors/feedforward_network.py
# https://raw.githubusercontent.com/google-research/pddm/master/pddm/regressors/dynamics_model.py
# https://raw.githubusercontent.com/google-research/pddm/master/pddm/config/cheetah.txt
#
# PDDM (Nagabandi et al. 2019, "Deep Dynamics Models for Learning Dexterous Manipulation")
# learns an ensemble of feedforward dynamics models used inside an MPPI/CEM model-predictive
# controller. The repo (google-research/pddm, TF1) requires MuJoCo + `tf.contrib`
# (`tf.layers.flatten`, `tf.contrib.layers.fully_connected`, `tf.contrib.layers.xavier_initializer`,
# `tf.variable_scope`) which is TF1-only API removed from modern TensorFlow and cannot be
# reasonably installed alongside this repo's torch stack -- so the network architecture (not
# the MuJoCo envs/MPPI controller, which are algorithm/task plumbing, not the learnable NN) is
# faithfully transcribed here into base-env torch.
#
# Ported faithfully from `pddm/regressors/feedforward_network.py`'s `feedforward_network()`
# (flatten -> `num_fc_layers` Linear layers with ReLU between hidden layers, Xavier/glorot-normal
# init on every Linear per `tf.contrib.layers.xavier_initializer(uniform=False)`, identity
# activation on the last layer) and `pddm/regressors/dynamics_model.py`'s `Dyn_Model` (a
# bootstrap ensemble of `ensemble_size` independently-initialized copies of that same
# feedforward network, each consuming a flattened `[K, inputSize]` window per the
# `create_placeholders` shape `[ensemble_size, None, K, inputSize]`, with the last `acSize`
# input channels clipped to [-1, 1] before the forward pass, per `Dyn_Model.__init__`).
# Hyperparameters (`num_fc_layers=2`, `depth_fc_layers=250`, `ensemble_size=3`, `K=1`) taken
# from `pddm/config/cheetah.txt`, the repo's flagship MuJoCo HalfCheetah config; `inputSize`/
# `outputSize`/`acSize` set to small stand-in dimensions (state+action -> next-state deltas)
# for tracing, matching the real placeholder shapes `[ensemble_size, batch, K, inputSize]` and
# `[batch, outputSize]`.
#
# MENAGERIE_ZOO = "ported-pytorch"

import torch
import torch.nn as nn

MENAGERIE_ZOO = "ported-pytorch"


class FeedforwardNetwork(nn.Module):
    """Faithful port of `feedforward_network()` in pddm/regressors/feedforward_network.py:
    flatten the [K, inputSize] window, then `num_fc_layers` Linear+ReLU hidden layers followed
    by one Linear output layer with no activation, all Xavier/glorot-normal initialized
    (matching `tf.contrib.layers.xavier_initializer(uniform=False)`).
    """

    def __init__(self, input_size: int, output_size: int, num_fc_layers: int, depth_fc_layers: int):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        layers = []
        in_dim = input_size
        for _ in range(num_fc_layers):
            layers.append(nn.Linear(in_dim, depth_fc_layers))
            in_dim = depth_fc_layers
        self.hidden_layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(in_dim, output_size)

        for layer in [*self.hidden_layers, self.output_layer]:
            nn.init.xavier_normal_(layer.weight)
            nn.init.xavier_normal_(layer.bias.unsqueeze(0))

    def forward(self, input_state: torch.Tensor) -> torch.Tensor:
        # input_state: (batch, K, input_size) -> tf.layers.flatten -> (batch, K * input_size)
        x = input_state.reshape(input_state.shape[0], -1)
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
        return self.output_layer(x)


class DynEnsemble(nn.Module):
    """Faithful port of the learnable core of `Dyn_Model` in
    pddm/regressors/dynamics_model.py: a bootstrap ensemble of `ensemble_size` independently
    initialized `FeedforwardNetwork`s (one per `scope=i` in `define_forward_pass`), each fed
    the same batch of `[K, inputSize]` windows (real training feeds each ensemble member a
    different bootstrap resample of the dataset; the architecture -- what this staging module
    exists to capture -- is identical per member), with the last `ac_size` input channels
    clipped to [-1, 1] first (mirrors `tf.clip_by_value(second, -1, 1)` on the action slice of
    `self.inputs_` in `Dyn_Model.__init__`, guarding against MPPI/CEM proposing actions outside
    the valid action range).
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        ac_size: int,
        num_fc_layers: int = 2,
        depth_fc_layers: int = 250,
        ensemble_size: int = 3,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.ac_size = ac_size
        self.ensemble_size = ensemble_size
        self.networks = nn.ModuleList(
            [
                FeedforwardNetwork(input_size, output_size, num_fc_layers, depth_fc_layers)
                for _ in range(ensemble_size)
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (ensemble_size, batch, K, input_size)
        state_part, action_part = torch.split(
            inputs, [self.input_size - self.ac_size, self.ac_size], dim=-1
        )
        action_part = torch.clamp(action_part, -1.0, 1.0)
        inputs_clipped = torch.cat([state_part, action_part], dim=-1)

        outputs = [self.networks[i](inputs_clipped[i]) for i in range(self.ensemble_size)]
        return torch.stack(outputs, dim=0)


# ---------------------------------------------------------------------------
# Menagerie staging entrypoints.
# ---------------------------------------------------------------------------


def build_pddm():
    torch.manual_seed(0)
    # pddm/config/cheetah.txt: num_fc_layers=2, depth_fc_layers=250, ensemble_size=3, K=1.
    # inputSize/outputSize/acSize are env-observation-dependent in the real repo (built from
    # MuJoCo HalfCheetah's obs/action dims); small stand-in dims used here for tracing.
    return DynEnsemble(
        input_size=20,
        output_size=17,
        ac_size=6,
        num_fc_layers=2,
        depth_fc_layers=250,
        ensemble_size=3,
    )


def example_input_pddm():
    torch.manual_seed(0)
    ensemble_size, batch, k, input_size = 3, 4, 1, 20
    return (torch.randn(ensemble_size, batch, k, input_size),)


MENAGERIE_ENTRIES = [
    ("PDDM (deep dynamics ensemble)", "build_pddm", "example_input_pddm", 2019, MENAGERIE_ZOO),
]
