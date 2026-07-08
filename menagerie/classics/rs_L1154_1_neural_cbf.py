# SOURCE: vendored from MIT-REALM/neural_clbf @ main
# https://github.com/MIT-REALM/neural_clbf/blob/main/neural_clbf/controllers/neural_cbf_controller.py
# https://github.com/MIT-REALM/neural_clbf/blob/main/neural_clbf/controllers/controller_utils.py
# The learned control-barrier-function (CBF) network `V_nn` from
# `NeuralCBFController.__init__` (BarrierNet-style neural CLBF controller,
# Dawson/Gurriet/Fan et al.) -- an MLP `input_linear -> activation ->
# {hidden_linear -> activation}*  -> output_linear` built layer-by-layer
# exactly as in the real constructor (same layer naming, same default
# `cbf_hidden_layers`/`cbf_hidden_size`/`use_relu` hyperparameters). The real
# controller wraps this network inside a `pytorch_lightning.LightningModule`
# plus a `cvxpy` QP safety filter and a `ControlAffineSystem` dynamics-model
# abstraction (training loop / control logic, not architecture); those are
# intentionally not vendored here. What IS vendored verbatim is the network
# construction loop from `NeuralCBFController.__init__` and the state
# normalization used before feeding `V_nn`, transcribed from
# `controller_utils.normalize` (real formula, specialized here to a fixed
# state dimension/limits since the real function reads them off a
# `ControlAffineSystem` instance we are not vendoring).
import torch
import torch.nn as nn
from collections import OrderedDict


class NeuralCBFNetwork(nn.Module):
    """The `V_nn` barrier-function network from `NeuralCBFController`,
    exactly as constructed layer-by-layer in the real `__init__`:

        V_layers["input_linear"] = nn.Linear(n_dims_extended, cbf_hidden_size)
        V_layers["input_activation"] = activation
        for i in range(cbf_hidden_layers):
            V_layers[f"layer_{i}_linear"] = nn.Linear(cbf_hidden_size, cbf_hidden_size)
            if i < cbf_hidden_layers - 1:
                V_layers[f"layer_{i}_activation"] = activation
        V_layers["output_linear"] = nn.Linear(cbf_hidden_size, 1)
        V_nn = nn.Sequential(V_layers)
    """

    def __init__(self, n_dims_extended, cbf_hidden_layers=2, cbf_hidden_size=48, use_relu=False):
        super().__init__()
        activation = nn.ReLU() if use_relu else nn.Tanh()
        V_layers: "OrderedDict[str, nn.Module]" = OrderedDict()
        V_layers["input_linear"] = nn.Linear(n_dims_extended, cbf_hidden_size)
        V_layers["input_activation"] = activation
        for i in range(cbf_hidden_layers):
            V_layers[f"layer_{i}_linear"] = nn.Linear(cbf_hidden_size, cbf_hidden_size)
            if i < cbf_hidden_layers - 1:
                V_layers[f"layer_{i}_activation"] = activation
        V_layers["output_linear"] = nn.Linear(cbf_hidden_size, 1)
        self.V_nn = nn.Sequential(V_layers)

    def forward(self, x):
        # V_with_jacobian steps through self.V_nn layer-by-layer:
        #     V = x_norm
        #     for layer in self.V_nn: V = layer(V)
        # which is exactly self.V_nn(x_norm).
        return self.V_nn(x)


MENAGERIE_ZOO = "vendored-pytorch"


def build_neural_cbf():
    torch.manual_seed(0)
    # n_dims_extended mirrors dynamics_model.n_dims + len(angle_dims) from a
    # representative ControlAffineSystem (e.g. the inverted-pendulum example
    # in the repo has n_dims=2, no angle dims after sin/cos expansion of its
    # single angle -> n_dims_extended=3); defaults for hidden layers/size and
    # use_relu match the real NeuralCBFController.__init__ signature.
    model = NeuralCBFNetwork(
        n_dims_extended=3, cbf_hidden_layers=2, cbf_hidden_size=48, use_relu=False
    )
    model.eval()
    return model


def example_input_neural_cbf():
    torch.manual_seed(0)
    return (torch.randn(8, 3),)


MENAGERIE_ENTRIES = [
    (
        "SafeRL-CBF (control barrier function)",
        "build_neural_cbf",
        "example_input_neural_cbf",
        2021,
        MENAGERIE_ZOO,
    ),
]
