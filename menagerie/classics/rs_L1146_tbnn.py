# SOURCE: vendored from https://github.com/nishantp9/TBNN_PyTorch @ 72304a8304
# Tensor Basis Neural Network (TBNN): "Reynolds averaged turbulence modelling
# using deep neural networks with embedded invariance" (Ling, Kurzawski &
# Templeton, JFM 2016, https://doi.org/10.1017/jfm.2016.615).
#
# The canonical sandialabs/tbnn reference implementation is Theano/Lasagne
# (unmaintained, pre-1.0 frameworks -- not runnable in a modern base env with
# no CPU/pure-torch fallback). This vendors the independent PyTorch
# reimplementation of the same TBNN architecture (invariant scalar inputs ->
# MLP -> per-basis-tensor coefficients -> linear combination with the tensor
# basis) from nishantp9/TBNN_PyTorch, src/model.py. Only change: the
# `params` argparse-namespace constructor argument is replaced with plain
# keyword arguments (identical fields, no architectural change) so the class
# can be built without the repo's CLI arg-parsing scaffolding.

import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


class TBNN(nn.Module):
    """Tensor Basis Neural Network (TBNN).

    input (lam) --> net --> coefficients --> dot with basis tensors --> output

    Parameters
    ----------
    n_lam : int
        number of tensor invariants (scalar network input dimension)
    n_basis : int
        number of tensors in the integrity basis
    hidden_layer_dims : list[int]
        hidden layer widths
    dropout : float
        dropout probability applied after each hidden layer
    """

    def __init__(self, n_lam, n_basis, hidden_layer_dims, dropout=0.0):
        super(TBNN, self).__init__()
        self.n_lam = n_lam
        self.n_basis = n_basis
        hidden_dim = hidden_layer_dims

        layers = []
        for dim1, dim2 in zip([self.n_lam] + hidden_dim, hidden_dim):
            layer = nn.Sequential(
                nn.Linear(dim1, dim2),
                nn.Dropout(dropout),
                nn.ReLU(),
            )
            layers.append(layer)

        # last layer outputs the basis coefficients
        layers.append(nn.Linear(hidden_dim[-1], self.n_basis))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        x : dict with keys
            'lam'   -- (batch, n_lam) tensor invariants
            'basis' -- (batch, n_basis, d, d) tensor integrity basis
        Returns a dict with the reconstructed output tensor and the raw
        per-basis coefficients (mirrors the real repo's output contract).
        """
        x_lam, x_basis = x["lam"], x["basis"]

        # forward pass through NN to get output coefficients
        C = self.net(x_lam)

        # Linear combination of x['basis'] with coefficients
        out = (C.view(*C.size(), 1, 1) * x_basis).sum(dim=1)
        return {"output": out, "coefficients": C}


def build_tbnn():
    # Real repo default network shape (see src/args.py hidden_layer_dims);
    # 5 scalar invariants and 10 tensor-basis elements are the standard
    # Ling et al. (2016) 3-D incompressible turbulence closure sizes.
    return TBNN(n_lam=5, n_basis=10, hidden_layer_dims=[30, 30], dropout=0.0)


def example_input_tbnn():
    import torch

    batch = 4
    lam = torch.randn(batch, 5)
    basis = torch.randn(batch, 10, 3, 3)
    return ({"lam": lam, "basis": basis},)


MENAGERIE_ENTRIES = [
    (
        "Tensor Basis Neural Network (TBNN)",
        build_tbnn,
        example_input_tbnn,
        2016,
        MENAGERIE_ZOO,
    ),
]
