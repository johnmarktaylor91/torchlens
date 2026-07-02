# SOURCE: vendored from THGLab/iShiftML @ main (nmrpred/models/MLP.py, nmrpred/layers/dense.py,
#         nmrpred/layers/renormalization/batchrenorm.py)
#
# iShiftML (Guan et al., 2024, "Real-Time Prediction of NMR Chemical Shifts with Structural
# Uncertainty Quantification") predicts per-atom NMR chemical shifts from AEV (atomic
# environment vector, ANI-style) descriptors of a molecule's local geometry. The `MLP` model
# vendored here is its atom-wise deep-ensemble shift-prediction network: a 4-layer stack of
# `Dense` (Linear + optional BatchRenorm + activation + dropout) blocks mapping a 384-dim AEV
# feature vector to a scalar shift prediction per atom. Vendored verbatim (architecture-relevant
# classes only). No architectural changes.
#
# MENAGERIE_ZOO = "vendored-pytorch"

import torch
from torch import nn
from torch.nn import ReLU
from torch.nn.init import xavier_uniform_, zeros_

MENAGERIE_ZOO = "vendored-pytorch"


# --- nmrpred/layers/renormalization/batchrenorm.py (verbatim; adapted upstream from
#     https://github.com/ludvb/batchrenorm) ---
class BatchRenorm(torch.nn.Module):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-3,
        momentum: float = 0.01,
        affine: bool = True,
    ):
        super(BatchRenorm, self).__init__()
        self.register_buffer("running_mean", torch.zeros(num_features, dtype=torch.float))
        self.register_buffer("running_std", torch.ones(num_features, dtype=torch.float))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))
        self.weight = torch.nn.Parameter(torch.ones(num_features, dtype=torch.float))
        self.bias = torch.nn.Parameter(torch.zeros(num_features, dtype=torch.float))
        self.affine = affine
        self.eps = eps
        self.step = 0
        self.momentum = momentum

    def _check_input_dim(self, x: torch.Tensor) -> None:
        raise NotImplementedError()  # pragma: no cover

    @property
    def rmax(self) -> torch.Tensor:
        return (2 / 35000 * self.num_batches_tracked + 25 / 35).clamp_(1.0, 3.0)

    @property
    def dmax(self) -> torch.Tensor:
        return (5 / 20000 * self.num_batches_tracked - 25 / 20).clamp_(0.0, 5.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input_dim(x)
        if x.dim() > 2:
            x = x.transpose(1, -1)
        if self.training:
            dims = [i for i in range(x.dim() - 1)]
            batch_mean = x.mean(dims)
            batch_std = x.std(dims, unbiased=False) + self.eps
            r = (batch_std.detach() / self.running_std.view_as(batch_std)).clamp_(
                1 / self.rmax, self.rmax
            )
            d = (
                (batch_mean.detach() - self.running_mean.view_as(batch_mean))
                / self.running_std.view_as(batch_std)
            ).clamp_(-self.dmax, self.dmax)
            x = (x - batch_mean) / batch_std * r + d
            self.running_mean += self.momentum * (batch_mean.detach() - self.running_mean)
            self.running_std += self.momentum * (batch_std.detach() - self.running_std)
            self.num_batches_tracked += 1
        else:
            x = (x - self.running_mean) / self.running_std
        if self.affine:
            x = self.weight * x + self.bias
        if x.dim() > 2:
            x = x.transpose(1, -1)
        return x


class BatchRenorm1d(BatchRenorm):
    def _check_input_dim(self, x: torch.Tensor) -> None:
        if x.dim() not in [2, 3]:
            raise ValueError("expected 2D or 3D input (got {x.dim()}D input)")


# --- nmrpred/layers/dense.py (verbatim) ---
class Dense(nn.Linear):
    r"""
    Fully connected linear layer with activation function.
    Originally borrowed from https://github.com/atomistic-machine-learning/schnetpack,
    and added dropout and batch normalization.
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        activation=None,
        weight_init=xavier_uniform_,
        xavier_init_gain=1.0,
        bias_init=zeros_,
        dropout=None,
        norm=None,
    ):
        self.weight_init = weight_init
        self.gain = xavier_init_gain
        self.bias_init = bias_init
        super(Dense, self).__init__(in_features, out_features, bias)
        self.activation = activation
        # initialize linear layer y = xW^T + b

        self.dropout = dropout
        if dropout:
            self.dropout = nn.Dropout(dropout)

        self.norm = norm
        if norm:
            self.norm = BatchRenorm1d(num_features=out_features)

    def reset_parameters(self):
        self.weight_init(self.weight, gain=self.gain)
        if self.bias is not None:
            self.bias_init(self.bias)

    def forward(self, x):
        # compute linear layer y = xW^T + b
        x = super(Dense, self).forward(x)

        # batch normalization
        if self.norm:
            x = self.norm(x)

        # add activation function
        if self.activation:
            x = self.activation(x)

        # dropout
        if self.dropout:
            x = self.dropout(x)

        return x


# --- nmrpred/models/MLP.py (verbatim) ---
class MLP(nn.Module):
    """Multiple Layer Perceptron model for AEV-based representation."""

    def __init__(self, dropout):
        super().__init__()
        self.hiddens = nn.ModuleList(
            [
                Dense(384, 128, activation=ReLU(), dropout=dropout),
                Dense(128, 128, activation=ReLU(), dropout=dropout),
                Dense(128, 128, activation=ReLU(), dropout=dropout),
                Dense(128, 1),
            ]
        )

    def forward(self, input):
        x = input["aev"]
        for i in range(len(self.hiddens)):
            x = self.hiddens[i](x)
        return x


def build_ishiftml_mlp():
    torch.manual_seed(0)
    return MLP(dropout=0.1)


def example_input_ishiftml_mlp():
    # MLP.forward reads a dict with key "aev": a per-atom 384-dim AEV (ANI-style atomic
    # environment vector) descriptor batch, matching the real xyz_to_aev featurizer's output
    # dimensionality used throughout iShiftML's training/inference pipeline.
    torch.manual_seed(0)
    n_atoms = 12
    return ({"aev": torch.randn(n_atoms, 384)},)


MENAGERIE_ENTRIES = [
    ("iShiftML_MLP", "build_ishiftml_mlp", "example_input_ishiftml_mlp", 2024, "vendored-pytorch"),
]
