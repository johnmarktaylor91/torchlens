# FAITHFUL PORT of https://github.com/netket/netket @ 49f400249f1b
# (netket/models/rbm.py::RBM + netket/models/autoreg.py::AbstractARNN/ARNNSequential/
#  ARNNDense + netket/nn/masked_linear.py::MaskedDense1D + netket/nn/activation.py::log_cosh)
# (original framework: JAX / Flax linen; NetKet's Neural Quantum State (NQS) models)
#
# NetKet (Carleo, Hackbusch, et al.; netket.readthedocs.io) is the standard JAX/Flax
# library for variational Monte Carlo quantum many-body simulation with neural-network
# wavefunctions. Two of its NQS architectures are ported here (JAX/Flax cannot be
# installed alongside this repo's torch-only environment, so vendoring the real Flax
# `nn.Module` subclasses directly is not possible; every layer/mechanism below is
# transcribed faithfully from the real netket source instead):
#
# 1. `RBM` (netket/models/rbm.py) -- Restricted Boltzmann Machine wavefunction ansatz
#    for Neural Quantum States: psi(s) = exp(sum_i a_i s_i) * prod_j cosh(sum_i W_ij s_i
#    + b_j), computed in log-space as log_psi = sum(log_cosh(Dense(s))) + s @ v_bias.
#    `log_cosh` is netket's numerically-stable log(cosh(x)) = |x| + log1p(exp(-2|x|))
#    - log(2) (netket/nn/activation.py::log_cosh).
#
# 2. `ARNNDense` (netket/models/autoreg.py + netket/nn/masked_linear.py) -- an
#    Autoregressive Neural Network (ARNN) NQS ansatz built from stacked
#    `MaskedDense1D` layers. Each `MaskedDense1D` applies a block-Kronecker
#    upper-triangular mask (`np.triu(ones(size,size), exclusive)` Kronecker-expanded
#    by `(in_features, out_features)`) to a single big kernel of shape
#    `(size*in_features, size*out_features)`, so output site i only depends on input
#    sites < i (first layer, `exclusive=True`) or <= i (later layers). The stack ends
#    with `Hilbert.local_size` (=2 for spin-1/2) output channels per site, giving
#    per-site conditional log-amplitudes; `AbstractARNN._normalize` (log-softmax-style
#    L2 renormalization along the local-Hilbert axis) turns those into a properly
#    normalized `conditionals_log_psi`, and `AbstractARNN.__call__` gathers the log-psi
#    of the actual sampled configuration and sums over sites (log psi(s) = sum_i
#    log_psi_cond(s_i | s_<i)). We assume the standard spin-1/2 Hilbert space
#    (`local_states=(-1, +1)`, `local_size=2`), so
#    `hilbert.states_to_local_indices(x)` is simply `(x + 1) / 2`.
#
# Both models operate on real-valued spin configurations in {-1, +1}^N and, for real
# `param_dtype` inputs (as used here), all intermediate activations (log_cosh,
# reim_selu) stay real-valued -- `reim(f)` in the real repo only branches to complex
# arithmetic when `jnp.iscomplexobj(x)`, which is false for our real recipe, so this
# port implements `selu`/`log_cosh` directly on real tensors without loss of fidelity.

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "ported-pytorch"


# ---------------------------------------------------------------------------
# netket/nn/activation.py::log_cosh
# ---------------------------------------------------------------------------


def log_cosh(x: torch.Tensor) -> torch.Tensor:
    """netket.nn.activation.log_cosh: numerically-stable log(cosh(x))."""
    sgn_x = torch.where(x < 0, -torch.ones_like(x), torch.ones_like(x))
    x = x * sgn_x
    return x + torch.log1p(torch.exp(-2.0 * x)) - math.log(2.0)


# ---------------------------------------------------------------------------
# netket/models/rbm.py::RBM
# ---------------------------------------------------------------------------


class RBM(nn.Module):
    r"""Faithful port of netket.models.RBM.

    A restricted Boltzmann Machine Neural Quantum State:
        log psi(s) = sum_j log_cosh( (W s)_j + b_j ) + v_bias . s
    equivalent to a 2-layer FFNN with log_cosh nonlinearity, summed over the
    hidden layer, plus an optional linear ("visible bias") skip term.
    """

    def __init__(
        self,
        n_sites: int,
        alpha: float = 1.0,
        use_hidden_bias: bool = True,
        use_visible_bias: bool = True,
    ):
        super().__init__()
        self.n_sites = n_sites
        n_hidden = int(alpha * n_sites)
        self.dense = nn.Linear(n_sites, n_hidden, bias=use_hidden_bias)
        self.use_visible_bias = use_visible_bias
        if use_visible_bias:
            self.visible_bias = nn.Parameter(torch.zeros(n_sites))
        nn.init.normal_(self.dense.weight, std=0.01)
        if use_hidden_bias:
            nn.init.normal_(self.dense.bias, std=0.01)
        if use_visible_bias:
            nn.init.normal_(self.visible_bias, std=0.01)

    def forward(self, spins: torch.Tensor) -> torch.Tensor:
        x = self.dense(spins)
        x = log_cosh(x)
        x = torch.sum(x, dim=-1)
        if self.use_visible_bias:
            out_bias = torch.matmul(spins, self.visible_bias)
            return x + out_bias
        return x


def build_netket_rbm():
    torch.manual_seed(0)
    model = RBM(n_sites=20, alpha=2.0, use_hidden_bias=True, use_visible_bias=True)
    model.eval()
    return model


def example_input_netket_rbm():
    # A batch of spin-1/2 configurations in {-1, +1}^N, the standard NetKet
    # Hilbert-space sample layout (real training samples come from netket's own
    # Monte Carlo sampler, which needs jax/flax; the model only consumes the
    # resulting (batch, N) real-valued spin tensor).
    torch.manual_seed(0)
    n_sites = 20
    batch_size = 8
    spins = (torch.randint(0, 2, (batch_size, n_sites)).float() * 2) - 1.0
    return (spins,)


# ---------------------------------------------------------------------------
# netket/nn/masked_linear.py::MaskedDense1D
# ---------------------------------------------------------------------------


class MaskedDense1D(nn.Module):
    """Faithful port of netket.nn.MaskedDense1D.

    1D linear transformation with an autoregressive block-Kronecker mask:
    output site i depends only on input sites < i (exclusive=True, used for
    the first ARNN layer) or <= i (exclusive=False, later layers).
    """

    def __init__(
        self, size: int, in_features: int, out_features: int, exclusive: bool, use_bias: bool = True
    ):
        super().__init__()
        self.size = size
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias

        mask = np.ones((size, size), dtype=np.float32)
        mask = np.triu(mask, int(exclusive))
        mask = np.kron(mask, np.ones((in_features, out_features), dtype=np.float32))
        self.register_buffer("mask", torch.from_numpy(mask))

        self.kernel = nn.Parameter(torch.empty(size * in_features, size * out_features))
        corr = math.sqrt(mask.size / mask.sum())
        nn.init.kaiming_normal_(self.kernel, mode="fan_in", nonlinearity="linear")
        with torch.no_grad():
            self.kernel.mul_(corr * self.mask)

        if use_bias:
            self.bias = nn.Parameter(torch.zeros(size, out_features))
        else:
            self.bias = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (batch, size, in_features)
        batch = inputs.shape[0]
        inputs_flat = inputs.reshape(batch, self.size * self.in_features)
        y = torch.matmul(inputs_flat, self.mask * self.kernel)
        y = y.reshape(batch, self.size, self.out_features)
        if self.use_bias:
            y = y + self.bias
        return y


# ---------------------------------------------------------------------------
# netket/models/autoreg.py::AbstractARNN / ARNNSequential / ARNNDense
# ---------------------------------------------------------------------------


def _selu_reim(x: torch.Tensor) -> torch.Tensor:
    # reim(selu) on a real-valued tensor: reim(f) only branches to complex
    # arithmetic when the input is complex; for real x it is exactly f(x).
    return F.selu(x)


def _normalize(log_psi: torch.Tensor, machine_pow: int) -> torch.Tensor:
    """netket.models.autoreg._normalize: L2-normalize log_psi along the local-Hilbert axis."""
    lse = torch.logsumexp(machine_pow * log_psi, dim=-1, keepdim=True)
    return log_psi - lse / machine_pow


class ARNNDense(nn.Module):
    """Faithful port of netket.models.ARNNDense (an ARNNSequential of MaskedDense1D layers).

    Assumes a spin-1/2 Hilbert space (local_states=(-1, +1), local_size=2), so
    `hilbert.states_to_local_indices(x)` is `(x + 1) / 2` and `machine_pow=2`
    (netket's default squared-amplitude normalization for real wavefunctions).
    """

    def __init__(
        self, n_sites: int, layers: int, features: int, local_size: int = 2, machine_pow: int = 2
    ):
        super().__init__()
        self.n_sites = n_sites
        self.local_size = local_size
        self.machine_pow = machine_pow

        feature_list = [features] * (layers - 1) + [local_size]
        self._layers = nn.ModuleList()
        in_features = 1
        for i in range(layers):
            self._layers.append(
                MaskedDense1D(
                    size=n_sites,
                    in_features=in_features,
                    out_features=feature_list[i],
                    exclusive=(i == 0),
                    use_bias=True,
                )
            )
            in_features = feature_list[i]

    def conditionals_log_psi(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (batch, n_sites) real spin configuration
        x = inputs.unsqueeze(-1)  # (batch, n_sites, 1)
        for i, layer in enumerate(self._layers):
            if i > 0:
                x = _selu_reim(x)
            x = layer(x)
        log_psi = _normalize(x, self.machine_pow)
        return log_psi  # (batch, n_sites, local_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # hilbert.states_to_local_indices for spin-1/2 (-1, +1) -> (0, 1)
        idx = ((inputs + 1.0) / 2.0).long().unsqueeze(-1)  # (batch, n_sites, 1)

        log_psi = self.conditionals_log_psi(inputs)  # (batch, n_sites, local_size)
        log_psi = torch.gather(log_psi, -1, idx)  # (batch, n_sites, 1)
        log_psi = log_psi.reshape(inputs.shape[0], -1).sum(dim=1)  # (batch,)
        return log_psi


def build_netket_arnn_dense():
    torch.manual_seed(0)
    model = ARNNDense(n_sites=16, layers=3, features=8, local_size=2, machine_pow=2)
    model.eval()
    return model


def example_input_netket_arnn_dense():
    torch.manual_seed(0)
    n_sites = 16
    batch_size = 8
    spins = (torch.randint(0, 2, (batch_size, n_sites)).float() * 2) - 1.0
    return (spins,)


MENAGERIE_ENTRIES = [
    (
        "Neural Quantum States RBM (NetKet)",
        "build_netket_rbm",
        "example_input_netket_rbm",
        2019,
        MENAGERIE_ZOO,
    ),
    (
        "NetKet Autoregressive NQS (ARNNDense)",
        "build_netket_arnn_dense",
        "example_input_netket_arnn_dense",
        2021,
        MENAGERIE_ZOO,
    ),
]
