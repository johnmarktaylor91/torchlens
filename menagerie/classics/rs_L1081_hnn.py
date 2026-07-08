# SOURCE: vendored from greydanus/hamiltonian-nn @ master
# https://github.com/greydanus/hamiltonian-nn
# Files combined: hnn.py, nn_models.py, utils.py (choose_nonlinearity, rk4)
# Sam Greydanus, Misko Dzamba, Jason Yosinski | "Hamiltonian Neural Networks" | NeurIPS 2019
#
# Architecture is unmodified from the original repo; only the cross-file imports
# (`from nn_models import MLP`, `from utils import rk4, choose_nonlinearity`) were
# inlined into a single staging module and forward() was adapted to keep autograd
# graph creation compatible with torchlens capture (the original relies on
# `torch.autograd.grad(..., create_graph=True)` inside `time_derivative`, which is
# not exercised by a plain `forward()` trace and is left untouched below).

import torch


def choose_nonlinearity(name):
    nl = None
    if name == "tanh":
        nl = torch.tanh
    elif name == "relu":
        nl = torch.relu
    elif name == "sigmoid":
        nl = torch.sigmoid
    elif name == "softplus":
        nl = torch.nn.functional.softplus
    elif name == "selu":
        nl = torch.nn.functional.selu
    elif name == "elu":
        nl = torch.nn.functional.elu
    elif name == "swish":
        nl = lambda x: x * torch.sigmoid(x)  # noqa: E731 (kept faithful to original)
    else:
        raise ValueError("nonlinearity not recognized")
    return nl


def rk4(fun, y0, t, dt, *args, **kwargs):
    dt2 = dt / 2.0
    k1 = fun(y0, t, *args, **kwargs)
    k2 = fun(y0 + dt2 * k1, t + dt2, *args, **kwargs)
    k3 = fun(y0 + dt2 * k2, t + dt2, *args, **kwargs)
    k4 = fun(y0 + dt * k3, t + dt, *args, **kwargs)
    dy = dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return dy


class MLP(torch.nn.Module):
    """Just a salt-of-the-earth MLP"""

    def __init__(self, input_dim, hidden_dim, output_dim, nonlinearity="tanh"):
        super(MLP, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = torch.nn.Linear(hidden_dim, output_dim, bias=None)

        for l in [self.linear1, self.linear2, self.linear3]:  # noqa: E741 (kept faithful to original)
            torch.nn.init.orthogonal_(l.weight)  # use a principled initialization

        self.nonlinearity = choose_nonlinearity(nonlinearity)

    def forward(self, x, separate_fields=False):
        h = self.nonlinearity(self.linear1(x))
        h = self.nonlinearity(self.linear2(h))
        return self.linear3(h)


class MLPAutoencoder(torch.nn.Module):
    """A salt-of-the-earth MLP Autoencoder + some edgy res connections"""

    def __init__(self, input_dim, hidden_dim, latent_dim, nonlinearity="tanh"):
        super(MLPAutoencoder, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = torch.nn.Linear(hidden_dim, latent_dim)

        self.linear5 = torch.nn.Linear(latent_dim, hidden_dim)
        self.linear6 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear7 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear8 = torch.nn.Linear(hidden_dim, input_dim)

        for l in [
            self.linear1,
            self.linear2,
            self.linear3,
            self.linear4,  # noqa: E741 (kept faithful to original)
            self.linear5,
            self.linear6,
            self.linear7,
            self.linear8,
        ]:
            torch.nn.init.orthogonal_(l.weight)  # use a principled initialization

        self.nonlinearity = choose_nonlinearity(nonlinearity)

    def encode(self, x):
        h = self.nonlinearity(self.linear1(x))
        h = h + self.nonlinearity(self.linear2(h))
        h = h + self.nonlinearity(self.linear3(h))
        return self.linear4(h)

    def decode(self, z):
        h = self.nonlinearity(self.linear5(z))
        h = h + self.nonlinearity(self.linear6(h))
        h = h + self.nonlinearity(self.linear7(h))
        return self.linear8(h)

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat


class HNN(torch.nn.Module):
    """Learn arbitrary vector fields that are sums of conservative and solenoidal fields"""

    def __init__(
        self,
        input_dim,
        differentiable_model,
        field_type="solenoidal",
        baseline=False,
        assume_canonical_coords=True,
    ):
        super(HNN, self).__init__()
        self.baseline = baseline
        self.differentiable_model = differentiable_model
        self.assume_canonical_coords = assume_canonical_coords
        self.M = self.permutation_tensor(input_dim)  # Levi-Civita permutation tensor
        self.field_type = field_type

    def forward(self, x):
        # traditional forward pass
        if self.baseline:
            return self.differentiable_model(x)

        y = self.differentiable_model(x)
        assert y.dim() == 2 and y.shape[1] == 2, "Output tensor should have shape [batch_size, 2]"
        return y.split(1, 1)

    def rk4_time_derivative(self, x, dt):
        return rk4(fun=self.time_derivative, y0=x, t=0, dt=dt)

    def time_derivative(self, x, t=None, separate_fields=False):
        """NEURAL ODE-STLE VECTOR FIELD"""
        if self.baseline:
            return self.differentiable_model(x)

        """NEURAL HAMILTONIAN-STLE VECTOR FIELD"""
        F1, F2 = self.forward(x)  # traditional forward pass

        conservative_field = torch.zeros_like(x)  # start out with both components set to 0
        solenoidal_field = torch.zeros_like(x)

        if self.field_type != "solenoidal":
            dF1 = torch.autograd.grad(F1.sum(), x, create_graph=True)[
                0
            ]  # gradients for conservative field
            conservative_field = dF1 @ torch.eye(*self.M.shape)

        if self.field_type != "conservative":
            dF2 = torch.autograd.grad(F2.sum(), x, create_graph=True)[
                0
            ]  # gradients for solenoidal field
            solenoidal_field = dF2 @ self.M.t()

        if separate_fields:
            return [conservative_field, solenoidal_field]

        return conservative_field + solenoidal_field

    def permutation_tensor(self, n):
        M = None
        if self.assume_canonical_coords:
            M = torch.eye(n)
            M = torch.cat([M[n // 2 :], -M[: n // 2]])
        else:
            """Constructs the Levi-Civita permutation tensor"""
            M = torch.ones(n, n)  # matrix of ones
            M *= 1 - torch.eye(n)  # clear diagonals
            M[::2] *= -1  # pattern of signs
            M[:, ::2] *= -1

            for i in range(n):  # make asymmetric
                for j in range(i + 1, n):
                    M[i, j] *= -1
        return M


class PixelHNN(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        autoencoder,
        field_type="solenoidal",
        nonlinearity="tanh",
        baseline=False,
    ):
        super(PixelHNN, self).__init__()
        self.autoencoder = autoencoder
        self.baseline = baseline

        output_dim = input_dim if baseline else 2
        nn_model = MLP(input_dim, hidden_dim, output_dim, nonlinearity)
        self.hnn = HNN(
            input_dim, differentiable_model=nn_model, field_type=field_type, baseline=baseline
        )

    def encode(self, x):
        return self.autoencoder.encode(x)

    def decode(self, z):
        return self.autoencoder.decode(z)

    def time_derivative(self, z, separate_fields=False):
        return self.hnn.time_derivative(z, separate_fields)

    def forward(self, x):
        z = self.encode(x)
        z_next = z + self.time_derivative(z)
        return self.decode(z_next)


MENAGERIE_ZOO = "vendored-pytorch"


def build_hnn():
    """Tiny HNN over a 2D (q,p) canonical phase space, baseline=False (Hamiltonian mode)."""
    nn_model = MLP(input_dim=2, hidden_dim=16, output_dim=2, nonlinearity="tanh")
    return HNN(input_dim=2, differentiable_model=nn_model, field_type="solenoidal", baseline=False)


def example_input_hnn():
    # HNN.time_derivative requires x.requires_grad_() to compute the field gradients.
    return torch.randn(4, 2, requires_grad=True)


# NOTE: PixelHNN.forward() internally calls self.time_derivative(), which invokes
# torch.autograd.grad(..., create_graph=True) on a non-leaf intermediate tensor (the
# encoder output `z`). That nested double-backward pattern is not compatible with a
# plain forward-pass torchlens trace (it fails inside torchlens' own backward-capture
# instrumentation, independent of this vendored code) and is left unregistered here;
# only the primary named candidate (HNN) is emitted.

MENAGERIE_ENTRIES = [
    ("HNN (Hamiltonian NN)", build_hnn, example_input_hnn, 2019, MENAGERIE_ZOO),
]
