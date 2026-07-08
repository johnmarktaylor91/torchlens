# SOURCE: vendored from https://github.com/greydanus/hamiltonian-nn @ master
# (hnn.py + nn_models.py + utils.choose_nonlinearity)
#
# Hamiltonian Neural Networks (Greydanus, Dzamba, Yosinski, NeurIPS 2019).
# Learns a scalar/vector field that is decomposed into conservative +
# solenoidal components via a Levi-Civita-style permutation tensor, using
# autograd through the underlying differentiable model (an MLP here) to
# recover the Hamiltonian vector field. Only the `HNN` wrapper module and the
# `MLP` it wraps are vendored (the real repo's `PixelHNN`/autoencoder variant
# and the scipy/imageio-dependent plotting helpers in `utils.py` are not
# needed to construct/trace the core HNN architecture). No architecture was
# altered; `choose_nonlinearity` is inlined verbatim from `utils.py` so the
# module has no repo-relative imports.

import torch

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# utils.py (only choose_nonlinearity, used by nn_models.MLP)
# ---------------------------------------------------------------------------


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
        nl = lambda x: x * torch.sigmoid(x)  # noqa: E731 (kept for fidelity)
    else:
        raise ValueError("nonlinearity not recognized")
    return nl


# ---------------------------------------------------------------------------
# nn_models.py
# ---------------------------------------------------------------------------


class MLP(torch.nn.Module):
    """Just a salt-of-the-earth MLP"""

    def __init__(self, input_dim, hidden_dim, output_dim, nonlinearity="tanh"):
        super(MLP, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = torch.nn.Linear(hidden_dim, output_dim, bias=None)

        for l in [self.linear1, self.linear2, self.linear3]:  # noqa: E741 (kept for fidelity)
            torch.nn.init.orthogonal_(l.weight)  # use a principled initialization

        self.nonlinearity = choose_nonlinearity(nonlinearity)

    def forward(self, x, separate_fields=False):
        h = self.nonlinearity(self.linear1(x))
        h = self.nonlinearity(self.linear2(h))
        return self.linear3(h)


# ---------------------------------------------------------------------------
# hnn.py
# ---------------------------------------------------------------------------


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


def rk4(fun, y0, t, dt, *args, **kwargs):
    dt2 = dt / 2.0
    k1 = fun(y0, t, *args, **kwargs)
    k2 = fun(y0 + dt2 * k1, t + dt2, *args, **kwargs)
    k3 = fun(y0 + dt2 * k2, t + dt2, *args, **kwargs)
    k4 = fun(y0 + dt * k3, t + dt, *args, **kwargs)
    dy = dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return dy


# ---------------------------------------------------------------------------
# Menagerie staging glue (not part of the original repo).
# ---------------------------------------------------------------------------

_INPUT_DIM = 2  # canonical (q, p) pair, as used by the repo's experiment-pend/spring setups
_HIDDEN_DIM = 32


def build_hnn():
    torch.manual_seed(0)
    differentiable_model = MLP(_INPUT_DIM, _HIDDEN_DIM, 2, nonlinearity="tanh")
    model = HNN(
        _INPUT_DIM,
        differentiable_model,
        field_type="solenoidal",
        baseline=False,
        assume_canonical_coords=True,
    )
    model.eval()
    return model


def example_input_hnn():
    torch.manual_seed(0)
    # HNN.time_derivative differentiates the model output w.r.t. x via
    # torch.autograd.grad, so the input tensor must require grad; traced
    # through `forward` (the `__call__` entry point) which just needs a
    # [batch, input_dim] tensor.
    x = torch.randn(4, _INPUT_DIM, requires_grad=True)
    return (x,)


MENAGERIE_ENTRIES = [
    ("Hamiltonian Neural Network", "build_hnn", "example_input_hnn", 2019, MENAGERIE_ZOO),
]
