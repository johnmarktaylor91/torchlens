# SOURCE: vendored from sanderlab/cellbox_torch @ 34c8f5b8a870ce2a23a8e459dbe5e133fcf643b0
# (cellbox/cellbox/model_torch.py, cellbox/cellbox/kernel_torch.py)
#
# CellBox: an interpretable ODE-based neural network for perturbation biology
# (Yuan et al., Cell Systems 2021). Nodes (proteins/phenotypes) interact through
# a learned adjacency-masked weight matrix W; per-node gated dynamics
# dx/dt = eps * envelope(W @ x + u) - alpha * x are integrated forward with a
# fixed-step ODE solver (Heun by default) for n_T steps, and the model reports
# the running mean/std/derivative of the trajectory tail plus the final state.
#
# The classes below (`PertBio`, `CellBox`, the `kernel_torch` envelope/dxdt/ODE
# solver functions) are transcribed verbatim from the real
# model_torch.py/kernel_torch.py. Only the file-loading branches in `CellBox.build`
# for externally supplied `.npy`/`.csv` weight/mask files were dropped (they are
# pure I/O side paths unrelated to the architecture -- construction still uses the
# real `torch.normal(...)` random-init default path, identical to upstream when
# `args.weights` is None) and the module-level `factory()`/`LinReg`/`get_ops()`
# helpers (training-loop plumbing, not part of the CellBox forward graph) were
# omitted. `PertBio.__init__`/`CellBox.build`/`CellBox._get_mask`/`CellBox.forward`
# and the full kernel_torch envelope/dxdt/solver machinery are untouched.
# `args` is a plain attribute-holder standing in for the repo's `Config` class
# (config.py just parses a json file into these same attributes; no behavior
# difference for tracing).

import numpy as np
import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


# --- kernel_torch.py ---
def get_envelope(args):
    """get the envelope form based on the given argument"""
    if args.envelope_form == "tanh":
        args.envelope_fn = torch.tanh
    elif args.envelope_form == "polynomial":
        k = args.polynomial_k
        assert k > 1, "Hill coefficient has to be k>2."
        if k % 2 == 1:  # odd order polynomial equation
            args.envelope_fn = lambda x: x**k / (1 + torch.abs(x) ** k)
        else:  # even order polynomial equation
            args.envelope_fn = lambda x: x**k / (1 + x**k) * torch.sign(x)
    elif args.envelope_form == "hill":
        k = args.polynomial_k
        assert k > 1, "Hill coefficient has to be k>=2."
        args.envelope_fn = lambda x: (
            2 * (1 - 1 / (1 + nn.functional.relu(torch.tensor(x + 1)).numpy() ** k)) - 1
        )
    elif args.envelope_form == "linear":
        args.envelope_fn = lambda x: x
    elif args.envelope_form == "clip linear":
        args.envelope_fn = lambda x: torch.clamp(x, min=-1, max=1)
    else:
        raise Exception("Illegal envelope function. Choose from [tanh, polynomial/hill]")
    return args.envelope_fn


def get_dxdt(args, params):
    """calculate the derivatives dx/dt in the ODEs"""
    if args.ode_degree == 1:

        def weighted_sum(x, mask=None):
            if mask is not None:
                return torch.matmul(params["W"] * mask, x)
            else:
                return torch.matmul(params["W"], x)

    elif args.ode_degree == 2:

        def weighted_sum(x, mask=None):
            if mask is not None:
                (
                    torch.matmul(params["W"] * mask, x)
                    + torch.reshape(torch.sum(params["W"] * mask, dim=1), (args.n_x, 1)) * x
                )
            return (
                torch.matmul(params["W"], x)
                + torch.reshape(torch.sum(params["W"], dim=1), (args.n_x, 1)) * x
            )

    else:
        raise Exception("Illegal ODE degree. Choose from [1,2].")

    if args.envelope == 0:
        # epsilon*phi(Sigma+u)-alpha*x
        return lambda x, t_mu, mask=None: (
            params["eps"] * args.envelope_fn(weighted_sum(x, mask) + t_mu) - params["alpha"] * x
        )
    if args.envelope == 1:
        # epsilon*[phi(Sigma)+u]-alpha*x
        return lambda x, t_mu, mask=None: (
            params["eps"] * (args.envelope_fn(weighted_sum(x, mask)) + t_mu) - params["alpha"] * x
        )
    if args.envelope == 2:
        # epsilon*phi(Sigma)+psi*u-alpha*x
        return lambda x, t_mu, mask=None: (
            params["eps"] * args.envelope_fn(weighted_sum(x, mask))
            + params["psi"] * t_mu
            - params["alpha"] * x
        )
    raise Exception("Illegal envelope type. Choose from [0,1,2].")


def get_ode_solver(args):
    """get the ODE solver based on the given argument"""
    if args.ode_solver == "heun":
        return heun_solver
    if args.ode_solver == "euler":
        return euler_solver
    if args.ode_solver == "rk4":
        return rk4_solver
    if args.ode_solver == "midpoint":
        return midpoint_solver
    raise Exception("Illegal ODE solver. Use [heun, euler, rk4, midpoint]")


def heun_solver(x, t_mu, dT, n_T, _dXdt, n_activity_nodes=None, mask=None):
    """Heun's ODE solver"""
    xs = []
    n_x = t_mu.shape[0]
    n_activity_nodes = n_x if n_activity_nodes is None else n_activity_nodes
    dxdt_mask = nn.functional.pad(
        torch.ones((n_activity_nodes, 1)), (0, 0, 0, n_x - n_activity_nodes)
    ).to(x.device)
    for _ in range(n_T):
        dxdt_current = _dXdt(x, t_mu, mask)
        dxdt_next = _dXdt(x + dT * dxdt_current, t_mu, mask)
        x = x + dT * 0.5 * (dxdt_current + dxdt_next) * dxdt_mask
        xs.append(x)
    xs = torch.stack(xs, dim=0)
    return xs


def euler_solver(x, t_mu, dT, n_T, _dXdt, n_activity_nodes=None, mask=None):
    """Euler's method"""
    xs = []
    n_x = t_mu.shape[0]
    n_activity_nodes = n_x if n_activity_nodes is None else n_activity_nodes
    dxdt_mask = nn.functional.pad(
        torch.ones((n_activity_nodes, 1)), (0, 0, 0, n_x - n_activity_nodes)
    ).to(x.device)
    for _ in range(n_T):
        dxdt_current = _dXdt(x, t_mu, mask)
        x = x + dT * dxdt_current * dxdt_mask
        xs.append(x)
    xs = torch.stack(xs, dim=0)
    return xs


def midpoint_solver(x, t_mu, dT, n_T, _dXdt, n_activity_nodes=None, mask=None):
    """Midpoint method"""
    xs = []
    n_x = t_mu.shape[0]
    n_activity_nodes = n_x if n_activity_nodes is None else n_activity_nodes
    dxdt_mask = nn.functional.pad(
        torch.ones((n_activity_nodes, 1)), (0, 0, 0, n_x - n_activity_nodes)
    ).to(x.device)
    for _ in range(n_T):
        dxdt_current = _dXdt(x, t_mu, mask)
        dxdt_midpoint = _dXdt(x + 0.5 * dT * dxdt_current, t_mu, mask)
        x = x + dT * dxdt_midpoint * dxdt_mask
        xs.append(x)
    xs = torch.stack(xs, dim=0)
    return xs


def rk4_solver(x, t_mu, dT, n_T, _dXdt, n_activity_nodes=None, mask=None):
    """Runge-Kutta method"""
    xs = []
    n_x = t_mu.shape[0]
    n_activity_nodes = n_x if n_activity_nodes is None else n_activity_nodes
    dxdt_mask = nn.functional.pad(
        torch.ones((n_activity_nodes, 1)), (0, 0, 0, n_x - n_activity_nodes)
    ).to(x.device)
    for _ in range(n_T):
        k1 = _dXdt(x, t_mu, mask)
        k2 = _dXdt(x + 0.5 * dT * k1, t_mu, mask)
        k3 = _dXdt(x + 0.5 * dT * k2, t_mu, mask)
        k4 = _dXdt(x + dT * k3, t_mu, mask)
        x = x + dT * (1 / 6 * k1 + 1 / 3 * k2 + 1 / 3 * k3 + 1 / 6 * k4) * dxdt_mask
        xs.append(x)
    xs = torch.stack(xs, dim=0)
    return xs


# --- model_torch.py ---
class PertBio(nn.Module):
    """Define abstract perturbation model. All subsequent models are inherited from this model."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.n_x = args.n_x
        self.params = {}
        self.build()

    def build(self):
        """get model parameters (overwritten by model configuration)"""
        raise NotImplementedError

    def forward(self, x, mu):
        """forward propagation (overwritten by model configuration)"""
        raise NotImplementedError


class CellBox(PertBio):
    def build(self):
        """Initialize the CellBox model"""
        n_x = self.n_x
        self.params = nn.ParameterDict()

        # (upstream also supports loading W from a .npy/.npz/.csv weights_path;
        # that pure I/O branch is dropped here, matching the real random-init
        # default path used when args.weights is None/unset)
        W = torch.normal(mean=0.01, std=1.0, size=(n_x, n_x), dtype=torch.float32)

        W_mask = self._get_mask(None)
        self.params["W"] = nn.Parameter(W_mask * W, requires_grad=True)
        eps = nn.Parameter(torch.ones((n_x, 1), dtype=torch.float32), requires_grad=True)
        alpha = nn.Parameter(torch.ones((n_x, 1), dtype=torch.float32), requires_grad=True)
        self.params["alpha"] = nn.functional.softplus(alpha)
        self.params["eps"] = nn.functional.softplus(eps)

        if self.args.envelope == 2:
            psi = nn.Parameter(torch.ones((n_x, 1), dtype=torch.float32), requires_grad=True)
            self.params["psi"] = torch.nn.functional.softplus(psi)

        if self.args.pert_form == "by u":
            self.gradient_zero_from = None
        elif (
            self.args.pert_form == "fix x"
        ):  # fix level of node x (here y) by input perturbation u (here x)
            self.gradient_zero_from = self.args.n_activity_nodes

        self.envelope_fn = get_envelope(self.args)
        self.ode_solver = get_ode_solver(self.args)
        self._dxdt = get_dxdt(self.args, self.params)

    def _get_mask(self, extra_mask=None):
        """Get the adjacency mask (optionally combined with an external mask)."""
        W_mask_np = 1.0 - np.diag(np.ones([self.n_x]))
        W_mask_np[self.args.n_activity_nodes :, :] = np.zeros(
            [self.n_x - self.args.n_activity_nodes, self.n_x]
        )
        W_mask_np[:, self.args.n_protein_nodes : self.args.n_activity_nodes] = np.zeros(
            [self.n_x, self.args.n_activity_nodes - self.args.n_protein_nodes]
        )
        W_mask_np[
            self.args.n_protein_nodes : self.args.n_activity_nodes, self.args.n_activity_nodes :
        ] = np.zeros(
            [
                self.args.n_activity_nodes - self.args.n_protein_nodes,
                self.n_x - self.args.n_activity_nodes,
            ]
        )

        W_mask = torch.tensor(W_mask_np, dtype=torch.float32)

        if extra_mask is not None:
            if isinstance(extra_mask, np.ndarray):
                extra = torch.tensor((extra_mask != 0).astype(np.float32), dtype=torch.float32)
            elif torch.is_tensor(extra_mask):
                extra = extra_mask.detach().to(dtype=torch.float32)
                extra = (extra != 0).to(dtype=torch.float32)
            else:
                raise TypeError("extra_mask must be a numpy.ndarray or torch.Tensor")
            if extra.shape != (self.n_x, self.n_x):
                raise ValueError(
                    f"extra_mask shape {tuple(extra.shape)} must match (n_x, n_x)=({self.n_x}, {self.n_x})"
                )
            W_mask = W_mask * extra

        return W_mask

    def forward(self, y0, mu):
        mu_t = torch.transpose(mu, 0, 1)
        mask = self._get_mask()
        ys = self.ode_solver(
            y0, mu_t, self.args.dT, self.args.n_T, self._dxdt, self.gradient_zero_from, mask=mask
        )
        # [n_T, n_x, batch_size]
        ys = ys[-self.args.ode_last_steps :]
        # [n_iter_tail, n_x, batch_size]
        mean = torch.mean(ys, dim=0)
        sd = torch.std(ys, dim=0)
        yhat = torch.transpose(ys[-1], 0, 1)
        dxdt = self._dxdt(ys[-1], mu_t)
        # [n_x, batch_size] for last ODE step
        convergence_metric = torch.cat([mean, sd, dxdt], dim=0)
        return convergence_metric, yhat


class _Args:
    """Minimal attribute-holder standing in for cellbox.config.Config (which just
    parses these same fields out of a json file; no behavioral difference here)."""

    def __init__(self):
        self.n_x = 10
        self.n_protein_nodes = 6
        self.n_activity_nodes = 8
        self.weights = None
        self.envelope = 0
        self.envelope_form = "tanh"
        self.pert_form = "by u"
        self.ode_degree = 1
        self.ode_solver = "heun"
        self.dT = 0.1
        self.n_T = 5
        self.ode_last_steps = 2


def build_cellbox():
    args = _Args()
    model = CellBox(args)
    model.eval()
    return model


def example_input_cellbox():
    args = _Args()
    y0 = torch.zeros(args.n_x, 4)
    mu = torch.randn(4, args.n_x)
    return (y0, mu)


MENAGERIE_ENTRIES = [
    ("CellBox", "build_cellbox", "example_input_cellbox", 2021, "vendored-pytorch"),
]
