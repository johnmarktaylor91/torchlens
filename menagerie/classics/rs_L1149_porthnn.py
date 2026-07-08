# SOURCE: vendored from shaandesai1/PortHNN @ main
# https://github.com/shaandesai1/PortHNN/blob/main/models/baseline.py
# https://github.com/shaandesai1/PortHNN/blob/main/models/TDHNN4.py
# "Port-Hamiltonian Neural Networks for Learning Explicit Time-Dependent
# Dynamical Systems" (Desai, Mattheakis, Roberts 2021). `base_model` (shared
# Hamiltonian MLP + symplectic permutation matrix + RK4 integrator) and
# `TDHNN4` (the port-Hamiltonian variant: adds a learned dissipation term D
# and a learned time-dependent forcing term F on top of the conservative
# Hamiltonian flow) are transcribed VERBATIM from the real repo, only
# rewriting the relative `from .baseline import *` into a same-file class
# reference. `TDHNN4.time_deriv` differentiates the Hamiltonian via
# `torch.autograd.grad(..., create_graph=True)`, so the example input must be
# a leaf tensor with `requires_grad_(True)`.
import torch
import torch.nn as nn


class base_model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_size, deltat):
        super(base_model, self).__init__()
        self.dt = deltat
        self.mlp1 = nn.Linear(input_dim, hidden_dim)
        self.mlp2 = nn.Linear(hidden_dim, hidden_dim)
        self.mlp3 = nn.Linear(hidden_dim, hidden_dim)
        self.mlp4 = nn.Linear(hidden_dim, output_size, bias=False)
        self.nonlin = torch.sin
        if input_dim % 2 == 0:
            self.M = self.permutation_tensor(input_dim)
        else:
            self.M = self.permutation_tensor(input_dim - 1)
        for layer in [self.mlp1, self.mlp2, self.mlp4, self.mlp3]:
            torch.nn.init.orthogonal_(layer.weight)

    def rk4(self, dx_dt_fn, x_t, t):
        k1 = self.dt * dx_dt_fn(x_t, t)
        k2 = self.dt * dx_dt_fn(x_t + (1.0 / 2) * k1, t + 0.5 * self.dt)
        k3 = self.dt * dx_dt_fn(x_t + (1.0 / 2) * k2, t + 0.5 * self.dt)
        k4 = self.dt * dx_dt_fn(x_t + k3, t + self.dt)
        x_tp1 = x_t + (1.0 / 6) * (k1 + k2 * 2.0 + k3 * 2.0 + k4)
        return x_tp1

    def get_H(self, x):
        h = self.nonlin(self.mlp1(x))
        h = self.nonlin(self.mlp2(h))
        h = self.nonlin(self.mlp3(h))
        fin = self.mlp4(h)
        return fin

    def permutation_tensor(self, n):
        M = torch.eye(n)
        M = torch.cat([M[n // 2 :], -M[: n // 2]])
        return M

    def next_step(self, x, t):
        init_param = x
        next_set = self.rk4(self.time_deriv, init_param, t)
        return next_set

    def time_deriv(self, x, t):
        input_vec = torch.cat([x, t.reshape(-1, 1)], 1)
        fin = self.get_H(input_vec)
        return fin


class TDHNN4(base_model):
    def __init__(self, input_dim, hidden_dim, output_size, deltat):
        super(TDHNN4, self).__init__(input_dim, hidden_dim, output_size, deltat)
        self.input_dim = input_dim
        self.f1 = nn.Linear(1, hidden_dim)
        self.f2 = nn.Linear(hidden_dim, hidden_dim)
        self.f2_ = nn.Linear(hidden_dim, hidden_dim)
        self.f3 = nn.Linear(hidden_dim, int(self.input_dim / 2), bias=False)
        self.nonlin = torch.sin

        self.d1 = nn.Parameter(torch.zeros(1, int(self.input_dim / 2)))
        self.d1 = nn.init.kaiming_normal_(self.d1)

    def get_F(self, x):
        h = self.nonlin(self.f1(x))
        h = self.nonlin(self.f2(h))
        h = self.nonlin(self.f2_(h))
        h = self.f3(h)
        return h

    def get_D(self):
        return self.d1

    def time_deriv(self, x, t):
        H0, F, D = self.get_H(x), self.get_F(t.reshape(-1, 1)), self.get_D()
        dH0 = torch.autograd.grad(H0.sum(), x, create_graph=True)[0]
        derivs = dH0 @ self.M.t()
        qdot = derivs[:, : int(self.input_dim / 2)]
        pdot = derivs[:, int(self.input_dim / 2) :]

        new_pdot = pdot + D * qdot + F
        fin_deriv = torch.cat([qdot, new_pdot], 1)
        return fin_deriv

    def forward(self, x, t):
        return self.time_deriv(x, t)


MENAGERIE_ZOO = "vendored-pytorch"


def build_porthnn():
    torch.manual_seed(0)
    model = TDHNN4(input_dim=2, hidden_dim=32, output_size=1, deltat=0.1)
    model.eval()
    return model


def example_input_porthnn():
    torch.manual_seed(0)
    x = torch.randn(4, 2, requires_grad=True)
    t = torch.rand(4)
    return (x, t)


MENAGERIE_ENTRIES = [
    ("PortHNN", "build_porthnn", "example_input_porthnn", 2021, MENAGERIE_ZOO),
]
