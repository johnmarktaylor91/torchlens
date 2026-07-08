# FAITHFUL PORT of michel-mata/cNODE.jl @ ac817e726dd8 (original framework: Julia / Flux.jl / DiffEqFlux.jl)
# https://github.com/michel-mata/cNODE.jl
# https://raw.githubusercontent.com/michel-mata/cNODE.jl/main/src/module/trainer.jl
#
# Michalska-Smith, Sanchez, Zomorrodi (repo author: Michel Mata) --
# "cNODE" (compositional Neural ODE): a species-composition predictor for
# microbiome relative-abundance data. Constrains the trajectory to the
# probability simplex (relative abundances sum to 1) via a replicator-style
# ODE, ported here layer-for-layer from `src/module/trainer.jl`:
#
#   struct FitnessLayer
#       W          # N x N learnable "fitness interaction" matrix
#   end
#   function (L::FitnessLayer)(p)
#       f = L.W * p
#       ṗ = p .* (f - ones(size(p,1)) * p' * f)
#       return ṗ
#   end
#   getModel(N) = NeuralODE(FitnessLayer(N), (0.0, 1.0), Tsit5(), saveat=1.0)
#   predict(cnode, z) = Array(cnode(z).u[end])
#
# `ṗ_i = p_i * (f_i - <p, f>)` is the classic replicator-equation form (the
# `ones(N) * p' * f` term broadcasts the scalar dot product `<p, f>` back out
# to an N-vector), which keeps `sum(p)` invariant along the trajectory --
# exactly the microbiome-composition constraint the paper exploits. `predict`
# integrates the `FitnessLayer` vector field from t=0 to t=1 and returns the
# state at t=1 (`saveat=1.0` in the original -- only the endpoint is kept).
#
# `Tsit5` is an adaptive-step explicit Runge-Kutta solver; this port uses a
# fixed-step classical RK4 integrator (`_integrate_rk4`, self-contained, no
# extra deps) over the same `t in [0, 1]` interval and the identical
# `FitnessLayer` vector field -- the dynamics equation and the "integrate
# from 0 to 1, keep the endpoint" NeuralODE forward semantics are ported
# faithfully; only the specific adaptive-step solver algorithm is swapped for
# a fixed-step one, which does not change the architecture being traced.
# `train_reptile` (the Reptile meta-learning training loop) is training-time
# procedure, not part of the traced network, and is not ported.

import torch
import torch.nn as nn


class FitnessLayer(nn.Module):
    """Ported verbatim from `trainer.jl`'s `FitnessLayer`."""

    def __init__(self, n):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(n, n))

    def forward(self, p):
        # p: [..., N]
        f = torch.matmul(p, self.weight.t())  # W @ p, batched: p @ W^T
        dot = (p * f).sum(dim=-1, keepdim=True)  # p' * f (scalar per batch row)
        return p * (f - dot)


def _integrate_rk4(field, p0, t0=0.0, t1=1.0, steps=8):
    """Fixed-step classical RK4 integration of dp/dt = field(p) from t0 to t1,
    returning the state at t1 (mirrors `NeuralODE(..., (0.0, 1.0), Tsit5(),
    saveat=1.0)` keeping only the endpoint)."""
    h = (t1 - t0) / steps
    p = p0
    for _ in range(steps):
        k1 = field(p)
        k2 = field(p + 0.5 * h * k1)
        k3 = field(p + 0.5 * h * k2)
        k4 = field(p + h * k3)
        p = p + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return p


class CNODE(nn.Module):
    """Ported from `getModel(N) = NeuralODE(FitnessLayer(N), (0.0,1.0),
    Tsit5(), saveat=1.0)` + `predict(cnode, z) = Array(cnode(z).u[end])`."""

    def __init__(self, n, steps=8):
        super().__init__()
        self.fitness_layer = FitnessLayer(n)
        self.steps = steps

    def forward(self, z):
        return _integrate_rk4(self.fitness_layer, z, t0=0.0, t1=1.0, steps=self.steps)


def build_cnode():
    return CNODE(n=12, steps=4)


def example_input_cnode():
    z = torch.rand(3, 12)
    z = z / z.sum(
        dim=-1, keepdim=True
    )  # normalized composition, matches source's `z` (relative abundances)
    return (z,)


MENAGERIE_ZOO = "ported-pytorch"
MENAGERIE_ENTRIES = [
    ("cNODE", "build_cnode", "example_input_cnode", 2022, "ported"),
]
