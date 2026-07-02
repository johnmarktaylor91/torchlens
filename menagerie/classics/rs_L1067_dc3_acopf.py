# SOURCE: vendored from https://github.com/locuslab/DC3 @ master (method.py,
# class NNSolver)
#
# DC3 ("Deep Constraint Completion and Correction", Donti, Rolnick, Kolter,
# ICLR 2021) applied to the AC Optimal Power Flow (AC-OPF) problem
# (--probType acopf57, the paper's flagship hard-constrained-optimization
# benchmark). The class below, `NNSolver`, is the REAL neural-network solver
# code from the official repo's `method.py` -- a BatchNorm-regularized MLP
# (Linear -> BatchNorm1d -> ReLU -> Dropout, repeated `hiddenSize`-wide) that
# maps problem parameters X (here: AC-OPF bus power demands) to a *partial*
# solution vector Y, which DC3 then completes via problem-specific
# (non-neural) power-flow equations and refines with gradient-based
# constraint-correction steps. No architecture was altered.
#
# The `data.complete_partial` / `data.process_output` completion step called
# at the end of the real `forward()` (the `ACOPFProblem` class in utils.py) is
# deterministic power-flow physics (admittance-matrix / Newton-style
# completion of voltage angles and reactive power from a real matpower case
# file) -- not a neural-network component, and it requires a real grid case
# (`.mat` matpower file, `pypower`/`scipy.io` case data) that is not available
# in the base menagerie env and cannot be fabricated without inventing
# electrical-grid data (forbidden by the ladder). `NNSolver` itself (both
# `__init__` and `forward`) is vendored verbatim below with NO architecture
# change. For the menagerie trace entry point, we call its real `.net`
# Sequential submodule directly instead of the full `forward()` -- `self.net`
# IS the entire neural network DC3 trains (Linear -> BatchNorm1d -> ReLU ->
# Dropout MLP); `complete_partial`/`process_output` are non-learned
# post-processing over real grid topology that this staging module does not
# attempt to fabricate.

import operator
from functools import reduce

import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


class NNSolver(nn.Module):
    """Real DC3 NNSolver, `__init__` and `forward` copied verbatim from
    method.py. `data` here only needs to expose the dimensional attributes
    the real `__init__`/`forward` read (xdim, ydim, nknowns, neq, and, for the
    real `forward`'s completion call, `complete_partial`); `args` only needs
    the hiddenSize/useCompl/probType keys the real code reads.
    """

    def __init__(self, data, args):
        super().__init__()
        self._data = data
        self._args = args
        layer_sizes = [data.xdim, self._args["hiddenSize"], self._args["hiddenSize"]]
        layers = reduce(
            operator.add,
            [
                [nn.Linear(a, b), nn.BatchNorm1d(b), nn.ReLU(), nn.Dropout(p=0.2)]
                for a, b in zip(layer_sizes[0:-1], layer_sizes[1:])
            ],
        )

        output_dim = data.ydim - data.nknowns

        if self._args["useCompl"]:
            layers += [nn.Linear(layer_sizes[-1], output_dim - data.neq)]
        else:
            layers += [nn.Linear(layer_sizes[-1], output_dim)]

        for layer in layers:
            if type(layer) == nn.Linear:
                nn.init.kaiming_normal_(layer.weight)

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # Verbatim from method.py.
        out = self.net(x)

        if self._args["useCompl"]:
            if "acopf" in self._args["probType"]:
                out = nn.Sigmoid()(out)  # used to interpolate between max and min values
            return self._data.complete_partial(x, out)
        else:
            return self._data.process_output(x, out)


# ---------------------------------------------------------------------------
# Menagerie staging glue (not part of the original repo).
# ---------------------------------------------------------------------------
class _DimsStub:
    """Exposes only the dimensional attributes NNSolver.__init__ reads
    (xdim, ydim, nknowns, neq) -- real acopf57-scale ratios (57-bus case:
    ng=7 generators, nbus=57), shrunk proportionally for a tiny CPU trace.
    """

    def __init__(self, xdim, ydim, nknowns, neq):
        self.xdim = xdim
        self.ydim = ydim
        self.nknowns = nknowns
        self.neq = neq


# acopf57 real dims (from DC3 utils.py ACOPFProblem, nbus=57 case): ydim =
# 2*ng + 2*nbus, neq = 2*nbus, nknowns = nslack (1 slack bus). Shrunk here to
# a tiny toy grid: nbus=6, ng=2, nslack=1.
_NBUS = 6
_NG = 2
_NSLACK = 1
_XDIM = 2 * _NBUS  # real+imag demand per bus
_YDIM = 2 * _NG + 2 * _NBUS  # pg, qg, |v|, angle
_NEQ = 2 * _NBUS
_BATCH = 4


def build_dc3_acopf_nnsolver():
    torch.manual_seed(0)
    data = _DimsStub(xdim=_XDIM, ydim=_YDIM, nknowns=_NSLACK, neq=_NEQ)
    args = {"hiddenSize": 16, "useCompl": True, "probType": "acopf57"}
    model = NNSolver(data, args)
    model.eval()
    # Trace the real .net Sequential directly (see module docstring: the full
    # forward() needs a real ACOPF grid case's complete_partial, unavailable
    # here). model.net IS the complete real neural architecture DC3 trains.
    return model.net


def example_input_dc3_acopf_nnsolver():
    torch.manual_seed(0)
    return torch.randn(_BATCH, _XDIM)


MENAGERIE_ENTRIES = [
    (
        "DC3-ACOPF-NNSolver-MLP",
        "build_dc3_acopf_nnsolver",
        "example_input_dc3_acopf_nnsolver",
        2021,
        MENAGERIE_ZOO,
    ),
]
