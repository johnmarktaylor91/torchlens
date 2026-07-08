# SOURCE: vendored from https://github.com/wadayama/TISTA @ master
# (TISTA.py, commit 1c2f9a2a8b880aae15ec3e56d15a37f9f00c30e4)
#
# Trainable ISTA (TISTA) for sparse signal recovery (Ito, Takabe, Wadayama,
# "Trainable ISTA for Sparse Signal Recovery", arXiv:1801.01978). Official
# PyTorch implementation by the original authors.
#
# The class below (``TISTA_NET``) is the REAL TISTA network: a per-layer
# learned step-size ``gamma`` drives an unrolled ISTA-style iteration with an
# analytic (Onsager-style) error-variance estimator ``eval_tau2`` and an
# MMSE shrinkage nonlinearity, applied against a fixed random Gaussian
# sensing matrix ``A`` (pseudo-inverse ``W`` precomputed once). No
# architecture was altered. The repo is a flat training script rather than a
# module: the module-level sensing-matrix setup (``A``, ``At``, ``W``, ``Wt``,
# ``taa``, ``tww``, ``sigma2``) that ``TISTA_NET`` reads from globals, and the
# ``generate_batch``/SNR-calibration loop used to derive ``sigma2``, are kept
# verbatim but moved into ``_setup_sensing_matrix``/``_generate_batch`` so
# they can be invoked with a small (N, M, batch_size) at staging time instead
# of the repo's N=500, M=250, batch_size=1000 defaults; only the problem size
# and device are parameterized, the math is untouched.

import math

import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"

device = torch.device("cpu")

# global variables (mirrors the repo's module-level state, sized down below)
N = 500  # length of a source signal vector
M = 250  # length of a observation vector
p = 0.1  # probability for occurrence of non-zero components
alpha2 = 1.0  # variance of non-zero component
max_layers = 12  # maximum number of layers
snr = 40.0  # SNR for the system in dB

At = None
Wt = None
taa = None
tww = None
sigma2 = None


def _generate_batch(batch_size):
    """Verbatim port of the repo's generate_batch()."""
    support = torch.bernoulli(p * torch.ones(batch_size, N))
    nonzero = torch.normal(0.0, math.sqrt(alpha2) * torch.ones(batch_size, N))
    return torch.mul(nonzero, support)


def _setup_sensing_matrix(n=N, m=M, seed=5, calib_batch=64):
    """Verbatim port of the repo's module-level sensing-matrix setup + SNR
    calibration loop (originally executed at import time for N=500, M=250,
    calibrated over 100 mini-batches of 1000); parameterized here so staging
    can use a tiny N/M/batch at staging time. Only the problem size and
    device are parameterized, the math is untouched."""
    global At, Wt, taa, tww, sigma2, N, M
    torch.manual_seed(seed)
    N, M = n, m

    # sensing matrix with small variance
    A = torch.normal(0.0, std=math.sqrt(1.0 / M) * torch.ones(M, N))

    At_ = A.t()
    W = At_.mm((A.mm(At_)).inverse())  # pseudo inverse matrix
    Wt_ = W.t()

    taa_ = (At_.mm(A)).trace().to(device)  # trace(A^T A)
    tww_ = (W.mm(Wt_)).trace().to(device)  # trace(W W^T)

    At = torch.Tensor(At_).to(device)
    Wt = torch.Tensor(Wt_).to(device)
    taa, tww = taa_, tww_

    # SNR calibration (repo's loop over generate_batch(), shrunk from
    # 100 batches of 1000 to a single small batch for staging speed).
    x = _generate_batch(calib_batch).to(device)
    y = x.mm(At)
    ave = (y.norm(2, 1).pow(2.0)).sum().item() / calib_batch
    sigma2 = ave / (M * math.pow(10.0, snr / 10.0))


class TISTA_NET(nn.Module):
    def __init__(self):
        super(TISTA_NET, self).__init__()
        self.gamma = nn.Parameter(torch.ones(max_layers))

    def gauss(self, x, var):
        return torch.exp(-torch.mul(x, x) / (2.0 * var)) / pow(2.0 * math.pi * var, 0.5)

    def MMSE_shrinkage(self, y, tau2):  # MMSE shrinkage function
        return (
            (y * alpha2 / (alpha2 + tau2))
            * p
            * self.gauss(y, (alpha2 + tau2))
            / ((1 - p) * self.gauss(y, tau2) + p * self.gauss(y, (alpha2 + tau2)))
        )

    def eval_tau2(self, t, i):  # error variance estimator
        v2 = (t.norm(2, 1).pow(2.0) - M * sigma2) / taa
        v2.clamp(min=1e-9)
        tau2 = (v2 / N) * (
            N + (self.gamma[i] * self.gamma[i] - 2.0 * self.gamma[i]) * M
        ) + self.gamma[i] * self.gamma[i] * tww * sigma2 / N
        tau2 = (tau2.expand(N, t.shape[0])).t()
        return tau2

    def forward(self, x, s, max_itr):  # TISTA network
        y = x.mm(At) + torch.normal(0.0, math.sqrt(sigma2) * torch.ones(x.shape[0], M)).to(device)
        for i in range(max_itr):
            t = y - s.mm(At)
            tau2 = self.eval_tau2(t, i)
            r = s + t.mm(Wt) * self.gamma[i]
            s = self.MMSE_shrinkage(r, tau2)
        return s


# ---------------------------------------------------------------------------
# Menagerie staging glue (not part of the original repo).
# ---------------------------------------------------------------------------

_N_TINY = 16
_M_TINY = 8
_BATCH = 2
_MAX_ITR = 3


def build_tista():
    _setup_sensing_matrix(n=_N_TINY, m=_M_TINY, seed=5)
    torch.manual_seed(0)
    model = TISTA_NET()
    model.eval()
    return model


def example_input_tista():
    torch.manual_seed(0)
    x = torch.zeros(_BATCH, _N_TINY)  # sparse source signal, same role as generate_batch() output
    s = torch.zeros(_BATCH, _N_TINY)  # initial estimate (s_zero in the repo)
    return (x, s, _MAX_ITR)


MENAGERIE_ENTRIES = [
    ("TISTA", "build_tista", "example_input_tista", 2019, MENAGERIE_ZOO),
]
