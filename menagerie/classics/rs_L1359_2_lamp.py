# FAITHFUL PORT of mborgerding/onsager_deep_learning @ master (original framework: TensorFlow 1.x / numpy)
# https://github.com/mborgerding/onsager_deep_learning/blob/master/LAMP.py
# https://github.com/mborgerding/onsager_deep_learning/blob/master/tools/networks.py (build_LAMP)
# https://github.com/mborgerding/onsager_deep_learning/blob/master/tools/shrinkage.py (shrink_bgest)
#
# Borgerding & Schniter, "AMP-Inspired Deep Networks for Sparse Linear Inverse Problems"
# (LAMP / Learned AMP), NeurIPS 2017. The original repo is Python 2 + TensorFlow 1.x
# session/graph-mode code (tf.Variable / tf.placeholder / `except X,e:` syntax) and cannot
# run in a modern base torch env, so this is a faithful architecture transcription, not a
# vendor-as-is: every step of build_LAMP()'s unrolled recurrence (the linear estimate B_@y,
# the Onsager-corrected residual update vt_, the re-estimated residual variance rvar_, the
# tied "rhat = xhat + B_@vt" linear step, and the theta-parameterized shrink_bgest nonlinearity)
# is transcribed 1:1 from tools/networks.py::build_LAMP and tools/shrinkage.py::shrink_bgest.
# The default driver config in LAMP.py is used for defaults: shrink='bg', untied=False, T=6
# (kept at T=3 here purely to shrink cost; untied stays False, matching the driver script, so
# the "Bt_" per-layer linear-stage weight is NOT introduced -- see the `if untied:` branch in
# the original build_LAMP, which is skipped here exactly as the driver leaves it).
#
# A ("the sensing/measurement matrix") is a fixed (non-trainable) buffer, matching how the
# original code treats prob.A as constant problem data, not a learned parameter. B_ (the
# LMMSE-like reconstruction operator, initialized as A^T / (1.01*||A||_2^2) exactly as in
# build_LAMP) and each layer's theta (Bernoulli-Gaussian shrinkage parameters, initialized to
# theta_init = (1, log(1/pnz - 1)) from get_shrinkage_function('bg') with pnz=.1, matching
# LAMP.py's problems.bernoulli_gaussian_trial(pnz=.1) call) ARE learned nn.Parameters, matching
# the original code's tf.Variable usage.

import math

import torch
import torch.nn as nn


def shrink_bgest(r: torch.Tensor, rvar: torch.Tensor, theta: torch.Tensor):
    """Bernoulli-Gaussian MMSE shrinkage, transcribed from tools/shrinkage.py::shrink_bgest.

    theta[..., 0] = xvar1 (variance of the nonzero entries of x)
    theta[..., 1] = loglam = log(1/lambda - 1), lambda = P(x_i != 0)
    """
    xvar1 = theta[..., 0].abs()
    loglam = theta[..., 1]
    beta = 1 / (1 + rvar / xvar1)
    r2scale = r * r * beta / rvar
    rho = torch.exp(loglam - 0.5 * r2scale) * torch.sqrt(1 + xvar1 / rvar)
    rho1 = rho + 1
    xhat = beta * r / rho1
    dxdr = beta * ((1 + rho * (1 + r2scale)) / torch.square(rho1))
    dxdr = dxdr.mean(dim=0)  # per-column average derivative, as in the original
    return xhat, dxdr


class LAMPLayer(nn.Module):
    """Holds the per-layer learnable shrinkage parameters theta_t (tied linear operator B_
    is shared across layers here, matching untied=False in the original build_LAMP)."""

    def __init__(self, theta_init: torch.Tensor):
        super().__init__()
        self.theta = nn.Parameter(theta_init.clone())


class LAMP(nn.Module):
    """Learned Approximate Message Passing network (Borgerding & Schniter 2017),
    faithfully ported from tools/networks.py::build_LAMP(prob, T, shrink='bg', untied=False).

    Solves y = A @ x + noise for sparse x via T unrolled AMP iterations with a learned
    tied linear stage (B_) and learned Bernoulli-Gaussian shrinkage nonlinearity per layer.
    """

    def __init__(self, M: int, N: int, T: int = 3, pnz: float = 0.1):
        super().__init__()
        self.M = M
        self.N = N
        self.T = T

        A = torch.randn(M, N) * (1.0 / math.sqrt(M))
        self.register_buffer("A", A)

        B0 = A.T / (1.01 * torch.linalg.matrix_norm(A, ord=2) ** 2)
        self.B_ = nn.Parameter(B0.clone())

        # theta_init for shrink='bg': (xvar1=1, loglam=log(1/pnz - 1)), per
        # get_shrinkage_function('bg') in tools/shrinkage.py
        theta_init = torch.tensor([1.0, math.log(1.0 / pnz - 1.0)], dtype=torch.float32)
        self.layers = nn.ModuleList([LAMPLayer(theta_init) for _ in range(T)])

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        # y: (M, L) -- L is the number of columns/signals in the batch (as in the original,
        # which stacks signals column-wise rather than row-wise)
        By_ = self.B_ @ y

        OneOverM = 1.0 / self.M
        NOverM = self.N / self.M
        rvar_ = torch.sum(torch.square(y), dim=0) * OneOverM

        xhat_, dxdr_ = shrink_bgest(By_, rvar_, self.layers[0].theta)

        vt_ = y
        for t in range(1, self.T):
            if dxdr_.dim() == 2:
                dxdr_ = dxdr_.mean(dim=0)
            bt_ = dxdr_ * NOverM
            vt_ = y - self.A @ xhat_ + bt_ * vt_
            rvar_ = torch.sum(torch.square(vt_), dim=0) * OneOverM

            rhat_ = xhat_ + self.B_ @ vt_  # tied B_ (untied=False)
            xhat_, dxdr_ = shrink_bgest(rhat_, rvar_, self.layers[t].theta)

        return xhat_


def build_lamp():
    return LAMP(M=8, N=16, T=3, pnz=0.1)


def example_input_lamp():
    return torch.randn(8, 4)


MENAGERIE_ZOO = "ported-pytorch"

MENAGERIE_ENTRIES = [
    ("LAMP (Learned AMP)", "build_lamp", "example_input_lamp", 2017, "ported-pytorch"),
]
