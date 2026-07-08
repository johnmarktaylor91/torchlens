# FAITHFUL PORT of STARainZ/CG-OAMP-NET @ main (original framework: TensorFlow
# 1.x graph-mode, tools/networks.py:build_CG_OAMP, version=1 branch ==
# "CG_OAMP_NET" in the paper/repo naming)
#
# CG-OAMP-Net (a Conjugate-Gradient variant of the learned Orthogonal
# Approximate Message Passing network for MIMO detection) is a deep-unfolded
# iterative-algorithm architecture: T learned layers, each performing (1) a
# bounded conjugate-gradient (CG) linear solve of
# XI @ u = p_noise  where  XI = H H^T + (sigma2/2) / v_sqr * I
# to approximate the OAMP linear-estimation step without an explicit matrix
# inverse, then (2) an OAMP-style divergence-free nonlinear estimator (`nle`,
# a Gaussian-posterior-mean QAM demapper) with per-layer learnable scalars
# (theta, gamma, beta, and -- in the "CG_OAMP_NET"/version==1 branch used here
# -- an additional learned affine correction phi*(x_hat - xi*r)).
#
# The real code (tools/networks.py:nle, tools/networks.py:CG,
# tools/networks.py:build_CG_OAMP) is raw TensorFlow-1.x graph-construction
# code (tf.Variable/tf.placeholder/tf.while_loop/tf.Session), not an nn.Module
# and not runnable/vendorable as-is in a torch-only env. It is TRANSCRIBED
# faithfully below: every tensor operation, learnable scalar, and the
# iteration structure (T outer unfolded layers, each with a CG solve bounded
# by `icg` iterations) matches the original math one-for-one. The TF
# `tf.while_loop`'s early-stop condition (`r_norm > rth`) is preserved as a
# per-sample-batch python-level early break -- since this is a fixed-shape,
# eager single-call trace (not a training loop), a bounded early-exit while
# loop is the direct semantic equivalent of `maximum_iterations=I` combined
# with the early-stop predicate.
#
# Config defaults transcribed from main.py's `SysIn`/`TrainSetting` classes:
#   mu=2 (QPSK), Mr=Nt=8 (8x8 MIMO), T=4 unfolded layers, icg=50 (CG cap),
#   version=1 (CG_OAMP_NET, 5 learnable scalars/layer: theta/gamma/beta/phi/xi).
#
# Only base-lib deps used: torch, torch.nn, math.

import math

import torch
import torch.nn as nn

MENAGERIE_ZOO = "ported-pytorch"

_MU = 2  # QPSK: {-1, +1} / sqrt(2) constellation (source SysIn.mu default)
_SQ2 = math.sqrt(2)
_EPSILON = 1e-10
_RTH = 1e-4  # CG residual-norm early-stop threshold (source: rth)
_PTH = 1e-100  # nle probability floor (source: pth)


def _nle_qpsk(mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    """Port of tools/networks.py:nle for mu==2 (QPSK). Gaussian-posterior-mean
    soft demapper over the {-1/sqrt2, +1/sqrt2} constellation."""
    p0 = torch.clamp(torch.exp(-torch.square(-1 / _SQ2 - mean) / (2 * var)), min=_PTH)
    p1 = torch.clamp(torch.exp(-torch.square(1 / _SQ2 - mean) / (2 * var)), min=_PTH)
    return (p1 - p0) / (p1 + p0) / _SQ2


def _cg_step(u, p, residual, r_norm, xi_mat):
    """Port of tools/networks.py:CG -- one conjugate-gradient iteration solving
    xi_mat @ u = p_noise (real-valued, batched)."""
    xi_p = torch.matmul(xi_mat, p)  # bs*2M*1
    a = r_norm / torch.matmul(p.transpose(-2, -1), xi_p)
    u = u + a * p
    residual = residual + (-a * xi_p)
    r_norm_last = r_norm
    r_norm = torch.square(torch.linalg.norm(residual, dim=(-2, -1), keepdim=False)).reshape(
        residual.shape[0], 1, 1
    )
    b = r_norm / r_norm_last
    p = residual + b * p
    return u, p, residual, r_norm


class CGOAMPNet(nn.Module):
    """Port of tools/networks.py:build_CG_OAMP (version=1 / "CG_OAMP_NET").

    Forward signature matches the real algorithm's per-call tensors: H (the
    real-valued 2M x 2N MIMO channel matrix), y (received signal, 2M x 1),
    sigma2 (per-sample noise variance scalar), and eigvalue (eigenvalues of
    H H^T, M x 1 -- used by the OAMP orthogonalization "estimate" term). x
    (the true transmit signal) is NOT a forward input in the real algorithm
    at inference/test time (`sys.sess, sys.x_hat_net = networks.build_CG_OAMP`
    only feeds x_ during TRAINING, to compute the loss; the detector itself
    -- x_hat -- never reads x_). This module implements the same detector
    (test-time) computation.
    """

    def __init__(self, T: int = 4, icg: int = 8):
        super().__init__()
        self.T = T
        self.icg = icg
        # version==1 ("CG_OAMP_NET"): 5 learnable scalars per unfolded layer
        self.theta = nn.ParameterList([nn.Parameter(torch.tensor(1.0)) for _ in range(T)])
        self.gamma = nn.ParameterList([nn.Parameter(torch.tensor(1.0)) for _ in range(T)])
        self.beta = nn.ParameterList([nn.Parameter(torch.tensor(0.5)) for _ in range(T)])
        self.phi = nn.ParameterList([nn.Parameter(torch.tensor(1.0)) for _ in range(T)])
        self.xi = nn.ParameterList([nn.Parameter(torch.tensor(0.0)) for _ in range(T)])

    def forward(
        self, H: torch.Tensor, y: torch.Tensor, sigma2: torch.Tensor, eigvalue: torch.Tensor
    ):
        bs, two_m, two_n = H.shape
        Mr = two_m // 2

        HT = H.transpose(-2, -1)
        HHT = torch.matmul(H, HT)
        # 1 / trace(H^T H), reshaped to bs*1*1 (source: OneOver_trHTH)
        trHTH = torch.diagonal(torch.matmul(HT, H), dim1=-2, dim2=-1).sum(-1)
        one_over_trHTH = (1.0 / trHTH).reshape(bs, 1, 1)
        eye = torch.eye(two_m, dtype=H.dtype, device=H.device).unsqueeze(0).expand(bs, -1, -1)
        sigma2_I = sigma2 / 2 * eye

        v_sqr_last = torch.zeros((), dtype=H.dtype, device=H.device)
        x_hat = torch.zeros_like(y)

        for t in range(self.T):
            theta_ = self.theta[t]
            gamma_ = self.gamma[t]
            beta_ = self.beta[t]
            phi_ = self.phi[t]
            xi_ = self.xi[t]

            p_noise = y - torch.matmul(H, x_hat)  # bs*2M*1
            v_sqr = (
                torch.square(torch.linalg.norm(p_noise, dim=(-2, -1))).reshape(bs, 1, 1)
                - Mr * sigma2
            ) * one_over_trHTH
            v_sqr = beta_ * v_sqr + (1 - beta_) * v_sqr_last
            v_sqr = torch.clamp(v_sqr, min=_EPSILON)
            v_sqr_last = v_sqr

            # CG solve: XI @ u = p_noise
            xi_mat = HHT + sigma2_I / v_sqr
            u = torch.zeros_like(y)
            residual = p_noise.clone()
            p = residual.clone()
            r_norm = torch.square(torch.linalg.norm(residual, dim=(-2, -1))).reshape(bs, 1, 1)
            for _ in range(self.icg):
                if bool(torch.all(r_norm <= _RTH)):
                    break
                u, p, residual, r_norm = _cg_step(u, p, residual, r_norm, xi_mat)

            estimate = (
                torch.mean(eigvalue / (eigvalue + sigma2 / 2 / v_sqr), dim=1) * 2 * Mr
            )  # bs*1
            nor_coef = 2 * (two_n // 2) / estimate.reshape(bs, 1, 1)
            r = x_hat + gamma_ * nor_coef * torch.matmul(HT, u)
            tau_sqr = v_sqr * ((theta_**2) * nor_coef - 2 * theta_ + 1)
            tau_sqr = torch.clamp(tau_sqr, min=_EPSILON)

            x_hat = _nle_qpsk(r, tau_sqr)
            x_hat = phi_ * (x_hat - xi_ * r)

        return x_hat


def build_cg_oamp_net():
    torch.manual_seed(0)
    model = CGOAMPNet(T=2, icg=4)  # source default: T=4, icg=50 (shrunk for a fast trace)
    model.eval()
    return model


def example_input_cg_oamp_net():
    torch.manual_seed(0)
    bs = 2
    Mr = Nt = 4  # source default: Mr=Nt=8 (shrunk for a fast trace)
    two_m, two_n = 2 * Mr, 2 * Nt
    H = torch.randn(bs, two_m, two_n) * 0.3
    y = torch.randn(bs, two_m, 1)
    sigma2 = torch.full((bs, 1, 1), 0.1)
    eigvalue = torch.rand(bs, Mr, 1) + 0.5
    return (H, y, sigma2, eigvalue)


MENAGERIE_ENTRIES = [
    ("CG-OAMP-Net", "build_cg_oamp_net", "example_input_cg_oamp_net", 2020, MENAGERIE_ZOO),
]
