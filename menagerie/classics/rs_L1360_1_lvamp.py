# FAITHFUL PORT of https://github.com/mborgerding/onsager_deep_learning @ 0ba1cd5 (original
# framework: TensorFlow 1.x)
#
# Ported function: tools/networks.py `build_LVAMP` (M. Borgerding, P. Schniter, S. Rangan,
# "AMP-Inspired Deep Networks for Sparse Linear Inverse Problems", IEEE Trans. Signal
# Processing 2017 / arXiv:1612.01183). LVAMP unrolls Learned Vector Approximate Message
# Passing with the SVD parameterization: given a fixed random sensing matrix A (M x N) and
# noisy linear measurements y = A x + noise, the eigendecomposition of A A^T (equivalently
# the SVD A = U S V^T, computed once at construction) lets each unrolled layer alternate a
# closed-form LMMSE ("linear") estimation step in the singular-value domain -- with an
# Onsager correction carried from the previous layer's residual variance -- and a per-layer
# learned Bernoulli-Gaussian ("bg") MMSE shrinkage nonlinearity (learned scalar `theta`).
# The nonlinear-step's Onsager gain `zeta_j = 1/(1 - mean(dxhat/dr))` is likewise recomputed
# every layer via the closed-form derivative of the BG-MMSE denoiser, exactly as
# `tools/shrinkage.py shrink_bgest` computes it analytically (it does NOT use TF's
# `tf.gradients`/`auto_gradients` path -- that path is only used by the `expo`/`spline`
# shrink variants, not `bg`).
#
# The TF1 code (`tf.Variable`/`tf.InteractiveSession`/`tf.matmul` static-graph API) cannot
# run in a modern base env (TF1 graph-mode session API was retired across TF2). No PyTorch
# port of LVAMP-SVD exists in the repo or elsewhere searched. Per the rung ladder this is a
# RUNG-3 FAITHFUL PORT: every op of `build_LVAMP` (SVD precompute, LMMSE scaling `scale_each`,
# per-column normalizer `zetai`, Onsager-adjusted `ri`/`tauri`, BG-MMSE shrink `shrink_bgest`,
# Onsager-corrected `rj`/`taurj` carried to the next layer) is transcribed faithfully into
# base-env torch. Batch convention: TF used `y_: [N|M, L]` (feature-major, L=batch); this
# port keeps the standard torch `[batch, dim]` convention and transposes the relevant
# matmuls/reductions accordingly (call-site verified: identical per-element algebra, only the
# batch axis is relocated, matching the same transpose convention already used in the
# GLISTA_cp sibling port in this same staging batch).

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

MENAGERIE_ZOO = "ported-pytorch"


def _shrink_bgest(r: torch.Tensor, rvar: torch.Tensor, theta: torch.Tensor):
    """Faithful port of tools/shrinkage.py `shrink_bgest` (Bernoulli-Gaussian MMSE estimator).

    r: [batch, dim], rvar: [batch, 1] (broadcasts over dim). TF's original layout was
    [dim, L=batch] with `tf.reduce_mean(dxdr, 0)` averaging OVER THE FEATURE AXIS to produce
    one scalar per batch-column [L]; with tensors transposed to batch-first here, that same
    per-batch-column average is `torch.mean(..., dim=1)` (dim=-1, the feature axis).
    theta: (xvar1, loglam) scalars (iid case).
    """
    xvar1 = torch.abs(theta[0])
    loglam = theta[1]  # log(1/lambda - 1)
    beta = 1.0 / (1.0 + rvar / xvar1)
    r2scale = r * r * beta / rvar
    rho = torch.exp(loglam - 0.5 * r2scale) * torch.sqrt(1 + xvar1 / rvar)
    rho1 = rho + 1
    xhat = beta * r / rho1
    dxdr_full = beta * ((1 + rho * (1 + r2scale)) / torch.square(rho1))
    dxdr = torch.mean(dxdr_full, dim=1)  # [batch]: per-column (per batch-sample) average
    return xhat, dxdr


class LVAMP(nn.Module):
    """Faithful port of `build_LVAMP(prob, T, shrink='bg')` (SVD-parameterized LVAMP).

    Constructor mirrors the original's problem setup (fixed random Gaussian sensing matrix
    A, computed once) plus T (# unrolled layers). `forward(y)` mirrors the original graph
    build: y -> T LMMSE+Onsager-correction+BG-shrinkage layers -> final sparse-code estimate.
    """

    def __init__(self, M: int, N: int, T: int, seed: int = 0):
        super().__init__()
        rng = np.random.RandomState(seed)
        A = rng.normal(size=(M, N), scale=1.0 / np.sqrt(M)).astype(np.float32)
        self._M, self._N, self._T = M, N, T

        AA = A @ A.T
        s2, U = np.linalg.eigh(AA)
        s2 = np.clip(s2, 1e-12, None)
        s = np.sqrt(s2)
        V = (A.T @ U) / s

        self.register_buffer("V", torch.from_numpy(V.astype(np.float32)))  # [N, M]
        self.register_buffer("Vt", torch.from_numpy(V.T.astype(np.float32).copy()))  # [M, N]
        self.register_buffer("Us", torch.from_numpy((U / s).astype(np.float32)))  # [M, M]
        self.register_buffer(
            "rS2", torch.from_numpy((1.0 / (s * s)).reshape(1, -1).astype(np.float32))
        )  # [1, M]

        self.logyvar = nn.Parameter(torch.zeros(()))
        # theta_init = (xvar1=1, loglam=log(1/pnz - 1)) matching get_shrinkage_function('bg')
        # default pnz=.1 used by build_LVAMP's caller (bernoulli_gaussian_trial default).
        pnz = 0.1
        theta_init = (1.0, float(np.log(1.0 / pnz - 1.0)))
        self.thetas = nn.ParameterList(
            [nn.Parameter(torch.tensor(theta_init, dtype=torch.float32)) for _ in range(T)]
        )

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """y: [batch, M] noisy linear measurements. Returns [batch, N] sparse-code estimate.

        Per-batch-element scalars (`taurj`/`tauri`/`zetai`/`zetaj`, one value per "column" in
        the original TF `[dim, L=batch]` layout) are kept as `[batch]` vectors here and
        broadcast against the `[batch, dim]` feature tensors -- exactly mirroring the
        original's `[1, L]`-vs-`[dim, L]` broadcasting, just batch-first.
        """
        batch = y.shape[0]
        N = self._N

        rj = torch.zeros(batch, N, dtype=y.dtype, device=y.device)
        taurj = torch.sum(y * y, dim=1) / N  # [batch], per original taurj_=reduce_sum(y*y,0)/N
        yvar = torch.exp(self.logyvar)
        ytilde = y @ self.Us  # [batch, M] = matmul(inv(S)U^T, y) with TF's (U/s).T convention
        xhat = torch.zeros(batch, N, dtype=y.dtype, device=y.device)

        for t in range(self._T):
            varRat = (yvar / taurj).unsqueeze(1)  # [batch, 1]
            scale_each = 1.0 / (1.0 + self.rS2 * varRat)  # [batch, M] (rS2: [1,M] broadcasts)
            zetai = N / torch.sum(scale_each, dim=1)  # [batch] (original: N/reduce_sum, NOT M)
            adjust = (scale_each * (ytilde - rj @ self.Vt.T)) * zetai.unsqueeze(1)  # [batch, M]
            ri = rj + adjust @ self.V.T  # bring back to x space: matmul(V, adjust) -> adjust @ V^T
            tauri = taurj * (zetai - 1)  # [batch]

            theta_t = self.thetas[t]
            xhat, dxdr = _shrink_bgest(ri, tauri.unsqueeze(1), theta_t)
            dxdr = torch.clamp(dxdr, min=0.5 / N)  # [batch]

            zetaj = 1.0 / (1.0 - dxdr)  # [batch]
            rj = (xhat - dxdr.unsqueeze(1) * ri) * zetaj.unsqueeze(1)  # [batch, N]
            taurj = tauri * (zetaj - 1)  # [batch]

        return xhat


# ---- tiny build/example (architecture unmodified from the real repo) ----


def build_lvamp():
    model = LVAMP(M=25, N=50, T=4, seed=0)
    model.eval()
    return model


def example_input_lvamp():
    """y: [batch, M] linear-measurement tensor (y = Ax + noise)."""
    batch = 4
    return torch.randn(batch, 25)


MENAGERIE_ENTRIES = [
    ("LVAMP", build_lvamp, example_input_lvamp, 2017, "ported-pytorch"),
]
