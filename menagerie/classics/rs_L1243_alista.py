# FAITHFUL PORT of uclaopt/alista @ master (original framework: TensorFlow 1.x)
# https://github.com/uclaopt/alista/blob/master/models/ALISTA.py
# https://github.com/uclaopt/alista/blob/master/utils/tf.py
# https://github.com/uclaopt/alista/blob/master/models/LISTA_base.py
# ALISTA ("Analytic Weights Are As Good As Learned Weights in LISTA", ICLR 2019) is
# a learned sparse-coding recovery network: given a measurement matrix `A` and a
# fixed *analytic* weight matrix `W` (computed offline, independent of training
# data, via the coherence-minimization procedure in `matlabs/CalculateW.m` --
# treated here as a non-trainable buffer since the offline solve itself is outside
# the traced forward graph), ALISTA unrolls T ISTA-style iterations. Each iteration
# t applies a single trainable scalar step size `alpha_t` and threshold `theta_t`:
#     residual = y - A @ x_hat
#     z = x_hat + alpha_t * W^T @ residual
#     x_hat = shrink_ss(z, theta_t, percent_t)
# where `shrink_ss` is the "support-selection" soft-threshold: entries in the top
# `percent_t`% by magnitude (and above `theta_t`) pass through unchanged (their
# support is presumed correct and exempted from shrinkage); all other entries get
# ordinary soft-thresholding. `percent_t` grows linearly with layer index (`(t+1) *
# percent`, clipped at `max_percent`), matching the real `ALISTA.__init__`.
#
# The real repo is TensorFlow 1.x graph-mode (`tf.variable_scope`, `tf.get_variable`,
# `tf.contrib.distributions.percentile`) and depends on `tf.contrib`, which was
# removed even from modern TF2 -- it cannot run in this base torch env, so the
# `ALISTA.__init__`/`setup_layers`/`inference` control flow (from models/ALISTA.py)
# and the `shrink`, `shrink_free`, `shrink_ss` shrinkage functions (from
# utils/tf.py) are FAITHFULLY TRANSCRIBED into self-contained torch, preserving
# every mechanism: the per-layer trainable (alpha, theta) pair, the fixed A/W
# buffers, the linearly-growing top-percent support-selection schedule, and the
# exact shrink_ss formula (index_ = (|z|>theta) & (|z|>percentile_thresh);
# output = index_*z + shrink_free((1-index_)*z, theta)). `tf.contrib.distributions
# .percentile(..., axis=0)` (a per-row/per-feature percentile across the batch
# dimension) is reproduced with `torch.quantile(..., dim=0)`, which computes the
# identical order statistic. No layer, iteration mechanism, or thresholding rule
# was added, removed, or altered relative to the real TF graph.
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

MENAGERIE_ZOO = "ported-pytorch"


# --- utils/tf.py shrinkage functions (ported) ---
def shrink_free(input_: Tensor, theta_: Tensor) -> Tensor:
    """Soft-shrinkage without the constraint that thresholds must be >= 0."""
    return torch.sign(input_) * torch.clamp(torch.abs(input_) - theta_, min=0.0)


def shrink_ss(inputs_: Tensor, theta_: Tensor, q: float) -> Tensor:
    """Special shrink that does not apply soft shrinkage to entries of top q% magnitudes.

    Entries that are greater than `theta_` AND in the top `q`% simultaneously are
    selected into the support and passed through unshrunk; everything else is soft-
    thresholded at `theta_`.
    """
    abs_ = torch.abs(inputs_)
    # tf.contrib.distributions.percentile(abs_, 100.0 - q, axis=0, keep_dims=True)
    # is the (100-q)-th percentile taken along the batch axis, per-row.
    q_frac = torch.clamp(torch.tensor((100.0 - q) / 100.0, dtype=inputs_.dtype), 0.0, 1.0)
    thres_ = torch.quantile(abs_, q_frac, dim=0, keepdim=True)

    index_ = (abs_ > theta_) & (abs_ > thres_)
    index_ = index_.to(inputs_.dtype).detach()  # tf.stop_gradient(index_)
    cindex_ = 1.0 - index_

    return index_ * inputs_ + shrink_free(cindex_ * inputs_, theta_)


# --- models/ALISTA.py (ported architecture) ---
class ALISTA(nn.Module):
    """Torch port of the ALISTA deep unrolled sparse-coding network."""

    def __init__(
        self,
        A: np.ndarray,
        T: int,
        lam: float,
        W: np.ndarray,
        percent: float,
        max_percent: float,
        coord: bool,
    ) -> None:
        """
        Parameters
        ----------
        A
            Measurement matrix, shape (M, N).
        T
            Number of unrolled layers (depth).
        lam
            Initial value of shrinkage thresholds.
        W
            Analytic weight matrix, shape (M, N) (precomputed offline; independent
            of training).
        percent
            Support-selection percent per layer (grows linearly with layer index).
        max_percent
            Clip ceiling for the per-layer support-selection percent.
        coord
            If True, use a per-coordinate (N,) threshold instead of a scalar.
        """
        super().__init__()
        A = A.astype(np.float32)
        W = W.astype(np.float32)
        self.T = T
        self.M, self.N = A.shape

        scale = 1.001 * np.linalg.norm(A, ord=2) ** 2
        theta_init = (
            (lam / scale).astype(np.float32)
            if isinstance(lam, np.ndarray)
            else np.float32(lam / scale)
        )
        if coord:
            theta_init = np.ones((self.N, 1), dtype=np.float32) * theta_init

        ps = [(t + 1) * percent for t in range(T)]
        ps = list(np.clip(ps, 0.0, max_percent))
        self.percents = ps

        # Fixed (non-trainable) buffers: measurement matrix and analytic weights.
        self.register_buffer("A", torch.from_numpy(A))
        self.register_buffer("W", torch.from_numpy(W))
        self.register_buffer("Wt", torch.from_numpy(W).t().contiguous())

        # Trainable per-layer step sizes and thresholds.
        self.alphas = nn.ParameterList([nn.Parameter(torch.tensor(1.0)) for _ in range(T)])
        if coord:
            self.thetas = nn.ParameterList(
                [nn.Parameter(torch.from_numpy(theta_init.copy())) for _ in range(T)]
            )
        else:
            self.thetas = nn.ParameterList(
                [nn.Parameter(torch.tensor(float(theta_init))) for _ in range(T)]
            )

    def forward(self, y: Tensor) -> Tensor:
        """Run T unrolled ALISTA iterations.

        Parameters
        ----------
        y
            Measurements, shape (M, batch).

        Returns
        -------
        Tensor
            Final sparse-code estimate, shape (N, batch).
        """
        batch_size = y.shape[-1]
        xh_ = torch.zeros(self.N, batch_size, dtype=y.dtype, device=y.device)

        for t in range(self.T):
            alpha_ = self.alphas[t]
            theta_ = self.thetas[t]
            percent = self.percents[t]

            res_ = y - torch.matmul(self.A, xh_)
            zh_ = xh_ + alpha_ * torch.matmul(self.Wt, res_)
            xh_ = shrink_ss(zh_, theta_, percent)

        return xh_


# --- staging entry points ---
def build_alista():
    rng = np.random.default_rng(0)
    M, N = 8, 16
    A = rng.standard_normal((M, N)).astype(np.float32)
    A /= (
        np.linalg.norm(A, axis=0, keepdims=True) + 1e-8
    )  # column-normalize, as in the real data setup
    # Real repo computes W offline via coherence minimization (CalculateW.m); a
    # random matrix of the same shape stands in for that precomputed asset without
    # altering the unrolled-network architecture that consumes it.
    W = rng.standard_normal((M, N)).astype(np.float32)
    model = ALISTA(A=A, T=4, lam=0.4, W=W, percent=0.1, max_percent=0.5, coord=False)
    model.eval()
    return model


def example_input_alista():
    torch.manual_seed(0)
    M, batch = 8, 4
    return torch.randn(M, batch)


MENAGERIE_ENTRIES = [
    (
        "ALISTA (Analytic-weights LISTA)",
        "build_alista",
        "example_input_alista",
        2019,
        MENAGERIE_ZOO,
    ),
]
