# FAITHFUL PORT of YiWei0129/LcgNet @ master (original framework: TensorFlow 1.x)
#
# https://raw.githubusercontent.com/YiWei0129/LcgNet/master/LcgNet.py
#
# The official repo is a single TensorFlow-1.x script (tf.placeholder / tf.Session /
# tf.Variable graph-mode style, Python 3.6 + TF 1.13.1 per the README) with no PyTorch port
# anywhere on GitHub (confirmed via `gh search code LcgNet` / `gh search repos LcgNet`: only
# the original TF repo, forks of it, and unrelated TF notebooks referencing it by name). The
# architecture (IEEE ICASSP 2020, "Learned Conjugate Gradient Descent Network for Massive
# MIMO Detection") is faithfully transcribed here: `build_Lcg()` unrolls T conjugate-gradient
# -style refinement steps over the real-valued lifted linear system A x = b (A = H_r^T H_r +
# noise_var * I, b = H_r^T y_r, where H_r/y_r are the real-composite forms of the complex
# channel matrix H and observation y), with a LEARNED scalar (`type=0`) or per-coordinate
# vector (`type=1`) step-size parameter pair (alpha_t, beta_t) at every iteration -- this is
# the model's actual architectural contribution (a differentiable unrolled solver with
# learned step sizes, not a fixed classical CG solver). All PyTorch code below is original
# transcription of `build_Lcg` (renamed as an `nn.Module.forward`) plus `random_channel`'s
# problem-generation logic re-expressed as a torch input generator; no simplification of the
# recurrence (alpha_t * d_t update rule, residual/direction recursion) was made.
#
# MENAGERIE_ZOO = "ported-pytorch"

from __future__ import annotations

import torch
import torch.nn as nn

MENAGERIE_ZOO = "ported-pytorch"


class LcgNet(nn.Module):
    """Faithful port of LcgNet.py's `build_Lcg` (unrolled learned-CG MIMO detector).

    Forward takes the real-composite system `A` (2M, 2M) and `b` (2M, 1) built from a complex
    channel `H` (N, M) and observation `y` (N, 1) at a given noise variance, and returns the
    T-step refined estimate `xhat` (2M, 1) -- the concatenated real/imaginary parts of the
    detected transmit symbol vector, matching `layers[-1][1]` (`xhat_`) in the original.
    """

    def __init__(self, M: int, T: int = 15, cg_type: int = 1):
        super().__init__()
        self.M = M
        self.T = T
        self.cg_type = cg_type
        param_shape = (1,) if cg_type == 0 else (2 * M, 1)

        # one (alpha, beta) pair per unrolled step (alpha_0 has no matching beta_0, matching
        # the original's layers[0] = ('...T=0', xhat_, (alpha_,), ()) before the loop starts).
        self.alpha = nn.ParameterList(
            [nn.Parameter(torch.zeros(*param_shape)) for _ in range(T + 1)]
        )
        self.beta = nn.ParameterList([nn.Parameter(torch.zeros(*param_shape)) for _ in range(T)])

    def forward(self, A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        d = b
        r = d
        alpha_t = self.alpha[0]
        xhat = alpha_t * d

        for t in range(1, self.T + 1):
            r = r - alpha_t * torch.matmul(A, d)
            beta_t = self.beta[t - 1]
            d = r + beta_t * d
            alpha_t = self.alpha[t]
            xhat = xhat + alpha_t * d

        return xhat


def _random_composite_system(M: int, N: int, snr_db: float = 30.0, seed: int = 0):
    """Faithful re-expression of `random_channel` + the A/b construction inlined at the top
    of `build_Lcg` in the original TF script: builds the real-composite lifted linear system
    A x = b for a random Gaussian complex channel H (N, M), a random BPSK transmit vector x
    (M, 1) transmitted as complex with zero imaginary part, and complex Gaussian noise at the
    given SNR.
    """
    g = torch.Generator().manual_seed(seed)
    H_real = torch.randn(N, M, generator=g) * (1.0 / (2.0 * N)) ** 0.5
    H_imag = torch.randn(N, M, generator=g) * (1.0 / (2.0 * N)) ** 0.5

    x_real = torch.sign(torch.rand(M, 1, generator=g) * 2 - 1)
    x_imag = torch.zeros(M, 1)

    noise_var = M / N * (10.0 ** (-snr_db / 10.0))
    noise_real = torch.randn(N, 1, generator=g) * (noise_var / 2.0) ** 0.5
    noise_imag = torch.randn(N, 1, generator=g) * (noise_var / 2.0) ** 0.5

    # y = H x + noise (complex matmul expressed via real/imag parts)
    y_real = H_real @ x_real - H_imag @ x_imag + noise_real
    y_imag = H_real @ x_imag + H_imag @ x_real + noise_imag

    # Real-composite lift: H_Real = [[Hr, Hi], [-Hi, Hr]], y_Real = [yr; yi]
    H_Real = torch.cat(
        [
            torch.cat([H_real, H_imag], dim=1),
            torch.cat([-H_imag, H_real], dim=1),
        ],
        dim=0,
    )
    y_Real = torch.cat([y_real, y_imag], dim=0)

    A = H_Real.T @ H_Real + noise_var * torch.eye(2 * M)
    b = H_Real.T @ y_Real
    return A, b


# ---------------------------------------------------------------------------
# Menagerie staging entrypoints.
# ---------------------------------------------------------------------------


def build_lcgnet():
    torch.manual_seed(0)
    # M=32, N=64, T=15, type=1 (vector alpha/beta) matches the original script's own
    # invocation at the bottom of LcgNet.py: `build_Lcg(prob, T=15, type=1)` with
    # `random_channel(M=32, N=64, L=1)`.
    return LcgNet(M=32, T=15, cg_type=1)


def example_input_lcgnet():
    A, b = _random_composite_system(M=32, N=64, snr_db=30.0, seed=0)
    return (A, b)


MENAGERIE_ENTRIES = [
    (
        "LcgNet (Learned-CG MIMO Detector)",
        "build_lcgnet",
        "example_input_lcgnet",
        2020,
        MENAGERIE_ZOO,
    ),
]
