# FAITHFUL PORT of mehrdadkhani/MMNet @ master (original framework: TensorFlow 1.x --
# `tf.Variable`/`tf.trace`/`tf.matrix_inverse`/`keep_dims=` throughout `learning_based/`;
# this TF1-era graph-mode API is not installable in the current base env, so the
# architecture is transcribed faithfully into self-contained torch below, layer for
# layer, rather than vendored).
#
# MMNet (Khani, Alizadeh, Hoydis & Fleming, "Adaptive Neural Signal Detection for
# Massive MIMO", arXiv:1906.04610 / IEEE Trans. Signal Processing 2020) is a deep
# unrolled iterative detector for massive-MIMO signal detection. Each of `L` unrolled
# layers (`learning_based/layer.py`'s `layer()`, driving `linear.MMNet` +
# `denoiser.MMNet` from `linear.py`/`denoiser.py`) performs:
#   1. a LEARNED LINEAR step (`linear.MMNet`): builds a real-valued NTxNR "pseudo-
#      inverse" matrix W per layer from a small complex-structured parameter
#      (Wr, Wi realified into the standard [[Wr,-Wi],[Wi,Wr]] block form used
#      throughout this codebase to represent complex linear maps as real 2x2-block
#      matrices), producing an intermediate estimate `zt = shatt + W @ rt` from the
#      current residual `rt = y - H @ shatt`;
#   2. an ONSAGER-CORRECTED GAUSSIAN DENOISER step (`denoiser.MMNet` -> `gaussian`):
#      computes a per-layer noise-variance proxy `tau2_t` from the Onsager formula
#      (trace of `C_t @ C_t^T` where `C_t = I - W@H`, weighted by an empirical
#      residual-power estimate `v2_t` and additive per-symbol noise term), scaled by
#      a small learned per-symbol correction (`tf.random_normal([1,NT,1],1.,0.1)`
#      -variance init), then projects `zt` onto the discrete constellation via a
#      softmax over squared per-symbol Gaussian log-likelihoods (`gaussian()`), giving
#      the next iteration's soft symbol estimate `shatt`.
# The full detector stacks `L` such layers (independent parameters per layer, i.e.
# `shared_parameters=False` in this repo's convention), starting from `shatt_0=0`,
# `rt_0=y`, producing the final detected soft-symbol vector. Architecture (per-layer
# W-parameterization, Onsager tau2_t formula, Gaussian-mixture denoiser) is reproduced
# faithfully from the real TF1 code; only the framework substrate changes.

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "ported-pytorch"


def batch_matvec_mul(A, b, transpose_a=False):
    """Matches utils.py's batch_matvec_mul: A (B,N,K) @ b (B,K) -> (B,N)."""
    if transpose_a:
        A = A.transpose(-2, -1)
    return torch.bmm(A, b.unsqueeze(-1)).squeeze(-1)


class MMNetLinear(nn.Module):
    """linear.py: linear.MMNet -- learned per-layer linear "pseudo-inverse" step.

    Wr, Wi are learned (NT/2 x NR/2) blocks realifying a complex linear map into the
    standard real 2x2-block form W = [[Wr,-Wi],[Wi,Wr]] used throughout this codebase
    to represent complex-valued linear operators over the realified (I/Q-stacked)
    channel representation.
    """

    def __init__(self, nt, nr):
        super().__init__()
        self.nt = nt
        self.nr = nr
        self.Wr = nn.Parameter(torch.randn(1, nt // 2, nr // 2) * 0.01)
        self.Wi = nn.Parameter(torch.randn(1, nt // 2, nr // 2) * 0.01)

    def forward(self, shatt, rt, H, batch_size):
        top = torch.cat([self.Wr, -self.Wi], dim=2)
        bot = torch.cat([self.Wi, self.Wr], dim=2)
        W = torch.cat([top, bot], dim=1)  # (1, NT, NR)
        W = W.expand(batch_size, -1, -1)
        zt = shatt + batch_matvec_mul(W, rt)
        I_WH = torch.eye(self.nt, device=H.device).expand(batch_size, -1, -1) - torch.bmm(W, H)
        return zt, {"W": W, "I_WH": I_WH}


class MMNetDenoiser(nn.Module):
    """denoiser.py: denoiser.MMNet -- Onsager-corrected Gaussian-mixture denoiser."""

    def __init__(self, nt, nr, constellation):
        super().__init__()
        self.nt = nt
        self.nr = nr
        self.register_buffer("constellation", constellation.float())
        self.m = constellation.shape[0]
        # tf.random_normal([1, NT, 1], mean=1., stddev=0.1) -- a learned per-symbol
        # multiplicative correction dividing the Onsager tau2_t estimate.
        self.tau_correction = nn.Parameter(torch.randn(1, nt, 1) * 0.1 + 1.0)

    def gaussian(self, zt, tau2_t):
        # arg: (B, NT, M) squared distance of each symbol estimate to each
        # constellation point, scaled by -1/(2*tau2_t), softmax-weighted expectation.
        arg = zt.unsqueeze(-1) - self.constellation.view(1, 1, -1)
        arg = -(arg**2) / 2.0 / tau2_t
        shatt1 = F.softmax(arg, dim=-1)
        shatt1 = torch.matmul(shatt1, self.constellation.view(-1, 1)).squeeze(-1)
        return shatt1

    def forward(self, zt, rt, H, noise_sigma, W_t, batch_size):
        HTH = torch.bmm(H.transpose(-2, -1), H)
        trace_HTH = HTH.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True)  # (B,1)
        noise_sigma_col = noise_sigma.pow(2).unsqueeze(-1)  # (B,1)
        v2_t = (rt.pow(2).sum(dim=1, keepdim=True) - self.nr * noise_sigma_col / 2.0) / trace_HTH
        v2_t = torch.clamp(v2_t, min=1e-9).unsqueeze(-1)  # (B,1,1)

        C_t = torch.eye(self.nt, device=H.device).expand(batch_size, -1, -1) - torch.bmm(W_t, H)
        trace_CCt = (
            torch.bmm(C_t, C_t.transpose(-2, -1)).diagonal(dim1=-2, dim2=-1).sum(-1).view(-1, 1, 1)
        )
        trace_WWt = (
            torch.bmm(W_t, W_t.transpose(-2, -1)).diagonal(dim1=-2, dim2=-1).sum(-1).view(-1, 1, 1)
        )
        noise_sigma_sq = noise_sigma.pow(2).view(-1, 1, 1)

        tau2_t = (1.0 / self.nt) * trace_CCt * v2_t + noise_sigma_sq / (2.0 * self.nt) * trace_WWt
        tau2_t = tau2_t / self.tau_correction

        shatt1 = self.gaussian(zt, tau2_t)
        return shatt1


class MMNetLayer(nn.Module):
    """learning_based/layer.py's `layer()` function, one unrolled iteration."""

    def __init__(self, nt, nr, constellation):
        super().__init__()
        self.linear = MMNetLinear(nt, nr)
        self.denoiser = MMNetDenoiser(nt, nr, constellation)

    def forward(self, xhat, r, H, y, noise_sigma, batch_size):
        zt, linear_helper = self.linear(xhat, r, H, batch_size)
        new_xhat = self.denoiser(zt, r, H, noise_sigma, linear_helper["W"], batch_size)
        new_r = y - batch_matvec_mul(H, new_xhat)
        return new_xhat, new_r


class MMNet(nn.Module):
    """Full MMNet detector: L independently-parameterized unrolled MMNetLayer
    iterations (learning_based/detector.py's `detector.create_graph`), starting
    from xhat_0 = 0, r_0 = y."""

    def __init__(self, nt=8, nr=8, num_layers=3, m_pam=4):
        super().__init__()
        self.nt = nt
        self.nr = nr
        # PAM constellation matching this repo's real+imag realified symbol alphabet
        # (e.g. 4-PAM: {-3,-1,1,3}/sqrt(5) normalized average unit energy per rail).
        levels = torch.arange(-(m_pam - 1), m_pam, 2, dtype=torch.float32)
        constellation = levels / levels.pow(2).mean().sqrt()
        self.layers = nn.ModuleList([MMNetLayer(nt, nr, constellation) for _ in range(num_layers)])

    def forward(self, y, H, noise_sigma):
        """
        y: (B, NR) real-valued (I/Q-stacked) received signal.
        H: (B, NR, NT) real-valued (I/Q-realified) channel matrix.
        noise_sigma: (B,) per-sample noise std.
        """
        batch_size = y.shape[0]
        xhat = torch.zeros(batch_size, self.nt, device=y.device, dtype=y.dtype)
        r = y.clone()
        for lyr in self.layers:
            xhat, r = lyr(xhat, r, H, y, noise_sigma, batch_size)
        return xhat


# ---- staging entry points ----


def build_mmnet():
    """MMNet at tiny size for tracing (NT=NR=8 realified antennas i.e. 4 complex Tx/Rx,
    3 unrolled layers). Architecture is unmodified from the real repo."""
    torch.manual_seed(0)
    model = MMNet(nt=8, nr=8, num_layers=3, m_pam=4)
    model.eval()
    return model


def example_input_mmnet():
    """Matches MMNet.forward(y, H, noise_sigma): a batch of realified received
    signals, channel matrices, and per-sample noise sigmas for a tiny 8x8 MIMO
    scenario."""
    torch.manual_seed(0)
    b, nt, nr = 2, 8, 8
    H = torch.randn(b, nr, nt) * 0.3
    x = torch.randn(b, nt)
    noise_sigma = torch.full((b,), 0.1)
    y = torch.bmm(H, x.unsqueeze(-1)).squeeze(-1) + noise_sigma.view(-1, 1) * torch.randn(b, nr)
    return (y, H, noise_sigma)


MENAGERIE_ENTRIES = [
    ("MMNet", "build_mmnet", "example_input_mmnet", 2020, MENAGERIE_ZOO),
]
