"""Misc faithful cores: coRNN, Decomposable Attention NLI, Deep Iterative Surface-Normal Est.

coRNN -- Rusch & Mishra, ICLR 2021, "Coupled Oscillatory Recurrent Neural Network (coRNN):
An accurate and (gradient) stable architecture for learning long time dependencies",
arXiv:2010.00951. Source: github.com/tk-rusch/coRNN.
  DISTINCTIVE: the hidden state is a network of DAMPED, FORCED harmonic oscillators. A
  second-order ODE y'' = sigma(W y + W_z z + V x + b) - gamma*y - epsilon*y' is discretized
  into two coupled first-order states (y = position, z = velocity):
      z_{n} = z_{n-1} + dt*(sigma(W y_{n-1} + W_z z_{n-1} + V x_n + b) - gamma*y_{n-1} - eps*z_{n-1})
      y_{n} = y_{n-1} + dt*z_{n}
  This oscillator physics gives gradient stability for long sequences.

Decomposable Attention NLI -- Parikh et al., EMNLP 2016, "A Decomposable Attention Model for
Natural Language Inference", arXiv:1606.01933.
  DISTINCTIVE: NLI with NO RNN/conv -- a 3-step Attend / Compare / Aggregate scheme. (1) ATTEND:
  soft-align premise & hypothesis tokens via a dot-product attention of per-token MLP features.
  (2) COMPARE: concatenate each token with its aligned counterpart, pass through an MLP. (3)
  AGGREGATE: sum the compared vectors of each sentence, concatenate, MLP -> 3-way entailment.

Deep Iterative Surface Normal Estimation -- Lenssen et al., CVPR 2020, "Deep Iterative Surface
Normal Estimation", arXiv:1904.07172. Source: github.com/mrhebamib/dnormae (and variants).
  DISTINCTIVE: estimate a point's surface normal by ITERATIVELY re-WEIGHTED least squares -- a
  small net predicts per-neighbor weights, a (differentiable) weighted plane fit gives a normal,
  the residuals re-weight the neighbors, and this repeats for K iterations. The learned IRLS
  weighting of a plane fit is the primitive.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# coRNN
# ============================================================


class coRNNCell(nn.Module):
    """One coupled-oscillator update step (discretized 2nd-order ODE: position y, velocity z)."""

    def __init__(
        self, n_inp: int, n_hid: int, dt: float = 0.042, gamma: float = 2.7, epsilon: float = 4.7
    ) -> None:
        super().__init__()
        self.dt, self.gamma, self.epsilon = dt, gamma, epsilon
        self.i2h = nn.Linear(n_inp + n_hid + n_hid, n_hid)

    def forward(
        self, x: torch.Tensor, hy: torch.Tensor, hz: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pre = self.i2h(torch.cat((x, hy, hz), dim=1))
        # velocity update (damped, forced): z += dt*(tanh(...) - gamma*y - epsilon*z)
        hz = hz + self.dt * (torch.tanh(pre) - self.gamma * hy - self.epsilon * hz)
        # position update: y += dt*z
        hy = hy + self.dt * hz
        return hy, hz


class coRNN(nn.Module):
    """Coupled Oscillatory RNN: oscillator-cell over a sequence -> linear readout."""

    def __init__(self, n_inp: int = 1, n_hid: int = 32, n_out: int = 10) -> None:
        super().__init__()
        self.n_hid = n_hid
        self.cell = coRNNCell(n_inp, n_hid)
        self.readout = nn.Linear(n_hid, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, B, n_inp)
        t, b, _ = x.shape
        hy = x.new_zeros(b, self.n_hid)
        hz = x.new_zeros(b, self.n_hid)
        for step in range(t):
            hy, hz = self.cell(x[step], hy, hz)
        return self.readout(hy)


# ============================================================
# Decomposable Attention NLI
# ============================================================


class DecomposableAttentionNLI(nn.Module):
    """Parikh et al. Attend-Compare-Aggregate NLI (no RNN; pure feed-forward + attention)."""

    def __init__(
        self, vocab: int = 500, emb_dim: int = 32, hidden: int = 32, n_classes: int = 3
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, emb_dim)
        self.attend = nn.Sequential(
            nn.Linear(emb_dim, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, hidden)
        )
        self.compare = nn.Sequential(
            nn.Linear(emb_dim * 2, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, hidden)
        )
        self.aggregate = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, n_classes)
        )

    def forward(self, premise: torch.Tensor, hypothesis: torch.Tensor) -> torch.Tensor:
        a = self.embed(premise)  # (B, La, E)
        b = self.embed(hypothesis)  # (B, Lb, E)
        # ATTEND: dot-product attention between attend-MLP features
        fa, fb = self.attend(a), self.attend(b)
        e = torch.bmm(fa, fb.transpose(1, 2))  # (B, La, Lb)
        beta = torch.bmm(F.softmax(e, dim=2), b)  # aligned hypothesis for each premise token
        alpha = torch.bmm(F.softmax(e, dim=1).transpose(1, 2), a)  # aligned premise per hyp token
        # COMPARE
        v_a = self.compare(torch.cat([a, beta], dim=2)).sum(1)  # aggregate premise
        v_b = self.compare(torch.cat([b, alpha], dim=2)).sum(1)  # aggregate hypothesis
        # AGGREGATE
        return self.aggregate(torch.cat([v_a, v_b], dim=1))


class _DecompAttnWrapper(nn.Module):
    """Forwardable from a single (1, 2L) int tensor: first L = premise, last L = hypothesis."""

    def __init__(self) -> None:
        super().__init__()
        self.model = DecomposableAttentionNLI()

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        half = tokens.shape[1] // 2
        return self.model(tokens[:, :half], tokens[:, half:])


# ============================================================
# Deep Iterative Surface Normal Estimation (learned IRLS plane fit)
# ============================================================


class DeepIterativeSurfaceNormal(nn.Module):
    """Iteratively re-weighted plane fit for surface normals from a local point neighborhood.

    Input: a point patch (B, 3, N) of N neighbor points. A small per-point net predicts an
    initial weight; K times: do a differentiable weighted plane fit (weighted covariance ->
    smallest eigenvector = normal), recompute residuals (point-to-plane distance), and feed
    them back to a weight-update net (the learned IRLS step). Returns the final normal (B, 3).
    """

    def __init__(self, num_iterations: int = 5, hidden: int = 32) -> None:
        super().__init__()
        self.num_iterations = num_iterations
        self.init_w = nn.Sequential(
            nn.Conv1d(3, hidden, 1), nn.ReLU(inplace=True), nn.Conv1d(hidden, 1, 1)
        )
        # weight-update net consumes [point(3), residual(1)] per neighbor
        self.upd_w = nn.Sequential(
            nn.Conv1d(4, hidden, 1), nn.ReLU(inplace=True), nn.Conv1d(hidden, 1, 1)
        )

    def _weighted_normal(self, pts: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # pts: (B, 3, N), w: (B, 1, N) -> normal (B, 3) via weighted-covariance smallest eigvec
        w = torch.softmax(w, dim=2)
        mean = (pts * w).sum(2, keepdim=True)
        centered = pts - mean
        wc = centered * w
        cov = torch.bmm(wc, centered.transpose(1, 2))  # (B, 3, 3)
        cov = cov + 1e-4 * torch.eye(3, device=pts.device)[None]
        eigvals, eigvecs = torch.linalg.eigh(cov)
        normal = eigvecs[:, :, 0]  # smallest-eigenvalue eigenvector
        return normal, mean.squeeze(2)

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        w = self.init_w(pts)  # (B, 1, N) initial weights
        normal, mean = self._weighted_normal(pts, w)
        for _ in range(self.num_iterations):
            # point-to-plane residual = |(p - mean) . n|
            resid = ((pts - mean[:, :, None]) * normal[:, :, None]).sum(1, keepdim=True).abs()
            w = self.upd_w(torch.cat([pts, resid], dim=1))  # learned re-weighting (IRLS step)
            normal, mean = self._weighted_normal(pts, w)
        return normal


def build_corNN() -> nn.Module:
    """Coupled Oscillatory RNN (coRNN): damped-forced-oscillator recurrent cell."""
    return coRNN().eval()


def build_decomposable_attention_nli() -> nn.Module:
    """Parikh et al. Decomposable Attention NLI (attend-compare-aggregate, no RNN)."""
    return _DecompAttnWrapper().eval()


def build_deep_iterative_surface_normal_estimation() -> nn.Module:
    """Deep iterative (learned IRLS) surface-normal estimation from a point neighborhood."""
    return DeepIterativeSurfaceNormal(num_iterations=5).eval()


def example_input_seq() -> torch.Tensor:
    """Sequence (40, 1, 1) -- (T, B, n_inp) for coRNN."""
    return torch.randn(40, 1, 1)


def example_input_nli() -> torch.Tensor:
    """Token ids (1, 24): first 12 = premise, last 12 = hypothesis."""
    return torch.randint(0, 500, (1, 24))


def example_input_points() -> torch.Tensor:
    """Local point neighborhood (1, 3, 64) for surface-normal estimation."""
    return torch.randn(1, 3, 64)


MENAGERIE_ENTRIES = [
    (
        "coRNN (Coupled Oscillatory RNN, damped-forced 2nd-order cell)",
        "build_corNN",
        "example_input_seq",
        "2020",
        "DC",
    ),
    (
        "Decomposable Attention NLI (attend-compare-aggregate)",
        "build_decomposable_attention_nli",
        "example_input_nli",
        "2016",
        "DC",
    ),
    (
        "Deep Iterative Surface Normal Estimation (learned IRLS plane fit)",
        "build_deep_iterative_surface_normal_estimation",
        "example_input_points",
        "2020",
        "DC",
    ),
]
