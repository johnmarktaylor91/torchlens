# FAITHFUL REIMPLEMENTATION from arXiv:2006.01125 (no public code) -- A/B sonnet
"""BCJRNet receiver: "Neural Network-Aided BCJR Algorithm for Joint Symbol Detection
and Channel Decoding" (Tsai, Teng, Ou, Wu, 2020).

Distinctive mechanisms implemented faithfully:
1. A JOINT trellis over (RSC-encoder-state, current-input-bit) so a single BCJR
   recursion handles ISI-channel symbol detection and RSC/turbo decoding at once,
   instead of two separate cascaded blocks (Sec. III.A, Fig. 3(c), Table I(c)).
2. A dedicated 3-layer fully-connected neural network ("BCJRNet") that replaces the
   channel-model-based computation of the branch probability P(y_k | s', s) --
   the classic BCJR still runs its log-domain forward/backward recursion, but the
   per-symbol transition likelihood is now learned end-to-end from the received
   signal instead of derived from a known channel model (Sec. III.B, Fig. 4,
   Eq. 17-19).
3. The full iterative two-decoder turbo-style receiver (Fig. 2(b)/(c)) that
   exchanges extrinsic LLRs between the two BCJR decoders over M iterations
   (Table II: M = 6).
"""

from __future__ import annotations

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Trellis construction (paper Sec. II.B / III.A, Table I)
# ---------------------------------------------------------------------------

# RSC encoder next-state table, Table I(a). Generator matrix
# [1, (1+D^2)/(1+D+D^2)] -> 4 states {00,01,10,11} indexed 0..3.
# NEXT_STATE_TABLE[s_prev][u_k] = next RSC state.
_RSC_NEXT_STATE = ((0, 2), (2, 0), (3, 1), (1, 3))


def _build_joint_trellis_edges() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the 8-state joint (RSC-state, current-bit) trellis, Sec. III.A.

    The paper redefines the BCJR state as s_k := (s_{k-1}^RSC, u_k) so that the
    branch probability at time k can depend on both the trellis code structure
    and the channel-corrupted received signal simultaneously (Fig. 3(c),
    Table I(c)). With a 4-state RSC and a binary input, this yields 8 joint
    states and 2 x 8 = 16 valid (s', s) transitions -- matching the "2x|s|"
    output width of the BCJRNet dense layer in Fig. 4.
    """
    edge_from: list[int] = []
    edge_to: list[int] = []
    edge_u: list[int] = []
    for s_prime in range(8):
        rsc_prev, u_prev = divmod(s_prime, 2)
        rsc_new = _RSC_NEXT_STATE[rsc_prev][u_prev]
        for u_k in (0, 1):
            s_to = rsc_new * 2 + u_k
            edge_from.append(s_prime)
            edge_to.append(s_to)
            edge_u.append(u_k)
    return (
        torch.tensor(edge_from, dtype=torch.long),
        torch.tensor(edge_to, dtype=torch.long),
        torch.tensor(edge_u, dtype=torch.long),
    )


_NEG = -1.0e9  # stand-in for -inf that stays finite under logsumexp/subtraction.


class BCJRNetGammaNet(nn.Module):
    """The dedicated branch-probability network of Fig. 4.

    "The dedicated model has three fully-connected layers ... The nonlinear
    function used by the first two layers [is] ReLU ... The softmax activation
    is used at the output layer" (Sec. III.B). Input = received signal
    [y_k^s, y_k^p] (2-dim); output = P(y_k | s', s) for each of the 2x|s| = 16
    valid trellis transitions (Eq. 17-19).
    """

    def __init__(self, num_edges: int = 16, hidden: int = 100):
        super().__init__()
        self.dense1 = nn.Linear(2, hidden)
        self.dense2 = nn.Linear(hidden, hidden)
        self.dense3 = nn.Linear(hidden, num_edges)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        # y: (..., 2) -> log P(y | s', s) per edge, (..., num_edges).
        h = torch.relu(self.dense1(y))
        h = torch.relu(self.dense2(h))
        logits = self.dense3(h)
        return torch.log_softmax(logits, dim=-1)  # Eq. 19, in log domain.


class BCJRNetDecoder(nn.Module):
    """One NN-aided BCJR decoder over the joint 8-state trellis.

    Runs the classic log-domain forward (Eq. 7) / backward (Eq. 9) recursion
    and posterior-LLR combination (Eq. 2-3), but Gamma_k(s', s) is built from
    the BCJRNetGammaNet's learned branch probability plus the extrinsic-info
    term u_k * Le(u_k) / 2 from the companion decoder (Eq. 5/8/17).
    """

    def __init__(self) -> None:
        super().__init__()
        edge_from, edge_to, edge_u = _build_joint_trellis_edges()
        self.register_buffer("edge_from", edge_from)
        self.register_buffer("edge_to", edge_to)
        self.register_buffer("edge_u", edge_u)
        self.register_buffer("edge_bpsk", (2 * edge_u - 1).to(torch.float32))
        self.gamma_net = BCJRNetGammaNet(num_edges=edge_from.numel())
        self.num_states = 8

    def forward(self, y: torch.Tensor, le_prior: torch.Tensor) -> torch.Tensor:
        """y: (B, K, 2) received [systematic, parity] pair per symbol.
        le_prior: (B, K) extrinsic LLR fed in from the companion decoder.
        Returns: (B, K) a posteriori LLR L(u_k), Eq. 2.
        """
        batch, k_len, _ = y.shape
        device = y.device

        # Branch-probability term from the neural network (Eq. 17-19).
        log_p_y = self.gamma_net(y)  # (B, K, 16)
        # Extrinsic-info term u_k * Le(u_k) / 2 (Eq. 5/8), same for every
        # edge sharing that edge's transmitted bit.
        extrinsic = 0.5 * le_prior.unsqueeze(-1) * self.edge_bpsk  # (B, K, 16)
        log_gamma_edges = log_p_y + extrinsic  # (B, K, 16), log Gamma_k(s', s)

        # Scatter edge-space Gamma into a dense (B, K, S', S) trellis matrix.
        gamma_mat = torch.full(
            (batch, k_len, self.num_states, self.num_states), _NEG, device=device
        )
        gamma_mat[:, :, self.edge_from, self.edge_to] = log_gamma_edges

        # Forward recursion A_k(s), Eq. 4/7: A_0(0) = 0, else -inf.
        alpha = [torch.full((batch, self.num_states), _NEG, device=device)]
        alpha[0][:, 0] = 0.0
        for t in range(k_len):
            prev = alpha[-1].unsqueeze(-1)  # (B, S', 1)
            step = torch.logsumexp(prev + gamma_mat[:, t, :, :], dim=1)  # (B, S)
            alpha.append(step)

        # Backward recursion B_k(s), Eq. 6/9: B_K(0) = 0, else -inf.
        beta = [None] * (k_len + 1)
        beta[k_len] = torch.full((batch, self.num_states), _NEG, device=device)
        beta[k_len][:, 0] = 0.0
        for t in range(k_len - 1, -1, -1):
            nxt = beta[t + 1].unsqueeze(1)  # (B, 1, S)
            step = torch.logsumexp(nxt + gamma_mat[:, t, :, :], dim=2)  # (B, S')
            beta[t] = step

        # Posterior LLR, Eq. 2-3: combine alpha_{k}, Gamma_{k+1}, beta_{k+1}
        # over edges grouped by the transmitted bit u_k (S+ = 1, S- = 0).
        u_is_one = self.edge_u == 1
        llrs = []
        for t in range(k_len):
            a_from = alpha[t][:, self.edge_from]  # (B, 16)
            b_to = beta[t + 1][:, self.edge_to]  # (B, 16)
            score = a_from + log_gamma_edges[:, t, :] + b_to  # (B, 16)
            pos = torch.logsumexp(score[:, u_is_one], dim=-1)
            neg = torch.logsumexp(score[:, ~u_is_one], dim=-1)
            llrs.append(pos - neg)
        return torch.stack(llrs, dim=1)  # (B, K)


class BCJRNetReceiver(nn.Module):
    """Joint symbol-detection-and-decoding BCJRNet receiver, Fig. 1(b)/2(b)(c).

    Two NN-aided BCJR decoders exchange extrinsic LLRs through a fixed random
    interleaver over M BCJR iterations (Table II: "BCJR iterations: 6",
    "Interleaver: Random"), exactly mirroring a standard turbo receiver except
    that each constituent decoder's branch-probability computation is the
    joint-trellis BCJRNet of `BCJRNetDecoder` above.
    """

    def __init__(self, k_len: int = 8, num_iterations: int = 6):
        super().__init__()
        self.k_len = k_len
        self.num_iterations = num_iterations
        self.decoder1 = BCJRNetDecoder()
        self.decoder2 = BCJRNetDecoder()
        perm = torch.randperm(k_len)
        self.register_buffer("interleave_idx", perm)
        self.register_buffer("deinterleave_idx", torch.argsort(perm))

    def _interleave(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, self.interleave_idx]

    def _deinterleave(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, self.deinterleave_idx]

    def forward(self, y1: torch.Tensor, y2: torch.Tensor) -> torch.Tensor:
        """y1, y2: (B, K, 2) received signal for decoder 1 / decoder 2 resp.
        (decoder 1 sees [y^s, y^p1], decoder 2 sees the interleaved-systematic
        pair [y^{s'}, y^{p2}], per Sec. III.A.) Returns final LLR (B, K).
        """
        batch = y1.shape[0]
        le1_prior = torch.zeros(batch, self.k_len, device=y1.device)
        l2 = None
        for _ in range(self.num_iterations):
            l1 = self.decoder1(y1, le1_prior)
            le1 = l1 - le1_prior
            le2_prior = self._interleave(le1)
            l2 = self.decoder2(y2, le2_prior)
            le2 = l2 - le2_prior
            le1_prior = self._deinterleave(le2)
        return self._deinterleave(l2)


def build_bcjrnet() -> BCJRNetReceiver:
    return BCJRNetReceiver(k_len=8, num_iterations=6)


def example_input_bcjrnet() -> tuple[torch.Tensor, torch.Tensor]:
    batch, k_len = 2, 8
    y1 = torch.randn(batch, k_len, 2)
    y2 = torch.randn(batch, k_len, 2)
    return (y1, y2)


MENAGERIE_ZOO = "reimpl-pytorch"
MENAGERIE_ENTRIES = [
    ("bcjrnet_receiver", "build_bcjrnet", "example_input_bcjrnet", 2020, "REIMPL"),
]
