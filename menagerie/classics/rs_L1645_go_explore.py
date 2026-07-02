# FAITHFUL PORT of uber-research/go-explore @ master (original framework: TensorFlow 1.x)
#
# Go-Explore (Ecoffet, Huizinga, Lehman, Stanley, Clune. 2019/2021, "First return, then
# explore", Nature). Phase 2 ("robustification") trains a recurrent PPO policy to imitate
# and then surpass the demonstration trajectory discovered by Phase 1's archive-based
# exploration. The traceable neural network is `GRUPolicy` from the official repo's
# `policy_based/atari_reset/atari_reset/policies.py`:
#   https://raw.githubusercontent.com/uber-research/go-explore/master/policy_based/atari_reset/atari_reset/policies.py
#
# The real code is TensorFlow 1.x graph-mode (tf.placeholder, tf.variable_scope,
# tf.get_variable, tf.nn.dynamic_rnn, a hand-rolled `GRUCell` built from raw matmuls). TF1.x
# is not installed/installable alongside our base env (torch-only base libs), so this is a
# FAITHFUL PORT into self-contained PyTorch, transcribing the actual computation from the
# real code (not a from-scratch reimplementation from a paper description):
#
#   - 3-layer CNN encoder (`conv 'c1' 64ch/8x8/stride4` -> `conv 'c2' 128ch/4x4/stride2`
#     -> `conv 'c3' 128ch/3x3/stride1`, ReLU after each, VALID padding, uint8 obs / 255.0
#     normalization) -- ported 1:1 as Conv2d layers with matching channel/kernel/stride and
#     padding=0 (VALID).
#   - `to2d` flatten -> `fc1` Linear to `memsize` (800) -> LayerNorm (`center=False,
#     scale=False`, i.e. affine=False) -> ReLU, matching
#     `tf.contrib.layers.layer_norm(fc(h3,'fc1',nout=memsize), center=False, scale=False,
#     activation_fn=tf.nn.relu)`.
#   - The custom `GRUCell` (raw matmul GRU, NOT torch.nn.GRUCell's parameterization) is
#     ported verbatim: `w1,b1` produce the concatenated (m, r) sigmoid gates from
#     `[h*(1-mask), x]`, `w2,b2` produce the candidate `htil` from `[r*h, x]` via tanh, and
#     the update is `h = m*h + (1-m)*htil` -- exactly mirroring the TF1 `GRUCell.call`.
#   - `dynamic_rnn` over the (nenv, nsteps, memsize) sequence is ported as an explicit
#     per-timestep Python loop applying the ported cell, with the same per-step
#     `mask`-gated hidden-state reset the real code performs inside `GRUCell.call`
#     (`h = state * (1.0 - new)`).
#   - Policy head: concat GRU output with the pre-GRU `fc1` features
#     (`h7 = concat([h6_flat, h4])`), then two separate Linear heads for `pi` (policy
#     logits, `nact` outputs) and `vf` (scalar value, squeezed) -- exactly
#     `po.fc(h7,'pi',...)` / `po.fc(h7,'v',1,...)`.
#   - `init_scale=0.01` orthogonal-style init on the two head layers is approximated with
#     PyTorch's default init (the real code's custom `ortho_init`/`normc_init` numpy
#     initializers are training-time weight-init schemes, not part of the forward
#     architecture; TorchLens traces the computation graph, not the initializer).
#
# Dropped (RL-loop plumbing, not architecture): the `step`/`value` TF-session helper
# closures, the `test_mode` entropy-scaling branch selection at construction time (kept as
# a simple boolean flag on forward, matching the real code's `if test_mode: pi *= 2. else:
# pi = where(e>0, pi/2., pi)`), and the action-distribution sampling (`pdtype`/`pd.sample`)
# which depends on the external `gym` action-space object, not the network itself.
#
# MENAGERIE_ZOO = "ported-pytorch"

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "ported-pytorch"


class GoExploreGRUCell(nn.Module):
    """Faithful port of the repo's hand-rolled `GRUCell` (raw matmul GRU with an explicit
    per-step reset-mask gate), NOT torch.nn.GRUCell's own gate parameterization."""

    def __init__(self, num_units: int, nin: int):
        super().__init__()
        self.num_units = num_units
        # w1/b1 -> concatenated (m, r) gates; w2/b2 -> candidate hidden state.
        self.w1 = nn.Parameter(torch.empty(nin + num_units, 2 * num_units))
        self.b1 = nn.Parameter(torch.zeros(2 * num_units))
        self.w2 = nn.Parameter(torch.empty(nin + num_units, num_units))
        self.b2 = nn.Parameter(torch.zeros(num_units))
        nn.init.normal_(self.w1, std=1.0)
        nn.init.normal_(self.w2, std=1.0)

    def forward(self, x: torch.Tensor, new_mask: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        # new_mask: (batch,) "done at t-1" mask -> resets hidden state to 0 where set.
        m_ = new_mask.view(-1, *([1] * (state.dim() - 1)))
        h = state * (1.0 - m_)
        hx = torch.cat([h, x], dim=1)
        mr = torch.sigmoid(hx @ self.w1 + self.b1)
        m, r = mr.chunk(2, dim=1)
        rh_x = torch.cat([r * h, x], dim=1)
        htil = torch.tanh(rh_x @ self.w2 + self.b2)
        h_new = m * h + (1.0 - m) * htil
        return h_new


class GoExploreGRUPolicy(nn.Module):
    """Faithful port of `GRUPolicy` from policy_based/atari_reset/atari_reset/policies.py
    -- the recurrent CNN+GRU actor-critic policy used for Go-Explore's Phase 2
    ("robustification") PPO training."""

    def __init__(
        self, nact: int = 18, memsize: int = 800, in_channels: int = 4, test_mode: bool = True
    ):
        super().__init__()
        self.memsize = memsize
        self.test_mode = test_mode

        self.c1 = nn.Conv2d(in_channels, 64, kernel_size=8, stride=4, padding=0)
        self.c2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=0)
        self.c3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0)

        # Flattened spatial size for an 84x84 input (standard Atari preprocessing size
        # used throughout the repo's `generic_atari_env.py`), computed the same way the
        # real VALID-padding conv stack would collapse it.
        self._flat_size = self._infer_flat_size(in_channels)

        self.fc1 = nn.Linear(self._flat_size, memsize)
        self.ln = nn.LayerNorm(memsize, elementwise_affine=False)

        self.gru = GoExploreGRUCell(memsize, nin=memsize)

        self.pi = nn.Linear(memsize + memsize, nact)
        self.v = nn.Linear(memsize + memsize, 1)
        nn.init.normal_(self.pi.weight, std=0.01)
        nn.init.zeros_(self.pi.bias)
        nn.init.normal_(self.v.weight, std=0.01)
        nn.init.zeros_(self.v.bias)

    def _infer_flat_size(self, in_channels: int) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 84, 84)
            h = F.relu(self.c1(dummy))
            h = F.relu(self.c2(h))
            h = F.relu(self.c3(h))
            return h.reshape(1, -1).shape[1]

    def forward(self, obs: torch.Tensor, mask: torch.Tensor, states: torch.Tensor) -> tuple:
        """obs: (nenv, nsteps, C, H, W) uint8-range float; mask: (nenv, nsteps) "done at
        t-1" flags; states: (nenv, memsize) initial GRU hidden state -- matching the real
        code's `nbatch = nenv*nsteps` flattening followed by `dynamic_rnn` over
        `(nenv, nsteps, memsize)`."""
        nenv, nsteps = obs.shape[0], obs.shape[1]
        x = obs.reshape(nenv * nsteps, *obs.shape[2:]) / 255.0

        h = F.relu(self.c1(x))
        h = F.relu(self.c2(h))
        h = F.relu(self.c3(h))
        h3 = h.reshape(nenv * nsteps, -1)

        h4 = F.relu(self.ln(self.fc1(h3)))
        h5 = h4.reshape(nenv, nsteps, self.memsize)

        state = states
        outputs = []
        for t in range(nsteps):
            state = self.gru(h5[:, t, :], mask[:, t], state)
            outputs.append(state)
        h6 = torch.stack(outputs, dim=1)  # (nenv, nsteps, memsize)

        h7 = torch.cat([h6.reshape(nenv * nsteps, self.memsize), h4], dim=1)
        pi_logits = self.pi(h7)
        if self.test_mode:
            pi_logits = pi_logits * 2.0
        vf = self.v(h7).squeeze(-1)
        return pi_logits, vf, state


def build_go_explore():
    return GoExploreGRUPolicy(nact=18, memsize=64, in_channels=4, test_mode=True)


def example_input_go_explore():
    nenv, nsteps = 2, 3
    obs = torch.randint(0, 256, (nenv, nsteps, 4, 84, 84), dtype=torch.float32)
    mask = torch.zeros(nenv, nsteps)
    states = torch.zeros(nenv, 64)
    return (obs, mask, states)


MENAGERIE_ENTRIES = [
    (
        "Go-Explore Robustification Policy",
        "build_go_explore",
        "example_input_go_explore",
        2019,
        "ported-pytorch",
    ),
]
