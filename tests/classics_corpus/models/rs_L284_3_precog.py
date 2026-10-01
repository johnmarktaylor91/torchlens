# FAITHFUL PORT of nrhinehart/precog @ master (original framework: TensorFlow 1.x)
"""PRECOG (ICCV 2019) -- "PREdiction Conditioned On Goals", the ESP (Estimating Social-
forecast Probability) multi-agent joint trajectory model. The real repo
(nrhinehart/precog) is TensorFlow 1.x graph-mode code (`tf.contrib.rnn.GRUCell`,
`tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell`, `tf.layers.Dense`, `tf.linalg.expm`,
`@six.add_metaclass`) that cannot run in this repo's torch env, so per the ladder this is a
faithful port (rung 3), not a vendor.

`precog/bijection/esp_bijection.py:ESPJointTrajectoryBijectionMixin.forward` is the shared
autoregressive rollout used by every bijection variant in the repo: at each future timestep
t it calls the subclass's `step_generate(S_history, phi)` to produce a per-agent mean-offset
`m_t` and a matrix-exponential-parameterized covariance factor `sigma_t = expm(sigel_t)`,
forms a constant-velocity motion prior `mu_t = m_t + 2*S[t-1] - S[t-2]`, and normalizing-flow
warps a per-step Gaussian latent `Z_t` into the next joint agent position:
`S_t = mu_t + sigma_t @ Z_t`.

This ports the SIMPLEST concrete bijection in the repo,
`precog/bijection/contextless_rnn.py:ContextlessRNNBijection` (the paper's own "debug
purposes" / no-map-context baseline: a per-agent `GRUCell(32)` consuming the last two
2D positions -> `tanh(32)` -> `Linear(6)` MLP producing `m_t` (2) and `sigel_t` (4,
reshaped to 2x2)), transcribed line-for-line from TF1 ops to torch (`tf.contrib.rnn.GRUCell`
-> `nn.GRUCell(input_size=4, hidden_size=32)`; `tf.layers.Dense(32, tanh)` + `Dense(6)` ->
`nn.Linear` + `torch.tanh`; `tf.linalg.expm` -> `torch.linalg.matrix_exp`; the per-agent
Python loop, RNN input concatenation `[S[t-2], S[t-1]]`, and the `A`-agent independent GRU
states are all preserved exactly as `step_generate` computes them). The shared
`ESPJointTrajectoryBijectionMixin.forward` autoregressive rollout (the `mu_t = m_t +
2*S[t-1] - S[t-2]` motion prior and `S_t = mu_t + sigma_t @ Z_t` bijection step, looped over
T future timesteps) is also transcribed verbatim as `PrecogContextlessRNN.forward`.
"""

import torch
from torch import nn

MENAGERIE_ZOO = "ported-pytorch"


class ContextlessRNNStepGenerate(nn.Module):
    """precog/bijection/contextless_rnn.py:ContextlessRNNBijection.

    Real TF1 code (per agent a):
        last_agent_position = S_history[-1][..., a, :]      # (B, 2)
        lastlast_agent_position = S_history[-2][..., a, :]  # (B, 2)
        rnn_feats = concat((lastlast_agent_position, last_agent_position), axis=-1)  # (B, 4)
        output, rnn_state[a] = GRUCell(32)(rnn_feats, rnn_state[a])
        predictions = mlp([Dense(32, tanh), Dense(6)], output)
        m_ta = predictions[..., :2]
        sigel = reshape(predictions[..., 2:], (-1, 2, 2))
        sigma_ta = expm(sigel)
    Stacked over agents (axis=-2) to produce m_t: (B, A, 2), sigel_t/sigma_t: (B, A, 2, 2).
    """

    def __init__(self, num_agents: int, hidden_size: int = 32):
        super().__init__()
        self.A = num_agents
        self.hidden_size = hidden_size
        # One independent GRUCell + MLP head per agent, matching the real per-agent
        # `self.rnn` (a single shared GRUCell called once per agent with a per-agent
        # hidden state) -- weight-shared across agents in the original, preserved here as
        # a single shared cell + shared MLP head, with per-agent hidden state carried by
        # the caller (PrecogContextlessRNN), exactly mirroring `self.rnn_states[a]`.
        self.rnn = nn.GRUCell(input_size=4, hidden_size=hidden_size)
        self.mlp_hidden = nn.Linear(hidden_size, 32)
        self.mlp_out = nn.Linear(32, 6)

    def step(self, S_tm1: torch.Tensor, S_tm2: torch.Tensor, rnn_states: list) -> tuple:
        """S_tm1, S_tm2: (B, A, 2). rnn_states: list of A tensors, each (B, hidden_size).

        Returns (m_t, sigel_t, sigma_t, new_rnn_states) with m_t: (B, A, 2),
        sigel_t/sigma_t: (B, A, 2, 2).
        """
        ms, sigels, sigmas, new_states = [], [], [], []
        for a in range(self.A):
            last_pos = S_tm1[:, a, :]
            lastlast_pos = S_tm2[:, a, :]
            rnn_feats = torch.cat((lastlast_pos, last_pos), dim=-1)
            new_state = self.rnn(rnn_feats, rnn_states[a])
            new_states.append(new_state)

            hidden = torch.tanh(self.mlp_hidden(new_state))
            predictions = self.mlp_out(hidden)

            m_ta = predictions[..., :2]
            sigel = predictions[..., 2:].reshape(-1, 2, 2)
            sigma_ta = torch.linalg.matrix_exp(sigel)

            ms.append(m_ta)
            sigels.append(sigel)
            sigmas.append(sigma_ta)

        m_t = torch.stack(ms, dim=-2)
        sigel_t = torch.stack(sigels, dim=-3)
        sigma_t = torch.stack(sigmas, dim=-3)
        return m_t, sigel_t, sigma_t, new_states


class PrecogContextlessRNN(nn.Module):
    """precog/bijection/esp_bijection.py:ESPJointTrajectoryBijectionMixin.forward,
    specialized to the ContextlessRNNBijection step_generate above.

    Real TF1 rollout (rank-4 case, no sample dimension K):
        S_0, S_m1 = phi.S_past_car_frames[..., -1, :], phi.S_past_car_frames[..., -2, :]
        S_history = [S_m1, S_0]
        for t in range(T):
            m_t, sigel_t, sigma_t = step_generate(S_history, phi)
            mu_t = m_t + 2*S_history[-1] - S_history[-2]
            Z_t = Z[..., t, :]
            S_t = mu_t + einsum('...ij,...j->...i', sigma_t, Z_t)
            S_history.append(S_t)
        return S_history[2:]   # (B, A, T, 2) stacked
    """

    def __init__(self, num_agents: int, hidden_size: int = 32):
        super().__init__()
        self.A = num_agents
        self.hidden_size = hidden_size
        self.step_generate = ContextlessRNNStepGenerate(num_agents, hidden_size)

    def forward(self, S_past: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        """S_past: (B, A, 2, 2) -- last two past positions [S_{-1}, S_0] per agent.
        Z: (B, A, T, 2) -- per-step Gaussian latents.
        Returns S_future: (B, A, T, 2).
        """
        B = S_past.shape[0]
        T = Z.shape[-2]

        S_m1 = S_past[:, :, 0, :]
        S_0 = S_past[:, :, 1, :]
        S_history = [S_m1, S_0]

        rnn_states = [
            torch.zeros(B, self.hidden_size, dtype=S_past.dtype, device=S_past.device)
            for _ in range(self.A)
        ]

        for t_idx in range(T):
            m_t, sigel_t, sigma_t, rnn_states = self.step_generate.step(
                S_history[-1], S_history[-2], rnn_states
            )
            mu_t = m_t + 2 * S_history[-1] - S_history[-2]

            Z_t = Z[..., t_idx, :]
            S_t = mu_t + torch.einsum("...ij,...j->...i", sigma_t, Z_t)
            S_history.append(S_t)

        return torch.stack(S_history[2:], dim=-2)


_NUM_AGENTS = 3
_HIDDEN = 32
_T_FUTURE = 4


def build_precog():
    return PrecogContextlessRNN(num_agents=_NUM_AGENTS, hidden_size=_HIDDEN)


def example_input_precog():
    torch.manual_seed(0)
    batch = 2
    s_past = torch.randn(batch, _NUM_AGENTS, 2, 2)
    z = torch.randn(batch, _NUM_AGENTS, _T_FUTURE, 2)
    return (s_past, z)


MENAGERIE_ENTRIES = [
    ("PRECOG", "build_precog", "example_input_precog", 2019, "ported-pytorch"),
]
