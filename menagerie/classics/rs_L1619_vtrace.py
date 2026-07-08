# FAITHFUL PORT of https://github.com/google-research/seed_rl @ master (original framework: TensorFlow 2 / tf.Module)
# (agents/vtrace/networks.py: MLPandLSTM, lines 1-119)
#
# V-trace (Espeholt et al. 2018, "IMPALA: Scalable Distributed Deep-RL with Importance
# Weighted Actor-Learner Architectures", arXiv:1802.01561) -- the off-policy correction
# used by SEED RL's `vtrace` agent. seed_rl is Google Research's TF2 distributed-RL
# framework; installing its full TPU/gRPC actor-learner infrastructure (and its
# `tf.nest`/`SingularMonitoredSession`-era TF1-Sonnet sibling that predates it) is
# unreasonable for a single traced module, so the real `agents/vtrace/networks.py`
# `MLPandLSTM` architecture is transcribed faithfully into base-env torch: an MLP encoder
# (`tf.keras.layers.Dense(size, 'relu')` per size in `mlp_sizes`) feeding a stacked-LSTM
# core (`tf.keras.layers.StackedRNNCells` of `LSTMCell`s, one per `lstm_sizes` entry, with
# per-timestep "reset state to zero on episode `done`" gating -- V-trace's defining
# recurrent-core mechanism for handling episode boundaries within an unroll), followed by
# separate `policy_logits` and `baseline` (value) linear heads -- exactly the network
# structure `_unroll`/`_head` build in the original. The `parametric_action_distribution`
# (an external, environment-specific action-sampling helper injected by the caller, not
# part of the network architecture) is replaced with a plain `argmax` over `policy_logits`
# to keep the port self-contained; every network layer/mechanism the original defines is
# preserved (MLP trunk, per-step done-masked stacked-LSTM core, policy/baseline heads).

import torch
import torch.nn as nn

MENAGERIE_ZOO = "ported-pytorch"


class MLPandLSTM(nn.Module):
    """Faithful torch port of seed_rl's agents/vtrace/networks.py MLPandLSTM: MLP trunk
    -> per-timestep done-masked stacked-LSTM core -> policy_logits/baseline heads."""

    def __init__(self, obs_dim, num_actions, mlp_sizes, lstm_sizes):
        super().__init__()
        self.num_actions = num_actions
        self.lstm_sizes = lstm_sizes

        # MLP trunk: tf.keras.Sequential([Dense(size, 'relu') for size in mlp_sizes])
        mlp_layers = []
        in_dim = obs_dim
        for size in mlp_sizes:
            mlp_layers.append(nn.Linear(in_dim, size))
            mlp_layers.append(nn.ReLU())
            in_dim = size
        self.mlp = nn.Sequential(*mlp_layers)

        # Stacked LSTM core: tf.keras.layers.StackedRNNCells([LSTMCell(size) for size in lstm_sizes])
        core_in = in_dim
        self.lstm_cells = nn.ModuleList()
        for size in lstm_sizes:
            self.lstm_cells.append(nn.LSTMCell(core_in, size))
            core_in = size
        core_output_size = lstm_sizes[-1]

        # _head: policy_logits / baseline linear heads
        self.policy_logits = nn.Linear(core_output_size, num_actions)
        self.baseline = nn.Linear(core_output_size, 1)

    def initial_state(self, batch_size, device):
        return [
            (
                torch.zeros(batch_size, size, device=device),
                torch.zeros(batch_size, size, device=device),
            )
            for size in self.lstm_sizes
        ]

    def _head(self, core_output):
        policy_logits = self.policy_logits(core_output)
        baseline = self.baseline(core_output).squeeze(-1)
        # Sample an action from the policy (parametric_action_distribution.sample in the
        # original is environment-supplied; argmax keeps this self-contained).
        action = torch.argmax(policy_logits, dim=-1)
        return action, policy_logits, baseline

    def forward(self, observation, done, core_state=None):
        """
        :param observation: (T, B, obs_dim) unrolled environment observations
        :param done: (T, B) bool episode-termination flags per timestep
        :param core_state: optional list of (h, c) tuples per LSTM layer; defaults to zeros
        :return: (action, policy_logits, baseline) each (T, B, ...), plus final core_state
        """
        t_len, batch = observation.shape[0], observation.shape[1]
        device = observation.device
        x = self.mlp(observation)  # (T, B, mlp_sizes[-1])

        if core_state is None:
            core_state = self.initial_state(batch, device)
        initial_core_state = self.initial_state(batch, device)

        core_output_list = []
        for t in range(t_len):
            input_t = x[t]
            d = done[t]
            # Reset core state to zero whenever the episode ended (per original _unroll).
            new_state = []
            for layer_idx, (h, c) in enumerate(core_state):
                init_h, init_c = initial_core_state[layer_idx]
                mask = (~d).float().unsqueeze(-1)
                h = mask * h + (1.0 - mask) * init_h
                c = mask * c + (1.0 - mask) * init_c
                new_state.append((h, c))
            core_state = new_state

            layer_input = input_t
            next_state = []
            for cell, (h, c) in zip(self.lstm_cells, core_state):
                h, c = cell(layer_input, (h, c))
                next_state.append((h, c))
                layer_input = h
            core_state = next_state
            core_output_list.append(layer_input)

        core_output = torch.stack(core_output_list)  # (T, B, lstm_sizes[-1])
        action, policy_logits, baseline = self._head(core_output)
        return action, policy_logits, baseline


def build_vtrace_mlp_lstm():
    return MLPandLSTM(obs_dim=16, num_actions=6, mlp_sizes=[64, 64], lstm_sizes=[64])


def example_input_vtrace_mlp_lstm():
    torch.manual_seed(0)
    t_len, batch, obs_dim = 4, 2, 16
    observation = torch.randn(t_len, batch, obs_dim)
    done = torch.zeros(t_len, batch, dtype=torch.bool)
    return (observation, done)


MENAGERIE_ENTRIES = [
    (
        "VTrace_MLPandLSTM",
        "build_vtrace_mlp_lstm",
        "example_input_vtrace_mlp_lstm",
        2018,
        "ported-pytorch",
    ),
]
