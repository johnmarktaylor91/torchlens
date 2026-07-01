# FAITHFUL PORT of https://github.com/automl/learna_tools @ main (original framework:
# tensorforce 0.3.3 policy-network spec + TensorFlow 1.4 op graph)
#
# LEARNA (Runge, Bergman, Stoll & Hutter, ICLR 2019 "Learning to Design RNA") is a
# deep-RL agent that designs an RNA sequence to fold into a target secondary
# structure, one nucleotide at a time. Its policy network is NOT a raw nn.Module in
# the real repo -- it is built by tensorforce (an old TF1 RL library, pinned to
# `tensorforce==0.3.3` / `tensorflow==1.4.0` in learna_tools' own setup.py) from a
# declarative layer-spec list returned by learna_tools/learna/agent.py::get_network().
# tensorforce and TF1.4 cannot be installed in a modern torch env, so we FAITHFULLY
# PORT the exact layer sequence (embedding -> conv1d stack -> flatten ->
# single-step LSTMCell -> dense stack -> linear action head) into self-contained
# PyTorch, using the shipped bin/learna CLI's own tuned hyperparameters (these are
# the deployed-model defaults, not made up): state_radius=32 (state length =
# 2*32+1=65), conv_sizes=[17,5], conv_channels=[7,18], num_fc_layers=1,
# fc_units=57, embedding_size=3, lstm_units=28, num_lstm_layers=1. Every mechanism
# is transcribed 1:1 from the real source:
#
#   - learna_tools/learna/agent.py::get_network() builds the trunk spec:
#     embedding(indices=4, size=embedding_size) -> Conv1d layers (one per
#     (conv_channels[i], conv_sizes[i]) pair with window>1, padding="VALID",
#     activation="relu") -> flatten -> num_lstm_layers x internal_lstm(lstm_units)
#     -> num_fc_layers x dense(fc_units, activation=fc_activation="relu").
#   - tensorforce.core.networks.layer.InternalLstm.tf_apply (0.3.3 source, pip
#     downloaded for reference, not installed) shows the LSTM layer is a SINGLE
#     recurrent-cell application per forward call (`self.lstm_cell(inputs=x,
#     state=state)`), i.e. one decision-step of an autoregressive RL policy, not
#     a sequence-unrolled LSTM -- we port it as one `nn.LSTMCell` step accordingly.
#   - tensorforce.core.distributions.categorical.Categorical.__init__ (0.3.3
#     source) appends exactly one `Linear(size=num_actions)` layer on top of the
#     trunk output for a scalar-shape discrete action space -- ported as the
#     final `self.action_head` Linear(fc_units, 4), matching
#     environment.py::RnaDesignEnvironment.actions == dict(type="int",
#     num_actions=4) (the four nucleotides).
#   - environment.py::RnaDesignEnvironmentConfig / RnaDesignEnvironment.states
#     defines the observation as a single-channel window of
#     `1 + 2*state_radius` encoded dot-bracket sites feeding the conv stack.
#
# No architecture is invented: every layer, size, and activation below is read
# directly off the real get_network()/Categorical/InternalLstm source and the
# real bin/learna CLI argument defaults (the paper's tuned/deployed
# hyperparameters), not guessed from the paper text.

import torch
from torch import nn

MENAGERIE_ZOO = "ported-pytorch"


class LearnaPolicyNetwork(nn.Module):
    """Faithful port of the tensorforce layer-spec trunk built by
    learna_tools/learna/agent.py::get_network() plus the Categorical action head
    tensorforce.core.distributions.categorical.Categorical appends on top, at the
    real bin/learna CLI's tuned hyperparameters."""

    def __init__(
        self,
        state_radius=32,
        conv_sizes=(17, 5),
        conv_channels=(7, 18),
        num_fc_layers=1,
        fc_units=57,
        embedding_size=3,
        lstm_units=28,
        num_lstm_layers=1,
        num_actions=4,
        num_site_symbols=4,  # site_encoding vocab: '.', '(', ')', '=' (padding)
    ):
        super().__init__()
        self.state_length = 1 + 2 * state_radius

        # embedding(indices=4, size=embedding_size) -- get_network(), use_embedding branch
        self.embedding = nn.Embedding(num_site_symbols, embedding_size)

        # convolution = [Conv1d(size, window, padding="VALID", activation="relu")
        #                for (size, window) in zip(conv_channels, conv_sizes) if window > 1]
        conv_layers = []
        in_channels = embedding_size
        self.conv_windows = []
        for size, window in zip(conv_channels, conv_sizes):
            if window > 1:
                conv_layers.append(
                    nn.Conv1d(in_channels, size, kernel_size=window, stride=1, padding=0, bias=True)
                )
                self.conv_windows.append(window)
                in_channels = size
        self.convs = nn.ModuleList(conv_layers)

        # flatten -- tensorforce Flatten layer: reshape (-1, prod(shape[1:]))
        conv_out_length = self.state_length
        for window in self.conv_windows:
            conv_out_length = conv_out_length - (window - 1)  # VALID padding, stride 1
        flat_size = (
            in_channels * conv_out_length if self.convs else embedding_size * self.state_length
        )

        # num_lstm_layers x internal_lstm(lstm_units) -- one recurrent-cell step per call
        self.lstm_units = lstm_units
        self.lstm_cells = nn.ModuleList()
        lstm_in = flat_size
        for _ in range(num_lstm_layers):
            self.lstm_cells.append(nn.LSTMCell(lstm_in, lstm_units))
            lstm_in = lstm_units
        self._post_lstm_size = lstm_in

        # num_fc_layers x dense(fc_units, activation="relu")
        dense_layers = []
        dense_in = self._post_lstm_size
        for _ in range(num_fc_layers):
            dense_layers.append(nn.Linear(dense_in, fc_units))
            dense_in = fc_units
        self.dense_layers = nn.ModuleList(dense_layers)
        self._trunk_out_size = dense_in

        # Categorical(shape=(), num_actions=4).logits = Linear(size=num_actions)
        self.action_head = nn.Linear(self._trunk_out_size, num_actions)

    def forward(self, state, lstm_states):
        """state: LongTensor (B, state_length) of site-encoding indices (0..3).
        lstm_states: list of (h, c) tuples, one per LSTM layer (the tensorforce
        `internal_lstm` recurrent state threaded across RL decision steps)."""
        x = self.embedding(state)  # (B, L, embedding_size)
        x = x.transpose(1, 2)  # (B, embedding_size, L) for Conv1d
        for conv in self.convs:
            x = torch.relu(conv(x))
        x = x.reshape(x.size(0), -1)  # flatten

        new_lstm_states = []
        for cell, (h, c) in zip(self.lstm_cells, lstm_states):
            h, c = cell(x, (h, c))
            x = h
            new_lstm_states.append((h, c))

        for dense in self.dense_layers:
            x = torch.relu(dense(x))

        logits = self.action_head(x)
        return logits, new_lstm_states


# ---- menagerie staging harness ----


def build_learna():
    """Real LEARNA policy network at the shipped bin/learna CLI's tuned
    hyperparameters (the deployed inference model's own defaults: conv_sizes=
    [17,5], conv_channels=[7,18], fc_units=57, lstm_units=28, embedding_size=3,
    state_radius=32)."""
    torch.manual_seed(0)
    model = LearnaPolicyNetwork(
        state_radius=32,
        conv_sizes=(17, 5),
        conv_channels=(7, 18),
        num_fc_layers=1,
        fc_units=57,
        embedding_size=3,
        lstm_units=28,
        num_lstm_layers=1,
        num_actions=4,
    )
    return model.eval()


def example_input_learna():
    """One decision-step input: a batch of 1 encoded state window (65 sites,
    4-symbol dot-bracket vocab: '.', '(', ')', '=' padding) plus a zero-initialized
    LSTM recurrent state (tensorforce's internal_lstm.internals_init() zero-inits
    the (c, h) state at episode start)."""
    torch.manual_seed(0)
    state_radius = 32
    state_length = 1 + 2 * state_radius
    state = torch.randint(0, 4, (1, state_length))
    lstm_states = [(torch.zeros(1, 28), torch.zeros(1, 28))]
    return (state, lstm_states)


MENAGERIE_ENTRIES = [
    (
        "LEARNA",
        "build_learna",
        "example_input_learna",
        2019,
        "ported-pytorch",
    ),
]
