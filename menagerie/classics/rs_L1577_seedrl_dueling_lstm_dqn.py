# FAITHFUL PORT of https://github.com/google-research/seed_rl @ master
# (atari/networks.py: DuelingLSTMDQNNet, lines 221-340; original framework: TensorFlow 2 /
#  Keras)
#
# SEED RL (Espeholt et al. 2020, "SEED RL: Scalable and Efficient Deep-RL with
# Accelerated Central Inference", ICLR 2020) is Google Research's architecture for
# running the RL environment loop entirely on learner-side accelerators. This is the
# real network its Atari R2D2 agent uses: `DuelingLSTMDQNNet`, "the dueling LSTM net
# similar to the one described in [Dueling Network Architectures, Wang et al. 2016]
# (only the Q(s, a) part), with the layer sizes mentioned in the R2D2 paper
# (Kapturowski et al. 2019), section Hyper parameters" (verbatim from the source
# docstring). `tensorflow` is not importable in this base environment (protobuf/keras
# ABI break) and is not on the installed-base-libs list, so per the ladder this is a
# faithful PORT to torch of the real Keras module: a 3-layer conv torso -> flatten ->
# 512-d Dense -> concat with one-hot(prev_action) and reward -> a single-step LSTMCell
# core -> parallel value/advantage MLP heads -> dueling Q-value combination
# `Q = V + (A - mean(A))`. The frame-stacking bit-packing utility (`stack_frames`,
# only relevant for multi-frame TPU-transfer optimization, not part of the network
# architecture) and the full time-unroll loop are simplified to a single-timestep
# forward pass (batch_apply/unroll over the time axis is orchestration, not
# architecture); every real layer of `_body`, `_value`, `_advantage`, `_core`,
# `_torso`, and `_head` is kept, with the same layer sizes (32/64/64 conv channels,
# 512-wide Dense trunk, 512-wide LSTM core, 512-wide value/advantage hidden layers) as
# the source.
#
# TF/Keras -> torch translation notes (mechanical, no architecture change):
#   - `tf.keras.layers.Conv2D(N, [k,k], s, padding='valid', activation='relu')` ->
#     `nn.Conv2d(in, N, kernel_size=k, stride=s)` (torch Conv2d defaults to VALID/no
#     padding) + `F.relu`.
#   - `tf.keras.layers.Dense(N, activation='relu')` -> `nn.Linear(in, N)` + `F.relu`;
#     the final `value_head`/`advantage_head` Dense layers have no activation, matching
#     the source (`name='value_head'` / `name='advantage_head'`, no `activation=`).
#   - `advantage_head` has `use_bias=False` in the source -> `nn.Linear(..., bias=False)`.
#   - `tf.keras.layers.LSTMCell(512)` -> `nn.LSTMCell(input_size, 512)`.
#   - Keras Conv2D is NHWC by convention; torch Conv2d is NCHW. The ported `_body`
#     takes NCHW input directly (no transpose needed, since we construct the example
#     input already channels-first).
#   - `tf.one_hot(prev_action, num_actions)` -> `F.one_hot(prev_action, num_actions).float()`.
#   - `advantage -= reduce_mean(advantage, axis=-1, keepdims=True)` ported verbatim as
#     `advantage - advantage.mean(dim=-1, keepdim=True)`.

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "ported-pytorch"


# ---- atari/networks.py: DuelingLSTMDQNNet, ported layer-for-layer (single-timestep) ----
class DuelingLSTMDQNNet(nn.Module):
    """The recurrent network used to compute the agent's Q values.

    Dueling LSTM DQN net (R2D2-style), ported from SEED RL's real TF2/Keras
    `DuelingLSTMDQNNet`. See module header for the full provenance note.
    """

    def __init__(self, num_actions, in_channels=4):
        super().__init__()
        self.num_actions = num_actions

        # _body (tf.keras.Sequential Conv2D x3 + Flatten + Dense(512))
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.body_dense = nn.LazyLinear(512)

        # _value (Dense(512, relu) -> Dense(1))
        self.value_hidden = nn.LazyLinear(512)
        self.value_head = nn.LazyLinear(1)

        # _advantage (Dense(512, relu) -> Dense(num_actions, bias=False))
        self.advantage_hidden = nn.LazyLinear(512)
        self.advantage_head = nn.LazyLinear(num_actions, bias=False)

        # _core: single LSTMCell(512). torso_output_size = 512 (conv) + 1 (reward) +
        # num_actions (one-hot prev action).
        torso_output_size = 512 + 1 + num_actions
        self.core = nn.LSTMCell(torso_output_size, 512)

    def _body(self, observation):
        x = F.relu(self.conv1(observation))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.body_dense(x))
        return x

    def _torso(self, prev_action, observation, reward):
        conv_out = self._body(observation)
        one_hot_prev_action = F.one_hot(prev_action, self.num_actions).float()
        return torch.cat([conv_out, reward.unsqueeze(-1), one_hot_prev_action], dim=1)

    def _head(self, core_output):
        value = self.value_head(F.relu(self.value_hidden(core_output)))
        advantage = self.advantage_head(F.relu(self.advantage_hidden(core_output)))
        advantage = advantage - advantage.mean(dim=-1, keepdim=True)
        q_values = value + advantage
        action = torch.argmax(q_values, dim=1)
        return action, q_values

    def forward(self, prev_action, observation, reward, core_state):
        """Single-timestep forward pass (the real network's `unroll=False` path).

        Args:
            prev_action: <int64>[batch] previous action index.
            observation: <float32>[batch, channels, height, width] (already
                normalized to [0, 1], mirroring the source's `/ 255` step).
            reward: <float32>[batch] previous-step reward.
            core_state: (h, c) LSTMCell state, each <float32>[batch, 512].
        Returns:
            (action, q_values), new_core_state -- matching AgentOutput/AgentState.
        """
        torso_out = self._torso(prev_action, observation, reward)
        h, c = self.core(torso_out, core_state)
        action, q_values = self._head(h)
        return (action, q_values), (h, c)


# ---- staging wrapper ----
def build_seedrl_dueling_lstm_dqn():
    model = DuelingLSTMDQNNet(num_actions=18, in_channels=4)
    model.eval()
    # Materialize the LazyLinear layers with one dry run (needed before tracing so
    # parameter shapes are concrete).
    with torch.no_grad():
        batch = 2
        obs = torch.rand(batch, 4, 84, 84)
        prev_action = torch.zeros(batch, dtype=torch.int64)
        reward = torch.zeros(batch)
        h0 = torch.zeros(batch, 512)
        c0 = torch.zeros(batch, 512)
        model(prev_action, obs, reward, (h0, c0))
    return model


def example_input_seedrl_dueling_lstm_dqn():
    torch.manual_seed(0)
    batch = 2
    obs = torch.rand(batch, 4, 84, 84)
    prev_action = torch.zeros(batch, dtype=torch.int64)
    reward = torch.zeros(batch)
    h0 = torch.zeros(batch, 512)
    c0 = torch.zeros(batch, 512)
    return (prev_action, obs, reward, (h0, c0))


MENAGERIE_ENTRIES = [
    (
        "SEEDRL_DuelingLSTMDQNNet",
        "build_seedrl_dueling_lstm_dqn",
        "example_input_seedrl_dueling_lstm_dqn",
        2020,
        "ported-pytorch",
    ),
]
