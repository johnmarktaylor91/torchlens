# FAITHFUL PORT of steffenvan/IMPALA-PopArt @ master (original framework: TensorFlow 1.x / DeepMind Sonnet)
#
# Source model file ported: popart/agent.py (class PopArtFeedForward), which builds directly on
# the shallow_convolution() torso helper also defined in that file. The original repo depends on
# `sonnet` (DeepMind Sonnet v1, TF1-only) and `tf.contrib`, neither of which is installable in the
# current base env (Sonnet v1 requires TF1.x; tf.contrib was removed from TF2). This module
# faithfully transcribes the PopArtFeedForward forward architecture -- the shallow conv torso,
# the concatenation of conv features with clipped reward and one-hot last action, the multi-task
# (per-game) PopArt-normalized value head with the affine un-normalization
# `un_normalized_vf = std * normalized_vf + mean`, and the policy-logits head -- into self-contained
# base-env torch. The `update_moments` PopArt statistics-adaptation routine (a training-time-only
# in-place parameter rescaling of the value head, not part of the forward compute graph) and the
# actor/learner unroll/V-trace plumbing are training infrastructure and are intentionally not
# ported; only the forward network architecture is in scope for TorchLens capture.
#
# Reference (paper): "Multi-task Deep Reinforcement Learning with PopArt" (Hessel et al., 2018),
# combined with IMPALA (Espeholt et al., 2018) as implemented in this repo.

import torch
import torch.nn as nn


class ShallowConvTorso(nn.Module):
    """Faithful port of shallow_convolution() in popart/agent.py."""

    def __init__(self, in_channels: int):
        super().__init__()
        # snt.Conv2D(16, 8, stride=4) -> snt.Conv2D(32, 4, stride=2), same-style padding
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=8, stride=4, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        conv_out = self.conv1(frame)
        conv_out = torch.relu(conv_out)
        conv_out = self.conv2(conv_out)
        return conv_out


class PopArtFeedForward(nn.Module):
    """Faithful port of PopArtFeedForward in popart/agent.py.

    Reproduces the `_torso` and `_head` forward methods of the original snt.AbstractModule:
      - shallow conv torso -> flatten -> Linear(256) -> ReLU
      - concat with clipped last reward and one-hot last action
      - multi-task PopArt value head: normalized_vf_games (Linear "baseline"),
        un_normalized_vf_games = std * normalized_vf_games + mean (PopArt Pop step, eq. in
        Hessel et al. 2018), then per-example task ("level") gather down to the active game
      - policy_logits head (Linear)

    The original network selects one of `number_of_games` output heads via a "level_name" id
    tensor at runtime (multi-task Atari agent, gathered via tf.batch_gather). We keep that
    exact per-example gather so the forward architecture (including the task-selection gather
    op) is preserved faithfully, rather than collapsing to a single-task head.
    """

    def __init__(self, num_actions: int, number_of_games: int, in_channels: int = 4):
        super().__init__()
        self.num_actions = num_actions
        self.number_of_games = number_of_games

        self.torso = ShallowConvTorso(in_channels)
        # Flattened conv-torso feature size for the shallow torso at (in_channels, 84, 84):
        # conv1 -> (16, 21, 21); conv2 -> (32, 10, 10) -> 3200
        self._conv_flat_dim = 32 * 10 * 10
        self.fc_torso = nn.Linear(self._conv_flat_dim, 256)

        # torso_output dim after concatenating clipped reward (1) and one-hot last action (num_actions)
        head_in_dim = 256 + 1 + num_actions

        self.baseline = nn.Linear(head_in_dim, number_of_games)
        self.policy_logits = nn.Linear(head_in_dim, num_actions)

        # PopArt running per-game moments (non-trainable statistics, as in the original
        # tf.get_variable(..., trainable=False) buffers).
        self.register_buffer("mean", torch.zeros(number_of_games))
        self.register_buffer("mean_squared", torch.ones(number_of_games))

    def _std(self) -> torch.Tensor:
        return torch.sqrt(self.mean_squared - self.mean.square()).detach()

    def _torso_forward(
        self, frame: torch.Tensor, last_reward: torch.Tensor, last_action: torch.Tensor
    ) -> torch.Tensor:
        frame = frame.float() / 255.0
        conv_out = self.torso(frame)
        conv_out = torch.relu(conv_out)
        conv_out = conv_out.reshape(conv_out.shape[0], -1)
        conv_out = self.fc_torso(conv_out)
        conv_out = torch.relu(conv_out)

        clipped_reward = torch.clamp(last_reward, -1.0, 1.0).unsqueeze(-1)
        one_hot_last_action = torch.nn.functional.one_hot(last_action, self.num_actions).float()
        return torch.cat([conv_out, clipped_reward, one_hot_last_action], dim=1)

    def forward(
        self,
        frame: torch.Tensor,
        last_reward: torch.Tensor,
        last_action: torch.Tensor,
        level_name: torch.Tensor,
    ):
        torso_output = self._torso_forward(frame, last_reward, last_action)

        normalized_vf_games = self.baseline(torso_output)
        std = self._std()
        un_normalized_vf_games = std * normalized_vf_games + self.mean

        # Per-example gather of the active game's value estimate (mirrors tf.batch_gather
        # against `level_name` in the original _head()).
        idx = level_name.view(-1, 1)
        normalized_vf = torch.gather(normalized_vf_games, 1, idx).squeeze(-1)
        un_normalized_vf = torch.gather(un_normalized_vf_games, 1, idx).squeeze(-1)

        policy_logits = self.policy_logits(torso_output)
        action = torch.distributions.Categorical(logits=policy_logits).sample()

        return action, policy_logits, un_normalized_vf, normalized_vf


MENAGERIE_ZOO = "ported-pytorch"


def build_popart_impala():
    return PopArtFeedForward(num_actions=6, number_of_games=3, in_channels=4)


def example_input_popart_impala():
    batch = 2
    frame = torch.randint(0, 256, (batch, 4, 84, 84), dtype=torch.uint8).float()
    last_reward = torch.zeros(batch)
    last_action = torch.zeros(batch, dtype=torch.long)
    level_name = torch.zeros(batch, dtype=torch.long)
    return frame, last_reward, last_action, level_name


MENAGERIE_ENTRIES = [
    (
        "popart_impala_feedforward",
        build_popart_impala,
        example_input_popart_impala,
        2018,
        MENAGERIE_ZOO,
    ),
]
