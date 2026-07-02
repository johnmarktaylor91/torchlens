# FAITHFUL PORT of uber-research/go-explore @ master (original framework: TensorFlow 1.x)
# https://github.com/uber-research/go-explore/blob/master/policy_based/atari_reset/atari_reset/policies.py
#
# Go-Explore (Ecoffet, Huizinga, Lehman, Stanley, Clune, "Go-Explore: a New
# Approach for Hard-Exploration Problems", 2019/2021) is a two-phase
# algorithm: Phase 1 ("explore until solved") builds an archive of visited
# "cells" via random exploration + return-to-cell restarts (requires raw
# emulator/game-state restore, not a neural net); Phase 2 ("robustification")
# trains an ordinary imitation-learning / PPO policy network to reliably
# reproduce the best discovered trajectory. The queue notes describe this
# Phase 2 network as "a standard BC net" -- that network is `CnnPolicy` in
# `atari_reset/atari_reset/policies.py`, the non-recurrent sibling of the
# repo's `GRUPolicy`.
#
# `policies.py` is TF1-only (`tf.placeholder`, `tf.variable_scope`,
# `tf.get_variable`, session-graph API) -- it cannot be run or vendored as-is
# in a modern base-env torch install. This module transcribes `CnnPolicy`
# faithfully into torch: the same three conv layers (`VALID` padding, exact
# channel counts/kernel sizes/strides as the `conv()` calls below), the same
# flatten -> 1024-wide FC trunk, and the same linear policy (`nact` outputs)
# and value (`1` output) heads. `init_scale`/`ortho_init`/`normc_init`
# (orthogonal-scaled and normalized-column weight inits) are TF1-training
# details that do not change the traced graph topology and are not
# reproduced; the layer shapes and connectivity are the load-bearing
# architecture and those are ported 1:1.
#
# Original TF1 reference (verbatim, for comparison):
#
#   def conv(x, scope, noutchannels, filtsize, stride, pad='VALID', init_scale=1.0):
#       with tf.variable_scope(scope):
#           nin = x.get_shape()[3].value
#           w = tf.get_variable("w", [filtsize, filtsize, nin, noutchannels], initializer=ortho_init(init_scale))
#           b = tf.get_variable("b", [noutchannels], initializer=tf.constant_initializer(0.0))
#           z = tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding=pad)+b
#           return z
#
#   class CnnPolicy(object):
#       def __init__(self, sess, ob_space, ac_space, nbatch, _nsteps, _test_mode=False, reuse=False):
#           nh, nw, nc = ob_space.shape
#           ob_shape = (nbatch, nh, nw, nc)
#           nact = ac_space.n
#           x = tf.placeholder(tf.uint8, ob_shape)
#           with tf.variable_scope("model", reuse=reuse):
#               h = tf.nn.relu(conv(tf.cast(x, tf.float32)/255., 'c1', noutchannels=64, filtsize=8, stride=4))
#               h2 = tf.nn.relu(conv(h, 'c2', noutchannels=128, filtsize=4, stride=2))
#               h3 = tf.nn.relu(conv(h2, 'c3', noutchannels=128, filtsize=3, stride=1))
#               h3 = to2d(h3)
#               h4 = tf.nn.relu(fc(h3, 'fc1', nout=1024))
#               pi = fc(h4, 'pi', nact, init_scale=0.01)
#               vf = fc(h4, 'v', 1, init_scale=0.01)[:, 0]
import torch
import torch.nn as nn


class GoExploreCnnPolicy(nn.Module):
    """Faithful torch port of Go-Explore's `CnnPolicy` (Phase 2 robustification
    policy): a standard "Nature DQN"-style conv trunk (8x8/4 -> 4x4/2 -> 3x3/1,
    all VALID/no padding, matching TF1 `conv(..., pad='VALID')`) feeding a
    1024-wide FC layer, then separate linear policy-logits and scalar-value
    heads -- exactly the graph `CnnPolicy.__init__` builds in the original
    TF1 code (see header)."""

    def __init__(self, in_channels: int, n_actions: int, fc_dim: int = 1024) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=8, stride=4, padding=0)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0)
        self.relu = nn.ReLU()

        # `to2d`/flatten dimension depends on the input spatial size; resolved
        # lazily on first forward (same effect as TF1's shape-inferred `fc`).
        self.fc_dim = fc_dim
        self.fc1: nn.Linear | None = None
        self.pi: nn.Linear | None = None
        self.vf: nn.Linear | None = None
        self.n_actions = n_actions

    def _build_head(self, flat_dim: int, device: torch.device) -> None:
        self.fc1 = nn.Linear(flat_dim, self.fc_dim).to(device)
        self.pi = nn.Linear(self.fc_dim, self.n_actions).to(device)
        self.vf = nn.Linear(self.fc_dim, 1).to(device)

    def forward(self, obs_uint8: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # obs_uint8: (batch, channels, H, W), uint8-valued floats in [0, 255],
        # matching the original `tf.cast(x, tf.float32) / 255.` normalization.
        x = obs_uint8.float() / 255.0

        h = self.relu(self.conv1(x))
        h2 = self.relu(self.conv2(h))
        h3 = self.relu(self.conv3(h2))

        flat = h3.reshape(h3.shape[0], -1)

        if self.fc1 is None:
            self._build_head(flat.shape[1], flat.device)

        h4 = self.relu(self.fc1(flat))
        pi_logits = self.pi(h4)
        vf = self.vf(h4)[:, 0]

        return pi_logits, vf


def build_goexplore_cnn_policy() -> nn.Module:
    """Tiny real CnnPolicy: 4-channel (stacked-frame) 84x84 Atari-style
    observation, 6 discrete actions (a common Atari action-space size, e.g.
    Pitfall/Montezuma's Revenge subsets used by the repo's `run_*` scripts)."""
    model = GoExploreCnnPolicy(in_channels=4, n_actions=6)
    # Materialize the lazily-built FC/head layers with one dry forward so the
    # returned module has a complete, traceable parameter set before tracing.
    with torch.no_grad():
        model(torch.randint(0, 256, (1, 4, 84, 84)).float())
    model.eval()
    return model


def example_input_goexplore_cnn_policy():
    return torch.randint(0, 256, (1, 4, 84, 84)).float()


MENAGERIE_ZOO = "ported-pytorch"
MENAGERIE_ENTRIES = [
    (
        "Go-Explore CnnPolicy (Phase 2 robustification policy)",
        build_goexplore_cnn_policy,
        example_input_goexplore_cnn_policy,
        2019,
        MENAGERIE_ZOO,
    ),
]
