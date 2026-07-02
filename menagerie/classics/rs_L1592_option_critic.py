# SOURCE: vendored from https://github.com/lweitkamp/option-critic-pytorch @ fab40f7aae0ff45cf5945b7de79d5ae5446d31a0
# (option_critic.py: OptionCriticConv, OptionCriticFeatures, lines 1-165)
#
# Option-Critic Architecture (Bacon, Harb & Precup, AAAI 2017,
# "The Option-Critic Architecture", arXiv:1609.05140). The paper's own
# reference code (jeanharb/option_critic) is Python-2 Theano/Lasagne and is
# not runnable in a modern base env. lweitkamp/option-critic-pytorch is the
# well-known, widely-cited community PyTorch reimplementation ("mostly a
# rewriting of the original Theano code ... into PyTorch") and is the
# canonical PyTorch reference used across option-critic reimplementations.
# Both `OptionCriticConv` (Atari-style conv feature extractor, for pixel
# observations) and `OptionCriticFeatures` (flat-MLP feature extractor, for
# vector observations e.g. CartPole/four-rooms) have no dependency beyond
# torch/numpy, so they are vendored verbatim (the RNG-consuming
# `get_action`/`predict_option_termination`/`epsilon` sampling helpers and the
# `critic_loss`/`actor_loss` training losses are the file's non-forward-pass
# logic and are omitted here -- they are training-loop utilities, not part of
# the traced architecture). Architecture: shared CNN/MLP torso -> (1) `Q`
# linear head (policy-over-options value), (2) `terminations` linear head
# (per-option termination probability), (3) `options_W`/`options_b` a
# per-option bilinear intra-option policy tensor (num_options x hidden x
# num_actions) -- the defining architectural move of option-critic is this
# per-option action-logit tensor sitting alongside the shared torso.

import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


# ---- option_critic.py (vendored verbatim, forward-pass architecture only) ----
class OptionCriticConv(nn.Module):
    def __init__(
        self,
        in_features,
        num_actions,
        num_options,
        temperature=1.0,
        eps_start=1.0,
        eps_min=0.1,
        eps_decay=int(1e6),
        eps_test=0.05,
        device="cpu",
        testing=False,
    ):
        super(OptionCriticConv, self).__init__()

        self.in_channels = in_features
        self.num_actions = num_actions
        self.num_options = num_options
        self.magic_number = 7 * 7 * 64
        self.device = device
        self.testing = testing

        self.temperature = temperature
        self.eps_min = eps_min
        self.eps_start = eps_start
        self.eps_decay = eps_decay
        self.eps_test = eps_test
        self.num_steps = 0

        self.features = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.magic_number, 512),
            nn.ReLU(),
        )

        self.Q = nn.Linear(512, num_options)  # Policy-Over-Options
        self.terminations = nn.Linear(512, num_options)  # Option-Termination
        self.options_W = nn.Parameter(torch.zeros(num_options, 512, num_actions))
        self.options_b = nn.Parameter(torch.zeros(num_options, num_actions))

        self.to(device)
        self.train(not testing)

    def get_state(self, obs):
        if obs.ndim < 4:
            obs = obs.unsqueeze(0)
        obs = obs.to(self.device)
        state = self.features(obs)
        return state

    def get_Q(self, state):
        return self.Q(state)

    def get_terminations(self, state):
        return self.terminations(state).sigmoid()

    def forward(self, obs):
        # Staging forward: exercises the real torso + both heads + the
        # per-option bilinear action-logit tensor (option 0) in one pass.
        state = self.get_state(obs)
        q = self.get_Q(state)
        terminations = self.get_terminations(state)
        logits = state @ self.options_W[0] + self.options_b[0]
        action_probs = (logits / self.temperature).softmax(dim=-1)
        return q, terminations, action_probs


class OptionCriticFeatures(nn.Module):
    def __init__(
        self,
        in_features,
        num_actions,
        num_options,
        temperature=1.0,
        eps_start=1.0,
        eps_min=0.1,
        eps_decay=int(1e6),
        eps_test=0.05,
        device="cpu",
        testing=False,
    ):
        super(OptionCriticFeatures, self).__init__()

        self.in_features = in_features
        self.num_actions = num_actions
        self.num_options = num_options
        self.device = device
        self.testing = testing

        self.temperature = temperature
        self.eps_min = eps_min
        self.eps_start = eps_start
        self.eps_decay = eps_decay
        self.eps_test = eps_test
        self.num_steps = 0

        self.features = nn.Sequential(
            nn.Linear(in_features, 32), nn.ReLU(), nn.Linear(32, 64), nn.ReLU()
        )

        self.Q = nn.Linear(64, num_options)  # Policy-Over-Options
        self.terminations = nn.Linear(64, num_options)  # Option-Termination
        self.options_W = nn.Parameter(torch.zeros(num_options, 64, num_actions))
        self.options_b = nn.Parameter(torch.zeros(num_options, num_actions))

        self.to(device)
        self.train(not testing)

    def get_state(self, obs):
        if obs.ndim < 4:
            obs = obs.unsqueeze(0)
        obs = obs.to(self.device)
        state = self.features(obs)
        return state

    def get_Q(self, state):
        return self.Q(state)

    def get_terminations(self, state):
        return self.terminations(state).sigmoid()

    def forward(self, obs):
        state = self.get_state(obs)
        q = self.get_Q(state)
        terminations = self.get_terminations(state)
        logits = state @ self.options_W[0] + self.options_b[0]
        action_probs = (logits / self.temperature).softmax(dim=-1)
        return q, terminations, action_probs


# ---- end vendored option_critic.py ----


def build_option_critic_conv():
    return OptionCriticConv(in_features=4, num_actions=6, num_options=8)


def example_input_option_critic_conv():
    return (torch.randn(2, 4, 84, 84),)


def build_option_critic_features():
    return OptionCriticFeatures(in_features=8, num_actions=4, num_options=4)


def example_input_option_critic_features():
    return (torch.randn(2, 8),)


MENAGERIE_ENTRIES = [
    (
        "Option-Critic (Conv)",
        build_option_critic_conv,
        example_input_option_critic_conv,
        2017,
        "vendored-pytorch",
    ),
    (
        "Option-Critic (MLP)",
        build_option_critic_features,
        example_input_option_critic_features,
        2017,
        "vendored-pytorch",
    ),
]
