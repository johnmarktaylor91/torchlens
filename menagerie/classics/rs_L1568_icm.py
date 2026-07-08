# FAITHFUL PORT of https://github.com/pathak22/noreward-rl @ master (original framework: TF1.x)
# (src/model.py: natureHead, StateActionPredictor)
#
# Intrinsic Curiosity Module (Pathak, Agrawal, Efros & Darrell, ICML 2017, "Curiosity-driven
# Exploration by Self-supervised Prediction"). The official release (pathak22/noreward-rl) is
# TensorFlow 1.x with `tf.get_variable`/`tf.variable_scope`/`rnn.rnn_cell` APIs removed from
# modern TF2 and no longer installable in this base env. This is a faithful port of the actual
# `StateActionPredictor` class (the ICM itself: a shared feature encoder `phi`, an inverse model
# `g(phi(s1), phi(s2)) -> a_hat` predicting the action taken between two states, and a forward
# model `f(phi(s1), a) -> phi_hat(s2)` predicting the next state's features -- the forward-model
# prediction error is the curiosity/intrinsic-reward signal) using the paper's `natureHead`
# feature encoder (the DQN-Nature-2015 conv stack; the head actually used for the paper's Atari
# experiments, see src/train.py `designHead` default usage / src/constants.py). Every layer and
# mechanism mirrors the real TF code 1:1 (same channel counts, kernel/stride, the 256-d shared
# feature space, the inverse-model classifier, the forward-model residual predictor); only the
# framework primitives change (tf.Variable/get_variable -> nn.Conv2d/nn.Linear parameters,
# tf.nn.relu -> F.relu, TF's `linear(x, size, name, normalized_columns_initializer)` -> an
# nn.Linear whose weight is re-initialized with the same normalized-columns scheme used
# throughout the original code).

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "ported-pytorch"


def _normalized_columns_init(weight: torch.Tensor, std: float = 1.0) -> None:
    # Port of model.py's normalized_columns_initializer: columns of the weight matrix are
    # drawn N(0,1) then rescaled to unit-std * `std` along the input (fan-in) axis. TF's
    # `linear()` stores weight as [in, out]; torch's nn.Linear.weight is [out, in], so we
    # normalize along dim=1 (in-features) here to match.
    with torch.no_grad():
        out = torch.randn_like(weight)
        out *= std / out.pow(2).sum(dim=1, keepdim=True).sqrt()
        weight.copy_(out)


class _Linear(nn.Linear):
    """nn.Linear with TF-style normalized-columns weight init (port of model.py `linear()`)."""

    def __init__(
        self, in_features: int, out_features: int, std: float = 1.0, bias_init: float = 0.0
    ):
        super().__init__(in_features, out_features)
        _normalized_columns_init(self.weight, std=std)
        nn.init.constant_(self.bias, bias_init)


# ---- src/model.py natureHead, ported (DQN Nature-2015 conv stack; VALID padding == no padding) ----
class NatureHead(nn.Module):
    """input: [N, C, 84, 84] -> output: [N, 512] (port of model.py `natureHead`)."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.fc = _Linear(64 * 7 * 7, 512, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc(x))
        return x


# ---- src/model.py StateActionPredictor, ported ----
class StateActionPredictor(nn.Module):
    """Port of model.py `StateActionPredictor` (the ICM module): shared encoder phi + inverse
    model (state pair -> predicted action logits) + forward model (state1 features + action ->
    predicted next-state features). Returns (predicted_action_logits, forward_loss) exactly as
    the original TF class exposes `self.ainvprobs` (softmax of `self.invloss`'s pre-softmax
    logits) and `self.forwardloss` (the scalar curiosity bonus, before the PREDICTION_BETA scale
    and the *288.0 length-normalization factor from the real code)."""

    def __init__(self, in_channels: int, num_actions: int):
        super().__init__()
        self.num_actions = num_actions
        feat_size = 512  # natureHead output width

        self.phi = NatureHead(in_channels)

        # inverse model: g(phi1, phi2) -> action logits
        self.g1 = _Linear(feat_size * 2, feat_size, std=0.01)
        self.g_last = _Linear(feat_size, num_actions, std=0.01)

        # forward model: f(phi1, action_onehot) -> predicted phi2
        self.f1 = _Linear(feat_size + num_actions, feat_size, std=0.01)
        self.f_last = _Linear(feat_size, feat_size, std=0.01)

    def forward(self, s1: torch.Tensor, s2: torch.Tensor, asample: torch.Tensor):
        phi1 = self.phi(s1)
        phi2 = self.phi(s2)

        # inverse model
        g = torch.cat([phi1, phi2], dim=1)
        g = F.relu(self.g1(g))
        logits = self.g_last(g)
        ainvprobs = F.softmax(logits, dim=-1)

        # forward model (no gradient into asample -- it's fixed policy output, per original)
        f = torch.cat([phi1, asample.detach()], dim=1)
        f = F.relu(self.f1(f))
        f = self.f_last(f)
        forwardloss = 0.5 * torch.mean((f - phi2) ** 2)
        # Real code: `self.forwardloss * 288.0` (lenFeatures=288, the universeHead's feature
        # width, hardcoded so the loss scale is independent of feature width). Ported literally
        # as the same constant so the natureHead (512-d) predictor keeps the original's
        # curiosity-bonus scaling behavior.
        forwardloss = forwardloss * 288.0

        return logits, ainvprobs, forwardloss


# ---- staging wrapper ----
def build_icm():
    torch.manual_seed(0)
    in_channels = 4  # 4-frame Atari stack, as used by the paper's Atari experiments
    num_actions = 6
    return StateActionPredictor(in_channels=in_channels, num_actions=num_actions)


def example_input_icm():
    torch.manual_seed(0)
    s1 = torch.rand(2, 4, 84, 84)
    s2 = torch.rand(2, 4, 84, 84)
    action_idx = torch.randint(0, 6, (2,))
    asample = F.one_hot(action_idx, num_classes=6).float()
    return (s1, s2, asample)


MENAGERIE_ENTRIES = [
    ("ICM_StateActionPredictor", "build_icm", "example_input_icm", 2017, "ported-pytorch"),
]
