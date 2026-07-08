# SOURCE: vendored from ssg-research/FLARE @ master
# https://github.com/ssg-research/FLARE
# File: src/agents/action_conditional_video_prediction.py (class `Network`)
# License: Apache-2.0. That file itself is adapted from wuyx/DeepRL_pytorch
# (credited in its header) as a PyTorch reimplementation of Oh, Guo, Lee, Lewis,
# Singh (NeurIPS 2015) "Action-Conditional Video Prediction using Deep Networks
# in Atari Games". The official repo (junhyukoh/nips2015-action-conditional-video-
# prediction) ships only Caffe/Torch7 code; this is the real, independently
# published PyTorch port of the paper's feedforward "action-conditional encoder"
# architecture (conv encoder -> multiplicative action gating -> deconv decoder),
# used here (as in FLARE) as the visual-foresight next-frame predictor module.
"""Action-Conditional Video Prediction network (Oh et al., NeurIPS 2015).

Convolutional encoder produces a latent code; the latent is multiplicatively
gated by an embedded action vector (the paper's core "action-conditional"
mechanism), then a deconvolutional decoder reconstructs the predicted next
frame difference.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionConditionalVideoPredictionNet(nn.Module):
    """Encoder-action-decoder next-frame predictor (verbatim `Network` from FLARE)."""

    def __init__(self, num_actions: int = 6) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, 8, 2, (1, 1))
        self.conv2 = nn.Conv2d(64, 128, 6, 2, (1, 1))
        self.conv3 = nn.Conv2d(128, 128, 7, 2, (1, 1))
        self.conv4 = nn.Conv2d(128, 128, 4, 2, (0, 0))

        self.hidden_units = 128 * 3 * 3

        self.fc5 = nn.Linear(self.hidden_units, 2048)
        self.fc_encode = nn.Linear(2048, 2048)
        self.fc_action = nn.Linear(num_actions, 2048)
        self.fc_decode = nn.Linear(2048, 2048)
        self.fc8 = nn.Linear(2048, self.hidden_units)

        self.deconv9 = nn.ConvTranspose2d(128, 128, 4, 2, (0, 0))
        self.deconv10 = nn.ConvTranspose2d(128, 128, 7, 2, (1, 1))
        self.deconv11 = nn.ConvTranspose2d(128, 64, 6, 2, (1, 1))
        self.deconv12 = nn.ConvTranspose2d(64, 1, 8, 2, (1, 1))

        self._init_weights()

    def _init_weights(self) -> None:
        for layer in self.children():
            if isinstance(layer, nn.Conv2d | nn.ConvTranspose2d):
                nn.init.xavier_uniform_(layer.weight.data)
            nn.init.constant_(layer.bias.data, 0)
        nn.init.uniform_(self.fc_encode.weight.data, -1, 1)
        nn.init.uniform_(self.fc_decode.weight.data, -1, 1)
        nn.init.uniform_(self.fc_action.weight.data, -0.1, 0.1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(obs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view((-1, self.hidden_units))
        x = F.relu(self.fc5(x))
        x = self.fc_encode(x)
        action_embed = self.fc_action(action)
        x = torch.mul(x, action_embed)
        x = self.fc_decode(x)
        x = F.relu(self.fc8(x))
        x = x.view((-1, 128, 3, 3))
        x = F.relu(self.deconv9(x))
        x = F.relu(self.deconv10(x))
        x = F.relu(self.deconv11(x))
        x = self.deconv12(x)
        return x


class _ACVPTraceWrapper(nn.Module):
    """Bundles `(obs, one_hot_action)` into a single-tensor-friendly forward call."""

    def __init__(self, num_actions: int = 6) -> None:
        super().__init__()
        self.net = ActionConditionalVideoPredictionNet(num_actions=num_actions)
        self.num_actions = num_actions

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(obs, action)


def build_action_cond_video_pred() -> nn.Module:
    """Build the action-conditional video-prediction network."""

    return _ACVPTraceWrapper(num_actions=6)


def example_input_action_cond_video_pred() -> tuple[torch.Tensor, torch.Tensor]:
    """Return an example `(obs, one_hot_action)` pair.

    obs: a single RGB Atari-style frame at the network's native 84x84 input
    resolution (chosen so the conv/deconv stack round-trips to 3x3 as in the
    original architecture). action: a one-hot action vector over 6 actions.
    """

    obs = torch.randn(1, 3, 84, 84)
    action = F.one_hot(torch.tensor([0]), num_classes=6).float()
    return obs, action


MENAGERIE_ENTRIES = [
    (
        "Action-Conditional Video Prediction (Oh et al. 2015, encoder-action-decoder)",
        "build_action_cond_video_pred",
        "example_input_action_cond_video_pred",
        "2015",
        "DC",
    ),
]
