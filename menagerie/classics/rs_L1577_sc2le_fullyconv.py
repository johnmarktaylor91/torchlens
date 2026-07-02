# FAITHFUL PORT of https://github.com/tuomaso/SC2LE-implementation @ master
# (agents/network.py: build_fcn, lines 79-180; original framework: TensorFlow 1.x via
#  tensorflow.contrib.layers)
#
# FullyConv agent for the StarCraft II Learning Environment minigames (Vinyals et al.
# 2017, "StarCraft II: A New Challenge for Reinforcement Learning", arXiv:1708.04782,
# Section 4.3 "Convolutional Agents", the FullyConv architecture). tuomaso's
# SC2LE-implementation is a direct, faithful reimplementation of the paper's FullyConv
# agent, itself adapted from xhujoy/pysc2-agents and updated for pysc2 v2. Its
# `network.build_fcn` is TF1.x (`tensorflow.contrib.layers`, unmaintained/unrunnable in
# a modern env), so this is a line-by-line PORT to torch of that exact function: two
# parallel 2-layer conv towers over minimap/screen spatial features, a non-spatial
# `info` (available-actions) feature vector broadcast-projected back into the spatial
# map, concatenated into one spatial feature map that yields the spatial (x,y) action
# location via a 1x1 conv + softmax, and a separate fully-connected trunk (from the
# flattened conv features + info) that yields the non-spatial action-type/argument
# heads (action_choice, queued, control_group_act/id, select_point_act, select_add,
# select_unit_act/id, select_worker, build_queue_id, unload_id) plus a scalar value
# head -- every head from the real `build_fcn` is kept, 1:1, with the same
# `num_outputs` per head as pysc2's actual argument-space sizes for these minigame
# argument types (see pysc2.lib.actions.TYPES for the canonical per-argument sizes
# tuomaso's fully-populated network uses; minimap/screen channel counts follow
# pysc2.lib.features.MINIMAP_FEATURES/SCREEN_FEATURES from the pysc2 v2 the repo
# targets).
#
# TF -> torch translation notes (mechanical, no architecture change):
#   - `layers.conv2d(..., activation_fn=default relu)` -> `nn.Conv2d` + `F.relu`
#     (TF-Slim's `conv2d` defaults to ReLU when `activation_fn` is left unset, as in
#     every conv call here).
#   - `layers.fully_connected` -> `nn.Linear` with the same explicit activation_fn
#     (`tanh`, `relu`, `softmax`, or `None` per call site).
#   - `tf.transpose(x, [0,2,3,1])` (NCHW->NHWC for TF's channel-last conv2d) is DROPPED
#     entirely: torch's `nn.Conv2d` is natively NCHW, so the input is used as-is.
#   - `layers.flatten` -> `torch.flatten(x, start_dim=1)`.
#   - `tf.reshape(info_projection, [-1, 64, 64, 1])` -> the torch equivalent is a
#     channels-first reshape `[-1, 1, msize, msize]` (kept general via `msize`/`ssize`
#     rather than the hardcoded `64` in the original, which only ever ran at 64x64).
#   - `SAME` padding is TF-Slim's `conv2d` default; ported as explicit `padding=k//2`
#     for the odd kernel sizes used here (5 and 3), which is exactly SAME for stride 1.

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "ported-pytorch"


# ---- agents/network.py: build_fcn, ported layer-for-layer ----
class SC2LEFullyConvNet(nn.Module):
    def __init__(self, minimap_channels, screen_channels, info_size, msize, ssize, num_action):
        super().__init__()
        self.msize = msize
        self.ssize = ssize

        # Extract features (mconv1/mconv2/sconv1/sconv2 in the original)
        self.mconv1 = nn.Conv2d(minimap_channels, 16, kernel_size=5, stride=1, padding=2)
        self.mconv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.sconv1 = nn.Conv2d(screen_channels, 16, kernel_size=5, stride=1, padding=2)
        self.sconv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)

        self.info_fc = nn.Linear(info_size, 256)

        # "made this a fully connected since details aren't very clear on original paper"
        self.info_projection = nn.Linear(info_size, ssize * ssize)

        # Compute spatial actions: 1x1 conv over concat(mconv2, sconv2, info_projection)
        self.spatial_choice_conv = nn.Conv2d(32 + 32 + 1, 1, kernel_size=1, stride=1)

        # Compute non spatial actions and value from flattened conv features + info_fc
        feat_fc_in = 32 * msize * msize + 32 * ssize * ssize + 256
        self.feat_fc = nn.Linear(feat_fc_in, 256)

        self.action_choice = nn.Linear(256, num_action)
        self.queued = nn.Linear(256, 2)
        self.control_group_act = nn.Linear(256, 5)
        self.control_group_id = nn.Linear(256, 10)
        self.select_point_act = nn.Linear(256, 4)
        self.select_add = nn.Linear(256, 2)
        self.select_unit_act = nn.Linear(256, 4)
        self.select_unit_id = nn.Linear(256, 500)
        self.select_worker = nn.Linear(256, 4)
        self.build_queue_id = nn.Linear(256, 10)
        self.unload_id = nn.Linear(256, 500)
        self.value = nn.Linear(256, 1)

    def forward(self, minimap, screen, info):
        # Extract features (channels-first NCHW throughout; the original's
        # tf.transpose(...,[0,2,3,1]) is dropped, see module docstring above)
        mconv1 = F.relu(self.mconv1(minimap))
        mconv2 = F.relu(self.mconv2(mconv1))
        sconv1 = F.relu(self.sconv1(screen))
        sconv2 = F.relu(self.sconv2(sconv1))

        info_flat = torch.flatten(info, start_dim=1)
        info_fc = torch.tanh(self.info_fc(info_flat))
        info_projection = torch.tanh(self.info_projection(info_flat))

        # Compute spatial actions
        info_proj_map = info_projection.view(-1, 1, self.ssize, self.ssize)
        feat_conv = torch.cat([mconv2, sconv2, info_proj_map], dim=1)
        spatial_choice = self.spatial_choice_conv(feat_conv)
        outputs = {}
        outputs["spatial_choice"] = F.softmax(torch.flatten(spatial_choice, start_dim=1), dim=-1)

        # Compute non spatial actions and value
        feat_fc_in = torch.cat(
            [
                torch.flatten(mconv2, start_dim=1),
                torch.flatten(sconv2, start_dim=1),
                info_fc,
            ],
            dim=1,
        )
        feat_fc = F.relu(self.feat_fc(feat_fc_in))

        outputs["action_choice"] = F.softmax(self.action_choice(feat_fc), dim=-1)
        outputs["queued"] = F.softmax(self.queued(feat_fc), dim=-1)
        outputs["control_group_act"] = F.softmax(self.control_group_act(feat_fc), dim=-1)
        outputs["control_group_id"] = F.softmax(self.control_group_id(feat_fc), dim=-1)
        outputs["select_point_act"] = F.softmax(self.select_point_act(feat_fc), dim=-1)
        outputs["select_add"] = F.softmax(self.select_add(feat_fc), dim=-1)
        outputs["select_unit_act"] = F.softmax(self.select_unit_act(feat_fc), dim=-1)
        outputs["select_unit_id"] = F.softmax(self.select_unit_id(feat_fc), dim=-1)
        outputs["select_worker"] = F.softmax(self.select_worker(feat_fc), dim=-1)
        outputs["build_queue_id"] = F.softmax(self.build_queue_id(feat_fc), dim=-1)
        outputs["unload_id"] = F.softmax(self.unload_id(feat_fc), dim=-1)
        outputs["value"] = self.value(feat_fc).view(-1)

        return outputs


# ---- staging wrapper ----
def build_sc2le_fullyconv():
    # pysc2 v2 minigame defaults: 64x64 minimap/screen; MINIMAP_FEATURES has 7
    # channels and SCREEN_FEATURES has 17 channels as of pysc2 v2 (features.py);
    # `info` is the one-hot available-actions vector, sized by len(actions.FUNCTIONS)
    # (524 in modern pysc2; kept small here for a fast trace since it is only ever
    # flattened/projected, never convolved).
    return SC2LEFullyConvNet(
        minimap_channels=7, screen_channels=17, info_size=64, msize=64, ssize=64, num_action=64
    )


def example_input_sc2le_fullyconv():
    torch.manual_seed(0)
    batch = 2
    minimap = torch.rand(batch, 7, 64, 64)
    screen = torch.rand(batch, 17, 64, 64)
    info = torch.zeros(batch, 64)
    info[:, :5] = 1.0
    return (minimap, screen, info)


MENAGERIE_ENTRIES = [
    (
        "SC2LE_FullyConv",
        "build_sc2le_fullyconv",
        "example_input_sc2le_fullyconv",
        2017,
        "ported-pytorch",
    ),
]
