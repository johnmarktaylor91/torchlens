# FAITHFUL PORT of TerrisGO/PilotNet @ master (original framework: TensorFlow 1.x)
#
# PilotNet (Bojarski et al., NVIDIA 2016 "End to End Learning for Self-Driving Cars"). The
# named repo (a fork of lhzlhz/PilotNet, itself sourced from SullyChen/Autopilot-TensorFlow)
# implements the network in raw TF1 graph-mode with tf.placeholder/tf.Variable and python2
# print syntax (src/nets/pilotNet.py, src/nets/model_nvidia.py) -- it cannot run in the
# torch-only base env. Every layer below is transcribed 1:1 from that TF1 code:
#   5 conv layers, exact kernel/stride/channel counts from pilotNet.py:
#     conv1: 5x5, stride 2, 3->24 channels, VALID padding, ReLU
#     conv2: 5x5, stride 2, 24->36 channels, VALID padding, ReLU
#     conv3: 5x5, stride 2, 36->48 channels, VALID padding, ReLU
#     conv4: 3x3, stride 1, 48->64 channels, VALID padding, ReLU
#     conv5: 3x3, stride 1, 64->64 channels, VALID padding, ReLU
#   flatten to 1152 (yielded by the 66x200x3 input geometry) -> 5 FC layers with dropout:
#     fc1: 1152->1164, ReLU, dropout
#     fc2: 1164->100, ReLU, dropout
#     fc3: 100->50, ReLU, dropout
#     fc4: 50->10, ReLU, dropout
#     fc5 (output): 10->1, then steering = atan(fc5_out) * 2  (matches
#       `y = tf.multiply(tf.atan(tf.matmul(...) + b_fc5), 2)` in the source exactly)
# Weight init in the source is `tf.truncated_normal(stddev=0.1)` / `tf.constant(0.1)` bias --
# reproduced here via explicit truncated-normal/constant init in __init__ rather than torch's
# default Kaiming init, to stay faithful to the original initialization scheme.
#
# MENAGERIE_ZOO = "ported-pytorch"

import torch
import torch.nn as nn


class PilotNet(nn.Module):
    """Faithful torch port of the TF1 PilotNet graph in src/nets/pilotNet.py."""

    def __init__(self, keep_prob=0.8):
        super().__init__()
        self.keep_prob = keep_prob

        # first convolutional layer: 5x5, 3->24, stride 2, VALID
        self.conv1 = nn.Conv2d(3, 24, kernel_size=5, stride=2, padding=0)
        # second convolutional layer: 5x5, 24->36, stride 2, VALID
        self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2, padding=0)
        # third convolutional layer: 5x5, 36->48, stride 2, VALID
        self.conv3 = nn.Conv2d(36, 48, kernel_size=5, stride=2, padding=0)
        # fourth convolutional layer: 3x3, 48->64, stride 1, VALID
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=0)
        # fifth convolutional layer: 3x3, 64->64, stride 1, VALID
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=1.0 - keep_prob)

        # FCL 1: 1152 -> 1164 (1152 = flattened 5th conv output for a 66x200x3 input)
        self.fc1 = nn.Linear(1152, 1164)
        # FCL 2: 1164 -> 100
        self.fc2 = nn.Linear(1164, 100)
        # FCL 3: 100 -> 50
        self.fc3 = nn.Linear(100, 50)
        # FCL 4: 50 -> 10
        self.fc4 = nn.Linear(50, 10)
        # output: 10 -> 1
        self.fc5 = nn.Linear(10, 1)

        self._init_weights()

    def _init_weights(self):
        # Matches the source's weight_variable (truncated_normal, stddev=0.1) /
        # bias_variable (constant 0.1) initializers.
        for layer in (
            self.conv1,
            self.conv2,
            self.conv3,
            self.conv4,
            self.conv5,
            self.fc1,
            self.fc2,
            self.fc3,
            self.fc4,
            self.fc5,
        ):
            nn.init.trunc_normal_(layer.weight, std=0.1)
            nn.init.constant_(layer.bias, 0.1)

    def forward(self, image_input):
        h_conv1 = self.relu(self.conv1(image_input))
        h_conv2 = self.relu(self.conv2(h_conv1))
        h_conv3 = self.relu(self.conv3(h_conv2))
        h_conv4 = self.relu(self.conv4(h_conv3))
        h_conv5 = self.relu(self.conv5(h_conv4))

        h_conv5_flat = h_conv5.reshape(h_conv5.size(0), -1)

        h_fc1 = self.relu(self.fc1(h_conv5_flat))
        h_fc1_drop = self.dropout(h_fc1)

        h_fc2 = self.relu(self.fc2(h_fc1_drop))
        h_fc2_drop = self.dropout(h_fc2)

        h_fc3 = self.relu(self.fc3(h_fc2_drop))
        h_fc3_drop = self.dropout(h_fc3)

        h_fc4 = self.relu(self.fc4(h_fc3_drop))
        h_fc4_drop = self.dropout(h_fc4)

        # scale the atan output (matches `tf.multiply(tf.atan(...), 2)` exactly)
        steering = torch.multiply(torch.atan(self.fc5(h_fc4_drop)), 2)
        return steering


# ---------------------------------------------------------------------------
# menagerie staging entry points
# ---------------------------------------------------------------------------
MENAGERIE_ZOO = "ported-pytorch"


def build_pilotnet():
    model = PilotNet()
    model.eval()
    return model


def example_input_pilotnet():
    # NHWC 66x200x3 in the source (tf.placeholder shape=[None, 66, 200, 3]); torch Conv2d
    # is NCHW, so the example input is permuted accordingly.
    batch = 2
    return torch.randn(batch, 3, 66, 200)


MENAGERIE_ENTRIES = [
    ("PilotNet", "build_pilotnet", "example_input_pilotnet", 2016, MENAGERIE_ZOO),
]
