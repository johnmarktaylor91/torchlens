# SOURCE: vendored from anir16293/Deep-Lesion @ 670c21e3eb4c65b0dab902db78b321249d69192e
# https://raw.githubusercontent.com/anir16293/Deep-Lesion/670c21e3eb4c65b0dab902db78b321249d69192e/panet.py
#
# JHU "Deep Learning" course project (Spring 2019) CAD network for the
# NIH DeepLesion CT dataset. The repo README states verbatim: "Our pytorch
# implementations of PANet and EncoderNet are present in panet.py." `PanNet`
# is a Path Aggregation Network (PANet)-style lesion-bounding-box regressor:
# a plain conv/maxpool encoder (`conv1`..`conv4`) producing a feature
# pyramid, a top-down FPN upsample path (`conv5`..`conv7`, transposed-conv
# upsample + skip concatenation) that IS the "Faster_RCNN.png"/FPN diagram in
# the repo, then PANet's signature bottom-up path augmentation
# (`conv8_init`/`conv8` + maxpool + concat back down the pyramid) feeding a
# `ChannelPool` (channel-wise max-pool, subclassing `nn.MaxPool1d`) at three
# pyramid levels, concatenated and passed through a 3-layer box-regression
# MLP (`fc1`/`fc2`/`fc3`) to a 4-value bounding box. This encoder + FPN +
# PANet bottom-up-augmentation + box-head design is the architectural
# contribution of the project (not a stock torchvision detector), so it is
# vendored rather than constructed from a base-library class.
#
# `ChannelPool`, `upsample`, and `PanNet` are the real, unmodified classes/
# functions from `panet.py` (layer composition and forward-pass control flow
# are byte-for-byte the original; only the dead `EncoderNet`/
# `autoencoder_improved` classes and the commented-out box-decoding lines at
# the end of `forward` were not needed and are omitted here since they are
# not part of `PanNet`'s architecture).
#   - Added `build_deeplesion_panet()`/`example_input_deeplesion_panet()`
#     staging entry points. `PanNet`'s linear layers hardcode the pyramid
#     feature-map sizes (`128*128`, `64*64`, `32*32`), which pins the input
#     spatial size to exactly 512x512 (4 stride-2 maxpools: 512 -> 256 -> 128
#     -> 64 -> 32, matching `linear8`/`linear9`/`linear10`'s in_features);
#     batch_size=2, 3-channel CT-window input, matching the original
#     `forward(x)` single-tensor-input call signature.

import numpy as np
import torch
from torch import nn
import sys
import torch.nn.functional as F


class ChannelPool(nn.MaxPool1d):
    def forward(self, input):
        n, c, w, h = input.size()
        input = input.view(n, c, w * h).permute(0, 2, 1)
        pooled = F.max_pool1d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.ceil_mode,
            self.return_indices,
        )
        _, _, c = pooled.size()
        pooled = pooled.permute(0, 2, 1)
        return pooled.view(n, c, w, h)


def upsample(ch_coarse, ch_fine):
    return nn.Sequential(
        nn.ConvTranspose2d(ch_coarse, ch_fine, kernel_size=4, stride=2, padding=1, bias=True),
        nn.ReLU(),
    )


class PanNet(nn.Module):
    def __init__(self):
        super(PanNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=2, padding=0)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv5 = upsample(64, 64)
        self.conv6 = upsample(128, 64)
        self.conv7 = upsample(128, 64)

        self.conv8_init = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1
        )
        self.conv8 = nn.Conv2d(
            in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1
        )
        # self.conv8_final = nn.Conv2d(in_channels = 256, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        self.batch_norm64 = nn.BatchNorm2d(64)
        self.batch_norm128 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU()

        self.linear8 = nn.Sequential(
            nn.Linear(in_features=1 * 128 * 128, out_features=1024), nn.BatchNorm2d(1024)
        )
        self.linear9 = nn.Sequential(
            nn.Linear(in_features=1 * 64 * 64, out_features=512), nn.BatchNorm2d(512)
        )
        self.linear10 = nn.Sequential(
            nn.Linear(in_features=1 * 32 * 32, out_features=256), nn.BatchNorm2d(256)
        )

        self.channel_pool256 = ChannelPool(256)
        self.channel_pool192 = ChannelPool(192)

        self.fc1 = nn.Sequential(
            nn.Linear(in_features=128 * 128 + 64 * 64 + 32 * 32, out_features=1024)
        )
        self.fc2 = nn.Sequential(nn.Linear(in_features=1024, out_features=32))
        self.fc3 = nn.Sequential(nn.Linear(in_features=32, out_features=4))

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # FPN
        out1 = self.conv1(x)
        out1 = self.batch_norm64(out1)
        out1 = self.max_pool(out1)  # 256
        out1 = self.relu(out1)

        out2 = self.conv2(out1)
        out2 = self.batch_norm64(out2)
        out2 = self.relu(out2)
        out2 = self.max_pool(out2)  # 128

        out3 = self.conv3(out2)
        out3 = self.batch_norm64(out3)
        out3 = self.relu(out3)
        out3 = self.max_pool(out3)  # 64

        out4 = self.conv4(out3)
        out4 = self.batch_norm64(out4)
        out4 = self.relu(out4)
        out4 = self.max_pool(out4)  # 32

        out5 = self.conv5(out4)
        out5 = self.batch_norm64(out5)
        out5 = self.relu(out5)
        out5 = torch.cat((out3, out5), dim=1)  # 64

        out6 = self.conv6(out5)
        out6 = self.batch_norm64(out6)
        out6 = self.relu(out6)
        out6 = torch.cat((out2, out6), dim=1)  # 128

        out7 = self.conv7(out6)
        out7 = self.batch_norm64(out7)
        out7 = self.relu(out7)
        out7 = torch.cat((out1, out7), dim=1)  # 256

        # Bottom up path augmentation
        out8 = self.conv8_init(out7)
        out8 = self.batch_norm128(out8)
        out8 = self.relu(out8)
        out8 = self.max_pool(out8)
        out8 = torch.cat((out6, out8), dim=1)  # 128
        out8_max = self.channel_pool256(out8)
        out8_max = out8_max.view(-1, 1, 128 * 128)

        out9 = self.conv8(out8)
        out9 = self.batch_norm128(out9)
        out9 = self.relu(out9)
        out9 = self.max_pool(out9)
        out9 = torch.cat((out5, out9), dim=1)  # 64
        out9_max = self.channel_pool256(out9)
        out9_max = out9_max.view(-1, 1, 64 * 64)

        out10 = self.conv8(out9)
        out10 = self.batch_norm128(out10)
        out10 = self.relu(out10)
        out10 = self.max_pool(out10)
        out10 = torch.cat((out4, out10), dim=1)  # 32
        out10_max = self.channel_pool192(out10)
        out10_max = out10_max.view(-1, 1, 32 * 32)

        # Adaptive pooling
        out_final = torch.cat((out8_max, out9_max, out10_max), dim=2)

        # Box regression network
        out_final = self.fc1(out_final)
        out_final = self.relu(out_final)
        out_final = self.fc2(out_final)
        out_final = self.relu(out_final)
        out_final = self.fc3(out_final)
        out_final = self.relu(out_final)
        return out_final


def build_deeplesion_panet():
    return PanNet()


def example_input_deeplesion_panet():
    # PanNet's linear/box-head layers pin the spatial size to exactly
    # 512x512 (4 stride-2 maxpools -> 32x32 deepest pyramid level, matching
    # linear10's in_features=32*32).
    return torch.randn(2, 3, 512, 512)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    (
        "DeepLesion-PanNet",
        "build_deeplesion_panet",
        "example_input_deeplesion_panet",
        2019,
        "vendored",
    ),
]
