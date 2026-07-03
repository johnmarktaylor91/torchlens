# SOURCE: vendored from cedriclmenard/irislandmarks.pytorch @ master (16f91b38)
# https://raw.githubusercontent.com/cedriclmenard/irislandmarks.pytorch/master/irislandmarks.py
"""MediaPipe Iris landmark model (PyTorch port).

Vendored real nn.Module code with only import/formatting touch-ups (no
architectural changes). Predicts 71 eye-contour landmarks and 5 iris
landmarks from a 64x64 eye crop.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class IrisBlock(nn.Module):
    """This is the main building block for architecture"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super(IrisBlock, self).__init__()

        self.stride = stride
        self.channel_pad = out_channels - in_channels

        padding = (kernel_size - 1) // 2
        if stride == 2:
            self.max_pool = nn.MaxPool2d(kernel_size=stride, stride=stride)

        self.convAct = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=int(out_channels / 2),
                kernel_size=stride,
                stride=stride,
                padding=0,
                bias=True,
            ),
            nn.PReLU(int(out_channels / 2)),
        )
        self.dwConvConv = nn.Sequential(
            nn.Conv2d(
                in_channels=int(out_channels / 2),
                out_channels=int(out_channels / 2),
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                groups=int(out_channels / 2),
                bias=True,
            ),
            nn.Conv2d(
                in_channels=int(out_channels / 2),
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
        )

        self.act = nn.PReLU(out_channels)

    def forward(self, x):
        h = self.convAct(x)
        if self.stride == 2:
            x = self.max_pool(x)

        h = self.dwConvConv(h)

        if self.channel_pad > 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.channel_pad), "constant", 0)

        return self.act(h + x)


class IrisLandmarks(nn.Module):
    """The IrisLandmark face landmark model from MediaPipe.

    Because we won't be training this model, it doesn't need to have
    batchnorm layers. These have already been "folded" into the conv
    weights by TFLite.
    """

    def __init__(self):
        super(IrisLandmarks, self).__init__()

        self.min_score_thresh = 0.75

        self._define_layers()

    def _define_layers(self):
        self.backbone = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=0, bias=True
            ),
            nn.PReLU(64),
            IrisBlock(64, 64),
            IrisBlock(64, 64),
            IrisBlock(64, 64),
            IrisBlock(64, 64),
            IrisBlock(64, 128, stride=2),
            IrisBlock(128, 128),
            IrisBlock(128, 128),
            IrisBlock(128, 128),
            IrisBlock(128, 128),
            IrisBlock(128, 128, stride=2),
        )
        self.split_eye = nn.Sequential(
            IrisBlock(128, 128),
            IrisBlock(128, 128),
            IrisBlock(128, 128, stride=2),
            IrisBlock(128, 128),
            IrisBlock(128, 128),
            IrisBlock(128, 128, stride=2),
            IrisBlock(128, 128),
            IrisBlock(128, 128),
            nn.Conv2d(
                in_channels=128, out_channels=213, kernel_size=2, stride=1, padding=0, bias=True
            ),
        )
        self.split_iris = nn.Sequential(
            IrisBlock(128, 128),
            IrisBlock(128, 128),
            IrisBlock(128, 128, stride=2),
            IrisBlock(128, 128),
            IrisBlock(128, 128),
            IrisBlock(128, 128, stride=2),
            IrisBlock(128, 128),
            IrisBlock(128, 128),
            nn.Conv2d(
                in_channels=128, out_channels=15, kernel_size=2, stride=1, padding=0, bias=True
            ),
        )

    def forward(self, x):
        # TFLite uses slightly different padding on the first conv layer
        # than PyTorch, so do it manually.
        x = F.pad(x, [0, 1, 0, 1], "constant", 0)
        b = x.shape[0]  # batch size, needed for reshaping later

        x = self.backbone(x)  # (b, 128, 8, 8)

        e = self.split_eye(x)  # (b, 213, 1, 1)
        e = e.view(b, -1)  # (b, 213)

        i = self.split_iris(x)  # (b, 15, 1, 1)
        i = i.reshape(b, -1)  # (b, 15)

        return [e, i]


def build_iris_landmarks():
    return IrisLandmarks()


def example_input_iris_landmarks():
    return torch.randn(1, 3, 64, 64)


MENAGERIE_ZOO = "vendored-pytorch"

MENAGERIE_ENTRIES = [
    ("MediaPipe Iris", "build_iris_landmarks", "example_input_iris_landmarks", 2020, "vendored"),
]
