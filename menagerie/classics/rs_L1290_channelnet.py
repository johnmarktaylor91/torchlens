# SOURCE: vendored from rohsequ/Deep-Learning-Model-for-Channel-Estimation-PyTorch @ main
#
# https://github.com/rohsequ/Deep-Learning-Model-for-Channel-Estimation-PyTorch
# https://raw.githubusercontent.com/rohsequ/Deep-Learning-Model-for-Channel-Estimation-PyTorch/main/ChannelNet_PyTorch/pytorch_models.py
#
# ChannelNet (Soltani, Pourahmadi, Mirzaei, Sheikhzadeh, "Deep Learning-Based Channel
# Estimation", IEEE Communications Letters 2019, arXiv:1810.05893) -- this is a
# community PyTorch reimplementation of the original Keras/TF ChannelNet (the official
# repo is Mehran-Soltani/ChannelNet, Keras/TF -- see needs_env_L1219.tsv), matching the
# same two-stage architecture described in the paper: the OFDM pilot-grid channel
# response is treated as a low-resolution 2D image, first up-sampled/denoised by an
# SRCNN (image super-resolution net, He et al.-style 3-conv-layer SRCNN: 9x9 -> 3x3 ->
# 5x5) and then refined by a DnCNN (20-layer residual denoiser: Conv+ReLU stem, 18x
# Conv+BN+ReLU, final Conv, with a global residual `x - out` subtraction) exactly as in
# `ChannelNet_PyTorch/pytorch_models.py`. `SRCNN` and `DnCNN` are transcribed verbatim
# from the real repo file above; only the unused training helpers
# (`train_SRCNN`/`train_DnCNN`) and the top-level `os.mkdir(...)` side effect executed at
# import time in the original file are dropped (not part of the model architecture). A
# thin `ChannelNet` wrapper chains the two real modules SRCNN -> DnCNN, matching the
# cascade the real repo's `train_DnCNN` sets up (SRCNN output feeds DnCNN input) and the
# paper's SR-then-IR pipeline description.

import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# From ChannelNet_PyTorch/pytorch_models.py
# ---------------------------------------------------------------------------
class SRCNN(torch.nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()

        def init_weights(m):
            if type(m) == torch.nn.Conv2d:
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                m.bias.data.fill_(0.01)

        # L1 ImgIn shape=(?, 28, 28, 1)
        # # Conv -> (?, :, :, 64)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, kernel_size=9, stride=1, padding=4),
            torch.nn.ReLU(),
        )
        self.layer1.apply(init_weights)
        # L2 ImgIn shape=(?, :, :, 64)
        # Conv      ->(?, :, :, 32)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
        )
        self.layer2.apply(init_weights)
        # L3 ImgIn shape=(?, :, :, 32)
        # Conv ->(?, :, :, 1)
        self.layer3 = torch.nn.Conv2d(32, 1, kernel_size=5, stride=1, padding=2)
        torch.nn.init.kaiming_normal_(self.layer3.weight, nonlinearity="relu")
        self.layer3.bias.data.fill_(0.01)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        return out


class DnCNN(torch.nn.Module):
    def __init__(self):
        super(DnCNN, self).__init__()

        def init_weights(m):
            if type(m) == torch.nn.Conv2d:
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                m.bias.data.fill_(0.01)

        # L1 ImgIn shape=(?, 28, 28, 1)
        # Conv -> (?, :, :, 64)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2),
            torch.nn.ReLU(),
        )
        self.layer1.apply(init_weights)
        # L2 ImgIn shape=(?, :, :, 64)
        # Conv      ->(?, :, :, 64)
        # 18 layers, Conv2d + BN + ReLu
        layers = []
        for i in range(18):
            layers.append(torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1))
            layers.append(torch.nn.BatchNorm2d(64, eps=1e-3))
            layers.append(torch.nn.ReLU())

        self.layer2 = torch.nn.Sequential(*layers)
        self.layer2.apply(init_weights)

        # L3 ImgIn shape=(?, :, :, 64)
        # Conv ->(?, :, :, 1)
        self.layer3 = torch.nn.Conv2d(64, 1, kernel_size=5, stride=1, padding=2)
        torch.nn.init.kaiming_normal_(self.layer3.weight, nonlinearity="relu")
        self.layer3.bias.data.fill_(0.01)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)

        # input - noise. Because the model learns the noise parameters.
        out = torch.sub(x, out)

        return out


class ChannelNet(nn.Module):
    """SRCNN -> DnCNN cascade, matching the real repo's train_DnCNN chaining (SRCNN
    output feeds DnCNN input) and the paper's super-resolution-then-denoising pipeline.
    """

    def __init__(self):
        super().__init__()
        self.srcnn = SRCNN()
        self.dncnn = DnCNN()

    def forward(self, x):
        sr = self.srcnn(x)
        return self.dncnn(sr)


def build_channelnet_tiny() -> ChannelNet:
    return ChannelNet().eval()


def example_input_channelnet_tiny():
    # Real repo treats the OFDM pilot/channel-response grid as a single-channel 2D
    # "image" (see README: "the time-frequency response ... as a two-dimensional
    # image"); a modest spatial size keeps the 20-conv-layer DnCNN cheap to trace.
    return torch.randn(1, 1, 32, 32)


MENAGERIE_ENTRIES = [
    (
        "ChannelNet",
        "build_channelnet_tiny",
        "example_input_channelnet_tiny",
        2019,
        "vendored-pytorch",
    ),
]
