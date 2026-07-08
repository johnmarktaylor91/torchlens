# SOURCE: vendored from Chinmayrane16/ReconNet-PyTorch @ 702586ed9a3d466b65c8ca801243a0f32382e299
# https://raw.githubusercontent.com/Chinmayrane16/ReconNet-PyTorch/master/model.py
#
# ReconNet, "ReconNet: Non-Iterative Reconstruction of Images from Compressively
# Sensed Measurements" (Kulkarni et al., CVPR 2016) -- pioneering compressed-sensing
# reconstruction CNN. A fully-connected layer maps a compressive measurement vector
# back to a fixed-size image patch, followed by 6 conv layers (3 conv-relu blocks
# each halved by an intermediate 1x1 "denoise" conv) that refine the reconstruction.
# Every layer/init call is transcribed verbatim from `model.py`. Only the unused
# `from torch.autograd import Variable` import is dropped (dead in the original
# file -- `Variable` is never referenced) since it is not part of the traced
# architecture.

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReconNet(nn.Module):
    def __init__(self, measurement_rate=0.25):
        super(ReconNet, self).__init__()

        self.measurement_rate = measurement_rate
        self.fc1 = nn.Linear(int(self.measurement_rate * 1089), 1089)
        nn.init.normal_(self.fc1.weight, mean=0, std=0.1)
        self.conv1 = nn.Conv2d(1, 64, 11, 1, padding=5)
        nn.init.normal_(self.conv1.weight, mean=0, std=0.1)
        self.conv2 = nn.Conv2d(64, 32, 1, 1, padding=0)
        nn.init.normal_(self.conv2.weight, mean=0, std=0.1)
        self.conv3 = nn.Conv2d(32, 1, 7, 1, padding=3)
        nn.init.normal_(self.conv3.weight, mean=0, std=0.1)
        self.conv4 = nn.Conv2d(1, 64, 11, 1, padding=5)
        nn.init.normal_(self.conv4.weight, mean=0, std=0.1)
        self.conv5 = nn.Conv2d(64, 32, 1, 1, padding=0)
        nn.init.normal_(self.conv5.weight, mean=0, std=0.1)
        self.conv6 = nn.Conv2d(32, 1, 7, 1, padding=3)
        nn.init.normal_(self.conv6.weight, mean=0, std=0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = x.view(-1, 33, 33)
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.conv6(x)

        return x


def build_reconnet():
    torch.manual_seed(0)
    model = ReconNet(measurement_rate=0.25)
    model.eval()
    return model


def example_input_reconnet():
    torch.manual_seed(0)
    # Compressive measurement vector: fc1 in-features = int(0.25 * 1089) = 272.
    return torch.randn(1, 272)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("ReconNet", "build_reconnet", "example_input_reconnet", 2016, MENAGERIE_ZOO),
]
