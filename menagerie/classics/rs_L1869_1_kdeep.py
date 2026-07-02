# SOURCE: vendored from https://github.com/abdulsalam-bande/KDeep @ main
# (pytorch/model.py, `CNNScore` class). Community PyTorch reimplementation of
# the KDeep 3D-CNN binding-affinity scorer (Jimenez et al., JCIM 2018), a
# SqueezeNet-style network of "fire" squeeze/expand Conv3d blocks operating
# on 16-channel voxelized protein-ligand pharmacophore grids. Copied verbatim
# (class body unmodified, only the module-level `cnnscore()` factory kept and
# imports trimmed); the accompanying `VGGNet` class in the same file is left
# out of staging because the upstream file marks it explicitly incomplete
# (`# TODO: Following`, commented-out final layers).
"""Vendored KDeep model definition (CNNScore SqueezeNet-style 3D CNN)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


class CNNScore(nn.Module):
    def __init__(self):
        super(CNNScore, self).__init__()
        self.conv1 = nn.Conv3d(16, 96, kernel_size=1, stride=2)
        self.fire2_squeeze = nn.Conv3d(96, 16, kernel_size=1)
        self.fire2_expand1 = nn.Conv3d(16, 64, kernel_size=1)
        # Padding = (k-1)/2 where k is the kernel size
        self.fire2_expand2 = nn.Conv3d(16, 64, kernel_size=3, padding=1)

        self.fire3_squeeze = nn.Conv3d(128, 16, kernel_size=1)
        self.fire3_expand1 = nn.Conv3d(16, 64, kernel_size=1)
        self.fire3_expand2 = nn.Conv3d(16, 64, kernel_size=3, padding=1)

        self.fire4_squeeze = nn.Conv3d(128, 32, kernel_size=1)
        self.fire4_expand1 = nn.Conv3d(32, 128, kernel_size=1)
        self.fire4_expand2 = nn.Conv3d(32, 128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool3d(kernel_size=3, stride=2)

        self.fire5_squeeze = nn.Conv3d(256, 32, kernel_size=1)
        self.fire5_expand1 = nn.Conv3d(32, 128, kernel_size=1)
        self.fire5_expand2 = nn.Conv3d(32, 128, kernel_size=3, padding=1)

        self.fire6_squeeze = nn.Conv3d(256, 48, kernel_size=1)
        self.fire6_expand1 = nn.Conv3d(48, 192, kernel_size=1)
        self.fire6_expand2 = nn.Conv3d(48, 192, kernel_size=3, padding=1)

        self.fire7_squeeze = nn.Conv3d(384, 48, kernel_size=1)
        self.fire7_expand1 = nn.Conv3d(48, 192, kernel_size=1)
        self.fire7_expand2 = nn.Conv3d(48, 192, kernel_size=3, padding=1)

        self.fire8_squeeze = nn.Conv3d(384, 64, kernel_size=1)
        self.fire8_expand1 = nn.Conv3d(64, 256, kernel_size=1)
        self.fire8_expand2 = nn.Conv3d(64, 256, kernel_size=3, padding=1)

        self.avg_pool = nn.AvgPool3d(kernel_size=3, padding=1)

        self.dense1 = nn.Linear(512 * 2 * 2 * 2, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.fire2_squeeze(x))
        expand1 = F.relu(self.fire2_expand1(x))
        expand2 = F.relu(self.fire2_expand2(x))
        merge1 = torch.cat((expand1, expand2), 1)

        x = F.relu(self.fire3_squeeze(merge1))
        expand1 = F.relu(self.fire3_expand1(x))
        expand2 = F.relu(self.fire3_expand2(x))
        merge2 = torch.cat((expand1, expand2), 1)

        x = F.relu(self.fire4_squeeze(merge2))
        expand1 = F.relu(self.fire4_expand1(x))
        expand2 = F.relu(self.fire4_expand2(x))
        merge3 = torch.cat((expand1, expand2), 1)
        pool1 = self.pool(merge3)

        x = F.relu(self.fire5_squeeze(pool1))
        expand1 = F.relu(self.fire5_expand1(x))
        expand2 = F.relu(self.fire5_expand2(x))
        merge4 = torch.cat((expand1, expand2), 1)

        x = F.relu(self.fire6_squeeze(merge4))
        expand1 = F.relu(self.fire6_expand1(x))
        expand2 = F.relu(self.fire6_expand2(x))
        merge5 = torch.cat((expand1, expand2), 1)

        x = F.relu(self.fire7_squeeze(merge5))
        expand1 = F.relu(self.fire7_expand1(x))
        expand2 = F.relu(self.fire7_expand2(x))
        merge6 = torch.cat((expand1, expand2), 1)

        x = F.relu(self.fire8_squeeze(merge6))
        expand1 = F.relu(self.fire8_expand1(x))
        expand2 = F.relu(self.fire8_expand2(x))
        merge7 = torch.cat((expand1, expand2), 1)

        pool2 = self.avg_pool(merge7)
        x = pool2.view(-1, 512 * 2 * 2 * 2)
        x = self.dense1(x)

        return x


def cnnscore(**kwargs):
    model = CNNScore(**kwargs)
    return model


# ---------------------------------------------------------------------------
# Staging build/example helpers. Input is a (batch, 16, 24, 24, 24) voxelized
# protein-ligand grid, matching the repo's own torchsummary smoke test
# (`summary(model, input_size=(16, 24, 24, 24))` in pytorch/model_test.py).
# ---------------------------------------------------------------------------


def build_kdeep():
    return cnnscore()


def example_input_kdeep():
    return (torch.rand(1, 16, 24, 24, 24),)


MENAGERIE_ENTRIES = [
    ("KDeep", "build_kdeep", "example_input_kdeep", "2018", "vendored-pytorch"),
]
