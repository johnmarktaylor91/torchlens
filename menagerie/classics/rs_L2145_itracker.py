# SOURCE: vendored from yihuacheng/Itracker @ main (Itracker/model.py)
# https://github.com/yihuacheng/Itracker
#
# PyTorch reimplementation of iTracker (Eye Tracking for Everyone, Krafka et al.,
# CVPR 2016) -- AlexNet-style multi-branch gaze estimator (left eye / right eye /
# face / face-grid pathways fused by FC layers). Architecture code below is
# unmodified from the upstream `Itracker/model.py`; only the `if __name__` demo
# block was replaced with `build_itracker()` / `example_input_itracker()` staging
# helpers for the menagerie build-bridge.

import torch
import torch.nn as nn


class ItrackerImageModel(nn.Module):
    # Used for both eyes (with shared weights) and the face (with unqiue weights)
    def __init__(self):
        super(ItrackerImageModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.CrossMapLRN2d(size=5, alpha=0.0001, beta=0.75, k=1.0),
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.CrossMapLRN2d(size=5, alpha=0.0001, beta=0.75, k=1.0),
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x


class FaceImageModel(nn.Module):
    def __init__(self):
        super(FaceImageModel, self).__init__()
        self.conv = ItrackerImageModel()
        self.fc = nn.Sequential(
            nn.Linear(12 * 12 * 64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


class FaceGridModel(nn.Module):
    # Model for the face grid pathway
    def __init__(self, gridSize=25):
        super(FaceGridModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(gridSize * gridSize, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ITrackerModel(nn.Module):
    def __init__(self):
        super(ITrackerModel, self).__init__()
        self.eyeModel = ItrackerImageModel()
        self.faceModel = FaceImageModel()
        self.gridModel = FaceGridModel()
        # Joining both eyes
        self.eyesFC = nn.Sequential(
            nn.Linear(2 * 12 * 12 * 64, 128),
            nn.ReLU(inplace=True),
        )
        # Joining everything
        self.fc = nn.Sequential(
            nn.Linear(128 + 64 + 128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
        )

    def forward(self, x_in):
        # Eye nets
        xEyeL = self.eyeModel(x_in["left"])
        xEyeR = self.eyeModel(x_in["right"])
        # Cat and FC
        xEyes = torch.cat((xEyeL, xEyeR), 1)
        xEyes = self.eyesFC(xEyes)

        # Face net
        xFace = self.faceModel(x_in["face"])
        xGrid = self.gridModel(x_in["grid"])

        # Cat all
        x = torch.cat((xEyes, xFace, xGrid), 1)
        x = self.fc(x)

        return x


# ---------------------------------------------------------------------------
# Menagerie staging helpers
# ---------------------------------------------------------------------------

MENAGERIE_ZOO = "vendored-pytorch"


def build_itracker():
    return ITrackerModel()


def example_input_itracker():
    return {
        "face": torch.zeros(2, 3, 224, 224),
        "left": torch.zeros(2, 3, 224, 224),
        "right": torch.zeros(2, 3, 224, 224),
        "grid": torch.zeros(2, 1, 25, 25),
    }


MENAGERIE_ENTRIES = [
    (
        "iTracker (Eye Tracking for Everyone)",
        build_itracker,
        example_input_itracker,
        2016,
        "vendored",
    ),
]
