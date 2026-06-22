"""Federated Learning Classic Client Networks.

McMahan et al. 2017 (FedAvg), AshwinRJ/Federated-Learning-PyTorch 2020.
Paper: https://arxiv.org/abs/1602.05629  (FedAvg)
Source: https://github.com/AshwinRJ/Federated-Learning-PyTorch

This module faithfully reproduces the exact small client architectures from the
canonical Federated-Learning-PyTorch repository (AshwinRJ), which implements
McMahan et al.'s Communication-Efficient Learning of Deep Networks from
Decentralized Data (FedAvg, AISTATS 2017).

Models included:
  CNNMnist      -- 2 conv (10/20 ch) + dropout + 2 fc; MNIST 1x28x28 input
  CNNFashion    -- same topology as CNNMnist applied to Fashion-MNIST
  CNNCifar      -- 2 conv (6/16 ch) + 3 fc; CIFAR 3x32x32 input
  MLP           -- 1 hidden layer MLP (200 hidden) for MNIST
  modelC        -- All-Conv-C (Springenberg et al. 2015, Striving for Simplicity)
                   used as modelC in the FedAvg repo; 9-layer all-convolutional net
  FedBN-ResNet18 -- ResNet-18 client net; FedBN (Li et al. 2021) keeps BatchNorm
                    parameters local (not aggregated). Architecturally identical to
                    ResNet-18; FedBN distinction is federated BN-non-aggregation,
                    not the forward graph. Compact (16-channel base) variant shown.
  FedPer        -- FedPer (Arivazhagan et al. 2019, arXiv:1912.00818) base +
                   personalized head CNN. Shared base = 3 conv blocks; personal
                   head = 2 fc layers. The shared/personalized split is the
                   architectural primitive.

All networks: random init, CPU, compact for fast TorchLens tracing.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------------------------------
# CNNMnist / CNNFashion_Mnist
# -------------------------------------------------------------------------


class CNNMnist(nn.Module):
    """2-conv + 2-fc CNN for MNIST/Fashion-MNIST (exact AshwinRJ repo topology)."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return self.fc2(x)


# -------------------------------------------------------------------------
# CNNCifar
# -------------------------------------------------------------------------


class CNNCifar(nn.Module):
    """2-conv + 3-fc CNN for CIFAR (exact AshwinRJ repo topology)."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# -------------------------------------------------------------------------
# MLP
# -------------------------------------------------------------------------


class MLP(nn.Module):
    """1-hidden-layer MLP for MNIST (AshwinRJ repo topology; 200 hidden units)."""

    def __init__(self, dim_in: int = 784, dim_hidden: int = 200, dim_out: int = 10) -> None:
        super().__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        return self.layer_hidden(x)


# -------------------------------------------------------------------------
# modelC = All-Conv-C (Springenberg et al., 2015)
# -------------------------------------------------------------------------


class AllConvC(nn.Module):
    """All-Convolutional Network C (Springenberg et al., Striving for Simplicity 2015).

    Used as ``modelC`` in the AshwinRJ FedAvg repo.
    9 conv layers; stride-2 convs replace pooling; global average pooling at end.
    Compact version: channels halved to [48,48,48,48,48,96,48,48,10] -> [24,24,24,24,24,48,24,24,10].
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        # Compact channel schedule (original * 0.5)
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(24, 24, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(24, 24, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Conv2d(24, 48, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(48, 48, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(48, 48, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Conv2d(48, 48, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(48, 48, 1),
            nn.ReLU(),
            nn.Conv2d(48, num_classes, 1),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.gap(x)
        return x.view(x.size(0), -1)


# -------------------------------------------------------------------------
# FedBN-ResNet18 (compact)
# -------------------------------------------------------------------------


class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_c: int, out_c: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.shortcut: nn.Module = nn.Identity()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class FedBNResNet18(nn.Module):
    """ResNet-18 client net for FedBN (Li et al. 2021, arXiv:2102.07623).

    FedBN's distinctive feature: BatchNorm statistics/params kept LOCAL per client
    (not aggregated in federated rounds). The forward graph is identical to ResNet-18.
    Compact variant with base_ch=16 (original=64) for fast tracing.
    """

    def __init__(self, num_classes: int = 10, base_ch: int = 16) -> None:
        super().__init__()
        c = base_ch
        self.stem = nn.Sequential(
            nn.Conv2d(3, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(),
        )
        self.layer1 = nn.Sequential(_BasicBlock(c, c), _BasicBlock(c, c))
        self.layer2 = nn.Sequential(_BasicBlock(c, c * 2, 2), _BasicBlock(c * 2, c * 2))
        self.layer3 = nn.Sequential(_BasicBlock(c * 2, c * 4, 2), _BasicBlock(c * 4, c * 4))
        self.layer4 = nn.Sequential(_BasicBlock(c * 4, c * 8, 2), _BasicBlock(c * 8, c * 8))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(c * 8, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


# -------------------------------------------------------------------------
# FedPer: shared base + personalized head
# -------------------------------------------------------------------------


class FedPerSharedBase(nn.Module):
    """Shared (globally aggregated) convolutional base for FedPer."""

    def __init__(self, in_ch: int = 3, base_ch: int = 32) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, padding=1),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch * 2, 3, padding=1),
            nn.BatchNorm2d(base_ch * 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch * 4, 3, padding=1),
            nn.BatchNorm2d(base_ch * 4),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(2),
        )
        self.out_ch = base_ch * 4 * 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x.flatten(1)


class FedPerPersonalHead(nn.Module):
    """Client-local personalized classification head for FedPer."""

    def __init__(self, in_features: int, num_classes: int = 10) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


class FedPerNet(nn.Module):
    """FedPer full network (Arivazhagan et al. 2019, arXiv:1912.00818).

    Federated learning with personalization: shared convolutional base (aggregated
    globally) + personalized head (kept local per client). The shared/personal split
    is the architectural primitive.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.shared_base = FedPerSharedBase()
        self.personal_head = FedPerPersonalHead(self.shared_base.out_ch, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.shared_base(x)
        return self.personal_head(features)


# -------------------------------------------------------------------------
# Build functions
# -------------------------------------------------------------------------


def build_cnn_mnist() -> nn.Module:
    return CNNMnist()


def build_cnn_fashion() -> nn.Module:
    return CNNMnist()  # Same topology applied to Fashion-MNIST


def build_cnn_cifar() -> nn.Module:
    return CNNCifar()


def build_mlp() -> nn.Module:
    return MLP()


def build_model_c() -> nn.Module:
    return AllConvC()


def build_fedbn_resnet18() -> nn.Module:
    return FedBNResNet18()


def build_fedper() -> nn.Module:
    return FedPerNet()


def build_fedavg_cnn() -> nn.Module:
    """FedAvg-CNN: the generic federated CNN from McMahan et al. 2017.

    The paper describes a CNN with 2 conv (32 ch each, 5x5) + 2 fc (512, 10)
    for MNIST. This is the same topology as CNNMnist (AshwinRJ repo).
    """
    return CNNMnist()


# -------------------------------------------------------------------------
# Example inputs
# -------------------------------------------------------------------------


def example_input_mnist() -> torch.Tensor:
    return torch.zeros(1, 1, 28, 28)


def example_input_cifar() -> torch.Tensor:
    return torch.zeros(1, 3, 32, 32)


def example_input_mlp() -> torch.Tensor:
    return torch.zeros(1, 1, 28, 28)


def example_input_model_c() -> torch.Tensor:
    return torch.zeros(1, 3, 32, 32)


def example_input_fedbn() -> torch.Tensor:
    return torch.zeros(1, 3, 32, 32)


def example_input_fedper() -> torch.Tensor:
    return torch.zeros(1, 3, 32, 32)


# -------------------------------------------------------------------------
# Registry
# -------------------------------------------------------------------------

MENAGERIE_ENTRIES = [
    ("FedAvg-CNN", "build_fedavg_cnn", "example_input_mnist", "2017", "DC"),
    ("FederatedLearningPyTorch-CNNMnist", "build_cnn_mnist", "example_input_mnist", "2020", "DC"),
    (
        "FederatedLearningPyTorch-CNNFashion_Mnist",
        "build_cnn_fashion",
        "example_input_mnist",
        "2020",
        "DC",
    ),
    ("FederatedLearningPyTorch-CNNCifar", "build_cnn_cifar", "example_input_cifar", "2020", "DC"),
    ("FederatedLearningPyTorch-MLP", "build_mlp", "example_input_mlp", "2020", "DC"),
    ("FederatedLearningPyTorch-modelC", "build_model_c", "example_input_model_c", "2020", "DC"),
    ("FedBN-ResNet18", "build_fedbn_resnet18", "example_input_fedbn", "2021", "DC"),
    ("FedPer-BasePersonalizedCNN", "build_fedper", "example_input_fedper", "2019", "DC"),
]
