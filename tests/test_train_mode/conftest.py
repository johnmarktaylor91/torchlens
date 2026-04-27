"""Shared fixtures for train-mode regression tests."""

from __future__ import annotations

import pytest
import torch
from torch import nn


class TwoLayerMlp(nn.Module):
    """Small MLP with a single hidden layer."""

    def __init__(self) -> None:
        """Initialize the layers."""

        super().__init__()
        self.fc1 = nn.Linear(4, 6)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(6, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass."""

        return self.fc2(self.relu(self.fc1(x)))


class TinyResnetWithProbe(nn.Module):
    """Tiny frozen-backbone model with a trainable probe head."""

    def __init__(self) -> None:
        """Initialize the backbone and probe."""

        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
        )
        self.probe = nn.Linear(8, 2)
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the frozen backbone and trainable probe."""

        return self.probe(self.backbone(x))


class TeacherStudentPair(nn.Module):
    """Two-head model used for teacher-student style captures."""

    def __init__(self) -> None:
        """Initialize teacher and student branches."""

        super().__init__()
        self.teacher = nn.Linear(4, 3)
        self.student = nn.Linear(4, 3)
        for param in self.teacher.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return teacher and student outputs."""

        return self.teacher(x), self.student(x)


class MultiTapModel(nn.Module):
    """Model returning intermediate and final activations."""

    def __init__(self) -> None:
        """Initialize the tap layers."""

        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a hidden activation and final output."""

        hidden = torch.relu(self.fc1(x))
        return hidden, self.fc2(hidden)


@pytest.fixture
def two_layer_mlp() -> TwoLayerMlp:
    """Return a two-layer MLP fixture."""

    return TwoLayerMlp()


@pytest.fixture
def tiny_resnet_with_probe() -> TinyResnetWithProbe:
    """Return a frozen-backbone model with a trainable probe."""

    return TinyResnetWithProbe()


@pytest.fixture
def teacher_student_pair() -> TeacherStudentPair:
    """Return a teacher-student pair fixture."""

    return TeacherStudentPair()


@pytest.fixture
def multi_tap_model() -> MultiTapModel:
    """Return a model with multiple output taps."""

    return MultiTapModel()
