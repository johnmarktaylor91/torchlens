# FAITHFUL REIMPLEMENTATION from detailed description (no public code): the 1D-CNN
# acoustic-emission (AE) pipeline-leak detector described in Z. Ahmad, T.-K. Nguyen,
# J.-M. Kim, "Leak detection and size identification in fluid pipelines using a novel
# vulnerability index and 1-D convolutional neural network", Engineering Applications
# of Computational Fluid Dynamics (2023). The paper's Table 1 specifies the network
# exactly: five convolutional layers (CL1..CL5, ReLU, channel/kernel sizes 128/16,
# 64/8, 32/4, 16/4, 8/4), four max-pooling layers (PL1..PL4, 2x downsampling each,
# after CL1-CL4), and dropout (p=0.25) applied at PL3 and after CL5, feeding two
# fully-connected classification layers. The paper's "five VIs" (vulnerability-index
# features computed per acoustic-emission-hit window) are reimplemented here as five
# input channels to CL1, matching the paper's stated "input feature vector ... five
# VIs" over a windowed AE-hit feature sequence. No unified public repository exists
# for this architecture; every layer/kernel/channel count below is taken directly
# from the paper's Table 1, not guessed.
from __future__ import annotations

import torch
from torch import Tensor, nn

MENAGERIE_ZOO = "reimpl-pytorch"


class AEPipelineLeakCNN(nn.Module):
    """1D-CNN for acoustic-emission-based pipeline leak detection and size ID.

    Reimplements the Ahmad et al. (2023) Table 1 architecture: CL1(128,k16) -> PL1 ->
    CL2(64,k8) -> PL2 -> CL3(32,k4) -> PL3 -> dropout -> CL4(16,k4) -> PL4 ->
    CL5(8,k4) -> dropout -> flatten -> FC -> FC (leak/leak-size classifier).
    """

    def __init__(self, in_channels: int = 5, num_classes: int = 2) -> None:
        """Initialize the five conv/pool blocks and two-layer FC classifier head."""
        super().__init__()
        self.cl1 = nn.Conv1d(in_channels, 128, kernel_size=16)
        self.pl1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.cl2 = nn.Conv1d(128, 64, kernel_size=8)
        self.pl2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.cl3 = nn.Conv1d(64, 32, kernel_size=4)
        self.pl3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout_pl3 = nn.Dropout(p=0.25)
        self.cl4 = nn.Conv1d(32, 16, kernel_size=4)
        self.pl4 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.cl5 = nn.Conv1d(16, 8, kernel_size=4)
        self.dropout_cl5 = nn.Dropout(p=0.25)
        self.relu = nn.ReLU()

        # Flatten size for a length-256 input over 5 VI channels (computed from the
        # valid-padding conv/pool chain above): 8 channels x 8 timesteps.
        self.fc1 = nn.Linear(8 * 8, 32)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Classify a windowed multi-channel AE vulnerability-index sequence.

        Parameters
        ----------
        x
            Vulnerability-index feature tensor of shape ``(batch, 5, length)``.

        Returns
        -------
        Tensor
            Leak/leak-size class logits of shape ``(batch, num_classes)``.
        """
        x = self.relu(self.cl1(x))
        x = self.pl1(x)
        x = self.relu(self.cl2(x))
        x = self.pl2(x)
        x = self.relu(self.cl3(x))
        x = self.pl3(x)
        x = self.dropout_pl3(x)
        x = self.relu(self.cl4(x))
        x = self.pl4(x)
        x = self.relu(self.cl5(x))
        x = self.dropout_cl5(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


def build_ae_pipeline_leak_cnn() -> AEPipelineLeakCNN:
    """Build a traceable AE pipeline-leak 1D-CNN classifier."""
    return AEPipelineLeakCNN(in_channels=5, num_classes=2)


def example_input_ae_pipeline_leak_cnn() -> Tensor:
    """Return a 5-channel vulnerability-index feature sequence example."""
    return torch.randn(2, 5, 256)


MENAGERIE_ENTRIES = [
    (
        "Acoustic-Emission CNN for Pipeline Leaks",
        build_ae_pipeline_leak_cnn,
        example_input_ae_pipeline_leak_cnn,
        2023,
        "RM3b-ae-pipeline-leak",
    ),
]
