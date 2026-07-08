# FAITHFUL PORT of rcarioni/gunshot-detection @ main
# (models/get_model.py, get_model_danaei -- Danaei, D. (2021), "Gunshot Detection in
# Wildlife using Deep Learning"; original framework: TensorFlow/Keras via
# tensorflow.keras.models.Sequential + Conv2D/MaxPool2D/Dense layers).
# TensorFlow/Keras is not in the installed base-env library set, so the original code
# cannot run directly; this is a layer-for-layer faithful port of get_model_danaei()
# into torch.nn, preserving every convolution/pooling/dense layer, kernel size,
# activation, and layer ordering from the real Keras source.
from __future__ import annotations

import torch
from torch import Tensor, nn

MENAGERIE_ZOO = "ported-pytorch"


class GunshotDanaeiCNN(nn.Module):
    """Port of Danaei (2021)'s spectrogram CNN for wildlife gunshot detection.

    Ported faithfully from ``get_model_danaei`` in rcarioni/gunshot-detection
    (models/get_model.py): three Conv2D+MaxPool2D blocks over a spectrogram input,
    followed by five dense layers with the original mixed sigmoid/tanh/relu
    activations, ending in a sigmoid binary gunshot/non-gunshot classifier.
    """

    def __init__(self, in_channels: int = 1) -> None:
        """Initialize the ported convolution stack and dense classifier head."""
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 40, kernel_size=(3, 3), stride=(1, 1))
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(40, 28, kernel_size=(4, 3), stride=(1, 1))
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv3 = nn.Conv2d(28, 24, kernel_size=(4, 3), stride=(1, 1))
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # Flatten size for a 64x64 input spectrogram (computed from the valid-padding
        # conv/pool chain above): 24 channels x 5 x 6.
        self.fc1 = nn.Linear(24 * 5 * 6, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 100)
        self.fc4 = nn.Linear(100, 300)
        self.fc5 = nn.Linear(300, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Run the ported Danaei CNN over a single-channel spectrogram.

        Parameters
        ----------
        x
            Spectrogram tensor of shape ``(batch, 1, 64, 64)``.

        Returns
        -------
        Tensor
            Gunshot probability of shape ``(batch, 1)``.
        """
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.sigmoid(self.conv2(x))
        x = self.pool2(x)
        x = torch.relu(self.conv3(x))
        x = self.pool3(x)
        x = torch.flatten(x, 1)
        x = torch.sigmoid(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.relu(self.fc4(x))
        return torch.sigmoid(self.fc5(x))


def build_gunshot_danaei_cnn() -> GunshotDanaeiCNN:
    """Build a traceable ported Danaei gunshot-detection CNN."""
    return GunshotDanaeiCNN(in_channels=1)


def example_input_gunshot_danaei_cnn() -> Tensor:
    """Return a single-channel spectrogram example for the Danaei CNN."""
    return torch.randn(1, 1, 64, 64)


MENAGERIE_ENTRIES = [
    (
        "Acoustic Gunshot (Danaei CNN)",
        build_gunshot_danaei_cnn,
        example_input_gunshot_danaei_cnn,
        2021,
        "RM3b-gunshot-danaei",
    ),
]
