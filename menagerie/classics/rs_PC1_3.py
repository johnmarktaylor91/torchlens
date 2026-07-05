# SOURCE: vendored from AbductiveLearning/ABLkit @ main: examples/mnist_add/models/nn.py
"""Staged real-source ABLkit LeNet5 perception model."""

import torch
from torch import nn

MENAGERIE_ZOO = "vendored-pytorch"


class LeNet5(nn.Module):
    """LeNet5 perception network from the official ABLkit examples."""

    def __init__(
        self, num_classes: int = 10, image_size: tuple[int, int, int] = (28, 28, 1)
    ) -> None:
        """Initialize LeNet5.

        Parameters
        ----------
        num_classes
            Number of output classes.
        image_size
            Input image size in upstream ``H, W, C`` order.
        """
        super().__init__()
        self.size = 16 * ((image_size[0] // 2 - 6) // 2) * ((image_size[1] // 2 - 6) // 2)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
            nn.Conv2d(6, 16, 5),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.size, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract convolutional features.

        Parameters
        ----------
        x
            Input image tensor.

        Returns
        -------
        torch.Tensor
            Encoded features.
        """
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run LeNet5.

        Parameters
        ----------
        x
            Input image tensor.

        Returns
        -------
        torch.Tensor
            Class logits.
        """
        x = self.encoder(x)
        x = x.view(-1, self.size)
        return self.classifier(x)


def build_abductive_learning_abl() -> nn.Module:
    """Build the staged ABLkit perception model.

    Returns
    -------
    nn.Module
        Model instance.
    """
    return LeNet5(num_classes=10)


def example_input_abductive_learning_abl() -> torch.Tensor:
    """Return an example MNIST image input.

    Returns
    -------
    torch.Tensor
        Example input tensor.
    """
    return torch.randn(1, 1, 28, 28)


MENAGERIE_ENTRIES = [
    (
        "Abductive Learning (ABL)",
        "build_abductive_learning_abl",
        "example_input_abductive_learning_abl",
        2023,
        "vendored from AbductiveLearning/ABLkit examples/mnist_add/models/nn.py",
    ),
]
