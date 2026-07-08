# SOURCE: vendored from acts-project/acts @ bc2e278
# SOURCE: vendored from dtpreda/colorful @ 8c53e95
"""CV3a vendored TorchLens menagerie staging models."""

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"

L_CENTER = 50.0
L_RANGE = 100.0


class DuplicateClassifier(nn.Module):
    """MLP model used to separate good seeds from duplicate seeds."""

    def __init__(self, input_dim: int, n_layers: list[int]) -> None:
        """Initialize the five-hidden-layer duplicate classifier.

        Parameters
        ----------
        input_dim
            Number of input seed features.
        n_layers
            Width of each hidden layer, as used by the source model.
        """
        super().__init__()
        self.linear1 = nn.Linear(input_dim, n_layers[0])
        self.linear2 = nn.Linear(n_layers[0], n_layers[1])
        self.linear3 = nn.Linear(n_layers[1], n_layers[2])
        self.linear4 = nn.Linear(n_layers[2], n_layers[3])
        self.linear5 = nn.Linear(n_layers[3], n_layers[4])
        self.output = nn.Linear(n_layers[4], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Score duplicate seed candidates.

        Parameters
        ----------
        z
            Seed feature matrix.

        Returns
        -------
        torch.Tensor
            One sigmoid score per seed.
        """
        z = F.relu(self.linear1(z))
        z = F.relu(self.linear2(z))
        z = F.relu(self.linear3(z))
        z = F.relu(self.linear4(z))
        z = F.relu(self.linear5(z))
        return self.sigmoid(self.output(z))


def normalize_l(l_channel: torch.Tensor) -> torch.Tensor:
    """Normalize LAB luminance values as in the source colorizer.

    Parameters
    ----------
    l_channel
        Luminance image tensor.

    Returns
    -------
    torch.Tensor
        Normalized luminance tensor.
    """
    return (l_channel - L_CENTER) / L_RANGE


class Colorizer(nn.Module):
    """Zhang-style colorization network from the colorful PyTorch repo."""

    def __init__(self) -> None:
        """Initialize the convolutional colorizer."""
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1, stride=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, stride=2, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, stride=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, stride=2, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1, stride=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, stride=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, stride=2, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1, stride=1, dilation=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1, stride=1, dilation=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1, stride=1, dilation=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=2, stride=1, dilation=2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, stride=1, dilation=2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, stride=1, dilation=2, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=2, stride=1, dilation=2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, stride=1, dilation=2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, stride=1, dilation=2, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1, stride=1, dilation=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1, stride=1, dilation=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1, stride=1, dilation=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
        )
        self.conv8 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, padding=1, stride=2, dilation=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, stride=1, dilation=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, stride=1, dilation=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.softmax = nn.Sequential(
            nn.Conv2d(256, 326, 1, padding=0, stride=1, dilation=1, bias=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict quantized color-bin logits from luminance.

        Parameters
        ----------
        x
            Luminance tensor in NCHW format.

        Returns
        -------
        torch.Tensor
            Per-pixel color-bin logits.
        """
        x = normalize_l(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        return self.softmax(x)


def build_acts_seed_filter() -> DuplicateClassifier:
    """Build the ACTS seed duplicate classifier.

    Returns
    -------
    DuplicateClassifier
        Randomly initialized source architecture.
    """
    return DuplicateClassifier(input_dim=10, n_layers=[16, 16, 16, 16, 16])


def example_input_acts_seed_filter() -> torch.Tensor:
    """Create an example ACTS seed-filter input.

    Returns
    -------
    torch.Tensor
        Batch of seed features.
    """
    return torch.randn(4, 10)


def build_zhang_colorizer() -> Colorizer:
    """Build the Zhang colorization network.

    Returns
    -------
    Colorizer
        Randomly initialized source architecture.
    """
    return Colorizer()


def example_input_zhang_colorizer() -> torch.Tensor:
    """Create an example luminance image tensor.

    Returns
    -------
    torch.Tensor
        Luminance image in NCHW format.
    """
    return torch.randn(1, 1, 32, 32)


MENAGERIE_ENTRIES = [
    (
        "ACTS Seed Filter NN",
        "build_acts_seed_filter",
        "example_input_acts_seed_filter",
        2023,
        "CV3a-15",
    ),
    (
        "Zhang Colorful Colorization",
        "build_zhang_colorizer",
        "example_input_zhang_colorizer",
        2016,
        "CV3a-16",
    ),
]
