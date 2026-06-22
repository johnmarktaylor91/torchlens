"""Compact learn2learn-style few-shot vision classics.

The models in this module reproduce public learn2learn/common few-shot cores:
Conv4/CNN4, Omniglot FC/CNN heads, ResNet12 and WRN-style backbones, Matching
Networks attention over a support set, and Prototypical Networks class means with
Euclidean distance.  They are small random-init tracing targets.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv4Backbone(nn.Module):
    """Four-block convolutional embedding network used in few-shot baselines."""

    def __init__(self, in_channels: int = 1, hidden: int = 16) -> None:
        """Initialize Conv-BN-ReLU-Pool blocks.

        Parameters
        ----------
        in_channels:
            Number of image channels.
        hidden:
            Hidden channel width.
        """

        super().__init__()
        layers: list[nn.Module] = []
        channels = in_channels
        for _ in range(4):
            layers.extend(
                [
                    nn.Conv2d(channels, hidden, 3, padding=1, bias=False),
                    nn.BatchNorm2d(hidden),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                ]
            )
            channels = hidden
        self.features = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed images.

        Parameters
        ----------
        x:
            Image batch.

        Returns
        -------
        torch.Tensor
            Flattened embeddings.
        """

        return self.features(x).flatten(1)


class CNN4Classifier(nn.Module):
    """Conv4 feature extractor with a linear classifier."""

    def __init__(self, in_channels: int = 1, classes: int = 5) -> None:
        """Initialize backbone and classifier.

        Parameters
        ----------
        in_channels:
            Number of image channels.
        classes:
            Output class count.
        """

        super().__init__()
        self.backbone = Conv4Backbone(in_channels)
        self.classifier = nn.Linear(16, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify images.

        Parameters
        ----------
        x:
            Image batch.

        Returns
        -------
        torch.Tensor
            Class logits.
        """

        return self.classifier(self.backbone(x))


class OmniglotFC(nn.Module):
    """Fully connected Omniglot baseline from learn2learn."""

    def __init__(self, classes: int = 5) -> None:
        """Initialize two hidden layers and classifier.

        Parameters
        ----------
        classes:
            Output class count.
        """

        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify Omniglot images.

        Parameters
        ----------
        x:
            Image batch.

        Returns
        -------
        torch.Tensor
            Class logits.
        """

        return self.net(x)


class _ResidualBlock(nn.Module):
    """Small residual block for ResNet12-style embeddings."""

    def __init__(self, in_channels: int, out_channels: int, pool: bool = True) -> None:
        """Initialize residual branch and shortcut.

        Parameters
        ----------
        in_channels:
            Input channels.
        out_channels:
            Output channels.
        pool:
            Whether to downsample spatially.
        """

        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual transformation.

        Parameters
        ----------
        x:
            Feature map.

        Returns
        -------
        torch.Tensor
            Updated feature map.
        """

        return self.pool(F.leaky_relu(self.main(x) + self.shortcut(x), 0.1))


class ResNet12Backbone(nn.Module):
    """Compact ResNet12 few-shot backbone."""

    def __init__(self, in_channels: int = 3) -> None:
        """Initialize four residual stages.

        Parameters
        ----------
        in_channels:
            Number of image channels.
        """

        super().__init__()
        self.stages = nn.Sequential(
            _ResidualBlock(in_channels, 16),
            _ResidualBlock(16, 32),
            _ResidualBlock(32, 48),
            _ResidualBlock(48, 64),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed images with global average pooling.

        Parameters
        ----------
        x:
            Image batch.

        Returns
        -------
        torch.Tensor
            Image embeddings.
        """

        return self.stages(x).mean(dim=(2, 3))


class ResNet12Classifier(nn.Module):
    """ResNet12 backbone with classifier."""

    def __init__(self, classes: int = 5) -> None:
        """Initialize backbone and head.

        Parameters
        ----------
        classes:
            Output class count.
        """

        super().__init__()
        self.backbone = ResNet12Backbone()
        self.head = nn.Linear(64, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify images.

        Parameters
        ----------
        x:
            Image batch.

        Returns
        -------
        torch.Tensor
            Class logits.
        """

        return self.head(self.backbone(x))


class WRNBackbone(nn.Module):
    """Compact wide-residual-network-style few-shot backbone."""

    def __init__(self, width: int = 2) -> None:
        """Initialize widened residual stages.

        Parameters
        ----------
        width:
            Width multiplier.
        """

        super().__init__()
        self.stem = nn.Conv2d(3, 16, 3, padding=1)
        self.blocks = nn.Sequential(
            _ResidualBlock(16, 16 * width, pool=False),
            _ResidualBlock(16 * width, 32 * width),
            _ResidualBlock(32 * width, 64 * width),
        )
        self.out_dim = 64 * width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed images with a WRN-style stack.

        Parameters
        ----------
        x:
            Image batch.

        Returns
        -------
        torch.Tensor
            Image embeddings.
        """

        return self.blocks(F.relu(self.stem(x))).mean(dim=(2, 3))


class WRNClassifier(nn.Module):
    """WRN backbone with classifier."""

    def __init__(self, classes: int = 5) -> None:
        """Initialize backbone and classifier.

        Parameters
        ----------
        classes:
            Output class count.
        """

        super().__init__()
        self.backbone = WRNBackbone()
        self.head = nn.Linear(self.backbone.out_dim, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify images.

        Parameters
        ----------
        x:
            Image batch.

        Returns
        -------
        torch.Tensor
            Class logits.
        """

        return self.head(self.backbone(x))


class MatchingNetConv4(nn.Module):
    """Matching Network with Conv4 embeddings and attention over support labels."""

    def __init__(self, classes: int = 3) -> None:
        """Initialize support/query embedding network.

        Parameters
        ----------
        classes:
            Number of episode classes.
        """

        super().__init__()
        self.classes = classes
        self.encoder = Conv4Backbone()

    def forward(self, episode: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Classify queries from support examples.

        Parameters
        ----------
        episode:
            Tuple ``(support_images, support_labels, query_images)``.

        Returns
        -------
        torch.Tensor
            Query class probabilities.
        """

        support, labels, query = episode
        support_z = F.normalize(self.encoder(support), dim=-1)
        query_z = F.normalize(self.encoder(query), dim=-1)
        attn = torch.softmax(query_z @ support_z.t(), dim=-1)
        one_hot = F.one_hot(labels.long(), self.classes).float()
        return attn @ one_hot


class PrototypicalConv4(nn.Module):
    """Prototypical Network with Conv4 embeddings and Euclidean distances."""

    def __init__(self, classes: int = 3) -> None:
        """Initialize embedding network.

        Parameters
        ----------
        classes:
            Number of episode classes.
        """

        super().__init__()
        self.classes = classes
        self.encoder = Conv4Backbone()

    def forward(self, episode: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Classify queries by distance to support prototypes.

        Parameters
        ----------
        episode:
            Tuple ``(support_images, support_labels, query_images)``.

        Returns
        -------
        torch.Tensor
            Negative squared distances to class prototypes.
        """

        support, labels, query = episode
        support_z = self.encoder(support)
        query_z = self.encoder(query)
        prototypes = []
        for cls in range(self.classes):
            weights = (labels == cls).float().unsqueeze(1)
            prototypes.append((support_z * weights).sum(dim=0) / weights.sum().clamp_min(1.0))
        proto = torch.stack(prototypes)
        return -((query_z.unsqueeze(1) - proto.unsqueeze(0)).pow(2).sum(dim=-1))


def build_cnn4_backbone() -> nn.Module:
    """Build a Conv4 backbone.

    Returns
    -------
    nn.Module
        Conv4 backbone.
    """

    return Conv4Backbone()


def build_cnn4() -> nn.Module:
    """Build a Conv4 classifier.

    Returns
    -------
    nn.Module
        Conv4 classifier.
    """

    return CNN4Classifier()


def build_omniglot_fc() -> nn.Module:
    """Build an Omniglot fully connected classifier.

    Returns
    -------
    nn.Module
        Omniglot FC model.
    """

    return OmniglotFC()


def build_resnet12_backbone() -> nn.Module:
    """Build a ResNet12 backbone.

    Returns
    -------
    nn.Module
        ResNet12 backbone.
    """

    return ResNet12Backbone()


def build_resnet12() -> nn.Module:
    """Build a ResNet12 classifier.

    Returns
    -------
    nn.Module
        ResNet12 classifier.
    """

    return ResNet12Classifier()


def build_wrn28_backbone() -> nn.Module:
    """Build a compact WRN28-style backbone.

    Returns
    -------
    nn.Module
        WRN-style backbone.
    """

    return WRNBackbone()


def build_wrn28() -> nn.Module:
    """Build a compact WRN28-style classifier.

    Returns
    -------
    nn.Module
        WRN-style classifier.
    """

    return WRNClassifier()


def build_matchingnet_conv4() -> nn.Module:
    """Build a Matching Network.

    Returns
    -------
    nn.Module
        Matching Network.
    """

    return MatchingNetConv4()


def build_prototypical_conv4() -> nn.Module:
    """Build a Prototypical Network.

    Returns
    -------
    nn.Module
        Prototypical Network.
    """

    return PrototypicalConv4()


def example_omniglot_image() -> torch.Tensor:
    """Create Omniglot image batch.

    Returns
    -------
    torch.Tensor
        Image tensor ``(2, 1, 28, 28)``.
    """

    return torch.randn(2, 1, 28, 28)


def example_rgb_image() -> torch.Tensor:
    """Create RGB few-shot image batch.

    Returns
    -------
    torch.Tensor
        Image tensor ``(2, 3, 32, 32)``.
    """

    return torch.randn(2, 3, 32, 32)


def example_episode() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a compact 3-way one-shot episode.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Support images, support labels, and query images.
    """

    return torch.randn(3, 1, 28, 28), torch.tensor([0, 1, 2]), torch.randn(2, 1, 28, 28)


MENAGERIE_ENTRIES = [
    ("learn2learn CNN4Backbone", "build_cnn4_backbone", "example_omniglot_image", "2017", "E5"),
    ("learn2learn CNN4", "build_cnn4", "example_omniglot_image", "2017", "E5"),
    ("learn2learn OmniglotCNN", "build_cnn4", "example_omniglot_image", "2017", "E5"),
    ("learn2learn OmniglotFC", "build_omniglot_fc", "example_omniglot_image", "2017", "E5"),
    ("MatchingNet-Conv4", "build_matchingnet_conv4", "example_episode", "2016", "E5"),
    ("PrototypicalNet-Conv4", "build_prototypical_conv4", "example_episode", "2017", "E5"),
    ("prototypical_conv4", "build_prototypical_conv4", "example_episode", "2017", "E5"),
    ("learn2learn ResNet12Backbone", "build_resnet12_backbone", "example_rgb_image", "2020", "E5"),
    ("learn2learn ResNet12", "build_resnet12", "example_rgb_image", "2020", "E5"),
    ("learn2learn WRN28Backbone", "build_wrn28_backbone", "example_rgb_image", "2020", "E5"),
    ("learn2learn WRN28", "build_wrn28", "example_rgb_image", "2020", "E5"),
]
