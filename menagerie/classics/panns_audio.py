"""PANNs audio classifiers: CNN14, MobileNetV1, ResNet38, and Wavegram-Logmel-CNN14.

Kong et al., IEEE/ACM TASLP 2020, arXiv:1912.10211.
Paper: PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition.

The original dependency-gated implementations use AudioSet-sized spectrogram
front-ends and large convolutional backbones.  These compact random-init models
preserve the distinctive PANNs forms: CNN14's six convolutional blocks with
global pooling, an audio MobileNetV1 depthwise-separable stack, a ResNet38-style
residual spectrogram encoder, and the two-branch Wavegram + Logmel fusion model.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Two-convolution PANNs block followed by average pooling."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize the convolutional block.

        Parameters
        ----------
        in_channels:
            Number of input feature maps.
        out_channels:
            Number of output feature maps.
        """

        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the PANNs convolution block.

        Parameters
        ----------
        x:
            Input spectrogram tensor of shape ``(B, C, T, F)``.

        Returns
        -------
        torch.Tensor
            Downsampled feature map.
        """

        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        return F.avg_pool2d(x, 2)


class DepthwiseSeparable(nn.Module):
    """MobileNetV1 depthwise-separable convolution block."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        """Initialize the depthwise and pointwise convolutions.

        Parameters
        ----------
        in_channels:
            Number of input channels.
        out_channels:
            Number of output channels.
        stride:
            Spatial stride for the depthwise stage.
        """

        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels, bias=False
        )
        self.dw_bn = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.pw_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run depthwise then pointwise convolution.

        Parameters
        ----------
        x:
            Input feature map.

        Returns
        -------
        torch.Tensor
            Output feature map.
        """

        x = F.relu_(self.dw_bn(self.depthwise(x)))
        return F.relu_(self.pw_bn(self.pointwise(x)))


class ResidualBlock(nn.Module):
    """Compact ResNet basic block used for the ResNet38 PANN variant."""

    def __init__(self, channels: int) -> None:
        """Initialize a same-width residual block.

        Parameters
        ----------
        channels:
            Number of feature channels.
        """

        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply two convolutions with an identity skip connection.

        Parameters
        ----------
        x:
            Input feature map.

        Returns
        -------
        torch.Tensor
            Residual output feature map.
        """

        residual = x
        x = F.relu_(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu_(x + residual)


class CNN14(nn.Module):
    """Compact PANNs CNN14 spectrogram classifier."""

    def __init__(self, classes: int = 10) -> None:
        """Initialize six two-convolution blocks and a classifier head.

        Parameters
        ----------
        classes:
            Number of output classes.
        """

        super().__init__()
        widths = [1, 16, 24, 32, 48, 64, 96]
        self.blocks = nn.ModuleList([ConvBlock(widths[i], widths[i + 1]) for i in range(6)])
        self.fc1 = nn.Linear(96, 64)
        self.fc_out = nn.Linear(64, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify a log-mel spectrogram.

        Parameters
        ----------
        x:
            Tensor of shape ``(B, 1, T, F)``.

        Returns
        -------
        torch.Tensor
            Class logits.
        """

        for block in self.blocks:
            x = block(x)
        x = torch.mean(x, dim=(2, 3))
        x = F.relu_(self.fc1(x))
        return self.fc_out(x)


class AudioMobileNetV1(nn.Module):
    """PANNs MobileNetV1-style audio classifier."""

    def __init__(self, classes: int = 10) -> None:
        """Initialize a compact MobileNetV1 spectrogram encoder.

        Parameters
        ----------
        classes:
            Number of output classes.
        """

        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            DepthwiseSeparable(16, 24),
            DepthwiseSeparable(24, 32, stride=2),
            DepthwiseSeparable(32, 48),
            DepthwiseSeparable(48, 64, stride=2),
            DepthwiseSeparable(64, 96),
        )
        self.head = nn.Linear(96, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify a log-mel spectrogram.

        Parameters
        ----------
        x:
            Tensor of shape ``(B, 1, T, F)``.

        Returns
        -------
        torch.Tensor
            Class logits.
        """

        x = self.blocks(self.stem(x))
        return self.head(torch.mean(x, dim=(2, 3)))


class ResNet38Audio(nn.Module):
    """Compact PANNs ResNet38-style spectrogram classifier."""

    def __init__(self, classes: int = 10) -> None:
        """Initialize residual stages and a classifier head.

        Parameters
        ----------
        classes:
            Number of output classes.
        """

        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 24, 3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(ResidualBlock(24), ResidualBlock(24))
        self.down1 = nn.Conv2d(24, 48, 3, stride=2, padding=1)
        self.stage2 = nn.Sequential(ResidualBlock(48), ResidualBlock(48))
        self.down2 = nn.Conv2d(48, 72, 3, stride=2, padding=1)
        self.stage3 = nn.Sequential(ResidualBlock(72), ResidualBlock(72))
        self.head = nn.Linear(72, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify a log-mel spectrogram with residual stages.

        Parameters
        ----------
        x:
            Tensor of shape ``(B, 1, T, F)``.

        Returns
        -------
        torch.Tensor
            Class logits.
        """

        x = self.stage1(self.stem(x))
        x = self.stage2(F.relu_(self.down1(x)))
        x = self.stage3(F.relu_(self.down2(x)))
        return self.head(torch.mean(x, dim=(2, 3)))


class WavegramLogmelCNN14(nn.Module):
    """Two-branch Wavegram + Logmel PANNs model."""

    def __init__(self, classes: int = 10) -> None:
        """Initialize waveform and log-mel branches.

        Parameters
        ----------
        classes:
            Number of output classes.
        """

        super().__init__()
        self.wave = nn.Sequential(
            nn.Conv1d(1, 16, 11, stride=4, padding=5),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 32, 5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )
        self.logmel = nn.Sequential(ConvBlock(1, 16), ConvBlock(16, 32), ConvBlock(32, 48))
        self.fuse = nn.Sequential(ConvBlock(80, 96), ConvBlock(96, 96))
        self.head = nn.Linear(96, classes)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Classify a raw waveform with learned wavegram and pseudo-logmel branches.

        Parameters
        ----------
        waveform:
            Tensor of shape ``(B, 1, samples)``.

        Returns
        -------
        torch.Tensor
            Class logits.
        """

        wave = self.wave(waveform).unsqueeze(-1)
        wave = F.interpolate(wave, size=(32, 16), mode="bilinear", align_corners=False)
        spec = waveform.unfold(-1, 16, 8).abs().mean(-1).unsqueeze(-1)
        spec = F.interpolate(spec, size=(128, 64), mode="bilinear", align_corners=False)
        logmel = self.logmel(torch.log1p(spec))
        wave = F.interpolate(wave, size=logmel.shape[2:], mode="bilinear", align_corners=False)
        fused = self.fuse(torch.cat([wave, logmel], dim=1))
        return self.head(torch.mean(fused, dim=(2, 3)))


def build_cnn14() -> nn.Module:
    """Build the compact CNN14 model.

    Returns
    -------
    nn.Module
        Random-init compact CNN14.
    """

    return CNN14()


def build_mobilenetv1_audio() -> nn.Module:
    """Build the compact MobileNetV1 audio model.

    Returns
    -------
    nn.Module
        Random-init compact MobileNetV1 audio classifier.
    """

    return AudioMobileNetV1()


def build_resnet38() -> nn.Module:
    """Build the compact ResNet38 audio model.

    Returns
    -------
    nn.Module
        Random-init compact ResNet38 audio classifier.
    """

    return ResNet38Audio()


def build_wavegram_logmel_cnn14() -> nn.Module:
    """Build the compact Wavegram-Logmel-CNN14 model.

    Returns
    -------
    nn.Module
        Random-init compact Wavegram-Logmel-CNN14.
    """

    return WavegramLogmelCNN14()


def example_spectrogram() -> torch.Tensor:
    """Create a compact log-mel spectrogram input.

    Returns
    -------
    torch.Tensor
        Spectrogram tensor ``(1, 1, 64, 64)``.
    """

    return torch.randn(1, 1, 64, 64)


def example_waveform() -> torch.Tensor:
    """Create a compact raw-waveform input.

    Returns
    -------
    torch.Tensor
        Waveform tensor ``(1, 1, 1024)``.
    """

    return torch.randn(1, 1, 1024)


MENAGERIE_ENTRIES = [
    ("PANNs-CNN14", "build_cnn14", "example_spectrogram", "2020", "DC"),
    ("panns_cnn14", "build_cnn14", "example_spectrogram", "2020", "DC"),
    ("PANNs-MobileNetV1-audio", "build_mobilenetv1_audio", "example_spectrogram", "2020", "DC"),
    ("PANNs-ResNet38", "build_resnet38", "example_spectrogram", "2020", "DC"),
    (
        "PANNs-Wavegram-Logmel-Cnn14",
        "build_wavegram_logmel_cnn14",
        "example_waveform",
        "2020",
        "DC",
    ),
]
