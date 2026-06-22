"""VisionLAN scene text recognizer.

Wang et al. (ICCV 2021), "From Two to One: A New Scene Text Recognizer with
Visual Language Modeling Network."  VisionLAN gives the visual recognizer
language-modeling capacity directly: visual sequence features are encoded, one
character position is replaced by a learned visual mask token, and a
language-aware decoder reconstructs character logits from visual context.  This
compact random-init version keeps the convolutional visual sequence, the
visual-language masked branch, and the parallel recognition head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    """Convolution, batch normalization, and ReLU."""

    def __init__(self, in_channels: int, out_channels: int, stride: tuple[int, int]) -> None:
        """Initialize the convolutional block.

        Parameters
        ----------
        in_channels:
            Input channel count.
        out_channels:
            Output channel count.
        stride:
            Spatial convolution stride.
        """

        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the convolutional block.

        Parameters
        ----------
        x:
            Input image or feature map.

        Returns
        -------
        torch.Tensor
            Transformed feature map.
        """

        return self.net(x)


class CompactVisionLAN(nn.Module):
    """Compact VisionLAN with visual-language masked modeling."""

    def __init__(self, vocab: int = 38, width: int = 48, max_steps: int = 12) -> None:
        """Initialize the compact recognizer.

        Parameters
        ----------
        vocab:
            Character vocabulary size.
        width:
            Hidden channel width.
        max_steps:
            Number of text positions decoded from the visual sequence.
        """

        super().__init__()
        self.max_steps = max_steps
        self.stem = nn.Sequential(
            ConvBNReLU(1, width // 2, (2, 2)),
            ConvBNReLU(width // 2, width, (2, 2)),
            ConvBNReLU(width, width, (2, 1)),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=4,
            dim_feedforward=width * 2,
            batch_first=True,
            activation="gelu",
        )
        self.visual_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.position = nn.Parameter(torch.randn(max_steps, width) * 0.02)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, width))
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=4,
            dim_feedforward=width * 2,
            batch_first=True,
            activation="gelu",
        )
        self.language_decoder = nn.TransformerEncoder(decoder_layer, num_layers=1)
        self.visual_head = nn.Linear(width, vocab)
        self.masked_head = nn.Linear(width, vocab)
        self.gate = nn.Linear(width * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Recognize text from a word image.

        Parameters
        ----------
        x:
            Grayscale word image of shape ``(B, 1, H, W)``.

        Returns
        -------
        torch.Tensor
            Character logits of shape ``(B, max_steps, vocab)``.
        """

        feat = self.stem(x).mean(dim=2).transpose(1, 2)
        feat = F.interpolate(feat.transpose(1, 2), size=self.max_steps, mode="linear").transpose(
            1, 2
        )
        visual = self.visual_encoder(feat + self.position.unsqueeze(0))
        main_logits = self.visual_head(visual)
        masked = visual.clone()
        masked[:, self.max_steps // 2 : self.max_steps // 2 + 1] = self.mask_token
        language = self.language_decoder(masked + self.position.unsqueeze(0))
        masked_logits = self.masked_head(language)
        mix = torch.sigmoid(self.gate(torch.cat([visual, language], dim=-1)))
        return mix * masked_logits + (1.0 - mix) * main_logits


def build() -> nn.Module:
    """Build the compact VisionLAN model.

    Returns
    -------
    nn.Module
        Random-init VisionLAN in evaluation mode.
    """

    return CompactVisionLAN().eval()


def example_input() -> torch.Tensor:
    """Return a small grayscale word image.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(1, 1, 32, 96)``.
    """

    return torch.randn(1, 1, 32, 96)


MENAGERIE_ENTRIES = [
    ("ppocr_visionlan", "build", "example_input", "2021", "DC"),
]
