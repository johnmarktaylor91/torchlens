"""Text-conditioned GAN generator classics.

Paper: AttnGAN (Xu et al., 2018), ControlGAN (Li et al., 2019), and
DM-GAN (Zhu et al., 2019).

These compact random-init reimplementations target the generator-side
architectures from three source-only text-to-image repositories that are not
base-environment installable.  They keep the load-bearing details rather than
using a generic image generator:

* AttnGAN: conditioning augmentation, multi-stage upsampling, and word-region
  attention at later stages.
* ControlGAN: word-conditioned channel gates that modulate generator features.
* DM-GAN: dynamic memory writing and reading to refine image features.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class ConditioningAugmentation(nn.Module):
    """AttnGAN-style conditioning augmentation for sentence embeddings."""

    def __init__(self, text_dim: int, cond_dim: int) -> None:
        """Initialize mean/log-variance projections.

        Parameters
        ----------
        text_dim:
            Sentence embedding dimension.
        cond_dim:
            Conditioning code dimension.
        """

        super().__init__()
        self.to_stats = nn.Linear(text_dim, cond_dim * 2)

    def forward(self, sentence: Tensor) -> Tensor:
        """Return a deterministic conditioning code from a sentence embedding.

        Parameters
        ----------
        sentence:
            Sentence embedding ``(batch, text_dim)``.

        Returns
        -------
        Tensor
            Conditioning code ``(batch, cond_dim)``.
        """

        mean, log_var = self.to_stats(sentence).chunk(2, dim=-1)
        return mean + torch.tanh(log_var) * 0.1


class UpsampleBlock(nn.Module):
    """Nearest-neighbor upsampling block with batch norm and GLU-style gating."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize an upsampling convolution block.

        Parameters
        ----------
        in_channels:
            Input feature channels.
        out_channels:
            Output feature channels.
        """

        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * 2, 3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels * 2)

    def forward(self, x: Tensor) -> Tensor:
        """Upsample and gate feature maps.

        Parameters
        ----------
        x:
            Feature tensor ``(batch, channels, height, width)``.

        Returns
        -------
        Tensor
            Upsampled feature tensor.
        """

        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        value, gate = self.norm(self.conv(x)).chunk(2, dim=1)
        return value * torch.sigmoid(gate)


class WordAttention(nn.Module):
    """Word-region attention used by AttnGAN refinement stages."""

    def __init__(self, channels: int, word_dim: int) -> None:
        """Initialize attention projections.

        Parameters
        ----------
        channels:
            Image feature channel count.
        word_dim:
            Word embedding dimension.
        """

        super().__init__()
        self.query = nn.Conv2d(channels, word_dim, 1)
        self.word_value = nn.Linear(word_dim, channels)

    def forward(self, features: Tensor, words: Tensor) -> Tensor:
        """Attend from image regions to word embeddings.

        Parameters
        ----------
        features:
            Image features ``(batch, channels, height, width)``.
        words:
            Word embeddings ``(batch, tokens, word_dim)``.

        Returns
        -------
        Tensor
            Word-context feature map with the same image shape.
        """

        batch, _, height, width = features.shape
        query = self.query(features).flatten(2).transpose(1, 2)
        weights = torch.softmax(torch.bmm(query, words.transpose(1, 2)), dim=-1)
        context = torch.bmm(weights, self.word_value(words))
        return context.transpose(1, 2).reshape(batch, -1, height, width)


class AttnGANGenerator(nn.Module):
    """Compact AttnGAN generator with CA and word-attended refinement."""

    def __init__(self, noise_dim: int = 32, text_dim: int = 64, base_channels: int = 32) -> None:
        """Initialize the AttnGAN generator.

        Parameters
        ----------
        noise_dim:
            Noise latent dimension.
        text_dim:
            Sentence and word embedding dimension.
        base_channels:
            Base feature channel count.
        """

        super().__init__()
        self.ca = ConditioningAugmentation(text_dim, text_dim)
        self.fc = nn.Linear(noise_dim + text_dim, base_channels * 8 * 4 * 4)
        self.up1 = UpsampleBlock(base_channels * 8, base_channels * 4)
        self.up2 = UpsampleBlock(base_channels * 4, base_channels * 2)
        self.attn = WordAttention(base_channels * 2, text_dim)
        self.refine = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 2, 3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.GELU(),
        )
        self.up3 = UpsampleBlock(base_channels * 2, base_channels)
        self.to_rgb = nn.Conv2d(base_channels, 3, 3, padding=1)

    def forward(self, noise: Tensor, sentence: Tensor, words: Tensor) -> Tensor:
        """Synthesize an image from noise, sentence, and word embeddings.

        Parameters
        ----------
        noise:
            Noise vector ``(batch, noise_dim)``.
        sentence:
            Sentence embedding ``(batch, text_dim)``.
        words:
            Word embeddings ``(batch, tokens, text_dim)``.

        Returns
        -------
        Tensor
            RGB image tensor.
        """

        cond = self.ca(sentence)
        x = self.fc(torch.cat([noise, cond], dim=-1)).reshape(noise.shape[0], -1, 4, 4)
        x = self.up1(x)
        x = self.up2(x)
        attended = self.attn(x, words)
        x = self.refine(torch.cat([x, attended], dim=1))
        x = self.up3(x)
        return torch.tanh(self.to_rgb(x))


class ControlGate(nn.Module):
    """ControlGAN word-level channel gate."""

    def __init__(self, channels: int, word_dim: int) -> None:
        """Initialize a word-conditioned channel gate.

        Parameters
        ----------
        channels:
            Feature channel count.
        word_dim:
            Word embedding dimension.
        """

        super().__init__()
        self.word_score = nn.Linear(word_dim, 1)
        self.to_gate = nn.Linear(word_dim, channels)

    def forward(self, features: Tensor, words: Tensor) -> Tensor:
        """Apply word-conditioned channel control.

        Parameters
        ----------
        features:
            Image features.
        words:
            Word embeddings.

        Returns
        -------
        Tensor
            Modulated features.
        """

        weights = torch.softmax(self.word_score(words).squeeze(-1), dim=-1)
        control = torch.bmm(weights.unsqueeze(1), words).squeeze(1)
        gate = torch.sigmoid(self.to_gate(control)).unsqueeze(-1).unsqueeze(-1)
        return features * gate


class ControlGANGenerator(nn.Module):
    """Compact ControlGAN generator with word-level controllable channels."""

    def __init__(self, noise_dim: int = 32, text_dim: int = 64, base_channels: int = 32) -> None:
        """Initialize the ControlGAN generator.

        Parameters
        ----------
        noise_dim:
            Noise latent dimension.
        text_dim:
            Text embedding dimension.
        base_channels:
            Base channel count.
        """

        super().__init__()
        self.fc = nn.Linear(noise_dim + text_dim, base_channels * 8 * 4 * 4)
        self.up1 = UpsampleBlock(base_channels * 8, base_channels * 4)
        self.gate1 = ControlGate(base_channels * 4, text_dim)
        self.up2 = UpsampleBlock(base_channels * 4, base_channels * 2)
        self.gate2 = ControlGate(base_channels * 2, text_dim)
        self.up3 = UpsampleBlock(base_channels * 2, base_channels)
        self.to_rgb = nn.Conv2d(base_channels, 3, 3, padding=1)

    def forward(self, noise: Tensor, sentence: Tensor, words: Tensor) -> Tensor:
        """Generate an image with word-controlled feature gates.

        Parameters
        ----------
        noise:
            Noise vector.
        sentence:
            Sentence embedding.
        words:
            Word embeddings.

        Returns
        -------
        Tensor
            RGB image tensor.
        """

        x = self.fc(torch.cat([noise, sentence], dim=-1)).reshape(noise.shape[0], -1, 4, 4)
        x = self.gate1(self.up1(x), words)
        x = self.gate2(self.up2(x), words)
        x = self.up3(x)
        return torch.tanh(self.to_rgb(x))


class DynamicMemory(nn.Module):
    """DM-GAN dynamic memory writer and reader."""

    def __init__(self, channels: int, word_dim: int, slots: int = 8) -> None:
        """Initialize memory projections.

        Parameters
        ----------
        channels:
            Feature channel count.
        word_dim:
            Word embedding dimension.
        slots:
            Number of memory slots.
        """

        super().__init__()
        self.slots = slots
        self.write = nn.Linear(word_dim, channels)
        self.read_query = nn.Conv2d(channels, channels, 1)
        self.slot_prior = nn.Parameter(torch.randn(slots, channels) * 0.02)

    def forward(self, features: Tensor, words: Tensor) -> Tensor:
        """Refine features by writing words to memory and reading by regions.

        Parameters
        ----------
        features:
            Image features.
        words:
            Word embeddings.

        Returns
        -------
        Tensor
            Memory-read context map.
        """

        batch, channels, height, width = features.shape
        word_memory = self.write(words[:, : self.slots])
        if word_memory.shape[1] < self.slots:
            pad = self.slot_prior[: self.slots - word_memory.shape[1]].expand(batch, -1, -1)
            word_memory = torch.cat([word_memory, pad], dim=1)
        memory = word_memory + self.slot_prior.unsqueeze(0)
        query = self.read_query(features).flatten(2).transpose(1, 2)
        weights = torch.softmax(torch.bmm(query, memory.transpose(1, 2)), dim=-1)
        read = torch.bmm(weights, memory)
        return read.transpose(1, 2).reshape(batch, channels, height, width)


class DMGANGenerator(nn.Module):
    """Compact DM-GAN generator with dynamic memory refinement."""

    def __init__(self, noise_dim: int = 32, text_dim: int = 64, base_channels: int = 32) -> None:
        """Initialize the DM-GAN generator.

        Parameters
        ----------
        noise_dim:
            Noise latent dimension.
        text_dim:
            Text embedding dimension.
        base_channels:
            Base channel count.
        """

        super().__init__()
        self.fc = nn.Linear(noise_dim + text_dim, base_channels * 8 * 4 * 4)
        self.up1 = UpsampleBlock(base_channels * 8, base_channels * 4)
        self.up2 = UpsampleBlock(base_channels * 4, base_channels * 2)
        self.memory = DynamicMemory(base_channels * 2, text_dim)
        self.refine = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 2, 3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.GELU(),
        )
        self.up3 = UpsampleBlock(base_channels * 2, base_channels)
        self.to_rgb = nn.Conv2d(base_channels, 3, 3, padding=1)

    def forward(self, noise: Tensor, sentence: Tensor, words: Tensor) -> Tensor:
        """Generate an image through dynamic-memory refinement.

        Parameters
        ----------
        noise:
            Noise vector.
        sentence:
            Sentence embedding.
        words:
            Word embeddings.

        Returns
        -------
        Tensor
            RGB image tensor.
        """

        x = self.fc(torch.cat([noise, sentence], dim=-1)).reshape(noise.shape[0], -1, 4, 4)
        x = self.up1(x)
        x = self.up2(x)
        read = self.memory(x, words)
        x = self.refine(torch.cat([x, read], dim=1))
        x = self.up3(x)
        return torch.tanh(self.to_rgb(x))


def build_attngan_generator() -> nn.Module:
    """Build a compact AttnGAN generator.

    Returns
    -------
    nn.Module
        Random-init AttnGAN generator.
    """

    return AttnGANGenerator().eval()


def build_controlgan_generator() -> nn.Module:
    """Build a compact ControlGAN generator.

    Returns
    -------
    nn.Module
        Random-init ControlGAN generator.
    """

    return ControlGANGenerator().eval()


def build_dmgan_generator() -> nn.Module:
    """Build a compact DM-GAN generator.

    Returns
    -------
    nn.Module
        Random-init DM-GAN generator.
    """

    return DMGANGenerator().eval()


def example_text_gan_input() -> tuple[Tensor, Tensor, Tensor]:
    """Return noise, sentence embedding, and word embeddings.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        ``(noise, sentence, words)`` example inputs.
    """

    return torch.randn(1, 32), torch.randn(1, 64), torch.randn(1, 8, 64)


MENAGERIE_ENTRIES = [
    (
        "AttnGAN generator (conditioning augmentation + word attention)",
        "build_attngan_generator",
        "example_text_gan_input",
        "2018",
        "text-to-image",
    ),
    (
        "ControlGAN generator (word-level controllable channels)",
        "build_controlgan_generator",
        "example_text_gan_input",
        "2019",
        "text-to-image",
    ),
    (
        "DM-GAN generator (dynamic memory text-to-image refinement)",
        "build_dmgan_generator",
        "example_text_gan_input",
        "2019",
        "text-to-image",
    ),
]
