"""Additional Spikformer/SpikeFormer catalog aliases backed by the Spikformer classic.

The target rows name CIFAR/ImageNet/DVS variants of the same Spikformer family:
spiking patch splitting, LIF spike neurons, and softmax-free Spiking
Self-Attention.  The faithful compact implementation lives in
``menagerie.classics.spikformer`` and is reused here with variant labels.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from menagerie.classics.spikformer import Spikformer


def build_cifar10() -> nn.Module:
    """Build a compact CIFAR Spikformer.

    Returns
    -------
    nn.Module
        Spikformer with ten output classes.
    """

    return Spikformer(embed_dim=48, depth=2, num_heads=4, num_classes=10, timesteps=2)


def build_cifar100() -> nn.Module:
    """Build a compact CIFAR-100 Spikformer.

    Returns
    -------
    nn.Module
        Spikformer with one hundred output classes.
    """

    return Spikformer(embed_dim=48, depth=2, num_heads=4, num_classes=100, timesteps=2)


def build_dvs() -> nn.Module:
    """Build a compact event/DVS Spikformer.

    Returns
    -------
    nn.Module
        Spikformer with two input polarity channels.
    """

    return Spikformer(embed_dim=48, depth=2, num_heads=4, num_classes=10, in_ch=2, timesteps=3)


def build_meta_384() -> nn.Module:
    """Build a compact Meta-SpikeFormer alias for nominal width 384.

    Returns
    -------
    nn.Module
        Compact Spikformer preserving SSA primitives.
    """

    return Spikformer(embed_dim=48, depth=2, num_heads=4, num_classes=1000, timesteps=2)


def build_meta_512() -> nn.Module:
    """Build a compact Meta-SpikeFormer alias for nominal width 512.

    Returns
    -------
    nn.Module
        Compact Spikformer preserving SSA primitives.
    """

    return Spikformer(embed_dim=64, depth=2, num_heads=4, num_classes=1000, timesteps=2)


def build_meta_768() -> nn.Module:
    """Build a compact Meta-SpikeFormer alias for nominal width 768.

    Returns
    -------
    nn.Module
        Compact Spikformer preserving SSA primitives.
    """

    return Spikformer(embed_dim=72, depth=2, num_heads=4, num_classes=1000, timesteps=2)


def example_rgb() -> torch.Tensor:
    """Create a small RGB image input.

    Returns
    -------
    torch.Tensor
        Example tensor ``(1, 3, 32, 32)``.
    """

    return torch.randn(1, 3, 32, 32)


def example_dvs() -> torch.Tensor:
    """Create a two-polarity event-frame input.

    Returns
    -------
    torch.Tensor
        Example tensor ``(1, 2, 32, 32)``.
    """

    return torch.randn(1, 2, 32, 32)


MENAGERIE_ENTRIES = [
    (
        "spikingformer_cifar10 (Spikformer SSA spiking ViT)",
        "build_cifar10",
        "example_rgb",
        "2023",
        "DC",
    ),
    (
        "spikingformer_cifar100 (Spikformer SSA spiking ViT)",
        "build_cifar100",
        "example_rgb",
        "2023",
        "DC",
    ),
    (
        "spikformer_cifar10dvs (Spikformer SSA event-camera ViT)",
        "build_dvs",
        "example_dvs",
        "2023",
        "DC",
    ),
    (
        "meta_spikeformer_8_384 (SpikeFormer 8-block SSA family)",
        "build_meta_384",
        "example_rgb",
        "2023",
        "DC",
    ),
    (
        "meta_spikeformer_8_512 (SpikeFormer 8-block SSA family)",
        "build_meta_512",
        "example_rgb",
        "2023",
        "DC",
    ),
    (
        "meta_spikeformer_8_768 (SpikeFormer 8-block SSA family)",
        "build_meta_768",
        "example_rgb",
        "2023",
        "DC",
    ),
]
