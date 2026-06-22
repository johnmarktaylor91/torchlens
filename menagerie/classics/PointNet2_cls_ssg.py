"""Exact-name PointNet++ SSG classifier entry.

Paper: Qi et al. 2017, "PointNet++: Deep Hierarchical Feature Learning on Point
Sets in a Metric Space." This module exposes the requested catalog spelling while
reusing the compact single-scale grouping classifier.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from menagerie.classics.Pointnet2_PyTorch_ClassificationSSG import build as _build
from menagerie.classics.Pointnet2_PyTorch_ClassificationSSG import example_input as _example_input


def build() -> nn.Module:
    """Build the compact PointNet++ SSG classifier.

    Returns
    -------
    nn.Module
        Random-initialized PointNet++ SSG classifier.
    """

    return _build()


def example_input() -> torch.Tensor:
    """Create a simple point-cloud tensor.

    Returns
    -------
    torch.Tensor
        Point cloud of shape ``(1, 32, 3)``.
    """

    return _example_input()


MENAGERIE_ENTRIES = [("PointNet2_cls_ssg", "build", "example_input", "2017", "DC")]
