"""Real architectures for the deck's "At scale and edge cases" section.

Copied from the visual audit pack's model file (``notebooks/audit/visual/_visual_models.py``):
torchvision models with ``weights=None`` (random initialisation; the structure is all a render
needs) in eval mode, traced on a small image. torchvision is imported inside each factory, so
the slide table and the coverage check never import it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import nn

from scripts.visual_language.models import _seeded


def _torchvision(name: str) -> Callable[[], tuple[nn.Module, Any]]:
    def factory() -> tuple[nn.Module, Any]:
        from torchvision import models

        return getattr(models, name)(weights=None).eval(), torch.randn(1, 3, 64, 64)

    return _seeded(factory)


SCALE_FIXTURES: dict[str, Callable[[], tuple[nn.Module, Any]]] = {
    "ResNet18": _torchvision("resnet18"),
    "ResNet50": _torchvision("resnet50"),
    "DenseNet121": _torchvision("densenet121"),
}
