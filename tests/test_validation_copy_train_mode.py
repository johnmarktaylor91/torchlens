"""Validation's deep copies run in their source's train/eval mode.

``torch.fx.GraphModule.__deepcopy__`` re-runs ``__init__``, and torchvision's
feature-extractor graph module puts itself in train mode there. The ground
truth then ran batch norm on batch statistics while the capture ran the
eval-mode model, and validation reported a large ground-truth mismatch on a
correct capture.
"""

from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._capture_state_helpers import (
    _model_for_ground_truth_validation,
    _model_for_validation_replay,
)


class _ReinitOnCopy(nn.Module):
    """A module whose deepcopy re-runs ``__init__``, which leaves it in train mode."""

    def __init__(self, conv: nn.Module | None = None, norm: nn.Module | None = None) -> None:
        """Build (or adopt) the layers, then enter train mode as ``__init__`` does."""

        super().__init__()
        self.conv = conv if conv is not None else nn.Conv2d(3, 4, 3, padding=1)
        self.norm = norm if norm is not None else nn.BatchNorm2d(4)
        self.train()

    def __deepcopy__(self, memo: dict[int, object]) -> _ReinitOnCopy:
        """Copy the layers and rebuild through ``__init__``."""

        return type(self)(copy.deepcopy(self.conv, memo), copy.deepcopy(self.norm, memo))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution, batch norm and a ReLU."""

        return torch.relu(self.norm(self.conv(x)))


def _eval_model() -> _ReinitOnCopy:
    """Return the module in eval mode with non-trivial running statistics."""

    torch.manual_seed(0)
    model = _ReinitOnCopy()
    with torch.no_grad():
        for _ in range(3):
            model(torch.randn(4, 3, 6, 6) * 3 + 1)
    return model.eval()


def test_reinit_copy_premise() -> None:
    """The fixture's plain deepcopy really comes back in train mode."""

    assert copy.deepcopy(_eval_model()).norm.training is True


@pytest.mark.parametrize(
    "make_copy",
    [
        lambda model: _model_for_ground_truth_validation(model)[0],
        lambda model: _model_for_validation_replay(model)[0],
    ],
    ids=["ground_truth", "replay"],
)
def test_validation_copies_keep_source_modes(make_copy: object) -> None:
    """Both validation copies match the source's per-module modes."""

    model = _eval_model()
    model.conv.train()  # a mixed-mode tree must survive too
    copied = make_copy(model)  # type: ignore[operator]
    assert copied is not model
    assert [m.training for m in copied.modules()] == [m.training for m in model.modules()]


def test_reinit_on_copy_model_validates() -> None:
    """The eval-mode capture is checked against an eval-mode ground truth."""

    model = _eval_model()
    assert tl.validate(model, torch.randn(1, 3, 6, 6), seed=0) is True
    assert model.training is False


def test_torchvision_feature_extractor_validates() -> None:
    """A torchvision feature extractor over a batch-norm network validates in eval mode."""

    feature_extraction = pytest.importorskip("torchvision.models.feature_extraction")
    torch.manual_seed(0)
    net = nn.Sequential(
        nn.Conv2d(3, 4, 3, padding=1),
        nn.BatchNorm2d(4),
        nn.ReLU(inplace=True),
        nn.Conv2d(4, 4, 3, padding=1),
        nn.BatchNorm2d(4),
        nn.ReLU(inplace=True),
    )
    with torch.no_grad():
        net(torch.randn(4, 3, 6, 6) * 3 + 1)
    extractor = feature_extraction.create_feature_extractor(
        net, return_nodes={"2": "first", "5": "second"}
    ).eval()
    assert tl.validate(extractor, torch.randn(1, 3, 6, 6), seed=0) is True
