"""F11 admission pinning on named real models (collapse memo item 15).

The U-gate makes admission MODE-STABLE: train-mode capture inflates raw op
count 12-40% (the round-1 "ceiling drift" measurement bug), but U is the
rendered universe, so a model admitted in eval mode stays admitted in train
mode. Pinned on named torchvision models, both modes, random init, offline.
"""

from __future__ import annotations

import warnings as warnings_module

import pytest
import torch

from torchlens.visualization import collapse_optimizer
from torchlens.visualization.collapse_plan import RenderContext

torchvision = pytest.importorskip("torchvision")

import torchlens as tl  # noqa: E402


def _admitted_planner(model: torch.nn.Module) -> str:
    """Capture and return the max-mode planner tier for ``model``."""

    trace = tl.trace(model, torch.randn(1, 3, 64, 64))
    with warnings_module.catch_warnings():
        warnings_module.simplefilter("ignore")
        result = collapse_optimizer.select_collapse_plan(trace, RenderContext(), mode="max")
    return result.planner


@pytest.mark.heavy
@pytest.mark.parametrize("mode", ["eval", "train"])
def test_resnet18_admitted_in_both_modes(mode: str) -> None:
    """resnet18 stays quality-planner-admitted in eval AND train mode."""

    model = torchvision.models.resnet18(weights=None)
    model.eval() if mode == "eval" else model.train()
    planner = _admitted_planner(model)
    assert planner not in {"linear_fallback", "floor_fallback"}, (
        f"resnet18 [{mode}] must stay admitted; got {planner}"
    )


@pytest.mark.slow
@pytest.mark.parametrize("mode", ["eval", "train"])
def test_densenet201_admitted_in_both_modes(mode: str) -> None:
    """densenet201's admission is mode-stable under the U-gate.

    The historical raw-op ceiling REFUSED train-mode densenet201 (2,120 raw
    ops) while admitting eval mode (1,517) at the same rendered universe
    (U=713 both modes) -- the exact defect memo D5 names. Under the U-gate
    both modes admit.
    """

    model = torchvision.models.densenet201(weights=None)
    model.eval() if mode == "eval" else model.train()
    trace = tl.trace(model, torch.randn(1, 3, 64, 64))
    with warnings_module.catch_warnings():
        warnings_module.simplefilter("ignore")
        result = collapse_optimizer.select_collapse_plan(trace, RenderContext(), mode="max")
    assert result.planner not in {"linear_fallback", "floor_fallback"}, (
        f"densenet201 [{mode}] must stay admitted; got {result.planner} "
        f"(estimator: {result.estimator})"
    )
