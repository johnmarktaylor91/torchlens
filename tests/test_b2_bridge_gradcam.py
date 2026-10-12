"""Grad-CAM bridge ``call_kwargs=`` against real pytorch-grad-cam (walk U45.3).

Smoothing options given in ``call_kwargs=`` reach the CAM call and match the
package called directly; an option given twice refuses typed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._errors import KeywordConflictError

pytorch_grad_cam = pytest.importorskip("pytorch_grad_cam")

pytestmark = [pytest.mark.optional]


class _SmallCnn(nn.Module):
    """Conv feature block plus a linear classifier."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU())
        self.head = nn.Linear(4, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pool the features and classify."""

        return self.head(self.features(x).mean(dim=(2, 3)))


def _targets() -> list[Any]:
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    return [ClassifierOutputTarget(2)] * 2


@pytest.fixture(scope="module")
def cnn() -> tuple[nn.Module, torch.Tensor, Any]:
    torch.manual_seed(0)
    model = _SmallCnn().eval()
    x = torch.randn(2, 3, 16, 16)
    return model, x, tl.trace(model, x, capture=tl.options.CaptureOptions(layers_to_save="all"))


@pytest.mark.parametrize(
    "options", [{"eigen_smooth": True}, {"aug_smooth": True}], ids=["eigen", "aug"]
)
def test_call_kwargs_reach_the_cam_call(cnn: Any, options: dict[str, Any]) -> None:
    model, x, log = cnn
    with pytorch_grad_cam.GradCAM(model=model, target_layers=[model.features]) as runner:
        direct = runner(input_tensor=x, targets=_targets(), **options)
    plain = tl.bridge.gradcam.cam(log, "features", inputs=x, targets=_targets())["cam"]
    bridged = tl.bridge.gradcam.cam(
        log, "features", inputs=x, targets=_targets(), call_kwargs=options
    )["cam"]
    assert np.max(np.abs(bridged - direct)) == 0.0
    # The option changed the result, so it was not silently dropped.
    assert np.max(np.abs(bridged - plain)) > 0.0


def test_an_option_given_twice_refuses(cnn: Any) -> None:
    _model, x, log = cnn
    with pytest.raises(KeywordConflictError) as info:
        tl.bridge.gradcam.cam(
            log,
            "features",
            inputs=x,
            targets=_targets(),
            call_kwargs={"aug_smooth": True},
            aug_smooth=False,
        )
    assert info.value.fields["code"] == "bridge_gradcam_call_option_conflict"
