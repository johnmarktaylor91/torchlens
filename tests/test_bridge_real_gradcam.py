"""Grad-CAM bridge against the REAL pytorch-grad-cam package.

Every CAM the bridge returns is compared bit-for-bit with the package called
directly on the same module. A TorchLens op label resolves to the outermost
module whose output its tensor is (``layer4`` for the last relu of a
resnet18), a label whose module runs more than once is refused, and call
options (``eigen_smooth``, ``aug_smooth``) reach the CAM call.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

import torchlens as tl
from torchlens._errors import InvalidArgumentError

pytorch_grad_cam = pytest.importorskip("pytorch_grad_cam")
torchvision = pytest.importorskip("torchvision")

pytestmark = [pytest.mark.optional, pytest.mark.heavy]


def _address(call: Any) -> str:
    """Return the module address of one ``output_of_module_calls`` entry."""

    if isinstance(call, tuple):
        return str(call[0])
    address, _, suffix = str(call).rpartition(":")
    return address if suffix.isdigit() else str(call)


def _outermost(layer: Any) -> str | None:
    """Return the outermost module address an op's tensor is the output of."""

    addresses = [_address(call) for call in getattr(layer, "output_of_module_calls", ()) or ()]
    return min(addresses, key=lambda a: a.count(".")) if addresses else None


@pytest.fixture(scope="module")
def resnet() -> tuple[torch.nn.Module, torch.Tensor, Any]:
    """Return a seeded resnet18, its input, and an all-saved trace."""

    torch.manual_seed(0)
    model = torchvision.models.resnet18(weights=None).eval()
    x = torch.randn(2, 3, 64, 64)
    log = tl.trace(model, x, capture=tl.options.CaptureOptions(layers_to_save="all"))
    return model, x, log


def _direct(model: torch.nn.Module, x: torch.Tensor, **call: Any) -> np.ndarray:
    """Run GradCAM directly on ``model.layer4``."""

    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    with pytorch_grad_cam.GradCAM(model=model, target_layers=[model.layer4]) as runner:
        return runner(input_tensor=x, targets=[ClassifierOutputTarget(3)] * 2, **call)


def _targets() -> list[Any]:
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    return [ClassifierOutputTarget(3)] * 2


def _layer4_label(log: Any) -> str:
    """Return the label of the last op whose outermost module is ``layer4``."""

    labels = [layer.layer_label for layer in log.layer_list if _outermost(layer) == "layer4"]
    assert labels, "resnet18 trace has no op that is layer4's output"
    return labels[-1]


def test_label_site_resolves_to_the_outermost_module(resnet) -> None:
    """The label of layer4's output tensor resolves to layer4 itself."""

    model, _x, log = resnet
    label = _layer4_label(log)
    assert tl.bridge.gradcam.layer(log, label) is model.layer4
    assert tl.bridge.gradcam.layer(log, "layer4") is model.layer4


def test_label_and_address_cams_are_bit_identical_to_direct(resnet) -> None:
    """Address and label sites give exactly GradCAM(target_layers=[m.layer4])."""

    model, x, log = resnet
    direct = _direct(model, x)
    by_address = tl.bridge.gradcam.cam(log, "layer4", inputs=x, targets=_targets())["cam"]
    by_label = tl.bridge.gradcam.cam(log, _layer4_label(log), inputs=x, targets=_targets())["cam"]
    assert np.max(np.abs(by_address - direct)) == 0.0
    assert np.max(np.abs(by_label - direct)) == 0.0


def test_call_options_reach_the_cam_call(resnet) -> None:
    """eigen_smooth/aug_smooth go to the CAM call and match the package exactly."""

    model, x, log = resnet
    for options in ({"eigen_smooth": True}, {"aug_smooth": True}):
        direct = _direct(model, x, **options)
        bridged = tl.bridge.gradcam.cam(log, "layer4", inputs=x, targets=_targets(), **options)
        assert np.max(np.abs(bridged["cam"] - direct)) == 0.0, options


def test_constructor_options_still_reach_the_constructor(resnet) -> None:
    """A keyword GradCAM's constructor names (reshape_transform) is not sent to the call."""

    model, x, log = resnet
    seen: list[torch.Tensor] = []

    def reshape(tensor: torch.Tensor) -> torch.Tensor:
        seen.append(tensor)
        return tensor

    tl.bridge.gradcam.cam(log, "layer4", inputs=x, targets=_targets(), reshape_transform=reshape)
    assert seen


def test_label_of_a_module_that_runs_twice_is_refused(resnet) -> None:
    """A basic block's relu runs twice; a label naming one of its calls refuses typed."""

    _model, _x, log = resnet
    labels = [layer.layer_label for layer in log.layer_list if _outermost(layer) == "layer4.1.relu"]
    assert labels, "expected an op whose outermost module is layer4.1.relu"
    with pytest.raises(InvalidArgumentError) as info:
        tl.bridge.gradcam.layer(log, labels[0])
    assert info.value.fields["code"] == "bridge_module_site_multi_call"
    with pytest.raises(InvalidArgumentError) as info:
        tl.bridge.gradcam.layer(log, "layer4.1.relu:2")
    assert info.value.fields["code"] == "bridge_module_site_multi_call"
