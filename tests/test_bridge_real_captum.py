"""Captum ``layer()`` against the REAL captum package.

``tl.bridge.captum.layer`` shares Grad-CAM's resolver: an op label resolves to
the outermost module whose output its tensor is, and LayerGradCam on that
module is bit-identical to LayerGradCam on the module named directly.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

import torchlens as tl
from torchlens._errors import InvalidArgumentError

captum_attr = pytest.importorskip("captum.attr")
torchvision = pytest.importorskip("torchvision")

pytestmark = [pytest.mark.optional]


def _outermost(layer: Any) -> str | None:
    """Return the outermost module address an op's tensor is the output of."""

    addresses = []
    for call in getattr(layer, "output_of_module_calls", ()) or ():
        if isinstance(call, tuple):
            addresses.append(str(call[0]))
        else:
            address, _, suffix = str(call).rpartition(":")
            addresses.append(address if suffix.isdigit() else str(call))
    return min(addresses, key=lambda a: a.count(".")) if addresses else None


@pytest.fixture(scope="module")
def resnet() -> tuple[torch.nn.Module, torch.Tensor, Any]:
    """Return a seeded resnet18, its input, and an all-saved trace."""

    torch.manual_seed(0)
    model = torchvision.models.resnet18(weights=None).eval()
    x = torch.randn(2, 3, 64, 64)
    log = tl.trace(model, x, capture=tl.options.CaptureOptions(layers_to_save="all"))
    return model, x, log


def test_label_site_layer_gradcam_matches_direct(resnet) -> None:
    """LayerGradCam on layer(log, label) equals LayerGradCam on model.layer4 exactly."""

    model, x, log = resnet
    label = [layer.layer_label for layer in log.layer_list if _outermost(layer) == "layer4"][-1]
    resolved = tl.bridge.captum.layer(log, label)
    assert resolved is model.layer4
    direct = captum_attr.LayerGradCam(model, model.layer4).attribute(x, target=3)
    bridged = tl.bridge.captum.attribute(
        log, captum_attr.LayerGradCam(model, resolved), 3, inputs=x
    )
    assert torch.equal(bridged, direct)


def test_label_of_a_module_that_runs_twice_is_refused(resnet) -> None:
    """A label whose module runs twice refuses typed instead of hooking both calls."""

    _model, _x, log = resnet
    label = [layer.layer_label for layer in log.layer_list if _outermost(layer) == "layer4.1.relu"][
        0
    ]
    with pytest.raises(InvalidArgumentError) as info:
        tl.bridge.captum.layer(log, label)
    assert info.value.fields["code"] == "bridge_module_site_multi_call"
