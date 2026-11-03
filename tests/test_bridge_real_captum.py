"""Captum ``layer()`` against the REAL captum package.

``tl.bridge.captum.layer`` shares Grad-CAM's resolver when a single-call
module returns the site's tensor; every other site (a functional op, one pass
of a repeated module) becomes a probe module that ``attribute()`` splices into
the forward, so Captum's layer methods hook exactly that tensor. Each result is
held to a hand-written hook reference and, where one exists, to Captum on the
module named directly.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._errors import InvalidArgumentError

captum_attr = pytest.importorskip("captum.attr")
torchvision = pytest.importorskip("torchvision")

pytestmark = [pytest.mark.optional]

_TARGET = 3
_N_STEPS = 4
# Captum's default approximation; its ``method=`` kwarg cannot travel through
# ``tl.bridge.captum.attribute``, whose second parameter is also ``method``.
_IG_METHOD = "gausslegendre"


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


def _module_calls(layer: Any) -> set[str]:
    """Return the pass-qualified module calls whose output an op's tensor is."""

    calls = set()
    for call in getattr(layer, "output_of_module_calls", ()) or ():
        calls.add(f"{call[0]}:{call[1]}" if isinstance(call, tuple) else str(call))
    return calls


@pytest.fixture(scope="module")
def resnet() -> Iterator[tuple[torch.nn.Module, torch.Tensor, Any]]:
    """Yield a seeded resnet18, its input, and an all-saved trace; clean up after."""

    torch.manual_seed(0)
    model = torchvision.models.resnet18(weights=None).eval()
    x = torch.randn(2, 3, 64, 64)
    log = tl.trace(model, x, save="all")
    try:
        yield model, x, log
    finally:
        log.cleanup()


def _run_hooked(
    model: nn.Module,
    x: torch.Tensor,
    module: nn.Module,
    call: int,
    where: str,
    replacement: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run ``model(x)``; read (and optionally replace) one tensor at ``module``'s ``call``-th call.

    ``where`` is ``"input"`` (the call's first positional input) or ``"output"``.
    Returns the model output and the tensor read (or the replacement), which
    the rest of the forward consumed through a clone.
    """

    state: dict[str, Any] = {"n": 0, "tensor": None}

    # The module downstream of the read tensor gets a clone: resnet's basic
    # block applies ``nn.ReLU(inplace=True)`` to the residual add, which would
    # otherwise overwrite the tensor read here before it is used.
    def pre(_mod: nn.Module, args: tuple[Any, ...]) -> tuple[Any, ...] | None:
        state["n"] += 1
        if state["n"] != call or where != "input":
            return None
        state["tensor"] = args[0] if replacement is None else replacement
        return (state["tensor"].clone(), *args[1:])

    def post(_mod: nn.Module, _args: tuple[Any, ...], out: torch.Tensor) -> torch.Tensor | None:
        if state["n"] != call or where != "output":
            return None
        state["tensor"] = out if replacement is None else replacement
        return state["tensor"].clone()

    handles = [module.register_forward_pre_hook(pre), module.register_forward_hook(post)]
    try:
        y = model(x)
    finally:
        for handle in handles:
            handle.remove()
    assert state["tensor"] is not None, f"call {call} of the module never ran"
    return y, state["tensor"]


def _gradcam_by_hand(
    model: nn.Module, x: torch.Tensor, module: nn.Module, call: int, where: str
) -> torch.Tensor:
    """Grad-CAM at one module call's tensor, written out from the definition."""

    y, act = _run_hooked(model, x, module, call, where)
    grad = torch.autograd.grad(y[:, _TARGET].sum(), act)[0]
    return (grad.mean(dim=(2, 3), keepdim=True) * act).sum(dim=1, keepdim=True)


def _lig_by_hand(
    model: nn.Module, x: torch.Tensor, module: nn.Module, call: int, where: str
) -> torch.Tensor:
    """Layer integrated gradients at one module call's tensor, written out from the definition."""

    from captum.attr._utils.approximation_methods import approximation_parameters

    step_sizes, alphas = approximation_parameters(_IG_METHOD)
    act = _run_hooked(model, x, module, call, where)[1].detach()
    base = _run_hooked(model, torch.zeros_like(x), module, call, where)[1].detach()
    total = torch.zeros_like(act)
    for step, alpha in zip(step_sizes(_N_STEPS), alphas(_N_STEPS), strict=True):
        scaled = (base + alpha * (act - base)).requires_grad_(True)
        y, _ = _run_hooked(model, x, module, call, where, replacement=scaled)
        total = total + step * torch.autograd.grad(y[:, _TARGET].sum(), scaled)[0]
    return total * (act - base)


def _bridged(log: Any, method_cls: Any, site: Any, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    """Attribute through the bridge at ``site`` with ``method_cls``."""

    model = tl.bridge.captum.source_model(log)
    layer = tl.bridge.captum.layer(
        log, site, **{k: kwargs.pop(k) for k in ("pass_index",) if k in kwargs}
    )
    method = method_cls(model, layer)
    return tl.bridge.captum.attribute(log, method, _TARGET, inputs=x, **kwargs)


def _add_label(log: Any) -> str:
    """Return the label of the residual add in ``layer4.1`` (the second relu call's parent)."""

    second_relu = [layer for layer in log._layer_list if "layer4.1.relu:2" in _module_calls(layer)]
    assert len(second_relu) == 1, second_relu
    parents = list(second_relu[0].parents)
    assert len(parents) == 1, parents
    label = str(parents[0])
    assert log[label].output_of_module_calls in ((), [], None), "the add is not a module output"
    return label


def test_label_site_layer_gradcam_matches_direct(resnet) -> None:
    """LayerGradCam on layer(log, label) equals LayerGradCam on model.layer4 exactly."""

    model, x, log = resnet
    label = [layer.layer_label for layer in log._layer_list if _outermost(layer) == "layer4"][-1]
    resolved = tl.bridge.captum.layer(log, label)
    assert resolved is model.layer4
    direct = captum_attr.LayerGradCam(model, model.layer4).attribute(x, target=_TARGET)
    bridged = tl.bridge.captum.attribute(
        log, captum_attr.LayerGradCam(model, resolved), _TARGET, inputs=x
    )
    assert torch.equal(bridged, direct)


def test_functional_add_site_is_a_captum_layer(resnet) -> None:
    """A residual add (no module returns it) attributes like a hand-written hook at that tensor."""

    model, x, log = resnet
    label = _add_label(log)
    probe = tl.bridge.captum.layer(log, label)
    assert isinstance(probe, nn.Module) and probe not in set(model.modules())
    relu = model.layer4[1].relu
    cam = _bridged(log, captum_attr.LayerGradCam, label, x)
    torch.testing.assert_close(cam, _gradcam_by_hand(model, x, relu, 2, "input"))
    lig = _bridged(log, captum_attr.LayerIntegratedGradients, label, x, n_steps=_N_STEPS)
    torch.testing.assert_close(lig, _lig_by_hand(model, x, relu, 2, "input"))
    assert lig.shape == cam.shape[:1] + model.layer4[1].conv2.weight.shape[:1] + cam.shape[2:]
    # The spliced forward left the model unhooked and its plain output unchanged.
    assert not relu._forward_hooks and not relu._forward_pre_hooks
    torch.testing.assert_close(model(x), log.output_ops[0].out)


def test_pass_index_picks_one_call_of_a_repeated_module(resnet) -> None:
    """``pass_index`` names one call of layer4.1's relu; each pass matches its own reference."""

    model, x, log = resnet
    relu = model.layer4[1].relu
    first = tl.bridge.captum.layer(log, "layer4.1.relu", pass_index=1)
    assert first is not relu
    assert repr(first) == repr(tl.bridge.captum.layer(log, "layer4.1.relu:1"))
    cam_1 = _bridged(log, captum_attr.LayerGradCam, "layer4.1.relu", x, pass_index=1)
    torch.testing.assert_close(cam_1, _gradcam_by_hand(model, x, relu, 1, "output"))
    lig_1 = _bridged(
        log,
        captum_attr.LayerIntegratedGradients,
        "layer4.1.relu",
        x,
        pass_index=1,
        n_steps=_N_STEPS,
    )
    torch.testing.assert_close(lig_1, _lig_by_hand(model, x, relu, 1, "output"))
    # The second call returns layer4's output, so it resolves to layer4 itself
    # and equals Captum on that module directly.
    assert tl.bridge.captum.layer(log, "layer4.1.relu", pass_index=2) is model.layer4
    cam_2 = _bridged(log, captum_attr.LayerGradCam, "layer4.1.relu", x, pass_index=2)
    direct = captum_attr.LayerGradCam(model, model.layer4).attribute(x, target=_TARGET)
    assert torch.equal(cam_2, direct)
    torch.testing.assert_close(cam_2, _gradcam_by_hand(model, x, relu, 2, "output"))
    assert not torch.allclose(cam_1, cam_2)


def test_pass_index_refusals_are_typed(resnet) -> None:
    """A bad ``pass_index`` or one on a non-module site refuses by code."""

    _model, _x, log = resnet
    with pytest.raises(InvalidArgumentError) as info:
        tl.bridge.captum.layer(log, "layer4.1.relu", pass_index=0)
    assert info.value.fields["code"] == "bridge_pass_index_invalid"
    with pytest.raises(InvalidArgumentError) as info:
        tl.bridge.captum.layer(log, _add_label(log), pass_index=1)
    assert info.value.fields["code"] == "bridge_pass_index_needs_module_address"
