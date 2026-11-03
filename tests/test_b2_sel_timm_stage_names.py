"""U26.1 acceptance: every timm stage name in both wrapper types, plus ``fc``, resolves.

Executed with timm installed (it is in the ``test`` and ``hf`` extras): resnet18
``features_only`` builds a ``FeatureListNet`` (5 stages), vit_tiny ``features_only``
builds a ``FeatureGetterNet`` (3 stages, names relative to its ``model`` child), and
resnet50's ``fc`` is the classifier whose output is the model output. Each name
resolves to the module's own forward output at ``tl.extract``, ``tl.pluck``,
``trace[...]``, ``tl.trace(save=[...])`` and ``trace.run(until=...)``.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from torch import nn

import torchlens as tl

timm = pytest.importorskip("timm")

pytestmark = pytest.mark.heavy


def _module_output(model: nn.Module, module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    store: dict[str, torch.Tensor] = {}
    handle = module.register_forward_hook(
        lambda _m, _i, out: store.setdefault("out", out.detach().clone())
    )
    with torch.no_grad():
        model(x)
    handle.remove()
    return store["out"]


def _until_outcome(trace: tl.Trace, x: torch.Tensor, site: str) -> tuple:
    """Where an ``until=`` run stopped, or the typed refusal it raised.

    The until= door must treat a stage name exactly as the layer label it names;
    eval-mode BatchNorm prefixes currently refuse ``buffer_sink_routing_unsupported``
    (lane RUN-A's bug 1), and the refusal must then be identical too.
    """

    try:
        truncation = trace.run(inputs=x, until=site).report.truncation
    except (RuntimeError, TypeError, ValueError) as exc:  # the outcome under comparison
        return ("refused", type(exc).__name__, getattr(exc, "fields", {}).get("code"))
    if truncation is None:
        return ("full run",)
    return (truncation.stopped_at, truncation.executed_count)


def _feature_list() -> tuple[nn.Module, torch.Tensor, Callable[[nn.Module, str], nn.Module]]:
    model = timm.create_model("resnet18", pretrained=False, features_only=True)
    return model, torch.randn(1, 3, 64, 64), lambda m, name: m.get_submodule(name)


def _feature_getter() -> tuple[nn.Module, torch.Tensor, Callable[[nn.Module, str], nn.Module]]:
    model = timm.create_model("vit_tiny_patch16_224", pretrained=False, features_only=True)
    return model, torch.randn(1, 3, 224, 224), lambda m, name: m.model.get_submodule(name)


_WRAPPERS = {
    "FeatureListNet": (_feature_list, 5),
    "FeatureGetterNet": (_feature_getter, 3),
}


@pytest.mark.parametrize("wrapper", sorted(_WRAPPERS))
def test_every_timm_stage_name_resolves_at_every_door(wrapper: str) -> None:
    build, stage_count = _WRAPPERS[wrapper]
    torch.manual_seed(0)
    model, x, owner = build()
    model.eval()
    assert type(model).__name__ == wrapper
    names = list(model.feature_info.module_name())
    assert len(names) == stage_count
    full = tl.trace(model, x)
    selective = tl.trace(model, x, save=names)
    extracted = tl.extract(model, x, {name: name for name in names})
    for name in names:
        expected = _module_output(model, owner(model, name), x)
        torch.testing.assert_close(extracted[name], expected)
        torch.testing.assert_close(tl.pluck(model, x, name), expected)
        torch.testing.assert_close(full[name].out, expected)
        torch.testing.assert_close(selective[name].out, expected)
        assert _until_outcome(full, x, name) == _until_outcome(full, x, full[name].layer_label)


@pytest.mark.parametrize("source", ["torchvision", "timm"])
def test_resnet50_fc_resolves_to_the_classifier_output(source: str) -> None:
    torch.manual_seed(0)
    if source == "timm":
        model = timm.create_model("resnet50", pretrained=False)
    else:
        torchvision = pytest.importorskip("torchvision")
        model = torchvision.models.resnet50(weights=None)
    model.eval()
    x = torch.randn(1, 3, 64, 64)
    expected = _module_output(model, model.fc, x)
    torch.testing.assert_close(tl.extract(model, x, ["fc"])["fc"], expected)
    torch.testing.assert_close(tl.trace(model, x, save=["fc"])["fc"].out, expected)
    trace = tl.trace(model, x)
    torch.testing.assert_close(trace["fc"].out, expected)
    assert _until_outcome(trace, x, "fc") == _until_outcome(trace, x, trace["fc"].layer_label)
