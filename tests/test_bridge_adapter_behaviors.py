"""Behavioral tests for dep-light bridge adapters and shared bridge helpers.

Grad-CAM and SAE Lens semantics run against minimal in-test stand-ins for the
optional packages, so the TorchLens-side resolution, dispatch, and refusal
logic executes on every environment; missing-dependency ImportError contracts
are asserted directly.
"""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.bridge import _utils as bridge_utils


class _ConvNet(nn.Module):
    """Small conv model with addressable modules."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 2, kernel_size=3, padding=1)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


@pytest.fixture()
def conv_log():
    """Freshly captured trace with a live source model."""

    torch.manual_seed(21)
    model = _ConvNet()
    log = tl.trace(model, torch.randn(1, 1, 4, 4))
    # Keep the model alive for the duration of the test via the log fixture.
    log._test_model_keepalive = model
    return log


@pytest.mark.smoke
def test_bridge_namespace_lazily_imports_and_rejects_unknown_names() -> None:
    """torchlens.bridge exposes adapters lazily and refuses unknown attributes."""

    import torchlens.bridge as bridge

    module = bridge.profiler
    assert module.__name__ == "torchlens.bridge.profiler"
    assert "gradcam" in dir(bridge)
    with pytest.raises(AttributeError, match="no attribute"):
        _ = bridge.does_not_exist


@pytest.mark.smoke
def test_bridge_utils_error_contracts(conv_log) -> None:
    """source_model/out_at/first_input_tensor fail with actionable errors."""

    with pytest.raises(ValueError, match="live source model"):
        bridge_utils.source_model(SimpleNamespace())

    with pytest.raises(ValueError, match="does not have a saved tensor out"):
        bridge_utils.out_at(conv_log, SimpleNamespace(out="not a tensor", layer_label="fake_1_1"))

    with pytest.raises(ValueError, match="tensor input"):
        bridge_utils.first_input_tensor(SimpleNamespace(layer_list=[]))

    layers = bridge_utils.tensor_layers(conv_log)
    assert layers
    assert all(isinstance(layer.out, torch.Tensor) for layer in layers)
    assert not any(layer.is_input for layer in layers)


@pytest.mark.smoke
def test_gradcam_layer_resolves_self_addresses_and_site_fallback(conv_log) -> None:
    """layer() honors 'self', module addresses, resolved sites, and refusals."""

    from torchlens.bridge import gradcam

    model = bridge_utils.source_model(conv_log)
    assert gradcam.layer(conv_log, "self") is model
    assert gradcam.layer(conv_log, "conv") is model.conv
    # A layer-label site falls back to the module the op executed in.
    assert gradcam.layer(conv_log, "conv2d_1_1") is model.conv

    with pytest.raises(ValueError, match="Could not resolve"):
        gradcam.layer(conv_log, conv_log["input_1"])


@pytest.mark.smoke
def test_gradcam_cam_runs_context_manager_runner_with_resolved_inputs(
    conv_log, monkeypatch: pytest.MonkeyPatch
) -> None:
    """cam() builds the runner from the resolved model/layer and captured input."""

    from torchlens.bridge import gradcam

    calls: dict[str, Any] = {}

    class _StubCam:
        def __init__(self, *, model: Any, target_layers: list[Any]) -> None:
            calls["model"] = model
            calls["target_layers"] = target_layers

        def __enter__(self) -> _StubCam:
            calls["entered"] = True
            return self

        def __exit__(self, *exc_info: Any) -> None:
            calls["exited"] = True

        def __call__(self, *, input_tensor: Any, targets: Any) -> str:
            calls["input_tensor"] = input_tensor
            calls["targets"] = targets
            return "cam-output"

    monkeypatch.setitem(sys.modules, "pytorch_grad_cam", SimpleNamespace(GradCAM=_StubCam))

    payload = gradcam.cam(conv_log, "conv")

    assert payload["schema"] == "torchlens.gradcam.v1"
    assert payload["cam"] == "cam-output"
    assert calls["model"] is bridge_utils.source_model(conv_log)
    assert calls["target_layers"] == [calls["model"].conv]
    assert torch.equal(calls["input_tensor"], conv_log["input_1"].out)
    assert calls["entered"] and calls["exited"]


def test_gradcam_missing_dependency_names_extra(conv_log, monkeypatch: pytest.MonkeyPatch) -> None:
    """cam() raises ImportError naming the gradcam extra when the package is absent."""

    from torchlens.bridge import gradcam

    monkeypatch.setitem(sys.modules, "pytorch_grad_cam", None)
    with pytest.raises(ImportError, match="torchlens\\[gradcam\\]"):
        gradcam.cam(conv_log, "conv")


@pytest.mark.smoke
def test_sae_lens_bridge_dispatches_encode_decode_and_refuses_junk(
    conv_log, monkeypatch: pytest.MonkeyPatch
) -> None:
    """encode()/decode() use the SAE's methods, fall back to callables, else refuse."""

    from torchlens.bridge import sae_lens as sae_bridge

    monkeypatch.setitem(sys.modules, "sae_lens", SimpleNamespace())
    site_out = conv_log["relu_1_2"].out

    class _Sae:
        def encode(self, value: torch.Tensor) -> torch.Tensor:
            return value * 2

        def decode(self, value: torch.Tensor) -> torch.Tensor:
            return value * 3

    encoded = sae_bridge.encode(conv_log, "relu_1_2", _Sae())
    assert torch.equal(encoded, site_out * 2)
    decoded = sae_bridge.decode(conv_log, "relu_1_2", _Sae())
    assert torch.equal(decoded, site_out * 3)

    called_with: list[torch.Tensor] = []
    fallback = sae_bridge.encode(conv_log, "relu_1_2", called_with.append)
    assert fallback is None and torch.equal(called_with[0], site_out)

    with pytest.raises(TypeError, match="encode"):
        sae_bridge.encode(conv_log, "relu_1_2", object())

    monkeypatch.setitem(sys.modules, "sae_lens", None)
    with pytest.raises(ImportError, match="torchlens\\[sae\\]"):
        sae_bridge.encode(conv_log, "relu_1_2", _Sae())


def test_profiler_execution_trace_writes_layer_nodes(conv_log, tmp_path) -> None:
    """execution_trace() writes the documented per-layer node schema."""

    from torchlens.bridge import profiler

    destination = tmp_path / "exec_trace.json"
    payload = profiler.execution_trace(conv_log, destination)

    on_disk = json.loads(destination.read_text(encoding="utf-8"))
    assert on_disk == payload
    assert payload["schema"] == "torchlens.execution_trace.v1"
    assert len(payload["nodes"]) == len(conv_log.layer_list)
    by_name = {node["name"]: node for node in payload["nodes"]}
    assert "conv2d_1_1" in by_name["relu_1_2"]["inputs"]
