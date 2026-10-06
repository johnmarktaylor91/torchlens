"""Phase 12b stretch-tier bridge contract tests."""

from __future__ import annotations

import builtins
import sys
from dataclasses import dataclass
from types import ModuleType
from typing import Any

import pytest
import torch
from torch import nn
from torch.fx import symbolic_trace

import torchlens as tl
import torchlens.compat as tl_compat
from torchlens.compat.torchextractor import Extractor


class _TinyBridgeModel(nn.Module):
    """Small model fixture for stretch bridge tests."""

    def __init__(self) -> None:
        """Initialize deterministic layers."""

        super().__init__()
        self.conv = nn.Conv2d(1, 2, kernel_size=3)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(72, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model forward pass.

        Parameters
        ----------
        x:
            Input image batch.

        Returns
        -------
        torch.Tensor
            Class logits.
        """

        return self.fc(self.flatten(self.relu(self.conv(x))))


class _NamedModuleModel(nn.Module):
    """Small model exposing named modules for compat tests."""

    def __init__(self) -> None:
        """Initialize layers."""

        super().__init__()
        self.fc1 = nn.Linear(3, 4)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model forward pass.

        Parameters
        ----------
        x:
            Input batch.

        Returns
        -------
        torch.Tensor
            Output batch.
        """

        return self.fc2(self.relu(self.fc1(x)))


def _bridge_log() -> tuple[_TinyBridgeModel, torch.Tensor, Any]:
    """Create a TorchLens log for stretch bridge contracts.

    Returns
    -------
    tuple[_TinyBridgeModel, torch.Tensor, Any]
        Model, input tensor, and log.
    """

    torch.manual_seed(120)
    model = _TinyBridgeModel().eval()
    x = torch.randn(2, 1, 8, 8)
    log = tl.trace(model, x, capture=tl.options.CaptureOptions(layers_to_save="all"))
    return model, x, log


def _module(name: str, **attrs: Any) -> ModuleType:
    """Build a fake importable module.

    Parameters
    ----------
    name:
        Module name.
    **attrs:
        Attributes to attach.

    Returns
    -------
    ModuleType
        Fake module.
    """

    module = ModuleType(name)
    for attr_name, value in attrs.items():
        setattr(module, attr_name, value)
    return module


def test_gradcam_bridge_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """Grad-CAM bridge calls the downstream CAM class with TorchLens-shaped inputs."""

    class FakeGradCAM:
        """pytorch-grad-cam fixture."""

        def __init__(self, *, model: nn.Module, target_layers: list[nn.Module]) -> None:
            """Store constructor inputs."""

            self.model = model
            self.target_layers = target_layers

        def __call__(self, *, input_tensor: torch.Tensor, targets: Any | None) -> torch.Tensor:
            """Return a deterministic CAM tensor."""

            assert targets == ["class-0"]
            return torch.ones(input_tensor.shape[0], 6, 6)

    model, x, log = _bridge_log()
    monkeypatch.setitem(
        sys.modules, "pytorch_grad_cam", _module("pytorch_grad_cam", GradCAM=FakeGradCAM)
    )

    payload = tl.bridge.gradcam.cam(log, "conv", inputs=x, targets=["class-0"])

    assert payload["schema"] == "torchlens.gradcam.v1"
    assert payload["model"] is model
    assert payload["target_layers"] == [model.conv]
    assert payload["cam"].shape == (2, 6, 6)


def test_shap_bridge_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """SHAP bridge creates an explainer and returns shap values."""

    class FakeDeepExplainer:
        """SHAP explainer fixture."""

        def __init__(self, model: nn.Module, background: torch.Tensor) -> None:
            """Store constructor inputs."""

            self.model = model
            self.background = background

        def shap_values(self, inputs: torch.Tensor) -> torch.Tensor:
            """Return deterministic SHAP values."""

            self.explained = inputs
            return torch.zeros_like(inputs)

    model, x, log = _bridge_log()
    monkeypatch.setitem(sys.modules, "shap", _module("shap", DeepExplainer=FakeDeepExplainer))
    background = torch.zeros_like(x)

    payload = tl.bridge.shap.explain(log, background=background)

    # The background goes to the constructor; the traced input is explained.
    assert payload["explainer"].background is background
    assert torch.equal(payload["explainer"].explained, x)

    assert payload["schema"] == "torchlens.shap.v1"
    assert payload["model"] is model
    assert payload["values"].shape == x.shape


def test_inseq_bridge_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """inseq bridge loads an attribution model and forwards attribution args."""

    class FakeAttributionModel:
        """inseq attribution fixture."""

        def attribute(
            self,
            input_texts: str,
            generated_texts: str | None = None,
            *,
            step_scores: list[str] | None = None,
        ) -> dict[str, Any]:
            """Return a deterministic attribution payload (inseq 0.7 keyword names)."""

            return {
                "inputs": input_texts,
                "generated_texts": generated_texts,
                "step_scores": step_scores,
            }

    loaded: list[str] = []

    def load_model(model_or_id: str, attribution_method: str) -> FakeAttributionModel:
        """Return a fake attribution model."""

        assert model_or_id == "tiny"
        loaded.append(attribution_method)
        return FakeAttributionModel()

    monkeypatch.setitem(sys.modules, "inseq", _module("inseq", load_model=load_model))

    payload = tl.bridge.inseq.attribute(
        "tiny",
        "hello",
        method="saliency",
        generated_texts="world",
        step_scores=["probability"],
    )

    assert payload["schema"] == "torchlens.inseq.v1"
    assert payload["method"] == "saliency"
    assert payload["attributions"]["generated_texts"] == "world"
    # The default method is inseq's own spelling, not "integrated_grads".
    assert tl.bridge.inseq.attribute("tiny", "hello")["method"] == "integrated_gradients"
    assert loaded == ["saliency", "integrated_gradients"]


def _contrastive_logs() -> tuple[Any, Any]:
    """Trace the tiny model on a positive and a negative batch of two inputs.

    Returns
    -------
    tuple[Any, Any]
        Positive and negative logs; ``"linear"`` outs are ``[2, 2]`` rows.
    """

    torch.manual_seed(121)
    model = _TinyBridgeModel().eval()
    options = tl.options.CaptureOptions(layers_to_save="all")
    log_pos = tl.trace(model, torch.randn(2, 1, 8, 8) + 1.0, capture=options)
    log_neg = tl.trace(model, torch.randn(2, 1, 8, 8), capture=options)
    return log_pos, log_neg


def test_steering_vectors_bridge_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """steering-vectors bridge aggregates contrastive rows from two traces."""

    @dataclass
    class SteeringVector:
        """steering_vectors.SteeringVector fixture."""

        layer_activations: dict[int, torch.Tensor]
        layer_type: str = "decoder_block"

    def mean_aggregator() -> Any:
        """Return the package's mean aggregator shape."""

        return lambda pos, neg: (pos - neg).mean(dim=0)

    log_pos, log_neg = _contrastive_logs()
    monkeypatch.setitem(
        sys.modules,
        "steering_vectors",
        _module("steering_vectors", mean_aggregator=mean_aggregator, SteeringVector=SteeringVector),
    )

    payload = tl.bridge.steering_vectors.vector(
        log_pos, "linear", negative_log=log_neg, read_token_index=None, layer=3
    )

    expected = (payload["positive"] - payload["negative"]).mean(dim=0)
    assert payload["positive"].shape == (2, 2)
    assert payload["schema"] == "torchlens.steering_vectors.v1"
    assert torch.equal(payload["vector"], expected)
    assert torch.equal(payload["steering_vector"].layer_activations[3], expected)


def _assert_signed_direction(
    direction: Any, positive: torch.Tensor, negative: torch.Tensor
) -> None:
    """Assert a unit direction that projects positives above negatives."""

    import numpy as np

    assert direction.dtype == np.float32
    assert np.isclose(np.linalg.norm(direction), 1.0, atol=1e-5)
    projected_pos = positive.numpy() @ direction
    projected_neg = negative.numpy() @ direction
    assert (projected_pos > projected_neg).mean() >= 0.5


def test_repeng_bridge_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """repeng bridge wraps the PCA direction of saved rows in a ControlVector."""

    pytest.importorskip("sklearn")

    @dataclass
    class ControlVector:
        """repeng.ControlVector fixture."""

        model_type: str
        directions: dict[int, Any]

    log_pos, log_neg = _contrastive_logs()
    monkeypatch.setitem(sys.modules, "repeng", _module("repeng", ControlVector=ControlVector))

    payload = tl.bridge.repeng.control_vector(
        log_pos,
        "linear",
        negative_log=log_neg,
        layer=2,
        read_token_index=None,
        model_type="tiny",
    )

    vector = payload["control_vector"]
    assert payload["schema"] == "torchlens.repeng.v1"
    assert vector.model_type == "tiny"
    assert list(vector.directions) == [2]
    _assert_signed_direction(vector.directions[2], payload["positive"], payload["negative"])
    with pytest.raises(ValueError, match="model_type="):
        tl.bridge.repeng.control_vector(
            log_pos, "linear", negative_log=log_neg, layer=2, read_token_index=None
        )


def test_dialz_bridge_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """dialz bridge follows the installed default method and builds a SteeringVector."""

    pytest.importorskip("sklearn")

    @dataclass
    class SteeringVector:
        """dialz.SteeringVector fixture."""

        model_type: str
        directions: dict[int, Any]

    def read_representations(model: Any, tokenizer: Any, inputs: Any, method: str = "pca") -> Any:
        """dialz 1.x signature shape: the default method name is read from here."""

        raise AssertionError("the bridge never runs the model")

    log_pos, log_neg = _contrastive_logs()
    vector_module = _module("dialz.vector", read_representations=read_representations)
    monkeypatch.setitem(
        sys.modules,
        "dialz",
        _module("dialz", SteeringVector=SteeringVector, vector=vector_module),
    )

    payload = tl.bridge.dialz.vector(
        log_pos, "linear", negative_log=log_neg, layer=1, read_token_index=None, model_type="t"
    )
    mean = tl.bridge.dialz.vector(
        log_pos,
        "linear",
        negative_log=log_neg,
        layer=1,
        read_token_index=None,
        model_type="t",
        method="mean_diff",
    )

    assert payload["schema"] == "torchlens.dialz.v2"
    _assert_signed_direction(
        payload["steering_vector"].directions[1], payload["positive"], payload["negative"]
    )
    expected_mean = (payload["positive"] - payload["negative"]).numpy().mean(axis=0)
    got_mean = mean["steering_vector"].directions[1]
    # read_representations flips the sign when most pairs project the wrong way.
    assert (got_mean == expected_mean).all() or (got_mean == -expected_mean).all()


# The former mock-based LIT contract test is deliberately GONE (lane F31,
# M(lit) item 2 "stop lying"): it mocked the entire lit_nlp module tree, so it
# passed green while real LIT 1.3.1 rejected the shipped wrapper at LitApp
# construction (round-2 card VQ1). The replacement contract tests live in
# tests/test_lit_bridge_*.py and run against REAL lit_nlp types.


def test_compat_shims_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """compat-shims helpers return migration payloads without real downstream packages."""

    monkeypatch.setitem(sys.modules, "torchextractor", _module("torchextractor"))
    monkeypatch.setitem(sys.modules, "sentence_transformers", _module("sentence_transformers"))
    model = _NamedModuleModel().eval()

    extractor = tl_compat.from_torchextractor(model, ["fc1"])
    fx_payload = tl_compat.from_fx(symbolic_trace(model))
    sentence_payload = tl_compat.from_sentence_transformers(model, prompt="query")

    assert isinstance(extractor, Extractor)
    assert extractor.layers == ["fc1"]
    assert fx_payload["schema"] == "torchlens.fx_migration.v1"
    assert "fc1" in sentence_payload["layers"]
    assert sentence_payload["prompt"] == "query"


def test_from_ilg_contract_and_optional_dep_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """ILG shim inverts return_layers and raises the documented optional-dep error."""

    model = _NamedModuleModel().eval()
    monkeypatch.setitem(sys.modules, "torchvision", _module("torchvision"))

    extractor = tl_compat.from_ilg(model, {"fc1": "features"})

    assert isinstance(extractor, Extractor)
    assert extractor.layers == {"features": "fc1"}

    real_import = builtins.__import__

    def fail_torchvision(name: str, *args: Any, **kwargs: Any) -> Any:
        """Raise ImportError for torchvision only."""

        if name == "torchvision":
            raise ImportError("missing torchvision")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "torchvision", raising=False)
    monkeypatch.setattr(builtins, "__import__", fail_torchvision)
    with pytest.raises(ImportError, match=r"torchlens\[vision-shims\]"):
        tl_compat.from_ilg(model, {"fc1": "features"})


def test_viz_compat_contracts(monkeypatch: pytest.MonkeyPatch) -> None:
    """viz compat adapters forward TorchLens tensor payloads to mocked viz packages."""

    class LayerFixture:
        """Layer-like object fixture."""

        out = torch.ones(1, 2)

    def show(tensor: torch.Tensor, *, title: str) -> dict[str, Any]:
        """Return a deterministic torchshow payload."""

        return {"shape": tuple(tensor.shape), "title": title}

    def lovely(tensor: torch.Tensor) -> str:
        """Return a deterministic lovely-tensors payload."""

        return f"lovely:{tuple(tensor.shape)}"

    monkeypatch.setitem(sys.modules, "torchshow", _module("torchshow", show=show))
    monkeypatch.setitem(sys.modules, "lovely_tensors", _module("lovely_tensors", lovely=lovely))

    assert tl_compat.torchshow.show(LayerFixture(), title="layer") == {
        "shape": (1, 2),
        "title": "layer",
    }
    assert tl_compat.lovely.str(LayerFixture()) == "lovely:(1, 2)"
