"""Phase 12b stretch-tier bridge contract tests."""

from __future__ import annotations

import builtins
import sys
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
    log = tl.log_forward_pass(model, x, layers_to_save="all")
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

            assert inputs is self.background
            return torch.zeros_like(inputs)

    model, x, log = _bridge_log()
    monkeypatch.setitem(sys.modules, "shap", _module("shap", DeepExplainer=FakeDeepExplainer))

    payload = tl.bridge.shap.explain(log, background=x)

    assert payload["schema"] == "torchlens.shap.v1"
    assert payload["model"] is model
    assert payload["values"].shape == x.shape


def test_inseq_bridge_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """inseq bridge loads an attribution model and forwards attribution args."""

    class FakeAttributionModel:
        """inseq attribution fixture."""

        def attribute(
            self,
            inputs: str,
            *,
            target_texts: str | None = None,
            step_scores: list[str] | None = None,
        ) -> dict[str, Any]:
            """Return a deterministic attribution payload."""

            return {"inputs": inputs, "target_texts": target_texts, "step_scores": step_scores}

    def load_model(model_or_id: str, method: str) -> FakeAttributionModel:
        """Return a fake attribution model."""

        assert model_or_id == "tiny"
        assert method == "saliency"
        return FakeAttributionModel()

    monkeypatch.setitem(sys.modules, "inseq", _module("inseq", load_model=load_model))

    payload = tl.bridge.inseq.attribute(
        "tiny",
        "hello",
        method="saliency",
        target_texts="world",
        step_scores=["probability"],
    )

    assert payload["schema"] == "torchlens.inseq.v1"
    assert payload["method"] == "saliency"
    assert payload["attributions"]["target_texts"] == "world"


def test_steering_vectors_bridge_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """steering-vectors bridge trains from saved activation tensors."""

    def train_steering_vector(
        positive: torch.Tensor, negative: torch.Tensor | None, *, normalize: bool
    ) -> dict[str, Any]:
        """Return a deterministic steering-vector payload."""

        return {
            "shape": tuple(positive.shape),
            "has_negative": negative is not None,
            "normalize": normalize,
        }

    _model, _x, log = _bridge_log()
    monkeypatch.setitem(
        sys.modules,
        "steering_vectors",
        _module("steering_vectors", train_steering_vector=train_steering_vector),
    )

    payload = tl.bridge.steering_vectors.vector(log, "conv2d", "linear", normalize=True)

    assert payload["schema"] == "torchlens.steering_vectors.v1"
    assert payload["vector"]["has_negative"] is True
    assert payload["vector"]["normalize"] is True


def test_repeng_bridge_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """repeng bridge builds a control vector from saved activations."""

    class FakeControlVector:
        """repeng ControlVector fixture."""

        @staticmethod
        def train(
            positive: torch.Tensor, negative: torch.Tensor | None, *, rank: int
        ) -> dict[str, Any]:
            """Return a deterministic control-vector payload."""

            return {
                "rank": rank,
                "positive_shape": tuple(positive.shape),
                "negative": negative is not None,
            }

    _model, _x, log = _bridge_log()
    monkeypatch.setitem(
        sys.modules,
        "repeng",
        _module("repeng", ControlVector=FakeControlVector),
    )

    payload = tl.bridge.repeng.control_vector(log, "conv2d", "linear", rank=1)

    assert payload["schema"] == "torchlens.repeng.v1"
    assert payload["control_vector"]["rank"] == 1
    assert payload["control_vector"]["negative"] is True


def test_dialz_bridge_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """dialz bridge sends labels and activation lists to the downstream analyzer."""

    def analyze(activations: list[torch.Tensor], *, labels: list[str]) -> dict[str, Any]:
        """Return a deterministic dialz analysis payload."""

        return {"labels": labels, "count": len(activations)}

    _model, _x, log = _bridge_log()
    monkeypatch.setitem(sys.modules, "dialz", _module("dialz", analyze=analyze))

    payload = tl.bridge.dialz.analyze(log, sites=["conv2d", "linear"])

    assert payload["schema"] == "torchlens.dialz.v1"
    assert payload["labels"] == payload["result"]["labels"]
    assert payload["result"]["count"] == 2


def test_lit_bridge_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """LIT bridge returns a model wrapper with LIT-shaped methods."""

    class FakeLitModel:
        """LIT base model fixture."""

    _model, _x, log = _bridge_log()
    lit_model_module = _module("lit_nlp.api.model", Model=FakeLitModel)
    lit_api_module = _module("lit_nlp.api", model=lit_model_module)
    monkeypatch.setitem(sys.modules, "lit_nlp", _module("lit_nlp", api=lit_api_module))
    monkeypatch.setitem(sys.modules, "lit_nlp.api", lit_api_module)
    monkeypatch.setitem(sys.modules, "lit_nlp.api.model", lit_model_module)

    payload = tl.bridge.lit.model(log, name="fixture")

    assert payload["schema"] == "torchlens.lit_model.v1"
    assert payload["model"].input_spec()
    assert payload["model"].output_spec()
    assert payload["model"].predict([{}])[0]["num_layers"] == len(log.layer_list)


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

        activation = torch.ones(1, 2)

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
