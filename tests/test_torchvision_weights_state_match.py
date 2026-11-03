"""Torchvision preset recovery from a model's loaded state (no handle needed)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
import torchlens.preprocessing as pp
from torchlens._autoroute._builtin_output import semantic_output_cache_key
from torchlens._utils import torchvision_weights as tvw


class _Fake(nn.Module):
    """Stand-in for a torchvision class (module name spoofed below)."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 3)
        self.register_buffer("scale", torch.arange(3.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.fc(x) * self.scale


_Fake.__module__ = "torchvision.models._state_match_test"


def _pil_image() -> Any:
    from PIL import Image

    return Image.new("RGB", (32, 32), color=(10, 200, 30))


class _Member:
    """A weights-member double with ``url``, ``meta`` and ``transforms``."""

    def __init__(self, name: str, url: str, preset: Any, num_params: int = 15) -> None:
        self.name = name
        self.url = url
        self.meta = {"num_params": num_params, "categories": [f"c{i}" for i in range(3)]}
        self._preset = preset

    def transforms(self) -> Any:  # noqa: D102
        return self._preset

    def __str__(self) -> str:
        return self.name


@pytest.fixture
def hub_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the torch hub cache at an empty temporary directory."""

    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))
    (tmp_path / "checkpoints").mkdir()
    return tmp_path / "checkpoints"


def _plant(hub_dir: Path, filename: str, state: dict[str, torch.Tensor]) -> None:
    torch.save(state, hub_dir / filename)


def test_non_torchvision_model_never_matches(hub_dir: Path) -> None:
    """A plain module is not consulted against the cache at all."""

    assert tvw.match_torchvision_weights(nn.Linear(2, 2)) is None
    assert tvw.is_torchvision_model(nn.Linear(2, 2)) is False
    assert tvw.is_torchvision_model(_Fake()) is True


def test_unique_cached_checkpoint_equal_to_the_state_matches(
    hub_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exactly one cached checkpoint equal to the loaded state names the preset."""

    torchvision_models = pytest.importorskip("torchvision.models")
    preset = torchvision_models.ResNet18_Weights.IMAGENET1K_V1.transforms()
    model = _Fake().eval()
    other = _Fake()
    v1 = _Member("FAKE_V1", "https://example.invalid/w/fake_v1.pth", preset)
    v2 = _Member("FAKE_V2", "https://example.invalid/w/fake_v2.pth", preset)
    absent = _Member("FAKE_V3", "https://example.invalid/w/fake_v3.pth", preset)
    _plant(hub_dir, "fake_v1.pth", model.state_dict())
    _plant(hub_dir, "fake_v2.pth", other.state_dict())
    monkeypatch.setattr(tvw, "_candidate_weights", lambda _model: [v1, v2, absent])

    assert tvw.cached_checkpoint_path(absent) is None
    assert tvw.match_torchvision_weights(model) is v1
    assert tvw.match_torchvision_weights(other) is v2


def test_ambiguous_or_mismatched_cache_names_nothing(
    hub_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two equal checkpoints, a shape mismatch or a missing key all decline."""

    torchvision_models = pytest.importorskip("torchvision.models")
    preset = torchvision_models.ResNet18_Weights.IMAGENET1K_V1.transforms()
    model = _Fake().eval()
    state = model.state_dict()
    twin_a = _Member("A", "https://example.invalid/w/twin_a.pth", preset)
    twin_b = _Member("B", "https://example.invalid/w/twin_b.pth", preset)
    _plant(hub_dir, "twin_a.pth", state)
    _plant(hub_dir, "twin_b.pth", state)
    monkeypatch.setattr(tvw, "_candidate_weights", lambda _model: [twin_a, twin_b])
    assert tvw.match_torchvision_weights(model) is None

    short = {key: value for key, value in state.items() if key != "scale"}
    _plant(hub_dir, "twin_b.pth", short)
    assert tvw.match_torchvision_weights(model) is twin_a
    wrong_shape = dict(state)
    wrong_shape["fc.weight"] = torch.zeros(3, 5)
    _plant(hub_dir, "twin_a.pth", wrong_shape)
    assert tvw.match_torchvision_weights(model) is None


def test_state_match_resolves_preprocessing_and_imagenet_labels(
    hub_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A matched preset drives PIL preprocessing and the output cache key."""

    torchvision_models = pytest.importorskip("torchvision.models")
    preset = torchvision_models.ResNet18_Weights.IMAGENET1K_V1.transforms()
    model = _Fake().eval()
    member = _Member("FAKE_V1", "https://example.invalid/w/fake_v1.pth", preset)
    _plant(hub_dir, "fake_v1.pth", model.state_dict())
    monkeypatch.setattr(tvw, "_candidate_weights", lambda _model: [member])

    resolution = pp.resolve(model=model)
    assert resolution.preprocessing.source == "torchvision_weights"
    assert resolution.preprocessing.verified is True
    assert resolution.preprocessing.config["resolution_method"] == "state_match"
    assert resolution.input_transform is not None

    fragment = semantic_output_cache_key(model)["torchvision"]
    assert fragment["weights_id"] == "FAKE_V1"
    assert fragment["categories_len"] == 3

    class _Image(_Fake):
        def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
            return self.fc(x.mean(dim=(1, 2))[:, :4]) * self.scale

    _Image.__module__ = _Fake.__module__
    image_model = _Image().eval()
    image_model.load_state_dict(model.state_dict())
    trace = tl.trace(
        image_model, _pil_image(), capture=tl.options.CaptureOptions(layers_to_save="none")
    )
    assert trace.provenance.input_preprocessor is not None
    assert trace.provenance.input_preprocessor.source == "torchvision_weights"
    assert trace.provenance.input_preprocessor.config["resolution_method"] == "state_match"


def test_planted_handle_still_wins_over_the_state_match(
    hub_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An attached or planted authority is used before the cache is consulted."""

    torchvision_models = pytest.importorskip("torchvision.models")
    weights = torchvision_models.ResNet18_Weights.IMAGENET1K_V1
    model = _Fake().eval()
    model._torchlens_weights = weights
    called: list[Any] = []
    monkeypatch.setattr(tvw, "_candidate_weights", lambda m: called.append(m) or [])

    resolution = pp.resolve(model=model)
    assert resolution.preprocessing.config["resolution_method"] == "model_metadata"
    assert called == []


@pytest.mark.slow
def test_real_resnet18_is_recognised_from_its_cached_checkpoint() -> None:
    """A torchvision resnet18 built with ``weights=`` resolves without attach_authority."""

    torchvision_models = pytest.importorskip("torchvision.models")
    weights = torchvision_models.ResNet18_Weights.IMAGENET1K_V1
    model = torchvision_models.resnet18(weights=weights).eval()

    assert tvw.match_torchvision_weights(model) is weights
    trace = tl.trace(model, _pil_image(), capture=tl.options.CaptureOptions(layers_to_save="none"))
    assert trace.provenance.input_preprocessor.source == "torchvision_weights"
    assert trace.provenance.output_postprocessor is not None
    assert trace.provenance.output_postprocessor.source == "torchvision_weights"
    assert trace.output_table(k=1).shape[0] == 1
