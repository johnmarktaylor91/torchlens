"""Tests for multimodal dict auto-routing in ``tl.trace``."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
import torchlens.bridge.transformers as hf_bridge


class _DictModel(nn.Module):
    """Tiny model that accepts a dict on the non-auto-routed path."""

    def forward(self, x: dict[str, Any]) -> torch.Tensor:
        """Return a tensor while accepting arbitrary dict input."""

        return torch.ones(1)


def _pil_image() -> Any:
    """Create a small RGB PIL image (Pillow is a required dependency).

    Returns
    -------
    Any
        PIL image instance.
    """

    from PIL import Image as image_module

    return image_module.new("RGB", (32, 32), color=(20, 130, 70))


@pytest.mark.slow
def test_clip_style_dict_routes_via_auto_processor() -> None:
    """CLIP-style text plus image dict should use ``AutoProcessor``."""

    transformers = pytest.importorskip("transformers")
    import PIL.Image  # noqa: F401

    model = transformers.CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    # The auto-route gate probes the local Hugging Face cache only (it never does
    # network I/O while routing), so fetch the processor files the way the model's
    # weights were fetched; a cache holding only the weights declines the route.
    transformers.AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

    log = tl.trace(
        model,
        {"text": "a cat", "images": _pil_image()},
        save="none",
    )

    assert log.provenance.input_preprocessor is not None
    assert log.provenance.input_preprocessor.source == "hf_auto_processor"
    assert log.provenance.input_preprocessor.verified is True


def test_wrong_modality_value_types_do_not_route() -> None:
    """Modality keys with implausible values should fail closed."""

    assert hf_bridge._is_hf_multimodal_input({"image": "not_a_pil"}) is False


def test_model_without_auto_processor_gate_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """A multimodal-looking dict should fall through when processor resolution fails."""

    import PIL.Image  # noqa: F401

    def fail_trace_multimodal(model: Any, input_dict: Any, **kwargs: Any) -> object:
        """Fail if multimodal bridge dispatch occurs."""

        raise AssertionError("multimodal bridge should not fire")

    monkeypatch.setattr(hf_bridge, "trace_multimodal", fail_trace_multimodal)
    monkeypatch.setattr(hf_bridge, "_can_resolve_hf_processor", lambda model: False)

    log = tl.trace(
        _DictModel(),
        {"images": _pil_image()},
        save="none",
    )

    assert log.provenance.input_preprocessor is None


class _FakePreTrainedModel(nn.Module):
    """Stand-in for ``transformers.PreTrainedModel`` (no transformers import)."""


class _UncachedHFModel(_FakePreTrainedModel):
    """HF-shaped model whose forward must never see the raw modality dict."""

    name_or_path = "fixture/uncached-multimodal"

    def __init__(self) -> None:
        super().__init__()
        self.forward_calls = 0

    def forward(self, **kwargs: Any) -> torch.Tensor:
        """Count calls; a real model would fail on the raw strings and images."""

        self.forward_calls += 1
        return torch.ones(1)


def _install_fake_transformers(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Install a fake ``transformers`` whose processor is never in the cache."""

    import sys
    from types import SimpleNamespace

    calls: list[dict[str, Any]] = []

    class _UncachedAutoProcessor:
        @classmethod
        def from_pretrained(cls, name_or_path: str, **kwargs: Any) -> Any:
            calls.append({"name_or_path": name_or_path, **kwargs})
            raise OSError("not found in the local cache")

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoProcessor=_UncachedAutoProcessor, PreTrainedModel=_FakePreTrainedModel),
    )
    return calls


def test_uncached_processor_on_hf_model_refuses_typed(monkeypatch: pytest.MonkeyPatch) -> None:
    """An uncached processor refuses typed instead of feeding the model raw input.

    The raw ``{"text": ..., "images": ...}`` dict used to reach the model's
    forward, which failed deep inside (a CLIP vision embedding read ``.shape``
    of a missing ``pixel_values``) with zero ops captured.
    """

    from torchlens.errors import InvalidArgumentError

    calls = _install_fake_transformers(monkeypatch)
    model = _UncachedHFModel()

    with pytest.raises(InvalidArgumentError) as excinfo:
        tl.trace(model, {"text": "a cat", "images": _pil_image()}, save="none")

    assert excinfo.value.fields["code"] == "preprocessing_unresolved"
    assert "fixture/uncached-multimodal" in str(excinfo.value)
    assert "AutoProcessor.from_pretrained" in str(excinfo.value)
    assert "trace_multimodal" in str(excinfo.value)
    assert model.forward_calls == 0
    # Detection stayed cache-only: no download was attempted.
    assert calls and all(call.get("local_files_only") is True for call in calls)


def test_uncached_processor_on_non_hf_model_keeps_plain_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A model outside transformers keeps the plain path even with a name_or_path."""

    _install_fake_transformers(monkeypatch)

    class _NamedDictModel(_DictModel):
        name_or_path = "fixture/not-a-transformers-model"

    log = tl.trace(_NamedDictModel(), {"images": _pil_image()}, save="none")

    assert log.provenance.input_preprocessor is None
