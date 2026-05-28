"""Tests for multimodal dict auto-routing in ``tl.trace``."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
import torchlens.bridge.hf as hf_bridge
import torchlens.user_funcs as user_funcs


class _DictModel(nn.Module):
    """Tiny model that accepts a dict on the non-auto-routed path."""

    def forward(self, x: dict[str, Any]) -> torch.Tensor:
        """Return a tensor while accepting arbitrary dict input."""

        return torch.ones(1)


def _pil_image() -> Any:
    """Create a small RGB PIL image, skipping when PIL is unavailable.

    Returns
    -------
    Any
        PIL image instance.
    """

    image_module = pytest.importorskip("PIL.Image")
    return image_module.new("RGB", (32, 32), color=(20, 130, 70))


@pytest.mark.slow
def test_clip_style_dict_routes_via_auto_processor() -> None:
    """CLIP-style text plus image dict should use ``AutoProcessor``."""

    transformers = pytest.importorskip("transformers")
    pytest.importorskip("PIL.Image")
    model = transformers.CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

    log = tl.trace(
        model,
        {"text": "a cat", "images": _pil_image()},
        layers_to_save="none",
    )

    assert log.input_preprocessor is not None
    assert log.input_preprocessor.source == "hf_auto_processor"
    assert log.input_preprocessor.verified is True


def test_non_modality_dict_does_not_route(monkeypatch: pytest.MonkeyPatch) -> None:
    """A plain dict should skip multimodal auto-routing."""

    def fail_trace_multimodal(model: Any, input_dict: Any, **kwargs: Any) -> object:
        """Fail if multimodal bridge dispatch occurs."""

        raise AssertionError("multimodal bridge should not fire")

    monkeypatch.setattr(hf_bridge, "trace_multimodal", fail_trace_multimodal)

    log = tl.trace(_DictModel(), {"key": "value"}, layers_to_save="none")

    assert log.input_preprocessor is None


def test_empty_dict_does_not_route() -> None:
    """An empty dict should not match the multimodal gate."""

    assert hf_bridge._is_hf_multimodal_input({}) is False


def test_wrong_modality_value_types_do_not_route() -> None:
    """Modality keys with implausible values should fail closed."""

    assert hf_bridge._is_hf_multimodal_input({"image": "not_a_pil"}) is False


def test_model_without_auto_processor_gate_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """A multimodal-looking dict should fall through when processor resolution fails."""

    pytest.importorskip("PIL.Image")

    def fail_trace_multimodal(model: Any, input_dict: Any, **kwargs: Any) -> object:
        """Fail if multimodal bridge dispatch occurs."""

        raise AssertionError("multimodal bridge should not fire")

    monkeypatch.setattr(hf_bridge, "trace_multimodal", fail_trace_multimodal)
    monkeypatch.setattr(user_funcs, "_can_resolve_hf_processor", lambda model: False)

    log = tl.trace(_DictModel(), {"images": _pil_image()}, layers_to_save="none")

    assert log.input_preprocessor is None


def test_dict_transform_override_skips_multimodal_autoroute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit ``transform=`` should bypass multimodal auto-routing."""

    pytest.importorskip("PIL.Image")

    def fail_trace_multimodal(model: Any, input_dict: Any, **kwargs: Any) -> object:
        """Fail if multimodal bridge dispatch occurs."""

        raise AssertionError("multimodal bridge should not fire")

    monkeypatch.setattr(hf_bridge, "trace_multimodal", fail_trace_multimodal)

    log = tl.trace(
        nn.Linear(1, 1),
        {"text": "hello", "images": _pil_image()},
        transform=lambda value: torch.ones(1),
        layers_to_save="none",
    )

    assert log.input_preprocessor is None
