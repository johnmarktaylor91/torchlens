"""Tests for multimodal dict auto-routing in ``tl.trace``."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
import torchlens.bridge._hf_processor as hf_processor
import torchlens.bridge.transformers as hf_bridge


class _DictModel(nn.Module):
    """Tiny model that accepts a dict on the non-auto-routed path."""

    def forward(self, x: dict[str, Any]) -> torch.Tensor:
        """Return a tensor while accepting arbitrary dict input."""

        return torch.ones(1)


_received: list[tuple[Any, Any]] = []
"""Inputs the CLIP-like forward saw; kept off the model, which capture rolls back."""


class _ClipLikeModel(nn.Module):
    """HF-shaped two-tower model: a ``name_or_path`` and a keyword-only forward.

    The forward reads ``pixel_values.shape`` first, the way ``CLIPModel`` does,
    so a modality dict that lands in the positional ``input_ids`` slot raises the
    notebook's ``'NoneType' object has no attribute 'shape'``.
    """

    name_or_path = "torchlens-tests/clip-like"

    def __init__(self) -> None:
        """Build the two tiny towers."""

        super().__init__()
        self.text_embed = nn.Embedding(16, 4)
        self.vision = nn.Conv2d(3, 4, kernel_size=8)

    def forward(
        self,
        input_ids: Any = None,
        pixel_values: Any = None,
        attention_mask: Any = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Return a text-image similarity row from keyword inputs only."""

        _received.append((input_ids, pixel_values))
        batch = pixel_values.shape[0]
        image = self.vision(pixel_values).reshape(batch, -1).mean(dim=1)
        text = self.text_embed(input_ids).mean(dim=(1, 2))
        return text * image


class _FakeAutoProcessor:
    """Stand-in for a resolved ``AutoProcessor`` returning paired tensors."""

    name_or_path = _ClipLikeModel.name_or_path

    def __init__(self) -> None:
        """Start the call ledger."""

        self.calls: list[dict[str, Any]] = []

    def __call__(self, *, return_tensors: str, **modalities: Any) -> dict[str, Any]:
        """Return CLIP-style keyword inputs for the given modalities."""

        self.calls.append({**modalities, "return_tensors": return_tensors})
        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
            "pixel_values": torch.ones(1, 3, 8, 8),
        }


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
    # No processor warm-up: the route resolves the AutoProcessor itself, so a
    # cache holding only the weights (the notebook's cold-cache run) still routes.

    log = tl.trace(
        model,
        {"text": "a cat", "images": _pil_image()},
        save="none",
    )

    assert log.provenance.input_preprocessor is not None
    assert log.provenance.input_preprocessor.source == "hf_auto_processor"
    assert log.provenance.input_preprocessor.verified is True


def test_non_modality_dict_does_not_route(monkeypatch: pytest.MonkeyPatch) -> None:
    """A plain dict should skip multimodal auto-routing."""

    def fail_trace_multimodal(model: Any, input_dict: Any, **kwargs: Any) -> object:
        """Fail if multimodal bridge dispatch occurs."""

        raise AssertionError("multimodal bridge should not fire")

    monkeypatch.setattr(hf_bridge, "trace_multimodal", fail_trace_multimodal)

    log = tl.trace(
        _DictModel(),
        {"key": "value"},
        save="none",
    )

    assert log.provenance.input_preprocessor is None


def test_empty_dict_does_not_route() -> None:
    """An empty dict should not match the multimodal gate."""

    assert hf_bridge._is_hf_multimodal_input({}) is False


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


def test_dict_transform_override_skips_multimodal_autoroute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit ``transform=`` should bypass multimodal auto-routing."""

    import PIL.Image  # noqa: F401

    def fail_trace_multimodal(model: Any, input_dict: Any, **kwargs: Any) -> object:
        """Fail if multimodal bridge dispatch occurs."""

        raise AssertionError("multimodal bridge should not fire")

    monkeypatch.setattr(hf_bridge, "trace_multimodal", fail_trace_multimodal)

    log = tl.trace(
        nn.Linear(1, 1),
        {"text": "hello", "images": _pil_image()},
        capture=tl.options.CaptureOptions(input_transform=lambda value: torch.ones(1)),
        save="none",
    )

    # The explicit transform bypasses the multimodal autoroute entirely: no
    # bridge stamped provenance, so the record is the honest user_transform
    # stamp (tvscope B1), never the multimodal bridge's richer record.
    assert log.provenance.input_preprocessor is not None
    assert log.provenance.input_preprocessor.source == "user_transform"
    assert log.provenance.input_preprocessor.verified is False


def test_clip_style_dict_routes_without_a_cached_processor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The route claims an HF model on shape and resolves the processor itself.

    Regression for the huggingface_tutorial CLIP cell on a cold cache: the gate
    used to probe the local cache only, so a model whose weights were just
    downloaded declined the route and the modality dict reached the forward as
    ``input_ids`` (``'NoneType' object has no attribute 'shape'``).
    """

    pytest.importorskip("transformers")
    processor = _FakeAutoProcessor()
    monkeypatch.setattr(hf_processor, "_load_hf_processor", lambda name_or_path: processor)
    model = _ClipLikeModel()
    image = _pil_image()
    _received.clear()

    log = tl.trace(model, {"text": "a cat", "images": image}, save="none")

    assert processor.calls == [{"text": "a cat", "images": image, "return_tensors": "pt"}]
    assert len(_received) == 1
    input_ids, pixel_values = _received[0]
    assert isinstance(input_ids, torch.Tensor) and isinstance(pixel_values, torch.Tensor)
    record = log.provenance.input_preprocessor
    assert record is not None
    assert record.source == "hf_auto_processor"
    assert record.identifier == _ClipLikeModel.name_or_path
    assert record.verified is True


def test_modality_dict_with_unresolvable_processor_refuses_typed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An HF model whose processor cannot load refuses before the forward runs."""

    pytest.importorskip("transformers")
    from torchlens.errors import InvalidArgumentError

    def fail_load(name_or_path: str) -> object:
        """Stand in for an offline Hub fetch of an uncached processor."""

        raise OSError(f"{name_or_path} is not cached and the Hub is unreachable")

    monkeypatch.setattr(hf_processor, "_load_hf_processor", fail_load)
    model = _ClipLikeModel()
    _received.clear()

    with pytest.raises(InvalidArgumentError) as excinfo:
        tl.trace(model, {"text": "a cat", "images": _pil_image()}, save="none")

    assert excinfo.value.fields["code"] == "preprocessing_unresolved"
    assert "model.processor = AutoProcessor.from_pretrained" in excinfo.value.fields["remedy"]
    assert excinfo.value.fields["resolution_attempt"]["name_or_path"] == model.name_or_path
    assert _received == []


def test_multimodal_gate_claims_on_model_shape_without_io(monkeypatch: pytest.MonkeyPatch) -> None:
    """Detection needs a ``name_or_path`` and ``transformers``; it never touches the Hub."""

    pytest.importorskip("transformers")

    def fail_load(name_or_path: str) -> object:
        """Fail if detection resolves the processor."""

        raise AssertionError("detection must not load the processor")

    monkeypatch.setattr(hf_processor, "_load_hf_processor", fail_load)
    assert hf_bridge._can_resolve_hf_processor(_ClipLikeModel()) is True
    assert hf_bridge._can_resolve_hf_processor(_DictModel()) is False
