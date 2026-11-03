"""Every entry that takes model inputs routes text, modality dicts and images as ``tl.trace`` does.

``tl.trace`` resolves a Hugging Face model's string, modality dict or PIL
image through its auto-route detectors. ``Trace.run``, ``tl.record``,
``tl.dry_run``, ``tl.validate_backward_pass`` and the ``tl.debug`` entries
that call the model themselves reach it through the input coercion door, so
the door carries the same resolution: the tests here give each entry the raw
payload a trace took and check the model saw the same keyword inputs and the
trace-shaped entries produced the same ops.
"""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

import torchlens as tl
import torchlens.bridge._hf_processor as hf_processor
import torchlens.bridge.transformers as hf_bridge
from torchlens._data_classes.trace import ResolvedPreprocessing
from torchlens._input_coerce import _chunk_rerun_plan, _coerce_input_args, _coerce_inputs
from torchlens._input_route import (
    _apply_hf_processor,
    _processor_parameter_names,
    _RoutedInputs,
    _spell_for_processor,
)
from torchlens.debug._graph_breaks import _model_call_args
from torchlens.errors import ChunkedForwardConfigError

_forward_calls: list[list[str]] = []
"""Keyword inputs the forward received per call; kept off the model, which capture rolls back."""
_processor_calls: list[dict[str, Any]] = []
"""Keyword arguments the fake processor received per call."""
_tokenizer_calls: list[Any] = []
"""Text payloads the fake tokenizer received per call."""


class _RouteParityModel(nn.Module):
    """HF-shaped model: a ``name_or_path`` and a keyword-only forward over three modalities."""

    name_or_path = "torchlens-tests/route-parity"

    def __init__(self) -> None:
        """Build the three tiny towers."""

        super().__init__()
        self.text_embed = nn.Embedding(16, 4)
        self.vision = nn.Conv2d(3, 4, kernel_size=8)
        self.audio = nn.Linear(8, 4)

    def forward(
        self,
        input_ids: Any = None,
        pixel_values: Any = None,
        attention_mask: Any = None,
        input_features: Any = None,
    ) -> torch.Tensor:
        """Sum whichever keyword inputs arrived."""

        given = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "attention_mask": attention_mask,
            "input_features": input_features,
        }
        _forward_calls.append(sorted(key for key, value in given.items() if value is not None))
        out = torch.zeros(1, 4)
        if input_ids is not None:
            out = out + self.text_embed(input_ids).mean(dim=1)
        if pixel_values is not None:
            out = out + self.vision(pixel_values).reshape(pixel_values.shape[0], -1)[:, :4].mean(0)
        if input_features is not None:
            out = out + self.audio(input_features).mean(dim=1)
        return out


def _image_tensor(images: Any) -> torch.Tensor:
    """Return a ``(batch, 3, 8, 8)`` float tensor from one PIL image or a list of them."""

    batch = images if isinstance(images, list) else [images]
    arrays = [np.asarray(image.convert("RGB").resize((8, 8)), dtype=np.float32) for image in batch]
    return torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2) / 255.0


class _NamedImagesProcessor:
    """``AutoProcessor`` stand-in whose call names ``images`` and ``text`` like ``CLIPProcessor``."""

    def __call__(
        self,
        images: Any = None,
        text: Any = None,
        audio: Any = None,
        *,
        return_tensors: str | None = None,
        sampling_rate: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Return keyword inputs for the given modalities."""

        call: dict[str, Any] = {"return_tensors": return_tensors, **kwargs}
        if images is not None:
            call["images"] = images
        if text is not None:
            call["text"] = text
        if audio is not None:
            call["audio"] = audio
        if sampling_rate is not None:
            call["sampling_rate"] = sampling_rate
        _processor_calls.append(call)
        out: dict[str, Any] = {}
        if text is not None:
            out["input_ids"] = torch.tensor([[1, 2, 3]])
            out["attention_mask"] = torch.ones(1, 3, dtype=torch.long)
        if images is not None:
            out["pixel_values"] = _image_tensor(images)
        if audio is not None:
            out["input_features"] = torch.as_tensor(audio, dtype=torch.float32).reshape(1, -1, 8)
        return out


class _WhisperLikeProcessor:
    """Processor whose call signature is ``(*args, **kwargs)``, as ``WhisperProcessor`` declares."""

    def __call__(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Record the keywords and return audio features."""

        _processor_calls.append(dict(kwargs))
        audio = kwargs["audio"]
        return {"input_features": torch.as_tensor(audio, dtype=torch.float32).reshape(1, -1, 8)}


class _FakeTokenizer:
    """Tokenizer stand-in returning a ``BatchEncoding``-shaped mapping."""

    name_or_path = _RouteParityModel.name_or_path
    padding_side = "right"
    model_max_length = 16

    def __call__(self, text: Any, *, return_tensors: str, padding: bool = True) -> dict[str, Any]:
        """Return token ids whose count depends on the text."""

        _tokenizer_calls.append(text)
        length = 2 + (len(text) if isinstance(text, str) else len(text[0])) % 3
        return {
            "input_ids": torch.arange(1, length + 1).reshape(1, length),
            "attention_mask": torch.ones(1, length, dtype=torch.long),
        }


def _rgb_image(color: tuple[int, int, int] = (10, 20, 30)) -> Any:
    """Return a small RGB PIL image."""

    from PIL import Image as pil_image

    return pil_image.new("RGB", (8, 8), color=color)


def _labels(trace: Any) -> list[str]:
    """Return the op labels of a trace."""

    return [op.label for op in trace.ops]


def _reset_ledgers() -> None:
    """Clear the module ledgers."""

    _forward_calls.clear()
    _processor_calls.clear()
    _tokenizer_calls.clear()


@pytest.fixture
def hf_route(monkeypatch: pytest.MonkeyPatch) -> _RouteParityModel:
    """Give the HF auto-route a fake processor, tokenizer and image recipe; return the model."""

    monkeypatch.setattr(hf_processor, "_load_hf_processor", lambda name: _NamedImagesProcessor())
    monkeypatch.setattr(hf_bridge, "_resolve_tokenizer", lambda model: _FakeTokenizer())
    monkeypatch.setattr(hf_bridge, "_transformers_importable", lambda: True)
    record = ResolvedPreprocessing(
        source="test_recipe", identifier="route-parity", verified=True, config={}, description=""
    )
    monkeypatch.setattr(
        hf_bridge,
        "_resolve_image_preprocessing",
        lambda model: (lambda image: _image_tensor(image)[0], record),
    )
    _reset_ledgers()
    return _RouteParityModel()


def _reference(model: nn.Module, raw: Any) -> tuple[list[str], list[str]]:
    """Trace ``raw`` and return the op labels and the forward's keyword inputs."""

    _forward_calls.clear()
    trace = tl.trace(model, raw, save="none")
    return _labels(trace), list(_forward_calls)


# ---------------------------------------------------------------- Trace.run


def test_run_with_stored_multimodal_transform_respells_singular_image(hf_route: Any) -> None:
    """``Trace.run`` on a text+image trace routes a new dict, ``image`` spelled as the processor does."""

    model = hf_route
    trace = tl.trace(model, {"text": "a cat", "images": _rgb_image()}, save="none")
    reference = _labels(trace)
    dog = _rgb_image((200, 100, 50))
    _reset_ledgers()

    rerun = trace.run(model, {"text": "a dog", "image": dog})

    assert _labels(rerun) == reference
    assert _forward_calls == [["attention_mask", "input_ids", "pixel_values"]]
    assert _processor_calls[-1]["images"] is dog
    assert "image" not in _processor_calls[-1]


def test_run_without_stored_transform_routes_through_the_door(hf_route: Any) -> None:
    """``Trace.run(..., input_transform=False)`` still resolves the dict at the coercion door."""

    model = hf_route
    raw = {"text": "a cat", "image": _rgb_image()}
    reference, reference_calls = _reference(model, raw)
    _reset_ledgers()

    rerun = tl.trace(model, raw, save="none").run(model, raw, input_transform=False)

    assert _labels(rerun) == reference
    assert _forward_calls[-1:] == reference_calls


def test_run_with_stored_text_transform_tokenizes_new_text(hf_route: Any) -> None:
    """``Trace.run`` on a text trace tokenizes the new string and feeds keyword inputs."""

    model = hf_route
    trace = tl.trace(model, "hello", save="none")
    reference = _labels(trace)
    _reset_ledgers()

    rerun = trace.run(model, "goodbye")

    assert _labels(rerun) == reference
    assert _tokenizer_calls == ["goodbye"]
    assert _forward_calls == [["attention_mask", "input_ids"]]


def test_fast_run_with_stored_text_transform(hf_route: Any) -> None:
    """The fast engine takes routed keyword inputs from a text trace's stored transform."""

    model = hf_route
    trace = tl.trace(
        model, "hello", save="none", capture=tl.options.CaptureOptions(replayable=True)
    )
    reference = _labels(trace)
    _reset_ledgers()

    rerun = trace.run(model, "goodbye", fast=True)

    assert _labels(rerun) == reference
    assert _tokenizer_calls == ["goodbye"]
    assert _forward_calls[-1] == ["attention_mask", "input_ids"]


class _ImageOnlyModel(nn.Module):
    """HF-shaped vision model whose forward takes the pixel tensor positionally."""

    name_or_path = "torchlens-tests/route-parity-vision"

    def __init__(self) -> None:
        """Build one convolution."""

        super().__init__()
        self.vision = nn.Conv2d(3, 4, kernel_size=8)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Reduce the image batch to one row per image."""

        return self.vision(pixel_values).reshape(pixel_values.shape[0], -1)


def test_pil_image_routes_at_run_and_record_as_at_trace(hf_route: Any) -> None:
    """A PIL image reaches run and record as the batched pixel tensor the trace saw."""

    model = _ImageOnlyModel()
    trace = tl.trace(model, _rgb_image(), save="all")
    reference = _labels(trace)
    assert trace.input is not None
    assert tuple(trace.input.shape) == (1, 3, 8, 8)

    rerun = trace.run(model, _rgb_image((5, 6, 7)), input_transform=False)
    recording = tl.record(model, [_rgb_image(), _rgb_image((9, 9, 9))], default_op=True)

    assert _labels(rerun) == reference
    assert recording.records


def test_chunked_rerun_refuses_routed_keyword_inputs(hf_route: Any) -> None:
    """Chunked reruns refuse a routed dict exactly as the trace door refuses keyword inputs."""

    model = hf_route
    with pytest.raises(ChunkedForwardConfigError):
        _chunk_rerun_plan(model, {"text": "a cat", "images": _rgb_image()}, 1, None)
    x = torch.randn(4, 3)
    routed, plan = _chunk_rerun_plan(nn.Linear(3, 2), x, 2, None)
    assert routed is x
    assert plan is not None


# ---------------------------------------------------------------- tl.record / dry_run / validation


def test_record_routes_modality_dict(hf_route: Any) -> None:
    """``tl.record`` resolves the dict and records the ops a trace of it has."""

    model = hf_route
    raw = {"text": "a cat", "image": _rgb_image()}
    reference, reference_calls = _reference(model, raw)
    _reset_ledgers()

    recording = tl.record(model, raw, default_op=True)

    assert recording.records
    assert reference
    assert _forward_calls == reference_calls


def test_dry_run_routes_modality_dict(hf_route: Any) -> None:
    """``tl.dry_run`` feeds the forward the same keyword inputs a trace does."""

    model = hf_route
    raw = {"text": "a cat", "image": _rgb_image()}
    _reference_labels, reference_calls = _reference(model, raw)
    _reset_ledgers()

    tl.dry_run(model, raw)

    assert _forward_calls[:1] == reference_calls


def test_validate_backward_pass_routes_modality_dict(hf_route: Any) -> None:
    """``tl.validate_backward_pass`` resolves the dict instead of handing it to the forward."""

    model = hf_route
    raw = {"text": "a cat", "image": _rgb_image()}

    assert tl.validate_backward_pass(model, raw, validate_metadata=False) is True
    assert _forward_calls[0] == ["attention_mask", "input_ids", "pixel_values"]


def test_record_routes_text(hf_route: Any) -> None:
    """``tl.record`` tokenizes a bare string through the model's tokenizer route."""

    model = hf_route
    recording = tl.record(model, "hello", default_op=True)

    assert recording.records
    assert _tokenizer_calls == ["hello"]
    assert _forward_calls == [["attention_mask", "input_ids"]]


# ---------------------------------------------------------------- tl.debug


def test_graph_breaks_call_args_route_modality_dict(hf_route: Any) -> None:
    """The Dynamo probe's model call gets the routed keyword inputs, not the raw dict."""

    model = hf_route
    raw = {"text": "a cat", "image": _rgb_image()}

    args, kwargs = _model_call_args(model, raw, {})

    assert args == ()
    assert set(kwargs) == {"input_ids", "attention_mask", "pixel_values"}
    assert _processor_calls[-1]["images"] is raw["image"]


def test_flops_vs_dispatch_routes_modality_dict(hf_route: Any) -> None:
    """The raw forward of the FLOP cross-check sees the inputs the trace saw."""

    model = hf_route
    raw = {"text": "a cat", "image": _rgb_image()}

    report = tl.debug.flops_vs_dispatch(model, raw)

    assert report.native_total is not None
    assert _forward_calls[-1] == ["attention_mask", "input_ids", "pixel_values"]


# ---------------------------------------------------------------- the door itself


def test_one_item_container_holding_a_routed_dict_becomes_keyword_inputs(hf_route: Any) -> None:
    """A one-item tuple or list around a modality dict routes like the bare dict."""

    model = hf_route
    raw = {"text": "a cat", "image": _rgb_image()}

    for container in ((raw,), [raw]):
        coerced = _coerce_input_args(model, container)
        assert isinstance(coerced, dict)
        assert set(coerced) == {"input_ids", "attention_mask", "pixel_values"}
        input_args, input_kwargs = _coerce_inputs(model, container, None)
        assert input_args == []
        assert input_kwargs is not None
        assert set(input_kwargs) == {"input_ids", "attention_mask", "pixel_values"}


def test_coerce_inputs_unwraps_routed_inputs_and_keeps_caller_keywords() -> None:
    """A ``_RoutedInputs`` carrier becomes the engine's ``(args, kwargs)``; caller keywords win."""

    routed = _RoutedInputs([], {"input_ids": torch.ones(1, 2, dtype=torch.long), "k": 1})

    input_args, input_kwargs = _coerce_inputs(nn.Linear(1, 1), routed, {"k": 2})

    assert input_args == []
    assert input_kwargs == {"input_ids": routed.input_kwargs["input_ids"], "k": 2}


def test_whisper_like_audio_dict_keeps_audio_key_and_adds_sampling_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``(*args, **kwargs)`` processor gets ``audio`` unchanged plus the model's sampling rate."""

    monkeypatch.setattr(hf_processor, "_load_hf_processor", lambda name: _WhisperLikeProcessor())
    monkeypatch.setattr(hf_bridge, "_transformers_importable", lambda: True)
    _reset_ledgers()
    model = _RouteParityModel()
    model.sampling_rate = 16_000
    waveform = np.zeros(64, dtype=np.float32)

    reference, reference_calls = _reference(model, {"audio": waveform})
    _reset_ledgers()
    rerun = tl.trace(model, {"audio": waveform}, save="none").run(
        model, {"audio": waveform}, input_transform=False
    )

    assert _labels(rerun) == reference
    assert _forward_calls[-1:] == reference_calls == [["input_features"]]
    assert _processor_calls[-1]["audio"] is waveform
    assert _processor_calls[-1]["sampling_rate"] == 16_000


def test_blip_and_llava_shaped_dicts_route_at_every_entry(hf_route: Any) -> None:
    """Image-only (BLIP captioning) and text plus image-batch (LLaVA) dicts route the same way."""

    model = hf_route
    cases = [
        ({"image": _rgb_image()}, ["pixel_values"]),
        ({"text": "describe", "image": [_rgb_image(), _rgb_image((1, 2, 3))]}, None),
    ]
    for raw, expected in cases:
        reference, reference_calls = _reference(model, raw)
        _reset_ledgers()
        rerun = tl.trace(model, raw, save="none").run(model, raw, input_transform=False)
        assert _labels(rerun) == reference
        assert _forward_calls[-1:] == reference_calls
        if expected is not None:
            assert reference_calls == [expected]
        assert _processor_calls[-1]["images"] is raw["image"]


# ---------------------------------------------------------------- processor signature oracle


def test_spelling_follows_the_processor_signature() -> None:
    """A key the processor names is kept; its plural or singular is used when that is what it names."""

    named = frozenset({"images", "text", "audio", "videos"})
    assert _spell_for_processor("image", named) == "images"
    assert _spell_for_processor("images", named) == "images"
    assert _spell_for_processor("video", named) == "videos"
    assert _spell_for_processor("texts", named) == "text"
    assert _spell_for_processor("audio", named) == "audio"
    assert _spell_for_processor("image", frozenset()) == "image"
    assert _spell_for_processor("image", None) == "image"


def test_apply_hf_processor_respells_only_for_a_processor_that_names_the_plural() -> None:
    """``image`` becomes ``images`` for a CLIP-shaped call and stays ``image`` for ``**kwargs``."""

    _reset_ledgers()
    image = _rgb_image()
    _apply_hf_processor(_NamedImagesProcessor(), {"text": "a", "image": image})
    assert _processor_calls[-1]["images"] is image
    assert _processor_calls[-1]["return_tensors"] == "pt"

    def passthrough(**kwargs: Any) -> dict[str, Any]:
        """Record whatever keys arrive."""

        _processor_calls.append(dict(kwargs))
        return {}

    _apply_hf_processor(passthrough, {"image": image})
    assert "image" in _processor_calls[-1]


def test_real_processor_signatures_are_the_oracle() -> None:
    """``CLIPProcessor`` names ``images``; ``WhisperProcessor`` declares ``*args, **kwargs``."""

    transformers = pytest.importorskip("transformers")

    clip_names = _processor_parameter_names(transformers.CLIPProcessor.__call__)
    assert clip_names is not None
    assert "images" in clip_names
    assert _spell_for_processor("image", clip_names) == "images"

    whisper_signature = inspect.signature(transformers.WhisperProcessor.__call__)
    kinds = {parameter.kind for parameter in whisper_signature.parameters.values()}
    if inspect.Parameter.VAR_KEYWORD in kinds and len(kinds) <= 3:
        whisper_names = _processor_parameter_names(transformers.WhisperProcessor.__call__)
        assert whisper_names is not None
        assert _spell_for_processor("audio", whisper_names) == "audio"


@pytest.mark.slow
def test_tiny_clip_singular_image_key_runs_and_records() -> None:
    """A real CLIP checkpoint takes ``{"text", "image"}`` through run and record as through trace."""

    transformers = pytest.importorskip("transformers")
    pytest.importorskip("PIL")

    model = transformers.CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    raw = {"text": "a cat", "image": _rgb_image()}

    trace = tl.trace(model, raw, save="none")
    reference = _labels(trace)
    rerun = trace.run(model, {"text": "a dog", "image": _rgb_image((200, 10, 10))})
    recording = tl.record(model, raw, default_op=True)

    assert _labels(rerun) == reference
    assert recording.records
