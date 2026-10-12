"""Vision logit lens and user-lens identity checks (batch-2 VISION: U55.2, U43.2).

Models are built from their vendored published configs, shrunk, weights from
a fixed seed, zero network:

- A ViT classifier's default lens is its own final norm plus classifier over
  the ``[CLS]`` token, validated exactly like a language-model lens.
- A CLIP capture run on images and label prompts decodes every vision-tower
  block against the captured text features: per-layer label scores whose
  final layer equals CLIP's own ``logits_per_image``.
- A user lens whose recorded model identity differs from the traced model
  refuses unless ``validate=False``; a lens with no recorded identity warns
  when its last-layer decode disagrees with the model's captured logits.
"""

from __future__ import annotations

import warnings
from typing import Any

import pytest
import torch

import torchlens as tl

pytest.importorskip("transformers")

from tests.real_model.r0.families import (  # noqa: E402
    FAMILY_BY_NAME,
    SEED,
    VOCAB,
    _hf_config,
    build_clip,
)
from torchlens.errors import TorchLensWarning  # noqa: E402
from torchlens.semantic.logit_lens import (  # noqa: E402
    LogitLensError,
    logit_lens,
    logit_lens_predictions,
)

pytestmark = pytest.mark.real_model

N_LABELS = 5
N_PROMPTS = 3


def _quiet_trace(model: Any, *args: Any, **kwargs: Any) -> Any:
    """Trace with upstream config chatter silenced."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return tl.trace(model, *args, **kwargs)


@pytest.fixture(scope="module")
def vit_classifier() -> tuple[Any, Any, Any]:
    """A shrunk ViTForImageClassification, its pixels, and its trace."""

    from transformers import ViTConfig, ViTForImageClassification

    shrink = {
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "hidden_size": 64,
        "intermediate_size": 128,
        "image_size": 32,
        "patch_size": 16,
        "id2label": {str(index): f"label_{index}" for index in range(N_LABELS)},
        "label2id": {f"label_{index}": index for index in range(N_LABELS)},
    }
    torch.manual_seed(SEED)
    model = ViTForImageClassification(_hf_config("vit", ViTConfig, shrink, "eager")).eval()
    pixels = torch.randn(2, 3, 32, 32, generator=torch.Generator().manual_seed(SEED))
    trace = _quiet_trace(model, (), {"pixel_values": pixels})
    return model, pixels, trace


@pytest.fixture(scope="module")
def clip_capture() -> tuple[Any, Any, Any]:
    """A shrunk CLIPModel run on two images and three label prompts."""

    model = build_clip("eager")
    generator = torch.Generator().manual_seed(SEED)
    inputs = {
        "input_ids": torch.randint(2, VOCAB, (N_PROMPTS, 8), generator=generator),
        "pixel_values": torch.randn(2, 3, 32, 32, generator=generator),
    }
    trace = _quiet_trace(model, (), inputs)
    return model, inputs, trace


@pytest.fixture(scope="module")
def llama_capture() -> tuple[Any, Any]:
    """The R0 tiny Llama and its trace (a language-model head)."""

    spec = FAMILY_BY_NAME["llama"]
    model = spec.build("eager")
    trace = _quiet_trace(model, spec.input_args(), spec.input_kwargs())
    return model, trace


def test_vit_classifier_default_lens_is_validated_and_ends_at_the_logits(
    vit_classifier: tuple[Any, Any, Any],
) -> None:
    """No lens=: the classifier head lens validates; its final layer is the model's logits."""

    model, pixels, trace = vit_classifier
    result = logit_lens(trace)
    assert result.lens_source == "classifier_head"
    assert result.validated is True
    blocks = [str(m.address) for m in trace.modules if m.class_name == "ViTLayer"]
    assert len(blocks) == 2
    assert [entry.address for entry in result.entries] == blocks
    with torch.no_grad():
        expected = model(pixel_values=pixels).logits
    for entry in result.entries:
        assert entry.logits.shape == (2, N_LABELS)
    assert torch.allclose(result.entries[-1].logits, expected, rtol=1e-5, atol=1e-6)
    assert result.final_logits is not None and torch.equal(result.final_logits, expected)
    head = trace.modules["self"].facets
    assert head["pooled_position"] == 0
    assert head["tower_address"] == "vit"
    assert head["final_norm_kind"] == "layer_norm"


def test_vit_classifier_predictions_stream_with_a_native_row(
    vit_classifier: tuple[Any, Any, Any],
) -> None:
    """The streaming extractor serves per-layer label rows plus the native row."""

    _model, _pixels, trace = vit_classifier
    preds = logit_lens_predictions(trace, k=2)
    assert preds.lens_source == "classifier_head"
    assert [row.layer_index for row in preds.rows] == [0, 1, None]
    native = preds.rows[-1]
    final = preds.rows[-2]
    assert native.provenance == "native output"
    assert torch.equal(native.top_ids, final.top_ids)


def test_vit_classifier_lens_validation_is_still_a_tripwire(
    vit_classifier: tuple[Any, Any, Any],
) -> None:
    """A head that no longer matches the captured logits refuses, as an LM head does."""

    model, _pixels, trace = vit_classifier
    original = model.classifier.weight.detach().clone()
    try:
        with torch.no_grad():
            model.classifier.weight.mul_(1.5)
        with pytest.raises(LogitLensError, match="failed validation"):
            logit_lens(trace)
        assert logit_lens(trace, validate=False).validated is False
    finally:
        with torch.no_grad():
            model.classifier.weight.copy_(original)


def test_trace_with_neither_head_still_refuses_typed() -> None:
    """A bare ViTModel has no LM head and no classifier: the typed refusal stands."""

    spec = FAMILY_BY_NAME["vit"]
    model = spec.build("eager")
    trace = _quiet_trace(model, spec.input_args(), spec.input_kwargs())
    with pytest.raises(LogitLensError) as excinfo:
        logit_lens(trace)
    assert excinfo.value.fields["code"] == "logit_lens_head_unavailable"


def test_clip_decodes_vision_blocks_against_label_prompt_features(
    clip_capture: tuple[Any, Any, Any],
) -> None:
    """Per-layer label scores; the final layer equals CLIP's own image-text similarity."""

    model, inputs, trace = clip_capture
    result = logit_lens(trace)
    assert result.lens_source == "contrastive_head"
    assert result.validated is True
    assert [entry.address for entry in result.entries] == [
        "vision_model.encoder.layers.0",
        "vision_model.encoder.layers.1",
    ]
    with torch.no_grad():
        expected = model(**inputs).logits_per_image
    assert expected.shape == (2, N_PROMPTS)
    for entry in result.entries:
        assert entry.logits.shape == (2, N_PROMPTS)
    assert torch.allclose(result.entries[-1].logits, expected, rtol=1e-5, atol=1e-5)
    rows = result.top_tokens(k=N_PROMPTS, batch_index=1)
    assert len(rows) == 2 and len(rows[-1][1]) == N_PROMPTS
    assert rows[-1][1][0][0] == int(expected[1].argmax())


def test_clip_head_lens_refuses_text_tower_layers(clip_capture: tuple[Any, Any, Any]) -> None:
    """The image head cannot decode the text tower's residual stream."""

    _model, _inputs, trace = clip_capture
    with pytest.raises(LogitLensError) as excinfo:
        logit_lens(trace, layers=["text_model.encoder.layers.1"])
    assert excinfo.value.fields["code"] == "logit_lens_layer_outside_head"


def _own_head_lens(model: Any) -> Any:
    """The Llama model's own final norm plus unembedding, as a user lens."""

    def lens(hidden: torch.Tensor) -> torch.Tensor:
        return model.lm_head(model.model.norm(hidden))

    return lens


def _random_lens(hidden_size: int) -> Any:
    """A lens unrelated to the model: a fixed random projection."""

    weight = torch.randn(hidden_size, VOCAB, generator=torch.Generator().manual_seed(1))

    def lens(hidden: torch.Tensor) -> torch.Tensor:
        return hidden @ weight

    return lens


def _identity_of(model: Any) -> dict[str, Any]:
    """The measured model-identity record an extraction manifest would carry."""

    from torchlens._data_substrate import compute_model_identity

    return compute_model_identity(model, level="measured")


def test_user_lens_with_a_different_recorded_identity_refuses(
    llama_capture: tuple[Any, Any],
) -> None:
    """A lens built for another model refuses typed; validate=False runs it."""

    model, trace = llama_capture
    other = FAMILY_BY_NAME["llama"].build("eager")
    with torch.no_grad():
        other.lm_head.weight.add_(1e-3)
    lens = _own_head_lens(model)
    lens.model_identity = _identity_of(other)  # type: ignore[attr-defined]
    with pytest.raises(LogitLensError) as excinfo:
        logit_lens(trace, lens=lens)
    assert excinfo.value.fields["code"] == "logit_lens_identity_mismatch"
    assert "validate=False" in str(excinfo.value)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = logit_lens(trace, lens=lens, validate=False)
    assert result.lens_source == "user" and len(result.entries) == 2


def test_user_lens_with_the_matching_identity_passes_silently(
    llama_capture: tuple[Any, Any],
) -> None:
    """A lens whose recorded identity matches the traced model runs with no warning."""

    model, trace = llama_capture
    lens = _random_lens(model.config.hidden_size)
    lens.model_identity = _identity_of(model)  # type: ignore[attr-defined]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = logit_lens(trace, lens=lens)
        preds = logit_lens_predictions(trace, k=1, lens=lens)
    assert result.lens_source == "user" and preds.lens_source == "user"


def test_identity_less_disagreeing_lens_emits_the_last_layer_agreement_warning(
    llama_capture: tuple[Any, Any],
) -> None:
    """No recorded identity and a last-layer decode that misses the logits: coded warning."""

    model, trace = llama_capture
    lens = _random_lens(model.config.hidden_size)
    with pytest.warns(TorchLensWarning) as record:
        result = logit_lens(trace, lens=lens)
    assert len(result.entries) == 2
    coded = [w.message for w in record if isinstance(w.message, TorchLensWarning)]
    assert [w.fields["code"] for w in coded] == ["logit_lens_identity_unrecorded"]
    message = str(coded[0])
    assert "could not confirm the user lens belongs to this model" in message
    assert "it records no model identity" in message
    assert "does not reproduce the captured logits" in message
    assert "top-1 agreement" in message
    assert "lens.model_identity" in coded[0].fields["remedy"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        logit_lens(trace, lens=lens, validate=False)


def test_identity_less_lens_that_reproduces_the_logits_is_silent(
    llama_capture: tuple[Any, Any],
) -> None:
    """The model's own head as an identity-less lens passes the agreement check quietly."""

    model, trace = llama_capture
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = logit_lens(trace, lens=_own_head_lens(model))
    assert result.lens_source == "user"


def test_asserted_identity_is_not_comparable_and_falls_back_to_the_warning(
    llama_capture: tuple[Any, Any],
) -> None:
    """Only a measured digest compares; an asserted record gets the agreement check."""

    model, trace = llama_capture
    lens = _random_lens(model.config.hidden_size)
    lens.model_identity = {  # type: ignore[attr-defined]
        "level": "asserted",
        "assertion": {"checkpoint": "elsewhere", "revision": "r1"},
    }
    with pytest.warns(TorchLensWarning, match="level 'asserted'"):
        logit_lens(trace, lens=lens)


def test_malformed_and_conflicting_identity_records_refuse(
    llama_capture: tuple[Any, Any],
) -> None:
    """A non-record identity is invalid; per-layer records that disagree mismatch."""

    model, trace = llama_capture
    bad = _own_head_lens(model)
    bad.model_identity = "llama-tiny"  # type: ignore[attr-defined]
    with pytest.raises(LogitLensError) as excinfo:
        logit_lens(trace, lens=bad)
    assert excinfo.value.fields["code"] == "logit_lens_identity_invalid"
    first, second = _own_head_lens(model), _own_head_lens(model)
    first.model_identity = _identity_of(model)  # type: ignore[attr-defined]
    second.model_identity = {**_identity_of(model), "digest": "0" * 64}  # type: ignore[attr-defined]
    mapping = {"model.layers.0": first, "model.layers.1": second}
    with pytest.raises(LogitLensError) as excinfo:
        logit_lens(trace, lens=mapping)
    assert excinfo.value.fields["code"] == "logit_lens_identity_mismatch"
