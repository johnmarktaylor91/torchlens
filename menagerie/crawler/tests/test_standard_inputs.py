"""Tests for canonical modality-aware input materialization."""

from __future__ import annotations

from menagerie.crawler.standard_inputs import materialize_standard_input


def test_standard_image_text_and_unknown_fallback_record_input_kind() -> None:
    """Known modalities use assets while unknown modalities record random fallback."""

    image = materialize_standard_input("vision", {"shape": [1, 3, 12, 10], "dtype": "float32"})
    text = materialize_standard_input("language", {"shape": [1, 16], "dtype": "int64"})
    unknown = materialize_standard_input("unknown", {"shape": [2, 5], "dtype": "float32"})

    assert image.input_kind == "standard-image"
    assert image.input_asset is not None
    assert tuple(getattr(image.value, "shape")) == (1, 3, 12, 10)
    assert text.input_kind == "standard-text"
    assert text.input_asset is not None
    assert tuple(getattr(text.value, "shape")) == (1, 16)
    assert unknown.input_kind == "random-fallback"
    assert unknown.input_asset is None
    assert "random fallback" in unknown.input_note
