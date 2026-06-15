"""Tests for ``Trace.input_preprocessor`` provenance."""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.data_classes.trace import ResolvedPreprocessing


class _TinyModel(nn.Module):
    """Small model for summary provenance checks."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a simple tensor expression."""

        return x * 2


@pytest.mark.slow
def test_text_autoroute_sets_input_preprocessor() -> None:
    """Text auto-route should record tokenizer preprocessing provenance."""

    transformers = pytest.importorskip("transformers")
    model = transformers.DistilBertModel.from_pretrained("distilbert-base-uncased")

    log = tl.trace(model, "Hello world!", layers_to_save="none")

    assert log.input_preprocessor is not None
    assert log.input_preprocessor.source == "hf_auto_tokenizer"
    assert log.input_preprocessor.verified is True
    assert log.input_preprocessor.config["tokenizer_name"]


def test_summary_includes_input_preprocessing_when_present() -> None:
    """Trace summary should show preprocessing provenance when present."""

    log = tl.trace(_TinyModel(), torch.ones(1), layers_to_save="none")
    log.input_preprocessor = ResolvedPreprocessing(
        source="imagenet_default",
        identifier="ImageNet-default-resize256-crop224",
        verified=False,
        config={},
        description="ImageNet default (UNVERIFIED): resize 256",
    )

    summary = log.summary()

    assert "Input preprocessing:" in summary
    assert "ImageNet default (UNVERIFIED): resize 256" in summary
    assert "WARNING: input preprocessing is UNVERIFIED" not in summary


def test_summary_can_include_unverified_input_preprocessing_detail() -> None:
    """Trace summary should optionally show unverified preprocessing detail."""

    log = tl.trace(_TinyModel(), torch.ones(1), layers_to_save="none")
    log.input_preprocessor = ResolvedPreprocessing(
        source="imagenet_default",
        identifier="ImageNet-default-resize256-crop224",
        verified=False,
        config={},
        description="ImageNet default (UNVERIFIED): resize 256",
    )

    summary = log.summary(show_input_preprocessing_details=True)

    assert "status: UNVERIFIED; source=imagenet_default" in summary
    assert "WARNING: input preprocessing is UNVERIFIED" in summary


def test_input_transform_summary_render_is_opt_in() -> None:
    """Input preprocessing text should render only with the draw affordance enabled."""

    log = tl.trace(_TinyModel(), torch.ones(1), layers_to_save="none")
    log.raw_input = "hello world"
    default_dot = log.draw(vis_save_only=True, vis_fileformat="dot")
    log.input_preprocessor = ResolvedPreprocessing(
        source="unit_test",
        identifier="tiny",
        verified=True,
        config={},
        description="unit-test transform",
    )

    off_dot = log.draw(vis_save_only=True, vis_fileformat="dot")
    on_dot = log.draw(
        vis_save_only=True,
        vis_fileformat="dot",
        show_input_transform_summary=True,
    )

    assert off_dot == default_dot
    assert "unit-test transform" not in off_dot
    assert "unit-test transform" in on_dot
    assert "preprocess" in on_dot


def test_summary_omits_input_preprocessing_when_absent() -> None:
    """Trace summary should omit preprocessing provenance when absent."""

    log = tl.trace(_TinyModel(), torch.ones(1), layers_to_save="none")

    summary = log.summary()

    assert "Input preprocessing:" not in summary
