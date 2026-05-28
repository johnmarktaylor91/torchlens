"""Tests for Hugging Face text-input auto-routing in ``tl.trace``."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
import torchlens.bridge.hf as hf_bridge


class _HFLikeTensorModel(nn.Module):
    """Small tensor model exposing Hugging Face tokenizer metadata."""

    def __init__(self) -> None:
        """Initialize a model with a resolvable HF-style path."""

        super().__init__()
        self.config = SimpleNamespace(name_or_path="distilbert-base-uncased")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a simple tensor expression."""

        return x * 2


class _TextFailingModel(nn.Module):
    """Tiny non-HF module whose existing text path raises before forward."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a simple tensor expression."""

        return x * torch.ones(1)


def _fail_trace_text(model: Any, text: Any, **kwargs: Any) -> object:
    """Fail if auto-routing fires in a pass-through regression test."""

    raise AssertionError("HF bridge auto-route should not fire")


@pytest.mark.slow
def test_string_input_hf_model_matches_trace_text() -> None:
    """String input should auto-route and match direct HF bridge tracing."""

    transformers = pytest.importorskip("transformers")
    model = transformers.DistilBertModel.from_pretrained("distilbert-base-uncased")

    auto_log = tl.trace(model, "Hello world!", layers_to_save="none")
    direct_log = tl.bridge.hf.trace_text(model, "Hello world!", layers_to_save="none")

    assert auto_log.num_ops == direct_log.num_ops
    assert len(auto_log.layer_logs) == len(direct_log.layer_logs)
    assert auto_log.num_modules == direct_log.num_modules


@pytest.mark.slow
def test_list_of_strings_input_hf_model_batches() -> None:
    """List-of-strings input should auto-route as a batch."""

    transformers = pytest.importorskip("transformers")
    model = transformers.DistilBertModel.from_pretrained("distilbert-base-uncased")

    log = tl.trace(model, ["a", "b", "c"], layers_to_save="none")

    assert log.num_ops > 0
    assert log.num_modules > 0


def test_chat_message_input_enables_chat_template(monkeypatch: pytest.MonkeyPatch) -> None:
    """Chat-message input should pass ``chat_template=True`` to the HF bridge."""

    calls: list[dict[str, Any]] = []
    sentinel = object()

    def fake_trace_text(model: Any, text: Any, **kwargs: Any) -> object:
        """Record bridge dispatch arguments."""

        calls.append({"model": model, "text": text, **kwargs})
        return sentinel

    monkeypatch.setattr(hf_bridge, "trace_text", fake_trace_text)
    model = _HFLikeTensorModel()
    messages = [{"role": "user", "content": "hi"}]

    result = tl.trace(model, messages, layers_to_save="none")

    assert result is sentinel
    assert calls[0]["model"] is model
    assert calls[0]["text"] == messages
    assert calls[0]["chat_template"] is True
    assert calls[0]["layers_to_save"] == "none"


def test_explicit_transform_overrides_autoroute(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit ``transform=`` should skip HF bridge dispatch."""

    monkeypatch.setattr(hf_bridge, "trace_text", _fail_trace_text)

    log = tl.trace(
        _HFLikeTensorModel(),
        "text",
        transform=lambda value: torch.ones(1),
        layers_to_save="none",
    )

    assert log.num_ops > 0


def test_tensor_input_passes_through_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tensor input should not auto-route even when the model has HF metadata."""

    monkeypatch.setattr(hf_bridge, "trace_text", _fail_trace_text)

    log = tl.trace(_HFLikeTensorModel(), torch.ones(1), layers_to_save="none")

    assert log.num_ops > 0


def test_non_hf_model_text_input_keeps_existing_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Text input for a non-HF model should fall through to the old path."""

    monkeypatch.setattr(hf_bridge, "trace_text", _fail_trace_text)

    with pytest.raises(TypeError, match="String input requires"):
        tl.trace(_TextFailingModel(), "text", layers_to_save="none")
