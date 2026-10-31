"""Unknown ``output_style`` refuses typed with a did-you-mean (walk ruling U06.5).

A typo used to decode nothing silently and then advise passing the option the
user had passed. Built-in styles still decode; a style only a registered
custom detector resolves still works; a style nobody resolves refuses --
before the forward when no custom detector could claim it, after the capture
when the registered custom detectors all declined.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._data_classes.trace import ResolvedPostprocessing
from torchlens._errors import InvalidArgumentError


def _logits_model() -> nn.Module:
    """A tiny classifier head."""

    torch.manual_seed(0)
    return nn.Linear(4, 3)


@pytest.mark.smoke
def test_unknown_style_refuses_before_the_forward_with_did_you_mean() -> None:
    """A typo names the nearest built-in style; the forward never runs."""

    calls: list[int] = []
    model = _logits_model()
    model.register_forward_hook(lambda *_: calls.append(1))
    options = tl.options.CaptureOptions(output_style="clasification")
    with pytest.raises(InvalidArgumentError) as excinfo:
        tl.trace(model, torch.randn(2, 4), capture=options)
    assert excinfo.value.fields["code"] == "output_style_unknown"
    assert "did you mean 'classification'?" in str(excinfo.value)
    assert calls == []


def test_builtin_style_is_unaffected() -> None:
    """A built-in style still resolves through the explicit-style detector."""

    options = tl.options.CaptureOptions(output_style="imagenet")
    trace = tl.trace(nn.Linear(4, 1000), torch.randn(1, 4), capture=options)
    assert trace.output_postprocessor is not None
    assert trace.output_postprocessor.style == "imagenet"


def test_custom_style_resolved_by_a_registered_detector_is_honored() -> None:
    """The documented custom-detector route keeps working."""

    @tl._autoroute.output.register(name="walk_labels", priority=5)
    def walk_labels(outputs: Any, meta: dict[str, Any]) -> ResolvedPostprocessing | None:
        """Resolve output_style='walk_labels'."""

        if meta.get("output_style") != "walk_labels":
            return None
        return ResolvedPostprocessing(
            source="project",
            identifier="walk_labels",
            verified=True,
            config={"id2label": {0: "a", 1: "b", 2: "c"}},
            description="test label bank",
            style="classification",
            selected_output_head=meta.get("output_head"),
            label_source="test",
            label_source_version="v1",
            confidence=1.0,
        )

    try:
        options = tl.options.CaptureOptions(output_style="walk_labels")
        trace = tl.trace(_logits_model(), torch.randn(2, 4), capture=options)
        assert trace.output_postprocessor is not None
        assert trace.output_postprocessor.identifier == "walk_labels"
    finally:
        tl._autoroute.output.unregister("walk_labels")


def test_style_declined_by_every_custom_detector_refuses_after_capture() -> None:
    """With custom detectors registered, the post-capture check refuses."""

    @tl._autoroute.output.register(name="walk_decliner", priority=5)
    def walk_decliner(outputs: Any, meta: dict[str, Any]) -> ResolvedPostprocessing | None:
        """Decline everything."""

        return None

    try:
        options = tl.options.CaptureOptions(output_style="walk_unclaimed")
        with pytest.raises(InvalidArgumentError) as excinfo:
            tl.trace(_logits_model(), torch.randn(2, 4), capture=options)
        assert excinfo.value.fields["code"] == "output_style_unknown"
        assert "no built-in or registered custom output detector" in str(excinfo.value)
    finally:
        tl._autoroute.output.unregister("walk_decliner")
