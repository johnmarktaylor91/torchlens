"""``torchlens.postprocessing``: the public home of custom output detectors."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
import torchlens.postprocessing
from torchlens._autoroute import output as _output
from torchlens.postprocessing import ResolvedPostprocessing


class _Fixed(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return torch.tensor([[0.2, 3.0]]) + (x.sum() + self.weight) * 0


def test_namespace_re_exports_the_output_registry() -> None:
    """Every public name is the registry's own object, and nothing else is exported."""

    assert set(torchlens.postprocessing.__all__) == {
        "Detector",
        "ResolvedPostprocessing",
        "info",
        "list",
        "register",
        "snapshot",
        "unregister",
    }
    for name in ("Detector", "info", "list", "register", "snapshot", "unregister"):
        assert getattr(torchlens.postprocessing, name) is getattr(_output, name)
    assert tl.postprocessing is torchlens.postprocessing
    names = [detector.name for detector in tl.postprocessing.list()]
    assert {"hf_config_classifier", "imagenet_verified", "hf_text"} <= set(names)


def test_custom_detector_registered_publicly_decodes_on_demand() -> None:
    """The documented recipe resolves ``decode_output(style=...)`` with no private import."""

    with tl.postprocessing.snapshot():

        @tl.postprocessing.register(name="my_labels", priority=5)
        def my_labels(outputs: Any, meta: dict[str, Any]) -> ResolvedPostprocessing | None:
            if meta.get("style") != "my_labels":
                return None
            return ResolvedPostprocessing(
                source="project",
                identifier="my_labels",
                verified=True,
                config={"id2label": {0: "cat", 1: "dog"}},
                description="project label bank",
                style="classification",
                selected_output_head=meta.get("head"),
                label_source="project-local",
                label_source_version="v1",
                confidence=1.0,
            )

        trace = tl.trace(_Fixed().eval(), torch.ones(1, 2))
        rows = trace.decode_output(style="my_labels")
        assert rows[0]["label"] == "dog"
        assert tl.postprocessing.info("my_labels")["priority"] == 5

        with pytest.raises(ValueError, match=r"tl\.postprocessing\.unregister\('my_labels'\)"):
            tl.postprocessing.register(name="my_labels", priority=5)(my_labels)
        tl.postprocessing.unregister("my_labels")
        assert "my_labels" not in [detector.name for detector in tl.postprocessing.list()]


def test_unknown_style_remedy_names_the_public_registration() -> None:
    """The refusal's remedy points at ``tl.postprocessing.register``."""

    trace = tl.trace(_Fixed().eval(), torch.ones(1, 2))
    with pytest.raises(Exception, match=r"tl\.postprocessing\.register") as excinfo:
        trace.decode_output(style="nonsense_style")
    assert getattr(excinfo.value, "code", None) == "decode_style_unknown"
