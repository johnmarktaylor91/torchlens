"""Tests for built-in MLP/FFN recipe behavior."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from torchlens import trace as trace_fn


class FFN(nn.Module):
    """Tiny transformers-4.57+ DistilBERT FFN block (lin1/lin2 children).

    transformers >= 4.57 renamed ``DistilBertFFN`` to ``FFN``; this surrogate
    matches the current class name so the recipe is exercised under it.
    """

    def __init__(self) -> None:
        """Initialize the tiny FFN block."""

        super().__init__()
        self.lin1 = nn.Linear(8, 16)
        self.lin2 = nn.Linear(16, 8)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the two-layer feed-forward block."""

        return self.lin2(self.activation(self.lin1(x)))


class DistilBertFFN(nn.Module):
    """Tiny legacy (transformers < 4.57) DistilBERT FFN block."""

    def __init__(self) -> None:
        """Initialize the tiny legacy FFN block."""

        super().__init__()
        self.lin1 = nn.Linear(8, 16)
        self.lin2 = nn.Linear(16, 8)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the two-layer feed-forward block."""

        return self.lin2(self.activation(self.lin1(x)))


class _FFNModel(nn.Module):
    """Wrapper exposing a named FFN module."""

    def __init__(self, ffn: nn.Module) -> None:
        """Initialize with an FFN-like child."""

        super().__init__()
        self.ffn = ffn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the FFN child."""

        return self.ffn(x)


@pytest.mark.slow
def test_distilbert_ffn_current_class_populates_facets() -> None:
    """The current transformers FFN class name populates non-empty FFN facets.

    Regression: transformers >= 4.57 renamed ``DistilBertFFN`` to ``FFN``. The
    recipe targeted only the legacy name, so DistilBERT FFN facets came back
    empty (silent) on current transformers.
    """

    log = trace_fn(_FFNModel(FFN()), torch.randn(2, 3, 8), layers_to_save="all")
    view = log.modules["ffn"].facets
    assert view.recipe_source == "distilbert_ffn"
    assert view.up_out.shape == (2, 3, 16)
    assert view.intermediate.shape == (2, 3, 16)
    assert view.down_out.shape == (2, 3, 8)
    assert view.output.shape == (2, 3, 8)


@pytest.mark.slow
def test_distilbert_ffn_legacy_class_still_populates_facets() -> None:
    """The legacy transformers FFN class name still populates FFN facets.

    Back-compat: older transformers (< 4.57) exposes ``DistilBertFFN``; the
    recipe must keep matching it so the fix is robust across versions.
    """

    log = trace_fn(_FFNModel(DistilBertFFN()), torch.randn(2, 3, 8), layers_to_save="all")
    view = log.modules["ffn"].facets
    assert view.recipe_source == "distilbert_ffn"
    assert view.up_out.shape == (2, 3, 16)
    assert view.intermediate.shape == (2, 3, 16)
    assert view.down_out.shape == (2, 3, 8)
    assert view.output.shape == (2, 3, 8)


@pytest.mark.slow
def test_real_distilbert_ffn_facets_populate() -> None:
    """A real cached DistilBERT model populates FFN facets on the live class.

    Skipped when the tiny-random DistilBERT checkpoint is not cached so the
    suite stays offline-deterministic.
    """

    pytest.importorskip("transformers")
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoModel  # noqa: PLC0415

    try:
        model = AutoModel.from_pretrained("hf-internal-testing/tiny-random-distilbert")
    except Exception as exc:  # pragma: no cover - cache/network dependent
        pytest.skip(f"tiny-random-distilbert unavailable offline: {exc}")
    model.eval()

    ffn_addrs = [
        name
        for name, mod in model.named_modules()
        if type(mod).__name__ in ("FFN", "DistilBertFFN")
    ]
    assert ffn_addrs, "expected at least one FFN module in DistilBERT"

    log = trace_fn(model, torch.randint(0, 100, (1, 8)), layers_to_save="all")
    view = log.modules[ffn_addrs[0]].facets
    assert view.recipe_source == "distilbert_ffn"
    assert view.up_out.ndim == 3 and view.up_out.shape[0] == 1
    assert view.intermediate.shape == view.up_out.shape
    assert view.down_out.ndim == 3 and view.down_out.shape[0] == 1
    assert view.output.shape == view.down_out.shape
