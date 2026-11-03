"""collapse 2 and 6: the boundary bundle and the K_CAP output flood.

collapse 2 (memo D3): in an overview render, three or more like model-output
leaves made inside ONE top-level module that renders as a box draw as one
node, counted 1 in the plan, in both backends. collapse 6: an output flood that
pushes the root's own units past ``K_CAP`` no longer empties the frontier; the
cap itself is unchanged.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._visualization.collapse_optimizer import select_collapse_plan
from torchlens._visualization.collapse_plan import RenderContext, collapse_plan_for_trace, count


class _Body(nn.Module):
    """Hidden path plus a cache of ``n`` like tensors, all made in this module."""

    def __init__(self, n: int) -> None:
        """Initialize the hidden path."""
        super().__init__()
        self.proj = nn.Linear(4, 4)
        self.mix = nn.Linear(4, 4)
        self.n = n

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Return the hidden state and the cache."""
        h = self.mix(torch.relu(self.proj(x)))
        return h, tuple(h * float(i + 1) for i in range(self.n))


class _CacheModel(nn.Module):
    """A body that returns a cache flood beside one named primary output."""

    def __init__(self, n: int = 6) -> None:
        """Initialize body and head."""
        super().__init__()
        self.body = _Body(n)
        self.head = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> dict[str, object]:
        """Return the logits and the cache."""
        h, cache = self.body(x)
        return {"logits": self.head(h), "cache": cache}


def _trace(n: int) -> tl.Trace:
    torch.manual_seed(0)
    return tl.trace(_CacheModel(n).eval(), torch.randn(2, 4))


def _boxes(address: str) -> Callable[[object], bool]:
    return lambda module: getattr(module, "address", None) == address


def _dot(trace: tl.Trace, tmp_path: Path, name: str, **kwargs: object) -> str:
    return str(trace.draw(path=str(tmp_path / name), file_format="dot", **kwargs).dot)


@pytest.mark.parametrize("mode", ["max"])
def test_bundle_keeps_scored_plan_and_emitted_counts_equal(mode: str) -> None:
    """Three-way parity holds in an overview context that bundles."""
    trace = _trace(6)
    try:
        context = RenderContext(collapse=mode)
        result = select_collapse_plan(trace, context, mode=mode)  # type: ignore[arg-type]
        plan_count = count(result.plan)
        assert result.visible_count == plan_count
        if result.scored_k is not None:
            assert result.scored_k == plan_count
        if result.segments:
            # Segment plans are emitted through renderer attributes, not the
            # predicate; the existing parity gate skips this leg for them too.
            return
        selected = set(result.selected)
        emitted = collapse_plan_for_trace(
            trace,
            (lambda module: module.address in selected) if selected else None,  # type: ignore[attr-defined]
            dict(result.repeat_folds),
            context,
        )
        assert count(emitted) == plan_count
    finally:
        trace.cleanup()


def test_gpt2_lm_head_cache_bundles_when_the_transformer_is_one_box(tmp_path: Path) -> None:
    """GPT-2 LM head with the cache on: one box makes every KV leaf, so they bundle."""
    transformers = pytest.importorskip("transformers")
    config = transformers.GPT2Config(
        n_layer=4, n_head=2, n_embd=16, n_positions=16, n_ctx=16, vocab_size=100
    )
    torch.manual_seed(0)
    model = transformers.GPT2LMHeadModel(config).eval()
    trace = tl.trace(model, torch.randint(0, 100, (1, 8)))
    try:
        overview = RenderContext(collapse="auto")
        boxed = count(collapse_plan_for_trace(trace, _boxes("transformer"), {}, overview))
        unbundled = count(
            collapse_plan_for_trace(trace, _boxes("transformer"), {}, RenderContext())
        )
        # 8 KV leaves: keys and values bundle separately (4 each), so 8 -> 2.
        assert unbundled - boxed == 6
        source = _dot(trace, tmp_path, "gpt2", collapse="auto", collapse_fn=_boxes("transformer"))
        assert "OUTPUTS: past_key_values.keys" in source
        assert "OUTPUTS: past_key_values.values" in source
        assert "OUTPUTS: logits" not in source
    finally:
        trace.cleanup()
