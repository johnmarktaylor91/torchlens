"""collapse 2 and 6: the boundary bundle and the K_CAP output flood.

collapse 2 (memo D3): in an overview render, three or more like model-output
leaves made inside ONE top-level module that renders as a box draw as one
node, counted 1 in the plan, in both backends. collapse 6: an output flood that
pushes the root's own units past ``K_CAP`` no longer empties the frontier; the
cap itself is unchanged.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.visualization._collapse_frontier import K_CAP
from torchlens.visualization.collapse_optimizer import select_collapse_plan
from torchlens.visualization.collapse_plan import RenderContext, collapse_plan_for_trace, count


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
    return str(
        trace.draw(
            vis_outpath=str(tmp_path / name), vis_save_only=True, vis_fileformat="dot", **kwargs
        )
    )


def _node_count(source: str) -> int:
    """Count DOT node statements (not edges or graph/node/edge defaults)."""
    statement = re.compile(r'^\s*("[^"]+"|[\w:]+)\s*\[')
    skip = {"graph", "node", "edge"}
    return sum(
        1
        for line in source.splitlines()
        if "->" not in line
        and (match := statement.match(line)) is not None
        and match.group(1) not in skip
    )


def test_bundle_counts_one_only_while_its_producer_is_one_box() -> None:
    """Six like cache leaves count 1 with ``body`` boxed and 6 with it expanded."""
    trace = _trace(6)
    try:
        overview, plain = RenderContext(collapse="auto"), RenderContext()
        boxed = count(collapse_plan_for_trace(trace, _boxes("body"), {}, overview))
        boxed_plain = count(collapse_plan_for_trace(trace, _boxes("body"), {}, plain))
        assert boxed == boxed_plain - 5
        # Expanded producer: six producing ops, so the bundle refuses.
        expanded = count(collapse_plan_for_trace(trace, None, {}, overview))
        assert expanded == count(collapse_plan_for_trace(trace, None, {}, plain))
    finally:
        trace.cleanup()


@pytest.mark.parametrize("mode", ["auto", "max"])
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


@pytest.mark.parametrize("engine", ["dot", "rank"])
def test_bundle_draws_one_node_in_both_backends(tmp_path: Path, engine: str) -> None:
    """A boxed producer draws the cache as one labeled node; logits stay individual."""
    trace = _trace(6)
    try:
        source = _dot(
            trace,
            tmp_path,
            f"bundle-{engine}",
            collapse="auto",
            collapse_fn=_boxes("body"),
            vis_node_placement=engine,
        )
        plain = _dot(trace, tmp_path, f"plain-{engine}", collapse_fn=_boxes("body"))
    finally:
        trace.cleanup()
    assert source.count("OUTPUTS: ") == 1
    assert "OUTPUTS: cache" in source
    assert "6 tensors, 2x4 each" in source
    assert "OUTPUTS: " not in plain
    assert _node_count(plain) - _node_count(source) == 5


def test_two_like_leaves_stay_individual(tmp_path: Path) -> None:
    """Fewer than three like leaves never bundle."""
    trace = _trace(2)
    try:
        source = _dot(trace, tmp_path, "two", collapse="auto", collapse_fn=_boxes("body"))
    finally:
        trace.cleanup()
    assert "OUTPUTS: " not in source


def test_output_flood_past_k_cap_no_longer_empties_the_frontier() -> None:
    """80 cache leaves: the root owned 81 units; bundled it owns 2 and plans normally."""
    assert K_CAP == 64
    trace = _trace(80)
    try:
        result = select_collapse_plan(trace, RenderContext(collapse="auto"), mode="auto")
        flooded = select_collapse_plan(trace, RenderContext(), mode="auto")
    finally:
        trace.cleanup()
    assert flooded.k_cap_exhausted is True
    assert flooded.root_own_units > K_CAP
    assert result.k_cap_exhausted is False
    assert result.planner != "floor_fallback"
    assert result.root_own_units <= 2
    assert "body" in result.selected
    assert result.visible_count < 15


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
