"""Tests for the R3a v2 collapse optimizer."""

from __future__ import annotations

import time

import pytest
import torch

import torchlens as tl
from torchlens.visualization import auto_collapse
from torchlens.visualization.auto_collapse import (
    analyze_collapse,
    resolve_collapse_fn,
    resolve_run_folds,
)
from torchlens.visualization.collapse_optimizer import (
    K_CAP,
    _FrontierPoint,
    _prune_frontier,
    build_role_components,
    select_collapse_plan,
)
from torchlens.visualization.collapse_plan import RenderContext, count, plan_from_v1


class TinyBlock(torch.nn.Module):
    """Small block used by optimizer tests."""

    def __init__(self, width: int = 8) -> None:
        """Initialize the block."""

        super().__init__()
        self.fc1 = torch.nn.Linear(width, width)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the block."""

        return self.fc2(self.relu(self.fc1(x)))


class UniformStack(torch.nn.Module):
    """Model with same-role sibling blocks."""

    def __init__(self, depth: int = 4, width: int = 8) -> None:
        """Initialize the stack."""

        super().__init__()
        self.b0 = TinyBlock(width)
        self.b1 = TinyBlock(width)
        self.b2 = TinyBlock(width)
        self.b3 = TinyBlock(width)
        self.extra = torch.nn.ModuleList(TinyBlock(width) for _ in range(max(depth - 4, 0)))
        self.out = torch.nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the stack."""

        for block in (self.b0, self.b1, self.b2, self.b3):
            x = block(x)
        for block in self.extra:
            x = block(x)
        return self.out(x)


class SequentialEncoderHead(torch.nn.Module):
    """Same-class children with deliberately different hidden mass."""

    def __init__(self, width: int = 4) -> None:
        """Initialize the encoder/head fixture."""

        super().__init__()
        encoder_layers: list[torch.nn.Module] = []
        for _ in range(40):
            encoder_layers.append(torch.nn.Linear(width, width))
            encoder_layers.append(torch.nn.ReLU())
        self.encoder = torch.nn.Sequential(*encoder_layers)
        self.head = torch.nn.Sequential(torch.nn.Linear(width, width), torch.nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model."""

        return self.head(self.encoder(x))


def _trace(model: torch.nn.Module, x: torch.Tensor) -> tl.Trace:
    """Trace a model in eval mode."""

    model.eval()
    grad_enabled = torch.is_grad_enabled()
    torch.set_grad_enabled(False)
    try:
        return tl.trace(model, x)
    finally:
        torch.set_grad_enabled(grad_enabled)


def test_role_components_connect_uniform_siblings() -> None:
    """Same-class siblings with comparable mass form one role component."""

    trace = _trace(UniformStack(depth=4), torch.randn(2, 8))
    try:
        analysis = analyze_collapse(trace)
        graph = analysis.child_flow_graphs["self"]
        children = [child for child in graph.flow_children if child.startswith("b")]
        components = build_role_components(trace, "self", children, analysis)
        assert tuple(component.members for component in components) == (tuple(children),)
    finally:
        trace.cleanup()


def test_role_components_split_heterogeneous_same_class_sequentials() -> None:
    """Same-class encoder/head siblings may receive different treatments."""

    trace = _trace(SequentialEncoderHead(), torch.randn(1, 4))
    try:
        analysis = analyze_collapse(trace)
        components = build_role_components(trace, "self", ("encoder", "head"), analysis)
        assert tuple(component.members for component in components) == (("encoder",), ("head",))
    finally:
        trace.cleanup()


def test_engine_flag_routes_v2_and_declines_rolled(monkeypatch: pytest.MonkeyPatch) -> None:
    """The v2 engine is opt-in and rolled mode falls back to v1."""

    trace = _trace(UniformStack(depth=4), torch.randn(2, 8))
    try:
        monkeypatch.delenv("TORCHLENS_COLLAPSE_ENGINE", raising=False)
        monkeypatch.setattr(auto_collapse, "COLLAPSE_ENGINE", "v1")
        v1_fn = resolve_collapse_fn(trace, "auto", "unrolled", context=RenderContext())
        assert not hasattr(v1_fn, "_torchlens_v2_result")

        monkeypatch.setenv("TORCHLENS_COLLAPSE_ENGINE", "v2")
        v2_fn = resolve_collapse_fn(trace, "auto", "unrolled", context=RenderContext())
        assert hasattr(v2_fn, "_torchlens_v2_result")

        rolled_context = RenderContext(vis_mode="rolled")
        rolled_fn = resolve_collapse_fn(trace, "auto", "rolled", context=rolled_context)
        assert not hasattr(rolled_fn, "_torchlens_v2_result")
    finally:
        trace.cleanup()


def test_v2_plan_parity_and_determinism(monkeypatch: pytest.MonkeyPatch) -> None:
    """The v2 selected plan is deterministic and matches renderer planning."""

    monkeypatch.setenv("TORCHLENS_COLLAPSE_ENGINE", "v2")
    trace = _trace(UniformStack(depth=12), torch.randn(2, 8))
    try:
        context = RenderContext()
        collapse_fn = resolve_collapse_fn(trace, "auto", "unrolled", context=context)
        folds = resolve_run_folds(trace, collapse_fn, context=context)
        result = getattr(collapse_fn, "_torchlens_v2_result")
        rendered_plan = plan_from_v1(trace, collapse_fn, folds, context)
        second = select_collapse_plan(trace, context)

        assert result.visible_count == count(result.plan)
        assert result.visible_count == count(rendered_plan)
        assert repr(result.plan) == repr(second.plan)
    finally:
        trace.cleanup()


def test_frontier_pruning_respects_caps() -> None:
    """Frontier pruning keeps at most the cap and drops over-cap counts."""

    points = tuple(
        _FrontierPoint(
            k=index,
            cost=float(100 - index),
            nodes=(),
            selected=frozenset({str(index)}),
            folds=(),
            box_costs=(),
        )
        for index in range(1, K_CAP + 20)
    )
    pruned = _prune_frontier(points)
    assert len(pruned) <= 32
    assert all(point.k <= K_CAP for point in pruned)


@pytest.mark.heavy
def test_v2_selection_latency_smoke() -> None:
    """Report a heavy latency smoke bound for a larger synthetic stack."""

    trace = _trace(UniformStack(depth=80), torch.randn(2, 8))
    try:
        start = time.perf_counter()
        result = select_collapse_plan(trace, RenderContext())
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        assert result.visible_count > 0
        assert elapsed_ms < 2000.0
    finally:
        trace.cleanup()
