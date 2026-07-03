"""Tests for the R3a v2 collapse optimizer."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import time
from types import ModuleType

import pytest
import torch
import torchvision.models as tvm

import torchlens as tl
from torchlens.visualization import auto_collapse
from torchlens.visualization.auto_collapse import (
    analyze_collapse,
    resolve_collapse_fn,
    resolve_run_folds,
)
from torchlens.visualization.collapse_optimizer import (
    K_CAP,
    MAX_SALIENCE_FLOOR,
    OptimizerWeights,
    _FrontierPoint,
    _OptimizerState,
    _RESULT_CACHE,
    _branch_salience,
    _child_address_map,
    _eligible_module_box,
    _max_box_salience_score,
    _optimizer_total_units,
    _plan_respects_max_dominance,
    _prune_frontier,
    _rendered_own_unit_map,
    _rendered_module_hidden_counts,
    _structural_digest_map,
    build_role_components,
    collapse_schedule,
    select_collapse_plan,
)
from torchlens.visualization.collapse_plan import (
    ChildSegment,
    CollapsePlan,
    ModuleBox,
    RenderContext,
    RunFold,
    SegmentDescriptor,
    count,
    plan_from_v1,
)


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


class SegmentPrefixTail(torch.nn.Module):
    """Same-role stack where only a prefix is legal to segment."""

    def __init__(self, width: int = 8) -> None:
        """Initialize the prefix/tail segment fixture."""

        super().__init__()
        self.b0 = TinyBlock(width)
        self.b1 = TinyBlock(width)
        self.b2 = TinyBlock(width)
        self.b3 = TinyBlock(width)
        self.head = torch.nn.Linear(width, width)
        self.aux = torch.nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a sequential prefix with a fan-out tail."""

        x = self.b0(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        return self.head(x) + self.aux(x)


class ReusedCallBlock(torch.nn.Module):
    """Small multi-op block reused across recurrent call counts."""

    def __init__(self, width: int = 4) -> None:
        """Initialize the reused block."""

        super().__init__()
        self.fc = torch.nn.Linear(width, width)
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the block."""

        return self.relu(self.fc(x))


class UnevenReusedSiblings(torch.nn.Module):
    """Digest-identical sibling modules called different numbers of times."""

    def __init__(self, width: int = 4) -> None:
        """Initialize uneven reused siblings."""

        super().__init__()
        self.short = ReusedCallBlock(width)
        self.long = ReusedCallBlock(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run sibling modules with different recurrence counts."""

        for _ in range(3):
            x = self.short(x)
        for _ in range(5):
            x = self.long(x)
        return x


class InteriorJunctionBlock(torch.nn.Module):
    """Block with a residual junction fully inside the module subtree."""

    def __init__(self, width: int = 8) -> None:
        """Initialize the interior-junction block."""

        super().__init__()
        self.pre = torch.nn.Linear(width, width)
        self.left = torch.nn.Linear(width, width)
        self.right = torch.nn.Linear(width, width)
        self.post = torch.nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the block."""

        hidden = self.pre(x)
        return self.post(self.left(hidden) + self.right(hidden))


class BoundaryJunctionBlock(torch.nn.Module):
    """Block with a module-output residual join fed by the module input."""

    def __init__(self, width: int = 8) -> None:
        """Initialize the boundary-junction block."""

        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(width, width),
            torch.nn.ReLU(),
            torch.nn.Linear(width, width),
            torch.nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the block."""

        return self.net(x) + x


class SingleBlockWrapper(torch.nn.Module):
    """Wrapper exposing one named block for signal assertions."""

    def __init__(self, block: torch.nn.Module, width: int = 8) -> None:
        """Initialize the wrapper."""

        super().__init__()
        self.block = block
        self.out = torch.nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the wrapped block and output layer."""

        return self.out(self.block(x))


class GRUWrapper(torch.nn.Module):
    """GRU fixture for rolled optimizer coverage."""

    def __init__(self) -> None:
        """Initialize the GRU wrapper."""

        super().__init__()
        self.rnn = torch.nn.GRU(8, 8, batch_first=True)
        self.head = torch.nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the GRU wrapper."""

        y, _ = self.rnn(x)
        return self.head(y[:, -1])


class ParallelBranch(torch.nn.Module):
    """Small convolution branch used by salience-floor fixtures."""

    def __init__(self, width: int = 4) -> None:
        """Initialize the branch."""

        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(width, width, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(width, width, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the branch."""

        return self.net(x)


class UniqueWideFanHead(torch.nn.Module):
    """One-off four-way fan head with a visible concat junction."""

    def __init__(self, width: int = 4) -> None:
        """Initialize the fan head."""

        super().__init__()
        self.branches = torch.nn.ModuleList([ParallelBranch(width) for _ in range(4)])
        self.project = torch.nn.Conv2d(width * 4, width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run four parallel branches and project their concatenation."""

        return self.project(torch.cat([branch(x) for branch in self.branches], dim=1))


class UniqueWideFanModel(torch.nn.Module):
    """Model with a single salient head/neck fan."""

    def __init__(self, width: int = 4) -> None:
        """Initialize the unique fan model."""

        super().__init__()
        self.stem = torch.nn.Conv2d(width, width, 1)
        self.head = UniqueWideFanHead(width)
        self.out = torch.nn.Conv2d(width, width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model."""

        return self.out(self.head(self.stem(x)))


class RepeatedWideFanModel(torch.nn.Module):
    """Model with repeated wide fans that should have low uniqueness."""

    def __init__(self, depth: int = 4, width: int = 4) -> None:
        """Initialize the repeated fan model."""

        super().__init__()
        self.blocks = torch.nn.ModuleList([UniqueWideFanHead(width) for _ in range(depth)])
        self.out = torch.nn.Conv2d(width, width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the repeated fan model."""

        for block in self.blocks:
            x = block(x)
        return self.out(x)


class WidthTwoResidualBlock(torch.nn.Module):
    """Residual block whose internal branch width is only two."""

    def __init__(self, width: int = 4) -> None:
        """Initialize the residual block."""

        super().__init__()
        self.left = ParallelBranch(width)
        self.right = ParallelBranch(width)
        self.project = torch.nn.Conv2d(width, width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a width-two residual-style junction."""

        return self.project(self.left(x) + self.right(x))


class WidthTwoResidualModel(torch.nn.Module):
    """Model containing one width-two residual junction."""

    def __init__(self, width: int = 4) -> None:
        """Initialize the width-two residual fixture."""

        super().__init__()
        self.block = WidthTwoResidualBlock(width)
        self.out = torch.nn.Conv2d(width, width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model."""

        return self.out(self.block(x))


def _trace(model: torch.nn.Module, x: torch.Tensor) -> tl.Trace:
    """Trace a model in eval mode."""

    model.eval()
    grad_enabled = torch.is_grad_enabled()
    torch.set_grad_enabled(False)
    try:
        return tl.trace(model, x)
    finally:
        torch.set_grad_enabled(grad_enabled)


def _plan_signature(plan: CollapsePlan) -> tuple[str, ...]:
    """Return a deterministic structural signature for a collapse plan."""

    return tuple(repr(node) for node in plan.nodes)


def _landmark_swallow_count(trace: tl.Trace, result: object) -> int:
    """Return selected boxes or child segments with hard landmark blockers."""

    analysis = analyze_collapse(trace)
    total = sum(
        1
        for address in getattr(result, "selected", ())
        if analysis.signals.get(address) is not None
        and analysis.signals[address].landmark_edges >= 2
    )
    for segment in (getattr(result, "segments", {}) or {}).values():
        if getattr(segment, "kind", "") != "child":
            continue
        for address in getattr(segment, "members", ()):
            signal = analysis.signals.get(address)
            if signal is not None and signal.landmark_edges >= 2:
                total += 1
    return total


def _optimizer_state_for_floor(trace: tl.Trace, context: RenderContext) -> _OptimizerState:
    """Return an optimizer state configured with the max-mode salience floor."""

    analysis = analyze_collapse(trace)
    child_addresses = _child_address_map(trace)
    return _OptimizerState(
        trace=trace,
        context=context,
        analysis=analysis,
        child_addresses=child_addresses,
        hidden_counts=_rendered_module_hidden_counts(trace, context),
        structural_digests=_structural_digest_map(trace, child_addresses, analysis),
        expanded_cache={},
        role_components_cache={},
        child_segments_cache={},
        single_member_expanded_cache={},
        box_cost_cache={},
        branch_salience_cache={},
        weights=OptimizerWeights(),
        g_star=1.0,
        total_ops=_optimizer_total_units(trace, context),
        allow_folds=True,
        allow_segments=True,
        max_salience_floor=MAX_SALIENCE_FLOOR,
        rendered_own_units=_rendered_own_unit_map(trace, context),
    )


def _require_transformers() -> ModuleType:
    """Import transformers or skip the calling test."""

    return pytest.importorskip("transformers")


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


def test_engine_default_routes_v2_env_override_and_rolled_auto(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default auto engine is v2, env override works, and rolled uses v2."""

    trace = _trace(UniformStack(depth=4), torch.randn(2, 8))
    try:
        monkeypatch.delenv("TORCHLENS_COLLAPSE_ENGINE", raising=False)
        v2_fn = resolve_collapse_fn(trace, "auto", "unrolled", context=RenderContext())
        assert hasattr(v2_fn, "_torchlens_v2_result")

        monkeypatch.setenv("TORCHLENS_COLLAPSE_ENGINE", "v1")
        v1_fn = resolve_collapse_fn(trace, "auto", "unrolled", context=RenderContext())
        assert not hasattr(v1_fn, "_torchlens_v2_result")

        monkeypatch.setenv("TORCHLENS_COLLAPSE_ENGINE", "v2")
        rolled_context = RenderContext(vis_mode="rolled")
        rolled_fn = resolve_collapse_fn(trace, "auto", "rolled", context=rolled_context)
        assert hasattr(rolled_fn, "_torchlens_v2_result")

        max_fn = resolve_collapse_fn(trace, "max", "unrolled", context=RenderContext())
        assert hasattr(max_fn, "_torchlens_v2_result")
    finally:
        trace.cleanup()


def test_float_collapse_level_validation_and_endpoints() -> None:
    """Float collapse levels validate and preserve none/max endpoint plans."""

    trace = _trace(UniformStack(depth=6), torch.randn(2, 8))
    context = RenderContext()
    try:
        none_plan = plan_from_v1(trace, None, None, context)
        max_result = select_collapse_plan(trace, context, mode="max")

        with pytest.raises(ValueError, match=r"\[0\.0, 1\.0\]"):
            trace.collapse_plan(mode=-0.1)
        with pytest.raises(ValueError, match=r"\[0\.0, 1\.0\]"):
            trace.draw(collapse=1.1, vis_save_only=True)

        assert trace.collapse_plan(mode=0.0) == none_plan
        assert trace.collapse_plan(mode=1.0) == max_result.plan
    finally:
        trace.cleanup()


def test_float_collapse_level_is_deterministic() -> None:
    """Same trace and float level produce identical plans across calls."""

    trace = _trace(UniformStack(depth=8), torch.randn(2, 8))
    context = RenderContext()
    try:
        first = trace.collapse_plan(mode=0.5, context=context)
        second = trace.collapse_plan(mode=0.5, context=context)
        assert _plan_signature(first) == _plan_signature(second)

        first_schedule = collapse_schedule(trace, context)
        second_schedule = collapse_schedule(trace, context)
        assert tuple(
            (step.t, step.visible_count, tuple(sorted(step.collapsed_addresses)))
            for step in first_schedule.steps
        ) == tuple(
            (step.t, step.visible_count, tuple(sorted(step.collapsed_addresses)))
            for step in second_schedule.steps
        )
    finally:
        trace.cleanup()


@pytest.mark.heavy
@pytest.mark.parametrize(
    ("name", "builder", "x"),
    [
        ("resnet50", lambda: tvm.resnet50(weights=None), torch.randn(1, 3, 224, 224)),
        ("vit_b_16", lambda: tvm.vit_b_16(weights=None), torch.randn(1, 3, 224, 224)),
        ("maxvit_t", lambda: tvm.maxvit_t(weights=None), torch.randn(1, 3, 224, 224)),
        ("mobilenet_v2", lambda: tvm.mobilenet_v2(weights=None), torch.randn(1, 3, 224, 224)),
        ("densenet201", lambda: tvm.densenet201(weights=None), torch.randn(1, 3, 224, 224)),
    ],
)
def test_float_collapse_schedule_monotone_and_nested(
    name: str,
    builder: Callable[[], torch.nn.Module],
    x: torch.Tensor,
) -> None:
    """Float collapse schedule is monotone and nesting-coherent on requested models."""

    _ = name
    torch.set_num_threads(4)
    trace = _trace(builder(), x)
    context = RenderContext()
    try:
        schedule = trace.collapse_schedule(context)
        none_plan = plan_from_v1(trace, None, None, context)
        max_plan = select_collapse_plan(trace, context, mode="max").plan

        sampled = [schedule.at(index / 10.0) for index in range(11)]
        assert sampled[0].plan == none_plan
        assert sampled[-1].plan == max_plan

        previous_count = sampled[0].visible_count
        previous_addresses = sampled[0].collapsed_addresses
        for step in sampled[1:]:
            assert step.visible_count <= previous_count
            assert previous_addresses <= step.collapsed_addresses
            previous_count = step.visible_count
            previous_addresses = step.collapsed_addresses
    finally:
        trace.cleanup()


def test_rolled_v2_memo_separates_digest_identical_different_num_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rolled v2 keeps recurrence counts in labels for digest-identical siblings."""

    monkeypatch.setenv("TORCHLENS_COLLAPSE_ENGINE", "v2")
    trace = _trace(UnevenReusedSiblings(), torch.randn(2, 4))
    try:
        context = RenderContext(vis_mode="rolled")
        collapse_fn = resolve_collapse_fn(trace, "auto", "rolled", context=context)
        result = getattr(collapse_fn, "_torchlens_v2_result")
        rendered_plan = plan_from_v1(trace, collapse_fn, result.run_folds, context)

        assert not result.declined
        assert trace.modules["short"].num_calls == 3
        assert trace.modules["long"].num_calls == 5
        assert count(rendered_plan) == result.visible_count
    finally:
        trace.cleanup()


def test_rolled_v2_gru_auto_is_non_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rolled v2 auto produces a non-empty recurrent cut."""

    monkeypatch.setenv("TORCHLENS_COLLAPSE_ENGINE", "v2")
    trace = _trace(GRUWrapper(), torch.randn(1, 6, 8))
    try:
        context = RenderContext(vis_mode="rolled")
        none_count = count(plan_from_v1(trace, None, None, context))
        collapse_fn = resolve_collapse_fn(trace, "auto", "rolled", context=context)
        result = getattr(collapse_fn, "_torchlens_v2_result")

        assert not result.declined
        assert result.selected
        assert 0 < result.visible_count < none_count
        assert "rnn" in result.selected
    finally:
        trace.cleanup()


def test_gpt2_small_config_auto_collapses_blocks_without_landmark_swallows() -> None:
    """GPT-2 small-config auto renders transformer blocks as honest boxes."""

    transformers = _require_transformers()
    config = transformers.GPT2Config(
        n_layer=2,
        n_head=2,
        n_embd=16,
        n_positions=16,
        n_ctx=16,
        vocab_size=100,
    )
    trace = _trace(transformers.GPT2Model(config), torch.randint(0, 100, (1, 8)))
    try:
        result = select_collapse_plan(trace, RenderContext(), mode="auto")
        analysis = analyze_collapse(trace)

        assert result.visible_count in range(18, 29)
        assert {"h.0", "h.1"}.issubset(result.selected)
        assert _landmark_swallow_count(trace, result) == 0
        assert analysis.signals["h.0"].landmark_edges == 0
        assert analysis.signals["h.1"].landmark_edges == 0
    finally:
        trace.cleanup()


def test_bert_and_distilbert_small_config_auto_cuts_stay_pinned() -> None:
    """BERT-family small-config cuts do not move while freeing GPT-2 blocks."""

    transformers = _require_transformers()
    bert_config = transformers.BertConfig(
        num_hidden_layers=2,
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=2,
        vocab_size=100,
        max_position_embeddings=16,
    )
    bert_trace = _trace(transformers.BertModel(bert_config), torch.randint(0, 100, (1, 8)))
    try:
        bert_result = select_collapse_plan(bert_trace, RenderContext(), mode="auto")
        assert bert_result.visible_count == 23
        assert bert_result.selected == {
            "encoder.layer.0",
            "encoder.layer.1",
        }
    finally:
        bert_trace.cleanup()

    distil_config = transformers.DistilBertConfig(
        n_layers=2,
        dim=16,
        hidden_dim=32,
        n_heads=2,
        vocab_size=100,
        max_position_embeddings=16,
    )
    distil_trace = _trace(
        transformers.DistilBertModel(distil_config),
        torch.randint(0, 100, (1, 8)),
    )
    try:
        distil_result = select_collapse_plan(distil_trace, RenderContext(), mode="auto")
        assert distil_result.visible_count == 18
        assert distil_result.selected == {
            "embeddings",
            "transformer.layer.0.attention",
            "transformer.layer.0.ffn",
            "transformer.layer.1.attention",
            "transformer.layer.1.ffn",
        }
    finally:
        distil_trace.cleanup()


def test_landmark_signals_ignore_interior_junctions_but_keep_boundary_merges() -> None:
    """Interior joins are free, while output merges with external input stay protected."""

    interior_trace = _trace(
        SingleBlockWrapper(InteriorJunctionBlock()),
        torch.randn(2, 8),
    )
    try:
        interior_signal = analyze_collapse(interior_trace).signals["block"]
        assert interior_signal.eligible
        assert interior_signal.landmark_edges == 0
        assert interior_signal.passthrough_edges == 0
    finally:
        interior_trace.cleanup()

    boundary_trace = _trace(
        SingleBlockWrapper(BoundaryJunctionBlock()),
        torch.randn(2, 8),
    )
    try:
        boundary_signal = analyze_collapse(boundary_trace).signals["block"]
        assert boundary_signal.eligible
        assert boundary_signal.landmark_edges == 0
        assert boundary_signal.passthrough_edges == 1
    finally:
        boundary_trace.cleanup()


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


def test_trace_collapse_plan_public_diagnostic(monkeypatch: pytest.MonkeyPatch) -> None:
    """Trace.collapse_plan returns the v2 plan without adding top-level API names."""

    monkeypatch.setenv("TORCHLENS_COLLAPSE_ENGINE", "v2")
    trace = _trace(UniformStack(depth=12), torch.randn(2, 8))
    try:
        plan = trace.collapse_plan(mode="auto")
        summary = repr(plan)

        assert isinstance(plan, CollapsePlan)
        assert count(plan) > 0
        assert summary.startswith("CollapsePlan(total=")
        assert "module_box=" in summary or "raw_op=" in summary

        with pytest.raises(ValueError, match="mode must be one of"):
            trace.collapse_plan(mode="none")  # type: ignore[arg-type]
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


def test_max_dominance_guard_rejects_oversized_visible_units() -> None:
    """Max-mode dominance validation rejects oversized boxes and segments."""

    trace = _trace(SequentialEncoderHead(), torch.randn(1, 4))
    try:
        analysis = analyze_collapse(trace)
        context = RenderContext()
        box_plan = CollapsePlan(nodes=(ModuleBox("encoder:1"),), context=context)
        assert not _plan_respects_max_dominance(trace, analysis, box_plan, {}, 0.1)

        segment = SegmentDescriptor(
            name="encoder_0__segment__encoder_10pass1",
            kind="child",
            label="encoder.0-10",
            members=("encoder.0", "encoder.2", "encoder.4"),
            ops=(),
            owner=None,
            num_ops=len(trace.ops),
            num_params=0,
        )
        segment_plan = CollapsePlan(
            nodes=(ChildSegment(segment.members),),
            context=context,
        )
        assert not _plan_respects_max_dominance(
            trace,
            analysis,
            segment_plan,
            {segment.name: segment},
            0.75,
        )
    finally:
        trace.cleanup()


def test_max_dp_segments_legal_prefix_and_keeps_fanout_tail_visible() -> None:
    """Max-mode DP can segment a legal prefix before a required visible tail."""

    trace = _trace(SegmentPrefixTail(), torch.randn(2, 8))
    try:
        context = RenderContext()
        auto = select_collapse_plan(trace, context, mode="auto")
        result = select_collapse_plan(trace, context, mode="max")

        child_segments = [node for node in result.plan.nodes if isinstance(node, ChildSegment)]

        assert count(result.plan) < count(auto.plan)
        assert child_segments
        assert any(segment.members == ("b0", "b1", "b2") for segment in child_segments)
        assert "b3" in result.selected
        assert result.level != "L3"
        assert _landmark_swallow_count(trace, result) == 0
    finally:
        trace.cleanup()


def test_max_salience_floor_fires_on_synthetic_unique_wide_fan() -> None:
    """Max mode keeps a parallel-fan hint for a unique wide head."""

    trace = _trace(UniqueWideFanModel(), torch.randn(1, 4, 8, 8))
    try:
        context = RenderContext()
        state = _optimizer_state_for_floor(trace, context)
        signal = state.analysis.signals["head"]
        result = select_collapse_plan(trace, context, mode="max")

        assert _branch_salience("head", state) == 0.75
        assert _max_box_salience_score("head", signal, state) == MAX_SALIENCE_FLOOR
        assert not _eligible_module_box(state, "head", signal)
        assert "head" not in result.selected
        assert any(isinstance(node, RunFold) for node in result.plan.nodes)
        assert any("head.branches" in repr(node) for node in result.plan.nodes)
    finally:
        trace.cleanup()


def test_max_salience_floor_fold_representative_uses_single_instance_stats(
    tmp_path: Path,
) -> None:
    """Max-mode parallel fold representatives display single-instance stats."""

    trace = _trace(UniqueWideFanModel(), torch.randn(1, 4, 8, 8))
    try:
        result = select_collapse_plan(trace, RenderContext(), mode="max")
        source = str(
            trace.draw(
                vis_outpath=str(tmp_path / "max_floor_rep_stats"),
                vis_save_only=True,
                vis_fileformat="svg",
                vis_node_placement="dot",
                collapse="max",
            )
        )
        run_fold = next(node for node in result.plan.nodes if isinstance(node, RunFold))
        fold = result.run_folds[run_fold.rep.call.rsplit(":", 1)[0]]
        representative = trace.modules[fold.representative]
        aggregate_layers = sum(
            int(getattr(trace.modules[address], "num_layers", 0) or 0) for address in fold.addresses
        )
        aggregate_params = sum(
            int(getattr(trace.modules[address], "num_params", 0) or 0) for address in fold.addresses
        )

        assert aggregate_layers != representative.num_layers
        assert aggregate_params != representative.num_params
        assert f"... +{fold.multiplicity - 1} more {fold.class_name}" in source
        assert f"{representative.num_layers} layers total" in source
        assert f"{representative.num_params} params (all trainable)" in source
        assert f"{aggregate_layers} layers total" not in source
        assert f"{aggregate_params} params (all trainable)" not in source
    finally:
        trace.cleanup()


def test_max_salience_floor_does_not_fire_on_repeated_fans() -> None:
    """Repeated wide fans have low uniqueness and remain box-eligible."""

    trace = _trace(RepeatedWideFanModel(), torch.randn(1, 4, 8, 8))
    try:
        context = RenderContext()
        state = _optimizer_state_for_floor(trace, context)
        signal = state.analysis.signals["blocks.0"]

        assert signal.peer_count >= 4
        assert _branch_salience("blocks.0", state) == 0.75
        assert _max_box_salience_score("blocks.0", signal, state) < MAX_SALIENCE_FLOOR
        assert _eligible_module_box(state, "blocks.0", signal)
    finally:
        trace.cleanup()


def test_max_salience_floor_does_not_fire_on_width_two_residual_junction() -> None:
    """Width-two residual-style junctions stay below the max salience floor."""

    trace = _trace(WidthTwoResidualModel(), torch.randn(1, 4, 8, 8))
    try:
        context = RenderContext()
        state = _optimizer_state_for_floor(trace, context)
        signal = state.analysis.signals["block"]

        assert _branch_salience("block", state) == 0.25
        assert _max_box_salience_score("block", signal, state) < MAX_SALIENCE_FLOOR
        assert _eligible_module_box(state, "block", signal)
    finally:
        trace.cleanup()


def test_auto_plan_unaffected_by_max_salience_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Auto plans do not consult the max-only salience floor constant."""

    trace = _trace(UniqueWideFanModel(), torch.randn(1, 4, 8, 8))
    context = RenderContext()
    try:
        baseline = select_collapse_plan(trace, context, mode="auto").plan
        monkeypatch.setattr(
            "torchlens.visualization.collapse_optimizer.MAX_SALIENCE_FLOOR",
            0.0,
        )
        _RESULT_CACHE[trace].pop((context, "auto"), None)
        auto_collapse_result = select_collapse_plan(trace, context, mode="auto")

        assert _plan_signature(auto_collapse_result.plan) == _plan_signature(baseline)
    finally:
        trace.cleanup()


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
