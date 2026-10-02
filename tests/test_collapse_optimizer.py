"""Tests for the R3a v2 collapse optimizer."""

from __future__ import annotations

import multiprocessing
import os
import queue
import random
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch

import torchlens as tl
import torchlens.visualization.auto_collapse as auto_collapse
import torchlens.visualization.collapse_optimizer as collapse_optimizer
import torchlens.visualization.source_graph as source_graph_module
from torchlens.data_classes._trace_accessors import TraceOpAccessor
from torchlens.visualization._render_common import format_collapsed_module_contents
from torchlens.visualization.auto_collapse import (
    _condensed_owner_for_op,
    _condensed_owner_map,
    _resolve_relationship_op,
    analyze_collapse,
    resolve_collapse_fn,
    resolve_repeat_folds,
)
from torchlens.visualization.collapse_optimizer import (
    _RESULT_CACHE,
    _SCHEDULE_CACHE,
    K_CAP,
    MAX_SALIENCE_FLOOR,
    OptimizerWeights,
    RoleComponent,
    _branch_salience,
    _child_address_map,
    _child_segment_covered_ops,
    _eligible_module_box,
    _FrontierPoint,
    _max_box_salience_score,
    _optimizer_total_units,
    _OptimizerState,
    _plan_respects_max_dominance,
    _prune_frontier,
    _rendered_module_hidden_counts,
    _rendered_own_unit_map,
    _same_role,
    _segment_is_legal,
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
    RepeatFold,
    SegmentDescriptor,
    collapse_plan_for_trace,
    count,
)

tvm = pytest.importorskip("torchvision.models")


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


class DenseFanUnit(torch.nn.Module):
    """Small unit consuming all earlier dense-stack features."""

    def __init__(self, input_width: int, growth_width: int = 4) -> None:
        """Initialize the dense fan-in unit."""

        super().__init__()
        self.linear = torch.nn.Linear(input_width, growth_width)
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project and activate concatenated earlier features."""

        return self.relu(self.linear(x))


class DenseFanStack(torch.nn.Module):
    """Compact dense-connectivity reproducer for collapse containment."""

    def __init__(self, depth: int = 18, width: int = 4) -> None:
        """Initialize a nested stack with growing dense fan-in."""

        super().__init__()
        self.units = torch.nn.ModuleList(
            DenseFanUnit(width * (index + 1), width) for index in range(depth)
        )
        self.out = torch.nn.Linear(width * (depth + 1), width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run every unit from the concatenation of all earlier features."""

        features = [x]
        for unit in self.units:
            features.append(unit(torch.cat(features, dim=-1)))
        return self.out(torch.cat(features, dim=-1))


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


def _clear_collapse_caches(trace: tl.Trace) -> None:
    """Clear analysis and optimizer caches for one trace.

    Parameters
    ----------
    trace:
        Trace whose next collapse operation must be uncached.
    """

    auto_collapse._ANALYSIS_CACHE.pop(trace, None)
    auto_collapse._OP_ADJACENCY_INDEX_CACHE.pop(trace, None)
    _RESULT_CACHE.pop(trace, None)
    _SCHEDULE_CACHE.pop(trace, None)


def _add_unambiguous_accessor_fallback_collision(trace: tl.Trace) -> str:
    """Add one relationship-form collision that preserves accessor resolution.

    Parameters
    ----------
    trace:
        Recurrent trace to modify for fallback coverage.

    Returns
    -------
    str
        The colliding relationship label.
    """

    input_op = next(op for op in trace.ops if op.is_input)
    collision_op = next(op for op in trace.ops if not op.is_input and not op.is_output)
    collision_op._label_raw = input_op.layer_label
    return str(input_op.layer_label)


def _assert_relationship_resolution_matches_accessor(trace: tl.Trace) -> None:
    """Assert internal relationship resolution is identical to the public accessor.

    Parameters
    ----------
    trace:
        Trace whose stored parent and child labels are being checked.
    """

    labels = {
        label
        for op in trace.ops
        for label in (*getattr(op, "parents", ()), *getattr(op, "children", ()))
    }
    for label in labels:
        try:
            expected = trace.ops[label]
        except Exception as expected_error:
            try:
                _resolve_relationship_op(trace, label)
            except Exception as actual_error:
                assert type(actual_error) is type(expected_error)
                assert str(actual_error) == str(expected_error)
            else:
                pytest.fail(f"relationship resolver accepted accessor error label {label!r}")
        else:
            assert _resolve_relationship_op(trace, label) is expected


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
        output_shape_cache={},
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


def _collapse_containment_child(trace: tl.Trace, result_queue: Any) -> None:
    """Run the dense collapse reproducer inside a spawned child process.

    Parameters
    ----------
    trace:
        Parent-captured trace for the child to optimize.
    result_queue:
        Multiprocessing queue used to return a success or failure payload.
    """

    try:
        torch.set_num_threads(2)
        context = RenderContext()
        _clear_collapse_caches(trace)
        max_result = select_collapse_plan(trace, context, mode="max")
        schedule = collapse_schedule(trace, context)
        if not max_result.plan.nodes or not schedule.steps:
            raise AssertionError("collapse containment produced an empty plan or schedule")
        if schedule.steps[-1].plan != max_result.plan:
            raise AssertionError("collapse schedule endpoint differs from the max plan")
        previous_count = schedule.steps[0].visible_count
        previous_addresses = schedule.steps[0].collapsed_addresses
        for step in schedule.steps[1:]:
            if step.visible_count > previous_count:
                raise AssertionError("collapse schedule visible counts are not monotone")
            if not previous_addresses <= step.collapsed_addresses:
                raise AssertionError("collapse schedule addresses are not nested")
            previous_count = step.visible_count
            previous_addresses = step.collapsed_addresses
        result_queue.put(
            (
                "ok",
                {
                    "max_visible_count": max_result.visible_count,
                    "num_ops": len(trace.ops),
                    "num_steps": len(schedule.steps),
                },
            )
        )
    except BaseException as exc:
        result_queue.put(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        trace.cleanup()


@pytest.mark.parametrize(
    ("builder", "x"),
    (
        (lambda: UniformStack(depth=12), torch.randn(2, 8)),
        (lambda: UniqueWideFanModel(width=4), torch.randn(1, 4, 8, 8)),
    ),
)
def test_collapse_adjacency_index_avoids_feed_forward_fuzzy_resolution(
    monkeypatch: pytest.MonkeyPatch,
    builder: Callable[[], torch.nn.Module],
    x: torch.Tensor,
) -> None:
    """Canonical analysis and rolled schedules never enter fuzzy Op lookup.

    Parameters
    ----------
    monkeypatch:
        Pytest patch helper.
    builder:
        Feed-forward model factory.
    x:
        Example model input.
    """

    analysis_trace = _trace(builder(), x)
    schedule_trace = _trace(builder(), x)

    def reject_fuzzy_lookup(self: TraceOpAccessor, key: str) -> Any:
        """Fail if canonical collapse work enters fuzzy Op lookup."""

        _ = self
        raise AssertionError(f"unexpected fuzzy Op lookup for {key!r}")

    try:
        _clear_collapse_caches(analysis_trace)
        monkeypatch.setattr(TraceOpAccessor, "_resolve_substring", reject_fuzzy_lookup)
        analysis = analyze_collapse(analysis_trace)
        assert analysis.signals
        assert analysis.child_flow_graphs

        _clear_collapse_caches(schedule_trace)
        schedule = collapse_schedule(schedule_trace, RenderContext(vis_mode="rolled"))
        assert schedule.steps
        assert schedule.steps[-1].plan.nodes
    finally:
        analysis_trace.cleanup()
        schedule_trace.cleanup()


def test_collapse_adjacency_index_build_and_visit_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Index construction is once-per-trace and relationship work has a derived bound.

    Parameters
    ----------
    monkeypatch:
        Pytest patch helper.
    """

    trace = _trace(UniformStack(depth=12), torch.randn(2, 8))
    original_index = auto_collapse._op_adjacency_index
    original_resolver = auto_collapse._resolve_relationship_op
    counts = {"index_builds": 0, "relationship_visits": 0}

    def counted_index(
        target: tl.Trace, revision: tuple[object, ...] | None = None
    ) -> Mapping[str, str]:
        """Count cache-miss index constructions."""

        if target not in auto_collapse._OP_ADJACENCY_INDEX_CACHE:
            counts["index_builds"] += 1
        return original_index(target, revision)

    def counted_resolver(
        target: tl.Trace, label: str, revision: tuple[object, ...] | None = None
    ) -> Any:
        """Count relationship-resolution visits."""

        counts["relationship_visits"] += 1
        return original_resolver(target, label, revision)

    def reject_fuzzy_lookup(self: TraceOpAccessor, key: str) -> Any:
        """Fail if an ordinary relationship label requires fuzzy lookup."""

        _ = self
        raise AssertionError(f"unexpected fuzzy Op lookup for {key!r}")

    try:
        _clear_collapse_caches(trace)
        monkeypatch.setattr(auto_collapse, "_op_adjacency_index", counted_index)
        monkeypatch.setattr(auto_collapse, "_resolve_relationship_op", counted_resolver)
        monkeypatch.setattr(TraceOpAccessor, "_resolve_substring", reject_fuzzy_lookup)
        analysis = analyze_collapse(trace)

        stored_references = sum(len(op.parents) + len(op.children) for op in trace.ops)
        hierarchy_multiplier = (
            max(
                (int(getattr(module, "address_depth", 0)) for module in trace.modules),
                default=0,
            )
            + 1
        )
        assert analysis.signals
        assert counts["index_builds"] == 1
        assert 0 < counts["relationship_visits"] <= stored_references * hierarchy_multiplier
    finally:
        trace.cleanup()


def test_recurrent_relationship_fallback_is_bounded_and_equivalent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A colliding recurrent label takes the bounded compatibility fallback.

    Parameters
    ----------
    monkeypatch:
        Pytest patch helper.
    """

    trace = _trace(UnevenReusedSiblings(), torch.randn(2, 4))
    fallback_label = _add_unambiguous_accessor_fallback_collision(trace)
    original = TraceOpAccessor._resolve_substring
    fuzzy_labels: list[str] = []

    def count_fuzzy_lookup(self: TraceOpAccessor, key: str) -> Any:
        """Count and delegate compatibility fallback lookups."""

        fuzzy_labels.append(key)
        return original(self, key)

    try:
        _clear_collapse_caches(trace)
        monkeypatch.setattr(TraceOpAccessor, "_resolve_substring", count_fuzzy_lookup)
        analysis = analyze_collapse(trace)

        assert analysis.signals
        assert 0 < len(fuzzy_labels) <= 4
        assert set(fuzzy_labels) == {fallback_label}
        _assert_relationship_resolution_matches_accessor(trace)
    finally:
        trace.cleanup()


def test_relationship_resolver_preserves_ambiguity_error() -> None:
    """A genuinely ambiguous recurrent relationship keeps the accessor error.

    Vehicle rebuilt for the immutable relation views (finding B1-19a). The plant
    used to be ``relationship_owner.children[0] = <ambiguous label>``, which the
    M6 type break turned into ``TypeError: 'tuple' object does not support item
    assignment`` -- a DEAD plant, i.e. a silently disarmed tripwire, not a fixed
    bug. Whole-sequence ASSIGNMENT is still the sanctioned mutation on a finished
    trace, so the plant now goes through that (and the raw tuple normalizes back
    to the view type), keeping the property under test intact: the internal
    relationship resolver must raise the SAME ambiguity error the public
    accessor raises, never silently pick one of the candidates.
    """

    from torchlens._errors import AmbiguousOpLookupError

    trace = _trace(UnevenReusedSiblings(), torch.randn(2, 4))
    try:
        recurrent_op = next(
            op for op in trace.ops if len(trace.ops.resolve_all(op.layer_label)) > 1
        )
        relationship_owner = next(op for op in trace.ops if op.children)
        original_children = tuple(relationship_owner.children)
        # Pre-flight: the plant must really BE ambiguous, or every assertion
        # below would pass vacuously on a resolvable label.
        with pytest.raises(AmbiguousOpLookupError):
            trace.ops[recurrent_op.layer_label]

        relationship_owner.children = (recurrent_op.layer_label, *original_children[1:])
        assert relationship_owner.children[0] == recurrent_op.layer_label
        auto_collapse._OP_ADJACENCY_INDEX_CACHE.pop(trace, None)

        # Explicit, non-vacuous form of the property: the resolver refuses the
        # planted ambiguous label rather than silently picking a pass.
        with pytest.raises(AmbiguousOpLookupError):
            _resolve_relationship_op(trace, recurrent_op.layer_label)
        _assert_relationship_resolution_matches_accessor(trace)

        relationship_owner.children = original_children
        assert tuple(relationship_owner.children) == original_children
    finally:
        trace.cleanup()


@pytest.mark.heavy
@pytest.mark.serial
def test_collapse_dense_fan_schedule_spawn_containment() -> None:
    """Dense max planning and scheduling cannot hang the parent test process."""

    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    trace = _trace(DenseFanStack(depth=18), torch.randn(2, 4))
    process = context.Process(target=_collapse_containment_child, args=(trace, result_queue))
    timed_out = False
    try:
        process.start()
        process.join(20)
        timed_out = process.is_alive()
        if timed_out:
            process.terminate()
            process.join(2)
            if process.is_alive():
                process.kill()
                process.join(2)

        assert not timed_out, "spawned dense collapse exceeded the 20-second containment deadline"
        assert process.exitcode == 0
        try:
            status, payload = result_queue.get(timeout=2)
        except queue.Empty:
            pytest.fail("spawned dense collapse exited without a result payload")
        assert status == "ok", payload
        assert payload["num_ops"] > 0
        assert payload["num_steps"] > 0
        assert payload["max_visible_count"] > 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join(2)
        result_queue.close()
        result_queue.join_thread()
        trace.cleanup()


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


def _pairwise_role_components(
    trace: Any,
    child_addresses: tuple[str, ...],
    analysis: Any,
    hidden_counts: Mapping[str, int] | None,
) -> tuple[RoleComponent, ...]:
    """Reference all-pairs union-find role partition driven by ``_same_role``."""

    children = tuple(child for child in child_addresses if child in trace.modules)
    parent_index = {child: index for index, child in enumerate(children)}
    parent = {child: child for child in children}

    def find(address: str) -> str:
        current = address
        while parent[current] != current:
            parent[current] = parent[parent[current]]
            current = parent[current]
        return current

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if parent_index[left_root] <= parent_index[right_root]:
            parent[right_root] = left_root
        else:
            parent[left_root] = right_root

    for left_index, left in enumerate(children):
        for right in children[left_index + 1 :]:
            if _same_role(trace, left, right, analysis, hidden_counts or {}):
                union(left, right)
    grouped: dict[str, list[str]] = {}
    for child in children:
        grouped.setdefault(find(child), []).append(child)
    return tuple(
        RoleComponent(tuple(members))
        for _, members in sorted(
            grouped.items(),
            key=lambda item: min(parent_index[member] for member in item[1]),
        )
    )


def _role_stub_inputs(
    labeled_counts: Mapping[str, tuple[str, int]],
) -> tuple[Any, Any]:
    """Return (trace, analysis) stubs for role-component inputs."""

    modules = {
        address: SimpleNamespace(class_name=class_name)
        for address, (class_name, _) in labeled_counts.items()
    }
    signals = {
        address: SimpleNamespace(address=address, hidden_ops=hidden_ops)
        for address, (_, hidden_ops) in labeled_counts.items()
    }
    return SimpleNamespace(modules=modules), SimpleNamespace(signals=signals)


@pytest.mark.smoke
def test_role_components_match_pairwise_union_property() -> None:
    """Sorted-run role components byte-match the all-pairs ``_same_role`` union."""

    for trial in range(200):
        rng = random.Random(trial)
        n_children = rng.randint(0, 32)
        labels = ["Block", "Stage", "Head", "Stem"][: rng.randint(1, 4)]
        scale = rng.choice([3, 40, 500, 100_000])
        labeled_counts = {
            f"m{index}": (rng.choice(labels), rng.randint(0, scale)) for index in range(n_children)
        }
        trace, analysis = _role_stub_inputs(labeled_counts)
        children = list(labeled_counts)
        rng.shuffle(children)
        # Unknown addresses must be filtered out identically by both paths.
        children.insert(rng.randint(0, len(children) or 1), "not_a_module")
        hidden_counts: Mapping[str, int] | None = None
        if rng.random() < 0.5:
            hidden_counts = {
                address: rng.randint(-2, scale) for address in labeled_counts if rng.random() < 0.5
            }
        fast = build_role_components(trace, "self", tuple(children), analysis, hidden_counts)
        reference = _pairwise_role_components(trace, tuple(children), analysis, hidden_counts)
        assert fast == reference, f"trial={trial}"


@pytest.mark.smoke
def test_role_components_chain_connects_beyond_tolerance() -> None:
    """Adjacent-in-mass siblings chain one component past the 1.5 pair tolerance."""

    # Masses log2(1+n) = 0,1,2,3,4: each adjacent gap is 1.0 <= 1.5, but the
    # extremes differ by 4.0, so connectivity must come from chaining.
    labeled_counts = {f"m{index}": ("Block", 2**index - 1) for index in range(5)}
    trace, analysis = _role_stub_inputs(labeled_counts)
    children = tuple(labeled_counts)
    fast = build_role_components(trace, "self", children, analysis)
    assert fast == _pairwise_role_components(trace, children, analysis, None)
    assert fast == (RoleComponent(children),)


def test_default_routes_v2_and_rolled_auto() -> None:
    """The default auto/max and rolled collapse paths use v2."""

    trace = _trace(UniformStack(depth=4), torch.randn(2, 8))
    try:
        v2_fn = resolve_collapse_fn(trace, "auto", "unrolled", context=RenderContext())
        assert hasattr(v2_fn, "_torchlens_v2_result")

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
        none_plan = collapse_plan_for_trace(trace, None, None, context)
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


def test_float_collapse_schedule_reuses_source_graph_and_plan_nodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cold float schedule shares one source graph and equal immutable plan nodes."""

    trace = _trace(UniformStack(depth=8), torch.randn(2, 8))
    context = RenderContext()
    original_build_source_graph = source_graph_module.build_source_graph
    source_graph_calls = 0

    def counted_build_source_graph(
        candidate_trace: tl.Trace,
        request: Any,
    ) -> Any:
        """Count normalized source-graph builds while preserving their result.

        Parameters
        ----------
        candidate_trace:
            Trace being normalized.
        request:
            Resolved rendering request.

        Returns
        -------
        Any
            Normalized source graph returned by the production builder.
        """

        nonlocal source_graph_calls
        source_graph_calls += 1
        return original_build_source_graph(candidate_trace, request)

    monkeypatch.setattr(source_graph_module, "build_source_graph", counted_build_source_graph)
    try:
        _clear_collapse_caches(trace)
        schedule = collapse_schedule(trace, context)

        assert source_graph_calls == 1
        canonical_nodes: dict[Any, Any] = {}
        reused_values = 0
        for step in schedule.steps:
            for node in step.plan.nodes:
                if node in canonical_nodes:
                    reused_values += 1
                    assert canonical_nodes[node] is node
                else:
                    canonical_nodes[node] = node
        assert reused_values > 0
    finally:
        trace.cleanup()


def test_condensed_owner_map_is_first_wins_and_built_once() -> None:
    """The condensed owner inversion reads each child set once and preserves flow priority."""

    class CountingChildSets(dict[str, set[str]]):
        """Child-set mapping that counts indexed reads."""

        reads = 0

        def __getitem__(self, key: str) -> set[str]:
            """Return one child set and count the indexed read.

            Parameters
            ----------
            key:
                Child address to read.

            Returns
            -------
            set[str]
                Operation labels in the requested child subtree.
            """

            type(self).reads += 1
            return super().__getitem__(key)

    flow_children = ("first", "second", "third")
    child_sets = CountingChildSets(
        {
            "first": {"shared", "first_only"},
            "second": {"shared", "second_only"},
            "third": {"third_only"},
        }
    )
    owner_by_op = _condensed_owner_map(flow_children, child_sets)

    assert CountingChildSets.reads == len(flow_children)
    for _ in range(50):
        assert _condensed_owner_for_op("shared", owner_by_op) == "first"
        assert _condensed_owner_for_op("second_only", owner_by_op) == "second"
        assert _condensed_owner_for_op("parent_owned", owner_by_op) == "parent_owned"
    assert CountingChildSets.reads == len(flow_children)


@pytest.fixture
def _restore_torch_num_threads() -> Iterator[None]:
    """Snapshot and restore torch's process-global thread count around a test.

    This test pins ``torch.set_num_threads(4)``; without restoration that value leaks
    into torch's global state for the rest of the pytest process (observed leak:
    baseline 10 -> 4), making later tests order-dependent. The fixture captures the
    count before the test and restores it in ``finally``. (The subprocess-scoped
    ``set_num_threads(2)`` elsewhere in this file is process-contained and needs no
    restoration.)
    """

    original = torch.get_num_threads()
    try:
        yield
    finally:
        torch.set_num_threads(original)


@pytest.mark.heavy
@pytest.mark.parametrize(
    ("name", "builder", "x"),
    [
        ("resnet50", lambda: tvm.resnet50(weights=None), torch.randn(1, 3, 224, 224)),
        ("vit_b_16", lambda: tvm.vit_b_16(weights=None), torch.randn(1, 3, 224, 224)),
        # slow cell (r3settle2 budget lint): the nested-schedule sweep on
        # maxvit_t measures far beyond heavy's 20s ceiling.
        pytest.param(
            "maxvit_t",
            lambda: tvm.maxvit_t(weights=None),
            torch.randn(1, 3, 224, 224),
            marks=pytest.mark.slow,
        ),
        ("mobilenet_v2", lambda: tvm.mobilenet_v2(weights=None), torch.randn(1, 3, 224, 224)),
        # slow cell (r3settle2 budget lint): densenet201's deep module tree
        # measures minutes under the schedule sweep.
        pytest.param(
            "densenet201",
            lambda: tvm.densenet201(weights=None),
            torch.randn(1, 3, 224, 224),
            marks=pytest.mark.slow,
        ),
    ],
)
def test_float_collapse_schedule_monotone_and_nested(
    name: str,
    builder: Callable[[], torch.nn.Module],
    x: torch.Tensor,
    _restore_torch_num_threads: None,
) -> None:
    """Float collapse schedule is monotone and nesting-coherent on requested models."""

    _ = name
    torch.set_num_threads(4)
    trace = _trace(builder(), x)
    context = RenderContext()
    try:
        schedule = trace.collapse_schedule(context)
        none_plan = collapse_plan_for_trace(trace, None, None, context)
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

    trace = _trace(UnevenReusedSiblings(), torch.randn(2, 4))
    try:
        context = RenderContext(vis_mode="rolled")
        collapse_fn = resolve_collapse_fn(trace, "auto", "rolled", context=context)
        result = getattr(collapse_fn, "_torchlens_v2_result")
        rendered_plan = collapse_plan_for_trace(trace, collapse_fn, result.repeat_folds, context)

        assert not result.declined
        assert trace.modules["short"].num_calls == 3
        assert trace.modules["long"].num_calls == 5
        assert count(rendered_plan) == result.visible_count
    finally:
        trace.cleanup()


# test_rolled_v2_gru_auto_is_non_noop moved to
# tests/test_collapse_power_ladder.py (F11 memo D8 rewrote its contract;
# the file sits at the 2000-line unledgered cap).
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
    # The pinned visible counts (BERT 23, DistilBERT 18) are calibrated for the DECLARED
    # supported Transformers range (transformers~=4.45, i.e. >=4.45,<5.0). Transformers v5's
    # "Bert-based Models Attention Refactor" (HF commit 155f7e2e / #38301, 2025-09-19) changed
    # the BERT/DistilBERT program graph, legitimately shifting the honest collapse cut (BERT ->
    # 21). That is an upstream model-graph change, NOT a TorchLens collapse regression (renders
    # verified honest at both counts), so this version-sensitive golden is scoped to the
    # supported range rather than rebaselined.
    if int(transformers.__version__.split(".")[0]) >= 5:
        pytest.skip(
            f"BERT/DistilBERT collapse pins are calibrated for transformers<5.0; installed "
            f"{transformers.__version__} changes the upstream model graph (HF #38301)."
        )
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

    trace = _trace(UniformStack(depth=12), torch.randn(2, 8))
    try:
        context = RenderContext()
        collapse_fn = resolve_collapse_fn(trace, "auto", "unrolled", context=context)
        folds = resolve_repeat_folds(trace, collapse_fn, context=context)
        result = getattr(collapse_fn, "_torchlens_v2_result")
        rendered_plan = collapse_plan_for_trace(trace, collapse_fn, folds, context)
        second = select_collapse_plan(trace, context)

        assert result.visible_count == count(result.plan)
        assert result.visible_count == count(rendered_plan)
        assert repr(result.plan) == repr(second.plan)
    finally:
        trace.cleanup()


def test_trace_collapse_plan_public_diagnostic(monkeypatch: pytest.MonkeyPatch) -> None:
    """Trace.collapse_plan returns the v2 plan without adding top-level API names."""

    trace = _trace(UniformStack(depth=12), torch.randn(2, 8))
    try:
        plan = trace.collapse_plan(mode="auto")
        summary = repr(plan)

        assert isinstance(plan, CollapsePlan)
        assert count(plan) > 0
        assert len(plan) == count(plan)
        assert plan.total == count(plan)
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
        assert any(isinstance(node, RepeatFold) for node in result.plan.nodes)
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
        run_fold = next(node for node in result.plan.nodes if isinstance(node, RepeatFold))
        fold = result.repeat_folds[run_fold.rep.call.rsplit(":", 1)[0]]
        representative = trace.modules[fold.representative]
        aggregate_layers = sum(
            int(getattr(trace.modules[address], "num_layers", 0) or 0) for address in fold.addresses
        )
        aggregate_params = sum(
            int(getattr(trace.modules[address], "num_params", 0) or 0) for address in fold.addresses
        )
        representative_label = format_collapsed_module_contents(
            representative.num_layers,
            sum(trace[label].is_buffer for label in representative.layer_labels),
        )
        aggregate_label = format_collapsed_module_contents(
            aggregate_layers,
            sum(
                trace[label].is_buffer
                for address in fold.addresses
                for label in trace.modules[address].layer_labels
            ),
        )

        assert aggregate_layers != representative.num_layers
        assert aggregate_params != representative.num_params
        assert f"... +{fold.multiplicity - 1} more {fold.class_name}" in source
        assert representative_label in source
        assert f"{representative.num_params} params (all trainable)" in source
        assert aggregate_label not in source
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
        _RESULT_CACHE.pop(trace, None)
        auto_collapse_result = select_collapse_plan(trace, context, mode="auto")

        assert _plan_signature(auto_collapse_result.plan) == _plan_signature(baseline)
    finally:
        trace.cleanup()


@pytest.mark.heavy
@pytest.mark.serial
def test_v2_selection_latency_smoke() -> None:
    """Report a load-scaled heavy latency smoke bound for a larger synthetic stack."""

    trace = _trace(UniformStack(depth=80), torch.randn(2, 8))
    try:
        start = time.perf_counter()
        result = select_collapse_plan(trace, RenderContext())
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        budget_factor = float(os.environ.get("TORCHLENS_TIMING_BUDGET_FACTOR", "2.0"))
        assert result.visible_count > 0
        assert elapsed_ms < 2000.0 * budget_factor
    finally:
        trace.cleanup()


def _reference_longest_legal_segment_prefix(
    members: Sequence[str],
    graph: Any,
    analysis: Any,
    hidden_counts: Mapping[str, int],
    total_ops: int,
    *,
    dominance_limit: float,
    vis_mode: str = "unrolled",
) -> tuple[str, ...] | None:
    """Naive forward prefix scan the optimized segment-run search must match.

    This is the pre-optimization algorithm, kept verbatim as the differential
    oracle: rescan the whole prefix for landmark members, probe chain-interval
    legality for every prefix, recount hidden units from scratch, and keep the
    last prefix that passed all three gates.
    """

    def hidden_units(addresses: tuple[str, ...]) -> int:
        covered_ops = _child_segment_covered_ops(analysis, addresses)
        if covered_ops:
            if vis_mode == "rolled":
                return len({str(label).rsplit(":", 1)[0] for label in covered_ops})
            return len(covered_ops)
        return sum(
            hidden_counts.get(address, analysis.signals[address].hidden_ops)
            for address in addresses
        )

    best: tuple[str, ...] | None = None
    for end in range(2, len(members) + 1):
        candidate = tuple(members[:end])
        if any(analysis.signals[address].landmark_edges >= 2 for address in candidate):
            continue
        if not _segment_is_legal(candidate, graph):
            continue
        hidden = hidden_units(candidate)
        if total_ops > 0 and hidden / total_ops > dominance_limit:
            continue
        best = candidate
    return best


class _SegmentPrefixProbe:
    """Differential + call-count probe around the segment-run prefix search."""

    def __init__(self) -> None:
        """Start an empty probe."""

        self.calls = 0
        self.candidates = 0
        self.legality_probes = 0
        self.hidden_recounts = 0

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Patch the optimizer to cross-check every prefix search."""

        module = "torchlens.visualization.collapse_optimizer"
        real_prefix = collapse_optimizer._longest_legal_segment_prefix
        real_legal = collapse_optimizer._segment_is_legal
        real_covered = collapse_optimizer._child_segment_covered_ops

        def counting_segment_is_legal(addresses: tuple[str, ...], graph: Any) -> bool:
            self.legality_probes += 1
            return real_legal(addresses, graph)

        def counting_covered_ops(analysis: Any, addresses: tuple[str, ...]) -> tuple[str, ...]:
            self.hidden_recounts += 1
            return real_covered(analysis, addresses)

        def checked_prefix(
            members: Sequence[str],
            graph: Any,
            analysis: Any,
            hidden_counts: Mapping[str, int],
            total_ops: int,
            **kwargs: Any,
        ) -> tuple[str, ...] | None:
            self.calls += 1
            self.candidates += max(len(members) - 1, 0)
            fast = real_prefix(members, graph, analysis, hidden_counts, total_ops, **kwargs)
            slow = _reference_longest_legal_segment_prefix(
                members, graph, analysis, hidden_counts, total_ops, **kwargs
            )
            assert fast == slow, (
                f"segment-run prefix diverged for {len(members)} members: {fast!r} != {slow!r}"
            )
            return fast

        monkeypatch.setattr(f"{module}._segment_is_legal", counting_segment_is_legal)
        monkeypatch.setattr(f"{module}._child_segment_covered_ops", counting_covered_ops)
        monkeypatch.setattr(f"{module}._longest_legal_segment_prefix", checked_prefix)


def _exercise_collapse_surfaces(
    model: torch.nn.Module,
    x: torch.Tensor,
    modes: tuple[str | float, ...] = ("auto", "max", 0.0, 0.25, 0.5, 0.75, 1.0),
) -> None:
    """Drive every collapse-planning surface that runs a segment-run search."""

    trace = _trace(model, x)
    try:
        for mode in modes:
            for vis_mode in ("unrolled", "rolled"):
                select_collapse_plan(trace, RenderContext(vis_mode=vis_mode), mode=mode)
        collapse_schedule(trace, RenderContext())
    finally:
        trace.cleanup()


@pytest.mark.heavy
@pytest.mark.parametrize(
    ("factory", "shape"),
    [
        (lambda: UniformStack(depth=24), (2, 8)),
        (lambda: UniqueWideFanModel(), (1, 4, 8, 8)),
        (lambda: SequentialEncoderHead(), (2, 4)),
        (lambda: SegmentPrefixTail(), (2, 8)),
        (lambda: UnevenReusedSiblings(), (2, 4)),
    ],
)
def test_segment_run_prefix_matches_naive_prefix_scan(
    monkeypatch: pytest.MonkeyPatch,
    factory: Callable[[], torch.nn.Module],
    shape: tuple[int, ...],
) -> None:
    """The linearized segment-run search returns the naive scan's exact answer."""

    probe = _SegmentPrefixProbe()
    probe.install(monkeypatch)
    _exercise_collapse_surfaces(factory(), torch.randn(*shape))

    assert probe.calls > 0, "probe never fired; the differential check was vacuous"


class FlatLinearChain(torch.nn.Module):
    """Long chain of sibling linears -- the worst case for segment-run search."""

    def __init__(self, depth: int = 64, width: int = 8) -> None:
        """Initialize the chain."""

        super().__init__()
        self.layers = torch.nn.Sequential(*[torch.nn.Linear(width, width) for _ in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the chain."""

        return self.layers(x)


@pytest.mark.heavy
@pytest.mark.serial
def test_segment_run_prefix_work_does_not_grow_with_component_size() -> None:
    """One long role component costs a bounded search, not one probe per prefix.

    A flat sibling chain puts every member in one role component, so the naive
    forward scan spent one chain-interval probe and one hidden-unit recount on
    every prefix -- quadratic in the chain length. The linearized search settles
    the landmark and dominance gates in one pass and probes down from the
    longest surviving candidate, so its work must stay flat as the chain grows.
    """

    counts: dict[int, tuple[int, int, int]] = {}
    for depth in (64, 128, 256):
        probe = _SegmentPrefixProbe()
        with pytest.MonkeyPatch.context() as monkeypatch:
            probe.install(monkeypatch)
            _exercise_collapse_surfaces(
                FlatLinearChain(depth=depth), torch.randn(1, 8), modes=("max",)
            )
        counts[depth] = (probe.candidates, probe.legality_probes, probe.hidden_recounts)

    for depth, (candidates, probes, recounts) in counts.items():
        assert probes <= candidates, f"depth {depth}: {probes} probes for {candidates} candidates"
        assert recounts <= candidates, f"depth {depth}: {recounts} recounts for {candidates}"
    assert counts[256][0] > 2 * counts[64][0], "the chain component never grew"
    assert counts[256][1] <= 2 * counts[64][1], (
        f"legality probes grew with component size: {counts[64][1]} -> {counts[256][1]}"
    )
    assert counts[256][2] <= 2 * counts[64][2] + 8, (
        f"hidden-unit recounts grew with component size: {counts[64][2]} -> {counts[256][2]}"
    )
    assert counts[256][1] * 10 < counts[256][0], (
        "probes did not fall far below the naive one-per-prefix count: "
        f"{counts[256][1]} probes for {counts[256][0]} candidate prefixes"
    )


def test_segment_descriptor_parity_guard_survives_python_O() -> None:
    """The label-honesty cardinality guards raise, not assert (r-b7 R24-2).

    ``python -O`` strips asserts; these two guards defend the segment-box /
    ``(xN)`` / ellipsis honesty contract on the DEFAULT ``draw(collapse=)``
    path, so they must fire with assertions disabled too.
    """

    from torchlens.visualization.collapse_optimizer import (
        _assert_segment_descriptor_parity,
    )

    with pytest.raises(RuntimeError, match="segment descriptor cardinality"):
        _assert_segment_descriptor_parity((), {"phantom": object()})  # type: ignore[arg-type]


def test_visible_plan_guards_survive_python_O() -> None:
    """The empty-plan honesty guards raise, not assert (T9, grind-p3).

    ``python -O`` strips asserts; the v2 select path and the floor fallback
    both publish plans on the DEFAULT ``draw(collapse=)`` path, so an empty
    plan must fail loudly with assertions disabled too.
    """

    from torchlens.visualization.collapse_optimizer import _assert_visible_plan

    _assert_visible_plan(1, "v2 collapse plan")
    with pytest.raises(RuntimeError, match="v2 collapse plan produced no visible nodes"):
        _assert_visible_plan(0, "v2 collapse plan")
    with pytest.raises(RuntimeError, match="collapse floor fallback produced no visible nodes"):
        _assert_visible_plan(0, "collapse floor fallback")


def test_rank_group_parity_guard_survives_python_O() -> None:
    """The sibling rank-group emission guard raises, not asserts (T9, grind-p3).

    ``python -O`` strips asserts; the queued-vs-emitted parity check runs on
    the DEFAULT ``draw()`` path, so a dropped sibling rank group must fail
    loudly with assertions disabled too.
    """

    # Moved out of _render_dot by 1876dde2 (renderer thinning) into the region
    # emitter that actually calls it; the guard itself is unchanged.
    from torchlens.visualization._render_regions import _assert_rank_group_parity

    _assert_rank_group_parity(3, 3)
    with pytest.raises(RuntimeError, match="sibling rank-group emission mismatch"):
        _assert_rank_group_parity(3, 2)


def test_collapse_optimizer_ops_ceiling_declines_disclosed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Above the ops ceiling the optimizer declines DISCLOSED, never computes.

    b8-sol R60: the frontier selection is superlinear (~n^1.75) with no
    preflight node ceiling, time budget, or typed refusal, so one
    ``draw(collapse="auto"|"max")`` on a several-thousand-op model burned
    CPU-hours. The ceiling must (a) warn and render uncollapsed on the draw
    path, (b) refuse typed from ``Trace.collapse_plan()``, and (c) degrade
    the schedule to its single full-graph step -- while sub-ceiling traces
    are untouched.
    """

    import warnings as warnings_module

    from torch import nn

    from torchlens._errors import InvalidArgumentError
    from torchlens.visualization import collapse_optimizer as optimizer_module

    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4), nn.ReLU())
    trace = tl.trace(model, torch.randn(2, 4))

    # Sub-ceiling: the real ceiling admits this trace and produces a plan.
    assert trace.collapse_plan(mode="max") is not None

    # Pathological arm (F11 memo D5(i)): above 20x the constant, smart
    # collapse declines outright (constant 0 makes any trace qualify).
    monkeypatch.setattr(optimizer_module, "COLLAPSE_OPTIMIZER_MAX_OPS", 0)
    fresh = tl.trace(model, torch.randn(2, 4))
    with pytest.warns(UserWarning, match="skipping smart collapse"):
        dot = fresh.draw(
            collapse="max",
            vis_save_only=True,
            vis_fileformat="dot",
            order_siblings=False,
        )
    assert dot  # uncollapsed render still ships

    with warnings_module.catch_warnings():
        warnings_module.simplefilter("ignore", UserWarning)
        with pytest.raises(InvalidArgumentError) as exc_info:
            fresh.collapse_plan(mode="auto")
    assert exc_info.value.fields["code"] == "collapse_plan_unavailable"
    assert "collapse_ops_ceiling" in exc_info.value.fields["reason"]

    with warnings_module.catch_warnings():
        warnings_module.simplefilter("ignore")
        schedule = fresh.collapse_schedule()
    assert len(schedule.steps) == 1

    # The budget-degrade arm (non-pathological over-budget -> the fallback
    # planner) is pinned in tests/test_collapse_power_ceiling.py.


def test_collapse_ceiling_warning_category_and_attribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ceiling-decline warning is a TorchLensWarning attributed to the caller.

    R19 (hunt-6): the decline advisory was a bare ``UserWarning`` with
    ``stacklevel=2``, which (a) could not be filtered/promoted via the
    ``TorchLensWarning`` taxonomy and (b) blamed an internal torchlens frame
    (``_trace_stats.py``) instead of the user's ``draw()`` call site.
    """

    import warnings as warnings_module

    from torchlens.errors import TorchLensWarning

    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU())
    # F11 memo D5(i): the decline arm is the pathological pre-gate now.
    monkeypatch.setattr(collapse_optimizer, "COLLAPSE_OPTIMIZER_MAX_OPS", 0)
    fresh = tl.trace(model, torch.randn(2, 4))

    with warnings_module.catch_warnings(record=True) as caught:
        warnings_module.simplefilter("always")
        fresh.draw(
            collapse="max",
            vis_save_only=True,
            vis_fileformat="dot",
            order_siblings=False,
        )
    declines = [w for w in caught if "skipping smart collapse" in str(w.message)]
    assert len(declines) == 1
    decline = declines[0]
    # Selectable via the package taxonomy, not just blanket UserWarning.
    assert issubclass(decline.category, TorchLensWarning)
    # Names the governing constant so users can see the threshold they crossed.
    assert "COLLAPSE_OPTIMIZER_MAX_OPS" in str(decline.message)
    # Attributed to the caller's frame (this file), not a torchlens internal.
    assert decline.filename == __file__


def test_collapse_ceiling_documented_lockstep() -> None:
    """The compute ceiling must stay documented everywhere user-facing (R19).

    The 0957027b ceiling changed the documented behaviour of
    ``collapse="auto"|"max"``, ``Trace.collapse_plan()`` and
    ``Trace.collapse_schedule()`` for large traces; per the LOCKED docs rule the
    constant (and its current value) must appear in the user-facing collapse
    docs, the limitations catalog, the glossary, and the agent guide (AGENTS.md;
    CLAUDE.md only imports it).
    """

    from torchlens.visualization.collapse_optimizer import COLLAPSE_OPTIMIZER_MAX_OPS

    repo_root = Path(__file__).resolve().parents[1]
    doc_pages = [
        repo_root / "docs" / "reference" / "collapse.md",
        repo_root / "docs" / "reference" / "limitations.md",
        repo_root / "docs" / "reference" / "glossary.md",
        # AGENTS.md no longer carries this prose directly (2026-10-01 docs move:
        # "docs: move agent reference material out of startup instructions"); it
        # lives in this pointer target now (same precedent as the __all__ count
        # claim repoint in test_docs_lockstep_names.py).
        repo_root / "docs" / "agent-reference" / "current-2-x-surface.md",
    ]
    for page in doc_pages:
        text = page.read_text(encoding="utf-8")
        assert "COLLAPSE_OPTIMIZER_MAX_OPS" in text, f"{page.name} misses the ceiling constant"
        assert str(COLLAPSE_OPTIMIZER_MAX_OPS) in text, (
            f"{page.name} misses the ceiling value {COLLAPSE_OPTIMIZER_MAX_OPS}"
        )


def test_repeat_fold_discovery_scans_module_table_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Descendant-aware fold discovery must not rescan ``trace.modules`` per child.

    Hunt-6 R29-3: ``_iter_collapsible_runs(allow_selected_descendant=True)``
    called ``_selected_descendants`` once per child and
    ``_iter_collapsible_child_path_runs`` once per sibling, and each call
    iterated EVERY ``trace.modules`` facade -- Theta(S*M) module visits before
    fold legality was even considered. Each discovery pass now builds one
    shared sorted index, so the module table is iterated a constant number of
    times per pass regardless of sibling count.
    """

    from torch import nn

    from torchlens.data_classes._accessor_base import Accessor

    model = nn.Sequential(*[nn.Sequential(nn.Linear(4, 4), nn.ReLU()) for _ in range(6)])
    trace = tl.trace(model, torch.randn(2, 4))
    children = [m.address for m in trace.modules if m.address.isdigit()]
    assert len(children) == 6

    def collapse_fn(module: Any) -> bool:
        """Select Linear leaves so children are only descendant-selected."""

        return str(getattr(module, "class_name", "")) == "Linear"

    scans = {"n": 0}
    original_iter = Accessor.__iter__

    def counting_iter(self: Any) -> Any:
        """Count full accessor iterations."""

        if self is trace.modules:
            scans["n"] += 1
        return original_iter(self)

    monkeypatch.setattr(Accessor, "__iter__", counting_iter)

    scans["n"] = 0
    list(
        auto_collapse._iter_collapsible_runs(
            trace, children, collapse_fn, allow_selected_descendant=True
        )
    )
    assert scans["n"] <= 2, (
        f"descendant-aware run discovery iterated trace.modules {scans['n']} times "
        f"for {len(children)} children -- the per-child full scan is back"
    )

    scans["n"] = 0
    list(auto_collapse._iter_collapsible_child_path_runs(trace, children, collapse_fn))
    assert scans["n"] <= 2, (
        f"child-path discovery iterated trace.modules {scans['n']} times "
        f"for {len(children)} siblings -- the per-sibling full scan is back"
    )
