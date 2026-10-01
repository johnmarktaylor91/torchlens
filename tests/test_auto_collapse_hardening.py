"""Hardening regressions for the intelligent auto-collapse subsystem (round 21+).

Covers the round-20 adversarial-audit defect classes:

- multi-pass submodule reuse must not crash the smart-collapse surface;
- segment descriptor identity must be injective (plan/render parity);
- schedule ``collapsed_addresses`` must not report an empty set while hiding nodes;
- range/count labels must never lie about hidden mass;
- weighted optimizer results must not be cache-order dependent;
- ``CollapseSchedule.at(0.0)`` must agree with ``select_collapse_level(0.0)``;
- plans, schedules, and orders must be deterministic across hash seeds.

Round-23 seal finding C1: when auto's band-pressure branch returns an
op-segment-condensed plan, the max ladder re-condenses it, so pre-existing
segment nodes must keep their descriptors (pass-through) and the L3 auto
fallback must carry ``auto.segments`` instead of stripping them; every
max/float/schedule/draw/order surface must return honestly for that class.

Round-24 seal finding C1: a max op segment spanning several CALLS of one
reused module declared ``owner=core:1`` and rendered physically inside the
first call's cluster while the other call clusters stayed empty -- counts
and cardinality remained exact, so only module-call containment could catch
it. Op-segment runs must split at module-call reuse boundaries, descriptor
owners must be the exact call-qualified LCA of their members' rendered
stacks, and runs across distinct single-call sibling modules must stay
merged at the honest top-level LCA.

Round-25 seal finding (MED): when an op segment absorbed a collapsed
module's surfaced atomic-exit op (the last sibling block's own relu followed
by a >=2-op trailing top-level chain), the op was double-represented --
counted inside the box content label AND claimed as the leading endpoint of
the segment range label -- so structurally identical sibling blocks rendered
inconsistently ("1 op" vs "2 ops") and one op was claimed twice while node
counts stayed exact. Segment runs must never absorb a box-owned surfaced
exit op, and the box remainder accounting must treat op-segment members as
rendered separately, exactly like standalone raw nodes.
"""

import re
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.visualization._render_edges import (
    _collapsed_module_should_show_remainder,
    _plan_separately_rendered_op_labels,
    _run_fold_ellipsis_label,
)
from torchlens.visualization._segment_descriptors import (
    _child_segment_covered_ops,
    _child_segment_label,
    _make_child_segment_descriptor,
    _op_segment_owner_key,
)
from torchlens.visualization.auto_collapse import _make_run_fold, analyze_collapse
from torchlens.visualization.collapse_optimizer import (
    _RESULT_CACHE,
    OptimizerWeights,
    _condense_plan_with_child_segments,
    _legal_plan_op_segment_run,
    _optimizer_total_units,
    _own_ops_segment_is_legal,
    _plan_box_owned_surfaced_labels,
    _rendered_module_hidden_counts,
    select_collapse_level,
    select_collapse_plan,
)
from torchlens.visualization.collapse_plan import (
    ChildSegment,
    CollapsePlan,
    ModuleBox,
    OpSegment,
    RawOp,
    RenderContext,
    RepeatFold,
)


class ResidualBlock(nn.Module):
    """Small residual block used to build multi-pass reuse models."""

    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + x


class DoubleCall(nn.Module):
    """Calls every block twice per forward, creating multi-pass submodules."""

    def __init__(self, depth: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([ResidualBlock() for _ in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
            x = block(x)
        return x


class SegmentLeaf(nn.Module):
    """Three-op leaf block used by segment-collision models."""

    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.lin = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(torch.relu(self.lin(x)))


class CollidingSegments(nn.Module):
    """Nested ``a.b0..a.b9`` and top-level ``a_b0..a_b9`` chains.

    Both child-segment endpoint pairs mangle to the same legacy Graphviz name,
    which historically made the descriptor dict non-injective.
    """

    class Inner(nn.Module):
        def __init__(self, width: int = 8) -> None:
            super().__init__()
            for index in range(10):
                setattr(self, f"b{index}", SegmentLeaf(width))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for index in range(10):
                x = getattr(self, f"b{index}")(x)
            return x

    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.a = self.Inner(width)
        for index in range(10):
            setattr(self, f"a_b{index}", SegmentLeaf(width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for index in range(10):
            x = getattr(self, f"a_b{index}")(x)
        return self.a(x)


class LoopedFunctionalAroundBlock(nn.Module):
    """Loop of shared functional ops around a boxable block.

    Every iteration re-executes the same functional layer labels, so the max
    plan contains several op segments whose BASE label runs are identical and
    only the pass identity distinguishes them.
    """

    def __init__(self, width: int = 8) -> None:
        super().__init__()
        layers = []
        for _ in range(5):
            layers += [nn.Linear(width, width), nn.ReLU()]
        self.blk = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(3):
            x = torch.tanh(torch.sigmoid(torch.relu(x)))
            x = self.blk(x)
        return x


class UnevenRecurrent(nn.Module):
    """Two sibling blocks called three and five times per forward."""

    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.first = SegmentLeaf(width)
        self.second = SegmentLeaf(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(3):
            x = self.first(x)
        for _ in range(5):
            x = self.second(x)
        return x


class MultiPassBlock(nn.Module):
    """Three-op leaf block (linear, relu, mul) reused by multi-pass fixtures."""

    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.lin = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.mul(torch.relu(self.lin(x)), 1.0)


class SharedMultiCall(nn.Module):
    """One shared n-child module called ``calls`` times plus a linear tail.

    Reconstructs the round-22 collapse-seal H1 fixture: functional
    relu/sigmoid/tanh before every shared call, a seven-child shared body,
    and a ten-leaf tail. Under max/1.0 the selected plan contains multi-pass
    module boxes and a fold whose representative is called ``calls`` times,
    which the pre-fix optimizer counted once per ADDRESS while the renderer
    emits one box per CALL.
    """

    class Shared(nn.Module):
        def __init__(self, width: int, children: int) -> None:
            super().__init__()
            self.body = nn.Sequential(*[MultiPassBlock(width) for _ in range(children)])

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.body(x)

    def __init__(
        self,
        width: int = 8,
        children: int = 7,
        calls: int = 4,
        tail: int = 10,
    ) -> None:
        super().__init__()
        self.calls = calls
        self.shared = self.Shared(width, children)
        self.tail = nn.Sequential(*[nn.Linear(width, width) for _ in range(tail)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(self.calls):
            x = torch.relu(x)
            x = torch.sigmoid(x)
            x = torch.tanh(x)
            x = self.shared(x)
        return self.tail(x)


def _svg_node_group_count(path: str) -> int:
    with open(path, encoding="utf-8") as handle:
        return len(re.findall(r'class="node', handle.read()))


def _plan_segment_node_count(plan) -> int:
    return sum(isinstance(node, (ChildSegment, OpSegment)) for node in plan.nodes)


@pytest.mark.parametrize("depth", [3, 4, 6])
def test_collapse_surface_survives_multipass_submodule(depth, tmp_path):
    """Every public smart-collapse surface must survive multi-pass reuse."""

    model = DoubleCall(depth).eval()
    trace = tl.trace(model, torch.randn(2, 8))
    for mode in ("auto", "max", 0.5):
        plan = trace.collapse_plan(mode=mode)
        assert plan.total > 0
    schedule = trace.collapse_schedule()
    assert schedule.steps[0].visible_count >= schedule.steps[-1].visible_count
    order = trace.module_collapse_order
    assert isinstance(order, list) and order
    for mode in ("auto", "max"):
        trace.draw(
            vis_save_only=True,
            vis_fileformat="svg",
            collapse=mode,
            vis_outpath=str(tmp_path / f"dc{depth}_{mode}"),
        )


def test_segment_descriptor_cardinality_matches_plan():
    """Colliding segment endpoints must yield one descriptor per plan segment."""

    model = CollidingSegments().eval()
    trace = tl.trace(model, torch.randn(2, 8))
    context = RenderContext()
    result = select_collapse_plan(trace, context, mode="max")
    plan_segments = _plan_segment_node_count(result.plan)
    descriptors = result.segments or {}
    assert plan_segments == 2
    assert len(descriptors) == plan_segments
    names = sorted(descriptors)
    assert len(set(names)) == plan_segments
    member_tuples = {descriptor.members for descriptor in descriptors.values()}
    assert len(member_tuples) == plan_segments


def test_segment_collision_render_parity(tmp_path):
    """The rendered SVG node count must equal the max plan's node count."""

    model = CollidingSegments().eval()
    trace = tl.trace(model, torch.randn(2, 8))
    plan = trace.collapse_plan(mode="max")
    out = tmp_path / "collision"
    trace.draw(
        vis_save_only=True,
        vis_fileformat="svg",
        collapse="max",
        vis_outpath=str(out),
    )
    assert _svg_node_group_count(str(out) + ".svg") == plan.total


@pytest.mark.parametrize(
    "builder",
    [lambda: DoubleCall(2), lambda: LoopedFunctionalAroundBlock()],
    ids=["double_call", "looped_functional"],
)
def test_multipass_op_segments_render_distinct(builder, tmp_path):
    """Per-pass op segments must not silently merge into one rendered node.

    ``looped_functional`` produces several op segments whose BASE label runs
    are byte-identical (only the pass differs), so any pass-free segment
    identity collides and silently drops rendered structure.
    """

    model = builder().eval()
    trace = tl.trace(model, torch.randn(2, 8))
    plan = trace.collapse_plan(mode="max")
    result = select_collapse_plan(trace, RenderContext(), mode="max")
    descriptors = result.segments or {}
    assert len(descriptors) == _plan_segment_node_count(result.plan)
    assert len(set(descriptors)) == len(descriptors)
    out = tmp_path / "multipass_max"
    trace.draw(
        vis_save_only=True,
        vis_fileformat="svg",
        collapse="max",
        vis_outpath=str(out),
    )
    assert _svg_node_group_count(str(out) + ".svg") == plan.total


def _plan_hidden_and_visible_ops(trace, result):
    """Split the trace's concrete ops into visible and hidden-with-witness sets."""

    concrete = {str(op.label) for op in trace.ops}
    hidden: set[str] = set()
    visible: set[str] = set()
    analysis = analyze_collapse(trace)
    occurrence: dict[str, int] = {}
    for node in result.plan.nodes:
        if isinstance(node, RawOp) and isinstance(node.op, str):
            base = node.op
            if base in concrete:
                visible.add(base)
                continue
            occurrence[base] = occurrence.get(base, 0) + 1
            qualified = f"{base}:{occurrence[base]}"
            visible.add(qualified if qualified in concrete else base)
        elif isinstance(node, OpSegment):
            for base in node.ops:
                occurrence[base] = occurrence.get(base, 0) + 1
    for descriptor in (result.segments or {}).values():
        hidden.update(str(op) for op in descriptor.ops)
    for node in result.plan.nodes:
        if isinstance(node, ChildSegment):
            for address in node.members:
                signal = analysis.signals.get(address)
                if signal is not None:
                    hidden.update(str(label) for label in signal.subtree_ops)
        elif isinstance(node, (ModuleBox, RepeatFold)):
            addresses = (
                node.members if isinstance(node, RepeatFold) else (node.call.rsplit(":", 1)[0],)
            )
            for address in addresses:
                signal = analysis.signals.get(address)
                if signal is not None:
                    hidden.update(str(label) for label in signal.subtree_ops)
    return visible, hidden


@pytest.mark.parametrize("mode", ["auto", "max"])
def test_no_silent_node_drop(mode):
    """Every concrete op is either visible or witnessed by a hiding unit."""

    battery = [
        DoubleCall(3).eval(),
        CollidingSegments().eval(),
        nn.Sequential(*[nn.Linear(8, 8) for _ in range(20)]).eval(),
    ]
    for model in battery:
        trace = tl.trace(model, torch.randn(2, 8))
        result = select_collapse_plan(trace, RenderContext(), mode=mode)
        if result.declined:
            continue
        visible, hidden = _plan_hidden_and_visible_ops(trace, result)
        concrete = {str(op.label) for op in trace.ops}
        unaccounted = {
            label
            for label in concrete
            if label not in hidden
            and label not in visible
            and label.rsplit(":", 1)[0] not in visible
        }
        assert not unaccounted, (
            f"{type(model).__name__} mode={mode}: ops silently dropped: {sorted(unaccounted)[:8]}"
        )


def _occurrence_witness_partition(trace, result):
    """Partition concrete PASS-QUALIFIED op occurrences into visible/hidden.

    Unlike :func:`_plan_hidden_and_visible_ops`, which witnesses whole
    addresses, this attributes every concrete pass-qualified occurrence:

    - visible: raw plan nodes, with the ``k``-th plan occurrence of a
      pass-free base label attributed to pass ``k`` (render order);
    - hidden: occurrences whose renderer-effective module stack (innermost
      dropped for atomic exits) contains an exact plan box CALL, any call of
      a fold's hidden members, or a segment member address, plus the exact
      pass-qualified ops of op-segment descriptors.

    An occurrence with no witness at all is a silently dropped node.
    """

    concrete = {str(op.label) for op in trace.ops}
    visible: set[str] = set()
    occurrence: dict[str, int] = {}
    box_calls: set[str] = set()
    fold_hidden_addresses: set[str] = set()
    child_members: set[str] = set()
    op_segment_ops: set[str] = set()
    for node in result.plan.nodes:
        if isinstance(node, RawOp):
            text = str(node.op)
            if text in concrete:
                visible.add(text)
                continue
            occurrence[text] = occurrence.get(text, 0) + 1
            qualified = f"{text}:{occurrence[text]}"
            if qualified in concrete:
                visible.add(qualified)
        elif isinstance(node, ModuleBox):
            box_calls.add(node.call)
        elif isinstance(node, RepeatFold):
            box_calls.add(node.rep.call)
            fold_hidden_addresses.update(node.members[1:])
        elif isinstance(node, ChildSegment):
            child_members.update(node.members)
    for descriptor in (result.segments or {}).values():
        if descriptor.kind == "op":
            op_segment_ops.update(str(op) for op in descriptor.ops)
        else:
            child_members.update(descriptor.members)
    hidden: set[str] = set()
    for op in trace.ops:
        label = str(op.label)
        modules = [str(call) for call in (getattr(op, "modules", ()) or ())]
        # Segment and fold absorption match the op's ORIGINAL module stack
        # (mirroring ``_segment_for_node`` / ``_run_fold_ancestor_for_node``);
        # box hiding matches the renderer-EFFECTIVE stack, where the innermost
        # module of an atomic exit op is dropped and the op stays visible.
        original_bases = {call.rsplit(":", 1)[0] for call in modules}
        effective = modules
        if getattr(op, "is_atomic_module", False) and effective:
            effective = effective[:-1]
        if (
            any(call in box_calls for call in effective)
            or original_bases & fold_hidden_addresses
            or original_bases & child_members
            or label in op_segment_ops
        ):
            hidden.add(label)
    return concrete, visible, hidden


def _multi_call_plan_boxes(trace, plan):
    """Return plan boxes and fold representatives of multi-call addresses."""

    boxes = []
    folds = []
    for node in plan.nodes:
        if isinstance(node, ModuleBox):
            address = node.call.rsplit(":", 1)[0]
        elif isinstance(node, RepeatFold):
            address = node.rep.call.rsplit(":", 1)[0]
        else:
            continue
        if address not in trace.modules:
            continue
        if int(getattr(trace.modules[address], "num_calls", 1) or 1) > 1:
            (folds if isinstance(node, RepeatFold) else boxes).append(node)
    return boxes, folds


@pytest.mark.parametrize(
    "builder, require_fold",
    [
        (lambda: SharedMultiCall(children=7, calls=4, tail=10), False),
        (lambda: SharedMultiCall(children=7, calls=2, tail=4), True),
    ],
    ids=["shared_four_calls", "folded_two_calls"],
)
def test_multipass_pass_occurrence_conservation_and_render_parity(builder, require_fold, tmp_path):
    """Multi-pass plans must count rendered CALLS and witness every occurrence.

    Round-22 seal finding H1: ``_module_box_plan_nodes`` and the folded
    instantiation emitted ``address:1`` once while the renderer collapses
    every call of a multi-pass address, so ``plan.total``, the max count
    gate, and ``collapse_schedule().steps[-1].visible_count`` under-counted
    the rendered graph and pass-2+ occurrences had no plan witness. For every
    public mode this pins:

    - ``plan.total`` equals the rendered SVG node count, and
    - the full concrete pass-qualified occurrence set equals the union of
      visible plan nodes and hidden-with-witness occurrences.
    """

    torch.manual_seed(0)
    model = builder().eval()
    trace = tl.trace(model, torch.randn(2, 8))
    context = RenderContext()

    max_result = select_collapse_plan(trace, context, mode="max")
    assert not max_result.declined
    multi_boxes, multi_folds = _multi_call_plan_boxes(trace, max_result.plan)
    assert multi_boxes, "fixture must exercise multi-call module boxes at max"
    if require_fold:
        assert multi_folds, "fixture must exercise a multi-call fold representative at max"
        rep_call = multi_folds[0].rep.call
        rep_address = rep_call.rsplit(":", 1)[0]
        later_pass_boxes = [
            node
            for node in max_result.plan.nodes
            if isinstance(node, ModuleBox)
            and node.call.rsplit(":", 1)[0] == rep_address
            and node.call != rep_call
        ]
        assert later_pass_boxes, "multi-call fold representative must render later passes"

    schedule = trace.collapse_schedule()
    assert schedule.steps[-1].visible_count == max_result.plan.total

    for mode in ("auto", "max", 0.5, 1.0):
        if isinstance(mode, float):
            result = select_collapse_level(trace, context, mode)
        else:
            result = select_collapse_plan(trace, context, mode=mode)
        if result.declined:
            continue
        out = tmp_path / f"multipass_{str(mode).replace('.', '_')}"
        trace.draw(
            vis_save_only=True,
            vis_fileformat="svg",
            collapse=mode,
            vis_outpath=str(out),
        )
        rendered = _svg_node_group_count(str(out) + ".svg")
        assert result.plan.total == rendered, (
            f"mode={mode}: plan.total={result.plan.total} but the renderer emitted {rendered} nodes"
        )
        assert result.visible_count == result.plan.total
        concrete, visible, hidden = _occurrence_witness_partition(trace, result)
        orphaned = concrete - visible - hidden
        assert not orphaned, (
            f"mode={mode}: {len(orphaned)} of {len(concrete)} concrete "
            f"pass-qualified occurrences have no visible or hidden plan "
            f"witness: {sorted(orphaned)[:8]}"
        )
        assert concrete == visible | hidden


def test_schedule_collapsed_addresses_not_empty_when_hiding():
    """Schedule steps that hide nodes must report a nonempty collapsed set."""

    model = nn.Sequential(*[nn.Linear(8, 8) for _ in range(20)]).eval()
    trace = tl.trace(model, torch.randn(2, 8))
    schedule = trace.collapse_schedule()
    full = schedule.steps[0].visible_count
    for step in schedule.steps:
        if step.visible_count < full:
            assert step.collapsed_addresses, (
                f"step t={step.t} hides {full - step.visible_count} nodes but "
                "reports an empty collapsed set"
            )


def test_schedule_addresses_monotone_superset():
    """Collapsed sets must stay nested as t increases (with the honest final set)."""

    for model in (
        nn.Sequential(*[nn.Linear(8, 8) for _ in range(20)]).eval(),
        nn.Sequential(*[ResidualBlock() for _ in range(6)]).eval(),
    ):
        trace = tl.trace(model, torch.randn(2, 8))
        schedule = trace.collapse_schedule()
        previous: frozenset = frozenset()
        for step in schedule.steps:
            assert step.collapsed_addresses >= previous
            previous = step.collapsed_addresses


def test_child_segment_range_label_honest():
    """Name-noncontiguous members must not be labeled as a numeric interval."""

    contiguous = _child_segment_label(("blocks.0", "blocks.1", "blocks.2"), 9, 0, 0)
    assert "blocks.0-2" in contiguous
    gapped = _child_segment_label(("blocks.0", "blocks.2", "blocks.4"), 9, 0, 0)
    assert "blocks.0-4" not in gapped
    assert "3 blocks" in gapped
    backwards = _child_segment_label(("stage.5", "stage.3", "stage.1"), 9, 0, 0)
    assert "stage.5-1" not in backwards
    assert "3 blocks" in backwards
    long_gapped = _child_segment_label(
        ("blocks.0", "blocks.2", "blocks.4", "blocks.6", "blocks.8"), 15, 0, 0
    )
    assert "blocks.0-8" not in long_gapped
    assert "5 blocks" in long_gapped


def test_child_segment_descriptor_counts_concrete_ops():
    """Segment descriptors must count every hidden concrete op, not base labels."""

    model = UnevenRecurrent().eval()
    trace = tl.trace(model, torch.randn(2, 8))
    analysis = analyze_collapse(trace)
    expected = len(analysis.signals["first"].subtree_ops) + len(
        analysis.signals["second"].subtree_ops
    )
    covered = _child_segment_covered_ops(analysis, ("first", "second"))
    assert len(covered) == expected
    descriptor = _make_child_segment_descriptor(
        trace, RenderContext(), ("first", "second"), covered
    )
    assert descriptor.num_ops + descriptor.num_buffers == expected
    assert f"{expected} ops" in descriptor.label


def test_fold_ellipsis_discloses_hidden_calls():
    """Repeat-fold ellipses must disclose hidden call mass beyond addresses."""

    model = UnevenRecurrent().eval()
    trace = tl.trace(model, torch.randn(2, 8))
    fold = _make_run_fold(trace, ("first", "second"))
    label = _run_fold_ellipsis_label(fold)
    assert label.startswith(f"... +{fold.multiplicity - 1} more ")
    hidden_calls = sum(
        int(getattr(trace.modules[address], "num_calls", 1) or 1) for address in fold.addresses[1:]
    )
    assert hidden_calls == 5
    assert "5 calls" in label

    single_model = nn.Sequential(*[ResidualBlock() for _ in range(4)]).eval()
    single_trace = tl.trace(single_model, torch.randn(2, 8))
    single_fold = _make_run_fold(single_trace, ("0", "1", "2", "3"))
    single_label = _run_fold_ellipsis_label(single_fold)
    assert single_label == f"... +3 more {single_fold.class_name}"


def test_weighted_result_cache_not_order_dependent():
    """Weighted plan selection must not be poisoned by earlier weighted calls."""

    model = nn.Sequential(*[ResidualBlock() for _ in range(6)]).eval()
    trace = tl.trace(model, torch.randn(2, 8))
    context = RenderContext()
    prefer_small = OptimizerWeights(w_k=100.0)
    prefer_large = OptimizerWeights(w_k=-100.0)

    def fingerprint(result):
        return (result.visible_count, tuple(sorted(result.selected)))

    _RESULT_CACHE.pop(trace, None)
    fresh_small = fingerprint(select_collapse_plan(trace, context, prefer_small))
    _RESULT_CACHE.pop(trace, None)
    fresh_large = fingerprint(select_collapse_plan(trace, context, prefer_large))
    assert fresh_small != fresh_large

    _RESULT_CACHE.pop(trace, None)
    first = fingerprint(select_collapse_plan(trace, context, prefer_small))
    second = fingerprint(select_collapse_plan(trace, context, prefer_large))
    assert (first, second) == (fresh_small, fresh_large)

    _RESULT_CACHE.pop(trace, None)
    first = fingerprint(select_collapse_plan(trace, context, prefer_large))
    second = fingerprint(select_collapse_plan(trace, context, prefer_small))
    assert (first, second) == (fresh_large, fresh_small)


def test_at_zero_agrees_with_select_level_zero():
    """``at(0.0)`` and ``select_collapse_level(0.0)`` must both report no collapse."""

    layers = []
    for _ in range(12):
        layers += [nn.Linear(8, 8), nn.ReLU()]
    model = nn.Sequential(*layers).eval()
    trace = tl.trace(model, torch.randn(2, 8))
    schedule = trace.collapse_schedule()
    full = schedule.steps[0].visible_count
    step = schedule.at(0.0)
    assert step.visible_count == full
    assert step.collapsed_addresses == frozenset()
    level = select_collapse_level(trace, RenderContext(), 0.0)
    assert level.visible_count == full
    assert level.selected == frozenset()
    assert level.selected == step.collapsed_addresses


_DETERMINISM_SNIPPET = textwrap.dedent(
    """
    import hashlib, warnings
    warnings.filterwarnings("ignore")
    import torch, torch.nn as nn
    import torchlens as tl

    torch.manual_seed(0)

    class Blk(nn.Module):
        def __init__(s, w=8):
            super().__init__()
            s.net = nn.Sequential(nn.Linear(w, w), nn.ReLU(), nn.Linear(w, w), nn.ReLU())

        def forward(s, x):
            return s.net(x) + x

    class DC(nn.Module):
        def __init__(s, depth):
            super().__init__()
            s.blocks = nn.ModuleList([Blk() for _ in range(depth)])

        def forward(s, x):
            for b in s.blocks:
                x = b(x)
                x = b(x)
            return x

    m = DC(3).eval()
    log = tl.trace(m, torch.randn(2, 8))
    pieces = [
        repr(log.collapse_plan(mode="max")),
        repr(log.collapse_plan(mode="auto")),
        repr(log.module_collapse_order),
        repr([(s.t, s.visible_count, tuple(sorted(s.collapsed_addresses)))
              for s in log.collapse_schedule().steps]),
    ]
    print(hashlib.sha256("\\n".join(pieces).encode()).hexdigest())
    """
)


@pytest.mark.heavy
def test_collapse_determinism_across_hashseed():
    """Plans, orders, and schedules must be byte-identical across hash seeds."""

    digests = set()
    for seed in ("0", "1", "12345"):
        proc = subprocess.run(
            [sys.executable, "-c", _DETERMINISM_SNIPPET],
            capture_output=True,
            text=True,
            env={
                "PYTHONHASHSEED": seed,
                "PATH": "/usr/bin:/bin",
                "CUDA_VISIBLE_DEVICES": "",
                "PYTHONPATH": ":".join(sys.path),
            },
            check=True,
        )
        digests.add(proc.stdout.strip())
    assert len(digests) == 1, f"hash-seed dependent collapse output: {digests}"


class PureFunctionalLoops(nn.Module):
    """Zero-submodule functional loop: tanh; sigmoid; relu(x + 0.01) per pass.

    At ten or more loops the full graph exceeds the readable band with no
    DP-selectable module, so auto's band-pressure branch returns an
    op-segment-condensed plan (round-23 seal C1 model class).
    """

    def __init__(self, loops: int) -> None:
        super().__init__()
        self.loops = loops

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(self.loops):
            x = torch.tanh(x)
            x = torch.sigmoid(x)
            x = torch.relu(x + 0.01)
        return x


class AtomicWrapper(nn.Module):
    """Bare-linear wrapper whose single op renders as an atomic raw op."""

    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.lin = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)


class SharedChildFunctionalLoops(nn.Module):
    """Real submodule called once per functional loop iteration.

    The child's ops join the surrounding functional run, so auto's op
    segments span multi-pass module ops rather than pure functional labels.
    """

    def __init__(self, loops: int, width: int = 8) -> None:
        super().__init__()
        self.child = AtomicWrapper(width)
        self.loops = loops

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(self.loops):
            x = torch.tanh(x)
            x = self.child(x)
            x = torch.sigmoid(x)
            x = torch.relu(x + 0.01)
        return x


def _op_segment_condensed_trace(kind: str, loops: int):
    torch.manual_seed(0)
    model = (
        PureFunctionalLoops(loops) if kind == "functional" else SharedChildFunctionalLoops(loops)
    ).eval()
    return tl.trace(model, torch.randn(1, 8))


_SEGMENTED_AUTO_CASES = [
    ("functional", 9),
    ("functional", 10),
    ("functional", 11),
    ("functional", 12),
    ("functional", 20),
    ("shared_child", 8),
    ("shared_child", 12),
    ("shared_child", 16),
]
_SEGMENTED_AUTO_IDS = [f"{kind}_{loops}" for kind, loops in _SEGMENTED_AUTO_CASES]


@pytest.mark.parametrize("kind, loops", _SEGMENTED_AUTO_CASES, ids=_SEGMENTED_AUTO_IDS)
def test_op_segment_condensed_auto_max_ladder_surfaces(kind, loops, tmp_path):
    """Round-23 C1: every max/float/schedule/draw/order surface must be honest.

    The size range spans the segmentation boundary (nine functional loops
    keep auto un-segmented; ten push it into the band-pressure branch) and
    both the zero-submodule and the real-submodule-in-functional-runs
    variants. The pre-fix max ladder re-condensed auto's already-segmented
    plan, dropped the pre-existing ``OpSegment`` descriptors, and crashed the
    r21 parity tripwire (``AssertionError``) on all six public surfaces.
    """

    trace = _op_segment_condensed_trace(kind, loops)
    context = RenderContext()

    auto_result = select_collapse_plan(trace, context, mode="auto")
    max_result = select_collapse_plan(trace, context, mode="max")
    results = {"auto": auto_result, "max": max_result}
    for t in (0.25, 0.5, 0.75, 1.0):
        results[f"t={t}"] = select_collapse_level(trace, context, t)
    for label, result in results.items():
        assert not result.declined, f"{label}: unexpectedly declined"
        assert result.visible_count == result.plan.total, (
            f"{label}: visible_count {result.visible_count} != plan.total {result.plan.total}"
        )
        descriptors = result.segments or {}
        plan_segments = _plan_segment_node_count(result.plan)
        assert len(descriptors) == plan_segments, (
            f"{label}: descriptor cardinality {len(descriptors)} != "
            f"plan segment nodes {plan_segments}"
        )
        assert len(set(descriptors)) == len(descriptors)
    assert results["t=1.0"].plan == max_result.plan

    schedule = trace.collapse_schedule()
    assert schedule.steps[-1].visible_count == max_result.plan.total
    for step in schedule.steps:
        assert step.visible_count == step.plan.total

    order = trace.collapse_order(mode="max")
    assert isinstance(order, list)

    concrete, visible, hidden = _occurrence_witness_partition(trace, max_result)
    assert concrete == visible | hidden, (
        f"max: {len(concrete - visible - hidden)} occurrences have no plan witness"
    )

    for mode in ("max", 0.5):
        out = tmp_path / f"segauto_{kind}_{loops}_{str(mode).replace('.', '_')}"
        trace.draw(
            vis_save_only=True,
            vis_fileformat="svg",
            collapse=mode,
            vis_outpath=str(out),
        )
        plan = trace.collapse_plan(mode)
        rendered = _svg_node_group_count(str(out) + ".svg")
        assert plan.total == rendered, (
            f"draw({mode}): plan.total {plan.total} != rendered SVG nodes {rendered}"
        )


def test_max_ladder_carries_auto_op_segment_descriptors():
    """A pass-through max plan must keep auto's descriptors byte-identical.

    For the op-segment-condensed auto class nothing further condenses, so the
    max ladder falls back to auto; the fallback must carry auto's plan AND
    auto's segment descriptors (the pre-r21 L3 fallback stripped
    ``segments={}``, a silent plan/render parity lie). The op segments must
    span multi-pass op labels, pinning the multi-pass absorption variant.
    """

    trace = _op_segment_condensed_trace("shared_child", 12)
    context = RenderContext()
    auto_result = select_collapse_plan(trace, context, mode="auto")
    max_result = select_collapse_plan(trace, context, mode="max")
    assert _plan_segment_node_count(auto_result.plan) > 0, (
        "fixture must produce an op-segment-condensed auto plan"
    )
    assert max_result.plan == auto_result.plan
    assert dict(max_result.segments or {}) == dict(auto_result.segments or {})
    assert max_result.segments, "max fallback stripped auto's segment descriptors"
    multi_pass = any(
        ":" in op and op.rsplit(":", 1)[1] not in ("", "1")
        for descriptor in max_result.segments.values()
        for op in descriptor.ops
    )
    assert multi_pass, "op segments must span multi-pass op labels"


def test_condense_pass_through_preserves_segment_descriptors():
    """Re-condensing an already-segmented plan must rebuild its descriptors.

    Feeds a max plan that carries ``ChildSegment`` nodes back through
    ``_condense_plan_with_child_segments``: pre-existing segment nodes must
    pass through verbatim with descriptors equal to the originals (the
    pre-fix loop appended them descriptor-less, firing the parity tripwire).
    """

    model = CollidingSegments().eval()
    trace = tl.trace(model, torch.randn(2, 8))
    context = RenderContext()
    max_result = select_collapse_plan(trace, context, mode="max")
    child_segments = sum(isinstance(node, ChildSegment) for node in max_result.plan.nodes)
    assert child_segments > 0, "fixture must produce child segments at max"
    analysis = analyze_collapse(trace)
    hidden_counts = _rendered_module_hidden_counts(trace, context)
    total_units = _optimizer_total_units(trace, context)
    replan, redescriptors = _condense_plan_with_child_segments(
        trace,
        context,
        analysis,
        max_result.plan,
        hidden_counts,
        total_units,
        dominance_limit=0.75,
        k_hi=20,
    )
    assert replan.nodes == max_result.plan.nodes
    assert len(redescriptors) == _plan_segment_node_count(replan)
    assert dict(redescriptors) == dict(max_result.segments or {})


class ReusedFunctionalCore(nn.Module):
    """Parameter-free seven-op block designed for multi-call reuse."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(x)
        x = torch.sigmoid(x)
        x = torch.relu(x + 0.01)
        x = torch.sin(x)
        x = torch.cos(x)
        return torch.abs(x)


class FunctionalCoreReuse(nn.Module):
    """Calls one parameter-free functional module ``calls`` times."""

    def __init__(self, calls: int) -> None:
        super().__init__()
        self.core = ReusedFunctionalCore()
        self.calls = calls

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(self.calls):
            x = self.core(x)
        return x


class ReusedResidualCell(nn.Module):
    """Real parameterized residual cell (Linear -> ReLU -> Linear -> add)."""

    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x))) + x


class ResidualCellReuse(nn.Module):
    """Calls one real residual cell ``calls`` times."""

    def __init__(self, calls: int) -> None:
        super().__init__()
        self.cell = ReusedResidualCell()
        self.calls = calls

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(self.calls):
            x = self.cell(x)
        return x


class NestedCoreReuse(nn.Module):
    """Nested reuse: outer called twice, the inner core twice per outer call."""

    class Outer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.core = ReusedFunctionalCore()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.core(x)
            x = self.core(x)
            return torch.log1p(torch.abs(x))

    def __init__(self) -> None:
        super().__init__()
        self.outer = self.Outer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.outer(x)
        x = self.outer(x)
        return x


class SiblingBlockChain(nn.Module):
    """Five DISTINCT single-call sibling blocks (the honest-LCA merge shape)."""

    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.Linear(width, width), nn.ReLU()) for _ in range(5)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


def _effective_stack(op) -> tuple[str, ...]:
    """Independently recompute the renderer's effective cluster stack."""

    modules = [str(module) for module in (getattr(op, "modules", ()) or ())]
    if getattr(op, "is_atomic_module", False) and modules:
        modules = modules[:-1]
    return tuple(modules)


def _exact_call_lca(stacks) -> str | None:
    """Return the deepest call-qualified entry shared by every stack."""

    if not stacks or any(not stack for stack in stacks):
        return None
    common = None
    for values in zip(*stacks):
        if len(set(values)) != 1:
            break
        common = values[0]
    return common


def _dot_node_clusters(source: str):
    """Map DOT node names to innermost clusters and clusters to member sets.

    Same-named reopened subgraph blocks (the BFS hierarchy walk reopens a
    parent cluster once per child branch) are merged by name, matching
    Graphviz semantics.
    """

    stack: list[str] = []
    node_cluster: dict[str, str | None] = {}
    members: dict[str, set[str]] = {}
    for raw_line in source.splitlines():
        line = raw_line.strip()
        opened = re.match(r'subgraph "?(cluster_[\w.]+)"?', line)
        if opened:
            stack.append(opened.group(1))
            members.setdefault(opened.group(1), set())
            continue
        if line.startswith("}"):
            if stack:
                stack.pop()
            continue
        node = re.match(r'"?([\w.:]+)"? \[', line)
        if node and "->" not in line and node.group(1) not in ("graph", "node", "edge"):
            node_cluster[node.group(1)] = stack[-1] if stack else None
            if stack:
                members[stack[-1]].add(node.group(1))
    return node_cluster, members


def _owner_cluster_name(owner: str | None) -> str | None:
    """Return the Graphviz cluster name for a call-qualified owner key."""

    if owner is None:
        return None
    return f"cluster_{owner.replace(':', '_pass')}"


_CONTAINMENT_CASES = [
    ("functional", 2),
    ("functional", 4),
    ("functional", 6),
    ("functional", 12),
    ("residual", 2),
    ("residual", 4),
    ("residual", 6),
    ("residual", 12),
    ("nested", 0),
    ("shared_multicall", 0),
    ("colliding", 0),
]
_CONTAINMENT_IDS = [f"{kind}_{calls}" if calls else kind for kind, calls in _CONTAINMENT_CASES]


def _containment_trace(kind: str, calls: int):
    torch.manual_seed(0)
    if kind == "functional":
        model = FunctionalCoreReuse(calls)
    elif kind == "residual":
        model = ResidualCellReuse(calls)
    elif kind == "nested":
        model = NestedCoreReuse()
    elif kind == "shared_multicall":
        model = SharedMultiCall(children=7, calls=4, tail=10)
    else:
        model = CollidingSegments()
    return tl.trace(model.eval(), torch.randn(1, 8))


@pytest.mark.parametrize("kind, calls", _CONTAINMENT_CASES, ids=_CONTAINMENT_IDS)
def test_op_segment_module_call_containment(kind, calls, tmp_path):
    """Round-24 C1: op segments must respect module-CALL containment.

    Pre-fix, a max op segment absorbed ops from several calls of one reused
    module and rendered inside only the FIRST call's cluster (owner=core:1,
    19/26 represented ops outside the owner, sibling call clusters empty)
    while counts and cardinality stayed exact. For every public surface this
    pins, per op-segment descriptor:

    - the owner equals the exact call-qualified LCA of its members' rendered
      module stacks (never a call the segment does not fully represent);
    - every member op's stack contains the owner when one is declared;
    - no two members sit in DIFFERENT CALLS of one module at a shared stack
      level (reuse absorption is the round-24 lie);
    - the max DOT places each segment node physically inside its owner's
      cluster (or at top level for owner ``None``);
    - counts/cardinality stay exact: plan == visible == rendered SVG and
      descriptor cardinality == plan segment nodes, with zero orphaned
      pass-qualified occurrences.
    """

    trace = _containment_trace(kind, calls)
    context = RenderContext()
    results = {
        "auto": select_collapse_plan(trace, context, mode="auto"),
        "max": select_collapse_plan(trace, context, mode="max"),
    }
    for level in (0.5, 1.0):
        results[f"t={level}"] = select_collapse_level(trace, context, level)

    for label, result in results.items():
        if result.declined:
            continue
        descriptors = result.segments or {}
        assert len(descriptors) == _plan_segment_node_count(result.plan), (
            f"{label}: descriptor cardinality != plan segment nodes"
        )
        assert result.visible_count == result.plan.total
        for name, segment in descriptors.items():
            if segment.kind != "op":
                continue
            stacks = [_effective_stack(trace.ops[str(op)]) for op in segment.ops]
            lca = _exact_call_lca(stacks)
            assert segment.owner == lca, (
                f"{label}/{name}: owner {segment.owner!r} is not the exact "
                f"call-qualified LCA {lca!r} of its members"
            )
            if segment.owner is not None:
                outside = [
                    op for op, stack in zip(segment.ops, stacks) if segment.owner not in stack
                ]
                assert not outside, (
                    f"{label}/{name}: {len(outside)} represented ops render "
                    f"outside declared owner {segment.owner!r}: {outside[:4]}"
                )
            for index, first in enumerate(stacks):
                for second in stacks[index + 1 :]:
                    for left, right in zip(first, second):
                        if left == right:
                            continue
                        assert left.rsplit(":", 1)[0] != right.rsplit(":", 1)[0], (
                            f"{label}/{name}: segment absorbs two calls of one "
                            f"module ({left} and {right})"
                        )
                        break

    max_result = results["max"]
    out = tmp_path / f"containment_{kind}_{calls}"
    source = trace.draw(
        vis_save_only=True,
        vis_fileformat="svg",
        collapse="max",
        vis_outpath=str(out),
    )
    assert _svg_node_group_count(str(out) + ".svg") == max_result.plan.total
    node_cluster, _ = _dot_node_clusters(source)
    for name, segment in (max_result.segments or {}).items():
        if segment.kind != "op":
            continue
        assert node_cluster.get(segment.name) == _owner_cluster_name(segment.owner), (
            f"max/{name}: DOT places segment in "
            f"{node_cluster.get(segment.name)!r}, owner requires "
            f"{_owner_cluster_name(segment.owner)!r}"
        )
    concrete, visible, hidden = _occurrence_witness_partition(trace, max_result)
    assert concrete == visible | hidden, "max: orphaned pass-qualified occurrences"


@pytest.mark.parametrize(
    "builder, address, calls",
    [
        (lambda: FunctionalCoreReuse(6), "core", 6),
        (lambda: ResidualCellReuse(12), "cell", 12),
    ],
    ids=["functional_6", "residual_12"],
)
def test_op_segment_per_call_containment_pin(builder, address, calls, tmp_path):
    """Round-24 C1 pin: exactly one per-call unit, inside that call's cluster.

    The executed seal repro: six calls of a parameter-free functional module
    (and twelve of a real residual cell) produced TWO max segments declared
    ``owner=<addr>:1`` / ``owner=<addr>:4`` with most represented ops outside
    the owner and the other call clusters empty. Post-fix, max must cover
    every call with exactly ONE unit that absorbs no cross-call ops. Honest
    per-call box pricing (megaplan row C05, collapse M2a) may now win with
    one ModuleBox per call instead of one op segment per call -- both shapes
    are legal; two spanning units for six calls never are. When segments win,
    each must be owned by and rendered inside THAT call's cluster, and every
    call cluster must be non-empty in the DOT.
    """

    torch.manual_seed(0)
    trace = tl.trace(builder().eval(), torch.randn(1, 8))
    context = RenderContext()
    result = select_collapse_plan(trace, context, mode="max")
    assert not result.declined
    descriptors = {
        name: segment for name, segment in (result.segments or {}).items() if segment.kind == "op"
    }
    box_calls = sorted(
        node.call
        for node in result.plan.nodes
        if isinstance(node, ModuleBox) and node.call.rsplit(":", 1)[0] == address
    )
    expected_calls = sorted(f"{address}:{index}" for index in range(1, calls + 1))
    if not descriptors:
        assert box_calls == expected_calls, (
            f"segment-free max plan must draw one box per call; got {box_calls}"
        )
    else:
        assert len(descriptors) == calls, (
            f"expected one per-call op segment per call, got {len(descriptors)}"
        )
        assert sorted(segment.owner for segment in descriptors.values()) == expected_calls
        for name, segment in descriptors.items():
            represented_calls = {_effective_stack(trace.ops[str(op)])[-1] for op in segment.ops}
            assert represented_calls == {segment.owner}, (
                f"{name}: represents calls {sorted(represented_calls)} but claims "
                f"sole ownership of {segment.owner}"
            )

    out = tmp_path / f"per_call_{address}"
    source = trace.draw(
        vis_save_only=True,
        vis_fileformat="svg",
        collapse="max",
        vis_outpath=str(out),
    )
    node_cluster, members = _dot_node_clusters(source)
    for segment in descriptors.values():
        assert node_cluster.get(segment.name) == _owner_cluster_name(segment.owner)
    if descriptors:
        for index in range(1, calls + 1):
            cluster = f"cluster_{address}_pass{index}"
            assert members.get(cluster), (
                f"call cluster {cluster} owns represented ops but contains no node"
            )
    assert _svg_node_group_count(str(out) + ".svg") == result.plan.total


def test_op_segment_sibling_merge_keeps_honest_top_level_lca(tmp_path):
    """Distinct single-call sibling modules may still merge into one segment.

    The containment boundary is module-call REUSE, not any stack change: a
    consecutive raw run across five different single-call blocks has no call
    boundary to cross, so it must stay merged with owner ``None`` (the exact
    call-qualified LCA) and render at top level. Splitting it would regress
    max-mode compression pinned by the render-identity oracle.
    """

    torch.manual_seed(0)
    trace = tl.trace(SiblingBlockChain().eval(), torch.randn(1, 8))
    context = RenderContext()
    result = select_collapse_plan(trace, context, mode="max")
    assert not result.declined
    descriptors = {
        name: segment for name, segment in (result.segments or {}).items() if segment.kind == "op"
    }
    assert descriptors, "fixture must produce op segments at max"
    spanning = [
        segment
        for segment in descriptors.values()
        if len({_effective_stack(trace.ops[str(op)])[:1] for op in segment.ops}) > 1
    ]
    assert spanning, "fixture must produce a segment spanning sibling modules"
    for segment in spanning:
        assert segment.owner is None, (
            f"sibling-spanning segment claims module owner {segment.owner!r}"
        )
    out = tmp_path / "sibling_merge"
    source = trace.draw(
        vis_save_only=True,
        vis_fileformat="svg",
        collapse="max",
        vis_outpath=str(out),
    )
    node_cluster, _ = _dot_node_clusters(source)
    for segment in spanning:
        assert node_cluster.get(segment.name) is None, (
            "owner-less sibling segment must render at top level"
        )
    assert _svg_node_group_count(str(out) + ".svg") == result.plan.total


def test_op_segment_owner_key_is_call_exact():
    """The owner key compares CALLS exactly, never pass-free commonality.

    Pre-fix ``_op_segment_owner_key`` stripped pass suffixes before testing
    commonality and then returned ``values[0]``, so a tuple spanning
    ``core:1..core:2`` was blessed with owner ``core:1`` -- the round-24
    false-containment root. Cross-call tuples must resolve to the honest LCA
    (``None`` here) and single-call tuples to that exact call.
    """

    torch.manual_seed(0)
    trace = tl.trace(FunctionalCoreReuse(6).eval(), torch.randn(1, 8))
    by_call: dict[str, list[str]] = {}
    for op in trace.ops:
        stack = _effective_stack(op)
        if stack:
            by_call.setdefault(stack[-1], []).append(str(op.label))
    assert set(by_call) == {f"core:{index}" for index in range(1, 7)}
    within = tuple(by_call["core:1"])
    assert _op_segment_owner_key(trace, within) == "core:1"
    assert _op_segment_owner_key(trace, within, "unrolled") == "core:1"
    cross = tuple(by_call["core:1"] + by_call["core:2"])
    assert _op_segment_owner_key(trace, cross) is None, (
        "cross-call tuple must own the honest LCA, not the first call"
    )


def test_own_ops_segment_refuses_module_call_reuse():
    """The DP own-ops segment producer must refuse reuse-crossing sequences.

    ``_instantiate_module`` replaces a module's own ops with ONE segment
    node when legal; for a multi-call module those ops span several rendered
    call clusters, so accepting them reproduces the round-24 containment lie
    through a second producer. Single-call sequences stay legal.
    """

    torch.manual_seed(0)
    trace = tl.trace(FunctionalCoreReuse(6).eval(), torch.randn(1, 8))
    by_call: dict[str, list[str]] = {}
    for op in trace.ops:
        stack = _effective_stack(op)
        if stack:
            by_call.setdefault(stack[-1], []).append(str(op.label))
    state = SimpleNamespace(trace=trace, context=RenderContext(), total_ops=10_000)
    assert _own_ops_segment_is_legal(state, tuple(by_call["core:1"]))
    cross = tuple(by_call["core:1"] + by_call["core:2"])
    assert not _own_ops_segment_is_legal(state, cross), (
        "own-ops producer accepted a sequence spanning two calls of one module"
    )


# ---------------------------------------------------------------------------
# Round-25: box + op-segment double-representation of surfaced atomic-exit ops
# ---------------------------------------------------------------------------


class AtomicExitBlock(nn.Module):
    """Linear child plus ONE own functional op: the relu is an atomic exit.

    The block's only own op is the relu, so the renderer drops the innermost
    box for it and keeps the op visible as the collapsed box's separate
    sibling node (the box remainder label subtracts it).
    """

    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.l = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.l(x))


class NonAtomicExitBlock(nn.Module):
    """Linear child plus TWO own functional ops: exits stay inside the box.

    ``is_atomic_module`` requires the innermost call to own exactly one op,
    so neither the tanh nor the relu surfaces; this is the classification
    control for the round-25 sweep.
    """

    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.l = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(torch.tanh(self.l(x)))


def _tail_op(index: int, x: torch.Tensor) -> torch.Tensor:
    """Apply the ``index``-th trailing chain op, resolving torch at call time.

    Pre-binding ``torch.relu`` and friends in a module-level tuple would
    capture the raw callables before TorchLens wraps torch and escape capture.
    """

    return (torch.relu, torch.tanh, torch.sigmoid)[index % 3](x)


class SiblingBlocksTrailingChain(nn.Module):
    """Round-25 seal repro (NetF): sibling blocks then a trailing chain.

    The last block's surfaced atomic-exit op is immediately followed by
    ``tail`` top-level ops, so a >=3-op run forms starting AT the exit op
    and, pre-fix, absorbed it into the adjacent op segment.
    """

    def __init__(
        self,
        block_cls: type[nn.Module] = AtomicExitBlock,
        width: int = 8,
        depth: int = 6,
        tail: int = 3,
    ) -> None:
        super().__init__()
        self.tail = tail
        self.blocks = nn.ModuleList([block_cls(width) for _ in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(x)
        x = torch.tanh(x)
        x = torch.sigmoid(x)
        x = x * 2
        x = x + 1
        for block in self.blocks:
            x = block(x)
        for index in range(self.tail):
            x = _tail_op(index, x)
        return x


class ReusedAtomicExitBlock(nn.Module):
    """ONE atomic-exit block called several times, then a trailing chain.

    Interplay with the round-24 containment class: each call renders its own
    box plus its own surfaced exit-op occurrence, and only the LAST call's
    occurrence is execution-adjacent to the trailing top-level chain.
    """

    def __init__(self, width: int = 8, calls: int = 4, tail: int = 3) -> None:
        super().__init__()
        self.blk = AtomicExitBlock(width)
        self.calls = calls
        self.tail = tail

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(self.calls):
            x = self.blk(x)
        for index in range(self.tail):
            x = _tail_op(index, x)
        return x


class NestedAtomicExitBlocks(nn.Module):
    """Sibling atomic-exit blocks and their trailing chain inside a container."""

    class Inner(nn.Module):
        def __init__(self, width: int = 8, depth: int = 4, tail: int = 3) -> None:
            super().__init__()
            self.tail = tail
            self.blocks = nn.ModuleList([AtomicExitBlock(width) for _ in range(depth)])

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for block in self.blocks:
                x = block(x)
            for index in range(self.tail):
                x = _tail_op(index, x)
            return x

    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.inner = self.Inner(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(torch.relu(x))


def _boxed_surfaced_labels(trace, plan) -> set[str]:
    """Return pass-free surfaced own-output labels of every plan module box.

    Independent test-side witness (not the implementation helper): an op is
    box-owned surfaced when it is a non-buffer atomic-exit op whose own
    module renders as a collapsed box or repeat-fold representative in the
    plan.
    """

    addresses = {node.call.rsplit(":", 1)[0] for node in plan.nodes if isinstance(node, ModuleBox)}
    addresses.update(
        node.rep.call.rsplit(":", 1)[0] for node in plan.nodes if isinstance(node, RepeatFold)
    )
    labels: set[str] = set()
    for op in trace.ops:
        modules = list(getattr(op, "modules", ()) or ())
        if (
            getattr(op, "is_atomic_module", False)
            and not getattr(op, "is_buffer", False)
            and modules
            and modules[-1].rsplit(":", 1)[0] in addresses
        ):
            labels.add(str(op.layer_label))
    return labels


def _plan_segment_member_labels(plan) -> set[str]:
    """Return pass-free member labels of every op segment in a plan."""

    labels: set[str] = set()
    for node in plan.nodes:
        if isinstance(node, OpSegment):
            labels.update(str(label).rsplit(":", 1)[0] for label in node.ops)
    return labels


def test_surfaced_exit_op_not_double_represented(tmp_path):
    """Round-25 pin: every op appears in EXACTLY ONE rendered label.

    Executed seal repro: six structurally identical sibling blocks whose last
    exit relu is followed by a 3-op trailing chain. Pre-fix, the max plan
    absorbed the last block's surfaced exit op into the trailing op segment
    while the box content label kept counting it: boxes read
    ``1/1/1/1/1/2 ops``, the segment claimed 4 ops, and 5 real ops in the
    region carried 6 label claims. Post-fix this asserts, on the max surface:

    - no op segment claims a box-owned surfaced exit op;
    - structurally identical sibling boxes render IDENTICAL op counts;
    - each rendered op is claimed by exactly one label (box remainder,
      segment range, or standalone raw node) with exact conservation;
    - node-count parity holds: ``plan.total == count == rendered SVG nodes``.
    """

    torch.manual_seed(0)
    trace = tl.trace(SiblingBlocksTrailingChain().eval(), torch.randn(2, 8))
    context = RenderContext()
    result = select_collapse_plan(trace, context, mode="max")
    plan = result.plan

    # 1. No segment member is a box-owned surfaced exit op.
    overlap = _boxed_surfaced_labels(trace, plan) & _plan_segment_member_labels(plan)
    assert not overlap, f"op segments absorb box-owned surfaced exit ops: {sorted(overlap)}"

    # 2. Exact one-claim-per-op conservation across all label kinds.
    box_addresses = [
        node.call.rsplit(":", 1)[0] for node in plan.nodes if isinstance(node, ModuleBox)
    ]
    stub = SimpleNamespace(_torchlens_v2_plan=plan, _torchlens_v2_mode="max")
    claimed: list[str] = []
    for address in box_addresses:
        module_call = trace.modules[address].ops[0]
        module_labels = {str(trace.ops[label].layer_label) for label in module_call.ops}
        if _collapsed_module_should_show_remainder(trace, address, module_call.ops, stub):
            surfaced = {
                str(op.layer_label)
                for op in trace.ops
                if getattr(op, "is_atomic_module", False)
                and (getattr(op, "modules", ()) or ("",))[-1].rsplit(":", 1)[0] == address
            }
            module_labels -= surfaced
        claimed.extend(sorted(module_labels))
    for node in plan.nodes:
        if isinstance(node, RawOp):
            claimed.append(str(node.op) if isinstance(node.op, str) else str(node.op.layer_label))
        elif isinstance(node, OpSegment):
            claimed.extend(str(label).rsplit(":", 1)[0] for label in node.ops)
    all_labels = sorted(str(op.layer_label) for op in trace.ops)
    assert sorted(claimed) == all_labels, (
        f"label claims must cover each op exactly once; claims={sorted(claimed)} ops={all_labels}"
    )

    # 3. SVG: sibling boxes identical, label-op totals honest, parity exact.
    out = tmp_path / "r25_seal"
    trace.draw(
        vis_save_only=True,
        vis_fileformat="svg",
        collapse="max",
        vis_outpath=str(out),
    )
    with open(str(out) + ".svg", encoding="utf-8") as handle:
        svg = handle.read()
    # Honest per-call box pricing (megaplan row C05, collapse M2a) may win
    # this toy's max plan with spanning segments instead of the historical
    # six sibling boxes -- the box checks bind to however many boxes the
    # winning plan actually draws, and the conservation check below carries
    # the seal either way.
    box_nodes = sum(isinstance(node, ModuleBox) for node in plan.nodes)
    box_counts = re.findall(r">(\d+) ops?<", svg)
    assert len(box_counts) == box_nodes, (
        f"expected one box content label per plan ModuleBox ({box_nodes}), got {box_counts}"
    )
    assert len(set(box_counts)) <= 1, (
        f"structurally identical sibling blocks render inconsistent op counts: {box_counts}"
    )
    # Segment labels may carry a trailing spanned-modules disclosure
    # (" -- spans @..."), so the op count is matched mid-run, not at the end.
    seg_counts = [
        int(match) for match in re.findall(r">[^<>]*?&#45;&#45; (\d+) ops?\b[^<>]*<", svg)
    ]
    standalone_raw = sum(isinstance(node, RawOp) for node in plan.nodes)
    label_total = sum(int(count) for count in box_counts) + sum(seg_counts) + standalone_raw
    assert label_total == len(list(trace.ops)), (
        f"total label op-count {label_total} != real op count {len(list(trace.ops))}"
    )
    assert _svg_node_group_count(str(out) + ".svg") == plan.total == result.visible_count


_R25_SWEEP_CASES = [
    ("seal_tail3", lambda: SiblingBlocksTrailingChain(tail=3)),
    ("seal_tail2", lambda: SiblingBlocksTrailingChain(tail=2)),
    ("seal_tail8", lambda: SiblingBlocksTrailingChain(tail=8)),
    ("seal_tail0", lambda: SiblingBlocksTrailingChain(tail=0)),
    ("depth2_tail3", lambda: SiblingBlocksTrailingChain(depth=2, tail=3)),
    ("nonatomic_exits", lambda: SiblingBlocksTrailingChain(block_cls=NonAtomicExitBlock)),
    ("reused_calls4", lambda: ReusedAtomicExitBlock(calls=4, tail=3)),
    ("reused_calls6_tail5", lambda: ReusedAtomicExitBlock(calls=6, tail=5)),
    ("nested_container", lambda: NestedAtomicExitBlocks()),
]


@pytest.mark.parametrize(
    "builder",
    [case[1] for case in _R25_SWEEP_CASES],
    ids=[case[0] for case in _R25_SWEEP_CASES],
)
def test_box_owned_exit_ops_never_absorbed_by_segments(builder):
    """Round-25 class sweep: op segments never claim box-owned surfaced ops.

    Every collapse surface (auto, max, float levels) in both vis modes must
    keep each box-owned surfaced atomic-exit op OUT of every op segment, so
    the box remainder label and the segment range label never both claim one
    op. Covers trailing chains too short to re-segment, sibling and reused
    blocks (round-24 interplay), nested containers, and the non-atomic
    classification control.
    """

    torch.manual_seed(0)
    trace = tl.trace(builder().eval(), torch.randn(2, 8))
    for vis_mode in ("unrolled", "rolled"):
        context = RenderContext(vis_mode=vis_mode)
        results = {
            "auto": select_collapse_plan(trace, context, mode="auto"),
            "max": select_collapse_plan(trace, context, mode="max"),
        }
        for level in (0.5, 1.0):
            results[f"t={level}"] = select_collapse_level(trace, context, level)
        for label, result in results.items():
            if result.declined:
                continue
            overlap = _boxed_surfaced_labels(trace, result.plan) & _plan_segment_member_labels(
                result.plan
            )
            assert not overlap, (
                f"{vis_mode}/{label}: op segments absorb box-owned surfaced "
                f"exit ops: {sorted(overlap)}"
            )
            assert result.visible_count == result.plan.total


def test_remainder_counts_segment_members_as_rendered_separately():
    """Round-25 belt: remainder accounting sees op-segment members as visible.

    ``_collapsed_module_should_show_remainder`` treated a surfaced exit op as
    rendered-separately ONLY when it was a standalone ``RawOp`` plan node;
    an op absorbed into an ``OpSegment`` returned False, so the box remainder
    was not subtracted and the op was double-counted. The remainder side must
    treat segment membership exactly like standalone raw visibility, even if
    a future producer path reintroduces absorption.
    """

    torch.manual_seed(0)
    trace = tl.trace(SiblingBlocksTrailingChain().eval(), torch.randn(2, 8))
    plan = select_collapse_plan(trace, RenderContext(), mode="max").plan
    module_call = trace.modules["blocks.5"].ops[0]
    exit_label = "relu_7_17"
    assert exit_label in {str(trace.ops[label].layer_label) for label in module_call.ops}

    # Synthesize the pre-fix absorbed shape: the exit op inside a segment.
    absorbed_nodes = []
    for node in plan.nodes:
        if isinstance(node, RawOp) and node.op == exit_label:
            continue
        if isinstance(node, OpSegment) and exit_label not in node.ops:
            last_seg = OpSegment((exit_label, *node.ops)) if "tanh_2_19" in node.ops else node
            absorbed_nodes.append(last_seg)
            continue
        absorbed_nodes.append(node)
    absorbed = CollapsePlan(nodes=tuple(absorbed_nodes), context=plan.context)
    assert exit_label in _plan_separately_rendered_op_labels(absorbed)
    stub = SimpleNamespace(_torchlens_v2_plan=absorbed)
    assert _collapsed_module_should_show_remainder(trace, "blocks.5", module_call.ops, stub), (
        "a surfaced exit op absorbed into an op segment must still count as "
        "rendered separately so the box remainder subtracts it"
    )

    # Control: with the op neither standalone nor in any segment, no remainder.
    hidden_nodes = tuple(
        node
        for node in absorbed_nodes
        if not (isinstance(node, OpSegment) and exit_label in node.ops)
    )
    hidden = CollapsePlan(nodes=hidden_nodes, context=plan.context)
    stub_hidden = SimpleNamespace(_torchlens_v2_plan=hidden)
    assert not _collapsed_module_should_show_remainder(
        trace, "blocks.5", module_call.ops, stub_hidden
    )


def test_run_builder_refuses_box_owned_exit_op():
    """Round-25 producer pin: runs refuse box-owned surfaced exit ops.

    ``_legal_plan_op_segment_run`` must not start (or continue) a run at a
    collapsed box's surfaced exit op; the run must instead start at the first
    genuinely top-level op, and without the guard set the historical
    absorbing 4-op run demonstrates what is being refused.
    """

    torch.manual_seed(0)
    trace = tl.trace(SiblingBlocksTrailingChain().eval(), torch.randn(2, 8))
    context = RenderContext()
    nodes = (
        ModuleBox("blocks.5:1"),
        RawOp("relu_7_17"),
        RawOp("relu_8_18"),
        RawOp("tanh_2_19"),
        RawOp("sigmoid_2_20"),
    )
    box_owned = _plan_box_owned_surfaced_labels(trace, context, nodes)
    assert "relu_7_17" in box_owned
    concrete = {1: "relu_7_17:1", 2: "relu_8_18:1", 3: "tanh_2_19:1", 4: "sigmoid_2_20:1"}
    concrete = {
        index: label if label in {str(op.label) for op in trace.ops} else label.rsplit(":", 1)[0]
        for index, label in concrete.items()
    }
    guarded = _legal_plan_op_segment_run(
        trace,
        context,
        nodes,
        1,
        total_ops=22,
        dominance_limit=0.75,
        concrete_labels=concrete,
        box_owned_labels=box_owned,
    )
    assert guarded is None, "run starting at a box-owned surfaced exit op must be refused"
    shifted = _legal_plan_op_segment_run(
        trace,
        context,
        nodes,
        2,
        total_ops=22,
        dominance_limit=0.75,
        concrete_labels=concrete,
        box_owned_labels=box_owned,
    )
    assert shifted == ("relu_8_18", "tanh_2_19", "sigmoid_2_20")
    unguarded = _legal_plan_op_segment_run(
        trace,
        context,
        nodes,
        1,
        total_ops=22,
        dominance_limit=0.75,
        concrete_labels=concrete,
        box_owned_labels=frozenset(),
    )
    assert unguarded == ("relu_7_17", "relu_8_18", "tanh_2_19", "sigmoid_2_20"), (
        "control: without the guard set the absorbing run forms, so the guard is what refuses it"
    )


class TestIndexedChildStemLinearParser:
    """The indexed-child stem parser (regex-backtracking replacement)."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("layer12", "layer"),
            ("block_3a", "block"),
            ("block.3", "block"),
            ("1234", ""),
            ("_5", ""),
            ("a.12b", "a."),
            ("3a7", "3a"),
            ("12ab", None),  # two trailing letters: not an indexed suffix
            ("layer", None),
            ("", None),
            ("layer12\n", None),  # regex `$` newline quirk deliberately dropped
        ],
    )
    def test_stem_semantics(self, name: str, expected: str | None) -> None:
        """Pin the lazy-stem/longest-suffix semantics of the old regex.

        The historical ``^(?P<stem>.*?)(?:\\.?\\d+|_?\\d+[a-z]?)$`` pattern
        backtracked quadratically on artifact-supplied names (measured 26s at
        40k chars); the manual parser was fuzz-checked equivalent over 20k
        random names from the {a, b, _, ., 0, 1, 9, z} alphabet.
        """

        from torchlens.visualization.auto_collapse import _indexed_child_stem

        assert _indexed_child_stem(name) == expected

    def test_pathological_input_is_linear(self) -> None:
        """The all-digits-plus-junk adversarial name parses instantly."""

        from torchlens.visualization.auto_collapse import _indexed_child_stem

        assert _indexed_child_stem("9" * 100_000 + "!!") is None


class TestNameConsecutiveTrailingDigitSplit:
    """collapse_optimizer._members_are_name_consecutive digit parsing."""

    def test_consecutive_and_rejections(self) -> None:
        """Trailing-digit semantics match the old (.*?)(\\d+) fullmatch."""

        from torchlens.visualization._segment_descriptors import _members_are_name_consecutive

        assert _members_are_name_consecutive(("m.block1", "m.block2", "m.block3"))
        assert not _members_are_name_consecutive(("m.block1", "m.block3"))
        assert not _members_are_name_consecutive(("m.block1", "n.block2"))
        assert not _members_are_name_consecutive(("m.block1", "m.layer2"))
        assert not _members_are_name_consecutive(("m.block", "m.block2"))
        # Adversarial leaf: linear, and correctly not consecutive.
        assert not _members_are_name_consecutive(("m." + "9" * 100_000 + "x", "m.b2"))


class TestRenderDisclosureCeiling:
    """R60: draw() disclosure warning above the Graphviz-hostile size."""

    def test_small_render_is_silent(self) -> None:
        import warnings as _warnings

        from torchlens.visualization.render_ir import (
            _warn_if_render_exceeds_disclosure_ceiling,
        )

        with _warnings.catch_warnings():
            _warnings.simplefilter("error")
            _warn_if_render_exceeds_disclosure_ceiling(100, 200)

    def test_oversized_render_warns_with_remedies(self) -> None:
        from torchlens.visualization.render_ir import (
            RENDER_DISCLOSURE_NODE_CEILING,
            _warn_if_render_exceeds_disclosure_ceiling,
        )

        with pytest.warns(UserWarning, match="collapse"):
            _warn_if_render_exceeds_disclosure_ceiling(RENDER_DISCLOSURE_NODE_CEILING + 1, 0)
