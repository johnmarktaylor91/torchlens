"""Tests for the R3a v2 collapse optimizer."""

from __future__ import annotations

import random
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch

import torchlens as tl
import torchlens._visualization.auto_collapse as auto_collapse
from torchlens._data_classes._trace_accessors import TraceOpAccessor
from torchlens._visualization._trace_collapse import (
    collapse_plan as _collapse_plan,
    collapse_schedule as _collapse_schedule,
)
from torchlens._visualization.auto_collapse import (
    _resolve_relationship_op,
    analyze_collapse,
)
from torchlens._visualization.collapse_optimizer import (
    _RESULT_CACHE,
    _SCHEDULE_CACHE,
    K_CAP,
    RoleComponent,
    _FrontierPoint,
    _prune_frontier,
    _same_role,
    build_role_components,
    collapse_schedule,
    select_collapse_plan,
)
from torchlens._visualization.collapse_plan import (
    CollapsePlan,
    RenderContext,
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

    def __init__(self, width: int = 4, branch: type[torch.nn.Module] = ParallelBranch) -> None:
        """Initialize the fan head."""

        super().__init__()
        self.branches = torch.nn.ModuleList([branch(width) for _ in range(4)])
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


def _trace(model: torch.nn.Module, x: torch.Tensor) -> tl.Trace:
    """Trace a model in eval mode."""

    model.eval()
    grad_enabled = torch.is_grad_enabled()
    torch.set_grad_enabled(False)
    try:
        return tl.trace(model, x)
    finally:
        torch.set_grad_enabled(grad_enabled)


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


def _require_transformers() -> ModuleType:
    """Import transformers or skip the calling test."""

    return pytest.importorskip("transformers")


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


def test_float_collapse_level_validation_and_endpoints() -> None:
    """Float collapse levels validate and preserve none/max endpoint plans."""

    trace = _trace(UniformStack(depth=6), torch.randn(2, 8))
    context = RenderContext()
    try:
        none_plan = collapse_plan_for_trace(trace, None, None, context)
        max_result = select_collapse_plan(trace, context, mode="max")

        with pytest.raises(ValueError, match=r"\[0\.0, 1\.0\]"):
            _collapse_plan(trace, mode=-0.1)
        with pytest.raises(ValueError, match=r"\[0\.0, 1\.0\]"):
            trace.draw(collapse=1.1)

        assert _collapse_plan(trace, mode=0.0) == none_plan
        assert _collapse_plan(trace, mode=1.0) == max_result.plan
    finally:
        trace.cleanup()


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


def test_trace_collapse_plan_public_diagnostic(monkeypatch: pytest.MonkeyPatch) -> None:
    """The collapse plan function returns the v2 plan without adding top-level API names."""

    trace = _trace(UniformStack(depth=12), torch.randn(2, 8))
    try:
        plan = _collapse_plan(trace, mode="auto")
        summary = repr(plan)

        assert isinstance(plan, CollapsePlan)
        assert count(plan) > 0
        assert len(plan) == count(plan)
        assert plan.total == count(plan)
        assert summary.startswith("CollapsePlan(total=")
        assert "module_box=" in summary or "raw_op=" in summary

        with pytest.raises(ValueError, match="mode must be one of"):
            _collapse_plan(trace, mode="none")  # type: ignore[arg-type]
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


def test_segment_descriptor_parity_guard_survives_python_O() -> None:
    """The label-honesty cardinality guards raise, not assert (r-b7 R24-2).

    ``python -O`` strips asserts; these two guards defend the segment-box /
    ``(xN)`` / ellipsis honesty contract on the DEFAULT ``draw(collapse=)``
    path, so they must fire with assertions disabled too.
    """

    from torchlens._visualization.collapse_optimizer import (
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

    from torchlens._visualization.collapse_optimizer import _assert_visible_plan

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
    from torchlens._visualization._render_regions import _assert_rank_group_parity

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
    path, (b) refuse typed from ``_collapse_plan(Trace)``, and (c) degrade
    the schedule to its single full-graph step -- while sub-ceiling traces
    are untouched.
    """

    import warnings as warnings_module

    from torch import nn

    from torchlens._errors import InvalidArgumentError
    from torchlens._visualization import collapse_optimizer as optimizer_module

    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4), nn.ReLU())
    trace = tl.trace(model, torch.randn(2, 4))

    # Sub-ceiling: the real ceiling admits this trace and produces a plan.
    assert _collapse_plan(trace, mode="max") is not None

    # Pathological arm (F11 memo D5(i)): above 20x the constant, smart
    # collapse declines outright (constant 0 makes any trace qualify).
    monkeypatch.setattr(optimizer_module, "COLLAPSE_OPTIMIZER_MAX_OPS", 0)
    fresh = tl.trace(model, torch.randn(2, 4))
    with pytest.warns(UserWarning, match="skipping smart collapse"):
        dot = fresh.draw(
            collapse="max",
            file_format="dot",
            order_siblings=False,
        ).dot
    assert dot  # uncollapsed render still ships

    with warnings_module.catch_warnings():
        warnings_module.simplefilter("ignore", UserWarning)
        with pytest.raises(InvalidArgumentError) as exc_info:
            _collapse_plan(fresh, mode="auto")
    assert exc_info.value.fields["code"] == "collapse_plan_unavailable"
    assert "collapse_ops_ceiling" in exc_info.value.fields["reason"]

    with warnings_module.catch_warnings():
        warnings_module.simplefilter("ignore")
        schedule = _collapse_schedule(fresh)
    assert len(schedule.steps) == 1

    # The budget-degrade arm (non-pathological over-budget -> the fallback
    # planner) is pinned in tests/test_collapse_power_ceiling.py.


# Kept: the test shrink removed this test; the docs move to docs/internal/ edited it after,
# so the edited version stays.
def test_collapse_ceiling_documented_lockstep() -> None:
    """The compute ceiling must stay documented everywhere user-facing (R19).

    The 0957027b ceiling changed the documented behaviour of
    ``collapse="auto"|"max"``, ``_collapse_plan(Trace)`` and
    ``_collapse_schedule(Trace)`` for large traces; per the LOCKED docs rule the
    constant (and its current value) must appear in the user-facing collapse
    docs, the limitations catalog, the glossary, and the agent guide (AGENTS.md;
    CLAUDE.md only imports it).
    """

    from torchlens._visualization.collapse_optimizer import COLLAPSE_OPTIMIZER_MAX_OPS

    repo_root = Path(__file__).resolve().parents[1]
    doc_pages = [
        repo_root / "docs" / "reference" / "collapse.md",
        repo_root / "docs" / "reference" / "limitations.md",
        repo_root / "docs" / "reference" / "glossary.md",
        # AGENTS.md no longer carries this prose directly (2026-10-01 docs move:
        # "docs: move agent reference material out of startup instructions"); it
        # lives in this pointer target now (same precedent as the __all__ count
        # claim repoint in test_docs_lockstep_names.py).
        repo_root / "docs" / "internal" / "agent-reference" / "current-2-x-surface.md",
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

    from torchlens._data_classes._accessor_base import Accessor

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
