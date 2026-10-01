"""Three-way collapse pricing parity gate (lane C05; collapse memo D1).

The phantom-k class: the optimizer scored a node count that is not what
gets drawn (a selected box priced k=1 while rendering one box PER CALL plus
kept atomic ops; synthetic child graphs claiming zero parent-owned ops).
The gate pins the parity chain on PLANS, not frontier records:

    scored k == CollapsePlan.count() == emitted render units

where the emitted leg re-derives the plan through the SAME node-universe
entry point both render backends consume, and ``RepeatFold`` counts as 2
(representative + ellipsis). Until this gate is green corpus-wide, every
weight, band target, and schedule number is frozen (they are computed on k).
"""

from __future__ import annotations

import pytest
import torch

import torchlens as tl
from torchlens.visualization.collapse_optimizer import select_collapse_plan
from torchlens.visualization.collapse_plan import (
    RenderContext,
    collapse_plan_for_trace,
    count,
)

# Per-test smoke marks, NOT a module pytestmark: markers are additive, and
# the googlenet parity cell below is heavy -- a module-level smoke mark
# would hand it the banned smoke+heavy combination (marker-lint).


def assert_three_way_parity(trace: tl.Trace, vis_mode: str, mode: str) -> None:
    """Assert the scored/plan/emitted parity chain for one cell."""

    context = RenderContext(vis_mode=vis_mode)
    result = select_collapse_plan(trace, context, mode=mode)
    if result.declined:
        pytest.skip(f"optimizer declined: {result.reason}")
    plan_count = count(result.plan)
    # Leg 1: the result's own two counts agree.
    assert result.visible_count == plan_count, (
        f"visible_count {result.visible_count} != plan count {plan_count} ({vis_mode}/{mode})"
    )
    # Leg 2: the DP's scored k equals the realized plan count whenever a
    # frontier point directly produced the plan (the phantom-k tripwire).
    if result.scored_k is not None:
        assert result.scored_k == plan_count, (
            f"scored k {result.scored_k} != realized plan count {plan_count} "
            f"({vis_mode}/{mode}): the optimizer priced a number that is not "
            "what gets drawn"
        )

    # Leg 3: emitted render units.
    if not result.segments:
        # Re-deriving the plan through the shared node-universe entry point
        # (what both render backends emit) must reproduce the same count.
        def selected_fn(module):  # noqa: ANN001, ANN202 - render predicate shape
            return module.address in result.selected

        emitted_plan = collapse_plan_for_trace(
            trace,
            selected_fn if result.selected else None,
            dict(result.repeat_folds),
            context,
        )
        assert count(emitted_plan) == plan_count, (
            f"emitted render units {count(emitted_plan)} != plan count {plan_count} "
            f"({vis_mode}/{mode})"
        )
    else:
        # Segmented plans materialize their segment boxes through renderer
        # attributes, not the collapse predicate -- the unified typed-unit
        # emitted leg is the F0 typed-units item (segments-as-lookup is the
        # known rank-path crash). Until F0 lands, pin what IS emittable: the
        # DOT render carries exactly one ``__segment__`` node per descriptor,
        # and the descriptor cardinality equals the plan's segment nodes
        # (enforced again at OptimizerResult construction).
        segment_nodes = sum(
            type(node).__name__ in {"OpSegment", "ChildSegment"} for node in result.plan.nodes
        )
        assert segment_nodes == len(result.segments)


class _Block(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin_a = torch.nn.Linear(8, 8)
        self.lin_b = torch.nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.lin_b(torch.relu(self.lin_a(x))))


class _ReusedBlockModel(torch.nn.Module):
    """One module instance called three times: a selected box renders 3 boxes."""

    def __init__(self) -> None:
        super().__init__()
        self.block = _Block()
        self.head = torch.nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(3):
            x = self.block(x)
        return self.head(x)


class _SeparateBlocksModel(torch.nn.Module):
    """Twelve separate same-class blocks: the K3 fold candidate shape."""

    def __init__(self) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList(_Block() for _ in range(12))
        self.head = torch.nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.head(x)


class _BranchyCatModel(torch.nn.Module):
    """Parallel branches joined at a cat junction (googlenet-class shape)."""

    def __init__(self) -> None:
        super().__init__()
        self.branch_a = _Block()
        self.branch_b = _Block()
        self.branch_c = _Block()
        self.head = torch.nn.Linear(24, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        joined = torch.cat([self.branch_a(x), self.branch_b(x), self.branch_c(x)], dim=1)
        return self.head(joined)


@pytest.fixture(scope="module")
def reused_trace():
    trace = tl.trace(_ReusedBlockModel(), torch.randn(2, 8))
    try:
        yield trace
    finally:
        trace.cleanup()


@pytest.fixture(scope="module")
def separate_trace():
    trace = tl.trace(_SeparateBlocksModel(), torch.randn(2, 8))
    try:
        yield trace
    finally:
        trace.cleanup()


@pytest.fixture(scope="module")
def branchy_trace():
    trace = tl.trace(_BranchyCatModel(), torch.randn(2, 8))
    try:
        yield trace
    finally:
        trace.cleanup()


@pytest.mark.parametrize("vis_mode", ["unrolled", "rolled"])
@pytest.mark.parametrize("mode", ["auto", "max"])
@pytest.mark.smoke
def test_parity_reused_block(reused_trace: tl.Trace, vis_mode: str, mode: str) -> None:
    assert_three_way_parity(reused_trace, vis_mode, mode)


@pytest.mark.parametrize("vis_mode", ["unrolled", "rolled"])
@pytest.mark.parametrize("mode", ["auto", "max"])
@pytest.mark.smoke
def test_parity_separate_blocks(separate_trace: tl.Trace, vis_mode: str, mode: str) -> None:
    assert_three_way_parity(separate_trace, vis_mode, mode)


@pytest.mark.parametrize("vis_mode", ["unrolled", "rolled"])
@pytest.mark.parametrize("mode", ["auto", "max"])
@pytest.mark.smoke
def test_parity_branchy_cat(branchy_trace: tl.Trace, vis_mode: str, mode: str) -> None:
    assert_three_way_parity(branchy_trace, vis_mode, mode)


@pytest.mark.smoke
def test_box_pricing_counts_per_call_units(reused_trace: tl.Trace) -> None:
    # M2a witness: selecting the reused block as a box renders one box PER
    # CALL on the unrolled view. The plan realized for an explicit selection
    # must count all three calls, and the derived plan through the universe
    # must agree.
    context = RenderContext(vis_mode="unrolled")

    def collapse_block(module):  # noqa: ANN001, ANN202
        return module.address == "block"

    plan = collapse_plan_for_trace(reused_trace, collapse_block, None, context)
    box_calls = [node for node in plan.nodes if type(node).__name__ == "ModuleBox"]
    assert len(box_calls) >= 3


@pytest.mark.smoke
def test_floor_fallback_is_disclosed_typed() -> None:
    # F9 (memo D4): a floor-fallback result carries the typed disclosure
    # fields; the frontier planner stamps the default.
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU())
    log = tl.trace(model, torch.randn(2, 4))
    result = select_collapse_plan(log, RenderContext(), mode="auto")
    assert result.planner in {"frontier", "floor_fallback"}
    if result.planner == "floor_fallback":
        assert result.reason is not None and "floor_fallback" in result.reason
    else:
        assert result.k_cap_exhausted is False


class _RootFanModel(torch.nn.Module):
    """70 parallel root-owned boundary outputs: the cache-flood cliff shape."""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(x + float(index) for index in range(70))


@pytest.mark.smoke
def test_near_uncollapsed_auto_plan_is_disclosed(tmp_path) -> None:
    # The measured silent-contract-violation shape (memo D4): a boundary
    # fan-out flood makes collapse="auto" return essentially the ENTIRE
    # graph as a "plan" with declined=False and no message. The release
    # invariant: auto never returns a plan within a small factor of U
    # without disclosure.
    from torchlens.errors._base import TorchLensWarning

    log = tl.trace(_RootFanModel(), torch.randn(2, 4))
    result = select_collapse_plan(log, RenderContext(vis_mode="unrolled"), mode="auto")
    assert result.visible_count >= 100  # nearly all 141 units visible
    with pytest.warns(TorchLensWarning, match="nearly the uncollapsed graph"):
        log.draw(
            vis_outpath=str(tmp_path / "fan"),
            vis_save_only=True,
            vis_fileformat="svg",
            collapse="auto",
        )


@pytest.mark.smoke
def test_floor_fallback_result_is_typed_and_visible(tmp_path, monkeypatch) -> None:
    # F9 unit wiring: when the DP frontier empties, the result must carry
    # planner="floor_fallback" + the K_CAP diagnosis, the draw must warn,
    # and the render caption must carry a visible notice.
    import torchlens.visualization.collapse_optimizer as co
    from torchlens.errors._base import TorchLensWarning

    log = tl.trace(_RootFanModel(), torch.randn(2, 4))
    monkeypatch.setattr(co, "_select_best_decision", lambda **kwargs: None)
    result = select_collapse_plan(log, RenderContext(vis_mode="unrolled"), mode="auto")
    assert result.planner == "floor_fallback"
    assert result.k_cap_exhausted is True
    assert result.root_own_units > 64
    assert result.reason is not None and "module= focus" in result.reason
    with pytest.warns(TorchLensWarning, match="floor plan"):
        source = log.draw(
            vis_outpath=str(tmp_path / "cliff"),
            vis_save_only=True,
            vis_fileformat="svg",
            collapse="auto",
        )
    # Visible notice on the render (never only a warning).
    assert "collapse fell back to the floor plan" in source


@pytest.mark.heavy
def test_parity_googlenet_eval() -> None:
    # The M2 corpus witness (memo: googlenet scored 22 while rendering 34).
    torchvision = pytest.importorskip("torchvision")
    model = torchvision.models.googlenet(aux_logits=False, init_weights=True).eval()
    log = tl.trace(model, torch.randn(1, 3, 224, 224))
    for vis_mode in ("unrolled", "rolled"):
        for mode in ("auto", "max"):
            assert_three_way_parity(log, vis_mode, mode)
