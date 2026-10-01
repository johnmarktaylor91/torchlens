"""L5 M3 pins: rank (stacking) channel — stack_by (a), explicit + licensed auto.

Covers the design-memo 4.1 contract: THE LOCKSTEP LICENSE adversarial matrix
(the case list IS the predicate — every globally monotone trace accepts,
every non-monotone trace refuses), the typed refusals, caption/legend
disclosure, rank=same emission under newrank=true, the sibling-ordering
no-op fence, and strict opt-in (plain draw() never stacks).

HONESTY: the license is a tripwire — auto-stacking must never claim "same
column = same timepoint" where no ground truth is derivable. Never weaken
it to make a figure render.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.visualization._stacking import (
    STACK_AUTO_DISPLAY,
    check_lockstep_license,
    resolve_stack_by,
)

# ---------------------------------------------------------------------------
# Fixtures: the license case list
# ---------------------------------------------------------------------------


class SmallMLP(nn.Module):
    """No multi-pass layers at all: auto is underivable."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


class Lockstep(nn.Module):
    """Vanilla lockstep: A1 B1 A2 B2 A3 B3 -> 1,1,2,2,3,3 (ACCEPTS)."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(4, 4)
        self.b = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(3):
            x = self.a(x)
            x = self.b(x)
        return x


class RaggedPrefix(nn.Module):
    """Early exit: A1 B1 A2 B2 A3 -> 1,1,2,2,3 (ragged monotone, ACCEPTS)."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(4, 4)
        self.b = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for step in range(3):
            x = self.a(x)
            if step < 2:
                x = self.b(x)
        return x


class MonotoneNested(nn.Module):
    """O1 I1 I2 O2 I3 I4 -> 1,1,2,2,3,4 (monotone nested interleaving, ACCEPTS)."""

    def __init__(self) -> None:
        super().__init__()
        self.outer = nn.Linear(4, 4)
        self.inner = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(2):
            x = self.outer(x)
            for _ in range(2):
                x = self.inner(x)
        return x


class ChainedLoops(nn.Module):
    """A1 A2 A3 B1 B2 B3 -> 1,2,3,1,2,3 (REFUSES)."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(4, 4)
        self.b = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(3):
            x = self.a(x)
        for _ in range(3):
            x = self.b(x)
        return x


class NestedNonMonotone(nn.Module):
    """O1 A1 A2 C1 O2 A3 A4 C2 -> 1,1,2,1,... breaks at C1 (REFUSES).

    The memo's exact nested-refusal fixture: an outer-body op running AFTER
    the inner loop in each outer iteration — nesting whose flat tally
    genuinely breaks monotonicity (nesting alone is not inherently
    non-monotone; MonotoneNested above accepts).
    """

    def __init__(self) -> None:
        super().__init__()
        self.opener = nn.Linear(4, 4)
        self.inner = nn.Linear(4, 4)
        self.closer = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(2):
            x = self.opener(x)
            for _ in range(2):
                x = self.inner(x)
            x = self.closer(x)
        return x


class LateResumption(nn.Module):
    """A1 B1 A2 A3 B2 -> 1,1,2,3,2 (interior skip resumes late, REFUSES)."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(4, 4)
        self.b = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.a(x)
        x = self.b(x)
        x = self.a(x)
        x = self.a(x)
        x = self.b(x)
        return x


class StemmedLockstep(nn.Module):
    """Single-pass stem + lockstep loop: stem ops stay un-annotated."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Linear(4, 4)
        self.a = nn.Linear(4, 4)
        self.b = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for _ in range(3):
            x = self.a(x)
            x = self.b(x)
        return x


def _trace(model: nn.Module) -> tl.Trace:
    return tl.trace(model, torch.randn(2, 4))


@pytest.fixture(scope="module")
def lockstep_log() -> Any:
    log = _trace(Lockstep())
    try:
        yield log
    finally:
        log.cleanup()


@pytest.fixture(scope="module")
def chained_log() -> Any:
    log = _trace(ChainedLoops())
    try:
        yield log
    finally:
        log.cleanup()


@pytest.fixture(scope="module")
def mlp_log() -> Any:
    log = _trace(SmallMLP())
    try:
        yield log
    finally:
        log.cleanup()


def _draw(log: tl.Trace, tmp_path: Path, **kwargs: Any) -> str:
    tmp_path.mkdir(parents=True, exist_ok=True)
    return log.draw(
        vis_save_only=True,
        vis_fileformat="svg",
        vis_outpath=str(tmp_path / "graph"),
        **kwargs,
    )


def _state(log: tl.Trace) -> Any:
    return log._last_encoding_state


# ---------------------------------------------------------------------------
# Option validation
# ---------------------------------------------------------------------------


def test_resolve_none_and_false_inactive() -> None:
    assert resolve_stack_by(None, "unrolled") is None
    assert resolve_stack_by(False, "unrolled") is None


def test_resolve_auto_forms() -> None:
    assert resolve_stack_by(True, "unrolled").source_kind == "auto"
    assert resolve_stack_by("auto", "unrolled").source_kind == "auto"
    assert resolve_stack_by(True, "unrolled").display_name == STACK_AUTO_DISPLAY


def test_resolve_field_and_callable_kinds() -> None:
    assert resolve_stack_by("pass_index", "unrolled").source_kind == "field"
    assert resolve_stack_by(lambda node: 1, "unrolled").source_kind == "callable"


def test_unknown_source_refuses() -> None:
    with pytest.raises(Exception) as excinfo:
        resolve_stack_by("no_such_field_anywhere", "unrolled")
    assert excinfo.value.fields["code"] == "encoding_source_invalid"


def test_non_string_non_callable_refuses() -> None:
    with pytest.raises(Exception) as excinfo:
        resolve_stack_by(42, "unrolled")
    assert excinfo.value.fields["code"] == "encoding_source_invalid"


def test_rolled_mode_refuses(lockstep_log: tl.Trace, tmp_path: Path) -> None:
    with pytest.raises(Exception) as excinfo:
        _draw(lockstep_log, tmp_path, stack_by=True, vis_mode="rolled")
    assert excinfo.value.fields["code"] == "stack_by_requires_unrolled"


# ---------------------------------------------------------------------------
# THE LOCKSTEP LICENSE — adversarial matrix (accepts AND refusals)
# ---------------------------------------------------------------------------


def _license_verdict(model: nn.Module) -> str:
    log = _trace(model)
    try:
        check_lockstep_license(log)
        return "accepts"
    # The matrix verifies TorchLens' structured code across refusal subclasses.
    except Exception as error:  # noqa: BLE001
        assert error.fields["code"] == "stack_by_auto_underivable"
        return "refuses"
    finally:
        log.cleanup()


def test_vanilla_lockstep_accepts() -> None:
    assert _license_verdict(Lockstep()) == "accepts"


def test_ragged_monotone_prefix_accepts() -> None:
    assert _license_verdict(RaggedPrefix()) == "accepts"


def test_monotone_nested_interleaving_accepts() -> None:
    assert _license_verdict(MonotoneNested()) == "accepts"


def test_chained_loops_refuse() -> None:
    assert _license_verdict(ChainedLoops()) == "refuses"


def test_nested_non_monotone_tally_refuses() -> None:
    assert _license_verdict(NestedNonMonotone()) == "refuses"


def test_late_resumption_interior_skip_refuses() -> None:
    assert _license_verdict(LateResumption()) == "refuses"


def test_no_multipass_layers_refuses(mlp_log: tl.Trace, tmp_path: Path) -> None:
    with pytest.raises(Exception) as excinfo:
        _draw(mlp_log, tmp_path, stack_by=True)
    assert excinfo.value.fields["code"] == "stack_by_auto_underivable"


# ---------------------------------------------------------------------------
# Auto annotation semantics + emission
# ---------------------------------------------------------------------------


def test_auto_stacks_passes_into_columns(lockstep_log: tl.Trace, tmp_path: Path) -> None:
    dot = _draw(lockstep_log, tmp_path, stack_by=True)
    state = _state(lockstep_log)
    groups = dict(state.stack_groups)
    # One rank group per execution window, both loop members in each.
    assert set(groups) == {"1", "2", "3"}
    for members in groups.values():
        assert len(members) >= 2
    assert "newrank=true" in dot
    assert dot.count("rank=same") == 3


def test_single_pass_stem_stays_unannotated(tmp_path: Path) -> None:
    log = _trace(StemmedLockstep())
    try:
        _draw(log, tmp_path, stack_by=True)
        stacked = {name for _, members in _state(log).stack_groups for name in members}
        # The stem renders but joins no rank group (classic diagrams draw
        # stems hanging free, not pinned to column 1).
        assert not any("linear_1_1" in name for name in stacked)
        assert stacked, "loop members must still stack"
    finally:
        log.cleanup()


def test_caption_and_legend_disclose_annotation(lockstep_log: tl.Trace, tmp_path: Path) -> None:
    dot = _draw(lockstep_log, tmp_path, stack_by=True)
    assert "stacked by: pass_index (auto)" in dot
    assert "cluster_torchlens_encoding_legend" in dot
    assert "stack_by: pass_index (auto)" in dot


def test_explicit_field_bypasses_license(chained_log: tl.Trace, tmp_path: Path) -> None:
    """Non-lockstep loops get their figure via explicit annotation."""

    dot = _draw(chained_log, tmp_path, stack_by="pass_index")
    state = _state(chained_log)
    assert state.stack_groups
    assert "stacked by: pass_index" in dot


def test_callable_bypasses_license_with_disclosure(chained_log: tl.Trace, tmp_path: Path) -> None:
    _draw(chained_log, tmp_path, stack_by=lambda node: int(node.pass_index))
    state = _state(chained_log)
    assert state.stack_groups
    assert any("callable" in note for note in state.stack_notes)


def test_callable_raise_chains_typed(lockstep_log: tl.Trace, tmp_path: Path) -> None:
    class Boom(RuntimeError):
        pass

    def source(node: Any) -> int:
        raise Boom("bad stack")

    with pytest.raises(Exception) as excinfo:
        _draw(lockstep_log, tmp_path, stack_by=source)
    assert excinfo.value.fields["code"] == "encoding_callable_error"
    assert isinstance(excinfo.value.__cause__, Boom)


def test_unhashable_value_refuses(lockstep_log: tl.Trace, tmp_path: Path) -> None:
    with pytest.raises(Exception) as excinfo:
        _draw(lockstep_log, tmp_path, stack_by=lambda node: [1, 2])
    assert excinfo.value.fields["code"] == "encoding_value_invalid"


def test_callable_none_leaves_node_unstacked(lockstep_log: tl.Trace, tmp_path: Path) -> None:
    _draw(lockstep_log, tmp_path, stack_by=lambda node: None)
    assert _state(lockstep_log).stack_groups == ()


# ---------------------------------------------------------------------------
# Strict opt-in + fences
# ---------------------------------------------------------------------------


def test_plain_draw_never_stacks(lockstep_log: tl.Trace, tmp_path: Path) -> None:
    dot = _draw(lockstep_log, tmp_path)
    assert "newrank" not in dot
    assert "stacked by" not in dot


def test_sibling_ordering_noops_while_stacking(lockstep_log: tl.Trace, tmp_path: Path) -> None:
    """Two independent rank-constraint systems must not fight (fence pin)."""

    _draw(lockstep_log, tmp_path, stack_by=True)
    decision = lockstep_log._last_sibling_ordering_decision
    assert decision.candidate_count == 0 and decision.survivor_count == 0


def test_explicit_rank_layout_refuses(lockstep_log: tl.Trace, tmp_path: Path) -> None:
    with pytest.raises(Exception) as excinfo:
        _draw(lockstep_log, tmp_path, stack_by=True, layout="rank")
    assert excinfo.value.fields["code"] == "encoding_requires_dot_layout"
    assert "stack_by" in str(excinfo.value)


def test_stack_composes_with_color(lockstep_log: tl.Trace, tmp_path: Path) -> None:
    # A VARYING color source: the lockstep toy's bytes are uniform, and a
    # degenerate (min == max) domain now honestly unencodes (themes memo
    # item 11), which would vacuously pass the composition half.
    _draw(lockstep_log, tmp_path, stack_by=True, color_by=lambda node: float(node.raw_index))
    state = _state(lockstep_log)
    assert state.stack_groups and state.colors
    assert state.active_channels() == ("color_by", "stack_by")


def test_request_hash_ignores_stack_fields() -> None:
    from torchlens.visualization.request import ResolvedRenderRequest

    base = ResolvedRenderRequest()
    stacked = ResolvedRenderRequest(stack_by=True, encoding=object())
    assert hash(base) == hash(stacked)


def test_rank_groups_ride_render_ir(lockstep_log: tl.Trace, tmp_path: Path) -> None:
    """The rank groups travel on RenderIR as a typed record (memo 4.1)."""

    from torchlens.visualization.render_ir import RenderIRRankGroup

    _draw(lockstep_log, tmp_path, stack_by=True)
    state = _state(lockstep_log)
    assert state.stack_groups
    group = RenderIRRankGroup(kind="stack", key="1", members=("a", "b"))
    assert group.kind == "stack"
