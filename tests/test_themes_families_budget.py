"""F12 pins: N16 view-aware source families and the N10 budget resolver.

Covers the memo's composition-test rows 5 (the resolver lands the band,
discloses the dial by name, deterministic) and 11 (the view x source
pairing four-cell matrix), plus the zero-coverage refusal in the F12 row
gate.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.visualization import lenses
from torchlens.visualization.lenses import _resolve as resolve_module


@pytest.fixture(scope="module")
def mlp_log() -> Any:
    """Small MLP trace shared by the family cells."""

    class Toy(nn.Module):
        """Two-linear toy."""

        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(8, 8)
            self.fc2 = nn.Linear(8, 4)

        def forward(self, x: Any) -> Any:
            return self.fc2(torch.relu(self.fc1(x)))

    log = tl.trace(Toy(), torch.randn(2, 8))
    yield log
    log.cleanup()


def test_family_table_is_closed_and_binds_per_view() -> None:
    """The four view x member cells resolve exactly per N16."""

    time_unrolled = lenses.resolve_source_family("time", "unrolled")
    assert time_unrolled.member == "func_duration"
    assert time_unrolled.aggregation_line is None
    time_rolled = lenses.resolve_source_family("time", "rolled")
    assert time_rolled.member == "total_func_duration"
    assert time_rolled.aggregation_line is not None
    assert "total across passes" in time_rolled.aggregation_line
    assert lenses.resolve_source_family("bytes", "rolled").member == "total_activation_memory"
    assert lenses.resolve_source_family("flops", "unrolled").member == "flops_forward"


def test_unknown_family_and_view_refuse_typed() -> None:
    """Closed-vocabulary refusals carry stable codes."""

    with pytest.raises(Exception) as excinfo:
        lenses.resolve_source_family("watts", "rolled")
    assert excinfo.value.fields["code"] == "lens_source_family_unknown"
    with pytest.raises(Exception) as excinfo:
        lenses.resolve_source_family("time", "sideways")
    assert excinfo.value.fields["code"] == "lens_view_invalid"


def test_speed_resolves_per_pass_member_on_unrolled(mlp_log: Any) -> None:
    """Unrolled speed binds func_duration with the rank transform."""

    resolution = lenses.resolve_lens(mlp_log, "speed")
    assert resolution.source is not None
    assert resolution.source.member == "func_duration"
    channel = resolution.draw_kwargs["color_by"]
    assert channel.transform == "rank"
    assert any(line.startswith("coverage: encoded") for line in resolution.disclosure)


@pytest.mark.smoke
def test_speed_resolves_summed_member_on_rolled(mlp_log: Any) -> None:
    """Rolled speed binds total_func_duration WITH the aggregation line."""

    resolution = lenses.resolve_lens(mlp_log, "speed", {"vis_mode": "rolled"})
    assert resolution.source is not None
    assert resolution.source.member == "total_func_duration"
    assert any("total across passes" in line for line in resolution.disclosure)


def test_zero_coverage_refuses_typed(mlp_log: Any, monkeypatch: Any) -> None:
    """ROW GATE: zero resolved coverage refuses; never a populated legend
    over an unencoded graph."""

    from torchlens.visualization.lenses import _families

    broken = dict(_families.SOURCE_FAMILIES)
    broken["time"] = _families.SourceFamily(
        name="time",
        per_pass_member="nonexistent_member_zzz",
        summed_member="nonexistent_member_total_zzz",
        unit_wording="",
        capture_remedy="re-capture with a plain tl.trace forward",
    )
    monkeypatch.setattr(_families, "SOURCE_FAMILIES", broken)
    with pytest.raises(Exception) as excinfo:
        lenses.resolve_lens(mlp_log, "speed")
    assert excinfo.value.fields["code"] == "lens_headline_evidence_missing"
    assert "re-capture" in str(excinfo.value)


@pytest.mark.smoke
def test_one_view_zero_coverage_names_the_zero_coverage_code(
    mlp_log: Any, monkeypatch: Any
) -> None:
    """When only the view-resolved member is unencoded (the other view has
    evidence), the refusal is the ADVERTISED-SCALE code, not the headline gap."""

    from torchlens.visualization.lenses import _families

    broken = dict(_families.SOURCE_FAMILIES)
    broken["time"] = _families.SourceFamily(
        name="time",
        per_pass_member="nonexistent_member_zzz",
        summed_member="total_func_duration",
        unit_wording="",
        capture_remedy="re-capture with a plain tl.trace forward",
    )
    monkeypatch.setattr(_families, "SOURCE_FAMILIES", broken)
    with pytest.raises(Exception) as excinfo:
        lenses.resolve_lens(mlp_log, "speed")
    assert excinfo.value.fields["code"] == "lens_source_zero_coverage"
    assert "re-capture" in str(excinfo.value)


def test_explicit_color_by_wins_over_family(mlp_log: Any) -> None:
    """Defaults-not-overrides: the user's channel beats the family binding."""

    resolution = lenses.resolve_lens(mlp_log, "speed", {"color_by": "activation_memory"})
    assert resolution.draw_kwargs["color_by"] == "activation_memory"


def test_source_coverage_counts_op_population(mlp_log: Any) -> None:
    """Coverage counts the op population, encoded and total."""

    coverage = lenses.source_coverage(mlp_log, "func_duration")
    assert coverage.total == len(mlp_log.ops)
    assert coverage.encoded == coverage.total
    assert coverage.fraction == 1.0


# ---------------------------------------------------------------------------
# N10 budget resolver.
# ---------------------------------------------------------------------------


def test_small_trace_renders_in_full(mlp_log: Any) -> None:
    """Below the band ceiling collapse='none' wins and is disclosed by name."""

    budget = lenses.resolve_budget(mlp_log)
    assert budget.dial == 'collapse="none"'
    assert budget.in_band
    assert budget.draw_kwargs == {"collapse": "none"}
    assert any("detail dial" in line for line in budget.disclosure)


@pytest.mark.heavy
def test_budget_search_is_deterministic_and_disclosed() -> None:
    """A >220-op trace searches the float schedule; same inputs, same dial."""

    class Block(nn.Module):
        """Ten-ish ops per block."""

        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(8, 8)

        def forward(self, x: Any) -> Any:
            h = torch.relu(self.fc(x))
            h = torch.sigmoid(h) + h
            h = torch.tanh(h) * h
            return h + 1.0

    model = nn.Sequential(*[Block() for _ in range(40)])
    log = tl.trace(model, torch.randn(2, 8))
    try:
        assert len(log.ops) > lenses.BAND_CEILING
        first = lenses.resolve_budget(log)
        second = lenses.resolve_budget(log)
        assert first.dial == second.dial
        assert first.visible_count == second.visible_count
        assert any("detail dial" in line for line in first.disclosure)
        # The dial must genuinely compact: never the full-graph count.
        assert first.visible_count < len(log.ops)
    finally:
        log.cleanup()


@pytest.mark.smoke
def test_overview_budget_rides_resolution(mlp_log: Any) -> None:
    """Overview (collapse='auto') routes through the resolver and discloses."""

    resolution = lenses.resolve_lens(mlp_log, "overview")
    assert resolution.budget is not None
    assert any("detail dial" in line for line in resolution.disclosure)


@pytest.mark.smoke
def test_band_missed_warns_coded_and_discloses(mlp_log: Any, monkeypatch: Any) -> None:
    """No schedule point in the band: nearest point served with the coded warning."""

    import warnings as warnings_module

    from torchlens.visualization.lenses import _budget

    # An impossible band (floor > ceiling admits no count) forces the miss path.
    monkeypatch.setattr(_budget, "BAND_FLOOR", 1)
    monkeypatch.setattr(_budget, "BAND_CEILING", 0)
    with warnings_module.catch_warnings(record=True) as caught:
        warnings_module.simplefilter("always")
        budget = lenses.resolve_budget(mlp_log)
    codes = [getattr(w.message, "fields", {}).get("code") for w in caught]
    assert "lens_budget_band_missed" in codes
    assert not budget.in_band
    assert any("band missed" in line for line in budget.disclosure)


@pytest.mark.smoke
def test_above_ceiling_fallback_warns_coded(mlp_log: Any, monkeypatch: Any) -> None:
    """Above the optimizer ceiling with no in-band depth: estimate served, coded."""

    import warnings as warnings_module

    from torchlens.visualization import collapse_optimizer

    # A tiny trace over a lowered ceiling: no call depth reaches the 100-220
    # band, so the nearest-depth estimate is served with the coded warning.
    monkeypatch.setattr(collapse_optimizer, "COLLAPSE_OPTIMIZER_MAX_OPS", 1)
    with warnings_module.catch_warnings(record=True) as caught:
        warnings_module.simplefilter("always")
        budget = lenses.resolve_budget(mlp_log)
    codes = [getattr(w.message, "fields", {}).get("code") for w in caught]
    assert "lens_budget_above_ceiling_fallback" in codes
    assert not budget.in_band
    assert "vis_call_depth" in budget.dial
    assert any("estimate" in line for line in budget.disclosure)


def test_band_constants_are_the_memo_band() -> None:
    """Two-sided band: floor 100 / target 160 / ceiling 220; coverage 0.80."""

    assert (lenses.BAND_FLOOR, lenses.BAND_TARGET, lenses.BAND_CEILING) == (100, 160, 220)
    assert lenses.COVERAGE_FLOOR == 0.80


def test_legend_dependence_gate_mechanism(mlp_log: Any, monkeypatch: Any) -> None:
    """A gate-ruled legend-DEPENDENT lens refuses explicit suppression."""

    monkeypatch.setattr(resolve_module, "LEGEND_DEPENDENT_LENSES", {"speed"})
    with pytest.raises(Exception) as excinfo:
        lenses.resolve_lens(mlp_log, "speed", {"show_legend": False})
    assert excinfo.value.fields["code"] == "lens_legend_suppression_refused"
    # Honored silently on a legend-optional lens (documented explicit value).
    resolution = lenses.resolve_lens(mlp_log, "overview", {"show_legend": False})
    assert resolution.draw_kwargs["show_legend"] is False
