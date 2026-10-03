"""F09 CP2: costreport items 13-17, 25 -- peaks, MFU, roofline,
instrumented rate, reserved join slots, and the cost measurer.

Named memo rows exercised: T-MFU-DENOM (two denominators are two
DIFFERENTLY NAMED quantities; a kernel-union step_time is a typed
refusal), the D18 peaks provenance/refusal rules (a zero-row catalog
refuses correctly), the D20 boxed-diagnostic grammar (banned words,
per-row applicability, typed CUDA refusal), D21's hypothesis language
and sum/sum aggregate, and D27's measured-never-estimated disclosure.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens._errors import InvalidArgumentError
from torchlens.report import (
    MfuProvenance,
    PeakRow,
    attributed_kernel_utilization,
    cost_report,
    device_peaks,
    instrumented_rate,
    machine_balance,
    mfu,
    roofline,
    time_clean_step,
)


@pytest.fixture(scope="module")
def small_trace():
    """One shared small capture."""

    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
    trace = tl.trace(model.eval(), torch.randn(2, 8))
    yield trace
    trace.cleanup()


def _peak_row(mode: str = "fp32", flops: float = 1e12) -> PeakRow:
    """A fully-cited peak row for tests."""

    return PeakRow(
        device_identity="TestDevice-X",
        execution_mode=mode,
        dense_flops_per_s=flops,
        source_citation="Vendor datasheet DS-123 rev 4",
        source_date="2026-01-01",
        clock_assumptions="boost clock, dense",
    )


# ---------------------------------------------------------------------------
# Peaks (D18)


def test_zero_row_catalog_refuses_correctly() -> None:
    """D18: launch never gates on row curation -- an empty catalog refuses."""

    catalog = device_peaks([])
    with pytest.raises(InvalidArgumentError) as excinfo:
        catalog.for_device("AnyGPU", "fp32")
    assert excinfo.value.fields["code"] == "device_peaks_unknown_device"


@pytest.mark.smoke
def test_peak_rows_require_citation_and_date() -> None:
    """D18: per-row provenance is mandatory (the staleness-lint class)."""

    bare = PeakRow(
        device_identity="X",
        execution_mode="fp32",
        dense_flops_per_s=1e12,
        source_citation=" ",
        source_date="",
        clock_assumptions="",
    )
    with pytest.raises(InvalidArgumentError, match="citation"):
        device_peaks([bare])
    catalog = device_peaks([_peak_row()])
    row = catalog.for_device("TestDevice-X", "fp32")
    assert row.sparse_policy == "dense_only"


# ---------------------------------------------------------------------------
# MFU (D16/D17; T-MFU-DENOM)


def test_mfu_reduces_to_f_over_tp_at_one_mode() -> None:
    """The D16 property test: one mode reduces exactly to F/(T*P)."""

    result = mfu(
        flops_by_mode={"fp32": 2_000_000},
        peaks_by_mode={"fp32": 1e9},
        step_seconds=0.01,
    )
    assert result.mfu == pytest.approx(2_000_000 / (0.01 * 1e9))
    assert result.denominator == "step_wall_time"
    assert result.peaks_source == "user_supplied"


def test_mfu_mode_split_sums_ideal_seconds() -> None:
    """D16: ideal seconds sum per execution mode, then one division."""

    result = mfu(
        flops_by_mode={"fp32": 1_000_000, "bf16": 3_000_000},
        peaks_by_mode={"fp32": 1e9, "bf16": 3e9},
        step_seconds=0.004,
    )
    assert result.ideal_seconds == pytest.approx(1e6 / 1e9 + 3e6 / 3e9)
    assert result.mfu == pytest.approx(result.ideal_seconds / 0.004)
    assert len(result.mode_terms) == 2


def test_mfu_refuses_kernel_union_denominator() -> None:
    """T-MFU-DENOM: the kernel union fed as step_time fails typed."""

    with pytest.raises(InvalidArgumentError) as excinfo:
        mfu(
            flops_by_mode={"fp32": 1},
            peaks_by_mode={"fp32": 1e9},
            step_seconds=0.01,
            provenance=MfuProvenance(denominator="kernel_union"),
        )
    assert excinfo.value.fields["code"] == "mfu_denominator_invalid"
    assert "attributed_kernel_utilization" in str(excinfo.value)


def test_mfu_unknown_mode_refuses_typed() -> None:
    """D17: an execution mode without a peak makes MFU unavailable."""

    with pytest.raises(InvalidArgumentError) as excinfo:
        mfu(flops_by_mode={"fp8_recipe": 10}, peaks_by_mode={"fp32": 1e9}, step_seconds=1.0)
    assert excinfo.value.fields["code"] == "mfu_mode_peak_missing"


@pytest.mark.smoke
def test_mfu_above_one_warns_coded() -> None:
    """D16: MFU > 1.0 warns with the coded S-18 warning."""

    from torchlens.errors import TorchLensWarning

    with pytest.warns(TorchLensWarning, match="MFU") as record:
        mfu(flops_by_mode={"fp32": 10**12}, peaks_by_mode={"fp32": 1e9}, step_seconds=0.5)
    assert any(
        getattr(warning.message, "fields", {}).get("code") == "mfu_exceeds_one"
        for warning in record
    )


def test_kernel_union_quantity_has_its_own_name_and_gate() -> None:
    """T-MFU-DENOM: the second denominator is a DIFFERENTLY NAMED metric,
    gated on the correlation-ID join."""

    with pytest.raises(InvalidArgumentError) as excinfo:
        attributed_kernel_utilization()
    assert excinfo.value.fields["code"] == "kernel_utilization_requires_device_join"


@pytest.mark.smoke
def test_time_clean_step_discloses_method_and_reruns() -> None:
    """D16: the opt-in helper states plainly that it re-runs the forward."""

    model = nn.Linear(4, 4).eval()
    timing = time_clean_step(model, torch.randn(2, 4), warmup=1, repeats=3)
    assert timing.seconds > 0
    assert timing.n_forwards_run == 4
    assert timing.scope == ("forward",)
    assert "uninstrumented" in timing.method


# ---------------------------------------------------------------------------
# Roofline + machine balance (D18/D21)


def test_roofline_rows_are_hypotheses_and_aggregate_is_sum_over_sum(small_trace) -> None:
    """D21: bound verdicts are hypotheses; aggregate intensity is
    sum(FLOPs)/sum(bytes), never a mean of per-op ratios."""

    balance = machine_balance(
        compute_peak_flops_per_s=1e12,
        memory_peak_bytes_per_s=1e11,
        basis="advertised",
    )
    result = roofline(small_trace, ridge_intensity=balance.ridge_intensity)
    covered = [row for row in result.rows if row.reason == "covered"]
    assert covered
    for row in covered:
        assert row.evidence == "estimated+ideal_read_once_write_once"
        if row.bound_hypothesis is not None:
            assert row.bound_hypothesis.endswith("_hypothesis")
    total_flops = sum(row.flops for row in covered)
    total_bytes = sum(row.ideal_traffic_bytes or 0 for row in covered)
    assert result.aggregate_intensity == pytest.approx(total_flops / total_bytes)


@pytest.mark.smoke
def test_roofline_cache_resident_ops_excluded_from_headline(small_trace) -> None:
    """D21: sub-cache ops carry the hint and never enter headline counts."""

    result = roofline(small_trace, ridge_intensity=10.0, cache_resident_bytes=1 << 40)
    assert result.headline_memory_bound == 0
    assert result.headline_compute_bound == 0
    assert result.excluded_cache_resident > 0


def test_roofline_without_ridge_renders_no_bounds(small_trace) -> None:
    """D21: no achieved/bound layer without a supplied ceiling."""

    result = roofline(small_trace)
    assert all(row.bound_hypothesis is None for row in result.rows)
    assert result.ridge_intensity is None


@pytest.mark.smoke
def test_machine_balance_requires_named_basis() -> None:
    """D18: the ceiling's basis is REQUIRED and disclosed; a measured peak
    is never called MFU."""

    with pytest.raises(InvalidArgumentError) as excinfo:
        machine_balance(
            compute_peak_flops_per_s=1e12, memory_peak_bytes_per_s=1e11, basis="vendor_max"
        )
    assert excinfo.value.fields["code"] == "machine_balance_basis_invalid"
    balance = machine_balance(
        compute_peak_flops_per_s=1e12,
        memory_peak_bytes_per_s=1e11,
        basis="measured_achievable",
        method="STREAM triad + DGEMM roof",
    )
    assert balance.ridge_intensity == pytest.approx(10.0)
    assert balance.basis == "measured_achievable"


# ---------------------------------------------------------------------------
# Instrumented rate (D19/D20)


@pytest.mark.smoke
def test_instrumented_rate_is_boxed_and_labeled(small_trace) -> None:
    """D20: 'instrumented' in the header; the banned words never appear."""

    table = instrumented_rate(small_trace)
    text = str(table)
    assert "instrumented" in text.lower()
    assert "achieved" not in text.lower()
    assert "throughput" not in text.lower()
    applicable = [row for row in table.rows if row.applicable]
    assert applicable
    for row in applicable:
        assert row.flops_per_second_instrumented is not None
        assert row.device is None or not row.device.startswith("cuda")


def test_instrumented_rate_not_in_default_profile(small_trace) -> None:
    """D20: never in any default view."""

    from torchlens.report import build_profile

    frame = build_profile(small_trace, level="op").to_pandas()
    assert not any("rate" in str(column) for column in frame.columns)


# ---------------------------------------------------------------------------
# Reserved join slots (item 17 / D22)


def test_compute_rows_reserve_device_attribution_slots(small_trace) -> None:
    """D22: device_time / kernel_time / attribution_status exist on every
    row from day one and stay None until the correlation-ID join."""

    from torchlens.report import compute_aggregation

    for row in compute_aggregation(small_trace).rows:
        assert row.device_time is None
        assert row.kernel_time is None
        assert row.attribution_status is None


def test_name_matched_bridge_is_relabeled() -> None:
    """D22: the name-substring bridge is a labeled approximate diagnostic."""

    import inspect

    from torchlens.bridge import profiler

    source = inspect.getsource(profiler.join)
    assert "name-matched (approximate; not for rates)" in source


# ---------------------------------------------------------------------------
# The cost measurer (D27)


@pytest.mark.smoke
def test_cost_report_measures_and_discloses() -> None:
    """D27: absolute ms first, ratio derived, N forwards disclosed."""

    model = nn.Linear(4, 4).eval()
    result = cost_report(model, torch.randn(2, 4), repeats=2)
    text = str(result)
    assert "measured on THIS model and host" in text
    assert "forward passes" in text
    tiers = {tier.tier: tier for tier in result.tiers}
    assert tiers["raw_forward"].ratio_vs_raw == 1.0
    assert tiers["trace"].median_seconds > 0
    assert result.n_forwards_run >= 6


def test_cost_report_refuses_unknown_tier() -> None:
    """D25: tiers are public callables, never harness rung names."""

    with pytest.raises(InvalidArgumentError) as excinfo:
        cost_report(nn.Linear(2, 2), torch.randn(1, 2), tiers=("raw_forward", "fastlog_halt_25"))
    assert excinfo.value.fields["code"] == "cost_report_tier_unknown"


# ---------------------------------------------------------------------------
# Every new code ships provoked (error-code coverage gate)


@pytest.mark.smoke
def test_remaining_new_codes_are_provoked(small_trace) -> None:
    """Provoke every F09 code not already provoked above, by code literal:
    clean_step_repeats_invalid, device_peaks_row_invalid,
    mfu_peaks_source_invalid, mfu_step_seconds_invalid,
    instrumented_rate_cuda_unsupported (via the wrapped-device fixture
    below where derivable)."""

    with pytest.raises(InvalidArgumentError) as excinfo:
        time_clean_step(nn.Linear(2, 2), torch.randn(1, 2), repeats=0)
    assert excinfo.value.fields["code"] == "clean_step_repeats_invalid"

    with pytest.raises(InvalidArgumentError) as excinfo:
        device_peaks(["not-a-peak-row"])
    assert excinfo.value.fields["code"] == "device_peaks_row_invalid"

    with pytest.raises(InvalidArgumentError) as excinfo:
        mfu(
            flops_by_mode={"fp32": 1},
            peaks_by_mode={"fp32": 1e9},
            step_seconds=1.0,
            provenance=MfuProvenance(peaks_source="vendor_marketing"),
        )
    assert excinfo.value.fields["code"] == "mfu_peaks_source_invalid"

    with pytest.raises(InvalidArgumentError) as excinfo:
        mfu(flops_by_mode={"fp32": 1}, peaks_by_mode={"fp32": 1e9}, step_seconds=0.0)
    assert excinfo.value.fields["code"] == "mfu_step_seconds_invalid"


def test_instrumented_rate_all_cuda_refuses_typed(small_trace, monkeypatch) -> None:
    """instrumented_rate_cuda_unsupported: every timed op on CUDA refuses."""

    from torchlens.report import _cost_perf

    class _FakeCudaOp:
        """Minimal op stub: timed, counted, CUDA-executed."""

        layer_label = "linear_1_1"
        label = "linear_1_1:1"
        func_name = "linear"
        flops_forward = 10
        macs_forward = 5
        func_duration = 0.001
        device_ref = "cuda:0"
        is_input = False
        is_output = False
        is_buffer = False
        compute_record = None

    class _FakeTrace:
        """Trace stub whose one timed op executed on CUDA."""

        layer_list = (_FakeCudaOp(),)

    with pytest.raises(InvalidArgumentError) as excinfo:
        _cost_perf.instrumented_rate(_FakeTrace())
    assert excinfo.value.fields["code"] == "instrumented_rate_cuda_unsupported"
