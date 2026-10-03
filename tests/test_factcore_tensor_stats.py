"""C02 TensorStats kernel + record + renderer pins (lovely items 2-5).

The two blocking round-1 numerical pins (offset cancellation, fp16
underflow), the N(1e6, 1) anti-regression golden, the sampler-aliasing
regression with its strided negative control, the end-bin invariant with a
failing-cap negative control, the exact integer route, magnitude labeling,
purity, the ``_version``-keyed cache, and the ascii==degrade byte contract.
"""

import math

import pytest
import torch

from torchlens.stats import (
    GLYPH_RAMP_ASCII,
    GLYPH_RAMP_UNICODE,
    TensorStats,
    degrade,
    format_sig,
    render_core_line,
    tensor_stats,
)
from torchlens.stats._stats_kernel import (
    HIST_BINS,
    SAMPLE_CAP,
    identity_seed,
    seeded_sample,
)
from torchlens.utils._torch_compat import get_cpu_half_kernels_support


def _oracle_moments(tensor: torch.Tensor) -> tuple[float, float]:
    """Two-pass float64 oracle: exact mean + population sd."""

    values = tensor.detach().reshape(-1).to(torch.float64)
    mean = float(values.mean())
    centered = values - mean
    return mean, float(((centered * centered).mean()) ** 0.5)


def _naive_variance(tensor: torch.Tensor) -> float:
    """The BANNED naive E[x^2]-E[x]^2 form (negative control only)."""

    values = tensor.detach().reshape(-1)
    return float((values * values).mean() - values.mean() ** 2)


def test_offset_cancellation_golden_n1e6_sd1() -> None:
    """N(1e6, 1): the sound kernel matches the oracle; naive f32 fails."""

    torch.manual_seed(0)
    tensor = (1e6 + torch.randn(100_000, dtype=torch.float64)).to(torch.float32)
    stats = tensor_stats(tensor)
    oracle_mean, oracle_sd = _oracle_moments(tensor)
    assert stats.mean == pytest.approx(oracle_mean, rel=1e-6)
    assert stats.sd == pytest.approx(oracle_sd, rel=1e-3)
    naive = _naive_variance(tensor)
    # The anti-regression: the banned form is catastrophically wrong here
    # (often NEGATIVE); any reintroduction flips the kernel assertion above.
    assert naive < 0 or abs(naive**0.5 - oracle_sd) / oracle_sd > 0.01


def test_offset_cancellation_golden_sd_1e_3_f64() -> None:
    """N(1e6, 1e-3) in f64: sound route survives; naive f64 does not."""

    torch.manual_seed(1)
    tensor = 1e6 + 1e-3 * torch.randn(65_536, dtype=torch.float64)
    stats = tensor_stats(tensor)
    oracle_mean, oracle_sd = _oracle_moments(tensor)
    assert stats.mean == pytest.approx(oracle_mean, rel=1e-9)
    assert stats.sd == pytest.approx(oracle_sd, rel=1e-3)
    naive = _naive_variance(tensor)
    assert naive < 0 or abs(naive**0.5 - oracle_sd) / oracle_sd > 0.01


@pytest.mark.skipif(
    not get_cpu_half_kernels_support(),
    reason="torch 2.1-2.2's CPU aminmax kernel does not cover float16 "
    "(aminmax_cpu not implemented for 'Half'), so the dense float kernel's "
    "leading nonfinite-proof reduction degrades tensor_stats to its honest "
    "unavailable shape (mean=None) instead of computing the pinned value",
)
def test_fp16_underflow_pin() -> None:
    """fp16 tiny values: chunked widening keeps the exact mean."""

    tensor = torch.full((50_000,), 1e-6, dtype=torch.float16)
    stats = tensor_stats(tensor)
    exact_value = float(tensor[0])
    assert exact_value > 0
    assert stats.mean == pytest.approx(exact_value, rel=1e-6)
    assert stats.sd == pytest.approx(0.0, abs=1e-12)
    torch.manual_seed(2)
    noisy = (1e-6 * (1 + 0.1 * torch.randn(50_000))).to(torch.float16)
    noisy_stats = tensor_stats(noisy)
    oracle_mean, oracle_sd = _oracle_moments(noisy)
    assert noisy_stats.mean == pytest.approx(oracle_mean, rel=1e-3)
    assert noisy_stats.sd == pytest.approx(oracle_sd, rel=1e-2)


def test_nonfinite_census_exact_and_finite_story_kept() -> None:
    """Exact NaN/+Inf/-Inf counts plus the surviving finite moments."""

    tensor = torch.tensor([1.0, -2.0, float("nan"), float("inf"), float("-inf"), 3.0, 0.0])
    stats = tensor_stats(tensor)
    assert (stats.nan_count, stats.posinf_count, stats.neginf_count) == (1, 1, 1)
    assert stats.finite_min == -2.0 and stats.finite_max == 3.0
    finite = torch.tensor([1.0, -2.0, 3.0, 0.0], dtype=torch.float64)
    assert stats.mean == pytest.approx(float(finite.mean()))
    assert stats.zero_count == 1


def test_sampler_aliasing_regression_with_strided_negative_control() -> None:
    """Gathered sample lands on iid theory; the strided slice FAILS (D21)."""

    period = 64
    n = 2**20
    k = 2**14
    values = torch.sin(2 * math.pi * (torch.arange(n, dtype=torch.float64) % period) / period)
    values = values.to(torch.float32)
    _, exact_sd = _oracle_moments(values)
    seed = identity_seed("aliasing-fixture", (n,), "torch.float32")
    gathered = seeded_sample(values, k, seed)
    gathered_sd = float(gathered.to(torch.float64).std(unbiased=False))
    se_sd = exact_sd / math.sqrt(2 * k)  # gaussian-ish bound; 4x slack below
    assert abs(gathered_sd - exact_sd) <= 4 * se_sd
    # Strided negative control: stride n//k is a multiple of the channel
    # period, so the slice hits ONE phase and its sd collapses.
    strided = values[:: n // k][:k]
    strided_sd = float(strided.to(torch.float64).std(unbiased=False))
    assert abs(strided_sd - exact_sd) > 4 * se_sd


@pytest.mark.smoke
def test_sampled_sd_above_gate_is_marked_and_bounded(monkeypatch) -> None:
    """Above the sd gate: sampled family marked, exact mean retained (D18/D19)."""

    import torchlens.stats._stats_kernel as kernel

    monkeypatch.setattr(kernel, "SD_EXACT_MAX", 2**12)
    torch.manual_seed(3)
    tensor = 5.0 + 2.0 * torch.randn(2**14)
    from torchlens.stats._tensor_stats import _record_from_kernel

    stats = _record_from_kernel(tensor, kernel.run_kernel(tensor, identity="gate-test"), None)
    assert stats.sd_evidence.sampled
    assert stats.sd_evidence.sample_size == min(SAMPLE_CAP, tensor.numel())
    assert stats.mean_evidence.policy == "exact"
    oracle_mean, oracle_sd = _oracle_moments(tensor)
    assert stats.mean == pytest.approx(oracle_mean, rel=1e-6)
    assert stats.sd_se is not None
    assert abs(stats.sd - oracle_sd) <= 6 * stats.sd_se


def test_end_bin_invariant_with_failing_control() -> None:
    """No bin the extremes prove occupied renders blank (D20)."""

    torch.manual_seed(4)
    core = torch.randn(5_000)
    tensor = torch.cat([core, torch.tensor([-100.0, 100.0, float("nan")])])
    stats = tensor_stats(tensor, identity="end-bin-fixture")
    assert stats.histogram_evidence.policy == "sampled"  # contaminated path
    assert stats.histogram_counts is not None
    assert stats.histogram_counts[0] >= 1
    assert stats.histogram_counts[-1] >= 1
    # Failing-cap negative control: a raw histogram of a with-replacement
    # sample under the same exact edges CAN blank an extreme-proven bin --
    # otherwise this invariant tests nothing. Find one seed where it does.
    flat = tensor[torch.isfinite(tensor)]
    saw_blank_end_bin = False
    for control_seed in range(24):
        sample = seeded_sample(flat, flat.numel(), control_seed)
        counts = torch.histc(sample, bins=HIST_BINS, min=-100.0, max=100.0)
        if int(counts[0]) == 0 or int(counts[-1]) == 0:
            saw_blank_end_bin = True
            break
    assert saw_blank_end_bin


def test_integer_route_exact_moments() -> None:
    """arange(8): exact rational mean 3.5, population sd 2.291 (D8/D27)."""

    tensor = torch.arange(8)
    stats = tensor_stats(tensor)
    assert stats.mean == pytest.approx(3.5)
    assert stats.sd == pytest.approx(math.sqrt(5.25), rel=1e-12)
    assert format_sig(stats.sd) == "2.291"
    assert stats.zero_count == 1
    assert stats.mean_evidence.policy == "exact"


def test_integer_overflow_guard_discloses_float_fallback() -> None:
    """Above the int64 guard the float fallback is DISCLOSED, never silent."""

    tensor = torch.tensor([2**31, -(2**31)] * 4, dtype=torch.int64)
    stats = tensor_stats(tensor)
    # f64 fallback: near-zero mean at 2e9 magnitude (3e-17 relative slack).
    assert stats.mean == pytest.approx(0.0, abs=1e-6)
    assert stats.mean_evidence.reason is not None
    assert "overflow guard" in stats.mean_evidence.reason


def test_bool_family_truth_rate_never_moments() -> None:
    """bool reports the truth rate; moments of a mask are noise (D12)."""

    tensor = torch.tensor([True, True, False, True])
    stats = tensor_stats(tensor)
    assert stats.true_count == 3
    assert stats.mean is None and stats.sd is None
    assert tensor_stats(torch.ones(4, dtype=torch.bool)).all_true
    assert tensor_stats(torch.zeros(4, dtype=torch.bool)).all_false


@pytest.mark.smoke
def test_complex_magnitude_labeled() -> None:
    """Complex moments are |z| statistics and say so (D12; lovely bug 3)."""

    torch.manual_seed(5)
    tensor = torch.complex(torch.randn(1_000), torch.randn(1_000))
    stats = tensor_stats(tensor)
    assert stats.magnitude_basis
    magnitude = tensor.abs().to(torch.float64)
    assert stats.mean == pytest.approx(float(magnitude.mean()), rel=1e-9)
    line = render_core_line(stats)
    assert "|mean|=" in line and "|sd|=" in line
    from torchlens.utils.display import tensor_stats_summary

    compat = tensor_stats_summary(tensor)
    assert "|mean|=" in compat and "neg=" not in compat


def test_kernel_purity_rng_grad_version() -> None:
    """The kernel never draws global RNG, mutates, or touches autograd."""

    base = torch.randn(64, requires_grad=True)
    tensor = base * 2
    rng_before = torch.get_rng_state()
    version_before = tensor._version
    stats = tensor_stats(tensor)
    assert stats.numel == 64
    assert torch.equal(rng_before, torch.get_rng_state())
    assert tensor._version == version_before
    assert tensor.grad_fn is not None
    assert base.grad is None


def test_version_keyed_cache_invalidates_on_mutation() -> None:
    """Cache serves one record per version; mutation recomputes (D30)."""

    tensor = torch.ones(32)
    first = tensor_stats(tensor)
    assert tensor_stats(tensor) is first
    tensor.add_(1.0)
    second = tensor_stats(tensor)
    assert second is not first
    assert second.constant_value == 2.0
    assert second.tensor_version != first.tensor_version


@pytest.mark.smoke
def test_never_raises_on_hostile_payloads() -> None:
    """Metadata-only is a first-class success; the kernel never raises."""

    hostile = [
        torch.empty(0),
        torch.tensor(8),
        torch.full((4,), float("nan")),
        torch.randn(2, 2).to_sparse(),
        torch.quantize_per_tensor(torch.randn(4), 0.1, 0, torch.qint8),
    ]
    for tensor in hostile:
        stats = tensor_stats(tensor)
        assert isinstance(stats, TensorStats)
        line = render_core_line(stats)
        assert isinstance(line, str) and line


def test_ascii_equals_degrade_unicode_byte_for_byte() -> None:
    """The two renders differ ONLY through the declared glyph table (D4)."""

    torch.manual_seed(6)
    specimens = [
        torch.randn(1, 64, 112, 112),
        torch.relu(torch.randn(1, 512, 7, 7)),
        torch.arange(8),
        torch.zeros(3, 4),
        torch.full((5, 5), 2.5),
        torch.cat([torch.randn(1_000), torch.tensor([float("nan")])]),
    ]
    for tensor in specimens:
        stats = tensor_stats(tensor)
        unicode_line = render_core_line(stats, style="unicode")
        ascii_line = render_core_line(stats, style="ascii")
        assert ascii_line == degrade(unicode_line)
        ascii_line.encode("ascii")  # ASCII canonical: must not raise


def test_glyph_table_is_nine_glyphs() -> None:
    """The package glyph table is the ramp and nothing else (D4)."""

    assert len(GLYPH_RAMP_ASCII) == 9
    assert len(GLYPH_RAMP_UNICODE) == 9
    assert GLYPH_RAMP_ASCII == " .:-=+*#%"


def test_width_degradation_order() -> None:
    """Width drops bytes, then n=, then sparkline -- never extrema/health (D14)."""

    torch.manual_seed(7)
    tensor = torch.cat([torch.randn(10_000), torch.tensor([float("inf")])]).reshape(73, 137)
    stats = tensor_stats(tensor)
    full = render_core_line(stats)
    assert "MiB" in full or "KiB" in full
    tight = render_core_line(stats, max_width=60)
    assert "KiB" not in tight and "MiB" not in tight
    for token in (stats.dtype, "+inf="):
        assert token in tight
    assert format_sig(stats.finite_min) in tight
    assert format_sig(stats.finite_max) in tight


def test_precision_law_formatter_pins() -> None:
    """D7: 4 sig digits, zeros KEPT, sci outside [1e-4, 1e4)."""

    assert format_sig(2.5) == "2.500"
    assert format_sig(1.1180339) == "1.118"
    assert format_sig(-2.0) == "-2.000"
    assert format_sig(0.0123456) == "0.01235"
    assert format_sig(12345.6) == "1.235e+04"
    assert format_sig(0.00005) == "5.000e-05"
    assert format_sig(0.0) == "0.000"


def test_special_case_lines() -> None:
    """empty / scalar / all_zero / constant / small-n inline forms (4.1)."""

    assert "empty" in render_core_line(tensor_stats(torch.empty(0)))
    assert "= 8" in render_core_line(tensor_stats(torch.tensor(8)))
    assert "all_zero" in render_core_line(tensor_stats(torch.zeros(3, 4)))
    assert "constant=2.500" in render_core_line(tensor_stats(torch.full((5, 5), 2.5)))
    line = render_core_line(tensor_stats(torch.arange(8)))
    assert "[0 1 2 3 4 5 6 7]" in line
    poisoned = render_core_line(tensor_stats(torch.full((4,), float("nan"))))
    assert "no_finite_values" in poisoned
    assert "nan=100%!" in poisoned


@pytest.mark.smoke
def test_compat_wrapper_preserves_field_layout() -> None:
    """tensor_stats_summary keeps its shape over the sound kernel."""

    from torchlens.utils.display import tensor_stats_summary

    line = tensor_stats_summary(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    assert line == (
        "Tensor[2, 2] float32 cpu mean=2.500 std=1.118 min=1.000 max=4.000 nan=0% inf=0% zero=0%"
    )
    assert "⚠" not in tensor_stats_summary(
        torch.tensor([1.0, float("nan")])
    )  # hazard marker is ASCII '!'
