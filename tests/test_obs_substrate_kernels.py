"""Kernel tests: Spine + signed-log2 Histogram (explorer item 4).

The memo-mandated rows: reference (vs direct torch reductions), merge
(integer exactness, split-vs-whole), nonfinite, dtype, count-width, and
RNG isolation.
"""

from __future__ import annotations

import math

import pytest
import torch

from torchlens.observability import (
    DEFAULT_DESCRIPTOR,
    Histogram,
    HistogramDescriptor,
    Spine,
    StatKernelError,
    merge_histogram_results,
    merge_spine_results,
)
from torchlens.observability._kernels import (
    INT32_COUNT_MAX,
    histogram_result_from_vector,
    merge_spine_vectors,
    sketch_vector,
    spine_result_from_vector,
    spine_vector,
)

pytestmark = pytest.mark.smoke


def _spine_of(*tensors: torch.Tensor) -> Spine:
    spine = Spine()
    for tensor in tensors:
        spine.update(tensor)
    return spine


class TestSpineReference:
    """Spine fields cross-checked against direct torch reductions."""

    def test_counts_and_extrema_match_torch(self) -> None:
        torch.manual_seed(0)
        data = torch.randn(1000)
        data[3] = 0.0
        result = _spine_of(data).result()
        assert result.count_total == 1000
        assert result.count_finite == 1000
        assert result.count_zero == int((data == 0).sum())
        assert result.count_negative == int((data < 0).sum())
        assert result.finite_min == pytest.approx(float(data.min()), rel=0, abs=0)
        assert result.finite_max == pytest.approx(float(data.max()), rel=0, abs=0)
        assert result.finite_absmax == pytest.approx(float(data.abs().max()))
        assert result.sum == pytest.approx(float(data.to(torch.float64).sum()))
        assert result.sum_squares == pytest.approx(float((data.to(torch.float64) ** 2).sum()))
        assert result.sum_abs == pytest.approx(float(data.abs().to(torch.float64).sum()))
        assert result.mean == pytest.approx(float(data.to(torch.float64).mean()))

    def test_moments_are_stable_not_naive(self) -> None:
        # The classic cancellation case: large offset, small variance. The
        # naive E[x^2]-E[x]^2 form loses every significant digit here.
        data = torch.randn(10_000, dtype=torch.float64) + 1.0e8
        result = _spine_of(data).result()
        expected_var = float(data.var(unbiased=False))
        assert result.m2 is not None
        assert result.m2 / result.count_finite == pytest.approx(expected_var, rel=1e-6)

    def test_empty_update_is_a_specified_noop(self) -> None:
        spine = _spine_of(torch.empty(0))
        result = spine.result()
        assert result.count_total == 0
        assert result.finite_min is None
        assert result.sum is None

    def test_batch_split_equals_whole_for_counts(self) -> None:
        torch.manual_seed(1)
        data = torch.randn(999)
        whole = _spine_of(data).result()
        split = _spine_of(data[:100], data[100:]).result()
        assert split.counts() == whole.counts()
        assert split.finite_min == whole.finite_min
        assert split.finite_max == whole.finite_max
        # Floating sums are documented approximate across regroupings.
        assert split.sum == pytest.approx(whole.sum)
        assert split.mean == pytest.approx(whole.mean)
        assert split.m2 == pytest.approx(whole.m2)


class TestSpineMerge:
    """Accumulator and result-level merges; integer counts EXACT."""

    def test_merge_integer_counts_exact(self) -> None:
        a = _spine_of(torch.tensor([1.0, float("nan"), -3.0, 0.0]))
        b = _spine_of(torch.tensor([float("inf"), 2.0]))
        a.merge(b)
        result = a.result()
        assert result.count_total == 6
        assert result.count_nan == 1
        assert result.count_posinf == 1
        assert result.count_zero == 1
        assert result.count_negative == 1

    def test_result_level_merge_matches_accumulator_merge(self) -> None:
        torch.manual_seed(2)
        x, y = torch.randn(500), torch.randn(300) + 2.0
        acc = _spine_of(x)
        acc.merge(_spine_of(y))
        via_acc = acc.result()
        via_results = merge_spine_results(_spine_of(x).result(), _spine_of(y).result())
        assert via_results.counts() == via_acc.counts()
        assert via_results.mean == pytest.approx(via_acc.mean)
        assert via_results.m2 == pytest.approx(via_acc.m2)

    def test_merge_type_refusal(self) -> None:
        with pytest.raises(StatKernelError) as excinfo:
            Spine().merge("nope")  # type: ignore[arg-type]
        assert excinfo.value.fields["code"] == "stat_merge_incompatible"


class TestHistogramReference:
    """Grid placement, specials, and edge conventions (D9)."""

    def test_exact_power_bin_placement(self) -> None:
        h = Histogram()
        h.update(torch.tensor([1.0]))  # log2 = 0 -> bin (0 - lo_exp) * bpo
        result = h.result()
        d = result.descriptor
        expected_bin = (0 - d.lo_exp) * d.bins_per_octave
        assert result.pos_counts[expected_bin] == 1
        assert sum(result.pos_counts) == 1

    def test_window_boundaries(self) -> None:
        d = DEFAULT_DESCRIPTOR
        h = Histogram()
        h.update(
            torch.tensor(
                [2.0**d.lo_exp, 2.0**d.hi_exp, 2.0 ** (d.lo_exp - 1), -(2.0 ** (d.hi_exp + 1))]
            )
        )
        result = h.result()
        assert result.pos_counts[0] == 1  # exactly 2^lo lands in the first bin
        assert result.specials["pos_overflow"] == 1  # 2^hi is outside [lo, hi)
        assert result.specials["pos_underflow"] == 1
        assert result.specials["neg_overflow"] == 1

    def test_specials_are_exact(self) -> None:
        h = Histogram()
        h.update(torch.tensor([0.0, 0.0, float("nan"), float("inf"), -float("inf"), 1.0, -1.0]))
        result = h.result()
        assert result.specials["zero"] == 2
        assert result.specials["nan"] == 1
        assert result.specials["posinf"] == 1
        assert result.specials["neginf"] == 1
        assert sum(result.pos_counts) == 1
        assert sum(result.neg_counts) == 1
        assert result.count_total == 7

    def test_signed_sides_are_independent(self) -> None:
        h = Histogram()
        h.update(torch.tensor([0.5, -8.0]))
        result = h.result()
        assert sum(result.pos_counts) == 1
        assert sum(result.neg_counts) == 1

    def test_population_conservation_on_real_telemetry_shape(self) -> None:
        torch.manual_seed(3)
        data = torch.randn(50_000) * 40.0
        h = Histogram()
        h.update(data)
        assert h.result().count_total == 50_000

    def test_bucket_edges_are_the_manifest_descriptor_values(self) -> None:
        d = HistogramDescriptor(bins_per_octave=1, lo_exp=-2, hi_exp=2)
        edges = d.bucket_edges()
        assert edges == (0.25, 0.5, 1.0, 2.0, 4.0)
        assert d.bins_per_side == 4


class TestHistogramMerge:
    """Integer merges are EXACT; descriptor mismatch refuses typed (D9)."""

    def test_split_vs_whole_is_bit_identical(self) -> None:
        torch.manual_seed(4)
        data = torch.randn(10_000)
        whole = Histogram()
        whole.update(data)
        parts = Histogram()
        first, second = Histogram(), Histogram()
        first.update(data[:3000])
        second.update(data[3000:])
        parts.merge(first)
        parts.merge(second)
        assert parts.result() == whole.result()

    def test_result_level_merge_matches(self) -> None:
        a, b = Histogram(), Histogram()
        a.update(torch.tensor([1.0, -2.0]))
        b.update(torch.tensor([4.0, 0.0]))
        merged = merge_histogram_results(a.result(), b.result())
        direct = Histogram()
        direct.update(torch.tensor([1.0, -2.0, 4.0, 0.0]))
        assert merged == direct.result()

    def test_descriptor_mismatch_refuses_typed(self) -> None:
        a = Histogram()
        b = Histogram(HistogramDescriptor(bins_per_octave=1))
        with pytest.raises(StatKernelError) as excinfo:
            a.merge(b)
        assert excinfo.value.fields["code"] == "sketch_descriptor_mismatch"
        with pytest.raises(StatKernelError) as excinfo2:
            merge_histogram_results(a.result(), b.result())
        assert excinfo2.value.fields["code"] == "sketch_descriptor_mismatch"

    def test_descriptor_geometry_refusals(self) -> None:
        with pytest.raises(StatKernelError) as excinfo:
            HistogramDescriptor(base=10)
        assert excinfo.value.fields["code"] == "sketch_descriptor_invalid"
        with pytest.raises(StatKernelError):
            HistogramDescriptor(lo_exp=4, hi_exp=4)
        with pytest.raises(StatKernelError):
            HistogramDescriptor(encoding="sparse")


class TestDtypes:
    """dtype rows: widening, integers, bool, complex refusal."""

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_half_widen_records_fp32_or_wider(self, dtype: torch.dtype) -> None:
        spine = _spine_of(torch.ones(8, dtype=dtype) * 0.5)
        result = spine.result()
        assert result.reduction_dtype in ("float32", "float64")
        assert result.sum == pytest.approx(4.0)

    def test_fp16_underflow_lands_in_low_bins_not_zero(self) -> None:
        # 2^-24 is subnormal in fp16 but well inside the [2^-48, 2^16) window.
        h = Histogram()
        h.update(torch.tensor([2.0**-24], dtype=torch.float16))
        result = h.result()
        assert result.specials["zero"] == 0
        assert sum(result.pos_counts) == 1

    def test_integer_and_bool_inputs(self) -> None:
        spine = _spine_of(torch.tensor([1, -2, 0], dtype=torch.int64))
        result = spine.result()
        assert result.count_zero == 1
        assert result.count_negative == 1
        bool_spine = _spine_of(torch.tensor([True, False, True]))
        assert bool_spine.result().count_zero == 1
        h = Histogram()
        h.update(torch.tensor([4, -16], dtype=torch.int32))
        hist = h.result()
        assert sum(hist.pos_counts) == 1
        assert sum(hist.neg_counts) == 1

    def test_complex_refuses_typed(self) -> None:
        with pytest.raises(StatKernelError) as excinfo:
            Spine().update(torch.complex(torch.ones(2), torch.ones(2)))
        assert excinfo.value.fields["code"] == "stat_dtype_unsupported"
        with pytest.raises(StatKernelError) as excinfo2:
            Histogram().update(torch.complex(torch.ones(2), torch.ones(2)))
        assert excinfo2.value.fields["code"] == "stat_dtype_unsupported"


class TestCountWidth:
    """Count-width row: int64 accumulators, int32-fit disclosure (D9)."""

    def test_counts_accumulate_in_int64(self) -> None:
        h = Histogram()
        assert h._pos.dtype == torch.int64
        assert h._neg.dtype == torch.int64

    def test_int32_fit_helper(self) -> None:
        h = Histogram()
        h.update(torch.ones(10))
        result = h.result()
        assert result.counts_fit_int32()
        oversized = result.specials.copy()
        oversized["zero"] = INT32_COUNT_MAX + 1
        from dataclasses import replace

        assert not replace(result, specials=oversized).counts_fit_int32()


class TestRngIsolation:
    """Kernels never consume model or global RNG."""

    def test_global_rng_state_bitwise_unchanged(self) -> None:
        torch.manual_seed(1234)
        before = torch.get_rng_state().clone()
        data = torch.arange(-100.0, 100.0)
        spine = Spine()
        spine.update(data)
        spine.merge(_spine_of(data))
        spine.result()
        h = Histogram()
        h.update(data)
        h.merge(Histogram())
        h.result()
        spine_vector(data)
        sketch_vector(data, DEFAULT_DESCRIPTOR)
        after = torch.get_rng_state()
        assert torch.equal(before, after)


class TestVectorPathParity:
    """The staged device-vector path agrees with the accumulator path."""

    def test_spine_vector_matches_accumulator(self) -> None:
        torch.manual_seed(5)
        data = torch.randn(2048)
        data[7] = float("nan")
        data[9] = float("inf")
        data[11] = 0.0
        via_vector = spine_result_from_vector(spine_vector(data), "float64")
        via_acc = _spine_of(data).result()
        assert via_vector.counts() == via_acc.counts()
        assert via_vector.finite_min == pytest.approx(via_acc.finite_min)
        assert via_vector.sum == pytest.approx(via_acc.sum)
        assert via_vector.m2 == pytest.approx(via_acc.m2)

    def test_merge_spine_vectors_matches_result_merge(self) -> None:
        torch.manual_seed(6)
        x, y = torch.randn(400), torch.randn(600) - 3.0
        merged_vec = merge_spine_vectors(spine_vector(x), spine_vector(y))
        via_vec = spine_result_from_vector(merged_vec, "float64")
        via_results = merge_spine_results(_spine_of(x).result(), _spine_of(y).result())
        assert via_vec.counts() == via_results.counts()
        assert via_vec.mean == pytest.approx(via_results.mean)
        assert via_vec.m2 == pytest.approx(via_results.m2)

    def test_sketch_vector_matches_accumulator(self) -> None:
        torch.manual_seed(7)
        data = torch.randn(4096) * 100
        data[0] = 0.0
        data[1] = float("nan")
        via_vector = histogram_result_from_vector(
            sketch_vector(data, DEFAULT_DESCRIPTOR), DEFAULT_DESCRIPTOR
        )
        acc = Histogram()
        acc.update(data)
        assert via_vector == acc.result()

    def test_quantile_sanity_from_grid(self) -> None:
        # The derived-quantile basis: cumulative counts over the fixed grid
        # place the median of a known population inside the right octave.
        data = torch.full((1000,), 3.0)
        result = histogram_result_from_vector(
            sketch_vector(data, DEFAULT_DESCRIPTOR), DEFAULT_DESCRIPTOR
        )
        d = result.descriptor
        bin_index = int(math.floor((math.log2(3.0) - d.lo_exp) * d.bins_per_octave))
        assert result.pos_counts[bin_index] == 1000
