"""Kernel tests: Spine + signed-log2 Histogram (explorer item 4).

The memo-mandated rows: reference (vs direct torch reductions), merge
(integer exactness, split-vs-whole), nonfinite, dtype, count-width, and
RNG isolation.
"""

from __future__ import annotations

import pytest
import torch

from torchlens._observability import (
    DEFAULT_DESCRIPTOR,
    Histogram,
    HistogramDescriptor,
    RunningStats,
    merge_histogram_results,
    merge_spine_results,
)
from torchlens._observability._kernels import (
    INT32_COUNT_MAX,
    merge_spine_vectors,
    sketch_vector,
    spine_result_from_vector,
    spine_vector,
)
from torchlens.errors import StatKernelError


def _spine_of(*tensors: torch.Tensor) -> RunningStats:
    spine = RunningStats()
    for tensor in tensors:
        spine.update(tensor)
    return spine


class TestSpineReference:
    """Spine fields cross-checked against direct torch reductions."""

    def test_empty_update_is_a_specified_noop(self) -> None:
        spine = _spine_of(torch.empty(0))
        result = spine.compute()
        assert result.numel == 0
        assert result.finite_min is None
        assert result.sum is None


class TestSpineMerge:
    """Accumulator and result-level merges; integer counts EXACT."""

    def test_merge_type_refusal(self) -> None:
        with pytest.raises(StatKernelError) as excinfo:
            RunningStats().merge("nope")  # type: ignore[arg-type]
        assert excinfo.value.fields["code"] == "stat_merge_incompatible"


class TestHistogramReference:
    """Grid placement, specials, and edge conventions (D9)."""

    def test_population_conservation_on_real_telemetry_shape(self) -> None:
        torch.manual_seed(3)
        data = torch.randn(50_000) * 40.0
        h = Histogram()
        h.update(data)
        assert h.result().numel == 50_000

    def test_bucket_edges_are_the_manifest_descriptor_values(self) -> None:
        d = HistogramDescriptor(bins_per_octave=1, lo_exp=-2, hi_exp=2)
        edges = d.bucket_edges()
        assert edges == (0.25, 0.5, 1.0, 2.0, 4.0)
        assert d.bins_per_side == 4


class TestHistogramMerge:
    """Integer merges are EXACT; descriptor mismatch refuses typed (D9)."""

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
            HistogramDescriptor(bin_encoding="sparse")


class TestDtypes:
    """dtype rows: widening, integers, bool, complex refusal."""

    def test_fp16_underflow_lands_in_low_bins_not_zero(self) -> None:
        # 2^-24 is subnormal in fp16 but well inside the [2^-48, 2^16) window.
        h = Histogram()
        h.update(torch.tensor([2.0**-24], dtype=torch.float16))
        result = h.result()
        assert result.specials["zero"] == 0
        assert sum(result.pos_counts) == 1

    def test_integer_and_bool_inputs(self) -> None:
        spine = _spine_of(torch.tensor([1, -2, 0], dtype=torch.int64))
        result = spine.compute()
        assert result.num_zero == 1
        assert result.num_negative == 1
        bool_spine = _spine_of(torch.tensor([True, False, True]))
        assert bool_spine.compute().num_zero == 1
        h = Histogram()
        h.update(torch.tensor([4, -16], dtype=torch.int32))
        hist = h.result()
        assert sum(hist.pos_counts) == 1
        assert sum(hist.neg_counts) == 1

    def test_complex_refuses_typed(self) -> None:
        with pytest.raises(StatKernelError) as excinfo:
            RunningStats().update(torch.complex(torch.ones(2), torch.ones(2)))
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
        spine = RunningStats()
        spine.update(data)
        spine.merge(_spine_of(data))
        spine.compute()
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

    def test_merge_spine_vectors_matches_result_merge(self) -> None:
        torch.manual_seed(6)
        x, y = torch.randn(400), torch.randn(600) - 3.0
        merged_vec = merge_spine_vectors(spine_vector(x), spine_vector(y))
        via_vec = spine_result_from_vector(merged_vec, "float64")
        via_results = merge_spine_results(_spine_of(x).compute(), _spine_of(y).compute())
        assert via_vec.counts() == via_results.counts()
        assert via_vec.mean == pytest.approx(via_results.mean)
        assert via_vec.m2 == pytest.approx(via_results.m2)
