"""F26: emission-view records, tag grammar, and the AMP closed-form oracle.

Covers memo 3.3-3.5 (grammar + tag safety + manifest) and 3.11 (the
bit-exact AMP correction, asserted with EQUALITY against the ``unscale_``
oracle, never allclose).
"""

from __future__ import annotations

import math

import pytest
import torch

import torchlens.trackers as trk
from torchlens._observability import (
    Histogram,
    HistogramDescriptor,
    ObservationRecord,
    RunningStats,
    StepBlockRecord,
)
from torchlens._observability._artifact import CommittedBlock
from torchlens._utils._torch_compat import HAS_AMP_GRADSCALER
from torchlens.trackers._errors import TagGrammarError, TrackersError

_requires_gradscaler = pytest.mark.skipif(
    not HAS_AMP_GRADSCALER,
    reason="torch.amp.GradScaler (device-agnostic) postdates the torch 2.1 floor",
)


def _spine_of(values: torch.Tensor):
    """Reduce one tensor through the C06 spine kernel."""

    kernel = RunningStats()
    kernel.update(values)
    return kernel.compute()


def _sketch_of(values: torch.Tensor, descriptor: HistogramDescriptor | None = None):
    """Reduce one tensor through the C06 histogram kernel."""

    kernel = Histogram(descriptor or HistogramDescriptor())
    kernel.update(values)
    return kernel.result()


class TestTagGrammar:
    """Memo 3.4: family first, statistic second, verbatim leaf last."""

    # Kept by the shrink-r4 coverage proof: sole branch coverage of trackers/_records.py.
    def test_empty_component_refuses(self) -> None:
        """Empty or dot-only path components refuse (grouping hazard)."""

        with pytest.raises(TagGrammarError):
            trk.assert_tag_safe("gradients//w")


class TestSpineScalars:
    """Memo 3.3: the portable scalar series derived from the spine."""

    # Kept by the shrink-r4 coverage proof: sole branch coverage of _observability/_kernels.py.
    def test_matches_torch_reference(self) -> None:
        """mean/std/norm/min/max/count agree with direct torch reductions."""

        values = torch.randn(128, dtype=torch.float64)
        scalars = trk.running_stats_scalars(_spine_of(values))
        assert scalars["count"] == 128.0
        assert math.isclose(scalars["mean"], values.mean().item(), rel_tol=1e-9)
        assert math.isclose(scalars["norm"], values.norm().item(), rel_tol=1e-9)
        assert math.isclose(scalars["min"], values.min().item(), rel_tol=1e-12)
        assert math.isclose(scalars["max"], values.max().item(), rel_tol=1e-12)
        assert math.isclose(
            scalars["std"],
            values.std(correction=0).item(),
            rel_tol=1e-9,
        )


class TestHistogramPoints:
    """The signed-log2 render: exact counts, explicit edges, honest bands."""

    # Kept by the shrink-r4 coverage proof: sole branch coverage of trackers/_records.py.
    def test_geometry_and_mass(self) -> None:
        """2n+1 bins, 2n+2 edges; in-grid + zero/underflow mass conserved."""

        descriptor = HistogramDescriptor(lo_exp=-4, hi_exp=4, bins_per_octave=1)
        values = torch.tensor([0.0, 0.5, -0.5, 3.0, -3.0, 1e-9])
        counts, edges = trk.histogram_points(_sketch_of(values, descriptor))
        n_side = descriptor.bins_per_side
        assert len(counts) == 2 * n_side + 1
        assert len(edges) == len(counts) + 1
        # Center band covers (-2^lo, +2^lo) and holds zero + underflow.
        assert edges[n_side] == -(2.0**-4)
        assert edges[n_side + 1] == 2.0**-4
        assert counts[n_side] == 2  # the exact zero and the 1e-9 underflow
        assert sum(counts) == 6


class TestAmpCorrection:
    """Memo 3.11: closed form, BIT-exact against the unscale_ oracle."""

    # Kept by the shrink-r4 coverage proof: sole branch coverage of trackers/_amp.py.
    def test_histogram_correction_exact_bin_shift(self) -> None:
        """Power-of-two scales shift the log2 grid by an exact bin offset."""

        grads = torch.tensor([0.5, 1.0, 2.0, -4.0], dtype=torch.float64)
        scale = 4.0
        descriptor = HistogramDescriptor(lo_exp=-8, hi_exp=8, bins_per_octave=2)
        corrected = trk.correct_histogram(_sketch_of(grads * scale, descriptor), scale)
        oracle = _sketch_of(grads, descriptor)
        assert corrected.pos_counts == oracle.pos_counts
        assert corrected.neg_counts == oracle.neg_counts
        assert corrected.specials == oracle.specials

    # Kept by the shrink-r4 coverage proof: sole branch coverage of _observability/_kernels.py, trackers/_amp.py.
    def test_underflow_spills_into_specials(self) -> None:
        """Counts shifted below the grid land in the underflow specials."""

        descriptor = HistogramDescriptor(lo_exp=-2, hi_exp=2, bins_per_octave=1)
        values = torch.tensor([0.5], dtype=torch.float64)  # bin near the floor
        corrected = trk.correct_histogram(_sketch_of(values, descriptor), 16.0)
        assert corrected.specials.get("pos_underflow", 0) >= 1
        assert sum(corrected.pos_counts) == 0

    # Kept by the shrink-r4 coverage proof: sole branch coverage of trackers/_amp.py.
    def test_non_power_of_two_refuses(self) -> None:
        """Rebinning is banned: a foreign scale refuses typed."""

        with pytest.raises(TrackersError) as info:
            trk.correct_histogram(_sketch_of(torch.randn(8)), 3.0)
        assert info.value.fields["code"] == "tracker_scale_invalid"

    # Kept by the shrink-r4 coverage proof: sole branch coverage of trackers/_amp.py.
    @_requires_gradscaler
    def test_disabled_scaler_is_observed_unscaled(self) -> None:
        """A disabled scaler is an observed fact, not an assumption."""

        scaler = torch.amp.GradScaler("cpu", enabled=False)
        assert trk.observed_grad_scale(scaler) == (1.0, "unscaled_observed")
        assert trk.observed_grad_scale(None) == (None, "unavailable")


class TestEmissionFromBlock:
    """Missing is never zero: non-observed presences emit nothing."""

    # Kept by the shrink-r4 coverage proof: sole branch coverage of _observability/_schema.py, trackers/_records.py.
    def test_non_observed_emits_no_points(self) -> None:
        """A budget_dropped observation contributes zero series points."""

        block = CommittedBlock(
            block=StepBlockRecord(segment_id="s", global_step=7, provenance="explicit"),
            observations=(
                ObservationRecord(
                    global_step=7,
                    site_id="w",
                    stream="param",
                    phase="pre_step",
                    presence="budget_dropped",
                    reason="cap",
                ),
            ),
            step_lo=7,
            step_hi=7,
        )
        emission = trk.emission_from_block(block, {}, trk.TagGrammar())
        data_scalars = [p for p in emission.scalars if not p.tag.startswith("torchlens/run/")]
        assert data_scalars == []
        assert emission.histograms == ()
