"""F26: emission-view records, tag grammar, and the AMP closed-form oracle.

Covers memo 3.3-3.5 (grammar + tag safety + manifest) and 3.11 (the
bit-exact AMP correction, asserted with EQUALITY against the ``unscale_``
oracle, never allclose).
"""

from __future__ import annotations

import json
import math

import pytest
import torch

import torchlens.trackers as trk
from torchlens.observability import (
    Histogram,
    HistogramDescriptor,
    ObservationRecord,
    RunRecord,
    SiteRecord,
    Spine,
    StepBlockRecord,
)
from torchlens.observability._artifact import CommittedBlock
from torchlens.trackers._errors import TagGrammarError, TrackersError

pytestmark = pytest.mark.smoke


def _spine_of(values: torch.Tensor):
    """Reduce one tensor through the C06 spine kernel."""

    kernel = Spine()
    kernel.update(values)
    return kernel.result()


def _sketch_of(values: torch.Tensor, descriptor: HistogramDescriptor | None = None):
    """Reduce one tensor through the C06 histogram kernel."""

    kernel = Histogram(descriptor or HistogramDescriptor())
    kernel.update(values)
    return kernel.result()


class TestTagGrammar:
    """Memo 3.4: family first, statistic second, verbatim leaf last."""

    def test_data_tag_shape(self) -> None:
        """The three-level grammar renders exactly family/statistic/leaf."""

        grammar = trk.TagGrammar()
        assert grammar.data("gradients", "norm", "h.0.attn.weight") == (
            "gradients/norm/h.0.attn.weight"
        )

    def test_name_inserts_one_component(self) -> None:
        """name= lands after the family (multi-model runs)."""

        grammar = trk.TagGrammar(name="ema")
        assert grammar.data("gradients", "norm", "w") == "gradients/ema/norm/w"
        # The reserved torchlens root never takes the name component.
        assert grammar.run_health("last_step") == "torchlens/run/last_step"

    def test_namespace_reroots_everything(self) -> None:
        """namespace= is the collision remedy: every tag re-roots."""

        grammar = trk.TagGrammar(namespace="torchlens")
        assert grammar.data("gradients", "norm", "w") == "torchlens/gradients/norm/w"
        assert grammar.check("dead_layer") == "torchlens/torchlens/check/dead_layer"

    def test_unsafe_tag_refuses_and_names(self) -> None:
        """D4 majority: refuse-and-name, never silently rewrite."""

        with pytest.raises(TagGrammarError) as info:
            trk.assert_tag_safe("gradients/norm/bad\x01leaf", site="bad leaf")
        assert info.value.fields["code"] == "tracker_tag_unsafe"
        assert "bad leaf" in str(info.value.fields.get("site"))

    def test_empty_component_refuses(self) -> None:
        """Empty or dot-only path components refuse (grouping hazard)."""

        with pytest.raises(TagGrammarError):
            trk.assert_tag_safe("gradients//w")

    def test_real_parameter_names_are_safe(self) -> None:
        """The measured basis: real module/param paths pass unmodified."""

        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.ReLU())
        for name, _ in model.named_parameters():
            trk.assert_tag_safe(f"gradients/norm/{name}")


class TestSpineScalars:
    """Memo 3.3: the portable scalar series derived from the spine."""

    def test_matches_torch_reference(self) -> None:
        """mean/std/norm/min/max/count agree with direct torch reductions."""

        values = torch.randn(128, dtype=torch.float64)
        scalars = trk.spine_scalars(_spine_of(values))
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

    def test_nonfinite_excluded_and_counted(self) -> None:
        """NaN/inf never poison the finite statistics; they count separately."""

        values = torch.tensor([1.0, float("nan"), float("inf"), -2.0])
        spine = _spine_of(values)
        scalars = trk.spine_scalars(spine)
        assert scalars["count"] == 2.0
        assert scalars["max"] == 1.0
        nonfinite = trk._records.nonfinite_scalars(spine)
        assert nonfinite == {"nan": 1.0, "posinf": 1.0, "neginf": 0.0}


class TestHistogramPoints:
    """The signed-log2 render: exact counts, explicit edges, honest bands."""

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

    def test_edges_ascend(self) -> None:
        """The full signed edge array is strictly ascending."""

        _counts, edges = trk.histogram_points(_sketch_of(torch.randn(64)))
        assert all(a < b for a, b in zip(edges, edges[1:], strict=False))


class TestAmpCorrection:
    """Memo 3.11: closed form, BIT-exact against the unscale_ oracle."""

    def test_spine_correction_bit_exact(self) -> None:
        """Corrected scaled-tensor spine EQUALS the unscaled tensor's spine."""

        grads = torch.randn(256, dtype=torch.float64)
        scale = 65536.0  # a real GradScaler power-of-two scale
        corrected = trk.correct_spine(_spine_of(grads * scale), scale)
        oracle = _spine_of(grads)
        assert corrected.mean == oracle.mean
        assert corrected.finite_min == oracle.finite_min
        assert corrected.finite_max == oracle.finite_max
        assert corrected.sum == oracle.sum
        assert corrected.sum_squares == oracle.sum_squares
        assert corrected.count_finite == oracle.count_finite

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

    def test_underflow_spills_into_specials(self) -> None:
        """Counts shifted below the grid land in the underflow specials."""

        descriptor = HistogramDescriptor(lo_exp=-2, hi_exp=2, bins_per_octave=1)
        values = torch.tensor([0.5], dtype=torch.float64)  # bin near the floor
        corrected = trk.correct_histogram(_sketch_of(values, descriptor), 16.0)
        assert corrected.specials.get("pos_underflow", 0) >= 1
        assert sum(corrected.pos_counts) == 0

    def test_non_power_of_two_refuses(self) -> None:
        """Rebinning is banned: a foreign scale refuses typed."""

        with pytest.raises(TrackersError) as info:
            trk.correct_histogram(_sketch_of(torch.randn(8)), 3.0)
        assert info.value.fields["code"] == "tracker_scale_invalid"

    def test_gradscaler_cycle_observed(self) -> None:
        """The full CPU GradScaler cycle: observed scale is the real factor."""

        scaler = torch.amp.GradScaler("cpu", init_scale=65536.0)
        scale, evidence = trk.observed_grad_scale(scaler)
        assert scale == 65536.0
        assert evidence == "scaled_unknown_factor"
        model = torch.nn.Linear(4, 4)
        loss = scaler.scale(model(torch.randn(2, 4)).sum())
        loss.backward()
        grad = next(model.parameters()).grad
        assert grad is not None
        scaled_spine = _spine_of(grad.detach().to(torch.float64))
        corrected = trk.correct_spine(scaled_spine, scale)
        # The unscale_ oracle: let the scaler unscale in place, re-reduce.
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        scaler.unscale_(optimizer)
        oracle = _spine_of(grad.detach().to(torch.float64))
        assert corrected.sum == oracle.sum
        assert corrected.finite_absmax == oracle.finite_absmax

    def test_disabled_scaler_is_observed_unscaled(self) -> None:
        """A disabled scaler is an observed fact, not an assumption."""

        scaler = torch.amp.GradScaler("cpu", enabled=False)
        assert trk.observed_grad_scale(scaler) == (1.0, "unscaled_observed")
        assert trk.observed_grad_scale(None) == (None, "unavailable")


class TestManifest:
    """Memo 3.5: the versioned series manifest + architecture fingerprint."""

    def _site(self, site_id: str, shape: tuple[int, ...]) -> SiteRecord:
        return SiteRecord(
            site_id=site_id,
            kind="param",
            display_label=site_id,
            param_name=site_id,
            shape=shape,
            dtype="float32",
            numel=int(torch.tensor(shape).prod()),
        )

    def test_manifest_payload(self) -> None:
        """The manifest carries identity, edge policy, and per-series rows."""

        run = RunRecord(run_id="r", segment_id="s")
        sites = {"w": self._site("w", (2, 2)), "b": self._site("b", (2,))}
        point = trk.build_manifest(run, sites, trk.TagGrammar())
        assert point.tag == "torchlens/meta/manifest"
        payload = json.loads(point.text)
        assert payload["kind"] == "torchlens.trackers.manifest"
        assert payload["edge_policy"]["base"] == 2
        assert len(payload["series"]) == 2
        assert payload["architecture_fingerprint"]

    def test_fingerprint_detects_architecture_change(self) -> None:
        """Different site censuses -> different fingerprints (3.5)."""

        a = {"w": self._site("w", (2, 2))}
        b = {"w": self._site("w", (2, 2)), "extra": self._site("extra", (3,))}
        assert trk.architecture_fingerprint(a) != trk.architecture_fingerprint(b)
        assert trk.architecture_fingerprint(a) == trk.architecture_fingerprint(dict(a))


class TestEmissionFromBlock:
    """Missing is never zero: non-observed presences emit nothing."""

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
