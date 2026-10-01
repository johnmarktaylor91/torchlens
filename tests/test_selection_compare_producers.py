"""Comparative selection producers: acceptance and honesty gates.

Covers the comparative-producer lane obligations: differential producers
(``changed`` / ``top_changed``) compare the SUBJECT (resolution trace)
against ONE explicit reference with a directional float64 delta, masks are
exact AS SETS against manually computed criteria, structure mismatches
refuse typed and never silently intersect (missing sites, unsaved payloads,
shape drift, structural-site-key disagreement, self-comparison), and
cross-pass producers (``stable_across_passes`` / ``pass_variance``) make
pass-qualified claims with the two-pass honesty floor, explicit windows,
intersection element-populations, and masks landing on every window pass.
Provenance relations are PINNED (every comparative claim is ``exact`` — a
statistic of complete retained evidence), NaN never satisfies a criterion,
rank tie-breaks are deterministic, and the end-to-end ``do()`` gallery rows
run (ablate the top movers; ablate a stable unit at every pass).
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.selection import SelectionError

_RELU = "relu_1_2"
_TANH = "tanh_1_2"


class _Probe(nn.Module):
    """CNN with an engineered input-invariant channel.

    encoder[0] channel 2 has zero weights and bias 0.7, so its ReLU output is
    the constant 0.7 on ANY input — the pinned "did not change when the input
    changed" control for the differential producers.
    """

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(1, 4, 3), nn.ReLU())
        self.head = nn.Conv2d(4, 2, 3)
        with torch.no_grad():
            self.encoder[0].weight[2] = 0.0
            self.encoder[0].bias[2] = 0.7

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.head(self.encoder(x)))


class _Recurrent(nn.Module):
    """Three-pass recurrence with an engineered pass-invariant unit.

    cell weight row 0 is zero with bias 0.5, so unit 0's tanh output is the
    constant tanh(0.5) at EVERY pass — the pinned "stable across timesteps"
    control for the cross-pass producers.
    """

    def __init__(self) -> None:
        super().__init__()
        self.cell = nn.Linear(4, 4)
        with torch.no_grad():
            self.cell.weight[0] = 0.0
            self.cell.bias[0] = 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(3):
            x = torch.tanh(self.cell(x))
        return x


@pytest.fixture(scope="module")
def model():
    """One shared deterministic probe model."""

    torch.manual_seed(0)
    return _Probe()


@pytest.fixture(scope="module")
def log(model):
    """SUBJECT capture (input A)."""

    torch.manual_seed(1)
    trace = tl.trace(model, torch.randn(1, 1, 10, 10))
    try:
        yield trace
    finally:
        trace.cleanup()


@pytest.fixture(scope="module")
def log_b(model):
    """REFERENCE capture (input B, same model)."""

    torch.manual_seed(2)
    trace = tl.trace(model, torch.randn(1, 1, 10, 10))
    try:
        yield trace
    finally:
        trace.cleanup()


@pytest.fixture(scope="module")
def rec_log():
    """Recurrent capture with three tanh passes."""

    torch.manual_seed(3)
    trace = tl.trace(_Recurrent(), torch.randn(1, 4))
    try:
        yield trace
    finally:
        trace.cleanup()


def _delta(log, log_b, site: str) -> torch.Tensor:
    """Manual subject-minus-reference delta in float64."""

    return log[site].out.to(torch.float64) - log_b[site].out.to(torch.float64)


def _relations(resolved):
    return [entry.provenance.relation for entry in resolved]


# ---------------------------------------------------------------------------
# Differential producers: masks are exact as sets against manual deltas.
# ---------------------------------------------------------------------------


def test_changed_default_is_the_moved_elements_mask(log, log_b):
    """Bare changed(ref) selects exactly the elements that moved at all."""

    resolved = tl.changed(log_b, _RELU).resolve(log)
    delta = _delta(log, log_b, _RELU)
    assert torch.equal(resolved[0].mask, delta.abs() > 0)
    assert _relations(resolved) == ["exact"]
    # the engineered input-invariant channel did not move
    assert not resolved[0].mask[0, 2].any()
    assert resolved[0].mask.any()


def test_changed_bounds_band_and_signed_direction(log, log_b):
    """Strict bounds on |delta|; signed selects increased/decreased elements."""

    delta = _delta(log, log_b, _RELU)
    above = tl.changed(log_b, _RELU, above=0.1).resolve(log)
    assert torch.equal(above[0].mask, delta.abs() > 0.1)
    band = tl.changed(log_b, _RELU, above=0.1, below=0.5).resolve(log)
    assert torch.equal(band[0].mask, (delta.abs() > 0.1) & (delta.abs() < 0.5))
    increased = tl.changed(log_b, _RELU, above=0.0, by="signed").resolve(log)
    assert torch.equal(increased[0].mask, delta > 0)
    decreased = tl.changed(log_b, _RELU, below=0.0, by="signed").resolve(log)
    assert torch.equal(decreased[0].mask, delta < 0)


def test_changed_selects_what_the_intervention_moved(model):
    """The intervention-effect mask: fork vs original after do()."""

    trace = tl.trace(
        model,
        torch.randn(1, 1, 10, 10),
        capture=tl.options.CaptureOptions(intervention_ready=True),
    )
    try:
        target = tl.units(_RELU, [(0, 0, 1, 1), (0, 1, 2, 2)])
        fork = trace.fork()
        fork.do(target.resolve(fork), tl.zero_ablate())
        moved = tl.changed(trace, _RELU).resolve(fork)
        expected = fork[_RELU].out.to(torch.float64) != trace[_RELU].out.to(torch.float64)
        assert torch.equal(moved[0].mask, expected)
        # only the ablated units (where the baseline was nonzero) moved
        assert bool(moved[0].mask.sum() <= 2)
    finally:
        trace.cleanup()


def test_top_changed_matches_manual_ranking(log, log_b):
    """top_changed selects exactly the k largest |delta| elements."""

    resolved = tl.top_changed(log_b, _RELU, 7).resolve(log)
    keys = _delta(log, log_b, _RELU).abs().reshape(-1)
    expected = torch.zeros_like(keys, dtype=torch.bool)
    expected[torch.argsort(keys, descending=True, stable=True)[:7]] = True
    assert torch.equal(resolved[0].mask.reshape(-1), expected)
    assert resolved[0].selected_count == 7
    assert _relations(resolved) == ["exact"]


def test_top_changed_fraction_least_moved_and_signed(log, log_b):
    """fraction=ceil rule; largest=False ranks the least moved; signed ranks deltas."""

    numel = log[_RELU].out.numel()
    fractional = tl.top_changed(log_b, _RELU, fraction=0.01).resolve(log)
    assert fractional[0].selected_count == math.ceil(0.01 * numel)
    least = tl.top_changed(log_b, _RELU, 4, largest=False).resolve(log)
    abs_keys = _delta(log, log_b, _RELU).abs().reshape(-1)
    expected_least = torch.zeros_like(abs_keys, dtype=torch.bool)
    expected_least[torch.argsort(abs_keys, stable=True)[:4]] = True
    assert torch.equal(least[0].mask.reshape(-1), expected_least)
    assert (abs_keys[least[0].mask.reshape(-1)] == 0).all()  # zero-delta ties win
    keys = _delta(log, log_b, _RELU).reshape(-1)
    signed = tl.top_changed(log_b, _RELU, 3, by="signed").resolve(log)
    expected = torch.zeros_like(keys, dtype=torch.bool)
    expected[torch.argsort(keys, descending=True, stable=True)[:3]] = True
    assert torch.equal(signed[0].mask.reshape(-1), expected)


def test_top_changed_population_too_small_refuses_typed(log, log_b):
    """k beyond the rankable delta population refuses with structured counts."""

    numel = log[_RELU].out.numel()
    with pytest.raises(SelectionError) as excinfo:
        tl.top_changed(log_b, _RELU, numel + 1).resolve(log)
    assert excinfo.value.fields["code"] == "selection_unresolvable"
    assert excinfo.value.fields["reason"] == "population_too_small"
    assert excinfo.value.fields["available"] == numel


def test_global_default_population_spans_saved_sites(log, log_b):
    """within=None ranks deltas across every retained tensor site."""

    resolved = tl.top_changed(log_b, None, 10).resolve(log)
    assert len(resolved) > 1
    assert sum(entry.selected_count for entry in resolved) == 10


def test_changed_nan_delta_never_satisfies():
    """Elements with a NaN on either side never satisfy or rank."""

    class _NaNs(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.sqrt(x)

    subject = tl.trace(_NaNs(), torch.tensor([[-1.0, 4.0, 9.0, 16.0]]))
    reference = tl.trace(_NaNs(), torch.tensor([[1.0, 1.0, -1.0, 4.0]]))
    try:
        site = "sqrt_1_1"
        nan_mask = torch.isnan(subject[site].out) | torch.isnan(reference[site].out)
        assert nan_mask.any()
        wide = tl.changed(reference, site, above=-1e30, by="signed").resolve(subject)
        assert not (wide[0].mask & nan_mask).any()
        ranked = tl.top_changed(reference, site, 2).resolve(subject)
        assert not (ranked[0].mask & nan_mask).any()
        with pytest.raises(SelectionError) as excinfo:
            tl.top_changed(reference, site, 3).resolve(subject)  # only 2 rankable
        assert excinfo.value.fields["reason"] == "population_too_small"
    finally:
        subject.cleanup()
        reference.cleanup()


def test_signed_complex_refuses_and_abs_works():
    """Complex deltas have no total order; magnitude comparison is legal."""

    class _Complex(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.fft.fft(x)

    subject = tl.trace(_Complex(), torch.randn(1, 8))
    reference = tl.trace(_Complex(), torch.randn(1, 8))
    try:
        site = "fft_1_1"
        with pytest.raises(SelectionError) as excinfo:
            tl.changed(reference, site, above=0.0, by="signed").resolve(subject)
        assert excinfo.value.fields["reason"] == "value_criterion_invalid"
        resolved = tl.changed(reference, site).resolve(subject)  # by='abs' default
        expected = (
            subject[site].out.to(torch.complex128) - reference[site].out.to(torch.complex128)
        ).abs() > 0
        assert torch.equal(resolved[0].mask, expected)
    finally:
        subject.cleanup()
        reference.cleanup()


# ---------------------------------------------------------------------------
# Differential structure honesty: refuse, never silently intersect.
# ---------------------------------------------------------------------------


def test_reference_must_be_one_trace(log, log_b):
    """Non-trace and evidence-set references refuse at construction, teaching."""

    with pytest.raises(ValueError, match="ONE reference trace"):
        tl.changed([log, log_b])
    with pytest.raises(ValueError, match="must be a Trace"):
        tl.changed("not a trace")


def test_self_comparison_refuses_typed(log):
    """Resolving against the reference itself is vacuous and refuses, teaching."""

    with pytest.raises(SelectionError) as excinfo:
        tl.changed(log, _RELU).resolve(log)
    assert excinfo.value.fields["reason"] == "value_criterion_invalid"
    assert "fork is a different trace object" in str(excinfo.value)


def test_reference_missing_site_refuses_named(log):
    """A reference lacking a population site refuses as a one-sided join key.

    Since leverage B4 the pairing authority is the guarded site join: a site
    the reference never executed is a DECLARED subject-only addition
    (``site_join_refused`` / ``one_sided_subject``), never a guess.
    """

    other = tl.trace(nn.Linear(4, 2), torch.randn(1, 4))
    try:
        with pytest.raises(SelectionError) as excinfo:
            tl.changed(other, _RELU).resolve(log)
        assert excinfo.value.fields["reason"] == "site_join_refused"
        assert excinfo.value.fields["join_verdict"] == "one_sided_subject"
    finally:
        other.cleanup()


def test_reference_shape_drift_refuses_named(model, log):
    """A shape-mismatched reference refuses mask_shape_mismatch, named."""

    other = tl.trace(model, torch.randn(1, 1, 12, 12))
    try:
        with pytest.raises(SelectionError) as excinfo:
            tl.changed(other, _RELU).resolve(log)
        assert excinfo.value.fields["reason"] == "mask_shape_mismatch"
        assert excinfo.value.fields["sample"] == "reference"
    finally:
        other.cleanup()


def test_reference_unsaved_payload_refuses(model, log):
    """A reference without the retained payload refuses value_not_saved."""

    other = tl.trace(model, torch.randn(1, 1, 10, 10), save=tl.func("conv2d"))
    try:
        with pytest.raises(SelectionError) as excinfo:
            tl.changed(other, _RELU).resolve(log)
        assert excinfo.value.fields["reason"] == "value_not_saved"
        assert excinfo.value.fields["sample"] == "reference"
    finally:
        other.cleanup()


def test_structural_site_key_mismatch_refuses():
    """A label+shape coincidence across different structures is caught, not compared."""

    class _A(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.lin(x)

    class _C(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.other = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.other(x)

    subject = tl.trace(_A(), torch.randn(1, 4))
    reference = tl.trace(_C(), torch.randn(1, 4))
    try:
        assert subject["linear_1_1"].site_key != reference["linear_1_1"].site_key
        with pytest.raises(SelectionError) as excinfo:
            tl.changed(reference, "linear_1_1").resolve(subject)
        # Since leverage B4 the guarded join is the pairing authority: the
        # coincident label's structural key exists on the subject only.
        assert excinfo.value.fields["reason"] == "site_join_refused"
        assert excinfo.value.fields["join_verdict"] == "one_sided_subject"
    finally:
        subject.cleanup()
        reference.cleanup()


def test_param_population_refuses_kind(log_b):
    """Differential producers read ACT payloads; PARAM populations refuse typed."""

    with pytest.raises(SelectionError) as excinfo:
        tl.changed(log_b, tl.params("encoder.0.weight"))
    assert excinfo.value.fields["code"] == "selection_kind_incompatible"


def test_constructor_validation_rows(log_b):
    """Construction-time parameter validation refuses early and plainly."""

    with pytest.raises(ValueError, match="'abs' or 'signed'"):
        tl.changed(log_b, by="relative")
    with pytest.raises(ValueError, match="must not be NaN"):
        tl.changed(log_b, above=float("nan"))
    with pytest.raises(ValueError, match="exactly one of"):
        tl.top_changed(log_b, _RELU)
    with pytest.raises(ValueError, match="exactly one of"):
        tl.top_changed(log_b, _RELU, 3, fraction=0.1)
    with pytest.raises(ValueError, match="non-negative int"):
        tl.top_changed(log_b, _RELU, -1)
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        tl.top_changed(log_b, _RELU, fraction=1.5)


def test_changed_composes_with_the_algebra(log, log_b):
    """Differential selections compose through the operator set."""

    composed = (tl.changed(log_b, _RELU, above=0.1) & tl.sign(_RELU, "nonzero")).resolve(log)
    delta = _delta(log, log_b, _RELU)
    expected = (delta.abs() > 0.1) & (log[_RELU].out != 0)
    assert torch.equal(composed[0].mask, expected)
    assert _relations(composed) == ["exact"]


def test_repr_stability_pins(log_b):
    """The constructor-shaped disclosures are pinned (readable AST spellings)."""

    label = log_b.trace_label
    assert repr(tl.changed(log_b)) == (
        f"Selection[ACT](changed(vs={label!r}, above=0.0, below=None, by='abs', "
        "within=saved_sites))"
    )
    assert repr(tl.stable_across_passes(_TANH, tol=0.5, passes=[1, 3])) == (
        "Selection[ACT](stable_across_passes(tol=0.5, passes=[1, 3], "
        f"within=Selection[ACT](site({_TANH!r}))))"
    )


# ---------------------------------------------------------------------------
# Cross-pass producers: pass-qualified claims with the honesty floor.
# ---------------------------------------------------------------------------


def _stacked_passes(rec_log, passes=(1, 2, 3)) -> torch.Tensor:
    """Manually stack the tanh layer's per-pass outputs in float64."""

    return torch.stack(
        [rec_log[f"{_TANH}:{pass_index}"].out.to(torch.float64) for pass_index in passes]
    )


def test_stable_selects_the_pass_invariant_unit(rec_log):
    """The engineered constant unit is stable; masks land on every pass site."""

    resolved = tl.stable_across_passes(_TANH, tol=1e-9).resolve(rec_log)
    assert [entry.site_key for entry in resolved] == [(_TANH, 1), (_TANH, 2), (_TANH, 3)]
    stacked = _stacked_passes(rec_log)
    expected = (stacked.max(dim=0).values - stacked.min(dim=0).values) <= 1e-9
    for entry in resolved:
        assert torch.equal(entry.mask, expected)
    assert expected[0, 0]  # the engineered unit
    assert _relations(resolved) == ["exact", "exact", "exact"]
    assert "n_passes=3" in resolved[0].provenance.source


def test_pass_variance_bounds_match_manual(rec_log):
    """Strict bounds on the unbiased float64 cross-pass variance."""

    variance = torch.var(_stacked_passes(rec_log), dim=0)
    low = tl.pass_variance(_TANH, below=1e-12).resolve(rec_log)
    assert torch.equal(low[0].mask, variance < 1e-12)
    assert low[0].mask[0, 0]  # the engineered unit has zero variance
    high = tl.pass_variance(_TANH, above=1e-12).resolve(rec_log)
    assert torch.equal(high[0].mask, variance > 1e-12)
    band = tl.pass_variance(_TANH, above=1e-12, below=1.0).resolve(rec_log)
    assert torch.equal(band[0].mask, (variance > 1e-12) & (variance < 1.0))
    assert _relations(low) == ["exact", "exact", "exact"]


def test_explicit_pass_window_restricts_evidence_and_family(rec_log):
    """passes=[2, 3] uses only that window and touches only those sites."""

    resolved = tl.stable_across_passes(_TANH, tol=1e-6, passes=[2, 3]).resolve(rec_log)
    assert [entry.site_key for entry in resolved] == [(_TANH, 2), (_TANH, 3)]
    stacked = _stacked_passes(rec_log, passes=(2, 3))
    expected = (stacked.max(dim=0).values - stacked.min(dim=0).values) <= 1e-6
    for entry in resolved:
        assert torch.equal(entry.mask, expected)
    assert "n_passes=2" in resolved[0].provenance.source


def test_missing_window_pass_refuses_named(rec_log):
    """A window naming an absent pass refuses site_not_in_trace, pass-qualified."""

    with pytest.raises(SelectionError) as excinfo:
        tl.stable_across_passes(_TANH, passes=[2, 9]).resolve(rec_log)
    assert excinfo.value.fields["reason"] == "site_not_in_trace"
    assert excinfo.value.fields["site"] == f"{_TANH}:9"


def test_window_construction_floor_and_validation():
    """Explicit windows need >= 2 distinct 1-based passes, refused at construction."""

    with pytest.raises(ValueError, match="at least 2"):
        tl.stable_across_passes(_TANH, passes=[2])
    with pytest.raises(ValueError, match="at least 2"):
        tl.pass_variance(_TANH, below=1.0, passes=[2, 2])
    with pytest.raises(ValueError, match="1-based"):
        tl.stable_across_passes(_TANH, passes=[0, 1])
    with pytest.raises(ValueError, match="at least one bound"):
        tl.pass_variance(_TANH)
    with pytest.raises(ValueError, match="non-negative"):
        tl.stable_across_passes(_TANH, tol=-1.0)


def test_single_pass_layer_refuses_vacuous_claim(log):
    """A cross-pass claim on a single-pass layer refuses with teaching, never
    a vacuous everything-mask."""

    with pytest.raises(SelectionError) as excinfo:
        tl.stable_across_passes(_RELU).resolve(log)
    assert excinfo.value.fields["reason"] == "population_too_small"
    assert excinfo.value.fields["available"] == 1
    assert "vacuously true" in str(excinfo.value)
    with pytest.raises(SelectionError):
        tl.pass_variance(below=1.0).resolve(log)  # default population, feedforward


def test_cross_pass_shape_drift_refuses():
    """A layer whose index space drifts across the window refuses typed."""

    from torchlens.selection import SelectionProvenance, SiteEntry, _mask_whole
    from torchlens.selection_compare import _pass_groups, _PassTerm

    def _entry(pass_index: int, shape: tuple[int, ...]) -> SiteEntry:
        return SiteEntry(
            kind="ACT",
            site_key=("grow_1_1", pass_index),
            provenance=SelectionProvenance(relation="exact", source="test"),
            _mask=_mask_whole(shape),
        )

    population = [_entry(1, (1, 4)), _entry(2, (1, 8))]
    node = _PassTerm(stat="stable", within=None)
    with pytest.raises(SelectionError) as excinfo:
        _pass_groups(node, population, "stable_across_passes")
    assert excinfo.value.fields["reason"] == "mask_shape_mismatch"
    assert "constant-shape window" in str(excinfo.value)


def test_cross_pass_complex_refuses_typed():
    """Complex payloads have no total order across passes; refuse typed."""

    class _RecComplex(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.pre = nn.Identity()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for _ in range(2):
                x = torch.mul(self.pre(x), 1j)
            return x

    trace = tl.trace(_RecComplex(), torch.randn(1, 4, dtype=torch.complex64))
    try:
        with pytest.raises(SelectionError) as excinfo:
            tl.stable_across_passes("mul_1_2").resolve(trace)
        assert excinfo.value.fields["reason"] == "value_criterion_invalid"
    finally:
        trace.cleanup()


def test_cross_pass_nan_never_selected():
    """An element NaN at any window pass has NaN spread/variance: never selected."""

    class _RecSqrt(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.pre = nn.Identity()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for _ in range(2):
                x = torch.sqrt(self.pre(x))
            return x

    trace = tl.trace(_RecSqrt(), torch.tensor([[-1.0, 4.0, 9.0, 16.0]]))
    try:
        site = "sqrt_1_2"
        nan_any = torch.isnan(trace[f"{site}:1"].out) | torch.isnan(trace[f"{site}:2"].out)
        assert nan_any.any()
        stable = tl.stable_across_passes(site, tol=1e30).resolve(trace)
        assert not (stable[0].mask & nan_any).any()
        swing = tl.pass_variance(site, above=-1.0).resolve(trace)
        assert not (swing[0].mask & nan_any).any()
    finally:
        trace.cleanup()


def test_population_restriction_intersects_across_the_window(rec_log):
    """The element population is the intersection of the window entries' masks."""

    population = tl.sign(_TANH, "positive")  # per-pass masks differ
    resolved = tl.stable_across_passes(population, tol=1e30).resolve(rec_log)
    per_pass = [rec_log[f"{_TANH}:{p}"].out > 0 for p in (1, 2, 3)]
    intersection = per_pass[0] & per_pass[1] & per_pass[2]
    for entry in resolved:
        assert torch.equal(entry.mask, intersection)  # tol=inf: criterion all-true


# ---------------------------------------------------------------------------
# End-to-end gallery rows: comparative selections drive interventions.
# ---------------------------------------------------------------------------


def test_do_zero_ablates_top_movers(model):
    """fork.do(top_changed(...), zero_ablate()) zeroes exactly the top movers."""

    torch.manual_seed(4)
    trace = tl.trace(
        model,
        torch.randn(1, 1, 10, 10),
        capture=tl.options.CaptureOptions(intervention_ready=True),
    )
    torch.manual_seed(5)
    reference = tl.trace(model, torch.randn(1, 1, 10, 10))
    try:
        selection = tl.top_changed(reference, _RELU, 6)
        mask = selection.resolve(trace)[0].mask
        baseline = trace[_RELU].out.clone()
        fork = trace.fork()
        fork.do(selection.resolve(fork), tl.zero_ablate())
        edited = fork[_RELU].out
        assert bool((edited[mask] == 0).all())
        assert torch.equal(edited[~mask], baseline[~mask])
        assert torch.equal(trace[_RELU].out, baseline)  # capture truth intact
    finally:
        trace.cleanup()
        reference.cleanup()


def test_do_zero_ablates_a_stable_unit_at_every_pass():
    """fork.do(stable_across_passes(...), zero_ablate()) edits every window pass."""

    torch.manual_seed(6)
    trace = tl.trace(
        _Recurrent(),
        torch.randn(1, 4),
        capture=tl.options.CaptureOptions(intervention_ready=True),
    )
    try:
        selection = tl.stable_across_passes(_TANH, tol=1e-9)
        resolved = selection.resolve(trace)
        assert resolved[0].mask[0, 0]
        fork = trace.fork()
        fork.do(selection, tl.zero_ablate())
        for pass_index in (1, 2, 3):
            edited = fork[f"{_TANH}:{pass_index}"].out
            baseline = trace[f"{_TANH}:{pass_index}"].out
            mask = resolved[pass_index - 1].mask
            assert bool((edited[mask] == 0).all())
            assert (baseline[0, 0] != 0).all()  # the ablation changed real values
    finally:
        trace.cleanup()
