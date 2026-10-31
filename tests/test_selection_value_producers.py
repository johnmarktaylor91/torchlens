"""Value-based + statistical selection producers: acceptance and honesty gates.

Covers the producer-lane obligations: every producer returns a composable
``Selection`` (operator mixin via the algebra), masks are exact AS SETS and
verified against manually computed criteria, the provenance relations are
PINNED per producer (a wrong claim fails here), composition flows through
the normative JOIN/FLIP/DIFFERENCE tables, the multi-sample boundary is
enforced (``dead`` with one sample refuses with the single-trace spelling
named), the typed refusal matrix (kind, trace binding, unsaved payloads,
missing/mismatched sample sites, complex ordered comparisons), NaN
exclusion, deterministic rank tie-breaks, and the end-to-end ``do()``
gallery row (ablate the top-k units).
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens._selection import _FLIP, SelectionError, SelectionProvenance
from torchlens.bundle import Bundle


class _Probe(nn.Module):
    """CNN with engineered pathologies: dead, constant, and saturated units.

    encoder[0] channel 3 has a huge negative bias (its ReLU output is dead on
    any bounded input); channel 2 has zero weights and bias 0.7 (constant
    across inputs); head channel 0 has a huge positive bias (sigmoid pinned
    at 1).
    """

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(1, 4, 3), nn.ReLU())
        self.head = nn.Conv2d(4, 2, 3)
        with torch.no_grad():
            self.encoder[0].bias[3] = -100.0
            self.encoder[0].weight[2] = 0.0
            self.encoder[0].bias[2] = 0.7
            self.head.bias[0] = 100.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.head(self.encoder(x)))


_RELU = "relu_1_2"
_SIGMOID = "sigmoid_1_4"


@pytest.fixture(scope="module")
def model():
    """One shared deterministic probe model."""

    torch.manual_seed(0)
    return _Probe()


@pytest.fixture(scope="module")
def log(model):
    """Primary capture (geometry + value source for single-capture producers)."""

    torch.manual_seed(1)
    trace = tl.trace(model, torch.randn(1, 1, 10, 10))
    try:
        yield trace
    finally:
        trace.cleanup()


@pytest.fixture(scope="module")
def log_b(model):
    """Second same-model capture (evidence set partner)."""

    torch.manual_seed(2)
    trace = tl.trace(model, torch.randn(1, 1, 10, 10))
    try:
        yield trace
    finally:
        trace.cleanup()


def _relations(resolved):
    return [entry.provenance.relation for entry in resolved]


# ---------------------------------------------------------------------------
# Value producers: masks are exact as sets against manual criteria.
# ---------------------------------------------------------------------------


def test_top_k_mask_matches_manual_ranking(log):
    """top_k selects exactly the k largest elements of the site's saved value."""

    resolved = tl.top_k(_RELU, 7).resolve(log)
    value = log[_RELU].out.reshape(-1).to(torch.float64)
    expected = torch.zeros_like(value, dtype=torch.bool)
    expected[torch.argsort(value, descending=True, stable=True)[:7]] = True
    assert torch.equal(resolved[0].mask.reshape(-1), expected)
    assert _relations(resolved) == ["exact"]
    assert resolved[0].num_selected == 7


def test_top_k_bottom_and_abs_variants(log):
    """largest=False ranks ascending; by='abs' ranks magnitudes."""

    value = log[_SIGMOID].out.reshape(-1)
    bottom = tl.top_k(_SIGMOID, 3, largest=False).resolve(log)
    assert bottom[0].mask.reshape(-1)[torch.argsort(value.to(torch.float64), stable=True)[:3]].all()
    magnitude = tl.top_k(_SIGMOID, 3, by="abs").resolve(log)
    expected = torch.zeros_like(value, dtype=torch.bool)
    expected[torch.argsort(value.abs().to(torch.float64), descending=True, stable=True)[:3]] = True
    assert torch.equal(magnitude[0].mask.reshape(-1), expected)


def test_top_k_tie_break_is_first_flat_indices():
    """Ties break deterministically by flat index (stable sort, pinned)."""

    class _AllZero(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.relu(x * 0.0)

    trace = tl.trace(_AllZero(), torch.randn(1, 2, 2))
    try:
        resolved = tl.top_k("relu_1_2", 2).resolve(trace)
        flat = resolved[0].mask.reshape(-1)
        assert flat[:2].all() and not flat[2:].any()
    finally:
        trace.cleanup()


def test_top_k_population_too_small_refuses_typed(log):
    """k beyond the rankable population refuses with structured counts."""

    numel = log[_RELU].out.numel()
    with pytest.raises(SelectionError) as excinfo:
        tl.top_k(_RELU, numel + 1).resolve(log)
    assert excinfo.value.fields["code"] == "selection_unresolvable"
    assert excinfo.value.fields["reason"] == "population_too_small"
    assert excinfo.value.fields["available"] == numel


def test_top_fraction_ceil_rule_pinned(log):
    """top_fraction selects ceil(fraction * population), never zero when nonempty."""

    numel = log[_RELU].out.numel()
    resolved = tl.top_fraction(_RELU, 0.01).resolve(log)
    assert resolved[0].num_selected == math.ceil(0.01 * numel)
    assert tl.top_fraction(_RELU, 0.0).resolve(log).empty


def test_threshold_band_and_abs(log):
    """Strict above/below bounds and their conjunction, plus magnitude form."""

    value = log[_RELU].out
    above = tl.threshold(_RELU, above=0.5).resolve(log)
    assert torch.equal(above[0].mask, value > 0.5)
    band = tl.threshold(_RELU, above=0.1, below=0.5).resolve(log)
    assert torch.equal(band[0].mask, (value > 0.1) & (value < 0.5))
    magnitude = tl.threshold(_RELU, above=0.5, by="abs").resolve(log)
    assert torch.equal(magnitude[0].mask, value.abs() > 0.5)
    assert _relations(above) == ["exact"]


def test_sign_classes_partition_the_site(log):
    """positive/negative/zero partition; zero is the sparsity mask."""

    value = log[_RELU].out
    zero = tl.sign(_RELU, "zero").resolve(log)
    positive = tl.sign(_RELU, "positive").resolve(log)
    negative = tl.sign(_RELU, "negative").resolve(log)
    nonzero = tl.sign(_RELU, "nonzero").resolve(log)
    assert torch.equal(zero[0].mask, value == 0)
    assert torch.equal(positive[0].mask, value > 0)
    assert negative[0].num_selected == 0  # ReLU output has no negatives
    assert torch.equal(nonzero[0].mask, ~zero[0].mask)
    total = zero[0].num_selected + positive[0].num_selected
    assert total == value.numel()
    # the engineered dead channel is all-zero on this capture
    assert zero[0].mask[0, 3].all()


def test_nan_never_satisfies_any_criterion():
    """NaN elements are excluded from bands, sign classes, and rankings."""

    class _NaNs(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.sqrt(x)

    trace = tl.trace(_NaNs(), torch.tensor([[-1.0, 4.0, 9.0, -1.0]]))
    try:
        site = "sqrt_1_1"
        nan_mask = torch.isnan(trace[site].out)
        assert nan_mask.any()
        wide = tl.threshold(site, above=-1e30).resolve(trace)
        assert not (wide[0].mask & nan_mask).any()
        for which in ("positive", "negative", "zero", "nonzero"):
            resolved = tl.sign(site, which).resolve(trace)
            assert not (resolved[0].mask & nan_mask).any()
        ranked = tl.top_k(site, 2, largest=False).resolve(trace)
        assert not (ranked[0].mask & nan_mask).any()
        with pytest.raises(SelectionError) as excinfo:
            tl.top_k(site, 3).resolve(trace)  # only 2 rankable elements
        assert excinfo.value.fields["reason"] == "population_too_small"
    finally:
        trace.cleanup()


def test_global_default_population_spans_saved_sites(log):
    """within=None ranks across every retained tensor site of the capture."""

    resolved = tl.top_k(None, 10).resolve(log)
    assert len(resolved) > 1  # touched family: all saved sites
    assert sum(entry.num_selected for entry in resolved) == 10


def test_value_producer_restricted_by_selection_population(log):
    """A Selection population restricts the candidate set, not just the mask."""

    zeros = tl.sign(_RELU, "zero")
    ranked = tl.top_k(zeros, 4, largest=False).resolve(log)
    value = log[_RELU].out
    relu_entry = [entry for entry in ranked if entry.key == (_RELU, 1)][0]
    assert relu_entry.num_selected == 4
    assert (value[relu_entry.mask] == 0).all()


# ---------------------------------------------------------------------------
# Statistical producers: multi-sample honesty.
# ---------------------------------------------------------------------------


def test_dead_requires_multi_sample_and_names_the_single_trace_spelling(log):
    """The single-capture form is a DIFFERENT spelling, enforced at construction."""

    with pytest.raises(ValueError, match=r"MULTI-SAMPLE.*sign\(site, 'zero'\)"):
        tl.dead([log])


def test_dead_mask_is_conjunction_across_samples(log, log_b):
    """dead selects exactly the elements silent in EVERY sample."""

    resolved = tl.dead([log, log_b], within=_RELU).resolve(log)
    expected = (log[_RELU].out == 0) & (log_b[_RELU].out == 0)
    assert torch.equal(resolved[0].mask, expected)
    assert resolved[0].mask[0, 3].all()  # the engineered dead channel
    assert _relations(resolved) == ["upper_bound"]
    assert "n_samples=2" in resolved[0].provenance.source


def test_dead_composes_with_module_selector(log, log_b):
    """The brief's acceptance row: tl.dead(...) & tl.in_module('encoder')."""

    composed = tl.dead([log, log_b]) & tl.in_module("encoder")
    resolved = composed.resolve(log)
    site_keys = {entry.key for entry in resolved}
    assert (_RELU, 1) in site_keys
    assert (_SIGMOID, 1) not in site_keys  # head is outside the encoder
    relu_entry = [entry for entry in resolved if entry.key == (_RELU, 1)][0]
    assert relu_entry.mask[0, 3].all()
    assert relu_entry.provenance.relation == "upper_bound"  # join(upper, exact)


def test_saturated_detects_pinned_sigmoid_channel(log, log_b):
    """Units within tol of a declared bound in every sample are saturated."""

    resolved = tl.saturated([log, log_b], low=0.0, high=1.0, tol=1e-4, within=_SIGMOID).resolve(log)
    assert resolved[0].mask[0, 0].all()  # bias=100 channel pinned at 1
    expected = ((log[_SIGMOID].out - 1.0).abs() <= 1e-4) & (
        (log_b[_SIGMOID].out - 1.0).abs() <= 1e-4
    )
    expected |= (log[_SIGMOID].out.abs() <= 1e-4) & (log_b[_SIGMOID].out.abs() <= 1e-4)
    assert torch.equal(resolved[0].mask, expected)
    assert _relations(resolved) == ["upper_bound"]


def test_saturated_requires_a_declared_bound(log, log_b):
    """Saturation is relative to a range; no bound refuses at construction."""

    with pytest.raises(ValueError, match="declared bound"):
        tl.saturated([log, log_b])


def test_low_variance_selects_constant_channel(log, log_b):
    """The zero-weight constant channel has zero variance across samples."""

    resolved = tl.low_variance([log, log_b], threshold=1e-12, within=_RELU).resolve(log)
    assert resolved[0].mask[0, 2].all()  # constant 0.7 channel
    stacked = torch.stack([log[_RELU].out.to(torch.float64), log_b[_RELU].out.to(torch.float64)])
    assert torch.equal(resolved[0].mask, torch.var(stacked, dim=0) < 1e-12)
    assert _relations(resolved) == ["exact"]
    with pytest.raises(ValueError, match="at least 2"):
        tl.low_variance([log], threshold=1e-12)


def test_samples_accept_a_bundle(log, log_b):
    """A Bundle is a legal evidence set (it iterates its member traces)."""

    bundled = tl.dead(Bundle([log, log_b]), within=_RELU).resolve(log)
    listed = tl.dead([log, log_b], within=_RELU).resolve(log)
    assert bundled == listed


def test_samples_must_be_traces(log):
    """Non-Trace evidence refuses at construction with the offending index."""

    with pytest.raises(ValueError, match=r"samples\[1\]"):
        tl.dead([log, "not a trace"])


def test_sample_missing_site_refuses_typed(log, log_b):
    """A sample lacking a population site refuses site_not_in_trace, named."""

    other = tl.trace(nn.Linear(4, 2), torch.randn(1, 4))
    try:
        with pytest.raises(SelectionError) as excinfo:
            tl.dead([log, other], within=_RELU).resolve(log)
        assert excinfo.value.fields["reason"] == "site_not_in_trace"
        assert excinfo.value.fields["sample"] == "samples[1]"
    finally:
        other.cleanup()


def test_sample_shape_drift_refuses_typed(model, log):
    """Shape-mismatched sample evidence refuses mask_shape_mismatch, named."""

    other = tl.trace(model, torch.randn(1, 1, 12, 12))
    try:
        with pytest.raises(SelectionError) as excinfo:
            tl.dead([log, other], within=_RELU).resolve(log)
        assert excinfo.value.fields["reason"] == "mask_shape_mismatch"
        assert excinfo.value.fields["sample"] == "samples[1]"
    finally:
        other.cleanup()


# ---------------------------------------------------------------------------
# Provenance honesty: pinned relations, tables, and the wrong-claim gate.
# ---------------------------------------------------------------------------


def test_provenance_relations_pinned_per_producer(log, log_b):
    """The honesty table: dispositional claims are upper_bound, statistics exact."""

    pinned = {
        "top_k": tl.top_k(_RELU, 3),
        "top_fraction": tl.top_fraction(_RELU, 0.05),
        "threshold": tl.threshold(_RELU, above=0.0),
        "sign": tl.sign(_RELU, "zero"),
        "dead": tl.dead([log, log_b], within=_RELU),
        "saturated": tl.saturated([log, log_b], low=0.0, within=_RELU),
        "low_variance": tl.low_variance([log, log_b], threshold=1e-6, within=_RELU),
    }
    expected = {
        "top_k": "exact",
        "top_fraction": "exact",
        "threshold": "exact",
        "sign": "exact",
        "dead": "upper_bound",
        "saturated": "upper_bound",
        "low_variance": "exact",
    }
    for name, selection in pinned.items():
        assert _relations(selection.resolve(log)) == [expected[name]], name


def test_wrong_provenance_claim_fails_closed():
    """An off-lattice relation cannot be constructed — the claim vocabulary is closed."""

    with pytest.raises(ValueError, match="relation must be one of"):
        SelectionProvenance(relation="approximately")


def test_relation_composition_through_flip_and_difference(log, log_b):
    """~dead flips upper->lower; exact - upper_bound lands lower_bound."""

    dead_sel = tl.dead([log, log_b], within=_RELU)
    inverted = (~dead_sel).resolve(log)
    assert _relations(inverted) == [_FLIP["upper_bound"]] == ["lower_bound"]
    difference = (tl.sign(_RELU, "zero") - dead_sel).resolve(log)
    assert _relations(difference) == ["lower_bound"]
    # element check: zeros-in-this-capture minus dead-across-both
    expected = (log[_RELU].out == 0) & ~((log[_RELU].out == 0) & (log_b[_RELU].out == 0))
    assert torch.equal(difference[0].mask, expected)


def test_complement_of_top_k_counts(log):
    """~top_k keeps the touched family and complements within the site."""

    resolved = (~tl.top_k(_RELU, 5)).resolve(log)
    assert resolved[0].num_selected == log[_RELU].out.numel() - 5


# ---------------------------------------------------------------------------
# Typed refusal matrix.
# ---------------------------------------------------------------------------


def test_param_population_refuses_kind(log):
    """Value producers read ACT payloads; a PARAM population refuses typed."""

    with pytest.raises(SelectionError) as excinfo:
        tl.top_k(tl.param("encoder.0.weight"), 3)
    assert excinfo.value.fields["code"] == "selection_kind_incompatible"


def test_unknown_site_refuses_typed(log):
    """A population site absent from the trace refuses site_not_in_trace."""

    with pytest.raises(SelectionError) as excinfo:
        tl.sign("nonexistent_9_9", "zero").resolve(log)
    assert excinfo.value.fields["reason"] == "site_not_in_trace"


def test_unsaved_payload_refuses_value_not_saved(model):
    """Value criteria on unsaved payloads refuse with the re-capture remedy."""

    trace = tl.trace(model, torch.randn(1, 1, 10, 10), save=tl.func("conv2d"))
    try:
        with pytest.raises(SelectionError) as excinfo:
            tl.sign(_RELU, "zero").resolve(trace)
        assert excinfo.value.fields["reason"] == "value_not_saved"
        # the DEFAULT population is the retained sites, so it still resolves
        resolved = tl.threshold(above=0.0).resolve(trace)
        labels = {entry.key[0] for entry in resolved}
        assert labels and all(label.startswith("conv2d") for label in labels)
    finally:
        trace.cleanup()


def test_foreign_resolved_population_refuses_trace_mismatch(log, log_b):
    """A within resolved on another trace refuses selection_trace_mismatch."""

    foreign = tl.sign(_RELU, "zero").resolve(log_b)
    with pytest.raises(SelectionError) as excinfo:
        tl.top_k(foreign, 2).resolve(log)
    assert excinfo.value.fields["code"] == "selection_trace_mismatch"


def test_complex_ordered_comparison_refuses_value_criterion_invalid():
    """Ordered criteria on complex payloads refuse; by='abs' works."""

    class _Complex(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.fft.fft(x)

    trace = tl.trace(_Complex(), torch.randn(1, 8))
    try:
        site = "fft_1_1"
        with pytest.raises(SelectionError) as excinfo:
            tl.top_k(site, 2).resolve(trace)
        assert excinfo.value.fields["reason"] == "value_criterion_invalid"
        resolved = tl.top_k(site, 2, by="abs").resolve(trace)
        assert resolved[0].num_selected == 2
        zero = tl.sign(site, "zero").resolve(trace)  # magnitude classes allowed
        assert zero[0].num_selected == 0
        with pytest.raises(SelectionError):
            tl.sign(site, "positive").resolve(trace)
    finally:
        trace.cleanup()


def test_constructor_validation_rows():
    """Construction-time parameter validation refuses early and plainly."""

    with pytest.raises(ValueError, match="non-negative int"):
        tl.top_k("s", -1)
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        tl.top_fraction("s", 1.5)
    with pytest.raises(ValueError, match="at least one bound"):
        tl.threshold("s")
    with pytest.raises(ValueError, match="must not be NaN"):
        tl.threshold("s", above=float("nan"))
    with pytest.raises(ValueError, match="which"):
        tl.sign("s", "negativeish")
    with pytest.raises(ValueError, match="'value' or 'abs'"):
        tl.top_k("s", 1, by="magnitude")


def test_repr_stability_pins(log, log_b):
    """The constructor-shaped disclosures are pinned (readable AST spellings)."""

    assert repr(tl.top_k(_RELU, 5)) == (
        "Selection[ACT](top_k(k=5, by='value', largest=True, "
        f"within=Selection[ACT](site({_RELU!r}))))"
    )
    assert repr(tl.dead([log, log_b], tol=0.0)) == (
        "Selection[ACT](dead(n_samples=2, tol=0.0, within=saved_sites))"
    )


# ---------------------------------------------------------------------------
# End-to-end gallery row: value selection drives an intervention.
# ---------------------------------------------------------------------------


def test_do_zero_ablates_top_k_units(model):
    """fork.do(top_k(...), zero_ablate()) zeroes exactly the selected units."""

    trace = tl.trace(
        model,
        torch.randn(1, 1, 10, 10),
        capture=tl.options.CaptureOptions(intervention_ready=True),
    )
    try:
        selection = tl.top_k(_RELU, 6)
        mask = selection.resolve(trace)[0].mask
        baseline = trace[_RELU].out.clone()
        fork = trace.fork()
        fork.do(selection, tl.zero_ablate())
        edited = fork[_RELU].out
        assert bool((edited[mask] == 0).all())
        assert torch.equal(edited[~mask], baseline[~mask])
        assert (baseline[mask] > 0).all()  # the ablation changed real values
        assert torch.equal(trace[_RELU].out, baseline)  # capture truth intact
    finally:
        trace.cleanup()
