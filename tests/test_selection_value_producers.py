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
from torchlens._selection.provenance import SelectionProvenance
from torchlens.bundle import Bundle
from torchlens.errors import SelectionError


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


@pytest.mark.smoke
def test_top_fraction_ceil_rule_pinned(log):
    """top_fraction selects ceil(fraction * population), never zero when nonempty."""

    numel = log[_RELU].out.numel()
    resolved = log[tl.top_fraction(_RELU, 0.01)]
    assert resolved[0].num_selected == math.ceil(0.01 * numel)
    assert log[tl.top_fraction(_RELU, 0.0)].is_empty


@pytest.mark.smoke
def test_threshold_band_and_abs(log):
    """Strict above/below bounds and their conjunction, plus magnitude form."""

    value = log[_RELU].out
    above = log[tl.threshold(_RELU, above=0.5)]
    assert torch.equal(above[0].mask, value > 0.5)
    band = log[tl.threshold(_RELU, above=0.1, below=0.5)]
    assert torch.equal(band[0].mask, (value > 0.1) & (value < 0.5))
    magnitude = log[tl.threshold(_RELU, above=0.5, by="abs")]
    assert torch.equal(magnitude[0].mask, value.abs() > 0.5)
    assert _relations(above) == ["exact"]


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
        wide = trace[tl.threshold(site, above=-1e30)]
        assert not (wide[0].mask & nan_mask).any()
        for which in ("positive", "negative", "zero", "nonzero"):
            resolved = trace[tl.sign(site, which)]
            assert not (resolved[0].mask & nan_mask).any()
        ranked = trace[tl.top_k(site, 2, largest=False)]
        assert not (ranked[0].mask & nan_mask).any()
        with pytest.raises(SelectionError) as excinfo:
            trace[tl.top_k(site, 3)]  # only 2 rankable elements
        assert excinfo.value.fields["reason"] == "population_too_small"
    finally:
        trace.cleanup()


# ---------------------------------------------------------------------------
# Statistical producers: multi-sample honesty.
# ---------------------------------------------------------------------------


def test_saturated_detects_pinned_sigmoid_channel(log, log_b):
    """Units within tol of a declared bound in every sample are saturated."""

    resolved = log[tl.saturated([log, log_b], low=0.0, high=1.0, atol=1e-4, within=_SIGMOID)]
    assert resolved[0].mask[0, 0].all()  # bias=100 channel pinned at 1
    expected = ((log[_SIGMOID].out - 1.0).abs() <= 1e-4) & (
        (log_b[_SIGMOID].out - 1.0).abs() <= 1e-4
    )
    expected |= (log[_SIGMOID].out.abs() <= 1e-4) & (log_b[_SIGMOID].out.abs() <= 1e-4)
    assert torch.equal(resolved[0].mask, expected)
    assert _relations(resolved) == ["upper_bound"]


@pytest.mark.smoke
def test_saturated_requires_a_declared_bound(log, log_b):
    """Saturation is relative to a range; no bound refuses at construction."""

    with pytest.raises(ValueError, match="declared bound"):
        tl.saturated([log, log_b])


@pytest.mark.smoke
def test_low_variance_selects_constant_channel(log, log_b):
    """The zero-weight constant channel has zero variance across samples."""

    resolved = log[tl.low_variance([log, log_b], threshold=1e-12, within=_RELU)]
    assert resolved[0].mask[0, 2].all()  # constant 0.7 channel
    stacked = torch.stack([log[_RELU].out.to(torch.float64), log_b[_RELU].out.to(torch.float64)])
    assert torch.equal(resolved[0].mask, torch.var(stacked, dim=0) < 1e-12)
    assert _relations(resolved) == ["exact"]
    with pytest.raises(ValueError, match="at least 2"):
        tl.low_variance([log], threshold=1e-12)


def test_samples_accept_a_bundle(log, log_b):
    """A Bundle is a legal evidence set (it iterates its member traces)."""

    bundled = log[tl.dead(Bundle([log, log_b]), within=_RELU)]
    listed = log[tl.dead([log, log_b], within=_RELU)]
    assert bundled == listed


@pytest.mark.smoke
def test_samples_must_be_traces(log):
    """Non-Trace evidence refuses at construction with the offending index."""

    with pytest.raises(ValueError, match=r"samples\[1\]"):
        tl.dead([log, "not a trace"])


def test_sample_missing_site_refuses_typed(log, log_b):
    """A sample lacking a population site refuses site_not_in_trace, named."""

    other = tl.trace(nn.Linear(4, 2), torch.randn(1, 4))
    try:
        with pytest.raises(SelectionError) as excinfo:
            log[tl.dead([log, other], within=_RELU)]
        assert excinfo.value.fields["reason"] == "site_not_in_trace"
        assert excinfo.value.fields["sample"] == "samples[1]"
    finally:
        other.cleanup()


def test_sample_shape_drift_refuses_typed(model, log):
    """Shape-mismatched sample evidence refuses mask_shape_mismatch, named."""

    other = tl.trace(model, torch.randn(1, 1, 12, 12))
    try:
        with pytest.raises(SelectionError) as excinfo:
            log[tl.dead([log, other], within=_RELU)]
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
        assert _relations(log[selection]) == [expected[name]], name


def test_wrong_provenance_claim_fails_closed():
    """An off-lattice relation cannot be constructed — the claim vocabulary is closed."""

    with pytest.raises(ValueError, match="relation must be one of"):
        SelectionProvenance(relation="approximately")


# ---------------------------------------------------------------------------
# Typed refusal matrix.
# ---------------------------------------------------------------------------


def test_param_population_refuses_kind(log):
    """Value producers read ACT payloads; a PARAM population refuses typed."""

    with pytest.raises(SelectionError) as excinfo:
        tl.top_k(tl.param("encoder.0.weight"), 3)
    assert excinfo.value.fields["code"] == "selection_kind_incompatible"


@pytest.mark.smoke
def test_unknown_site_refuses_typed(log):
    """A population site absent from the trace refuses site_not_in_trace."""

    with pytest.raises(SelectionError) as excinfo:
        log[tl.sign("nonexistent_9_9", "zero")]
    assert excinfo.value.fields["reason"] == "site_not_in_trace"


def test_unsaved_payload_refuses_value_not_saved(model):
    """Value criteria on unsaved payloads refuse with the re-capture remedy."""

    trace = tl.trace(model, torch.randn(1, 1, 10, 10), save=tl.func("conv2d"))
    try:
        with pytest.raises(SelectionError) as excinfo:
            trace[tl.sign(_RELU, "zero")]
        assert excinfo.value.fields["reason"] == "value_not_saved"
        # the DEFAULT population is the retained sites, so it still resolves
        resolved = trace[tl.threshold(above=0.0)]
        labels = {entry.key[0] for entry in resolved}
        assert labels and all(label.startswith("conv2d") for label in labels)
    finally:
        trace.cleanup()


def test_foreign_resolved_population_refuses_trace_mismatch(log, log_b):
    """A within resolved on another trace refuses selection_trace_mismatch."""

    foreign = log_b[tl.sign(_RELU, "zero")]
    with pytest.raises(SelectionError) as excinfo:
        log[tl.top_k(foreign, 2)]
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
            trace[tl.top_k(site, 2)]
        assert excinfo.value.fields["reason"] == "value_criterion_invalid"
        resolved = trace[tl.top_k(site, 2, by="abs")]
        assert resolved[0].num_selected == 2
        zero = trace[tl.sign(site, "zero")]  # magnitude classes allowed
        assert zero[0].num_selected == 0
        with pytest.raises(SelectionError):
            trace[tl.sign(site, "positive")]
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


@pytest.mark.smoke
def test_repr_stability_pins(log, log_b):
    """The constructor-shaped disclosures are pinned (readable AST spellings)."""

    assert repr(tl.top_k(_RELU, 5)) == (
        "Selection[ACT](top_k(k=5, by='value', largest=True, "
        f"within=Selection[ACT](site({_RELU!r}))))"
    )
    assert repr(tl.dead([log, log_b], atol=0.0)) == (
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
        capture=tl.options.CaptureOptions(replayable=True),
    )
    try:
        selection = tl.top_k(_RELU, 6)
        mask = trace[selection][0].mask
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
