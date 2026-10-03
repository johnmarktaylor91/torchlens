"""B5: the neighbourhood acceptance test for the guarded site join (leverage D-3).

The identity contract is about a RELATION, so the primary assertion is the
JOIN'S REFUSAL on the touched cohorts — plus graph-neighbourhood (parent /
child site-key signature) agreement on every corroborated key, IN ADDITION TO
shape and payload where retained (review NEW-13's wording).

The three POSITIVE CONTROLS THAT MUST FAIL are retained as tests asserting
each plausible artifact-at-a-time oracle WRONGLY PASSES on a measured broken
join (memo section 6, controls #1-#3): the naive (key, ordinal) join pairs
everything, set membership reports every baseline key preserved, and payload
equality is blind — while the guarded join refuses and the neighbourhood
signature catches the mis-pair.
"""

from __future__ import annotations

import pytest
import torch
from test_leverage_div_fixtures import make_insertion_pair

import torchlens as tl
from torchlens.postprocess._site_join import (
    join_site_profiles,
    occurrence_coordinates,
    site_profile,
)

_SAVE_ALL = {"capture": tl.options.CaptureOptions(layers_to_save="all")}


_SMOKE = pytest.mark.smoke


def _capture_pair():
    baseline_model, variant_model, x = make_insertion_pair()
    baseline = tl.trace(baseline_model, x, **_SAVE_ALL)
    variant = tl.trace(variant_model, x, **_SAVE_ALL)
    return baseline, variant


def _naive_ordinal_pairs(baseline, variant):
    """The panel-banned naive join: pair same-key occurrences by capture order."""

    profiles = (site_profile(baseline), site_profile(variant))
    by_key: list[dict[str, list[str]]] = [{}, {}]
    for side, (trace, profile) in enumerate(zip((baseline, variant), profiles, strict=True)):
        for label in trace.op_labels:
            key = profile.keys.get(label)
            if key is not None:
                by_key[side].setdefault(key, []).append(label)
    pairs = []
    for key, base_labels in by_key[0].items():
        # strict=False IS the naive join's defect on display: silent
        # truncation when the occurrence counts drift.
        for base_label, variant_label in zip(base_labels, by_key[1].get(key, []), strict=False):
            pairs.append((key, base_label, variant_label))
    return pairs


def _key_of(trace, label):
    """One op's site key through the flexible lookup (bare or pass-qualified)."""

    return getattr(trace.ops[label], "site_key", None) or "?"


def _neighbourhood_signature(trace, keys, label):
    """One op's (sorted parent keys, sorted child keys) signature."""

    op = trace.ops[label]
    parents = tuple(sorted(_key_of(trace, parent) for parent in op.parents))
    children = tuple(sorted(_key_of(trace, child) for child in op.children))
    return parents, children


@_SMOKE
def test_guarded_join_refuses_insertion_cohort():
    """PRIMARY: the touched cohort refuses; untouched conv cohorts corroborate."""

    baseline, variant = _capture_pair()
    rows = join_site_profiles(site_profile(baseline), site_profile(variant))
    relu_rows = {key: row for key, row in rows.items() if "|relu|" in key}
    assert relu_rows, "fixture lost its reused-relu cohort"
    assert all(row.verdict.value == "refused_cardinality" for row in relu_rows.values())
    conv_rows = {key: row for key, row in rows.items() if "conv2d" in key}
    assert conv_rows and all(row.verdict.value == "corroborated" for row in conv_rows.values())
    # D-2 pricing: the positional tier is exactly the graph-boundary pseudo-ops.
    positional = [key for key, row in rows.items() if row.verdict.value == "positional"]
    assert all(("input" in key) or ("output" in key) for key in positional)


@_SMOKE
def test_clean_recapture_joins_without_refusals():
    """A clean recapture of the same model corroborates every real-op key."""

    baseline_model, _, x = make_insertion_pair()
    first = tl.trace(baseline_model, x, **_SAVE_ALL)
    second = tl.trace(baseline_model, x, **_SAVE_ALL)
    rows = join_site_profiles(site_profile(first), site_profile(second))
    assert rows and all(row.joined for row in rows.values())


@_SMOKE
def test_neighbourhood_agreement_on_corroborated_keys():
    """Corroborated keys agree on parent/child site-key signatures across runs."""

    baseline, variant = _capture_pair()
    baseline_profile, variant_profile = site_profile(baseline), site_profile(variant)
    rows = join_site_profiles(baseline_profile, variant_profile)
    baseline_coords = occurrence_coordinates(baseline, baseline_profile.keys)
    variant_by_coord = {
        coord: label
        for label, coord in occurrence_coordinates(variant, variant_profile.keys).items()
    }
    checked = 0
    for label, coord in baseline_coords.items():
        key = coord[0]
        if not rows.get(key) or rows[key].verdict.value != "corroborated":
            continue
        partner = variant_by_coord.get(coord)
        assert partner is not None
        assert _neighbourhood_signature(
            baseline, baseline_profile.keys, label
        ) == _neighbourhood_signature(variant, variant_profile.keys, partner)
        # NEW-13's wording: neighbourhood IN ADDITION TO shape and payload.
        assert baseline.ops[label].shape == variant.ops[partner].shape
        assert torch.equal(baseline.ops[label].out, variant.ops[partner].out)
        checked += 1
    assert checked >= 2, "no corroborated keys were actually checked"


@_SMOKE
def test_positive_control_naive_ordinal_join_pairs_everything():
    """MUST-FAIL CONTROL #1: the naive (key, ordinal) join reports success.

    It pairs every baseline occurrence — including one WRONG pair the
    neighbourhood signature exposes — on the exact case the guarded join
    refuses. This control existing and passing is the acceptance evidence
    that artifact-at-a-time oracles cannot guard the identity contract.
    """

    baseline, variant = _capture_pair()
    pairs = _naive_ordinal_pairs(baseline, variant)
    baseline_profile, variant_profile = site_profile(baseline), site_profile(variant)
    baseline_real = [
        label
        for label, key in baseline_profile.keys.items()
        if "input" not in key and "output" not in key
    ]
    paired_baseline = {base_label for _, base_label, _ in pairs}
    # The naive join happily pairs every real baseline op ("151/151 preserved").
    assert set(baseline_real) <= paired_baseline
    # ... yet at least one of its pairs disagrees on the graph neighbourhood.
    mismatched = [
        (base_label, variant_label)
        for _, base_label, variant_label in pairs
        if _neighbourhood_signature(baseline, baseline_profile.keys, base_label)
        != _neighbourhood_signature(variant, variant_profile.keys, variant_label)
    ]
    assert mismatched, "the naive join found no wrong pair; the fixture regressed"


@_SMOKE
def test_positive_control_set_membership_blind():
    """MUST-FAIL CONTROL #2: key-set membership reports every key preserved."""

    baseline, variant = _capture_pair()
    baseline_keys = set(site_profile(baseline).keys.values())
    variant_keys = set(site_profile(variant).keys.values())
    assert baseline_keys <= variant_keys  # "preserved", says the blind oracle


@_SMOKE
def test_positive_control_payload_equality_blind():
    """MUST-FAIL CONTROL #3: value equality ALONE passes on the broken join.

    Every naive pair is bit-identical by construction (identity convs), so a
    value-equality-ONLY oracle certifies the wrong pairing. Payload checks
    stay valid ALONGSIDE neighbourhood agreement — this control shows they
    are insufficient alone, not that they are invalid.
    """

    baseline, variant = _capture_pair()
    for _, base_label, variant_label in _naive_ordinal_pairs(baseline, variant):
        assert torch.equal(baseline.ops[base_label].out, variant.ops[variant_label].out)


@pytest.mark.heavy
def test_resnet18_real_weights_insertion_arm():
    """B5's real-checkpoint fixture: resnet18 + a value-preserving in-block insertion."""

    import types

    torchvision_models = pytest.importorskip("torchvision.models")
    weights = torchvision_models.ResNet18_Weights.IMAGENET1K_V1
    baseline_model = torchvision_models.resnet18(weights=weights).eval()
    variant_model = torchvision_models.resnet18(weights=weights).eval()
    block = variant_model.layer1[0]
    block_forward = type(block).forward

    def patched(self, x):
        return self.relu(block_forward(self, x))  # relu after relu: value-preserving

    block.forward = types.MethodType(patched, block)
    x = torch.rand(1, 3, 64, 64)
    baseline = tl.trace(baseline_model, x)
    variant = tl.trace(variant_model, x)
    rows = join_site_profiles(site_profile(baseline), site_profile(variant))
    refused = {key for key, row in rows.items() if not row.joined}
    assert refused == {"s1|layer1/layer1.0/layer1.0.relu|relu||1"}
    verdicts = [row.verdict.value for row in rows.values()]
    assert verdicts.count("corroborated") >= 100
    positional = [key for key, row in rows.items() if row.verdict.value == "positional"]
    assert all(("input" in key) or ("output" in key) for key in positional)
