"""F03 ledger memo item 5: bundle.vary — one explicit edit or identity per member.

Pins: complete coverage by default (typed preflight refusals BEFORE any
mutation), explicit None identity distinguishable from unmentioned, sugar
normalization, whole-mapping donor-group normalization (one reused
SamplingPlan = one donor group across members), disclosed no-rollback
partial outcomes, mutator return, and the EVENT audit rows the experiment
layer writes on each varied member.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.errors.episode import BundleExperimentError
from torchlens.intervention.audit import event_audit_rows


class _Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.linear(x))


def _pair() -> tl.Bundle:
    torch.manual_seed(9)
    x = torch.randn(2, 3)

    def _capture() -> tl.Trace:
        return tl.trace(
            _Tiny(), x, capture=tl.options.CaptureOptions(intervention_ready=True)
        ).fork()

    return tl.Bundle({"a": _capture(), "b": _capture()}, baseline="a")


def test_vary_applies_one_spec_per_member_and_returns_self() -> None:
    bundle = _pair()
    out = bundle.vary(
        {
            "a": None,
            "b": tl.when(tl.func("relu"), tl.scale(0.0)),
        }
    )
    assert out is bundle
    assert bundle["a"].intervention_audit == []
    assert bundle["b"].intervention_audit, "varied member has no audit evidence"
    # The experiment layer wrote the canonical EVENT row on the varied member.
    events = event_audit_rows(bundle["b"])
    assert events and events[-1]["door"] == "do"
    operation = bundle.operations[-1]
    assert operation.kind == "vary"
    assert operation.params["outcomes"] == {"a": "identity", "b": "completed"}
    assert operation.params["spec_digests"]["b"]
    assert bundle.member_construction["b"]["origin"] == "varied"
    assert bundle.member_construction["a"]["origin"] == "constructed"


def test_vary_sugar_pair_normalizes_to_spec() -> None:
    bundle = _pair()
    bundle.vary({"a": None, "b": (tl.func("relu"), tl.scale(2.0))})
    assert bundle.operations[-1].params["outcomes"]["b"] == "completed"


def test_vary_coverage_refusals_fire_before_any_mutation() -> None:
    bundle = _pair()
    with pytest.raises(BundleExperimentError) as excinfo:
        bundle.vary({"a": tl.when(tl.func("relu"), tl.scale(0.5))})
    assert excinfo.value.fields["code"] == "vary_coverage_incomplete"
    assert excinfo.value.fields["missing_members"] == ["b"]
    # Nothing mutated, nothing recorded.
    assert bundle["a"].intervention_audit == []
    assert bundle.operations == ()

    with pytest.raises(BundleExperimentError) as excinfo:
        bundle.vary({"ghost": None, "a": None, "b": None})
    assert excinfo.value.fields["code"] == "vary_member_unknown"

    with pytest.raises(BundleExperimentError) as excinfo:
        bundle.vary({"a": None, "b": "not a spec"})
    assert excinfo.value.fields["code"] == "vary_mapping_invalid"
    assert bundle.operations == ()


def test_vary_subset_requires_explicit_unmentioned() -> None:
    bundle = _pair()
    bundle.vary({"b": tl.when(tl.func("relu"), tl.scale(0.5))}, unmentioned="unchanged")
    operation = bundle.operations[-1]
    # Explicit None identity and unmentioned are distinguishable records.
    assert operation.params["outcomes"] == {"a": "unmentioned", "b": "completed"}
    assert operation.params["unmentioned"] == "unchanged"


def test_vary_duplicate_after_normalization_refuses() -> None:
    bundle = _pair()
    trace_b = bundle["b"]
    with pytest.raises(BundleExperimentError) as excinfo:
        bundle.vary({"a": None, "b": None, trace_b: None})
    assert excinfo.value.fields["code"] == "vary_member_duplicate"


@pytest.mark.smoke
def test_vary_partial_failure_disclosed_no_rollback() -> None:
    bundle = _pair()
    good = tl.when(tl.func("relu"), tl.scale(0.0))
    bad = tl.when(tl.label("nonexistent_site_9_9"), tl.scale(0.0))
    with pytest.raises(BundleExperimentError) as excinfo:
        bundle.vary({"a": good, "b": bad})
    fields = excinfo.value.fields
    assert fields["code"] == "vary_partial_failure"
    assert fields["material_action_completed"] is True
    assert fields["outcomes"]["a"] == "completed"
    assert fields["outcomes"]["b"] == "failed"
    # The completed member keeps its edit (no rollback claimed) and the
    # chronology row records the partial outcome.
    assert bundle["a"].intervention_audit
    assert bundle.operations[-1].params["outcomes"]["b"] == "failed"


@pytest.mark.smoke
def test_vary_shared_plan_keeps_one_donor_group() -> None:
    bundle = _pair()
    population = tl.intervention.reference(
        [torch.randn(2, 3) for _ in range(4)],
        origin="unit-test population, seeded randn rows",
    )
    plan = tl.intervention.sample_from(population, seed=7)
    bundle.vary(
        {
            "a": (tl.func("relu"), tl.patch_from(plan)),
            "b": (tl.func("relu"), tl.patch_from(plan)),
        }
    )
    from torchlens.intervention.stochastic import sampling_records

    groups = set()
    for name in ("a", "b"):
        rows = sampling_records(bundle[name])
        assert rows, f"member {name} has no sampling records"
        groups.update(row["donor_group_id"] for row in rows)
    assert len(groups) == 1, f"one reused plan must keep ONE donor group, got {groups}"
