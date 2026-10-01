"""B13 (persistable collapse plans) + B17 (harvest schema) + B7 pins.

The composition row "collapse plan x edit": save a plan, recapture identical
units (reapplies, zero changed cohorts), insert an op (guarded changed-set
report / strict refusal) — never label failure, ordinal guessing, or silent
replanning. B17: schema lands now, execution refuses per ungated address
class, finer addresses never coarsen. B7 pins: the persisted spec carries
the user's expression as a provenance rider and stamps ``spec_derived``
by the door (predicate-door addressing is derived labels either way).
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest
from test_leverage_div_fixtures import make_insertion_pair

import torchlens as tl
from torchlens._errors import InvalidArgumentError
from torchlens._extraction.harvest_schema import (
    HarvestSpec,
    harvest_block,
    require_executable,
    validate_harvest_block,
)
from torchlens.visualization.collapse_plan import (
    export_collapse_plan,
    reapply_collapse_plan,
)

pytestmark = pytest.mark.smoke

_SAVE_ALL = {"capture": tl.options.CaptureOptions(layers_to_save="all")}


# ---------------------------------------------------------------------------
# B13: persistable collapse plans.
# ---------------------------------------------------------------------------


def test_collapse_plan_roundtrips_and_reapplies_on_clean_recapture():
    baseline_model, _, x = make_insertion_pair()
    first = tl.trace(baseline_model, x, **_SAVE_ALL)
    exported = export_collapse_plan(first, mode="auto")
    payload = json.loads(json.dumps(exported))  # must survive real JSON
    second = tl.trace(baseline_model, x, **_SAVE_ALL)
    report = reapply_collapse_plan(second, payload)
    assert report.applicable
    assert report.changed_cohorts == ()
    assert report.collapsed_addresses  # the plan's units survived the trip


def test_collapse_plan_reapply_refuses_across_insertion():
    """Strict reapply refuses on the join's changed cohorts, teaching."""

    baseline_model, variant_model, x = make_insertion_pair()
    baseline = tl.trace(baseline_model, x, **_SAVE_ALL)
    variant = tl.trace(variant_model, x, **_SAVE_ALL)
    exported = export_collapse_plan(baseline, mode="auto")
    with pytest.raises(InvalidArgumentError) as excinfo:
        reapply_collapse_plan(variant, exported)
    assert excinfo.value.fields["code"] == "collapse_plan_cohorts_changed"
    report = reapply_collapse_plan(variant, exported, strict=False)
    assert not report.applicable
    assert any("relu" in key for key, _ in report.changed_cohorts)
    assert all(
        disposition
        in {"refused_cardinality", "refused_witness", "removed_on_target", "added_on_target"}
        for _, disposition in report.changed_cohorts
    )


def test_collapse_plan_payload_fail_closed():
    baseline_model, _, x = make_insertion_pair()
    trace = tl.trace(baseline_model, x, **_SAVE_ALL)
    with pytest.raises(InvalidArgumentError) as excinfo:
        reapply_collapse_plan(trace, {"wrong": {}})
    assert excinfo.value.fields["code"] == "collapse_plan_payload_invalid"
    exported = export_collapse_plan(trace)
    exported["tl_collapse_plan_v1"]["profile"] = {"torn": True}
    with pytest.raises(InvalidArgumentError) as excinfo:
        reapply_collapse_plan(trace, exported)
    assert excinfo.value.fields["code"] == "collapse_plan_payload_invalid"


# ---------------------------------------------------------------------------
# B17: counterfactual harvest schema.
# ---------------------------------------------------------------------------


def test_harvest_clean_policy_is_explicit_and_roundtrips():
    spec = HarvestSpec()
    block = harvest_block(spec)
    assert block["tl_counterfactual_harvest_v1"]["policy"] == "none"
    assert (
        validate_harvest_block(json.loads(json.dumps(block))["tl_counterfactual_harvest_v1"])
        == spec
    )
    require_executable(spec)  # clean harvests are never gated


def test_harvest_intervened_requires_receipt_and_site_keys():
    with pytest.raises(InvalidArgumentError) as excinfo:
        HarvestSpec(policy="intervened", engine="replay", address_class="module")
    assert excinfo.value.fields["code"] == "harvest_schema_invalid"
    with pytest.raises(InvalidArgumentError) as excinfo:
        HarvestSpec(
            policy="intervened",
            engine="replay",
            address_class="module",
            target_site_keys=("s1|relu|relu||1",),
        )
    assert excinfo.value.fields["code"] == "harvest_schema_invalid"  # missing receipt


def test_harvest_clean_policy_refuses_intervention_fields():
    with pytest.raises(InvalidArgumentError) as excinfo:
        HarvestSpec(policy="none", engine="replay")
    assert excinfo.value.fields["code"] == "harvest_schema_invalid"


def test_harvest_execution_gated_per_address_class_never_coarsens():
    spec = HarvestSpec(
        policy="intervened",
        engine="replay",
        address_class="selection",
        target_site_keys=("s1|relu|relu||1",),
        receipt_digest="deadbeef",
        verification="movers_subset_of_cone",
    )
    with pytest.raises(InvalidArgumentError) as excinfo:
        require_executable(spec)
    assert excinfo.value.fields["code"] == "harvest_address_class_ungated"
    assert "never silently coarsened" in str(excinfo.value)


def test_harvest_block_fail_closed_on_foreign_fields():
    block = harvest_block(HarvestSpec())["tl_counterfactual_harvest_v1"]
    block["extra"] = 1
    with pytest.raises(InvalidArgumentError) as excinfo:
        validate_harvest_block(block)
    assert excinfo.value.fields["code"] == "harvest_schema_invalid"


# ---------------------------------------------------------------------------
# B7 pins (landed by C03; pinned here as the leverage acceptance).
# ---------------------------------------------------------------------------


def test_spec_persistence_carries_expression_and_resolution_disclosure():
    """Predicate-door saves carry the WHERE expression + site-key disclosure."""

    import torchlens.intervention.save as intervention_save

    baseline_model, _, x = make_insertion_pair()
    spec = tl.when(tl.func("relu"), tl.zero_ablate())
    trace = tl.trace(baseline_model, x, intervene=spec)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "spec.json")
        intervention_save.save_intervention(trace, path)
        with open(os.path.join(path, "spec.json")) as spec_file:
            data = json.load(spec_file)
    # The predicate door lowers the fired selector into per-site label
    # targets, so the persisted ADDRESSING is derived (the pinned C03
    # disclosure contract). B7's expression provenance rides BESIDE it.
    assert data["spec_derived"] is True
    blob = json.dumps(data["intervention_spec"])
    assert "spec_where_repr" in blob and "func" in blob  # the user's expression
    for entry in data["target_manifest"]:
        if entry["resolved_status"] == "resolved":
            assert entry["resolved_site_keys"]  # disclosure: structural keys
            assert entry["graph_shape_hash"]  # the resolution digest
