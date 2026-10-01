"""F09 CP3: sumfam items 11, 13-16, 19-20 -- the family re-base, the
SurfaceRegistry, the capability card, PartialTrace typed refusals, and
the family-gate composition rows.

Named rows exercised: D22 (executable-spelling CI: every registry
spelling runs on a real fixture in its declared state), D24 (the
capability card is metadata-only), D25 (typed refusal, never
AttributeError, on the five partial members), D17/item 14 (explain-json
x agent_json overlapping-field parity), D4 (work-metric: no
metadata-only member triggers a payload scan), D3 (count vocabulary:
one value per grain across surfaces), and composition row 10
(non-mutation).
"""

from __future__ import annotations

import pathlib

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens._errors import InvalidArgumentError
from torchlens.report import (
    SURFACE_REGISTRY,
    capability_card,
    factcore,
    registry_entry,
    which_do_i_use,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


class FamilyModel(nn.Module):
    """Small nested model shared by the family gate."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(6, 12), nn.ReLU())
        self.head = nn.Linear(12, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Plain nested forward."""

        return self.head(self.encoder(x))


@pytest.fixture(scope="module")
def family_trace():
    """One shared finished capture for the registry CI."""

    trace = tl.trace(FamilyModel().eval(), torch.randn(2, 6))
    yield trace
    trace.cleanup()


@pytest.fixture()
def partial_trace():
    """A real failed capture recovered as a PartialTrace."""

    class Boom(nn.Module):
        """Fails after one real op."""

        def __init__(self) -> None:
            super().__init__()
            self.a = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run one op then fail."""

            _ = self.a(x)
            raise RuntimeError("planted forward failure")

    with pytest.warns(Warning):
        try:
            tl.trace(Boom(), torch.randn(2, 4))
        except RuntimeError as exc:
            return tl.partial.from_failed_capture(exc)
    raise AssertionError("the planted failure did not raise")


# ---------------------------------------------------------------------------
# SurfaceRegistry (D22) + executable-spelling CI


@pytest.mark.smoke
def test_registry_vocabularies_are_closed() -> None:
    """Every row uses the closed subject/register/invariant/cost vocab."""

    from torchlens.report import COST_CLASSES, INVARIANTS, REGISTERS, SUBJECTS

    assert SURFACE_REGISTRY
    keys = [entry.key for entry in SURFACE_REGISTRY]
    assert len(keys) == len(set(keys))
    for entry in SURFACE_REGISTRY:
        assert entry.subject in SUBJECTS, entry.key
        assert entry.register in REGISTERS, entry.key
        assert entry.cost_class in COST_CLASSES, entry.key
        assert set(entry.invariants) <= set(INVARIANTS), entry.key
        assert entry.answers.endswith(".")
        assert entry.refuses_when


@pytest.mark.smoke
def test_registry_unknown_key_refuses_typed() -> None:
    """Lookups teach the roster."""

    with pytest.raises(InvalidArgumentError) as excinfo:
        registry_entry("nonexistent_surface")
    assert excinfo.value.fields["code"] == "surface_registry_unknown"


@pytest.mark.heavy  # executes every registry spelling; crossed the 7s smoke budget at T58
def test_every_registry_spelling_executes(family_trace) -> None:
    """D22 executable-spelling CI: the hand-written map rotted; this one
    cannot -- every example runs on a real fixture in its declared state."""

    model = FamilyModel().eval()
    x = torch.randn(2, 6)
    namespace = {"tl": tl, "trace": family_trace, "model": model, "x": x}
    for entry in SURFACE_REGISTRY:
        result = eval(entry.example, namespace)  # noqa: S307 -- registry-owned strings
        assert result is not None, entry.key


@pytest.mark.smoke
def test_which_do_i_use_docs_page_is_lockstepped() -> None:
    """Item 20: the docs page IS the generator output (no rot possible)."""

    page = REPO_ROOT / "docs" / "reference" / "report_family.md"
    assert page.read_text(encoding="utf-8") == which_do_i_use()


# ---------------------------------------------------------------------------
# Capability card (D24)


@pytest.mark.smoke
def test_capability_card_is_metadata_only(family_trace, monkeypatch) -> None:
    """D24: building the card never scans, captures, or runs a forward."""

    import torchlens.data_classes._nonfinite as nonfinite_module

    def _no_scan(*_args, **_kwargs):
        raise AssertionError("capability_card triggered a payload scan")

    monkeypatch.setattr(nonfinite_module, "_scan", _no_scan)
    card = family_trace.capability_card()
    text = str(card)
    assert "capability card" in text
    assert any(row.available for row in card.rows)


@pytest.mark.smoke
def test_capability_card_on_partial_names_remedies(partial_trace) -> None:
    """D24 x D25: the card tells a partial-capture user where to go."""

    card = capability_card(partial_trace)
    assert card.object_kind == "partial_trace"
    refused = {row.key: row for row in card.rows if not row.available}
    assert "summary" in refused
    assert "explain" not in refused  # explain degrades, never refuses
    assert "audit" in str(refused["summary"].reason) or "explain" in str(refused["summary"].reason)


# ---------------------------------------------------------------------------
# PartialTrace typed refusals (D25 / item 15)


@pytest.mark.smoke
def test_partial_members_refuse_typed_never_attributeerror(partial_trace) -> None:
    """D25: the five bare AttributeErrors are gone; refusals teach."""

    for member in ("summary", "profile", "to_pandas", "to_agent_json", "output_table"):
        with pytest.raises(InvalidArgumentError) as excinfo:
            getattr(partial_trace, member)()
        assert excinfo.value.fields["code"] == "partial_trace_member_unavailable"
        assert "explain" in str(excinfo.value)
        assert "audit" in str(excinfo.value)


@pytest.mark.smoke
def test_partial_explain_and_audit_still_answer(partial_trace) -> None:
    """The two remedies the refusal names actually work."""

    text = tl.report.explain(partial_trace)
    assert "partial capture" in text
    audit = partial_trace.audit()
    assert audit is not None


# ---------------------------------------------------------------------------
# Machine-core parity (item 14 / D17)


@pytest.mark.smoke
def test_explain_json_agent_json_overlapping_fields_agree(family_trace) -> None:
    """One machine core: every overlapping fact is equal across the two
    projections (the 69/151 bug class regression gate)."""

    explain_json = tl.report.explain(family_trace, format="json")
    agent_json = family_trace.to_agent_json()
    assert explain_json["layer_count"] == agent_json["counts"]["layers"]
    assert explain_json["operation_count"] == agent_json["counts"]["operations"]
    assert explain_json["saved_tensor_count"] == agent_json["counts"]["tensors_saved"]
    assert explain_json["model_class"] == agent_json["capture"]["model_class"]
    assert explain_json["capture_status"] == agent_json["capture"]["capture_status"]
    assert explain_json["has_backward_pass"] == agent_json["capture"]["has_backward_pass"]


@pytest.mark.smoke
def test_bom_is_a_factcore_projection(family_trace) -> None:
    """Item 14/D16: BOM counts come from FactCore, both grains named."""

    core = factcore(family_trace)
    bom = family_trace.bill_of_materials()
    assert bom["graph"]["num_compute_ops"] == core.counts.compute_ops
    assert bom["graph"]["num_alias_rows"] == core.counts.alias_rows
    assert bom["graph"]["num_ops"] == core.counts.tracked_tensor_rows
    assert bom["parameters"]["num_params"] == core.params.total
    assert bom["activations"]["payload_scope"] == "retained_now"


# ---------------------------------------------------------------------------
# Family gate rows (item 19)


@pytest.mark.smoke
def test_count_vocabulary_one_value_per_grain(family_trace) -> None:
    """D3 (composition row 3): each grain has exactly ONE value across
    summary counts, agent_json, BOM, and profile row counts."""

    core = factcore(family_trace)
    agent_json = family_trace.to_agent_json()
    bom = family_trace.bill_of_materials()
    profile_frame = family_trace.profile().to_pandas()
    assert (
        agent_json["counts"]["operations"]
        == bom["graph"]["num_ops"]
        == core.counts.tracked_tensor_rows
        == len(profile_frame)
    )
    assert agent_json["counts"]["compute_ops"] == bom["graph"]["num_compute_ops"]
    assert agent_json["counts"]["layers"] == bom["graph"]["num_layers"] == core.counts.layers
    assert agent_json["counts"]["modules"] == bom["graph"]["num_modules"]


@pytest.mark.smoke
def test_metadata_only_members_trigger_no_scan(family_trace, monkeypatch) -> None:
    """D4 work-metric (composition row 6): no metadata_only member pays for
    a payload scan -- explain, profile, BOM, agent_json, summary included."""

    import torchlens.data_classes._nonfinite as nonfinite_module

    calls: list[str] = []
    real_scan = nonfinite_module._scan

    def _counting_scan(log, kind, stop_at_first):
        calls.append(kind)
        return real_scan(log, kind, stop_at_first)

    monkeypatch.setattr(nonfinite_module, "_scan", _counting_scan)
    tl.report.explain(family_trace)
    family_trace.profile()
    family_trace.bill_of_materials()
    family_trace.to_agent_json()
    tl.report.cost_tree(family_trace)
    tl.report.flops_report(family_trace)
    family_trace.capability_card()
    assert calls == [], f"metadata-only members scanned: {calls}"


@pytest.mark.smoke
def test_family_non_mutation(family_trace) -> None:
    """Composition row 10: no family member mutates params or RNG."""

    rng_before = torch.get_rng_state().clone()
    tl.report.explain(family_trace)
    family_trace.profile()
    family_trace.bill_of_materials()
    family_trace.to_agent_json()
    tl.report.cost_tree(family_trace)
    tl.report.flops_report(family_trace)
    tl.report.backward_status(family_trace)
    tl.report.backward_estimate(family_trace)
    assert torch.equal(torch.get_rng_state(), rng_before)


@pytest.mark.smoke
def test_shared_facts_object_equal_before_render(family_trace) -> None:
    """Composition row 1: every surface reads THE one FactCore instance."""

    assert factcore(family_trace) is factcore(family_trace)
    report = tl.report.flops_report(family_trace)
    core = factcore(family_trace)
    assert report.forward_flops == int(core.compute.partition_total)
    assert report.params_unique == core.params.total


@pytest.mark.smoke
def test_health_evidence_survives_annotations_roundtrip(family_trace) -> None:
    """Composition row 8 (session form): a derived health record persists
    on the annotations channel and is served back as the authority."""

    from torchlens.report import HEALTH_FACTS_ANNOTATIONS_KEY, health_facts

    derived = health_facts(family_trace)  # the explicit door may scan
    assert HEALTH_FACTS_ANNOTATIONS_KEY in family_trace.annotations
    served = health_facts(family_trace, allow_scan=False)
    assert served.basis == "persisted"
    assert served.source_basis == (derived.source_basis or derived.basis)
    assert served.verdict == derived.verdict
