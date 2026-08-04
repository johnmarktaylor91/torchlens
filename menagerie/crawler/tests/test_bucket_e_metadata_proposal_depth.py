"""Regression tests for bucket-E metadata and anti-slop proposal depth."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from menagerie.crawler.constants import MODEL_SCHEMA_VERSION_V3
from menagerie.crawler.identity import hash_bytes, stable_hash
from menagerie.crawler.metadata import (
    MANDATORY_EXTERNAL_FIELDS,
    MetadataValidationError,
    TORCHLENS_DERIVABLE_FIELDS,
    authored_fact_leaves,
    recompute_accepted_identities,
    validate_authored_facts_for_write,
    validate_external_metadata,
    validate_external_metadata_for_write,
)
from menagerie.crawler.proposal import (
    KEYWORD_CLAIM,
    ProposalValidationError,
    model_code_manifest,
    required_metadata_field_checks,
    validate_author_proposal,
)
from menagerie.crawler.tests.conftest import (
    attach_paper_evidence,
    make_author_proposal,
    make_model,
)

NEWLY_GATED_EXTERNAL_FIELDS = (
    "field",
    "subfield",
    "predecessors",
    "family",
    "era",
    "original_framework",
    "run_framework",
    "modes",
)


def _accurate_gate_item() -> dict[str, Any]:
    """Return exhaustive accurate leaf checks for canonical metadata write.

    Returns
    -------
    dict[str, Any]
        Independently accurate metadata gate item.
    """

    return {
        "verdict": "accurate",
        "integrity": {"verdict": "accurate"},
        "field_checks": [
            {"field": f"external_metadata.{field}", "verdict": "accurate"}
            for field in MANDATORY_EXTERNAL_FIELDS
        ],
    }


def _accepted_metadata() -> dict[str, Any]:
    """Return complete accepted external metadata.

    Returns
    -------
    dict[str, Any]
        Schema-valid authored metadata fixture.
    """

    return deepcopy(make_model(accepted=True)["external_metadata"])


@pytest.mark.parametrize("field", NEWLY_GATED_EXTERNAL_FIELDS)
def test_new_external_field_missing_blocks_canonical_write(field: str) -> None:
    """Every newly covered authored field is mandatory on the write path.

    Parameters
    ----------
    field:
        Required external metadata field removed from the proposal.
    """

    metadata = _accepted_metadata()
    del metadata[field]
    with pytest.raises(MetadataValidationError, match="missing mandatory"):
        validate_external_metadata_for_write(metadata, _accurate_gate_item())


@pytest.mark.parametrize("field", NEWLY_GATED_EXTERNAL_FIELDS)
def test_new_external_field_ungated_blocks_canonical_write(field: str) -> None:
    """Every newly covered authored field needs its own accurate leaf check.

    Parameters
    ----------
    field:
        Required external metadata field whose independent check is removed.
    """

    gate_item = _accurate_gate_item()
    gate_item["field_checks"] = [
        check
        for check in gate_item["field_checks"]
        if check["field"] != f"external_metadata.{field}"
    ]
    with pytest.raises(MetadataValidationError, match="ungated mandatory"):
        validate_external_metadata_for_write(_accepted_metadata(), gate_item)


def test_duplicate_external_leaf_check_blocks_canonical_write() -> None:
    """A duplicated leaf cannot hide behind an otherwise exhaustive accurate gate."""

    gate_item = _accurate_gate_item()
    gate_item["field_checks"].append(deepcopy(gate_item["field_checks"][0]))
    with pytest.raises(MetadataValidationError, match="duplicate metadata field check"):
        validate_external_metadata_for_write(_accepted_metadata(), gate_item)


def test_torchlens_derivable_fields_remain_optional_at_write() -> None:
    """Missing TorchLens structural observations do not block authored metadata."""

    metadata = _accepted_metadata()
    assert TORCHLENS_DERIVABLE_FIELDS.isdisjoint(metadata)
    report = validate_external_metadata_for_write(metadata, _accurate_gate_item())
    assert report.derivable_fields_present == frozenset()


def _accurate_authored_gate(facts: dict[str, Any]) -> dict[str, Any]:
    """Return exhaustive gated-claim checks with keyword relevance semantics.

    The covered set is the machine-derived closed claim vocabulary
    (:func:`required_metadata_field_checks`) -- the same list the envelope
    boundary stamps onto every checker item and the same list canonical write
    validates against. It was previously every authored schema leaf (180+ of
    them), which no real proposal could ever ground: the author contract tags
    excerpts at CLAIM granularity, so the only thing that ever satisfied a
    per-leaf gate was this fixture stuffing every leaf path into one excerpt's
    ``supports``. Tagging ~34 claim strings is what a real proposal does.

    Parameters
    ----------
    facts:
        Complete proposed fact tree.

    Returns
    -------
    dict[str, Any]
        Accurate gate item suitable for block-at-write validation.
    """

    fields = required_metadata_field_checks(facts)
    facts["evidence"]["excerpts"][0]["supports"] = list(fields)
    return {
        "verdict": "accurate",
        "integrity": {"verdict": "accurate"},
        "campaign_root_work_id": "campaign-1",
        "terminal_disposition": None,
        "rung_check": {
            "selected_rung": facts["source_resolution"]["rung"],
            "highest_applicable": facts["source_resolution"]["rung"],
            "verdict": "accurate",
            "findings": [],
        },
        "field_checks": [
            {
                "field": field,
                "verdict": "accurate",
                "evidence_ids": [] if field == KEYWORD_CLAIM else ["evidence-1"],
                "checked_source_ids": ["source-1"],
                "reason": "relevant user search term" if field == KEYWORD_CLAIM else "supported",
            }
            for field in fields
        ],
    }


@pytest.mark.parametrize(
    "spelling",
    [
        pytest.param("identity", id="section-name"),
        pytest.param("licenses", id="section-name-second"),
        pytest.param("citation; external_metadata.citation", id="semicolon-joined"),
        pytest.param("citation,dates", id="comma-joined"),
    ],
)
def test_a_section_or_grouped_field_name_is_refused_and_names_the_checker(
    spelling: str,
) -> None:
    """A gate must name ONE claim, and the refusal must say whose defect it is.

    Every spelling here is verbatim from the pilot's real ``m5915``/``m7362``
    metadata gates, which terminalized the model. The engine is right to refuse:
    one verdict cannot carry two claims' provenance, and the exhaustiveness check
    downstream is keyed on the machine-derived claim paths. What it was NOT doing
    is saying so -- the old message, "extraneous authored field check: identity",
    reads as an author emitting a stray field, and misdirected an investigation
    that way.

    The owed spelling moved from "one authored leaf path" to "one claim from the
    envelope item's machine-derived ``required_field_checks``", so the message
    assertion tracks the new contract. The HAZARD is unchanged and is asserted
    directly below the refusal: a grouped or section-named verdict must never
    blanket-bless the claims it swallowed. Two independent layers catch it -- the
    substituted name is refused as outside the required set, and the claim it
    displaced is separately reported as ungated.
    """

    facts = make_author_proposal()["proposed_facts"]
    gate_item = _accurate_authored_gate(facts)
    displaced = gate_item["field_checks"][0]["field"]
    gate_item["field_checks"][0]["field"] = spelling

    with pytest.raises(MetadataValidationError) as refused:
        validate_authored_facts_for_write(facts, gate_item)
    message = str(refused.value)
    assert "checker gate" in message, "the refusal must name the checker as the owner"
    assert repr(spelling) in message, "the offending spelling must be quoted back"
    assert "exactly one claim" in message, "the owed spelling must be stated"
    assert "required_field_checks" in message, "the owed spelling's OWNER must be named"

    # The blanket-blessing hazard itself: even with the offending name dropped
    # rather than refused, the claim it stood in for is still uncovered.
    dropped = deepcopy(gate_item)
    dropped["field_checks"] = dropped["field_checks"][1:]
    with pytest.raises(MetadataValidationError, match="ungated authored facts") as ungated:
        validate_authored_facts_for_write(facts, dropped)
    assert displaced in str(ungated.value), "the swallowed claim must be named as ungated"


def test_a_leaf_path_check_with_the_proposed_facts_prefix_is_still_accepted() -> None:
    """The tolerated prefix must survive the sharper refusal around it."""

    facts = make_author_proposal()["proposed_facts"]
    gate_item = _accurate_authored_gate(facts)
    original = gate_item["field_checks"][0]["field"]
    gate_item["field_checks"][0]["field"] = f"proposed_facts.{original}"

    report = validate_authored_facts_for_write(facts, gate_item)
    assert report is not None


def test_authored_input_dtype_is_gated_and_changes_vet_identity() -> None:
    """An authored dtype collision cannot bypass write gating or vet staleness.

    The gated unit for the input contract is the single ``input_contract`` claim,
    not one check per contract leaf, so the dtype's gate is the contract verdict.
    Both halves of the original hazard still land. No canonical write happens
    unless a checker returned an accurate verdict covering the authored input
    contract -- the dtype rides inside it -- and a checker cannot quietly narrow
    that obligation to a single leaf, because a per-leaf expansion is refused and
    leaves the contract claim ungated. The leaf-name ownership assertions below
    are unaffected: they exercise the schema's collision rules, not gate coverage.
    """

    facts = make_author_proposal()["proposed_facts"]
    dtype_path = "input_contract.args[].dtype"
    contract_claim = "input_contract"
    kwarg = deepcopy(facts["input_contract"]["args"][0])
    kwarg["path"] = "kwargs.image"
    facts["input_contract"]["kwargs"].append(kwarg)
    leaves = authored_fact_leaves(facts, schema_version=MODEL_SCHEMA_VERSION_V3)
    assert leaves[dtype_path] == ["float32"]
    assert leaves["input_contract.kwargs[].dtype"] == ["float32"]

    accurate = _accurate_authored_gate(facts)
    assert contract_claim in required_metadata_field_checks(facts)

    inaccurate = deepcopy(accurate)
    contract_check = next(
        check for check in inaccurate["field_checks"] if check["field"] == contract_claim
    )
    contract_check["verdict"] = "inaccurate"
    with pytest.raises(MetadataValidationError, match="non-accurate authored facts"):
        validate_authored_facts_for_write(facts, inaccurate)

    narrowed = deepcopy(accurate)
    next(
        check for check in narrowed["field_checks"] if check["field"] == contract_claim
    )["field"] = dtype_path
    with pytest.raises(MetadataValidationError, match="per-leaf expansions"):
        validate_authored_facts_for_write(facts, narrowed)

    before = recompute_accepted_identities(
        facts,
        checker_prompt_hash="sha256:" + "1" * 64,
        checker_model="codex",
        checker_version="test",
        schema_version=MODEL_SCHEMA_VERSION_V3,
    )
    changed = deepcopy(facts)
    changed["input_contract"]["args"][0]["dtype"] = "float64"
    after = recompute_accepted_identities(
        changed,
        checker_prompt_hash="sha256:" + "1" * 64,
        checker_model="codex",
        checker_version="test",
        schema_version=MODEL_SCHEMA_VERSION_V3,
    )
    assert after.vet != before.vet


@pytest.mark.parametrize("name", sorted(TORCHLENS_DERIVABLE_FIELDS))
def test_derivable_name_collision_requires_schema_ownership(name: str) -> None:
    """A future path cannot acquire authored or observed policy by leaf-name collision.

    Parameters
    ----------
    name:
        Mechanically derivable leaf name reused below both roots.
    """

    with pytest.raises(MetadataValidationError, match="unowned schema leaf"):
        authored_fact_leaves(
            {
                "authored_future": {name: "source-read claim"},
                "observed": {name: "mechanical fact"},
            },
            schema_version=MODEL_SCHEMA_VERSION_V3,
        )


def test_keywords_use_nonverbatim_relevance_checks_but_remain_gated() -> None:
    """Reasonable search terms pass without excerpts; irrelevant terms still block."""

    facts = make_author_proposal()["proposed_facts"]
    facts["external_metadata"]["keywords"] = ["image feature extractor"]
    gate_item = _accurate_authored_gate(facts)
    report = validate_authored_facts_for_write(facts, gate_item)
    keyword_path = KEYWORD_CLAIM
    assert keyword_path in report.gated_fields

    keyword_check = next(
        check for check in gate_item["field_checks"] if check["field"] == keyword_path
    )
    keyword_check["verdict"] = "inaccurate"
    keyword_check["reason"] = "unrelated to the verified model"
    with pytest.raises(MetadataValidationError, match="non-accurate authored facts"):
        validate_authored_facts_for_write(facts, gate_item)


def test_v3_schema_ownership_gates_mode_divergence_and_resolves_references() -> None:
    """External mode claims are gated while reducer mode results stay outside vet identity.

    The schema-ownership half is untouched: the authored ``external_metadata``
    mode claim is inside vet identity and the reducer's own ``modes`` results are
    outside it. Only the gated UNIT moved -- the checker owes one verdict on the
    ``external_metadata.modes`` claim rather than one per mode leaf -- so the
    authored divergence assertion is still covered by a required check.
    """

    facts = make_author_proposal()["proposed_facts"]
    leaves = authored_fact_leaves(facts, schema_version=MODEL_SCHEMA_VERSION_V3)
    assert "external_metadata.modes.train_eval_divergence" in leaves
    assert "modes.meaningful_modes[]" in leaves
    assert "modes.train_eval_divergence" not in leaves
    assert "modes.divergence_evidence" not in leaves
    gate_item = _accurate_authored_gate(facts)
    gate_item.update(
        {
            "campaign_root_work_id": "campaign-1",
            "terminal_disposition": None,
            "rung_check": {
                "selected_rung": "R1_LIBRARY",
                "highest_applicable": "R1_LIBRARY",
                "verdict": "accurate",
                "findings": [],
            },
        }
    )
    report = validate_authored_facts_for_write(facts, gate_item)
    assert "external_metadata.modes" in report.gated_fields
    # The authored divergence leaf lives under -- and is therefore gated by --
    # that claim; the reducer's own mode results still are not authored facts.
    assert "modes.train_eval_divergence" not in report.gated_fields

    fabricated = deepcopy(gate_item)
    fabricated["field_checks"][0]["evidence_ids"] = ["fabricated"]
    with pytest.raises(MetadataValidationError, match="fabricated evidence"):
        validate_authored_facts_for_write(facts, fabricated)

    extraneous = deepcopy(gate_item)
    extraneous["field_checks"][0]["checked_source_ids"] = ["source-elsewhere"]
    with pytest.raises(MetadataValidationError, match="extraneous sources"):
        validate_authored_facts_for_write(facts, extraneous)


def test_v3_accurate_r4_requires_enumerated_search_attestation() -> None:
    """An accurate R4 result consumes each exact re-executed search assertion."""

    facts = make_author_proposal()["proposed_facts"]
    facts["source_resolution"]["rung"] = "R4_REIMPLEMENT"
    links = facts["source_resolution"]["search_report"]["links_checked"]
    gate_item = _accurate_authored_gate(facts)
    gate_item.update(
        {
            "campaign_root_work_id": "campaign-1",
            "terminal_disposition": None,
            "rung_check": {
                "selected_rung": "R4_REIMPLEMENT",
                "highest_applicable": "R4_REIMPLEMENT",
                "verdict": "accurate",
                "findings": [],
            },
        }
    )
    with pytest.raises(MetadataValidationError, match="re-executed search attestations"):
        validate_authored_facts_for_write(facts, gate_item)
    gate_item["rung_check"]["findings"] = [f"search-attested:{link}" for link in links]
    validate_authored_facts_for_write(facts, gate_item)


def test_empty_keywords_are_rejected() -> None:
    """The required user-search vocabulary cannot be an empty array."""

    metadata = _accepted_metadata()
    metadata["keywords"] = []
    with pytest.raises(MetadataValidationError, match="keywords.*non-empty"):
        validate_external_metadata(metadata)


def _proposal_source(
    tmp_path: Path, text: str, *, extra_supports: tuple[str, ...] = ()
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build an exact-source R1 proposal for focused proposal-gate tests.

    Parameters
    ----------
    tmp_path:
        Isolated source and staged-code directory.
    text:
        Exact controlled source bytes and verbatim excerpt.
    extra_supports:
        Additional self-declared labels attached to the excerpt.

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any]]
        Schema-valid proposal and exact controlled-fetch manifest.
    """

    proposal = make_author_proposal()
    source_path = tmp_path / "source.txt"
    source_path.write_text(text)
    source_hash = hash_bytes(text.encode())
    excerpt = proposal["proposed_facts"]["evidence"]["excerpts"][0]
    excerpt.update(
        {
            "locator": f"bytes:0-{len(text.encode())}",
            "text": text,
            "text_sha256": source_hash,
            "supports": [
                "external_metadata.citation",
                "external_metadata.family",
                *extra_supports,
            ],
            "family_level": True,
        }
    )
    proposal["proposed_facts"]["evidence"]["coverage"].update(
        {
            "all_agent_fields_have_support": True,
            "missing_support": [],
            "family_grounding_complete": True,
        }
    )
    manifest = {
        "sources": [
            {
                "source_id": "source-1",
                "url": "https://example.com/model",
                "revision": "v1",
                "content_sha256": source_hash,
                "cas_path": str(source_path),
                "retrieval_status": "fetched",
            }
        ]
    }
    attach_paper_evidence(proposal, manifest, tmp_path)
    return proposal, manifest


def _validate_focused(proposal: dict[str, Any], manifest: dict[str, Any], tmp_path: Path) -> None:
    """Validate a proposal while focusing mandatory support on its family claim.

    Parameters
    ----------
    proposal:
        Complete author proposal.
    manifest:
        Exact controlled-fetch source manifest.
    tmp_path:
        Allowed staged-code directory.
    """

    validate_author_proposal(
        proposal,
        allowed_model_dir=tmp_path,
        source_manifest=manifest,
        required_claims={"external_metadata.family"},
    )


def test_family_value_verdict_belongs_to_the_checker_not_token_overlap(
    tmp_path: Path,
) -> None:
    """The family label binds provenance; its value verdict is the checker's.

    The deleted token oracle refused any excerpt that failed to token-match the family
    string -- the same mechanism that certified ``country="US"`` on the pronoun "us".
    The deterministic layer now enforces the binding (an excerpt must name the claim)
    and leaves entailment to the per-leaf accuracy checker, which remains mandatory
    for every external field.
    """

    from menagerie.crawler.proposal import CHECKER_EVALUATED_CLAIMS

    assert "external_metadata.family" in CHECKER_EVALUATED_CLAIMS
    proposal, manifest = _proposal_source(
        tmp_path, "Example Model was published at TestConf in 2020 about weather forecasting."
    )
    _validate_focused(proposal, manifest, tmp_path)
    unbound, unbound_manifest = _proposal_source(
        tmp_path, "Example Model was published at TestConf in 2020 about weather forecasting."
    )
    for excerpt in unbound["proposed_facts"]["evidence"]["excerpts"]:
        excerpt["supports"] = [
            support for support in excerpt["supports"] if support != "external_metadata.family"
        ]
    with pytest.raises(ProposalValidationError, match="ungrounded claim categories"):
        _validate_focused(unbound, unbound_manifest, tmp_path)


def test_excerpt_genuinely_supporting_proposed_value_passes(tmp_path: Path) -> None:
    """Literal text naming the proposed family satisfies deterministic binding."""

    proposal, manifest = _proposal_source(
        tmp_path,
        "Example Model introduced the ExampleNet family at TestConf in 2020.",
    )
    _validate_focused(proposal, manifest, tmp_path)


def _stage_typed_adapter(proposal: dict[str, Any], tmp_path: Path, code: str, *, rung: str) -> None:
    """Convert an R1 fixture into a schema-valid typed staged-code rung.

    Parameters
    ----------
    proposal:
        Proposal modified in place.
    tmp_path:
        Adapter staging directory.
    code:
        Typed adapter source.
    rung:
        Selected canonical source rung.
    """

    code_path = tmp_path / "adapter.py"
    code_path.write_text(code)
    facts = proposal["proposed_facts"]
    resolution = facts["source_resolution"]
    resolution["rung"] = rung
    selected_index = ("R1_LIBRARY", "R2_VENDOR", "R3_PORT", "R4_REIMPLEMENT").index(rung)
    resolution["attempted_rungs"] = [
        {
            "rung": attempted,
            "result": "selected" if attempted == rung else "unavailable",
            "reason_code": "documented-search",
            "evidence_ids": ["evidence-1"],
        }
        for attempted in ("R1_LIBRARY", "R2_VENDOR", "R3_PORT", "R4_REIMPLEMENT")[
            : selected_index + 1
        ]
    ]
    implementation = facts["implementation"]
    implementation.update(
        {
            "recipe_type": "typed-adapter" if rung == "R2_VENDOR" else "reimplementation",
            "code_path": "adapter.py",
            "code_sha256": hash_bytes(code.encode()),
            "builder_symbol": "build_model",
            "dummy_call_symbol": "make_dummy_call",
            "library_recipe": None,
        }
    )
    code_manifest = [dict(row) for row in model_code_manifest(code_path, tmp_path)]
    implementation["code_manifest"] = code_manifest
    proposal["verified_hashes"]["code"] = hash_bytes(code.encode())
    proposal["verified_hashes"]["code_manifest"] = stable_hash(code_manifest)


def _source_map(source_locator: str = "bytes:0-82") -> list[dict[str, Any]]:
    """Return one exact source-to-adapter mapping.

    Parameters
    ----------
    source_locator:
        Exact verified excerpt locator in the controlled source bytes.

    Returns
    -------
    list[dict[str, Any]]
        Complete source-map row.
    """

    return [
        {
            "material_item": "upstream forward implementation",
            "source_id": "source-1",
            "source_locator": source_locator,
            "evidence_ids": ["evidence-1"],
            "code_path": "adapter.py",
            "code_locator": "lines 1-5",
            "disposition": "wrapped-exactly",
        }
    ]


def test_generic_sequential_mlp_as_exotic_family_trips_tripwire(tmp_path: Path) -> None:
    """A plain Sequential MLP cannot masquerade as the named ExampleNet family."""

    proposal, manifest = _proposal_source(
        tmp_path,
        "Example Model introduced the complete ExampleNet architecture at TestConf in 2020.",
        extra_supports=("implementation.architecture",),
    )
    code = (
        "import torch.nn as nn\n\n"
        "class ExampleNet(nn.Module):\n"
        "    def __init__(self) -> None:\n"
        "        super().__init__()\n"
        "        self.layers = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 2))\n\n"
        "    def forward(self, value: object) -> object:\n"
        "        return self.layers(value)\n\n"
        "def build_model() -> object:\n"
        "    return ExampleNet()\n\n"
        "def make_dummy_call(seed: int, device: str) -> tuple[tuple[()], dict[str, object]]:\n"
        "    return (), {}\n"
    )
    _stage_typed_adapter(proposal, tmp_path, code, rung="R4_REIMPLEMENT")
    facts = proposal["proposed_facts"]
    facts["source_resolution"]["sources"][0].update({"role": "introducing-paper", "kind": "paper"})
    facts["implementation"]["source_to_code_map"] = _source_map()
    facts["fidelity"].update(
        {"required": True, "reason": "source-faithful architecture", "current": False}
    )
    with pytest.raises(ProposalValidationError, match="structural slop tripwire"):
        _validate_focused(proposal, manifest, tmp_path)


def _r2_proposal(tmp_path: Path, *, bound: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build an R2 proposal with optional exact byte/map binding.

    Parameters
    ----------
    tmp_path:
        Isolated source and adapter directory.
    bound:
        Whether exact upstream bytes and source-map rows are supplied.

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any]]
        R2 proposal and exact source manifest.
    """

    proposal, manifest = _proposal_source(
        tmp_path,
        "Example Model introduced the official upstream ExampleNet architecture in 2020.",
        extra_supports=("implementation.architecture",),
    )
    code = (
        "def build_model() -> object:\n"
        "    return object()\n\n"
        "def make_dummy_call(seed: int, device: str) -> tuple[tuple[()], dict[str, object]]:\n"
        "    return (), {}\n"
    )
    _stage_typed_adapter(proposal, tmp_path, code, rung="R2_VENDOR")
    implementation = proposal["proposed_facts"]["implementation"]
    if bound:
        proposal["proposed_facts"]["source_resolution"]["sources"][0]["content_sha256"] = manifest[
            "sources"
        ][0]["content_sha256"]
        implementation["upstream_files"] = [
            {
                "source_id": "source-1",
                "path": "upstream/model.py",
                "sha256": manifest["sources"][0]["content_sha256"],
                "use": "exact upstream model source",
            }
        ]
        excerpt = proposal["proposed_facts"]["evidence"]["excerpts"][0]
        implementation["source_to_code_map"] = _source_map(excerpt["locator"])
    return proposal, manifest


def test_r2_without_exact_source_binding_is_refused(tmp_path: Path) -> None:
    """A named vendor adapter alone cannot earn the R2 rung."""

    proposal, manifest = _r2_proposal(tmp_path, bound=False)
    with pytest.raises(ProposalValidationError, match="exact mirrored upstream files"):
        _validate_focused(proposal, manifest, tmp_path)


def test_r2_with_exact_source_bytes_and_map_passes(tmp_path: Path) -> None:
    """R2 passes when adapter code maps to hash-matched controlled source bytes."""

    proposal, manifest = _r2_proposal(tmp_path, bound=True)
    _validate_focused(proposal, manifest, tmp_path)


def test_r2_checked_candidate_withheld_from_fetch_is_rejected(tmp_path: Path) -> None:
    """R2 search coverage cannot omit a checked implementation candidate."""

    proposal, manifest = _r2_proposal(tmp_path, bound=True)
    proposal["proposed_facts"]["source_resolution"]["search_report"]["links_checked"].append(
        "https://code.example.org/alternate-example-net"
    )
    with pytest.raises(ProposalValidationError, match="checked-link coverage gap"):
        _validate_focused(proposal, manifest, tmp_path)


_UNTYPED_VENDORED_CODE = (
    "import torch.nn as nn\n\n"
    "class ExampleNet(nn.Module):\n"
    "    def __init__(self):\n"
    "        super().__init__()\n"
    "        self.conv = nn.Conv2d(3, 8, 3)\n"
    "        self.act = nn.ReLU()\n"
    "        self.head = nn.Linear(8, 2)\n\n"
    "    def forward(self, value):\n"
    "        hidden = self.act(self.conv(value))\n"
    "        return self.head(hidden.mean(dim=(2, 3)))\n"
)

_TYPED_VENDOR_ADAPTER = (
    "from __future__ import annotations\n\n"
    "import vendored_net\n\n"
    "def build_model() -> object:\n"
    "    return vendored_net.ExampleNet()\n\n"
    "def make_dummy_call(seed: int, device: str) -> tuple[tuple[()], dict[str, object]]:\n"
    "    return (), {}\n"
)


def _r2_vendored_proposal(
    tmp_path: Path,
    *,
    frozen_code: str = _UNTYPED_VENDORED_CODE,
    staged_code: str | None = None,
    adapter_code: str = _TYPED_VENDOR_ADAPTER,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build an R2 proposal whose adapter imports a really staged vendored file.

    Parameters
    ----------
    tmp_path:
        Isolated source and adapter directory.
    frozen_code:
        Exact upstream implementation bytes frozen in the controlled manifest.
    staged_code:
        Bytes actually staged as ``vendored_net.py``; defaults to the frozen
        bytes, i.e. a verbatim vendoring.
    adapter_code:
        Staged entry-point adapter source.

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any]]
        R2 proposal and exact source manifest.
    """

    proposal, manifest = _r2_proposal(tmp_path, bound=True)
    facts = proposal["proposed_facts"]
    frozen_bytes = frozen_code.encode()
    frozen_digest = hash_bytes(frozen_bytes)
    impl_cas = tmp_path / "impl-source.py"
    impl_cas.write_text(frozen_code)
    declared_impl = dict(facts["source_resolution"]["sources"][0])
    declared_impl.update(
        {
            "source_id": "source-impl",
            "role": "implementation",
            "kind": "repository",
            "url": "https://example.com/upstream/vendored_net.py",
            "locator": "vendored_net.py",
            "content_sha256": frozen_digest,
            "byte_count": len(frozen_bytes),
        }
    )
    facts["source_resolution"]["sources"].append(dict(declared_impl))
    manifest["sources"].append({**declared_impl, "cas_path": str(impl_cas)})
    manifest["manifest_sha256"] = stable_hash(manifest["sources"])
    proposal["verified_hashes"]["source_manifest"] = manifest["manifest_sha256"]
    facts["evidence"]["excerpts"].append(
        {
            "evidence_id": "evidence-impl",
            "source_id": "source-impl",
            "locator": f"bytes:0-{len(frozen_bytes)}",
            "text": frozen_code,
            "text_sha256": frozen_digest,
            "supports": ["implementation.architecture"],
            "family_level": False,
            "disposition": "supporting",
            "license_disposition": "short-excerpt-committed",
        }
    )
    (tmp_path / "vendored_net.py").write_text(
        frozen_code if staged_code is None else staged_code
    )
    (tmp_path / "adapter.py").write_text(adapter_code)
    implementation = facts["implementation"]
    implementation["code_sha256"] = hash_bytes(adapter_code.encode())
    implementation["upstream_files"] = [
        {
            "source_id": "source-impl",
            "path": "vendored_net.py",
            "sha256": frozen_digest,
            "use": "vendored verbatim as the model definition module",
        }
    ]
    implementation["source_to_code_map"] = [
        {
            "material_item": "complete upstream forward architecture",
            "source_id": "source-impl",
            "source_locator": f"bytes:0-{len(frozen_bytes)}",
            "evidence_ids": ["evidence-impl"],
            "code_path": "vendored_net.py",
            "code_locator": "lines 1-12",
            "disposition": "wrapped-exactly",
        }
    ]
    code_manifest = [dict(row) for row in model_code_manifest(tmp_path / "adapter.py", tmp_path)]
    implementation["code_manifest"] = code_manifest
    proposal["verified_hashes"]["code"] = implementation["code_sha256"]
    proposal["verified_hashes"]["code_manifest"] = stable_hash(code_manifest)
    return proposal, manifest


@pytest.mark.smoke
def test_r2_verbatim_vendored_member_may_be_untyped(tmp_path: Path) -> None:
    """The m8189 shape is satisfiable: verbatim upstream bytes need no annotations.

    The vendored module is unannotated, exactly as real upstream PyTorch source
    is, and its staged bytes reproduce the frozen manifest digest byte for
    byte. The typed entry adapter plus the byte-proven vendored member now pass
    the gate; the typing rule was a legibility constraint that made R2
    structurally unreachable, never a safety one.
    """

    proposal, manifest = _r2_vendored_proposal(tmp_path)
    _validate_focused(proposal, manifest, tmp_path)


@pytest.mark.smoke
def test_r2_modified_vendored_bytes_restore_the_typing_requirement(tmp_path: Path) -> None:
    """One changed byte voids the exemption: the file is author-edited code now."""

    proposal, manifest = _r2_vendored_proposal(
        tmp_path, staged_code=_UNTYPED_VENDORED_CODE.replace("8, 2", "8, 3")
    )
    with pytest.raises(ProposalValidationError, match="must be fully typed"):
        _validate_focused(proposal, manifest, tmp_path)


@pytest.mark.smoke
def test_r2_vendored_bytes_never_escape_the_safety_checks(tmp_path: Path) -> None:
    """A frozen source carrying eval() is refused even as a byte-proven vendoring.

    The exemption covers the LEGIBILITY rule only. If the upstream bytes
    themselves contain dynamic execution, byte-fidelity to the frozen manifest
    changes nothing: the safety AST check runs on every closure member.
    """

    hostile = (
        "from __future__ import annotations\n\n"
        "def build_upstream() -> object:\n"
        "    return eval('object()')\n"
    )
    proposal, manifest = _r2_vendored_proposal(tmp_path, frozen_code=hostile)
    with pytest.raises(ProposalValidationError, match="forbidden dynamic execution"):
        _validate_focused(proposal, manifest, tmp_path)


@pytest.mark.smoke
def test_r2_entry_adapter_is_never_exempt_from_typing(tmp_path: Path) -> None:
    """Binding the entry point itself as an upstream file earns no exemption.

    The adapter is author-written by definition; even bytes that reproduce a
    frozen source digest do not lift the annotation contract from it.
    """

    untyped_adapter = (
        "import vendored_net\n\n"
        "def build_model():\n"
        "    return vendored_net.ExampleNet()\n\n"
        "def make_dummy_call(seed, device):\n"
        "    return (), {}\n"
    )
    proposal, manifest = _r2_vendored_proposal(tmp_path, adapter_code=untyped_adapter)
    implementation = proposal["proposed_facts"]["implementation"]
    adapter_digest = hash_bytes(untyped_adapter.encode())
    adapter_cas = tmp_path / "adapter-source.py"
    adapter_cas.write_text(untyped_adapter)
    declared = dict(proposal["proposed_facts"]["source_resolution"]["sources"][-1])
    declared.update(
        {
            "source_id": "source-adapter",
            "locator": "adapter.py",
            "content_sha256": adapter_digest,
            "byte_count": len(untyped_adapter.encode()),
        }
    )
    proposal["proposed_facts"]["source_resolution"]["sources"].append(declared)
    manifest["sources"].append({**declared, "cas_path": str(adapter_cas)})
    manifest["manifest_sha256"] = stable_hash(manifest["sources"])
    proposal["verified_hashes"]["source_manifest"] = manifest["manifest_sha256"]
    implementation["upstream_files"].append(
        {
            "source_id": "source-adapter",
            "path": "adapter.py",
            "sha256": adapter_digest,
            "use": "attempted exemption of the entry adapter itself",
        }
    )
    with pytest.raises(ProposalValidationError, match="must be fully typed"):
        _validate_focused(proposal, manifest, tmp_path)


@pytest.mark.parametrize(
    "phrase",
    [
        "an approximate implementation",
        "a toy model substitute",
        "a knowingly simplified version",
        "a placeholder for the real architecture",
        "a stand-in for the source model",
    ],
)
def test_broadened_approximation_language_is_refused(tmp_path: Path, phrase: str) -> None:
    """Expanded explicit approximation vocabulary blocks acceptance.

    Parameters
    ----------
    tmp_path:
        Isolated source directory.
    phrase:
        Explicit approximation admission inserted into authored prose.
    """

    proposal, manifest = _proposal_source(
        tmp_path, "Example Model introduced the ExampleNet family at TestConf in 2020."
    )
    # Implementation-fidelity surfaces stay armed; descriptions are no longer scanned
    # because dozens of real roster models are *named* with this vocabulary.
    proposal["proposed_facts"]["source_resolution"]["decision"] = f"ExampleNet is {phrase}."
    with pytest.raises(ProposalValidationError, match="forbidden approximation language"):
        _validate_focused(proposal, manifest, tmp_path)


def test_divergent_top_level_citation_is_refused(tmp_path: Path) -> None:
    """An unvetted public citation cannot differ from checked external metadata."""

    proposal, manifest = _proposal_source(
        tmp_path, "Example Model introduced the ExampleNet family at TestConf in 2020."
    )
    proposal["proposed_facts"]["citation"]["title"] = "Different Unvetted Work"
    with pytest.raises(ProposalValidationError, match="top-level citation differs"):
        _validate_focused(proposal, manifest, tmp_path)
