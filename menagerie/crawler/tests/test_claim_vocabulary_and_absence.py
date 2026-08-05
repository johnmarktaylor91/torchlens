"""Claim-category vocabulary and typed-absence coverage.

The c1-mech author rung refused 8 of 8 proposals with ``ungrounded claim categories``.
Coverage is NOMINAL -- ``supported.update(supports)`` over raw strings -- so satisfying
it costs nothing; the authors failed because the required vocabulary was a closed
exact-match set that existed only in Python and was invisible to them. These tests pin
the three fixes and, more importantly, pin the ways the gate must STILL refuse.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from menagerie.crawler.constants import AUTHOR_PROMPT_NAME, MODEL_SCHEMA_VERSION_V3
from menagerie.crawler.evidence import EvidenceValidationError, validate_evidence
from menagerie.crawler.identity import hash_bytes, stable_hash
from menagerie.crawler.metadata import (
    AVAILABILITY_BASES,
    AVAILABILITY_KEYS,
    AVAILABILITY_STATUSES,
)
from menagerie.crawler.proposal import (
    AUTHORED_LEAF_COVERAGE_LANES,
    CLAIM_VOCABULARY_BEGIN,
    CLAIM_VOCABULARY_END,
    CONDITIONAL_GATED_CLAIMS,
    DEFAULT_GATED_CLAIMS,
    EMPTIABLE_CLAIMS,
    KEYWORD_CLAIM,
    ProposalValidationError,
    _validate_availability_record,
    gated_claim_vocabulary_block,
    validate_author_proposal,
)
from menagerie.crawler.schema import SchemaOwner, load_schema, schema_owner_paths
from menagerie.crawler.tests.conftest import attach_paper_evidence, make_author_proposal
from menagerie.crawler.tools import render_claim_vocabulary

_CRAWLER_ROOT = Path(__file__).resolve().parents[1]
_PROMPT_PATH = _CRAWLER_ROOT / "prompts" / f"{AUTHOR_PROMPT_NAME}.txt"


# --------------------------------------------------------------------------------
# 1. The vocabulary is derived from the enforcing code, not hand-copied.
# --------------------------------------------------------------------------------


def test_prompt_claim_vocabulary_is_derived_from_the_enforcing_code() -> None:
    """The shipped prompt's generated region equals the code-derived block.

    A hand-maintained list is what built this wall: the required strings changed in
    Python and the prompt never followed. This assertion is the drift guard -- adding
    or removing a gated claim without regenerating the prompt fails here.
    """

    prompt = _PROMPT_PATH.read_text(encoding="utf-8")
    begin = prompt.index(CLAIM_VOCABULARY_BEGIN)
    end = prompt.index(CLAIM_VOCABULARY_END) + len(CLAIM_VOCABULARY_END)
    assert prompt[begin:end] == gated_claim_vocabulary_block()


def test_render_tool_reports_the_shipped_prompt_as_current() -> None:
    """``render_claim_vocabulary`` with no ``--write`` is a clean drift check."""

    assert render_claim_vocabulary.main(["--prompt", str(_PROMPT_PATH)]) == 0


def test_render_tool_rewrites_a_stale_region_and_detects_it_first(tmp_path: Path) -> None:
    """A stale region is REPORTED before it is rewritten, never silently accepted."""

    stale = _PROMPT_PATH.read_text(encoding="utf-8")
    begin = stale.index(CLAIM_VOCABULARY_BEGIN)
    end = stale.index(CLAIM_VOCABULARY_END)
    corrupted = stale[: begin + len(CLAIM_VOCABULARY_BEGIN)] + "\n  stale.claim\n" + stale[end:]
    scratch = tmp_path / "prompt.txt"
    scratch.write_text(corrupted, encoding="utf-8")

    assert render_claim_vocabulary.main(["--prompt", str(scratch)]) == 1
    assert render_claim_vocabulary.main(["--prompt", str(scratch), "--write"]) == 0
    assert render_claim_vocabulary.main(["--prompt", str(scratch)]) == 0
    assert scratch.read_text(encoding="utf-8") == stale


def test_every_gated_claim_string_is_stated_verbatim_in_the_prompt() -> None:
    """Each required string appears in the prompt exactly as the gate matches it."""

    prompt = _PROMPT_PATH.read_text(encoding="utf-8")
    missing = sorted(claim for claim in DEFAULT_GATED_CLAIMS if claim not in prompt)
    assert missing == []


@pytest.mark.parametrize(
    "schema_version",
    ["menagerie.crawler.author-proposal.v3", "menagerie.crawler.model.v3"],
)
def test_both_schemas_declare_exactly_the_enforced_availability_keys(
    schema_version: str,
) -> None:
    """The availability register is one closed key set, declared in three places.

    ``AVAILABILITY_KEYS`` in code, the proposal schema, and the model schema must agree;
    a claim that can declare an absence in one but not another is a silent refusal
    waiting to happen.
    """

    schema = load_schema(schema_version)
    declared = set(schema["$defs"]["availability_claims"]["properties"])
    assert declared == set(AVAILABILITY_KEYS)


@pytest.mark.parametrize(
    "schema_version",
    ["menagerie.crawler.author-proposal.v3", "menagerie.crawler.model.v3"],
)
def test_both_schemas_declare_exactly_the_enforced_availability_statuses(
    schema_version: str,
) -> None:
    """The status vocabulary is closed and identical across code and both schemas."""

    schema = load_schema(schema_version)
    declared = set(schema["$defs"]["availability_claim"]["properties"]["status"]["enum"])
    assert declared == set(AVAILABILITY_STATUSES)


@pytest.mark.parametrize(
    "schema_version",
    ["menagerie.crawler.author-proposal.v3", "menagerie.crawler.model.v3"],
)
def test_both_schemas_declare_exactly_the_enforced_availability_bases(
    schema_version: str,
) -> None:
    """The basis vocabulary is a declared ``enum``, not prose promising one.

    ``status`` was always an enum and no author has ever failed on it. ``basis`` was
    ``nonempty_string`` with the description "Mandatory closed-vocabulary basis for the
    disposition" -- a schema that TELLS the author the vocabulary is closed and then
    declines to say what is in it, leaving the eight members visible only in Python.
    That is precisely the wall the generated claim vocabulary exists to prevent, and it
    killed two of twenty models in the 2026-08-05 rung on their one and only attempt
    (``bounded-source-read`` on m8245, ``bounded-frozen-source-read`` on m9617), both of
    which were honest descriptions of reading the frozen sources where the enum wanted
    ``search-exhausted``. The enum is now declared where a schema-reading author looks.
    """

    schema = load_schema(schema_version)
    declared = schema["$defs"]["availability_claim"]["properties"]["basis"]["enum"]
    assert set(declared) == set(AVAILABILITY_BASES)
    assert len(declared) == len(set(declared))


def test_a_non_canonical_basis_refusal_names_the_vocabulary_it_wanted() -> None:
    """The refusal must carry the closed set, not only the rejected spelling.

    A terminal refusal is read by a human, never by the author that caused it, so the
    message is the only artifact that can explain which of eight strings was wanted.
    """

    proposal = make_author_proposal()
    facts = proposal["proposed_facts"]
    facts["external_metadata"]["country"] = None
    facts["external_metadata"].setdefault("availability", {})["country"] = {
        "status": "not-found-after-search",
        "values": [],
        "basis": "bounded-frozen-source-read",
        "evidence": [],
    }

    with pytest.raises(ProposalValidationError) as excinfo:
        _validate_availability_record(
            "external_metadata.country",
            facts["external_metadata"]["availability"]["country"],
            None,
            True,
            facts,
            frozenset(),
        )
    message = str(excinfo.value)
    assert "bounded-frozen-source-read" in message
    assert "search-exhausted" in message
    for basis in AVAILABILITY_BASES:
        assert basis in message


@pytest.mark.parametrize(
    "schema_version",
    ["menagerie.crawler.author-proposal.v3", "menagerie.crawler.model.v3"],
)
def test_the_dotted_availability_key_cannot_collide_with_a_nested_object(
    schema_version: str,
) -> None:
    """``taxonomy.novel_ops`` is one flat key, and nothing can spell it two ways.

    The ownership walker joins instance keys with a dot, so a NESTED
    ``availability.taxonomy.novel_ops`` object would normalize to the same path as the
    flat dotted key. A closed key set with no bare ``taxonomy`` member is what keeps
    that ambiguity impossible.
    """

    claims = load_schema(schema_version)["$defs"]["availability_claims"]
    assert claims["additionalProperties"] is False
    assert "taxonomy" not in claims["properties"]
    assert "taxonomy.novel_ops" in claims["properties"]


def test_schema_publishes_exactly_the_enforced_gated_vocabulary() -> None:
    """The machine-discoverable enum equals the set the gate enforces."""

    schema = load_schema("menagerie.crawler.author-proposal.v3")
    published = schema["$defs"]["gated_claim_category"]["enum"]
    assert sorted(published) == sorted(DEFAULT_GATED_CLAIMS)
    assert len(published) == len(set(published))


# --------------------------------------------------------------------------------
# 2. Nominal coverage: exact equality, and NO prefix roll-up.
# --------------------------------------------------------------------------------


def _multi_source_evidence(tmp_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a two-source evidence block with DISTINCT bytes per source.

    Two sources, not one: a single frozen source hides width-dependent indexing bugs.
    The two texts differ so that binding an excerpt to the wrong source cannot pass by
    coincidence, and so a placeholder in either slot cannot match the other.
    """

    paper_text = "PaperNet was introduced at TestConf 2020 by A. Author at Example Lab."
    code_text = "class PaperNet(nn.Module):  # reference PyTorch implementation, Apache-2.0"
    assert paper_text != code_text
    paper_path = tmp_path / "paper.txt"
    code_path = tmp_path / "code.py"
    paper_path.write_text(paper_text)
    code_path.write_text(code_text)

    evidence = {
        "excerpts": [
            {
                "evidence_id": "ev-paper",
                "source_id": "paper-1",
                "locator": f"bytes:0-{len(paper_text.encode())}",
                "text": paper_text,
                "text_sha256": hash_bytes(paper_text.encode()),
                "supports": ["external_metadata.year"],
                "family_level": True,
                "disposition": "supporting",
                "license_disposition": "quoted-under-fair-use",
            },
            {
                "evidence_id": "ev-code",
                "source_id": "code-1",
                "locator": f"bytes:0-{len(code_text.encode())}",
                "text": code_text,
                "text_sha256": hash_bytes(code_text.encode()),
                "supports": ["external_metadata.license"],
                "family_level": False,
                "disposition": "supporting",
                "license_disposition": "quoted-under-fair-use",
            },
        ],
        "coverage": {
            "all_agent_fields_have_support": True,
            "missing_support": [],
            "family_grounding_complete": True,
        },
    }
    manifest: dict[str, Any] = {
        "sources": [
            {
                "source_id": "paper-1",
                "url": "https://example.com/paper",
                "revision": "v1",
                "content_sha256": hash_bytes(paper_text.encode()),
                "cas_path": str(paper_path),
                "retrieval_status": "fetched",
            },
            {
                "source_id": "code-1",
                "url": "https://example.com/code",
                "revision": "v1",
                "content_sha256": hash_bytes(code_text.encode()),
                "cas_path": str(code_path),
                "retrieval_status": "fetched",
            },
        ]
    }
    manifest["manifest_sha256"] = stable_hash(manifest["sources"])
    return evidence, manifest


@pytest.mark.parametrize(
    ("tagged", "required"),
    [
        ("external_metadata.citation.arxiv_id", "external_metadata.citation"),
        ("external_metadata.modes.train_eval_divergence", "external_metadata.modes"),
        ("input_contract.args", "input_contract"),
    ],
)
def test_a_leaf_path_never_satisfies_its_aggregate(
    tmp_path: Path, tagged: str, required: str
) -> None:
    """Roll-up is refused BY DESIGN, and the fix is the instruction, not the check.

    An excerpt supporting only an arXiv ID genuinely does not support the citation as a
    whole. Rolling leaves up into their parent would silently launder partial support
    into full coverage, so the gate must keep refusing this and the prompt must keep
    telling the author to tag the aggregate.
    """

    evidence, manifest = _multi_source_evidence(tmp_path)
    evidence["excerpts"][0]["supports"] = [tagged]
    with pytest.raises(EvidenceValidationError) as excinfo:
        validate_evidence(evidence, manifest, [required])
    assert str(excinfo.value) == f"ungrounded claim categories: ['{required}']"


@pytest.mark.parametrize(
    ("tagged", "required"),
    [
        ("implementation.original_framework", "external_metadata.original_framework"),
        ("taxonomy.family", "external_metadata.family"),
        ("implementation.run_framework", "external_metadata.run_framework"),
    ],
)
def test_a_cross_block_spelling_never_satisfies_the_gated_claim(
    tmp_path: Path, tagged: str, required: str
) -> None:
    """The right fact under the wrong schema home covers nothing.

    ``taxonomy.family`` and ``external_metadata.family`` are BOTH gated categories and
    they are DIFFERENT ones; neither stands in for the other.
    """

    evidence, manifest = _multi_source_evidence(tmp_path)
    evidence["excerpts"][0]["supports"] = [tagged]
    with pytest.raises(EvidenceValidationError) as excinfo:
        validate_evidence(evidence, manifest, [required])
    assert str(excinfo.value) == f"ungrounded claim categories: ['{required}']"


def test_the_exact_aggregate_string_does_satisfy_the_claim(tmp_path: Path) -> None:
    """The positive arm: the string the vocabulary names is what passes."""

    evidence, manifest = _multi_source_evidence(tmp_path)
    evidence["excerpts"][0]["supports"] = ["external_metadata.citation"]
    report = validate_evidence(evidence, manifest, ["external_metadata.citation"])
    assert "external_metadata.citation" in report.supported_claims


# --------------------------------------------------------------------------------
# 3. Typed absence discharges coverage; bare emptiness still does not.
# --------------------------------------------------------------------------------


def test_a_typed_absence_discharges_coverage_without_any_excerpt_tag(
    tmp_path: Path,
) -> None:
    """Class E: the record's own evidence IDs satisfy the claim it is about.

    There is no excerpt that says a fact is not there, so requiring the excerpt to
    back-tag the claim made the honest declaration fail while a fabricated tag on an
    unrelated excerpt succeeded.
    """

    evidence, manifest = _multi_source_evidence(tmp_path)
    with pytest.raises(EvidenceValidationError) as excinfo:
        validate_evidence(evidence, manifest, ["external_metadata.venue"])
    assert str(excinfo.value) == "ungrounded claim categories: ['external_metadata.venue']"

    report = validate_evidence(
        evidence,
        manifest,
        ["external_metadata.venue"],
        declared_absences={"external_metadata.venue": ["ev-paper"]},
    )
    assert report.absence_covered_claims == frozenset({"external_metadata.venue"})
    assert "external_metadata.venue" not in report.supported_claims


def test_an_absence_citing_fabricated_evidence_buys_nothing(tmp_path: Path) -> None:
    """A record may only cite excerpts that actually validated in this same pass."""

    evidence, manifest = _multi_source_evidence(tmp_path)
    with pytest.raises(EvidenceValidationError) as excinfo:
        validate_evidence(
            evidence,
            manifest,
            ["external_metadata.venue"],
            declared_absences={"external_metadata.venue": ["ev-does-not-exist"]},
        )
    assert str(excinfo.value) == (
        "declared absence for external_metadata.venue cites missing or fabricated "
        "evidence: ['ev-does-not-exist']"
    )


def test_a_bare_evidence_id_string_is_not_a_list_of_ids(tmp_path: Path) -> None:
    """A bare string iterates into characters; it must be rejected as a Sequence."""

    evidence, manifest = _multi_source_evidence(tmp_path)
    with pytest.raises(EvidenceValidationError) as excinfo:
        validate_evidence(
            evidence,
            manifest,
            ["external_metadata.venue"],
            declared_absences={"external_metadata.venue": "ev-paper"},
        )
    assert str(excinfo.value) == (
        "declared absence for external_metadata.venue must carry a list of evidence IDs"
    )


def _ground(tmp_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a full schema-valid proposal whose gated claims are all grounded."""

    proposal = make_author_proposal()
    text = (
        "Example Model introduced ExampleNet in TestConf 2020 by A. Author at Example Lab in "
        "the US. ExampleNet is an official PyTorch library CNN architecture for supervised "
        "computer vision classification in machine learning. This modern ExampleNet family "
        "uses vision modality and has the example and cnn keywords. It is a small "
        "source-grounded example network whose grounded contribution uses the Apache-2.0 "
        "license. It runs in PyTorch eval mode with no train eval divergence. The input "
        "contract is one small RGB image and the output is class scores."
    )
    source_path = tmp_path / "source.txt"
    source_path.write_text(text)
    excerpt = proposal["proposed_facts"]["evidence"]["excerpts"][0]
    excerpt.update(
        {
            "locator": f"bytes:0-{len(text.encode())}",
            "text": text,
            "text_sha256": hash_bytes(text.encode()),
            "supports": [*sorted(DEFAULT_GATED_CLAIMS), "implementation.architecture"],
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
    manifest: dict[str, Any] = {
        "sources": [
            {
                "source_id": "source-1",
                "url": "https://example.com/model",
                "revision": "v1",
                "content_sha256": hash_bytes(text.encode()),
                "cas_path": str(source_path),
                "retrieval_status": "fetched",
            }
        ]
    }
    manifest["manifest_sha256"] = stable_hash(manifest["sources"])
    proposal["verified_hashes"]["source_manifest"] = manifest["manifest_sha256"]
    attach_paper_evidence(proposal, manifest, tmp_path)
    return proposal, manifest


def _availability(proposal: dict[str, Any]) -> dict[str, Any]:
    """Return the proposal's availability register."""

    return proposal["proposed_facts"]["external_metadata"]["availability"]


@pytest.mark.parametrize("claim", sorted(EMPTIABLE_CLAIMS))
def test_an_empty_emptiable_claim_without_a_typed_record_is_still_refused(
    tmp_path: Path, claim: str
) -> None:
    """Class D negative: emptiness is an assertion, and an undeclared one fails.

    ``EMPTIABLE_CLAIMS`` used to mean "empty passes for free", which combined with the
    nominal coverage gate meant the ONLY way through was to tag an unrelated excerpt.
    Removing the free pass must not reintroduce a silent one.
    """

    proposal, manifest = _ground(tmp_path)
    key = claim.removeprefix("external_metadata.") if "external_metadata." in claim else claim
    del _availability(proposal)[key]
    with pytest.raises(ProposalValidationError) as excinfo:
        validate_author_proposal(proposal, allowed_model_dir=tmp_path, source_manifest=manifest)
    assert str(excinfo.value) == (
        f"gated claim {claim} is bare null/empty; a value must be present or the claim "
        "must declare a typed availability state (external_metadata.availability) of "
        "none-exist, not-found-after-search, or not-applicable with its evidence"
    )


def test_taxonomy_era_null_without_a_typed_record_is_still_refused(tmp_path: Path) -> None:
    """Scalar taxonomy era has an availability route, but bare null still fails."""

    proposal, manifest = _ground(tmp_path)
    proposal["proposed_facts"]["taxonomy"]["era"] = None
    with pytest.raises(ProposalValidationError) as excinfo:
        validate_author_proposal(proposal, allowed_model_dir=tmp_path, source_manifest=manifest)
    assert str(excinfo.value) == (
        "gated claim taxonomy.era is bare null/empty; a value must be present or the "
        "claim must declare a typed availability state (external_metadata.availability) of "
        "none-exist, not-found-after-search, or not-applicable with its evidence"
    )


@pytest.mark.parametrize("claim", sorted(EMPTIABLE_CLAIMS))
def test_an_empty_emptiable_claim_passes_on_a_typed_none_exist_record(
    tmp_path: Path, claim: str
) -> None:
    """Class D positive: the honest spelling of "there are none" now exists."""

    proposal, manifest = _ground(tmp_path)
    key = claim.removeprefix("external_metadata.") if "external_metadata." in claim else claim
    assert _availability(proposal)[key]["status"] == "none-exist"
    report = validate_author_proposal(
        proposal, allowed_model_dir=tmp_path, source_manifest=manifest
    )
    assert report is not None


def test_none_exist_is_a_distinct_recorded_state_not_an_alias(tmp_path: Path) -> None:
    """``none-exist`` and ``not-found-after-search`` are different recorded answers.

    One says the fact does not exist; the other says it could not be established. Both
    must be expressible and both must survive into the record, or the catalog cannot
    tell an answered-empty field from an unanswered one.
    """

    assert {"none-exist", "not-found-after-search"} <= AVAILABILITY_STATUSES
    proposal, manifest = _ground(tmp_path)
    _availability(proposal)["lineage"]["status"] = "not-found-after-search"
    validate_author_proposal(proposal, allowed_model_dir=tmp_path, source_manifest=manifest)
    assert _availability(proposal)["lineage"]["status"] == "not-found-after-search"
    assert _availability(proposal)["predecessors"]["status"] == "none-exist"


@pytest.mark.parametrize("claim", sorted(EMPTIABLE_CLAIMS))
def test_a_none_exist_claim_without_a_bounded_search_is_refused(
    tmp_path: Path, claim: str
) -> None:
    """Asserting "there are none" is a finding, and a finding needs its search.

    Exactly one claim under test carries the search-backed status; the other two are
    parked on ``not-applicable``, which no search informs. Without that isolation the
    validator's set iteration decides which claim the message names, and the assertion
    becomes hash-seed dependent.
    """

    proposal, manifest = _ground(tmp_path)
    for other in EMPTIABLE_CLAIMS:
        key = other.removeprefix("external_metadata.") if "external_metadata." in other else other
        _availability(proposal)[key]["status"] = (
            "none-exist" if other == claim else "not-applicable"
        )
    proposal["proposed_facts"]["source_resolution"]["search_report"]["queries"] = []
    with pytest.raises(ProposalValidationError) as excinfo:
        validate_author_proposal(proposal, allowed_model_dir=tmp_path, source_manifest=manifest)
    assert str(excinfo.value) == (
        "an absence state (none-exist or not-found-after-search) requires a recorded "
        f"bounded search that could have found the fact: ['{claim}']"
    )


@pytest.mark.parametrize("claim", sorted(EMPTIABLE_CLAIMS))
def test_not_applicable_also_discharges_coverage_and_needs_no_search(
    tmp_path: Path, claim: str
) -> None:
    """Every typed absence discharges coverage, not only the searched ones.

    ``not-applicable`` is just as declared, recorded, and queryable as ``none-exist``.
    Withholding the coverage route from it would rebuild the same trap one status over:
    the author declares honestly and is refused anyway.
    """

    proposal, manifest = _ground(tmp_path)
    key = claim.removeprefix("external_metadata.") if "external_metadata." in claim else claim
    # Every emptiable claim goes to not-applicable so no search-backed status remains;
    # the parametrization then asserts each claim individually still reaches coverage.
    for other in EMPTIABLE_CLAIMS:
        other_key = (
            other.removeprefix("external_metadata.") if "external_metadata." in other else other
        )
        _availability(proposal)[other_key]["status"] = "not-applicable"
        _availability(proposal)[other_key]["basis"] = "not-applicable"
    assert _availability(proposal)[key]["status"] == "not-applicable"
    proposal["proposed_facts"]["source_resolution"]["search_report"]["queries"] = []
    report = validate_author_proposal(
        proposal, allowed_model_dir=tmp_path, source_manifest=manifest
    )
    assert report is not None


def test_a_non_empty_claim_is_never_laundered_by_an_absence_record(tmp_path: Path) -> None:
    """The absence route covers ABSENCE only; a carried value must still be grounded.

    Observed on real data: one proposal declared a non-empty lineage and never tagged
    it. That must keep failing -- an absence record cannot stand in for a claim that
    actually asserts something.
    """

    proposal, manifest = _ground(tmp_path)
    proposal["proposed_facts"]["external_metadata"]["lineage"] = ["PriorNet"]
    with pytest.raises(ProposalValidationError) as excinfo:
        validate_author_proposal(proposal, allowed_model_dir=tmp_path, source_manifest=manifest)
    assert str(excinfo.value) == (
        "availability state for external_metadata.lineage declares none-exist but the "
        "field carries a value"
    )


# --------------------------------------------------------------------------------
# 4. Characterization: _SUPPORT_ALIASES are NOT honoured by the coverage gate.
# --------------------------------------------------------------------------------


def test_support_aliases_do_not_reach_the_coverage_gate(tmp_path: Path) -> None:
    """Pin the reported trap: ``_SUPPORT_ALIASES`` is dead at ``validate_evidence``.

    ``_validate_claim_support`` canonicalizes ``citation`` -> ``external_metadata.citation``
    through ``_SUPPORT_ALIASES``, but ``validate_evidence`` compares raw strings and
    never consults the map. The alias is therefore inert for coverage today. This test
    records that asymmetry rather than papering over it: honouring aliases here would
    be a real behaviour change, and it should be a deliberate one.
    """

    evidence, manifest = _multi_source_evidence(tmp_path)
    evidence["excerpts"][0]["supports"] = ["citation"]
    with pytest.raises(EvidenceValidationError) as excinfo:
        validate_evidence(evidence, manifest, ["external_metadata.citation"])
    assert str(excinfo.value) == (
        "ungrounded claim categories: ['external_metadata.citation']"
    )


def test_the_gated_vocabulary_has_no_leaf_of_another_member(tmp_path: Path) -> None:
    """No gated category is a dotted child of another, so roll-up is never needed.

    If a future change nests one gated claim under another, the no-roll-up rule would
    become genuinely ambiguous for authors. Fail here instead.
    """

    nested = sorted(
        claim
        for claim in DEFAULT_GATED_CLAIMS
        for other in DEFAULT_GATED_CLAIMS
        if claim != other and claim.startswith(f"{other}.")
    )
    assert nested == []


def test_the_generated_block_round_trips_through_json_safe_text() -> None:
    """The prompt block stays plain ASCII text the author can copy verbatim."""

    block = gated_claim_vocabulary_block()
    assert block == json.loads(json.dumps(block))
    assert block.isascii()


# --------------------------------------------------------------------------------
# 5. The audited coverage map: no author-gated leaf without a declared lane.
# --------------------------------------------------------------------------------


def _lane_covers(path: str, prefix: str) -> bool:
    """Return whether one dotted/indexed schema path falls under one lane prefix."""

    return path == prefix or path.startswith(f"{prefix}.") or path.startswith(f"{prefix}[")


def test_every_author_gated_schema_path_has_a_declared_verification_lane() -> None:
    """Every authored leaf is claim-gated or carries an audited lane assignment.

    The claim-vocabulary gate deliberately covers claims, not the ~290 author-gated
    schema paths; the 2026-08 audit mapped every remaining path to the lane that
    actually verifies it (or recorded why it is deliberately unverified) in
    ``AUTHORED_LEAF_COVERAGE_LANES``. This test is the tripwire that keeps the map
    exhaustive: a NEW author-gated schema path that lands in no gated claim and no
    audited lane fails here, forcing its change to declare who verifies it instead
    of shipping a leaf an author can state freely and nobody checks.
    """

    claim_prefixes = set(DEFAULT_GATED_CLAIMS) | {KEYWORD_CLAIM}
    lane_prefixes = [prefix for prefix, _ in AUTHORED_LEAF_COVERAGE_LANES]
    uncovered = []
    for raw in schema_owner_paths(MODEL_SCHEMA_VERSION_V3, SchemaOwner.AUTHOR_GATED):
        path = raw.removeprefix("$.")
        claim_covered = any(_lane_covers(path, claim) for claim in claim_prefixes)
        lane_covered = any(_lane_covers(path, prefix) for prefix in lane_prefixes)
        if not claim_covered and not lane_covered:
            uncovered.append(path)
    assert uncovered == []


def test_every_declared_lane_prefix_names_a_real_schema_path() -> None:
    """No dead map rows: every lane prefix matches at least one author-gated path."""

    paths = [
        raw.removeprefix("$.")
        for raw in schema_owner_paths(MODEL_SCHEMA_VERSION_V3, SchemaOwner.AUTHOR_GATED)
    ]
    dead = [
        prefix
        for prefix, _ in AUTHORED_LEAF_COVERAGE_LANES
        if not any(_lane_covers(path, prefix) for path in paths)
    ]
    assert dead == []


def test_website_prose_is_an_always_required_gated_claim() -> None:
    """The public-facing website block is claim-gated, unconditionally.

    Website prose was the one authored block no lane verified structurally: the
    checker reviewed it only under its holistic item verdict. The block claim closes
    that, and it needs no availability route because the schema requires non-empty
    website prose for every proposal.
    """

    assert "website" in DEFAULT_GATED_CLAIMS
    assert "website" not in CONDITIONAL_GATED_CLAIMS
    assert "website" in gated_claim_vocabulary_block()


# --------------------------------------------------------------------------------
# 6. website.family_grounding_id is a dereferenced reference, not free text.
# --------------------------------------------------------------------------------


def test_a_fabricated_family_grounding_id_is_refused(tmp_path: Path) -> None:
    """The one field naming the family-grounding excerpt must actually name one."""

    proposal, manifest = _ground(tmp_path)
    proposal["proposed_facts"]["website"]["family_grounding_id"] = "grounding-nowhere"
    with pytest.raises(ProposalValidationError) as excinfo:
        validate_author_proposal(proposal, allowed_model_dir=tmp_path, source_manifest=manifest)
    assert str(excinfo.value) == (
        "website.family_grounding_id references missing or fabricated evidence: "
        "grounding-nowhere"
    )


def test_a_non_family_level_grounding_reference_is_refused(tmp_path: Path) -> None:
    """Pointing at a real excerpt that is not family-level is still a false claim."""

    proposal, manifest = _ground(tmp_path)
    # ``attach_paper_evidence`` adds a real, validated, NON-family-level excerpt.
    proposal["proposed_facts"]["website"]["family_grounding_id"] = "evidence-source-paper"
    with pytest.raises(ProposalValidationError) as excinfo:
        validate_author_proposal(proposal, allowed_model_dir=tmp_path, source_manifest=manifest)
    assert str(excinfo.value) == (
        "website.family_grounding_id must name a family_level excerpt: "
        "evidence-source-paper is not family-level"
    )


def test_a_family_level_grounding_reference_passes(tmp_path: Path) -> None:
    """The positive arm: the fixture references its family-level excerpt and passes."""

    proposal, manifest = _ground(tmp_path)
    website = proposal["proposed_facts"]["website"]
    excerpts = proposal["proposed_facts"]["evidence"]["excerpts"]
    referenced = [
        excerpt
        for excerpt in excerpts
        if excerpt["evidence_id"] == website["family_grounding_id"]
    ]
    assert len(referenced) == 1 and referenced[0]["family_level"] is True
    report = validate_author_proposal(
        proposal, allowed_model_dir=tmp_path, source_manifest=manifest
    )
    assert "website" in report.supported_claims
