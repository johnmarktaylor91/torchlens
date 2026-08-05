"""The ONE supplementary broker pack must ground evidence, and nothing else may.

Context (pilot rung, 2026-07-31). Five ``c1-mech`` models were refused
``failed:author / malformed-result`` on evidence bookkeeping. Three of them --
and a fourth behind an earlier refusal -- died on ``references unknown source``
while citing a source OUR OWN broker had fetched and hash-pinned for them: the
executor grants at most one supplementary source round mid-session, resumes the
author with the resulting manifest, and nothing carried that manifest back to
the lane. The frozen manifest never learned the source existed, so an honest
author quoting a document we handed it was recorded as a fabricator, with no
trace anywhere of the pack that proved otherwise.

That is strictly worse than the fabrication the check exists to catch, so these
tests pin BOTH directions:

* a supplementary source is citable, and its identity binding does not move; and
* a source in neither set, and a digest that does not cover the quoted string,
  are still refused exactly as before.

The digest cases are transcribed from the two real refusals so the tripwire is
pinned against the mechanisms that actually occurred, not an invented one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from menagerie.crawler.artifact_transactions import ArtifactBindingError, _validate_context_result
from menagerie.crawler.author_attempts import new_attempt
from menagerie.crawler.author_dispatch import (
    AuthorEffortGrant,
    ProposedAuthorResult,
    build_author_envelope,
    serialize_author_result_cache,
    validate_author_result,
    validate_author_result_cache,
    write_envelope_atomic,
)
from menagerie.crawler.constants import (
    AUTHOR_PROPOSAL_SCHEMA_VERSION_V3,
    EnvironmentPhase,
)
from menagerie.crawler.driver_admission import _AuthorLaneBase
from menagerie.crawler.driver_contracts import DriverIntegrationError, IntentRoute, WorkItem
from menagerie.crawler.evidence import EvidenceValidationError, validate_evidence
from menagerie.crawler.identity import hash_bytes, stable_hash
from menagerie.crawler.intake import IntakeItem
from menagerie.crawler.models import manifest_source_rows
from menagerie.crawler.proposal import ProposalValidationError, validate_author_proposal
from menagerie.crawler.source_broker import broker_source_pack, write_broker_outputs
from menagerie.crawler.tests.conftest import PAPER_SOURCE_ID
from menagerie.crawler.tests.test_slice_d_proposal_author import (
    _author_context,
    _author_result,
    _ground_proposal,
)
from menagerie.crawler.tests.test_source_cas_promotion import MapTransport

#: A no-break space, byte-for-byte what the m4334 ar5iv paragraph really held and
#: what the author's quoted text silently flattened to an ASCII space.
NBSP = " "


def _work_item(stable_id: str) -> WorkItem:
    """Return a minimal routed work item; only its stable identity is read here."""

    intake = IntakeItem(
        stable_id=stable_id,
        name="SupplementFixture",
        zoo="fixture",
        variant="base",
        discovery_source="crawl_roster",
        legacy_row_sha256="0" * 64,
        preserved_legacy_flags=(),
        variant_scope="standalone",
        family_representative_id=stable_id,
    )
    return WorkItem(
        intake=intake,
        route=IntentRoute(stable_id=stable_id, intent="core", phase=EnvironmentPhase.PYTORCH),
    )


def _demote_paper_source_to_supplement(
    proposal: dict[str, Any], manifest: dict[str, Any]
) -> dict[str, Any]:
    """Move the paper row out of ``sources`` into the supplementary pack.

    Reproduces the shipped shape exactly: the paper was NOT in the frozen
    stage-1 manifest, the author asked for it, our broker fetched it, and the
    excerpt cites it. The frozen identity is recomputed from what actually
    remains frozen, so the fixture cannot pass by accident on a stale digest.

    Parameters
    ----------
    proposal:
        Author proposal mutated in place.
    manifest:
        Frozen manifest mutated in place.

    Returns
    -------
    dict[str, Any]
        The manifest, now carrying ``supplementary_sources``.
    """

    frozen = [row for row in manifest["sources"] if row["source_id"] != PAPER_SOURCE_ID]
    supplementary = [row for row in manifest["sources"] if row["source_id"] == PAPER_SOURCE_ID]
    assert len(supplementary) == 1, "fixture must move exactly the paper row"
    assert len(frozen) >= 1, "a frozen row must remain so the split is real"
    manifest["sources"] = frozen
    manifest["manifest_sha256"] = stable_hash(frozen)
    manifest["supplementary_sources"] = supplementary
    manifest["supplementary_manifest_sha256"] = stable_hash(supplementary)
    proposal["verified_hashes"]["source_manifest"] = manifest["manifest_sha256"]
    return manifest


def _frozen_only(manifest: dict[str, Any]) -> dict[str, Any]:
    """Return the manifest as the lane saw it BEFORE the supplement was ingested."""

    return {
        "sources": list(manifest["sources"]),
        "manifest_sha256": manifest["manifest_sha256"],
    }


def _paper_excerpt(proposal: dict[str, Any]) -> dict[str, Any]:
    """Return the excerpt that cites the demoted paper source."""

    for excerpt in proposal["proposed_facts"]["evidence"]["excerpts"]:
        if excerpt["source_id"] == PAPER_SOURCE_ID:
            return excerpt
    raise AssertionError("fixture lost its paper excerpt")


# -- the defect: a source we fetched ourselves must be citable ---------------


@pytest.mark.smoke
def test_a_supplementary_source_is_refused_until_the_lane_ingests_it(
    tmp_path: Path,
) -> None:
    """The exact shipped refusal, and the exact fix, on one fixture.

    Same proposal, same excerpt, same bytes. The only difference is whether the
    lane carried the supplementary pack back -- which is precisely the defect.
    """

    proposal, manifest = _ground_proposal(tmp_path)
    _demote_paper_source_to_supplement(proposal, manifest)
    evidence_id = _paper_excerpt(proposal)["evidence_id"]

    with pytest.raises(ProposalValidationError) as refused:
        validate_author_proposal(
            proposal,
            allowed_model_dir=tmp_path,
            source_manifest=_frozen_only(manifest),
        )
    assert str(refused.value) == f"{evidence_id} references unknown source {PAPER_SOURCE_ID}"

    report = validate_author_proposal(
        proposal,
        allowed_model_dir=tmp_path,
        source_manifest={"sources": manifest_source_rows(manifest)},
    )
    assert report.stable_id == proposal["stable_id"]


def test_ingesting_the_supplement_never_moves_the_bound_manifest_identity(
    tmp_path: Path,
) -> None:
    """``source_manifest_identity`` is what the author echoed; it must not move.

    Growing ``sources`` would have been the obvious fix and is the wrong one: it
    re-derives the identity the author already bound into its result, so a good
    proposal would die at the binder on an identity it was never shown.
    """

    proposal, manifest = _ground_proposal(tmp_path)
    frozen_identity = stable_hash(
        [row for row in manifest["sources"] if row["source_id"] != PAPER_SOURCE_ID]
    )
    _demote_paper_source_to_supplement(proposal, manifest)

    assert manifest["manifest_sha256"] == frozen_identity
    assert manifest["manifest_sha256"] == stable_hash(manifest["sources"])
    assert manifest["supplementary_manifest_sha256"] != manifest["manifest_sha256"]
    assert all(
        row["source_id"] != PAPER_SOURCE_ID for row in manifest["sources"]
    ), "a supplementary row must never join the identity-bearing list"


def test_manifest_source_rows_returns_both_halves_in_order(tmp_path: Path) -> None:
    """Grounding reads both halves; identity readers still read ``sources``."""

    proposal, manifest = _ground_proposal(tmp_path)
    _demote_paper_source_to_supplement(proposal, manifest)

    rows = manifest_source_rows(manifest)
    identifiers = [row["source_id"] for row in rows]

    assert identifiers == [
        *(row["source_id"] for row in manifest["sources"]),
        *(row["source_id"] for row in manifest["supplementary_sources"]),
    ]
    assert PAPER_SOURCE_ID in identifiers
    assert len(set(identifiers)) == len(identifiers)


def test_a_manifest_with_no_supplement_reads_exactly_its_frozen_rows(
    tmp_path: Path,
) -> None:
    """The overwhelmingly common case must be byte-identical to before."""

    _, manifest = _ground_proposal(tmp_path)

    assert manifest_source_rows(manifest) == manifest["sources"]
    assert "supplementary_sources" not in manifest


# -- the failing direction, which must stay failing -------------------------


def test_a_source_in_neither_half_is_still_refused(tmp_path: Path) -> None:
    """Widening grounding must not make an unknown citation reachable."""

    proposal, manifest = _ground_proposal(tmp_path)
    _demote_paper_source_to_supplement(proposal, manifest)
    excerpt = _paper_excerpt(proposal)
    excerpt["source_id"] = "paper-never-fetched"

    with pytest.raises(ProposalValidationError) as refused:
        validate_author_proposal(
            proposal,
            allowed_model_dir=tmp_path,
            source_manifest={"sources": manifest_source_rows(manifest)},
        )
    assert (
        str(refused.value)
        == f"{excerpt['evidence_id']} references unknown source paper-never-fetched"
    )


@pytest.mark.smoke
def test_a_digest_over_a_wider_line_range_than_the_quote_is_still_refused(
    tmp_path: Path,
) -> None:
    """The m7362 mechanism: hash the locator's lines, quote fewer of them.

    ``def forward`` through ``return x`` is lines 371-382 of the upstream file
    only if the trailing blank line is counted. The author hashed all twelve and
    quoted eleven. The excerpt text IS verbatim in the source, so the locator
    check passes it -- the digest is the only thing standing between a partial
    quote and the record, and it must keep standing.
    """

    quoted = "    def forward(self, x: torch.Tensor) -> torch.Tensor:\n        return x\n"
    source_text = quoted + "\n"
    source_path = tmp_path / "wider-range.py"
    source_path.write_bytes(source_text.encode("utf-8"))
    manifest = {
        "sources": [
            {
                "source_id": "impl-upstream",
                "content_sha256": hash_bytes(source_text.encode("utf-8")),
                "cas_path": str(source_path),
            }
        ]
    }
    evidence = {
        "excerpts": [
            {
                "evidence_id": "ev-upstream-forward",
                "source_id": "impl-upstream",
                "locator": "mobileone.py lines 371-382",
                "text": quoted,
                # The digest of the WIDER range the locator names.
                "text_sha256": hash_bytes(source_text.encode("utf-8")),
                "supports": ["implementation.architecture"],
                "family_level": True,
            }
        ],
        "coverage": {"all_agent_fields_have_support": True, "missing_support": []},
    }

    assert source_text.encode("utf-8").find(quoted.encode("utf-8")) == 0, (
        "the quote must really be verbatim in the source, or this test proves nothing"
    )
    with pytest.raises(EvidenceValidationError) as refused:
        validate_evidence(evidence, manifest, ["implementation.architecture"])
    assert (
        str(refused.value)
        == "ev-upstream-forward text_sha256 does not match the verbatim UTF-8 bytes"
    )


@pytest.mark.smoke
def test_a_digest_over_bytes_the_quote_normalized_away_is_still_refused(
    tmp_path: Path,
) -> None:
    """The m4334 mechanism: the source held U+00A0, the quote held a space.

    The digest was RIGHT about the real bytes and the quoted text was wrong, so
    accepting it would have published a paraphrase under a correct-looking hash.
    """

    source_text = f"We first train our networks on the ImageNet 2012 training set{NBSP}[34].\n"
    flattened = source_text.replace(NBSP, " ")
    source_path = tmp_path / "paper-body.html"
    source_path.write_bytes(source_text.encode("utf-8"))
    manifest = {
        "sources": [
            {
                "source_id": "paper-body",
                "content_sha256": hash_bytes(source_text.encode("utf-8")),
                "cas_path": str(source_path),
            }
        ]
    }
    evidence = {
        "excerpts": [
            {
                "evidence_id": "ev-paper-imagenet",
                "source_id": "paper-body",
                "locator": "ar5iv HTML line 533",
                "text": flattened,
                # The digest of the REAL bytes, which the quoted text no longer is.
                "text_sha256": hash_bytes(source_text.encode("utf-8")),
                "supports": ["external_metadata.citation"],
                "family_level": True,
            }
        ],
        "coverage": {"all_agent_fields_have_support": True, "missing_support": []},
    }

    assert flattened != source_text, "the fixture must actually differ from its source"
    with pytest.raises(EvidenceValidationError) as refused:
        validate_evidence(evidence, manifest, ["external_metadata.citation"])
    assert (
        str(refused.value)
        == "ev-paper-imagenet text_sha256 does not match the verbatim UTF-8 bytes"
    )


def test_a_supplementary_source_cannot_launder_a_bad_digest(tmp_path: Path) -> None:
    """Being reachable is not being trusted: the digest check still runs on it."""

    proposal, manifest = _ground_proposal(tmp_path)
    _demote_paper_source_to_supplement(proposal, manifest)
    excerpt = _paper_excerpt(proposal)
    excerpt["text_sha256"] = hash_bytes((excerpt["text"] + " tampered").encode("utf-8"))

    with pytest.raises(ProposalValidationError) as refused:
        validate_author_proposal(
            proposal,
            allowed_model_dir=tmp_path,
            source_manifest={"sources": manifest_source_rows(manifest)},
        )
    assert (
        str(refused.value)
        == f"{excerpt['evidence_id']} text_sha256 does not match the verbatim UTF-8 bytes"
    )


def test_a_supplementary_source_cannot_launder_a_non_verbatim_excerpt(
    tmp_path: Path,
) -> None:
    """The locator check runs on the supplementary half too."""

    proposal, manifest = _ground_proposal(tmp_path)
    _demote_paper_source_to_supplement(proposal, manifest)
    excerpt = _paper_excerpt(proposal)
    excerpt["text"] = "A sentence that never appeared in the fetched paper page."
    excerpt["text_sha256"] = hash_bytes(excerpt["text"].encode("utf-8"))
    excerpt["locator"] = "abstract"

    with pytest.raises(ProposalValidationError) as refused:
        validate_author_proposal(
            proposal,
            allowed_model_dir=tmp_path,
            source_manifest={"sources": manifest_source_rows(manifest)},
        )
    assert (
        str(refused.value)
        == f"{excerpt['evidence_id']} excerpt is not verbatim in its fetched source"
    )


# -- the transport: the extension survives result, cache, and rehydration ----


def test_the_result_binding_and_its_cache_both_accept_the_ingested_pack(
    tmp_path: Path,
) -> None:
    """End to end: dispatch binding, result validation, and cache reload.

    The cache path matters on its own. It rebuilds an envelope from the stored
    manifest, so if the supplementary rows did not ride along, a model that used
    its supplement round would validate once and then fail every rehydration.
    """

    proposal, manifest = _ground_proposal(tmp_path)
    _demote_paper_source_to_supplement(proposal, manifest)
    prompt_hash = hash_bytes(
        (Path(__file__).parents[1] / "prompts" / "claude_crawler_author_v2.txt").read_bytes()
    )
    proposal["author"]["prompt_sha256"] = prompt_hash
    proposal.update(
        {
            "schema_version": AUTHOR_PROPOSAL_SCHEMA_VERSION_V3,
            "campaign_id": "campaign-1",
            "intake_snapshot_id": "intake-1",
            "intake_snapshot_sha256": "sha256:" + "1" * 64,
            "intake_item_sha256": stable_hash(
                {"stable_id": proposal["stable_id"], "variant": "base"}
            ),
            "source_manifest_identity": str(manifest["manifest_sha256"]),
            "dispatcher_identity": "sha256:" + "2" * 64,
        }
    )
    proposal["proposal_sha256"] = stable_hash(
        {key: value for key, value in proposal.items() if key != "proposal_sha256"}
    )
    context = _author_context(proposal, prompt_hash)
    result_path = tmp_path / "output" / "result.json"
    envelope = build_author_envelope(
        context=context,
        work_id=proposal["work_id"],
        stable_id=proposal["stable_id"],
        campaign_id="campaign-1",
        created_at="2026-07-16T00:00:00Z",
        untrusted_hints={"legacy": "hint"},
        source_manifest=manifest,
        allowed_model_dir=tmp_path,
        output_path=result_path,
    )
    # The identity the author echoes is still the FROZEN one.
    assert envelope["source_manifest_identity"] == stable_hash(manifest["sources"])

    result = _author_result(envelope, "PROPOSED", {"arm": "PROPOSED", "proposal": proposal})
    result_path.parent.mkdir()
    result_path.write_text(json.dumps(result))

    validated = validate_author_result(
        result_path,
        envelope,
        supplementary_sources=manifest["supplementary_sources"],
    )
    assert isinstance(validated, ProposedAuthorResult)

    cache = serialize_author_result_cache(validated, source_manifest=manifest, model_dir=tmp_path)
    reloaded = validate_author_result_cache(cache, envelope)
    assert isinstance(reloaded, ProposedAuthorResult)


def test_a_result_validated_without_its_pack_still_names_the_unknown_source(
    tmp_path: Path,
) -> None:
    """A lane that forgets to ingest must fail loudly, not pass quietly."""

    proposal, manifest = _ground_proposal(tmp_path)
    _demote_paper_source_to_supplement(proposal, manifest)
    evidence_id = _paper_excerpt(proposal)["evidence_id"]
    prompt_hash = hash_bytes(
        (Path(__file__).parents[1] / "prompts" / "claude_crawler_author_v2.txt").read_bytes()
    )
    proposal["author"]["prompt_sha256"] = prompt_hash
    proposal.update(
        {
            "schema_version": AUTHOR_PROPOSAL_SCHEMA_VERSION_V3,
            "campaign_id": "campaign-1",
            "intake_snapshot_id": "intake-1",
            "intake_snapshot_sha256": "sha256:" + "1" * 64,
            "intake_item_sha256": stable_hash(
                {"stable_id": proposal["stable_id"], "variant": "base"}
            ),
            "source_manifest_identity": str(manifest["manifest_sha256"]),
            "dispatcher_identity": "sha256:" + "2" * 64,
        }
    )
    proposal["proposal_sha256"] = stable_hash(
        {key: value for key, value in proposal.items() if key != "proposal_sha256"}
    )
    context = _author_context(proposal, prompt_hash)
    result_path = tmp_path / "output" / "result.json"
    envelope = build_author_envelope(
        context=context,
        work_id=proposal["work_id"],
        stable_id=proposal["stable_id"],
        campaign_id="campaign-1",
        created_at="2026-07-16T00:00:00Z",
        untrusted_hints={"legacy": "hint"},
        source_manifest=_frozen_only(manifest),
        allowed_model_dir=tmp_path,
        output_path=result_path,
    )
    result = _author_result(envelope, "PROPOSED", {"arm": "PROPOSED", "proposal": proposal})
    result_path.parent.mkdir()
    result_path.write_text(json.dumps(result))

    with pytest.raises(Exception) as refused:
        validate_author_result(result_path, envelope)
    assert str(refused.value) == f"{evidence_id} references unknown source {PAPER_SOURCE_ID}"


# -- the artifact binding: the same split, one lane further downstream -------


def _bindable(proposal: dict[str, Any], manifest: dict[str, Any], prompt_hash: str) -> None:
    """Stamp the transaction-authority fields ``_validate_context_result`` reads."""

    proposal["author"]["prompt_sha256"] = prompt_hash
    proposal.update(
        {
            "schema_version": AUTHOR_PROPOSAL_SCHEMA_VERSION_V3,
            "campaign_id": "campaign-1",
            "intake_snapshot_id": "intake-1",
            "intake_snapshot_sha256": "sha256:" + "1" * 64,
            "intake_item_sha256": stable_hash(
                {"stable_id": proposal["stable_id"], "variant": "base"}
            ),
            "source_manifest_identity": str(manifest["manifest_sha256"]),
            "dispatcher_identity": "sha256:" + "2" * 64,
        }
    )
    proposal["proposal_sha256"] = stable_hash(
        {key: value for key, value in proposal.items() if key != "proposal_sha256"}
    )


def _echo_manifest_fields(proposal: dict[str, Any], manifest: dict[str, Any]) -> None:
    """Copy the five machine-owned source fields verbatim, as a correct author does.

    ``_ground_proposal`` is a proposal-validation fixture and its declared rows
    carry placeholder digests, which the binding's per-row field check refuses on
    its own. Normalizing them here keeps these three cases pinned to the SET
    comparison under test; the field check keeps its own coverage elsewhere.
    """

    rows: list[dict[str, Any]] = [
        row  # the fixture's own mutable manifest dicts, reached through one reading
        for group in ("sources", "supplementary_sources")
        for row in manifest.get(group, [])
    ]
    by_id = {str(row["source_id"]): row for row in rows}
    for declared in proposal["proposed_facts"]["source_resolution"]["sources"]:
        fetched = by_id.get(str(declared["source_id"]))
        if fetched is None:
            continue
        for field in ("url", "revision", "content_sha256", "media_type", "retrieved_at"):
            # Reconcile toward whichever side actually carries the value; the
            # fixture's manifest rows are partial and its declared rows carry
            # placeholders, and inventing a third value on either side would make
            # the pair agree on something neither ever said.
            if fetched.get(field) is None:
                fetched[field] = declared.get(field)
            else:
                declared[field] = fetched[field]
        if fetched.get("fetched_bytes_len") is None and fetched.get("byte_count") is None:
            fetched["fetched_bytes_len"] = declared["byte_count"]
        else:
            declared["byte_count"] = fetched.get("fetched_bytes_len", fetched.get("byte_count"))
    # Reconciliation edited manifest rows, so re-freeze both halves' digests.
    # `_bindable` reads `manifest_sha256` afterwards, and the binder recomputes
    # it from the rows -- a stale digest would fail on identity, not on the set.
    manifest["manifest_sha256"] = stable_hash(manifest["sources"])
    if "supplementary_sources" in manifest:
        manifest["supplementary_manifest_sha256"] = stable_hash(manifest["supplementary_sources"])
    proposal["verified_hashes"]["source_manifest"] = manifest["manifest_sha256"]


def test_artifact_binding_enforces_retrieved_at_when_the_manifest_carries_it(
    tmp_path: Path,
) -> None:
    """A source timestamp becomes part of the transaction echo only when present."""

    proposal, manifest = _ground_proposal(tmp_path)
    manifest["sources"][0]["retrieved_at"] = "2026-08-05T15:00:00Z"
    manifest["manifest_sha256"] = stable_hash(manifest["sources"])
    source = proposal["proposed_facts"]["source_resolution"]["sources"][0]
    source["retrieved_at"] = "2026-08-05T14:59:00Z"

    with pytest.raises(
        ArtifactBindingError,
        match="proposal source fields differ from controlled-fetch manifest",
    ):
        _bind(proposal, manifest, echo_manifest=False)


def test_artifact_binding_does_not_invent_retrieved_at_for_legacy_manifests(
    tmp_path: Path,
) -> None:
    """A legacy manifest without ``retrieved_at`` keeps the old echo surface."""

    proposal, manifest = _ground_proposal(tmp_path)
    manifest["sources"][0].pop("retrieved_at", None)

    assert "source-1" in _bind(proposal, manifest)


def _bind(
    proposal: dict[str, Any], manifest: dict[str, Any], *, echo_manifest: bool = True
) -> tuple[str, ...]:
    """Run the REAL artifact binding over one proposal and return its source set."""

    prompt_hash = hash_bytes(
        (Path(__file__).parents[1] / "prompts" / "claude_crawler_author_v2.txt").read_bytes()
    )
    if echo_manifest:
        _echo_manifest_fields(proposal, manifest)
    _bindable(proposal, manifest, prompt_hash)
    context = _author_context(proposal, prompt_hash)
    envelope = build_author_envelope(
        context=context,
        work_id=proposal["work_id"],
        stable_id=proposal["stable_id"],
        campaign_id="campaign-1",
        created_at="2026-07-16T00:00:00Z",
        untrusted_hints={},
        source_manifest=manifest,
        allowed_model_dir=Path(proposal["stable_id"]),
        output_path=Path(proposal["stable_id"]) / "result.json",
    )
    result = _author_result(envelope, "PROPOSED", {"arm": "PROPOSED", "proposal": proposal})
    return _validate_context_result(
        context,
        proposal["stable_id"],
        proposal["work_id"],
        result,
        proposal,
        manifest,
    )[6]


@pytest.mark.smoke
def test_the_artifact_binding_accepts_the_supplementary_source_the_author_cited(
    tmp_path: Path,
) -> None:
    """A proposal echoing frozen PLUS supplementary rows binds.

    This is the m538/m9304 shape from the ten-model pilot, where the author did
    exactly what ``stage2_author.md`` promises -- "sources the ONE supplementary
    round fetched for you are citable exactly like frozen ones" -- and the
    binding, still reading ``sources`` alone, refused it as "proposal and source
    manifest source sets differ". Grounding had already been widened; this lane
    had not, so a correct proposal still died permanently.
    """

    proposal, manifest = _ground_proposal(tmp_path)
    _demote_paper_source_to_supplement(proposal, manifest)
    declared = {
        row["source_id"]
        for row in proposal["proposed_facts"]["source_resolution"]["sources"]
    }
    assert PAPER_SOURCE_ID in declared, "the fixture must actually exercise the supplement"

    bound = _bind(proposal, manifest)
    assert set(bound) == {row["source_id"] for row in manifest_source_rows(manifest)}
    assert PAPER_SOURCE_ID in bound


def test_the_binding_still_refuses_a_source_that_is_in_neither_half(tmp_path: Path) -> None:
    """Widening added a citable row; it must not have opened the set.

    The tripwire's whole point is that an author cannot name a document the
    machine never fetched, and a proposal-side row with no manifest row behind it
    is exactly that.
    """

    proposal, manifest = _ground_proposal(tmp_path)
    _demote_paper_source_to_supplement(proposal, manifest)
    invented = dict(proposal["proposed_facts"]["source_resolution"]["sources"][0])
    invented["source_id"] = "source-nobody-fetched"
    proposal["proposed_facts"]["source_resolution"]["sources"].append(invented)

    with pytest.raises(Exception) as refused:
        _bind(proposal, manifest)
    assert "source sets differ" in str(refused.value)


def test_the_binding_still_refuses_a_proposal_that_drops_a_fetched_source(
    tmp_path: Path,
) -> None:
    """The echo stays exhaustive in the other direction too.

    This is the m9617 shape: the author cited its supplement correctly but
    silently omitted seven frozen sources it had decided not to use. Dropping a
    source we fetched is not an editorial choice -- it removes the custody
    attestation for those bytes -- so it must keep failing, and the obligation is
    now stated on ``source_resolution.sources`` where the author reads it.
    """

    proposal, manifest = _ground_proposal(tmp_path)
    _demote_paper_source_to_supplement(proposal, manifest)
    declared = proposal["proposed_facts"]["source_resolution"]["sources"]
    dropped = next(row for row in declared if row["source_id"] != PAPER_SOURCE_ID)
    declared.remove(dropped)

    with pytest.raises(Exception) as refused:
        _bind(proposal, manifest)
    assert "source sets differ" in str(refused.value)


# -- the lane: ingesting the pack the executor actually wrote ----------------


ABS_URL = "https://arxiv.org/abs/2111.11418"
RECORD_URL = "https://api.example.org/works/10.1109-ACCESS.2020.3025372"
ABS_BODY = b"<html><body>Submitted 22 November 2021. MetaFormer Is Actually What You Need.</body></html>"
RECORD_BODY = b'{"venue": "IEEE Access", "year": 2020, "title": "MA-Net"}'


class _IngestOnlyLane(_AuthorLaneBase):
    """The author lane reduced to the one member the supplement ingest reads."""

    def __init__(self, effort_grant: AuthorEffortGrant) -> None:
        self.effort_grant = effort_grant


def _supplement_pack(root: Path) -> tuple[Path, list[dict[str, Any]]]:
    """Broker a REAL two-source supplementary pack under one attempt.

    Two sources, not one: a pack frozen at width one cannot catch an ingest that
    keeps only the first row, which is exactly the class of bug that hides in a
    seam like this one.

    Parameters
    ----------
    root:
        Per-model author custody root.

    Returns
    -------
    tuple[Path, list[dict[str, Any]]]
        Attempt directory, and the brokered supplementary rows.
    """

    attempt = new_attempt(root, stable_id="m-supp", campaign_id="c1-mech", kind="author")
    transport = MapTransport({ABS_URL: (200, ABS_BODY), RECORD_URL: (200, RECORD_BODY)})
    pack = broker_source_pack(
        [
            {
                "source_id": "paper-abs-page",
                "kind": "raw-url",
                "url": ABS_URL,
                "requested_role": "paper",
                "basis": "The introducing paper's own abstract page carries the year.",
            },
            {
                "source_id": "paper-record-s2",
                "kind": "raw-url",
                "url": RECORD_URL,
                "requested_role": "paper",
                "basis": "The DOI-keyed bibliographic record states venue and authors.",
            },
        ],
        broker_dir=attempt.paths.broker / "supplement",
        transport=transport,
    )
    assert len(pack.rows) == 2, "fixture must broker BOTH supplementary sources"
    write_broker_outputs(pack, attempt.paths.broker / "supplement")
    manifest_path = attempt.paths.directory / "supplement-manifest.json"
    manifest_path.write_text(json.dumps(pack.to_dict()), encoding="utf-8")
    attempt.update(supplement={"manifest_path": str(manifest_path)})
    return attempt.paths.directory, list(pack.rows)


@pytest.mark.smoke
def test_the_lane_ingests_the_pack_the_executor_granted(tmp_path: Path) -> None:
    """The defect, at the seam where it lived: the pack reaches private custody.

    Before this, ``supplement-manifest.json`` was written, resumed against, and
    then read by nobody. Both rows must land in the CAS and in the extended
    manifest, and the frozen identity must be untouched.
    """

    root = tmp_path / "author"
    root.mkdir()
    _supplement_pack(root)
    frozen: dict[str, Any] = {
        "sources": [
            {
                "source_id": "impl-frozen",
                "url": "https://example.com/impl",
                "revision": "v1",
                "content_sha256": hash_bytes(b"frozen"),
            }
        ]
    }
    frozen["manifest_sha256"] = stable_hash(frozen["sources"])

    extended = _IngestOnlyLane(AuthorEffortGrant())._extend_manifest_with_supplement(
        _work_item("m-supp"), root, dict(frozen)
    )

    assert extended["sources"] == frozen["sources"]
    assert extended["manifest_sha256"] == frozen["manifest_sha256"]
    assert [row["source_id"] for row in extended["supplementary_sources"]] == [
        "paper-abs-page",
        "paper-record-s2",
    ]
    # The bytes are really in the model's own CAS, not merely named.
    by_id = {str(row["source_id"]): row for row in extended["supplementary_sources"]}
    assert Path(by_id["paper-abs-page"]["cas_path"]).read_bytes() == ABS_BODY
    assert Path(by_id["paper-record-s2"]["cas_path"]).read_bytes() == RECORD_BODY
    assert by_id["paper-abs-page"]["content_sha256"] == hash_bytes(ABS_BODY)
    assert by_id["paper-record-s2"]["content_sha256"] == hash_bytes(RECORD_BODY)


def test_the_ingest_is_recorded_so_a_source_can_never_vanish_silently(
    tmp_path: Path,
) -> None:
    """The defect was silence as much as loss; the custody file must say so."""

    root = tmp_path / "author"
    root.mkdir()
    attempt_dir, _ = _supplement_pack(root)
    custody_path = root / "source-custody.json"
    write_envelope_atomic(
        {
            "provenance_version": "menagerie.crawler.source-custody-provenance.v1",
            "stable_id": "m-supp",
            "work_id": "work-m-supp",
            "promotions": [{"source_id": "impl-frozen", "disposition": "promoted"}],
            "upstream_drift": [],
        },
        custody_path,
    )
    frozen: dict[str, Any] = {"sources": [{"source_id": "impl-frozen"}]}
    frozen["manifest_sha256"] = stable_hash(frozen["sources"])

    _IngestOnlyLane(AuthorEffortGrant())._extend_manifest_with_supplement(
        _work_item("m-supp"), root, dict(frozen)
    )

    custody = json.loads(custody_path.read_text(encoding="utf-8"))
    assert custody["supplement_manifest_path"] == str(attempt_dir / "supplement-manifest.json")
    assert [entry["source_id"] for entry in custody["supplement_promotions"]] == [
        "paper-abs-page",
        "paper-record-s2",
    ]
    assert all(entry["disposition"] == "promoted" for entry in custody["supplement_promotions"])
    # The stage-1 record is preserved, not overwritten by the extension.
    assert custody["promotions"] == [{"source_id": "impl-frozen", "disposition": "promoted"}]


def test_a_supplementary_row_may_never_redefine_a_frozen_source_id(
    tmp_path: Path,
) -> None:
    """Repointing an already-quoted identifier is the one thing freezing prevents."""

    root = tmp_path / "author"
    root.mkdir()
    _supplement_pack(root)
    frozen: dict[str, Any] = {"sources": [{"source_id": "paper-record-s2"}]}
    frozen["manifest_sha256"] = stable_hash(frozen["sources"])

    with pytest.raises(DriverIntegrationError) as refused:
        _IngestOnlyLane(AuthorEffortGrant())._extend_manifest_with_supplement(
            _work_item("m-supp"), root, dict(frozen)
        )
    assert str(refused.value) == "supplementary source may not redefine a frozen manifest source_id"


def test_a_lane_with_no_supplement_round_returns_the_manifest_untouched(
    tmp_path: Path,
) -> None:
    """The common path must not acquire a key, a file, or a fetch."""

    root = tmp_path / "author"
    root.mkdir()
    new_attempt(root, stable_id="m-plain", campaign_id="c1-mech", kind="author")
    frozen: dict[str, Any] = {"sources": [{"source_id": "impl-frozen"}]}
    frozen["manifest_sha256"] = stable_hash(frozen["sources"])

    extended = _IngestOnlyLane(AuthorEffortGrant())._extend_manifest_with_supplement(
        _work_item("m-plain"), root, dict(frozen)
    )

    assert extended == frozen
    assert not (root / "source-custody.json").exists()
