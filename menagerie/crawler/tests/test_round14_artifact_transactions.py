"""Round-14 Root-B artifact transaction and checkpoint regressions."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from menagerie.crawler.artifact_transactions import (
    ArtifactCheckpointError,
    ArtifactEventKind,
    ArtifactEventLedger,
    ArtifactInput,
    ArtifactPublicationError,
    ReconstructionInputs,
    append_artifact_authorization,
    derive_artifact_claims,
    derive_publication_authorization_id,
    publish_authorized_artifact,
    stage_private_artifact,
    validate_artifact_checkpoint,
)
from menagerie.crawler.authority import (
    ArtifactClaimId,
    AuthorityContext,
    DependencyState,
    DependencyValue,
    DependencyVector,
    PublicationAuthorization,
    PublicationAuthorizationId,
)
from menagerie.crawler.constants import (
    AUTHOR_PROPOSAL_SCHEMA_VERSION_V3,
    AUTHOR_RESULT_SCHEMA_VERSION,
)
from menagerie.crawler.identity import hash_bytes, stable_hash
from menagerie.crawler.licenses import LicenseDecision, RedistributionClass
from menagerie.crawler.mirrors import ArtifactOrigin, MirrorClass, MirrorStore
from menagerie.crawler.tests.conftest import make_author_proposal

HASH = "sha256:" + "a" * 64
NOW = "2026-07-16T12:00:00Z"


def _context(*stable_ids: str) -> AuthorityContext:
    """Build exact active authority for focused artifact tests.

    Parameters
    ----------
    stable_ids:
        Trusted intake identities.

    Returns
    -------
    AuthorityContext
        Frozen test authority.
    """

    intake = {
        stable_id: {"stable_id": stable_id, "natural_key": f"fixture:{stable_id}"}
        for stable_id in stable_ids
    }
    return AuthorityContext(
        active_intake_snapshot_id="intake-test",
        active_intake_snapshot_sha256=HASH,
        intake_by_stable_id=intake,
        family_bindings={},
        author_prompt_identity=HASH,
        author_model_identity=HASH,
        author_schema_identity=HASH,
        author_dispatcher_identity=HASH,
        checker_prompt_identity=HASH,
        checker_model_identity=HASH,
        checker_schema_identity=HASH,
        environment_generations={},
        reducer_policy_identity=HASH,
        runner_policy_identity=HASH,
        terminal_policy_identity=HASH,
        publication_policy_identity=HASH,
    )


def _source_manifest(content: bytes) -> dict[str, Any]:
    """Build one exact controlled-fetch source manifest.

    Parameters
    ----------
    content:
        Frozen upstream bytes.

    Returns
    -------
    dict[str, Any]
        Hash-bound source manifest.
    """

    source = {
        "source_id": "src-upstream",
        "url": "https://example.test/source.tar",
        "revision": "v1",
        "content_sha256": hash_bytes(content),
        "fetched_bytes_len": len(content),
        "media_type": "application/x-tar",
    }
    return {"sources": [source], "manifest_sha256": stable_hash([source])}


def _blocked_result(
    stable_id: str, context: AuthorityContext, source_manifest: dict[str, Any]
) -> dict[str, Any]:
    """Build one valid typed blocked recommendation.

    Parameters
    ----------
    stable_id, context, source_manifest:
        Exact model, active trust roots, and source manifest.

    Returns
    -------
    dict[str, Any]
        Valid ``author-result.v3`` mapping.
    """

    payload = {
        "arm": "BLOCKED",
        "stage": "source",
        "reason_code": "missing-mandatory-link",
        "prerequisite_ids": ["src-upstream"],
        "evidence_ids": ["ev-source"],
        "evidence_identity": HASH,
        "license_identity": HASH,
        "recommendation_sha256": HASH,
    }
    result = {
        "schema_version": AUTHOR_RESULT_SCHEMA_VERSION,
        "result_id": f"result-{stable_id}",
        "result_sha256": HASH,
        "kind": "BLOCKED",
        "stable_id": stable_id,
        "work_id": f"work-{stable_id}",
        "campaign_id": "campaign-test",
        "created_at": NOW,
        "author_identity": context.author_model_identity,
        "prompt_identity": context.author_prompt_identity,
        "dispatcher_identity": context.author_dispatcher_identity,
        "source_manifest_identity": source_manifest["manifest_sha256"],
        "intake_snapshot_id": context.active_intake_snapshot_id,
        "intake_snapshot_sha256": context.active_intake_snapshot_sha256,
        "intake_item_sha256": stable_hash(context.intake_by_stable_id[stable_id]),
        "payload": payload,
    }
    result["result_sha256"] = stable_hash(
        {key: value for key, value in result.items() if key != "result_sha256"}
    )
    return result


def _proposed_result(
    stable_id: str, context: AuthorityContext, content: bytes
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build a valid proposed result and matching controlled source manifest.

    Parameters
    ----------
    stable_id, context, content:
        Exact model, active trust roots, and upstream bytes.

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any], dict[str, Any]]
        Typed author result, embedded proposal, and source manifest.
    """

    proposal = make_author_proposal(stable_id)
    digest = hash_bytes(content)
    source = proposal["proposed_facts"]["source_resolution"]["sources"][0]
    source.update(
        {
            "source_id": "src-upstream",
            "url": "https://example.test/source.tar",
            "revision": "v1",
            "content_sha256": digest,
            "byte_count": len(content),
            "media_type": "application/x-tar",
        }
    )
    fetched_source = {
        "source_id": "src-upstream",
        "url": "https://example.test/source.tar",
        "revision": "v1",
        "content_sha256": digest,
        "fetched_bytes_len": len(content),
        "media_type": "application/x-tar",
    }
    source_manifest = {
        "sources": [fetched_source],
        "manifest_sha256": stable_hash([fetched_source]),
    }
    proposal.update(
        {
            "schema_version": AUTHOR_PROPOSAL_SCHEMA_VERSION_V3,
            "campaign_id": "campaign-test",
            "intake_snapshot_id": context.active_intake_snapshot_id,
            "intake_snapshot_sha256": context.active_intake_snapshot_sha256,
            "intake_item_sha256": stable_hash(context.intake_by_stable_id[stable_id]),
            "source_manifest_identity": source_manifest["manifest_sha256"],
            "dispatcher_identity": context.author_dispatcher_identity,
        }
    )
    proposal["author"]["prompt_sha256"] = context.author_prompt_identity
    proposal["verified_hashes"]["source_manifest"] = source_manifest["manifest_sha256"]
    proposal["proposal_sha256"] = stable_hash(
        {key: value for key, value in proposal.items() if key != "proposal_sha256"}
    )
    result = {
        "schema_version": AUTHOR_RESULT_SCHEMA_VERSION,
        "result_id": f"result-{stable_id}",
        "result_sha256": HASH,
        "kind": "PROPOSED",
        "stable_id": stable_id,
        "work_id": f"work-{stable_id}",
        "campaign_id": "campaign-test",
        "created_at": NOW,
        "author_identity": context.author_model_identity,
        "prompt_identity": context.author_prompt_identity,
        "dispatcher_identity": context.author_dispatcher_identity,
        "source_manifest_identity": source_manifest["manifest_sha256"],
        "intake_snapshot_id": context.active_intake_snapshot_id,
        "intake_snapshot_sha256": context.active_intake_snapshot_sha256,
        "intake_item_sha256": stable_hash(context.intake_by_stable_id[stable_id]),
        "payload": {"arm": "PROPOSED", "proposal": proposal},
    }
    result["result_sha256"] = stable_hash(
        {key: value for key, value in result.items() if key != "result_sha256"}
    )
    return result, proposal, source_manifest


def _mirrors(tmp_path: Path) -> MirrorStore:
    """Create separated physical mirrors.

    Parameters
    ----------
    tmp_path:
        Test root.

    Returns
    -------
    MirrorStore
        Separated public/private/local roots.
    """

    return MirrorStore(
        tmp_path / "mirror-public",
        tmp_path / "mirror-private",
        tmp_path / "mirror-local",
    )


def _vector(
    context: AuthorityContext,
    stable_id: str,
    transaction_id: str,
    result_id: str,
    source_manifest_identity: str,
    claim_ids: tuple[ArtifactClaimId, ...] = (),
    proposal_identity: DependencyValue = DependencyState.NOT_APPLICABLE,
) -> DependencyVector:
    """Build a transaction-bound dependency vector.

    Parameters
    ----------
    context, stable_id, transaction_id, result_id, source_manifest_identity,
    claim_ids, proposal_identity:
        Exact authority axes and optional accepted claim set.

    Returns
    -------
    DependencyVector
        Frozen dependency vector.
    """

    return DependencyVector(
        intake_snapshot_id=context.active_intake_snapshot_id,
        intake_snapshot_sha256=context.active_intake_snapshot_sha256,
        intake_item_sha256=stable_hash(context.intake_by_stable_id[stable_id]),
        author_result_schema_identity=context.author_schema_identity,
        author_dispatcher_identity=context.author_dispatcher_identity,
        author_prompt_identity=context.author_prompt_identity,
        checker_prompt_identity=context.checker_prompt_identity,
        terminal_rule_identity=context.terminal_policy_identity,
        status_proof_identity=HASH,
        source_manifest_identity=source_manifest_identity,
        proposal_identity=proposal_identity,
        author_result_identity=result_id,
        checker_gate_identity="gate-terminal",
        recipe_revision=DependencyState.NOT_APPLICABLE,
        runner_identity=DependencyState.NOT_APPLICABLE,
        award_closure_identity=DependencyState.NOT_APPLICABLE,
        environment_generation=DependencyState.NOT_APPLICABLE,
        accepted_attempt_ids=(),
        artifact_transaction_id=transaction_id,
        artifact_claim_ids=claim_ids,
        representative_revision=DependencyState.NOT_APPLICABLE,
        publication_policy_identity=context.publication_policy_identity,
    )


def _gate_item(stable_id: str, result: dict[str, Any], source_identity: str) -> dict[str, Any]:
    """Build one accepted terminal gate item.

    Parameters
    ----------
    stable_id, result, source_identity:
        Exact model, author result, and source-manifest identity.

    Returns
    -------
    dict[str, Any]
        Accepted terminal-disposition item.
    """

    return {
        "stable_id": stable_id,
        "work_id": f"work-{stable_id}",
        "terminal_disposition": {
            "verdict": "accepted",
            "author_result_id": result["result_id"],
            "author_result_sha256": result["result_sha256"],
            "source_manifest_identity": source_identity,
            "source_ids": ["src-upstream"],
        },
    }


def _commit_private(
    stable_id: str,
    *,
    content: bytes,
    context: AuthorityContext,
    mirrors: MirrorStore,
    ledger: ArtifactEventLedger,
    canonical_root: Path,
    repository_root: Path,
    evidence_id: str,
) -> tuple[Any, Any]:
    """Stage, authorize, and privately commit one test transaction.

    Parameters
    ----------
    stable_id, content, context, mirrors, ledger, canonical_root,
    repository_root, evidence_id:
        Complete deterministic transaction inputs.

    Returns
    -------
    tuple[StagedArtifact, PublishedArtifact]
        Private stage and completed commitment.
    """

    source_manifest = _source_manifest(content)
    result = _blocked_result(stable_id, context, source_manifest)
    digest = hash_bytes(content)
    staged = stage_private_artifact(
        (
            ArtifactInput(
                content=content,
                content_sha256=digest,
                logical_role="source",
                logical_path=f"menagerie/crawler/source_cas/{digest[7:]}.source",
                source_id="src-upstream",
                origin=ArtifactOrigin("https://example.test/source.tar", "v1"),
                fetch_recipe="GET exact source.tar@v1",
                evidence_ids=(evidence_id,),
                media_type="application/x-tar",
            ),
        ),
        context=context,
        stable_id=stable_id,
        work_id=f"work-{stable_id}",
        author_result=result,
        proposal=None,
        source_manifest=source_manifest,
        mirrors=mirrors,
        ledger=ledger,
        created_at=NOW,
    )
    gate_item = _gate_item(stable_id, result, source_manifest["manifest_sha256"])
    provisional_vector = _vector(
        context,
        stable_id,
        str(staged.transaction_id),
        result["result_id"],
        source_manifest["manifest_sha256"],
    )
    decision = LicenseDecision(
        content_sha256=digest,
        redistribution_class=RedistributionClass.RESTRICTED_PRIVATE,
        evidence_ids=(evidence_id,),
        rationale="restricted fixture",
    )
    decisions = {staged.custody_claims[0].claim_id: decision}
    authorization_id = derive_publication_authorization_id(
        staged,
        accepted_gate_id="gate-terminal",
        accepted_gate_item_sha256=stable_hash(gate_item),
        dependency_vector=provisional_vector,
        decisions=decisions,
        publication_policy_identity=context.publication_policy_identity,
    )
    claims = derive_artifact_claims(
        staged,
        accepted_gate_id="gate-terminal",
        authorization_id=authorization_id,
        decisions=decisions,
        mirrors=mirrors,
    )
    vector = replace(
        provisional_vector, artifact_claim_ids=tuple(claim.claim_id for claim in claims)
    )
    assert authorization_id == derive_publication_authorization_id(
        staged,
        accepted_gate_id="gate-terminal",
        accepted_gate_item_sha256=stable_hash(gate_item),
        dependency_vector=vector,
        decisions=decisions,
        publication_policy_identity=context.publication_policy_identity,
    )
    authorization = PublicationAuthorization(
        authorization_id=authorization_id,
        stable_id=stable_id,
        work_id=f"work-{stable_id}",
        transaction_id=staged.transaction_id,
        accepted_gate_id="gate-terminal",
        accepted_gate_item_sha256=stable_hash(gate_item),
        dependency_vector=vector,
        claim_ids=tuple(claim.claim_id for claim in claims),
        public_object_ids=(),
        private_object_ids=tuple(claim.object_id for claim in claims),
        publication_policy_identity=context.publication_policy_identity,
    )
    append_artifact_authorization(
        staged,
        authorization,
        claims,
        accepted_gate_item=gate_item,
        event_kind=ArtifactEventKind.TERMINAL_AUTHORIZED,
        context=context,
        mirrors=mirrors,
        ledger=ledger,
        created_at=NOW,
    )
    published = publish_authorized_artifact(
        staged,
        authorization,
        reconstruction_inputs=ReconstructionInputs(
            author_result=result,
            proposal=None,
            source_manifest=source_manifest,
            accepted_gate_item=gate_item,
        ),
        context=context,
        mirrors=mirrors,
        ledger=ledger,
        canonical_root=canonical_root,
        repository_root=repository_root,
        created_at=NOW,
    )
    return staged, published


def test_private_first_transaction_commits_immutable_reconstruction(tmp_path: Path) -> None:
    """All bytes remain private while reconstruction is ledger anchored."""

    repo = tmp_path / "repo"
    canonical = repo / "menagerie" / "crawler"
    ledger_path = canonical / "records" / "artifacts" / "shard.jsonl"
    context = _context("m_one")
    mirrors = _mirrors(tmp_path)
    with ArtifactEventLedger(ledger_path) as ledger:
        staged, published = _commit_private(
            "m_one",
            content=b"shared upstream bytes",
            context=context,
            mirrors=mirrors,
            ledger=ledger,
            canonical_root=canonical,
            repository_root=repo,
            evidence_id="ev-one",
        )
        assert [event["event_kind"] for event in ledger.events] == [
            "staged-private",
            "terminal-authorized",
            "reconstruction-committed",
            "private-committed",
        ]
        assert published.reconstruction_path.is_file()
        assert not (repo / staged.custody_claims[0].logical_path).exists()
        validate_artifact_checkpoint(
            (ledger_path,),
            context=context,
            mirrors=mirrors,
            canonical_root=canonical,
            repository_root=repo,
        )


def test_publication_occurs_only_after_committed_public_authorization(tmp_path: Path) -> None:
    """Accepted proposed bytes publish from retained private custody."""

    repo = tmp_path / "repo"
    canonical = repo / "menagerie" / "crawler"
    ledger_path = canonical / "records" / "artifacts" / "shard.jsonl"
    context = _context("m_public")
    mirrors = _mirrors(tmp_path)
    content = b"public-compatible source"
    result, proposal, source_manifest = _proposed_result("m_public", context, content)
    digest = hash_bytes(content)
    logical_path = f"menagerie/crawler/source_cas/{digest[7:]}.source"
    with ArtifactEventLedger(ledger_path) as ledger:
        staged = stage_private_artifact(
            (
                ArtifactInput(
                    content=content,
                    content_sha256=digest,
                    logical_role="source",
                    logical_path=logical_path,
                    source_id="src-upstream",
                    origin=ArtifactOrigin("https://example.test/source.tar", "v1"),
                    fetch_recipe="GET exact source.tar@v1",
                    evidence_ids=("ev-license",),
                    media_type="application/x-tar",
                ),
            ),
            context=context,
            stable_id="m_public",
            work_id="work-m_public",
            author_result=result,
            proposal=proposal,
            source_manifest=source_manifest,
            mirrors=mirrors,
            ledger=ledger,
            created_at=NOW,
        )
        assert tuple(mirrors.iter_objects(MirrorClass.PUBLIC)) == ()
        gate_item = {
            "stable_id": "m_public",
            "work_id": "work-m_public",
            "verdict": "accurate",
            "integrity": {"verdict": "accurate"},
            "rung_check": {"verdict": "accurate"},
            "verified_hashes": {"source_manifest": source_manifest["manifest_sha256"]},
        }
        provisional_vector = _vector(
            context,
            "m_public",
            str(staged.transaction_id),
            result["result_id"],
            source_manifest["manifest_sha256"],
            proposal_identity=proposal["proposal_id"],
        )
        decision = LicenseDecision(
            content_sha256=digest,
            redistribution_class=RedistributionClass.PUBLIC_OK,
            evidence_ids=("ev-license",),
            rationale="public fixture",
        )
        decisions = {staged.custody_claims[0].claim_id: decision}
        authorization_id = derive_publication_authorization_id(
            staged,
            accepted_gate_id="gate-public",
            accepted_gate_item_sha256=stable_hash(gate_item),
            dependency_vector=provisional_vector,
            decisions=decisions,
            publication_policy_identity=context.publication_policy_identity,
        )
        claims = derive_artifact_claims(
            staged,
            accepted_gate_id="gate-public",
            authorization_id=authorization_id,
            decisions=decisions,
            mirrors=mirrors,
        )
        vector = replace(
            provisional_vector,
            artifact_claim_ids=tuple(claim.claim_id for claim in claims),
        )
        authorization = PublicationAuthorization(
            authorization_id=authorization_id,
            stable_id="m_public",
            work_id="work-m_public",
            transaction_id=staged.transaction_id,
            accepted_gate_id="gate-public",
            accepted_gate_item_sha256=stable_hash(gate_item),
            dependency_vector=vector,
            claim_ids=tuple(claim.claim_id for claim in claims),
            public_object_ids=tuple(claim.object_id for claim in claims),
            private_object_ids=(),
            publication_policy_identity=context.publication_policy_identity,
        )
        append_artifact_authorization(
            staged,
            authorization,
            claims,
            accepted_gate_item=gate_item,
            event_kind=ArtifactEventKind.PUBLICATION_AUTHORIZED,
            context=context,
            mirrors=mirrors,
            ledger=ledger,
            created_at=NOW,
        )
        assert tuple(mirrors.iter_objects(MirrorClass.PUBLIC)) == ()
        publish_authorized_artifact(
            staged,
            authorization,
            reconstruction_inputs=ReconstructionInputs(
                author_result=result,
                proposal=proposal,
                source_manifest=source_manifest,
                accepted_gate_item=gate_item,
            ),
            context=context,
            mirrors=mirrors,
            ledger=ledger,
            canonical_root=canonical,
            repository_root=repo,
            created_at=NOW,
        )
        assert (repo / logical_path).read_bytes() == content
        assert len(tuple(mirrors.iter_objects(MirrorClass.PRIVATE))) == 1
        assert len(tuple(mirrors.iter_objects(MirrorClass.PUBLIC))) == 1
        validate_artifact_checkpoint(
            (ledger_path,),
            context=context,
            mirrors=mirrors,
            canonical_root=canonical,
            repository_root=repo,
        )


def test_publication_capability_without_ledger_authorization_writes_nothing(
    tmp_path: Path,
) -> None:
    """A directly constructed capability cannot cross the public API boundary."""

    repo = tmp_path / "repo"
    canonical = repo / "menagerie" / "crawler"
    ledger_path = canonical / "records" / "artifacts" / "shard.jsonl"
    context = _context("m_one")
    mirrors = _mirrors(tmp_path)
    content = b"private first"
    source_manifest = _source_manifest(content)
    result = _blocked_result("m_one", context, source_manifest)
    with ArtifactEventLedger(ledger_path) as ledger:
        staged = stage_private_artifact(
            (
                ArtifactInput(
                    content=content,
                    content_sha256=hash_bytes(content),
                    logical_role="source",
                    logical_path="menagerie/crawler/source_cas/object.source",
                    source_id="src-upstream",
                    origin=ArtifactOrigin("https://example.test/source.tar", "v1"),
                    fetch_recipe="GET exact source.tar@v1",
                    media_type="application/x-tar",
                ),
            ),
            context=context,
            stable_id="m_one",
            work_id="work-m_one",
            author_result=result,
            proposal=None,
            source_manifest=source_manifest,
            mirrors=mirrors,
            ledger=ledger,
            created_at=NOW,
        )
        vector = _vector(
            context,
            "m_one",
            str(staged.transaction_id),
            result["result_id"],
            source_manifest["manifest_sha256"],
        )
        forged = PublicationAuthorization(
            authorization_id=PublicationAuthorizationId("forged"),
            stable_id="m_one",
            work_id="work-m_one",
            transaction_id=staged.transaction_id,
            accepted_gate_id="gate-forged",
            accepted_gate_item_sha256=HASH,
            dependency_vector=vector,
            claim_ids=(),
            public_object_ids=(),
            private_object_ids=(),
            publication_policy_identity=context.publication_policy_identity,
        )
        with pytest.raises(ArtifactPublicationError, match="prior authorization"):
            publish_authorized_artifact(
                staged,
                forged,
                reconstruction_inputs=ReconstructionInputs(
                    author_result=result,
                    proposal=None,
                    source_manifest=source_manifest,
                    accepted_gate_item={},
                ),
                context=context,
                mirrors=mirrors,
                ledger=ledger,
                canonical_root=canonical,
                repository_root=repo,
            )
    assert tuple(mirrors.iter_objects(MirrorClass.PUBLIC)) == ()


def test_two_models_share_one_object_but_retain_independent_claims(tmp_path: Path) -> None:
    """Shared bytes deduplicate physically without source-claim collision."""

    repo = tmp_path / "repo"
    canonical = repo / "menagerie" / "crawler"
    ledger_path = canonical / "records" / "artifacts" / "shard.jsonl"
    context = _context("m_one", "m_two")
    mirrors = _mirrors(tmp_path)
    with ArtifactEventLedger(ledger_path) as ledger:
        first, _published_first = _commit_private(
            "m_one",
            content=b"same source object",
            context=context,
            mirrors=mirrors,
            ledger=ledger,
            canonical_root=canonical,
            repository_root=repo,
            evidence_id="ev-one",
        )
        second, _published_second = _commit_private(
            "m_two",
            content=b"same source object",
            context=context,
            mirrors=mirrors,
            ledger=ledger,
            canonical_root=canonical,
            repository_root=repo,
            evidence_id="ev-two",
        )
        assert first.objects[0].object_id == second.objects[0].object_id
        assert first.custody_claims[0].claim_id != second.custody_claims[0].claim_id
        assert len(tuple(mirrors.iter_objects(MirrorClass.PRIVATE))) == 1
        validate_artifact_checkpoint(
            (ledger_path,),
            context=context,
            mirrors=mirrors,
            canonical_root=canonical,
            repository_root=repo,
        )


def test_checkpoint_rejects_reconstruction_rewrite_and_mirror_orphan(tmp_path: Path) -> None:
    """Independent ledger anchors catch coherent-file rewrites and unknown objects."""

    repo = tmp_path / "repo"
    canonical = repo / "menagerie" / "crawler"
    ledger_path = canonical / "records" / "artifacts" / "shard.jsonl"
    context = _context("m_one")
    mirrors = _mirrors(tmp_path)
    with ArtifactEventLedger(ledger_path) as ledger:
        _staged, published = _commit_private(
            "m_one",
            content=b"checkpoint object",
            context=context,
            mirrors=mirrors,
            ledger=ledger,
            canonical_root=canonical,
            repository_root=repo,
            evidence_id="ev-one",
        )
    original = published.reconstruction_path.read_bytes()
    published.reconstruction_path.write_bytes(original + b" ")
    with pytest.raises(ArtifactCheckpointError, match="reconstruction bytes changed"):
        validate_artifact_checkpoint(
            (ledger_path,),
            context=context,
            mirrors=mirrors,
            canonical_root=canonical,
            repository_root=repo,
        )
    published.reconstruction_path.write_bytes(original)
    orphan = mirrors.root(MirrorClass.PRIVATE) / "unexpected.bin"
    orphan.parent.mkdir(parents=True, exist_ok=True)
    orphan.write_bytes(b"orphan")
    with pytest.raises(ArtifactCheckpointError, match="orphan"):
        validate_artifact_checkpoint(
            (ledger_path,),
            context=context,
            mirrors=mirrors,
            canonical_root=canonical,
            repository_root=repo,
        )
