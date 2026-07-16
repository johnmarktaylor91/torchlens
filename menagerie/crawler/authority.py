"""Frozen round-14 authority, artifact, capability, and lifecycle contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping, NewType, Optional

from menagerie.crawler.models import JsonObject

MirrorObjectId = NewType("MirrorObjectId", str)
ArtifactObjectId = MirrorObjectId
ObjectId = MirrorObjectId
ArtifactClaimId = NewType("ArtifactClaimId", str)
ClaimId = ArtifactClaimId
ArtifactTransactionId = NewType("ArtifactTransactionId", str)
PublicationAuthorizationId = NewType("PublicationAuthorizationId", str)


class DependencyState(str, Enum):
    """Typed non-identity states permitted on a dependency-vector axis."""

    NOT_APPLICABLE = "not-applicable"
    PENDING_UNTRUSTED = "pending-untrusted"


DependencyValue = str | DependencyState


@dataclass(frozen=True)
class AuthorityContext:
    """Mandatory active trust roots and policy-closure identities.

    Parameters
    ----------
    active_intake_snapshot_id, active_intake_snapshot_sha256:
        Exact validated active intake snapshot identity.
    intake_by_stable_id:
        Full verified intake rows keyed by stable model identity.
    family_bindings:
        Trusted intake-derived family bindings keyed by stable identity.
    author_prompt_identity, author_model_identity, author_schema_identity,
    author_dispatcher_identity:
        Current author contract identities.
    checker_prompt_identity, checker_model_identity, checker_schema_identity:
        Current checker contract identities.
    environment_generations:
        Current exact environment identities keyed by environment name.
    reducer_policy_identity, runner_policy_identity, terminal_policy_identity,
    publication_policy_identity:
        Versioned closure identities for reducer-owned decisions.
    """

    active_intake_snapshot_id: str
    active_intake_snapshot_sha256: str
    intake_by_stable_id: Mapping[str, JsonObject]
    family_bindings: Mapping[str, JsonObject]
    author_prompt_identity: str
    author_model_identity: str
    author_schema_identity: str
    author_dispatcher_identity: str
    checker_prompt_identity: str
    checker_model_identity: str
    checker_schema_identity: str
    environment_generations: Mapping[str, str]
    reducer_policy_identity: str
    runner_policy_identity: str
    terminal_policy_identity: str
    publication_policy_identity: str


@dataclass(frozen=True)
class DependencyVector:
    """Closed stage-sensitive identity vector for one canonical revision."""

    intake_snapshot_id: DependencyValue
    intake_snapshot_sha256: DependencyValue
    intake_item_sha256: DependencyValue
    author_result_schema_identity: DependencyValue
    author_dispatcher_identity: DependencyValue
    author_prompt_identity: DependencyValue
    checker_prompt_identity: DependencyValue
    terminal_rule_identity: DependencyValue
    status_proof_identity: DependencyValue
    source_manifest_identity: DependencyValue
    proposal_identity: DependencyValue
    author_result_identity: DependencyValue
    checker_gate_identity: DependencyValue
    recipe_revision: DependencyValue
    runner_identity: DependencyValue
    award_closure_identity: DependencyValue
    environment_generation: DependencyValue
    accepted_attempt_ids: tuple[str, ...]
    artifact_transaction_id: DependencyValue
    artifact_claim_ids: tuple[ArtifactClaimId, ...]
    representative_revision: DependencyValue
    publication_policy_identity: DependencyValue


@dataclass(frozen=True)
class AttemptAuthority:
    """Reducer-verified association between one attempt and its raw proof."""

    attempt_id: str
    stable_id: str
    work_id: str
    execution_identity: str
    request_identity: str
    raw_award_receipt_sha256: str
    parent_attestation_sha256: str


@dataclass(frozen=True)
class ModeSummary:
    """Reducer-derived comparison over authenticated per-mode attempts."""

    comparison_state: str
    classification: str
    train_attempt_id: Optional[str]
    eval_attempt_id: Optional[str]
    compared_fields: tuple[str, ...]
    evidence_sha256: str


@dataclass(frozen=True)
class TerminalProof:
    """Closed reducer-derived semantic proof for one terminal disposition."""

    proof_id: str
    proof_rule_identity: str
    stable_id: str
    work_id: str
    status_code: str
    decisive_attempt_ids: tuple[str, ...]
    gate_id: DependencyValue
    source_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    failure_stage: DependencyValue
    reason_code: DependencyValue
    root_cause_fingerprint: DependencyValue
    platform_claim: DependencyValue
    per_mode_attempt_ids: tuple[tuple[str, str], ...]
    terminal_observation_sha256: str


@dataclass(frozen=True)
class FamilyAuthority:
    """Trusted intake-derived representative binding for every family member."""

    stable_id: str
    representative_stable_id: DependencyValue
    representative_revision: DependencyValue
    representative_gate_id: DependencyValue
    representative_proposal_id: DependencyValue
    variant_token: DependencyValue
    template_source_revision: DependencyValue
    derivation_rule_identity: DependencyValue


@dataclass(frozen=True)
class MirrorObject:
    """Intrinsic physical-object identity, independent of model provenance."""

    object_id: MirrorObjectId
    mirror_class: str
    content_sha256: str
    byte_count: int
    media_type: str
    object_key: str


@dataclass(frozen=True)
class ArtifactClaim:
    """Model-specific provenance and license claim over one mirror object."""

    claim_id: ArtifactClaimId
    object_id: MirrorObjectId
    stable_id: str
    work_id: str
    proposal_id: DependencyValue
    gate_id: DependencyValue
    authorization_id: DependencyValue
    logical_role: str
    logical_path: str
    source_id: str
    origin: str
    revision: str
    fetch_recipe_sha256: str
    evidence_ids: tuple[str, ...]
    license_disposition: str


@dataclass(frozen=True)
class PublicationAuthorization:
    """Reducer-created capability required for any public artifact write."""

    authorization_id: PublicationAuthorizationId
    stable_id: str
    work_id: str
    transaction_id: ArtifactTransactionId
    accepted_gate_id: str
    accepted_gate_item_sha256: str
    dependency_vector: DependencyVector
    claim_ids: tuple[ArtifactClaimId, ...]
    public_object_ids: tuple[MirrorObjectId, ...]
    private_object_ids: tuple[MirrorObjectId, ...]
    publication_policy_identity: str


@dataclass(frozen=True)
class ExecutionReadManifest:
    """Trusted, digest-bound semantic read capability for one worker request."""

    manifest_id: str
    stable_id: str
    work_id: str
    execution_identity: str
    code_manifest_identity: str
    code_members: tuple[tuple[Path, str, str], ...]
    standard_input_asset: Optional[tuple[Path, str, str]]
    runtime_support: tuple[tuple[Path, str], ...]


@dataclass(frozen=True)
class WorkerLease:
    """Durable metadata that augments the child-held worker kernel lock."""

    lease_id: str
    nonce: str
    run_id: str
    stable_id: str
    work_id: str
    request_identity: str
    execution_identity: str
    boot_id: str
    driver_pid: int
    driver_start_token: str
    child_pid: Optional[int]
    child_start_token: Optional[str]
    child_pgid: Optional[int]
    receipt_path: Path
    opened_at: str
    deadline_at: str


@dataclass(frozen=True)
class WakeEpisode:
    """Durable recurring usage-limit wake episode derived from operations."""

    episode_id: str
    provider: str
    reset_at: str
    reset_observation: str
    not_before: str
    retry_interval_seconds: int
    callback_identity: str
    callback_argv: tuple[str, ...]
    opened_event_id: str
    supersedes_episode_id: Optional[str]


@dataclass(frozen=True)
class DependencyCurrencyProjection:
    """Single dependency-current projection consumed by every read surface."""

    current_records: Mapping[str, JsonObject]
    stale_reasons: Mapping[str, str]
    stale_stable_ids: frozenset[str]

    def __init__(
        self,
        current_records: Mapping[str, JsonObject],
        stale_reasons: Mapping[str, str],
        stale_stable_ids: Optional[frozenset[str]] = None,
    ) -> None:
        """Initialize the projection and derive its closed stale-ID set.

        Parameters
        ----------
        current_records:
            Highest revisions that remain dependency-current.
        stale_reasons:
            Stable-ID keyed reasons for excluding highest revisions.
        stale_stable_ids:
            Exact stale identity set. Omission derives it from ``stale_reasons``
            for source compatibility during the interface freeze.
        """

        object.__setattr__(self, "current_records", current_records)
        object.__setattr__(self, "stale_reasons", stale_reasons)
        object.__setattr__(
            self,
            "stale_stable_ids",
            stale_stable_ids if stale_stable_ids is not None else frozenset(stale_reasons),
        )
