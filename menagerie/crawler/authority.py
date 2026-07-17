"""Frozen round-14 authority contracts and pure proof derivations."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, NewType, Optional, Sequence

from menagerie.crawler.constants import FAILURE_REASON_CODES
from menagerie.crawler.identity import compute_execution_identity, hash_bytes, stable_hash
from menagerie.crawler.models import JsonObject

_RAW_AWARD_RECEIPT_VERSION = "menagerie.crawler.raw-award-receipt.v3"
_PARENT_ATTESTATION_VERSION = "menagerie.crawler.parent-attestation.v2"
_WORKER_COMPLETION_PREFIX = "MENAGERIE_WORKER_COMPLETION_V3 "
_RAW_RECEIPT_FIELDS = frozenset(
    {
        "receipt_version",
        "request_nonce",
        "request_sha256",
        "stable_id",
        "work_id",
        "execution_identity",
        "recipe_revision",
        "code_manifest_identity",
        "input_identity",
        "requested_mode",
        "observation",
    }
)
_PARENT_ATTESTATION_FIELDS = frozenset(
    {
        "attestation_version",
        "request_nonce",
        "request_sha256",
        "completion_line_sha256",
        "named_raw_award_receipt_sha256",
        "exit_code",
        "signal",
        "timed_out",
        "rss_exceeded",
        "peak_rss_bytes",
        "stdout_sha256",
        "stderr_sha256",
        "started_at",
        "finished_at",
        "attestation_sha256",
    }
)
_HASH_PREFIX = "sha256:"
_MODE_ORDER = {"train": 0, "eval": 1}
_POLICY_FIELDS = (
    "network_attempted",
    "checkpoint_or_weight_read_attempted",
    "write_outside_scratch_attempted",
    "credentials_present",
    "torchlens_import_attempted",
    "cache_read_attempted",
)
_POLICY_SEQUENCE_FIELDS = ("socket_targets", "checkpoint_paths", "write_paths")
_STATUS_RUNNER_STAGES = frozenset(
    {"environment", "import", "constructor", "input", "forward", "resource", "policy", "runner"}
)


class AuthorityDerivationError(ValueError):
    """Raised when retained facts do not form the frozen replayable proof graph."""


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
class RuntimeMember:
    """One exact digest-bound executable or runtime file in manifest v2.

    Parameters
    ----------
    path:
        Absolute regular unaliased member path.
    sha256:
        Canonical digest of the exact member bytes.
    kind:
        Closed executable/runtime member kind selected by the closure compiler.
    provenance:
        Exact compiler inventory or seed that admitted the member.
    """

    path: Path
    sha256: str
    kind: str
    provenance: str


@dataclass(frozen=True)
class RuntimeLookupDirectory:
    """Lookup-only directory scaffold that grants no child-file read authority."""

    path: Path
    provenance: str


@dataclass(frozen=True)
class ExecutionReadManifestV2:
    """Frozen v2 worker capability with no semantic filesystem-root grants.

    Every executable/runtime file is named by path and digest. Lookup directories
    exist only to support import and mount traversal and never authorize descendants.
    The environment generation and installed-package inventory digest are identity
    inputs so later producers can recompile and stale manifests from their real closure.
    """

    manifest_version: str
    manifest_id: str
    stable_id: str
    work_id: str
    execution_identity: str
    code_manifest_identity: str
    environment_generation: str
    installed_package_inventory_sha256: str
    code_members: tuple[RuntimeMember, ...]
    runtime_members: tuple[RuntimeMember, ...]
    standard_input_asset: Optional[tuple[Path, str, str]]
    lookup_directories: tuple[RuntimeLookupDirectory, ...]


@dataclass(frozen=True)
class ShutdownInterruptionFact:
    """Operational-only fact for one shutdown-interrupted worker invocation.

    A fact may describe a pre-spawn interruption, in which case lease, process,
    parent observation, and partial receipt fields are null. It never represents an
    attempt or model row and any partial receipt remains non-awarding diagnostics.
    """

    invocation_id: str
    admission_boundary: str
    stable_id: Optional[str]
    work_id: Optional[str]
    execution_identity: Optional[str]
    request_identity: Optional[str]
    lease_id: Optional[str]
    child_pid: Optional[int]
    child_start_token: Optional[str]
    child_pgid: Optional[int]
    signal: Optional[int]
    parent_observation: Optional[Mapping[str, Any]]
    partial_receipt: Optional[Mapping[str, Any]]


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


def build_authority_context(
    *,
    active_intake_snapshot_id: str,
    active_intake_snapshot_sha256: str,
    intake_rows: Iterable[Mapping[str, Any]],
    author_model: str,
    author_version: str,
    checker_model: str,
    checker_version: str,
    environment_generations: Optional[Mapping[str, str]] = None,
) -> AuthorityContext:
    """Build the one production authority context from exact shipped bytes.

    Parameters
    ----------
    active_intake_snapshot_id, active_intake_snapshot_sha256:
        Canonically validated active intake identity.
    intake_rows:
        Full trusted intake rows.
    author_model, author_version, checker_model, checker_version:
        Configured author/checker identities.
    environment_generations:
        Exact currently materialized environment generations keyed by name.

    Returns
    -------
    AuthorityContext
        Mandatory context shared by every reducer and projection consumer.
    """

    package_root = Path(__file__).parent
    rows = tuple(dict(row) for row in intake_rows)
    intake_by_stable_id = {str(row["stable_id"]): row for row in rows}
    if len(intake_by_stable_id) != len(rows):
        raise AuthorityDerivationError("active intake contains duplicate stable IDs")
    family_bindings: dict[str, JsonObject] = {}
    for stable_id, row in intake_by_stable_id.items():
        representative = str(row.get("family_representative_id") or stable_id)
        if row.get("variant_scope", "family") == "family" and representative != stable_id:
            family_bindings[stable_id] = {
                "binding_state": "variant",
                "representative_stable_id": representative,
                "variant_token": str(row.get("variant", "")),
                "derivation_rule_identity": stable_hash("menagerie-family-variant-derivation-v1"),
            }

    def content_identity(relative: str) -> str:
        """Hash one exact shipped authority file."""

        try:
            return hash_bytes((package_root / relative).read_bytes())
        except OSError as exc:
            raise AuthorityDerivationError(
                f"authority component is unavailable: {relative}"
            ) from exc

    from menagerie.crawler.constants import (  # noqa: PLC0415
        AUTHOR_PROMPT_NAME,
        CHECKER_PROMPT_NAME,
    )

    author_prompt = content_identity(f"prompts/{AUTHOR_PROMPT_NAME}.txt")
    checker_prompt = content_identity(f"prompts/{CHECKER_PROMPT_NAME}.txt")
    author_identity = stable_hash(
        {
            "provider": "anthropic",
            "model": author_model,
            "version": author_version,
            "prompt_sha256": author_prompt,
        }
    )
    checker_identity = stable_hash(
        {
            "provider": "openai",
            "model": checker_model,
            "version": checker_version,
            "prompt_sha256": checker_prompt,
        }
    )
    reducer_policy = stable_hash(
        {
            "reducer": content_identity("reducer.py"),
            "metadata": content_identity("metadata.py"),
            "gates": content_identity("gates.py"),
        }
    )
    runner_policy = stable_hash(
        {
            "worker": content_identity("worker.py"),
            "supervisor": content_identity("worker_supervisor.py"),
            "policy": content_identity("policy.py"),
        }
    )
    terminal_policy = stable_hash(
        {
            "authority": content_identity("authority.py"),
            "gate_schema": content_identity("schemas/gate-v3.schema.json"),
        }
    )
    publication_policy = stable_hash(
        {
            "transactions": content_identity("artifact_transactions.py"),
            "artifact_schema": content_identity("schemas/artifact-event-v1.schema.json"),
            "licenses": content_identity("licenses.py"),
        }
    )
    return AuthorityContext(
        active_intake_snapshot_id=active_intake_snapshot_id,
        active_intake_snapshot_sha256=active_intake_snapshot_sha256,
        intake_by_stable_id=intake_by_stable_id,
        family_bindings=family_bindings,
        author_prompt_identity=author_prompt,
        author_model_identity=author_identity,
        author_schema_identity=content_identity("schemas/author-result-v3.schema.json"),
        author_dispatcher_identity=content_identity("author_dispatch.py"),
        checker_prompt_identity=checker_prompt,
        checker_model_identity=checker_identity,
        checker_schema_identity=content_identity("schemas/gate-v3.schema.json"),
        environment_generations=dict(environment_generations or {}),
        reducer_policy_identity=reducer_policy,
        runner_policy_identity=runner_policy,
        terminal_policy_identity=terminal_policy,
        publication_policy_identity=publication_policy,
    )


def _require_nonempty_string(value: object, field: str) -> str:
    """Return one required non-empty string.

    Parameters
    ----------
    value:
        Candidate value.
    field:
        Field name used in the failure.

    Returns
    -------
    str
        Validated string.

    Raises
    ------
    AuthorityDerivationError
        If the value is absent or empty.
    """

    if not isinstance(value, str) or not value:
        raise AuthorityDerivationError(f"{field} must be a non-empty string")
    return value


def _require_hash(value: object, field: str) -> str:
    """Return one required prefixed SHA-256 identity.

    Parameters
    ----------
    value:
        Candidate value.
    field:
        Field name used in the failure.

    Returns
    -------
    str
        Validated prefixed digest.

    Raises
    ------
    AuthorityDerivationError
        If the value is not a canonical SHA-256 identity.
    """

    digest = _require_nonempty_string(value, field)
    if len(digest) != 71 or not digest.startswith(_HASH_PREFIX):
        raise AuthorityDerivationError(f"{field} must be a prefixed SHA-256 identity")
    try:
        int(digest.removeprefix(_HASH_PREFIX), 16)
    except ValueError as exc:
        raise AuthorityDerivationError(f"{field} must be a prefixed SHA-256 identity") from exc
    if digest != digest.lower():
        raise AuthorityDerivationError(f"{field} must use lowercase hexadecimal")
    return digest


def _closed_mapping(value: Mapping[str, Any], expected_fields: frozenset[str], field: str) -> None:
    """Require an exact closed mapping key set.

    Parameters
    ----------
    value:
        Mapping being authenticated.
    expected_fields:
        Exact allowed and required key set.
    field:
        Field name used in the failure.

    Raises
    ------
    AuthorityDerivationError
        If keys are missing or extraneous.
    """

    actual = frozenset(value)
    if actual != expected_fields:
        missing = sorted(expected_fields - actual)
        extra = sorted(actual - expected_fields)
        raise AuthorityDerivationError(
            f"{field} is not closed (missing={missing!r}, extra={extra!r})"
        )


def _canonical_completion_payload(payload: Mapping[str, Any]) -> str:
    """Serialize a completion payload in its single canonical representation.

    Parameters
    ----------
    payload:
        Closed completion payload.

    Returns
    -------
    str
        Canonical JSON text.
    """

    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def raw_award_receipt_sha256(raw_award_receipt: Mapping[str, Any]) -> str:
    """Hash the exact canonical raw award receipt.

    Parameters
    ----------
    raw_award_receipt:
        Closed v3 raw worker receipt without a self-referential digest field.

    Returns
    -------
    str
        Canonical receipt digest.
    """

    return stable_hash(dict(raw_award_receipt))


def completion_line_for_raw_award_receipt(raw_award_receipt: Mapping[str, Any]) -> str:
    """Build the canonical completion line naming one raw receipt.

    Parameters
    ----------
    raw_award_receipt:
        Closed v3 raw receipt.

    Returns
    -------
    str
        Canonical parent-visible completion line without its trailing newline.
    """

    payload = {
        "raw_award_receipt_sha256": raw_award_receipt_sha256(raw_award_receipt),
        "request_nonce": _require_nonempty_string(
            raw_award_receipt.get("request_nonce"), "raw_award_receipt.request_nonce"
        ),
        "request_sha256": _require_hash(
            raw_award_receipt.get("request_sha256"), "raw_award_receipt.request_sha256"
        ),
    }
    return _WORKER_COMPLETION_PREFIX + _canonical_completion_payload(payload)


def _parse_completion_line(completion_line: str) -> JsonObject:
    """Parse and canonicalize one v3 worker completion line.

    Parameters
    ----------
    completion_line:
        Exact parent-observed line without its trailing newline.

    Returns
    -------
    dict[str, Any]
        Closed parsed completion payload.

    Raises
    ------
    AuthorityDerivationError
        If the marker, JSON, fields, or canonical representation is invalid.
    """

    if not completion_line.startswith(_WORKER_COMPLETION_PREFIX):
        raise AuthorityDerivationError("parent completion line has the wrong protocol marker")
    encoded = completion_line.removeprefix(_WORKER_COMPLETION_PREFIX)
    try:
        parsed = json.loads(encoded)
    except json.JSONDecodeError as exc:
        raise AuthorityDerivationError("parent completion line is not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise AuthorityDerivationError("parent completion payload must be an object")
    expected = {
        "raw_award_receipt_sha256",
        "request_nonce",
        "request_sha256",
    }
    if set(parsed) != expected:
        raise AuthorityDerivationError("parent completion payload is not closed")
    if encoded != _canonical_completion_payload(parsed):
        raise AuthorityDerivationError("parent completion payload is not canonically encoded")
    _require_hash(parsed.get("raw_award_receipt_sha256"), "completion.raw_award_receipt_sha256")
    _require_nonempty_string(parsed.get("request_nonce"), "completion.request_nonce")
    _require_hash(parsed.get("request_sha256"), "completion.request_sha256")
    return parsed


def derive_parent_attestation(
    raw_award_receipt: Mapping[str, Any],
    completion_line: str,
    supervisor_observation: Mapping[str, Any],
    *,
    started_at: str,
    finished_at: str,
) -> JsonObject:
    """Derive the closed parent attestation from parent-observed facts.

    Parameters
    ----------
    raw_award_receipt:
        Exact raw receipt named by the completion line.
    completion_line:
        Exact observed completion line without its trailing newline.
    supervisor_observation:
        Parent-owned exit, signal, resource, and stream facts.
    started_at, finished_at:
        Parent-observed UTC process boundaries.

    Returns
    -------
    dict[str, Any]
        Closed v2 parent attestation with its canonical self hash.
    """

    parsed = _parse_completion_line(completion_line)
    receipt_digest = raw_award_receipt_sha256(raw_award_receipt)
    if parsed["raw_award_receipt_sha256"] != receipt_digest:
        raise AuthorityDerivationError("completion line names different raw receipt bytes")
    for field in ("request_nonce", "request_sha256"):
        if parsed[field] != raw_award_receipt.get(field):
            raise AuthorityDerivationError(f"completion line {field} disagrees with raw receipt")
    attestation: JsonObject = {
        "attestation_version": _PARENT_ATTESTATION_VERSION,
        "request_nonce": parsed["request_nonce"],
        "request_sha256": parsed["request_sha256"],
        "completion_line_sha256": hash_bytes((completion_line + "\n").encode("utf-8")),
        "named_raw_award_receipt_sha256": receipt_digest,
        "exit_code": supervisor_observation.get("exit_code"),
        "signal": supervisor_observation.get("signal"),
        "timed_out": supervisor_observation.get("timed_out") is True,
        "rss_exceeded": supervisor_observation.get("rss_exceeded") is True,
        "peak_rss_bytes": supervisor_observation.get("peak_rss_bytes"),
        "stdout_sha256": _require_hash(
            supervisor_observation.get("stdout_sha256"), "supervisor_observation.stdout_sha256"
        ),
        "stderr_sha256": _require_hash(
            supervisor_observation.get("stderr_sha256"), "supervisor_observation.stderr_sha256"
        ),
        "started_at": _require_nonempty_string(started_at, "started_at"),
        "finished_at": _require_nonempty_string(finished_at, "finished_at"),
    }
    attestation["attestation_sha256"] = stable_hash(attestation)
    return attestation


def _validate_raw_receipt(raw_award_receipt: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate closed receipt invariants and return its observation.

    Parameters
    ----------
    raw_award_receipt:
        Candidate closed raw receipt.

    Returns
    -------
    Mapping[str, Any]
        Validated raw observation.

    Raises
    ------
    AuthorityDerivationError
        If any association or success fact is invalid.
    """

    _closed_mapping(raw_award_receipt, _RAW_RECEIPT_FIELDS, "raw_award_receipt")
    if raw_award_receipt.get("receipt_version") != _RAW_AWARD_RECEIPT_VERSION:
        raise AuthorityDerivationError("raw award receipt has the wrong protocol version")
    for field in ("request_nonce", "stable_id", "work_id"):
        _require_nonempty_string(raw_award_receipt.get(field), f"raw_award_receipt.{field}")
    for field in (
        "request_sha256",
        "execution_identity",
        "recipe_revision",
        "code_manifest_identity",
        "input_identity",
    ):
        _require_hash(raw_award_receipt.get(field), f"raw_award_receipt.{field}")
    requested_mode = raw_award_receipt.get("requested_mode")
    if requested_mode not in _MODE_ORDER:
        raise AuthorityDerivationError("raw award receipt has an invalid requested mode")
    observation = raw_award_receipt.get("observation")
    if not isinstance(observation, Mapping):
        raise AuthorityDerivationError("raw award receipt observation must be an object")
    if observation.get("present") is not True:
        raise AuthorityDerivationError("raw award receipt must contain a present observation")
    if observation.get("receipt_sha256") is not None:
        raise AuthorityDerivationError(
            "raw observation receipt_sha256 must be null; the raw digest is separately named"
        )
    if observation.get("mode") != requested_mode:
        raise AuthorityDerivationError("raw observation mode disagrees with requested mode")
    if observation.get("observed_recipe_revision") != raw_award_receipt.get("recipe_revision"):
        raise AuthorityDerivationError("raw observation recipe identity is stale")
    if observation.get("observed_code_manifest_sha256") != raw_award_receipt.get(
        "code_manifest_identity"
    ):
        raise AuthorityDerivationError("raw observation code-manifest identity is stale")
    for field in (
        "constructor_started",
        "constructor_completed",
        "input_completed",
        "forward_started",
        "forward_completed",
    ):
        if observation.get(field) is not True:
            raise AuthorityDerivationError(f"raw success observation requires {field}=true")
    return observation


def _validate_parent_attestation(
    raw_award_receipt: Mapping[str, Any],
    parent_attestation: Mapping[str, Any],
    completion_line: str,
) -> str:
    """Validate a parent attestation and return the raw receipt digest.

    Parameters
    ----------
    raw_award_receipt:
        Exact raw worker receipt.
    parent_attestation:
        Candidate closed parent attestation.
    completion_line:
        Exact parent-observed completion line.

    Returns
    -------
    str
        Recomputed raw receipt digest.

    Raises
    ------
    AuthorityDerivationError
        If any parent or child association fails.
    """

    _closed_mapping(parent_attestation, _PARENT_ATTESTATION_FIELDS, "parent_attestation")
    if parent_attestation.get("attestation_version") != _PARENT_ATTESTATION_VERSION:
        raise AuthorityDerivationError("parent attestation has the wrong protocol version")
    unhashed = {
        key: value for key, value in parent_attestation.items() if key != "attestation_sha256"
    }
    if parent_attestation.get("attestation_sha256") != stable_hash(unhashed):
        raise AuthorityDerivationError("parent attestation self hash is invalid")
    parsed = _parse_completion_line(completion_line)
    raw_digest = raw_award_receipt_sha256(raw_award_receipt)
    expected = {
        "request_nonce": raw_award_receipt.get("request_nonce"),
        "request_sha256": raw_award_receipt.get("request_sha256"),
        "raw_award_receipt_sha256": raw_digest,
    }
    if parsed != expected:
        raise AuthorityDerivationError(
            "completion line does not name the exact raw request/receipt"
        )
    if parent_attestation.get("request_nonce") != expected["request_nonce"]:
        raise AuthorityDerivationError("parent attestation request nonce is mismatched")
    if parent_attestation.get("request_sha256") != expected["request_sha256"]:
        raise AuthorityDerivationError("parent attestation request digest is mismatched")
    if parent_attestation.get("named_raw_award_receipt_sha256") != raw_digest:
        raise AuthorityDerivationError("parent attestation names different raw receipt bytes")
    line_digest = hash_bytes((completion_line + "\n").encode("utf-8"))
    if parent_attestation.get("completion_line_sha256") != line_digest:
        raise AuthorityDerivationError("parent attestation completion-line hash is invalid")
    if (
        parent_attestation.get("exit_code") != 0
        or parent_attestation.get("signal") is not None
        or parent_attestation.get("timed_out") is not False
        or parent_attestation.get("rss_exceeded") is not False
    ):
        raise AuthorityDerivationError("non-clean parent observation cannot attest a success")
    return raw_digest


def _candidate_projection_error(
    candidate_attempt: Mapping[str, Any],
    raw_award_receipt: Mapping[str, Any],
    parent_attestation: Mapping[str, Any],
    completion_line: str,
    raw_digest: str,
) -> Optional[str]:
    """Return the first candidate/raw projection disagreement.

    Parameters
    ----------
    candidate_attempt:
        Reducer admission candidate.
    raw_award_receipt, parent_attestation, completion_line, raw_digest:
        Already authenticated proof graph.

    Returns
    -------
    str | None
        Mismatched path, or ``None`` when every consumed projection agrees.
    """

    observation = raw_award_receipt["observation"]
    identities = candidate_attempt.get("identities")
    invocation = candidate_attempt.get("invocation")
    supervisor = candidate_attempt.get("supervisor_observation")
    comparisons: tuple[tuple[str, object, object], ...] = (
        ("result", candidate_attempt.get("result"), "succeeded"),
        ("stage", candidate_attempt.get("stage"), "forward"),
        ("stable_id", candidate_attempt.get("stable_id"), raw_award_receipt["stable_id"]),
        ("work_id", candidate_attempt.get("work_id"), raw_award_receipt["work_id"]),
        ("mode", candidate_attempt.get("mode"), raw_award_receipt["requested_mode"]),
        ("worker_receipt", candidate_attempt.get("worker_receipt"), observation),
        ("raw_award_receipt", candidate_attempt.get("raw_award_receipt"), raw_award_receipt),
        (
            "raw_award_receipt_sha256",
            candidate_attempt.get("raw_award_receipt_sha256"),
            raw_digest,
        ),
        (
            "parent_attestation",
            candidate_attempt.get("parent_attestation"),
            parent_attestation,
        ),
        ("unattested_partial", candidate_attempt.get("unattested_partial"), None),
        (
            "identities.execution",
            identities.get("execution") if isinstance(identities, Mapping) else None,
            raw_award_receipt["execution_identity"],
        ),
        (
            "identities.recipe",
            identities.get("recipe") if isinstance(identities, Mapping) else None,
            raw_award_receipt["recipe_revision"],
        ),
        (
            "invocation.mode",
            invocation.get("mode") if isinstance(invocation, Mapping) else None,
            raw_award_receipt["requested_mode"],
        ),
        (
            "supervisor_observation.stdout_completion_line",
            supervisor.get("stdout_completion_line") if isinstance(supervisor, Mapping) else None,
            completion_line,
        ),
        (
            "supervisor_observation.exit_code",
            supervisor.get("exit_code") if isinstance(supervisor, Mapping) else None,
            parent_attestation["exit_code"],
        ),
        (
            "supervisor_observation.signal",
            supervisor.get("signal") if isinstance(supervisor, Mapping) else None,
            parent_attestation["signal"],
        ),
        (
            "supervisor_observation.peak_rss_bytes",
            supervisor.get("peak_rss_bytes") if isinstance(supervisor, Mapping) else None,
            parent_attestation["peak_rss_bytes"],
        ),
        (
            "supervisor_observation.stdout_sha256",
            supervisor.get("stdout_sha256") if isinstance(supervisor, Mapping) else None,
            parent_attestation["stdout_sha256"],
        ),
        (
            "supervisor_observation.stderr_sha256",
            supervisor.get("stderr_sha256") if isinstance(supervisor, Mapping) else None,
            parent_attestation["stderr_sha256"],
        ),
        ("started_at", candidate_attempt.get("started_at"), parent_attestation["started_at"]),
        ("finished_at", candidate_attempt.get("finished_at"), parent_attestation["finished_at"]),
    )
    for path, candidate, derived in comparisons:
        if candidate != derived:
            return path
    policy = candidate_attempt.get("policy_observation")
    if not isinstance(policy, Mapping):
        return "policy_observation"
    if any(policy.get(field) is not False for field in _POLICY_FIELDS):
        return "policy_observation.clean_flags"
    if any(policy.get(field) != [] for field in _POLICY_SEQUENCE_FIELDS):
        return "policy_observation.clean_details"
    return None


def derive_attempt_projection(
    raw_award_receipt: Mapping[str, Any],
    parent_attestation: Mapping[str, Any],
    *,
    completion_line: Optional[str] = None,
    candidate_attempt: Optional[Mapping[str, Any]] = None,
) -> AttemptAuthority:
    """Authenticate a raw success receipt and its complete persisted projection.

    Parameters
    ----------
    raw_award_receipt:
        Retained closed v3 worker receipt.
    parent_attestation:
        Separately retained v2 parent attestation.
    completion_line:
        Exact parent-observed completion line. When a candidate is supplied, it
        may be read from ``supervisor_observation.stdout_completion_line``.
    candidate_attempt:
        Optional persisted candidate whose every award-consumed projection is
        required to equal the authenticated proof.

    Returns
    -------
    AttemptAuthority
        Immutable verified attempt/raw/parent association.

    Raises
    ------
    AuthorityDerivationError
        If any raw, parent, completion, association, or projection fact fails.
    """

    _validate_raw_receipt(raw_award_receipt)
    if completion_line is None and candidate_attempt is not None:
        supervisor = candidate_attempt.get("supervisor_observation")
        candidate_line = (
            supervisor.get("stdout_completion_line") if isinstance(supervisor, Mapping) else None
        )
        completion_line = candidate_line if isinstance(candidate_line, str) else None
    if completion_line is None:
        raise AuthorityDerivationError("exact parent-observed completion line is required")
    raw_digest = _validate_parent_attestation(
        raw_award_receipt, parent_attestation, completion_line
    )
    if candidate_attempt is not None:
        mismatch = _candidate_projection_error(
            candidate_attempt,
            raw_award_receipt,
            parent_attestation,
            completion_line,
            raw_digest,
        )
        if mismatch is not None:
            raise AuthorityDerivationError(
                f"attempt projection contradicts authenticated receipt at {mismatch}"
            )
        attempt_id = _require_nonempty_string(
            candidate_attempt.get("attempt_id"), "candidate_attempt.attempt_id"
        )
    else:
        attempt_id = stable_hash(
            {
                "request_sha256": raw_award_receipt["request_sha256"],
                "raw_award_receipt_sha256": raw_digest,
                "parent_attestation_sha256": parent_attestation["attestation_sha256"],
            }
        )
    return AttemptAuthority(
        attempt_id=attempt_id,
        stable_id=str(raw_award_receipt["stable_id"]),
        work_id=str(raw_award_receipt["work_id"]),
        execution_identity=str(raw_award_receipt["execution_identity"]),
        request_identity=str(raw_award_receipt["request_sha256"]),
        raw_award_receipt_sha256=raw_digest,
        parent_attestation_sha256=str(parent_attestation["attestation_sha256"]),
    )


def _authenticated_observation(
    attempt: Mapping[str, Any],
) -> tuple[AttemptAuthority, Mapping[str, Any]]:
    """Return authenticated authority and raw observation for one attempt.

    Parameters
    ----------
    attempt:
        Candidate admitted v3 attempt.

    Returns
    -------
    tuple[AttemptAuthority, Mapping[str, Any]]
        Verified association and exact raw observation.
    """

    raw = attempt.get("raw_award_receipt")
    parent = attempt.get("parent_attestation")
    if not isinstance(raw, Mapping) or not isinstance(parent, Mapping):
        raise AuthorityDerivationError("mode comparison requires retained v3 raw proof")
    authority = derive_attempt_projection(raw, parent, candidate_attempt=attempt)
    observation = raw.get("observation")
    if not isinstance(observation, Mapping):
        raise AuthorityDerivationError("authenticated raw observation is missing")
    return authority, observation


def derive_mode_summary(
    train_attempt: Optional[Mapping[str, Any]],
    eval_attempt: Optional[Mapping[str, Any]],
) -> ModeSummary:
    """Derive train/eval divergence only from authenticated raw observations.

    Parameters
    ----------
    train_attempt, eval_attempt:
        Canonical v3 attempts selected for the two meaningful modes.

    Returns
    -------
    ModeSummary
        Structured comparison, including honest unverifiable/not-applicable states.

    Raises
    ------
    AuthorityDerivationError
        If a supplied attempt is unauthenticated or associated with the wrong mode.
    """

    if train_attempt is None or eval_attempt is None:
        supplied = train_attempt if train_attempt is not None else eval_attempt
        supplied_id: Optional[str] = None
        if supplied is not None:
            authority, observation = _authenticated_observation(supplied)
            expected_mode = "train" if train_attempt is not None else "eval"
            if observation.get("mode") != expected_mode:
                raise AuthorityDerivationError(
                    "single-mode attempt is associated with the wrong mode"
                )
            supplied_id = authority.attempt_id
        train_attempt_id = supplied_id if train_attempt is not None else None
        eval_attempt_id = supplied_id if eval_attempt is not None else None
        payload: JsonObject = {
            "comparison_state": "not-applicable",
            "classification": "not-applicable",
            "train_attempt_id": train_attempt_id,
            "eval_attempt_id": eval_attempt_id,
            "compared_fields": [],
        }
        return ModeSummary(
            comparison_state="not-applicable",
            classification="not-applicable",
            train_attempt_id=train_attempt_id,
            eval_attempt_id=eval_attempt_id,
            compared_fields=(),
            evidence_sha256=stable_hash(payload),
        )

    train_authority, train_observation = _authenticated_observation(train_attempt)
    eval_authority, eval_observation = _authenticated_observation(eval_attempt)
    if train_observation.get("mode") != "train" or eval_observation.get("mode") != "eval":
        raise AuthorityDerivationError("mode comparison attempts are cross-associated")
    if train_authority.stable_id != eval_authority.stable_id:
        raise AuthorityDerivationError("mode comparison attempts belong to different models")
    if train_authority.work_id != eval_authority.work_id:
        raise AuthorityDerivationError(
            "mode comparison attempts belong to different work generations"
        )
    train_signature = train_observation.get("output_signature")
    eval_signature = eval_observation.get("output_signature")
    compared_fields: tuple[str, ...] = ("output_signature",)
    if train_signature != eval_signature:
        comparison_state = "verified"
        classification = "structural"
    else:
        train_digest = train_observation.get("output_value_sha256")
        eval_digest = eval_observation.get("output_value_sha256")
        if not isinstance(train_digest, str) or not isinstance(eval_digest, str):
            comparison_state = "unverifiable"
            classification = "unverifiable"
        else:
            _require_hash(train_digest, "train output value digest")
            _require_hash(eval_digest, "eval output value digest")
            compared_fields = ("output_signature", "output_value_sha256")
            comparison_state = "verified"
            classification = "none" if train_digest == eval_digest else "statistical"
    payload = {
        "comparison_state": comparison_state,
        "classification": classification,
        "train_attempt_id": train_authority.attempt_id,
        "eval_attempt_id": eval_authority.attempt_id,
        "compared_fields": list(compared_fields),
    }
    return ModeSummary(
        comparison_state=comparison_state,
        classification=classification,
        train_attempt_id=train_authority.attempt_id,
        eval_attempt_id=eval_authority.attempt_id,
        compared_fields=compared_fields,
        evidence_sha256=stable_hash(payload),
    )


def mode_summary_projection(summary: ModeSummary) -> JsonObject:
    """Render one reducer-derived mode summary into the model-v3 mode fields.

    Parameters
    ----------
    summary:
        Authenticated structured mode comparison.

    Returns
    -------
    dict[str, Any]
        Canonical classification and a stable JSON evidence string retaining
        comparison state, exact attempts, and compared fields.
    """

    evidence = {
        "comparison_state": summary.comparison_state,
        "train_attempt_id": summary.train_attempt_id,
        "eval_attempt_id": summary.eval_attempt_id,
        "compared_fields": list(summary.compared_fields),
        "evidence_sha256": summary.evidence_sha256,
    }
    return {
        "train_eval_divergence": summary.classification,
        "divergence_evidence": json.dumps(
            evidence,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
    }


def dependency_vector_projection(vector: DependencyVector) -> JsonObject:
    """Render a frozen dependency vector into its canonical schema mapping.

    Parameters
    ----------
    vector:
        Reducer-derived closed dependency vector.

    Returns
    -------
    dict[str, Any]
        JSON-compatible vector with typed states encoded by their stable values.
    """

    payload = asdict(vector)
    payload["accepted_attempt_ids"] = list(vector.accepted_attempt_ids)
    payload["artifact_claim_ids"] = [str(value) for value in vector.artifact_claim_ids]
    for key, value in tuple(payload.items()):
        if isinstance(value, DependencyState):
            payload[key] = value.value
    return payload


def _attempt_order(attempt: Mapping[str, Any]) -> tuple[int, int, str]:
    """Return the deterministic decisive-attempt ordering key.

    Parameters
    ----------
    attempt:
        Canonical attempt.

    Returns
    -------
    tuple[int, int, str]
        Attempt number, ledger sequence, and stable attempt ID.
    """

    attempt_no = attempt.get("attempt_no")
    ledger_seq = attempt.get("ledger_seq")
    return (
        attempt_no if isinstance(attempt_no, int) else -1,
        ledger_seq if isinstance(ledger_seq, int) else -1,
        str(attempt.get("attempt_id", "")),
    )


def derive_per_mode_attempt_ids(
    attempts: Sequence[Mapping[str, Any]],
    *,
    stable_id: str,
    work_id: str,
    meaningful_modes: Iterable[str] = ("train", "eval"),
) -> tuple[tuple[str, str], ...]:
    """Select the complete deterministic per-mode terminal attempt map.

    Parameters
    ----------
    attempts:
        Canonical attempt history.
    stable_id, work_id:
        Exact terminal model and work generation.
    meaningful_modes:
        Closed ordered meaningful-mode set.

    Returns
    -------
    tuple[tuple[str, str], ...]
        Mode-to-decisive-attempt-ID pairs in canonical mode order.

    Raises
    ------
    AuthorityDerivationError
        If a mode attempt is malformed or reused.
    """

    modes = tuple(dict.fromkeys(str(mode) for mode in meaningful_modes))
    if any(mode not in _MODE_ORDER for mode in modes):
        raise AuthorityDerivationError("meaningful modes contain an unknown value")
    selected: list[tuple[str, str]] = []
    used_ids: set[str] = set()
    for mode in sorted(modes, key=_MODE_ORDER.__getitem__):
        candidates = [
            attempt
            for attempt in attempts
            if attempt.get("stable_id") == stable_id
            and attempt.get("work_id") == work_id
            and attempt.get("mode") == mode
        ]
        if not candidates:
            continue
        decisive = max(candidates, key=_attempt_order)
        attempt_id = _require_nonempty_string(decisive.get("attempt_id"), "attempt.attempt_id")
        if attempt_id in used_ids:
            raise AuthorityDerivationError("one attempt cannot represent two terminal modes")
        if decisive.get("stage") != "forward":
            raise AuthorityDerivationError("terminal mode map contains a non-forward attempt")
        if decisive.get("result") not in {"succeeded", "failed", "observed"}:
            raise AuthorityDerivationError("terminal mode map contains an invalid result")
        used_ids.add(attempt_id)
        selected.append((mode, attempt_id))
    return tuple(selected)


def derive_per_mode_run(
    attempts: Sequence[Mapping[str, Any]],
    *,
    stable_id: str,
    work_id: str,
    meaningful_modes: Iterable[str] = ("train", "eval"),
) -> JsonObject:
    """Derive the exact schema-shaped terminal per-mode outcome map.

    Parameters
    ----------
    attempts:
        Canonical attempt history.
    stable_id, work_id:
        Exact terminal model and work generation.
    meaningful_modes:
        Closed ordered meaningful-mode set.

    Returns
    -------
    dict[str, Any]
        Complete deterministic ``modes.per_mode_run`` projection.
    """

    selected = derive_per_mode_attempt_ids(
        attempts,
        stable_id=stable_id,
        work_id=work_id,
        meaningful_modes=meaningful_modes,
    )
    index = {str(attempt.get("attempt_id")): attempt for attempt in attempts}
    return {
        mode: {
            "attempt_id": attempt_id,
            "status": str(index[attempt_id]["result"]),
        }
        for mode, attempt_id in selected
    }


def derive_terminal_observation(
    attempts: Sequence[Mapping[str, Any]], *, stable_id: str, work_id: str
) -> JsonObject:
    """Derive schema-complete terminal observations from exact attempt history.

    Parameters
    ----------
    attempts:
        Canonical attempt history.
    stable_id, work_id:
        Exact terminal model/work generation.

    Returns
    -------
    dict[str, Any]
        Reducer-owned terminal observation; no worker fact is fabricated.
    """

    relevant = sorted(
        (
            attempt
            for attempt in attempts
            if attempt.get("stable_id") == stable_id and attempt.get("work_id") == work_id
        ),
        key=_attempt_order,
    )
    receipt: Mapping[str, Any] = {}
    supervisor: Mapping[str, Any] = {}
    for attempt in reversed(relevant):
        candidate = attempt.get("worker_receipt")
        if isinstance(candidate, Mapping) and candidate.get("present") is True:
            receipt = candidate
            parent = attempt.get("supervisor_observation")
            supervisor = parent if isinstance(parent, Mapping) else {}
            break
    output = receipt.get("output_signature")
    normalized_output = (
        dict(output) if isinstance(output, Mapping) else {"tree": None, "leaves": []}
    )
    if not {"tree", "leaves"}.issubset(normalized_output):
        normalized_output = {"tree": None, "leaves": []}
    snippet = "driver-owned terminal disposition; no run awarded"
    return {
        "parameter_count_total": int(receipt.get("parameter_count_total") or 0),
        "parameter_count_trainable": int(receipt.get("parameter_count_trainable") or 0),
        "native_framework": receipt.get("native_framework"),
        "delegated_method": receipt.get("delegated_method"),
        "output_signature": normalized_output,
        "input_kind": str(receipt.get("input_kind") or "random-fallback"),
        "input_asset": receipt.get("input_asset"),
        "input_note": str(receipt.get("input_note") or "No complete worker input receipt."),
        "constructor_seconds": float(receipt.get("constructor_seconds") or 0.0),
        "forward_seconds": float(receipt.get("forward_seconds") or 0.0),
        "peak_rss_bytes": int(supervisor.get("peak_rss_bytes") or 0),
        "measurement_attempt_ids": [str(attempt["attempt_id"]) for attempt in relevant],
        "snippet": snippet,
        "snippet_sha256": stable_hash(snippet),
    }


def _gate_item_fingerprint(item: Mapping[str, Any]) -> str:
    """Derive the checker root-cause fingerprint for one exact gate item.

    Parameters
    ----------
    item:
        Canonical checker item.

    Returns
    -------
    str
        Reducer-owned root-cause fingerprint.
    """

    return stable_hash(
        {
            "verdict": item.get("verdict"),
            "integrity": item.get("integrity"),
            "field_checks": item.get("field_checks"),
            "rung_check": item.get("rung_check"),
            "fidelity": item.get("fidelity"),
            "terminal_disposition": item.get("terminal_disposition"),
            "unsupported_claims": item.get("unsupported_claims"),
            "required_repairs": item.get("required_repairs"),
        }
    )


def _gate_order(gate: Mapping[str, Any]) -> tuple[int, int, str]:
    """Return the deterministic gate ordering key.

    Parameters
    ----------
    gate:
        Canonical gate envelope.

    Returns
    -------
    tuple[int, int, str]
        Gate round, ledger sequence, and gate ID.
    """

    gate_round = gate.get("gate_round")
    ledger_seq = gate.get("ledger_seq")
    return (
        gate_round if isinstance(gate_round, int) else -1,
        ledger_seq if isinstance(ledger_seq, int) else -1,
        str(gate.get("gate_id", "")),
    )


def _matching_gate_items(
    gates: Sequence[Mapping[str, Any]],
    *,
    stable_id: str,
    work_id: str,
    gate_kind: str,
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    """Return exact one-item gate matches in deterministic history order.

    Parameters
    ----------
    gates:
        Canonical gate history.
    stable_id, work_id, gate_kind:
        Exact item association.

    Returns
    -------
    list[tuple[Mapping[str, Any], Mapping[str, Any]]]
        Matching envelopes and items.
    """

    matches: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for gate in sorted(gates, key=_gate_order):
        if gate.get("gate_kind") != gate_kind:
            continue
        items = gate.get("items")
        if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
            continue
        exact = [
            item
            for item in items
            if isinstance(item, Mapping)
            and item.get("stable_id") == stable_id
            and item.get("work_id") == work_id
        ]
        if len(exact) == 1:
            matches.append((gate, exact[0]))
    return matches


def _terminal_gate(
    gates: Sequence[Mapping[str, Any]],
    *,
    stable_id: str,
    work_id: str,
    predicate: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    """Resolve the latest exact accepted terminal-disposition gate.

    Parameters
    ----------
    gates:
        Canonical gate history.
    stable_id, work_id, predicate:
        Exact terminal recommendation association.

    Returns
    -------
    tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]
        Gate, item, and accepted terminal-disposition block.

    Raises
    ------
    AuthorityDerivationError
        If no exact accepted predicate exists.
    """

    accepted: list[tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]] = []
    for gate, item in _matching_gate_items(
        gates,
        stable_id=stable_id,
        work_id=work_id,
        gate_kind="terminal_disposition",
    ):
        disposition = item.get("terminal_disposition")
        if (
            isinstance(disposition, Mapping)
            and disposition.get("predicate") == predicate
            and disposition.get("verdict") == "accepted"
            and item.get("verdict") == "accurate"
            and item.get("integrity", {}).get("verdict") == "accurate"
        ):
            accepted.append((gate, item, disposition))
    if not accepted:
        raise AuthorityDerivationError(
            f"terminal {predicate} lacks an exact accepted terminal-disposition gate"
        )
    return accepted[-1]


def _validate_terminal_references(
    disposition: Mapping[str, Any],
    *,
    predicate: str,
    source_manifest: Sequence[Mapping[str, Any]],
    evidence_excerpts: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Resolve terminal source/evidence IDs and re-check the typed predicate.

    Parameters
    ----------
    disposition:
        Accepted terminal gate disposition.
    predicate:
        Closed terminal predicate.
    source_manifest, evidence_excerpts:
        Exact canonical source and literal-evidence facts.

    Returns
    -------
    tuple[tuple[str, ...], tuple[str, ...]]
        Exact resolved source and evidence ID sequences.

    Raises
    ------
    AuthorityDerivationError
        If IDs are missing, duplicated, cross-bound, or fail the predicate.
    """

    source_ids = tuple(str(value) for value in disposition.get("source_ids", ()))
    evidence_ids = tuple(str(value) for value in disposition.get("evidence_ids", ()))
    if not source_ids or len(source_ids) != len(set(source_ids)):
        raise AuthorityDerivationError("terminal gate must bind a non-empty unique source set")
    if len(evidence_ids) != len(set(evidence_ids)):
        raise AuthorityDerivationError("terminal gate evidence IDs must be unique")
    source_index = {
        str(source.get("source_id")): source
        for source in source_manifest
        if isinstance(source, Mapping) and source.get("source_id") is not None
    }
    evidence_index = {
        str(excerpt.get("evidence_id")): excerpt
        for excerpt in evidence_excerpts
        if isinstance(excerpt, Mapping) and excerpt.get("evidence_id") is not None
    }
    if set(source_ids) != {source_id for source_id in source_ids if source_id in source_index}:
        raise AuthorityDerivationError("terminal gate names a missing source fact")
    if set(evidence_ids) != {
        evidence_id for evidence_id in evidence_ids if evidence_id in evidence_index
    }:
        raise AuthorityDerivationError("terminal gate names a missing literal-evidence fact")
    if any(
        str(evidence_index[evidence_id].get("source_id")) not in source_ids
        for evidence_id in evidence_ids
    ):
        raise AuthorityDerivationError("terminal evidence is not bound to the exact source set")
    if predicate in {"needs-cuda", "needs-x86"}:
        accepted_supports = {
            predicate,
            f"platform.{predicate}",
            f"deferred:{predicate}",
            f"defer_evidence.{predicate}",
        }
        supported = any(
            bool(
                accepted_supports
                & {str(value) for value in evidence_index[evidence_id].get("supports", ())}
            )
            for evidence_id in evidence_ids
        )
        if not supported:
            raise AuthorityDerivationError(
                f"literal evidence does not support typed platform claim {predicate}"
            )
    return source_ids, evidence_ids


def _validate_terminal_gate_identities(
    disposition: Mapping[str, Any],
    *,
    source_manifest_identity: Optional[str],
    evidence_identity: Optional[str],
    license_identity: Optional[str],
) -> None:
    """Require the terminal gate to bind all exact frozen input identities.

    Parameters
    ----------
    disposition:
        Accepted terminal-disposition block.
    source_manifest_identity, evidence_identity, license_identity:
        Reducer-resolved identities of the exact canonical input facts.

    Raises
    ------
    AuthorityDerivationError
        If a resolved identity is absent or differs from the accepted gate.
    """

    expected = {
        "source_manifest_identity": source_manifest_identity,
        "evidence_identity": evidence_identity,
        "license_identity": license_identity,
    }
    for field, value in expected.items():
        if value is None:
            raise AuthorityDerivationError(f"terminal proof requires resolved {field}")
        _require_hash(value, field)
        if disposition.get(field) != value:
            raise AuthorityDerivationError(f"terminal gate has stale {field}")


def _validate_skip_predicate(
    predicate: str,
    source_resolution: Mapping[str, Any],
    evidence_excerpts: Sequence[Mapping[str, Any]],
    evidence_ids: tuple[str, ...],
) -> None:
    """Re-check the exact R5 semantic predicate behind one accepted skip.

    Parameters
    ----------
    predicate:
        Closed skip suffix.
    source_resolution:
        Canonical accepted R5 source/search facts.
    evidence_excerpts:
        Canonical literal evidence pack.
    evidence_ids:
        Exact gate-selected evidence IDs.

    Raises
    ------
    AuthorityDerivationError
        If the accepted facts do not prove the typed skip predicate.
    """

    if source_resolution.get("rung") != "R5_SKIP":
        raise AuthorityDerivationError("skip proof does not resolve to R5 source facts")
    search_report = source_resolution.get("search_report")
    if not isinstance(search_report, Mapping) or not search_report.get("conclusion"):
        raise AuthorityDerivationError("skip proof lacks its exact bounded search report")
    evidence_index = {
        str(excerpt.get("evidence_id")): excerpt
        for excerpt in evidence_excerpts
        if isinstance(excerpt, Mapping) and excerpt.get("evidence_id") is not None
    }
    selected = [evidence_index[evidence_id] for evidence_id in evidence_ids]
    if predicate == "insufficient-description":
        if not source_resolution.get("sufficiency_gap"):
            raise AuthorityDerivationError(
                "insufficient-description lacks its material sufficiency gap"
            )
        if not any(
            excerpt.get("disposition") == "insufficient-for-faithful-reimpl" for excerpt in selected
        ):
            raise AuthorityDerivationError(
                "insufficient-description lacks its exact vague literal excerpt"
            )
    elif predicate == "no-description":
        if source_resolution.get("sufficiency_gap") not in {None, ""}:
            raise AuthorityDerivationError("no-description cannot carry a sufficiency gap")
    elif predicate == "not-a-real-NN":
        supported = any(
            {
                "not-a-real-NN",
                "skipped:not-a-real-NN",
                "source_resolution.not-a-real-NN",
            }
            & {str(value) for value in excerpt.get("supports", ())}
            for excerpt in selected
        )
        if not supported:
            raise AuthorityDerivationError("not-a-real-NN lacks literal scope evidence")


def _derive_gate_failure(
    gates: Sequence[Mapping[str, Any]], *, stable_id: str, work_id: str, stage: str
) -> tuple[str, str, str]:
    """Derive a capped accuracy/fidelity failure from exact rejected history.

    Parameters
    ----------
    gates:
        Canonical gate history.
    stable_id, work_id, stage:
        Exact model/work and ``accuracy-gate`` or ``fidelity`` stage.

    Returns
    -------
    tuple[str, str, str]
        Gate ID, reason code, and root-cause fingerprint.

    Raises
    ------
    AuthorityDerivationError
        If the rejection lineage has not reached the bounded terminal rule.
    """

    gate_kind = "metadata_batch" if stage == "accuracy-gate" else "fidelity"
    active = _matching_gate_items(
        gates,
        stable_id=stable_id,
        work_id=work_id,
        gate_kind=gate_kind,
    )
    campaign_ids = {
        str(item.get("campaign_root_work_id"))
        for _gate, item in active
        if isinstance(item.get("campaign_root_work_id"), str)
    }
    if len(campaign_ids) > 1:
        raise AuthorityDerivationError(f"failed:{stage} spans multiple repair campaigns")
    campaign_id = next(iter(campaign_ids), None)
    rejected: list[tuple[Mapping[str, Any], Mapping[str, Any], str]] = []
    history = (
        [
            (gate, item)
            for gate in sorted(gates, key=_gate_order)
            if gate.get("gate_kind") == gate_kind
            for item in gate.get("items", ())
            if isinstance(item, Mapping)
            and item.get("stable_id") == stable_id
            and item.get("campaign_root_work_id") == campaign_id
        ]
        if campaign_id is not None
        else active
    )
    for gate, item in history:
        if gate_kind == "metadata_batch":
            accepted = bool(
                item.get("verdict") == "accurate"
                and item.get("integrity", {}).get("verdict") == "accurate"
                and item.get("rung_check", {}).get("verdict") == "accurate"
            )
        else:
            accepted = bool(
                item.get("fidelity", {}).get("verdict") in {"match", "minor-drift"}
                and item.get("rung_check", {}).get("verdict") == "accurate"
            )
        if not accepted:
            rejected.append((gate, item, _gate_item_fingerprint(item)))
    if not rejected:
        raise AuthorityDerivationError(f"failed:{stage} lacks exact rejected gate evidence")
    fingerprints = [fingerprint for _gate, _item, fingerprint in rejected]
    if len(rejected) < 3 and fingerprints[-1] not in fingerprints[:-1]:
        raise AuthorityDerivationError(f"failed:{stage} has not reached its bounded cap")
    gate, item, fingerprint = rejected[-1]
    if stage == "accuracy-gate":
        reason = (
            "cannot-verify-cap-exhausted"
            if item.get("verdict") == "cannot-verify"
            else "identity-mismatch"
            if item.get("integrity", {}).get("verdict") != "accurate"
            else "inaccurate-cap-exhausted"
        )
    else:
        verdict = item.get("fidelity", {}).get("verdict")
        reason = {
            "major-drift": "major-drift-cap-exhausted",
            "slop": "slop-cap-exhausted",
            "cannot-verify": "cannot-verify-cap-exhausted",
        }.get(str(verdict), "identity-mismatch")
    return str(gate["gate_id"]), reason, fingerprint


def derive_terminal_proof(
    stable_id: str,
    work_id: str,
    status_code: str,
    *,
    attempts: Sequence[Mapping[str, Any]],
    gates: Sequence[Mapping[str, Any]] = (),
    source_manifest: Sequence[Mapping[str, Any]] = (),
    evidence_excerpts: Sequence[Mapping[str, Any]] = (),
    source_resolution: Optional[Mapping[str, Any]] = None,
    source_manifest_identity: Optional[str] = None,
    evidence_identity: Optional[str] = None,
    license_identity: Optional[str] = None,
    meaningful_modes: Iterable[str] = ("train", "eval"),
    proof_rule_identity: str,
) -> TerminalProof:
    """Resolve one terminal status to its exact semantic proof graph.

    Parameters
    ----------
    stable_id, work_id, status_code:
        Exact terminal association and closed public status code.
    attempts, gates:
        Canonical append-only facts available to the reducer.
    source_manifest, evidence_excerpts:
        Exact source and literal-evidence facts for terminal predicates.
    source_resolution:
        Exact accepted R5 facts required for epistemic skips.
    source_manifest_identity, evidence_identity, license_identity:
        Reducer-resolved frozen identities required by terminal gates.
    meaningful_modes:
        Ordered meaningful-mode set used for the complete per-mode map.
    proof_rule_identity:
        Versioned terminal-rule closure from the mandatory authority context.

    Returns
    -------
    TerminalProof
        Immutable reducer-derived terminal authority.

    Raises
    ------
    AuthorityDerivationError
        If the status is unknown or lacks its specific status-proving predicate.
    """

    stable_id = _require_nonempty_string(stable_id, "stable_id")
    work_id = _require_nonempty_string(work_id, "work_id")
    proof_rule_identity = _require_nonempty_string(proof_rule_identity, "proof_rule_identity")
    relevant = tuple(
        attempt
        for attempt in attempts
        if attempt.get("stable_id") == stable_id and attempt.get("work_id") == work_id
    )
    per_mode = derive_per_mode_attempt_ids(
        attempts,
        stable_id=stable_id,
        work_id=work_id,
        meaningful_modes=meaningful_modes,
    )
    gate_id: DependencyValue = DependencyState.NOT_APPLICABLE
    source_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    failure_stage: DependencyValue = DependencyState.NOT_APPLICABLE
    reason_code: DependencyValue = DependencyState.NOT_APPLICABLE
    root_cause: DependencyValue = DependencyState.NOT_APPLICABLE
    platform_claim: DependencyValue = DependencyState.NOT_APPLICABLE
    decisive_ids: tuple[str, ...] = ()
    gate_proof_identity: DependencyValue = DependencyState.NOT_APPLICABLE
    resolved_reference_identity: DependencyValue = DependencyState.NOT_APPLICABLE

    if status_code == "runs":
        expected_modes = tuple(dict.fromkeys(str(mode) for mode in meaningful_modes))
        if {mode for mode, _attempt_id in per_mode} != set(expected_modes):
            raise AuthorityDerivationError("runs proof does not cover every meaningful mode")
        attempt_index = {str(attempt.get("attempt_id")): attempt for attempt in relevant}
        for mode, attempt_id in per_mode:
            attempt = attempt_index.get(attempt_id)
            if (
                attempt is None
                or attempt.get("mode") != mode
                or attempt.get("result") != "succeeded"
            ):
                raise AuthorityDerivationError("runs proof contains a non-successful mode attempt")
            _authenticated_observation(attempt)
        decisive_ids = tuple(attempt_id for _mode, attempt_id in per_mode)
    elif status_code.startswith("failed:"):
        stage = status_code.removeprefix("failed:")
        failure_stage = stage
        if stage in {"accuracy-gate", "fidelity"}:
            gate_id, reason_code, root_cause = _derive_gate_failure(
                gates, stable_id=stable_id, work_id=work_id, stage=stage
            )
        else:
            candidates = []
            for attempt in relevant:
                error = attempt.get("error")
                if (
                    attempt.get("result") == "failed"
                    and attempt.get("stage") == stage
                    and isinstance(error, Mapping)
                    and error.get("stage") == stage
                    and isinstance(error.get("reason_code"), str)
                    and isinstance(error.get("root_cause_fingerprint"), str)
                ):
                    candidates.append(attempt)
            if not candidates:
                raise AuthorityDerivationError(
                    f"{status_code} lacks an exact same-stage failed attempt"
                )
            decisive = max(candidates, key=_attempt_order)
            error = decisive["error"]
            assert isinstance(error, Mapping)
            reason_code = str(error["reason_code"])
            if reason_code not in FAILURE_REASON_CODES.get(stage, frozenset()):
                raise AuthorityDerivationError("failed attempt reason is not closed for its stage")
            root_cause = str(error["root_cause_fingerprint"])
            decisive_ids = (str(decisive["attempt_id"]),)
            if stage == "source" and reason_code == "missing-mandatory-link":
                if decisive.get("stage") != "source":
                    raise AuthorityDerivationError(
                        "missing-mandatory-link must bind an exact source-stage attempt"
                    )
    elif status_code.startswith("deferred:"):
        predicate = status_code.removeprefix("deferred:")
        if predicate not in {"needs-cuda", "needs-x86"}:
            raise AuthorityDerivationError("unknown platform deferral status")
        gate, item, disposition = _terminal_gate(
            gates,
            stable_id=stable_id,
            work_id=work_id,
            predicate=predicate,
        )
        gate_id = str(gate["gate_id"])
        gate_proof_identity = stable_hash({"gate_id": gate_id, "item": item})
        platform_claim = predicate
        _validate_terminal_gate_identities(
            disposition,
            source_manifest_identity=source_manifest_identity,
            evidence_identity=evidence_identity,
            license_identity=license_identity,
        )
        source_ids, evidence_ids = _validate_terminal_references(
            disposition,
            predicate=predicate,
            source_manifest=source_manifest,
            evidence_excerpts=evidence_excerpts,
        )
        resolved_reference_identity = stable_hash(
            {
                "sources": [
                    source
                    for source in source_manifest
                    if str(source.get("source_id")) in source_ids
                ],
                "evidence": [
                    excerpt
                    for excerpt in evidence_excerpts
                    if str(excerpt.get("evidence_id")) in evidence_ids
                ],
            }
        )
        probe_ids: list[str] = []
        for attempt in relevant:
            defer = attempt.get("defer_evidence")
            if not isinstance(defer, Mapping) or defer.get("target_status") != status_code:
                continue
            if set(str(value) for value in defer.get("source_ids", ())) != set(source_ids):
                raise AuthorityDerivationError("deferral attempt source set is not gate-exact")
            named_probes = tuple(str(value) for value in defer.get("probe_attempt_ids", ()))
            for probe_id in named_probes:
                probe = next(
                    (
                        candidate
                        for candidate in relevant
                        if candidate.get("attempt_id") == probe_id
                    ),
                    None,
                )
                capability = probe.get("capability_observation") if probe is not None else None
                if (
                    probe is None
                    or probe.get("result") not in {"observed", "succeeded"}
                    or not isinstance(capability, Mapping)
                    or capability.get("claim") != predicate
                    or capability.get("supported") is not True
                ):
                    raise AuthorityDerivationError(
                        "deferral probe lacks a structured positive same-work capability observation"
                    )
                probe_ids.append(probe_id)
        decisive_ids = tuple(dict.fromkeys(probe_ids))
    elif status_code.startswith("skipped:"):
        predicate = status_code.removeprefix("skipped:")
        if predicate not in {"insufficient-description", "no-description", "not-a-real-NN"}:
            raise AuthorityDerivationError("unknown epistemic skip status")
        gate, item, disposition = _terminal_gate(
            gates,
            stable_id=stable_id,
            work_id=work_id,
            predicate=predicate,
        )
        if (
            item.get("rung_check", {}).get("selected_rung") != "R5_SKIP"
            or item.get("rung_check", {}).get("verdict") != "accurate"
        ):
            raise AuthorityDerivationError("skip gate does not prove an accurate R5 decision")
        gate_id = str(gate["gate_id"])
        gate_proof_identity = stable_hash({"gate_id": gate_id, "item": item})
        _validate_terminal_gate_identities(
            disposition,
            source_manifest_identity=source_manifest_identity,
            evidence_identity=evidence_identity,
            license_identity=license_identity,
        )
        source_ids, evidence_ids = _validate_terminal_references(
            disposition,
            predicate=predicate,
            source_manifest=source_manifest,
            evidence_excerpts=evidence_excerpts,
        )
        if source_resolution is None:
            raise AuthorityDerivationError(
                "skip proof requires exact accepted source-resolution facts"
            )
        _validate_skip_predicate(
            predicate,
            source_resolution,
            evidence_excerpts,
            evidence_ids,
        )
        resolved_reference_identity = stable_hash(
            {
                "source_resolution": source_resolution,
                "sources": [
                    source
                    for source in source_manifest
                    if str(source.get("source_id")) in source_ids
                ],
                "evidence": [
                    excerpt
                    for excerpt in evidence_excerpts
                    if str(excerpt.get("evidence_id")) in evidence_ids
                ],
            }
        )
    else:
        raise AuthorityDerivationError("status code has no closed terminal proof rule")

    terminal_observation = derive_terminal_observation(
        attempts, stable_id=stable_id, work_id=work_id
    )
    proof_payload = {
        "proof_rule_identity": proof_rule_identity,
        "stable_id": stable_id,
        "work_id": work_id,
        "status_code": status_code,
        "decisive_attempt_ids": list(decisive_ids),
        "gate_id": gate_id,
        "source_ids": list(source_ids),
        "evidence_ids": list(evidence_ids),
        "failure_stage": failure_stage,
        "reason_code": reason_code,
        "root_cause_fingerprint": root_cause,
        "platform_claim": platform_claim,
        "per_mode_attempt_ids": [list(value) for value in per_mode],
        "terminal_observation_sha256": stable_hash(terminal_observation),
        "gate_proof_identity": gate_proof_identity,
        "resolved_reference_identity": resolved_reference_identity,
    }
    return TerminalProof(
        proof_id=stable_hash(proof_payload),
        proof_rule_identity=proof_rule_identity,
        stable_id=stable_id,
        work_id=work_id,
        status_code=status_code,
        decisive_attempt_ids=decisive_ids,
        gate_id=gate_id,
        source_ids=source_ids,
        evidence_ids=evidence_ids,
        failure_stage=failure_stage,
        reason_code=reason_code,
        root_cause_fingerprint=root_cause,
        platform_claim=platform_claim,
        per_mode_attempt_ids=per_mode,
        terminal_observation_sha256=str(proof_payload["terminal_observation_sha256"]),
    )


def derive_family_authority(
    context: AuthorityContext,
    stable_id: str,
    *,
    representative_record: Optional[Mapping[str, Any]] = None,
) -> FamilyAuthority:
    """Derive ordinary/variant family authority from trusted intake binding.

    Parameters
    ----------
    context:
        Mandatory active authority context.
    stable_id:
        Current model identity.
    representative_record:
        Exact dependency-current representative revision for a bound variant.

    Returns
    -------
    FamilyAuthority
        Trusted ordinary or exact representative binding.

    Raises
    ------
    AuthorityDerivationError
        If a trusted variant binding is incomplete or unresolved.
    """

    if stable_id not in context.intake_by_stable_id:
        raise AuthorityDerivationError("family authority stable ID is outside active intake")
    binding = context.family_bindings.get(stable_id)
    if binding is None or (
        isinstance(binding, Mapping) and binding.get("binding_state") == "ordinary"
    ):
        state = DependencyState.NOT_APPLICABLE
        return FamilyAuthority(stable_id, state, state, state, state, state, state, state)
    if not isinstance(binding, Mapping):
        raise AuthorityDerivationError("trusted family binding must be an object")
    representative_id = binding.get("representative_stable_id")
    if representative_id is None:
        representative_id = binding.get("family_representative_id")
    representative_id = _require_nonempty_string(
        representative_id, "family_binding.representative_stable_id"
    )
    if representative_id == stable_id:
        state = DependencyState.NOT_APPLICABLE
        return FamilyAuthority(stable_id, state, state, state, state, state, state, state)
    variant_token = binding.get("variant_token", binding.get("variant"))
    variant_token = _require_nonempty_string(variant_token, "family_binding.variant_token")
    if representative_record is None or representative_record.get("stable_id") != representative_id:
        raise AuthorityDerivationError("variant lacks its exact current representative record")
    revision = _require_hash(
        representative_record.get("record_revision"), "representative.record_revision"
    )
    gate_id = _require_nonempty_string(
        representative_record.get("accuracy_gate", {}).get("gate_id"),
        "representative.accuracy_gate.gate_id",
    )
    vector = representative_record.get("dependency_vector")
    proposal_id = vector.get("proposal_identity") if isinstance(vector, Mapping) else None
    proposal_id = _require_nonempty_string(
        proposal_id, "representative.dependency_vector.proposal_identity"
    )
    derivation_identity = str(
        binding.get("derivation_rule_identity")
        or stable_hash("menagerie-family-variant-derivation-v1")
    )
    return FamilyAuthority(
        stable_id=stable_id,
        representative_stable_id=representative_id,
        representative_revision=revision,
        representative_gate_id=gate_id,
        representative_proposal_id=proposal_id,
        variant_token=variant_token,
        template_source_revision=revision,
        derivation_rule_identity=derivation_identity,
    )


def family_authority_projection(authority: FamilyAuthority) -> JsonObject:
    """Render a frozen ``FamilyAuthority`` into the model-v3 schema shape.

    Parameters
    ----------
    authority:
        Reducer-derived trusted family authority.

    Returns
    -------
    dict[str, Any]
        Exact schema-owned family-authority block.
    """

    variant = authority.representative_stable_id != DependencyState.NOT_APPLICABLE
    return {
        "binding_state": "variant" if variant else "ordinary",
        "representative_stable_id": authority.representative_stable_id,
        "representative_revision": authority.representative_revision,
        "representative_gate_id": authority.representative_gate_id,
        "representative_proposal_id": authority.representative_proposal_id,
        "variant_token": authority.variant_token,
        "template_source_revision": authority.template_source_revision,
        "derivation_rule_identity": authority.derivation_rule_identity,
    }


def derive_dependency_vector(
    context: AuthorityContext,
    *,
    stable_id: str,
    terminal_proof: TerminalProof,
    source_manifest_identity: DependencyValue = DependencyState.NOT_APPLICABLE,
    proposal_identity: DependencyValue = DependencyState.NOT_APPLICABLE,
    author_result_identity: DependencyValue = DependencyState.NOT_APPLICABLE,
    checker_gate_identity: DependencyValue = DependencyState.NOT_APPLICABLE,
    recipe_revision: DependencyValue = DependencyState.NOT_APPLICABLE,
    environment_id: Optional[str] = None,
    accepted_attempt_ids: Iterable[str] = (),
    artifact_transaction_id: DependencyValue = DependencyState.NOT_APPLICABLE,
    artifact_claim_ids: Iterable[ArtifactClaimId] = (),
    family_authority: Optional[FamilyAuthority] = None,
) -> DependencyVector:
    """Derive the closed stage-sensitive vector from resolved authority facts.

    Parameters
    ----------
    context:
        Mandatory active trust roots and policy closures.
    stable_id, terminal_proof:
        Exact canonical model and its reducer-derived status proof.
    source_manifest_identity, proposal_identity, author_result_identity,
    checker_gate_identity, recipe_revision:
        Exact resolved canonical references or typed states.
    environment_id:
        Current environment key; the generation is taken only from ``context``.
    accepted_attempt_ids:
        Exact reducer-admitted attempt identities participating in the status.
    artifact_transaction_id, artifact_claim_ids:
        Exact artifact-ledger authority references.
    family_authority:
        Trusted family derivation, or ordinary authority derived from context.

    Returns
    -------
    DependencyVector
        Closed reducer-owned dependency vector.

    Raises
    ------
    AuthorityDerivationError
        If a resolved reference is outside the active authority context.
    """

    intake_item = context.intake_by_stable_id.get(stable_id)
    if not isinstance(intake_item, Mapping):
        raise AuthorityDerivationError("dependency vector stable ID is outside active intake")
    if terminal_proof.stable_id != stable_id:
        raise AuthorityDerivationError("terminal proof belongs to another stable ID")
    family = family_authority or derive_family_authority(context, stable_id)
    status_stage = (
        terminal_proof.status_code.removeprefix("failed:")
        if terminal_proof.status_code.startswith("failed:")
        else "terminal"
    )
    runner_applicable = (
        terminal_proof.status_code == "runs" or status_stage in _STATUS_RUNNER_STAGES
    )
    if environment_id is None:
        environment_generation: DependencyValue = DependencyState.NOT_APPLICABLE
    else:
        generation = context.environment_generations.get(environment_id)
        if generation is None:
            raise AuthorityDerivationError("dependency vector names an unknown environment")
        environment_generation = generation
    checker_prompt: DependencyValue = (
        DependencyState.NOT_APPLICABLE
        if checker_gate_identity == DependencyState.NOT_APPLICABLE
        else context.checker_prompt_identity
    )
    representative_revision = family.representative_revision
    return DependencyVector(
        intake_snapshot_id=context.active_intake_snapshot_id,
        intake_snapshot_sha256=context.active_intake_snapshot_sha256,
        intake_item_sha256=stable_hash(dict(intake_item)),
        author_result_schema_identity=context.author_schema_identity,
        author_dispatcher_identity=context.author_dispatcher_identity,
        author_prompt_identity=context.author_prompt_identity,
        checker_prompt_identity=checker_prompt,
        terminal_rule_identity=context.terminal_policy_identity,
        status_proof_identity=terminal_proof.proof_id,
        source_manifest_identity=source_manifest_identity,
        proposal_identity=proposal_identity,
        author_result_identity=author_result_identity,
        checker_gate_identity=checker_gate_identity,
        recipe_revision=recipe_revision,
        runner_identity=(
            context.runner_policy_identity if runner_applicable else DependencyState.NOT_APPLICABLE
        ),
        award_closure_identity=(
            context.reducer_policy_identity if runner_applicable else DependencyState.NOT_APPLICABLE
        ),
        environment_generation=environment_generation,
        accepted_attempt_ids=tuple(dict.fromkeys(str(value) for value in accepted_attempt_ids)),
        artifact_transaction_id=artifact_transaction_id,
        artifact_claim_ids=tuple(dict.fromkeys(artifact_claim_ids)),
        representative_revision=representative_revision,
        publication_policy_identity=context.publication_policy_identity,
    )


def _vector_mapping(vector: object) -> Mapping[str, Any]:
    """Normalize a stored vector dataclass or mapping for currency comparison.

    Parameters
    ----------
    vector:
        Candidate dependency vector.

    Returns
    -------
    Mapping[str, Any]
        Field mapping, or an empty mapping for an invalid candidate.
    """

    if isinstance(vector, DependencyVector):
        return asdict(vector)
    return vector if isinstance(vector, Mapping) else {}


def validate_currency(
    context: AuthorityContext,
    record: Mapping[str, Any],
    *,
    terminal_proof: Optional[TerminalProof] = None,
    family_authority: Optional[FamilyAuthority] = None,
) -> Optional[str]:
    """Return the exact first stale reason for one canonical revision.

    Parameters
    ----------
    context:
        Mandatory active trust roots.
    record:
        Latest canonical model revision.
    terminal_proof:
        Replayed proof when available.
    family_authority:
        Replayed family authority when available.

    Returns
    -------
    str | None
        Stable stale reason, or ``None`` when all directly replayable axes are current.
    """

    if record.get("schema_version") != "menagerie.crawler.model.v3":
        return "legacy-untrusted: model revision lacks v3 authority"
    stable_id = str(record.get("stable_id", ""))
    intake_item = context.intake_by_stable_id.get(stable_id)
    if not isinstance(intake_item, Mapping):
        return "intake: stable ID is absent from active snapshot"
    vector = _vector_mapping(record.get("dependency_vector"))
    if not vector:
        return "dependency-vector: missing closed v3 vector"
    expected_axes: tuple[tuple[str, object], ...] = (
        ("intake_snapshot_id", context.active_intake_snapshot_id),
        ("intake_snapshot_sha256", context.active_intake_snapshot_sha256),
        ("intake_item_sha256", stable_hash(dict(intake_item))),
        ("author_result_schema_identity", context.author_schema_identity),
        ("author_dispatcher_identity", context.author_dispatcher_identity),
        ("author_prompt_identity", context.author_prompt_identity),
        ("terminal_rule_identity", context.terminal_policy_identity),
        ("publication_policy_identity", context.publication_policy_identity),
    )
    for axis, expected in expected_axes:
        if vector.get(axis) != expected:
            return f"dependency-vector: stale {axis}"
    checker = vector.get("checker_prompt_identity")
    if checker not in {context.checker_prompt_identity, DependencyState.NOT_APPLICABLE.value}:
        return "dependency-vector: stale checker_prompt_identity"
    if terminal_proof is not None:
        if terminal_proof.stable_id != stable_id:
            return "status-proof: proof belongs to another stable ID"
        if vector.get("status_proof_identity") != terminal_proof.proof_id:
            return "dependency-vector: stale status_proof_identity"
    status = record.get("status")
    status_code = status.get("code") if isinstance(status, Mapping) else ""
    stage = str(status_code).removeprefix("failed:")
    runner_applicable = status_code == "runs" or stage in _STATUS_RUNNER_STAGES
    expected_runner: DependencyValue = (
        context.runner_policy_identity if runner_applicable else DependencyState.NOT_APPLICABLE
    )
    expected_award: DependencyValue = (
        context.reducer_policy_identity if runner_applicable else DependencyState.NOT_APPLICABLE
    )
    if vector.get("runner_identity") != expected_runner:
        return "dependency-vector: stale runner_identity"
    if vector.get("award_closure_identity") != expected_award:
        return "dependency-vector: stale award_closure_identity"
    environment_generation = vector.get("environment_generation")
    if (
        environment_generation != DependencyState.NOT_APPLICABLE.value
        and environment_generation not in set(context.environment_generations.values())
    ):
        return "dependency-vector: stale environment_generation"
    binding = context.family_bindings.get(stable_id)
    trusted_variant = bool(
        isinstance(binding, Mapping)
        and binding.get("binding_state") != "ordinary"
        and binding.get("representative_stable_id", binding.get("family_representative_id"))
        not in {None, stable_id}
    )
    if trusted_variant and family_authority is None:
        return "family-authority: trusted variant binding was not replayed"
    replayed_family = family_authority or derive_family_authority(context, stable_id)
    if record.get("family_authority") != family_authority_projection(replayed_family):
        return "family-authority: canonical projection contradicts trusted intake binding"
    if family_authority is not None:
        if family_authority.stable_id != stable_id:
            return "family-authority: binding belongs to another stable ID"
        if vector.get("representative_revision") != family_authority.representative_revision:
            return "dependency-vector: stale representative_revision"
    return None


def derive_runner_identity(
    semantic_components: Mapping[str, str],
    *,
    platform_name: str,
    selected_asset_identity: DependencyValue = DependencyState.NOT_APPLICABLE,
) -> str:
    """Hash a caller-collected exact runtime semantic closure without I/O.

    Parameters
    ----------
    semantic_components:
        Component name to exact semantic-AST/content identity.
    platform_name:
        Exact execution-host platform.
    selected_asset_identity:
        Exact selected standard-input object or typed state.

    Returns
    -------
    str
        Versioned runner closure identity.
    """

    return stable_hash(
        {
            "closure_version": "menagerie-runner-closure-v3",
            "platform": platform_name,
            "semantic_components": dict(sorted(semantic_components.items())),
            "selected_asset_identity": selected_asset_identity,
        }
    )


def derive_award_closure_identity(
    semantic_components: Mapping[str, str], schema_identities: Mapping[str, str]
) -> str:
    """Hash caller-collected reducer/parent award semantics without I/O.

    Parameters
    ----------
    semantic_components:
        Component name to exact semantic-AST identity.
    schema_identities:
        Current award-consumed schema name to exact content identity.

    Returns
    -------
    str
        Versioned award closure identity.
    """

    return stable_hash(
        {
            "closure_version": "menagerie-award-closure-v3",
            "semantic_components": dict(sorted(semantic_components.items())),
            "schema_identities": dict(sorted(schema_identities.items())),
        }
    )


def derive_execution_identity(
    *,
    stable_id: str,
    recipe_revision: str,
    environment_generation: str,
    runner_identity: str,
    target: str,
    machine_class: str,
    input_seed: int,
    framework: str,
    recipe_type: str,
    award_closure_identity: str,
    runtime_dependencies_identity: str,
    device: str,
) -> str:
    """Derive execution identity from resolved facts without filesystem I/O.

    Parameters
    ----------
    stable_id, recipe_revision, environment_generation:
        Exact model, accepted recipe, and current environment identities.
    runner_identity, target, machine_class:
        Exact runtime closure and execution-host facts.
    input_seed:
        Accepted deterministic input seed.
    framework, recipe_type:
        Exact runtime adapter selection.
    award_closure_identity, runtime_dependencies_identity:
        Parent/reducer decision closure and accepted runtime-fact closure.
    device:
        Exact accepted device policy.

    Returns
    -------
    str
        Canonical controlled execution identity.
    """

    return compute_execution_identity(
        stable_id=stable_id,
        recipe_revision=recipe_revision,
        env_generation=environment_generation,
        runner_version=runner_identity,
        target=target,
        machine_class=machine_class,
        seed_policy={
            "input_seed": input_seed,
            "cold_seed_reuse": "single-accepted-input-manifest",
            "version": 3,
        },
        framework_adapter={
            "framework": framework,
            "recipe_type": recipe_type,
            "award_closure_sha256": award_closure_identity,
            "runtime_dependencies_sha256": runtime_dependencies_identity,
        },
        device=device,
    )
