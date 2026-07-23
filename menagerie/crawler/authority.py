"""Frozen round-14 authority contracts and pure proof derivations."""

from __future__ import annotations

import ast
from contextlib import contextmanager
import json
import os
import stat
import unicodedata
from dataclasses import asdict, dataclass, field as dataclass_field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, NewType, Optional, Sequence

from menagerie.crawler.constants import (
    ATTEMPT_SCHEMA_VERSION_V3,
    ENVIRONMENT_AUTHORITY_VERSION_V1,
    ENVIRONMENT_CONTENT_MANIFEST_VERSION_V1,
    ENVIRONMENT_GENERATION_VERSION_V2,
    EXECUTION_READ_MANIFEST_VERSION_V2,
    EXECUTION_READ_MANIFEST_VERSION_V3,
    FAILURE_REASON_CODES,
    GATE_SCHEMA_VERSION_V3,
)
from menagerie.crawler.identity import (
    compute_execution_identity,
    hash_bytes,
    payload_hash,
    stable_hash,
)
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
_LEGACY_PROOF_RULE = "legacy rows lack v3 proof material required for current authority"
_CODE_MEMBER_KINDS = frozenset(
    {
        "native-extension",
        "native-source",
        "python-bytecode",
        "python-source",
    }
)
_RUNTIME_MEMBER_KINDS = frozenset(
    {
        "import-metadata",
        "interpreter",
        "native-extension",
        "native-library",
        "package-data",
        "python-bytecode",
        "python-source",
    }
)
_MEMBER_SUFFIXES = {
    "native-extension": frozenset({".dylib", ".pyd", ".so"}),
    "native-source": frozenset({".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp"}),
    "python-bytecode": frozenset({".pyc"}),
    "python-source": frozenset({".py", ".pyi", ".pyx"}),
}
_ENVIRONMENT_TREE_WALK_REGISTRY: Mapping[str, int] = {
    "EnvironmentAuthorityCache._walk_and_validate": 1,
    "_seal_environment_content": 2,
}


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
    reason: Optional[str]
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

    def __iter__(self) -> Iterator[Path | str]:
        """Yield the legacy code-member tuple for enforcement adapters.

        Yields
        ------
        pathlib.Path | str
            Path, digest, then kind. Provenance remains available only on the
            typed member and is never discarded from manifest identity.
        """

        yield self.path
        yield self.sha256
        yield self.kind


@dataclass(frozen=True)
class RuntimeLookupDirectory:
    """Lookup-only directory scaffold that grants no child-file read authority."""

    path: Path
    provenance: str


@dataclass(frozen=True)
class EnvironmentExternalTarget:
    """One exact regular file reached by a sealed-prefix symlink escape."""

    path: Path
    sha256: str
    kind: str = "regular-file"


@dataclass(frozen=True)
class EnvironmentContentEntry:
    """One canonical directory, regular file, or symlink in a prefix seal."""

    relative_path: str
    entry_type: str
    sha256: Optional[str]
    executable: Optional[bool]
    link_text: Optional[str]
    resolved_target_relative_path: Optional[str]


@dataclass(frozen=True)
class EnvironmentContentManifestV1:
    """Complete stable content seal for one materialized environment prefix."""

    manifest_version: str
    path_normalization: str
    entries: tuple[EnvironmentContentEntry, ...]
    selected_interpreter_relative_path: str
    selected_interpreter_target_relative_path: str
    selected_interpreter_digest: str
    startup_pth_relative_paths: tuple[str, ...]
    external_targets: tuple[EnvironmentExternalTarget, ...]
    content_manifest_sha256: str
    cheap_tree_fingerprint: str


@dataclass(frozen=True)
class EnvironmentAuthorityV1:
    """Digest-bound read-only capability for one canonical environment prefix."""

    authority_version: str
    authority_id: str
    prefix: Path
    base_environment_generation: str
    environment_generation: str
    content_manifest_sha256: str
    selected_interpreter: Path
    selected_interpreter_relative_path: str
    selected_interpreter_target_relative_path: str
    selected_interpreter_digest: str
    startup_pth_paths: tuple[Path, ...]
    external_targets: tuple[EnvironmentExternalTarget, ...]
    content_manifest: EnvironmentContentManifestV1
    _cache: EnvironmentAuthorityCache = dataclass_field(compare=False, repr=False)


_VERIFICATION_TOKEN_SECRET = object()


class EnvironmentVerificationToken:
    """Opaque cache-created proof of one complete current prefix observation."""

    __slots__ = (
        "_active",
        "_authority_id",
        "_cache",
        "_epoch",
        "_observed_fingerprint",
        "_purpose",
        "_sequence",
        "_spawn_marked",
    )

    def __init__(
        self,
        *,
        cache: EnvironmentAuthorityCache,
        authority_id: str,
        epoch: int,
        sequence: int,
        purpose: str,
        observed_fingerprint: str,
        secret: object,
    ) -> None:
        """Initialize a cache-owned token inaccessible to ordinary callers.

        Parameters
        ----------
        cache, authority_id, epoch, sequence, purpose, observed_fingerprint:
            Exact cache, authority, lifecycle epoch, issuance sequence, closed purpose,
            and complete observed cheap fingerprint bound by this proof.
        secret:
            Module-private constructor capability held only by the cache.
        """

        if secret is not _VERIFICATION_TOKEN_SECRET:
            raise TypeError("environment verification tokens are cache-created only")
        self._cache = cache
        self._authority_id = authority_id
        self._epoch = epoch
        self._sequence = sequence
        self._purpose = purpose
        self._observed_fingerprint = observed_fingerprint
        self._active = True
        self._spawn_marked = False

    def __reduce__(self) -> str | tuple[Any, ...]:
        """Reject serialization of process-local verification authority.

        Raises
        ------
        TypeError
            Always; verification tokens cannot cross a process boundary.
        """

        raise TypeError("environment verification tokens are not serializable")


def _environment_entry_payload(entry: EnvironmentContentEntry) -> JsonObject:
    """Return one content entry's path-neutral canonical identity payload.

    Parameters
    ----------
    entry:
        Stable prefix-tree entry.

    Returns
    -------
    dict[str, Any]
        Closed JSON-compatible entry payload.
    """

    return {
        "relative_path": entry.relative_path,
        "entry_type": entry.entry_type,
        "sha256": entry.sha256,
        "executable": entry.executable,
        "link_text": entry.link_text,
        "resolved_target_relative_path": entry.resolved_target_relative_path,
    }


def _hash_regular_file_stably(
    path: Path,
    before: os.stat_result,
) -> tuple[str, os.stat_result]:
    """Hash a regular file and reject concurrent metadata changes.

    Parameters
    ----------
    path:
        Lexical file path opened without following a different tree entry.
    before:
        ``lstat`` result captured immediately before the read.

    Returns
    -------
    tuple[str, os.stat_result]
        Exact prefixed SHA-256 digest and the stable post-read metadata baseline.

    Raises
    ------
    AuthorityDerivationError
        If the file changes while being hashed.
    """

    import hashlib

    semantic_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_size",
        "st_mtime_ns",
    )
    for attempt in range(2):
        digest = hashlib.sha256()
        try:
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            after = path.lstat()
        except OSError as exc:
            raise AuthorityDerivationError(f"environment member cannot be sealed: {path}") from exc
        if any(getattr(before, name) != getattr(after, name) for name in semantic_fields):
            raise AuthorityDerivationError(f"environment member changed while sealing: {path}")
        if before.st_ctime_ns == after.st_ctime_ns:
            return f"sha256:{digest.hexdigest()}", after
        if attempt == 1:
            raise AuthorityDerivationError(f"environment member changed while sealing: {path}")
        before = after
    raise AssertionError("stable environment hash loop exhausted")


def _canonical_relative_path(prefix: Path, path: Path) -> str:
    """Return a canonical NFC POSIX relative path or fail closed.

    Parameters
    ----------
    prefix, path:
        Canonical prefix and one lexical descendant.

    Returns
    -------
    str
        NFC-normalized POSIX relative path.
    """

    relative = path.relative_to(prefix).as_posix()
    normalized = unicodedata.normalize("NFC", relative)
    if relative != normalized or not relative or relative.startswith("/"):
        raise AuthorityDerivationError(f"environment member path is noncanonical: {relative!r}")
    return relative


def _scan_environment_tree(
    prefix: Path,
    *,
    hash_files: bool,
) -> tuple[
    tuple[EnvironmentContentEntry, ...],
    tuple[EnvironmentExternalTarget, ...],
    str,
]:
    """Enumerate one prefix without following tree symlinks.

    Parameters
    ----------
    prefix:
        Canonical materialized environment root.
    hash_files:
        Whether to hash regular-file and external-target bytes.

    Returns
    -------
    tuple
        Canonical entries, exact external targets, and cheap complete-tree fingerprint.
    """

    entries: list[EnvironmentContentEntry] = []
    external_by_path: dict[Path, EnvironmentExternalTarget] = {}
    fingerprint_rows: list[JsonObject] = []
    pending = [prefix]
    while pending:
        directory = pending.pop()
        try:
            children = sorted(directory.iterdir(), key=lambda value: value.name.encode("utf-8"))
        except (OSError, UnicodeError) as exc:
            raise AuthorityDerivationError(
                f"environment directory cannot be sealed: {directory}"
            ) from exc
        for path in children:
            relative = _canonical_relative_path(prefix, path)
            try:
                status = path.lstat()
            except OSError as exc:
                raise AuthorityDerivationError(
                    f"environment member cannot be inspected: {path}"
                ) from exc
            common_fingerprint: JsonObject = {
                "relative_path": relative,
                "mode": status.st_mode,
                "size": status.st_size,
                "mtime_ns": status.st_mtime_ns,
                "ctime_ns": status.st_ctime_ns,
                "st_dev": status.st_dev,
                "st_ino": status.st_ino,
            }
            if stat.S_ISDIR(status.st_mode):
                entry_type = "directory"
                entries.append(
                    EnvironmentContentEntry(relative, entry_type, None, None, None, None)
                )
                pending.append(path)
            elif stat.S_ISREG(status.st_mode):
                entry_type = "regular-file"
                if hash_files:
                    digest, status = _hash_regular_file_stably(path, status)
                    common_fingerprint.update(
                        {
                            "mode": status.st_mode,
                            "size": status.st_size,
                            "mtime_ns": status.st_mtime_ns,
                            "ctime_ns": status.st_ctime_ns,
                            "st_dev": status.st_dev,
                            "st_ino": status.st_ino,
                        }
                    )
                else:
                    digest = None
                entries.append(
                    EnvironmentContentEntry(
                        relative,
                        entry_type,
                        digest,
                        bool(status.st_mode & 0o111),
                        None,
                        None,
                    )
                )
            elif stat.S_ISLNK(status.st_mode):
                entry_type = "symlink"
                try:
                    link_text = os.readlink(path)
                    resolved = path.resolve(strict=True)
                except OSError as exc:
                    raise AuthorityDerivationError(
                        f"environment symlink target is unavailable: {path}"
                    ) from exc
                target_relative: Optional[str] = None
                if resolved.is_relative_to(prefix):
                    target_relative = _canonical_relative_path(prefix, resolved)
                else:
                    try:
                        target_status = resolved.stat()
                    except OSError as exc:
                        raise AuthorityDerivationError(
                            f"environment symlink escape is unavailable: {path}"
                        ) from exc
                    if not stat.S_ISREG(target_status.st_mode):
                        raise AuthorityDerivationError(
                            f"environment symlink escapes to a non-file target: {path}"
                        )
                    if hash_files:
                        digest, target_status = _hash_regular_file_stably(
                            resolved,
                            target_status,
                        )
                    else:
                        digest = ""
                    external_by_path[resolved] = EnvironmentExternalTarget(
                        path=resolved,
                        sha256=digest,
                    )
                    common_fingerprint["external_target"] = str(resolved)
                    common_fingerprint["external_entry_type"] = "regular-file"
                    common_fingerprint["external_size"] = target_status.st_size
                    common_fingerprint["external_mode"] = target_status.st_mode
                    common_fingerprint["external_mtime_ns"] = target_status.st_mtime_ns
                    common_fingerprint["external_ctime_ns"] = target_status.st_ctime_ns
                    common_fingerprint["external_st_dev"] = target_status.st_dev
                    common_fingerprint["external_st_ino"] = target_status.st_ino
                common_fingerprint["link_text"] = link_text
                entries.append(
                    EnvironmentContentEntry(
                        relative,
                        entry_type,
                        None,
                        None,
                        link_text,
                        target_relative,
                    )
                )
            else:
                raise AuthorityDerivationError(
                    f"environment contains a forbidden special entry: {path}"
                )
            common_fingerprint["entry_type"] = entry_type
            fingerprint_rows.append(common_fingerprint)
    entries.sort(key=lambda entry: entry.relative_path.encode("utf-8"))
    external = tuple(
        external_by_path[path] for path in sorted(external_by_path, key=lambda value: str(value))
    )
    fingerprint_rows.sort(key=lambda row: str(row["relative_path"]).encode("utf-8"))
    return tuple(entries), external, stable_hash(fingerprint_rows)


def _seal_environment_content(
    prefix: Path,
    selected_interpreter: Path,
) -> EnvironmentContentManifestV1:
    """Build one stable complete prefix seal with two tree enumerations.

    Parameters
    ----------
    prefix, selected_interpreter:
        Canonical environment root and its lexical interpreter member.

    Returns
    -------
    EnvironmentContentManifestV1
        Complete content identity and cheap validation token.
    """

    try:
        canonical_prefix = prefix.resolve(strict=True)
    except OSError as exc:
        raise AuthorityDerivationError(f"environment prefix is unavailable: {prefix}") from exc
    lexical_interpreter = selected_interpreter.absolute()
    if not lexical_interpreter.is_relative_to(canonical_prefix):
        raise AuthorityDerivationError("selected interpreter is outside the canonical prefix")
    try:
        resolved_interpreter = lexical_interpreter.resolve(strict=True)
        interpreter_status = resolved_interpreter.stat()
    except OSError as exc:
        raise AuthorityDerivationError("selected interpreter is unavailable") from exc
    if not resolved_interpreter.is_relative_to(canonical_prefix):
        raise AuthorityDerivationError("selected interpreter resolves outside the canonical prefix")
    if not stat.S_ISREG(interpreter_status.st_mode) or not interpreter_status.st_mode & 0o111:
        raise AuthorityDerivationError("selected interpreter is not an executable regular file")

    entries, external, fingerprint = _scan_environment_tree(
        canonical_prefix,
        hash_files=True,
    )
    second_entries, second_external, second_fingerprint = _scan_environment_tree(
        canonical_prefix,
        hash_files=False,
    )
    stable_shape = tuple(
        (
            entry.relative_path,
            entry.entry_type,
            entry.executable,
            entry.link_text,
            entry.resolved_target_relative_path,
        )
        for entry in entries
    )
    second_shape = tuple(
        (
            entry.relative_path,
            entry.entry_type,
            entry.executable,
            entry.link_text,
            entry.resolved_target_relative_path,
        )
        for entry in second_entries
    )
    if stable_shape != second_shape or fingerprint != second_fingerprint:
        raise AuthorityDerivationError("environment tree changed during content sealing")
    if tuple(target.path for target in external) != tuple(
        target.path for target in second_external
    ):
        raise AuthorityDerivationError("environment symlink escapes changed during sealing")

    relative_interpreter = _canonical_relative_path(canonical_prefix, lexical_interpreter)
    target_relative = _canonical_relative_path(canonical_prefix, resolved_interpreter)
    entries_by_path = {entry.relative_path: entry for entry in entries}
    interpreter_entry = entries_by_path.get(target_relative)
    if (
        interpreter_entry is None
        or interpreter_entry.entry_type != "regular-file"
        or interpreter_entry.sha256 is None
        or not interpreter_entry.executable
    ):
        raise AuthorityDerivationError("selected interpreter is absent from the content seal")
    startup_pth = tuple(
        entry.relative_path
        for entry in entries
        if entry.entry_type == "regular-file"
        and entry.relative_path.endswith(".pth")
        and "/site-packages/" in f"/{entry.relative_path}"
    )
    payload: JsonObject = {
        "manifest_version": ENVIRONMENT_CONTENT_MANIFEST_VERSION_V1,
        "path_normalization": "NFC POSIX relative paths; UTF-8 byte order; lexical-case identity",
        "entries": [_environment_entry_payload(entry) for entry in entries],
        "selected_interpreter_relative_path": relative_interpreter,
        "selected_interpreter_target_relative_path": target_relative,
        "selected_interpreter_digest": interpreter_entry.sha256,
        "startup_pth_relative_paths": list(startup_pth),
        "external_targets": [
            {"path": str(target.path), "sha256": target.sha256, "kind": target.kind}
            for target in external
        ],
    }
    return EnvironmentContentManifestV1(
        manifest_version=ENVIRONMENT_CONTENT_MANIFEST_VERSION_V1,
        path_normalization=str(payload["path_normalization"]),
        entries=entries,
        selected_interpreter_relative_path=relative_interpreter,
        selected_interpreter_target_relative_path=target_relative,
        selected_interpreter_digest=interpreter_entry.sha256,
        startup_pth_relative_paths=startup_pth,
        external_targets=external,
        content_manifest_sha256=stable_hash(payload),
        cheap_tree_fingerprint=fingerprint,
    )


class EnvironmentAuthorityCache:
    """Parent-owned seal cache for one active immutable environment prefix."""

    def __init__(self) -> None:
        """Initialize empty deterministic counters and no active authority."""

        self.full_seals = 0
        self.cheap_validations = 0
        self.cheap_tree_walks = 0
        self.lstat_tree_walks = 0
        self.currentness_passes = 0
        self.spawn_validations = 0
        self.real_spawns = 0
        self.invalidations = 0
        self.rehashes = 0
        self.rejected_rebinds = 0
        self._epoch = 0
        self._verification_sequence = 0
        self._active_currentness_token: Optional[EnvironmentVerificationToken] = None
        self._manifest: Optional[EnvironmentContentManifestV1] = None
        self._authority: Optional[EnvironmentAuthorityV1] = None

    @property
    def manifest(self) -> Optional[EnvironmentContentManifestV1]:
        """Return the current complete cached manifest, if any."""

        return self._manifest

    @property
    def authority(self) -> Optional[EnvironmentAuthorityV1]:
        """Return the active authority so quarantine can retain its stale identity."""

        return self._authority

    def _seal(self, prefix: Path, interpreter: Path) -> EnvironmentContentManifestV1:
        """Seal one prefix and increment the deterministic full-seal counter."""

        self.full_seals += 1
        self.lstat_tree_walks += 2
        return _seal_environment_content(prefix, interpreter)

    def bind(
        self,
        *,
        prefix: Path,
        selected_interpreter: Path,
        base_environment_generation: str,
        validate_active: bool = True,
    ) -> EnvironmentAuthorityV1:
        """Bind a stable complete seal into generation v2 and one local authority.

        Parameters
        ----------
        prefix, selected_interpreter:
            Materialized environment and its selected lexical interpreter.
        base_environment_generation:
            Existing exact lock/export/package/probe generation.
        validate_active:
            Whether an already-bound matching authority receives an immediate standalone
            validation. The driver defers this one check to its cache-owned pass token.

        Returns
        -------
        EnvironmentAuthorityV1
            Complete digest-bound prefix capability.
        """

        base_generation = _require_hash(
            base_environment_generation,
            "base_environment_generation",
        )
        canonical_prefix = prefix.resolve(strict=True)
        active = self._authority
        if active is not None:
            if (
                active.prefix != canonical_prefix
                or active.selected_interpreter != selected_interpreter.absolute()
                or active.base_environment_generation != base_generation
            ):
                self.rejected_rebinds += 1
                raise AuthorityDerivationError(
                    "active environment authority cache cannot be rebound to different inputs"
                )
            if validate_active:
                self.verify(active)
            return active
        manifest = self._seal(prefix, selected_interpreter)
        final_generation = stable_hash(
            {
                "version": ENVIRONMENT_GENERATION_VERSION_V2,
                "base_environment_generation": base_generation,
                "environment_content_sha256": manifest.content_manifest_sha256,
            }
        )
        selected = canonical_prefix / manifest.selected_interpreter_relative_path
        authority_payload = {
            "version": ENVIRONMENT_AUTHORITY_VERSION_V1,
            "prefix": str(canonical_prefix),
            "selected_interpreter_relative_path": (manifest.selected_interpreter_relative_path),
            "selected_interpreter_digest": manifest.selected_interpreter_digest,
            "environment_generation": final_generation,
            "environment_content_sha256": manifest.content_manifest_sha256,
            "external_targets": [
                {"path": str(target.path), "sha256": target.sha256, "kind": target.kind}
                for target in manifest.external_targets
            ],
        }
        authority = EnvironmentAuthorityV1(
            authority_version=ENVIRONMENT_AUTHORITY_VERSION_V1,
            authority_id=stable_hash(authority_payload),
            prefix=canonical_prefix,
            base_environment_generation=base_generation,
            environment_generation=final_generation,
            content_manifest_sha256=manifest.content_manifest_sha256,
            selected_interpreter=selected,
            selected_interpreter_relative_path=manifest.selected_interpreter_relative_path,
            selected_interpreter_target_relative_path=(
                manifest.selected_interpreter_target_relative_path
            ),
            selected_interpreter_digest=manifest.selected_interpreter_digest,
            startup_pth_paths=tuple(
                canonical_prefix / relative for relative in manifest.startup_pth_relative_paths
            ),
            external_targets=manifest.external_targets,
            content_manifest=manifest,
            _cache=self,
        )
        self._manifest = manifest
        self._authority = authority
        return authority

    def verify(
        self,
        authority: EnvironmentAuthorityV1,
        *,
        verification_token: Optional[EnvironmentVerificationToken] = None,
    ) -> None:
        """Cheaply validate or fully rehash one bound active authority.

        Parameters
        ----------
        authority:
            Previously bound prefix capability.
        verification_token:
            Cache-created pass or spawn proof that reuses its complete observation.

        Raises
        ------
        AuthorityDerivationError
            If the complete current seal differs from the bound identity.
        """

        if verification_token is not None:
            self._assert_verification_token(authority, verification_token)
            return
        self._walk_and_validate(authority)

    def _walk_and_validate(self, authority: EnvironmentAuthorityV1) -> None:
        """Perform exactly one complete cheap walk and validate its fingerprint.

        Parameters
        ----------
        authority:
            Active authority whose entire prefix and external targets are observed.
        """

        self.assert_active(authority)
        manifest = self._manifest
        assert manifest is not None
        self.cheap_validations += 1
        self.cheap_tree_walks += 1
        self.lstat_tree_walks += 1
        _entries, _external, fingerprint = _scan_environment_tree(
            authority.prefix,
            hash_files=False,
        )
        if fingerprint == manifest.cheap_tree_fingerprint:
            return
        self._epoch += 1
        if self._active_currentness_token is not None:
            self._active_currentness_token._active = False
            self._active_currentness_token = None
        self.rehashes += 1
        current = self._seal(
            authority.prefix, authority.prefix / authority.selected_interpreter_relative_path
        )
        if (
            current.content_manifest_sha256 != authority.content_manifest_sha256
            or current.selected_interpreter_digest != authority.selected_interpreter_digest
        ):
            self.invalidations += 1
            self._manifest = None
            self._authority = None
            raise AuthorityDerivationError("environment content seal changed; authority is stale")
        # Hardlink-clone creation can change source-inode ctime without changing bytes.
        # Keep the bound semantic identity and re-baseline only this authority's cheap fields.
        self._manifest = current

    def _issue_token(
        self,
        authority: EnvironmentAuthorityV1,
        purpose: str,
    ) -> EnvironmentVerificationToken:
        """Return one opaque token after a fresh complete cheap walk.

        Parameters
        ----------
        authority:
            Exact active prefix capability.
        purpose:
            Closed ``currentness-pass`` or ``spawn`` use class.

        Returns
        -------
        EnvironmentVerificationToken
            Process-local proof bound to the post-validation cache epoch and fingerprint.
        """

        self._walk_and_validate(authority)
        manifest = self._manifest
        if manifest is None:
            raise AuthorityDerivationError(
                "environment authority was invalidated during validation"
            )
        self._verification_sequence += 1
        return EnvironmentVerificationToken(
            cache=self,
            authority_id=authority.authority_id,
            epoch=self._epoch,
            sequence=self._verification_sequence,
            purpose=purpose,
            observed_fingerprint=manifest.cheap_tree_fingerprint,
            secret=_VERIFICATION_TOKEN_SECRET,
        )

    @contextmanager
    def currentness_pass(
        self,
        authority: EnvironmentAuthorityV1,
    ) -> Iterator[EnvironmentVerificationToken]:
        """Yield one reusable complete observation for a scheduling/currentness pass.

        Parameters
        ----------
        authority:
            Exact active prefix capability shared by all models in the pass.

        Yields
        ------
        EnvironmentVerificationToken
            Pass-only proof that cannot authorize a worker spawn.
        """

        if self._active_currentness_token is not None:
            raise AuthorityDerivationError("environment currentness passes cannot be nested")
        token = self._issue_token(authority, "currentness-pass")
        self.currentness_passes += 1
        self._active_currentness_token = token
        try:
            yield token
        finally:
            token._active = False
            if self._active_currentness_token is token:
                self._active_currentness_token = None

    @contextmanager
    def spawn_verification(
        self,
        authority: EnvironmentAuthorityV1,
    ) -> Iterator[EnvironmentVerificationToken]:
        """Yield one fresh one-shot observation for a single worker spawn.

        Parameters
        ----------
        authority:
            Exact active prefix capability checked immediately before spawn setup.

        Yields
        ------
        EnvironmentVerificationToken
            Spawn-only proof reusable by compiler, projection, renderer, and supervisor.
        """

        self.spawn_validations += 1
        token = self._issue_token(authority, "spawn")
        try:
            yield token
        finally:
            token._active = False

    def mark_spawned(self, verification_token: EnvironmentVerificationToken) -> None:
        """Consume one spawn token at the actual subprocess boundary.

        Parameters
        ----------
        verification_token:
            Active spawn-purpose proof passed through every pre-spawn consumer.
        """

        authority = self._authority
        if authority is None:
            raise AuthorityDerivationError("environment authority is unavailable at spawn")
        self._assert_verification_token(
            authority,
            verification_token,
            required_purpose="spawn",
        )
        if verification_token._spawn_marked:
            raise AuthorityDerivationError("environment spawn token was already consumed")
        verification_token._spawn_marked = True
        self.real_spawns += 1

    def _assert_verification_token(
        self,
        authority: EnvironmentAuthorityV1,
        verification_token: EnvironmentVerificationToken,
        *,
        required_purpose: Optional[str] = None,
    ) -> None:
        """Require one active token to match every cache/authority binding field.

        Parameters
        ----------
        authority:
            Authority entering a verification consumer.
        verification_token:
            Opaque cache-issued proof.
        required_purpose:
            Optional exact purpose required by a spawn-only boundary.
        """

        self.assert_active(authority)
        manifest = self._manifest
        if (
            not isinstance(verification_token, EnvironmentVerificationToken)
            or not verification_token._active
            or verification_token._cache is not self
            or verification_token._authority_id != authority.authority_id
            or verification_token._epoch != self._epoch
            or not 0 < verification_token._sequence <= self._verification_sequence
            or manifest is None
            or verification_token._observed_fingerprint != manifest.cheap_tree_fingerprint
            or verification_token._purpose not in {"currentness-pass", "spawn"}
            or (required_purpose is not None and verification_token._purpose != required_purpose)
        ):
            raise AuthorityDerivationError("environment verification token is stale or mismatched")

    def assert_active(self, authority: EnvironmentAuthorityV1) -> None:
        """Require one authority to belong to this active lifecycle cache.

        Parameters
        ----------
        authority:
            Prefix authority already validated for the current scheduling pass.

        Raises
        ------
        AuthorityDerivationError
            If the cache was invalidated or belongs to another authority.
        """

        if (
            authority._cache is not self
            or self._manifest is None
            or self._authority is not authority
        ):
            raise AuthorityDerivationError("environment authority cache association is invalid")

    def invalidate(self) -> None:
        """Invalidate the cached active prefix on teardown or quarantine."""

        if self._manifest is None and self._authority is None:
            return
        self.invalidations += 1
        self._epoch += 1
        if self._active_currentness_token is not None:
            self._active_currentness_token._active = False
            self._active_currentness_token = None
        self._manifest = None
        self._authority = None


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

    @property
    def closure_identity(self) -> str:
        """Return the pre-execution identity of every verified closure member.

        Returns
        -------
        str
            Cycle-free executable closure identity.
        """

        return _executable_closure_identity(
            code_manifest_identity=self.code_manifest_identity,
            environment_generation=self.environment_generation,
            installed_package_inventory_sha256=self.installed_package_inventory_sha256,
            code_members=self.code_members,
            runtime_members=self.runtime_members,
            standard_input_asset=self.standard_input_asset,
            lookup_directories=self.lookup_directories,
        )

    @property
    def runtime_support(self) -> tuple[tuple[Path, str], ...]:
        """Return exact runtime files through the legacy enforcement adapter.

        Returns
        -------
        tuple[tuple[pathlib.Path, str], ...]
            Every exact runtime member labeled ``runtime-file``. Lookup directories
            are intentionally absent because they grant no descendant authority.
        """

        return tuple((member.path, "runtime-file") for member in self.runtime_members)


@dataclass(frozen=True)
class ExecutionReadManifestV3:
    """Live worker capability with one complete sealed environment authority."""

    manifest_version: str
    manifest_id: str
    stable_id: str
    work_id: str
    execution_identity: str
    code_manifest_identity: str
    environment_generation: str
    code_members: tuple[RuntimeMember, ...]
    worker_members: tuple[RuntimeMember, ...]
    environment_authority: EnvironmentAuthorityV1
    standard_input_asset: Optional[tuple[Path, str, str]]
    lookup_directories: tuple[RuntimeLookupDirectory, ...]

    @property
    def closure_identity(self) -> str:
        """Return the cycle-free identity of all four read-authority partitions."""

        return _executable_closure_v3_identity(
            code_manifest_identity=self.code_manifest_identity,
            environment_authority=self.environment_authority,
            code_members=self.code_members,
            worker_members=self.worker_members,
            standard_input_asset=self.standard_input_asset,
            lookup_directories=self.lookup_directories,
        )

    @property
    def runtime_members(self) -> tuple[RuntimeMember, ...]:
        """Return exact outside-prefix crawler/bootstrap members."""

        return self.worker_members

    @property
    def runtime_support(self) -> tuple[tuple[Path, str], ...]:
        """Return only exact outside-prefix files through the legacy adapter."""

        return tuple((member.path, "runtime-file") for member in self.worker_members)


@dataclass(frozen=True)
class ExecutableClosure:
    """Verified executable members collected before execution identity exists."""

    identity: str
    code_manifest_identity: str
    environment_generation: str
    installed_package_inventory_sha256: str
    code_members: tuple[RuntimeMember, ...]
    runtime_members: tuple[RuntimeMember, ...]
    standard_input_asset: Optional[tuple[Path, str, str]]
    lookup_directories: tuple[RuntimeLookupDirectory, ...]


@dataclass(frozen=True)
class ExecutableClosureV3:
    """Verified four-part execution closure collected before request identity."""

    identity: str
    code_manifest_identity: str
    environment_generation: str
    code_members: tuple[RuntimeMember, ...]
    worker_members: tuple[RuntimeMember, ...]
    environment_authority: EnvironmentAuthorityV1
    standard_input_asset: Optional[tuple[Path, str, str]]
    lookup_directories: tuple[RuntimeLookupDirectory, ...]


@dataclass(frozen=True)
class ExactReadCapability:
    """Verified exact-member projection shared by every enforcement layer."""

    manifest_id: str
    closure_identity: str
    members: tuple[RuntimeMember, ...]
    standard_input_asset: Optional[tuple[Path, str, str]]
    lookup_directories: tuple[RuntimeLookupDirectory, ...]

    @property
    def member_paths(self) -> tuple[Path, ...]:
        """Return exact authorized regular files in canonical order.

        Returns
        -------
        tuple[pathlib.Path, ...]
            Exact semantic file capabilities.
        """

        return tuple(member.path for member in self.members)


@dataclass(frozen=True)
class EnvironmentReadCapability:
    """Single verified read projection shared by v3 enforcement consumers."""

    manifest_id: str
    closure_identity: str
    exact_members: tuple[RuntimeMember, ...]
    environment_prefix: Path
    selected_interpreter: Path
    startup_pth_paths: tuple[Path, ...]
    external_targets: tuple[EnvironmentExternalTarget, ...]
    standard_input_asset: Optional[tuple[Path, str, str]]
    lookup_directories: tuple[RuntimeLookupDirectory, ...]

    @property
    def exact_member_paths(self) -> tuple[Path, ...]:
        """Return exact model, crawler, external, and asset file capabilities."""

        paths = [member.path for member in self.exact_members]
        paths.extend(target.path for target in self.external_targets)
        if self.standard_input_asset is not None:
            paths.append(self.standard_input_asset[0])
        return tuple(dict.fromkeys(paths))


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
        author_schema_identity=content_identity("schemas/author-result-v4.schema.json"),
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


def _verified_member(
    member: RuntimeMember,
    *,
    allowed_kinds: frozenset[str],
    field: str,
) -> RuntimeMember:
    """Return one normalized exact executable-closure member.

    Parameters
    ----------
    member:
        Typed member supplied by the trusted closure collector.
    allowed_kinds:
        Closed kinds valid for the member's manifest partition.
    field:
        Diagnostic partition name.

    Returns
    -------
    RuntimeMember
        Normalized member with an absolute canonical path.

    Raises
    ------
    AuthorityDerivationError
        If the path is a root/directory/alias, the kind is invalid, or bytes differ.
    """

    if not isinstance(member, RuntimeMember):
        raise AuthorityDerivationError(f"{field} must contain RuntimeMember values")
    path = member.path
    if not path.is_absolute():
        raise AuthorityDerivationError(f"{field} member path must be absolute: {path}")
    if member.kind not in allowed_kinds:
        raise AuthorityDerivationError(
            f"{field} member has a non-file or unknown kind: {member.kind}"
        )
    _require_nonempty_string(member.provenance, f"{field}.provenance")
    digest = _require_hash(member.sha256, f"{field}.sha256")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise AuthorityDerivationError(f"{field} member is unavailable: {path}") from exc
    if path.is_symlink() or resolved != path or not path.is_file():
        raise AuthorityDerivationError(f"{field} member is not an exact unaliased file: {path}")
    suffixes = _MEMBER_SUFFIXES.get(member.kind)
    lowered_name = path.name.lower()
    if suffixes is not None and path.suffix.lower() not in suffixes:
        if member.kind not in {"native-extension"} or ".so." not in lowered_name:
            raise AuthorityDerivationError(
                f"{field} member kind does not match its file suffix: {path}"
            )
    if member.kind == "native-library" and not (
        path.suffix.lower() in {".dylib", ".so"} or ".so." in lowered_name
    ):
        raise AuthorityDerivationError(f"{field} native library has an invalid suffix: {path}")
    try:
        observed = hash_bytes(path.read_bytes())
    except OSError as exc:
        raise AuthorityDerivationError(f"{field} member cannot be read: {path}") from exc
    if observed != digest:
        raise AuthorityDerivationError(f"{field} member digest changed: {path}")
    return RuntimeMember(
        path=path,
        sha256=digest,
        kind=member.kind,
        provenance=member.provenance,
    )


def _manifest_v2_payload(
    *,
    stable_id: str,
    work_id: str,
    execution_identity: str,
    code_manifest_identity: str,
    environment_generation: str,
    installed_package_inventory_sha256: str,
    code_members: Sequence[RuntimeMember],
    runtime_members: Sequence[RuntimeMember],
    standard_input_asset: Optional[tuple[Path, str, str]],
    lookup_directories: Sequence[RuntimeLookupDirectory],
) -> JsonObject:
    """Build the canonical JSON identity payload for manifest v2.

    Parameters
    ----------
    stable_id, work_id, execution_identity, code_manifest_identity,
    environment_generation, installed_package_inventory_sha256:
        Exact request, implementation, and environment associations.
    code_members, runtime_members:
        Exact verified semantic file capabilities.
    standard_input_asset:
        Optional exact selected standard-input member.
    lookup_directories:
        Non-authorizing lookup and mount scaffolding.

    Returns
    -------
    dict[str, Any]
        Canonical JSON-compatible identity payload.
    """

    def member_payload(member: RuntimeMember) -> JsonObject:
        """Render one typed member into canonical JSON."""

        return {
            "path": str(member.path),
            "sha256": member.sha256,
            "kind": member.kind,
            "provenance": member.provenance,
        }

    return {
        "manifest_version": EXECUTION_READ_MANIFEST_VERSION_V2,
        "stable_id": stable_id,
        "work_id": work_id,
        "execution_identity": execution_identity,
        "code_manifest_identity": code_manifest_identity,
        "environment_generation": environment_generation,
        "installed_package_inventory_sha256": installed_package_inventory_sha256,
        "code_members": [member_payload(member) for member in code_members],
        "runtime_members": [member_payload(member) for member in runtime_members],
        "standard_input_asset": (
            None
            if standard_input_asset is None
            else {
                "path": str(standard_input_asset[0]),
                "sha256": standard_input_asset[1],
                "asset_id": standard_input_asset[2],
            }
        ),
        "lookup_directories": [
            {"path": str(directory.path), "provenance": directory.provenance}
            for directory in lookup_directories
        ],
    }


def _executable_closure_identity(
    *,
    code_manifest_identity: str,
    environment_generation: str,
    installed_package_inventory_sha256: str,
    code_members: Sequence[RuntimeMember],
    runtime_members: Sequence[RuntimeMember],
    standard_input_asset: Optional[tuple[Path, str, str]],
    lookup_directories: Sequence[RuntimeLookupDirectory],
) -> str:
    """Hash verified executable members without an execution-identity cycle.

    Parameters
    ----------
    code_manifest_identity, environment_generation, installed_package_inventory_sha256:
        Frozen implementation and environment associations.
    code_members, runtime_members, standard_input_asset, lookup_directories:
        Exact semantic files, selected input, and non-authorizing traversal scaffolds.

    Returns
    -------
    str
        Canonical pre-execution closure identity.
    """

    def member_payload(member: RuntimeMember) -> JsonObject:
        """Render one exact member for closure hashing.

        Parameters
        ----------
        member:
            Verified executable member.

        Returns
        -------
        dict[str, Any]
            Canonical member payload.
        """

        return {
            "path": str(member.path),
            "sha256": member.sha256,
            "kind": member.kind,
            "provenance": member.provenance,
        }

    return stable_hash(
        {
            "closure_version": "menagerie.crawler.executable-closure.v1",
            "code_manifest_identity": code_manifest_identity,
            "environment_generation": environment_generation,
            "installed_package_inventory_sha256": installed_package_inventory_sha256,
            "code_members": [member_payload(member) for member in code_members],
            "runtime_members": [member_payload(member) for member in runtime_members],
            "standard_input_asset": (
                None
                if standard_input_asset is None
                else {
                    "path": str(standard_input_asset[0]),
                    "sha256": standard_input_asset[1],
                    "asset_id": standard_input_asset[2],
                }
            ),
            "lookup_directories": [
                {"path": str(directory.path), "provenance": directory.provenance}
                for directory in lookup_directories
            ],
        }
    )


def _import_names(path: Path, lookup_directories: Sequence[Path]) -> tuple[str, ...]:
    """Return statically resolvable module names imported by one Python source.

    Parameters
    ----------
    path:
        Exact Python source member.
    lookup_directories:
        Ordered import lookup scaffolding.

    Returns
    -------
    tuple[str, ...]
        Absolute module names named by imports and literal dynamic-import calls.

    Raises
    ------
    AuthorityDerivationError
        If source parsing or a relative-import package association fails.
    """

    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeError, SyntaxError) as exc:
        raise AuthorityDerivationError(
            f"executable Python member cannot be parsed: {path}"
        ) from exc
    containing_roots = [root for root in lookup_directories if path.is_relative_to(root)]
    containing_roots.sort(key=lambda root: len(root.parts), reverse=True)
    package_parts: tuple[str, ...] = ()
    if containing_roots:
        relative = path.relative_to(containing_roots[0])
        package_parts = relative.parent.parts
        if path.name == "__init__.py":
            package_parts = relative.parent.parts
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
            continue
        if isinstance(node, ast.ImportFrom):
            if node.level:
                if not containing_roots or node.level > len(package_parts) + 1:
                    raise AuthorityDerivationError(
                        f"relative import cannot be bound to a lookup package: {path}"
                    )
                retained = len(package_parts) - (node.level - 1)
                base_parts = (*package_parts[:retained],)
            else:
                base_parts = ()
            module_parts = tuple(node.module.split(".")) if node.module else ()
            base = ".".join((*base_parts, *module_parts))
            if base:
                names.append(base)
            for alias in node.names:
                if alias.name != "*":
                    child = ".".join(value for value in (base, alias.name) if value)
                    if child:
                        names.append(child)
            continue
        if not isinstance(node, ast.Call) or not node.args:
            continue
        function_name: Optional[str] = None
        if isinstance(node.func, ast.Name):
            function_name = node.func.id
        elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            function_name = f"{node.func.value.id}.{node.func.attr}"
        literal = node.args[0]
        if (
            function_name in {"__import__", "importlib.import_module"}
            and isinstance(literal, ast.Constant)
            and isinstance(literal.value, str)
            and literal.value
        ):
            names.append(literal.value)
    return tuple(dict.fromkeys(names))


def _module_files(module_name: str, lookup_directories: Sequence[Path]) -> tuple[Path, ...]:
    """Resolve one module name to its exact importable files under lookup scaffolding.

    Parameters
    ----------
    module_name:
        Absolute dotted module name.
    lookup_directories:
        Ordered non-authorizing import roots.

    Returns
    -------
    tuple[Path, ...]
        Package initializers plus the module source/bytecode/native file, or empty
        when the module is built-in, frozen, or outside the supplied lookup closure.

    Raises
    ------
    AuthorityDerivationError
        If multiple lookup roots resolve the same named module differently.
    """

    parts = tuple(part for part in module_name.split(".") if part)
    if not parts:
        return ()
    resolutions: list[tuple[Path, ...]] = []
    for root in lookup_directories:
        package_files: list[Path] = []
        for index in range(1, len(parts)):
            initializer = root.joinpath(*parts[:index], "__init__.py")
            if initializer.is_file():
                package_files.append(initializer.resolve())
        leaf_base = root.joinpath(*parts)
        candidates = [leaf_base.with_suffix(".py"), leaf_base / "__init__.py"]
        try:
            candidates.extend(sorted(leaf_base.parent.glob(f"{leaf_base.name}*.so")))
            candidates.extend(sorted(leaf_base.parent.glob(f"{leaf_base.name}*.pyd")))
        except OSError:
            pass
        leaves = tuple(candidate.resolve() for candidate in candidates if candidate.is_file())
        if len(leaves) > 1:
            raise AuthorityDerivationError(
                f"static import resolves to multiple executable files: {module_name}"
            )
        if leaves:
            resolutions.append(tuple(dict.fromkeys((*package_files, leaves[0]))))
    if len(resolutions) > 1:
        raise AuthorityDerivationError(
            f"static import is ambiguous across lookup directories: {module_name}"
        )
    return resolutions[0] if resolutions else ()


def _validate_static_import_closure(
    members: Sequence[RuntimeMember],
    lookup_directories: Sequence[RuntimeLookupDirectory],
) -> None:
    """Require every statically resolvable Python import to be an exact member.

    Parameters
    ----------
    members:
        Complete model/runtime member inventory.
    lookup_directories:
        Non-authorizing roots used only to resolve named imports.

    Raises
    ------
    AuthorityDerivationError
        If a source imports executable bytes that are absent from the inventory.
    """

    member_paths = {member.path for member in members}
    roots = tuple(directory.path for directory in lookup_directories)
    for member in members:
        if member.kind != "python-source" or member.path.suffix != ".py":
            continue
        for module_name in _import_names(member.path, roots):
            missing = [
                path for path in _module_files(module_name, roots) if path not in member_paths
            ]
            if missing:
                rendered = ", ".join(str(path) for path in missing)
                raise AuthorityDerivationError(
                    f"static import is outside the executable member inventory: {rendered}"
                )


def compile_execution_read_manifest_v2(
    *,
    stable_id: str,
    work_id: str,
    execution_identity: str,
    code_manifest_identity: str,
    environment_generation: str,
    installed_package_inventory_sha256: str,
    code_members: Sequence[RuntimeMember],
    runtime_members: Sequence[RuntimeMember],
    standard_input_asset: Optional[tuple[Path, str, str]] = None,
    lookup_directories: Sequence[RuntimeLookupDirectory] = (),
) -> ExecutionReadManifestV2:
    """Compile an exact byte-inventoried executable closure into manifest v2.

    Parameters
    ----------
    stable_id, work_id:
        Exact model and work-generation association.
    execution_identity, code_manifest_identity, environment_generation,
    installed_package_inventory_sha256:
        Exact execution, model-code, environment, and installed-inventory identities.
    code_members:
        Accepted model-code files, each already classified with exact provenance.
    runtime_members:
        Worker/bootstrap, interpreter, import, native-loader, metadata, and package-data files.
    standard_input_asset:
        Optional selected standard input as absolute path, digest, and asset ID.
    lookup_directories:
        Directory scaffolding used for lookup or mounts. It grants no descendant capability.

    Returns
    -------
    ExecutionReadManifestV2
        Frozen manifest whose identity includes every semantic file and association.

    Raises
    ------
    AuthorityDerivationError
        If an identity, member, digest, path, kind, alias, or partition is invalid.

    Notes
    -----
    This producer intentionally accepts only an exact member inventory. Repository or
    environment roots cannot be expressed as semantic members, and lookup directories
    are excluded from the authorized member set consumed by enforcement layers.
    """

    stable_id = _require_nonempty_string(stable_id, "stable_id")
    work_id = _require_nonempty_string(work_id, "work_id")
    execution_identity = _require_hash(execution_identity, "execution_identity")
    code_manifest_identity = _require_hash(code_manifest_identity, "code_manifest_identity")
    environment_generation = _require_hash(environment_generation, "environment_generation")
    installed_package_inventory_sha256 = _require_hash(
        installed_package_inventory_sha256,
        "installed_package_inventory_sha256",
    )
    normalized_code = tuple(
        sorted(
            (
                _verified_member(member, allowed_kinds=_CODE_MEMBER_KINDS, field="code_members")
                for member in code_members
            ),
            key=lambda member: (str(member.path), member.kind, member.provenance),
        )
    )
    normalized_runtime = tuple(
        sorted(
            (
                _verified_member(
                    member,
                    allowed_kinds=_RUNTIME_MEMBER_KINDS,
                    field="runtime_members",
                )
                for member in runtime_members
            ),
            key=lambda member: (str(member.path), member.kind, member.provenance),
        )
    )
    member_paths = [member.path for member in (*normalized_code, *normalized_runtime)]
    if len(member_paths) != len(set(member_paths)):
        raise AuthorityDerivationError(
            "execution closure contains a duplicate or cross-partition member path"
        )

    normalized_asset: Optional[tuple[Path, str, str]] = None
    if standard_input_asset is not None:
        path, digest, asset_id = standard_input_asset
        digest = _require_hash(digest, "standard_input_asset.sha256")
        asset_id = _require_nonempty_string(asset_id, "standard_input_asset.asset_id")
        if not path.is_absolute():
            raise AuthorityDerivationError("standard input asset path must be absolute")
        try:
            resolved = path.resolve(strict=True)
            observed = hash_bytes(path.read_bytes())
        except OSError as exc:
            raise AuthorityDerivationError("standard input asset is unavailable") from exc
        if path.is_symlink() or resolved != path or not path.is_file() or path in member_paths:
            raise AuthorityDerivationError("standard input asset is aliased or overlaps code")
        if observed != digest:
            raise AuthorityDerivationError("standard input asset digest changed")
        normalized_asset = (path, digest, asset_id)

    normalized_lookup: list[RuntimeLookupDirectory] = []
    seen_lookup: set[Path] = set()
    for directory in lookup_directories:
        if not isinstance(directory, RuntimeLookupDirectory):
            raise AuthorityDerivationError(
                "lookup_directories must contain RuntimeLookupDirectory values"
            )
        path = directory.path
        provenance = _require_nonempty_string(directory.provenance, "lookup_directories.provenance")
        try:
            resolved = path.resolve(strict=True)
        except OSError as exc:
            raise AuthorityDerivationError(f"lookup directory is unavailable: {path}") from exc
        if (
            not path.is_absolute()
            or path.is_symlink()
            or resolved != path
            or not path.is_dir()
            or path in seen_lookup
        ):
            raise AuthorityDerivationError(f"lookup directory is unsafe or duplicated: {path}")
        seen_lookup.add(path)
        normalized_lookup.append(RuntimeLookupDirectory(path=path, provenance=provenance))
    normalized_lookup.sort(key=lambda directory: (str(directory.path), directory.provenance))
    _validate_static_import_closure(
        (*normalized_code, *normalized_runtime),
        normalized_lookup,
    )

    payload = _manifest_v2_payload(
        stable_id=stable_id,
        work_id=work_id,
        execution_identity=execution_identity,
        code_manifest_identity=code_manifest_identity,
        environment_generation=environment_generation,
        installed_package_inventory_sha256=installed_package_inventory_sha256,
        code_members=normalized_code,
        runtime_members=normalized_runtime,
        standard_input_asset=normalized_asset,
        lookup_directories=normalized_lookup,
    )
    return ExecutionReadManifestV2(
        manifest_version=EXECUTION_READ_MANIFEST_VERSION_V2,
        manifest_id=stable_hash(payload),
        stable_id=stable_id,
        work_id=work_id,
        execution_identity=execution_identity,
        code_manifest_identity=code_manifest_identity,
        environment_generation=environment_generation,
        installed_package_inventory_sha256=installed_package_inventory_sha256,
        code_members=normalized_code,
        runtime_members=normalized_runtime,
        standard_input_asset=normalized_asset,
        lookup_directories=tuple(normalized_lookup),
    )


def verify_execution_read_manifest_v2(manifest: ExecutionReadManifestV2) -> None:
    """Reverify every manifest-v2 byte and association immediately before spawn.

    Parameters
    ----------
    manifest:
        Previously compiled frozen v2 executable closure.

    Raises
    ------
    AuthorityDerivationError
        If the version, identity, path, kind, provenance, or any member byte changed.
    """

    if manifest.manifest_version != EXECUTION_READ_MANIFEST_VERSION_V2:
        raise AuthorityDerivationError("execution read manifest has the wrong v2 discriminator")
    rebuilt = compile_execution_read_manifest_v2(
        stable_id=manifest.stable_id,
        work_id=manifest.work_id,
        execution_identity=manifest.execution_identity,
        code_manifest_identity=manifest.code_manifest_identity,
        environment_generation=manifest.environment_generation,
        installed_package_inventory_sha256=manifest.installed_package_inventory_sha256,
        code_members=manifest.code_members,
        runtime_members=manifest.runtime_members,
        standard_input_asset=manifest.standard_input_asset,
        lookup_directories=manifest.lookup_directories,
    )
    if rebuilt != manifest:
        raise AuthorityDerivationError("execution read manifest identity is stale or rewritten")


def collect_executable_closure(
    *,
    code_manifest_identity: str,
    environment_generation: str,
    installed_package_inventory_sha256: str,
    code_members: Sequence[RuntimeMember],
    runtime_members: Sequence[RuntimeMember],
    standard_input_asset: Optional[tuple[Path, str, str]] = None,
    lookup_directories: Sequence[RuntimeLookupDirectory] = (),
) -> ExecutableClosure:
    """Collect and verify an executable closure before execution identity derivation.

    Parameters
    ----------
    code_manifest_identity, environment_generation, installed_package_inventory_sha256:
        Frozen implementation and environment associations.
    code_members, runtime_members, standard_input_asset, lookup_directories:
        Candidate exact members, selected input, and non-authorizing scaffolds.

    Returns
    -------
    ExecutableClosure
        Normalized byte-verified closure with a cycle-free identity.
    """

    probe_identity = stable_hash("executable-closure-pre-identity-probe")
    probe = compile_execution_read_manifest_v2(
        stable_id="closure-collection",
        work_id="closure-collection",
        execution_identity=probe_identity,
        code_manifest_identity=code_manifest_identity,
        environment_generation=environment_generation,
        installed_package_inventory_sha256=installed_package_inventory_sha256,
        code_members=code_members,
        runtime_members=runtime_members,
        standard_input_asset=standard_input_asset,
        lookup_directories=lookup_directories,
    )
    return ExecutableClosure(
        identity=probe.closure_identity,
        code_manifest_identity=probe.code_manifest_identity,
        environment_generation=probe.environment_generation,
        installed_package_inventory_sha256=probe.installed_package_inventory_sha256,
        code_members=probe.code_members,
        runtime_members=probe.runtime_members,
        standard_input_asset=probe.standard_input_asset,
        lookup_directories=probe.lookup_directories,
    )


def compile_execution_read_manifest_from_closure(
    closure: ExecutableClosure,
    *,
    stable_id: str,
    work_id: str,
    execution_identity: str,
) -> ExecutionReadManifestV2:
    """Bind one pre-identity closure to final request associations.

    Parameters
    ----------
    closure:
        Verified cycle-free executable closure.
    stable_id, work_id, execution_identity:
        Final model, work-generation, and execution associations.

    Returns
    -------
    ExecutionReadManifestV2
        Final exact-member execution capability.
    """

    manifest = compile_execution_read_manifest_v2(
        stable_id=stable_id,
        work_id=work_id,
        execution_identity=execution_identity,
        code_manifest_identity=closure.code_manifest_identity,
        environment_generation=closure.environment_generation,
        installed_package_inventory_sha256=closure.installed_package_inventory_sha256,
        code_members=closure.code_members,
        runtime_members=closure.runtime_members,
        standard_input_asset=closure.standard_input_asset,
        lookup_directories=closure.lookup_directories,
    )
    if manifest.closure_identity != closure.identity:
        raise AuthorityDerivationError("execution closure changed during final manifest binding")
    return manifest


def exact_read_capability(manifest: ExecutionReadManifestV2) -> ExactReadCapability:
    """Verify and project one manifest into exact semantic read authority.

    Parameters
    ----------
    manifest:
        Final v2 manifest to reverify.

    Returns
    -------
    ExactReadCapability
        Exact member paths and non-authorizing lookup scaffolds.
    """

    verify_execution_read_manifest_v2(manifest)
    return ExactReadCapability(
        manifest_id=manifest.manifest_id,
        closure_identity=manifest.closure_identity,
        members=(*manifest.code_members, *manifest.runtime_members),
        standard_input_asset=manifest.standard_input_asset,
        lookup_directories=manifest.lookup_directories,
    )


def _environment_authority_payload(authority: EnvironmentAuthorityV1) -> JsonObject:
    """Return the closed local environment-authority identity payload.

    Parameters
    ----------
    authority:
        Bound environment prefix authority.

    Returns
    -------
    dict[str, Any]
        Canonical authority payload.
    """

    return {
        "version": authority.authority_version,
        "prefix": str(authority.prefix),
        "selected_interpreter_relative_path": authority.selected_interpreter_relative_path,
        "selected_interpreter_digest": authority.selected_interpreter_digest,
        "environment_generation": authority.environment_generation,
        "environment_content_sha256": authority.content_manifest_sha256,
        "external_targets": [
            {"path": str(target.path), "sha256": target.sha256, "kind": target.kind}
            for target in authority.external_targets
        ],
    }


def verify_environment_authority(
    authority: EnvironmentAuthorityV1,
    *,
    verification_token: Optional[EnvironmentVerificationToken] = None,
) -> None:
    """Verify one authority's identities and current complete prefix seal.

    Parameters
    ----------
    authority:
        Previously bound environment capability.
    verification_token:
        Optional cache-created pass or spawn proof reusing one complete tree walk.

    Raises
    ------
    AuthorityDerivationError
        If an association, interpreter, or sealed member is stale.
    """

    if not isinstance(authority, EnvironmentAuthorityV1):
        raise AuthorityDerivationError("execution manifest lacks EnvironmentAuthorityV1")
    if authority.authority_version != ENVIRONMENT_AUTHORITY_VERSION_V1:
        raise AuthorityDerivationError("environment authority has the wrong discriminator")
    if authority.authority_id != stable_hash(_environment_authority_payload(authority)):
        raise AuthorityDerivationError("environment authority identity is rewritten")
    expected_generation = stable_hash(
        {
            "version": ENVIRONMENT_GENERATION_VERSION_V2,
            "base_environment_generation": authority.base_environment_generation,
            "environment_content_sha256": authority.content_manifest_sha256,
        }
    )
    if authority.environment_generation != expected_generation:
        raise AuthorityDerivationError("environment generation omits or rewrites the content seal")
    lexical_interpreter = authority.prefix / authority.selected_interpreter_relative_path
    try:
        resolved_interpreter = lexical_interpreter.resolve(strict=True)
    except OSError as exc:
        raise AuthorityDerivationError("selected interpreter is unavailable") from exc
    expected_target = authority.prefix / authority.selected_interpreter_target_relative_path
    if resolved_interpreter != expected_target:
        raise AuthorityDerivationError("selected interpreter association changed")
    if not resolved_interpreter.is_relative_to(authority.prefix):
        raise AuthorityDerivationError("selected interpreter resolves outside the canonical prefix")
    authority._cache.verify(authority, verification_token=verification_token)


def _v3_member(
    value: RuntimeMember | tuple[Path, str, str],
    *,
    provenance: str,
) -> RuntimeMember:
    """Normalize a typed member or public three-tuple into one RuntimeMember.

    Parameters
    ----------
    value:
        Typed member or path/digest/kind tuple.
    provenance:
        Trusted compiler provenance for tuple callers.

    Returns
    -------
    RuntimeMember
        Typed candidate ready for exact verification.
    """

    if isinstance(value, RuntimeMember):
        return value
    path, digest, kind = value
    return RuntimeMember(path=path, sha256=digest, kind=kind, provenance=provenance)


def _normalized_lookup_directories(
    lookup_directories: Sequence[RuntimeLookupDirectory],
) -> tuple[RuntimeLookupDirectory, ...]:
    """Verify lookup scaffolds without granting descendant read authority.

    Parameters
    ----------
    lookup_directories:
        Candidate non-authorizing import roots.

    Returns
    -------
    tuple[RuntimeLookupDirectory, ...]
        Canonically sorted unique roots.
    """

    normalized: list[RuntimeLookupDirectory] = []
    seen: set[Path] = set()
    for directory in lookup_directories:
        if not isinstance(directory, RuntimeLookupDirectory):
            raise AuthorityDerivationError(
                "lookup_directories must contain RuntimeLookupDirectory values"
            )
        try:
            resolved = directory.path.resolve(strict=True)
        except OSError as exc:
            raise AuthorityDerivationError(
                f"lookup directory is unavailable: {directory.path}"
            ) from exc
        if (
            not directory.path.is_absolute()
            or directory.path.is_symlink()
            or resolved != directory.path
            or not directory.path.is_dir()
            or directory.path in seen
        ):
            raise AuthorityDerivationError(
                f"lookup directory is unsafe or duplicated: {directory.path}"
            )
        seen.add(directory.path)
        normalized.append(
            RuntimeLookupDirectory(
                path=directory.path,
                provenance=_require_nonempty_string(
                    directory.provenance,
                    "lookup_directories.provenance",
                ),
            )
        )
    normalized.sort(key=lambda directory: (str(directory.path), directory.provenance))
    return tuple(normalized)


def _normalized_standard_asset(
    standard_input_asset: Optional[tuple[Path, str, str]],
    occupied: set[Path],
) -> Optional[tuple[Path, str, str]]:
    """Verify the exact selected standard input independently of prefix authority.

    Parameters
    ----------
    standard_input_asset:
        Optional exact asset tuple.
    occupied:
        Exact code/worker paths that the asset must not overlap.

    Returns
    -------
    tuple[pathlib.Path, str, str] | None
        Normalized exact asset.
    """

    if standard_input_asset is None:
        return None
    path, digest, asset_id = standard_input_asset
    digest = _require_hash(digest, "standard_input_asset.sha256")
    asset_id = _require_nonempty_string(asset_id, "standard_input_asset.asset_id")
    try:
        resolved = path.resolve(strict=True)
        observed = hash_bytes(path.read_bytes())
    except OSError as exc:
        raise AuthorityDerivationError("standard input asset is unavailable") from exc
    if not path.is_absolute() or path.is_symlink() or resolved != path or path in occupied:
        raise AuthorityDerivationError("standard input asset is aliased or overlaps code")
    if observed != digest:
        raise AuthorityDerivationError("standard input asset digest changed")
    return path, digest, asset_id


def _validate_static_import_closure_v3(
    members: Sequence[RuntimeMember],
    lookup_directories: Sequence[RuntimeLookupDirectory],
    environment_prefix: Path,
    external_targets: Sequence[EnvironmentExternalTarget],
) -> None:
    """Require static imports outside the sealed prefix to remain exact members.

    Parameters
    ----------
    members:
        Exact model and crawler/bootstrap files.
    lookup_directories:
        Import resolution scaffolds.
    environment_prefix:
        Sole digest-bound semantic root.
    external_targets:
        Exact digest-bound regular files reached by sealed-prefix symlinks.
    """

    member_paths = {
        *(member.path for member in members),
        *(target.path for target in external_targets),
    }
    roots = tuple(directory.path for directory in lookup_directories)
    for member in members:
        if member.kind != "python-source" or member.path.suffix != ".py":
            continue
        for module_name in _import_names(member.path, roots):
            missing = [
                path
                for path in _module_files(module_name, roots)
                if not path.is_relative_to(environment_prefix) and path not in member_paths
            ]
            if missing:
                rendered = ", ".join(str(path) for path in missing)
                raise AuthorityDerivationError(
                    f"static import is outside sealed environment and exact inventory: {rendered}"
                )


def _manifest_v3_payload(
    *,
    stable_id: str,
    work_id: str,
    execution_identity: str,
    code_manifest_identity: str,
    environment_authority: EnvironmentAuthorityV1,
    code_members: Sequence[RuntimeMember],
    worker_members: Sequence[RuntimeMember],
    standard_input_asset: Optional[tuple[Path, str, str]],
    lookup_directories: Sequence[RuntimeLookupDirectory],
) -> JsonObject:
    """Build the closed canonical execution-read-manifest v3 payload."""

    def member_payload(member: RuntimeMember) -> JsonObject:
        """Render one exact outside-prefix member."""

        return {
            "path": str(member.path),
            "sha256": member.sha256,
            "kind": member.kind,
            "provenance": member.provenance,
        }

    return {
        "manifest_version": EXECUTION_READ_MANIFEST_VERSION_V3,
        "stable_id": stable_id,
        "work_id": work_id,
        "execution_identity": execution_identity,
        "code_manifest_identity": code_manifest_identity,
        "environment_generation": environment_authority.environment_generation,
        "environment_authority_id": environment_authority.authority_id,
        "environment_content_sha256": environment_authority.content_manifest_sha256,
        "selected_interpreter_relative_path": (
            environment_authority.selected_interpreter_relative_path
        ),
        "selected_interpreter_digest": environment_authority.selected_interpreter_digest,
        "code_members": [member_payload(member) for member in code_members],
        "worker_members": [member_payload(member) for member in worker_members],
        "standard_input_asset": (
            None
            if standard_input_asset is None
            else {
                "path": str(standard_input_asset[0]),
                "sha256": standard_input_asset[1],
                "asset_id": standard_input_asset[2],
            }
        ),
        "lookup_directories": [
            {"path": str(directory.path), "provenance": directory.provenance}
            for directory in lookup_directories
        ],
    }


def _executable_closure_v3_identity(
    *,
    code_manifest_identity: str,
    environment_authority: EnvironmentAuthorityV1,
    code_members: Sequence[RuntimeMember],
    worker_members: Sequence[RuntimeMember],
    standard_input_asset: Optional[tuple[Path, str, str]],
    lookup_directories: Sequence[RuntimeLookupDirectory],
) -> str:
    """Hash a v3 closure without the final model/request execution identity."""

    payload = _manifest_v3_payload(
        stable_id="closure-collection",
        work_id="closure-collection",
        execution_identity=stable_hash("executable-closure-v3-probe"),
        code_manifest_identity=code_manifest_identity,
        environment_authority=environment_authority,
        code_members=code_members,
        worker_members=worker_members,
        standard_input_asset=standard_input_asset,
        lookup_directories=lookup_directories,
    )
    payload.pop("manifest_version")
    payload.pop("stable_id")
    payload.pop("work_id")
    payload.pop("execution_identity")
    payload["closure_version"] = "menagerie.crawler.executable-closure.v2"
    return stable_hash(payload)


def compile_execution_read_manifest_v3(
    *,
    stable_id: str,
    work_id: str,
    execution_identity: str,
    code_manifest_identity: str,
    environment_authority: EnvironmentAuthorityV1,
    code_members: Sequence[RuntimeMember | tuple[Path, str, str]],
    worker_members: Sequence[RuntimeMember | tuple[Path, str, str]] = (),
    standard_input_asset: Optional[tuple[Path, str, str]] = None,
    lookup_directories: Sequence[RuntimeLookupDirectory] = (),
    verification_token: Optional[EnvironmentVerificationToken] = None,
) -> ExecutionReadManifestV3:
    """Compile the sole live four-part execution read capability.

    Parameters
    ----------
    stable_id, work_id, execution_identity, code_manifest_identity:
        Exact request and accepted implementation associations.
    environment_authority:
        Complete current content-sealed prefix authority.
    code_members, worker_members:
        Exact model and crawler/bootstrap files outside that prefix.
    standard_input_asset:
        Optional exact selected standard asset.
    lookup_directories:
        Non-authorizing import traversal scaffolds.
    verification_token:
        Optional cache-created proof shared across the current pass or spawn.

    Returns
    -------
    ExecutionReadManifestV3
        Current live execution capability.
    """

    stable_id = _require_nonempty_string(stable_id, "stable_id")
    work_id = _require_nonempty_string(work_id, "work_id")
    execution_identity = _require_hash(execution_identity, "execution_identity")
    code_manifest_identity = _require_hash(code_manifest_identity, "code_manifest_identity")
    verify_environment_authority(
        environment_authority,
        verification_token=verification_token,
    )
    normalized_code = tuple(
        sorted(
            (
                _verified_member(
                    _v3_member(value, provenance="accepted-model-code-manifest"),
                    allowed_kinds=_CODE_MEMBER_KINDS,
                    field="code_members",
                )
                for value in code_members
            ),
            key=lambda member: (str(member.path), member.kind, member.provenance),
        )
    )
    normalized_worker = tuple(
        sorted(
            (
                _verified_member(
                    _v3_member(value, provenance="crawler-worker-import-closure"),
                    allowed_kinds=_RUNTIME_MEMBER_KINDS,
                    field="worker_members",
                )
                for value in worker_members
            ),
            key=lambda member: (str(member.path), member.kind, member.provenance),
        )
    )
    member_paths = [member.path for member in (*normalized_code, *normalized_worker)]
    if len(member_paths) != len(set(member_paths)):
        raise AuthorityDerivationError("v3 manifest has duplicate exact member paths")
    if any(path.is_relative_to(environment_authority.prefix) for path in member_paths):
        raise AuthorityDerivationError(
            "environment descendants belong to the sealed unit, not exact-member partitions"
        )
    normalized_asset = _normalized_standard_asset(standard_input_asset, set(member_paths))
    normalized_lookup = _normalized_lookup_directories(lookup_directories)
    _validate_static_import_closure_v3(
        (*normalized_code, *normalized_worker),
        normalized_lookup,
        environment_authority.prefix,
        environment_authority.external_targets,
    )
    payload = _manifest_v3_payload(
        stable_id=stable_id,
        work_id=work_id,
        execution_identity=execution_identity,
        code_manifest_identity=code_manifest_identity,
        environment_authority=environment_authority,
        code_members=normalized_code,
        worker_members=normalized_worker,
        standard_input_asset=normalized_asset,
        lookup_directories=normalized_lookup,
    )
    return ExecutionReadManifestV3(
        manifest_version=EXECUTION_READ_MANIFEST_VERSION_V3,
        manifest_id=stable_hash(payload),
        stable_id=stable_id,
        work_id=work_id,
        execution_identity=execution_identity,
        code_manifest_identity=code_manifest_identity,
        environment_generation=environment_authority.environment_generation,
        code_members=normalized_code,
        worker_members=normalized_worker,
        environment_authority=environment_authority,
        standard_input_asset=normalized_asset,
        lookup_directories=normalized_lookup,
    )


def verify_execution_read_manifest_v3(
    manifest: ExecutionReadManifestV3,
    *,
    verification_token: Optional[EnvironmentVerificationToken] = None,
) -> None:
    """Reverify a live v3 manifest and interpreter with one shared observation."""

    if manifest.manifest_version != EXECUTION_READ_MANIFEST_VERSION_V3:
        raise AuthorityDerivationError("execution read manifest has the wrong v3 discriminator")
    if manifest.environment_generation != manifest.environment_authority.environment_generation:
        raise AuthorityDerivationError("manifest and environment authority generations differ")
    rebuilt = compile_execution_read_manifest_v3(
        stable_id=manifest.stable_id,
        work_id=manifest.work_id,
        execution_identity=manifest.execution_identity,
        code_manifest_identity=manifest.code_manifest_identity,
        environment_authority=manifest.environment_authority,
        code_members=manifest.code_members,
        worker_members=manifest.worker_members,
        standard_input_asset=manifest.standard_input_asset,
        lookup_directories=manifest.lookup_directories,
        verification_token=verification_token,
    )
    if rebuilt != manifest:
        raise AuthorityDerivationError("execution read manifest v3 is stale or rewritten")


def collect_executable_closure_v3(
    *,
    code_manifest_identity: str,
    environment_authority: EnvironmentAuthorityV1,
    code_members: Sequence[RuntimeMember | tuple[Path, str, str]],
    worker_members: Sequence[RuntimeMember | tuple[Path, str, str]],
    standard_input_asset: Optional[tuple[Path, str, str]] = None,
    lookup_directories: Sequence[RuntimeLookupDirectory] = (),
    verification_token: Optional[EnvironmentVerificationToken] = None,
) -> ExecutableClosureV3:
    """Collect a verified v3 closure before execution identity derivation.

    Parameters
    ----------
    code_manifest_identity, environment_authority, code_members, worker_members,
    standard_input_asset, lookup_directories:
        Exact four-part execution closure inputs.
    verification_token:
        Optional cache-created proof shared by the enclosing pass or spawn.

    Returns
    -------
    ExecutableClosureV3
        Verified pre-execution closure.
    """

    probe = compile_execution_read_manifest_v3(
        stable_id="closure-collection",
        work_id="closure-collection",
        execution_identity=stable_hash("executable-closure-v3-probe"),
        code_manifest_identity=code_manifest_identity,
        environment_authority=environment_authority,
        code_members=code_members,
        worker_members=worker_members,
        standard_input_asset=standard_input_asset,
        lookup_directories=lookup_directories,
        verification_token=verification_token,
    )
    return ExecutableClosureV3(
        identity=probe.closure_identity,
        code_manifest_identity=probe.code_manifest_identity,
        environment_generation=probe.environment_generation,
        code_members=probe.code_members,
        worker_members=probe.worker_members,
        environment_authority=probe.environment_authority,
        standard_input_asset=probe.standard_input_asset,
        lookup_directories=probe.lookup_directories,
    )


def compile_execution_read_manifest_v3_from_closure(
    closure: ExecutableClosureV3,
    *,
    stable_id: str,
    work_id: str,
    execution_identity: str,
    verification_token: Optional[EnvironmentVerificationToken] = None,
) -> ExecutionReadManifestV3:
    """Bind one pre-identity v3 closure to its final request associations.

    Parameters
    ----------
    closure:
        Previously verified executable closure.
    stable_id, work_id, execution_identity:
        Exact final request associations.
    verification_token:
        Optional cache-created proof shared by the enclosing pass or spawn.

    Returns
    -------
    ExecutionReadManifestV3
        Final request-bound execution capability.
    """

    manifest = compile_execution_read_manifest_v3(
        stable_id=stable_id,
        work_id=work_id,
        execution_identity=execution_identity,
        code_manifest_identity=closure.code_manifest_identity,
        environment_authority=closure.environment_authority,
        code_members=closure.code_members,
        worker_members=closure.worker_members,
        standard_input_asset=closure.standard_input_asset,
        lookup_directories=closure.lookup_directories,
        verification_token=verification_token,
    )
    if manifest.closure_identity != closure.identity:
        raise AuthorityDerivationError("execution closure v3 changed during final binding")
    return manifest


def environment_read_capability(
    manifest: ExecutionReadManifestV3,
    *,
    verification_token: Optional[EnvironmentVerificationToken] = None,
) -> EnvironmentReadCapability:
    """Verify and project the single v3 capability for every enforcement layer."""

    verify_execution_read_manifest_v3(
        manifest,
        verification_token=verification_token,
    )
    authority = manifest.environment_authority
    return EnvironmentReadCapability(
        manifest_id=manifest.manifest_id,
        closure_identity=manifest.closure_identity,
        exact_members=(*manifest.code_members, *manifest.worker_members),
        environment_prefix=authority.prefix,
        selected_interpreter=authority.selected_interpreter,
        startup_pth_paths=authority.startup_pth_paths,
        external_targets=authority.external_targets,
        standard_input_asset=manifest.standard_input_asset,
        lookup_directories=manifest.lookup_directories,
    )


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


def _validate_current_ledger_binding(record: Mapping[str, Any], field: str) -> None:
    """Validate the immutable ledger sequence and payload self-hash.

    Parameters
    ----------
    record:
        Canonical persisted attempt or gate row.
    field:
        Diagnostic record kind.

    Raises
    ------
    AuthorityDerivationError
        If the record lacks a positive sequence or exact payload digest.
    """

    ledger_seq = record.get("ledger_seq")
    if not isinstance(ledger_seq, int) or isinstance(ledger_seq, bool) or ledger_seq < 1:
        raise AuthorityDerivationError(f"current {field} proof requires a positive ledger_seq")
    digest = _require_hash(record.get("payload_sha256"), f"{field}.payload_sha256")
    if digest != payload_hash(record):
        raise AuthorityDerivationError(f"current {field} proof has an invalid payload self-hash")


def _validate_nonaward_parent_projection(attempt: Mapping[str, Any]) -> None:
    """Authenticate parent-owned proof for a non-awarding current attempt.

    Parameters
    ----------
    attempt:
        Current failed or observed attempt with no award receipt.

    Raises
    ------
    AuthorityDerivationError
        If the parent proof is incomplete, rewritten, or disagrees with projections.
    """

    parent = attempt.get("parent_attestation")
    if not isinstance(parent, Mapping):
        raise AuthorityDerivationError("current non-award attempt lacks parent proof material")
    _closed_mapping(parent, _PARENT_ATTESTATION_FIELDS, "parent_attestation")
    if parent.get("attestation_version") != _PARENT_ATTESTATION_VERSION:
        raise AuthorityDerivationError("parent attestation has the wrong protocol version")
    unhashed = {key: value for key, value in parent.items() if key != "attestation_sha256"}
    if parent.get("attestation_sha256") != stable_hash(unhashed):
        raise AuthorityDerivationError("parent attestation self hash is invalid")
    _require_nonempty_string(parent.get("request_nonce"), "parent_attestation.request_nonce")
    for name in ("request_sha256", "stdout_sha256", "stderr_sha256"):
        _require_hash(parent.get(name), f"parent_attestation.{name}")
    for name in ("completion_line_sha256", "named_raw_award_receipt_sha256"):
        value = parent.get(name)
        if value is not None:
            _require_hash(value, f"parent_attestation.{name}")
    if parent.get("named_raw_award_receipt_sha256") is not None:
        raise AuthorityDerivationError("non-award parent proof cannot name an award receipt")
    if any(
        attempt.get(name) is not None for name in ("raw_award_receipt", "raw_award_receipt_sha256")
    ):
        raise AuthorityDerivationError("non-success attempt cannot retain award proof material")
    supervisor = attempt.get("supervisor_observation")
    if not isinstance(supervisor, Mapping):
        raise AuthorityDerivationError("current non-award attempt lacks supervisor proof")
    comparisons = (
        ("exit_code", supervisor.get("exit_code"), parent.get("exit_code")),
        ("signal", supervisor.get("signal"), parent.get("signal")),
        ("peak_rss_bytes", supervisor.get("peak_rss_bytes"), parent.get("peak_rss_bytes")),
        (
            "stdout_sha256",
            supervisor.get("stdout_sha256") or hash_bytes(b""),
            parent.get("stdout_sha256"),
        ),
        (
            "stderr_sha256",
            supervisor.get("stderr_sha256") or hash_bytes(b""),
            parent.get("stderr_sha256"),
        ),
        ("started_at", attempt.get("started_at"), parent.get("started_at")),
        ("finished_at", attempt.get("finished_at"), parent.get("finished_at")),
    )
    mismatch = next((name for name, projected, proved in comparisons if projected != proved), None)
    if mismatch is not None:
        raise AuthorityDerivationError(f"non-award attempt contradicts parent proof at {mismatch}")
    result = attempt.get("result")
    error = attempt.get("error")
    if result == "failed":
        if (
            not isinstance(error, Mapping)
            or error.get("stage") != attempt.get("stage")
            or error.get("reason_code")
            not in FAILURE_REASON_CODES.get(str(attempt.get("stage")), frozenset())
        ):
            raise AuthorityDerivationError("current failed attempt lacks exact typed error proof")
        _require_hash(error.get("root_cause_fingerprint"), "attempt.error.root_cause_fingerprint")
    elif result != "observed" or error is not None:
        raise AuthorityDerivationError("current non-award attempt has an invalid result/error arm")


def load_current_attempt_proof(
    attempt: Mapping[str, Any], *, require_award: bool = False
) -> Optional[AttemptAuthority]:
    """Load one persisted attempt only when it carries current v3 proof authority.

    Parameters
    ----------
    attempt:
        Canonical persisted attempt row.
    require_award:
        Whether the consumer requires independently authenticated success authority.

    Returns
    -------
    AttemptAuthority | None
        Authenticated success projection, or ``None`` for a valid failed/observed proof.

    Raises
    ------
    AuthorityDerivationError
        If the row is legacy, unbound, malformed, or unsuitable for award authority.
    """

    if attempt.get("schema_version") != ATTEMPT_SCHEMA_VERSION_V3:
        raise AuthorityDerivationError(_LEGACY_PROOF_RULE)
    _validate_current_ledger_binding(attempt, "attempt")
    _require_nonempty_string(attempt.get("attempt_id"), "attempt.attempt_id")
    result = attempt.get("result")
    if result == "succeeded":
        raw = attempt.get("raw_award_receipt")
        parent = attempt.get("parent_attestation")
        if not isinstance(raw, Mapping) or not isinstance(parent, Mapping):
            raise AuthorityDerivationError("current success attempt lacks retained v3 raw proof")
        return derive_attempt_projection(raw, parent, candidate_attempt=attempt)
    if require_award:
        raise AuthorityDerivationError("accepted award attempt is not a v3 authenticated success")
    _validate_nonaward_parent_projection(attempt)
    return None


def authenticate_accepted_attempts(
    accepted_attempt_ids: Sequence[str],
    attempts: Sequence[Mapping[str, Any]],
    *,
    stable_id: Optional[str] = None,
    work_id: Optional[str] = None,
    execution_identity: Optional[str] = None,
) -> tuple[AttemptAuthority, ...]:
    """Independently authenticate every attempt counted toward a run award.

    Parameters
    ----------
    accepted_attempt_ids:
        Exact unique ordered award-counted attempt IDs, including confirmation slots.
    attempts:
        Canonical append-only attempt history.
    stable_id, work_id, execution_identity:
        Optional exact associations the integrator already resolved for the candidate model.

    Returns
    -------
    tuple[AttemptAuthority, ...]
        Verified projections in the caller's accepted-attempt order.

    Raises
    ------
    AuthorityDerivationError
        If an ID is duplicated/missing/ambiguous or any individual proof fails replay.
    """

    if isinstance(accepted_attempt_ids, (str, bytes)):
        raise AuthorityDerivationError("accepted attempt IDs must be a unique ordered sequence")
    ordered_ids = tuple(
        _require_nonempty_string(value, "accepted_attempt_ids[]") for value in accepted_attempt_ids
    )
    if len(ordered_ids) != len(set(ordered_ids)):
        raise AuthorityDerivationError("accepted attempt IDs must be unique")
    index: dict[str, list[Mapping[str, Any]]] = {}
    for attempt in attempts:
        attempt_id = attempt.get("attempt_id")
        if isinstance(attempt_id, str):
            index.setdefault(attempt_id, []).append(attempt)
    projections: list[AttemptAuthority] = []
    for attempt_id in ordered_ids:
        matches = index.get(attempt_id, [])
        if len(matches) != 1:
            raise AuthorityDerivationError(
                f"accepted attempt {attempt_id} is missing or ambiguous in the immutable ledger"
            )
        projection = load_current_attempt_proof(matches[0], require_award=True)
        assert projection is not None
        if stable_id is not None and projection.stable_id != stable_id:
            raise AuthorityDerivationError("accepted attempt belongs to another stable ID")
        if work_id is not None and projection.work_id != work_id:
            raise AuthorityDerivationError("accepted attempt belongs to another work generation")
        if execution_identity is not None and projection.execution_identity != execution_identity:
            raise AuthorityDerivationError("accepted attempt has a stale execution identity")
        projections.append(projection)
    return tuple(projections)


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
            reason=None,
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
    reason: Optional[str] = None
    if train_signature != eval_signature:
        comparison_state = "verified"
        classification = "structural"
    else:
        train_digest = train_observation.get("output_value_sha256")
        eval_digest = eval_observation.get("output_value_sha256")
        if not isinstance(train_digest, str) or not isinstance(eval_digest, str):
            comparison_state = "unverifiable"
            classification = "unverifiable"
            reason = "matching output signatures lack stable output value digests"
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
    if reason is not None:
        payload["reason"] = reason
    return ModeSummary(
        comparison_state=comparison_state,
        classification=classification,
        reason=reason,
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
    if summary.reason is not None:
        evidence["reason"] = summary.reason
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


def load_current_gate_proof(gate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Load one gate envelope only when its current v3 proof bindings replay.

    Parameters
    ----------
    gate:
        Canonical persisted gate envelope.

    Returns
    -------
    Mapping[str, Any]
        The exact input gate after proof validation.

    Raises
    ------
    AuthorityDerivationError
        If the gate is legacy or any immutable checker/gate binding is absent or stale.
    """

    if gate.get("schema_version") != GATE_SCHEMA_VERSION_V3:
        raise AuthorityDerivationError(_LEGACY_PROOF_RULE)
    _validate_current_ledger_binding(gate, "gate")
    _require_nonempty_string(gate.get("gate_id"), "gate.gate_id")
    for field in (
        "gate_identity",
        "result_envelope_sha256",
        "author_result_schema_identity",
        "dispatcher_identity",
    ):
        _require_hash(gate.get(field), f"gate.{field}")
    checker = gate.get("checker")
    if not isinstance(checker, Mapping):
        raise AuthorityDerivationError("current gate proof lacks its checker envelope")
    for field in ("provider", "model", "version"):
        _require_nonempty_string(checker.get(field), f"gate.checker.{field}")
    _require_hash(checker.get("prompt_sha256"), "gate.checker.prompt_sha256")
    items = gate.get("items")
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        raise AuthorityDerivationError("current gate proof has a malformed item sequence")
    if gate.get("batch_size") != len(items):
        raise AuthorityDerivationError("current gate proof has a partial item batch")
    result_payload = {
        key: value
        for key, value in gate.items()
        if key not in {"result_envelope_sha256", "payload_sha256", "ledger_seq"}
    }
    if gate.get("result_envelope_sha256") != stable_hash(result_payload):
        raise AuthorityDerivationError("current gate result-envelope self-hash is invalid")
    return gate


def resolve_exact_gate_item_membership(
    gates: Sequence[Mapping[str, Any]],
    *,
    accepted_gate_item: Mapping[str, Any],
    accepted_gate_item_sha256: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Resolve one exact item to its unique immutable current-v3 owning gate.

    Parameters
    ----------
    gates:
        Canonical append-only gate history.
    accepted_gate_item:
        Exact caller-held item bytes. Caller-shaped ``gate_id`` is never consulted.
    accepted_gate_item_sha256:
        Independently retained digest of the exact accepted item.

    Returns
    -------
    tuple[Mapping[str, Any], Mapping[str, Any]]
        Unique authenticated owning gate and its exact ledger item.

    Raises
    ------
    AuthorityDerivationError
        If the digest is stale, authority is legacy, or membership is missing/ambiguous.
    """

    item_digest = _require_hash(accepted_gate_item_sha256, "accepted_gate_item_sha256")
    if stable_hash(dict(accepted_gate_item)) != item_digest:
        raise AuthorityDerivationError("accepted gate item digest does not bind caller bytes")
    matches: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    legacy_match = False
    for gate in gates:
        items = gate.get("items")
        if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
            continue
        for item in items:
            if (
                isinstance(item, Mapping)
                and stable_hash(dict(item)) == item_digest
                and item == accepted_gate_item
            ):
                if gate.get("schema_version") != GATE_SCHEMA_VERSION_V3:
                    legacy_match = True
                    continue
                load_current_gate_proof(gate)
                matches.append((gate, item))
    if not matches and legacy_match:
        raise AuthorityDerivationError(_LEGACY_PROOF_RULE)
    if len(matches) != 1:
        raise AuthorityDerivationError(
            "accepted gate item has zero or multiple exact current-v3 ledger memberships"
        )
    return matches[0]


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


@dataclass
class _TerminalProofPipeline:
    """Run terminal-proof checks in their frozen diagnostic order.

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
        Ordered meaningful-mode iterable used for the complete per-mode map.
    proof_rule_identity:
        Versioned terminal-rule closure from the mandatory authority context.
    """

    stable_id: str
    work_id: str
    status_code: str
    attempts: Sequence[Mapping[str, Any]]
    gates: Sequence[Mapping[str, Any]]
    source_manifest: Sequence[Mapping[str, Any]]
    evidence_excerpts: Sequence[Mapping[str, Any]]
    source_resolution: Optional[Mapping[str, Any]]
    source_manifest_identity: Optional[str]
    evidence_identity: Optional[str]
    license_identity: Optional[str]
    meaningful_modes: Iterable[str]
    proof_rule_identity: str
    relevant: tuple[Mapping[str, Any], ...] = dataclass_field(init=False, default=())
    per_mode: tuple[tuple[str, str], ...] = dataclass_field(init=False, default=())
    gate_id: DependencyValue = dataclass_field(init=False, default=DependencyState.NOT_APPLICABLE)
    source_ids: tuple[str, ...] = dataclass_field(init=False, default=())
    evidence_ids: tuple[str, ...] = dataclass_field(init=False, default=())
    failure_stage: DependencyValue = dataclass_field(
        init=False, default=DependencyState.NOT_APPLICABLE
    )
    reason_code: DependencyValue = dataclass_field(
        init=False, default=DependencyState.NOT_APPLICABLE
    )
    root_cause: DependencyValue = dataclass_field(
        init=False, default=DependencyState.NOT_APPLICABLE
    )
    platform_claim: DependencyValue = dataclass_field(
        init=False, default=DependencyState.NOT_APPLICABLE
    )
    decisive_ids: tuple[str, ...] = dataclass_field(init=False, default=())
    gate_proof_identity: DependencyValue = dataclass_field(
        init=False, default=DependencyState.NOT_APPLICABLE
    )
    resolved_reference_identity: DependencyValue = dataclass_field(
        init=False, default=DependencyState.NOT_APPLICABLE
    )

    def run(self) -> TerminalProof:
        """Execute the ordered proof pipeline and return its immutable projection.

        Returns
        -------
        TerminalProof
            Immutable reducer-derived terminal authority.
        """

        self._initialize()
        self._derive_status_proof()
        return self._build_proof()

    def _initialize(self) -> None:
        """Validate roots and derive shared attempt projections in original order."""

        self.stable_id = _require_nonempty_string(self.stable_id, "stable_id")
        self.work_id = _require_nonempty_string(self.work_id, "work_id")
        self.proof_rule_identity = _require_nonempty_string(
            self.proof_rule_identity, "proof_rule_identity"
        )
        self.relevant = tuple(
            attempt
            for attempt in self.attempts
            if attempt.get("stable_id") == self.stable_id and attempt.get("work_id") == self.work_id
        )
        self.per_mode = derive_per_mode_attempt_ids(
            self.attempts,
            stable_id=self.stable_id,
            work_id=self.work_id,
            meaningful_modes=self.meaningful_modes,
        )

    def _derive_status_proof(self) -> None:
        """Dispatch exactly one closed status rule without changing precedence."""

        if self.status_code == "runs":
            self._derive_runs()
        elif self.status_code.startswith("failed:"):
            self._derive_failure()
        elif self.status_code.startswith("deferred:"):
            self._derive_deferral()
        elif self.status_code.startswith("skipped:"):
            self._derive_skip()
        else:
            raise AuthorityDerivationError("status code has no closed terminal proof rule")

    def _derive_runs(self) -> None:
        """Validate complete successful per-mode authority for a runs status."""

        expected_modes = tuple(dict.fromkeys(str(mode) for mode in self.meaningful_modes))
        if {mode for mode, _attempt_id in self.per_mode} != set(expected_modes):
            raise AuthorityDerivationError("runs proof does not cover every meaningful mode")
        attempt_index = {str(attempt.get("attempt_id")): attempt for attempt in self.relevant}
        for mode, attempt_id in self.per_mode:
            attempt = attempt_index.get(attempt_id)
            if (
                attempt is None
                or attempt.get("mode") != mode
                or attempt.get("result") != "succeeded"
            ):
                raise AuthorityDerivationError("runs proof contains a non-successful mode attempt")
            _authenticated_observation(attempt)
        self.decisive_ids = tuple(attempt_id for _mode, attempt_id in self.per_mode)

    def _derive_failure(self) -> None:
        """Validate the exact gate or attempt proof for a failed status."""

        stage = self.status_code.removeprefix("failed:")
        self.failure_stage = stage
        if stage in {"accuracy-gate", "fidelity"}:
            self.gate_id, self.reason_code, self.root_cause = _derive_gate_failure(
                self.gates,
                stable_id=self.stable_id,
                work_id=self.work_id,
                stage=stage,
            )
            return
        candidates = []
        for attempt in self.relevant:
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
                f"{self.status_code} lacks an exact same-stage failed attempt"
            )
        decisive = max(candidates, key=_attempt_order)
        error = decisive["error"]
        assert isinstance(error, Mapping)
        self.reason_code = str(error["reason_code"])
        if self.reason_code not in FAILURE_REASON_CODES.get(stage, frozenset()):
            raise AuthorityDerivationError("failed attempt reason is not closed for its stage")
        self.root_cause = str(error["root_cause_fingerprint"])
        self.decisive_ids = (str(decisive["attempt_id"]),)
        if stage == "source" and self.reason_code == "missing-mandatory-link":
            if decisive.get("stage") != "source":
                raise AuthorityDerivationError(
                    "missing-mandatory-link must bind an exact source-stage attempt"
                )

    def _derive_deferral(self) -> None:
        """Validate exact gate, source, evidence, and probe authority for deferral."""

        predicate = self.status_code.removeprefix("deferred:")
        if predicate not in {"needs-cuda", "needs-x86"}:
            raise AuthorityDerivationError("unknown platform deferral status")
        gate, item, disposition = _terminal_gate(
            self.gates,
            stable_id=self.stable_id,
            work_id=self.work_id,
            predicate=predicate,
        )
        self.gate_id = str(gate["gate_id"])
        self.gate_proof_identity = stable_hash({"gate_id": self.gate_id, "item": item})
        self.platform_claim = predicate
        _validate_terminal_gate_identities(
            disposition,
            source_manifest_identity=self.source_manifest_identity,
            evidence_identity=self.evidence_identity,
            license_identity=self.license_identity,
        )
        self.source_ids, self.evidence_ids = _validate_terminal_references(
            disposition,
            predicate=predicate,
            source_manifest=self.source_manifest,
            evidence_excerpts=self.evidence_excerpts,
        )
        self.resolved_reference_identity = stable_hash(
            {
                "sources": [
                    source
                    for source in self.source_manifest
                    if str(source.get("source_id")) in self.source_ids
                ],
                "evidence": [
                    excerpt
                    for excerpt in self.evidence_excerpts
                    if str(excerpt.get("evidence_id")) in self.evidence_ids
                ],
            }
        )
        probe_ids: list[str] = []
        for attempt in self.relevant:
            defer = attempt.get("defer_evidence")
            if not isinstance(defer, Mapping) or defer.get("target_status") != self.status_code:
                continue
            if set(str(value) for value in defer.get("source_ids", ())) != set(self.source_ids):
                raise AuthorityDerivationError("deferral attempt source set is not gate-exact")
            named_probes = tuple(str(value) for value in defer.get("probe_attempt_ids", ()))
            for probe_id in named_probes:
                probe = next(
                    (
                        candidate
                        for candidate in self.relevant
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
                        "deferral probe lacks a structured positive same-work capability "
                        "observation"
                    )
                probe_ids.append(probe_id)
        self.decisive_ids = tuple(dict.fromkeys(probe_ids))

    def _derive_skip(self) -> None:
        """Validate exact R5 gate, reference, and typed skip-predicate authority."""

        predicate = self.status_code.removeprefix("skipped:")
        if predicate not in {
            "insufficient-description",
            "no-description",
            "not-a-real-NN",
        }:
            raise AuthorityDerivationError("unknown epistemic skip status")
        gate, item, disposition = _terminal_gate(
            self.gates,
            stable_id=self.stable_id,
            work_id=self.work_id,
            predicate=predicate,
        )
        if (
            item.get("rung_check", {}).get("selected_rung") != "R5_SKIP"
            or item.get("rung_check", {}).get("verdict") != "accurate"
        ):
            raise AuthorityDerivationError("skip gate does not prove an accurate R5 decision")
        self.gate_id = str(gate["gate_id"])
        self.gate_proof_identity = stable_hash({"gate_id": self.gate_id, "item": item})
        _validate_terminal_gate_identities(
            disposition,
            source_manifest_identity=self.source_manifest_identity,
            evidence_identity=self.evidence_identity,
            license_identity=self.license_identity,
        )
        self.source_ids, self.evidence_ids = _validate_terminal_references(
            disposition,
            predicate=predicate,
            source_manifest=self.source_manifest,
            evidence_excerpts=self.evidence_excerpts,
        )
        if self.source_resolution is None:
            raise AuthorityDerivationError(
                "skip proof requires exact accepted source-resolution facts"
            )
        _validate_skip_predicate(
            predicate,
            self.source_resolution,
            self.evidence_excerpts,
            self.evidence_ids,
        )
        self.resolved_reference_identity = stable_hash(
            {
                "source_resolution": self.source_resolution,
                "sources": [
                    source
                    for source in self.source_manifest
                    if str(source.get("source_id")) in self.source_ids
                ],
                "evidence": [
                    excerpt
                    for excerpt in self.evidence_excerpts
                    if str(excerpt.get("evidence_id")) in self.evidence_ids
                ],
            }
        )

    def _build_proof(self) -> TerminalProof:
        """Derive the final observation hash and immutable proof payload."""

        terminal_observation = derive_terminal_observation(
            self.attempts,
            stable_id=self.stable_id,
            work_id=self.work_id,
        )
        proof_payload = {
            "proof_rule_identity": self.proof_rule_identity,
            "stable_id": self.stable_id,
            "work_id": self.work_id,
            "status_code": self.status_code,
            "decisive_attempt_ids": list(self.decisive_ids),
            "gate_id": self.gate_id,
            "source_ids": list(self.source_ids),
            "evidence_ids": list(self.evidence_ids),
            "failure_stage": self.failure_stage,
            "reason_code": self.reason_code,
            "root_cause_fingerprint": self.root_cause,
            "platform_claim": self.platform_claim,
            "per_mode_attempt_ids": [list(value) for value in self.per_mode],
            "terminal_observation_sha256": stable_hash(terminal_observation),
            "gate_proof_identity": self.gate_proof_identity,
            "resolved_reference_identity": self.resolved_reference_identity,
        }
        return TerminalProof(
            proof_id=stable_hash(proof_payload),
            proof_rule_identity=self.proof_rule_identity,
            stable_id=self.stable_id,
            work_id=self.work_id,
            status_code=self.status_code,
            decisive_attempt_ids=self.decisive_ids,
            gate_id=self.gate_id,
            source_ids=self.source_ids,
            evidence_ids=self.evidence_ids,
            failure_stage=self.failure_stage,
            reason_code=self.reason_code,
            root_cause_fingerprint=self.root_cause,
            platform_claim=self.platform_claim,
            per_mode_attempt_ids=self.per_mode,
            terminal_observation_sha256=str(proof_payload["terminal_observation_sha256"]),
        )


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

    return _TerminalProofPipeline(
        stable_id=stable_id,
        work_id=work_id,
        status_code=status_code,
        attempts=attempts,
        gates=gates,
        source_manifest=source_manifest,
        evidence_excerpts=evidence_excerpts,
        source_resolution=source_resolution,
        source_manifest_identity=source_manifest_identity,
        evidence_identity=evidence_identity,
        license_identity=license_identity,
        meaningful_modes=meaningful_modes,
        proof_rule_identity=proof_rule_identity,
    ).run()


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
