"""Canonical reducer and sole model/attempt/gate ledger writer."""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Iterable, Mapping, Optional, Sequence, Union

from menagerie.crawler.artifact_transactions import (
    ArtifactEventKind,
    ArtifactEventLedger,
    StagedArtifact,
    append_artifact_authorization,
    derive_artifact_claims,
    derive_publication_authorization_id,
)
from menagerie.crawler.authority import (
    AuthorityContext,
    AuthorityDerivationError,
    DependencyCurrencyProjection,
    DependencyState,
    DependencyVector,
    DependencyValue,
    FamilyAuthority,
    PublicationAuthorization,
    TerminalProof,
    authenticate_accepted_attempts,
    derive_attempt_projection,
    dependency_vector_projection,
    derive_dependency_vector,
    derive_family_authority,
    derive_mode_summary,
    derive_per_mode_run,
    derive_terminal_observation,
    derive_terminal_proof,
    family_authority_projection,
    mode_summary_projection,
    load_current_attempt_proof,
    load_current_gate_proof,
    resolve_exact_gate_item_membership,
    validate_currency,
)

from menagerie.crawler.constants import (
    ATTEMPT_SCHEMA_VERSION_V3,
    CHECKER_PROMPT_NAME,
    FAILURE_REASON_CODES,
    GATE_SCHEMA_VERSION_V3,
    MODEL_SCHEMA_VERSION_V3,
    TERMINAL_STATUS_CODES,
)
from menagerie.crawler.env_lifecycle import (
    EnvironmentExactnessError,
    EnvironmentProbeError,
    materialized_environment_generation,
    parse_probe_receipt_bytes,
    parse_resolved_export,
)
from menagerie.crawler.envs import EnvironmentSpecError, load_environment_registry
from menagerie.crawler.family_templates import (
    FamilyTemplateError,
    family_representative_is_usable,
    family_variant_currency_error,
    validate_size_variant,
    validate_size_variant_derivation,
)
from menagerie.crawler.identity import canonical_json_bytes, hash_bytes, stable_hash
from menagerie.crawler.intake import (
    IntakeError,
    legacy_requires_fidelity_audit,
    load_intake_snapshot,
)
from menagerie.crawler.metadata import (
    MetadataValidationError,
    input_signature_matches_contract,
    recompute_accepted_identities,
    validate_authored_facts_for_write,
)
from menagerie.crawler.models import AppendResult, JsonObject, LedgerPaths
from menagerie.crawler.licenses import LicenseDecision, RedistributionClass
from menagerie.crawler.mirrors import MirrorStore
from menagerie.crawler.recordio import JsonlLedger, LedgerConflictError, scan_jsonl
from menagerie.crawler.state import _select_current as _select_current_by_parent
from menagerie.crawler.standard_inputs import ASSET_ROOT


class ReductionError(ValueError):
    """Raised when a proposed canonical fact violates reducer invariants."""


@dataclass(frozen=True)
class ColdForwardPolicy:
    """Reducer-owned cold-forward admission policy.

    Parameters
    ----------
    confirmation_policy:
        Canonical policy label persisted in a run record.
    required_cold_forwards:
        Exact number of independent forwards required for every meaningful mode.
    """

    confirmation_policy: str
    required_cold_forwards: int


def cold_forward_policy(stable_id: str, rung: object) -> ColdForwardPolicy:
    """Derive the authoritative deterministic cold-forward policy for one run.

    Parameters
    ----------
    stable_id:
        Durable model identity used for reproducible canary membership.
    rung:
        Canonical source rung.

    Returns
    -------
    ColdForwardPolicy
        Two cold forwards for R3/R4 and for the deterministic two-percent R1/R2
        canary; one cold forward for the remaining R1/R2 population.
    """

    if str(rung) in {"R3_PORT", "R4_REIMPLEMENT"}:
        return ColdForwardPolicy("two-cold-r3-r4", 2)
    canary_digest = hashlib.sha256(f"{stable_id}mechanical-canary-v1".encode("utf-8")).digest()
    if int.from_bytes(canary_digest, "big") % 100 < 2:
        return ColdForwardPolicy("mechanical-canary", 2)
    return ColdForwardPolicy("single-mechanical", 1)


class _ReplayLedger:
    """Minimal in-memory ledger used to replay reducer admission without writes."""

    def __init__(self, records: Iterable[Mapping[str, Any]]) -> None:
        """Retain immutable records in an append-compatible replay ledger.

        Replay never mutates canonical input rows, so sharing their mappings
        avoids a full-ledger deepcopy for every projected model.
        """

        self.records: list[Mapping[str, Any]] = list(records)

    def append(self, record: Mapping[str, Any]) -> AppendResult:
        """Record one already hash-validated replay candidate in memory."""

        identity = next(
            (
                (field, record.get(field))
                for field in ("attempt_id", "gate_id", "record_revision")
                if record.get(field) is not None
            ),
            None,
        )
        if identity is not None:
            field, value = identity
            existing = next(
                (candidate for candidate in self.records if candidate.get(field) == value),
                None,
            )
            if existing is not None:
                return AppendResult(deepcopy(dict(existing)), appended=False)
        copied = deepcopy(dict(record))
        self.records.append(copied)
        return AppendResult(copied, appended=True)


class _ReplayArtifactLedger:
    """Read-only artifact-event facade used during semantic projection replay."""

    def __init__(self, records: Iterable[Mapping[str, Any]]) -> None:
        """Retain immutable artifact events without acquiring another writer lock."""

        self.events = tuple(records)


_FACT_ROOTS = (
    "identity",
    "taxonomy",
    "external_metadata",
    "website",
    "people_and_origin",
    "dates",
    "citation",
    "licenses",
    "source_resolution",
    "evidence",
    "implementation",
    "input_contract",
    "modes",
    "fidelity",
)

_MODALITY_STANDARD_ASSETS = {
    "audio": "audio.csv",
    "computer-vision": "image.ppm",
    "image": "image.ppm",
    "language": "text.txt",
    "nlp": "text.txt",
    "recsys": "tabular.csv",
    "speech": "audio.csv",
    "tabular": "tabular.csv",
    "text": "text.txt",
    "video": "image.ppm",
    "vision": "image.ppm",
}
_WORKER_COMPLETION_PREFIX = "MENAGERIE_WORKER_COMPLETION_V1 "
_DIAGNOSTIC_REDACTION_MARKER = "externally-controlled-text-v1"
_HASH_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
_SAFE_EXCERPT_DISPOSITIONS = frozenset(
    {"public-compatible", "public-domain", "short-excerpt-committed"}
)
_EXTERNALLY_CONTROLLED_RECORD_FIELDS = frozenset(
    {
        "message",
        "mode_error",
        "observed_response",
        "receipt_error",
        "response_excerpt",
        "stderr_tail",
        "stdout_tail",
        "traceback",
    }
)


def _is_safe_diagnostic_redaction(value: Any) -> bool:
    """Return whether a C-07 field contains no raw externally controlled text.

    Parameters
    ----------
    value:
        Candidate externally controlled field value.

    Returns
    -------
    bool
        ``True`` for empty values or the checkpoint-approved sidecar reference shape.
    """

    if value is None or value == "":
        return True
    if not isinstance(value, Mapping):
        return False
    if value.get("license_disposition") in _SAFE_EXCERPT_DISPOSITIONS:
        return bool(
            _HASH_PATTERN.fullmatch(str(value.get("text_sha256", "")))
            and isinstance(value.get("locator"), str)
        )
    required = {"redaction", "content_sha256", "local_path", "diagnostic_key"}
    allowed = required | {"stream_sha256"}
    local_path = str(value.get("local_path", ""))
    pure_local = PurePosixPath(local_path)
    stream_sha256 = value.get("stream_sha256")
    return bool(
        set(value) <= allowed
        and required <= set(value)
        and value.get("redaction") == _DIAGNOSTIC_REDACTION_MARKER
        and _HASH_PATTERN.fullmatch(str(value.get("content_sha256", "")))
        and pure_local.parts[:2] == (".crawl-local", "diagnostics")
        and ".." not in pure_local.parts
        and pure_local.suffix == ".json"
        and re.fullmatch(r"\$[A-Za-z0-9_.\[\]-]+", str(value.get("diagnostic_key", "")))
        and (stream_sha256 is None or _HASH_PATTERN.fullmatch(str(stream_sha256)) is not None)
    )


def _validate_c07_diagnostic_fields(record: Mapping[str, Any], *, model: bool) -> None:
    """Reject raw C-07 diagnostic values before a canonical append.

    Parameters
    ----------
    record:
        Proposed attempt or model record.
    model:
        Whether model-only failed-detail and human-review fields are in scope.

    Raises
    ------
    ReductionError
        If any protected field contains raw text or a forged redaction reference.
    """

    findings: list[str] = []

    def visit(value: Any, location: str = "$") -> None:
        """Collect unsafe protected values recursively."""

        if isinstance(value, Mapping):
            for key, nested in value.items():
                nested_location = f"{location}.{key}"
                failed_status_detail = bool(
                    model
                    and key == "detail"
                    and location.endswith(".status")
                    and value.get("kind") == "failed"
                )
                human_review_reason = bool(
                    model and key == "reason" and location.endswith(".status.human_review")
                )
                if (
                    key in _EXTERNALLY_CONTROLLED_RECORD_FIELDS
                    or failed_status_detail
                    or human_review_reason
                ) and not _is_safe_diagnostic_redaction(nested):
                    findings.append(nested_location)
                visit(nested, nested_location)
        elif isinstance(value, list):
            for index, nested in enumerate(value):
                visit(nested, f"{location}[{index}]")

    visit(record)
    if findings:
        raise ReductionError(
            f"canonical record contains unredacted externally controlled text: fields={findings}"
        )


def expected_standard_asset(modality: object) -> Optional[dict[str, str]]:
    """Return the exact standard asset selected by input materialization.

    Parameters
    ----------
    modality:
        Accepted modality string or sequence.

    Returns
    -------
    dict[str, str] | None
        Asset path, byte digest, and content-addressed identifier, or ``None``
        when input materialization must use deterministic random fallback.
    """

    values = (modality,) if isinstance(modality, str) else modality
    if not isinstance(values, (list, tuple)):
        return None
    normalized = tuple(str(value).strip().lower() for value in values)
    selected_name: Optional[str] = None
    for candidates in (
        {"vision", "image", "computer-vision"},
        {"language", "text", "nlp"},
        {"audio", "speech"},
        {"video"},
        {"tabular", "recsys"},
    ):
        selected = next((value for value in normalized if value in candidates), None)
        if selected is not None:
            selected_name = _MODALITY_STANDARD_ASSETS[selected]
            break
    if selected_name is None:
        return None
    path = ASSET_ROOT / selected_name
    try:
        digest = hash_bytes(path.read_bytes())
    except OSError as exc:
        raise ReductionError(f"selected standard input asset is unavailable: {path}") from exc
    return {
        "path": str(path.resolve()),
        "sha256": digest,
        "asset_id": f"standard:{selected_name}:{digest}",
    }


def output_signature_error(signature: object) -> Optional[str]:
    """Return why an output signature is not a complete pytree contract.

    Parameters
    ----------
    signature:
        Candidate ``output_signature`` mapping.

    Returns
    -------
    str | None
        Stable validation error, or ``None`` for a complete signature.
    """

    if not isinstance(signature, Mapping):
        return "signature is not an object"
    tree = signature.get("tree")
    leaves = signature.get("leaves")
    if tree is None or not isinstance(leaves, list) or not leaves:
        return "signature tree and leaves must be non-empty"
    referenced: list[int] = []

    def visit(node: object) -> bool:
        """Validate one tree node and collect its leaf indices."""

        if not isinstance(node, Mapping) or not node:
            return False
        if set(node) == {"leaf"} and isinstance(node.get("leaf"), int):
            index = node.get("leaf")
            if isinstance(index, bool):
                return False
            assert isinstance(index, int)
            referenced.append(index)
            return 0 <= index < len(leaves)
        if set(node) in ({"tuple"}, {"list"}) and isinstance(next(iter(node.values())), list):
            children = next(iter(node.values()))
            assert isinstance(children, list)
            return all(visit(child) for child in children)
        return all(isinstance(key, str) and visit(child) for key, child in node.items())

    if not visit(tree) or sorted(referenced) != list(range(len(leaves))):
        return "signature tree does not reference each leaf exactly once"
    for leaf in leaves:
        if not isinstance(leaf, Mapping):
            return "signature leaf is not an object"
        path = leaf.get("path")
        kind = leaf.get("kind")
        python_type = leaf.get("python_type")
        if (
            not isinstance(path, str)
            or not path
            or kind not in {"tensor", "python"}
            or not isinstance(python_type, str)
            or not python_type
        ):
            return "signature leaf lacks path, kind, or python type"
        if kind == "tensor":
            shape = leaf.get("shape")
            dtype = leaf.get("dtype")
            if (
                not isinstance(shape, list)
                or any(
                    not isinstance(dimension, int) or isinstance(dimension, bool) or dimension < 0
                    for dimension in shape
                )
                or not isinstance(dtype, str)
                or not dtype
            ):
                return "tensor signature leaf lacks a concrete shape or dtype"
    return None


def _select_current(records: Iterable[Mapping[str, Any]]) -> dict[str, JsonObject]:
    """Validate public supersession lineage and select current revisions.

    Parameters
    ----------
    records:
        Canonical model revisions in append order.

    Returns
    -------
    dict[str, dict[str, Any]]
        Current revisions selected after exact public-lineage validation.
    """

    materialized = tuple(records)
    for record in materialized:
        parent = record.get("parent_revision")
        supersedes = record.get("status", {}).get("supersedes_revision")
        if parent is None and supersedes is not None:
            raise ReductionError("a first model revision cannot supersede another revision")
        if parent is not None and supersedes != parent:
            raise ReductionError(
                "persisted status.supersedes_revision does not match parent_revision"
            )
    return _select_current_by_parent(materialized)


def _records_root(ledgers: LedgerPaths) -> Path:
    """Return the canonical records root for sibling operational evidence.

    Parameters
    ----------
    ledgers:
        Canonical model/attempt/gate ledger paths.

    Returns
    -------
    pathlib.Path
        Records root containing the durable operational directory.
    """

    parent = ledgers.models.resolve().parent
    return parent.parent if parent.name == "models" else parent


def _revision_work_ids(
    revision: Mapping[str, Any],
    attempts: Iterable[Mapping[str, Any]],
    gates: Iterable[Mapping[str, Any]],
) -> frozenset[str]:
    """Collect durable work identities referenced by one model revision.

    Parameters
    ----------
    revision:
        Candidate or persisted model revision.
    attempts, gates:
        Canonical execution and checker evidence.

    Returns
    -------
    frozenset[str]
        Work identities bound by the revision's referenced evidence.
    """

    stable_id = revision.get("stable_id")
    attempt_ids = set(revision.get("status", {}).get("attempt_ids", [])) | set(
        revision.get("execution", {}).get("accepted_attempt_ids", [])
    )
    work_ids = {
        str(attempt.get("work_id"))
        for attempt in attempts
        if attempt.get("attempt_id") in attempt_ids and attempt.get("work_id") is not None
    }
    gate_ids = {
        revision.get("accuracy_gate", {}).get("gate_id"),
        revision.get("fidelity", {}).get("gate_id"),
    } - {None}
    for gate in gates:
        if gate.get("gate_id") not in gate_ids:
            continue
        for item in gate.get("items", []):
            if item.get("stable_id") == stable_id:
                for field in ("work_id", "campaign_root_work_id"):
                    if item.get(field) is not None:
                        work_ids.add(str(item[field]))
    untrusted = revision.get("untrusted_attempt")
    if isinstance(untrusted, Mapping):
        proposal = untrusted.get("proposal")
        if isinstance(proposal, Mapping) and proposal.get("work_id") is not None:
            work_ids.add(str(proposal["work_id"]))
    return frozenset(work_ids)


def _validate_persisted_requeue_lineage(
    records: Iterable[Mapping[str, Any]],
    ledgers: LedgerPaths,
    attempts: Iterable[Mapping[str, Any]],
    gates: Iterable[Mapping[str, Any]],
) -> None:
    """Validate every explicit grant introduction against canonical durable proof.

    Parameters
    ----------
    records:
        Persisted revisions plus an optional append candidate.
    ledgers:
        Canonical ledger paths used to locate operational evidence.
    attempts, gates:
        Canonical evidence carrying new-work identities.

    Raises
    ------
    ReductionError
        If a grant, parent binding, generation, or new-work identity is inconsistent.
    """

    materialized = tuple(records)
    if not any(record.get("budget", {}).get("explicit_grants", []) for record in materialized):
        return
    operational_root = _records_root(ledgers) / "operational"
    grant_rows = scan_jsonl(operational_root / "requeue-grants.jsonl", validate=False)
    event_rows = scan_jsonl(operational_root / "events.jsonl")
    grants = {str(grant.get("grant_id")): grant for grant in grant_rows}
    consumptions = {
        str(event.get("details", {}).get("grant_id")): event
        for event in event_rows
        if event.get("event_kind") == "requeue-grant-consumed"
    }
    prior_grants: dict[str, set[str]] = {}
    for revision in materialized:
        stable_id = str(revision.get("stable_id"))
        inherited = prior_grants.get(stable_id, set())
        current_grants = set(revision.get("budget", {}).get("explicit_grants", []))
        if not inherited <= current_grants:
            raise ReductionError("explicit requeue grants cannot be removed from a lineage")
        introduced = current_grants - inherited
        if len(introduced) > 1:
            raise ReductionError("one model revision cannot consume multiple requeue grants")
        for grant_id in introduced:
            grant = grants.get(grant_id)
            event = consumptions.get(grant_id)
            if grant is None or event is None:
                raise ReductionError("explicit requeue grant lacks canonical durable proof")
            if "new_work_generation" in grant:
                expected_grant_id = stable_hash(
                    {
                        "generation": grant.get("new_work_generation"),
                        "stable_id": grant.get("stable_id"),
                        "stage": grant.get("stage"),
                        "reason": grant.get("reason"),
                        "attempts": grant.get("attempts"),
                        "granted_by": grant.get("granted_by"),
                    }
                )
            else:
                expected_grant_id = stable_hash(
                    {
                        "stable_id": grant.get("stable_id"),
                        "stage": grant.get("stage"),
                        "reason": grant.get("reason"),
                        "grant": grant.get("attempts"),
                    }
                )
            if expected_grant_id != grant_id:
                raise ReductionError("explicit requeue grant identity is invalid")
            details = event.get("details", {})
            if (
                grant.get("stable_id") != stable_id
                or details.get("stable_id") != stable_id
                or any(
                    details.get(field) != grant.get(field)
                    for field in ("stage", "reason", "attempts")
                )
            ):
                raise ReductionError("explicit requeue grant facts conflict with consumption")
            parent_revision = revision.get("parent_revision")
            if parent_revision is None or details.get("source_record_revision") != parent_revision:
                raise ReductionError("explicit requeue grant is bound to the wrong parent revision")
            generation = details.get("new_work_generation")
            if not isinstance(generation, int) or isinstance(generation, bool) or generation < 1:
                raise ReductionError("explicit requeue grant generation is invalid")
            expected_work_id = stable_hash(
                {
                    "stable_id": stable_id,
                    "grant_id": grant_id,
                    "parent_revision": parent_revision,
                    "generation": generation,
                }
            )
            if details.get("new_work_id") != expected_work_id:
                raise ReductionError("explicit requeue grant new-work identity is invalid")
            if expected_work_id not in _revision_work_ids(revision, attempts, gates):
                raise ReductionError(
                    "superseding revision does not bind the granted new-work identity"
                )
        prior_grants[stable_id] = current_grants


def _model_facts(model: Mapping[str, Any]) -> Mapping[str, Any]:
    """Extract the complete accepted fact tree from a canonical model revision."""

    return {field: model.get(field) for field in _FACT_ROOTS}


def _authored_facts_for_vet(model: Mapping[str, Any]) -> Mapping[str, Any]:
    """Recover authored facts before a fidelity gate projected its verdict fields."""

    facts = deepcopy(dict(_model_facts(model)))
    fidelity = facts.get("fidelity")
    if isinstance(fidelity, dict) and fidelity.get("gate_id") is not None:
        fidelity.update(
            {
                "current": False,
                "deviations": [],
                "gate_id": None,
                "permanent_scar": False,
                "verdict": None,
            }
        )
    return facts


def _checker_prompt_hash() -> str:
    """Hash the exact current checker prompt bytes used for freshness."""

    path = Path(__file__).with_name("prompts") / f"{CHECKER_PROMPT_NAME}.txt"
    try:
        return hash_bytes(path.read_bytes())
    except OSError as exc:
        raise ReductionError(f"checker prompt bytes are unavailable: {exc}") from exc


def _gate_item_fingerprint(item: Mapping[str, Any]) -> str:
    """Return the canonical checker root-cause fingerprint for one item."""

    return stable_hash(
        {
            "verdict": item.get("verdict"),
            "integrity": item.get("integrity"),
            "field_checks": item.get("field_checks"),
            "rung_check": item.get("rung_check"),
            "fidelity": item.get("fidelity"),
            "unsupported_claims": item.get("unsupported_claims"),
            "required_repairs": item.get("required_repairs"),
        }
    )


def _record_attempt_ids(record: Mapping[str, Any]) -> tuple[str, ...]:
    """Return every attempt reference carried by a terminal or run record."""

    referenced: list[str] = []
    blocks = (
        record.get("status", {}).get("attempt_ids", []),
        record.get("execution", {}).get("accepted_attempt_ids", []),
        record.get("observed", {}).get("measurement_attempt_ids", []),
        [
            value.get("attempt_id")
            for value in record.get("modes", {}).get("per_mode_run", {}).values()
            if isinstance(value, Mapping)
        ],
    )
    for values in blocks:
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            continue
        for value in values:
            if value is not None and str(value) not in referenced:
                referenced.append(str(value))
    return tuple(referenced)


class CanonicalReducer:
    """Exclusive canonical writer enforcing parentage, gates, runs, and statuses.

    Parameters
    ----------
    ledgers:
        Paths to the three canonical append-only ledgers.
    context:
        Mandatory active intake, contract, environment, and policy authority.
    """

    def __init__(
        self,
        ledgers: LedgerPaths,
        context: AuthorityContext,
    ) -> None:
        """Acquire all canonical writer locks and load current facts.

        Parameters
        ----------
        ledgers:
            Paths to canonical ledgers.
        context:
            Mandatory active authority shared by admission and every projection.
        """

        self.ledger_paths = ledgers
        self.context = context
        self.intake_ids = frozenset(context.intake_by_stable_id)
        self.intake_variant_bindings = {
            stable_id: (
                str(
                    binding.get("representative_stable_id", binding.get("family_representative_id"))
                ),
                str(binding.get("variant_token", binding.get("variant", ""))),
            )
            for stable_id, binding in context.family_bindings.items()
            if isinstance(binding, Mapping)
            and binding.get("binding_state") != "ordinary"
            and binding.get("representative_stable_id", binding.get("family_representative_id"))
            not in {None, stable_id}
        }
        unknown_bindings = set(context.family_bindings) - self.intake_ids
        if unknown_bindings:
            raise ReductionError(
                f"family variant bindings exist outside intake: {sorted(unknown_bindings)}"
            )
        opened: list[JsonlLedger] = []
        try:
            self._models = JsonlLedger(ledgers.models, MODEL_SCHEMA_VERSION_V3)
            opened.append(self._models)
            self._attempts = JsonlLedger(ledgers.attempts, ATTEMPT_SCHEMA_VERSION_V3)
            opened.append(self._attempts)
            self._gates = JsonlLedger(ledgers.gates, GATE_SCHEMA_VERSION_V3)
            opened.append(self._gates)
            self._artifacts = ArtifactEventLedger(ledgers.artifacts)
        except Exception:
            artifact_ledger = getattr(self, "_artifacts", None)
            if artifact_ledger is not None:
                artifact_ledger.close()
            for ledger in reversed(opened):
                ledger.close()
            raise
        try:
            for attempt in self._attempts.records:
                _validate_c07_diagnostic_fields(attempt, model=False)
            for model in self._models.records:
                _validate_c07_diagnostic_fields(model, model=True)
        except ReductionError:
            self.close()
            raise
        self._current = _select_current(self._models.records)
        self._projection_cache: Optional[tuple[str, DependencyCurrencyProjection]] = None
        self._attempt_index_cache: Optional[Mapping[str, Mapping[str, Any]]] = None
        self._gate_index_cache: Optional[Mapping[str, Mapping[str, Any]]] = None
        _validate_persisted_requeue_lineage(
            self._models.records,
            self.ledger_paths,
            self._attempts.records,
            self._gates.records,
        )
        unknown = set(self._current) - self.intake_ids
        if unknown:
            self.close()
            raise ReductionError(f"model revisions exist outside intake: {sorted(unknown)}")

    def __enter__(self) -> "CanonicalReducer":
        """Return this exclusive reducer.

        Returns
        -------
        CanonicalReducer
            This reducer.
        """

        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Release every canonical writer lock.

        Parameters
        ----------
        exc_type, exc_value, traceback:
            Context-manager exception state.
        """

        self.close()

    def close(self) -> None:
        """Release every canonical writer lock idempotently."""

        for ledger_name in ("_artifacts", "_gates", "_attempts", "_models"):
            ledger = getattr(self, ledger_name, None)
            if ledger is not None:
                ledger.close()

    @property
    def current_records(self) -> Mapping[str, JsonObject]:
        """Return defensive copies of materialized current revisions.

        Returns
        -------
        Mapping[str, dict[str, Any]]
            Stable-ID keyed current model revisions.
        """

        token = _dependency_bytes_token(self.ledger_paths)
        cached = self._projection_cache
        if cached is None or cached[0] != token:
            projection = project_dependency_current(
                self.ledger_paths,
                context=self.context,
                model_records=self._models.records,
                attempt_records=self._attempts.records,
                gate_records=self._gates.records,
                artifact_records=self._artifacts.events,
            )
            self._projection_cache = (token, projection)
        else:
            projection = cached[1]
        return deepcopy(projection.current_records)

    @property
    def artifact_ledger(self) -> ArtifactEventLedger:
        """Return the reducer-owned artifact writer held in canonical lock order."""

        return self._artifacts

    def _attempt_index(self) -> Mapping[str, Mapping[str, Any]]:
        """Return the shared immutable attempt-ID index for reducer validation."""

        cached = getattr(self, "_attempt_index_cache", None)
        if cached is None:
            cached = {str(record.get("attempt_id")): record for record in self._attempts.records}
            self._attempt_index_cache = cached
        return cached

    def update_context(self, context: AuthorityContext) -> None:
        """Replace only runtime-current environment axes without changing intake roots.

        Parameters
        ----------
        context:
            Refreshed authority after exact environment materialization.

        Raises
        ------
        ReductionError
            If a caller attempts to change the active intake or contract roots
            during one locked reducer lifetime.
        """

        stable_axes = (
            "active_intake_snapshot_id",
            "active_intake_snapshot_sha256",
            "intake_by_stable_id",
            "family_bindings",
            "author_prompt_identity",
            "author_model_identity",
            "author_schema_identity",
            "author_dispatcher_identity",
            "checker_prompt_identity",
            "checker_model_identity",
            "checker_schema_identity",
            "reducer_policy_identity",
            "runner_policy_identity",
            "terminal_policy_identity",
            "publication_policy_identity",
        )
        if any(getattr(context, axis) != getattr(self.context, axis) for axis in stable_axes):
            raise ReductionError("locked reducer context changed a non-environment authority axis")
        self.context = context
        self._projection_cache = None

    def _gate_index(self) -> Mapping[str, Mapping[str, Any]]:
        """Return the shared immutable gate-ID index for reducer validation."""

        cached = getattr(self, "_gate_index_cache", None)
        if cached is None:
            cached = {str(record.get("gate_id")): record for record in self._gates.records}
            self._gate_index_cache = cached
        return cached

    def _validation_current_records(self) -> Mapping[str, Mapping[str, Any]]:
        """Return the dependency-current map selected for this admission pass."""

        projected = getattr(self, "_validation_current", None)
        if isinstance(projected, Mapping):
            return projected
        return self._current

    def _current_attempt_records(self) -> tuple[Mapping[str, Any], ...]:
        """Return only attempts carrying authenticated current-v3 proof authority.

        Returns
        -------
        tuple[Mapping[str, Any], ...]
            Current authenticated attempts in immutable ledger order. Legacy rows
            remain readable audit history but cannot participate in model authority.

        Raises
        ------
        ReductionError
            If a purported current-v3 row fails its proof replay.
        """

        current: list[Mapping[str, Any]] = []
        for attempt in self._attempts.records:
            if attempt.get("schema_version") != ATTEMPT_SCHEMA_VERSION_V3:
                continue
            try:
                load_current_attempt_proof(attempt)
            except AuthorityDerivationError as exc:
                raise ReductionError(str(exc)) from exc
            current.append(attempt)
        return tuple(current)

    def _current_gate_records(self) -> tuple[Mapping[str, Any], ...]:
        """Return only gates carrying authenticated current-v3 proof authority.

        Returns
        -------
        tuple[Mapping[str, Any], ...]
            Current authenticated gates in immutable ledger order.

        Raises
        ------
        ReductionError
            If a purported current-v3 gate fails its proof replay.
        """

        current: list[Mapping[str, Any]] = []
        for gate in self._gates.records:
            if gate.get("schema_version") != GATE_SCHEMA_VERSION_V3:
                continue
            try:
                load_current_gate_proof(gate)
            except AuthorityDerivationError as exc:
                raise ReductionError(str(exc)) from exc
            current.append(gate)
        return tuple(current)

    def append_attempt(self, attempt: Mapping[str, Any]) -> AppendResult:
        """Append one immutable attempt after parent/work validation.

        Parameters
        ----------
        attempt:
            Attempt payload; ledger sequence and payload hash may be omitted.

        Returns
        -------
        AppendResult
            Idempotent append result.
        """

        _validate_c07_diagnostic_fields(attempt, model=False)
        if attempt.get("schema_version") != ATTEMPT_SCHEMA_VERSION_V3:
            raise ReductionError("attempt appends require the current v3 authority contract")
        stable_id = attempt.get("stable_id")
        if stable_id is not None and stable_id not in self.intake_ids:
            raise ReductionError(f"attempt stable_id is outside intake: {stable_id}")
        parent_id = attempt.get("parent_attempt_id")
        by_id = self._attempt_index()
        if parent_id is not None:
            parent = by_id.get(parent_id)
            if parent is None:
                raise ReductionError(f"missing parent attempt {parent_id}")
            if parent["work_id"] != attempt.get("work_id"):
                raise ReductionError("parent attempt belongs to a different work identity")
            if int(attempt.get("attempt_no", 0)) <= int(parent["attempt_no"]):
                raise ReductionError("attempt_no must increase after its parent")
        error = attempt.get("error")
        if attempt.get("result") == "failed":
            if not isinstance(error, Mapping):
                raise ReductionError("failed attempt requires a structured error")
            stage = error.get("stage")
            if stage != attempt.get("stage"):
                raise ReductionError("attempt and error stages must match")
            if error.get("reason_code") not in FAILURE_REASON_CODES.get(str(stage), frozenset()):
                raise ReductionError("attempt error reason is not allowed for its stage")
        elif error is not None:
            raise ReductionError("non-failed attempt cannot carry an error")
        environment = attempt.get("environment")
        identities = attempt.get("identities", {})
        receipt = attempt.get("worker_receipt", {})
        observation = attempt.get("supervisor_observation", {})
        if environment is None:
            if identities.get("environment") is not None or identities.get("execution") is not None:
                raise ReductionError(
                    "an unobserved environment cannot carry environment/execution identities"
                )
            if receipt.get("present"):
                raise ReductionError("a present worker receipt requires observed environment facts")
            if (
                observation.get("stdout_sha256") is not None
                or observation.get("stderr_sha256") is not None
            ):
                raise ReductionError("unobserved process streams cannot carry artifact digests")
        elif not all(
            environment.get(field)
            for field in (
                "lock_sha256",
                "resolved_export_sha256",
                "packages_manifest_sha256",
            )
        ):
            raise ReductionError("observed environment facts require exact artifact digests")
        elif observation.get("stdout_sha256") is None or observation.get("stderr_sha256") is None:
            raise ReductionError("an observed subprocess requires exact stream-byte digests")
        if environment is not None:
            self._validate_environment_generation(attempt)
        if attempt.get("result") == "succeeded" and (
            receipt.get("observed_recipe_revision") != identities.get("recipe")
            or "observed_adapter_sha256" not in receipt
        ):
            raise ReductionError("successful worker receipt lacks current observed recipe bindings")
        if attempt.get("result") == "succeeded":
            raw_receipt = attempt.get("raw_award_receipt")
            parent_attestation = attempt.get("parent_attestation")
            if not isinstance(raw_receipt, Mapping) or not isinstance(parent_attestation, Mapping):
                raise ReductionError("successful attempt lacks retained v3 raw proof")
            try:
                derive_attempt_projection(
                    raw_receipt,
                    parent_attestation,
                    candidate_attempt=attempt,
                )
            except AuthorityDerivationError as exc:
                raise ReductionError(str(exc)) from exc
        elif any(
            attempt.get(field) is not None
            for field in ("raw_award_receipt", "raw_award_receipt_sha256")
        ):
            raise ReductionError("non-success attempt cannot retain award-eligible raw proof")
        result = self._attempts.append(attempt)
        if result.appended:
            self._attempt_index_cache = None
        return result

    def _validate_environment_generation(self, attempt: Mapping[str, Any]) -> None:
        """Recompute production environment identity from committed exact artifacts.

        Parameters
        ----------
        attempt:
            Candidate attempt carrying observed environment and generation facts.

        Raises
        ------
        ReductionError
            If a canonical campaign attempt is not backed by exact committed bytes.
        """

        records_root = self.ledger_paths.models.parent.parent
        canonical_root = records_root.parent
        production_layout = (
            records_root.name == "records"
            and canonical_root.name == "crawler"
            and canonical_root.parent.name == "menagerie"
        )
        if not production_layout:
            return
        environment = attempt.get("environment")
        identities = attempt.get("identities")
        if not isinstance(environment, Mapping) or not isinstance(identities, Mapping):
            raise ReductionError("environment generation facts are malformed")
        family = environment.get("family")
        target = environment.get("target")
        if not isinstance(family, str) or not isinstance(target, str):
            raise ReductionError("environment generation lacks family/target")
        try:
            registry = load_environment_registry(canonical_root / "envs", target=target)
            intent = registry.intents[family]
            lock_bytes = intent.lock.lock_path.read_bytes()
            export_bytes = intent.lock.export_path.read_bytes()
            export_hash = intent.lock.export_hash_path.read_text(encoding="utf-8").strip()
            receipt_path = intent.lock.lock_path.with_name(f"{target}.probes.json")
            probe_results = parse_probe_receipt_bytes(intent.probes, receipt_path.read_bytes())
            package_bytes = parse_resolved_export(export_bytes)
            generation = materialized_environment_generation(
                intent,
                lock_bytes=lock_bytes,
                export_bytes=export_bytes,
                package_bytes=package_bytes,
                python_version=str(environment.get("python")),
                compiler_identity=str(environment.get("compiler_identity")),
                sdk_identity=str(environment.get("sdk_identity")),
                probe_results=probe_results,
            )
        except (
            KeyError,
            OSError,
            UnicodeError,
            EnvironmentSpecError,
            EnvironmentExactnessError,
            EnvironmentProbeError,
        ) as exc:
            raise ReductionError(
                "observed environment lacks exact committed generation artifacts"
            ) from exc
        if (
            environment.get("lock_sha256") != hash_bytes(lock_bytes)
            or environment.get("resolved_export_sha256") != hash_bytes(export_bytes)
            or environment.get("packages_manifest_sha256") != hash_bytes(package_bytes)
            or export_hash != hash_bytes(export_bytes)
            or identities.get("environment") != generation
        ):
            raise ReductionError(
                "observed environment generation is stale or self-attested: "
                f"expected={generation}, observed={identities.get('environment')}, "
                f"lock={environment.get('lock_sha256') == hash_bytes(lock_bytes)}, "
                f"export={environment.get('resolved_export_sha256') == hash_bytes(export_bytes)}, "
                f"packages={environment.get('packages_manifest_sha256') == hash_bytes(package_bytes)}"
            )

    def append_gate(self, gate: Mapping[str, Any]) -> AppendResult:
        """Append one immutable checker gate after intake/integrity checks.

        Parameters
        ----------
        gate:
            Gate payload; ledger sequence and payload hash may be omitted.

        Returns
        -------
        AppendResult
            Idempotent append result.
        """

        if gate.get("schema_version") != GATE_SCHEMA_VERSION_V3:
            raise ReductionError("gate appends require the current v3 authority contract")
        items = gate.get("items", [])
        stable_ids = [item.get("stable_id") for item in items]
        if gate.get("batch_size") != len(items):
            raise ReductionError("gate batch_size must equal the number of items")
        if len(stable_ids) != len(set(stable_ids)):
            raise ReductionError("gate items must have unique stable IDs")
        if gate.get("checker", {}).get("prompt_sha256") != _checker_prompt_hash():
            raise ReductionError("checker gate does not bind the current prompt bytes")
        for item in items:
            if item.get("stable_id") not in self.intake_ids:
                raise ReductionError(
                    f"gate item stable_id is outside intake: {item.get('stable_id')}"
                )
            if not isinstance(item.get("campaign_root_work_id"), str) or not item.get(
                "campaign_root_work_id"
            ):
                raise ReductionError("gate item has no campaign/root-work lineage identity")
            verdict = item.get("verdict")
            integrity = item.get("integrity", {}).get("verdict")
            field_verdicts = [check.get("verdict") for check in item.get("field_checks", [])]
            if verdict == "accurate" and (
                integrity != "accurate" or any(value != "accurate" for value in field_verdicts)
            ):
                raise ReductionError(
                    "an accurate gate item requires accurate integrity/field checks"
                )
        result = self._gates.append(gate)
        if result.appended:
            self._gate_index_cache = None
        return result

    def _artifact_authority_inputs(self, model: Mapping[str, Any]) -> JsonObject:
        """Resolve and validate one model's exact append-only artifact authority.

        Parameters
        ----------
        model:
            Candidate v3 model revision.

        Returns
        -------
        dict[str, Any]
            Latest event plus immutable reconstruction inputs when applicable.

        Raises
        ------
        ReductionError
            If the model names no exact transaction state or immutable bytes.
        """

        authority = model.get("artifact_authority")
        if not isinstance(authority, Mapping):
            raise ReductionError("model lacks its mandatory artifact authority block")
        if authority.get("state") == DependencyState.NOT_APPLICABLE.value:
            expected = {
                "state": DependencyState.NOT_APPLICABLE.value,
                "transaction_id": DependencyState.NOT_APPLICABLE.value,
                "committed_event_id": DependencyState.NOT_APPLICABLE.value,
                "authorization_id": DependencyState.NOT_APPLICABLE.value,
                "reconstruction_sha256": DependencyState.NOT_APPLICABLE.value,
                "claim_ids": [],
            }
            if dict(authority) != expected:
                raise ReductionError("not-applicable artifact authority is not closed")
            return {
                "event": None,
                "document": None,
                "transaction_id": DependencyState.NOT_APPLICABLE,
                "claim_ids": (),
            }
        transaction_id = authority.get("transaction_id")
        events = [
            event
            for event in self._artifacts.events
            if event.get("transaction_id") == transaction_id
        ]
        if not events:
            raise ReductionError("model artifact transaction is absent from the append-only ledger")
        latest = events[-1]
        claims = tuple(sorted(str(value.get("claim_id")) for value in latest.get("claims", [])))
        reconstruction = latest.get("reconstruction")
        reconstruction_sha256: object = DependencyState.NOT_APPLICABLE.value
        document: Optional[JsonObject] = None
        if isinstance(reconstruction, Mapping):
            reconstruction_sha256 = reconstruction.get("sha256")
            path_value = reconstruction.get("path")
            if not isinstance(path_value, str):
                raise ReductionError("artifact reconstruction lacks its immutable path")
            canonical_root = _production_canonical_root(self.ledger_paths)
            if canonical_root is not None:
                repository_root = canonical_root.parents[1]
                path = (repository_root / path_value).resolve()
                try:
                    raw = path.read_bytes()
                    parsed = json.loads(raw)
                except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ReductionError("artifact reconstruction bytes are unavailable") from exc
                if hash_bytes(raw) != reconstruction_sha256 or not isinstance(parsed, dict):
                    raise ReductionError("artifact reconstruction digest changed")
                document = parsed
        expected_authority: JsonObject = {
            "state": latest.get("event_kind"),
            "transaction_id": transaction_id,
            "committed_event_id": latest.get("artifact_event_id"),
            "authorization_id": latest.get("authorization_id")
            or DependencyState.PENDING_UNTRUSTED.value,
            "reconstruction_sha256": reconstruction_sha256,
            "claim_ids": list(claims),
        }
        if dict(authority) != expected_authority:
            raise ReductionError("model artifact authority contradicts the latest ledger event")
        return {
            "event": latest,
            "document": document,
            "transaction_id": str(transaction_id),
            "claim_ids": claims,
        }

    def _model_work_id(self, model: Mapping[str, Any], artifact_inputs: Mapping[str, Any]) -> str:
        """Resolve one exact work generation from all model authority references.

        Parameters
        ----------
        model, artifact_inputs:
            Candidate model and its resolved artifact transaction.

        Returns
        -------
        str
            Unique work identity shared by attempts, gates, proposal, and custody.
        """

        stable_id = str(model.get("stable_id"))
        work_ids: set[str] = set()
        referenced_attempts = set(_record_attempt_ids(model))
        for attempt in self._attempts.records:
            if (
                attempt.get("attempt_id") in referenced_attempts
                and attempt.get("stable_id") == stable_id
                and isinstance(attempt.get("work_id"), str)
            ):
                work_ids.add(str(attempt["work_id"]))
        event = artifact_inputs.get("event")
        if isinstance(event, Mapping) and isinstance(event.get("work_id"), str):
            work_ids.add(str(event["work_id"]))
        for block_name in ("accuracy_gate", "fidelity"):
            gate_id = model.get(block_name, {}).get("gate_id")
            found = self._gate_item(gate_id, stable_id)
            if found is not None and isinstance(found[1].get("work_id"), str):
                work_ids.add(str(found[1]["work_id"]))
        retained = model.get("untrusted_attempt")
        proposal = retained.get("proposal") if isinstance(retained, Mapping) else None
        if isinstance(proposal, Mapping) and isinstance(proposal.get("work_id"), str):
            work_ids.add(str(proposal["work_id"]))
        if len(work_ids) != 1:
            raise ReductionError(
                f"model authority does not resolve one work identity: {sorted(work_ids)}"
            )
        return next(iter(work_ids))

    def _derive_model_authority(
        self, model: Mapping[str, Any]
    ) -> tuple[TerminalProof, FamilyAuthority, JsonObject, JsonObject]:
        """Derive terminal, family, dependency, mode, and artifact projections.

        Parameters
        ----------
        model:
            Candidate current v3 model revision.

        Returns
        -------
        tuple[TerminalProof, FamilyAuthority, dict[str, Any], dict[str, Any]]
            Exact proof objects plus the derived vector and artifact inputs.
        """

        artifact_inputs = self._artifact_authority_inputs(model)
        stable_id = str(model.get("stable_id"))
        work_id = self._model_work_id(model, artifact_inputs)
        meaningful_modes = tuple(
            str(value) for value in model.get("modes", {}).get("meaningful_modes", ())
        )
        event = artifact_inputs.get("event")
        document = artifact_inputs.get("document")
        source_manifest: Sequence[Mapping[str, Any]] = ()
        evidence_excerpts: Sequence[Mapping[str, Any]] = ()
        source_resolution: Optional[Mapping[str, Any]] = None
        source_manifest_identity: Optional[str] = None
        evidence_identity: Optional[str] = None
        license_identity: Optional[str] = None
        if isinstance(event, Mapping):
            source_manifest_identity = str(event.get("source_manifest_identity"))
        if isinstance(document, Mapping):
            manifest = document.get("source_manifest")
            raw_sources = manifest.get("sources") if isinstance(manifest, Mapping) else None
            if isinstance(raw_sources, list):
                source_manifest = tuple(
                    value for value in raw_sources if isinstance(value, Mapping)
                )
            author_result = document.get("author_result")
            payload = author_result.get("payload") if isinstance(author_result, Mapping) else None
            if isinstance(payload, Mapping):
                evidence_identity = payload.get("evidence_identity")
                license_identity = payload.get("license_identity")
            proposal = document.get("proposal")
            facts = proposal.get("proposed_facts") if isinstance(proposal, Mapping) else None
            if isinstance(facts, Mapping):
                source_resolution_value = facts.get("source_resolution")
                evidence = facts.get("evidence")
                if isinstance(source_resolution_value, Mapping):
                    source_resolution = source_resolution_value
                excerpts = evidence.get("excerpts") if isinstance(evidence, Mapping) else None
                if isinstance(excerpts, list):
                    evidence_excerpts = tuple(
                        value for value in excerpts if isinstance(value, Mapping)
                    )
                if evidence_identity is None and isinstance(evidence, Mapping):
                    evidence_identity = evidence.get("evidence_identity")
                if license_identity is None:
                    licenses = facts.get("licenses")
                    if isinstance(licenses, Mapping):
                        license_identity = stable_hash(licenses)
        if source_resolution is None:
            candidate_resolution = model.get("source_resolution")
            if isinstance(candidate_resolution, Mapping):
                source_resolution = candidate_resolution
        if not source_manifest and isinstance(source_resolution, Mapping):
            candidate_sources = source_resolution.get("sources")
            if isinstance(candidate_sources, list):
                source_manifest = tuple(
                    value for value in candidate_sources if isinstance(value, Mapping)
                )
        if not evidence_excerpts:
            evidence = model.get("evidence")
            excerpts = evidence.get("excerpts") if isinstance(evidence, Mapping) else None
            if isinstance(excerpts, list):
                evidence_excerpts = tuple(value for value in excerpts if isinstance(value, Mapping))
            if evidence_identity is None and isinstance(evidence, Mapping):
                evidence_identity = evidence.get("evidence_identity")
        if license_identity is None:
            for gate in self._gates.records:
                for item in gate.get("items", []):
                    if item.get("stable_id") != stable_id or item.get("work_id") != work_id:
                        continue
                    terminal = item.get("terminal_disposition")
                    if isinstance(terminal, Mapping):
                        license_identity = terminal.get("license_identity")
                        evidence_identity = terminal.get("evidence_identity", evidence_identity)
        try:
            current_attempts = self._current_attempt_records()
            current_gates = self._current_gate_records()
            terminal_proof = derive_terminal_proof(
                stable_id,
                work_id,
                str(model.get("status", {}).get("code")),
                attempts=current_attempts,
                gates=current_gates,
                source_manifest=source_manifest,
                evidence_excerpts=evidence_excerpts,
                source_resolution=source_resolution,
                source_manifest_identity=source_manifest_identity,
                evidence_identity=(
                    str(evidence_identity) if evidence_identity is not None else None
                ),
                license_identity=(str(license_identity) if license_identity is not None else None),
                meaningful_modes=meaningful_modes,
                proof_rule_identity=self.context.terminal_policy_identity,
            )
            representative: Optional[Mapping[str, Any]] = None
            binding = self.context.family_bindings.get(stable_id)
            if isinstance(binding, Mapping):
                representative_id = binding.get(
                    "representative_stable_id", binding.get("family_representative_id")
                )
                if isinstance(representative_id, str) and representative_id != stable_id:
                    representative = self._validation_current_records().get(representative_id)
            family_authority = derive_family_authority(
                self.context,
                stable_id,
                representative_record=representative,
            )
        except AuthorityDerivationError as exc:
            raise ReductionError(str(exc)) from exc

        relevant = tuple(
            attempt
            for attempt in current_attempts
            if attempt.get("stable_id") == stable_id and attempt.get("work_id") == work_id
        )
        expected_per_mode = derive_per_mode_run(
            relevant,
            stable_id=stable_id,
            work_id=work_id,
            meaningful_modes=meaningful_modes,
        )
        if model.get("modes", {}).get("per_mode_run") != expected_per_mode:
            raise ReductionError("model per_mode_run contradicts reducer-derived attempt authority")
        by_mode = {
            str(attempt.get("mode")): attempt
            for attempt in relevant
            if attempt.get("attempt_id")
            in {value["attempt_id"] for value in expected_per_mode.values()}
            and attempt.get("result") == "succeeded"
        }
        try:
            mode_projection = mode_summary_projection(
                derive_mode_summary(by_mode.get("train"), by_mode.get("eval"))
            )
        except AuthorityDerivationError as exc:
            raise ReductionError(str(exc)) from exc
        for field, expected in mode_projection.items():
            if model.get("modes", {}).get(field) != expected:
                raise ReductionError(f"modes.{field} contradicts reducer-derived mode authority")
        if model.get("status", {}).get("kind") != "runs":
            expected_observed = derive_terminal_observation(
                current_attempts,
                stable_id=stable_id,
                work_id=work_id,
            )
            if model.get("observed") != expected_observed:
                raise ReductionError("terminal observed facts contradict reducer-derived authority")

        checker_gate: DependencyValue = terminal_proof.gate_id
        if checker_gate == DependencyState.NOT_APPLICABLE:
            accuracy_gate = model.get("accuracy_gate", {}).get("gate_id")
            checker_gate = (
                accuracy_gate
                if isinstance(accuracy_gate, str) and accuracy_gate
                else DependencyState.PENDING_UNTRUSTED
                if model.get("authored_metadata_state") == "pending"
                else DependencyState.NOT_APPLICABLE
            )
        environment_id: Optional[str] = None
        env_generation = model.get("execution", {}).get("env_generation")
        proof_attempt_ids = set(terminal_proof.decisive_attempt_ids) | {
            attempt_id for _mode, attempt_id in terminal_proof.per_mode_attempt_ids
        }
        proof_environment_generations = {
            attempt.get("identities", {}).get("environment")
            for attempt in current_attempts
            if attempt.get("attempt_id") in proof_attempt_ids
            and isinstance(attempt.get("identities", {}).get("environment"), str)
        }
        if len(proof_environment_generations) > 1:
            raise ReductionError("terminal proof spans multiple environment generations")
        proof_environment_generation = next(iter(proof_environment_generations), None)
        if proof_environment_generation is not None:
            if env_generation != proof_environment_generation:
                raise ReductionError("model environment generation contradicts its attempt proof")
            environment_id = next(
                (
                    name
                    for name, generation in self.context.environment_generations.items()
                    if generation == proof_environment_generation
                ),
                None,
            )
            if environment_id is None:
                raise ReductionError("model environment generation is absent from active context")
        elif model.get("status", {}).get("kind") == "runs":
            raise ReductionError("runs proof has no environment generation")
        accepted_ids = (
            model.get("execution", {}).get("accepted_attempt_ids", [])
            if model.get("status", {}).get("kind") == "runs"
            else model.get("status", {}).get("attempt_ids", [])
        )
        proposal_identity: DependencyValue = DependencyState.NOT_APPLICABLE
        author_result_identity: DependencyValue = DependencyState.NOT_APPLICABLE
        if isinstance(event, Mapping):
            event_proposal = event.get("proposal_id")
            event_result = event.get("author_result_id")
            if isinstance(event_proposal, str) and event_proposal:
                proposal_identity = event_proposal
            if isinstance(event_result, str) and event_result:
                author_result_identity = event_result
        candidate_recipe = model.get("implementation", {}).get("recipe_revision")
        recipe_revision: DependencyValue = (
            candidate_recipe
            if isinstance(candidate_recipe, str)
            else DependencyState.NOT_APPLICABLE
        )
        try:
            vector = derive_dependency_vector(
                self.context,
                stable_id=stable_id,
                terminal_proof=terminal_proof,
                source_manifest_identity=(
                    source_manifest_identity or DependencyState.NOT_APPLICABLE
                ),
                proposal_identity=proposal_identity,
                author_result_identity=author_result_identity,
                checker_gate_identity=checker_gate,
                recipe_revision=recipe_revision,
                environment_id=environment_id,
                accepted_attempt_ids=accepted_ids,
                artifact_transaction_id=artifact_inputs["transaction_id"],
                artifact_claim_ids=artifact_inputs["claim_ids"],
                family_authority=family_authority,
            )
        except AuthorityDerivationError as exc:
            raise ReductionError(str(exc)) from exc
        vector_projection = dependency_vector_projection(vector)
        return terminal_proof, family_authority, vector_projection, artifact_inputs

    def _validate_derived_model_authority(self, model: Mapping[str, Any]) -> None:
        """Require exact reducer-derived proof, family, and dependency projections.

        Parameters
        ----------
        model:
            Candidate v3 model revision.

        Raises
        ------
        ReductionError
            If any copied driver projection differs from replayed authority.
        """

        terminal_proof, family_authority, vector, _artifact_inputs = self._derive_model_authority(
            model
        )
        if model.get("family_authority") != family_authority_projection(family_authority):
            raise ReductionError("model family authority contradicts trusted intake")
        if model.get("dependency_vector") != vector:
            raise ReductionError("model dependency vector contradicts reducer derivation")
        status = model.get("status")
        if not isinstance(status, Mapping):
            raise ReductionError("model status is malformed")
        if status.get("code") != terminal_proof.status_code:
            raise ReductionError("model status code contradicts its terminal proof")
        if terminal_proof.status_code.startswith("failed:") and (
            status.get("stage") != terminal_proof.failure_stage
            or status.get("reason_code") != terminal_proof.reason_code
            or status.get("root_cause_fingerprint") != terminal_proof.root_cause_fingerprint
        ):
            raise ReductionError("failed status fields contradict its decisive proof")
        stale_reason = validate_currency(
            self.context,
            model,
            terminal_proof=terminal_proof,
            family_authority=family_authority,
        )
        if stale_reason is not None:
            raise ReductionError(stale_reason)

    def prepare_model(self, model: Mapping[str, Any]) -> JsonObject:
        """Project one driver candidate exclusively from reducer-owned authority.

        Parameters
        ----------
        model:
            Structurally assembled candidate without copied authority fields.

        Returns
        -------
        dict[str, Any]
            Candidate with exact artifact, mode, family, and dependency projections.
        """

        candidate = deepcopy(dict(model))
        stable_id = str(candidate.get("stable_id"))
        events = [event for event in self._artifacts.events if event.get("stable_id") == stable_id]
        referenced_attempt_ids = {
            str(value) for value in candidate.get("status", {}).get("attempt_ids", [])
        }
        referenced_work_ids = {
            str(attempt.get("work_id"))
            for attempt in self._attempts.records
            if attempt.get("attempt_id") in referenced_attempt_ids
            and isinstance(attempt.get("work_id"), str)
        }
        untrusted = candidate.get("untrusted_attempt")
        proposal = untrusted.get("proposal") if isinstance(untrusted, Mapping) else None
        if isinstance(proposal, Mapping) and isinstance(proposal.get("work_id"), str):
            referenced_work_ids.add(str(proposal["work_id"]))
        if len(referenced_work_ids) == 1:
            exact_work_id = next(iter(referenced_work_ids))
            exact_events = [event for event in events if event.get("work_id") == exact_work_id]
            if exact_events:
                events = exact_events
        if not events:
            candidate["artifact_authority"] = {
                "state": DependencyState.NOT_APPLICABLE.value,
                "transaction_id": DependencyState.NOT_APPLICABLE.value,
                "committed_event_id": DependencyState.NOT_APPLICABLE.value,
                "authorization_id": DependencyState.NOT_APPLICABLE.value,
                "reconstruction_sha256": DependencyState.NOT_APPLICABLE.value,
                "claim_ids": [],
            }
        else:
            latest = events[-1]
            reconstruction = latest.get("reconstruction")
            candidate["artifact_authority"] = {
                "state": latest["event_kind"],
                "transaction_id": latest["transaction_id"],
                "committed_event_id": latest["artifact_event_id"],
                "authorization_id": latest.get("authorization_id")
                or DependencyState.PENDING_UNTRUSTED.value,
                "reconstruction_sha256": (
                    reconstruction.get("sha256")
                    if isinstance(reconstruction, Mapping)
                    else DependencyState.NOT_APPLICABLE.value
                ),
                "claim_ids": sorted(str(value["claim_id"]) for value in latest.get("claims", [])),
            }
        artifact_inputs = self._artifact_authority_inputs(candidate)
        work_id = self._model_work_id(candidate, artifact_inputs)
        meaningful_modes = tuple(
            str(value) for value in candidate.get("modes", {}).get("meaningful_modes", ())
        )
        relevant = tuple(
            attempt
            for attempt in self._attempts.records
            if attempt.get("stable_id") == stable_id and attempt.get("work_id") == work_id
        )
        per_mode = derive_per_mode_run(
            relevant,
            stable_id=stable_id,
            work_id=work_id,
            meaningful_modes=meaningful_modes,
        )
        modes = candidate.get("modes")
        if not isinstance(modes, dict):
            raise ReductionError("model modes must be mutable during authority projection")
        modes["per_mode_run"] = per_mode
        by_mode = {
            str(attempt.get("mode")): attempt
            for attempt in relevant
            if attempt.get("attempt_id") in {value["attempt_id"] for value in per_mode.values()}
            and attempt.get("result") == "succeeded"
        }
        try:
            modes.update(
                mode_summary_projection(
                    derive_mode_summary(by_mode.get("train"), by_mode.get("eval"))
                )
            )
        except AuthorityDerivationError as exc:
            raise ReductionError(str(exc)) from exc
        if candidate.get("status", {}).get("kind") != "runs":
            candidate["observed"] = derive_terminal_observation(
                relevant,
                stable_id=stable_id,
                work_id=work_id,
            )
        terminal_proof, family_authority, vector, _ = self._derive_model_authority(candidate)
        candidate["family_authority"] = family_authority_projection(family_authority)
        candidate["dependency_vector"] = vector
        status = candidate.get("status")
        if not isinstance(status, dict):
            raise ReductionError("model status must be mutable during authority projection")
        status["code"] = terminal_proof.status_code
        status["kind"] = terminal_proof.status_code.split(":", 1)[0]
        if terminal_proof.status_code.startswith("failed:"):
            status["stage"] = terminal_proof.failure_stage
            status["reason_code"] = terminal_proof.reason_code
            status["root_cause_fingerprint"] = terminal_proof.root_cause_fingerprint
        return candidate

    def authorize_publication(
        self,
        model: Mapping[str, Any],
        staged: StagedArtifact,
        accepted_gate_item: Mapping[str, Any],
        decisions: Mapping[Any, LicenseDecision],
        mirrors: MirrorStore,
        *,
        terminal: bool = False,
    ) -> PublicationAuthorization:
        """Derive and append the sole artifact publication capability.

        Parameters
        ----------
        model:
            Structurally complete candidate whose proof/vector can be replayed.
        staged:
            Exact private-custody transaction.
        accepted_gate_item:
            Independently accepted v3 gate item.
        decisions:
            Exact claim-keyed license decisions.
        mirrors:
            Separated physical mirror store.
        terminal:
            Whether the gate authorizes a terminal private/public commitment.

        Returns
        -------
        PublicationAuthorization
            Frozen capability already committed to the artifact ledger.
        """

        prepared = self.prepare_model(model)
        raw_vector = dict(prepared["dependency_vector"])
        raw_vector["accepted_attempt_ids"] = tuple(raw_vector["accepted_attempt_ids"])
        raw_vector["artifact_claim_ids"] = ()
        provisional = DependencyVector(**raw_vector)
        gate_item_sha256 = stable_hash(accepted_gate_item)
        try:
            owning_gate, ledger_gate_item = resolve_exact_gate_item_membership(
                self._gates.records,
                accepted_gate_item=accepted_gate_item,
                accepted_gate_item_sha256=gate_item_sha256,
            )
        except AuthorityDerivationError as exc:
            raise ReductionError(str(exc)) from exc
        gate_id = str(owning_gate["gate_id"])
        authorization_id = derive_publication_authorization_id(
            staged,
            accepted_gate_id=gate_id,
            accepted_gate_item_sha256=gate_item_sha256,
            dependency_vector=provisional,
            decisions=decisions,
            publication_policy_identity=self.context.publication_policy_identity,
        )
        claims = derive_artifact_claims(
            staged,
            accepted_gate_id=gate_id,
            authorization_id=authorization_id,
            decisions=decisions,
            mirrors=mirrors,
        )
        final_vector = replace(
            provisional,
            artifact_claim_ids=tuple(claim.claim_id for claim in claims),
        )
        public_object_ids = tuple(
            claim.object_id
            for claim in claims
            if claim.license_disposition == RedistributionClass.PUBLIC_OK.value
        )
        private_object_ids = tuple(
            claim.object_id
            for claim in claims
            if claim.license_disposition != RedistributionClass.PUBLIC_OK.value
        )
        authorization = PublicationAuthorization(
            authorization_id=authorization_id,
            stable_id=str(staged.event["stable_id"]),
            work_id=str(staged.event["work_id"]),
            transaction_id=staged.transaction_id,
            accepted_gate_id=gate_id,
            accepted_gate_item_sha256=gate_item_sha256,
            dependency_vector=final_vector,
            claim_ids=tuple(claim.claim_id for claim in claims),
            public_object_ids=public_object_ids,
            private_object_ids=private_object_ids,
            publication_policy_identity=self.context.publication_policy_identity,
        )
        append_artifact_authorization(
            staged,
            authorization,
            claims,
            accepted_gate_item=ledger_gate_item,
            event_kind=(
                ArtifactEventKind.TERMINAL_AUTHORIZED
                if terminal
                else ArtifactEventKind.PUBLICATION_AUTHORIZED
            ),
            context=self.context,
            mirrors=mirrors,
            ledger=self._artifacts,
        )
        self._projection_cache = None
        return authorization

    def append_model(self, model: Mapping[str, Any]) -> AppendResult:
        """Validate and append a full canonical model revision.

        Parameters
        ----------
        model:
            Complete model revision. The reducer computes its canonical revision hash.

        Returns
        -------
        AppendResult
            Idempotent append result.

        Raises
        ------
        ReductionError
            If parentage, taxonomy, gate, attempt, mode, or partition rules fail.
        """

        candidate = deepcopy(dict(model))
        if candidate.get("schema_version") != MODEL_SCHEMA_VERSION_V3:
            raise ReductionError("model appends require the current v3 authority contract")
        _validate_c07_diagnostic_fields(candidate, model=True)
        stable_id = candidate.get("stable_id")
        if stable_id not in self.intake_ids:
            raise ReductionError(f"model stable_id is outside intake: {stable_id}")
        semantic_replay = bool(getattr(self, "_semantic_replay", False))
        projection_before: Optional[tuple[str, DependencyCurrencyProjection]] = None
        if not semantic_replay:
            self._validation_current = self.current_records
            projection_before = self._projection_cache
        for existing in self._models.records:
            if existing["stable_id"] != stable_id:
                continue
            logical_existing = {
                key: value
                for key, value in existing.items()
                if key not in {"record_seq", "record_revision"}
            }
            logical_candidate = {
                key: value
                for key, value in candidate.items()
                if key not in {"record_seq", "record_revision"}
            }
            if logical_existing == logical_candidate:
                return AppendResult(deepcopy(existing), appended=False)
        previous = self._current.get(str(stable_id))
        expected_parent = previous["record_revision"] if previous is not None else None
        if (
            previous is not None
            and candidate.get("parent_revision") is None
            and not semantic_replay
            and str(stable_id) not in self.current_records
        ):
            candidate["parent_revision"] = expected_parent
            status = candidate.get("status")
            if isinstance(status, dict):
                status["supersedes_revision"] = expected_parent
        if candidate.get("parent_revision") != expected_parent:
            raise ReductionError(
                f"bad parentage for {stable_id}: expected {expected_parent!r}, "
                f"received {candidate.get('parent_revision')!r}"
            )
        supersedes_revision = candidate.get("status", {}).get("supersedes_revision")
        if previous is None and supersedes_revision is not None:
            raise ReductionError("a first model revision cannot supersede another revision")
        if previous is not None and supersedes_revision != expected_parent:
            raise ReductionError("status.supersedes_revision must exactly match parent_revision")
        if previous is None and candidate.get("record_seq") not in {None, 1}:
            raise ReductionError("a first model revision must start at record_seq 1 or omit it")
        if previous is not None and candidate.get("record_seq") is not None:
            if int(candidate["record_seq"]) <= int(previous["record_seq"]):
                raise ReductionError("record_seq must increase over the current revision")
        if previous is not None and previous.get("status", {}).get("human_review", {}).get(
            "required"
        ):
            previous_grants = set(previous.get("budget", {}).get("explicit_grants", []))
            candidate_grants = set(candidate.get("budget", {}).get("explicit_grants", []))
            if not candidate_grants > previous_grants:
                raise ReductionError(
                    "human-review terminal supersession requires a new explicit grant"
                )
        if not semantic_replay:
            _validate_persisted_requeue_lineage(
                (*self._models.records, candidate),
                self.ledger_paths,
                self._attempts.records,
                self._gates.records,
            )
        self._validate_status(candidate)
        self._validate_source(candidate)
        self._validate_derived_model_authority(candidate)
        self._validate_family_template(candidate)
        self._validate_gates(candidate)
        self._validate_execution(candidate)
        self._validate_completeness(candidate)
        try:
            result = self._models.append(candidate)
        except LedgerConflictError as exc:
            raise ReductionError(str(exc)) from exc
        self._current[str(stable_id)] = deepcopy(result.record)
        if not semantic_replay:
            if projection_before is None:
                self._projection_cache = None
            else:
                token, prior_projection = projection_before
                projected = dict(prior_projection.current_records)
                stale = dict(prior_projection.stale_reasons)
                projected[str(stable_id)] = deepcopy(result.record)
                stale.pop(str(stable_id), None)
                for variant_id, (
                    representative_id,
                    _variant_token,
                ) in self.intake_variant_bindings.items():
                    if representative_id != stable_id or variant_id == stable_id:
                        continue
                    projected.pop(variant_id, None)
                    stale[variant_id] = (
                        "family variant representative changed dependency-current revision"
                    )
                self._projection_cache = (
                    token,
                    DependencyCurrencyProjection(projected, stale),
                )
            del self._validation_current
        else:
            self._projection_cache = None
        return result

    def _validate_status(self, model: Mapping[str, Any]) -> None:
        """Validate closed terminal code/kind/stage/reason relationships.

        Parameters
        ----------
        model:
            Proposed model revision.
        """

        status = model.get("status", {})
        code = status.get("code")
        kind = status.get("kind")
        if code not in TERMINAL_STATUS_CODES:
            raise ReductionError(f"unknown terminal status code: {code!r}")
        expected_kind = str(code).split(":", 1)[0]
        if kind != expected_kind:
            raise ReductionError(f"status kind {kind!r} does not match code {code!r}")
        stage = status.get("stage")
        reason = status.get("reason_code")
        if kind == "failed":
            expected_stage = str(code).split(":", 1)[1]
            if stage != expected_stage:
                raise ReductionError("failure stage must match the failed status code")
            if reason not in FAILURE_REASON_CODES[expected_stage]:
                raise ReductionError(
                    f"reason {reason!r} is not allowed for failure stage {expected_stage!r}"
                )
            if status.get("traceback") is None and status.get("no_traceback_reason") is None:
                raise ReductionError("failure requires a traceback or explicit no_traceback_reason")
            if code == "failed:accuracy-gate" and not status.get("human_review", {}).get(
                "required"
            ):
                raise ReductionError("failed:accuracy-gate requires human review")
        elif stage is not None or reason is not None:
            raise ReductionError("non-failed statuses cannot carry failure stage/reason")

    def _validate_source(self, model: Mapping[str, Any]) -> None:
        """Enforce the mandatory public source-link invariant.

        Parameters
        ----------
        model:
            Proposed model revision.
        """

        resolution = model.get("source_resolution", {})
        sources = resolution.get("sources", [])
        primary = resolution.get("primary_source_id")
        exact_primary = bool(sources) and any(
            source.get("source_id") == primary
            and str(source.get("url", "")).startswith(("http://", "https://"))
            for source in sources
        )
        source_failure = model.get("status", {}).get("code") == "failed:source"
        if not source_failure and not exact_primary:
            raise ReductionError("missing mandatory exact public primary source link")
        if source_failure and resolution.get("mandatory_link_status") != (
            "ok" if exact_primary else "failed"
        ):
            raise ReductionError("failed:source mandatory-link status contradicts source evidence")
        if not source_failure and resolution.get("mandatory_link_status") != "ok":
            raise ReductionError(
                "non-source-failure terminal records require mandatory_link_status=ok"
            )

    def _gate_item(
        self, gate_id: Optional[str], stable_id: str
    ) -> Optional[tuple[Mapping[str, Any], Mapping[str, Any]]]:
        """Find a gate and its unique item for a stable ID.

        Parameters
        ----------
        gate_id:
            Referenced gate ID.
        stable_id:
            Model stable ID.

        Returns
        -------
        tuple[Mapping[str, Any], Mapping[str, Any]] | None
            Gate envelope and matching item, if present.
        """

        if gate_id is None:
            return None
        gate = self._gate_index().get(str(gate_id))
        if gate is None:
            return None
        if gate.get("schema_version") != GATE_SCHEMA_VERSION_V3:
            return None
        try:
            load_current_gate_proof(gate)
        except AuthorityDerivationError as exc:
            raise ReductionError(str(exc)) from exc
        matches = [item for item in gate["items"] if item["stable_id"] == stable_id]
        if len(matches) != 1:
            return None
        return gate, matches[0]

    def _validate_gates(self, model: Mapping[str, Any]) -> None:
        """Enforce block-at-write metadata and required fidelity gates.

        Parameters
        ----------
        model:
            Proposed model revision.
        """

        stable_id = str(model["stable_id"])
        accuracy = model.get("accuracy_gate", {})
        rung = model.get("source_resolution", {}).get("rung")
        status_kind = model.get("status", {}).get("kind")
        metadata_state = model.get("authored_metadata_state")
        family_variant = stable_id in self.intake_variant_bindings
        representative_id = str(model.get("identity", {}).get("family_representative_id", ""))
        representative = (
            self._validation_current_records().get(representative_id) if family_variant else None
        )
        identities = None
        rung_gate_current = False
        gate_stable_id = representative_id if family_variant else stable_id
        rung_found = self._gate_item(accuracy.get("gate_id"), gate_stable_id)
        if rung_found is not None:
            rung_gate, rung_item = rung_found
            rung_check = rung_item.get("rung_check")
            verified_hashes = rung_item.get("verified_hashes", {})
            rung_gate_current = bool(
                rung_gate.get("gate_kind") == "metadata_batch"
                and rung_item.get("vet_identity") == accuracy.get("vet_identity")
                and isinstance(rung_check, Mapping)
                and rung_check.get("selected_rung") == rung
                and rung_check.get("verdict") == "accurate"
                and verified_hashes.get("code")
                == model.get("implementation", {}).get("code_sha256")
                and rung_gate.get("checker", {}).get("prompt_sha256") == _checker_prompt_hash()
            )
        if status_kind == "runs" and not rung_gate_current:
            if metadata_state == "pending":
                self._validate_pending_run_rung(model)
            else:
                raise ReductionError(
                    "runs requires a current identity-tight anti-slop/rung check gate"
                )
        if metadata_state == "accepted":
            found = self._gate_item(accuracy.get("gate_id"), gate_stable_id)
            if found is None:
                raise ReductionError("accepted authored metadata is missing its gate")
            gate, item = found
            if gate["gate_kind"] != "metadata_batch":
                raise ReductionError("authored metadata must reference a metadata_batch gate")
            rung_check = item.get("rung_check")
            if (
                item["vet_identity"] != accuracy.get("vet_identity")
                or item["verdict"] != "accurate"
                or item["integrity"]["verdict"] != "accurate"
                or not isinstance(rung_check, Mapping)
                or rung_check.get("selected_rung") != rung
                or rung_check.get("verdict") != "accurate"
                or accuracy.get("verdict") != "accurate"
                or not accuracy.get("current")
                or gate.get("checker", {}).get("prompt_sha256") != accuracy.get("prompt_sha256")
            ):
                raise ReductionError(
                    "authored metadata gate is missing, stale, inaccurate, or has a blocked rung check"
                )
            facts = _authored_facts_for_vet(model)
            if family_variant:
                representative_accuracy = (
                    representative.get("accuracy_gate", {})
                    if isinstance(representative, Mapping)
                    else {}
                )
                if (
                    not isinstance(representative, Mapping)
                    or representative.get("authored_metadata_state") != "accepted"
                    or representative_accuracy.get("current") is not True
                    or accuracy != representative_accuracy
                ):
                    raise ReductionError(
                        "family variant does not inherit its current representative accuracy gate"
                    )
                try:
                    identities = recompute_accepted_identities(
                        facts,
                        checker_prompt_hash=_checker_prompt_hash(),
                        checker_model=str(gate.get("checker", {}).get("model")),
                        checker_version=str(gate.get("checker", {}).get("version")),
                        schema_version=MODEL_SCHEMA_VERSION_V3,
                    )
                except MetadataValidationError as exc:
                    raise ReductionError(str(exc)) from exc
                if (
                    model.get("evidence", {}).get("evidence_identity") != identities.evidence
                    or model.get("implementation", {}).get("recipe_revision") != identities.recipe
                ):
                    raise ReductionError(
                        "family variant source/evidence/recipe identities are stale"
                    )
            else:
                try:
                    validate_authored_facts_for_write(facts, item)
                    identities = recompute_accepted_identities(
                        facts,
                        checker_prompt_hash=_checker_prompt_hash(),
                        checker_model=str(gate.get("checker", {}).get("model")),
                        checker_version=str(gate.get("checker", {}).get("version")),
                        schema_version=MODEL_SCHEMA_VERSION_V3,
                    )
                except MetadataValidationError as exc:
                    raise ReductionError(str(exc)) from exc
                if (
                    model.get("evidence", {}).get("evidence_identity") != identities.evidence
                    or model.get("implementation", {}).get("recipe_revision") != identities.recipe
                    or item.get("vet_identity") != identities.vet
                    or accuracy.get("vet_identity") != identities.vet
                ):
                    stale = [
                        name
                        for name, actual, expected in (
                            (
                                "evidence",
                                model.get("evidence", {}).get("evidence_identity"),
                                identities.evidence,
                            ),
                            (
                                "recipe",
                                model.get("implementation", {}).get("recipe_revision"),
                                identities.recipe,
                            ),
                            ("gate-vet", item.get("vet_identity"), identities.vet),
                            ("model-vet", accuracy.get("vet_identity"), identities.vet),
                        )
                        if actual != expected
                    ]
                    raise ReductionError(
                        "accepted source/evidence/recipe/vet identities are stale: "
                        + ", ".join(stale)
                    )
        fidelity = model.get("fidelity", {})
        legacy_flags = model.get("intake", {}).get("preserved_legacy_flags", [])
        required = (
            bool(fidelity.get("required"))
            or rung in {"R3_PORT", "R4_REIMPLEMENT"}
            or legacy_requires_fidelity_audit(legacy_flags)
        )
        if required:
            found = self._gate_item(fidelity.get("gate_id"), stable_id)
            if found is None:
                status = model.get("status", {})
                if self._is_pre_fidelity_terminal(model):
                    return
                if (
                    status.get("code") == "failed:fidelity"
                    and status.get("reason_code") == "identity-mismatch"
                    and not fidelity.get("current")
                ):
                    return
                raise ReductionError("required fidelity is missing its gate")
            gate, item = found
            if identities is None:
                try:
                    identities = recompute_accepted_identities(
                        _model_facts(model),
                        checker_prompt_hash=_checker_prompt_hash(),
                        checker_model=str(gate.get("checker", {}).get("model")),
                        checker_version=str(gate.get("checker", {}).get("version")),
                        schema_version=MODEL_SCHEMA_VERSION_V3,
                    )
                except MetadataValidationError as exc:
                    raise ReductionError(str(exc)) from exc
            if gate["gate_kind"] != "fidelity":
                raise ReductionError("fidelity must reference a per-model fidelity gate")
            rung_check = item.get("rung_check")
            rejected_terminal = model.get("status", {}).get(
                "code"
            ) == "failed:fidelity" or self._is_fidelity_repair_failure(model)
            allowed_verdicts = (
                {"major-drift", "slop", "cannot-verify"}
                if rejected_terminal
                else {"match", "minor-drift"}
            )
            rung_accepted = isinstance(rung_check, Mapping) and (
                rejected_terminal
                or (
                    rung_check.get("verdict") == "accurate"
                    and rung_check.get("selected_rung") == rung
                )
            )
            if (
                item.get("fidelity_identity") != fidelity.get("fidelity_identity")
                or item.get("fidelity_identity") != identities.fidelity
                or item["fidelity"]["verdict"] != fidelity.get("verdict")
                or fidelity.get("verdict") not in allowed_verdicts
                or not rung_accepted
                or not fidelity.get("current")
            ):
                raise ReductionError(
                    "required fidelity gate is stale, unacceptable, or has a blocked rung check"
                )

    def _validate_pending_run_rung(self, model: Mapping[str, Any]) -> None:
        """Validate a driver-owned R1/R2 run while authored metadata is pending.

        Parameters
        ----------
        model:
            Proposed pending-metadata run revision.
        """

        rung = model.get("source_resolution", {}).get("rung")
        if rung not in {"R1_LIBRARY", "R2_VENDOR"}:
            raise ReductionError("pending metadata runs are restricted to mechanical R1/R2")
        accuracy = model.get("accuracy_gate", {})
        if (
            accuracy.get("gate_id") is not None
            or accuracy.get("vet_identity") is not None
            or accuracy.get("verdict") is not None
            or accuracy.get("current")
            or accuracy.get("prompt_sha256") != _checker_prompt_hash()
        ):
            raise ReductionError("pending metadata run carries false accuracy-gate authority")
        for field in (
            "taxonomy",
            "external_metadata",
            "website",
            "people_and_origin",
            "dates",
            "citation",
            "licenses",
        ):
            if model.get(field) is not None:
                raise ReductionError("pending metadata run exposes ungated source-read fields")
        untrusted = model.get("untrusted_attempt")
        proposal = untrusted.get("proposal") if isinstance(untrusted, Mapping) else None
        if not isinstance(untrusted, Mapping) or not isinstance(proposal, Mapping):
            raise ReductionError("pending metadata run lacks its exact untrusted proposal")
        proposal_sha256 = stable_hash(
            {key: value for key, value in proposal.items() if key != "proposal_sha256"}
        )
        if (
            untrusted.get("proposal_sha256") != proposal_sha256
            or proposal.get("proposal_sha256") != proposal_sha256
            or proposal.get("stable_id") != model.get("stable_id")
        ):
            raise ReductionError("pending metadata run proposal identity is stale")
        proposed_facts = proposal.get("proposed_facts")
        if not isinstance(proposed_facts, Mapping):
            raise ReductionError("pending metadata run proposal facts are incomplete")
        for field in (
            "identity",
            "source_resolution",
            "evidence",
            "implementation",
            "input_contract",
            "fidelity",
        ):
            if model.get(field) != proposed_facts.get(field):
                raise ReductionError(f"pending metadata run changed mechanical fact root {field}")
        proposed_modes = proposed_facts.get("modes")
        model_modes = model.get("modes")
        if not isinstance(proposed_modes, Mapping) or not isinstance(model_modes, Mapping):
            raise ReductionError("pending metadata run modes are incomplete")
        if model_modes.get("meaningful_modes") != proposed_modes.get("meaningful_modes"):
            raise ReductionError("pending metadata run changed authored meaningful modes")
        try:
            identities = recompute_accepted_identities(
                proposed_facts,
                checker_prompt_hash=_checker_prompt_hash(),
                checker_model=str(accuracy.get("checker_model")),
                checker_version=str(accuracy.get("checker_version")),
                schema_version=MODEL_SCHEMA_VERSION_V3,
            )
        except MetadataValidationError as exc:
            raise ReductionError(str(exc)) from exc
        if (
            proposal.get("source_identity") != identities.source
            or proposal.get("evidence_identity") != identities.evidence
            or proposal.get("recipe_revision") != identities.recipe
            or model.get("evidence", {}).get("evidence_identity") != identities.evidence
            or model.get("implementation", {}).get("recipe_revision") != identities.recipe
        ):
            raise ReductionError("pending metadata mechanical identities are stale")

    def _is_fidelity_repair_failure(self, model: Mapping[str, Any]) -> bool:
        """Return whether a runner terminal is an evidenced fidelity-repair failure.

        Parameters
        ----------
        model:
            Proposed terminal revision.

        Returns
        -------
        bool
            True only when a driver-owned runner attempt failed after a current
            rejected fidelity gate.
        """

        if model.get("status", {}).get("code") != "failed:runner":
            return False
        fidelity = model.get("fidelity", {})
        if not fidelity.get("current") or fidelity.get("verdict") not in {
            "major-drift",
            "slop",
            "cannot-verify",
        }:
            return False
        attempts_by_id = self._attempt_index()
        return any(
            (attempt := attempts_by_id.get(str(attempt_id))) is not None
            and attempt.get("actor") == "driver"
            and attempt.get("stage") == "runner"
            and attempt.get("result") == "failed"
            and attempt.get("environment") is None
            for attempt_id in model.get("status", {}).get("attempt_ids", [])
        )

    def _is_pre_fidelity_terminal(self, model: Mapping[str, Any]) -> bool:
        """Return whether a terminal record stopped before fidelity or execution.

        Parameters
        ----------
        model:
            Proposed terminal revision with an R3/R4-required fidelity block.

        Returns
        -------
        bool
            Whether durable attempts prove that no fidelity/run stage was reached.
        """

        status = model.get("status", {})
        fidelity = model.get("fidelity", {})
        execution = model.get("execution", {})
        pre_fidelity_failure_stages = {
            "intake",
            "source",
            "fetch",
            "evidence",
            "accuracy-gate",
            "runner",
        }
        if (
            status.get("kind") == "runs"
            or status.get("code") == "failed:fidelity"
            or (
                status.get("kind") == "failed"
                and status.get("stage") not in pre_fidelity_failure_stages
            )
            or fidelity.get("current")
            or fidelity.get("gate_id") is not None
            or fidelity.get("verdict") is not None
            or execution.get("current")
            or execution.get("accepted_attempt_ids")
        ):
            return False
        attempts_by_id = self._attempt_index()
        for attempt_id in status.get("attempt_ids", []):
            attempt = attempts_by_id.get(attempt_id)
            if attempt is None:
                return False
            identities = attempt.get("identities", {})
            receipt = attempt.get("worker_receipt", {})
            if (
                attempt.get("environment") is not None
                or identities.get("environment") is not None
                or identities.get("execution") is not None
                or receipt.get("present")
                or receipt.get("forward_started")
            ):
                return False
        return True

    def _validate_family_template(self, model: Mapping[str, Any]) -> None:
        """Validate the trusted-intake family branch before record-shaped hints."""

        stable_id = str(model.get("stable_id", ""))
        website = model.get("website")
        binding = self.intake_variant_bindings.get(stable_id)
        if binding is None:
            if isinstance(website, Mapping) and website.get("kind") == "size-variant-template":
                raise ReductionError("ordinary intake item cannot claim family variant authority")
            return
        if not isinstance(website, Mapping) or website.get("kind") != "size-variant-template":
            raise ReductionError("trusted family variant cannot omit its structural template")
        trusted_representative_id, trusted_variant_token = binding
        representative_id = model.get("identity", {}).get("family_representative_id")
        if (
            representative_id != trusted_representative_id
            or website.get("template_source_model_id") != trusted_representative_id
        ):
            raise ReductionError("family variant contradicts its trusted intake family binding")
        representative = self._validation_current_records().get(str(representative_id))
        if not isinstance(representative, Mapping) or not family_representative_is_usable(
            representative, str(representative_id)
        ):
            raise ReductionError("family variant representative is not a usable accepted record")
        try:
            terminal_variant = model.get("status", {}).get("kind") != "runs"
            validate_size_variant(
                representative,
                model,
                str(representative_id),
                parameter_count_total=(
                    None
                    if terminal_variant
                    else model.get("observed", {}).get("parameter_count_total")
                ),
                input_contract=(None if terminal_variant else model.get("input_contract", {})),
            )
            validate_size_variant_derivation(
                representative,
                model,
                str(representative_id),
                trusted_variant_token=trusted_variant_token,
            )
        except FamilyTemplateError as exc:
            raise ReductionError(f"family template validation failed: {exc}") from exc

    def _validate_completeness(self, model: Mapping[str, Any]) -> None:
        """Derive every completeness bit and reject producer contradictions.

        Parameters
        ----------
        model:
            Candidate already validated for source, gates, family, and execution.

        Raises
        ------
        ReductionError
            If any supplied completeness value differs from reducer-derived facts.
        """

        completeness = model.get("completeness")
        if not isinstance(completeness, Mapping):
            raise ReductionError("model completeness block is missing")
        status = model.get("status", {})
        status_kind = status.get("kind")
        metadata_accepted = model.get("authored_metadata_state") == "accepted"
        source_fields = (
            "taxonomy",
            "external_metadata",
            "website",
            "people_and_origin",
            "dates",
            "citation",
            "licenses",
        )
        source_read_complete = metadata_accepted and all(
            model.get(field) is not None for field in source_fields
        )
        coverage = model.get("evidence", {}).get("coverage", {})
        evidence_complete = bool(
            metadata_accepted
            and coverage.get("all_agent_fields_have_support") is True
            and coverage.get("family_grounding_complete") is True
            and coverage.get("missing_support") == []
        )
        accuracy = model.get("accuracy_gate", {})
        accuracy_current = bool(
            metadata_accepted
            and accuracy.get("current") is True
            and accuracy.get("verdict") == "accurate"
            and accuracy.get("gate_id")
            and accuracy.get("vet_identity")
        )
        fidelity = model.get("fidelity", {})
        fidelity_current = bool(not fidelity.get("required") or fidelity.get("current") is True)
        execution_current = bool(
            status_kind == "runs" and model.get("execution", {}).get("current") is True
        )
        family_valid = True
        human_review_pending = status.get("human_review", {}).get("required") is True
        release_eligible = bool(
            status_kind == "runs"
            and source_read_complete
            and evidence_complete
            and accuracy_current
            and fidelity_current
            and execution_current
            and family_valid
            and not human_review_pending
        )
        status_code = str(status.get("code"))
        if status_kind == "runs":
            expected_issues = [] if metadata_accepted else ["authored-metadata-pending"]
        else:
            expected_issues = [status_code]
        resolution = model.get("source_resolution", {})
        sources = resolution.get("sources", []) if isinstance(resolution, Mapping) else []
        primary = resolution.get("primary_source_id") if isinstance(resolution, Mapping) else None
        mandatory_source_present = bool(sources) and any(
            isinstance(source, Mapping)
            and source.get("source_id") == primary
            and str(source.get("url", "")).startswith(("http://", "https://"))
            for source in sources
        )
        expected: JsonObject = {
            "schema_valid": True,
            "mandatory_source_present": mandatory_source_present,
            "source_read_fields_complete": source_read_complete,
            "evidence_coverage_complete": evidence_complete,
            "accuracy_gate_current": accuracy_current,
            "required_fidelity_current": fidelity_current,
            "execution_current": execution_current,
            "family_template_valid": family_valid,
            "release_eligible": release_eligible,
            "issues": expected_issues,
        }
        for field, expected_value in expected.items():
            if completeness.get(field) != expected_value:
                raise ReductionError(
                    f"completeness.{field} contradicts reducer-derived value {expected_value!r}"
                )

    def _terminal_gate_evidence_valid(self, model: Mapping[str, Any], gate_kind: str) -> bool:
        """Return whether a rejected checker history proves one gate terminal.

        Parameters
        ----------
        model:
            Candidate failed terminal.
        gate_kind:
            ``metadata_batch`` or ``fidelity``.

        Returns
        -------
        bool
            True only for the exact rejected item after bounded cap exhaustion
            or a repeated root cause.
        """

        stable_id = str(model.get("stable_id"))
        block_name = "accuracy_gate" if gate_kind == "metadata_batch" else "fidelity"
        gate_id = model.get(block_name, {}).get("gate_id")
        found = self._gate_item(gate_id, stable_id)
        if found is None:
            return False
        gate, item = found
        if gate.get("gate_kind") != gate_kind:
            return False
        if gate_kind == "metadata_batch":
            rejected = not (
                item.get("verdict") == "accurate"
                and item.get("integrity", {}).get("verdict") == "accurate"
                and item.get("rung_check", {}).get("verdict") == "accurate"
            )
        else:
            rejected = not (
                item.get("fidelity", {}).get("verdict") in {"match", "minor-drift"}
                and item.get("rung_check", {}).get("verdict") == "accurate"
            )
        status = model.get("status", {})
        if not rejected or status.get("root_cause_fingerprint") != _gate_item_fingerprint(item):
            return False
        lineage = item.get("campaign_root_work_id")
        rejected_fingerprints: list[str] = []
        for candidate_gate in self._gates.records:
            if candidate_gate.get("gate_kind") != gate_kind:
                continue
            matching = [
                candidate
                for candidate in candidate_gate.get("items", [])
                if candidate.get("stable_id") == stable_id
                and candidate.get("campaign_root_work_id") == lineage
            ]
            if len(matching) != 1:
                continue
            candidate = matching[0]
            if gate_kind == "metadata_batch":
                accepted = bool(
                    candidate.get("verdict") == "accurate"
                    and candidate.get("integrity", {}).get("verdict") == "accurate"
                    and candidate.get("rung_check", {}).get("verdict") == "accurate"
                )
            else:
                accepted = bool(
                    candidate.get("fidelity", {}).get("verdict") in {"match", "minor-drift"}
                    and candidate.get("rung_check", {}).get("verdict") == "accurate"
                )
            if not accepted:
                rejected_fingerprints.append(_gate_item_fingerprint(candidate))
            if candidate_gate.get("gate_id") == gate_id:
                break
        if not rejected_fingerprints:
            return False
        latest = rejected_fingerprints[-1]
        return len(rejected_fingerprints) > 2 or latest in rejected_fingerprints[:-1]

    def _validate_execution(self, model: Mapping[str, Any]) -> None:
        """Enforce attempt/receipt and meaningful-mode rules for run awards.

        Parameters
        ----------
        model:
            Proposed model revision.
        """

        if model.get("status", {}).get("kind") != "runs":
            return
        execution = model.get("execution", {})
        if not execution.get("current"):
            raise ReductionError("runs requires a current execution identity")
        if execution.get("network_attempted") or execution.get("checkpoint_accessed"):
            raise ReductionError("policy-contaminated execution cannot earn runs")
        modes = model.get("modes", {})
        meaningful_order = tuple(str(mode) for mode in modes.get("meaningful_modes", []))
        meaningful = set(meaningful_order)
        per_mode = modes.get("per_mode_run", {})
        if meaningful != set(per_mode):
            raise ReductionError("per_mode_run must cover exactly every meaningful mode")
        attempts_by_id = self._attempt_index()
        accepted = set(execution.get("accepted_attempt_ids", []))
        stable_id = model["stable_id"]
        rung = model.get("source_resolution", {}).get("rung")
        cold_policy = cold_forward_policy(str(stable_id), rung)
        signatures: dict[str, list[Any]] = {mode: [] for mode in meaningful}
        counts: dict[str, int] = {mode: 0 for mode in meaningful}
        cold_indexes: dict[str, set[int]] = {mode: set() for mode in meaningful}
        parent_witnesses: set[str] = set()
        invocations: set[bytes] = set()
        accepted_work_ids: set[str] = set()
        implementation = model.get("implementation", {})
        evidence = model.get("evidence", {})
        untrusted = model.get("untrusted_attempt")
        pending_proposal = untrusted.get("proposal") if isinstance(untrusted, Mapping) else None
        pending_facts = (
            pending_proposal.get("proposed_facts")
            if isinstance(pending_proposal, Mapping)
            and model.get("authored_metadata_state") == "pending"
            else None
        )
        code_manifest = implementation.get("code_manifest")
        expected_manifest_digest = (
            stable_hash(code_manifest)
            if isinstance(code_manifest, list) and code_manifest
            else None
        )
        external_metadata = (
            pending_facts.get("external_metadata")
            if isinstance(pending_facts, Mapping)
            else model.get("external_metadata", {})
        )
        modality = (
            external_metadata.get("modality") if isinstance(external_metadata, Mapping) else None
        )
        expected_asset = (
            expected_standard_asset(modality)
            if implementation.get("recipe_type") == "declarative-library"
            else None
        )
        expected_asset_digest = expected_asset["sha256"] if expected_asset is not None else None
        expected_asset_id = expected_asset["asset_id"] if expected_asset is not None else None
        try:
            accepted_identities = recompute_accepted_identities(
                pending_facts if isinstance(pending_facts, Mapping) else _model_facts(model),
                checker_prompt_hash=_checker_prompt_hash(),
                checker_model=str(model.get("accuracy_gate", {}).get("checker_model")),
                checker_version=str(model.get("accuracy_gate", {}).get("checker_version")),
                schema_version=MODEL_SCHEMA_VERSION_V3,
            )
        except MetadataValidationError as exc:
            raise ReductionError(str(exc)) from exc
        input_contract = model.get("input_contract", {})
        accepted_in_order = execution.get("accepted_attempt_ids", [])
        if not isinstance(accepted_in_order, list) or len(accepted_in_order) != len(accepted):
            raise ReductionError("accepted execution attempt IDs must be a unique ordered list")
        try:
            authenticated_attempts = authenticate_accepted_attempts(
                accepted_in_order,
                self._attempts.records,
                stable_id=str(stable_id),
                execution_identity=str(execution.get("execution_identity")),
            )
        except AuthorityDerivationError as exc:
            raise ReductionError(str(exc)) from exc
        authenticated_by_id = {
            authority.attempt_id: authority for authority in authenticated_attempts
        }
        for attempt_id in accepted:
            attempt = attempts_by_id.get(attempt_id)
            if attempt is None:
                raise ReductionError("accepted execution attempt is missing")
            authority = authenticated_by_id[attempt_id]
            accepted_work_ids.add(authority.work_id)
            identities = attempt.get("identities", {})
            receipt = attempt.get("worker_receipt", {})
            policy = attempt.get("policy_observation", {})
            if not isinstance(policy, Mapping) or any(
                policy.get(key)
                for key in (
                    "network_attempted",
                    "checkpoint_or_weight_read_attempted",
                    "cache_read_attempted",
                    "write_outside_scratch_attempted",
                    "credentials_present",
                    "torchlens_import_attempted",
                )
            ):
                raise ReductionError("accepted attempt lacks a clean successful worker receipt")
            observed_asset_pair = (
                receipt.get("observed_input_asset_sha256"),
                receipt.get("input_asset"),
            )
            raw_receipt = attempt.get("raw_award_receipt")
            expected_observed_manifest = expected_manifest_digest
            if expected_observed_manifest is None and isinstance(raw_receipt, Mapping):
                expected_observed_manifest = raw_receipt.get("code_manifest_identity")
            if (
                identities.get("source") != accepted_identities.source
                or identities.get("recipe") != accepted_identities.recipe
                or identities.get("recipe") != implementation.get("recipe_revision")
                or identities.get("evidence") != accepted_identities.evidence
                or identities.get("evidence") != evidence.get("evidence_identity")
                or identities.get("environment") != execution.get("env_generation")
                or identities.get("checker_prompt") != _checker_prompt_hash()
                or receipt.get("observed_recipe_revision") != accepted_identities.recipe
                or receipt.get("observed_adapter_sha256") != implementation.get("code_sha256")
                or receipt.get("observed_code_manifest_sha256") != expected_observed_manifest
                or observed_asset_pair
                not in {(None, None), (expected_asset_digest, expected_asset_id)}
            ):
                raise ReductionError("accepted attempt identities are stale for the current model")
            mode = str(attempt.get("mode"))
            observation = attempt.get("supervisor_observation", {})
            signature = receipt.get("output_signature")
            retries = attempt.get("retries", {})
            stage_attempt = retries.get("stage_attempt")
            cold_index = stage_attempt - 1 if isinstance(stage_attempt, int) else -1
            mode_index = meaningful_order.index(mode) if mode in meaningful_order else -1
            expected_attempt_id = stable_hash(
                {
                    "work_id": attempt.get("work_id"),
                    "execution_identity": execution.get("execution_identity"),
                    "cold_index": cold_index,
                    "mode": mode,
                }
            )
            completion_line = observation.get("stdout_completion_line")
            invocation = attempt.get("invocation", {})
            argv = invocation.get("argv") if isinstance(invocation, Mapping) else None
            if (
                mode not in meaningful
                or attempt.get("result") != "succeeded"
                or observation.get("exit_code") != 0
                or observation.get("signal") is not None
                or not receipt.get("constructor_started")
                or not receipt.get("constructor_completed")
                or not receipt.get("input_completed")
                or not receipt.get("forward_started")
                or not receipt.get("forward_completed")
                or not input_signature_matches_contract(
                    receipt.get("input_signature"), input_contract
                )
                or output_signature_error(signature) is not None
                or (
                    cold_policy.required_cold_forwards > 1
                    and (
                        cold_index < 0
                        or attempt.get("attempt_no")
                        != cold_index * len(meaningful_order) + mode_index + 1
                        or attempt.get("attempt_id") != expected_attempt_id
                        or not isinstance(completion_line, str)
                        or not isinstance(argv, list)
                        or any(not isinstance(value, str) for value in argv)
                    )
                )
            ):
                raise ReductionError("accepted attempt lacks a complete zero-exit receipt")
            counts[mode] += 1
            signatures[mode].append(signature)
            if cold_policy.required_cold_forwards > 1:
                cold_indexes[mode].add(cold_index)
                if completion_line in parent_witnesses:
                    raise ReductionError("accepted cold forwards reuse a parent process witness")
                parent_witnesses.add(completion_line)
                invocation_bytes = canonical_json_bytes(argv)
                if invocation_bytes in invocations:
                    raise ReductionError(
                        "accepted cold forwards reuse a scratch/request invocation"
                    )
                invocations.add(invocation_bytes)
            else:
                cold_indexes[mode].add(0)
        if len(accepted_work_ids) != 1:
            raise ReductionError("accepted attempts span multiple proposal work identities")
        if isinstance(pending_proposal, Mapping):
            if str(pending_proposal.get("work_id")) not in accepted_work_ids:
                raise ReductionError("pending run proposal is stale for the accepted work identity")
        else:
            family_variant = str(stable_id) in self.intake_variant_bindings
            if family_variant:
                representative_id = str(
                    model.get("identity", {}).get("family_representative_id", "")
                )
                representative = self._validation_current_records().get(representative_id)
                if not isinstance(representative, Mapping) or (
                    model.get("accuracy_gate") != representative.get("accuracy_gate")
                ):
                    raise ReductionError(
                        "family variant anti-slop authority is not its current representative"
                    )
            else:
                rung_gate = self._gate_item(
                    model.get("accuracy_gate", {}).get("gate_id"), str(stable_id)
                )
                if rung_gate is None or str(rung_gate[1].get("work_id")) not in accepted_work_ids:
                    raise ReductionError(
                        "anti-slop/rung gate is stale for the accepted work identity"
                    )
        for mode in meaningful:
            reference = per_mode[mode]
            attempt_id = reference.get("attempt_id")
            attempt = attempts_by_id.get(attempt_id)
            if attempt_id not in accepted or attempt is None:
                raise ReductionError(f"mode {mode} references an unaccepted/missing attempt")
            receipt = attempt["worker_receipt"]
            policy = attempt["policy_observation"]
            if (
                attempt["stable_id"] != stable_id
                or attempt["stage"] != "forward"
                or attempt["mode"] != mode
                or attempt["result"] != "succeeded"
                or reference.get("status") != "succeeded"
                or attempt["identities"]["execution"] != execution.get("execution_identity")
                or not receipt["present"]
                or not receipt["constructor_started"]
                or not receipt["constructor_completed"]
                or not receipt["input_completed"]
                or not receipt["forward_started"]
                or not receipt["forward_completed"]
                or not input_signature_matches_contract(
                    receipt.get("input_signature"), input_contract
                )
                or receipt["mode"] != mode
                or any(
                    policy[key]
                    for key in (
                        "network_attempted",
                        "checkpoint_or_weight_read_attempted",
                        "cache_read_attempted",
                        "write_outside_scratch_attempted",
                        "credentials_present",
                        "torchlens_import_attempted",
                    )
                )
            ):
                raise ReductionError(f"mode {mode} lacks a clean successful worker receipt")
        meaningful_order = model.get("modes", {}).get("meaningful_modes", [])
        if not isinstance(meaningful_order, list) or not meaningful_order:
            raise ReductionError("observed facts require an ordered meaningful-mode set")
        measurement_mode = str(meaningful_order[0])
        measurement_reference = per_mode.get(measurement_mode, {})
        measurement_attempt = attempts_by_id.get(measurement_reference.get("attempt_id"))
        if measurement_attempt is None:
            raise ReductionError("observed facts lack their designated measurement attempt")
        measurement_receipt = measurement_attempt.get("worker_receipt", {})
        measurement_supervisor = measurement_attempt.get("supervisor_observation", {})
        observed = model.get("observed", {})
        receipt_observations = {
            "parameter_count_total": measurement_receipt.get("parameter_count_total"),
            "parameter_count_trainable": measurement_receipt.get("parameter_count_trainable"),
            "native_framework": measurement_receipt.get("native_framework"),
            "delegated_method": measurement_receipt.get("delegated_method"),
            "output_signature": measurement_receipt.get("output_signature"),
            "input_kind": measurement_receipt.get("input_kind"),
            "input_asset": measurement_receipt.get("input_asset"),
            "input_note": measurement_receipt.get("input_note"),
            "constructor_seconds": measurement_receipt.get("constructor_seconds"),
            "forward_seconds": measurement_receipt.get("forward_seconds"),
            "peak_rss_bytes": measurement_supervisor.get("peak_rss_bytes"),
            "measurement_attempt_ids": accepted_in_order,
        }
        if not isinstance(observed, Mapping) or any(
            observed.get(field) != value for field, value in receipt_observations.items()
        ):
            raise ReductionError("observed runtime facts contradict accepted worker receipts")
        snippet = "driver-owned isolated forward"
        if observed.get("snippet") != snippet or observed.get("snippet_sha256") != stable_hash(
            snippet
        ):
            raise ReductionError("observed snippet is not the driver-owned mechanical recipe")
        confirmation_policy = execution.get("confirmation_policy")
        if confirmation_policy != cold_policy.confirmation_policy:
            raise ReductionError("execution confirmation policy contradicts reducer policy")
        expected_indexes = set(range(cold_policy.required_cold_forwards))
        if any(
            counts[mode] != cold_policy.required_cold_forwards
            or cold_indexes[mode] != expected_indexes
            for mode in meaningful
        ):
            raise ReductionError("accepted attempts do not satisfy reducer cold-forward policy")
        if any(
            any(
                canonical_json_bytes(signature) != canonical_json_bytes(mode_signatures[0])
                for signature in mode_signatures[1:]
            )
            for mode_signatures in signatures.values()
        ):
            raise ReductionError("non-deterministic cold-forward output signature")

    def _validate_deferral(self, model: Mapping[str, Any]) -> None:
        """Require positive source/probe evidence for either closed platform deferral.

        Parameters
        ----------
        model:
            Proposed model revision.
        """

        status_code = model.get("status", {}).get("code")
        if status_code not in {"deferred:needs-cuda", "deferred:needs-x86"}:
            return
        attempts_by_id = self._attempt_index()
        for attempt_id in model.get("status", {}).get("attempt_ids", []):
            attempt = attempts_by_id.get(attempt_id)
            evidence = attempt.get("defer_evidence") if attempt is not None else None
            if (
                isinstance(evidence, Mapping)
                and evidence.get("target_status") == status_code
                and (evidence.get("source_ids") or evidence.get("probe_attempt_ids"))
            ):
                return
        raise ReductionError("platform deferral requires positive source or focused-probe evidence")


def _production_canonical_root(ledgers: LedgerPaths) -> Optional[Path]:
    """Return the canonical crawler root for the production records layout.

    Parameters
    ----------
    ledgers:
        Candidate canonical ledger paths.

    Returns
    -------
    pathlib.Path | None
        ``menagerie/crawler`` root, or ``None`` for isolated test layouts.
    """

    records_root = ledgers.models.resolve().parent.parent
    canonical_root = records_root.parent
    if (
        records_root.name == "records"
        and canonical_root.name == "crawler"
        and canonical_root.parent.name == "menagerie"
    ):
        return canonical_root
    return None


def _dependency_bytes_token(ledgers: LedgerPaths) -> str:
    """Return a cheap live-cache token for shared projection authorities.

    Parameters
    ----------
    ledgers:
        Canonical ledger and records-root paths.

    Returns
    -------
    str
        Stable token over prompts, authority code, schemas, and environment manifests.

        Per-model reconstruction/source bytes are validated while projecting that
        model and by every one-shot checkpoint/rebuild projection. They are not
        rescanned globally on each driver hot-path lookup.
    """

    package_root = Path(__file__).parent
    authority_paths = [
        *package_root.glob("*.py"),
        *(package_root / "schemas").glob("*.json"),
        *(package_root / "prompts").glob("*.txt"),
    ]
    canonical_root = _production_canonical_root(ledgers)
    if canonical_root is not None:
        envs_root = canonical_root / "envs"
        if envs_root.exists():
            authority_paths.extend(path for path in envs_root.rglob("*") if path.is_file())
    facts: list[tuple[str, str]] = []
    for path in sorted(set(authority_paths)):
        try:
            digest = hash_bytes(path.read_bytes())
        except OSError:
            continue
        facts.append((str(path), digest))
    return stable_hash(facts)


def intake_variant_bindings_from_rows(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, tuple[str, str]]:
    """Derive trusted non-representative family bindings from intake rows.

    Parameters
    ----------
    rows:
        Trusted immutable intake items.

    Returns
    -------
    dict[str, tuple[str, str]]
        Variant stable ID to exact representative ID and variant token.
    """

    bindings: dict[str, tuple[str, str]] = {}
    for row in rows:
        stable_id = row.get("stable_id")
        representative_id = row.get("family_representative_id") or stable_id
        if (
            isinstance(stable_id, str)
            and isinstance(representative_id, str)
            and representative_id != stable_id
            and row.get("variant_scope", "family") == "family"
        ):
            bindings[stable_id] = (representative_id, str(row.get("variant", "")))
    return bindings


def _canonical_intake_rows(ledgers: LedgerPaths) -> dict[str, JsonObject]:
    """Load trusted intake rows committed beside production ledgers.

    Parameters
    ----------
    ledgers:
        Canonical records paths.

    Returns
    -------
    dict[str, dict[str, Any]]
        Stable-ID keyed immutable intake authority, empty outside production.
    """

    canonical_root = _production_canonical_root(ledgers)
    if canonical_root is None:
        return {}
    rows: dict[str, JsonObject] = {}
    for manifest in sorted((canonical_root / "records" / "intake").glob("*/manifest.json")):
        try:
            snapshot = load_intake_snapshot(manifest.parent)
        except (OSError, KeyError, TypeError, ValueError, IntakeError) as exc:
            raise ReductionError(
                f"canonical intake snapshot is invalid: {manifest.parent}"
            ) from exc
        for item in snapshot.items:
            value = item.to_dict()
            previous = rows.get(item.stable_id)
            if previous is not None and (
                previous.get("family_representative_id") != value.get("family_representative_id")
                or previous.get("variant") != value.get("variant")
            ):
                raise ReductionError(
                    f"canonical intake family authority conflicts for {item.stable_id}"
                )
            rows[item.stable_id] = value
    return rows


def _replay_reducer(
    ledgers: LedgerPaths,
    *,
    context: AuthorityContext,
    prior_model_records: Sequence[Mapping[str, Any]],
    attempt_records: Sequence[Mapping[str, Any]],
    gate_records: Sequence[Mapping[str, Any]],
    artifact_records: Sequence[Mapping[str, Any]],
    raw_prior_current: Mapping[str, Mapping[str, Any]],
    dependency_current: Mapping[str, Mapping[str, Any]],
    attempt_index: Mapping[str, Mapping[str, Any]],
    gate_index: Mapping[str, Mapping[str, Any]],
) -> CanonicalReducer:
    """Build a write-free reducer positioned immediately before one candidate.

    Parameters
    ----------
    ledgers, context:
        Canonical paths and mandatory active authority.
    prior_model_records, attempt_records, gate_records, artifact_records:
        Hash-validated append-only history.
    raw_prior_current, dependency_current:
        Raw parent-lineage state and already dependency-current representatives.
    attempt_index, gate_index:
        Projection-wide immutable evidence indexes shared across every replay.

    Returns
    -------
    CanonicalReducer
        In-memory reducer whose ordinary ``append_model`` performs the replay.
    """

    replay: Any = CanonicalReducer.__new__(CanonicalReducer)
    replay.ledger_paths = ledgers
    replay._semantic_replay = True
    replay.context = context
    replay.intake_ids = frozenset(context.intake_by_stable_id)
    replay.intake_variant_bindings = {
        stable_id: (
            str(binding.get("representative_stable_id", binding.get("family_representative_id"))),
            str(binding.get("variant_token", binding.get("variant", ""))),
        )
        for stable_id, binding in context.family_bindings.items()
        if isinstance(binding, Mapping)
        and binding.get("binding_state") != "ordinary"
        and binding.get("representative_stable_id", binding.get("family_representative_id"))
        not in {None, stable_id}
    }
    replay._models = _ReplayLedger(prior_model_records)
    replay._attempts = _ReplayLedger(attempt_records)
    replay._gates = _ReplayLedger(gate_records)
    replay._artifacts = _ReplayArtifactLedger(artifact_records)
    replay._current = {stable_id: record for stable_id, record in raw_prior_current.items()}
    replay._validation_current = dependency_current
    replay._attempt_index_cache = attempt_index
    replay._gate_index_cache = gate_index
    replay._projection_cache = None
    return replay


def _referenced_evidence_repasses(
    replay: CanonicalReducer,
    record: Mapping[str, Any],
    attempts: Sequence[Mapping[str, Any]],
    gates: Sequence[Mapping[str, Any]],
) -> None:
    """Replay append admission for every attempt and gate used by one record.

    Parameters
    ----------
    replay:
        Side-effect-free reducer carrying canonical authority inputs.
    record:
        Candidate current model revision.
    attempts, gates:
        Complete canonical evidence histories.

    Raises
    ------
    ReductionError
        If referenced evidence no longer passes its original admission predicate.
    """

    attempt_ids = set(_record_attempt_ids(record))
    attempts_by_id = {str(value.get("attempt_id")): value for value in attempts}
    pending = list(attempt_ids)
    while pending:
        referenced_id = pending.pop()
        referenced = attempts_by_id.get(str(referenced_id))
        if referenced is None:
            continue
        defer_evidence = referenced.get("defer_evidence")
        probe_ids = (
            defer_evidence.get("probe_attempt_ids", [])
            if isinstance(defer_evidence, Mapping)
            else []
        )
        for probe_id in probe_ids:
            normalized = str(probe_id)
            if normalized not in attempt_ids:
                attempt_ids.add(normalized)
                pending.append(normalized)
    for attempt_id in sorted(str(value) for value in attempt_ids):
        attempt = attempts_by_id.get(attempt_id)
        if attempt is None:
            raise ReductionError(f"referenced attempt is missing: {attempt_id}")
        replay.append_attempt(attempt)

    gate_ids = {
        record.get("accuracy_gate", {}).get("gate_id"),
        record.get("fidelity", {}).get("gate_id"),
    } - {None}
    gates_by_id = {str(value.get("gate_id")): value for value in gates}
    for gate_id in sorted(str(value) for value in gate_ids):
        gate = gates_by_id.get(gate_id)
        if gate is None:
            raise ReductionError(f"referenced gate is missing: {gate_id}")
        replay.append_gate(gate)


def project_dependency_current(
    ledgers: LedgerPaths,
    *,
    context: AuthorityContext,
    model_records: Optional[Sequence[Mapping[str, Any]]] = None,
    attempt_records: Optional[Sequence[Mapping[str, Any]]] = None,
    gate_records: Optional[Sequence[Mapping[str, Any]]] = None,
    artifact_records: Optional[Sequence[Mapping[str, Any]]] = None,
) -> DependencyCurrencyProjection:
    """Project the one authoritative current view from current dependency bytes.

    Parameters
    ----------
    ledgers:
        Canonical ledger paths and records-root identity.
    context:
        Mandatory active authority shared with every reducer consumer.
    model_records, attempt_records, gate_records, artifact_records:
        Optional already scanned rows used by the live reducer without another disk read.

    Returns
    -------
    DependencyCurrencyProjection
        Replayed dependency-current records and fail-closed stale reasons.
    """

    models = tuple(model_records) if model_records is not None else scan_jsonl(ledgers.models)
    attempts = (
        tuple(attempt_records) if attempt_records is not None else scan_jsonl(ledgers.attempts)
    )
    gates = tuple(gate_records) if gate_records is not None else scan_jsonl(ledgers.gates)
    artifacts = (
        tuple(artifact_records)
        if artifact_records is not None
        else tuple(scan_jsonl(ledgers.artifacts))
    )
    _validate_persisted_requeue_lineage(models, ledgers, attempts, gates)
    raw_current = _select_current(models)
    by_revision = {str(record.get("record_revision")): index for index, record in enumerate(models)}
    attempts_by_id = {str(record.get("attempt_id")): record for record in attempts}
    gates_by_id = {str(record.get("gate_id")): record for record in gates}
    current: dict[str, JsonObject] = {}
    stale: dict[str, str] = {}
    ordered_current = sorted(
        raw_current.items(), key=lambda value: by_revision[str(value[1].get("record_revision"))]
    )
    prior_current: dict[str, JsonObject] = {}
    prior_models_by_stable_id: dict[str, list[Mapping[str, Any]]] = {}
    cursor = 0
    for stable_id, record in ordered_current:
        try:
            index = by_revision[str(record.get("record_revision"))]
            while cursor < index:
                prior = models[cursor]
                prior_current[str(prior.get("stable_id"))] = deepcopy(dict(prior))
                prior_models_by_stable_id.setdefault(str(prior.get("stable_id")), []).append(prior)
                cursor += 1
            replay = _replay_reducer(
                ledgers,
                context=context,
                prior_model_records=prior_models_by_stable_id.get(stable_id, ()),
                attempt_records=attempts,
                gate_records=gates,
                artifact_records=artifacts,
                raw_prior_current=prior_current,
                dependency_current=current,
                attempt_index=attempts_by_id,
                gate_index=gates_by_id,
            )
            _referenced_evidence_repasses(replay, record, attempts, gates)
            replay_candidate = {
                key: value
                for key, value in record.items()
                if key not in {"record_seq", "record_revision"}
            }
            replay.append_model(replay_candidate)
            family_error = family_variant_currency_error(record, current)
            if family_error is not None:
                raise ReductionError(family_error)
            current[stable_id] = deepcopy(dict(record))
        except (KeyError, ReductionError, TypeError, ValueError) as exc:
            stale[stable_id] = str(exc)
    return DependencyCurrencyProjection(current, stale)


def materialize_current(
    ledgers: LedgerPaths,
    *,
    context: AuthorityContext,
) -> Mapping[str, JsonObject]:
    """Read and deterministically materialize current model revisions.

    Parameters
    ----------
    ledgers:
        Canonical ledger paths.

    Returns
    -------
    Mapping[str, dict[str, Any]]
        Highest valid dependency-current revision per stable ID. Stale family
        variants remain in append-only history but are excluded from this view.
    """

    return project_dependency_current(
        ledgers,
        context=context,
    ).current_records


def default_ledger_paths(root: Union[str, Path]) -> LedgerPaths:
    """Return conventional Slice-A ledger paths below a root directory.

    Parameters
    ----------
    root:
        Canonical records root.

    Returns
    -------
    LedgerPaths
        Models, attempts, and gates JSONL paths.
    """

    base = Path(root)
    return LedgerPaths(
        models=base / "models" / "current-shard.jsonl",
        attempts=base / "attempts" / "local.jsonl",
        gates=base / "gates" / "current-shard.jsonl",
    )
