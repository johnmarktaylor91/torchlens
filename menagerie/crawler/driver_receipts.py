"""Worker request, supervision, receipt projection, and diagnostic boundaries."""

from __future__ import annotations
import ast
import os
import platform
import re
import secrets
import threading
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional, Sequence
from menagerie.crawler.authority import (
    AuthorityDerivationError,
    EnvironmentVerificationToken,
    ExecutableClosureV3,
    ExecutionReadManifestV3,
    RuntimeLookupDirectory,
    RuntimeMember,
    ShutdownInterruptionFact,
    WorkerLease,
    collect_executable_closure_v3,
    compile_execution_read_manifest_v3_from_closure,
    load_current_attempt_proof,
)
from menagerie.crawler.constants import (
    ATTEMPT_SCHEMA_VERSION_V3,
    DEFAULT_FORWARD_TIMEOUT_SECONDS,
    OPERATIONAL_EVENT_SCHEMA_VERSION,
    OperationalEventKind,
    OperationalEventStatus,
)
from menagerie.crawler.identity import (
    canonical_json_bytes,
    hash_bytes,
    stable_hash,
    utc_now,
)
from menagerie.crawler.metadata import (
    input_signature_matches_contract,
)
from menagerie.crawler.modes import classify_observed_mode_receipts
from menagerie.crawler.models import JsonObject
from menagerie.crawler.recordio import JsonlLedger, resolve_attempt_slot, scan_jsonl
from menagerie.crawler.reducer import (
    CanonicalReducer,
    cold_forward_policy,
    expected_standard_asset,
    output_signature_error,
)
from menagerie.crawler.worker_supervisor import (
    VerifiedWorkerResult,
    clear_worker_lease,
    current_boot_id,
    open_worker_lease,
    process_start_token,
    supervise_worker,
    verify_supervised_worker_result,
    worker_result_outer_for_diagnostics,
)
from menagerie.crawler.worker_supervisor import (
    SupervisorObservation,
    SupervisedResult,
)
from menagerie.crawler.driver_contracts import (
    ActivatedHandoffArtifact,
    AuthorArtifact,
    DriverIntegrationError,
    DriverPaused,
    DriverShutdown,
    EnvironmentBinding,
    WorkItem,
)
from menagerie.crawler.driver_progress import (
    _is_sandbox_unavailable,
    _read_json,
    _write_driver_state,
    _write_json_atomic,
)
from menagerie.crawler.driver_models import (
    _assemble_run_model,
    _attempt_policy_satisfied,
    _detected_mode_expansion,
    _driver_facade,
    _driver_failure_attempt,
    _matching_attempts,
    _matching_model_attempts,
    _without_ledger_fields,
)


_WORKER_COMPLETION_PREFIX = "MENAGERIE_WORKER_COMPLETION_V3 "

_EXTERNALLY_CONTROLLED_ATTEMPT_FIELDS = frozenset(
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

_DIAGNOSTIC_REDACTION_MARKER = "externally-controlled-text-v1"

_FORBIDDEN_CACHE_ROOT_NAMES = frozenset(
    {
        ".cache",
        ".keras",
        ".paddle",
        ".torch",
        "huggingface",
        "huggingface-hub",
        "torch-hub",
        "transformers",
    }
)


def _forward_timeout_seconds(proposal: Mapping[str, Any], default_seconds: float) -> float:
    """Return the bounded proposal-declared forward timeout.

    Parameters
    ----------
    proposal:
        Current author proposal.
    default_seconds:
        Normal lane timeout used when no override is declared.

    Returns
    -------
    float
        Effective parent-enforced timeout, never greater than 1,800 seconds.

    Raises
    ------
    DriverIntegrationError
        If a declared override is not a positive bounded integer.
    """

    implementation = proposal.get("proposed_facts", {}).get("implementation", {})
    declared = (
        implementation.get("declared_timeout_seconds")
        if isinstance(implementation, Mapping)
        else None
    )
    if declared is None:
        return default_seconds
    if isinstance(declared, bool) or not isinstance(declared, int) or not 1 <= declared <= 1800:
        raise DriverIntegrationError(
            "implementation.declared_timeout_seconds must be an integer in [1, 1800]"
        )
    return float(declared)


class SupervisedForwardLane:
    """Production forward lane backed by the isolated Slice-B worker supervisor."""

    def __init__(
        self,
        *,
        timeout_seconds: float = DEFAULT_FORWARD_TIMEOUT_SECONDS,
        rss_limit_bytes: int = 12 * 1024**3,
        cwd: Optional[Path] = None,
    ) -> None:
        """Configure parent-enforced resource caps and the read-only source root."""

        self.timeout_seconds = timeout_seconds
        self.rss_limit_bytes = rss_limit_bytes
        self.cwd = cwd

    def forward(
        self,
        artifact: AuthorArtifact,
        environment: EnvironmentBinding,
        cold_runs: int,
        work_root: Path,
        *,
        worker_lock_path: Optional[Path] = None,
        worker_lease_path: Optional[Path] = None,
        run_id: str = "direct-forward",
        shutdown_event: Optional[threading.Event] = None,
        lifecycle_event: Optional[Callable[[str, str, WorkerLease], None]] = None,
        attempt_sink: Optional[Callable[[Mapping[str, Any]], None]] = None,
        attempt_resolver: Optional[Callable[[int, str], Optional[Mapping[str, Any]]]] = None,
        closure: Optional[ExecutableClosureV3] = None,
    ) -> Sequence[Mapping[str, Any]]:
        """Run each cold confirmation and fan its receipt into immutable mode attempts.

        Parameters
        ----------
        artifact, environment, cold_runs, work_root:
            Exact staged proposal, bound runtime, requested confirmations, and work root.
        worker_lock_path, worker_lease_path, run_id, shutdown_event, lifecycle_event,
        attempt_sink, attempt_resolver:
            Existing driver-owned worker lifecycle, persistence, and resume boundaries.
        closure:
            Optional closure already collected once for this artifact in the scheduling pass.

        Returns
        -------
        collections.abc.Sequence[collections.abc.Mapping[str, Any]]
            Authenticated immutable mode attempts.
        """

        if cold_runs < 1:
            raise ValueError("cold_runs must be positive")
        proposal = artifact.proposal
        stable_id = str(proposal["stable_id"])
        authority = environment.environment_authority
        cache = environment.environment_authority_cache
        if authority is None or cache is None:
            raise DriverIntegrationError(
                "live supervised execution requires a lifecycle-owned environment authority"
            )
        if closure is None:
            with cache.currentness_pass(authority) as verification_token:
                closure = _collect_worker_executable_closure(
                    artifact,
                    environment,
                    verification_token=verification_token,
                )
        execution_identity = _driver_facade()._execution_identity(
            artifact, environment, closure_identity=closure.identity
        )
        rung = proposal.get("proposed_facts", {}).get("source_resolution", {}).get("rung")
        reducer_policy = cold_forward_policy(stable_id, rung)
        required_cold_runs = reducer_policy.required_cold_forwards
        effective_timeout = _forward_timeout_seconds(proposal, self.timeout_seconds)
        attempts: list[JsonObject] = []
        observed_receipts: dict[int, dict[str, Mapping[str, object]]] = defaultdict(dict)
        lock_path = worker_lock_path or work_root / "locks" / "worker.lock"
        lease_path = worker_lease_path or work_root / "locks" / "worker-lease.json"
        shutdown = shutdown_event or threading.Event()
        # These modes are gate-authoritative. A worker-discovered expansion is a
        # contract failure below and must be re-proposed/re-gated before it can run.
        modes = tuple(
            str(value) for value in proposal["proposed_facts"]["modes"]["meaningful_modes"]
        )
        for cold_index in range(required_cold_runs):
            for mode in modes:
                if shutdown.is_set():
                    raise DriverShutdown(
                        ShutdownInterruptionFact(
                            invocation_id=run_id,
                            admission_boundary="pre-slot-resolution",
                            stable_id=stable_id,
                            work_id=str(proposal["work_id"]),
                            execution_identity=execution_identity,
                            request_identity=None,
                            lease_id=None,
                            child_pid=None,
                            child_start_token=None,
                            child_pgid=None,
                            signal=None,
                            parent_observation=None,
                            partial_receipt=None,
                        )
                    )
                resolved_attempt = (
                    attempt_resolver(cold_index, mode) if attempt_resolver is not None else None
                )
                if resolved_attempt is not None:
                    attempts.append(dict(resolved_attempt))
                    continue
                root = work_root / stable_id / "forward" / f"cold-{cold_index + 1}" / mode
                request_path = root / "request.json"
                receipt_path = root / "result" / "receipt.json"
                with cache.spawn_verification(authority) as spawn_verification:
                    manifest = _compile_worker_read_manifest(
                        artifact,
                        environment,
                        execution_identity,
                        closure=closure,
                        verification_token=spawn_verification,
                    )
                    request = _worker_request(
                        artifact,
                        root,
                        receipt_path,
                        execution_identity,
                        manifest,
                        cold_index,
                        mode,
                    )
                    _write_json_atomic(request_path, request)
                    request_identity = hash_bytes(request_path.read_bytes())
                    driver_token = process_start_token(os.getpid())
                    if driver_token is None:
                        raise DriverIntegrationError("cannot establish driver process identity")
                    opened = datetime.now(timezone.utc)
                    lease = WorkerLease(
                        lease_id=stable_hash(
                            {
                                "run_id": run_id,
                                "stable_id": stable_id,
                                "work_id": proposal["work_id"],
                                "execution_identity": execution_identity,
                                "cold_index": cold_index,
                                "mode": mode,
                                "request_identity": request_identity,
                            }
                        ),
                        nonce=str(request["request_nonce"]),
                        run_id=run_id,
                        stable_id=stable_id,
                        work_id=str(proposal["work_id"]),
                        request_identity=request_identity,
                        execution_identity=execution_identity,
                        boot_id=current_boot_id(),
                        driver_pid=os.getpid(),
                        driver_start_token=driver_token,
                        child_pid=None,
                        child_start_token=None,
                        child_pgid=None,
                        receipt_path=receipt_path,
                        opened_at=opened.isoformat().replace("+00:00", "Z"),
                        deadline_at=(opened + timedelta(seconds=effective_timeout))
                        .isoformat()
                        .replace("+00:00", "Z"),
                    )

                    def on_opened(value: WorkerLease) -> None:
                        """Append the lock-ordered opened lifecycle event."""

                        if lifecycle_event is not None:
                            lifecycle_event(
                                OperationalEventKind.WORKER_LEASE_OPENED.value,
                                OperationalEventStatus.WORKER_LEASE_OPEN.value,
                                value,
                            )

                    handle = open_worker_lease(
                        lock_path,
                        lease_path,
                        lease,
                        on_lock_acquired=on_opened,
                    )

                    def on_started(value: WorkerLease) -> None:
                        """Append the exact child-start lifecycle event."""

                        if lifecycle_event is not None:
                            lifecycle_event(
                                OperationalEventKind.WORKER_LEASE_STARTED.value,
                                OperationalEventStatus.WORKER_LEASE_ACTIVE.value,
                                value,
                            )

                    result = supervise_worker(
                        request_path,
                        receipt_path,
                        root / "supervisor",
                        timeout_seconds=effective_timeout,
                        rss_limit_bytes=self.rss_limit_bytes,
                        cwd=self.cwd,
                        execution_read_manifest=manifest,
                        worker_lease_handle=handle,
                        shutdown_event=shutdown,
                        on_lease_started=on_started,
                        verification_token=spawn_verification,
                    )
                if result.observation.shutdown_requested:
                    if lifecycle_event is not None:
                        lifecycle_event(
                            OperationalEventKind.WORKER_LEASE_CLOSED.value,
                            OperationalEventStatus.WORKER_LEASE_CLOSED.value,
                            handle.lease,
                        )
                    interrupted_lease = handle.lease
                    clear_worker_lease(handle)
                    partial_receipt = worker_result_outer_for_diagnostics(result)
                    raise DriverShutdown(
                        ShutdownInterruptionFact(
                            invocation_id=run_id,
                            admission_boundary="worker-supervision",
                            stable_id=stable_id,
                            work_id=str(proposal["work_id"]),
                            execution_identity=execution_identity,
                            request_identity=request_identity,
                            lease_id=interrupted_lease.lease_id,
                            child_pid=interrupted_lease.child_pid,
                            child_start_token=interrupted_lease.child_start_token,
                            child_pgid=interrupted_lease.child_pgid,
                            signal=result.observation.signal_number,
                            parent_observation=result.observation.to_dict(),
                            partial_receipt=(
                                dict(partial_receipt)
                                if isinstance(partial_receipt, Mapping)
                                else None
                            ),
                        )
                    )
                generated = _attempts_from_supervised(
                    artifact,
                    result,
                    environment,
                    execution_identity,
                    cold_index,
                    effective_timeout,
                    self.rss_limit_bytes,
                    requested_mode=mode,
                    execution_read_manifest_identity=manifest.manifest_id,
                    diagnostics_root=_diagnostics_root_for_work_root(work_root),
                )
                for attempt in generated:
                    if attempt_sink is not None:
                        attempt_sink(attempt)
                    attempts.append(dict(attempt))
                if lifecycle_event is not None:
                    lifecycle_event(
                        OperationalEventKind.WORKER_LEASE_CLOSED.value,
                        OperationalEventStatus.WORKER_LEASE_CLOSED.value,
                        handle.lease,
                    )
                clear_worker_lease(handle)
                verified, _verification_error = _verified_worker_result(
                    result,
                    proposal,
                    execution_identity,
                    requested_mode=mode,
                )
                raw_per_mode = verified.diagnostic.get("per_mode") if verified is not None else None
                raw_mode_receipt = (
                    raw_per_mode.get(mode) if isinstance(raw_per_mode, Mapping) else None
                )
                if isinstance(raw_mode_receipt, Mapping):
                    observed_receipts[cold_index][mode] = raw_mode_receipt

        observation_failures: list[JsonObject] = []
        for mode in modes:
            signatures = [
                observed_receipts[index][mode].get("output_signature")
                for index in range(required_cold_runs)
                if mode in observed_receipts[index]
            ]
            if len(signatures) == required_cold_runs and any(
                signature != signatures[0] for signature in signatures[1:]
            ):
                observation_failures.append(
                    {
                        "kind": "cold-forward-nondeterminism",
                        "mode": mode,
                        "required_cold_forwards": required_cold_runs,
                    }
                )
        declared_divergence = str(
            proposal.get("proposed_facts", {}).get("modes", {}).get("train_eval_divergence", "none")
        )
        if set(modes) == {"train", "eval"}:
            for cold_index in range(required_cold_runs):
                per_mode = observed_receipts[cold_index]
                if not {"train", "eval"}.issubset(per_mode):
                    continue
                divergence = classify_observed_mode_receipts(per_mode["train"], per_mode["eval"])
                signatures_differ = per_mode["train"].get("output_signature") != per_mode[
                    "eval"
                ].get("output_signature")
                contradicted = (
                    (declared_divergence == "structural" and not signatures_differ)
                    or (declared_divergence != "structural" and signatures_differ)
                    or (divergence is not None and divergence.classification != declared_divergence)
                )
                if contradicted:
                    observation_failures.append(
                        {
                            "kind": "train-eval-divergence-mismatch",
                            "cold_index": cold_index,
                            "declared": declared_divergence,
                            "observed": (
                                divergence.classification
                                if divergence is not None
                                else "signature-compatible"
                            ),
                        }
                    )
        elif declared_divergence != "none":
            observation_failures.append(
                {
                    "kind": "single-mode-divergence-mismatch",
                    "declared": declared_divergence,
                }
            )
        if observation_failures:
            failure = _attempt_error_fields(
                "forward",
                "confirmation-mismatch",
                None,
                "mechanical forward observations contradict the accepted run contract",
                native_crash=False,
                details={"observations": observation_failures},
            )
            failure["root_cause_fingerprint"] = stable_hash(failure)
            for index, attempt in enumerate(attempts):
                if attempt.get("result") != "succeeded":
                    continue
                attempt["result"] = "failed"
                attempt["error"] = deepcopy(failure)
                attempts[index] = _redact_attempt_diagnostics(
                    attempt,
                    None,
                    _diagnostics_root_for_work_root(work_root),
                )
        return tuple(attempts)


def _collect_worker_executable_closure(
    artifact: AuthorArtifact,
    environment: EnvironmentBinding,
    *,
    verification_token: Optional[EnvironmentVerificationToken] = None,
) -> ExecutableClosureV3:
    """Collect the sole executable closure before execution identity derivation.

    Parameters
    ----------
    artifact:
        Privately staged proposed result.
    environment:
        Exact materialized runtime generation.
    verification_token:
        Optional cache-created proof shared by the enclosing pass or spawn.
    Returns
    -------
    ExecutableClosureV3
        Closed exact code/crawler partitions and one sealed environment unit.
    """

    if environment.environment_authority is None:
        raise DriverIntegrationError(
            "live supervised execution requires a sealed EnvironmentAuthorityV1 binding"
        )

    proposal = artifact.proposal
    facts = proposal["proposed_facts"]
    implementation = facts["implementation"]
    raw_manifest = implementation.get("code_manifest", [])
    if not isinstance(raw_manifest, list):
        raise DriverIntegrationError("implementation code manifest is malformed")
    members: list[RuntimeMember] = []
    for row in raw_manifest:
        if not isinstance(row, Mapping):
            raise DriverIntegrationError("implementation code manifest row is malformed")
        path = (artifact.model_dir / str(row.get("path", ""))).resolve()
        if path.suffix in {".py", ".pyi", ".pyx"}:
            kind = "python-source"
        elif path.suffix in {".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp"}:
            kind = "native-source"
        elif path.suffix in {".a", ".dylib", ".pyd", ".so"}:
            kind = "native-library"
        elif path.suffix == ".pyc":
            kind = "python-bytecode"
        else:
            raise DriverIntegrationError(f"execution code member has forbidden suffix: {path}")
        members.append(
            RuntimeMember(
                path=path,
                sha256=str(row.get("sha256")),
                kind=kind,
                provenance="accepted-model-code-manifest",
            )
        )
    selected = (
        expected_standard_asset(facts["external_metadata"]["modality"])
        if implementation.get("recipe_type") == "declarative-library"
        else None
    )
    asset = (
        (Path(selected["path"]), selected["sha256"], selected["asset_id"])
        if selected is not None
        else None
    )
    code_identity = stable_hash(raw_manifest)
    worker_members = tuple(
        RuntimeMember(
            path=path,
            sha256=hash_bytes(path.read_bytes()),
            kind=_runtime_member_kind(path, environment.python_executable.resolve()),
            provenance="crawler-worker-import-closure",
        )
        for path in _crawler_worker_runtime_paths()
    )
    lookup_candidates = {
        Path(__file__).resolve().parents[2],
        environment.prefix.resolve(),
        *(path.resolve() for path in environment.prefix.glob("lib/python*")),
        *(path.resolve() for path in environment.prefix.glob("lib/python*/site-packages")),
    }
    lookup_directories = tuple(
        RuntimeLookupDirectory(path=path, provenance="import-lookup-scaffold")
        for path in sorted(lookup_candidates, key=str)
        if path.is_dir() and not path.is_symlink()
    )
    return collect_executable_closure_v3(
        code_manifest_identity=code_identity,
        environment_authority=environment.environment_authority,
        code_members=tuple(members),
        worker_members=worker_members,
        standard_input_asset=asset,
        lookup_directories=lookup_directories,
        verification_token=verification_token,
    )


def _compile_worker_read_manifest(
    artifact: AuthorArtifact,
    environment: EnvironmentBinding,
    execution_identity: str,
    *,
    closure: Optional[ExecutableClosureV3] = None,
    verification_token: Optional[EnvironmentVerificationToken] = None,
) -> ExecutionReadManifestV3:
    """Bind a verified pre-identity closure into the final worker manifest.

    Parameters
    ----------
    artifact, environment:
        Exact executable artifact and materialized environment generation.
    execution_identity:
        Reducer-compatible execution identity derived from the closure.
    closure:
        Optional already-collected closure reused across meaningful modes.
    verification_token:
        Optional cache-created proof shared across every compilation consumer.

    Returns
    -------
    ExecutionReadManifestV3
        Final exact model/crawler/asset plus sealed-environment capability.
    """

    proposal = artifact.proposal
    collected = closure or _collect_worker_executable_closure(
        artifact,
        environment,
        verification_token=verification_token,
    )
    return compile_execution_read_manifest_v3_from_closure(
        collected,
        stable_id=str(proposal["stable_id"]),
        work_id=str(proposal["work_id"]),
        execution_identity=execution_identity,
        verification_token=verification_token,
    )


def _runtime_member_kind(path: Path, interpreter: Path) -> str:
    """Classify one exact runtime file for execution-manifest v2.

    Parameters
    ----------
    path:
        Canonical regular runtime member.
    interpreter:
        Canonical selected environment interpreter.

    Returns
    -------
    str
        Closed v2 runtime-member kind.
    """

    if path == interpreter:
        return "interpreter"
    suffix = path.suffix.lower()
    if suffix in {".py", ".pyi", ".pyx"}:
        return "python-source"
    if suffix == ".pyc":
        return "python-bytecode"
    if suffix in {".pyd", ".so"}:
        return "native-extension"
    if suffix == ".dylib" or ".so." in path.name.lower():
        return "native-library"
    if (
        path.name
        in {
            "INSTALLER",
            "METADATA",
            "RECORD",
            "WHEEL",
            "entry_points.txt",
            "pyvenv.cfg",
        }
        or path.parent.name == "conda-meta"
    ):
        return "import-metadata"
    return "package-data"


def _crawler_worker_runtime_paths() -> tuple[Path, ...]:
    """Collect the exact recursive crawler-local worker import closure.

    Returns
    -------
    tuple[pathlib.Path, ...]
        Canonical Python files seeded by the worker/supervisor/policy entry points.
    """

    package_root = Path(__file__).resolve().parent
    repository_root = package_root.parents[1]
    pending = [
        package_root / "worker.py",
        package_root / "worker_supervisor.py",
        package_root / "policy.py",
    ]
    members: set[Path] = {
        repository_root / "menagerie" / "__init__.py",
        package_root / "__init__.py",
    }
    while pending:
        path = pending.pop().resolve()
        if path in members or not path.is_file():
            continue
        members.add(path)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, UnicodeError, SyntaxError) as exc:
            raise DriverIntegrationError(f"worker runtime source cannot be parsed: {path}") from exc
        module_names: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                module_names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                module_names.append(node.module)
        for module_name in module_names:
            if not module_name.startswith("menagerie.crawler"):
                continue
            parts = module_name.split(".")[2:]
            candidate = package_root.joinpath(*parts).with_suffix(".py")
            if candidate.is_file():
                pending.append(candidate)
    return tuple(sorted(members, key=str))


def _worker_request(
    artifact: AuthorArtifact,
    scratch_root: Path,
    receipt_path: Path,
    execution_identity: str,
    execution_manifest: ExecutionReadManifestV3,
    cold_index: int,
    mode: str,
) -> JsonObject:
    """Build one explicit-mode v3 request bound to an out-of-band manifest."""

    proposal = artifact.proposal
    facts = proposal["proposed_facts"]
    implementation = facts["implementation"]
    input_contract = deepcopy(dict(facts["input_contract"]))
    if "code_path" in input_contract:
        raise DriverIntegrationError("v3 execution forbids input_contract.code_path presence")
    builder_symbol = input_contract.get("builder_symbol")
    if not isinstance(builder_symbol, str) or not re.fullmatch(
        r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", builder_symbol
    ):
        raise DriverIntegrationError("worker input_contract.builder_symbol is malformed")
    non_tensor_values = input_contract.get("non_tensor_values")
    if not isinstance(non_tensor_values, list):
        raise DriverIntegrationError("worker input_contract.non_tensor_values is malformed")
    for leaf in non_tensor_values:
        value = leaf.get("value") if isinstance(leaf, Mapping) else None
        if isinstance(value, str):
            possible_path = Path(value)
            if possible_path.is_absolute() or ".." in possible_path.parts:
                raise DriverIntegrationError(
                    "worker refuses path-like non-tensor values outside the model root"
                )
            value_type = str(leaf.get("type", "")).casefold().replace("_", "-")
            if value_type in {"file", "file-path", "filepath", "path", "pathlib.path"}:
                resolved_value = (artifact.model_dir / possible_path).resolve()
                if not resolved_value.is_relative_to(artifact.model_dir.resolve()) or not (
                    resolved_value.is_file()
                ):
                    raise DriverIntegrationError(
                        "worker path-valued non-tensor input is not a model-local regular file"
                    )
    if implementation["recipe_type"] == "declarative-library":
        recipe: JsonObject = {
            "kind": "declarative-library",
            "recipe": implementation["library_recipe"],
        }
    else:
        code_path = Path(str(implementation["code_path"]))
        if code_path.is_absolute():
            raise DriverIntegrationError(
                "worker refuses a legacy absolute adapter path; re-propose and re-gate"
            )
        code_path = (artifact.model_dir / code_path).resolve()
        if not code_path.is_relative_to(artifact.model_dir.resolve()):
            raise DriverIntegrationError("worker adapter path escapes private custody")
        manifest_rows: list[JsonObject] = []
        raw_manifest = implementation.get("code_manifest")
        if not isinstance(raw_manifest, list) or not raw_manifest:
            raise DriverIntegrationError("worker adapter code manifest is missing")
        for member in raw_manifest:
            if not isinstance(member, Mapping) or not isinstance(member.get("path"), str):
                raise DriverIntegrationError("worker adapter code manifest path is malformed")
            member_path = Path(str(member["path"]))
            if member_path.is_absolute():
                raise DriverIntegrationError("worker adapter code manifest path must be relative")
            resolved_member = (artifact.model_dir / member_path).resolve()
            if (
                not resolved_member.is_relative_to(artifact.model_dir.resolve())
                or not resolved_member.is_file()
            ):
                raise DriverIntegrationError(
                    "worker adapter code manifest path is not a private regular file"
                )
            manifest_rows.append(
                {
                    "path": str(resolved_member),
                    "identity_path": member["path"],
                    "sha256": member["sha256"],
                }
            )
        recipe = {
            "kind": "typed-adapter",
            "path": str(code_path),
            "adapter_sha256": implementation["code_sha256"],
            "code_manifest": manifest_rows,
            "code_manifest_sha256": stable_hash(implementation["code_manifest"]),
        }
    input_seed = int(input_contract.get("seed", 0))
    try:
        selected_asset = (
            expected_standard_asset(facts["external_metadata"]["modality"])
            if implementation["recipe_type"] == "declarative-library"
            else None
        )
    except ValueError as exc:
        raise DriverIntegrationError(str(exc)) from exc
    standard_input_asset = (
        {"sha256": selected_asset["sha256"], "asset_id": selected_asset["asset_id"]}
        if selected_asset is not None
        else None
    )
    return {
        "protocol_version": "menagerie.crawler.worker-request.v3",
        "stable_id": proposal["stable_id"],
        "work_id": proposal["work_id"],
        "request_nonce": secrets.token_hex(32),
        "execution_read_manifest_identity": execution_manifest.manifest_id,
        "code_manifest_identity": execution_manifest.code_manifest_identity,
        "input_identity": stable_hash(
            {
                "input_contract": input_contract,
                "standard_input_asset": standard_input_asset,
                "seed": input_seed,
            }
        ),
        "recipe": recipe,
        "modality": facts["external_metadata"]["modality"],
        "input_spec": None,
        "input_contract": input_contract,
        "scratch_root": str(scratch_root),
        "receipt_path": str(receipt_path),
        "seed": input_seed,
        "input_seed": input_seed,
        "standard_input_asset": standard_input_asset,
        "device": implementation["device_policy"],
        "framework": implementation["run_framework"],
        "mode": mode,
        "meaningful_modes": list(facts["modes"]["meaningful_modes"]),
        "source_identity": proposal["source_identity"],
        "recipe_revision": proposal["recipe_revision"],
        "recipe_identity_payload": {
            "implementation": {
                key: value for key, value in implementation.items() if key != "recipe_revision"
            },
            "input_contract": deepcopy(input_contract),
            "modes": {
                "meaningful_modes": list(facts["modes"]["meaningful_modes"]),
            },
        },
        "execution_identity": execution_identity,
    }


def _expected_adapter_sha256(proposal: Mapping[str, Any]) -> Optional[str]:
    """Return the accepted adapter digest, or null for declarative recipes.

    Parameters
    ----------
    proposal:
        Current author proposal bound into a worker request.

    Returns
    -------
    str | None
        Exact accepted code digest for typed adapters.
    """

    implementation = proposal.get("proposed_facts", {}).get("implementation", {})
    if not isinstance(implementation, Mapping):
        return None
    if implementation.get("recipe_type") == "declarative-library":
        return None
    value = implementation.get("code_sha256")
    return str(value) if isinstance(value, str) else None


def _expected_code_manifest_sha256(proposal: Mapping[str, Any]) -> Optional[str]:
    """Return the identity-bound recursive code-manifest digest.

    Parameters
    ----------
    proposal:
        Current author proposal bound into a worker request.

    Returns
    -------
    str | None
        Aggregate manifest digest, including the canonical empty manifest for
        declarative recipes, or ``None`` for malformed/unbound adapters.
    """

    implementation = proposal.get("proposed_facts", {}).get("implementation", {})
    manifest = implementation.get("code_manifest") if isinstance(implementation, Mapping) else None
    if isinstance(manifest, list):
        return stable_hash(manifest)
    if isinstance(implementation, Mapping) and implementation.get("recipe_type") == (
        "declarative-library"
    ):
        return stable_hash([])
    return None


def _expected_input_asset_sha256(proposal: Mapping[str, Any]) -> Optional[str]:
    """Return the digest of the selected request-bound standard input asset.

    Parameters
    ----------
    proposal:
        Current author proposal bound into a worker request.

    Returns
    -------
    str | None
        Selected standard-asset digest, or ``None`` for typed dummy calls and
        random fallback.
    """

    facts = proposal.get("proposed_facts", {})
    implementation = facts.get("implementation", {}) if isinstance(facts, Mapping) else {}
    if not isinstance(implementation, Mapping) or implementation.get("recipe_type") != (
        "declarative-library"
    ):
        return None
    external = facts.get("external_metadata", {}) if isinstance(facts, Mapping) else {}
    modality = external.get("modality") if isinstance(external, Mapping) else None
    selected = expected_standard_asset(modality)
    return selected["sha256"] if selected is not None else None


def _expected_input_asset_id(proposal: Mapping[str, Any]) -> Optional[str]:
    """Return the content-addressed selected standard input identifier.

    Parameters
    ----------
    proposal:
        Current author proposal bound into a worker request.

    Returns
    -------
    str | None
        Expected worker ``input_asset`` value.
    """

    facts = proposal.get("proposed_facts", {})
    implementation = facts.get("implementation", {}) if isinstance(facts, Mapping) else {}
    if not isinstance(implementation, Mapping) or implementation.get("recipe_type") != (
        "declarative-library"
    ):
        return None
    external = facts.get("external_metadata", {}) if isinstance(facts, Mapping) else {}
    modality = external.get("modality") if isinstance(external, Mapping) else None
    selected = expected_standard_asset(modality)
    return selected["asset_id"] if selected is not None else None


def _verified_worker_result(
    result: SupervisedResult,
    proposal: Mapping[str, Any],
    execution_identity: str,
    *,
    requested_mode: Optional[str],
) -> tuple[Optional[VerifiedWorkerResult], Optional[str]]:
    """Project a live worker-result.v3 against current driver associations.

    Parameters
    ----------
    result:
        Parent-observed live worker result.
    proposal:
        Current accepted author proposal.
    execution_identity:
        Parent-computed execution identity.
    requested_mode:
        Explicit mode assigned to the subprocess, or ``None`` for all modes.

    Returns
    -------
    tuple[VerifiedWorkerResult | None, str | None]
        Typed projection or its closed protocol error.
    """

    return verify_supervised_worker_result(
        result,
        expected_stable_id=str(proposal.get("stable_id")),
        expected_work_id=str(proposal.get("work_id")),
        expected_source_identity=str(proposal.get("source_identity")),
        expected_recipe_revision=str(proposal.get("recipe_revision")),
        expected_execution_identity=execution_identity,
        expected_code_manifest_identity=_expected_code_manifest_sha256(proposal),
        requested_mode=requested_mode,
    )


def _attempts_from_supervised(
    artifact: AuthorArtifact,
    result: SupervisedResult,
    environment: EnvironmentBinding,
    execution_identity: str,
    cold_index: int,
    timeout_seconds: float,
    rss_limit_bytes: int,
    *,
    requested_mode: Optional[str] = None,
    execution_read_manifest_identity: Optional[str] = None,
    diagnostics_root: Optional[Path] = None,
) -> tuple[JsonObject, ...]:
    """Convert one parent observation and honest receipt into per-mode attempts.

    Parameters
    ----------
    artifact, result, environment, execution_identity, cold_index:
        Bound worker request, parent result, environment, and run identity facts.
    timeout_seconds, rss_limit_bytes:
        Parent-enforced resource limits.
    requested_mode:
        Single mode isolated in this subprocess, when applicable.
    diagnostics_root:
        Gitignored local root for exact external-text diagnostic sidecars. Production
        callers always provide ``.crawl-local/diagnostics``; tests that only exercise
        receipt classification may omit it when all controlled fields are empty.

    Returns
    -------
    tuple[dict[str, Any], ...]
        Canonical attempts containing only redacted references to model-controlled text.
    """

    proposal = artifact.proposal
    facts = proposal["proposed_facts"]
    verified, verification_error = _verified_worker_result(
        result,
        proposal,
        execution_identity,
        requested_mode=requested_mode,
    )
    receipt = deepcopy(verified.diagnostic) if verified is not None else {}
    policy_value = receipt.get("policy_observation", {})
    policy = dict(policy_value) if isinstance(policy_value, Mapping) else {}
    policy["cache_read_attempted"] = bool(policy.get("cache_read_attempted")) or (
        _parent_cache_read_attempted(policy)
    )
    receipt["policy_observation"] = policy
    envelope_error = verification_error or _receipt_envelope_error(
        result, proposal, execution_identity, requested_mode=requested_mode
    )
    if policy.get("cache_read_attempted"):
        envelope_error = "failed:policy-cache-read"
    effective_result = (
        result
        if envelope_error is None
        else SupervisedResult(result.observation, None, envelope_error, None)
    )
    per_mode = receipt.get("per_mode", {})
    receipt_modes = receipt.get("meaningful_modes", [])
    detected_modes = tuple(
        str(value)
        for value in receipt.get("detected_meaningful_modes", [])
        if isinstance(value, str)
    )
    proposal_mode_set = {str(value) for value in facts["modes"]["meaningful_modes"]}
    missing_proposal_modes = tuple(
        mode for mode in ("train", "eval") if mode in set(detected_modes) - proposal_mode_set
    )
    modes = (
        (requested_mode,)
        if requested_mode is not None
        else tuple(
            dict.fromkeys(
                [
                    *(str(value) for value in facts["modes"]["meaningful_modes"]),
                    *(str(value) for value in receipt_modes if isinstance(receipt_modes, list)),
                ]
            )
        )
    )
    attempts: list[JsonObject] = []
    declared_modes = tuple(str(value) for value in facts["modes"]["meaningful_modes"])
    for mode in modes:
        mode_index = declared_modes.index(mode) if mode in declared_modes else len(declared_modes)
        mode_receipt = per_mode.get(mode, {}) if isinstance(per_mode, Mapping) else {}
        succeeded = bool(
            envelope_error is None
            and verified is not None
            and result.observation.exit_code == 0
            and result.observation.signal_number is None
            and verified.raw_award_receipt is not None
            and verified.raw_observation is not None
            and verified.parent_attestation is not None
            and mode_receipt.get("constructor_started")
            and mode_receipt.get("constructor_completed")
            and mode_receipt.get("input_completed")
            and mode_receipt.get("forward_started")
            and mode_receipt.get("forward_completed")
            and input_signature_matches_contract(
                mode_receipt.get("input_signature"), facts["input_contract"]
            )
        )
        attempt_id = stable_hash(
            {
                "work_id": proposal["work_id"],
                "execution_identity": execution_identity,
                "cold_index": cold_index,
                "mode": mode,
            }
        )
        observation = result.observation
        error: Optional[JsonObject] = None
        attempt_stage = "forward"
        attempt_mode: Optional[str] = mode
        if not succeeded:
            if missing_proposal_modes:
                failure = _attempt_error_fields(
                    "input",
                    "contract-invalid",
                    None,
                    "worker detected meaningful modes absent from the gated proposal",
                    native_crash=False,
                    details={
                        "route": "recipe-and-gate-revision-required",
                        "proposal_meaningful_modes": list(declared_modes),
                        "detected_meaningful_modes": list(detected_modes),
                        "missing_proposal_modes": list(missing_proposal_modes),
                    },
                )
            else:
                failure = _supervised_failure(
                    effective_result,
                    receipt,
                    mode_receipt,
                    policy,
                    worker_result_present=verified is not None,
                )
            attempt_stage = failure["stage"]
            attempt_mode = mode if attempt_stage == "forward" else None
            error = {
                **failure,
                "root_cause_fingerprint": stable_hash(failure),
            }
        worker_receipt = {
            "present": verified is not None,
            "receipt_sha256": (verified.result_sha256 if verified is not None else None),
            "observed_recipe_revision": receipt.get("observed_recipe_revision"),
            "observed_adapter_sha256": receipt.get("observed_adapter_sha256"),
            "observed_code_manifest_sha256": receipt.get("observed_code_manifest_sha256"),
            "observed_input_asset_sha256": receipt.get("observed_input_asset_sha256"),
            "constructor_started": bool(mode_receipt.get("constructor_started")),
            "constructor_completed": bool(mode_receipt.get("constructor_completed")),
            "input_completed": bool(mode_receipt.get("input_completed")),
            "forward_started": bool(mode_receipt.get("forward_started")),
            "forward_completed": bool(mode_receipt.get("forward_completed")),
            "mode": mode,
            "input_signature": mode_receipt.get("input_signature"),
            "output_signature": mode_receipt.get("output_signature"),
            "output_value_sha256": mode_receipt.get("output_value_sha256"),
            "input_kind": mode_receipt.get("input_kind"),
            "input_asset": mode_receipt.get("input_asset"),
            "input_note": str(mode_receipt.get("input_note") or "worker receipt unavailable"),
            "parameter_count_total": mode_receipt.get("parameter_count_total"),
            "parameter_count_trainable": mode_receipt.get("parameter_count_trainable"),
            "native_framework": mode_receipt.get("native_framework"),
            "delegated_method": mode_receipt.get("delegated_method"),
            "constructor_seconds": mode_receipt.get("constructor_seconds"),
            "forward_seconds": mode_receipt.get("forward_seconds"),
        }
        if succeeded:
            if verified is None or verified.raw_observation is None:
                raise DriverIntegrationError("successful worker result lacks a raw award receipt")
            worker_receipt = deepcopy(verified.raw_observation)
        parent_attestation = (
            deepcopy(verified.parent_attestation)
            if verified is not None
            else deepcopy(result.parent_attestation)
        )
        attempt: JsonObject = {
            "schema_version": ATTEMPT_SCHEMA_VERSION_V3,
            "attempt_id": attempt_id,
            "work_id": proposal["work_id"],
            "stable_id": proposal["stable_id"],
            "attempt_no": cold_index * len(declared_modes) + mode_index + 1,
            "parent_attempt_id": None,
            "actor": "worker",
            "stage": attempt_stage,
            "mode": attempt_mode,
            "started_at": (
                parent_attestation.get("started_at")
                if isinstance(parent_attestation, Mapping)
                else proposal["created_at"]
            ),
            "finished_at": (
                parent_attestation.get("finished_at")
                if isinstance(parent_attestation, Mapping)
                else utc_now()
            ),
            "result": "succeeded" if succeeded else "failed",
            "attempted_rungs": [facts["source_resolution"]["rung"]],
            "retries": {
                "stage_attempt": cold_index + 1,
                "root_cause_repeat": 0,
                "author_round": 1,
                "gate_round": 1,
            },
            "identities": {
                "source": proposal["source_identity"],
                "evidence": proposal["evidence_identity"],
                "recipe": proposal["recipe_revision"],
                "environment": environment.env_generation,
                "execution": execution_identity,
                "runner": _driver_facade()._runner_identity(facts["external_metadata"]["modality"]),
                "author_prompt": proposal["author"]["prompt_sha256"],
                "checker_prompt": _driver_facade()._checker_prompt_hash(),
            },
            "environment": {
                "family": environment.family,
                "target": environment.target,
                "env_id": str(environment.prefix),
                "lock_sha256": environment.lock_sha256,
                "resolved_export_sha256": environment.resolved_export_sha256,
                "python": environment.python_version,
                "packages_manifest_sha256": environment.packages_manifest_sha256,
                "compiler_identity": environment.compiler_identity,
                "sdk_identity": environment.sdk_identity,
                "authority_epoch": environment.authority_epoch,
                "base_environment_generation": environment.base_environment_generation,
                "environment_content_sha256": environment.environment_content_sha256,
                "environment_authority_id": environment.environment_authority_id,
                "selected_interpreter_relative_path": (
                    environment.selected_interpreter_relative_path
                ),
                "selected_interpreter_digest": environment.selected_interpreter_digest,
                "external_escape_records": [
                    {
                        "path": str(record.path),
                        "sha256": record.sha256,
                        "kind": record.kind,
                    }
                    for record in environment.external_escape_records
                ],
            },
            "host": {
                "machine_id": platform.node() or "unknown-machine",
                "os": platform.system().lower(),
                "os_build": platform.version(),
                "architecture": platform.machine(),
                "cpu": platform.processor() or "unknown-cpu",
                "ram_bytes": _physical_memory_bytes(),
                "accelerator": None,
                "accelerator_runtime": None,
            },
            "invocation": {
                "argv": list(observation.argv),
                "cwd": observation.cwd,
                "safe_env": {"MENAGERIE_EXECUTION_OFFLINE": "1"},
                "seed": int(facts["input_contract"].get("seed", 0)),
                "device": facts["implementation"]["device_policy"],
                "mode": attempt_mode,
                "network_policy": "offline",
                "timeout_seconds": timeout_seconds,
                "rss_limit_bytes": rss_limit_bytes,
                "scratch_limit_bytes": rss_limit_bytes,
            },
            "worker_receipt": worker_receipt,
            "supervisor_observation": {
                "exit_code": observation.exit_code,
                "signal": observation.signal_number,
                "wall_seconds": observation.wall_seconds,
                "cpu_seconds": observation.cpu_seconds,
                "peak_rss_bytes": observation.peak_rss_bytes,
                "stdout_sha256": observation.stdout_sha256,
                "stdout_bytes": observation.stdout_bytes,
                "stdout_tail": observation.stdout_tail,
                "stdout_completion_line": (
                    _attested_completion_line(observation.stdout_tail) if succeeded else None
                ),
                "stderr_sha256": observation.stderr_sha256,
                "stderr_bytes": observation.stderr_bytes,
                "stderr_tail": observation.stderr_tail,
                "full_log_local_path": observation.stderr_path,
                "full_log_retention": "campaign",
            },
            "policy_observation": {
                "network_attempted": bool(policy.get("network_attempted")),
                "socket_targets": list(policy.get("socket_targets", [])),
                "checkpoint_or_weight_read_attempted": bool(
                    policy.get("checkpoint_or_weight_read_attempted")
                ),
                "checkpoint_paths": list(policy.get("checkpoint_paths", [])),
                "write_outside_scratch_attempted": bool(
                    policy.get("write_outside_scratch_attempted")
                ),
                "write_paths": list(policy.get("write_paths", [])),
                "credentials_present": bool(policy.get("credentials_present")),
                "torchlens_import_attempted": bool(policy.get("torchlens_import_attempted")),
                "cache_read_attempted": bool(policy.get("cache_read_attempted")),
            },
            "error": error,
            "defer_evidence": None,
            "capability_observation": None,
            "execution_read_manifest_identity": (
                execution_read_manifest_identity
                or stable_hash("direct-supervised-execution-manifest")
            ),
            "raw_award_receipt": (
                deepcopy(verified.raw_award_receipt) if succeeded and verified is not None else None
            ),
            "raw_award_receipt_sha256": (
                verified.raw_award_receipt_sha256 if succeeded and verified is not None else None
            ),
            "parent_attestation": parent_attestation,
            "unattested_partial": (None if succeeded else deepcopy(result.unattested_partial)),
        }
        attempts.append(_redact_attempt_diagnostics(attempt, observation, diagnostics_root))
    return tuple(attempts)


def _attested_completion_line(stdout_tail: str) -> Optional[str]:
    """Return the final TorchLens-owned completion marker from a bounded stdout tail.

    Parameters
    ----------
    stdout_tail:
        Live parent-observed stdout tail.

    Returns
    -------
    str | None
        Final completion line, or ``None`` when no marker is present.
    """

    lines = stdout_tail.splitlines()
    if not lines or not lines[-1].startswith(_WORKER_COMPLETION_PREFIX):
        return None
    return lines[-1]


def _diagnostic_relative_path(diagnostics_root: Path, attempt_id: str) -> str:
    """Return a checkpoint-safe repository-relative diagnostic sidecar locator.

    Parameters
    ----------
    diagnostics_root:
        Local diagnostics root.
    attempt_id:
        Stable attempt identifier used as the sidecar filename.

    Returns
    -------
    str
        Relative locator rooted at ``.crawl-local``.

    Raises
    ------
    DriverIntegrationError
        If diagnostics are not rooted below the gitignored runtime directory.
    """

    resolved = diagnostics_root.resolve()
    if ".crawl-local" not in resolved.parts:
        raise DriverIntegrationError("diagnostic sidecars must live below .crawl-local")
    index = max(index for index, part in enumerate(resolved.parts) if part == ".crawl-local")
    relative_root = Path(*resolved.parts[index:])
    return (relative_root / f"{attempt_id}.json").as_posix()


def _diagnostics_root_for_work_root(work_root: Path) -> Path:
    """Return a campaign-local C-07 sidecar root below ``.crawl-local``.

    Parameters
    ----------
    work_root:
        Driver work-envelope root below its runtime directory.

    Returns
    -------
    pathlib.Path
        Production's sibling diagnostics directory, or an isolated nested
        ``.crawl-local`` directory for an explicitly relocated dry-run runtime.
    """

    runtime_root = work_root.parent
    if ".crawl-local" in runtime_root.resolve().parts:
        return runtime_root / "diagnostics"
    return runtime_root / ".crawl-local" / "diagnostics"


def _redact_attempt_diagnostics(
    attempt: JsonObject,
    observation: Optional[SupervisorObservation],
    diagnostics_root: Optional[Path],
) -> JsonObject:
    """Persist exact local diagnostics and redact their canonical attempt projections.

    Parameters
    ----------
    attempt:
        Newly assembled attempt before canonical persistence.
    observation:
        Live :class:`SupervisorObservation` retaining exact bounded stream tails and paths,
        or ``None`` for a driver-originated failure with no child process.
    diagnostics_root:
        Gitignored local sidecar root. It may be omitted only when every controlled value
        is empty, as in receipt-contract unit tests.

    Returns
    -------
    dict[str, Any]
        Attempt whose externally controlled values are explicit redaction references.

    Raises
    ------
    DriverIntegrationError
        If nonempty diagnostics would otherwise be lost.
    """

    attempt_id = str(attempt["attempt_id"])
    controlled: dict[str, Any] = {}
    has_nonempty_controlled = False

    def collect(value: Any, location: str = "$") -> None:
        """Collect every external-text field before replacing it."""

        nonlocal has_nonempty_controlled
        if isinstance(value, Mapping):
            for key, nested in value.items():
                nested_location = f"{location}.{key}"
                if key in _EXTERNALLY_CONTROLLED_ATTEMPT_FIELDS:
                    if _is_diagnostic_redaction_reference(nested):
                        continue
                    controlled[nested_location] = deepcopy(nested)
                    if nested is not None and nested != "":
                        has_nonempty_controlled = True
                collect(nested, nested_location)
        elif isinstance(value, list):
            for index, nested in enumerate(value):
                collect(nested, f"{location}[{index}]")

    collect(attempt)
    if not has_nonempty_controlled:
        return attempt
    if diagnostics_root is None:
        if has_nonempty_controlled:
            raise DriverIntegrationError(
                "externally controlled attempt text requires a local diagnostic sidecar"
            )
        return attempt

    local_path = _diagnostic_relative_path(diagnostics_root, attempt_id)
    sidecar_path = diagnostics_root / f"{attempt_id}.json"
    supervisor_value = attempt.get("supervisor_observation", {})
    supervisor = supervisor_value if isinstance(supervisor_value, Mapping) else {}
    sidecar: JsonObject = {
        "schema_version": "menagerie.crawler.local-diagnostics.v1",
        "attempt_id": attempt_id,
        "stdout": {
            "stream_sha256": (
                observation.stdout_sha256
                if observation is not None
                else supervisor.get("stdout_sha256")
            ),
            "stream_bytes": (
                observation.stdout_bytes
                if observation is not None
                else supervisor.get("stdout_bytes", 0)
            ),
            "tail": observation.stdout_tail
            if observation is not None
            else supervisor.get("stdout_tail", ""),
            "full_log_path": (
                observation.stdout_path
                if observation is not None
                else supervisor.get("full_log_local_path")
            ),
        },
        "stderr": {
            "stream_sha256": (
                observation.stderr_sha256
                if observation is not None
                else supervisor.get("stderr_sha256")
            ),
            "stream_bytes": (
                observation.stderr_bytes
                if observation is not None
                else supervisor.get("stderr_bytes", 0)
            ),
            "tail": observation.stderr_tail
            if observation is not None
            else supervisor.get("stderr_tail", ""),
            "full_log_path": (
                observation.stderr_path
                if observation is not None
                else supervisor.get("full_log_local_path")
            ),
        },
        "externally_controlled_fields": controlled,
    }
    _write_json_atomic(sidecar_path, sidecar)
    sidecar_path.chmod(0o600)

    def redact(value: Any, location: str = "$") -> Any:
        """Replace controlled values with hash-bound local references."""

        if isinstance(value, Mapping):
            redacted: dict[str, Any] = {}
            for key, nested in value.items():
                nested_location = f"{location}.{key}"
                if (
                    key in _EXTERNALLY_CONTROLLED_ATTEMPT_FIELDS
                    and nested is not None
                    and nested != ""
                    and not _is_diagnostic_redaction_reference(nested)
                ):
                    reference: dict[str, Any] = {
                        "redaction": _DIAGNOSTIC_REDACTION_MARKER,
                        "content_sha256": hash_bytes(canonical_json_bytes(nested)),
                        "local_path": local_path,
                        "diagnostic_key": nested_location,
                    }
                    if key == "stdout_tail":
                        reference["stream_sha256"] = (
                            observation.stdout_sha256
                            if observation is not None
                            else supervisor.get("stdout_sha256")
                        )
                    elif key == "stderr_tail":
                        reference["stream_sha256"] = (
                            observation.stderr_sha256
                            if observation is not None
                            else supervisor.get("stderr_sha256")
                        )
                    redacted[key] = reference
                else:
                    redacted[key] = redact(nested, nested_location)
            return redacted
        if isinstance(value, list):
            return [redact(nested, f"{location}[{index}]") for index, nested in enumerate(value)]
        return value

    redacted_attempt = redact(attempt)
    if not isinstance(redacted_attempt, dict):
        raise AssertionError("attempt redaction must preserve the top-level object")
    redacted_supervisor = redacted_attempt.get("supervisor_observation")
    if isinstance(redacted_supervisor, dict):
        redacted_supervisor["full_log_local_path"] = local_path
    return redacted_attempt


def _is_diagnostic_redaction_reference(value: Any) -> bool:
    """Return whether a value is an already-redacted C-07 sidecar reference.

    Parameters
    ----------
    value:
        Candidate diagnostic field value.

    Returns
    -------
    bool
        Whether the value carries the closed redaction marker and required locator fields.
    """

    return bool(
        isinstance(value, Mapping)
        and value.get("redaction") == _DIAGNOSTIC_REDACTION_MARKER
        and all(
            isinstance(value.get(field), str) and value.get(field)
            for field in ("content_sha256", "local_path", "diagnostic_key")
        )
    )


def _parent_cache_read_attempted(policy: Mapping[str, Any]) -> bool:
    """Detect forbidden cache roots in parent-owned successful-read telemetry.

    Parameters
    ----------
    policy:
        Receipt policy merged with parent-owned syscall path observations.

    Returns
    -------
    bool
        True when a recorded read path falls below a closed cache root.
    """

    paths = policy.get("checkpoint_paths", [])
    if not isinstance(paths, list):
        return False
    for value in paths:
        if not isinstance(value, str):
            continue
        parts = {part.lower() for part in Path(value).parts}
        if parts & _FORBIDDEN_CACHE_ROOT_NAMES:
            return True
        normalized = value.replace("\\", "/").lower()
        if "/.crawl-local/caches/" in normalized or "/caches/" in normalized:
            return True
    return False


def _receipt_envelope_error(
    result: SupervisedResult,
    proposal: Mapping[str, Any],
    execution_identity: str,
    *,
    requested_mode: Optional[str] = None,
) -> Optional[str]:
    """Return a protocol error unless the requested child envelope is current.

    Parameters
    ----------
    result:
        Parent-owned supervisor observation and child receipt.
    proposal:
        Current accepted author proposal.
    execution_identity:
        Parent-computed execution identity.
    requested_mode:
        Explicit single mode assigned to this subprocess, or ``None`` for a
        legacy all-modes request.
    """

    verified, verification_error = _verified_worker_result(
        result,
        proposal,
        execution_identity,
        requested_mode=requested_mode,
    )
    if verified is None:
        return verification_error or "invalid-receipt:worker-result-v3"
    receipt = verified.diagnostic
    successful_exit = result.observation.exit_code == 0 and result.observation.signal_number is None
    expected_adapter = _expected_adapter_sha256(proposal)
    expected_manifest = _expected_code_manifest_sha256(proposal)
    observed_asset_pair = (
        receipt.get("observed_input_asset_sha256"),
        next(
            (
                value.get("input_asset")
                for value in receipt.get("per_mode", {}).values()
                if isinstance(value, Mapping)
            ),
            None,
        ),
    )
    expected_asset_pair = (
        _expected_input_asset_sha256(proposal),
        _expected_input_asset_id(proposal),
    )
    if (
        receipt.get("stable_id") != proposal.get("stable_id")
        or receipt.get("source_identity") != proposal.get("source_identity")
        or receipt.get("recipe_revision") != proposal.get("recipe_revision")
        or receipt.get("execution_identity") != execution_identity
        or (
            receipt.get("observed_recipe_revision") is not None
            and receipt.get("observed_recipe_revision") != proposal.get("recipe_revision")
        )
        or (
            receipt.get("observed_adapter_sha256") is not None
            and receipt.get("observed_adapter_sha256") != expected_adapter
        )
        or (
            receipt.get("observed_code_manifest_sha256") is not None
            and receipt.get("observed_code_manifest_sha256") != expected_manifest
        )
        or observed_asset_pair
        not in {
            (None, None),
            expected_asset_pair,
            (expected_asset_pair[0], None) if not successful_exit else expected_asset_pair,
        }
    ):
        return "invalid-receipt:identity"
    modes = receipt.get("meaningful_modes")
    detected = receipt.get("detected_meaningful_modes")
    declared = receipt.get("declared_meaningful_modes")
    per_mode = receipt.get("per_mode")
    if (
        not isinstance(modes, list)
        or not isinstance(detected, list)
        or not isinstance(declared, list)
        or any(not isinstance(value, str) for value in (*modes, *detected, *declared))
        or len(modes) != len(set(modes))
        or len(detected) != len(set(detected))
        or len(declared) != len(set(declared))
    ):
        return "invalid-receipt:mode-envelope"
    proposal_modes = set(
        str(value)
        for value in proposal.get("proposed_facts", {}).get("modes", {}).get("meaningful_modes", [])
    )
    mode_set = set(modes)
    detected_set = set(detected)
    declared_set = set(declared)
    valid_modes = {"train", "eval"}
    if (
        not mode_set <= valid_modes
        or not detected_set <= valid_modes
        or declared_set != proposal_modes
        or mode_set != proposal_modes | detected_set
    ):
        return "invalid-receipt:mode-envelope"
    if detected_set - proposal_modes:
        return "invalid-receipt:meaningful-mode-contract"
    if not isinstance(per_mode, Mapping):
        return "invalid-receipt:mode-envelope"
    receipt_mode = receipt.get("mode")
    validated_modes: tuple[str, ...]
    if not successful_exit:
        if requested_mode is not None and (
            requested_mode not in proposal_modes
            or receipt_mode != requested_mode
            or not set(per_mode) <= {requested_mode}
        ):
            return "invalid-receipt:mode-envelope"
        if requested_mode is None and receipt_mode is not None:
            return "invalid-receipt:mode-envelope"
        validated_modes = tuple(str(mode) for mode in per_mode)
    elif requested_mode is not None:
        if verified.parent_attestation is None:
            return "missing-parent-success-attestation"
        if (
            requested_mode not in proposal_modes
            or requested_mode not in mode_set
            or receipt_mode != requested_mode
            or set(per_mode) != {requested_mode}
        ):
            return "invalid-receipt:mode-envelope"
        validated_modes = (requested_mode,)
    else:
        if verified.parent_attestation is None:
            return "missing-parent-success-attestation"
        if receipt_mode is not None or set(per_mode) != mode_set:
            return "invalid-receipt:mode-envelope"
        validated_modes = tuple(str(mode) for mode in modes)
    required_mode = {
        "mode",
        "constructor_started",
        "constructor_completed",
        "input_completed",
        "input_signature",
        "forward_started",
        "forward_completed",
        "output_signature",
        "error",
    }
    for mode in validated_modes:
        value = per_mode.get(mode)
        if not isinstance(value, Mapping) or not required_mode <= set(value):
            return "invalid-receipt:incomplete-mode"
        if not successful_exit:
            if value.get("mode") != mode:
                return "invalid-receipt:mode-envelope"
            continue
        if (
            value.get("mode") != mode
            or value.get("error") is not None
            or not value.get("constructor_started")
            or not value.get("constructor_completed")
            or not value.get("input_completed")
            or not value.get("forward_started")
            or not value.get("forward_completed")
            or not input_signature_matches_contract(
                value.get("input_signature"),
                proposal.get("proposed_facts", {}).get("input_contract", {}),
            )
            or (
                receipt.get("observed_input_asset_sha256"),
                value.get("input_asset"),
            )
            not in {(None, None), expected_asset_pair}
        ):
            return "invalid-receipt:incomplete-mode"
        if output_signature_error(value.get("output_signature")) is not None:
            return "invalid-receipt:output-signature"
    policy = receipt.get("policy_observation")
    required_policy = {
        "network_attempted",
        "socket_targets",
        "checkpoint_or_weight_read_attempted",
        "checkpoint_paths",
        "write_outside_scratch_attempted",
        "write_paths",
        "credentials_present",
        "torchlens_import_attempted",
        "cache_read_attempted",
    }
    if not isinstance(policy, Mapping) or not required_policy <= set(policy):
        return "invalid-receipt:policy-envelope"
    if successful_exit and verified.raw_award_receipt is None:
        return "missing-authenticated-raw-award-receipt"
    if successful_exit and receipt.get("error") is not None:
        return "invalid-receipt:success-with-error"
    if not successful_exit and not (
        isinstance(receipt.get("error"), Mapping)
        or any(
            isinstance(value, Mapping) and isinstance(value.get("error"), Mapping)
            for value in per_mode.values()
        )
        or any(policy.get(field) for field in required_policy if field.endswith("attempted"))
    ):
        return "invalid-receipt:failure-without-error"
    return None


def _supervised_failure(
    result: SupervisedResult,
    receipt: Mapping[str, Any],
    mode_receipt: Mapping[str, Any],
    policy: Mapping[str, Any],
    *,
    worker_result_present: bool,
) -> JsonObject:
    """Classify a failed worker observation into its actual closed stage and reason.

    Parameters
    ----------
    result:
        Parent-observed result, possibly replaced with a projection error.
    receipt, mode_receipt, policy:
        Verified non-awarding diagnostic facts.
    worker_result_present:
        Whether the central v3 projection accepted an outer worker result.

    Returns
    -------
    dict[str, Any]
        Closed attempt failure fields.
    """

    policy_reasons = (
        ("network_attempted", "network-attempt"),
        ("checkpoint_or_weight_read_attempted", "checkpoint-read"),
        ("cache_read_attempted", "checkpoint-read"),
        ("write_outside_scratch_attempted", "write-outside-scratch"),
        ("credentials_present", "credentials-exposed"),
        ("torchlens_import_attempted", "torchlens-import"),
    )
    for field, reason in policy_reasons:
        if policy.get(field):
            return _attempt_error_fields(
                "policy",
                reason,
                receipt.get("error"),
                f"worker policy violation: {reason}",
                native_crash=False,
                details={"policy_field": field},
            )
    observation = result.observation
    if observation.timed_out:
        return _attempt_error_fields(
            "resource",
            "timeout",
            None,
            "worker exceeded the parent wall timeout",
            native_crash=False,
            details={"wall_seconds": observation.wall_seconds},
        )
    if observation.rss_exceeded:
        return _attempt_error_fields(
            "resource",
            "rss-cap",
            None,
            "worker exceeded the parent RSS limit",
            native_crash=False,
            details={"peak_rss_bytes": observation.peak_rss_bytes},
        )
    if observation.signal_number is not None:
        return _attempt_error_fields(
            "runner",
            "signal",
            None,
            f"worker terminated by signal {observation.signal_number}",
            native_crash=True,
            details={"signal": observation.signal_number},
        )
    if not worker_result_present:
        if result.receipt_error in {"failed:policy", "failed:sandbox-unavailable"}:
            return _attempt_error_fields(
                "policy",
                "sandbox-unavailable-v1",
                None,
                "required operating-system sandbox is unavailable",
                native_crash=False,
                details={"receipt_error": result.receipt_error},
            )
        reason = (
            "missing-receipt" if result.receipt_error == "missing-receipt" else "protocol-violation"
        )
        return _attempt_error_fields(
            "runner",
            reason,
            None,
            str(result.receipt_error or "worker receipt unavailable"),
            native_crash=False,
            details={"receipt_error": result.receipt_error},
        )
    global_error = receipt.get("error")
    if not receipt.get("constructor_started"):
        return _attempt_error_fields(
            "import",
            "import-exception",
            global_error,
            "worker failed while loading the recipe",
            native_crash=False,
            details={"receipt_error": global_error},
        )
    if not receipt.get("constructor_completed"):
        return _attempt_error_fields(
            "constructor",
            "exception",
            global_error,
            "model constructor failed",
            native_crash=False,
            details={"receipt_error": global_error},
        )
    if not receipt.get("input_completed"):
        return _attempt_error_fields(
            "input",
            "generation-exception",
            global_error,
            "dummy input generation failed",
            native_crash=False,
            details={"receipt_error": global_error},
        )
    return _attempt_error_fields(
        "forward",
        "mode-run",
        mode_receipt.get("error"),
        "meaningful mode forward failed",
        native_crash=False,
        details={"mode_error": mode_receipt.get("error")},
    )


def _attempt_error_fields(
    stage: str,
    reason_code: str,
    worker_error: Any,
    fallback_message: str,
    *,
    native_crash: bool,
    details: Mapping[str, Any],
) -> JsonObject:
    """Build complete attempt error fields from optional worker Python evidence."""

    error = worker_error if isinstance(worker_error, Mapping) else {}
    traceback_text = error.get("traceback")
    return {
        "stage": stage,
        "reason_code": reason_code,
        "exception_type": error.get("exception_type"),
        "message": str(error.get("message") or fallback_message),
        "traceback": traceback_text,
        "no_traceback_reason": None if traceback_text else "no Python traceback was available",
        "native_crash": native_crash,
        "details": dict(details),
    }


def _physical_memory_bytes() -> int:
    """Return host physical memory when POSIX page counters are available."""

    try:
        return int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError):
        return 0


class ReceiptDriverMixin:
    """Driver-side worker lifecycle, receipt, and run-award orchestration."""

    if TYPE_CHECKING:
        _reduced: int

        def __getattr__(self, name: str) -> Any:
            """Describe collaborators supplied by the concrete driver facade."""

            raise AttributeError(name)

    def _append_worker_lifecycle_event(
        self,
        operational: JsonlLedger,
        *,
        event_kind: str,
        status: str,
        lease_id: str,
        stable_id: str,
        details: Mapping[str, Any],
    ) -> None:
        """Append one idempotent worker lifecycle event."""

        identity = stable_hash(
            {"event_kind": event_kind, "lease_id": lease_id, "details": dict(details)}
        )[7:31]
        operational.append(
            {
                "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
                "event_id": f"worker-{identity}",
                "created_at": self.dependencies.clock(),
                "event_kind": event_kind,
                "status": status,
                "provider": None,
                "observed_response": None,
                "reset_at": None,
                "queued_work_counts": {"models": 0},
                "current_environment": None,
                "run_id": self.config.run_id,
                "machine_id": self.config.machine_id,
                "details": {"lease_id": lease_id, "stable_id": stable_id, **dict(details)},
            }
        )

    def _ensure_pending_run_anchors(
        self,
        work: Sequence[WorkItem],
        artifacts: dict[str, AuthorArtifact],
        reducer: CanonicalReducer,
        operational: JsonlLedger,
        state: JsonObject,
    ) -> None:
        """Require private custody before checker backoff.

        Parameters
        ----------
        work, artifacts:
            Current scheduled items and mutable normalized artifact map.
        reducer, operational, state:
            Locked canonical stores used to terminalize one failed staging transaction.

        Notes
        -----
        Pending mechanical work remains private. Publication is authorized only after the
        accepted checker decision is part of the reducer authority projection.
        """

        del reducer, operational, state
        for item in work:
            artifact = artifacts.get(item.stable_id)
            if artifact is not None and artifact.staged is None:
                raise DriverIntegrationError(
                    "pending mechanical work must execute from verified private custody"
                )

    def _forward_and_reduce(
        self,
        item: WorkItem,
        artifact: AuthorArtifact,
        environment: EnvironmentBinding,
        reducer: CanonicalReducer,
        operational: JsonlLedger,
        state: JsonObject,
        *,
        award_run: bool,
        closure: Optional[ExecutableClosureV3] = None,
        verification_token: Optional[EnvironmentVerificationToken] = None,
    ) -> Optional[str]:
        """Append honest worker attempts and return a checker pause from run repair.

        Parameters
        ----------
        item, artifact, environment, reducer, operational, state:
            Exact scheduled work, authority, ledgers, and durable driver state.
        award_run:
            Whether successful attempts may advance to run award.
        closure:
            Optional closure already collected once for this artifact in the pass.
        verification_token:
            Optional cache-created currentness-pass proof used for closure collection.

        Returns
        -------
        str | None
            Checker pause reason, or ``None`` after ordinary reduction.
        """

        if isinstance(self.dependencies.forward, SupervisedForwardLane):
            collected_closure = closure or _collect_worker_executable_closure(
                artifact,
                environment,
                verification_token=verification_token,
            )
            closure_identity = collected_closure.identity
        else:
            collected_closure = None
            closure_identity = _driver_facade()._INJECTED_FORWARD_CLOSURE_IDENTITY
        execution_identity = _driver_facade()._execution_identity(
            artifact, environment, closure_identity=closure_identity
        )
        self.dependencies.boundary_hook("pre-forward", item.stable_id)
        self._check_shutdown(
            "forward-admission",
            item=item,
            work_id=str(artifact.proposal["work_id"]),
            execution_identity=execution_identity,
        )
        attempts = _matching_attempts(
            self.paths.ledgers.attempts,
            artifact.proposal,
            environment,
            execution_identity,
        )
        rung = artifact.proposal.get("proposed_facts", {}).get("source_resolution", {}).get("rung")
        cold_runs = cold_forward_policy(item.stable_id, rung).required_cold_forwards
        if not _attempt_policy_satisfied(attempts, artifact.proposal, cold_runs):
            generated: tuple[Mapping[str, Any], ...]
            cache_identity = stable_hash(
                {
                    "execution_identity": execution_identity,
                    "work_id": artifact.proposal.get("work_id"),
                }
            )
            cache = (
                self.paths.work_root
                / item.stable_id
                / f"driver-forward-attempts-{cache_identity[7:23]}.json"
            )
            persisted_by_lane = isinstance(self.dependencies.forward, SupervisedForwardLane)
            attempts_persisted_by_lane = False

            def persist_worker_attempt(attempt: Mapping[str, Any]) -> None:
                """Persist one honest attempt before its worker lease closes."""

                candidate = _without_ledger_fields(attempt)
                reducer.append_attempt(
                    _redact_attempt_diagnostics(
                        candidate,
                        None,
                        _diagnostics_root_for_work_root(self.paths.work_root),
                    )
                )
                self.dependencies.boundary_hook("after-attempt", item.stable_id)

            def persist_worker_lifecycle(event_kind: str, status: str, lease: WorkerLease) -> None:
                """Persist a lock-ordered worker lifecycle transition."""

                self._append_worker_lifecycle_event(
                    operational,
                    event_kind=event_kind,
                    status=status,
                    lease_id=lease.lease_id,
                    stable_id=lease.stable_id,
                    details={
                        "work_id": lease.work_id,
                        "execution_identity": lease.execution_identity,
                        "request_identity": lease.request_identity,
                        "child_pid": lease.child_pid,
                    },
                )
                if event_kind == OperationalEventKind.WORKER_LEASE_STARTED.value:
                    self.dependencies.boundary_hook(event_kind, lease.stable_id)

            def resolve_worker_attempt(cold_index: int, mode: str) -> Optional[Mapping[str, Any]]:
                """Authenticate one canonical deterministic slot before any capability opens."""

                resolved = resolve_attempt_slot(
                    scan_jsonl(self.paths.ledgers.attempts),
                    work_id=str(artifact.proposal["work_id"]),
                    execution_identity=execution_identity,
                    cold_index=cold_index,
                    mode=mode,
                )
                if resolved is None:
                    return None
                try:
                    authority = load_current_attempt_proof(resolved)
                except AuthorityDerivationError as exc:
                    raise DriverIntegrationError(str(exc)) from exc
                if resolved.get("result") == "succeeded" and authority is None:
                    raise DriverIntegrationError(
                        "canonical execution slot lacks authenticated success authority"
                    )
                return resolved

            cached: JsonObject | None = None
            if cache.is_file() and not persisted_by_lane:
                try:
                    candidate = _read_json(cache)
                except Exception:  # noqa: BLE001 -- disposable replay cache is regenerable
                    cache.unlink(missing_ok=True)
                else:
                    if (
                        candidate.get("work_id") == artifact.proposal.get("work_id")
                        and candidate.get("execution_identity") == execution_identity
                    ):
                        cached = candidate
                    else:
                        cache.unlink(missing_ok=True)
            if cached is not None:
                generated = tuple(
                    dict(value)
                    for value in cached.get("attempts", [])
                    if isinstance(value, Mapping)
                )
            else:
                try:
                    if isinstance(self.dependencies.forward, SupervisedForwardLane):
                        generated = tuple(
                            self.dependencies.forward.forward(
                                artifact,
                                environment,
                                cold_runs,
                                self.paths.work_root,
                                worker_lock_path=self.paths.worker_lock,
                                worker_lease_path=self.paths.worker_lease,
                                run_id=self.config.run_id,
                                shutdown_event=self._shutdown_event,
                                lifecycle_event=persist_worker_lifecycle,
                                attempt_sink=persist_worker_attempt,
                                attempt_resolver=resolve_worker_attempt,
                                closure=collected_closure,
                            )
                        )
                        attempts_persisted_by_lane = True
                    else:
                        generated = tuple(
                            self.dependencies.forward.forward(
                                artifact,
                                environment,
                                cold_runs,
                                self.paths.work_root,
                            )
                        )
                except Exception as exc:  # noqa: BLE001 -- supervisor failure is model-local
                    stage, reason = (
                        ("policy", "sandbox-unavailable-v1")
                        if _is_sandbox_unavailable(exc)
                        else ("runner", "internal-error")
                    )
                    generated = (
                        _driver_failure_attempt(
                            item,
                            artifact,
                            stage,
                            reason,
                            exc,
                            self.config,
                            diagnostics_root=_diagnostics_root_for_work_root(self.paths.work_root),
                            environment=environment.family,
                            created_at=self.dependencies.clock(),
                        ),
                    )
                _write_json_atomic(
                    cache,
                    {
                        "work_id": artifact.proposal["work_id"],
                        "execution_identity": execution_identity,
                        "attempts": list(generated),
                    },
                )
            if not attempts_persisted_by_lane:
                for attempt in generated:
                    persist_worker_attempt(attempt)
            attempts = _matching_attempts(
                self.paths.ledgers.attempts,
                artifact.proposal,
                environment,
                execution_identity,
            )
            if not _attempt_policy_satisfied(attempts, artifact.proposal, cold_runs):
                all_attempts = _matching_model_attempts(
                    self.paths.ledgers.attempts, artifact.proposal
                )
                expansion = _detected_mode_expansion(all_attempts, artifact.proposal)
                if expansion is not None:
                    if not any(
                        attempt.get("error", {}).get("details", {}).get("route")
                        == "recipe-and-gate-revision-required"
                        for attempt in all_attempts
                        if isinstance(attempt.get("error"), Mapping)
                    ):
                        expansion_attempt = _driver_failure_attempt(
                            item,
                            artifact,
                            "input",
                            "contract-invalid",
                            DriverIntegrationError(
                                "worker detected meaningful modes absent from the gated proposal"
                            ),
                            self.config,
                            diagnostics_root=_diagnostics_root_for_work_root(self.paths.work_root),
                            environment=environment.family,
                            created_at=self.dependencies.clock(),
                        )
                        expansion_attempt["attempt_id"] = stable_hash(
                            {
                                "work_id": artifact.proposal["work_id"],
                                "route": "recipe-and-gate-revision-required",
                                "detected_meaningful_modes": expansion["detected_meaningful_modes"],
                            }
                        )
                        expansion_attempt["error"]["details"] = deepcopy(expansion)
                        expansion_attempt["error"]["root_cause_fingerprint"] = stable_hash(
                            expansion
                        )
                        persisted_expansion = reducer.append_attempt(expansion_attempt).record
                        all_attempts = (*all_attempts, persisted_expansion)
                    try:
                        repaired = self._repair_author_for_detected_modes(
                            item,
                            artifact,
                            expansion,
                            reducer,
                        )
                    except Exception as exc:  # noqa: BLE001 -- bounded repair is model-local
                        reason = (
                            "protocol-violation"
                            if isinstance(exc, DriverIntegrationError)
                            and not self._is_infrastructure_error(exc)
                            else "internal-error"
                        )
                        repair_failure = reducer.append_attempt(
                            _driver_failure_attempt(
                                item,
                                artifact,
                                "runner",
                                reason,
                                exc,
                                self.config,
                                diagnostics_root=_diagnostics_root_for_work_root(
                                    self.paths.work_root
                                ),
                                environment=environment.family,
                                created_at=self.dependencies.clock(),
                            )
                        ).record
                        self._terminalize(
                            item,
                            artifact,
                            "failed:runner",
                            reason,
                            str(exc),
                            (*all_attempts, repair_failure),
                            reducer,
                            operational,
                            state,
                        )
                        return None
                    repaired_artifacts = {item.stable_id: repaired}
                    pause = self._ensure_gates(
                        (item,), repaired_artifacts, reducer, operational, state
                    )
                    current = reducer.current_records.get(item.stable_id)
                    if current is not None and current.get("status", {}).get("kind") != "runs":
                        return pause
                    if pause is not None:
                        return pause
                    repaired = repaired_artifacts[item.stable_id]
                    self._family_artifacts[item.stable_id] = repaired
                    repaired_closure = (
                        _collect_worker_executable_closure(
                            repaired,
                            environment,
                            verification_token=verification_token,
                        )
                        if isinstance(self.dependencies.forward, SupervisedForwardLane)
                        else None
                    )
                    return self._forward_and_reduce(
                        item,
                        repaired,
                        environment,
                        reducer,
                        operational,
                        state,
                        award_run=award_run,
                        closure=repaired_closure,
                        verification_token=verification_token,
                    )
                failure = next(
                    (
                        attempt
                        for attempt in reversed(all_attempts)
                        if attempt["result"] == "failed"
                    ),
                    None,
                )
                if failure is None:
                    integration_error = DriverIntegrationError(
                        f"worker attempts do not satisfy modes/cold policy for {item.stable_id}"
                    )
                    failure = reducer.append_attempt(
                        _driver_failure_attempt(
                            item,
                            artifact,
                            "runner",
                            "protocol-violation",
                            integration_error,
                            self.config,
                            diagnostics_root=_diagnostics_root_for_work_root(self.paths.work_root),
                            environment=environment.family,
                            created_at=self.dependencies.clock(),
                        )
                    ).record
                    all_attempts = (*all_attempts, failure)
                error = failure["error"]
                if not isinstance(error, Mapping):
                    raise DriverIntegrationError("failed attempt lost its structured error")
                stage = str(error["stage"])
                self._terminalize(
                    item,
                    artifact,
                    f"failed:{stage}",
                    str(error["reason_code"]),
                    None,
                    all_attempts,
                    reducer,
                    operational,
                    state,
                )
                return None
            self.dependencies.boundary_hook("after-forward", item.stable_id)
        if not award_run:
            return None
        self.dependencies.boundary_hook("post-attempt-pre-award", item.stable_id)
        self._check_shutdown(
            "post-attempt-pre-award",
            item=item,
            work_id=str(artifact.proposal["work_id"]),
            execution_identity=execution_identity,
        )
        gates = scan_jsonl(self.paths.ledgers.gates)
        representative_model = (
            reducer.current_records.get(item.family_representative_id)
            if item.is_family_variant
            else None
        )
        try:
            model = _assemble_run_model(
                item,
                artifact,
                attempts,
                gates,
                self.config,
                representative_model=representative_model,
            )
        except DriverIntegrationError as exc:
            if artifact.template_source_revision is None:
                raise
            failure = reducer.append_attempt(
                _driver_failure_attempt(
                    item,
                    artifact,
                    "runner",
                    "protocol-violation",
                    exc,
                    self.config,
                    diagnostics_root=_diagnostics_root_for_work_root(self.paths.work_root),
                    environment=environment.family,
                    created_at=self.dependencies.clock(),
                )
            ).record
            self._terminalize(
                item,
                artifact,
                "failed:runner",
                "protocol-violation",
                str(exc),
                (*attempts, failure),
                reducer,
                operational,
                state,
            )
            return None
        current_model = reducer.current_records.get(item.stable_id)
        model["parent_revision"] = (
            current_model["record_revision"] if current_model is not None else None
        )
        if current_model is not None:
            model["status"]["supersedes_revision"] = current_model["record_revision"]

        self.dependencies.boundary_hook("pre-publication", item.stable_id)
        self._check_shutdown(
            "pre-publication-admission",
            item=item,
            work_id=str(artifact.proposal["work_id"]),
            execution_identity=execution_identity,
        )
        self.dependencies.boundary_hook("pre-award-commit", item.stable_id)
        self._check_shutdown(
            "pre-award-commit",
            item=item,
            work_id=str(artifact.proposal["work_id"]),
            execution_identity=execution_identity,
        )
        self.dependencies.boundary_hook("award-commit-entered", item.stable_id)
        # Graceful-shutdown atomic award section: publication authorization and
        # materialization must remain check-free through the canonical model append.
        if model.get("authored_metadata_state") == "accepted" and not isinstance(
            artifact, ActivatedHandoffArtifact
        ):
            self._authorize_and_publish_artifact(artifact, model, gates, reducer)
        result = reducer.append_model(reducer.prepare_model(model))
        if result.appended:
            self._reduced += 1
        self.dependencies.boundary_hook("post-award-commit", item.stable_id)
        self._check_shutdown("post-award-commit")
        self.dependencies.boundary_hook("after-reduce", item.stable_id)
        current_records = reducer.current_records
        snapshot = self._policy_snapshot()
        self._handle_progress(operational, current_records, snapshot, state=state)
        if self._maybe_pause_for_review(operational, current_records, snapshot, state):
            raise DriverPaused("review checkpoint reached")
        state["last_terminal_count"] = len(current_records)
        state["status"] = "running"
        _write_driver_state(self.paths.driver_state, state)
        return None
