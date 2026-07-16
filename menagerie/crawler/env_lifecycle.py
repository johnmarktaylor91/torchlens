"""Sequential target-solved environment creation, probing, use, and teardown."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from menagerie.crawler.envs import EnvironmentIntent, IntentProbes
from menagerie.crawler.effort import EffortTracker
from menagerie.crawler.identity import canonical_json_bytes, compute_env_generation, hash_bytes


class EnvironmentLifecycleError(RuntimeError):
    """Base class for typed lifecycle failures."""


class EnvironmentSolveError(EnvironmentLifecycleError):
    """Raised when an on-target intent solve fails."""


class EnvironmentProbeError(EnvironmentLifecycleError):
    """Raised when an exact environment probe fails."""


class EnvironmentBusyError(EnvironmentLifecycleError):
    """Raised when a second environment is requested before teardown."""


class DiskRecoveryError(EnvironmentLifecycleError):
    """Raised when teardown does not recover the required disk space."""


class EnvironmentCleanupError(EnvironmentLifecycleError):
    """Raised when removal of a completed environment fails."""


class EnvironmentExactnessError(EnvironmentLifecycleError):
    """Raised when materialized environment bytes disagree with declared facts."""


@dataclass(frozen=True)
class ArtifactReceipt:
    """One solver artifact whose digest was verified from materialized bytes."""

    url: str
    sha256: str


@dataclass(frozen=True)
class SolveResult:
    """Exact bytes and measured effort returned by an on-target solver."""

    lock_bytes: bytes
    resolved_export_bytes: bytes
    elapsed_seconds: float
    artifact_bytes: int
    artifact_receipts: tuple[ArtifactReceipt, ...] = ()


@dataclass(frozen=True)
class ProbeResult:
    """Result of one declared environment probe."""

    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class LifecycleResult:
    """Completed sequential lifecycle metrics and generated artifacts."""

    intent: str
    target: str
    export_sha256: str
    probe_results: tuple[ProbeResult, ...]
    disk_before: int
    disk_after_create: int
    disk_after_teardown: int
    disk_recovery_checked: bool


class EnvironmentBackend(Protocol):
    """Backend boundary implemented by conda tooling or a test fake."""

    def solve(self, environment_file: Path, target: str) -> SolveResult:
        """Solve direct dependencies on the actual target machine."""

        ...

    def create(self, lock_file: Path, prefix: Path) -> bytes:
        """Create an immutable environment and return its installed inventory bytes."""

        ...

    def probe(self, prefix: Path, probes: IntentProbes) -> Sequence[ProbeResult]:
        """Run all declared imports, export checks, and build canaries."""

        ...

    def remove(self, prefix: Path) -> None:
        """Remove only the named environment and its dedicated scratch/cache."""

        ...


DiskFree = Callable[[Path], int]
UseEnvironment = Callable[[Path, tuple[ProbeResult, ...]], None]


def disk_free_bytes(path: Path) -> int:
    """Return available filesystem bytes at a path.

    Parameters
    ----------
    path:
        Existing path on the environment filesystem.

    Returns
    -------
    int
        Available bytes.
    """

    stats = os.statvfs(path)
    return stats.f_bavail * stats.f_frsize


class SequentialEnvironmentLifecycle:
    """Own exactly one active environment from target solve through teardown."""

    def __init__(
        self,
        backend: EnvironmentBackend,
        effort: EffortTracker,
        *,
        env_root: Path,
        disk_free: DiskFree = disk_free_bytes,
        minimum_free_bytes: int = 30 * 1024**3,
        recovery_tolerance_bytes: int = 1024**2,
    ) -> None:
        """Configure the backend, effort caps, roots, and disk invariants."""

        self._backend = backend
        self._effort = effort
        self._env_root = env_root
        self._disk_free = disk_free
        self._minimum_free_bytes = minimum_free_bytes
        self._recovery_tolerance_bytes = recovery_tolerance_bytes
        self._active: Path | None = None

    def run(
        self,
        intent: EnvironmentIntent,
        *,
        use: UseEnvironment,
    ) -> LifecycleResult:
        """Solve, lock, create, probe, use, teardown, and verify one environment.

        Parameters
        ----------
        intent:
            Loaded intent whose target paths receive setup-time artifacts.
        use:
            Driver callback that processes assigned models sequentially.

        Returns
        -------
        LifecycleResult
            Completed lifecycle evidence.

        Raises
        ------
        EnvironmentBusyError
            If another environment remains active.
        EnvironmentLifecycleError
            If solve, probe, or disk invariants fail.
        EffortCapExceeded
            If solve/build work exceeds the configured environment cap.
        """

        if self._active is not None:
            raise EnvironmentBusyError(f"environment already active at {self._active}")
        intent.lock.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._env_root.mkdir(parents=True, exist_ok=True)
        disk_before = self._disk_free(self._env_root)
        if disk_before < self._minimum_free_bytes:
            raise DiskRecoveryError("less than the required 30 GiB is available")
        self._effort.consume("environment", attempts=1)
        try:
            solved = self._backend.solve(
                intent.lock.lock_path.parent.parent / "environment.yml", intent.lock.target
            )
        except Exception as exc:
            raise EnvironmentSolveError(f"target solve failed for {intent.name}: {exc}") from exc
        self._effort.consume(
            "environment",
            seconds=solved.elapsed_seconds,
            bytes_used=solved.artifact_bytes,
        )
        if not solved.lock_bytes or not solved.resolved_export_bytes:
            raise EnvironmentSolveError("solver returned an empty lock or resolved export")
        lock_receipts = parse_exact_lock(solved.lock_bytes)
        if tuple(solved.artifact_receipts) != lock_receipts:
            raise EnvironmentExactnessError(
                "materialized solver artifacts do not exactly match the target lock"
            )
        declared_inventory = parse_resolved_export(solved.resolved_export_bytes)
        _require_lock_inventory_match(solved.lock_bytes, declared_inventory)
        export_hash = hash_bytes(solved.resolved_export_bytes)
        _atomic_write(intent.lock.lock_path, solved.lock_bytes)
        _atomic_write(intent.lock.export_path, solved.resolved_export_bytes)
        _atomic_write(intent.lock.export_hash_path, f"{export_hash}\n".encode())

        prefix = self._env_root / f"{intent.name}-{export_hash[7:19]}"
        self._active = prefix
        disk_after_create = disk_before
        probe_results: tuple[ProbeResult, ...] = ()
        try:
            installed_inventory = self._backend.create(intent.lock.lock_path, prefix)
            if installed_inventory != declared_inventory:
                raise EnvironmentExactnessError(
                    "created-prefix package inventory differs from the declared resolved export"
                )
            disk_after_create = self._disk_free(self._env_root)
            raw_probe_results = tuple(self._backend.probe(prefix, intent.probes))
            probe_results = validate_probe_receipts(intent.probes, raw_probe_results)
            receipt_path = intent.lock.lock_path.with_name(f"{intent.lock.target}.probes.json")
            _atomic_write(receipt_path, canonical_probe_receipt_bytes(probe_results))
            use(prefix, probe_results)
        finally:
            try:
                try:
                    self._backend.remove(prefix)
                except Exception as exc:
                    raise EnvironmentCleanupError(
                        f"environment removal failed for {intent.name}: {exc}"
                    ) from exc
            finally:
                self._active = None
        try:
            disk_after_teardown = self._disk_free(self._env_root)
        except Exception as exc:
            raise EnvironmentCleanupError(
                f"post-removal disk recovery measurement failed for {intent.name}: {exc}"
            ) from exc
        if disk_after_teardown + self._recovery_tolerance_bytes < disk_before:
            raise DiskRecoveryError(
                f"teardown recovered {disk_after_teardown} bytes; expected at least {disk_before}"
            )
        return LifecycleResult(
            intent=intent.name,
            target=intent.lock.target,
            export_sha256=export_hash,
            probe_results=probe_results,
            disk_before=disk_before,
            disk_after_create=disk_after_create,
            disk_after_teardown=disk_after_teardown,
            disk_recovery_checked=True,
        )


_HASH_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
_LOCK_LINE_PATTERN = re.compile(r"(?P<url>\S+?)(?:#sha256=|#)(?P<digest>[0-9a-f]{64})\Z")
_PACKAGE_FIELDS = ("name", "version", "build", "url", "sha256")


def parse_exact_lock(content: bytes) -> tuple[ArtifactReceipt, ...]:
    """Parse an explicit lock and require one canonical SHA-256 per artifact.

    Parameters
    ----------
    content:
        Exact lock bytes produced by the target solver.

    Returns
    -------
    tuple[ArtifactReceipt, ...]
        Ordered URL and digest receipts from the lock.

    Raises
    ------
    EnvironmentExactnessError
        If the lock is malformed, empty, unhashed, or contains duplicates.
    """

    try:
        lines = content.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise EnvironmentExactnessError("environment lock is not UTF-8") from exc
    receipts: list[ArtifactReceipt] = []
    for line in lines:
        value = line.strip()
        if not value or value.startswith(("#", "@EXPLICIT")):
            continue
        match = _LOCK_LINE_PATTERN.fullmatch(value)
        if match is None:
            raise EnvironmentExactnessError(
                f"environment lock entry lacks a canonical SHA-256 digest: {value!r}"
            )
        receipts.append(ArtifactReceipt(match.group("url"), f"sha256:{match.group('digest')}"))
    if not receipts:
        raise EnvironmentExactnessError("environment lock has no artifact entries")
    if len({receipt.url for receipt in receipts}) != len(receipts):
        raise EnvironmentExactnessError("environment lock contains duplicate artifact URLs")
    return tuple(receipts)


def parse_resolved_export(content: bytes) -> bytes:
    """Parse and canonicalize the declared installed-package export.

    Parameters
    ----------
    content:
        Declared resolved-export bytes.

    Returns
    -------
    bytes
        Canonical installed inventory bytes suitable for byte comparison.

    Raises
    ------
    EnvironmentExactnessError
        If the export is not the closed exact package-inventory format.
    """

    try:
        value = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EnvironmentExactnessError("resolved export is not valid JSON") from exc
    if not isinstance(value, Mapping) or set(value) != {"packages"}:
        raise EnvironmentExactnessError(
            "resolved export must contain exactly one packages inventory"
        )
    return _canonical_package_inventory(value.get("packages"), "resolved export")


def installed_package_inventory_bytes(prefix: Path) -> bytes:
    """Derive canonical package evidence only from immutable ``conda-meta`` files.

    Parameters
    ----------
    prefix:
        Created environment prefix.

    Returns
    -------
    bytes
        Canonical package inventory derived from installed metadata.

    Raises
    ------
    EnvironmentExactnessError
        If installed metadata is absent, malformed, or lacks exact artifact hashes.
    """

    metadata_paths = sorted((prefix / "conda-meta").glob("*.json"))
    if not metadata_paths:
        raise EnvironmentExactnessError(f"installed conda metadata is missing below {prefix}")
    rows: list[Mapping[str, Any]] = []
    for path in metadata_paths:
        try:
            value = json.loads(path.read_bytes())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise EnvironmentExactnessError(
                f"installed package metadata is invalid: {path}"
            ) from exc
        if not isinstance(value, Mapping):
            raise EnvironmentExactnessError(f"installed package metadata must be an object: {path}")
        rows.append(value)
    return _canonical_package_inventory(rows, "installed conda metadata")


def _canonical_package_inventory(value: Any, label: str) -> bytes:
    """Return canonical bytes for a closed exact package inventory.

    Parameters
    ----------
    value:
        Candidate package-row sequence.
    label:
        Error-context label.

    Returns
    -------
    bytes
        Canonical newline-terminated JSON inventory.

    Raises
    ------
    EnvironmentExactnessError
        If rows are absent, ambiguous, or lack immutable artifact fields.
    """

    if not isinstance(value, list) or not value:
        raise EnvironmentExactnessError(f"{label} has no package rows")
    rows: list[dict[str, str]] = []
    for raw in value:
        if not isinstance(raw, Mapping):
            raise EnvironmentExactnessError(f"{label} contains a non-object package row")
        row: dict[str, str] = {}
        for field in _PACKAGE_FIELDS:
            item = raw.get(field)
            if not isinstance(item, str) or not item:
                raise EnvironmentExactnessError(f"{label} package row lacks nonempty {field}")
            row[field] = item
        if re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
            row["sha256"] = f"sha256:{row['sha256']}"
        if _HASH_PATTERN.fullmatch(row["sha256"]) is None:
            raise EnvironmentExactnessError(
                f"{label} package row has a noncanonical SHA-256 digest"
            )
        rows.append(row)
    rows.sort(key=lambda row: tuple(row[field] for field in _PACKAGE_FIELDS))
    identities = [tuple(row[field] for field in _PACKAGE_FIELDS) for row in rows]
    if len(identities) != len(set(identities)):
        raise EnvironmentExactnessError(f"{label} contains duplicate package rows")
    return canonical_json_bytes({"packages": rows}) + b"\n"


def _require_lock_inventory_match(lock_bytes: bytes, package_bytes: bytes) -> None:
    """Require lock artifacts and installed/exported packages to be one-to-one.

    Parameters
    ----------
    lock_bytes, package_bytes:
        Exact target lock and canonical package-inventory bytes.

    Raises
    ------
    EnvironmentExactnessError
        If either side names an artifact URL/digest absent from the other.
    """

    lock_receipts = {(receipt.url, receipt.sha256) for receipt in parse_exact_lock(lock_bytes)}
    try:
        value = json.loads(package_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EnvironmentExactnessError("canonical package inventory is invalid") from exc
    packages = value.get("packages") if isinstance(value, Mapping) else None
    if not isinstance(packages, list):
        raise EnvironmentExactnessError("canonical package inventory has no package rows")
    package_receipts = {
        (str(package.get("url")), str(package.get("sha256")))
        for package in packages
        if isinstance(package, Mapping)
    }
    if lock_receipts != package_receipts or len(package_receipts) != len(packages):
        raise EnvironmentExactnessError(
            "target lock artifacts differ from the created-prefix package inventory"
        )


def expected_probe_names(probes: IntentProbes) -> tuple[str, ...]:
    """Return the exact ordered receipt names for one declared probe contract.

    Parameters
    ----------
    probes:
        Declared import, export, and source-build probes.

    Returns
    -------
    tuple[str, ...]
        Closed ordered receipt-name set.
    """

    return (
        *(f"import:{name}" for name in probes.imports),
        *(f"export:{check.module}:{check.attribute}" for check in probes.export_checks),
        *(f"source-build:{build.name}" for build in probes.source_build),
    )


def validate_probe_receipts(
    probes: IntentProbes, receipts: Sequence[ProbeResult]
) -> tuple[ProbeResult, ...]:
    """Require one successful named receipt for every declared probe and no others.

    Parameters
    ----------
    probes, receipts:
        Declared probe contract and backend observations.

    Returns
    -------
    tuple[ProbeResult, ...]
        Receipts reordered to the canonical declaration order.

    Raises
    ------
    EnvironmentProbeError
        If the declared set is empty, duplicated, missing, extra, or failed.
    """

    expected = expected_probe_names(probes)
    if not expected or len(set(expected)) != len(expected):
        raise EnvironmentProbeError("environment probe contract is empty or ambiguous")
    by_name: dict[str, ProbeResult] = {}
    for receipt in receipts:
        if receipt.name in by_name:
            raise EnvironmentProbeError(f"duplicate environment probe receipt: {receipt.name}")
        by_name[receipt.name] = receipt
    if set(by_name) != set(expected):
        missing = sorted(set(expected) - set(by_name))
        extra = sorted(set(by_name) - set(expected))
        raise EnvironmentProbeError(
            f"environment probe receipts do not match intent: missing={missing}, extra={extra}"
        )
    failed = [name for name in expected if not by_name[name].passed]
    if failed:
        raise EnvironmentProbeError(f"environment probes failed: {', '.join(failed)}")
    return tuple(by_name[name] for name in expected)


def canonical_probe_receipt_bytes(receipts: Sequence[ProbeResult]) -> bytes:
    """Serialize canonical successful probe receipts for durable recomputation.

    Parameters
    ----------
    receipts:
        Already validated ordered receipts.

    Returns
    -------
    bytes
        Canonical newline-terminated receipt artifact.
    """

    return (
        canonical_json_bytes(
            {
                "probes": [
                    {"name": receipt.name, "passed": receipt.passed, "detail": receipt.detail}
                    for receipt in receipts
                ]
            }
        )
        + b"\n"
    )


def parse_probe_receipt_bytes(probes: IntentProbes, content: bytes) -> tuple[ProbeResult, ...]:
    """Parse a durable receipt artifact against its declared probe contract.

    Parameters
    ----------
    probes, content:
        Declared probes and committed canonical receipt bytes.

    Returns
    -------
    tuple[ProbeResult, ...]
        Exact validated receipts in declaration order.

    Raises
    ------
    EnvironmentProbeError
        If the artifact is malformed, noncanonical, or does not match the contract.
    """

    try:
        value = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EnvironmentProbeError("environment probe receipt artifact is invalid JSON") from exc
    if not isinstance(value, Mapping) or set(value) != {"probes"}:
        raise EnvironmentProbeError("environment probe receipt artifact has an invalid schema")
    raw_receipts = value.get("probes")
    if not isinstance(raw_receipts, list):
        raise EnvironmentProbeError("environment probe receipt list is missing")
    receipts: list[ProbeResult] = []
    for raw in raw_receipts:
        if (
            not isinstance(raw, Mapping)
            or set(raw) != {"name", "passed", "detail"}
            or not isinstance(raw.get("name"), str)
            or not isinstance(raw.get("passed"), bool)
            or not isinstance(raw.get("detail"), str)
        ):
            raise EnvironmentProbeError("environment probe receipt row is malformed")
        receipts.append(ProbeResult(str(raw["name"]), bool(raw["passed"]), str(raw["detail"])))
    validated = validate_probe_receipts(probes, receipts)
    if content != canonical_probe_receipt_bytes(validated):
        raise EnvironmentProbeError("environment probe receipt artifact is not canonical")
    return validated


def materialized_environment_generation(
    intent: EnvironmentIntent,
    *,
    lock_bytes: bytes,
    export_bytes: bytes,
    package_bytes: bytes,
    python_version: str,
    compiler_identity: str,
    sdk_identity: str,
    probe_results: Sequence[ProbeResult],
) -> str:
    """Recompute one generation solely from exact artifacts and canonical receipts.

    Parameters
    ----------
    intent:
        Registry-derived intent and probe contract.
    lock_bytes, export_bytes, package_bytes:
        Exact lock, declared export, and created-prefix package inventory bytes.
    python_version, compiler_identity, sdk_identity:
        Observed interpreter and toolchain facts.
    probe_results:
        Exact one-to-one declared probe receipts.

    Returns
    -------
    str
        Canonical environment generation.

    Raises
    ------
    EnvironmentExactnessError
        If committed artifacts do not form one exact environment.
    """

    parse_exact_lock(lock_bytes)
    if parse_resolved_export(export_bytes) != package_bytes:
        raise EnvironmentExactnessError(
            "created-prefix package inventory differs from the declared resolved export"
        )
    _require_lock_inventory_match(lock_bytes, package_bytes)
    receipts = validate_probe_receipts(intent.probes, probe_results)
    probe_intent = {
        "imports": list(intent.probes.imports),
        "export_checks": [vars(value) for value in intent.probes.export_checks],
        "source_build": [vars(value) for value in intent.probes.source_build],
    }
    observed_probes = [
        {"name": result.name, "passed": result.passed, "detail": result.detail}
        for result in receipts
    ]
    platform_facts = {
        "target": intent.lock.target,
        "python": python_version,
        "compiler": compiler_identity,
        "sdk": sdk_identity,
        "packages_manifest_sha256": hash_bytes(package_bytes),
    }
    return compute_env_generation(
        {
            "name": intent.name,
            "framework": intent.framework,
            "target": intent.lock.target,
            "channels": list(intent.channels),
            "dependencies": list(intent.dependencies),
            "probe_intent": probe_intent,
        },
        hash_bytes(lock_bytes),
        hash_bytes(export_bytes),
        platform_facts,
        observed_probes,
    )


def _atomic_write(path: Path, content: bytes) -> None:
    """Atomically replace one setup-time artifact and fsync its contents."""

    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
