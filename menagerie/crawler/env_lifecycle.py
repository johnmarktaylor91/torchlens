"""Sequential target-solved environment creation, probing, use, and teardown."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol, Sequence

from menagerie.crawler.envs import EnvironmentIntent, IntentProbes
from menagerie.crawler.effort import EffortTracker
from menagerie.crawler.identity import hash_bytes


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


@dataclass(frozen=True)
class SolveResult:
    """Exact bytes and measured effort returned by an on-target solver."""

    lock_bytes: bytes
    resolved_export_bytes: bytes
    elapsed_seconds: float
    artifact_bytes: int


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

    def create(self, lock_file: Path, prefix: Path) -> None:
        """Create a new immutable environment from an exact lock."""

        ...

    def probe(self, prefix: Path, probes: IntentProbes) -> Sequence[ProbeResult]:
        """Run all declared imports, export checks, and build canaries."""

        ...

    def remove(self, prefix: Path) -> None:
        """Remove only the named environment and its dedicated scratch/cache."""

        ...


DiskFree = Callable[[Path], int]
UseEnvironment = Callable[[Path], None]


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
        export_hash = hash_bytes(solved.resolved_export_bytes)
        _atomic_write(intent.lock.lock_path, solved.lock_bytes)
        _atomic_write(intent.lock.export_path, solved.resolved_export_bytes)
        _atomic_write(intent.lock.export_hash_path, f"{export_hash}\n".encode())

        prefix = self._env_root / f"{intent.name}-{export_hash[7:19]}"
        self._active = prefix
        disk_after_create = disk_before
        probe_results: tuple[ProbeResult, ...] = ()
        try:
            self._backend.create(intent.lock.lock_path, prefix)
            disk_after_create = self._disk_free(self._env_root)
            probe_results = tuple(self._backend.probe(prefix, intent.probes))
            failed = [result for result in probe_results if not result.passed]
            if failed:
                names = ", ".join(result.name for result in failed)
                raise EnvironmentProbeError(f"environment probes failed: {names}")
            use(prefix)
        finally:
            self._backend.remove(prefix)
            self._active = None
        disk_after_teardown = self._disk_free(self._env_root)
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


def _atomic_write(path: Path, content: bytes) -> None:
    """Atomically replace one setup-time artifact and fsync its contents."""

    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
