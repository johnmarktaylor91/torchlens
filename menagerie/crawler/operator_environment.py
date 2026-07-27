"""Exact-lock conda environment operator for crawler lifecycle actions."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

import yaml

from menagerie.crawler.env_lifecycle import (
    ArtifactReceipt,
    EnvironmentExactnessError,
    parse_exact_lock,
    parse_resolved_export,
)
from menagerie.crawler.identity import (
    atomic_replace_bytes,
    canonical_json_bytes,
    fsync_directory,
    hash_bytes,
)

EXIT_SUCCESS = 0
EXIT_PERMANENT = 64
EXIT_RETRYABLE = 75
PROTOCOL_VERSION = "menagerie.crawler.environment-operator.v1"

_DEFAULT_CONDA_LOCK = Path("/opt/homebrew/Caskroom/miniforge/base/bin/conda-lock")
_DEFAULT_CONDA = Path("/opt/homebrew/Caskroom/miniforge/base/bin/conda")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_COMPONENT_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_GLOB_CHARACTERS = frozenset("*?[]{}")
_TRANSIENT_MARKERS = (
    "connection reset",
    "connection refused",
    "connection aborted",
    "connection error",
    "could not connect",
    "failed to establish a new connection",
    "http 408",
    "http 425",
    "http 429",
    "http 500",
    "http 502",
    "http 503",
    "http 504",
    "name or service not known",
    "network is unreachable",
    "remote end closed connection",
    "resource temporarily unavailable",
    "temporary failure",
    "temporarily unavailable",
    "timed out",
    "timeout",
    "tls handshake",
)
_DETAIL_LIMIT = 1_500
_PROBE_DETAIL_LIMIT = 1_000
_SOLVE_TIMEOUT_SECONDS = 30 * 60
_CREATE_TIMEOUT_SECONDS = 30 * 60
_PROBE_TIMEOUT_SECONDS = 5 * 60
_DOWNLOAD_TIMEOUT_SECONDS = 5 * 60
_TRANSIENT_RETRIES = 2


class EnvironmentOperatorError(RuntimeError):
    """Base class for typed environment-operator failures."""


class PermanentEnvironmentOperatorError(EnvironmentOperatorError):
    """A contract, configuration, or exactness failure that must not be retried."""


class TransientEnvironmentOperatorError(EnvironmentOperatorError):
    """An exhausted spawn, transport, or temporary service failure."""


@dataclass(frozen=True)
class SolvedPackage:
    """One target-specific conda package parsed from the unified lock."""

    name: str
    version: str
    build: str
    url: str
    sha256: str
    filename: str


@dataclass(frozen=True)
class CommandObservation:
    """One completed child process with bounded captured streams."""

    returncode: int
    stdout: str
    stderr: str


CommandRunner = Callable[[Sequence[str], Mapping[str, str], float], CommandObservation]
Sleeper = Callable[[float], None]


class EnvironmentOperator:
    """Solve, create, probe, and remove exact conda environments."""

    def __init__(
        self,
        *,
        state_root: Path | None = None,
        conda_lock_command: Sequence[str] | None = None,
        conda_command: Sequence[str] | None = None,
        runner: CommandRunner | None = None,
        sleeper: Sleeper = time.sleep,
    ) -> None:
        """Configure durable state, external commands, and retry injection points.

        Parameters
        ----------
        state_root:
            Optional absolute operator-state root. When omitted, the root is inferred
            from the action path and placed below the repository's ``.crawl-local``.
        conda_lock_command, conda_command:
            Optional argv prefixes used instead of the installed tools.
        runner:
            Injectable non-shell subprocess boundary.
        sleeper:
            Injectable retry backoff.
        """

        if state_root is not None and not state_root.is_absolute():
            raise PermanentEnvironmentOperatorError("operator state root must be absolute")
        self._state_root = state_root.resolve() if state_root is not None else None
        self._conda_lock_command = tuple(
            conda_lock_command or _default_tool_command("MENAGERIE_CONDA_LOCK", _DEFAULT_CONDA_LOCK)
        )
        self._conda_command = tuple(
            conda_command or _default_tool_command("MENAGERIE_CONDA", _DEFAULT_CONDA)
        )
        self._runner = runner or _default_command_runner
        self._sleeper = sleeper

    def solve(
        self,
        environment_file: Path,
        target: str,
        *,
        force_resolve: bool = False,
    ) -> dict[str, Any]:
        """Solve one intent/target pair and return exact artifact receipts.

        Parameters
        ----------
        environment_file:
            Absolute conda environment specification.
        target:
            Exact conda-lock platform.
        force_resolve:
            Whether an operator explicitly requested replacement of a valid solve cache.

        Returns
        -------
        dict[str, Any]
            Driver-facing lock/export paths, measurements, cache state, and receipts.
        """

        started = time.monotonic()
        environment_file = _require_absolute_file(environment_file, "environment specification")
        _require_safe_component(target, "target")
        intent = environment_file.parent.name
        _require_safe_component(intent, "intent")
        specification = _load_mapping(environment_file, "environment specification")
        _refuse_pip_sections(specification)
        state_root = self._state_root_for(environment_file)
        solve_root = state_root / "solves" / intent / target
        solve_root.mkdir(parents=True, exist_ok=True)
        with _exclusive_file_lock(solve_root / ".solve.lock"):
            if not force_resolve:
                cached = self._cached_solve_result(solve_root, state_root)
                if cached is not None:
                    cached["elapsed_seconds"] = time.monotonic() - started
                    cached["cache_hit"] = True
                    return cached
            result = self._fresh_solve(
                environment_file=environment_file,
                target=target,
                solve_root=solve_root,
                state_root=state_root,
                started=started,
            )
            result["cache_hit"] = False
            return result

    def create(self, lock_file: Path, prefix: Path) -> dict[str, Any]:
        """Create one prefix offline from a verified exact lock and local CAS.

        Parameters
        ----------
        lock_file:
            Absolute engine-facing SHA-256 explicit lock.
        prefix:
            Absolute exact environment prefix.

        Returns
        -------
        dict[str, Any]
            Created-prefix and generation-local package-cache paths.
        """

        lock_file = _require_absolute_file(lock_file, "exact lock")
        prefix = _require_removable_prefix(prefix)
        lock_bytes = lock_file.read_bytes()
        try:
            receipts = parse_exact_lock(lock_bytes)
        except EnvironmentExactnessError as exc:
            raise PermanentEnvironmentOperatorError(str(exc)) from exc
        state_root = self._state_root_for(lock_file)
        registration_path = _registration_path(state_root, prefix)
        with _exclusive_file_lock(registration_path.with_suffix(".lock")):
            if registration_path.is_file():
                self._remove_registered(prefix, state_root, registration_path)
            if prefix.exists() or prefix.is_symlink():
                raise PermanentEnvironmentOperatorError(
                    "create prefix already exists without an operator registration"
                )
            lock_sha256 = hash_bytes(lock_bytes)
            package_cache = (
                state_root
                / "package-caches"
                / f"{_prefix_identifier(prefix)}-{lock_sha256[7:19]}"
            )
            package_cache.mkdir(parents=True, exist_ok=False)
            registration = {
                "protocol_version": PROTOCOL_VERSION,
                "prefix": str(prefix),
                "package_cache": str(package_cache),
                "lock_sha256": lock_sha256,
            }
            atomic_replace_bytes(
                registration_path, canonical_json_bytes(registration) + b"\n"
            )
            try:
                self._stage_package_cache(receipts, package_cache, state_root)
                environment = _clean_environment(
                    prefix=prefix,
                    state_root=state_root,
                    package_cache=package_cache,
                    tool_commands=(self._conda_command,),
                )
                command = (
                    *self._conda_command,
                    "create",
                    "--yes",
                    "--offline",
                    "--prefix",
                    str(prefix),
                    "--file",
                    str(lock_file),
                )
                self._run_required_command(
                    command,
                    environment=environment,
                    timeout_seconds=_CREATE_TIMEOUT_SECONDS,
                    label="conda create",
                )
            except Exception:
                # The registration intentionally survives so lifecycle ``finally`` can
                # remove a partial prefix and cache through the same exact target.
                raise
        return {
            "protocol_version": PROTOCOL_VERSION,
            "prefix": str(prefix),
            "package_cache_path": str(package_cache),
        }

    def probe(self, prefix: Path, probes_value: str) -> dict[str, Any]:
        """Run declared canaries with the prefix interpreter and a clean environment.

        Parameters
        ----------
        prefix:
            Absolute created environment prefix.
        probes_value:
            Either the probe JSON itself or an absolute path to a JSON file.

        Returns
        -------
        dict[str, Any]
            Exactly one ordered boolean receipt per declared canary.
        """

        prefix = _require_absolute_directory(prefix, "environment prefix")
        interpreter = prefix / "bin" / "python"
        if not interpreter.is_file():
            raise PermanentEnvironmentOperatorError(
                f"environment interpreter is missing below {prefix}"
            )
        probes = _load_probe_request(probes_value)
        state_root = self._state_root_for(prefix)
        environment = _clean_environment(prefix=prefix, state_root=state_root)
        results: list[dict[str, Any]] = []
        for module in _string_list(probes.get("imports"), "probe imports"):
            observation = self._run_probe_command(
                (
                    str(interpreter),
                    "-I",
                    "-c",
                    "import importlib,sys; importlib.import_module(sys.argv[1])",
                    module,
                ),
                environment=environment,
                attempts=1,
            )
            results.append(
                _probe_result(
                    f"import:{module}",
                    observation.returncode == 0,
                    observation,
                )
            )
        for raw in _mapping_list(probes.get("export_checks"), "probe export checks"):
            module = _nonempty_string(raw.get("module"), "probe export module")
            attribute = _nonempty_string(raw.get("attribute"), "probe export attribute")
            observation = self._run_probe_command(
                (
                    str(interpreter),
                    "-I",
                    "-c",
                    (
                        "import importlib,sys; "
                        "getattr(importlib.import_module(sys.argv[1]),sys.argv[2])"
                    ),
                    module,
                    attribute,
                ),
                environment=environment,
                attempts=1,
            )
            results.append(
                _probe_result(
                    f"export:{module}:{attribute}",
                    observation.returncode == 0,
                    observation,
                )
            )
        for raw in _mapping_list(
            probes.get("source_build", []), "probe source-build checks"
        ):
            name = _nonempty_string(raw.get("name"), "source-build probe name")
            command = _string_list(raw.get("command"), "source-build probe command")
            max_attempts = raw.get("max_attempts")
            if not isinstance(max_attempts, int) or isinstance(max_attempts, bool):
                raise PermanentEnvironmentOperatorError(
                    "source-build max_attempts must be a positive integer"
                )
            if max_attempts < 1:
                raise PermanentEnvironmentOperatorError(
                    "source-build max_attempts must be a positive integer"
                )
            resolved_command = _prefix_python_command(command, interpreter)
            observation = self._run_probe_command(
                resolved_command,
                environment=environment,
                attempts=max_attempts,
            )
            results.append(
                _probe_result(
                    f"source-build:{name}",
                    observation.returncode == 0,
                    observation,
                )
            )
        return {"protocol_version": PROTOCOL_VERSION, "results": results}

    def remove(self, prefix: Path) -> dict[str, Any]:
        """Remove only an exact registered prefix and its generation-local cache.

        Parameters
        ----------
        prefix:
            Absolute exact prefix previously registered by :meth:`create`.

        Returns
        -------
        dict[str, Any]
            Exact removed prefix; shared CAS objects are retained.
        """

        prefix = _require_removable_prefix(prefix)
        state_root = self._state_root_for(prefix)
        registration_path = _registration_path(state_root, prefix)
        with _exclusive_file_lock(registration_path.with_suffix(".lock")):
            self._remove_registered(prefix, state_root, registration_path)
        return {
            "protocol_version": PROTOCOL_VERSION,
            "prefix": str(prefix),
            "cas_retained": True,
        }

    def _fresh_solve(
        self,
        *,
        environment_file: Path,
        target: str,
        solve_root: Path,
        state_root: Path,
        started: float,
    ) -> dict[str, Any]:
        """Run conda-lock, materialize CAS bytes, and publish a complete solve."""

        with tempfile.TemporaryDirectory(prefix=".solve-", dir=solve_root) as temporary_value:
            temporary_root = Path(temporary_value)
            unified_lock = temporary_root / "conda-lock.yml"
            command = (
                *self._conda_lock_command,
                "lock",
                "--file",
                str(environment_file),
                "-p",
                target,
                "--lockfile",
                str(unified_lock),
                "--mamba",
            )
            self._run_required_command(
                command,
                environment=_clean_environment(
                    state_root=state_root,
                    tool_commands=(self._conda_lock_command, self._conda_command),
                ),
                timeout_seconds=_SOLVE_TIMEOUT_SECONDS,
                label="conda-lock solve",
            )
            if not unified_lock.is_file():
                raise PermanentEnvironmentOperatorError(
                    "conda-lock exited successfully without its requested lockfile"
                )
            packages = _parse_solved_packages(unified_lock, target)
            lock_bytes = _explicit_sha256_lock_bytes(packages)
            export_bytes = _resolved_export_bytes(packages)
            _require_lock_export_match(lock_bytes, export_bytes)
            artifact_bytes = 0
            artifacts: list[dict[str, str]] = []
            for package in packages:
                path, downloaded_bytes = self._materialize_artifact(package, state_root)
                artifact_bytes += downloaded_bytes
                artifacts.append(
                    {
                        "url": package.url,
                        "path": str(path),
                        "sha256": f"sha256:{package.sha256}",
                    }
                )
            published_lock = solve_root / "explicit.lock"
            published_export = solve_root / "resolved.json"
            published_unified = solve_root / "conda-lock.yml"
            atomic_replace_bytes(published_lock, lock_bytes)
            atomic_replace_bytes(published_export, export_bytes)
            atomic_replace_bytes(published_unified, unified_lock.read_bytes())
            manifest = {
                "protocol_version": PROTOCOL_VERSION,
                "intent": environment_file.parent.name,
                "target": target,
                "lock_sha256": hash_bytes(lock_bytes),
                "resolved_export_sha256": hash_bytes(export_bytes),
                "artifacts": artifacts,
            }
            atomic_replace_bytes(
                solve_root / "manifest.json", canonical_json_bytes(manifest) + b"\n"
            )
        return {
            "protocol_version": PROTOCOL_VERSION,
            "lock_path": str(published_lock),
            "resolved_export_path": str(published_export),
            "artifacts": artifacts,
            "artifact_bytes": artifact_bytes,
            "elapsed_seconds": time.monotonic() - started,
        }

    def _cached_solve_result(
        self, solve_root: Path, state_root: Path
    ) -> dict[str, Any] | None:
        """Return a fully reverified solve cache or ``None`` for any cache miss."""

        lock_path = solve_root / "explicit.lock"
        export_path = solve_root / "resolved.json"
        if not lock_path.is_file() or not export_path.is_file():
            return None
        try:
            lock_bytes = lock_path.read_bytes()
            export_bytes = export_path.read_bytes()
            receipts = parse_exact_lock(lock_bytes)
            if parse_resolved_export(export_bytes) != export_bytes:
                return None
            _require_lock_export_match(lock_bytes, export_bytes)
        except (OSError, EnvironmentExactnessError, PermanentEnvironmentOperatorError):
            return None
        artifacts: list[dict[str, str]] = []
        for receipt in receipts:
            path = _cas_path(state_root, receipt.sha256)
            if not path.is_file() or _stream_sha256(path) != receipt.sha256:
                return None
            artifacts.append(
                {"url": receipt.url, "path": str(path), "sha256": receipt.sha256}
            )
        return {
            "protocol_version": PROTOCOL_VERSION,
            "lock_path": str(lock_path),
            "resolved_export_path": str(export_path),
            "artifacts": artifacts,
            "artifact_bytes": 0,
        }

    def _materialize_artifact(
        self, package: SolvedPackage, state_root: Path
    ) -> tuple[Path, int]:
        """Download one immutable artifact into the content-addressed store."""

        destination = _cas_path(state_root, f"sha256:{package.sha256}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file():
            if _stream_sha256(destination) == f"sha256:{package.sha256}":
                return destination, 0
            destination.unlink()
            fsync_directory(destination.parent)
        last_error: BaseException | None = None
        for attempt in range(_TRANSIENT_RETRIES + 1):
            temporary = destination.with_name(
                f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
            )
            try:
                request = urllib.request.Request(
                    package.url,
                    headers={"User-Agent": "torchlens-menagerie-environment-operator/1"},
                )
                digest = hashlib.sha256()
                size = 0
                with urllib.request.urlopen(  # noqa: S310 -- lock-authorized artifact URL
                    request, timeout=_DOWNLOAD_TIMEOUT_SECONDS
                ) as response:
                    with temporary.open("wb") as handle:
                        while True:
                            chunk = response.read(1024 * 1024)
                            if not chunk:
                                break
                            digest.update(chunk)
                            handle.write(chunk)
                            size += len(chunk)
                        handle.flush()
                        os.fsync(handle.fileno())
                if digest.hexdigest() != package.sha256:
                    raise PermanentEnvironmentOperatorError(
                        f"downloaded artifact digest mismatch for {package.filename}"
                    )
                os.replace(temporary, destination)
                fsync_directory(destination.parent)
                return destination, size
            except PermanentEnvironmentOperatorError:
                temporary.unlink(missing_ok=True)
                raise
            except urllib.error.HTTPError as exc:
                temporary.unlink(missing_ok=True)
                if exc.code not in {408, 425, 429} and exc.code < 500:
                    raise PermanentEnvironmentOperatorError(
                        f"artifact service permanently rejected {package.filename}: HTTP {exc.code}"
                    ) from exc
                last_error = exc
            except (OSError, TimeoutError, urllib.error.URLError) as exc:
                temporary.unlink(missing_ok=True)
                last_error = exc
            if attempt < _TRANSIENT_RETRIES:
                self._sleeper(float(2**attempt))
        raise TransientEnvironmentOperatorError(
            f"artifact download remained unavailable for {package.filename}"
        ) from last_error

    def _stage_package_cache(
        self,
        receipts: Sequence[ArtifactReceipt],
        package_cache: Path,
        state_root: Path,
    ) -> None:
        """Stage verified CAS objects under conda's artifact filenames."""

        filenames: dict[str, str] = {}
        for receipt in receipts:
            filename = _artifact_filename(receipt.url)
            previous = filenames.setdefault(filename, receipt.sha256)
            if previous != receipt.sha256:
                raise PermanentEnvironmentOperatorError(
                    f"lock maps artifact filename {filename!r} to multiple digests"
                )
            source = _cas_path(state_root, receipt.sha256)
            if not source.is_file() or _stream_sha256(source) != receipt.sha256:
                raise PermanentEnvironmentOperatorError(
                    f"verified CAS artifact is missing for {filename}"
                )
            destination = package_cache / filename
            shutil.copyfile(source, destination)
            if _stream_sha256(destination) != receipt.sha256:
                raise PermanentEnvironmentOperatorError(
                    f"staged package-cache artifact changed for {filename}"
                )
        fsync_directory(package_cache)

    def _run_required_command(
        self,
        command: Sequence[str],
        *,
        environment: Mapping[str, str],
        timeout_seconds: float,
        label: str,
    ) -> CommandObservation:
        """Run a required subprocess with typed transient-only retries."""

        last_error: BaseException | None = None
        for attempt in range(_TRANSIENT_RETRIES + 1):
            try:
                observation = self._runner(command, environment, timeout_seconds)
            except FileNotFoundError as exc:
                raise PermanentEnvironmentOperatorError(
                    f"{label} executable is unavailable"
                ) from exc
            except (OSError, subprocess.SubprocessError) as exc:
                last_error = exc
            else:
                if observation.returncode == 0:
                    return observation
                detail = _combined_detail(observation)
                if observation.returncode != EXIT_RETRYABLE and not _is_transient_detail(
                    detail
                ):
                    raise PermanentEnvironmentOperatorError(
                        f"{label} failed permanently: {_bounded_detail(detail)}"
                    )
                last_error = RuntimeError(_bounded_detail(detail))
            if attempt < _TRANSIENT_RETRIES:
                self._sleeper(float(2**attempt))
        raise TransientEnvironmentOperatorError(
            f"{label} remained unavailable after {_TRANSIENT_RETRIES + 1} attempts"
        ) from last_error

    def _run_probe_command(
        self,
        command: Sequence[str],
        *,
        environment: Mapping[str, str],
        attempts: int,
    ) -> CommandObservation:
        """Run a canary, retrying only spawn/timeout infrastructure failures."""

        last_error: BaseException | None = None
        for attempt in range(attempts):
            try:
                return self._runner(command, environment, _PROBE_TIMEOUT_SECONDS)
            except FileNotFoundError as exc:
                raise PermanentEnvironmentOperatorError(
                    "probe executable is unavailable"
                ) from exc
            except (OSError, subprocess.SubprocessError) as exc:
                last_error = exc
                if attempt + 1 < attempts:
                    self._sleeper(float(2**attempt))
        raise TransientEnvironmentOperatorError(
            f"probe subprocess remained unavailable after {attempts} attempts"
        ) from last_error

    def _remove_registered(
        self, prefix: Path, state_root: Path, registration_path: Path
    ) -> None:
        """Validate and remove one prefix registration without touching shared CAS."""

        if not registration_path.is_file():
            raise PermanentEnvironmentOperatorError(
                "remove target is not an exact registered environment prefix"
            )
        registration = _load_mapping(registration_path, "prefix registration")
        if registration.get("protocol_version") != PROTOCOL_VERSION:
            raise PermanentEnvironmentOperatorError("prefix registration version is invalid")
        if registration.get("prefix") != str(prefix):
            raise PermanentEnvironmentOperatorError("prefix registration target mismatch")
        package_cache_value = registration.get("package_cache")
        if not isinstance(package_cache_value, str):
            raise PermanentEnvironmentOperatorError(
                "prefix registration package cache is invalid"
            )
        package_cache = Path(package_cache_value)
        expected_cache_root = (state_root / "package-caches").resolve()
        if (
            not package_cache.is_absolute()
            or package_cache.resolve(strict=False).parent != expected_cache_root
            or package_cache.is_symlink()
        ):
            raise PermanentEnvironmentOperatorError(
                "registered package cache escapes its exact operator root"
            )
        if prefix.is_symlink():
            raise PermanentEnvironmentOperatorError("registered prefix became a symlink")
        if prefix.exists():
            shutil.rmtree(prefix)
        if package_cache.exists():
            shutil.rmtree(package_cache)
        registration_path.unlink()
        fsync_directory(registration_path.parent)

    def _state_root_for(self, anchor: Path) -> Path:
        """Return and create the explicit or repository-inferred state root."""

        if self._state_root is not None:
            root = self._state_root
        else:
            configured = os.environ.get("MENAGERIE_ENVIRONMENT_STATE_ROOT")
            if configured:
                candidate = Path(configured)
                if not candidate.is_absolute():
                    raise PermanentEnvironmentOperatorError(
                        "MENAGERIE_ENVIRONMENT_STATE_ROOT must be absolute"
                    )
                root = candidate.resolve()
            else:
                repository = _find_repository_root(anchor)
                root = repository / ".crawl-local" / "environment-operator"
        root.mkdir(parents=True, exist_ok=True)
        return root


def _default_tool_command(environment_key: str, fixed_path: Path) -> tuple[str, ...]:
    """Resolve one configured, PATH-provided, or machine-baseline tool command."""

    configured = os.environ.get(environment_key)
    if configured:
        return (configured,)
    discovered = shutil.which(fixed_path.name)
    if discovered is not None:
        return (discovered,)
    return (str(fixed_path),)


def _default_command_runner(
    command: Sequence[str], environment: Mapping[str, str], timeout_seconds: float
) -> CommandObservation:
    """Run one argv-only child with captured bounded text streams."""

    completed = subprocess.run(
        list(command),
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        start_new_session=True,
        env=dict(environment),
    )
    return CommandObservation(
        returncode=completed.returncode,
        stdout=completed.stdout[-_DETAIL_LIMIT:],
        stderr=completed.stderr[-_DETAIL_LIMIT:],
    )


def _load_mapping(path: Path, label: str) -> Mapping[str, Any]:
    """Load one JSON/YAML file and require a string-keyed mapping."""

    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise PermanentEnvironmentOperatorError(f"{label} is invalid: {path}") from exc
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise PermanentEnvironmentOperatorError(f"{label} must be a string-keyed mapping")
    return value


def _refuse_pip_sections(value: Any) -> None:
    """Refuse any mapping-declared pip installation lane recursively."""

    if isinstance(value, Mapping):
        if "pip" in value:
            raise PermanentEnvironmentOperatorError(
                "pip sections are unsupported by exact conda-meta inventory"
            )
        for nested in value.values():
            _refuse_pip_sections(nested)
    elif isinstance(value, list):
        for nested in value:
            _refuse_pip_sections(nested)


def _parse_solved_packages(path: Path, target: str) -> tuple[SolvedPackage, ...]:
    """Parse target conda packages and derive build strings from artifact filenames."""

    value = _load_mapping(path, "conda-lock output")
    raw_packages = value.get("package")
    if not isinstance(raw_packages, list) or not raw_packages:
        raise PermanentEnvironmentOperatorError("conda-lock output has no package rows")
    packages: list[SolvedPackage] = []
    for raw in raw_packages:
        if not isinstance(raw, Mapping):
            raise PermanentEnvironmentOperatorError(
                "conda-lock output contains a non-object package row"
            )
        platform = _nonempty_string(raw.get("platform"), "locked package platform")
        if platform != target:
            continue
        manager = _nonempty_string(raw.get("manager"), "locked package manager")
        if manager != "conda":
            raise PermanentEnvironmentOperatorError(
                f"locked manager {manager!r} is unsupported; pip is out of scope"
            )
        name = _nonempty_string(raw.get("name"), "locked package name")
        version = _nonempty_string(raw.get("version"), "locked package version")
        url = _nonempty_string(raw.get("url"), "locked package URL")
        raw_hash = raw.get("hash")
        if not isinstance(raw_hash, Mapping):
            raise PermanentEnvironmentOperatorError("locked package hash is missing")
        sha256 = _nonempty_string(raw_hash.get("sha256"), "locked package SHA-256")
        if _SHA256_PATTERN.fullmatch(sha256) is None:
            raise PermanentEnvironmentOperatorError(
                "locked package SHA-256 must be 64 lowercase hex characters"
            )
        filename = _artifact_filename(url)
        build = _build_from_filename(filename, name, version)
        packages.append(
            SolvedPackage(
                name=name,
                version=version,
                build=build,
                url=url,
                sha256=sha256,
                filename=filename,
            )
        )
    if not packages:
        raise PermanentEnvironmentOperatorError(
            f"conda-lock output has no conda packages for target {target!r}"
        )
    packages.sort(key=lambda package: (package.url, package.sha256))
    urls = [package.url for package in packages]
    if len(urls) != len(set(urls)):
        raise PermanentEnvironmentOperatorError(
            "conda-lock output contains duplicate target artifact URLs"
        )
    return tuple(packages)


def _artifact_filename(url: str) -> str:
    """Return a safe conda artifact filename from one exact URL."""

    filename = unquote(Path(urlsplit(url).path).name)
    if (
        not filename
        or filename in {".", ".."}
        or "/" in filename
        or "\\" in filename
        or not filename.endswith((".conda", ".tar.bz2"))
    ):
        raise PermanentEnvironmentOperatorError(
            "locked artifact URL has no supported conda filename"
        )
    return filename


def _build_from_filename(filename: str, name: str, version: str) -> str:
    """Derive the conda build string absent from unified conda-lock rows."""

    stem = (
        filename[: -len(".tar.bz2")]
        if filename.endswith(".tar.bz2")
        else filename[: -len(".conda")]
    )
    prefix = f"{name}-{version}-"
    if not stem.startswith(prefix) or not stem[len(prefix) :]:
        raise PermanentEnvironmentOperatorError(
            f"artifact filename does not encode name/version/build: {filename}"
        )
    return stem[len(prefix) :]


def _explicit_sha256_lock_bytes(packages: Sequence[SolvedPackage]) -> bytes:
    """Synthesize the engine-facing SHA-256 explicit lock without conda-lock render."""

    lines = [f"{package.url}#{package.sha256}" for package in packages]
    if len(lines) != len(set(lines)):
        raise PermanentEnvironmentOperatorError(
            "solved packages do not form a unique explicit lock"
        )
    return ("@EXPLICIT\n" + "\n".join(lines) + "\n").encode()


def _resolved_export_bytes(packages: Sequence[SolvedPackage]) -> bytes:
    """Serialize the five-field inventory from the same solved package rows."""

    rows = [
        {
            "name": package.name,
            "version": package.version,
            "build": package.build,
            "url": package.url,
            "sha256": f"sha256:{package.sha256}",
        }
        for package in packages
    ]
    rows.sort(
        key=lambda row: (
            row["name"],
            row["version"],
            row["build"],
            row["url"],
            row["sha256"],
        )
    )
    return canonical_json_bytes({"packages": rows}) + b"\n"


def _require_lock_export_match(lock_bytes: bytes, export_bytes: bytes) -> None:
    """Require exact one-to-one URL/digest identity across lock and export."""

    try:
        lock_receipts = parse_exact_lock(lock_bytes)
        canonical_export = parse_resolved_export(export_bytes)
        value = json.loads(canonical_export)
    except (EnvironmentExactnessError, json.JSONDecodeError) as exc:
        raise PermanentEnvironmentOperatorError(str(exc)) from exc
    packages = value.get("packages") if isinstance(value, Mapping) else None
    if not isinstance(packages, list):
        raise PermanentEnvironmentOperatorError("resolved export has no package rows")
    export_receipts = tuple(
        (row.get("url"), row.get("sha256"))
        for row in packages
        if isinstance(row, Mapping)
    )
    if len(export_receipts) != len(packages) or set(export_receipts) != {
        (receipt.url, receipt.sha256) for receipt in lock_receipts
    }:
        raise PermanentEnvironmentOperatorError(
            "target lock and resolved export artifact identities differ"
        )


def _cas_path(state_root: Path, sha256: str) -> Path:
    """Return the content-addressed path for one canonical SHA-256."""

    digest = sha256.removeprefix("sha256:")
    if _SHA256_PATTERN.fullmatch(digest) is None:
        raise PermanentEnvironmentOperatorError("CAS digest is not canonical SHA-256")
    return state_root / "artifact-cas" / "sha256" / digest


def _stream_sha256(path: Path) -> str:
    """Hash one artifact without loading the package into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _clean_environment(
    *,
    state_root: Path,
    prefix: Path | None = None,
    package_cache: Path | None = None,
    tool_commands: Sequence[Sequence[str]] = (),
) -> dict[str, str]:
    """Build a bounded child environment with no inherited proxy or Python paths."""

    operator_home = state_root / "home"
    temporary = state_root / "tmp"
    operator_home.mkdir(parents=True, exist_ok=True)
    temporary.mkdir(parents=True, exist_ok=True)
    path_entries = [
        str(Path(command[0]).parent)
        for command in tool_commands
        if command and Path(command[0]).is_absolute()
    ]
    if prefix is not None:
        path_entries.insert(0, str(prefix / "bin"))
    path_entries.extend(("/usr/bin", "/bin"))
    environment = {
        "HOME": str(operator_home),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
        "PATH": os.pathsep.join(path_entries),
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
        "PIP_NO_INDEX": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "TMPDIR": str(temporary),
        "TRANSFORMERS_OFFLINE": "1",
        "WANDB_MODE": "offline",
    }
    for key in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"):
        value = os.environ.get(key)
        if value:
            environment[key] = value
    if package_cache is not None:
        environment["CONDA_PKGS_DIRS"] = str(package_cache)
        environment["CONDA_OFFLINE"] = "true"
    return environment


def _load_probe_request(value: str) -> Mapping[str, Any]:
    """Parse an inline probe object or an absolute JSON request path."""

    candidate = Path(value)
    if candidate.is_absolute() and candidate.is_file():
        try:
            parsed = json.loads(candidate.read_bytes())
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise PermanentEnvironmentOperatorError("probe request file is invalid") from exc
    else:
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise PermanentEnvironmentOperatorError(
                "probe request must be inline JSON or an absolute existing path"
            ) from exc
    if not isinstance(parsed, dict) or set(parsed) != {
        "imports",
        "export_checks",
        "source_build",
    }:
        raise PermanentEnvironmentOperatorError(
            "probe request must contain exactly imports, export_checks, and source_build"
        )
    return parsed


def _mapping_list(value: Any, label: str) -> tuple[Mapping[str, Any], ...]:
    """Require one list containing only string-keyed mappings."""

    if not isinstance(value, list):
        raise PermanentEnvironmentOperatorError(f"{label} must be a list")
    rows: list[Mapping[str, Any]] = []
    for raw in value:
        if not isinstance(raw, dict) or not all(isinstance(key, str) for key in raw):
            raise PermanentEnvironmentOperatorError(
                f"{label} must contain string-keyed objects"
            )
        rows.append(raw)
    return tuple(rows)


def _string_list(value: Any, label: str) -> tuple[str, ...]:
    """Require one list containing only non-empty strings."""

    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise PermanentEnvironmentOperatorError(
            f"{label} must contain only non-empty strings"
        )
    return tuple(value)


def _nonempty_string(value: Any, label: str) -> str:
    """Require and return one non-empty string."""

    if not isinstance(value, str) or not value:
        raise PermanentEnvironmentOperatorError(f"{label} must be a non-empty string")
    return value


def _prefix_python_command(command: Sequence[str], interpreter: Path) -> tuple[str, ...]:
    """Bind a declared source-build Python canary to the prefix interpreter."""

    if not command or command[0] not in {"python", "python3"}:
        raise PermanentEnvironmentOperatorError(
            "source-build canaries must invoke the prefix Python interpreter"
        )
    return (str(interpreter), *command[1:])


def _probe_result(
    name: str, passed: bool, observation: CommandObservation
) -> dict[str, Any]:
    """Build one bounded, single-line canary receipt."""

    detail = _bounded_detail(_combined_detail(observation), limit=_PROBE_DETAIL_LIMIT)
    if not detail:
        detail = "ok" if passed else f"exit {observation.returncode}"
    return {"name": name, "passed": passed, "detail": detail}


def _combined_detail(observation: CommandObservation) -> str:
    """Merge both child streams so one non-empty stream cannot mask the other."""

    return "\n".join(
        value for value in (observation.stderr, observation.stdout) if value
    )


def _bounded_detail(value: str, *, limit: int = _DETAIL_LIMIT) -> str:
    """Collapse and bound externally controlled diagnostic text."""

    collapsed = " ".join(value.split())
    return collapsed[-limit:]


def _is_transient_detail(value: str) -> bool:
    """Return whether child diagnostics carry a known retryable transport marker."""

    lowered = value.lower()
    return any(marker in lowered for marker in _TRANSIENT_MARKERS)


def _require_safe_component(value: str, label: str) -> None:
    """Require a path-component-safe intent or target identifier."""

    if _SAFE_COMPONENT_PATTERN.fullmatch(value) is None:
        raise PermanentEnvironmentOperatorError(f"{label} is not a safe path component")


def _require_absolute_file(path: Path, label: str) -> Path:
    """Require one absolute regular file and return its normalized path."""

    if not path.is_absolute():
        raise PermanentEnvironmentOperatorError(f"{label} path must be absolute")
    resolved = path.resolve()
    if not resolved.is_file():
        raise PermanentEnvironmentOperatorError(f"{label} is missing: {resolved}")
    return resolved


def _require_absolute_directory(path: Path, label: str) -> Path:
    """Require one absolute directory and return its normalized path."""

    if not path.is_absolute():
        raise PermanentEnvironmentOperatorError(f"{label} path must be absolute")
    resolved = path.resolve()
    if not resolved.is_dir():
        raise PermanentEnvironmentOperatorError(f"{label} is missing: {resolved}")
    return resolved


def _require_removable_prefix(path: Path) -> Path:
    """Refuse globbed, relative, symlinked, root, or otherwise broad prefixes."""

    raw = str(path)
    if any(character in raw for character in _GLOB_CHARACTERS):
        raise PermanentEnvironmentOperatorError("environment prefix cannot contain glob syntax")
    if not path.is_absolute():
        raise PermanentEnvironmentOperatorError("environment prefix must be absolute")
    normalized = path.resolve(strict=False)
    if normalized == Path(normalized.anchor) or normalized.parent == Path(normalized.anchor):
        raise PermanentEnvironmentOperatorError("environment prefix is too broad to remove")
    return normalized


def _find_repository_root(anchor: Path) -> Path:
    """Find the containing TorchLens repository from an action path."""

    resolved = anchor.resolve(strict=False)
    start = resolved if resolved.is_dir() else resolved.parent
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "menagerie" / "crawler"
        ).is_dir():
            return candidate
    cwd = Path.cwd().resolve()
    if (cwd / "pyproject.toml").is_file() and (cwd / "menagerie" / "crawler").is_dir():
        return cwd
    raise PermanentEnvironmentOperatorError(
        "cannot infer operator state root; pass --state-root explicitly"
    )


def _prefix_identifier(prefix: Path) -> str:
    """Return a stable filesystem-safe identifier for one exact prefix."""

    return hashlib.sha256(str(prefix).encode()).hexdigest()[:24]


def _registration_path(state_root: Path, prefix: Path) -> Path:
    """Return the exact durable registration path for one prefix."""

    root = state_root / "prefix-registrations"
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{_prefix_identifier(prefix)}.json"


@contextmanager
def _exclusive_file_lock(path: Path) -> Iterator[None]:
    """Hold one process-exclusive solve or prefix mutation lock."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _emit_telemetry(
    state_root: Path | None, action: str, status: str, error_kind: str | None = None
) -> None:
    """Append one bounded JSONL operator event without external diagnostic text."""

    if state_root is None:
        return
    path = state_root / "telemetry.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "protocol_version": PROTOCOL_VERSION,
        "action": action,
        "status": status,
        "error_kind": error_kind,
    }
    line = canonical_json_bytes(event) + b"\n"
    with path.open("ab") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def build_parser() -> argparse.ArgumentParser:
    """Build the direct argv action interface consumed by the crawler driver."""

    parser = argparse.ArgumentParser(prog="operator_environment.py")
    parser.add_argument("--state-root", type=Path)
    parser.add_argument("--version", action="version", version=PROTOCOL_VERSION)
    subparsers = parser.add_subparsers(dest="action", required=True)
    solve = subparsers.add_parser("solve")
    solve.add_argument("environment_file", type=Path)
    solve.add_argument("target")
    solve.add_argument("--force-resolve", action="store_true")
    create = subparsers.add_parser("create")
    create.add_argument("lock_file", type=Path)
    create.add_argument("prefix", type=Path)
    probe = subparsers.add_parser("probe")
    probe.add_argument("prefix", type=Path)
    probe.add_argument("probes")
    remove = subparsers.add_parser("remove")
    remove.add_argument("prefix", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch one action and map typed outcomes to the common operator exits."""

    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        operator = EnvironmentOperator(state_root=arguments.state_root)
        action = str(arguments.action)
        if action == "solve":
            result = operator.solve(
                arguments.environment_file,
                arguments.target,
                force_resolve=bool(arguments.force_resolve),
            )
        elif action == "create":
            result = operator.create(arguments.lock_file, arguments.prefix)
        elif action == "probe":
            result = operator.probe(arguments.prefix, arguments.probes)
        elif action == "remove":
            result = operator.remove(arguments.prefix)
        else:
            raise AssertionError(f"unhandled environment action: {action}")
    except TransientEnvironmentOperatorError as exc:
        _emit_telemetry(arguments.state_root, str(arguments.action), "retryable", type(exc).__name__)
        print(
            json.dumps(
                {"status": "retryable", "error": _bounded_detail(str(exc))},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return EXIT_RETRYABLE
    except (PermanentEnvironmentOperatorError, OSError) as exc:
        _emit_telemetry(arguments.state_root, str(arguments.action), "permanent", type(exc).__name__)
        print(
            json.dumps(
                {"status": "permanent", "error": _bounded_detail(str(exc))},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return EXIT_PERMANENT
    _emit_telemetry(arguments.state_root, str(arguments.action), "success")
    print(json.dumps(result, sort_keys=True))
    return EXIT_SUCCESS


if __name__ == "__main__":
    raise SystemExit(main())
