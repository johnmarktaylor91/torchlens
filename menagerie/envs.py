"""Pixi environment manager for clean menagerie validation islands."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

import yaml

from menagerie.catalog import CatalogRow, build_canonical_rows
from menagerie.ledger import (
    Status,
    VerificationRun,
    append_verification_run,
    connect as connect_ledger,
    python_version,
    runner_host,
    utc_now,
)
from menagerie.recipe import build_input_for_row, instantiate_model
from menagerie.runtime import dependency_plan


REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = Path(__file__).resolve().parent / "data" / "validation_envs.yml"
LOCKS_DIR = Path(__file__).resolve().parent / "locks"
DEFAULT_CACHE_ROOT = Path.home() / ".cache" / "torchlens" / "menagerie" / "envs"
BUILD_MARKER = "torchlens_env_build.json"
LOCK_MANIFEST_SUFFIX = ".pixi.toml"
LOCK_FILE_SUFFIX = ".pixi.lock"
BASE_STATUS_MODULES = {"builtins", "menagerie", "pathlib", "torch"}
TORCHLENS_RUNTIME_PYPI = (
    "graphviz",
    "safetensors>=0.4",
    "tqdm",
    "typing_extensions>=4.0",
    "pillow>=9",
)
BuildStatus = Literal["built", "cached", "install_failed", "env_unavailable"]


@dataclass(frozen=True)
class StatusPolicy:
    """Status policy for one validation island.

    Parameters
    ----------
    default_status:
        Default posture for an unproven island.
    unavailable_status:
        Ledger status used when the environment cannot be made available.
    failure_status:
        Ledger status used when installation or smoke fails.
    notes:
        Optional human-readable policy notes.
    """

    default_status: str
    unavailable_status: str
    failure_status: str
    notes: str = ""


@dataclass(frozen=True)
class EnvSpec:
    """Normalized validation island specification.

    Parameters
    ----------
    key:
        Registry island key.
    description:
        Human-readable island description.
    selectors:
        Raw selector mapping from the registry.
    python:
        Python version string.
    torch_pin:
        Torch package pin summary.
    torch_index_url:
        Optional torch wheel index URL.
    pypi_options:
        PyPI resolver options for pixi.
    conda_packages:
        Conda dependencies.
    pypi_packages:
        PyPI dependencies.
    smoke_stable_ids:
        Stable IDs used for import and trace smoke tests.
    expected_device:
        Expected device for validation.
    disk_estimate_gib:
        Projected disk footprint.
    status_policy:
        Status policy for unavailable or failed envs.
    """

    key: str
    description: str
    selectors: dict[str, Any]
    python: str
    torch_pin: str
    torch_index_url: str | None
    pypi_options: dict[str, Any]
    conda_packages: tuple[str, ...]
    pypi_packages: tuple[str, ...]
    smoke_stable_ids: tuple[str, ...]
    expected_device: str
    disk_estimate_gib: float
    status_policy: StatusPolicy


@dataclass(frozen=True)
class EnvRegistry:
    """Loaded validation environment registry.

    Parameters
    ----------
    path:
        Source YAML path.
    cache_root:
        Managed cache root.
    disk_free_reserve_gib:
        Required free-space floor.
    base_modules:
        Top-level modules considered base-capable.
    fallback_dependency_env:
        Island used for unmatched external dependency rows.
    islands:
        Normalized island mapping.
    """

    path: Path
    cache_root: Path
    disk_free_reserve_gib: float
    base_modules: frozenset[str]
    fallback_dependency_env: str
    islands: dict[str, EnvSpec]


@dataclass(frozen=True)
class LockResult:
    """Result from locking one pixi island.

    Parameters
    ----------
    env_key:
        Island key.
    manifest_path:
        Committed pixi manifest path.
    lock_path:
        Committed pixi lock path.
    lock_hash:
        SHA-256 over manifest and lock bytes.
    returncode:
        Pixi command return code.
    message:
        Command output or status note.
    """

    env_key: str
    manifest_path: Path
    lock_path: Path
    lock_hash: str
    returncode: int
    message: str


@dataclass(frozen=True)
class BuildResult:
    """Result from building or reusing a managed pixi environment.

    Parameters
    ----------
    env_key:
        Island key.
    status:
        Build status.
    env_hash:
        Environment identity stored in ledger rows.
    lock_hash:
        Lock hash used for cache keying.
    project_dir:
        Managed pixi project directory.
    message:
        Human-readable result detail.
    """

    env_key: str
    status: BuildStatus
    env_hash: str
    lock_hash: str
    project_dir: Path
    message: str


@dataclass(frozen=True)
class DiskCheckResult:
    """Result from enforcing the managed-cache disk floor.

    Parameters
    ----------
    allowed:
        Whether the projected build may proceed.
    free_gib:
        Free GiB after any managed-root eviction.
    evicted:
        Managed cache directories removed.
    reason:
        Optional refusal reason.
    """

    allowed: bool
    free_gib: float
    evicted: tuple[Path, ...]
    reason: str = ""


@dataclass(frozen=True)
class CommandResult:
    """Small subprocess result used by the env manager.

    Parameters
    ----------
    returncode:
        Process return code.
    stdout:
        Captured standard output.
    stderr:
        Captured standard error.
    """

    returncode: int
    stdout: str
    stderr: str


def _expand_path(path: str | Path) -> Path:
    """Expand a registry path.

    Parameters
    ----------
    path:
        Path string or object.

    Returns
    -------
    pathlib.Path
        Expanded path.
    """

    return Path(path).expanduser()


def load_registry(path: Path = REGISTRY_PATH) -> EnvRegistry:
    """Load the validation environment registry.

    Parameters
    ----------
    path:
        Registry YAML path.

    Returns
    -------
    EnvRegistry
        Normalized registry.
    """

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    pixi = payload.get("pixi", {})
    assignment = payload.get("assignment", {})
    islands = {
        key: _normalize_env_spec(key, value) for key, value in payload.get("islands", {}).items()
    }
    cache_root = _expand_path(pixi.get("managed_cache_root", DEFAULT_CACHE_ROOT))
    return EnvRegistry(
        path=path,
        cache_root=cache_root,
        disk_free_reserve_gib=float(pixi.get("disk_free_reserve_gib", 25.0)),
        base_modules=frozenset(str(item) for item in assignment.get("base_modules", [])),
        fallback_dependency_env=str(assignment.get("fallback_dependency_env", "gpu_tail")),
        islands=islands,
    )


def _normalize_env_spec(key: str, payload: dict[str, Any]) -> EnvSpec:
    """Normalize one raw registry island.

    Parameters
    ----------
    key:
        Island key.
    payload:
        Raw YAML mapping.

    Returns
    -------
    EnvSpec
        Normalized island spec.
    """

    packages = payload.get("packages", {})
    policy = payload.get("status_policy", {})
    return EnvSpec(
        key=key,
        description=str(payload.get("description", "")),
        selectors=dict(payload.get("selectors", {})),
        python=str(payload.get("python", "3.11")),
        torch_pin=str(payload.get("torch_pin", "current")),
        torch_index_url=payload.get("torch_index_url"),
        pypi_options=dict(payload.get("pypi_options", {})),
        conda_packages=tuple(str(item) for item in packages.get("conda", [])),
        pypi_packages=tuple(str(item) for item in packages.get("pypi", [])),
        smoke_stable_ids=tuple(str(item) for item in payload.get("smoke_stable_ids", [])),
        expected_device=str(payload.get("expected_device", "cpu")),
        disk_estimate_gib=float(payload.get("disk_estimate_gib", 0.0)),
        status_policy=StatusPolicy(
            default_status=str(policy.get("default_status", "env_unavailable")),
            unavailable_status=str(policy.get("unavailable_status", "env_unavailable")),
            failure_status=str(policy.get("failure_status", "install_failed")),
            notes=str(policy.get("notes", "")),
        ),
    )


def _regexes(spec: EnvSpec, selector_key: str) -> tuple[re.Pattern[str], ...]:
    """Compile selector regexes.

    Parameters
    ----------
    spec:
        Environment spec.
    selector_key:
        Selector list key.

    Returns
    -------
    tuple[re.Pattern[str], ...]
        Compiled regular expressions.
    """

    return tuple(
        re.compile(str(pattern), re.IGNORECASE) for pattern in spec.selectors.get(selector_key, [])
    )


def _matches_any(patterns: Sequence[re.Pattern[str]], values: Sequence[str]) -> bool:
    """Return whether any pattern matches any value.

    Parameters
    ----------
    patterns:
        Compiled regexes.
    values:
        Values to test.

    Returns
    -------
    bool
        Whether at least one match exists.
    """

    return any(pattern.search(value) for pattern in patterns for value in values)


def _row_matches_env(row: CatalogRow, spec: EnvSpec) -> bool:
    """Return whether a catalog row matches a specialized island.

    Parameters
    ----------
    row:
        Catalog row.
    spec:
        Environment spec.

    Returns
    -------
    bool
        Whether the row matches.
    """

    if spec.key == "base":
        return False
    plan = dependency_plan(row)
    if _matches_any(_regexes(spec, "exclude_zoo_patterns"), (row.zoo,)):
        return False
    if _matches_any(_regexes(spec, "exclude_name_patterns"), (row.name,)):
        return False
    if _matches_any(_regexes(spec, "exclude_module_patterns"), plan.top_modules):
        return False
    return (
        _matches_any(_regexes(spec, "cluster_patterns"), (plan.cluster_key,))
        or _matches_any(_regexes(spec, "module_patterns"), plan.top_modules)
        or _matches_any(_regexes(spec, "zoo_patterns"), (row.zoo,))
        or _matches_any(_regexes(spec, "name_patterns"), (row.name,))
    )


def _row_needs_dependency_env(row: CatalogRow, registry: EnvRegistry) -> bool:
    """Return whether a row has non-base external dependency signals.

    Parameters
    ----------
    row:
        Catalog row.
    registry:
        Environment registry.

    Returns
    -------
    bool
        Whether a non-base dependency island is needed.
    """

    plan = dependency_plan(row)
    external_modules = [
        module for module in plan.top_modules if module not in registry.base_modules
    ]
    return bool(plan.packages or external_modules)


def env_for_row(row: CatalogRow, registry: EnvRegistry | None = None) -> str:
    """Assign one catalog row to exactly one validation island.

    Parameters
    ----------
    row:
        Catalog row.
    registry:
        Optional loaded registry.

    Returns
    -------
    str
        Island key.

    Raises
    ------
    ValueError
        If specialized selectors overlap.
    """

    active_registry = registry or load_registry()
    if not _row_needs_dependency_env(row, active_registry):
        return "base"
    matches = [key for key, spec in active_registry.islands.items() if _row_matches_env(row, spec)]
    unique_matches = sorted(set(matches))
    if len(unique_matches) > 1:
        raise ValueError(
            f"{row.stable_id} matched multiple validation envs: {', '.join(unique_matches)}"
        )
    if unique_matches:
        return unique_matches[0]
    return active_registry.fallback_dependency_env


def assign(rows: Sequence[CatalogRow], registry: EnvRegistry | None = None) -> dict[str, str]:
    """Assign rows to validation islands.

    Parameters
    ----------
    rows:
        Catalog rows.
    registry:
        Optional loaded registry.

    Returns
    -------
    dict[str, str]
        Mapping from stable ID to island key.
    """

    active_registry = registry or load_registry()
    assignments: dict[str, str] = {}
    for row in rows:
        env_key = env_for_row(row, active_registry)
        if row.stable_id in assignments:
            raise ValueError(f"duplicate stable_id in assignment input: {row.stable_id}")
        assignments[row.stable_id] = env_key
    return assignments


def pixi_bin() -> Path:
    """Return the pixi binary path.

    Returns
    -------
    pathlib.Path
        Pixi executable path.

    Raises
    ------
    FileNotFoundError
        If pixi cannot be found.
    """

    configured = os.environ.get("PIXI_BIN")
    candidates = [
        Path(configured) if configured else None,
        shutil.which("pixi"),
        Path("/tmp/tlpixi/bin/pixi"),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        path = Path(candidate)
        if path.exists():
            return path
    raise FileNotFoundError("pixi not found; install via conda-forge or set PIXI_BIN")


def _split_package_spec(spec: str) -> tuple[str, str]:
    """Split a package spec into name and version constraint.

    Parameters
    ----------
    spec:
        Package spec.

    Returns
    -------
    tuple[str, str]
        Package name and constraint.
    """

    for marker in ("==", ">=", "<=", "~=", ">", "<"):
        if marker in spec:
            name, constraint = spec.split(marker, 1)
            return name.strip(), f"{marker}{constraint.strip()}"
    if "=" in spec and not spec.startswith(("http://", "https://")):
        name, version_part = spec.split("=", 1)
        if name.strip() == "python":
            return name.strip(), f"{version_part.strip()}.*"
        return name.strip(), f"=={version_part.strip()}"
    return spec.strip(), "*"


def _split_pypi_package_spec(spec: str) -> tuple[str, str, tuple[str, ...]]:
    """Split a PyPI package spec into name, constraint, and extras.

    Parameters
    ----------
    spec:
        PyPI package spec.

    Returns
    -------
    tuple[str, str, tuple[str, ...]]
        Package name, constraint, and extras.
    """

    name, constraint = _split_package_spec(spec)
    if "[" not in name or not name.endswith("]"):
        return name, constraint, ()
    base_name, raw_extras = name[:-1].split("[", 1)
    extras = tuple(extra.strip() for extra in raw_extras.split(",") if extra.strip())
    return base_name, constraint, extras


def _toml_quote(value: str) -> str:
    """Return a TOML string literal.

    Parameters
    ----------
    value:
        String value.

    Returns
    -------
    str
        JSON-compatible TOML quoted string.
    """

    return json.dumps(value)


def _toml_key(value: str) -> str:
    """Return a TOML key.

    Parameters
    ----------
    value:
        Raw key.

    Returns
    -------
    str
        Quoted TOML key.
    """

    return _toml_quote(value)


def render_pixi_manifest(spec: EnvSpec) -> str:
    """Render a deterministic pixi manifest for one island.

    Parameters
    ----------
    spec:
        Environment spec.

    Returns
    -------
    str
        Pixi TOML manifest text.
    """

    lines = [
        "[workspace]",
        f"name = {_toml_quote(f'torchlens-menagerie-{spec.key}')}",
        'channels = ["conda-forge"]',
        'platforms = ["linux-64"]',
        'version = "0.1.0"',
        "",
        "[dependencies]",
    ]
    conda_specs = spec.conda_packages or (f"python={spec.python}", "pip")
    for package_spec in conda_specs:
        name, constraint = _split_package_spec(package_spec)
        lines.append(f"{_toml_key(name)} = {_toml_quote(constraint)}")
    if spec.pypi_options:
        lines.extend(["", "[pypi-options]"])
        extra_urls = spec.pypi_options.get("extra_index_urls", [])
        if extra_urls:
            rendered = ", ".join(_toml_quote(str(url)) for url in extra_urls)
            lines.append(f"extra-index-urls = [{rendered}]")
        index_strategy = spec.pypi_options.get("index_strategy")
        if index_strategy:
            lines.append(f"index-strategy = {_toml_quote(str(index_strategy))}")
        elif extra_urls:
            lines.append('index-strategy = "unsafe-best-match"')
        no_build_isolation = spec.pypi_options.get("no_build_isolation", [])
        if no_build_isolation:
            rendered = ", ".join(_toml_quote(str(item)) for item in no_build_isolation)
            lines.append(f"no-build-isolation = [{rendered}]")
    pypi_packages = _with_torchlens_runtime_deps(spec.pypi_packages)
    if pypi_packages:
        lines.extend(["", "[pypi-dependencies]"])
        for package_spec in pypi_packages:
            name, constraint, extras = _split_pypi_package_spec(package_spec)
            if extras:
                rendered_extras = ", ".join(_toml_quote(extra) for extra in extras)
                lines.append(
                    f"{_toml_key(name)} = "
                    f"{{ version = {_toml_quote(constraint)}, extras = [{rendered_extras}] }}"
                )
            else:
                lines.append(f"{_toml_key(name)} = {_toml_quote(constraint)}")
    return "\n".join(lines) + "\n"


def _with_torchlens_runtime_deps(package_specs: Sequence[str]) -> tuple[str, ...]:
    """Return package specs with TorchLens runtime deps included.

    Parameters
    ----------
    package_specs:
        Island package specs.

    Returns
    -------
    tuple[str, ...]
        Package specs including TorchLens runtime dependencies.
    """

    existing = {_split_pypi_package_spec(spec)[0] for spec in package_specs}
    runtime_specs = [
        package_spec
        for package_spec in TORCHLENS_RUNTIME_PYPI
        if _split_pypi_package_spec(package_spec)[0] not in existing
    ]
    return (*runtime_specs, *package_specs)


def _run_command(
    command: Sequence[str],
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: float | None = None,
) -> CommandResult:
    """Run a subprocess and capture output.

    Parameters
    ----------
    command:
        Command arguments.
    cwd:
        Optional working directory.
    env:
        Optional environment overrides.
    timeout:
        Optional timeout in seconds.

    Returns
    -------
    CommandResult
        Captured process result.
    """

    run_env = dict(os.environ)
    if env:
        run_env.update(env)
    completed = subprocess.run(
        list(command),
        cwd=cwd,
        env=run_env,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return CommandResult(completed.returncode, completed.stdout, completed.stderr)


def _manifest_path(env_key: str) -> Path:
    """Return the committed manifest path for an island.

    Parameters
    ----------
    env_key:
        Island key.

    Returns
    -------
    pathlib.Path
        Manifest path.
    """

    return LOCKS_DIR / f"{env_key}{LOCK_MANIFEST_SUFFIX}"


def _lock_path(env_key: str) -> Path:
    """Return the committed pixi lock path for an island.

    Parameters
    ----------
    env_key:
        Island key.

    Returns
    -------
    pathlib.Path
        Lock path.
    """

    return LOCKS_DIR / f"{env_key}{LOCK_FILE_SUFFIX}"


def compute_lock_hash(manifest_path: Path, lock_path: Path) -> str:
    """Compute the environment lock hash.

    Parameters
    ----------
    manifest_path:
        Pixi manifest path.
    lock_path:
        Pixi lock path.

    Returns
    -------
    str
        SHA-256 digest over committed lock inputs.
    """

    digest = sha256()
    digest.update(manifest_path.read_bytes())
    if lock_path.exists():
        digest.update(lock_path.read_bytes())
    return digest.hexdigest()


def env_hash(env_key: str, lock_hash: str) -> str:
    """Return the ledger environment hash string.

    Parameters
    ----------
    env_key:
        Island key.
    lock_hash:
        Lock hash.

    Returns
    -------
    str
        Ledger environment identity.
    """

    return f"pixi:{env_key}:{lock_hash}"


def lock(env_key: str, registry: EnvRegistry | None = None) -> LockResult:
    """Write and solve the committed pixi lock for an island.

    Parameters
    ----------
    env_key:
        Island key.
    registry:
        Optional loaded registry.

    Returns
    -------
    LockResult
        Lock result with hash.
    """

    active_registry = registry or load_registry()
    spec = active_registry.islands[env_key]
    LOCKS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = _manifest_path(env_key)
    lock_path = _lock_path(env_key)
    manifest_text = render_pixi_manifest(spec)
    manifest_path.write_text(manifest_text, encoding="utf-8")
    work_dir = active_registry.cache_root / "_lock_work" / env_key
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    work_manifest = work_dir / "pixi.toml"
    work_lock = work_dir / "pixi.lock"
    work_manifest.write_text(manifest_text, encoding="utf-8")
    command = [
        str(pixi_bin()),
        "lock",
        "--manifest-path",
        str(work_manifest),
        "--no-progress",
    ]
    result = _run_command(command, timeout=1800.0)
    if work_lock.exists():
        shutil.copy2(work_lock, lock_path)
    elif not lock_path.exists():
        lock_path.write_text("", encoding="utf-8")
    lock_hash = compute_lock_hash(manifest_path, lock_path)
    return LockResult(
        env_key=env_key,
        manifest_path=manifest_path,
        lock_path=lock_path,
        lock_hash=lock_hash,
        returncode=result.returncode,
        message=(result.stdout + result.stderr).strip(),
    )


def _project_dir(cache_root: Path, env_key: str, lock_hash: str) -> Path:
    """Return the managed project directory for a cache key.

    Parameters
    ----------
    cache_root:
        Managed cache root.
    env_key:
        Island key.
    lock_hash:
        Lock hash.

    Returns
    -------
    pathlib.Path
        Managed project directory.
    """

    return cache_root / f"{env_key}-{lock_hash[:16]}"


def _disk_free_gib(path: Path) -> float:
    """Return free GiB for a filesystem path.

    Parameters
    ----------
    path:
        Filesystem path.

    Returns
    -------
    float
        Free GiB.
    """

    check_path = path
    while not check_path.exists() and check_path != check_path.parent:
        check_path = check_path.parent
    return shutil.disk_usage(check_path).free / (1024**3)


def _cache_entry_mtime(path: Path) -> float:
    """Return an LRU timestamp for a managed cache entry.

    Parameters
    ----------
    path:
        Cache entry path.

    Returns
    -------
    float
        Last-used timestamp.
    """

    marker = path / BUILD_MARKER
    if marker.exists():
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
            return float(payload.get("last_used", marker.stat().st_mtime))
        except (OSError, ValueError, TypeError):
            return marker.stat().st_mtime
    return path.stat().st_mtime


def _managed_entries(cache_root: Path) -> list[Path]:
    """Return managed environment cache entries.

    Parameters
    ----------
    cache_root:
        Managed cache root.

    Returns
    -------
    list[pathlib.Path]
        Managed cache directories.
    """

    if not cache_root.exists():
        return []
    return [
        path for path in cache_root.iterdir() if path.is_dir() and not path.name.startswith("_")
    ]


def enforce_disk_lru(
    cache_root: Path,
    reserve_gib: float,
    projected_gib: float,
    protected: Path | None = None,
    free_gib: Callable[[Path], float] = _disk_free_gib,
    remove_tree: Callable[[Path], None] = shutil.rmtree,
) -> DiskCheckResult:
    """Evict managed cache entries until the projected disk floor is satisfied.

    Parameters
    ----------
    cache_root:
        Managed cache root.
    reserve_gib:
        Hard free-space reserve.
    projected_gib:
        Projected build footprint.
    protected:
        Optional cache entry that must not be evicted.
    free_gib:
        Free-space provider used by tests.
    remove_tree:
        Directory removal function used by tests.

    Returns
    -------
    DiskCheckResult
        Disk-floor enforcement result.
    """

    cache_root.mkdir(parents=True, exist_ok=True)
    target_free = reserve_gib + max(projected_gib, 0.0)
    current_free = free_gib(cache_root)
    evicted: list[Path] = []
    if current_free >= target_free:
        return DiskCheckResult(True, current_free, tuple(evicted))
    entries = sorted(_managed_entries(cache_root), key=_cache_entry_mtime)
    for entry in entries:
        if protected is not None and entry.resolve() == protected.resolve():
            continue
        remove_tree(entry)
        evicted.append(entry)
        current_free = free_gib(cache_root)
        if current_free >= target_free:
            return DiskCheckResult(True, current_free, tuple(evicted))
    current_free = free_gib(cache_root)
    if current_free >= reserve_gib:
        return DiskCheckResult(
            False,
            current_free,
            tuple(evicted),
            "projected_build_exceeds_floor",
        )
    return DiskCheckResult(False, current_free, tuple(evicted), "disk_floor")


def _write_build_marker(project_dir: Path, result: BuildResult) -> None:
    """Write the managed-cache build marker.

    Parameters
    ----------
    project_dir:
        Managed project directory.
    result:
        Build result.
    """

    payload = {
        "env_key": result.env_key,
        "status": result.status,
        "env_hash": result.env_hash,
        "lock_hash": result.lock_hash,
        "last_used": time.time(),
        "message": result.message,
    }
    (project_dir / BUILD_MARKER).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _cached_build(project_dir: Path, env_key: str, lock_hash: str) -> BuildResult | None:
    """Return a cached build result when the marker matches.

    Parameters
    ----------
    project_dir:
        Managed project directory.
    env_key:
        Island key.
    lock_hash:
        Expected lock hash.

    Returns
    -------
    BuildResult | None
        Cached result or ``None``.
    """

    marker = project_dir / BUILD_MARKER
    if not marker.exists():
        return None
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if payload.get("env_key") != env_key or payload.get("lock_hash") != lock_hash:
        return None
    result = BuildResult(
        env_key=env_key,
        status="cached",
        env_hash=str(payload.get("env_hash", env_hash(env_key, lock_hash))),
        lock_hash=lock_hash,
        project_dir=project_dir,
        message="cached",
    )
    _write_build_marker(project_dir, result)
    return result


def _copy_lock_inputs(project_dir: Path, env_key: str) -> None:
    """Copy committed lock inputs into a managed pixi project directory.

    Parameters
    ----------
    project_dir:
        Managed project directory.
    env_key:
        Island key.
    """

    project_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(_manifest_path(env_key), project_dir / "pixi.toml")
    shutil.copy2(_lock_path(env_key), project_dir / "pixi.lock")


def _pixi_run_command(project_dir: Path, command: Sequence[str]) -> list[str]:
    """Build a pixi-run command for a managed project.

    Parameters
    ----------
    project_dir:
        Managed project directory.
    command:
        Command to execute inside the pixi env.

    Returns
    -------
    list[str]
        Full pixi command.
    """

    return [
        str(pixi_bin()),
        "run",
        "--manifest-path",
        str(project_dir / "pixi.toml"),
        "--frozen",
        "--",
        *command,
    ]


def _tail_output(result: CommandResult) -> str:
    """Return compact command output for status messages.

    Parameters
    ----------
    result:
        Command result.

    Returns
    -------
    str
        Last output lines.
    """

    lines = (result.stdout + result.stderr).splitlines()
    return " | ".join(lines[-12:])


def build(env_key: str, registry: EnvRegistry | None = None) -> BuildResult:
    """Build or reuse a managed pixi island environment.

    Parameters
    ----------
    env_key:
        Island key.
    registry:
        Optional loaded registry.

    Returns
    -------
    BuildResult
        Build status.
    """

    active_registry = registry or load_registry()
    spec = active_registry.islands[env_key]
    lock_result = lock(env_key, active_registry)
    lock_hash = lock_result.lock_hash
    project_dir = _project_dir(active_registry.cache_root, env_key, lock_hash)
    disk = enforce_disk_lru(
        active_registry.cache_root,
        active_registry.disk_free_reserve_gib,
        spec.disk_estimate_gib,
        protected=project_dir,
    )
    identity = env_hash(env_key, lock_hash)
    if not disk.allowed:
        return BuildResult(
            env_key, "env_unavailable", identity, lock_hash, project_dir, disk.reason
        )
    cached = _cached_build(project_dir, env_key, lock_hash)
    if cached is not None:
        return cached
    if lock_result.returncode != 0:
        result = BuildResult(
            env_key,
            "install_failed",
            identity,
            lock_hash,
            project_dir,
            lock_result.message,
        )
        return result
    _copy_lock_inputs(project_dir, env_key)
    install_result = _run_command(
        [
            str(pixi_bin()),
            "install",
            "--manifest-path",
            str(project_dir / "pixi.toml"),
            "--frozen",
            "--no-progress",
        ],
        timeout=3600.0,
    )
    if install_result.returncode != 0:
        return BuildResult(
            env_key,
            "install_failed",
            identity,
            lock_hash,
            project_dir,
            _tail_output(install_result),
        )
    editable_result = _run_command(
        _pixi_run_command(
            project_dir, ["python", "-m", "pip", "install", "--no-deps", "-e", str(REPO_ROOT)]
        ),
        timeout=900.0,
    )
    if editable_result.returncode != 0:
        return BuildResult(
            env_key,
            "install_failed",
            identity,
            lock_hash,
            project_dir,
            _tail_output(editable_result),
        )
    import_result = _run_command(
        _pixi_run_command(project_dir, ["python", "-c", "import torchlens, torch"]),
        timeout=120.0,
    )
    if import_result.returncode != 0:
        return BuildResult(
            env_key,
            "install_failed",
            identity,
            lock_hash,
            project_dir,
            _tail_output(import_result),
        )
    result = BuildResult(env_key, "built", identity, lock_hash, project_dir, "built")
    _write_build_marker(project_dir, result)
    return result


def _rows_by_stable_id(stable_ids: Sequence[str]) -> list[CatalogRow]:
    """Load catalog rows by stable ID from canonical JSONL sources.

    Parameters
    ----------
    stable_ids:
        Stable IDs to load.

    Returns
    -------
    list[CatalogRow]
        Rows in requested order.
    """

    wanted = list(stable_ids)
    by_id = {row.stable_id: row for row in build_canonical_rows()}
    missing = [stable_id for stable_id in wanted if stable_id not in by_id]
    if missing:
        raise ValueError(f"unknown stable_id(s): {', '.join(missing)}")
    return [by_id[stable_id] for stable_id in wanted]


def smoke_worker(env_key: str, stable_ids: Sequence[str]) -> int:
    """Run smoke validation inside an already-active pixi environment.

    Parameters
    ----------
    env_key:
        Island key.
    stable_ids:
        Stable IDs to smoke.

    Returns
    -------
    int
        Process exit code.
    """

    import torch
    import torchlens as tl

    torch.set_num_threads(1)
    for row in _rows_by_stable_id(stable_ids):
        model = instantiate_model(row)
        if hasattr(model, "eval"):
            model.eval()
        input_value = build_input_for_row(row)
        with torch.no_grad():
            trace = tl.trace(model, input_value, inference_only=True)
        if int(getattr(trace, "num_ops", 0) or 0) <= 0:
            raise RuntimeError(f"{row.stable_id} produced an empty trace in {env_key}")
        result = tl.validate_forward_pass(model, input_value, validate_metadata=True)
        if not bool(result):
            raise RuntimeError(f"{row.stable_id} smoke validation failed: {result!r}")
    print(json.dumps({"event": "smoke_passed", "env_key": env_key, "stable_ids": list(stable_ids)}))
    return 0


def smoke(env_key: str, registry: EnvRegistry | None = None) -> CommandResult:
    """Run import and trace smoke tests in one island.

    Parameters
    ----------
    env_key:
        Island key.
    registry:
        Optional loaded registry.

    Returns
    -------
    CommandResult
        Smoke command result.
    """

    active_registry = registry or load_registry()
    build_result = build(env_key, active_registry)
    if build_result.status not in {"built", "cached"}:
        return CommandResult(1, "", build_result.message)
    spec = active_registry.islands[env_key]
    command = _pixi_run_command(
        build_result.project_dir,
        [
            "python",
            "-m",
            "menagerie.envs",
            "_smoke_worker",
            env_key,
            "--stable-ids",
            *spec.smoke_stable_ids,
        ],
    )
    return _run_command(
        command, env={"TORCHLENS_MENAGERIE_ENV_HASH": build_result.env_hash}, timeout=900.0
    )


def ledger_status_for_build_status(status: BuildStatus) -> Status:
    """Map env-manager build statuses to ledger terminal statuses.

    Parameters
    ----------
    status:
        Build status.

    Returns
    -------
    Status
        Ledger status.
    """

    if status == "env_unavailable":
        return "env_unavailable"
    if status == "install_failed":
        return "install_failed"
    return "deferred"


def _package_version(package: str, fallback: str) -> str:
    """Return an installed package version with fallback.

    Parameters
    ----------
    package:
        Package name.
    fallback:
        Fallback version.

    Returns
    -------
    str
        Version string.
    """

    try:
        return version(package)
    except PackageNotFoundError:
        return fallback


def _torchlens_version() -> str:
    """Return the local TorchLens version.

    Returns
    -------
    str
        TorchLens version.
    """

    import torchlens as tl

    return _package_version("torchlens", str(getattr(tl, "__version__", "unknown")))


def _torch_version(fallback: str) -> str:
    """Return the local torch version with fallback.

    Parameters
    ----------
    fallback:
        Fallback version string.

    Returns
    -------
    str
        Torch version.
    """

    try:
        import torch
    except ImportError:
        return fallback
    return str(torch.__version__)


def append_env_status_rows(
    stable_ids: Sequence[str],
    status: Status,
    env_hash_value: str,
    reason: str,
    device: str = "cpu",
) -> int:
    """Append terminal env status rows to the verification ledger.

    Parameters
    ----------
    stable_ids:
        Stable IDs to record.
    status:
        Ledger terminal status.
    env_hash_value:
        Environment hash to store.
    reason:
        Failure or unavailable reason.
    device:
        Requested device.

    Returns
    -------
    int
        Number of rows appended.
    """

    rows = _rows_by_stable_id(stable_ids)
    started_at = utc_now()
    finished_at = utc_now()
    with connect_ledger() as conn:
        for row in rows:
            append_verification_run(
                conn,
                VerificationRun(
                    stable_id=row.stable_id,
                    recipe_revision_sha256=row.recipe_revision_sha256,
                    name=row.name,
                    zoo=row.zoo,
                    variant=row.variant,
                    scope="forward",
                    status=status,
                    forward_pass=None,
                    backward_pass=None,
                    backward_na_reason=None,
                    metadata_ok=None,
                    n_ops=None,
                    graph_shape_hash=None,
                    svg_sha256=None,
                    torchlens_version=_torchlens_version(),
                    torch_version=_torch_version(env_hash_value),
                    python_version=python_version(),
                    device_requested=device,
                    device_actual=None,
                    env_hash=env_hash_value,
                    runner_host=runner_host(),
                    started_at=started_at,
                    finished_at=finished_at,
                    duration_sec=0.0,
                    error_class=status,
                    error_message=reason,
                ),
            )
    return len(rows)


def run_validate(
    env_key: str,
    stable_ids: Sequence[str],
    registry: EnvRegistry | None = None,
    extra_args: Sequence[str] = (),
) -> CommandResult:
    """Run menagerie validation inside a managed island.

    Parameters
    ----------
    env_key:
        Island key.
    stable_ids:
        Stable IDs to validate.
    registry:
        Optional loaded registry.
    extra_args:
        Extra CLI arguments forwarded to the validator.

    Returns
    -------
    CommandResult
        Validator command result.
    """

    active_registry = registry or load_registry()
    build_result = build(env_key, active_registry)
    if build_result.status not in {"built", "cached"}:
        ledger_status = ledger_status_for_build_status(build_result.status)
        append_env_status_rows(
            stable_ids,
            ledger_status,
            build_result.env_hash,
            build_result.message,
            active_registry.islands[env_key].expected_device,
        )
        return CommandResult(1, "", build_result.message)
    spec = active_registry.islands[env_key]
    command = _pixi_run_command(
        build_result.project_dir,
        [
            "python",
            "-m",
            "menagerie.validate_menagerie",
            "--no-install-deps",
            "--stable-ids",
            *stable_ids,
            "--db",
            str(build_result.project_dir / "catalog.db"),
            "--device",
            spec.expected_device,
            *extra_args,
        ],
    )
    return _run_command(
        command,
        env={"TORCHLENS_MENAGERIE_ENV_HASH": build_result.env_hash},
        timeout=None,
    )


def _assignment_summary(rows: Sequence[CatalogRow], registry: EnvRegistry) -> dict[str, Any]:
    """Build a JSON-compatible assignment summary.

    Parameters
    ----------
    rows:
        Catalog rows.
    registry:
        Loaded registry.

    Returns
    -------
    dict[str, Any]
        Assignment summary.
    """

    assignments = assign(rows, registry)
    counts: dict[str, int] = {key: 0 for key in registry.islands}
    for env_key in assignments.values():
        counts[env_key] = counts.get(env_key, 0) + 1
    dep_count = sum(1 for row in rows if _row_needs_dependency_env(row, registry))
    return {"rows": len(rows), "dependency_rows": dep_count, "counts": counts}


def build_parser() -> argparse.ArgumentParser:
    """Build the env-manager CLI parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("assign")
    lock_parser = subparsers.add_parser("lock")
    lock_parser.add_argument("env_key")
    build_parser_obj = subparsers.add_parser("build")
    build_parser_obj.add_argument("env_key")
    smoke_parser = subparsers.add_parser("smoke")
    smoke_parser.add_argument("env_key")
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("env_key")
    validate_parser.add_argument("stable_ids", nargs="+")
    worker_parser = subparsers.add_parser("_smoke_worker")
    worker_parser.add_argument("env_key")
    worker_parser.add_argument("--stable-ids", nargs="+", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the env-manager CLI.

    Parameters
    ----------
    argv:
        Optional argument vector.

    Returns
    -------
    int
        Process exit code.
    """

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "_smoke_worker":
        return smoke_worker(args.env_key, args.stable_ids)
    registry = load_registry(args.registry)
    if args.command == "assign":
        print(json.dumps(_assignment_summary(build_canonical_rows(), registry), sort_keys=True))
        return 0
    if args.command == "lock":
        result = lock(args.env_key, registry)
        print(
            json.dumps(
                result.__dict__
                | {"manifest_path": str(result.manifest_path), "lock_path": str(result.lock_path)},
                sort_keys=True,
            )
        )
        return 0 if result.returncode == 0 else 1
    if args.command == "build":
        result = build(args.env_key, registry)
        print(
            json.dumps(result.__dict__ | {"project_dir": str(result.project_dir)}, sort_keys=True)
        )
        return 0 if result.status in {"built", "cached"} else 1
    if args.command == "smoke":
        result = smoke(args.env_key, registry)
        print(result.stdout, end="")
        print(result.stderr, end="", file=sys.stderr)
        return result.returncode
    if args.command == "validate":
        result = run_validate(args.env_key, args.stable_ids, registry)
        print(result.stdout, end="")
        print(result.stderr, end="", file=sys.stderr)
        return result.returncode
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
