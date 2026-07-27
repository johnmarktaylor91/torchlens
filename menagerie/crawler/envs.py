"""Typed loading for crawler environment intents and target-solved artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from menagerie.crawler.constants import EnvironmentPhase
from menagerie.crawler.identity import hash_bytes, stable_hash

DEFAULT_ENVS_ROOT = Path(__file__).with_name("envs")


class EnvironmentSpecError(ValueError):
    """Raised when an environment registry or intent artifact is malformed."""


@dataclass(frozen=True)
class ExportCheck:
    """One expected top-level attribute exposed by an imported module."""

    module: str
    attribute: str


@dataclass(frozen=True)
class SourceBuildProbe:
    """A bounded compiled/source-build probe declaration."""

    name: str
    packages: tuple[str, ...]
    command: tuple[str, ...]
    max_attempts: int


@dataclass(frozen=True)
class IntentProbes:
    """Import, export, and optional source-build checks for one intent."""

    imports: tuple[str, ...]
    export_checks: tuple[ExportCheck, ...]
    source_build: tuple[SourceBuildProbe, ...]


@dataclass(frozen=True)
class LockArtifacts:
    """Optional exact target lock and resolved export for one intent."""

    target: str
    lock_path: Path
    export_path: Path
    export_hash_path: Path
    lock_bytes: Optional[bytes]
    export_bytes: Optional[bytes]
    declared_export_hash: Optional[str]

    @property
    def status(self) -> str:
        """Return the target artifact readiness state.

        Returns
        -------
        str
            ``unlocked``, ``incomplete``, ``hash-mismatch``, or ``locked``.
        """

        present = (self.lock_bytes is not None, self.export_bytes is not None)
        if not any(present) and self.declared_export_hash is None:
            return "unlocked"
        if not all(present) or self.declared_export_hash is None:
            return "incomplete"
        if hash_bytes(self.export_bytes or b"") != self.declared_export_hash:
            return "hash-mismatch"
        return "locked"


@dataclass(frozen=True)
class EnvironmentIntent:
    """Complete in-memory view of one environment intent and target state."""

    name: str
    phase: EnvironmentPhase
    framework: str
    description: str
    split_guidance: str
    channels: tuple[str, ...]
    dependencies: tuple[str, ...]
    probes: IntentProbes
    lock: LockArtifacts
    generation: Optional[str]

    @property
    def lock_status(self) -> str:
        """Return the exact target lock status.

        Returns
        -------
        str
            Delegated lock readiness state.
        """

        return self.lock.status


@dataclass(frozen=True)
class EnvironmentRegistry:
    """Ordered small-set environment registry without a hard intent cap."""

    intents: Mapping[str, EnvironmentIntent]
    phase_order: tuple[EnvironmentPhase, ...]
    small_set_target: bool
    hard_cap: Optional[int]
    global_split_guidance: str


def load_environment_registry(
    root: Path = DEFAULT_ENVS_ROOT, *, target: str = "osx-arm64"
) -> EnvironmentRegistry:
    """Load registry, direct dependencies, probes, and optional target locks.

    Parameters
    ----------
    root:
        Directory containing ``registry.yml`` and intent subdirectories.
    target:
        Exact target lock basename.

    Returns
    -------
    EnvironmentRegistry
        Validated typed environment view.
    """

    registry = _load_json_yaml(root / "registry.yml")
    probes = _load_json_yaml(root / "probes.yml")
    raw_intents = _mapping(registry.get("intents"), "registry intents")
    raw_probes = _mapping(probes.get("intents"), "probe intents")
    intents: dict[str, EnvironmentIntent] = {}
    for name, raw_value in raw_intents.items():
        raw = _mapping(raw_value, f"intent {name}")
        env = _load_json_yaml(root / name / "environment.yml")
        probe = _parse_probes(name, _mapping(raw_probes.get(name), f"probes for {name}"))
        lock = _load_lock_artifacts(root / name / "locks", target)
        generation = compute_environment_generation(name, lock, probe)
        intents[name] = EnvironmentIntent(
            name=name,
            phase=EnvironmentPhase(_string(raw.get("phase"), f"phase for {name}")),
            framework=_string(raw.get("framework"), f"framework for {name}"),
            description=_string(raw.get("description"), f"description for {name}"),
            split_guidance=_string(raw.get("split_guidance"), f"split guidance for {name}"),
            channels=_strings(env.get("channels"), f"channels for {name}"),
            dependencies=_strings(env.get("dependencies"), f"dependencies for {name}"),
            probes=probe,
            lock=lock,
            generation=generation,
        )
    if set(raw_probes) != set(raw_intents):
        raise EnvironmentSpecError("registry and probes must name exactly the same intents")
    phase_order = tuple(
        EnvironmentPhase(value) for value in _strings(registry.get("phase_order"), "phase order")
    )
    hard_cap = registry.get("hard_cap")
    if hard_cap is not None and not isinstance(hard_cap, int):
        raise EnvironmentSpecError("hard_cap must be null or an integer")
    return EnvironmentRegistry(
        intents=intents,
        phase_order=phase_order,
        small_set_target=bool(registry.get("small_set_target")),
        hard_cap=hard_cap,
        global_split_guidance=_string(registry.get("split_guidance"), "split guidance"),
    )


def compute_environment_generation(
    intent: str, lock: LockArtifacts, probes: IntentProbes
) -> Optional[str]:
    """Hash exact lock, resolved export, declared hash, and probe contract.

    Parameters
    ----------
    intent, lock, probes:
        Environment intent name and its target-solved generation inputs.

    Returns
    -------
    str or None
        Stable generation hash, or ``None`` until artifacts are complete and valid.
    """

    if lock.status != "locked":
        return None
    return stable_hash(
        {
            "intent": intent,
            "target": lock.target,
            "lock_sha256": hash_bytes(lock.lock_bytes or b""),
            "resolved_export_sha256": lock.declared_export_hash,
            "probes": {
                "imports": probes.imports,
                "export_checks": [vars(check) for check in probes.export_checks],
                "source_build": [vars(probe) for probe in probes.source_build],
            },
        }
    )


def _load_lock_artifacts(lock_dir: Path, target: str) -> LockArtifacts:
    """Load optional setup-time target artifacts from one lock directory."""

    lock_path = lock_dir / f"{target}.lock"
    export_path = lock_dir / f"{target}.resolved.json"
    export_hash_path = lock_dir / f"{target}.resolved.sha256"
    return LockArtifacts(
        target=target,
        lock_path=lock_path,
        export_path=export_path,
        export_hash_path=export_hash_path,
        lock_bytes=lock_path.read_bytes() if lock_path.is_file() else None,
        export_bytes=export_path.read_bytes() if export_path.is_file() else None,
        declared_export_hash=(
            export_hash_path.read_text(encoding="utf-8").strip()
            if export_hash_path.is_file()
            else None
        ),
    )


def _parse_probes(name: str, raw: Mapping[str, Any]) -> IntentProbes:
    """Parse one intent's probe declaration."""

    checks: list[ExportCheck] = []
    for value in _sequence(raw.get("export_checks"), f"export checks for {name}"):
        item = _mapping(value, f"export check for {name}")
        checks.append(
            ExportCheck(
                module=_string(item.get("module"), "export module"),
                attribute=_string(item.get("attribute"), "export attribute"),
            )
        )
    builds: list[SourceBuildProbe] = []
    for value in _sequence(raw.get("source_build", []), f"source builds for {name}"):
        item = _mapping(value, f"source build for {name}")
        max_attempts = item.get("max_attempts")
        if not isinstance(max_attempts, int) or max_attempts < 1:
            raise EnvironmentSpecError("source-build max_attempts must be positive")
        builds.append(
            SourceBuildProbe(
                name=_string(item.get("name"), "source-build name"),
                packages=_strings(item.get("packages"), "source-build packages"),
                command=_strings(item.get("command"), "source-build command"),
                max_attempts=max_attempts,
            )
        )
    imports = _strings(raw.get("imports"), f"imports for {name}")
    if len(imports) < 3:
        raise EnvironmentSpecError(f"intent {name!r} must declare at least three import canaries")
    return IntentProbes(imports=imports, export_checks=tuple(checks), source_build=tuple(builds))


def _load_json_yaml(path: Path) -> Mapping[str, Any]:
    """Load a dependency-free JSON-form YAML document."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EnvironmentSpecError(f"cannot load {path}: {exc}") from exc
    return _mapping(value, str(path))


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    """Require a string-keyed mapping."""

    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise EnvironmentSpecError(f"{label} must be a string-keyed mapping")
    return value


def _sequence(value: Any, label: str) -> list[Any]:
    """Require a list value."""

    if not isinstance(value, list):
        raise EnvironmentSpecError(f"{label} must be a list")
    return value


def _strings(value: Any, label: str) -> tuple[str, ...]:
    """Require a non-empty list of non-empty strings."""

    sequence = _sequence(value, label)
    if not sequence or not all(isinstance(item, str) and item.strip() for item in sequence):
        raise EnvironmentSpecError(f"{label} must contain non-empty strings")
    return tuple(sequence)


def _string(value: Any, label: str) -> str:
    """Require one non-empty string."""

    if not isinstance(value, str) or not value.strip():
        raise EnvironmentSpecError(f"{label} must be a non-empty string")
    return value
