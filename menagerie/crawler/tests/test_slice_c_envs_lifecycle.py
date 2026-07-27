"""Slice C environment registry and sequential lifecycle tests."""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import pytest

from menagerie.crawler.env_lifecycle import (
    ArtifactReceipt,
    DiskRecoveryError,
    EnvironmentCleanupError,
    EnvironmentExactnessError,
    EnvironmentProbeError,
    PerIntentGenerationEffort,
    ProbeResult,
    SequentialEnvironmentLifecycle,
    SolveResult,
    expected_probe_names,
    parse_exact_lock,
    parse_resolved_export,
)
from menagerie.crawler.envs import DEFAULT_ENVS_ROOT, IntentProbes, load_environment_registry
from menagerie.crawler.effort import CapFailureRecord, EffortCapExceeded, EffortTracker, StageCap
from menagerie.crawler.identity import hash_bytes

_ARTIFACT_BYTES = b"exact conda artifact"
_ARTIFACT_SHA256 = hash_bytes(_ARTIFACT_BYTES)
_ARTIFACT_URL = "https://conda.example.test/python.conda"
_LOCK_BYTES = f"{_ARTIFACT_URL}#{_ARTIFACT_SHA256.removeprefix('sha256:')}\n".encode()
_EXPORT_BYTES = (
    b'{"packages":[{"build":"h1_0","name":"python","sha256":"'
    + _ARTIFACT_SHA256.encode()
    + b'","url":"https://conda.example.test/python.conda","version":"3.11"}]}\n'
)


class FakeEnvironmentBackend:
    """Record lifecycle calls without invoking a real package solver."""

    def __init__(self, events: list[str], *, elapsed_seconds: float = 1.0) -> None:
        """Configure event storage and measured fake solve duration."""

        self.events = events
        self.elapsed_seconds = elapsed_seconds

    def solve(self, environment_file: Path, target: str) -> SolveResult:
        """Return deterministic exact artifact bytes."""

        assert environment_file.is_file()
        self.events.append(f"solve:{target}")
        return SolveResult(
            _LOCK_BYTES,
            _EXPORT_BYTES,
            self.elapsed_seconds,
            40,
            (ArtifactReceipt(_ARTIFACT_URL, _ARTIFACT_SHA256),),
        )

    def create(self, lock_file: Path, prefix: Path) -> bytes:
        """Record creation after proving the lock was written."""

        del prefix
        assert lock_file.read_bytes() == _LOCK_BYTES
        self.events.append("create")
        return _EXPORT_BYTES

    def probe(self, prefix: Path, probes: IntentProbes) -> Sequence[ProbeResult]:
        """Return successful results for all declared import canaries."""

        self.events.append("probe")
        return tuple(ProbeResult(name, True, "ok") for name in expected_probe_names(probes))

    def remove(self, prefix: Path) -> None:
        """Record deterministic teardown."""

        self.events.append("remove")


def _copy_env_specs(tmp_path: Path) -> Path:
    """Copy intent inputs so lifecycle outputs never modify the source tree."""

    target = tmp_path / "envs"
    shutil.copytree(DEFAULT_ENVS_ROOT, target)
    return target


def _release_lock_provenance_errors(env_root: Path) -> tuple[str, ...]:
    """Return violations in committed release-lock provenance.

    Parameters
    ----------
    env_root:
        Environment registry root containing optional committed release locks.

    Returns
    -------
    tuple[str, ...]
        Located anti-fabrication diagnostics, empty only for fully attested lock families.
    """

    generated_outputs = {
        path
        for path in env_root.rglob("*")
        if path.is_file()
        and (
            path.suffix == ".lock"
            or path.name.endswith(".resolved.json")
            or path.name.endswith(".resolved.sha256")
        )
    }
    accounted_outputs: set[Path] = set()
    errors: list[str] = []
    for lock_path in sorted(env_root.rglob("*.lock")):
        family = {
            "resolved": lock_path.with_suffix(".resolved.json"),
            "resolved_hash": lock_path.with_suffix(".resolved.sha256"),
            "provenance": lock_path.with_suffix(".provenance.json"),
            "probes": lock_path.with_suffix(".probes.json"),
        }
        accounted_outputs.add(lock_path)
        accounted_outputs.update(
            path for name, path in family.items() if name in {"resolved", "resolved_hash"}
        )
        missing = [
            name
            for name, path in family.items()
            if not path.is_file()
        ]
        if missing:
            errors.append(f"{lock_path}:missing-{','.join(sorted(missing))}")
            continue
        try:
            lock_bytes = lock_path.read_bytes()
            resolved_bytes = family["resolved"].read_bytes()
            parse_exact_lock(lock_bytes)
            if parse_resolved_export(resolved_bytes) != resolved_bytes:
                errors.append(f"{lock_path}:noncanonical-resolved-export")
            declared_resolved_hash = family["resolved_hash"].read_text(encoding="utf-8").strip()
            provenance = json.loads(family["provenance"].read_bytes())
        except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{lock_path}:invalid-lock-family:{type(exc).__name__}")
            continue
        if not isinstance(provenance, dict):
            errors.append(f"{lock_path}:provenance-not-object")
            continue
        target = provenance.get("target")
        spec_path = provenance.get("spec_path")
        probe_contract_path = provenance.get("probe_contract_path")
        if not isinstance(target, str) or not lock_path.name.endswith(f"-{target}.lock"):
            errors.append(f"{lock_path}:target-not-bound-to-lock")
        if provenance.get("schema_version") != "menagerie.crawler.release-lock-provenance.v1":
            errors.append(f"{lock_path}:invalid-provenance-schema")
        if provenance.get("lock_path") != (
            f"menagerie/crawler/envs/{lock_path.relative_to(env_root).as_posix()}"
        ):
            errors.append(f"{lock_path}:lock-path-not-bound")
        if provenance.get("lock_sha256") != hash_bytes(lock_bytes):
            errors.append(f"{lock_path}:lock-hash-mismatch")
        if declared_resolved_hash != hash_bytes(resolved_bytes):
            errors.append(f"{lock_path}:resolved-hash-file-mismatch")
        if provenance.get("resolved_export_sha256") != declared_resolved_hash:
            errors.append(f"{lock_path}:resolved-provenance-mismatch")
        if provenance.get("resolved_export_path") != (
            f"menagerie/crawler/envs/{family['resolved'].relative_to(env_root).as_posix()}"
        ):
            errors.append(f"{lock_path}:resolved-path-not-bound")
        if family["probes"].is_file():
            if provenance.get("probe_receipt_path") != (
                f"menagerie/crawler/envs/{family['probes'].relative_to(env_root).as_posix()}"
            ):
                errors.append(f"{lock_path}:probe-receipt-path-not-bound")
            if provenance.get("probe_receipt_sha256") != hash_bytes(family["probes"].read_bytes()):
                errors.append(f"{lock_path}:probe-receipt-hash-mismatch")
        elif provenance.get("probe_observation") != {
            "committed_on_linux": False,
            "producer": "hosted macOS release CI attestation",
        }:
            errors.append(f"{lock_path}:hosted-probe-observation-not-bound")
        for label, declared_path, digest_key in (
            ("spec", spec_path, "spec_sha256"),
            ("probe-contract", probe_contract_path, "probe_contract_sha256"),
        ):
            expected_parent = Path("menagerie/crawler/envs/specs")
            if not isinstance(declared_path, str) or Path(declared_path).parent != expected_parent:
                errors.append(f"{lock_path}:{label}-path-outside-registry")
                continue
            local_path = env_root / "specs" / Path(declared_path).name
            if not local_path.is_file() or provenance.get(digest_key) != hash_bytes(
                local_path.read_bytes()
            ):
                errors.append(f"{lock_path}:{label}-hash-mismatch")
        solver = provenance.get("solver")
        solver_command = solver.get("command") if isinstance(solver, dict) else None
        if (
            not isinstance(solver_command, list)
            or spec_path not in solver_command
            or (
                solver_command[:3] != ["conda", "env", "create"]
                # Validate the conda-lock invocation signature ("<python> -m conda_lock")
                # without pinning the solver's absolute interpreter path (host-specific).
                and solver_command[1:3] != ["-m", "conda_lock"]
            )
        ):
            errors.append(f"{lock_path}:solver-command-not-bound")
        clean_create = provenance.get("clean_create")
        artifact_validation = provenance.get("artifact_hash_validation")
        if not isinstance(clean_create, dict):
            errors.append(f"{lock_path}:clean-create-unattested")
        elif target == "linux-64" and clean_create.get("validated") is not True:
            errors.append(f"{lock_path}:clean-create-unattested")
        elif target == "osx-arm64" and clean_create.get("validation_host") != (
            "hosted macOS release CI"
        ):
            errors.append(f"{lock_path}:clean-create-unattested")
        if (
            not isinstance(artifact_validation, dict)
            or artifact_validation.get("algorithm") != "sha256"
            or artifact_validation.get("verified") is not True
        ):
            errors.append(f"{lock_path}:artifact-hashes-unattested")
        target_host = {"linux-64": ("Linux", "x86_64", "__linux")}.get(target)
        host = provenance.get("host")
        virtual_packages = provenance.get("virtual_packages")
        if target == "linux-64" and (
            target_host is None
            or not isinstance(host, dict)
            or (host.get("system"), host.get("machine")) != target_host[:2]
            or not isinstance(virtual_packages, list)
            or not any(
                isinstance(row, list) and row and row[0] == target_host[2]
                for row in virtual_packages
            )
        ):
            errors.append(f"{lock_path}:solve-host-not-on-target")
        if target == "osx-arm64" and (
            not isinstance(host, dict)
            or host.get("system") != "Linux"
            or "virtual_package_spec_path" not in provenance
            or not isinstance(virtual_packages, list)
            or not any(
                isinstance(row, list) and row and row[0] == "__osx" for row in virtual_packages
            )
            or provenance.get("virtual_package_spec_sha256")
            != hash_bytes(
                (env_root / "specs" / "round21-release.virtual-packages.yml").read_bytes()
            )
        ):
            errors.append(f"{lock_path}:cross-solve-not-bound")
    for path in sorted(generated_outputs - accounted_outputs):
        errors.append(f"{path}:orphan-hand-authored-output")
    return tuple(errors)


def test_registry_loads_all_intents_without_committed_locks() -> None:
    """All eleven intents have dependencies, probes, and honest unlocked state."""

    registry = load_environment_registry()
    assert len(registry.intents) == 11
    assert registry.small_set_target
    assert registry.hard_cap is None
    assert [phase.value for phase in registry.phase_order] == ["pytorch", "native-tail"]
    for intent in registry.intents.values():
        assert len(intent.probes.imports) >= 3
        assert intent.dependencies
        assert intent.lock_status == "unlocked"
        assert intent.generation is None
        assert (DEFAULT_ENVS_ROOT / intent.name / "environment.yml").is_file()


def test_generation_is_stable_and_changes_with_exact_lock(tmp_path: Path) -> None:
    """Identical target artifacts hash stably and a changed lock is a new generation."""

    root = _copy_env_specs(tmp_path)
    lock_dir = root / "core" / "locks"
    lock_dir.mkdir()
    export = b'{"python":"3.11","torch":"2.4"}\n'
    (lock_dir / "osx-arm64.lock").write_bytes(b"url hash-one\n")
    (lock_dir / "osx-arm64.resolved.json").write_bytes(export)
    (lock_dir / "osx-arm64.resolved.sha256").write_text(f"{hash_bytes(export)}\n", encoding="utf-8")
    first = load_environment_registry(root).intents["core"].generation
    second = load_environment_registry(root).intents["core"].generation
    assert first == second
    (lock_dir / "osx-arm64.lock").write_bytes(b"url hash-two\n")
    assert load_environment_registry(root).intents["core"].generation != first


def test_repository_contains_no_hand_authored_lock_or_hash_outputs(tmp_path: Path) -> None:
    """Only solve-provenance-bound exact lock families may be committed."""

    assert _release_lock_provenance_errors(DEFAULT_ENVS_ROOT) == ()
    assert (DEFAULT_ENVS_ROOT / "locks" / "README.md").is_file()
    copied_root = _copy_env_specs(tmp_path)
    fabricated_lock = copied_root / "locks" / "hand-authored-linux-64.lock"
    fabricated_lock.write_text("@EXPLICIT\nhttps://example.test/fake#sha256=00\n", encoding="utf-8")
    fabricated_lock.with_suffix(".resolved.sha256").write_text(
        "sha256:0000000000000000000000000000000000000000000000000000000000000000\n",
        encoding="utf-8",
    )
    assert any(
        "hand-authored-linux-64.lock:missing-" in error
        for error in _release_lock_provenance_errors(copied_root)
    )


def test_lifecycle_orders_solve_create_probe_use_and_teardown(tmp_path: Path) -> None:
    """One exact environment runs sequentially and checks disk after removal."""

    root = _copy_env_specs(tmp_path)
    intent = load_environment_registry(root).intents["core"]
    events: list[str] = []
    disk_values = iter((1000, 700, 1000))
    lifecycle = SequentialEnvironmentLifecycle(
        FakeEnvironmentBackend(events),
        EffortTracker({"environment": StageCap(attempts=1, seconds=5, bytes=100)}),
        env_root=tmp_path / "active",
        disk_free=lambda _path: next(disk_values),
        minimum_free_bytes=0,
    )
    observed_probes: list[tuple[ProbeResult, ...]] = []

    def use(_prefix: Path, probe_results: tuple[ProbeResult, ...]) -> None:
        """Record the exact successful canary observations passed to the driver."""

        observed_probes.append(probe_results)
        events.append("use")

    result = lifecycle.run(intent, use=use)
    assert events == ["solve:osx-arm64", "create", "probe", "use", "remove"]
    assert result.disk_recovery_checked
    assert result.disk_after_teardown == result.disk_before
    assert observed_probes == [result.probe_results]
    assert intent.lock.lock_path.is_file()
    assert intent.lock.export_hash_path.read_text(encoding="utf-8").startswith("sha256:")


def test_environment_effort_cap_is_independent_per_intent_generation(tmp_path: Path) -> None:
    """Exhausting one intent cannot consume another intent's attempt budget."""

    root = _copy_env_specs(tmp_path)
    registry = load_environment_registry(root)
    core = registry.intents["core"]
    graph = registry.intents["graph"]
    effort = PerIntentGenerationEffort(StageCap(attempts=1, seconds=5, bytes=100))
    lifecycle = SequentialEnvironmentLifecycle(
        FakeEnvironmentBackend([]),
        effort,
        env_root=tmp_path / "active",
        disk_free=lambda _path: 1000,
        minimum_free_bytes=0,
    )

    lifecycle.run(core, use=lambda _prefix, _probes: None)
    lifecycle.run(graph, use=lambda _prefix, _probes: None)

    assert effort(core).usage("environment").attempts == 1
    assert effort(graph).usage("environment").attempts == 1
    with pytest.raises(EffortCapExceeded):
        lifecycle.run(core, use=lambda _prefix, _probes: None)


def test_cap_exceeding_solve_is_typed_recorded_and_not_materialized(tmp_path: Path) -> None:
    """Measured solve work over cap stops before any lock/create operation."""

    root = _copy_env_specs(tmp_path)
    intent = load_environment_registry(root).intents["core"]
    events: list[str] = []
    recorded: list[CapFailureRecord] = []
    tracker = EffortTracker(
        {"environment": StageCap(attempts=1, seconds=1)}, recorder=recorded.append
    )
    lifecycle = SequentialEnvironmentLifecycle(
        FakeEnvironmentBackend(events, elapsed_seconds=2),
        tracker,
        env_root=tmp_path / "active",
        disk_free=lambda _path: 1000,
        minimum_free_bytes=0,
    )
    with pytest.raises(EffortCapExceeded):
        lifecycle.run(intent, use=lambda _prefix, _probes: events.append("use"))
    assert recorded[0].actual_stage == "environment"
    assert recorded[0].metric == "seconds"
    assert events == ["solve:osx-arm64"]
    assert not intent.lock.lock_path.exists()


def test_failed_teardown_releases_lifecycle_for_a_later_retry(tmp_path: Path) -> None:
    """A remove exception cannot leave the sequential lifecycle permanently busy."""

    class RemoveFailingBackend(FakeEnvironmentBackend):
        """Fail only the first teardown after otherwise successful use."""

        def __init__(self, events: list[str]) -> None:
            """Initialize one-shot teardown failure state."""

            super().__init__(events)
            self.remove_calls = 0

        def remove(self, prefix: Path) -> None:
            """Raise on the first removal and succeed on the next."""

            del prefix
            self.remove_calls += 1
            self.events.append("remove")
            if self.remove_calls == 1:
                raise DiskRecoveryError("synthetic teardown failure")

    root = _copy_env_specs(tmp_path)
    intent = load_environment_registry(root).intents["core"]
    events: list[str] = []
    lifecycle = SequentialEnvironmentLifecycle(
        RemoveFailingBackend(events),
        EffortTracker({"environment": StageCap(attempts=2, seconds=10, bytes=200)}),
        env_root=tmp_path / "active",
        disk_free=lambda _path: 1000,
        minimum_free_bytes=0,
    )

    with pytest.raises(EnvironmentCleanupError, match="synthetic teardown"):
        lifecycle.run(intent, use=lambda _prefix, _probes: events.append("use"))
    lifecycle.run(intent, use=lambda _prefix, _probes: events.append("use"))

    assert events.count("use") == 2


def test_lifecycle_rejects_empty_probe_contract_and_unverified_lock(tmp_path: Path) -> None:
    """Neither zero probe receipts nor solver-asserted artifact hashes are admissible."""

    root = _copy_env_specs(tmp_path)
    base_intent = load_environment_registry(root).intents["core"]
    empty_intent = replace(base_intent, probes=IntentProbes((), (), ()))
    lifecycle = SequentialEnvironmentLifecycle(
        FakeEnvironmentBackend([]),
        EffortTracker({"environment": StageCap(attempts=1, seconds=5, bytes=100)}),
        env_root=tmp_path / "empty-probes",
        disk_free=lambda _path: 1000,
        minimum_free_bytes=0,
    )
    with pytest.raises(EnvironmentProbeError, match="empty"):
        lifecycle.run(empty_intent, use=lambda _prefix, _probes: None)

    class UnverifiedArtifactBackend(FakeEnvironmentBackend):
        """Return a lock receipt that was not verified from the named artifact bytes."""

        def solve(self, environment_file: Path, target: str) -> SolveResult:
            """Change the materialized receipt while leaving lock bytes unchanged."""

            solved = super().solve(environment_file, target)
            return replace(
                solved,
                artifact_receipts=(ArtifactReceipt(_ARTIFACT_URL, "sha256:" + "f" * 64),),
            )

    lifecycle = SequentialEnvironmentLifecycle(
        UnverifiedArtifactBackend([]),
        EffortTracker({"environment": StageCap(attempts=1, seconds=5, bytes=100)}),
        env_root=tmp_path / "unverified-lock",
        disk_free=lambda _path: 1000,
        minimum_free_bytes=0,
    )
    with pytest.raises(EnvironmentExactnessError, match="materialized solver artifacts"):
        lifecycle.run(base_intent, use=lambda _prefix, _probes: None)
