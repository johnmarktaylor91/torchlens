"""Slice C environment registry and sequential lifecycle tests."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Sequence

import pytest

from menagerie.crawler.env_lifecycle import (
    ProbeResult,
    SequentialEnvironmentLifecycle,
    SolveResult,
)
from menagerie.crawler.envs import DEFAULT_ENVS_ROOT, IntentProbes, load_environment_registry
from menagerie.crawler.effort import CapFailureRecord, EffortCapExceeded, EffortTracker, StageCap
from menagerie.crawler.identity import hash_bytes


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
        return SolveResult(b"exact-lock", b'{"python":"3.11"}\n', self.elapsed_seconds, 40)

    def create(self, lock_file: Path, prefix: Path) -> None:
        """Record creation after proving the lock was written."""

        assert lock_file.read_bytes() == b"exact-lock"
        self.events.append("create")

    def probe(self, prefix: Path, probes: IntentProbes) -> Sequence[ProbeResult]:
        """Return successful results for all declared import canaries."""

        self.events.append("probe")
        return tuple(ProbeResult(name, True, "ok") for name in probes.imports)

    def remove(self, prefix: Path) -> None:
        """Record deterministic teardown."""

        self.events.append("remove")


def _copy_env_specs(tmp_path: Path) -> Path:
    """Copy intent inputs so lifecycle outputs never modify the source tree."""

    target = tmp_path / "envs"
    shutil.copytree(DEFAULT_ENVS_ROOT, target)
    return target


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


def test_repository_contains_no_hand_authored_lock_or_hash_outputs() -> None:
    """Only setup documentation is committed under lock directories."""

    forbidden = [
        path
        for path in DEFAULT_ENVS_ROOT.rglob("*")
        if path.is_file()
        and (
            path.suffix == ".lock"
            or path.name.endswith(".resolved.json")
            or path.name.endswith(".resolved.sha256")
        )
    ]
    assert forbidden == []
    assert (DEFAULT_ENVS_ROOT / "locks" / "README.md").is_file()


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
