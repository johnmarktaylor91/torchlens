"""Tests for pixi-based menagerie validation environments."""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from menagerie import envs
from menagerie.catalog import CatalogRow, build_canonical_rows


def _registry(tmp_path: Path) -> envs.EnvRegistry:
    """Return the real registry redirected to a temporary cache root.

    Parameters
    ----------
    tmp_path:
        Pytest temporary path.

    Returns
    -------
    envs.EnvRegistry
        Registry using a temporary managed cache root.
    """

    registry = envs.load_registry()
    return replace(registry, cache_root=tmp_path / "envs")


def test_assign_is_one_to_one_over_catalog(tmp_path: Path) -> None:
    """Every canonical catalog row maps to exactly one validation island."""

    registry = _registry(tmp_path)
    rows = build_canonical_rows()
    assignments = envs.assign(rows, registry)

    assert len(assignments) == len(rows)
    assert set(assignments) == {row.stable_id for row in rows}
    assert set(assignments.values()).issubset(registry.islands)

    base_capable = [
        row
        for row in rows
        if not envs._row_needs_dependency_env(row, registry)  # noqa: SLF001
    ]
    assert base_capable
    assert all(assignments[row.stable_id] == "base" for row in base_capable)


def test_forecast_tab_assignment_examples(tmp_path: Path) -> None:
    """Known forecast/tabular rows route to the forecast_tab island."""

    registry = _registry(tmp_path)
    rows_by_id = {row.stable_id: row for row in build_canonical_rows()}

    assert envs.env_for_row(rows_by_id["m3392"], registry) == "forecast_tab"
    assert envs.env_for_row(rows_by_id["m1907"], registry) == "forecast_tab"


def test_lock_hash_is_stable_with_mocked_pixi(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Locking the same manifest twice produces the same lock hash."""

    registry = _registry(tmp_path)
    locks_dir = tmp_path / "locks"

    def fake_run(
        command: list[str] | tuple[str, ...],
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> envs.CommandResult:
        """Write a deterministic pixi lock next to the work manifest."""

        del cwd, env, timeout
        manifest_path = Path(command[command.index("--manifest-path") + 1])
        (manifest_path.parent / "pixi.lock").write_text("locked = true\n", encoding="utf-8")
        return envs.CommandResult(0, "locked", "")

    monkeypatch.setattr(envs, "LOCKS_DIR", locks_dir)
    monkeypatch.setattr(envs, "pixi_bin", lambda: Path("/tmp/fake-pixi"))
    monkeypatch.setattr(envs, "_run_command", fake_run)

    first = envs.lock("forecast_tab", registry)
    second = envs.lock("forecast_tab", registry)

    assert first.returncode == 0
    assert first.lock_hash == second.lock_hash
    assert first.lock_path.read_text(encoding="utf-8") == "locked = true\n"


def test_disk_lru_evicts_managed_roots_until_projected_floor(tmp_path: Path) -> None:
    """The LRU removes only managed env roots until the projected floor is met."""

    cache_root = tmp_path / "envs"
    old = cache_root / "old-env"
    new = cache_root / "new-env"
    old.mkdir(parents=True)
    new.mkdir()
    (old / envs.BUILD_MARKER).write_text(json.dumps({"last_used": 1.0}), encoding="utf-8")
    (new / envs.BUILD_MARKER).write_text(json.dumps({"last_used": 2.0}), encoding="utf-8")
    free_values = iter([20.0, 24.0, 32.0])
    removed: list[Path] = []

    def fake_free(path: Path) -> float:
        """Return scripted free-space values."""

        assert path == cache_root
        return next(free_values)

    def fake_remove(path: Path) -> None:
        """Record and remove one managed directory."""

        removed.append(path)
        shutil.rmtree(path)

    result = envs.enforce_disk_lru(
        cache_root,
        reserve_gib=25.0,
        projected_gib=4.0,
        free_gib=fake_free,
        remove_tree=fake_remove,
    )

    assert result.allowed
    assert removed == [old, new]
    assert result.evicted == (old, new)


def test_disk_lru_refuses_when_reserve_still_below_floor(tmp_path: Path) -> None:
    """The disk guard refuses when managed eviction cannot restore reserve."""

    cache_root = tmp_path / "envs"
    stale = cache_root / "stale-env"
    stale.mkdir(parents=True)
    (stale / envs.BUILD_MARKER).write_text(json.dumps({"last_used": 1.0}), encoding="utf-8")
    free_values = iter([10.0, 12.0, 12.0])

    def fake_free(path: Path) -> float:
        """Return scripted free-space values."""

        assert path == cache_root
        return next(free_values)

    result = envs.enforce_disk_lru(
        cache_root,
        reserve_gib=25.0,
        projected_gib=1.0,
        free_gib=fake_free,
    )

    assert not result.allowed
    assert result.reason == "disk_floor"
    assert result.evicted == (stale,)


def test_env_manager_statuses_map_to_ledger_statuses() -> None:
    """Build statuses map to honest ledger terminal statuses."""

    assert envs.ledger_status_for_build_status("install_failed") == "install_failed"
    assert envs.ledger_status_for_build_status("env_unavailable") == "env_unavailable"
    assert envs.ledger_status_for_build_status("built") == "deferred"
    assert envs.ledger_status_for_build_status("cached") == "deferred"


def test_assign_rejects_duplicate_stable_ids(tmp_path: Path) -> None:
    """Assignment refuses duplicate stable IDs in the input sequence."""

    registry = _registry(tmp_path)
    row = CatalogRow(
        model_id=1,
        display_index=1,
        stable_id="duplicate",
        name="Toy",
        variant="",
        family="toy",
        family_normalized="toy",
        domain="toy",
        zoo="classics-pytorch",
        constructor_call="torch.nn.Linear(1, 1)",
        input_shape="(1, 1)",
        input_dtype="float32",
        era="2024",
        verified=True,
        notes="",
        source="catalog",
        recipe_revision_sha256="recipe",
    )

    with pytest.raises(ValueError, match="duplicate stable_id"):
        envs.assign([row, row], registry)
