"""The notebook runner runs ``HEAVY_NOTEBOOKS`` alone, after the parallel pool.

Two GPT-2 recipes that spend minutes in batched backward passes ran past the 600 s cell
timeout when they shared a 4-CPU runner, and each passed alone. These tests pin the
schedule (no heavy notebook overlaps any other notebook) and that every heavy entry is a
committed user notebook, with ``_run_child`` replaced by a recorder: no kernel starts.
"""

from __future__ import annotations

import importlib.util
import sys
import threading
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _runner() -> ModuleType:
    """Import ``scripts/run_notebooks.py`` as a module."""

    spec = importlib.util.spec_from_file_location(
        "_run_notebooks_under_test", REPO_ROOT / "scripts" / "run_notebooks.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_heavy_notebooks_are_user_notebooks() -> None:
    """Every heavy entry names a notebook the runner would select by default."""

    runner = _runner()
    selected = {path.relative_to(REPO_ROOT).as_posix() for path in runner.user_notebooks()}
    assert runner.HEAVY_NOTEBOOKS
    assert runner.HEAVY_NOTEBOOKS.issubset(selected)


def test_heavy_notebooks_run_alone_after_the_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    """With ``-j 3`` the light notebooks overlap, and no heavy notebook overlaps anything."""

    runner = _runner()
    heavy = sorted(runner.HEAVY_NOTEBOOKS)
    light = [
        rel
        for rel in (path.relative_to(REPO_ROOT).as_posix() for path in runner.user_notebooks())
        if rel not in runner.HEAVY_NOTEBOOKS
    ][:4]
    order = [heavy[0], light[0], heavy[1], light[1], light[2], heavy[2], light[3]]
    lock = threading.Lock()
    active: set[str] = set()
    overlaps: dict[str, set[str]] = {rel: set() for rel in order}
    finished: list[str] = []

    def fake_child(path: Path, args: Any) -> Any:
        """Record which notebooks run at the same time as this one."""

        rel = path.relative_to(REPO_ROOT).as_posix()
        with lock:
            for other in active:
                overlaps[other].add(rel)
                overlaps[rel].add(other)
            active.add(rel)
        time.sleep(0.05)
        with lock:
            active.discard(rel)
            finished.append(rel)
        return runner.NotebookResult(rel, "ok", 0.05)

    monkeypatch.setattr(runner, "_run_child", fake_child)
    assert runner.main(["-j", "3", *(str(REPO_ROOT / rel) for rel in order)]) == 0
    assert all(not overlaps[rel] for rel in heavy), overlaps
    assert any(overlaps[rel] for rel in light), "the light notebooks lost their parallel pool"
    assert finished[-3:] == heavy
