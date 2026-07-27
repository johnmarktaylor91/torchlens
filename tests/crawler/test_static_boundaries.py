"""Cross-cutting static boundaries over the complete crawler source tree."""

from __future__ import annotations

import subprocess
import shutil
from pathlib import Path

import pytest

from menagerie.crawler.checkpoint import (
    NonAllowlistedPath,
    WrongCheckpointBranch,
    create_checkpoint_set,
)
from menagerie.crawler.mirrors import MirrorStore
from menagerie.crawler.policy import static_source_check
from menagerie.crawler.tests.test_slice_e_checkpoint_retro import RecordingGit
from .support import (
    fabricated_crawler_locks,
    repository_root,
    tracked_paths,
)


def _mirror_store(root: Path) -> MirrorStore:
    """Return isolated empty mirror classes for checkpoint refusal tests."""

    return MirrorStore(root / "public", root / "private", root / "local")


def test_worker_execution_sources_hold_the_torchlens_import_ban() -> None:
    """Every worker/execution source passes the shared static policy check."""

    crawler_root = repository_root() / "menagerie" / "crawler"
    execution_paths = [
        crawler_root / name
        for name in (
            "worker.py",
            "worker_supervisor.py",
            "frameworks.py",
            "recipe.py",
            "standard_inputs.py",
        )
    ]
    for optional_root in ("adapters", "ports", "patches"):
        execution_paths.extend((crawler_root / optional_root).rglob("*.py"))
    for path in execution_paths:
        static_source_check(path)


def test_checkpoint_refuses_wrong_branch_and_nonallowlisted_paths(tmp_path: Path) -> None:
    """The cross-cutting staging boundary rejects both independent authority failures."""

    common = {
        "ledger_paths": [],
        "derived_view_checks": [],
        "public_artifacts": [],
        "mirrors": _mirror_store(tmp_path),
        "git_runner": RecordingGit(),
    }
    with pytest.raises(WrongCheckpointBranch):
        create_checkpoint_set(tmp_path, [], branch="main", **common)
    with pytest.raises(NonAllowlistedPath):
        create_checkpoint_set(
            tmp_path,
            [Path(".crawl-local/runtime.json")],
            branch="menagerie/crawler-pipeline",
            **common,
        )


def test_git_tree_ships_no_runtime_solve_or_secret_artifacts(tmp_path: Path) -> None:
    """Only source/spec facts are tracked; runtime, fabricated locks, and secrets stay local."""

    repo_root = repository_root()
    tracked = tracked_paths(repo_root)
    assert all(".crawl-local" not in path.parts for path in tracked)
    assert fabricated_crawler_locks(repo_root) == ()
    crawler_paths = [path for path in tracked if path.parts[:2] == ("menagerie", "crawler")]
    secret_parts = {"credential", "credentials", "secret", "secrets", "token"}
    assert all(
        not secret_parts.intersection(part.lower() for part in path.parts) for path in crawler_paths
    )
    baseline = tmp_path / ".secrets.baseline"
    shutil.copyfile(repo_root / ".secrets.baseline", baseline)
    secret_scan = subprocess.run(
        ["detect-secrets", "scan", "--baseline", str(baseline)],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert secret_scan.returncode == 0, secret_scan.stdout + secret_scan.stderr
