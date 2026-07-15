"""End-to-end fail-closed crawler checkpoint transaction tests."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Sequence

import pytest

import menagerie.crawler.checkpoint as checkpoint_module
from menagerie.crawler.checkpoint import (
    CheckpointValidationError,
    GitCommandResult,
    RestrictedPublicArtifact,
    WrongCheckpointBranch,
    create_canonical_checkpoint,
)
from menagerie.crawler.cli import main
from menagerie.crawler.constants import MODEL_SCHEMA_VERSION
from menagerie.crawler.driver import AuthorArtifact, _promote_accepted_code, default_driver_paths
from menagerie.crawler.driver import (
    AuthorLane,
    CrawlerDriver,
    DriverConfig,
    DriverDependencies,
    EnvironmentBinding,
    WorkItem,
)
from menagerie.crawler.envs import load_environment_registry
from menagerie.crawler.identity import canonical_json_bytes
from menagerie.crawler.intake import IntakeSnapshot, create_intake_snapshot
from menagerie.crawler.licenses import (
    LicenseEvidence,
    LicenseEvidenceStatus,
    LicensedArtifact,
    pre_public_merge_sweep,
    store_licensed_artifact,
)
from menagerie.crawler.mirrors import ArtifactOrigin, MirrorStore
from menagerie.crawler.recordio import JsonlLedger, scan_jsonl
from menagerie.crawler.tests.conftest import make_author_proposal, make_model
from menagerie.crawler.tests.test_slice_f_driver import (
    FakeChecker,
    FakeEnvironments,
    FakeForward,
    FakeNotifier,
    _refresh_proposal_identities,
)
from menagerie.crawler.tools.rebuild_views import rebuild_views


class RecordingGit:
    """Record checkpoint Git commands and emulate an empty index."""

    def __init__(self, staged: Sequence[Path] = ()) -> None:
        """Initialize with optional already-staged repository-relative paths."""

        self.staged = tuple(staged)
        self.commands: list[tuple[str, ...]] = []

    def __call__(self, argv: Sequence[str], cwd: Path) -> GitCommandResult:
        """Return deterministic branch, index, and add results."""

        del cwd
        command = tuple(argv)
        self.commands.append(command)
        if command[:4] == ("git", "diff", "--cached", "--name-only"):
            return GitCommandResult(0, "\0".join(path.as_posix() for path in self.staged), "")
        return GitCommandResult(0, "", "")


class CanonicalTypedAuthor(AuthorLane):
    """Author one typed adapter plus exact fetched-source CAS bytes."""

    def __init__(self) -> None:
        """Initialize invocation telemetry."""

        self.calls = 0

    def author(self, item: WorkItem, work_root: Path, config: DriverConfig) -> AuthorArtifact:
        """Return a repository-relative accepted-code proposal fixture."""

        self.calls += 1
        model_dir = work_root / item.stable_id / "author" / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        code_path = model_dir / "adapter.py"
        code_path.write_text(
            "def build_model() -> object:\n"
            "    return object()\n\n"
            "def make_dummy_call(seed: int, device: str) -> "
            "tuple[tuple[object, ...], dict[str, object]]:\n"
            "    return (), {}\n",
            encoding="utf-8",
        )
        source_path = work_root / item.stable_id / "author" / "source-cas" / "source.bin"
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_bytes(b"exact public implementation source")
        proposal = make_author_proposal(item.stable_id)
        proposal["work_id"] = f"work-{item.stable_id}"
        implementation = proposal["proposed_facts"]["implementation"]
        implementation.update(
            {
                "recipe_type": "typed-adapter",
                "code_path": "adapter.py",
                "code_sha256": checkpoint_module.hash_bytes(code_path.read_bytes()),
                "builder_symbol": "build_model",
                "dummy_call_symbol": "make_dummy_call",
                "library_recipe": None,
            }
        )
        proposal["verified_hashes"]["code"] = implementation["code_sha256"]
        proposal["proposed_facts"]["modes"]["meaningful_modes"] = ["train", "eval"]
        proposal["proposed_facts"]["external_metadata"]["modes"]["meaningful_modes"] = [
            "train",
            "eval",
        ]
        _refresh_proposal_identities(
            proposal,
            checker_model=config.checker_model,
            checker_version=config.checker_version,
        )
        source = {
            "source_id": "source-1",
            "url": "https://example.test/repository",
            "revision": "abc123",
            "content_sha256": checkpoint_module.hash_bytes(source_path.read_bytes()),
            "byte_count": source_path.stat().st_size,
            "media_type": "application/octet-stream",
            "cas_path": str(source_path),
        }
        return AuthorArtifact(proposal, {"sources": [source]}, model_dir)


class TypedFakeForward(FakeForward):
    """Echo the exact accepted adapter digest in fake successful receipts."""

    def forward(
        self,
        artifact: AuthorArtifact,
        environment: EnvironmentBinding,
        cold_runs: int,
        work_root: Path,
    ) -> Sequence[dict[str, Any]]:
        """Return clean attempts rebound to typed adapter bytes."""

        attempts = [
            dict(value) for value in super().forward(artifact, environment, cold_runs, work_root)
        ]
        digest = artifact.proposal["proposed_facts"]["implementation"]["code_sha256"]
        for attempt in attempts:
            attempt["worker_receipt"] = dict(attempt["worker_receipt"])
            attempt["worker_receipt"]["observed_adapter_sha256"] = digest
        return attempts


class DisabledAuthor(AuthorLane):
    """Fail if clean-clone resume attempts the author/network lane."""

    def author(self, item: WorkItem, work_root: Path, config: DriverConfig) -> AuthorArtifact:
        """Raise because reconstruction must satisfy resume before authoring."""

        del item, work_root, config
        raise AssertionError("author/network lane must remain disabled on clean-clone resume")


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Write canonical newline-terminated JSON objects."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in rows))


def _mirrors(root: Path) -> MirrorStore:
    """Return conventional separated runtime mirror roots."""

    runtime = root / ".crawl-local" / "mirrors"
    return MirrorStore(runtime / "public", runtime / "private", runtime / "local")


def _snapshot(root: Path, count: int = 1) -> IntakeSnapshot:
    """Create one canonical intake snapshot below the records allowlist."""

    master = root / "master.jsonl"
    deferred = root / "deferred.jsonl"
    _write_jsonl(
        master,
        [{"name": f"Example-{index}", "zoo": "tests", "variant": "base"} for index in range(count)],
    )
    _write_jsonl(deferred, [])
    return create_intake_snapshot(
        master,
        deferred,
        root / "menagerie" / "crawler" / "records" / "intake",
    )


def _clean_state(
    root: Path, *, intake_count: int = 1, status_code: str = "runs"
) -> tuple[IntakeSnapshot, MirrorStore]:
    """Materialize a complete canonical mini-campaign and matching derived facts."""

    snapshot = _snapshot(root, intake_count)
    crawler = root / "menagerie" / "crawler"
    records = crawler / "records"
    model = make_model(snapshot.items[0].stable_id, accepted=True, status_code=status_code)
    if status_code != "runs":
        model["execution"]["current"] = False
        model["completeness"]["execution_current"] = False
        model["completeness"]["release_eligible"] = False
        model["completeness"]["issues"] = [status_code]
    if status_code == "failed:forward":
        model["status"]["reason_code"] = "exception"
    with JsonlLedger(records / "models" / "current-shard.jsonl", MODEL_SCHEMA_VERSION) as ledger:
        ledger.append(model)
    _write_jsonl(records / "attempts" / "local.jsonl", [])
    _write_jsonl(records / "gates" / "current-shard.jsonl", [])
    rebuild_views(
        snapshot.root / "items.jsonl",
        records,
        crawler / "views",
        root / ".crawl-local" / "state.sqlite",
    )
    mirrors = _mirrors(root)
    artifacts: list[LicensedArtifact] = []
    for relative_root in (Path("records"), Path("source_manifests"), Path("evidence")):
        for path in sorted((crawler / relative_root).rglob("*")):
            if not path.is_file() or path.stat().st_size == 0:
                continue
            artifacts.append(
                store_licensed_artifact(
                    mirrors,
                    path.read_bytes(),
                    staged_path=path.relative_to(root),
                    origin=ArtifactOrigin("https://example.test/fixture", "v1"),
                    evidence=(
                        LicenseEvidence(
                            f"license-{len(artifacts)}",
                            "source-fixture",
                            "LICENSE:1",
                            "MIT License",
                            LicenseEvidenceStatus.DECLARED,
                            "MIT",
                        ),
                    ),
                )
            )
    rows = [
        {
            "staged_path": artifact.staged_path.as_posix(),
            "manifest": artifact.manifest.to_dict(),
            "decision": artifact.decision.to_dict(),
        }
        for artifact in artifacts
    ]
    _write_jsonl(crawler / "mirrors" / "public-manifest.jsonl", rows)
    _write_jsonl(crawler / "mirrors" / "private-manifest.jsonl", [])
    report = pre_public_merge_sweep(artifacts, mirrors)
    _write_jsonl(crawler / "license_reports" / "current.json", [report.to_dict()])
    return snapshot, mirrors


def test_checkpoint_refuses_wrong_branch_before_staging(tmp_path: Path) -> None:
    """A non-crawler branch is a typed refusal and cannot reach Git add."""

    git = RecordingGit()
    with pytest.raises(WrongCheckpointBranch):
        create_canonical_checkpoint(
            tmp_path,
            tmp_path / "missing-intake",
            branch="main",
            git_runner=git,
        )
    assert all(command[:3] != ("git", "add", "--") for command in git.commands)


def test_checkpoint_cli_wrong_branch_is_nonzero_and_never_claims_verified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI maps a typed branch refusal to nonzero without a success claim."""

    git = RecordingGit()

    def wrong_branch_git(argv: Sequence[str], cwd: Path) -> GitCommandResult:
        """Return ``main`` for branch inspection and record every Git command."""

        result = git(argv, cwd)
        if tuple(argv) == ("git", "branch", "--show-current"):
            return GitCommandResult(0, "main\n", "")
        return result

    monkeypatch.setattr(checkpoint_module, "_run_git", wrong_branch_git)
    exit_code = main(
        [
            "--repo-root",
            str(tmp_path),
            "checkpoint",
            "--intake",
            str(tmp_path / "missing-intake"),
            "--verify-ledgers",
            "--verify-views",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code != 0
    assert '"verified": true' not in captured.out.lower()
    assert "requires branch" in captured.err
    assert all(command[:3] != ("git", "add", "--") for command in git.commands)


def test_checkpoint_allows_consistent_incomplete_prefix(tmp_path: Path) -> None:
    """A valid current terminal prefix checkpoints before the full crawl completes."""

    snapshot, mirrors = _clean_state(tmp_path, intake_count=2)
    result = create_canonical_checkpoint(
        tmp_path,
        snapshot.root,
        mirrors=mirrors,
        branch="menagerie/crawler-pipeline",
        git_runner=RecordingGit(),
    )
    assert result.license_report.passed


def test_checkpoint_derives_and_refuses_restricted_public_artifact(tmp_path: Path) -> None:
    """A restricted public-manifest row is found without a caller artifact list."""

    snapshot, mirrors = _clean_state(tmp_path)
    artifact = store_licensed_artifact(
        mirrors,
        b"restricted source",
        staged_path=Path("menagerie/crawler/evidence/restricted.txt"),
        origin=ArtifactOrigin("https://example.test/restricted", "v1"),
        evidence=(
            LicenseEvidence(
                "license-gpl",
                "source-gpl",
                "LICENSE:1",
                "GPL license text",
                LicenseEvidenceStatus.DECLARED,
                "GPL-3.0-only",
            ),
        ),
    )
    row = {
        "staged_path": artifact.staged_path.as_posix(),
        "manifest": artifact.manifest.to_dict(),
        "decision": artifact.decision.to_dict(),
    }
    _write_jsonl(tmp_path / "menagerie" / "crawler" / "mirrors" / "public-manifest.jsonl", [row])
    with pytest.raises(RestrictedPublicArtifact):
        create_canonical_checkpoint(
            tmp_path,
            snapshot.root,
            mirrors=mirrors,
            branch="menagerie/crawler-pipeline",
            git_runner=RecordingGit(),
        )


def test_checkpoint_refuses_missing_license_report(tmp_path: Path) -> None:
    """A fresh sweep cannot substitute for its required persisted report."""

    snapshot, mirrors = _clean_state(tmp_path)
    (tmp_path / "menagerie" / "crawler" / "license_reports" / "current.json").unlink()
    with pytest.raises(CheckpointValidationError, match="persisted license report"):
        create_canonical_checkpoint(
            tmp_path,
            snapshot.root,
            mirrors=mirrors,
            branch="menagerie/crawler-pipeline",
            git_runner=RecordingGit(),
        )


def test_clean_checkpoint_stages_only_derived_allowlist_and_never_pushes(tmp_path: Path) -> None:
    """Clean canonical state stages every derived fact, no runtime state, and never pushes."""

    snapshot, mirrors = _clean_state(tmp_path)
    runtime_secret = tmp_path / ".crawl-local" / "secret.json"
    runtime_secret.parent.mkdir(parents=True, exist_ok=True)
    runtime_secret.write_text(json.dumps({"token": "must-not-stage"}), encoding="utf-8")
    git = RecordingGit()
    result = create_canonical_checkpoint(
        tmp_path,
        snapshot.root,
        mirrors=mirrors,
        branch="menagerie/crawler-pipeline",
        git_runner=git,
    )
    add = next(command for command in git.commands if command[:3] == ("git", "add", "--"))
    staged = set(add[3:])
    assert staged == {path.as_posix() for path in result.paths}
    assert "menagerie/crawler/license_reports/current.json" in staged
    assert all(not path.startswith(".crawl-local/") for path in staged)
    assert all("push" not in command for command in git.commands)


def test_checkpoint_includes_promoted_code_and_exact_environment_for_clean_reproduction(
    tmp_path: Path,
) -> None:
    """Accepted code plus target lock/export survive checkpoint and a clean-tree copy."""

    snapshot, mirrors = _clean_state(tmp_path)
    stable_id = snapshot.items[0].stable_id
    staging = tmp_path / ".crawl-local" / "work" / stable_id / "author" / "model"
    staging.mkdir(parents=True)
    adapter = staging / "adapter.py"
    adapter.write_text("def build_model() -> object:\n    return object()\n", encoding="utf-8")
    proposal = make_author_proposal(stable_id)
    proposal["proposed_facts"]["source_resolution"]["rung"] = "R2_VENDOR"
    proposal["proposed_facts"]["implementation"].update(
        {
            "recipe_type": "typed-adapter",
            "code_path": "adapter.py",
            "code_sha256": checkpoint_module.hash_bytes(adapter.read_bytes()),
            "library_recipe": None,
        }
    )
    promoted = _promote_accepted_code(
        AuthorArtifact(proposal, {"sources": []}, staging),
        default_driver_paths(tmp_path, snapshot.root),
    )
    promoted_path = promoted.model_dir / "adapter.py"
    assert promoted_path.is_file()
    assert ".crawl-local" not in promoted_path.parts

    env_root = tmp_path / "menagerie" / "crawler" / "envs" / "core"
    lock_path = env_root / "locks" / "linux-x86_64-cuda.lock"
    export_path = env_root / "resolved-exports" / "linux-x86_64-cuda.json"
    lock_path.parent.mkdir(parents=True)
    export_path.parent.mkdir(parents=True)
    (env_root / "environment.yml").write_text("name: core\n", encoding="utf-8")
    lock_path.write_bytes(b"exact target lock\n")
    export_path.write_bytes(b'{"python":"3.11","torch":"test"}\n')

    code_artifact = store_licensed_artifact(
        mirrors,
        promoted_path.read_bytes(),
        staged_path=promoted_path.relative_to(tmp_path),
        origin=ArtifactOrigin("https://example.test/authored-adapter", "v1"),
        evidence=(
            LicenseEvidence(
                "license-authored-adapter",
                "source-authored-adapter",
                "LICENSE:1",
                "MIT License",
                LicenseEvidenceStatus.DECLARED,
                "MIT",
            ),
        ),
    )
    manifest_path = tmp_path / "menagerie" / "crawler" / "mirrors" / "public-manifest.jsonl"
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    rows.append(
        {
            "staged_path": code_artifact.staged_path.as_posix(),
            "manifest": code_artifact.manifest.to_dict(),
            "decision": code_artifact.decision.to_dict(),
        }
    )
    _write_jsonl(manifest_path, rows)
    inventory = [checkpoint_module._licensed_artifact(row) for row in rows]
    report = pre_public_merge_sweep(inventory, mirrors)
    _write_jsonl(
        tmp_path / "menagerie" / "crawler" / "license_reports" / "current.json",
        [report.to_dict()],
    )

    result = create_canonical_checkpoint(
        tmp_path,
        snapshot.root,
        mirrors=mirrors,
        branch="menagerie/crawler-pipeline",
        git_runner=RecordingGit(),
    )
    relative_code = promoted_path.relative_to(tmp_path)
    relative_lock = lock_path.relative_to(tmp_path)
    relative_export = export_path.relative_to(tmp_path)
    assert {relative_code, relative_lock, relative_export} <= set(result.paths)

    clean = tmp_path / "clean-clone"
    for relative in result.paths:
        destination = clean / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(tmp_path / relative, destination)
    assert (clean / relative_code).read_bytes() == promoted_path.read_bytes()
    assert (clean / relative_lock).read_bytes() == lock_path.read_bytes()
    assert (clean / relative_export).read_bytes() == export_path.read_bytes()


def test_driver_checkpoint_clean_clone_resume_without_author_or_network(tmp_path: Path) -> None:
    """Production promotion checkpoints all facts needed for offline clean-clone resume."""

    snapshot = _snapshot(tmp_path)
    author = CanonicalTypedAuthor()
    paths = default_driver_paths(tmp_path, snapshot.root)
    driver = CrawlerDriver(
        paths,
        DriverConfig(
            run_id="canonical-first-run",
            machine_id="mac",
            review_checkpoint_at=None,
            progress_milestones=(),
        ),
        DriverDependencies(
            author,
            FakeChecker(),
            TypedFakeForward(),
            FakeEnvironments(tmp_path / "fake-envs"),
            FakeNotifier(),
            lambda: "2026-07-14T12:00:00Z",
        ),
        registry=load_environment_registry(target="osx-arm64"),
    )
    assert driver.run().status == "complete"
    assert author.calls == 1

    crawler = tmp_path / "menagerie" / "crawler"
    rebuild_views(
        snapshot.root / "items.jsonl",
        crawler / "records",
        crawler / "views",
        tmp_path / ".crawl-local" / "checkpoint-state.sqlite",
    )
    mirrors = _mirrors(tmp_path)
    checkpoint = create_canonical_checkpoint(
        tmp_path,
        snapshot.root,
        mirrors=mirrors,
        branch="menagerie/crawler-pipeline",
        git_runner=RecordingGit(),
    )
    assert any(path.parts[-3] == "reconstruction" for path in checkpoint.paths)
    assert any("source_cas" in path.parts for path in checkpoint.paths)
    assert any("adapters" in path.parts for path in checkpoint.paths)

    clean = tmp_path / "clean-clone"
    for relative in checkpoint.paths:
        destination = clean / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(tmp_path / relative, destination)
    shutil.rmtree(tmp_path / ".crawl-local")

    clean_intake = clean / "menagerie" / "crawler" / "records" / "intake" / snapshot.snapshot_id
    clean_paths = default_driver_paths(clean, clean_intake)
    forward = TypedFakeForward()
    resumed = CrawlerDriver(
        clean_paths,
        DriverConfig(
            run_id="clean-clone-resume",
            machine_id="mac",
            review_checkpoint_at=None,
            progress_milestones=(),
        ),
        DriverDependencies(
            DisabledAuthor(),
            FakeChecker(),
            forward,
            FakeEnvironments(clean / "fake-envs"),
            FakeNotifier(),
            lambda: "2026-07-14T12:00:00Z",
        ),
        registry=load_environment_registry(target="osx-arm64"),
    )
    assert resumed.run().status == "complete"
    assert forward.calls == {}


def test_environment_metadata_gets_safe_provenance_and_rejects_private_url(
    tmp_path: Path,
) -> None:
    """Locks/exports use the safe package-metadata class and reject private URLs."""

    snapshot, mirrors = _clean_state(tmp_path)
    env = tmp_path / "menagerie" / "crawler" / "envs" / "core"
    lock = env / "locks" / "osx-arm64.lock"
    lock.parent.mkdir(parents=True)
    lock.write_text("https://conda.example.test/pkg.conda#sha256=abc\n", encoding="utf-8")
    create_canonical_checkpoint(
        tmp_path,
        snapshot.root,
        mirrors=mirrors,
        branch="menagerie/crawler-pipeline",
        git_runner=RecordingGit(),
    )
    generated = scan_jsonl(
        tmp_path / "menagerie" / "crawler" / "mirrors" / "generated-metadata-manifest.jsonl",
        validate=False,
    )
    row = next(value for value in generated if value["staged_path"].endswith("osx-arm64.lock"))
    assert row["disposition"] == "safe-package-metadata-v1"
    assert row["content_sha256"] == checkpoint_module.hash_bytes(lock.read_bytes())

    lock.write_text(
        "https://user:secret@private.example.test/pkg.conda\n",  # pragma: allowlist secret
        encoding="utf-8",
    )
    with pytest.raises(checkpoint_module.SecretBearingPath, match="private/credential URL"):
        create_canonical_checkpoint(
            tmp_path,
            snapshot.root,
            mirrors=mirrors,
            branch="menagerie/crawler-pipeline",
            git_runner=RecordingGit(),
        )
