"""End-to-end fail-closed crawler checkpoint transaction tests."""

from __future__ import annotations

import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import pytest

import menagerie.crawler.checkpoint as checkpoint_module
import menagerie.crawler.driver as driver_module
from menagerie.crawler.checkpoint import (
    CheckpointValidationError,
    GeneratedMetadataDisposition,
    GitCommandResult,
    RestrictedPublicArtifact,
    WrongCheckpointBranch,
    create_checkpoint_set,
    create_canonical_checkpoint,
)
from menagerie.crawler.cli import main
from menagerie.crawler.constants import ATTEMPT_SCHEMA_VERSION
from menagerie.crawler.driver import (
    AuthorArtifact,
    DriverIntegrationError,
    _artifact_cache_identity,
    _execution_identity,
    _gated_path_license_bindings,
    _normalize_artifact_modes,
    _promote_and_publish_accepted_artifact,
    _promote_accepted_code,
    _publish_licensed_paths,
    _rehydrate_canonical_artifact,
    _validate_artifact_identities,
    default_driver_paths,
)
from menagerie.crawler.driver import (
    AuthorLane,
    CrawlerDriver,
    DriverConfig,
    DriverDependencies,
    DriverLock,
    EnvironmentBinding,
    WorkItem,
)
from menagerie.crawler.envs import DEFAULT_ENVS_ROOT, load_environment_registry
from menagerie.crawler.env_lifecycle import (
    materialized_environment_generation,
    parse_probe_receipt_bytes,
    parse_resolved_export,
)
from menagerie.crawler.identity import canonical_json_bytes, stable_hash
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
from menagerie.crawler.reducer import CanonicalReducer, default_ledger_paths
from menagerie.crawler.routing import ModelRequirements, route_model
from menagerie.crawler.tests.conftest import make_attempt, make_author_proposal, make_model
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
        source_digest = checkpoint_module.hash_bytes(source_path.read_bytes())
        proposal["proposed_facts"]["source_resolution"]["sources"][0].update(
            {
                "url": "https://example.test/repository",
                "revision": "abc123",
                "content_sha256": source_digest,
            }
        )
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
        source = {
            "source_id": "source-1",
            "url": "https://example.test/repository",
            "revision": "abc123",
            "content_sha256": source_digest,
            "byte_count": source_path.stat().st_size,
            "media_type": "application/octet-stream",
            "cas_path": str(source_path),
        }
        source_manifest = {
            "sources": [source],
            "manifest_sha256": stable_hash([source]),
        }
        proposal["verified_hashes"]["source_manifest"] = source_manifest["manifest_sha256"]
        _refresh_proposal_identities(
            proposal,
            checker_model=config.checker_model,
            checker_version=config.checker_version,
        )
        return AuthorArtifact(proposal, source_manifest, model_dir)


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
        implementation = artifact.proposal["proposed_facts"]["implementation"]
        digest = implementation["code_sha256"]
        manifest_digest = stable_hash(implementation["code_manifest"])
        for attempt in attempts:
            attempt["worker_receipt"] = dict(attempt["worker_receipt"])
            attempt["worker_receipt"]["observed_adapter_sha256"] = digest
            attempt["worker_receipt"]["observed_code_manifest_sha256"] = manifest_digest
            attempt["worker_receipt"]["observed_input_asset_sha256"] = None
            attempt["worker_receipt"]["input_asset"] = None
        return attempts


class DisabledAuthor(AuthorLane):
    """Fail if clean-clone resume attempts the author/network lane."""

    def author(self, item: WorkItem, work_root: Path, config: DriverConfig) -> AuthorArtifact:
        """Raise because reconstruction must satisfy resume before authoring."""

        del item, work_root, config
        raise AssertionError("author/network lane must remain disabled on clean-clone resume")


class DeferredCanonicalTypedAuthor(CanonicalTypedAuthor):
    """Author reconstructable typed code but defer its first run to Linux."""

    def author(self, item: WorkItem, work_root: Path, config: DriverConfig) -> AuthorArtifact:
        """Return an evidenced CUDA deferral over accepted pre-deferral facts."""

        artifact = super().author(item, work_root, config)
        return AuthorArtifact(
            artifact.proposal,
            artifact.source_manifest,
            artifact.model_dir,
            terminal_status="deferred:needs-cuda",
            terminal_detail="source proves an unavoidable CUDA operator",
            defer_evidence={
                "target_status": "deferred:needs-cuda",
                "source_ids": ["source-1"],
                "probe_attempt_ids": [],
                "explanation": "source proves an unavoidable CUDA operator",
            },
        )


def test_crash_mid_promotion_rolls_back_before_exposure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure before license publication exposes no incomplete model transaction."""

    snapshot = _snapshot(tmp_path)
    paths = default_driver_paths(tmp_path, snapshot.root)
    intake = snapshot.items[0]
    item = WorkItem(
        intake,
        route_model(ModelRequirements(intake.stable_id, "pytorch")),
    )
    config = DriverConfig(review_checkpoint_at=None, progress_milestones=())
    artifact = _normalize_artifact_modes(
        CanonicalTypedAuthor().author(item, paths.work_root, config),
        config,
    )
    original_publish = driver_module._publish_licensed_paths

    def crash_before_inventory(*args: object, **kwargs: object) -> None:
        """Simulate a process failure between byte promotion and inventory publication."""

        del args, kwargs
        raise RuntimeError("injected promotion crash")

    monkeypatch.setattr(driver_module, "_publish_licensed_paths", crash_before_inventory)
    with pytest.raises(RuntimeError, match="injected promotion crash"):
        _promote_and_publish_accepted_artifact(item, artifact, paths)
    prefix = intake.stable_id.removeprefix("m_")[:2]
    crawler = tmp_path / "menagerie" / "crawler"
    assert not (crawler / "adapters" / prefix / intake.stable_id).exists()
    assert not (crawler / "reconstruction" / prefix / f"{intake.stable_id}.json").exists()
    assert not (paths.runtime_root / "promotion-transactions" / intake.stable_id).exists()

    monkeypatch.setattr(driver_module, "_publish_licensed_paths", original_publish)
    promoted = _promote_and_publish_accepted_artifact(item, artifact, paths)
    assert promoted.canonical_code_root is not None
    assert (crawler / "reconstruction" / prefix / f"{intake.stable_id}.commit.json").is_file()


@pytest.mark.parametrize(
    "tamper",
    [
        "marker-transaction",
        "both-transaction-fields",
        "legacy-v1-downgrade",
        "proposal-digest",
        "missing-source-byte",
    ],
)
def test_reconstruction_tampering_is_refused_by_staging_and_rehydration(
    tmp_path: Path,
    tamper: str,
) -> None:
    """Checkpoint staging and offline rehydration share exact byte validation."""

    snapshot = _snapshot(tmp_path)
    paths = default_driver_paths(tmp_path, snapshot.root)
    intake = snapshot.items[0]
    item = WorkItem(
        intake,
        route_model(ModelRequirements(intake.stable_id, "pytorch")),
    )
    config = DriverConfig(review_checkpoint_at=None, progress_milestones=())
    artifact = _normalize_artifact_modes(
        CanonicalTypedAuthor().author(item, paths.work_root, config),
        config,
    )
    _promote_and_publish_accepted_artifact(item, artifact, paths)
    crawler = tmp_path / "menagerie" / "crawler"
    prefix = intake.stable_id.removeprefix("m_")[:2]
    reconstruction_path = crawler / "reconstruction" / prefix / f"{intake.stable_id}.json"
    marker_path = reconstruction_path.with_name(f"{intake.stable_id}.commit.json")
    reconstruction = json.loads(reconstruction_path.read_text(encoding="utf-8"))
    if tamper == "marker-transaction":
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        marker["transaction_id"] = "sha256:" + "f" * 64
        marker_path.write_bytes(canonical_json_bytes(marker) + b"\n")
    elif tamper == "both-transaction-fields":
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        marker["transaction_id"] = "sha256:" + "f" * 64
        reconstruction["transaction_id"] = marker["transaction_id"]
        marker_path.write_bytes(canonical_json_bytes(marker) + b"\n")
        reconstruction_path.write_bytes(canonical_json_bytes(reconstruction) + b"\n")
    elif tamper == "legacy-v1-downgrade":
        reconstruction["schema_version"] = "menagerie.crawler.reconstruction.v1"
        reconstruction.pop("transaction_id")
        reconstruction_path.write_bytes(canonical_json_bytes(reconstruction) + b"\n")
        marker_path.unlink()
    elif tamper == "proposal-digest":
        reconstruction["proposal_sha256"] = "sha256:" + "f" * 64
        reconstruction_path.write_bytes(canonical_json_bytes(reconstruction) + b"\n")
    else:
        source_path = tmp_path / reconstruction["source_manifest"]["sources"][0]["cas_path"]
        source_path.unlink()

    with pytest.raises(checkpoint_module.ReconstructionValidationError):
        checkpoint_module._derive_candidate_paths(tmp_path, crawler)
    with pytest.raises(DriverIntegrationError):
        _rehydrate_canonical_artifact(item, paths)


def test_coherent_reconstruction_rewrite_without_canonical_anchor_is_refused(
    tmp_path: Path,
) -> None:
    """Rehashing a rewritten proposal and marker cannot replace append-only gate authority."""

    snapshot = _snapshot(tmp_path)
    paths = default_driver_paths(tmp_path, snapshot.root)
    intake = snapshot.items[0]
    item = WorkItem(intake, route_model(ModelRequirements(intake.stable_id, "pytorch")))
    config = DriverConfig(review_checkpoint_at=None, progress_milestones=())
    artifact = _normalize_artifact_modes(
        CanonicalTypedAuthor().author(item, paths.work_root, config), config
    )
    _promote_and_publish_accepted_artifact(item, artifact, paths)
    gate = FakeChecker().check_metadata((artifact,), paths.work_root, config).gate
    assert gate is not None
    with JsonlLedger(paths.ledgers.gates, gate["schema_version"]) as ledger:
        ledger.append(gate)

    crawler = tmp_path / "menagerie" / "crawler"
    prefix = intake.stable_id.removeprefix("m_")[:2]
    reconstruction_path = crawler / "reconstruction" / prefix / f"{intake.stable_id}.json"
    marker_path = reconstruction_path.with_name(f"{intake.stable_id}.commit.json")
    reconstruction = json.loads(reconstruction_path.read_text(encoding="utf-8"))
    rewritten = reconstruction["proposal"]
    rewritten["work_id"] = "coherently-rewritten-work"
    rewritten["proposal_sha256"] = stable_hash(
        {key: value for key, value in rewritten.items() if key != "proposal_sha256"}
    )
    reconstruction["proposal_sha256"] = rewritten["proposal_sha256"]
    transaction_id = checkpoint_module.reconstruction_transaction_id(
        intake.stable_id,
        rewritten["proposal_sha256"],
        reconstruction["source_manifest"],
        reconstruction["intake_item_sha256"],
    )
    reconstruction["transaction_id"] = transaction_id
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["transaction_id"] = transaction_id
    marker["proposal_sha256"] = rewritten["proposal_sha256"]
    reconstruction_path.write_bytes(canonical_json_bytes(reconstruction) + b"\n")
    marker_path.write_bytes(canonical_json_bytes(marker) + b"\n")

    with pytest.raises(checkpoint_module.ReconstructionValidationError, match="not anchored"):
        checkpoint_module._derive_candidate_paths(tmp_path, crawler)
    with pytest.raises(DriverIntegrationError, match="not anchored"):
        _rehydrate_canonical_artifact(item, paths)


def test_reconstruction_source_rewrite_cannot_escape_anchored_proposal(
    tmp_path: Path,
) -> None:
    """A coherently rehashed source transaction remains bound to gated source facts."""

    snapshot = _snapshot(tmp_path)
    paths = default_driver_paths(tmp_path, snapshot.root)
    intake = snapshot.items[0]
    item = WorkItem(intake, route_model(ModelRequirements(intake.stable_id, "pytorch")))
    config = DriverConfig(review_checkpoint_at=None, progress_milestones=())
    artifact = _normalize_artifact_modes(
        CanonicalTypedAuthor().author(item, paths.work_root, config), config
    )
    _promote_and_publish_accepted_artifact(item, artifact, paths)
    gate = FakeChecker().check_metadata((artifact,), paths.work_root, config).gate
    assert gate is not None
    with JsonlLedger(paths.ledgers.gates, gate["schema_version"]) as ledger:
        ledger.append(gate)

    crawler = tmp_path / "menagerie" / "crawler"
    prefix = intake.stable_id.removeprefix("m_")[:2]
    reconstruction_path = crawler / "reconstruction" / prefix / f"{intake.stable_id}.json"
    marker_path = reconstruction_path.with_name(f"{intake.stable_id}.commit.json")
    reconstruction = json.loads(reconstruction_path.read_text(encoding="utf-8"))
    replacement = b"different already-public source bytes"
    replacement_digest = checkpoint_module.hash_bytes(replacement)
    replacement_path = (
        crawler / "source_cas" / (f"{replacement_digest.removeprefix('sha256:')}.source")
    )
    replacement_path.write_bytes(replacement)
    source = reconstruction["source_manifest"]["sources"][0]
    source["content_sha256"] = replacement_digest
    source["cas_path"] = replacement_path.relative_to(tmp_path).as_posix()
    reconstruction["source_manifest"]["manifest_sha256"] = stable_hash(
        reconstruction["source_manifest"]["sources"]
    )
    source_manifest_path = tmp_path / reconstruction["source_manifest_path"]
    source_manifest_path.write_bytes(
        canonical_json_bytes(reconstruction["source_manifest"]) + b"\n"
    )
    transaction_id = checkpoint_module.reconstruction_transaction_id(
        intake.stable_id,
        reconstruction["proposal_sha256"],
        reconstruction["source_manifest"],
        reconstruction["intake_item_sha256"],
    )
    reconstruction["transaction_id"] = transaction_id
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["transaction_id"] = transaction_id
    reconstruction_path.write_bytes(canonical_json_bytes(reconstruction) + b"\n")
    marker_path.write_bytes(canonical_json_bytes(marker) + b"\n")

    with pytest.raises(
        checkpoint_module.ReconstructionValidationError,
        match="anchored proposal",
    ):
        checkpoint_module.validate_canonical_reconstruction(
            reconstruction_path,
            crawler,
            canonical_gates=(gate,),
            current_models={},
        )


def test_reconstructed_proposal_requires_current_author_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Canonical reconstruction cannot bypass live author-prompt staleness."""

    snapshot = _snapshot(tmp_path)
    paths = default_driver_paths(tmp_path, snapshot.root)
    intake = snapshot.items[0]
    item = WorkItem(
        intake,
        route_model(ModelRequirements(intake.stable_id, "pytorch")),
    )
    config = DriverConfig(review_checkpoint_at=None, progress_milestones=())
    artifact = _normalize_artifact_modes(
        CanonicalTypedAuthor().author(item, paths.work_root, config),
        config,
    )
    _promote_and_publish_accepted_artifact(item, artifact, paths)
    gate = FakeChecker().check_metadata((artifact,), paths.work_root, config).gate
    assert gate is not None
    with JsonlLedger(paths.ledgers.gates, gate["schema_version"]) as ledger:
        ledger.append(gate)
    reconstructed = _rehydrate_canonical_artifact(item, paths)
    assert reconstructed is not None
    environment = EnvironmentBinding(
        prefix=tmp_path / "env",
        python_executable=Path(sys.executable),
        family="core",
        target="test",
        env_generation="sha256:" + "a" * 64,
        lock_sha256="sha256:" + "b" * 64,
        resolved_export_sha256="sha256:" + "c" * 64,
        packages_manifest_sha256="sha256:" + "d" * 64,
        python_version="3.11",
        compiler_identity="test",
        sdk_identity="test",
    )
    prior_execution = _execution_identity(reconstructed.proposal, environment)
    original_read_bytes = Path.read_bytes

    def changed_author_prompt(path: Path) -> bytes:
        """Return revised bytes only for the live frozen author prompt."""

        value = original_read_bytes(path)
        if path.name == "claude_crawler_author_v2.txt":
            return value + b"\nRequire one more source-bound fact.\n"
        return value

    monkeypatch.setattr(Path, "read_bytes", changed_author_prompt)
    with pytest.raises(DriverIntegrationError, match="current frozen prompt"):
        _validate_artifact_identities(reconstructed, config)
    assert _execution_identity(reconstructed.proposal, environment) != prior_execution


def test_imported_helper_change_stales_cache_and_execution_identity(tmp_path: Path) -> None:
    """Every imported helper path and byte digest participates in run identity."""

    snapshot = _snapshot(tmp_path)
    paths = default_driver_paths(tmp_path, snapshot.root)
    intake = snapshot.items[0]
    item = WorkItem(
        intake,
        route_model(ModelRequirements(intake.stable_id, "pytorch")),
    )
    config = DriverConfig(review_checkpoint_at=None, progress_milestones=())
    artifact = CanonicalTypedAuthor().author(item, paths.work_root, config)
    adapter = artifact.model_dir / "adapter.py"
    adapter.write_text(
        "import helper\n\n"
        "def build_model() -> object:\n"
        "    return object()\n\n"
        "def make_dummy_call(seed: int, device: str) -> "
        "tuple[tuple[object, ...], dict[str, object]]:\n"
        "    return (), {}\n",
        encoding="utf-8",
    )
    helper = artifact.model_dir / "helper.py"
    helper.write_text("def helper_value() -> int:\n    return 1\n", encoding="utf-8")
    artifact.proposal["proposed_facts"]["implementation"]["code_sha256"] = (
        checkpoint_module.hash_bytes(adapter.read_bytes())
    )
    artifact.proposal["verified_hashes"]["code"] = artifact.proposal["proposed_facts"][
        "implementation"
    ]["code_sha256"]
    _refresh_proposal_identities(
        artifact.proposal,
        checker_model=config.checker_model,
        checker_version=config.checker_version,
    )
    first = _normalize_artifact_modes(artifact, config)
    environment = EnvironmentBinding(
        prefix=tmp_path / "env",
        python_executable=Path(sys.executable),
        family="core",
        target="test",
        env_generation="sha256:" + "a" * 64,
        lock_sha256="sha256:" + "b" * 64,
        resolved_export_sha256="sha256:" + "c" * 64,
        packages_manifest_sha256="sha256:" + "d" * 64,
        python_version="3.11",
        compiler_identity="test",
        sdk_identity="test",
    )
    first_execution = _execution_identity(first.proposal, environment)
    first_cache = _artifact_cache_identity(item, first, config)

    helper.write_text("def helper_value() -> int:\n    return 2\n", encoding="utf-8")
    second = _normalize_artifact_modes(artifact, config)
    assert (
        second.proposal["proposed_facts"]["implementation"]["code_manifest"]
        != (first.proposal["proposed_facts"]["implementation"]["code_manifest"])
    )
    assert _execution_identity(second.proposal, environment) != first_execution
    assert _artifact_cache_identity(item, first, config) != first_cache


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Write canonical newline-terminated JSON objects."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in rows))


def _append_environment_attestation(
    records_root: Path,
    stable_id: str,
    *,
    family: str,
    target: str,
    lock_sha256: str,
    export_sha256: str,
) -> None:
    """Append one canonical attempt binding target lock, toolchain, and passed probes."""

    attempt = make_attempt(stable_id)
    env_root = records_root.parent / "envs"
    intent = load_environment_registry(env_root, target=target).intents[family]
    assert intent.lock.lock_bytes is not None
    assert intent.lock.export_bytes is not None
    package_bytes = parse_resolved_export(intent.lock.export_bytes)
    probe_results = parse_probe_receipt_bytes(
        intent.probes,
        intent.lock.lock_path.with_name(f"{target}.probes.json").read_bytes(),
    )
    generation = materialized_environment_generation(
        intent,
        lock_bytes=intent.lock.lock_bytes,
        export_bytes=intent.lock.export_bytes,
        package_bytes=package_bytes,
        python_version=str(attempt["environment"]["python"]),
        compiler_identity=str(attempt["environment"]["compiler_identity"]),
        sdk_identity=str(attempt["environment"]["sdk_identity"]),
        probe_results=probe_results,
    )
    attempt["environment"].update(
        {
            "family": family,
            "target": target,
            "lock_sha256": lock_sha256,
            "resolved_export_sha256": export_sha256,
            "packages_manifest_sha256": checkpoint_module.hash_bytes(package_bytes),
        }
    )
    attempt["identities"]["environment"] = generation
    with JsonlLedger(
        records_root / "attempts" / "environment-attestation.jsonl", ATTEMPT_SCHEMA_VERSION
    ) as ledger:
        ledger.append(attempt)


def _write_exact_environment_artifacts(env_root: Path, target: str) -> tuple[Path, Path]:
    """Write one canonical lock/export/probe fixture for the core intent.

    Parameters
    ----------
    env_root, target:
        Copied environment registry root and exact target basename.

    Returns
    -------
    tuple[pathlib.Path, pathlib.Path]
        Lock and resolved-export paths.
    """

    locks = env_root / "core" / "locks"
    locks.mkdir(parents=True, exist_ok=True)
    artifact_sha256 = "sha256:" + "a" * 64
    artifact_url = "https://conda.example.test/core.conda"
    lock = locks / f"{target}.lock"
    export = locks / f"{target}.resolved.json"
    lock.write_text(f"{artifact_url}#{artifact_sha256.removeprefix('sha256:')}\n", encoding="utf-8")
    export.write_bytes(
        canonical_json_bytes(
            {
                "packages": [
                    {
                        "name": "python",
                        "version": "3.11",
                        "build": "h1_0",
                        "url": artifact_url,
                        "sha256": artifact_sha256,
                    }
                ]
            }
        )
        + b"\n"
    )
    (locks / f"{target}.resolved.sha256").write_text(
        f"{checkpoint_module.hash_bytes(export.read_bytes())}\n", encoding="utf-8"
    )
    intent = load_environment_registry(env_root, target=target).intents["core"]
    names = [
        *(f"import:{name}" for name in intent.probes.imports),
        *(f"export:{check.module}:{check.attribute}" for check in intent.probes.export_checks),
        *(f"source-build:{build.name}" for build in intent.probes.source_build),
    ]
    (locks / f"{target}.probes.json").write_bytes(
        canonical_json_bytes(
            {"probes": [{"name": name, "passed": True, "detail": "ok"} for name in names]}
        )
        + b"\n"
    )
    return lock, export


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
    root: Path, *, intake_count: int = 1, status_code: str = "failed:source"
) -> tuple[IntakeSnapshot, MirrorStore]:
    """Materialize a complete canonical mini-campaign and matching derived facts."""

    snapshot = _snapshot(root, intake_count)
    crawler = root / "menagerie" / "crawler"
    records = crawler / "records"
    model = make_model(snapshot.items[0].stable_id, accepted=False, status_code=status_code)
    if status_code != "runs":
        model["execution"]["current"] = False
        model["completeness"]["execution_current"] = False
        model["completeness"]["release_eligible"] = False
        model["completeness"]["issues"] = [status_code]
    if status_code == "failed:forward":
        model["status"]["reason_code"] = "exception"
        # A bare digest is not a retrievable diagnostic reference. Public failure
        # details are empty unless they carry the explicit local-sidecar shape.
        model["status"]["detail"] = None
    ledgers = default_ledger_paths(records)
    with CanonicalReducer(ledgers, (item.stable_id for item in snapshot.items)) as reducer:
        if status_code.startswith("deferred:"):
            attempt = make_attempt(snapshot.items[0].stable_id)
            attempt["environment"] = None
            attempt["identities"]["environment"] = None
            attempt["identities"]["execution"] = None
            attempt["worker_receipt"]["present"] = False
            attempt["supervisor_observation"]["stdout_sha256"] = None
            attempt["supervisor_observation"]["stderr_sha256"] = None
            attempt["defer_evidence"] = {
                "target_status": status_code,
                "source_ids": ["source-1"],
                "probe_attempt_ids": [],
                "explanation": "source requires the deferred platform",
            }
            reducer.append_attempt(attempt)
        reducer.append_model(model)
    rebuild_views(
        snapshot.root / "items.jsonl",
        records,
        crawler / "views",
        root / ".crawl-local" / "state.sqlite",
    )
    mirrors = _mirrors(root)
    artifacts: list[LicensedArtifact] = []
    _write_jsonl(crawler / "mirrors" / "public-manifest.jsonl", [])
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


def test_checkpoint_cli_respects_busy_driver_lock(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Checkpoint refuses before deriving candidates while the driver owns its lock."""

    lock = tmp_path / ".crawl-local" / "locks" / "driver.lock"
    with DriverLock(lock, {"pid": 1, "run_id": "live-driver"}):
        exit_code = main(
            [
                "--repo-root",
                str(tmp_path),
                "checkpoint",
                "--intake",
                str(tmp_path / "missing-intake"),
            ]
        )
    captured = capsys.readouterr()
    assert exit_code == 3
    assert "another driver owns" in captured.err


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


def test_checkpoint_refuses_restricted_excerpt_embedded_in_failure_record(
    tmp_path: Path,
) -> None:
    """Whole-file safe-metadata labels cannot hide restricted excerpt bytes."""

    candidate = Path("menagerie/crawler/records/models/failed.jsonl")
    content = (
        canonical_json_bytes(
            {
                "status": {"code": "failed:forward"},
                "evidence": {
                    "excerpts": [
                        {
                            "text": "restricted third-party excerpt",
                            "license_disposition": "restricted-private",
                        }
                    ]
                },
            }
        )
        + b"\n"
    )
    path = tmp_path / candidate
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    generated = GeneratedMetadataDisposition(
        candidate,
        checkpoint_module.hash_bytes(content),
        len(content),
        "safe-generated-metadata-v1",
        "test",
        "claimed safe metadata",
    )
    with pytest.raises(RestrictedPublicArtifact, match="embedded excerpts"):
        create_checkpoint_set(
            tmp_path,
            [candidate],
            ledger_paths=[],
            derived_view_checks=[],
            public_artifacts=[],
            mirrors=_mirrors(tmp_path),
            generated_metadata_inventory=[generated],
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


def test_promotion_uses_exact_per_source_license_and_origin(tmp_path: Path) -> None:
    """Heterogeneous source paths cannot inherit the code repository's provenance."""

    proposal = make_author_proposal("m_heterogeneous")
    facts = proposal["proposed_facts"]
    facts["source_resolution"]["sources"].append(
        {
            **facts["source_resolution"]["sources"][0],
            "source_id": "paper-2",
            "url": "https://example.test/paper",
            "revision": "paper-v2",
        }
    )
    facts["evidence"]["excerpts"].append(
        {
            **facts["evidence"]["excerpts"][0],
            "evidence_id": "paper-license",
            "source_id": "paper-2",
            "locator": "LICENSE:1",
            "text": "Creative Commons Zero",
        }
    )
    facts["licenses"]["source_dispositions"] = [
        {
            "spdx": "CC0-1.0",
            "status": "declared",
            "source_id": "paper-2",
            "locator": "LICENSE:1",
            "evidence_ids": ["paper-license"],
        }
    ]
    source_manifest = {
        "sources": [
            {
                "source_id": "source-1",
                "url": "https://example.com/model",
                "revision": "abc123",
            },
            {
                "source_id": "paper-2",
                "url": "https://example.test/paper",
                "revision": "paper-v2",
            },
        ]
    }
    bindings = _gated_path_license_bindings(proposal, source_manifest)
    code_path = tmp_path / "menagerie" / "crawler" / "adapters" / "code.py"
    paper_path = tmp_path / "menagerie" / "crawler" / "source_cas" / "paper.source"
    code_path.parent.mkdir(parents=True)
    paper_path.parent.mkdir(parents=True)
    code_path.write_text("licensed code", encoding="utf-8")
    paper_path.write_text("licensed paper archive", encoding="utf-8")
    _publish_licensed_paths(
        tmp_path,
        tmp_path / "menagerie" / "crawler",
        (
            (code_path, *bindings["__code__"]),
            (paper_path, *bindings["paper-2"]),
        ),
    )
    rows = scan_jsonl(
        tmp_path / "menagerie" / "crawler" / "mirrors" / "public-manifest.jsonl",
        validate=False,
    )
    origins = {row["staged_path"]: row["manifest"]["origin"] for row in rows}
    assert origins[code_path.relative_to(tmp_path).as_posix()]["url"] == (
        "https://example.com/model"
    )
    assert origins[paper_path.relative_to(tmp_path).as_posix()]["url"] == (
        "https://example.test/paper"
    )
    facts["licenses"]["source_dispositions"].append(
        dict(facts["licenses"]["source_dispositions"][0])
    )
    with pytest.raises(DriverIntegrationError, match="exactly one gated disposition"):
        _gated_path_license_bindings(proposal, source_manifest)


def test_public_license_rows_append_in_arrival_order_and_preserve_prefix(tmp_path: Path) -> None:
    """A later lexically earlier path extends, rather than rewrites, canonical history."""

    proposal = make_author_proposal("m_license_order")
    binding = _gated_path_license_bindings(proposal, {"sources": []})["__code__"]
    canonical_root = tmp_path / "menagerie" / "crawler"
    late = canonical_root / "adapters" / "z.py"
    early = canonical_root / "adapters" / "a.py"
    late.parent.mkdir(parents=True)
    late.write_text("late arrival", encoding="utf-8")
    early.write_text("early lexical path", encoding="utf-8")

    _publish_licensed_paths(tmp_path, canonical_root, ((late, *binding),))
    manifest = canonical_root / "mirrors" / "public-manifest.jsonl"
    committed_prefix = manifest.read_bytes()
    _publish_licensed_paths(tmp_path, canonical_root, ((early, *binding),))
    extended = manifest.read_bytes()
    assert extended.startswith(committed_prefix)
    assert [row["staged_path"] for row in scan_jsonl(manifest, validate=False)] == [
        late.relative_to(tmp_path).as_posix(),
        early.relative_to(tmp_path).as_posix(),
    ]

    _publish_licensed_paths(tmp_path, canonical_root, ((early, *binding),))
    assert manifest.read_bytes() == extended


@pytest.mark.parametrize(
    ("spdx", "status", "redistribution"),
    (
        ("GPL-3.0-only", "declared", "restricted-private"),
        ("NOASSERTION", "custom", "manifest-only"),
    ),
)
def test_restricted_and_unknown_promotion_use_only_private_mirror(
    tmp_path: Path,
    spdx: str,
    status: str,
    redistribution: str,
) -> None:
    """Restricted/unknown source and code bytes never enter public roots."""

    snapshot = _snapshot(tmp_path)
    paths = default_driver_paths(tmp_path, snapshot.root)
    intake = snapshot.items[0]
    item = WorkItem(intake, route_model(ModelRequirements(intake.stable_id, "pytorch")))
    config = DriverConfig(review_checkpoint_at=None, progress_milestones=())
    artifact = _normalize_artifact_modes(
        CanonicalTypedAuthor().author(item, paths.work_root, config), config
    )
    licenses = artifact.proposal["proposed_facts"]["licenses"]
    licenses["code"]["spdx"] = spdx
    licenses["code"]["status"] = status
    licenses["redistribution_class"] = redistribution
    _refresh_proposal_identities(
        artifact.proposal,
        checker_model=config.checker_model,
        checker_version=config.checker_version,
    )
    artifact = replace(artifact, proposal=artifact.proposal)
    gate = FakeChecker().check_metadata((artifact,), paths.work_root, config).gate
    assert gate is not None
    with JsonlLedger(paths.ledgers.gates, gate["schema_version"]) as ledger:
        ledger.append(gate)

    promoted = _promote_and_publish_accepted_artifact(item, artifact, paths)
    crawler = tmp_path / "menagerie" / "crawler"
    source_digest = artifact.source_manifest["sources"][0]["content_sha256"]
    source_path = crawler / "source_cas" / f"{source_digest.removeprefix('sha256:')}.source"
    prefix = intake.stable_id.removeprefix("m_")[:2]
    code_path = crawler / "adapters" / prefix / intake.stable_id / "adapter.py"

    assert not source_path.exists()
    assert not code_path.exists()
    assert promoted.model_dir.is_relative_to(paths.runtime_root)
    assert scan_jsonl(crawler / "mirrors" / "public-manifest.jsonl", validate=False) == []
    private_rows = scan_jsonl(crawler / "mirrors" / "private-manifest.jsonl", validate=False)
    assert {row["staged_path"] for row in private_rows} == {
        source_path.relative_to(tmp_path).as_posix(),
        code_path.relative_to(tmp_path).as_posix(),
    }
    mirrors = _mirrors(tmp_path)
    for row in private_rows:
        artifact_manifest = checkpoint_module._licensed_artifact(row)
        assert mirrors.fetch(artifact_manifest.manifest)
    shutil.rmtree(artifact.model_dir)
    for source in artifact.source_manifest["sources"]:
        Path(source["cas_path"]).unlink()
    reconstructed = _rehydrate_canonical_artifact(item, paths)
    assert reconstructed is not None
    assert (reconstructed.model_dir / "adapter.py").is_file()


def test_promotion_uses_snapshot_identity_when_intake_directory_is_renamed(
    tmp_path: Path,
) -> None:
    """An exact renamed snapshot promotes and rehydrates under its canonical ID."""

    snapshot = _snapshot(tmp_path)
    renamed_root = tmp_path / "renamed-exact-snapshot"
    shutil.copytree(snapshot.root, renamed_root)
    renamed_snapshot = replace(snapshot, root=renamed_root)
    paths = default_driver_paths(tmp_path, renamed_root)
    intake = renamed_snapshot.items[0]
    item = WorkItem(intake, route_model(ModelRequirements(intake.stable_id, "pytorch")))
    config = DriverConfig(review_checkpoint_at=None, progress_milestones=())
    artifact = _normalize_artifact_modes(
        CanonicalTypedAuthor().author(item, paths.work_root, config), config
    )
    gate = FakeChecker().check_metadata((artifact,), paths.work_root, config).gate
    assert gate is not None
    with JsonlLedger(paths.ledgers.gates, gate["schema_version"]) as ledger:
        ledger.append(gate)

    _promote_and_publish_accepted_artifact(item, artifact, paths)
    prefix = intake.stable_id.removeprefix("m_")[:2]
    reconstruction_path = (
        tmp_path / "menagerie" / "crawler" / "reconstruction" / prefix / f"{intake.stable_id}.json"
    )
    reconstruction = json.loads(reconstruction_path.read_text(encoding="utf-8"))

    assert reconstruction["intake_snapshot_id"] == snapshot.snapshot_id
    assert (
        tmp_path / "menagerie" / "crawler" / "records" / "intake" / snapshot.snapshot_id
    ).is_dir()
    assert not (
        tmp_path / "menagerie" / "crawler" / "records" / "intake" / renamed_root.name
    ).exists()
    assert _rehydrate_canonical_artifact(item, paths) is not None


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


def test_checkpoint_rejects_self_attested_promoted_code_license(
    tmp_path: Path,
) -> None:
    """Promoted code cannot use a license decision absent from current gated facts."""

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

    crawler_envs = tmp_path / "menagerie" / "crawler" / "envs"
    shutil.copytree(DEFAULT_ENVS_ROOT, crawler_envs)
    lock_path, export_path = _write_exact_environment_artifacts(crawler_envs, "linux-x86_64-cuda")
    _append_environment_attestation(
        tmp_path / "menagerie" / "crawler" / "records",
        stable_id,
        family="core",
        target="linux-x86_64-cuda",
        lock_sha256=checkpoint_module.hash_bytes(lock_path.read_bytes()),
        export_sha256=checkpoint_module.hash_bytes(export_path.read_bytes()),
    )

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
            "artifact_role": "code",
            "source_id": "source-authored-adapter",
            "fetch_recipe": "https-get",
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

    with pytest.raises(RestrictedPublicArtifact, match="closed dependency-current"):
        create_canonical_checkpoint(
            tmp_path,
            snapshot.root,
            mirrors=mirrors,
            branch="menagerie/crawler-pipeline",
            git_runner=RecordingGit(),
        )


def test_driver_checkpoint_clean_clone_resume_without_author_or_network(tmp_path: Path) -> None:
    """Production promotion checkpoints all facts needed for offline clean-clone resume."""

    snapshot = _snapshot(tmp_path)
    author = CanonicalTypedAuthor()
    paths = default_driver_paths(tmp_path, snapshot.root)
    crawler_envs = tmp_path / "menagerie" / "crawler" / "envs"
    shutil.copytree(DEFAULT_ENVS_ROOT, crawler_envs)
    _write_exact_environment_artifacts(crawler_envs, "osx-arm64")
    registry = load_environment_registry(crawler_envs, target="osx-arm64")

    class ExactFakeEnvironments(FakeEnvironments):
        """Materialize installed metadata matching the canonical test export."""

        def run(self, intent: Any, *, use: Any) -> object:
            """Write immutable installed metadata before invoking the fake lifecycle."""

            prefix = self.root / intent.name
            metadata = prefix / "conda-meta"
            metadata.mkdir(parents=True, exist_ok=True)
            export = json.loads(intent.lock.export_path.read_bytes())
            (metadata / "python.json").write_bytes(canonical_json_bytes(export["packages"][0]))
            return super().run(intent, use=use)

    environments = ExactFakeEnvironments(tmp_path / "fake-envs")
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
            environments,
            FakeNotifier(),
            lambda: "2026-07-14T12:00:00Z",
        ),
        registry=registry,
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
            ExactFakeEnvironments(clean / "fake-envs"),
            FakeNotifier(),
            lambda: "2026-07-14T12:00:00Z",
        ),
        registry=load_environment_registry(
            clean / "menagerie" / "crawler" / "envs", target="osx-arm64"
        ),
    )
    assert resumed.run().status == "complete"
    assert forward.calls == {}


def test_deferred_ungated_promotion_cannot_checkpoint_public_code(tmp_path: Path) -> None:
    """A pre-gate deferral cannot self-attest promoted code as public."""

    snapshot = _snapshot(tmp_path)
    paths = default_driver_paths(tmp_path, snapshot.root)
    author = DeferredCanonicalTypedAuthor()
    first = CrawlerDriver(
        paths,
        DriverConfig(review_checkpoint_at=None, progress_milestones=()),
        DriverDependencies(
            author,
            FakeChecker(),
            TypedFakeForward(),
            FakeEnvironments(tmp_path / "mac-envs"),
            FakeNotifier(),
            lambda: "2026-07-14T12:00:00Z",
        ),
        registry=load_environment_registry(target="osx-arm64"),
    )
    assert first.run().status == "complete"
    assert author.calls == 1
    crawler = tmp_path / "menagerie" / "crawler"
    rebuild_views(
        snapshot.root / "items.jsonl",
        crawler / "records",
        crawler / "views",
        tmp_path / ".crawl-local" / "checkpoint-state.sqlite",
    )
    with pytest.raises(RestrictedPublicArtifact, match="closed dependency-current"):
        create_canonical_checkpoint(
            tmp_path,
            snapshot.root,
            mirrors=_mirrors(tmp_path),
            branch="menagerie/crawler-pipeline",
            git_runner=RecordingGit(),
        )


def test_environment_metadata_gets_safe_provenance_and_rejects_private_url(
    tmp_path: Path,
) -> None:
    """Locks/exports use the safe package-metadata class and reject private URLs."""

    snapshot, mirrors = _clean_state(tmp_path)
    crawler_envs = tmp_path / "menagerie" / "crawler" / "envs"
    shutil.copytree(DEFAULT_ENVS_ROOT, crawler_envs)
    lock, export = _write_exact_environment_artifacts(crawler_envs, "osx-arm64")
    with pytest.raises(CheckpointValidationError, match="compiler/SDK/probe-bound"):
        create_canonical_checkpoint(
            tmp_path,
            snapshot.root,
            mirrors=mirrors,
            branch="menagerie/crawler-pipeline",
            git_runner=RecordingGit(),
        )
    _append_environment_attestation(
        tmp_path / "menagerie" / "crawler" / "records",
        snapshot.items[0].stable_id,
        family="core",
        target="osx-arm64",
        lock_sha256=checkpoint_module.hash_bytes(lock.read_bytes()),
        export_sha256=checkpoint_module.hash_bytes(export.read_bytes()),
    )
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

    lock.write_text("https://conda.example.test/pkg.conda#sha256=abc\n", encoding="utf-8")
    with pytest.raises(CheckpointValidationError, match="canonical SHA-256"):
        create_canonical_checkpoint(
            tmp_path,
            snapshot.root,
            mirrors=mirrors,
            branch="menagerie/crawler-pipeline",
            git_runner=RecordingGit(),
        )

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


def test_requeue_cli_reports_success_only_after_canonical_append(tmp_path: Path) -> None:
    """Operator authorization survives deletion of all disposable runtime state."""

    snapshot = _snapshot(tmp_path)
    stable_id = snapshot.items[0].stable_id
    result = main(
        [
            "--repo-root",
            str(tmp_path),
            "requeue",
            stable_id,
            "--reason",
            "reviewed retry",
            "--grant",
            "1",
            "--stage",
            "forward",
        ]
    )
    assert result == 0
    canonical = (
        tmp_path / "menagerie" / "crawler" / "records" / "operational" / "requeue-grants.jsonl"
    )
    grants = scan_jsonl(canonical, validate=False)
    assert len(grants) == 1
    assert grants[0]["new_work_generation"] == 1
    shutil.rmtree(tmp_path / ".crawl-local")
    assert scan_jsonl(canonical, validate=False) == grants
