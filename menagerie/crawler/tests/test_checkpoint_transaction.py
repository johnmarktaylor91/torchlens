"""End-to-end fail-closed crawler checkpoint transaction tests."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Sequence

import pytest

import menagerie.crawler.checkpoint as checkpoint_module
from menagerie.crawler.artifact_transactions import ArtifactInput, stage_private_artifact
from menagerie.crawler.authority import AuthorityContext
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
from menagerie.crawler.constants import ATTEMPT_SCHEMA_VERSION_V3 as ATTEMPT_SCHEMA_VERSION
from menagerie.crawler.driver import (
    AuthorArtifact,
)
from menagerie.crawler.driver import (
    AuthorLane,
    DriverConfig,
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
from menagerie.crawler.identity import canonical_json_bytes, hash_bytes, stable_hash
from menagerie.crawler.intake import IntakeSnapshot, create_intake_snapshot
from menagerie.crawler.licenses import (
    LicenseEvidence,
    LicenseEvidenceStatus,
    LicensedArtifact,
    pre_public_merge_sweep,
    store_licensed_artifact,
)
from menagerie.crawler.mirrors import ArtifactOrigin, MirrorClass, MirrorStore
from menagerie.crawler.recordio import JsonlLedger, scan_jsonl
from menagerie.crawler.reducer import CanonicalReducer, default_ledger_paths
from menagerie.crawler.tests.conftest import (
    HASH,
    NOW,
    _bind_model_identities,
    bind_terminal_attempts,
    make_attempt,
    make_authority_context,
    make_author_proposal,
    make_failed_attempt,
    make_gate,
    make_model,
)
from menagerie.crawler.tests.test_slice_f_driver import (
    FakeForward,
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
    root: Path,
    *,
    intake_count: int = 1,
    status_code: str = "failed:source",
    accepted_metadata: bool = False,
) -> tuple[IntakeSnapshot, MirrorStore]:
    """Materialize a complete canonical mini-campaign and matching derived facts.

    Parameters
    ----------
    root:
        Temporary repository root.
    intake_count:
        Number of intake rows in the checkpoint prefix.
    status_code:
        Current terminal status for the materialized model.
    accepted_metadata:
        Whether a non-skip terminal has accepted gated metadata without a
        canonical promotion transaction.

    Returns
    -------
    tuple[IntakeSnapshot, MirrorStore]
        Canonical intake snapshot and empty separated mirror stores.
    """

    snapshot = _snapshot(root, intake_count)
    crawler = root / "menagerie" / "crawler"
    records = crawler / "records"
    stable_id = snapshot.items[0].stable_id
    model = make_model(stable_id, accepted=accepted_metadata, status_code=status_code)
    terminal_attempts: list[dict[str, Any]] = []
    gate: dict[str, Any] | None = None
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
        terminal_attempts = [
            make_failed_attempt(stable_id, stage="forward", reason_code="exception")
        ]
        bind_terminal_attempts(model, terminal_attempts)
    elif status_code.startswith("failed:"):
        stage = status_code.split(":", 1)[1]
        terminal_attempts = [
            make_failed_attempt(
                stable_id,
                stage=stage,
                reason_code=str(model["status"]["reason_code"]),
            )
        ]
        bind_terminal_attempts(model, terminal_attempts)
    elif status_code.startswith("skipped:"):
        model = make_model(stable_id, accepted=True, status_code=status_code)
        model["source_resolution"]["rung"] = "R5_SKIP"
        model["source_resolution"]["attempted_rungs"][0]["rung"] = "R5_SKIP"
        model["execution"]["accepted_attempt_ids"] = []
        bind_terminal_attempts(model, [])
        _bind_model_identities(model)
    if model["authored_metadata_state"] == "accepted":
        gate = make_gate([stable_id])
        gate["items"][0]["vet_identity"] = model["accuracy_gate"]["vet_identity"]
        gate["items"][0]["rung_check"]["selected_rung"] = model["source_resolution"]["rung"]
    ledgers = default_ledger_paths(records)
    context = make_authority_context(
        (item.stable_id for item in snapshot.items),
        snapshot_id=snapshot.snapshot_id,
        snapshot_sha256=snapshot.snapshot_sha256,
    )
    mirrors = _mirrors(root)
    with CanonicalReducer(ledgers, context) as reducer:
        if status_code.startswith(("skipped:", "deferred:")):
            model["authored_metadata_state"] = "pending"
            model["accuracy_gate"].update(
                {
                    "vet_identity": None,
                    "gate_id": None,
                    "verdict": None,
                    "current": False,
                }
            )
            model["completeness"].update(
                {
                    "source_read_fields_complete": False,
                    "evidence_coverage_complete": False,
                    "accuracy_gate_current": False,
                }
            )
            for field in (
                "taxonomy",
                "external_metadata",
                "website",
                "people_and_origin",
                "dates",
                "citation",
                "licenses",
            ):
                model[field] = None
            source_bytes = b"round-14 terminal source fixture"
            source = model["source_resolution"]["sources"][0]
            source["content_sha256"] = hash_bytes(source_bytes)
            source["byte_count"] = len(source_bytes)
            source_manifest = {
                "sources": [dict(source)],
            }
            source_manifest["manifest_sha256"] = stable_hash(source_manifest["sources"])
            evidence_ids = ["evidence-1"] if status_code.startswith("deferred:") else []
            if evidence_ids:
                model["evidence"]["excerpts"][0]["supports"].append(
                    status_code.removeprefix("deferred:")
                )
            _bind_model_identities(model)
            payload = (
                {
                    "arm": "DEFER_RECOMMENDATION",
                    "platform": status_code.removeprefix("deferred:needs-"),
                    "source_ids": ["source-1"],
                    "evidence_ids": evidence_ids,
                    "evidence_identity": model["evidence"]["evidence_identity"],
                    "license_identity": stable_hash(model["licenses"]),
                }
                if status_code.startswith("deferred:")
                else {
                    "arm": "SKIP_RECOMMENDATION",
                    "status_code": status_code,
                    "source_ids": ["source-1"],
                    "evidence_ids": evidence_ids,
                    "evidence_identity": model["evidence"]["evidence_identity"],
                    "search_report_identity": stable_hash(
                        model["source_resolution"]["search_report"]
                    ),
                    "license_identity": stable_hash(model["licenses"]),
                }
            )
            payload["recommendation_sha256"] = stable_hash(payload)
            author_result = {
                "schema_version": "menagerie.crawler.author-result.v3",
                "result_id": f"result-{stable_id}",
                "result_sha256": HASH,
                "kind": payload["arm"],
                "stable_id": stable_id,
                "work_id": f"work-{stable_id}",
                "campaign_id": f"campaign-{stable_id}",
                "created_at": NOW,
                "author_identity": context.author_model_identity,
                "prompt_identity": context.author_prompt_identity,
                "dispatcher_identity": context.author_dispatcher_identity,
                "source_manifest_identity": source_manifest["manifest_sha256"],
                "intake_snapshot_id": context.active_intake_snapshot_id,
                "intake_snapshot_sha256": context.active_intake_snapshot_sha256,
                "intake_item_sha256": stable_hash(context.intake_by_stable_id[stable_id]),
                "payload": payload,
            }
            author_result["result_sha256"] = stable_hash(
                {key: value for key, value in author_result.items() if key != "result_sha256"}
            )
            stage_private_artifact(
                (
                    ArtifactInput(
                        content=source_bytes,
                        content_sha256=hash_bytes(source_bytes),
                        logical_role="source",
                        logical_path=f"menagerie/crawler/source_cas/{stable_id}.source",
                        source_id="source-1",
                        origin=ArtifactOrigin(
                            url=str(source["url"]), revision=str(source["revision"])
                        ),
                        fetch_recipe=str(source["fetch_recipe"]),
                        evidence_ids=tuple(evidence_ids),
                        media_type=str(source["media_type"]),
                    ),
                ),
                context=context,
                stable_id=stable_id,
                work_id=f"work-{stable_id}",
                author_result=author_result,
                proposal=None,
                source_manifest=source_manifest,
                mirrors=mirrors,
                ledger=reducer.artifact_ledger,
                created_at=NOW,
            )
            gate = make_gate([stable_id], gate_kind="fidelity")
            gate["gate_kind"] = "terminal_disposition"
            gate["items"][0]["rung_check"]["selected_rung"] = model["source_resolution"]["rung"]
            gate["items"][0]["terminal_disposition"] = {
                "author_result_id": author_result["result_id"],
                "author_result_sha256": author_result["result_sha256"],
                "kind": author_result["kind"],
                "predicate": status_code.split(":", 1)[1],
                "verdict": "accepted",
                "source_manifest_identity": source_manifest["manifest_sha256"],
                "source_ids": ["source-1"],
                "evidence_identity": payload["evidence_identity"],
                "evidence_ids": evidence_ids,
                "license_identity": payload["license_identity"],
                "findings": [],
            }
        if status_code.startswith("deferred:"):
            attempt = make_attempt(stable_id)
            attempt["environment"] = None
            attempt["identities"]["environment"] = None
            attempt["identities"]["execution"] = None
            attempt["worker_receipt"]["present"] = False
            attempt["result"] = "observed"
            attempt["raw_award_receipt"] = None
            attempt["raw_award_receipt_sha256"] = None
            attempt["supervisor_observation"]["stdout_sha256"] = None
            attempt["supervisor_observation"]["stderr_sha256"] = None
            attempt["supervisor_observation"]["stdout_completion_line"] = None
            attempt["defer_evidence"] = {
                "target_status": status_code,
                "source_ids": ["source-1"],
                "probe_attempt_ids": [],
                "explanation": "source requires the deferred platform",
            }
            terminal_attempts = [attempt]
            bind_terminal_attempts(model, terminal_attempts)
        if gate is not None:
            reducer.append_gate(gate)
        for attempt in terminal_attempts:
            reducer.append_attempt(attempt)
        reducer.append_model(reducer.prepare_model(model))
    rebuild_views(
        snapshot.root / "items.jsonl",
        records,
        crawler / "views",
        root / ".crawl-local" / "state.sqlite",
        context=context,
    )
    artifacts: list[LicensedArtifact] = []
    _write_jsonl(crawler / "mirrors" / "public-manifest.jsonl", [])
    _write_jsonl(crawler / "mirrors" / "private-manifest.jsonl", [])
    report = pre_public_merge_sweep(artifacts, mirrors)
    _write_jsonl(crawler / "license_reports" / "current.json", [report.to_dict()])
    return snapshot, mirrors


def _authority_context(snapshot: IntakeSnapshot) -> AuthorityContext:
    """Build the exact checkpoint authority context for a materialized snapshot."""

    return make_authority_context(
        (item.stable_id for item in snapshot.items),
        snapshot_id=snapshot.snapshot_id,
        snapshot_sha256=snapshot.snapshot_sha256,
    )


def test_checkpoint_refuses_wrong_branch_before_staging(tmp_path: Path) -> None:
    """A non-crawler branch is a typed refusal and cannot reach Git add."""

    git = RecordingGit()
    with pytest.raises(WrongCheckpointBranch):
        create_canonical_checkpoint(
            tmp_path,
            tmp_path / "missing-intake",
            authority_context=make_authority_context(()),
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
    snapshot = _snapshot(tmp_path)
    exit_code = main(
        [
            "--repo-root",
            str(tmp_path),
            "checkpoint",
            "--intake",
            str(snapshot.root),
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
        authority_context=_authority_context(snapshot),
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
            authority_context=_authority_context(snapshot),
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
            authority_context=_authority_context(snapshot),
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
        authority_context=_authority_context(snapshot),
        branch="menagerie/crawler-pipeline",
        git_runner=git,
    )
    add = next(command for command in git.commands if command[:3] == ("git", "add", "--"))
    staged = set(add[3:])
    assert staged == {path.as_posix() for path in result.paths}
    assert "menagerie/crawler/license_reports/current.json" in staged
    assert all(not path.startswith(".crawl-local/") for path in staged)
    assert all("push" not in command for command in git.commands)


def test_deferred_ungated_promotion_cannot_checkpoint_public_code(tmp_path: Path) -> None:
    """A checked deferral checkpoints durable private custody without public bytes."""

    snapshot, mirrors = _clean_state(
        tmp_path,
        status_code="deferred:needs-cuda",
        accepted_metadata=True,
    )
    events = scan_jsonl(
        tmp_path / "menagerie" / "crawler" / "records" / "artifacts" / "current-shard.jsonl"
    )
    assert [event["event_kind"] for event in events] == ["staged-private"]
    assert tuple(mirrors.iter_objects(MirrorClass.PUBLIC)) == ()
    assert tuple(mirrors.iter_objects(MirrorClass.PRIVATE))
    result = create_canonical_checkpoint(
        tmp_path,
        snapshot.root,
        mirrors=mirrors,
        authority_context=_authority_context(snapshot),
        branch="menagerie/crawler-pipeline",
        git_runner=RecordingGit(),
    )
    assert result.license_report.passed


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
            authority_context=_authority_context(snapshot),
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
        authority_context=_authority_context(snapshot),
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
            authority_context=_authority_context(snapshot),
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
            authority_context=_authority_context(snapshot),
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
