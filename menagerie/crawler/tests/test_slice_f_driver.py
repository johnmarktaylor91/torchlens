"""Slice F single-writer driver, award, pause, and notification tests."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import pytest

import menagerie.crawler.driver as driver_module
from menagerie.crawler.checker_dispatch import CheckerBackoffSignal
from menagerie.crawler.constants import (
    CheckerPauseReason,
    EnvironmentPhase,
)
from menagerie.crawler.driver import (
    AuthorArtifact,
    AuthorLane,
    CheckerLane,
    CheckerOutcome,
    CommandAuthorLane,
    CommandNotifier,
    CrawlerDriver,
    DriverConfig,
    DriverDependencies,
    DriverIntegrationError,
    DriverPaths,
    EnvironmentLane,
    EnvironmentBinding,
    ForwardLane,
    Notifier,
    UsagePauseScheduler,
    WorkItem,
    _execution_identity,
    _environment_binding,
    _runner_identity,
    _supervise_environment_worker,
)
from menagerie.crawler.envs import (
    EnvironmentIntent,
    IntentProbes,
    LockArtifacts,
    load_environment_registry,
)
from menagerie.crawler.env_lifecycle import EnvironmentProbeError
from menagerie.crawler.intake import IntakeSnapshot, create_intake_snapshot
from menagerie.crawler.identity import hash_bytes, stable_hash
from menagerie.crawler.fetcher import fetch_targets as controlled_fetch_targets
from menagerie.crawler.metadata import MANDATORY_EXTERNAL_FIELDS
from menagerie.crawler.models import LedgerPaths
from menagerie.crawler.recordio import JsonlLedger, scan_jsonl
from menagerie.crawler.reducer import CanonicalReducer
from menagerie.crawler.status import assert_partition
from menagerie.crawler.tests.conftest import (
    HASH,
    NOW,
    make_attempt,
    make_author_proposal,
    make_gate,
    make_model,
)
from menagerie.crawler.wakeup import (
    OperationalContext,
    WakeupBackend,
    WakeupManager,
)
from menagerie.crawler.worker_supervisor import SupervisorObservation


class InjectedKill(RuntimeError):
    """Simulate an uncatchable process boundary failure in a deterministic test."""


class FakeAuthor(AuthorLane):
    """Return complete synthetic proposals without a live author session."""

    def __init__(self) -> None:
        """Initialize per-model invocation counts."""

        self.calls: dict[str, int] = {}

    def author(self, item: WorkItem, work_root: Path, config: DriverConfig) -> AuthorArtifact:
        """Return one accepted two-mode proposal."""

        del config
        self.calls[item.stable_id] = self.calls.get(item.stable_id, 0) + 1
        proposal = make_author_proposal(item.stable_id)
        proposal["work_id"] = f"work-{item.stable_id}"
        facts = proposal["proposed_facts"]
        facts["modes"]["meaningful_modes"] = ["train", "eval"]
        facts["external_metadata"]["modes"]["meaningful_modes"] = ["train", "eval"]
        model_dir = work_root / item.stable_id / "fake-model"
        model_dir.mkdir(parents=True, exist_ok=True)
        return AuthorArtifact(proposal, {"sources": []}, model_dir)


class OneModelAuthorFailure(FakeAuthor):
    """Raise from the author lane for exactly one model."""

    def __init__(self, failed_id: str) -> None:
        """Store the model-local author failure identity."""

        super().__init__()
        self.failed_id = failed_id

    def author(self, item: WorkItem, work_root: Path, config: DriverConfig) -> AuthorArtifact:
        """Raise for one item and author every later item normally."""

        if item.stable_id == self.failed_id:
            raise RuntimeError("synthetic author command failure")
        return super().author(item, work_root, config)


class TerminalOutcomeAuthor(FakeAuthor):
    """Return one evidenced deferral and one epistemic skip recommendation."""

    def __init__(self) -> None:
        """Initialize deterministic outcome order."""

        super().__init__()
        self._index = 0

    def author(self, item: WorkItem, work_root: Path, config: DriverConfig) -> AuthorArtifact:
        """Return a driver-consumable terminal outcome artifact."""

        artifact = super().author(item, work_root, config)
        self._index += 1
        if self._index == 1:
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
        artifact.proposal["proposed_facts"]["source_resolution"]["rung"] = "R5_SKIP"
        return AuthorArtifact(
            artifact.proposal,
            artifact.source_manifest,
            artifact.model_dir,
            terminal_status="skipped:no-description",
            terminal_detail="bounded search found no architecture description",
        )


class FakeChecker(CheckerLane):
    """Return accurate synthetic gates or one configured quota signal."""

    def __init__(self, *, quota: bool = False) -> None:
        """Configure whether metadata checking reports quota exhaustion."""

        self.quota = quota
        self.metadata_calls = 0
        self.fidelity_calls = 0

    def check_metadata(
        self, artifacts: Sequence[AuthorArtifact], work_root: Path, config: DriverConfig
    ) -> CheckerOutcome:
        """Return one exhaustive accurate metadata gate."""

        del work_root, config
        self.metadata_calls += 1
        if self.quota:
            return CheckerOutcome(
                backoff=CheckerBackoffSignal(
                    CheckerPauseReason.QUOTA_EXHAUSTED,
                    None,
                    "2026-07-14T13:00:00Z",
                    "quota exhausted",
                )
            )
        stable_ids = [str(artifact.proposal["stable_id"]) for artifact in artifacts]
        gate = make_gate(stable_ids, gate_id=f"gate-metadata-{stable_ids[0]}")
        gate.pop("ledger_seq", None)
        gate.pop("payload_sha256", None)
        for item in gate["items"]:
            artifact = next(
                value for value in artifacts if value.proposal["stable_id"] == item["stable_id"]
            )
            proposal = artifact.proposal
            item["work_id"] = proposal["work_id"]
            item["vet_identity"] = proposal["vet_identity"]
            item["fidelity_identity"] = None
            item["verified_hashes"] = {
                "proposal": proposal["proposal_sha256"],
                **proposal["verified_hashes"],
            }
            item["rung_check"]["selected_rung"] = proposal["proposed_facts"]["source_resolution"][
                "rung"
            ]
            item["field_checks"] = [
                {
                    "field": f"external_metadata.{field}",
                    "verdict": "accurate",
                    "evidence_ids": ["evidence-1"],
                    "checked_source_ids": ["source-1"],
                    "reason": "supported",
                    "required_repair": None,
                }
                for field in MANDATORY_EXTERNAL_FIELDS
            ]
        return CheckerOutcome(gate=gate)

    def check_fidelity(
        self, artifact: AuthorArtifact, work_root: Path, config: DriverConfig
    ) -> CheckerOutcome:
        """Return an accurate match fidelity gate."""

        del work_root, config
        self.fidelity_calls += 1
        stable_id = str(artifact.proposal["stable_id"])
        gate = make_gate(
            [stable_id],
            gate_id=f"gate-fidelity-{stable_id}",
            gate_kind="fidelity",
            fidelity_identity=str(artifact.proposal["fidelity_identity"]),
        )
        gate.pop("ledger_seq", None)
        gate.pop("payload_sha256", None)
        item = gate["items"][0]
        item["work_id"] = artifact.proposal["work_id"]
        item["vet_identity"] = artifact.proposal["vet_identity"]
        item["verified_hashes"] = {
            "proposal": artifact.proposal["proposal_sha256"],
            **artifact.proposal["verified_hashes"],
        }
        item["rung_check"]["selected_rung"] = artifact.proposal["proposed_facts"][
            "source_resolution"
        ]["rung"]
        return CheckerOutcome(gate=gate)


class FailingMetadataChecker(FakeChecker):
    """Raise a checker-contract error for every metadata batch."""

    def check_metadata(
        self, artifacts: Sequence[AuthorArtifact], work_root: Path, config: DriverConfig
    ) -> CheckerOutcome:
        """Simulate a checker process that returned no valid envelope."""

        del artifacts, work_root, config
        raise RuntimeError("synthetic invalid checker envelope")


class FakeForward(ForwardLane):
    """Return schema-valid attempts for every mode/cold invocation."""

    def __init__(self) -> None:
        """Initialize per-model invocation counts."""

        self.calls: dict[str, int] = {}

    def forward(
        self,
        artifact: AuthorArtifact,
        environment: EnvironmentBinding,
        cold_runs: int,
        work_root: Path,
    ) -> Sequence[Mapping[str, Any]]:
        """Build clean train/eval attempts with complete model output signatures."""

        del work_root
        stable_id = str(artifact.proposal["stable_id"])
        self.calls[stable_id] = self.calls.get(stable_id, 0) + 1
        execution_identity = _execution_identity(artifact.proposal, environment)
        output_signature = make_model()["observed"]["output_signature"]
        attempts: list[dict[str, Any]] = []
        attempt_no = 0
        for cold in range(cold_runs):
            for mode in ("train", "eval"):
                attempt_no += 1
                attempt = make_attempt(
                    stable_id,
                    attempt_id=(f"attempt-{stable_id}-{execution_identity[7:19]}-{cold}-{mode}"),
                    mode=mode,
                )
                attempt["attempt_no"] = attempt_no
                attempt.pop("ledger_seq", None)
                attempt.pop("payload_sha256", None)
                attempt["worker_receipt"]["output_signature"] = output_signature
                attempt["identities"].update(
                    {
                        "source": artifact.proposal["source_identity"],
                        "evidence": artifact.proposal["evidence_identity"],
                        "recipe": artifact.proposal["recipe_revision"],
                        "environment": environment.env_generation,
                        "execution": execution_identity,
                        "runner": _runner_identity(),
                        "author_prompt": artifact.proposal["author"]["prompt_sha256"],
                    }
                )
                attempt["environment"].update(
                    {
                        "family": environment.family,
                        "target": environment.target,
                        "env_id": str(environment.prefix),
                        "lock_sha256": environment.lock_sha256,
                        "resolved_export_sha256": environment.resolved_export_sha256,
                        "packages_manifest_sha256": environment.packages_manifest_sha256,
                    }
                )
                attempts.append(attempt)
        return attempts


class OneModelForwardFailure(FakeForward):
    """Return a real failed mode attempt for exactly one model."""

    def __init__(self, failed_id: str) -> None:
        """Store the one model whose forward must fail."""

        super().__init__()
        self.failed_id = failed_id

    def forward(
        self,
        artifact: AuthorArtifact,
        environment: EnvironmentBinding,
        cold_runs: int,
        work_root: Path,
    ) -> Sequence[Mapping[str, Any]]:
        """Return complete attempts, changing one mode to a schema-valid failure."""

        attempts = [
            dict(value) for value in super().forward(artifact, environment, cold_runs, work_root)
        ]
        stable_id = str(artifact.proposal["stable_id"])
        if stable_id != self.failed_id:
            return attempts
        failed = attempts[0]
        failed["result"] = "failed"
        failed["worker_receipt"] = dict(failed["worker_receipt"])
        failed["worker_receipt"]["forward_completed"] = False
        failed["error"] = {
            "stage": "forward",
            "reason_code": "mode-run",
            "exception_type": "builtins.RuntimeError",
            "message": "synthetic forward failure",
            "traceback": "Traceback: synthetic forward failure",
            "no_traceback_reason": None,
            "native_crash": False,
            "root_cause_fingerprint": HASH,
            "details": {"mode": failed["mode"]},
        }
        return attempts


class InaccurateChecker(FakeChecker):
    """Return the same inaccurate root cause until the driver terminalizes it."""

    def check_metadata(
        self, artifacts: Sequence[AuthorArtifact], work_root: Path, config: DriverConfig
    ) -> CheckerOutcome:
        """Return one complete inaccurate metadata gate for every requested item."""

        outcome = super().check_metadata(artifacts, work_root, config)
        assert outcome.gate is not None
        for item in outcome.gate["items"]:
            item["verdict"] = "inaccurate"
            item["required_repairs"] = ["correct the unsupported metadata claim"]
            item["field_checks"][0]["verdict"] = "inaccurate"
            item["field_checks"][0]["required_repair"] = "correct the claim"
        return outcome


class FidelityAuthor(FakeAuthor):
    """Mark every proposal as an R3 port requiring fidelity review."""

    def author(self, item: WorkItem, work_root: Path, config: DriverConfig) -> AuthorArtifact:
        """Return a complete proposal with required fidelity enabled."""

        artifact = super().author(item, work_root, config)
        fidelity_identity = HASH
        artifact.proposal["fidelity_identity"] = fidelity_identity
        facts = artifact.proposal["proposed_facts"]
        facts["source_resolution"]["rung"] = "R3_PORT"
        facts["fidelity"].update(
            {
                "required": True,
                "reason": "synthetic R3 port",
                "verdict": None,
                "fidelity_identity": fidelity_identity,
                "gate_id": None,
                "current": False,
            }
        )
        return artifact


class ChangedInputAuthor(FakeAuthor):
    """Return a new source/input-bound recipe generation for resume tests."""

    def author(self, item: WorkItem, work_root: Path, config: DriverConfig) -> AuthorArtifact:
        """Change source, recipe, and dummy-input dependencies together."""

        artifact = super().author(item, work_root, config)
        artifact.proposal["source_identity"] = "sha256:" + "b" * 64
        artifact.proposal["recipe_revision"] = "sha256:" + "c" * 64
        facts = artifact.proposal["proposed_facts"]
        facts["implementation"]["recipe_revision"] = artifact.proposal["recipe_revision"]
        facts["input_contract"]["args"][0]["shape"] = [1, 3, 9, 9]
        artifact.proposal["proposal_sha256"] = stable_hash(
            {key: value for key, value in artifact.proposal.items() if key != "proposal_sha256"}
        )
        return artifact


class RejectingFidelityChecker(FakeChecker):
    """Return accurate metadata and a material-drift fidelity rejection."""

    def check_fidelity(
        self, artifact: AuthorArtifact, work_root: Path, config: DriverConfig
    ) -> CheckerOutcome:
        """Return a complete major-drift fidelity gate."""

        outcome = super().check_fidelity(artifact, work_root, config)
        assert outcome.gate is not None
        item = outcome.gate["items"][0]
        item["verdict"] = "inaccurate"
        item["fidelity"]["verdict"] = "major-drift"
        item["fidelity"]["contradictions"] = ["material topology mismatch"]
        return outcome


class FailingNotifier(Notifier):
    """Record attempted messages while reporting delivery failure."""

    def __init__(self) -> None:
        """Initialize attempted messages."""

        self.messages: list[str] = []

    def notify(self, summary: str) -> bool:
        """Record one call and report that delivery failed."""

        self.messages.append(summary)
        return False


class FakeEnvironments(EnvironmentLane):
    """Exercise environment callbacks sequentially without conda."""

    def __init__(self, root: Path) -> None:
        """Store the fake prefix root and lifecycle event log."""

        self.root = root
        self.events: list[str] = []
        self.active = 0

    def run(self, intent: EnvironmentIntent, *, use: Any) -> object:
        """Call use under one active fake prefix and always tear down."""

        assert self.active == 0
        self.active += 1
        self.events.append(f"create:{intent.name}")
        prefix = self.root / intent.name
        prefix.mkdir(parents=True, exist_ok=True)
        try:
            use(prefix)
        finally:
            self.events.append(f"remove:{intent.name}")
            self.active -= 1
        return object()


class FailingEnvironments(FakeEnvironments):
    """Fail one intent before any model callback begins."""

    def run(self, intent: EnvironmentIntent, *, use: Any) -> object:
        """Raise a typed probe failure instead of invoking the model callback."""

        del intent, use
        raise EnvironmentProbeError("synthetic environment probe failure")


class FakeNotifier(Notifier):
    """Capture ASCII summaries without external messaging."""

    def __init__(self) -> None:
        """Initialize the delivered message list."""

        self.messages: list[str] = []

    def notify(self, summary: str) -> bool:
        """Capture one summary and report success."""

        summary.encode("ascii")
        self.messages.append(summary)
        return True


class FakePauseScheduler(UsagePauseScheduler):
    """Use the real idempotent wakeup records with a no-op OS activator."""

    def __init__(self, root: Path) -> None:
        """Store the fake wakeup definition root and call count."""

        self.root = root
        self.calls = 0

    def schedule(
        self,
        signal: CheckerBackoffSignal,
        operational: JsonlLedger,
        context: OperationalContext,
        created_at: str,
        reset_at: str,
    ) -> None:
        """Record pause/wakeup events without installing an OS scheduler."""

        self.calls += 1
        manager = WakeupManager(
            self.root,
            operational,
            ["python", "-m", "menagerie.crawler", "run", "--resume"],
            backend=WakeupBackend.CRON,
            activator=lambda _spec: None,
        )
        manager.record_pause_and_schedule(
            provider="openai",
            observed_response=signal.response_excerpt,
            reset_at=reset_at,
            context=context,
            created_at=created_at,
        )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write small discovery inputs for an intake snapshot."""

    path.write_text("".join(json.dumps(dict(row)) + "\n" for row in rows), encoding="utf-8")


def _snapshot(tmp_path: Path, count: int = 10) -> IntakeSnapshot:
    """Create one immutable synthetic intake snapshot."""

    master = tmp_path / "master.jsonl"
    deferred = tmp_path / "deferred.jsonl"
    _write_jsonl(
        master,
        [
            {"name": f"Example{index}", "zoo": "fixtures", "variant": "base"}
            for index in range(count)
        ],
    )
    _write_jsonl(deferred, [])
    return create_intake_snapshot(master, deferred, tmp_path / "intake")


def _mixed_phase_snapshot(tmp_path: Path) -> IntakeSnapshot:
    """Create ten PyTorch rows and one unprocessed native-tail row."""

    master = tmp_path / "master.jsonl"
    deferred = tmp_path / "deferred.jsonl"
    rows = [
        {"name": f"Example{index}", "zoo": "fixtures", "variant": "base"} for index in range(10)
    ]
    rows.append({"name": "TensorFlowNativeExample", "zoo": "tensorflow", "variant": "base"})
    _write_jsonl(master, rows)
    _write_jsonl(deferred, [])
    return create_intake_snapshot(master, deferred, tmp_path / "intake")


def _paths(tmp_path: Path, snapshot: IntakeSnapshot) -> DriverPaths:
    """Return isolated runtime and canonical ledger paths."""

    return DriverPaths(
        tmp_path / "runtime",
        snapshot.root,
        LedgerPaths(
            tmp_path / "records" / "models.jsonl",
            tmp_path / "records" / "attempts.jsonl",
            tmp_path / "records" / "gates.jsonl",
        ),
    )


def _test_environment(prefix: Path) -> EnvironmentBinding:
    """Build exact synthetic byte identities for a fake active environment."""

    return EnvironmentBinding(
        prefix=prefix,
        python_executable=Path(sys.executable),
        family="core",
        target="test",
        env_generation=HASH,
        lock_sha256=HASH,
        resolved_export_sha256=HASH,
        packages_manifest_sha256=HASH,
        python_version="3.11",
        compiler_identity="test-compiler",
        sdk_identity="test-sdk",
    )


def test_environment_binding_hashes_real_bytes_and_launches_prefix_python(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Runtime provenance and sandbox argv use the verified environment itself."""

    lock_bytes = b"exact lock bytes"
    export_bytes = b"exact resolved export bytes"
    package_bytes = b'{"packages":["torch==test"]}'
    lock_path = tmp_path / "target.lock"
    export_path = tmp_path / "target.resolved.json"
    lock_path.write_bytes(lock_bytes)
    export_path.write_bytes(export_bytes)
    prefix = tmp_path / "env-prefix"
    (prefix / "bin").mkdir(parents=True)
    (prefix / "bin" / "python").symlink_to(Path(sys.executable))
    (prefix / "packages-manifest.json").write_bytes(package_bytes)
    intent = EnvironmentIntent(
        name="core",
        phase=EnvironmentPhase.PYTORCH,
        framework="pytorch",
        description="test",
        split_guidance="test",
        channels=("conda-forge",),
        dependencies=("python",),
        probes=IntentProbes((), (), ()),
        lock=LockArtifacts(
            target="linux-64",
            lock_path=lock_path,
            export_path=export_path,
            export_hash_path=tmp_path / "target.sha256",
            lock_bytes=lock_bytes,
            export_bytes=export_bytes,
            declared_export_hash=hash_bytes(export_bytes),
        ),
        generation=None,
    )
    binding = _environment_binding(intent, prefix, strict=True)
    assert binding.lock_sha256 == hash_bytes(lock_bytes)
    assert binding.resolved_export_sha256 == hash_bytes(export_bytes)
    assert binding.packages_manifest_sha256 == hash_bytes(package_bytes)

    receipt_path = tmp_path / "result" / "receipt.json"
    captured: list[str] = []

    def fake_isolated(argv: Sequence[str], scratch_root: Path, **kwargs: Any) -> Any:
        """Capture sandbox input and publish one self-hashed receipt."""

        del scratch_root, kwargs
        captured.extend(argv)
        payload = {"receipt_version": "test"}
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(
            json.dumps({**payload, "receipt_sha256": stable_hash(payload)}),
            encoding="utf-8",
        )
        return SupervisorObservation(
            argv=tuple(argv),
            cwd=str(tmp_path),
            exit_code=0,
            signal_number=None,
            wall_seconds=0.1,
            cpu_seconds=0.1,
            peak_rss_bytes=1,
            timed_out=False,
            rss_exceeded=False,
            stdout_sha256=hash_bytes(b""),
            stdout_bytes=0,
            stdout_tail="",
            stderr_sha256=hash_bytes(b""),
            stderr_bytes=0,
            stderr_tail="",
            stdout_path=str(tmp_path / "stdout"),
            stderr_path=str(tmp_path / "stderr"),
        )

    monkeypatch.setattr(driver_module, "run_isolated_subprocess", fake_isolated)
    result = _supervise_environment_worker(
        tmp_path / "request.json",
        receipt_path,
        tmp_path / "supervisor",
        binding.python_executable,
        timeout_seconds=1,
        rss_limit_bytes=1024,
        cwd=tmp_path,
    )
    assert result.receipt_error is None
    assert captured[0] == str(prefix / "bin" / "python")


def test_author_source_handshake_freezes_nonempty_cas_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Author pins are controlled-fetched before the evidence envelope is frozen."""

    snapshot = _snapshot(tmp_path)
    driver = _driver(tmp_path, snapshot)
    item = driver._ordered_work(snapshot, {})[0]
    content = b"ExampleNet is a source-grounded architecture."
    digest = hash_bytes(content)

    def fake_run(argv: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        """Write the author-requested exact source target."""

        del kwargs
        request = json.loads(Path(argv[-1]).read_text(encoding="utf-8"))
        output = Path(request["required_output_path"])
        output.write_text(
            json.dumps(
                {
                    "sources": [
                        {
                            "source_id": "source-1",
                            "url": "https://example.com/model.txt",
                            "revision": "v1",
                            "expected_sha256": digest,
                            "media_type": "text/plain",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(list(argv), 0, "", "")

    monkeypatch.setattr(driver_module.subprocess, "run", fake_run)
    monkeypatch.setattr(
        driver_module,
        "fetch_targets",
        lambda targets, root: controlled_fetch_targets(
            targets, root, fetch_bytes=lambda _url: content
        ),
    )
    lane = CommandAuthorLane(("fake-author",))
    manifest = lane._fetch_author_sources(item, tmp_path / "author")
    assert manifest["sources"]
    assert Path(manifest["sources"][0]["cas_path"]).read_bytes() == content

    def empty_run(argv: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        """Write an invalid empty author source request."""

        del kwargs
        request = json.loads(Path(argv[-1]).read_text(encoding="utf-8"))
        Path(request["required_output_path"]).write_text(
            json.dumps({"sources": []}), encoding="utf-8"
        )
        return subprocess.CompletedProcess(list(argv), 0, "", "")

    monkeypatch.setattr(driver_module.subprocess, "run", empty_run)
    with pytest.raises(DriverIntegrationError, match="at least one pinned source"):
        lane._fetch_author_sources(item, tmp_path / "empty-author")


def _driver(
    tmp_path: Path,
    snapshot: IntakeSnapshot,
    *,
    author: Optional[AuthorLane] = None,
    checker: Optional[CheckerLane] = None,
    forward: Optional[ForwardLane] = None,
    environments: Optional[EnvironmentLane] = None,
    notifier: Optional[Notifier] = None,
    boundary: Any = lambda _boundary, _stable_id: None,
    review_at: Optional[int] = None,
    milestones: tuple[int, ...] = (),
    pause_scheduler: Optional[FakePauseScheduler] = None,
    phase: Optional[str] = None,
) -> CrawlerDriver:
    """Build a fully fake deterministic driver."""

    dependencies = DriverDependencies(
        author or FakeAuthor(),
        checker or FakeChecker(),
        forward or FakeForward(),
        environments or FakeEnvironments(tmp_path / "fake-envs"),
        notifier or FakeNotifier(),
        lambda: NOW,
        boundary,
        pause_scheduler,
    )
    return CrawlerDriver(
        _paths(tmp_path, snapshot),
        DriverConfig(
            target="osx-arm64",
            phase=phase,
            run_id="run-test",
            machine_id="machine-test",
            review_checkpoint_at=review_at,
            progress_milestones=milestones,
        ),
        dependencies,
        registry=load_environment_registry(target="osx-arm64"),
    )


@pytest.mark.parametrize(
    "boundary", ["after-author", "after-gate", "after-forward", "after-reduce"]
)
def test_resume_is_lock_safe_and_duplicate_free_at_critical_boundaries(
    tmp_path: Path, boundary: str
) -> None:
    """A kill after every durable boundary resumes the first unsatisfied identity."""

    snapshot = _snapshot(tmp_path)
    killed = False

    def kill_once(observed: str, stable_id: str) -> None:
        """Raise exactly once at the selected boundary."""

        nonlocal killed
        del stable_id
        if observed == boundary and not killed:
            killed = True
            raise InjectedKill(boundary)

    with pytest.raises(InjectedKill):
        _driver(tmp_path, snapshot, boundary=kill_once).run()
    result = _driver(tmp_path, snapshot).run()
    assert result.status == "complete"
    paths = _paths(tmp_path, snapshot)
    assert len(scan_jsonl(paths.ledgers.models)) == 10
    assert len(scan_jsonl(paths.ledgers.attempts)) == 20
    assert len(scan_jsonl(paths.ledgers.gates)) == 1


def test_changed_source_and_input_supersede_run_and_force_reexecution(tmp_path: Path) -> None:
    """Stable-ID membership cannot reuse a run after dependent bytes change."""

    snapshot = _snapshot(tmp_path)
    first_forward = FakeForward()
    first = _driver(tmp_path, snapshot, forward=first_forward).run()
    assert first.status == "complete"
    paths = _paths(tmp_path, snapshot)
    first_models = {record["stable_id"]: record for record in scan_jsonl(paths.ledgers.models)}
    for item in snapshot.items:
        (paths.work_root / item.stable_id / "driver-author-artifact.json").unlink()

    second_forward = FakeForward()
    second = _driver(
        tmp_path,
        snapshot,
        author=ChangedInputAuthor(),
        forward=second_forward,
    ).run()

    assert second.status == "complete"
    assert set(second_forward.calls) == {item.stable_id for item in snapshot.items}
    revisions = scan_jsonl(paths.ledgers.models)
    assert len(revisions) == 2 * len(snapshot.items)
    for item in snapshot.items:
        current = [record for record in revisions if record["stable_id"] == item.stable_id][-1]
        assert current["parent_revision"] == first_models[item.stable_id]["record_revision"]
        assert current["implementation"]["recipe_revision"] == "sha256:" + "c" * 64


def test_only_driver_awards_runs_after_gate_receipts_and_both_modes(tmp_path: Path) -> None:
    """Gates and worker attempts alone never create a canonical runs record."""

    snapshot = _snapshot(tmp_path)
    paths = _paths(tmp_path, snapshot)
    checker = FakeChecker()
    author = FakeAuthor()
    item_driver = _driver(tmp_path, snapshot, author=author, checker=checker)
    work = item_driver._ordered_work(snapshot, {})
    artifacts = [author.author(item, paths.work_root, item_driver.config) for item in work]
    artifact = artifacts[0]
    gate = checker.check_metadata(artifacts, paths.work_root, item_driver.config).gate
    assert gate is not None
    gate["ledger_seq"] = 1
    gate["payload_sha256"] = HASH
    attempts = FakeForward().forward(
        artifact, _test_environment(tmp_path / "env"), 1, paths.work_root
    )
    with CanonicalReducer(paths.ledgers, [item.stable_id for item in snapshot.items]) as reducer:
        reducer.append_gate(gate)
        for attempt in attempts:
            reducer.append_attempt(attempt)
        assert reducer.current_records == {}
    result = _driver(tmp_path, snapshot, author=author, checker=checker).run()
    assert result.status == "complete"
    assert scan_jsonl(paths.ledgers.models)[0]["status"]["code"] == "runs"


def test_quota_pause_records_event_wakeup_and_no_partial_award(tmp_path: Path) -> None:
    """A checker quota signal visibly pauses, schedules once, and awards no run."""

    snapshot = _snapshot(tmp_path)
    scheduler = FakePauseScheduler(tmp_path / "wakeups")
    result = _driver(
        tmp_path,
        snapshot,
        checker=FakeChecker(quota=True),
        pause_scheduler=scheduler,
    ).run()
    assert result.status == "paused:usage-limit"
    assert scheduler.calls == 1
    paths = _paths(tmp_path, snapshot)
    assert scan_jsonl(paths.ledgers.models) == []
    events = scan_jsonl(paths.operational_ledger)
    assert [event["event_kind"] for event in events] == ["usage-pause", "wakeup"]


def test_review_checkpoint_blocks_notifies_and_is_one_shot(tmp_path: Path) -> None:
    """The configured terminal count blocks until explicit sign-off, then never re-blocks."""

    snapshot = _snapshot(tmp_path, count=10)
    notifier = FakeNotifier()
    first = _driver(tmp_path, snapshot, notifier=notifier, review_at=1).run()
    assert first.status == "paused:review-checkpoint"
    paths = _paths(tmp_path, snapshot)
    assert len(scan_jsonl(paths.ledgers.models)) == 1
    events = scan_jsonl(paths.operational_ledger)
    review = next(event for event in events if event["event_kind"] == "checkpoint-review")
    assert Path(review["report_path"]).is_file()
    assert len(notifier.messages) == 1
    still_paused = _driver(tmp_path, snapshot, review_at=1).run()
    assert still_paused.status == "paused:review-checkpoint"
    resumed = _driver(tmp_path, snapshot, review_at=1).run(after_review=True)
    assert resumed.status == "complete"
    assert len(scan_jsonl(paths.ledgers.models)) == 10
    final_events = scan_jsonl(paths.operational_ledger)
    assert [event["event_kind"] for event in final_events].count("checkpoint-review") == 1
    assert [event["event_kind"] for event in final_events].count("review-signoff") == 1


def test_progress_milestones_fire_once_without_pausing(tmp_path: Path) -> None:
    """Every crossed milestone emits one event/message and resume does not re-fire it."""

    snapshot = _snapshot(tmp_path, count=10)
    notifier = FakeNotifier()
    first = _driver(tmp_path, snapshot, notifier=notifier, milestones=(1, 2)).run()
    assert first.status == "complete"
    assert len(notifier.messages) == 2
    second_notifier = FakeNotifier()
    second = _driver(tmp_path, snapshot, notifier=second_notifier, milestones=(1, 2)).run()
    assert second.status == "complete"
    assert second_notifier.messages == []
    events = scan_jsonl(_paths(tmp_path, snapshot).operational_ledger)
    assert [
        event["milestone"] for event in events if event["event_kind"] == "progress-notification"
    ] == [1, 2]


def test_empty_milestones_and_missing_notifier_never_block(tmp_path: Path) -> None:
    """No milestones means no pings; a missing notifier remains a nonfatal fallback."""

    snapshot = _snapshot(tmp_path)
    result = _driver(tmp_path, snapshot, milestones=()).run()
    assert result.status == "complete"
    events = scan_jsonl(_paths(tmp_path, snapshot).operational_ledger)
    assert all(event["event_kind"] != "progress-notification" for event in events)
    assert CommandNotifier("/definitely/missing/send-to-jmt.sh").notify("milestone 1") is False


def test_failed_forward_terminalizes_and_campaign_continues(tmp_path: Path) -> None:
    """One failed forward records its real error while later models still run."""

    snapshot = _snapshot(tmp_path)
    failed_id = snapshot.items[0].stable_id
    result = _driver(
        tmp_path,
        snapshot,
        forward=OneModelForwardFailure(failed_id),
    ).run()
    assert result.status == "complete"
    paths = _paths(tmp_path, snapshot)
    models = {record["stable_id"]: record for record in scan_jsonl(paths.ledgers.models)}
    assert models[failed_id]["status"]["code"] == "failed:forward"
    assert len(models) == len(snapshot.items)
    assert any(record["status"]["code"] == "runs" for record in models.values())
    failed_attempt = next(
        record
        for record in scan_jsonl(paths.ledgers.attempts)
        if record["stable_id"] == failed_id and record["result"] == "failed"
    )
    assert failed_attempt["stage"] == "forward"
    assert failed_attempt["error"] == {
        "stage": "forward",
        "reason_code": "mode-run",
        "exception_type": "builtins.RuntimeError",
        "message": "synthetic forward failure",
        "traceback": "Traceback: synthetic forward failure",
        "no_traceback_reason": None,
        "native_crash": False,
        "root_cause_fingerprint": HASH,
        "details": {"mode": failed_attempt["mode"]},
    }


def test_author_failure_terminalizes_one_model_and_later_models_continue(tmp_path: Path) -> None:
    """An author command failure becomes failed:runner without aborting the campaign."""

    snapshot = _snapshot(tmp_path, count=20)
    failed_id = snapshot.items[0].stable_id
    result = _driver(
        tmp_path,
        snapshot,
        author=OneModelAuthorFailure(failed_id),
    ).run()
    assert result.status == "complete"
    models = {
        record["stable_id"]: record
        for record in scan_jsonl(_paths(tmp_path, snapshot).ledgers.models)
    }
    assert models[failed_id]["status"]["code"] == "failed:runner"
    assert sum(record["status"]["code"] == "runs" for record in models.values()) == 19


def test_checker_contract_failure_terminalizes_batch(tmp_path: Path) -> None:
    """An invalid checker envelope records failed:accuracy-gate for every batch item."""

    snapshot = _snapshot(tmp_path)
    result = _driver(tmp_path, snapshot, checker=FailingMetadataChecker()).run()
    assert result.status == "complete"
    models = scan_jsonl(_paths(tmp_path, snapshot).ledgers.models)
    assert {record["status"]["code"] for record in models} == {"failed:accuracy-gate"}
    assert {record["status"]["reason_code"] for record in models} == {"checker-contract-invalid"}


def test_environment_failure_terminalizes_intent(tmp_path: Path) -> None:
    """A typed environment probe failure terminalizes all assigned models."""

    snapshot = _snapshot(tmp_path)
    result = _driver(
        tmp_path,
        snapshot,
        environments=FailingEnvironments(tmp_path / "fake-envs"),
    ).run()
    assert result.status == "complete"
    models = scan_jsonl(_paths(tmp_path, snapshot).ledgers.models)
    assert {record["status"]["code"] for record in models} == {"failed:environment"}
    assert {record["status"]["reason_code"] for record in models} == {"probe-failed"}


def test_skip_and_evidenced_deferral_use_driver_terminalization(tmp_path: Path) -> None:
    """Ruled skip/deferral outcomes enter distinct terminal partition buckets."""

    snapshot = _snapshot(tmp_path, count=2)
    result = _driver(tmp_path, snapshot, author=TerminalOutcomeAuthor()).run()
    assert result.status == "complete"
    paths = _paths(tmp_path, snapshot)
    models = scan_jsonl(paths.ledgers.models)
    assert {record["status"]["code"] for record in models} == {
        "deferred:needs-cuda",
        "skipped:no-description",
    }
    deferral_attempt = next(
        record for record in scan_jsonl(paths.ledgers.attempts) if record["defer_evidence"]
    )
    assert deferral_attempt["defer_evidence"]["target_status"] == "deferred:needs-cuda"
    assert deferral_attempt["result"] == "observed"


def test_phase_filtered_run_is_not_global_complete(tmp_path: Path) -> None:
    """A PyTorch-only pass reports phase completion while native intake remains."""

    snapshot = _mixed_phase_snapshot(tmp_path)
    result = _driver(tmp_path, snapshot, phase="pytorch").run()
    assert result.status == "phase-complete:pytorch"
    paths = _paths(tmp_path, snapshot)
    current = {record["stable_id"]: record for record in scan_jsonl(paths.ledgers.models)}
    pytorch_ids = [
        item.stable_id for item in snapshot.items if "tensorflow" not in item.name.lower()
    ]
    assert_partition(pytorch_ids, current)
    assert len(current) == 10
    assert len(current) < len(snapshot.items)


def test_inaccurate_metadata_gate_repairs_are_bounded(tmp_path: Path) -> None:
    """A repeated inaccurate root cause stops in human-review terminal failures."""

    snapshot = _snapshot(tmp_path)
    checker = InaccurateChecker()
    result = _driver(tmp_path, snapshot, checker=checker).run()
    assert result.status == "complete"
    assert checker.metadata_calls == 2
    paths = _paths(tmp_path, snapshot)
    models = scan_jsonl(paths.ledgers.models)
    assert {record["status"]["code"] for record in models} == {"failed:accuracy-gate"}
    assert all(record["status"]["human_review"]["required"] for record in models)
    assert len(scan_jsonl(paths.ledgers.gates)) == 2


def test_fidelity_rejection_terminalizes_without_aborting(tmp_path: Path) -> None:
    """A material fidelity rejection becomes failed:fidelity for every model."""

    snapshot = _snapshot(tmp_path)
    result = _driver(
        tmp_path,
        snapshot,
        author=FidelityAuthor(),
        checker=RejectingFidelityChecker(),
    ).run()
    assert result.status == "complete"
    models = scan_jsonl(_paths(tmp_path, snapshot).ledgers.models)
    assert {record["status"]["code"] for record in models} == {"failed:fidelity"}
    assert {record["status"]["reason_code"] for record in models} == {"major-drift-cap-exhausted"}


def test_partial_multi_attempt_append_resumes_idempotently(tmp_path: Path) -> None:
    """A kill after attempt one lets the ledger assign sequence two on replay."""

    snapshot = _snapshot(tmp_path)
    killed = False

    def kill_after_first_attempt(boundary: str, stable_id: str) -> None:
        """Interrupt exactly after the first durable attempt append."""

        nonlocal killed
        del stable_id
        if boundary == "after-attempt" and not killed:
            killed = True
            raise InjectedKill("after-attempt")

    with pytest.raises(InjectedKill):
        _driver(tmp_path, snapshot, boundary=kill_after_first_attempt).run()
    result = _driver(tmp_path, snapshot).run()
    assert result.status == "complete"
    attempts = scan_jsonl(_paths(tmp_path, snapshot).ledgers.attempts)
    assert len(attempts) == 20
    assert [record["ledger_seq"] for record in attempts] == list(range(1, 21))


def test_failed_notification_is_separate_and_state_wipe_cannot_hide_milestone(
    tmp_path: Path,
) -> None:
    """Canonical count recovers a missed identity and records failed delivery separately."""

    snapshot = _snapshot(tmp_path)
    killed = False

    def kill_before_progress(boundary: str, stable_id: str) -> None:
        """Interrupt after the terminal model append but before milestone handling."""

        nonlocal killed
        del stable_id
        if boundary == "after-reduce" and not killed:
            killed = True
            raise InjectedKill("after-reduce")

    with pytest.raises(InjectedKill):
        _driver(
            tmp_path,
            snapshot,
            boundary=kill_before_progress,
            milestones=(1,),
        ).run()
    paths = _paths(tmp_path, snapshot)
    paths.driver_state.unlink(missing_ok=True)
    notifier = FailingNotifier()
    result = _driver(
        tmp_path,
        snapshot,
        notifier=notifier,
        milestones=(1,),
    ).run()
    assert result.status == "complete"
    assert len(notifier.messages) == 1
    events = scan_jsonl(paths.operational_ledger)
    progress = [event for event in events if event["event_kind"] == "progress-notification"]
    deliveries = [event for event in events if event["event_kind"] == "notification-delivery"]
    assert len(progress) == 1
    assert progress[0]["status"] == "progress-recorded"
    assert len(deliveries) == 1
    assert deliveries[0]["status"] == "notification-failed"
    assert deliveries[0]["details"]["delivered"] is False


def test_command_notifier_timeout_is_short_and_nonblocking(tmp_path: Path) -> None:
    """A hung notifier is killed by its short timeout and returns promptly."""

    del tmp_path
    notifier = CommandNotifier(
        f'{sys.executable} -c "import time; time.sleep(5)"', timeout_seconds=0.05
    )
    started = time.monotonic()
    assert notifier.notify("milestone 1") is False
    assert time.monotonic() - started < 1.0
